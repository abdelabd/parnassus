"""Generate CMS-full-sim-like pseudodata for the differentiable tuning harness.

This script produces a ROOT file in the cms-flow ``event_tree`` schema
(the same schema as Zenodo record 11389651) so that
``tune_cms_fullsim.py`` can be run end-to-end on reproducible input
without needing network access.

The generator runs in two clean phases: **all** Pythia events are
generated up front (and in parallel) into a single HepMC3 file, and only
then are those events streamed through TorchDelphes. This keeps the
expensive event generation cleanly separated from (and parallel to) the
calorimeter simulation.

Event generation
----------------
1. :class:`parnassus.pythia.HepMC3Generator` generates events of the
   selected ``--process`` at ``sqrt(s) = 13 TeV`` (``dijet`` = hard QCD
   2->2, ``HZZ4l`` = VBF H->ZZ->4l; calibration resonance guns, one parent
   per event decayed by Pythia: ``muongun`` = J/psi + Z -> mu mu, ``electrongun``
   = J/psi + Z -> e e, ``ksgun`` = K_S -> pi+ pi-, ``photongun`` = flat-in-log-E
   photons out to |eta| = 5 for the ECal; the Pythia ``.cmnd`` files
   live in ``processes/``) across
   ``--n-workers`` parallel Pythia8 processes, retries failed events so
   the merged HepMC3 file contains *exactly* ``--n-events`` events, and
   merges the per-worker outputs into one file. For ``dijet`` a
   ``PhaseSpace:pTHatMin`` override (``--pt-hat-min``, default 20) is
   always appended; the other processes only get it when the flag is
   given. Hadronization and final-state radiation are on, producing a
   realistic mix of charged hadrons, electrons, muons, photons,
   K-shorts, Lambdas, and neutrons -- the exact particle zoo that
   exercises every learnable parameter in ``CMSEnergyFlowDefault``.

2. The merged HepMC3 file is read back event by event; each event's
   final-state particles (HepMC ``status == 1``) are converted to the
   ``(N, N_FEATURES)`` particle tensor via
   :func:`parnassus.data.particle_io.hepmc_particles_to_tensor` and
   reduced to the class-based ``truth_*`` branches.

3. The truth particles are run through a
   :class:`CMSEnergyFlowDefault` card to produce a CMS-PF-like reco
   object collection. This phase runs on CUDA when a GPU is available
   (auto-detected; override with ``--device``), while phase 1 stays
   CPU-parallel. Two cards are used:

   - A **default** (non-learnable) card which gives us the truth
     particles themselves as the ``truth_*`` branches.
   - A **learnable card initialized from a parameter config** which
     plays the role of "CMS full simulation" for the fitting target.
     The config (``--param-config``, see
     :mod:`parnassus.torch_delphes.param_config`) may be PARTIAL: the
     scalars it lists are set to their physical ground-truth ``value``,
     every other learnable parameter keeps its card default. The shipped
     default (``cms_target_default.yaml``) is the pure card default. These
     values are exactly what Adam should recover when we fit the trainee
     against the ``pflow_*`` branches.

Output schema
-------------
The output ROOT file has a TTree named ``event_tree`` with jagged
per-event branches:

- ``truth_pt, truth_eta, truth_phi, truth_class``
- ``pflow_pt, pflow_eta, pflow_phi, pflow_class``

Class values follow :func:`parnassus.utils.pid_to_class` (0=charged
hadron, 1=electron, 2=muon, 3=neutral hadron, 4=photon), matching the
cms-flow loader.

Debug mode (``--debug``) additionally writes a per-module breakdown of
the target card's *intermediate* outputs (``ParticleAfterProp``,
``ChargedHadronEfficiency``, ``ECalTower``, …) into the same tree,
following the
:mod:`parnassus.torch_delphes.tune_cms_fullsim.debug` schema (branch
names like ``"Track.PT"``, ``"ECalTower.E"`` mirror the
:mod:`validation.validate_torch_delphes` convention).
:mod:`parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results`'s own
``--debug`` flag reads those branches back and overlays them with the
trainee's intermediate outputs at init / best.

Event count
-----------
The ``--n-events`` flag sets exactly how many events are generated.
Pythia generation is parallelised across ``--n-workers`` processes via
:class:`parnassus.pythia.HepMC3Generator`, which retries failed events so
the merged HepMC3 file always contains exactly ``--n-events`` events
regardless of per-event acceptance.

Usage
-----
.. code-block:: shell

    # Generate 20k events at pT-hat > 100 GeV using 32 parallel workers
    uv run python -m parnassus.torch_delphes.generate_pseudodata \
        --output src/parnassus/tests/benchmark_data/cms_pseudodata.root \
        --n-events 20000 \
        --n-workers 32 \
        --pt-hat-min 20

By default, the output lives in ``src/parnassus/tests/benchmark_data/``
so it can be committed and consumed by
``test_tune_cms_fullsim_real_fit``.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import shutil
import tempfile
from pathlib import Path

import awkward as ak
import joblib
import numpy as np
import pyhepmc
import torch
import uproot
from tqdm import tqdm

from parnassus.data.particle_io import ColumnMap, hepmc_particles_to_tensor
from parnassus.pythia import HepMC3Generator
from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.tune_cms_fullsim.data import load_truth_events
from parnassus.torch_delphes.tune_cms_fullsim.debug import (
    INTERMEDIATE_BRANCHES,
    debug_branch_name,
    tensor_to_per_event_arrays,
)
from parnassus.utils import pid_to_class
from parnassus.utils.logger import is_terminal

# The "target" (fake full-sim) card's parameters come from a declarative param
# config (see parnassus.torch_delphes.param_config) laid over the card defaults,
# so a config only needs to list the scalars it perturbs. The shipped default is
# the pure card default (every value = CMS card constant). Point --param-config
# at a different (possibly partial) file to change which knobs the generated
# sample perturbs.
_DEFAULT_PARAM_CONFIG: Path = (
    Path(__file__).resolve().parent / "param_configs" / "cms_target_default.yaml"
)

# --process name -> shipped Pythia8 .cmnd consumed by HepMC3Generator. The random
# seed is injected per-worker by the generator, so these files only carry the
# process / beam definition. Only dijet has a hard-process pT-hat cut.
_PROCESS_DIR: Path = Path(__file__).resolve().parent / "processes"
PROCESS_CMND = {
    "dijet": "qcd_dijet.cmnd",
    "HZZ4l": "HZZ4l.cmnd",
    "muongun": "muon_gun.cmnd",  # J/psi + Z -> mu mu resonance gun (muon calibration)
    "electrongun": "electron_gun.cmnd",  # J/psi + Z -> e e (electron calibration)
    "ksgun": "kshort_gun.cmnd",  # K_S -> pi+ pi- (charged-hadron calibration)
    # Energy-sampled photon gun: the only sample that reaches the FORWARD ECal
    # (|eta| > 2.5), since a neutral particle needs no tracker and the other guns
    # stop at the tracking edge. Constrains the ECal scale/resolution params only.
    "photongun": "photon_gun.cmnd",
}
DIJET_PT_HAT_MIN = 20.0  # 100 -> 20 (2026-08-18): populate the calo thresholds (soft photons / NH) and broaden HT


def resolve_device(device=None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_effective_cmnd(
    base_cmnd: str | Path,
    pt_hat_min: float,
    dest_dir: str | Path,
) -> Path:
    """Copy ``base_cmnd`` and append a ``pTHatMin`` override.

    Pythia applies settings in order, so appending
    ``PhaseSpace:pTHatMin = <pt_hat_min>`` after the base file's contents makes
    the ``--pt-hat-min`` CLI flag always win over whatever the shipped process
    ``.cmnd`` declares. The effective file is written into ``dest_dir``
    (typically the run's scratch directory) and its path returned.

    Returns
    -------
    Path
        Path to the written effective ``.cmnd`` file.
    """
    base = Path(base_cmnd).read_text()
    override = (
        "\n! ---- Override injected by generate_pseudodata.py ----\n"
        f"PhaseSpace:pTHatMin = {pt_hat_min}\n"
    )
    dest = Path(dest_dir) / "effective.cmnd"
    dest.write_text(base + override)
    return dest


@contextlib.contextmanager
def _tqdm_joblib(tqdm_bar):
    """Make a :class:`joblib.Parallel` call drive a tqdm progress bar.

    Monkey-patches joblib's (private) ``BatchCompletionCallBack`` so each
    completed task advances ``tqdm_bar``, then restores the original in a
    ``finally``. If a future joblib renames that symbol this raises
    ``AttributeError`` at setup (fails loudly) rather than silently no-op'ing.

    The Pythia phase dispatches exactly one task per worker, so the bar ticks
    once per finished worker (use ``total=n_workers``). It is therefore coarse
    and back-loaded: workers get near-equal event counts and finish around the
    same time, so the bar sits near empty for most of the (long) generation and
    then snaps toward 100%. That is expected, not a hang.

    Yields
    ------
    tqdm.tqdm
        The same ``tqdm_bar`` passed in, now advanced by joblib task completions.
    """
    old_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallBack(old_callback):
        def __call__(self, *args, **kwargs):
            # HepMC3Generator.generate() runs a second (file-merge) Parallel
            # after the per-worker generation; clamp to total so those extra
            # task completions don't push the bar past the n_workers total.
            if tqdm_bar.total is None:
                tqdm_bar.update(self.batch_size)
            elif tqdm_bar.n < tqdm_bar.total:
                tqdm_bar.update(min(self.batch_size, tqdm_bar.total - tqdm_bar.n))
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_bar.close()


def generate_truth_events(
    cmnd_file: str | Path,
    n_events: int,
    n_workers: int,
    work_dir: str | Path,
    seed_offset: int = 0,
) -> Path:
    """Generate ``n_events`` Pythia events in parallel into one HepMC3 file.

    Thin wrapper around :class:`parnassus.pythia.HepMC3Generator`: it launches
    ``n_workers`` single-core Pythia8 jobs (worker ``i`` seeded with
    ``seed_offset + i``), retries failed events so exactly ``n_events`` are
    produced, and merges the per-worker outputs into a single HepMC3 file.

    Parameters
    ----------
    cmnd_file : str | Path
        Pythia ``.cmnd`` configuration (see :func:`build_effective_cmnd`).
    n_events : int
        Total number of events to generate across all workers.
    n_workers : int
        Number of parallel Pythia8 processes.
    work_dir : str | Path
        Scratch directory; the HepMC files and per-job logs are written under
        ``<work_dir>/hepmc`` and ``<work_dir>/hepmc_logs``.
    seed_offset : int
        Added to every worker's Pythia ``Random:seed`` so independent samples
        (e.g. SLURM array tasks) draw disjoint seed ranges.

    Returns
    -------
    Path
        Path to the merged HepMC3 file holding all ``n_events`` events.
    """
    work_dir = Path(work_dir)
    generator = HepMC3Generator(
        cmnd_file=str(cmnd_file),
        output_dir=str(work_dir / "hepmc"),
        log_dir=str(work_dir / "hepmc_logs"),
    )
    # Drive a coarse progress bar off joblib task completions (one task per
    # worker) and silence joblib's own verbose=100 output (verbose=0).
    with _tqdm_joblib(
        tqdm(
            total=n_workers,
            desc="[1/3] Pythia (CPU workers)",
            unit="worker",
            disable=not is_terminal,
        )
    ):
        return generator.generate(
            n_events=n_events,
            max_workers=n_workers,
            debug=False,
            verbose=0,
            seed_offset=seed_offset,
        )


def make_target_card(
    param_config: str | Path = _DEFAULT_PARAM_CONFIG,
    debug: bool = False,
    device: str | torch.device = "cpu",
) -> CMSEnergyFlowDefault:
    """Build a learnable CMS card initialized from a parameter config.

    Every scalar listed in ``param_config`` (see
    :mod:`parnassus.torch_delphes.param_config`) is set to its physical
    ``value``; the config may be PARTIAL -- unlisted parameters keep the card
    defaults (:func:`param_config.load_param_config_over_defaults`). The card
    is then frozen and put in eval mode. These values play the role of the
    ground-truth detector response that the tuning harness should recover.

    Parameters
    ----------
    param_config : str | Path
        Path to a (possibly partial) YAML parameter config. Defaults to the
        shipped ``param_configs/cms_target_default.yaml`` (= card defaults).
    debug : bool
        If True, build the card in debug mode so it returns every
        intermediate per-module tensor in addition to the final
        ``EFlowObject`` (see
        :class:`parnassus.torch_delphes.defaults.CMSEnergyFlowDefault`). This
        is what enables ``generate_pseudodata --debug`` to save the
        per-module breakdown into the ROOT file.
    device : str | torch.device
        Device to place the card on (e.g. ``"cuda"`` or ``"cpu"``).

    Returns
    -------
    CMSEnergyFlowDefault
        A frozen learnable card whose parameters match the config values.
    """
    card = CMSEnergyFlowDefault(debug=debug, learnable=True)
    cfg = pc.load_param_config_over_defaults(param_config, card)
    pc.apply_param_config(card, cfg)
    for p in card.parameters():
        p.requires_grad_(False)
    card.eval()
    card.to(device)
    return card


def eflow_to_class_arrays(
    eflow: torch.Tensor,
    event_numbers: torch.Tensor,
    n_events: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split an EFlowObject tensor into per-event jagged arrays.

    Filters out zero-pT Gumbel-ST ghost tracks and converts PID -> class
    via :func:`parnassus.utils.pid_to_class` (0=chad, 1=e, 2=mu,
    3=neutral hadron, 4=photon).

    Returns
    -------
    tuple
        Four per-event jagged lists: pt, eta, phi, class.
    """
    pt_np = eflow[:, ColumnMap.PT].cpu().numpy().astype(np.float32)
    eta_np = eflow[:, ColumnMap.ETA].cpu().numpy().astype(np.float32)
    phi_np = eflow[:, ColumnMap.PHI].cpu().numpy().astype(np.float32)
    pid_np = eflow[:, ColumnMap.PID].cpu().numpy().astype(np.int64)
    ev_np = event_numbers.cpu().numpy().astype(np.int64)

    keep = pt_np > 1e-6
    pt_np = pt_np[keep]
    eta_np = eta_np[keep]
    phi_np = phi_np[keep]
    pid_np = pid_np[keep]
    ev_np = ev_np[keep]

    # Vectorize pid_to_class via a small lookup over |PID| mapped to
    # {11, 13, 22} -> {1, 2, 4} with charged/neutral hadrons filled in
    # by sign-of-charge considerations via pid_to_class itself.
    cls_np = np.fromiter(
        (pid_to_class(int(p)) for p in pid_np),
        dtype=np.int32,
        count=pid_np.shape[0],
    )

    pt_list = [pt_np[ev_np == i] for i in range(n_events)]
    eta_list = [eta_np[ev_np == i] for i in range(n_events)]
    phi_list = [phi_np[ev_np == i] for i in range(n_events)]
    cls_list = [cls_np[ev_np == i] for i in range(n_events)]
    return pt_list, eta_list, phi_list, cls_list


def truth_tensor_to_class_arrays(
    truth: torch.Tensor,
    n_events: int,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
]:
    """Split a per-particle truth tensor into per-event jagged arrays.

    The tensor is the direct output of ``pythia_particles_to_tensor``,
    so it contains EVERY particle in the event (including intermediate
    / unstable ones). We filter to final-state particles
    (``STATUS == 1``) and drop neutrinos.

    Returns
    -------
    tuple
        Five per-event jagged lists: pt, eta, phi, class, pdgid. ``pdgid`` is
        the real PDG id (int64) carried through unchanged for the
        ``truth_pdgid`` ROOT branch, so the trainee can route particles by true
        species instead of the lossy class label.
    """
    status = truth[:, ColumnMap.STATUS].cpu().numpy().astype(np.int64)
    pid = truth[:, ColumnMap.PID].cpu().numpy().astype(np.int64)
    abs_pid = np.abs(pid)
    # Keep status==1 and exclude electron/mu/tau neutrinos.
    keep = (status == 1) & ~np.isin(abs_pid, [12, 14, 16])

    pt_np = truth[:, ColumnMap.PT].cpu().numpy().astype(np.float32)[keep]
    eta_np = truth[:, ColumnMap.ETA].cpu().numpy().astype(np.float32)[keep]
    phi_np = truth[:, ColumnMap.PHI].cpu().numpy().astype(np.float32)[keep]
    pid_np = pid[keep]
    ev_np = truth[:, ColumnMap.EVENT_NUMBER].cpu().numpy().astype(np.int64)[keep]

    # Drop any particle with |eta| beyond something absurd (Pythia emits
    # ±999.9 for pT=0 particles). Those confuse the calorimeter and our
    # binning, and they carry no physics information.
    finite_eta = np.abs(eta_np) < 10.0
    pt_np = pt_np[finite_eta]
    eta_np = eta_np[finite_eta]
    phi_np = phi_np[finite_eta]
    pid_np = pid_np[finite_eta]
    ev_np = ev_np[finite_eta]

    cls_np = np.fromiter(
        (pid_to_class(int(p)) for p in pid_np),
        dtype=np.int32,
        count=pid_np.shape[0],
    )

    pt_list = [pt_np[ev_np == i] for i in range(n_events)]
    eta_list = [eta_np[ev_np == i] for i in range(n_events)]
    phi_list = [phi_np[ev_np == i] for i in range(n_events)]
    cls_list = [cls_np[ev_np == i] for i in range(n_events)]
    # Real PDG id (int64) carried through unchanged for the ``truth_pdgid``
    # ROOT branch so the trainee routes by true species, not the lossy class.
    pid_list = [pid_np[ev_np == i].astype(np.int64) for i in range(n_events)]
    return pt_list, eta_list, phi_list, cls_list, pid_list


def hepmc_to_truth_class_arrays(
    hepmc_path: str | Path,
    n_events: int | None = None,
) -> dict[str, list[np.ndarray]]:
    """Read a merged HepMC3 file into per-event class-based ``truth_*`` arrays.

    Each event's final-state particles (HepMC ``status == 1``) are converted to
    a ``(N, N_FEATURES)`` tensor via
    :func:`parnassus.data.particle_io.hepmc_particles_to_tensor` and reduced to
    the ``(pt, eta, phi, class, pdgid)`` representation written to the ROOT file
    -- what the trainee gets to see. Pre-filtering to ``status ==
    1`` before tensor construction skips the (large) intermediate shower
    history, which is what :func:`truth_tensor_to_class_arrays` would discard
    anyway.

    Parameters
    ----------
    hepmc_path : str | Path
        Path to the merged HepMC3 file produced by :func:`generate_truth_events`.
    n_events : int | None
        If given, stop after reading this many events (a safety cap matching the
        number requested from Pythia). ``None`` reads the whole file.

    Returns
    -------
    dict
        ``{"truth_pt", "truth_eta", "truth_phi", "truth_class", "truth_pdgid"}``
        -> list of per-event numpy arrays, one entry per event (empty events
        kept as zero-length arrays so the per-event alignment is preserved).
    """
    truth_pt: list[np.ndarray] = []
    truth_eta: list[np.ndarray] = []
    truth_phi: list[np.ndarray] = []
    truth_class: list[np.ndarray] = []
    truth_pdgid: list[np.ndarray] = []

    with pyhepmc.open(str(hepmc_path), "r") as reader:
        for event_idx, event in enumerate(
            tqdm(
                reader,
                total=n_events,
                desc="[2/3] HepMC -> truth",
                unit="evt",
                disable=not is_terminal,
            )
        ):
            if n_events is not None and event_idx >= n_events:
                break
            # Keep only final-state particles; event_number=0 so the single
            # event splits trivially under truth_tensor_to_class_arrays below.
            final_particles = [p for p in event.particles if p.status == 1]
            truth = hepmc_particles_to_tensor(final_particles, 0, dtype=torch.float64)
            pts, etas, phis, clss, pids = truth_tensor_to_class_arrays(truth, n_events=1)
            truth_pt.append(pts[0])
            truth_eta.append(etas[0])
            truth_phi.append(phis[0])
            truth_class.append(clss[0])
            truth_pdgid.append(pids[0])

    return {
        "truth_pt": truth_pt,
        "truth_eta": truth_eta,
        "truth_phi": truth_phi,
        "truth_class": truth_class,
        "truth_pdgid": truth_pdgid,
    }


def truth_arrays_to_pflow(
    truth_arrays: dict[str, list[np.ndarray]],
    target_card: CMSEnergyFlowDefault,
    batch_size: int = 512,
    debug: bool = False,
    device: str | torch.device = "cpu",
) -> dict[str, list[np.ndarray]]:
    """Run the class-based truth events through the target card in batches.

    The trainee only ever sees the ``truth_*`` branches, so we feed the target
    card the SAME class-based reconstruction the trainee will rebuild from those
    branches via :func:`load_truth_events` (representative PID, pi-mass, derived
    charge, zero vertex, recomputed E). This keeps generation-input identical to
    trainee-input, so the fit can recover the truth parameters; otherwise the
    full-Pythia species (real PID/mass) make the target unreachable from the
    class-only truth. See doc ``truth_input_fidelity_issue.md``.

    Events are processed in slices of ``batch_size`` for memory; per-event
    alignment with ``truth_arrays`` is preserved because each slice is split
    back out with ``n_events`` equal to the slice length (empty events yield
    empty per-event arrays on both sides).

    When ``debug`` is True, the target card is expected to have been built with
    ``debug=True`` so its forward pass returns every intermediate per-module
    tensor in addition to ``EFlowObject``; each is split into per-event jagged
    arrays under ``"<ModuleName>.<Var>"`` branch names (see
    :func:`parnassus.torch_delphes.tune_cms_fullsim.debug.debug_branch_name`).

    Parameters
    ----------
    device : str | torch.device
        Device the ``target_card`` lives on; each batch's input tensor is moved
        here before the forward pass. Outputs are moved back to CPU when split
        into per-event arrays.

    Returns
    -------
    dict
        ``{"pflow_pt", "pflow_eta", "pflow_phi", "pflow_class"}`` (plus the
        ``"<ModuleName>.<Var>"`` debug branches when ``debug`` is True) -> list
        of per-event numpy arrays, index-aligned with ``truth_arrays``.
    """
    n_events = len(truth_arrays["truth_pt"])

    branches: dict[str, list[np.ndarray]] = {
        "pflow_pt": [],
        "pflow_eta": [],
        "pflow_phi": [],
        "pflow_class": [],
    }
    if debug:
        for module_name, variables in INTERMEDIATE_BRANCHES:
            for var in variables:
                branches[debug_branch_name(module_name, var)] = []

    for start in tqdm(
        range(0, n_events, batch_size),
        desc="[3/3] TorchDelphes",
        unit="batch",
        disable=not is_terminal,
    ):
        end = min(start + batch_size, n_events)
        n_batch = end - start
        batch_truth = {k: v[start:end] for k, v in truth_arrays.items()}

        truth_tensor = load_truth_events(batch_truth)
        # Flatten back to (N, N_FEATURES) and drop the zero-padding rows.
        reco_input = truth_tensor[torch.any(truth_tensor != 0, dim=-1)]
        if reco_input.numel() == 0:
            # Whole slice was empty events: emit empty per-event arrays so the
            # output stays index-aligned with the truth side.
            for key in branches:
                branches[key].extend([np.empty(0, dtype=np.float32) for _ in range(n_batch)])
            continue

        with torch.no_grad():
            out = target_card(reco_input.to(device))
        eflow = out["EFlowObject"]

        pflow_pt, pflow_eta, pflow_phi, pflow_class = eflow_to_class_arrays(
            eflow, eflow[:, ColumnMap.EVENT_NUMBER], n_events=n_batch
        )
        branches["pflow_pt"].extend(pflow_pt)
        branches["pflow_eta"].extend(pflow_eta)
        branches["pflow_phi"].extend(pflow_phi)
        branches["pflow_class"].extend(pflow_class)

        # In debug mode, every key in INTERMEDIATE_BRANCHES is expected to be
        # present in the card's output dict (the target card was built with
        # ``debug=True``); split each per-module tensor into per-event jagged
        # arrays for ROOT writing. Branch names follow the
        # ``"<ModuleName>.<Var>"`` convention used by validate_torch_delphes,
        # which uproot accepts verbatim.
        if debug:
            for module_name, variables in INTERMEDIATE_BRANCHES:
                tensor = out.get(module_name)
                if tensor is None:
                    # The target card didn't return this module's output. This
                    # shouldn't happen with the CMS card in debug=True mode, but
                    # we skip rather than crash to keep the script forward-
                    # compatible with cards that expose fewer intermediates.
                    continue
                per_var = tensor_to_per_event_arrays(
                    tensor, module_name, variables, n_events=n_batch
                )
                for var, lists in per_var.items():
                    branches[debug_branch_name(module_name, var)].extend(lists)

    return branches


def generate(
    output_path: Path,
    n_events: int = 20_000,
    process: str = "dijet",
    pt_hat_min: float | None = None,
    n_workers: int | None = None,
    batch_size: int = 512,
    seed: int = 1,
    param_config: str | Path = _DEFAULT_PARAM_CONFIG,
    debug: bool = False,
    work_dir: str | Path | None = None,
    keep_hepmc: bool = False,
    device: str | torch.device | None = None,
) -> int:
    """Generate the pseudodataset and write it to ``output_path``.

    Runs the two-phase pipeline: (1) generate ALL ``n_events`` Pythia events up
    front and in parallel into one HepMC3 file, then (2) stream those events
    through the perturbed TorchDelphes target card and write the cms-flow-schema
    ROOT file.

    Parameters
    ----------
    n_events : int
        Exact number of events to generate.
    process : str
        Key into :data:`PROCESS_CMND` selecting the shipped Pythia ``.cmnd``
        (``"dijet"``, ``"HZZ4l"``, ``"muongun"``, ``"electrongun"``, ``"ksgun"``
        or ``"photongun"``).
    pt_hat_min : float | None
        ``PhaseSpace:pTHatMin`` override appended to the process ``.cmnd``.
        ``None`` means no override, except for ``dijet`` which always runs
        with :data:`DIJET_PT_HAT_MIN`.
    n_workers : int | None
        Number of parallel Pythia8 processes. ``None`` defaults to
        ``os.cpu_count()`` (capped at ``n_events`` so no worker gets zero
        events).
    batch_size : int
        Number of events per TorchDelphes forward pass in phase 2.
    seed : int
        Seeds BOTH phases: Pythia workers use ``Random:seed`` in
        ``seed*n_workers+1 .. (seed+1)*n_workers`` (disjoint across seeds at a
        fixed ``n_workers``), and torch is seeded for the card's smearing.
    param_config : str | Path
        Generation/truth param config; may be PARTIAL (see
        :func:`make_target_card`).
    work_dir : str | Path | None
        Scratch directory for the intermediate HepMC files / logs. ``None``
        creates (and, unless ``keep_hepmc``, removes) a temporary directory.
    keep_hepmc : bool
        If True, do not delete the intermediate HepMC files/logs when a
        temporary ``work_dir`` was created. No effect when ``work_dir`` is given.
    device : str | torch.device | None
        Device for the phase-2 TorchDelphes forward pass. ``None`` auto-detects
        (``"cuda"`` if a GPU is available, else ``"cpu"``). Phase-1 Pythia
        generation is always CPU-parallel and unaffected.
    debug : bool
        If True, run the target card in debug mode and additionally write
        every intermediate per-module output into the ROOT file under branch
        names ``"<ModuleName>.<Var>"`` (see
        :mod:`parnassus.torch_delphes.tune_cms_fullsim.debug`). The file
        size will grow considerably (one branch per kinematic variable per
        intermediate module, ~150 extra branches), so leave this off when
        generating samples for plain training.

    Returns
    -------
    int
        Number of events actually written to the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_device = resolve_device(device)

    # Resolve worker count: default to all cores, never more than n_events
    # (HepMC3Generator would otherwise hand some workers zero events).
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, n_events))

    # Resolve scratch dir. A user-provided dir is never auto-deleted.
    cleanup = False
    if work_dir is None:
        work_dir = Path(tempfile.mkdtemp(prefix="generate_pseudodata_"))
        cleanup = not keep_hepmc
    else:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ----- Phase 1: generate ALL Pythia events up front, in parallel. -----
        cmnd = _PROCESS_DIR / PROCESS_CMND[process]
        if process == "dijet" and pt_hat_min is None:
            pt_hat_min = DIJET_PT_HAT_MIN  # dijet always runs with an explicit pTHatMin
        if pt_hat_min is not None:
            cmnd = build_effective_cmnd(cmnd, pt_hat_min, work_dir)
        print(
            f"[1/3] Generating {n_events} Pythia {process} events on {n_workers} CPU "
            f"worker(s) (pTHatMin={pt_hat_min})..."
        )
        # Pythia seeds seed*n_workers+1 .. (seed+1)*n_workers: disjoint across
        # --seed values (at fixed n_workers), so array tasks give distinct events.
        hepmc_path = generate_truth_events(
            cmnd, n_events, n_workers, work_dir, seed_offset=seed * n_workers
        )

        # ----- Phase 2a: read HepMC back into class-based truth arrays. -----
        print(f"[2/3] Reading {hepmc_path.name} -> truth particles...")
        truth_arrays = hepmc_to_truth_class_arrays(hepmc_path, n_events=n_events)
        n_read = len(truth_arrays["truth_pt"])

        # ----- Phase 2b: pass truth events through TorchDelphes. -----
        print(
            f"[3/3] Passing {n_read} events through TorchDelphes (target card) "
            f"on {resolved_device}..."
        )
        target_card = make_target_card(
            param_config=param_config, debug=debug, device=resolved_device
        )
        torch.manual_seed(seed)  # the card's smearing / Gumbel-ST is stochastic
        pflow_arrays = truth_arrays_to_pflow(
            truth_arrays,
            target_card,
            batch_size=batch_size,
            debug=debug,
            device=resolved_device,
        )

        # ----- Write the cms-flow-schema ROOT file. -----
        all_branches = {**truth_arrays, **pflow_arrays}
        with uproot.recreate(str(output_path)) as f:
            f["event_tree"] = {k: ak.Array(v) for k, v in all_branches.items()}
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  wrote {n_read} events ({size_mb:.2f} MB) to {output_path}")

        return n_read
    finally:
        if cleanup:
            shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    """Entry point for ``python -m parnassus.torch_delphes.generate_pseudodata``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/parnassus/tests/benchmark_data/cms_pseudodata.root"),
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=20_000,
        help="Exact number of events to generate.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=(
            "Number of parallel Pythia8 processes for event generation. "
            "Defaults to all available CPU cores (capped at --n-events)."
        ),
    )
    parser.add_argument(
        "--process",
        choices=PROCESS_CMND,
        default="dijet",
        help="Physics process; selects the shipped processes/<name>.cmnd (default: dijet).",
    )
    parser.add_argument(
        "--pt-hat-min",
        type=float,
        default=None,
        help=(
            "Append a PhaseSpace:pTHatMin override to the process .cmnd. Default: "
            f"{DIJET_PT_HAT_MIN} for dijet (always applied), no override otherwise."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of events per TorchDelphes forward pass.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help=(
            "Seeds both phases: Pythia workers get Random:seed in "
            "seed*n_workers+1 .. (seed+1)*n_workers (disjoint across seeds at fixed "
            "--n-workers, so SLURM array tasks produce distinct events) and torch "
            "is seeded for the target card's stochastic smearing."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device for the TorchDelphes forward pass (e.g. 'cuda', 'cuda:0', "
            "'cpu'). Defaults to auto-detect: 'cuda' if a GPU is available, "
            "else 'cpu'. Pythia generation is always CPU-parallel."
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help=(
            "Scratch directory for the intermediate HepMC files / per-job logs. "
            "Defaults to a temporary directory that is removed on completion "
            "(unless --keep-hepmc is given)."
        ),
    )
    parser.add_argument(
        "--keep-hepmc",
        action="store_true",
        help=(
            "Keep the intermediate HepMC files / logs instead of deleting the "
            "auto-created temporary work directory. No effect with --work-dir."
        ),
    )
    parser.add_argument(
        "--param-config",
        type=Path,
        default=_DEFAULT_PARAM_CONFIG,
        help=(
            "Path to a YAML parameter config whose physical 'value' fields define "
            "the ground-truth detector response written into the pflow_* branches. "
            "May be PARTIAL: only the listed scalars are changed, everything else "
            "keeps the card default. Defaults to the shipped "
            "param_configs/cms_target_default.yaml (= card defaults)."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Also write every intermediate per-module output of the target card "
            "into the ROOT file under '<ModuleName>.<Var>' branches "
            "(ParticleAfterProp, ChargedHadronEfficiency, ECalTower, ...). "
            "Mirrors the --debug branch list of "
            "validation/validate_torch_delphes.py and is what enables "
            "tune_cms_fullsim/plot_fit_results.py --debug. Increases the file "
            "size considerably (~150 extra branches), so leave off for plain "
            "training runs."
        ),
    )
    args = parser.parse_args()

    resolved_device = resolve_device(args.device)
    print(
        f"Generating pseudodata -> {args.output}\n"
        f"  n_events={args.n_events}  cpu-workers={args.n_workers or 'auto'} (Pythia)  "
        f"process={args.process}  pTHatMin={args.pt_hat_min}  seed={args.seed}\n"
        f"  device={resolved_device} (TorchDelphes phase-2 only; Pythia always CPU)  "
        f"param-config={args.param_config}  debug={args.debug}"
    )
    n = generate(
        args.output,
        n_events=args.n_events,
        process=args.process,
        pt_hat_min=args.pt_hat_min,
        n_workers=args.n_workers,
        batch_size=args.batch_size,
        seed=args.seed,
        param_config=args.param_config,
        debug=args.debug,
        work_dir=args.work_dir,
        keep_hepmc=args.keep_hepmc,
        device=resolved_device,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {n} events, {size_mb:.2f} MB, to {args.output}")


if __name__ == "__main__":
    main()
