"""Generate CMS-full-sim-like pseudodata for the differentiable tuning harness.

This script produces a ROOT file in the cms-flow ``event_tree`` schema
(the same schema as Zenodo record 11389651) so that
``tune_cms_fullsim.py`` can be run end-to-end on reproducible input
without needing network access.

Event generation
----------------
1. Pythia8 generates hard QCD dijet events at ``sqrt(s) = 13 TeV`` with
   ``pTHatMin`` set by ``--pt-hat-min``. Hadronization is on; final-state
   radiation is on. This produces a realistic mix of charged hadrons,
   electrons, muons, photons, K-shorts, Lambdas, and neutrons -- the
   exact particle zoo that exercises every learnable parameter in
   ``CMSEnergyFlowDefault``.

2. Each event's stable particles are converted to the
   ``(N, N_FEATURES)`` particle tensor via
   :func:`parnassus.data.particle_io.pythia_particles_to_tensor`,
   with ``EVENT_NUMBER`` set to the (0-indexed) global event number so
   the calorimeter's per-event aggregation works correctly.

3. The stable particles are run through a
   :class:`CMSEnergyFlowDefault` card to produce a CMS-PF-like reco
   object collection. Two cards are used:

   - A **default** (non-learnable) card which gives us the truth
     particles themselves as the ``truth_*`` branches.
   - A **learnable card initialized from a parameter config** which
     plays the role of "CMS full simulation" for the fitting target.
     The config (``--param-config``, see
     :mod:`parnassus.torch_delphes.param_config`) sets every one of the
     66 learnable parameters to a physical ground-truth value; the
     shipped default perturbs the three energy/momentum scales (CHAD
     1.25, ECal 1.20, HCal 0.90). These values are exactly what Adam
     should recover when we fit the trainee against the ``pflow_*``
     branches.

Output schema
-------------
The output ROOT file has a TTree named ``event_tree`` with jagged
per-event branches:

- ``truth_pt, truth_eta, truth_phi, truth_class``
- ``pflow_pt, pflow_eta, pflow_phi, pflow_class``

Class values follow :func:`parnassus.utils.pid_to_class` (0=charged
hadron, 1=electron, 2=muon, 3=neutral hadron, 4=photon), matching the
cms-flow loader.

Size targeting
--------------
The ``--target-size-mb`` flag controls how many events are generated.
The script measures the current file size after every 200 events and
stops once the target size is reached, so the output is always at
roughly the requested size regardless of event multiplicity.

Usage
-----
.. code-block:: shell

    # Generate ~20 MB of pseudodata at pT-hat > 100 GeV
    uv run python -m parnassus.torch_delphes.generate_pseudodata \
        --output src/parnassus/tests/benchmark_data/cms_pseudodata.root \
        --target-size-mb 20 \
        --pt-hat-min 100

By default, the output lives in ``src/parnassus/tests/benchmark_data/``
so it can be committed and consumed by
``test_tune_cms_fullsim_real_fit``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import awkward as ak
import numpy as np
import pythia8mc
import torch
import uproot

from parnassus.data.particle_io import ColumnMap, pythia_particles_to_tensor
from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.tune_cms_fullsim.data import load_truth_events
from parnassus.utils import pid_to_class

# The "target" (fake full-sim) card's parameters are taken verbatim from a
# declarative parameter config (see parnassus.torch_delphes.param_config). The
# shipped default sets the three energy/momentum scales to reachable
# ground-truth values (CHAD 1.25, ECal 1.20, HCal 0.90) and leaves everything
# else at its model default. Point --param-config at a different file to change
# which knobs the generated sample perturbs.
_DEFAULT_PARAM_CONFIG: Path = (
    Path(__file__).resolve().parent / "param_configs" / "cms_target_default.yaml"
)


def build_pythia(pt_hat_min: float, seed: int) -> object:
    """Configure and initialise a Pythia8 instance for QCD dijets.

    Returns
    -------
    pythia8mc.Pythia
        An initialised Pythia8 instance ready for :meth:`next` calls.
    """
    pythia = pythia8mc.Pythia()
    settings = [
        "Beams:eCM = 13000.",
        "HardQCD:all = on",
        f"PhaseSpace:pTHatMin = {pt_hat_min}",
        f"Random:seed = {seed}",
        "Random:setSeed = on",
        "Next:numberShowInfo = 0",
        "Next:numberShowProcess = 0",
        "Next:numberShowEvent = 0",
        "Init:showProcesses = off",
        "Init:showMultipartonInteractions = off",
        "Init:showChangedSettings = off",
        "Init:showChangedParticleData = off",
        "Stat:showProcessLevel = off",
        "Stat:showPartonLevel = off",
        "Stat:showErrors = off",
    ]
    for setting in settings:
        pythia.readString(setting)
    pythia.init()
    return pythia


def make_target_card(
    param_config: str | Path = _DEFAULT_PARAM_CONFIG,
) -> CMSEnergyFlowDefault:
    """Build a learnable CMS card initialized from a parameter config.

    Every one of the card's 66 learnable parameters is set to the physical
    ``value`` of the matching entry in ``param_config`` (see
    :mod:`parnassus.torch_delphes.param_config`), then the card is frozen and
    put in eval mode. These values play the role of the ground-truth detector
    response that the tuning harness should recover.

    Parameters
    ----------
    param_config : str | Path
        Path to a YAML parameter config. Defaults to the shipped
        ``param_configs/cms_target_default.yaml``.

    Returns
    -------
    CMSEnergyFlowDefault
        A frozen learnable card whose parameters match the config values.
    """
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    cfg = pc.load_param_config(param_config)
    pc.apply_param_config(card, cfg)
    for p in card.parameters():
        p.requires_grad_(False)
    card.eval()
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
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Split a per-particle truth tensor into per-event jagged arrays.

    The tensor is the direct output of ``pythia_particles_to_tensor``,
    so it contains EVERY particle in the event (including intermediate
    / unstable ones). We filter to final-state particles
    (``STATUS == 1``) and drop neutrinos.

    Returns
    -------
    tuple
        Four per-event jagged lists: pt, eta, phi, class.
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
    return pt_list, eta_list, phi_list, cls_list


def _flush_batch(
    pythia: object,
    target_card: CMSEnergyFlowDefault,
    start_idx: int,
    batch_size: int,
) -> tuple[dict[str, list[np.ndarray]], int]:
    """Generate one batch of events and return cms-flow-schema lists.

    Runs ``batch_size`` events through Pythia, converts them to a truth
    tensor, passes them through the perturbed target card, and splits
    both sides into per-event jagged arrays.

    Returns
    -------
    branches : dict
        Mapping from branch name (e.g. ``"truth_pt"``) to a list of
        per-event numpy arrays for the batch.
    n_generated : int
        Actual number of events generated (may be smaller than
        ``batch_size`` if Pythia skipped events).
    """
    del start_idx  # event numbers are shifted to [0, n_generated) locally
    batch_truth_rows: list[torch.Tensor] = []
    n_generated = 0
    for _ in range(batch_size):
        if not pythia.next():  # type: ignore[attr-defined]
            continue
        # Use a *local* event number in [0, n_generated) so the helpers
        # below can split jagged arrays with a straightforward
        # ``ev == i`` comparison. The absolute event index across
        # batches is tracked by the caller.
        row = pythia_particles_to_tensor(pythia.event, n_generated, dtype=torch.float64)  # type: ignore[attr-defined]
        batch_truth_rows.append(row)
        n_generated += 1

    if n_generated == 0:
        return {}, 0

    # Flatten all events in the batch into a single tensor.
    batch_truth = torch.cat(batch_truth_rows, dim=0)

    # The truth_* branches (status==1, non-neutrino, |eta|<10, PID->class) are
    # the ONLY thing the trainee gets to see. Build them first, then feed the
    # target card the SAME class-based reconstruction the trainee will rebuild
    # from those branches via load_truth_events. This keeps generation-input
    # identical to trainee-input (representative PID, pi-mass, derived charge,
    # zero vertex, recomputed E), so the fit can recover the truth parameters;
    # otherwise the full-Pythia species (real PID/mass) make the target
    # unreachable from the class-only truth. See
    # doc truth_input_fidelity_issue.md.
    truth_pt, truth_eta, truth_phi, truth_class = truth_tensor_to_class_arrays(
        batch_truth, n_events=n_generated
    )
    truth_tensor = load_truth_events(
        {
            "truth_pt": truth_pt,
            "truth_eta": truth_eta,
            "truth_phi": truth_phi,
            "truth_class": truth_class,
        }
    )  # padded (n_events, max_n_particles, N_FEATURES)
    # Flatten back to (N, N_FEATURES) and drop the zero-padding rows.
    reco_input = truth_tensor[torch.any(truth_tensor != 0, dim=-1)]

    with torch.no_grad():
        out = target_card(reco_input)
    eflow = out["EFlowObject"]

    pflow_pt, pflow_eta, pflow_phi, pflow_class = eflow_to_class_arrays(
        eflow, eflow[:, ColumnMap.EVENT_NUMBER], n_events=n_generated
    )

    return (
        {
            "truth_pt": truth_pt,
            "truth_eta": truth_eta,
            "truth_phi": truth_phi,
            "truth_class": truth_class,
            "pflow_pt": pflow_pt,
            "pflow_eta": pflow_eta,
            "pflow_phi": pflow_phi,
            "pflow_class": pflow_class,
        },
        n_generated,
    )


def generate(
    output_path: Path,
    target_size_mb: float = 20.0,
    pt_hat_min: float = 100.0,
    batch_size: int = 200,
    max_events: int = 100_000,
    seed: int = 1,
    param_config: str | Path = _DEFAULT_PARAM_CONFIG,
) -> int:
    """Generate the pseudodataset and write it to ``output_path``.

    Returns
    -------
    int
        Number of events actually written to the output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pythia = build_pythia(pt_hat_min=pt_hat_min, seed=seed)
    target_card = make_target_card(param_config=param_config)

    accumulated: dict[str, list[np.ndarray]] = {
        b: []
        for b in (
            "truth_pt",
            "truth_eta",
            "truth_phi",
            "truth_class",
            "pflow_pt",
            "pflow_eta",
            "pflow_phi",
            "pflow_class",
        )
    }

    total_written = 0
    while total_written < max_events:
        batch, n_gen = _flush_batch(pythia, target_card, total_written, batch_size)
        if n_gen == 0:
            break
        for key, val in batch.items():
            accumulated[key].extend(val)
        total_written += n_gen

        # Write a checkpoint so we can measure the file size on disk.
        with uproot.recreate(str(output_path)) as f:
            f["event_tree"] = {k: ak.Array(v) for k, v in accumulated.items()}
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(
            f"  generated {total_written} events, "
            f"file size = {size_mb:.2f} MB "
            f"(target {target_size_mb:.1f} MB)"
        )
        if size_mb >= target_size_mb:
            break

    return total_written


def main() -> None:
    """Entry point for ``python -m parnassus.torch_delphes.generate_pseudodata``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/parnassus/tests/benchmark_data/cms_pseudodata.root"),
    )
    parser.add_argument("--target-size-mb", type=float, default=20.0)
    parser.add_argument("--pt-hat-min", type=float, default=100.0)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--max-events", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--param-config",
        type=Path,
        default=_DEFAULT_PARAM_CONFIG,
        help=(
            "Path to a YAML parameter config whose physical 'value' fields define "
            "the ground-truth detector response written into the pflow_* branches. "
            "Defaults to the shipped param_configs/cms_target_default.yaml."
        ),
    )
    args = parser.parse_args()

    print(
        f"Generating pseudodata -> {args.output}\n"
        f"  pTHatMin={args.pt_hat_min}  target={args.target_size_mb} MB  seed={args.seed}\n"
        f"  param-config={args.param_config}"
    )
    n = generate(
        args.output,
        target_size_mb=args.target_size_mb,
        pt_hat_min=args.pt_hat_min,
        batch_size=args.batch_size,
        max_events=args.max_events,
        seed=args.seed,
        param_config=args.param_config,
    )
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {n} events, {size_mb:.2f} MB, to {args.output}")


if __name__ == "__main__":
    main()
