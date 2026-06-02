"""ROOT I/O and observable construction for ``tune_cms_fullsim``.

This module turns a cms-flow-format ROOT file into the two things the fit
loop needs:

- the ``(N_particles, N_FEATURES)`` truth particle tensor that is fed to the
  trainee card (:func:`load_cms_flow_root`, :func:`truth_to_particle_tensor`);
- the per-observable target / prediction dictionaries used by the loss
  (:func:`pflow_target_observables` from the frozen ``pflow_*`` reco, and
  :func:`trainee_observables` from the trainee card's ``EFlowObject`` output).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import uproot

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.utils import class_to_pid_vectorized

from .config import PFLOW_BRANCHES, TRUTH_BRANCHES

# =============================================================================
# ROOT I/O
# =============================================================================


def load_cms_flow_root(
    path: Path,
    n_events: int,
    tree_name: str = "event_tree",
    entry_start: int = 0,
) -> dict[str, np.ndarray]:
    """Load up to ``n_events`` events from a cms-flow-format ROOT file.

    Reads the contiguous event range ``[entry_start, entry_start + n_events)``.
    This is used by the DDP code path to give each rank a disjoint shard
    of the file without re-reading the whole tree on every rank.
    """
    with uproot.open(str(path)) as f:
        if tree_name not in f:
            raise KeyError(
                f"ROOT file {path} has no tree named {tree_name!r}. "
                f"Available keys: {list(f.keys())[:10]}"
            )
        tree = f[tree_name]
        arrays = tree.arrays(
            list(TRUTH_BRANCHES + PFLOW_BRANCHES),
            library="np",
            entry_start=entry_start,
            entry_stop=entry_start + n_events,
        )
    return arrays


def truth_to_particle_tensor(
    arrays: dict[str, np.ndarray],
    n_events: int,
) -> torch.Tensor:
    """Convert per-event truth jagged arrays into a flat particle tensor.

    Each truth particle becomes one row of an ``(N_total, N_FEATURES)``
    tensor suitable to be fed directly into
    :class:`CMSEnergyFlowDefault.forward`. ``EVENT_NUMBER`` is set to the
    (0-indexed) event so that the calorimeter's per-event tower
    aggregation treats events independently.

    Particle class is mapped to PDG via
    :func:`parnassus.utils.class_to_pid_vectorized`. Electrons / muons
    are given small masses from the PDG table; everything else gets the
    charged-pion mass (0.14 GeV), which matches what the Parnassus
    pipeline does for minimum-bias-like particles.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(N_total, N_FEATURES)`` ready for
        :class:`CMSEnergyFlowDefault.forward`.
    """
    rows_list: list[np.ndarray] = []
    for i in range(n_events):
        pt = np.asarray(arrays["truth_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["truth_eta"][i], dtype=np.float64)
        phi = np.asarray(arrays["truth_phi"][i], dtype=np.float64)
        cls = np.asarray(arrays["truth_class"][i], dtype=np.int64)
        n_p = pt.shape[0]
        if n_p == 0:
            continue
        pids = class_to_pid_vectorized(cls)
        # Rough masses: electrons/muons use their PDG masses, everything
        # else uses the charged-pion mass as a stand-in.
        abs_pid = np.abs(pids)
        mass = np.where(abs_pid == 11, 0.000511, np.where(abs_pid == 13, 0.10566, 0.13957)).astype(
            np.float64
        )
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
        charge = np.where(
            abs_pid == 211,
            1.0,
            np.where(
                abs_pid == 11,
                -1.0,
                np.where(abs_pid == 13, -1.0, 0.0),
            ),
        ).astype(np.float64)
        # Sign the charge based on the sign of the underlying PDG (class->PDG
        # returns positive PIDs, so we conservatively keep |charge| = {0, 1}).
        row = np.zeros((n_p, N_FEATURES), dtype=np.float64)
        row[:, ColumnMap.PID] = pids
        row[:, ColumnMap.STATUS] = 1
        row[:, ColumnMap.CHARGE] = charge
        row[:, ColumnMap.E] = e
        row[:, ColumnMap.PX] = px
        row[:, ColumnMap.PY] = py
        row[:, ColumnMap.PZ] = pz
        row[:, ColumnMap.PT] = pt
        row[:, ColumnMap.ETA] = eta
        row[:, ColumnMap.PHI] = phi
        row[:, ColumnMap.MASS] = mass
        row[:, ColumnMap.EVENT_NUMBER] = i
        rows_list.append(row)
    if not rows_list:
        return torch.zeros((0, N_FEATURES), dtype=torch.float64)
    return torch.from_numpy(np.concatenate(rows_list, axis=0))


# =============================================================================
# Observables
# =============================================================================


def pflow_target_observables(
    arrays: dict[str, np.ndarray],
    n_events: int,
    log_pt_floor: float = -1,
) -> dict[str, torch.Tensor]:
    """Build target observable tensors from the ``pflow_*`` branches.

    These are the fitting target. Because they come straight from the
    full-simulation CMS PF reco in the ROOT file, they do not depend on
    any learnable parameters and are computed once per call.

    The ROOT schema does not store ``pflow_e``, so the per-particle
    energy is reconstructed analytically from ``(pt, eta, phi, class)``
    using the same PDG-mass assignment as
    :func:`truth_to_particle_tensor` (electrons / muons get their PDG
    masses, everything else uses the charged-pion mass). This is the
    same convention the trainee card applies internally, so the two
    energy distributions are directly comparable.

    Parameters
    ----------
    log_pt_floor : float
        ``pflow_pt`` values below this threshold are clipped before
        taking the logarithm. Real PF objects sit well above 0.1 GeV,
        so this only affects pathological zero entries.

    Returns
    -------
    dict[str, torch.Tensor]
        Per-particle 1-D tensors: ``"pt"``, ``"eta"``, ``"phi"``,
        ``"E"``, ``"log_pt"``. Per-event tensors of length
        ``n_events``: ``"multiplicity"`` and ``"ht"`` (scalar-pT sum).
    """
    all_pt: list[np.ndarray] = []
    all_eta: list[np.ndarray] = []
    # all_phi: list[np.ndarray] = [] # phi doesn't contribute to gradient
    all_e: list[np.ndarray] = []
    per_event_mult = np.zeros(n_events, dtype=np.float64)
    per_event_ht = np.zeros(n_events, dtype=np.float64)
    for i in range(n_events):
        pt = np.asarray(arrays["pflow_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["pflow_eta"][i], dtype=np.float64)
        # phi = np.asarray(arrays["pflow_phi"][i], dtype=np.float64)
        cls = np.asarray(arrays["pflow_class"][i], dtype=np.int64)
        # Reconstruct per-particle energy: E = sqrt(p^2 + m^2) with
        # |p| = pt * cosh(eta) and m from the class -> PDG -> mass map.
        pids = class_to_pid_vectorized(cls) if pt.size else np.empty(0, dtype=np.int64)
        abs_pid = np.abs(pids)
        mass = np.where(
            abs_pid == 11,
            0.000511,
            np.where(abs_pid == 13, 0.10566, 0.13957),
        ).astype(np.float64)
        p_mag = pt * np.cosh(eta)
        e = np.sqrt(p_mag * p_mag + mass * mass)
        all_pt.append(pt)
        all_eta.append(eta)
        # all_phi.append(phi)
        all_e.append(e)
        per_event_mult[i] = float(pt.shape[0])
        per_event_ht[i] = float(pt.sum())
    pt_cat = np.concatenate(all_pt) if all_pt else np.empty(0, dtype=np.float64)
    eta_cat = np.concatenate(all_eta) if all_eta else np.empty(0, dtype=np.float64)
    # phi_cat = np.concatenate(all_phi) if all_phi else np.empty(0, dtype=np.float64)
    e_cat = np.concatenate(all_e) if all_e else np.empty(0, dtype=np.float64)
    log_pt_cat = np.log(np.maximum(pt_cat, log_pt_floor))
    return {
        "pt": torch.from_numpy(pt_cat),
        "eta": torch.from_numpy(eta_cat),
        # "phi": torch.from_numpy(phi_cat),
        "E": torch.from_numpy(e_cat),
        "log_pt": torch.from_numpy(log_pt_cat),
        # "multiplicity": torch.from_numpy(per_event_mult),
        "ht": torch.from_numpy(per_event_ht),
    }


def trainee_observables(
    card_out: dict[str, torch.Tensor],
    n_events: int,
    min_pt: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Build the same observable dict from the trainee card's output.

    Operates on the ``EFlowObject`` branch of the card output, which is
    the PF-like final collection. Zero-pT ghost tracks (a byproduct of
    the Gumbel-ST efficiency mask; see ``learnable.py``) are filtered
    out before any statistic is computed; the filter is a hard boolean
    mask on values only, not on gradients, so it does not break
    backprop.

    The energy column is read directly from ``EFlowObject`` (the
    smearing modules update it in-place). ``log_pt`` is taken on the
    already-filtered (strictly positive) ``pt_kept`` so the logarithm
    is always well-defined and its gradient propagates back to the
    learnable smearing parameters.

    Returns
    -------
    dict[str, torch.Tensor]
        Same keys as :func:`pflow_target_observables`. All tensors
        carry gradients back to the learnable parameters.
    """
    eflow = card_out["EFlowObject"]
    pt_all = eflow[:, ColumnMap.PT]
    eta_all = eflow[:, ColumnMap.ETA]
    # phi_all = eflow[:, ColumnMap.PHI] # phi doesn't contribute to gradient
    e_all = eflow[:, ColumnMap.E]
    event_all = eflow[:, ColumnMap.EVENT_NUMBER]

    valid = pt_all > min_pt
    pt_kept = pt_all[valid]
    eta_kept = eta_all[valid]
    # phi_kept = phi_all[valid]
    e_kept = e_all[valid]
    log_pt_kept = torch.log(pt_kept)
    ev_kept = event_all[valid].long()

    # Per-event multiplicity and HT via scatter-add. Using
    # torch.zeros(..., dtype=pt.dtype) on the same device as pt so the
    # accumulation stays in the autograd graph.
    device = pt_all.device
    dtype = pt_all.dtype
    mult = torch.zeros(n_events, dtype=dtype, device=device)
    ht = torch.zeros(n_events, dtype=dtype, device=device)
    if ev_kept.numel() > 0:
        mult.scatter_add_(0, ev_kept, torch.ones_like(pt_kept))
        ht.scatter_add_(0, ev_kept, pt_kept)

    return {
        "pt": pt_kept,
        "eta": eta_kept,
        # "phi": phi_kept,
        "E": e_kept,
        "log_pt": log_pt_kept,
        # "multiplicity": mult,
        "ht": ht,
    }
