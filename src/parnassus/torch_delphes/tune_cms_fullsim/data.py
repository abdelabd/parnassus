"""ROOT I/O and observable construction for ``tune_cms_fullsim``.

This module turns a cms-flow-format ROOT file into the things the fit loop needs:

- :func:`load_cms_flow_root` reads the ``truth_*`` / ``pflow_*`` branches into
  per-event numpy arrays;
- :func:`load_truth_events` builds the padded ``(n_events, max_n_particles,
  N_FEATURES)`` truth particle tensor fed to the trainee card;
- :func:`load_pflow_targets` builds the per-observable target dict from the
  frozen ``pflow_*`` reco, and :func:`load_pflow_targets_from_tensor` builds the
  differentiable prediction dict from the trainee card's ``EFlowObject`` output
  (:func:`restore_event_format` regroups that flat output per event);
- :func:`split_truth_objects` / :func:`split_pflow_targets` do the train/val split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import uproot

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.utils import class_to_pid_vectorized, pid_to_class_vectorized

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
        if n_events < 0: # load all events from entry_start to the end of the tree
            n_events = tree.num_entries - entry_start
        arrays = tree.arrays(
            list(TRUTH_BRANCHES + PFLOW_BRANCHES),
            library="np",
            entry_start=entry_start,
            entry_stop=entry_start + n_events,
        )
    return arrays


# =============================================================================
# Constructing inputs
# =============================================================================

def load_truth_events(arrays: dict[str, np.ndarray],):
    """
    This task will pick truth particles from the input array, then it will
    pad the objects in each event to the max number of particles across events, and finally
    the output has shape (n_events, max_n_particles, n_features) in tensor form.
    """
    rows_list: list[np.ndarray] = []
    key_0 = arrays.keys().__iter__().__next__()
    for i in range(len(arrays[key_0])):
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

    # rows_list has shape (n_events, n_particles, n_features)
    # n_particles can vary across events
    # pad to the max n_particles across events and stack into a single tensor of shape (n_events, max_n_particles, n_features)
    max_n_particles = max(row.shape[0] for row in rows_list)
    padded_rows = []
    for row in rows_list:
        n_particles = row.shape[0]
        if n_particles < max_n_particles:
            padding = np.zeros((max_n_particles - n_particles, N_FEATURES), dtype=np.float64)
            padded_row = np.vstack([row, padding])
        else:
            padded_row = row
        padded_rows.append(padded_row)
    stacked_rows = np.stack(padded_rows, axis=0)
    
    return torch.from_numpy(stacked_rows)


def restore_event_format(
    eflow_objects: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Regroup the flat ``EFlowObject`` tensor back into per-event format.

    The input ``eflow_objects`` has shape ``(all_objects, n_features)`` with each
    object's source event index stored in the ``EVENT_NUMBER`` column. The output
    has shape ``(n_events, max_n_objects, n_features)``.

    ``mask`` is the ``(n_events, n_particles)`` truth-particle validity mask; only
    its first dim is used, to recover ``n_events`` (the batch size). Event numbers
    inside a batch are shuffled/global and non-contiguous, so they are remapped to
    dense ids; events that produced zero objects are absent from the input and end
    up as all-zero rows. The op is fully vectorized (no loop) and
    differentiable w.r.t. the ``eflow_objects`` values.
    """
    device = eflow_objects.device
    n_events = mask.shape[0]  # batch size, including events with zero objects
    n_objects, n_features = eflow_objects.shape

    if n_objects == 0:  # nothing reconstructed this batch
        return torch.zeros(
            (n_events, 0, n_features), dtype=eflow_objects.dtype, device=device
        )

    event_numbers = eflow_objects[:, ColumnMap.EVENT_NUMBER].long()
    # Map the shuffled/global event numbers to dense ids in [0, n_unique). Events
    # with no objects are absent here and stay all-zero in the padded output.
    _, local_event = torch.unique(event_numbers, sorted=True, return_inverse=True)

    counts = torch.bincount(local_event, minlength=n_events)  # objects per event
    max_n_objects = int(counts.max().item())

    # Within-event slot for each object, no loop: sort objects by event, then the
    # slot is "global sorted position - start offset of that event".
    offsets = torch.cumsum(counts, dim=0) - counts  # start index of each event
    order = torch.argsort(local_event, stable=True)
    slot = torch.empty(n_objects, dtype=torch.long, device=device)
    slot[order] = torch.arange(n_objects, device=device) - offsets[local_event[order]]

    out = torch.zeros(
        (n_events, max_n_objects, n_features), dtype=eflow_objects.dtype, device=device
    )
    out[local_event, slot] = eflow_objects  # differentiable index_put
    return out


# =============================================================================
# Constructing targets
# =============================================================================

def load_pflow_targets(arrays: dict[str, np.ndarray], log_pt_floor: float = -1):
    """
    This task will pick the pflow objects from the input array, then it will
    pad the objects in each event to the max number of particles across events, and finally
    the output is a dict of tensors, each with shape (n_events, max_n_particles) except for multiplicity and ht which have shape (n_events,).
    """
    # get num of events
    key_0 = arrays.keys().__iter__().__next__()
    n_events = len(arrays[key_0])

    all_pt: list[np.ndarray] = []
    all_eta: list[np.ndarray] = []
    # all_phi: list[np.ndarray] = [] # phi doesn't contribute to gradient
    all_e: list[np.ndarray] = []
    all_pids: list[np.ndarray] = []
    per_event_mult = np.zeros(n_events, dtype=np.float64)
    per_event_ht = np.zeros(n_events, dtype=np.float64)
    # Per-event reconstructed charged-hadron (pid 211) count in each of the 4 RECO
    # (pt, |eta|) bins [barrel-lowpt, barrel-highpt, endcap-lowpt, endcap-highpt].
    # This is the (realistic, data-only) TARGET for the differentiable charged-hadron
    # count term: the trainee builds a differentiable expected count in these SAME
    # reco bins from its own reco-bin <- pre-reco-region migration (see
    # CMSEnergyFlowDefault._charged_hadron_expected_reco_counts) and matches it here.
    per_event_chad_region_counts = np.zeros((n_events, 4), dtype=np.float64)
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
        all_pids.append(pids)
        per_event_mult[i] = float(pt.shape[0])
        per_event_ht[i] = float(pt.sum())

        # Per-region charged-hadron counts (regions match the learnable efficiency).
        is_chad = abs_pid == 211
        abs_eta = np.abs(eta)
        pt_low = (pt > 0.1) & (pt <= 1.0)
        pt_high = pt > 1.0
        barrel = abs_eta <= 1.5
        endcap = (abs_eta > 1.5) & (abs_eta <= 2.5)
        per_event_chad_region_counts[i, 0] = float(np.sum(is_chad & barrel & pt_low))
        per_event_chad_region_counts[i, 1] = float(np.sum(is_chad & barrel & pt_high))
        per_event_chad_region_counts[i, 2] = float(np.sum(is_chad & endcap & pt_low))
        per_event_chad_region_counts[i, 3] = float(np.sum(is_chad & endcap & pt_high))

    # shape of all_pt, all_eta, all_e is (num_events, num_particles_in_event); num_particles_in_event can vary across events
    # pad to the max num_particles across events and stack into a single tensor of shape (num_events, max_num_particles)
    max_n_particles = max(pt.shape[0] for pt in all_pt)

    # pad the last dimension (num_particles_in_event) with zeros to max_n_particles
    # and build a boolean mask so downstream code can ignore the padded slots
    # (padded pt=0 would otherwise corrupt log_pt and the per-bin multiplicity).
    def _pad_stack(arrays: list[np.ndarray]) -> np.ndarray:
        out = np.zeros((n_events, max_n_particles), dtype=np.float64)
        for i, arr in enumerate(arrays):
            out[i, : arr.shape[0]] = arr
        return out

    pt_pad = _pad_stack(all_pt)
    eta_pad = _pad_stack(all_eta)
    # phi_pad = _pad_stack(all_phi)  # phi doesn't contribute to gradient
    e_pad = _pad_stack(all_e)
    pids_pad = _pad_stack(all_pids)

    # True where a real particle exists, False for padding.
    mask = np.zeros((n_events, max_n_particles), dtype=bool)
    for i, arr in enumerate(all_pt):
        mask[i, : arr.shape[0]] = True

    # log(pt) only where valid; padded slots are left at 0 and excluded by the mask.
    log_pt_pad = np.zeros_like(pt_pad)
    log_pt_pad[mask] = np.log(np.maximum(pt_pad[mask], log_pt_floor))

    # log(E) only where valid; padded slots are left at 0 and excluded by the mask.
    log_E_pad = np.zeros_like(e_pad)
    log_E_pad[mask] = np.log(np.maximum(e_pad[mask], 1e-6))

    # log(HT); floor guards log(0) on the rare empty event (mirrors log_E).
    per_event_log_ht = np.log(np.maximum(per_event_ht, 1e-6))

    return {
        "pt": torch.from_numpy(pt_pad),
        "eta": torch.from_numpy(eta_pad),
        # "phi": torch.from_numpy(phi_pad),
        # "E": torch.from_numpy(e_pad),
        "log_pt": torch.from_numpy(log_pt_pad),
        "log_E": torch.from_numpy(log_E_pad),
        "pid": torch.from_numpy(pids_pad),
        "multiplicity": torch.from_numpy(per_event_mult),
        "ht": torch.from_numpy(per_event_ht),
        "log_ht": torch.from_numpy(per_event_log_ht),
        "chad_region_counts": torch.from_numpy(per_event_chad_region_counts),
    }


def load_pflow_targets_from_tensor(arrays: torch.Tensor, log_pt_floor: float = -1):
    """Build the per-observable dict from a padded ``EFlowObject`` tensor.

    The differentiable counterpart of :func:`load_pflow_targets`: it operates
    on the trainee card's output instead of a numpy ROOT dict, so gradients
    flow back to the learnable card parameters.

    The input has shape ``(n_events, max_n_objects, N_FEATURES)`` with two kinds
    of empty slots that must be excluded:

    - genuine zero-padding rows added by :func:`restore_event_format`, and
    - efficiency-killed "ghost" rows, where ``LearnableEfficiency`` has zeroed
      ``(PT, PX, PY, PZ, E)`` via a Gumbel-ST mask but left the row in place
      with nonzero ``ETA`` / ``PID``.

    Both are caught by a single ``pt != 0`` test (a real reconstructed object
    always has ``pt > 0``), which matches the target side's ``pflow_pt > 1e-6``
    cut. The output stays on ``arrays``'s device and dtype and is fully
    differentiable wrt ``arrays``.

    Returns
    -------
    dict[str, torch.Tensor]
        ``"pt"``, ``"eta"``, ``"log_pt"``, ``"log_E"`` of shape
        ``(n_events, max_n_objects)`` (zero on invalid slots), and
        ``"multiplicity"``, ``"ht"``, ``"log_ht"`` of shape ``(n_events,)``.
    """
    pt = arrays[..., ColumnMap.PT]  # (n_events, max_n_objects)
    eta = arrays[..., ColumnMap.ETA]

    # Normalize the trainee card's mixed Delphes PID column (neutral-hadron
    # marker 0, photon 22, real PDG on tracks) to the SAME canonical PDG codes
    # the target side carries (load_pflow_targets): pid -> class id -> canonical
    # pid (211/11/13/111/22). PID is discrete bookkeeping (not in the loss; it
    # only selects the mass below), so route it through numpy off the graph and
    # place the result back on the input device.
    pid_np = arrays[..., ColumnMap.PID].detach().cpu().numpy().astype(np.int64)
    pid_np = class_to_pid_vectorized(pid_to_class_vectorized(pid_np))
    pid = torch.from_numpy(pid_np).to(device=arrays.device)

    # Real particle <=> nonzero pt (drops padding and efficiency-killed ghosts).
    valid = pt != 0

    # Reconstruct E = sqrt((pt * cosh(eta))^2 + m^2) with the mass derived from
    # the canonical PDG in the PID column (211/11/13/111/22), mirroring
    # load_pflow_targets. PID is already an off-graph int64 tensor (no gradient
    # path), so the mass is a per-row constant as required.
    abs_pid = pid.abs()
    mass = torch.where(
        abs_pid == 11,
        torch.full_like(pt, 0.000511),
        torch.where(
            abs_pid == 13,
            torch.full_like(pt, 0.10566),
            torch.full_like(pt, 0.13957),
        ),
    )
    p_mag = pt * torch.cosh(eta)
    e = torch.sqrt(p_mag * p_mag + mass * mass)
    e = torch.where(valid, e, torch.zeros_like(e))

    # log(pt) on valid slots only. Clamp the log ARGUMENT to a positive value
    # before taking the log so the backward (1/arg) stays finite on the pt == 0
    # slots; a plain torch.where(valid, log(pt), 0) would still backprop
    # log(0)'s -inf derivative through the masked branch and poison the
    # gradient. (log_pt_floor is a no-op for pt >= 0, mirroring the reference;
    # it is kept in the signature for parity.)
    pt_safe = torch.where(valid, pt, torch.ones_like(pt))  # 1.0 on invalid slots
    log_pt = torch.where(valid, torch.log(pt_safe), torch.zeros_like(pt))

    # Same gradient-safety as log_pt: e was zeroed on invalid slots above, so a
    # plain torch.where(valid, log(e), 0) would backprop log(0)'s -inf derivative
    # through the masked branch (0 * inf = NaN). Clamp the log argument to 1.0 on
    # invalid slots before taking the log.
    e_safe = torch.where(valid, e, torch.ones_like(e))  # 1.0 on invalid slots
    log_E = torch.where(valid, torch.log(e_safe), torch.zeros_like(e))

    pt_out = torch.where(valid, pt, torch.zeros_like(pt))
    eta_out = torch.where(valid, eta, torch.zeros_like(eta))
    pid_out = torch.where(valid, pid, torch.zeros_like(pid))

    valid_f = valid.to(pt.dtype)
    multiplicity = valid_f.sum(dim=1)  # (n_events,) -- a count (no gradient)
    ht = (pt * valid_f).sum(dim=1)  # (n_events,) -- differentiable
    # log(HT): clamp keeps the forward value positive so 1/value stays finite
    # in the backward (and the clamp grad is 0 below the floor) -- no NaN on the
    # rare empty event. Floor matches the target side.
    log_ht = torch.log(torch.clamp(ht, min=1e-6))

    return {
        "pt": pt_out,
        "eta": eta_out,
        "log_E": log_E,
        # "E": e,
        "log_pt": log_pt,
        "pid": pid_out,
        "multiplicity": multiplicity,
        "ht": ht,
        "log_ht": log_ht,
    }


# =============================================================================
# Train validation splitting
# =============================================================================

def split_truth_objects(truth_tensor: torch.Tensor, train_fraction: float = 0.8, seed: int = 42):
    """Split the truth tensor into train and validation parts along the event axis.

    The split is deterministic based on the provided random seed. The same split
    is applied to all events, so the train and validation sets are disjoint at
    the event level (no event is partially in train and partially in val).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The train and validation truth tensors, each with shape
        ``(n_events_subset, max_n_particles, n_features)``.
    """
    n_events = truth_tensor.shape[0]
    indices = torch.randperm(n_events, generator=torch.Generator().manual_seed(seed))
    split_idx = int(train_fraction * n_events)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    train_tensor = truth_tensor[train_indices]
    val_tensor = truth_tensor[val_indices]
    return train_tensor, val_tensor


def split_pflow_targets(target: dict[str, torch.Tensor], train_fraction: float = 0.8, seed: int = 42):
    """Split the target dict into train and validation parts along the event axis.

    The split is deterministic based on the provided random seed. The same split
    is applied to all observables, so the train and validation sets are disjoint
    at the event level (no event is partially in train and partially in val).

    Returns
    -------
    dict[str, torch.Tensor], dict[str, torch.Tensor]
        The train and validation target dicts, each with the same keys as the
        input and tensors with shape ``(n_events_subset, ...)``.
    """
    n_events = next(iter(target.values())).shape[0]  # number of events from any observable
    indices = torch.randperm(n_events, generator=torch.Generator().manual_seed(seed))
    split_idx = int(train_fraction * n_events)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_target = {k: v[train_indices] for k, v in target.items()}
    val_target = {k: v[val_indices] for k, v in target.items()}
    return train_target, val_target