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

from parnassus.data.particle_io import (
    N_FEATURES,
    ColumnMap,
    get_charge_from_pdg_id,
    get_mass_from_pdg_id,
)
from parnassus.torch_delphes.learnable import CMS_EFF_REGION_SPECS
from parnassus.torch_delphes.SimpleCalorimeter import (
    calo_count_eta_edges,
    calo_count_region_masks,
)
from parnassus.utils import class_to_pid_vectorized, pid_to_class_vectorized

from .config import PFLOW_BRANCHES, TRUTH_BRANCHES

# Per-species reconstructed-data count targets for the differentiable count terms.
# (region-spec key, |pid| selecting that species in the reco data, target dict key).
# Each target is matched against ``CMSEnergyFlowDefault._expected_reco_counts`` for
# the same species, in the SAME reco (pt, |eta|) bins. Charged hadrons collapse to
# pid 211, electrons to 11, muons to 13 (see parnassus.utils.class_to_pid).
_COUNT_TERM_SPECIES: tuple[tuple[str, int, str], ...] = (
    ("charged_hadron", 211, "chad_region_counts"),
    ("electron", 11, "electron_region_counts"),
    ("muon", 13, "muon_region_counts"),
)

# Per-species calorimeter object-count targets for the differentiable resolution-param
# count term. Unlike the efficiency regions above these bin by |eta| ONLY, matching
# the soft significance gate in ``SimpleCalorimeter.forward``: ECal photons
# (|pid|==22) and HCal neutral hadrons (|pid|==111). Tuple = (|pid| selecting the
# species, is_ecal, target dict key). The region edges are derived from the SAME
# shared helpers (calo_count_eta_edges / calo_count_region_masks in
# parnassus.torch_delphes.SimpleCalorimeter) that the soft gate uses, optionally
# bounded by the reco |eta| acceptance cut, so pred and target region layouts
# cannot drift.
_CALO_COUNT_SPECIES: tuple[tuple[int, bool, str], ...] = (
    (22, True, "ecal_photon_region_counts"),
    (111, False, "hcal_nh_region_counts"),
)

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

    Branches in :data:`TRUTH_BRANCHES` / :data:`PFLOW_BRANCHES` that are not
    present in the tree are silently skipped here; the required ``truth_pdgid``
    branch is then enforced downstream by :func:`load_truth_events`, which
    raises if it is absent (regenerate the pseudodata to add it).
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
        available = set(tree.keys())
        requested = [b for b in (TRUTH_BRANCHES + PFLOW_BRANCHES) if b in available]
        arrays = tree.arrays(
            requested,
            library="np",
            entry_start=entry_start,
            entry_stop=entry_start + n_events,
        )
    return arrays


# =============================================================================
# Constructing inputs
# =============================================================================

def _build_truth_rows(
    arrays: dict[str, np.ndarray],
    truth_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
) -> list[np.ndarray]:
    """Build the per-event ``(n_particles, N_FEATURES)`` truth rows (float64).

    Shared by :func:`load_truth_events` (pads + stacks them into a dense tensor)
    and :func:`load_truth_events_ragged` (keeps them ragged). The returned list
    has exactly one entry per event, in file order; an event with zero truth
    particles contributes a ``(0, N_FEATURES)`` row so the list stays aligned
    with the pflow target loader (which also keeps every event).

    ``truth_pt_cut`` / ``abs_eta_cut`` apply an explicit acceptance selection
    (``pt >= truth_pt_cut`` and ``|eta| <= abs_eta_cut``) to ALL truth species
    before the rows are built. At the CLI defaults (0.25 / 2.7) this is a no-op
    on the externally preprocessed files, but it makes the harmonization
    explicit in code. ``None`` disables the respective part.
    """
    rows_list: list[np.ndarray] = []
    key_0 = arrays.keys().__iter__().__next__()
    if "truth_pdgid" not in arrays:
        raise KeyError(
            "truth_pdgid branch is required but missing from the loaded ROOT "
            "arrays. Regenerate the pseudodata with "
            "parnassus.torch_delphes.generate_pseudodata (which now writes "
            "truth_pdgid), or point --root-file at a sample that carries the "
            "real truth PDG ids."
        )
    for i in range(len(arrays[key_0])):
        pt = np.asarray(arrays["truth_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["truth_eta"][i], dtype=np.float64)
        phi = np.asarray(arrays["truth_phi"][i], dtype=np.float64)
        # Real PDG IDs (K_L0=130, n=2112, K+/-=321, p=2212, ...), carried
        # through unchanged so the ECal/HCal energy-fraction LUT routes each
        # species correctly instead of collapsing to the lossy class buckets.
        pids = np.asarray(arrays["truth_pdgid"][i], dtype=np.int64)
        if truth_pt_cut is not None or abs_eta_cut is not None:
            sel = np.ones(pt.shape[0], dtype=bool)
            if truth_pt_cut is not None:
                sel &= pt >= truth_pt_cut
            if abs_eta_cut is not None:
                sel &= np.abs(eta) <= abs_eta_cut
            pt, eta, phi, pids = pt[sel], eta[sel], phi[sel], pids[sel]
        n_p = pt.shape[0]
        if n_p == 0:
            # Keep the empty event as a zero-row entry (NOT dropped): the pflow
            # target loader keeps every event, so dropping here would silently
            # desync truth[i] from target[i] for all subsequent events.
            rows_list.append(np.zeros((0, N_FEATURES), dtype=np.float64))
            continue
        # PDG-aware mass and charge so the energy reconstruction and the
        # B-field track curvature use the correct values per species.
        mass = get_mass_from_pdg_id(pids)
        charge = get_charge_from_pdg_id(pids)
        # Unknown PDGs return -999 (C++ Delphes convention); clamp to neutral
        # instead of routing unknowns into charged streams.
        charge = np.where(charge == -999, 0.0, charge).astype(np.float64)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        e = np.sqrt(px * px + py * py + pz * pz + mass * mass)
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
    return rows_list


def load_truth_events(
    arrays: dict[str, np.ndarray],
    truth_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
):
    """
    This task will pick truth particles from the input array, then it will
    pad the objects in each event to the max number of particles across events, and finally
    the output has shape (n_events, max_n_particles, n_features) in tensor form.

    ``truth_pt_cut`` / ``abs_eta_cut``: see :func:`_build_truth_rows`.
    """
    rows_list = _build_truth_rows(arrays, truth_pt_cut=truth_pt_cut, abs_eta_cut=abs_eta_cut)
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


def load_truth_events_ragged(
    arrays: dict[str, np.ndarray],
    truth_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
) -> list[torch.Tensor]:
    """Ragged counterpart of :func:`load_truth_events`.

    Returns the per-event truth particles as a ``list`` of ``(n_particles_i,
    N_FEATURES)`` float64 tensors (one per event, file order; empty events give
    ``(0, N_FEATURES)`` entries) instead of padding every event up to the GLOBAL
    max multiplicity and stacking into a dense ``(n_events, max_n_particles,
    N_FEATURES)`` tensor. The dense form wastes ~max/mean memory (it pads to the
    busiest event in the whole file, ~50 GB at 100k events) and the fit loop
    immediately un-pads it anyway, so the ragged list + per-batch padding (see
    ``DelphesDataSet`` / ``delphes_collate_fn``) is numerically identical and an
    order of magnitude smaller.

    ``truth_pt_cut`` / ``abs_eta_cut``: see :func:`_build_truth_rows`.
    """
    return [
        torch.from_numpy(row)
        for row in _build_truth_rows(arrays, truth_pt_cut=truth_pt_cut, abs_eta_cut=abs_eta_cut)
    ]


def restore_event_format(
    eflow_objects: torch.Tensor,
    mask: torch.Tensor,
    event_ids: torch.Tensor | None = None,
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

    ``event_ids`` (optional): the ``(n_events,)`` global event numbers of the
    batch IN BATCH ORDER (see :func:`batch_event_ids`). When given, output row
    ``i`` holds the objects of event ``event_ids[i]``, so the prediction rows
    line up with the target batch rows -- REQUIRED by any per-event pairing such
    as :func:`apply_chad_truncation` (without it rows are ordered by ascending
    global event number, which under a shuffled sampler is NOT the batch order;
    purely distributional losses are unaffected either way). When ``None``,
    the legacy ascending-event-number layout is used.
    """
    device = eflow_objects.device
    n_events = mask.shape[0]  # batch size, including events with zero objects
    n_objects, n_features = eflow_objects.shape

    if n_objects == 0:  # nothing reconstructed this batch
        return torch.zeros(
            (n_events, 0, n_features), dtype=eflow_objects.dtype, device=device
        )

    event_numbers = eflow_objects[:, ColumnMap.EVENT_NUMBER].long()
    if event_ids is not None:
        # Map each object's global event number to its BATCH position, so pred
        # row i == target row i. searchsorted over the sorted ids, then undo the
        # sort permutation.
        sorted_ids, perm = torch.sort(event_ids.long())
        pos = torch.searchsorted(sorted_ids, event_numbers)
        local_event = perm[pos]
    else:
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

def _build_pflow_event_data(
    arrays: dict[str, np.ndarray],
    reco_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
    truncate_chads: bool = False,
):
    """Build the per-event pflow target arrays shared by the dense and ragged loaders.

    Returns ``(n_events, all_pt, all_eta, all_e, all_pids, per_event_mult,
    per_event_ht, per_event_n_truth_chad, per_event_region_counts)`` where the
    ``all_*`` are per-event lists of variable-length 1-D float64 arrays
    (``all_pids`` int64) and the ``per_event_*`` are dense ``(n_events, ...)``
    arrays. Every event is kept (including empty ones), so the first axis is the
    full event count.

    Acceptance / truncation (all default off = legacy behavior):

    - ``reco_pt_cut`` / ``abs_eta_cut``: keep reco objects (ALL classes) with
      ``pt >= reco_pt_cut`` and ``|eta| <= abs_eta_cut``. At the CLI defaults
      (1.0 / 2.7) this is a no-op on the externally preprocessed target files but
      mirrors the cut applied to the trainee output in the fit loop.
    - ``per_event_n_truth_chad``: count of truth charged hadrons inside the SAME
      acceptance (``pt >= reco_pt_cut``, ``|eta| <= abs_eta_cut``) -- the
      per-event ceiling on how many reco chads a truth-fed sim could produce.
      Zero when the truth branches are absent from ``arrays``.
    - ``truncate_chads``: per event, keep only the top-``n_truth_chad``
      charged hadrons by pt (other classes untouched). Removes the reco chads
      the data contains but a truth-fed sim can never make (decay-in-flight
      daughters, baryons missing from the truth record, GEANT4 material
      secondaries). Everything downstream (mult, ht, region counts -- including
      the chad count-term targets) is built from the cut + truncated set.
    """
    # get num of events
    key_0 = arrays.keys().__iter__().__next__()
    n_events = len(arrays[key_0])

    has_truth = "truth_pt" in arrays and "truth_pdgid" in arrays
    if truncate_chads and not has_truth:
        raise KeyError(
            "truncate_chads=True requires the truth_pt/truth_eta/truth_pdgid "
            "branches in the loaded arrays to count the per-event truth "
            "charged-hadron ceiling, but they are absent."
        )

    all_pt: list[np.ndarray] = []
    all_eta: list[np.ndarray] = []
    all_phi: list[np.ndarray] = []
    all_e: list[np.ndarray] = []
    all_pids: list[np.ndarray] = []
    per_event_mult = np.zeros(n_events, dtype=np.float64)
    per_event_ht = np.zeros(n_events, dtype=np.float64)
    per_event_n_truth_chad = np.zeros(n_events, dtype=np.float64)
    # Per-event reconstructed per-species count in each RECO (pt, |eta|) region.
    # These are the (realistic, data-only) TARGETS for the differentiable count terms:
    # the trainee builds a differentiable expected count in these SAME reco bins from
    # its own reco-bin <- pre-reco-region migration (see
    # CMSEnergyFlowDefault._expected_reco_counts) and matches it here. Binning comes
    # from the shared CMS_EFF_REGION_SPECS so the three call sites cannot drift.
    per_event_region_counts: dict[str, np.ndarray] = {
        key: np.zeros((n_events, CMS_EFF_REGION_SPECS[spec_key].n_regions), dtype=np.float64)
        for spec_key, _pid, key in _COUNT_TERM_SPECIES
    }
    # Calorimeter object-count targets share the same dict, so they spread into the
    # loader output (and the loss) automatically alongside the efficiency counts.
    # Region layout (possibly |eta|-bounded) comes from the shared helpers so it
    # matches the card's soft-count regions when both get the same abs_eta_cut.
    per_event_region_counts.update({
        key: np.zeros(
            (n_events, len(calo_count_eta_edges(is_ecal, abs_eta_cut)) + 1),
            dtype=np.float64,
        )
        for _pid, is_ecal, key in _CALO_COUNT_SPECIES
    })
    for i in range(n_events):
        pt = np.asarray(arrays["pflow_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["pflow_eta"][i], dtype=np.float64)
        phi = np.asarray(arrays["pflow_phi"][i], dtype=np.float64)
        cls = np.asarray(arrays["pflow_class"][i], dtype=np.int64)
        # Reco acceptance cut (all classes) BEFORE anything downstream, so the
        # observables, ht and every region-count target see the cut set only.
        if reco_pt_cut is not None or abs_eta_cut is not None:
            sel = np.ones(pt.shape[0], dtype=bool)
            if reco_pt_cut is not None:
                sel &= pt >= reco_pt_cut
            if abs_eta_cut is not None:
                sel &= np.abs(eta) <= abs_eta_cut
            pt, eta, phi, cls = pt[sel], eta[sel], phi[sel], cls[sel]
        pids = class_to_pid_vectorized(cls) if pt.size else np.empty(0, dtype=np.int64)
        abs_pid = np.abs(pids)

        # Per-event truth charged-hadron ceiling, counted in the SAME acceptance
        # as the reco cut (a sub-cut truth chad cannot make an in-acceptance reco
        # chad; momentum smearing is percent-level). Loop index i is file order in
        # BOTH collections, so this stays aligned with the reco arrays.
        if has_truth:
            t_pt = np.asarray(arrays["truth_pt"][i], dtype=np.float64)
            if t_pt.size:
                t_eta = np.asarray(arrays["truth_eta"][i], dtype=np.float64)
                t_pids = np.asarray(arrays["truth_pdgid"][i], dtype=np.int64)
                t_sel = pid_to_class_vectorized(t_pids) == 0  # any charged hadron
                if reco_pt_cut is not None:
                    t_sel &= t_pt >= reco_pt_cut
                if abs_eta_cut is not None:
                    t_sel &= np.abs(t_eta) <= abs_eta_cut
                per_event_n_truth_chad[i] = float(np.sum(t_sel))

        # Truncate reco charged hadrons at the truth ceiling: keep the top-k by
        # pt (stable sort => deterministic on ties), all other classes untouched.
        if truncate_chads:
            k = int(per_event_n_truth_chad[i])
            chad_idx = np.flatnonzero(abs_pid == 211)
            if chad_idx.size > k:
                order = np.argsort(-pt[chad_idx], kind="stable")
                keep = np.ones(pt.shape[0], dtype=bool)
                keep[chad_idx[order[k:]]] = False
                pt, eta, phi, pids, abs_pid = (
                    pt[keep], eta[keep], phi[keep], pids[keep], abs_pid[keep]
                )

        # Reconstruct per-particle energy: E = sqrt(p^2 + m^2) with
        # |p| = pt * cosh(eta) and m from the class -> PDG -> mass map.
        mass = np.where(
            abs_pid == 11,
            0.000511,
            np.where(abs_pid == 13, 0.10566, 0.13957),
        ).astype(np.float64)
        p_mag = pt * np.cosh(eta)
        e = np.sqrt(p_mag * p_mag + mass * mass)
        all_pt.append(pt)
        all_eta.append(eta)
        all_phi.append(phi)
        all_e.append(e)
        all_pids.append(pids)
        per_event_mult[i] = float(pt.shape[0])
        per_event_ht[i] = float(pt.sum())

        # Per-region per-species counts (regions match the learnable efficiency,
        # via the shared spec; binning is duck-typed so it runs on these numpy arrays).
        abs_eta = np.abs(eta)
        for spec_key, pid_sel, key in _COUNT_TERM_SPECIES:
            spec = CMS_EFF_REGION_SPECS[spec_key]
            is_species = abs_pid == pid_sel
            for b, region_mask in enumerate(spec.region_masks(pt, abs_eta)):
                per_event_region_counts[key][i, b] = float(np.sum(is_species & region_mask))
        # Calorimeter object counts: |eta|-only regions matching the soft gate.
        for pid_sel, is_ecal, key in _CALO_COUNT_SPECIES:
            is_species = abs_pid == pid_sel
            edges = calo_count_eta_edges(is_ecal, abs_eta_cut)
            for b, region_mask in enumerate(
                calo_count_region_masks(abs_eta, edges, abs_eta_cut)
            ):
                per_event_region_counts[key][i, b] = float(np.sum(is_species & region_mask))

    return (
        n_events,
        all_pt,
        all_eta,
        all_phi,
        all_e,
        all_pids,
        per_event_mult,
        per_event_ht,
        per_event_n_truth_chad,
        per_event_region_counts,
    )


def load_pflow_targets(
    arrays: dict[str, np.ndarray],
    log_pt_floor: float = 1e-6,
    reco_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
    truncate_chads: bool = False,
):
    """
    This task will pick the pflow objects from the input array, then it will
    pad the objects in each event to the max number of particles across events, and finally
    the output is a dict of tensors, each with shape (n_events, max_n_particles) except for multiplicity and ht which have shape (n_events,).

    ``reco_pt_cut`` / ``abs_eta_cut`` / ``truncate_chads``: see
    :func:`_build_pflow_event_data`.
    """
    (
        n_events,
        all_pt,
        all_eta,
        all_phi,
        all_e,
        all_pids,
        per_event_mult,
        per_event_ht,
        per_event_n_truth_chad,
        per_event_region_counts,
    ) = _build_pflow_event_data(
        arrays,
        reco_pt_cut=reco_pt_cut,
        abs_eta_cut=abs_eta_cut,
        truncate_chads=truncate_chads,
    )

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
    phi_pad = _pad_stack(all_phi)
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
        "phi": torch.from_numpy(phi_pad),
        # "E": torch.from_numpy(e_pad),
        "log_pt": torch.from_numpy(log_pt_pad),
        "log_E": torch.from_numpy(log_E_pad),
        "pid": torch.from_numpy(pids_pad),
        "multiplicity": torch.from_numpy(per_event_mult),
        "ht": torch.from_numpy(per_event_ht),
        "log_ht": torch.from_numpy(per_event_log_ht),
        "n_truth_chad": torch.from_numpy(per_event_n_truth_chad),
        **{key: torch.from_numpy(arr) for key, arr in per_event_region_counts.items()},
    }


def load_pflow_targets_ragged(
    arrays: dict[str, np.ndarray],
    log_pt_floor: float = 1e-6,
    reco_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
    truncate_chads: bool = False,
):
    """Ragged counterpart of :func:`load_pflow_targets`.

    The per-particle observables (``pt``, ``eta``, ``log_pt``, ``log_E``, ``pid``)
    are returned as a ``list`` of per-event 1-D float64 tensors instead of dense
    ``(n_events, max_n_particles)`` arrays padded to the GLOBAL max multiplicity.
    The per-event scalars (``multiplicity``, ``ht``, ``log_ht``, ``n_truth_chad``)
    and the per-region count targets (``*_region_counts``) stay dense
    ``(n_events, ...)`` exactly as in the dense loader. Per-batch padding in
    ``delphes_collate_fn`` reconstructs the same batch the fit loop expects
    (padded slots carry ``pid == 0`` and are dropped by the loss), so this is
    numerically identical at a fraction of the memory.

    ``reco_pt_cut`` / ``abs_eta_cut`` / ``truncate_chads``: see
    :func:`_build_pflow_event_data`.
    """
    (
        n_events,
        all_pt,
        all_eta,
        all_phi,
        all_e,
        all_pids,
        per_event_mult,
        per_event_ht,
        per_event_n_truth_chad,
        per_event_region_counts,
    ) = _build_pflow_event_data(
        arrays,
        reco_pt_cut=reco_pt_cut,
        abs_eta_cut=abs_eta_cut,
        truncate_chads=truncate_chads,
    )

    # Per-event log(pt) / log(E) on the real particles only -- no padded slots to
    # mask out. The np.maximum floors guard log(0): a genuine pt == 0 pflow object
    # would otherwise become -inf and poison the bin-free Wasserstein loss (its
    # torch.quantile interpolation turns -inf into NaN). log_pt_floor mirrors the
    # 1e-6 argument-floor used for log_E just below, on both this target loader and
    # the pred-side load_pflow_targets_from_tensor.
    log_pt_list = [np.log(np.maximum(pt, log_pt_floor)) for pt in all_pt]
    log_E_list = [np.log(np.maximum(e, 1e-6)) for e in all_e]

    # log(HT); floor guards log(0) on the rare empty event (mirrors the dense loader).
    per_event_log_ht = np.log(np.maximum(per_event_ht, 1e-6))

    def _to_list(arrs: list[np.ndarray]) -> list[torch.Tensor]:
        # float64 throughout so per-batch pad_sequence and the loss's torch.stack
        # agree on dtype (the dense loader's pids_pad is float64 too).
        return [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float64)) for a in arrs]

    return {
        "pt": _to_list(all_pt),
        "eta": _to_list(all_eta),
        "phi": _to_list(all_phi),
        "log_pt": _to_list(log_pt_list),
        "log_E": _to_list(log_E_list),
        "pid": _to_list(all_pids),
        "multiplicity": torch.from_numpy(per_event_mult),
        "ht": torch.from_numpy(per_event_ht),
        "log_ht": torch.from_numpy(per_event_log_ht),
        "n_truth_chad": torch.from_numpy(per_event_n_truth_chad),
        **{key: torch.from_numpy(arr) for key, arr in per_event_region_counts.items()},
    }


def load_pflow_targets_from_tensor(arrays: torch.Tensor, log_pt_floor: float = 1e-6):
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
    phi = arrays[..., ColumnMap.PHI]

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
    # gradient. log_pt_floor (1e-6) also floors a genuine valid pt == 0 object so
    # its log_pt is log(1e-6) instead of -inf, matching the target loaders bit for
    # bit (the bin-free Wasserstein loss turns a -inf target into NaN otherwise).
    pt_safe = torch.clamp(
        torch.where(valid, pt, torch.ones_like(pt)),  # 1.0 on invalid slots
        min=log_pt_floor,
    )
    log_pt = torch.where(valid, torch.log(pt_safe), torch.zeros_like(pt))

    # Same gradient-safety as log_pt: e was zeroed on invalid slots above, so a
    # plain torch.where(valid, log(e), 0) would backprop log(0)'s -inf derivative
    # through the masked branch (0 * inf = NaN). Clamp the log argument to 1.0 on
    # invalid slots before taking the log.
    e_safe = torch.where(valid, e, torch.ones_like(e))  # 1.0 on invalid slots
    log_E = torch.where(valid, torch.log(e_safe), torch.zeros_like(e))

    pt_out = torch.where(valid, pt, torch.zeros_like(pt))
    eta_out = torch.where(valid, eta, torch.zeros_like(eta))
    phi_out = torch.where(valid, phi, torch.zeros_like(phi))
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
        "phi": phi_out,
        "log_E": log_E,
        # "E": e,
        "log_pt": log_pt,
        "pid": pid_out,
        "multiplicity": multiplicity,
        "ht": ht,
        "log_ht": log_ht,
    }


# =============================================================================
# Acceptance cut + truth-ceiling chad truncation (loss-side filters)
# =============================================================================

# Per-object observable keys zeroed by the filters below. Dropped slots get
# pt == 0 AND pid == 0 -- identical to the padding/ghost convention -- so
# _group_objects_by_pid (loss) and the ``pt != 0`` plot cuts exclude them with
# no downstream changes.
_OBJECT_OBS_KEYS: tuple[str, ...] = ("pt", "eta", "phi", "log_E", "log_pt", "pid")


def _zero_dropped_and_recompute(
    obs: dict[str, torch.Tensor], keep: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Shared tail of the loss-side filters: zero dropped slots on the six
    per-object keys and recompute ``multiplicity`` / ``ht`` / ``log_ht`` from the
    kept objects (1e-6 floor matches the loaders). Shallow copy -- every other
    key (``*_region_counts``, ``*_expected_counts``, ``n_truth_chad``, ...)
    passes through untouched; the input dict is never mutated.
    """
    out = dict(obs)
    for key in _OBJECT_OBS_KEYS:
        if key in out:
            out[key] = torch.where(keep, obs[key], torch.zeros_like(obs[key]))
    out["multiplicity"] = keep.sum(dim=1).to(obs["multiplicity"].dtype)
    ht = out["pt"].sum(dim=1)  # differentiable through the kept pred pt
    out["ht"] = ht
    out["log_ht"] = torch.log(torch.clamp(ht, min=1e-6))
    return out


def apply_reco_acceptance_cut(
    obs: dict[str, torch.Tensor],
    reco_pt_cut: float | None,
    abs_eta_cut: float | None,
) -> dict[str, torch.Tensor]:
    """Apply the reco acceptance cut (``pt >= reco_pt_cut``, ``|eta| <=
    abs_eta_cut``) to a per-observable dict, all classes.

    The pred-side runtime counterpart of the loader-side cut in
    :func:`_build_pflow_event_data`: the target files already carry the cut (or
    get it at load time), while the trainee output must be cut here so both
    sides of the loss live in the same acceptance. Gradients flow through the
    kept slots only (the keep mask is built off-graph); the differentiable
    membership gradient at the thresholds is supplied by the count terms'
    soft/hard gates instead (``count_pt_min`` in the card).
    """
    if (reco_pt_cut is None and abs_eta_cut is None) or obs["pt"].shape[1] == 0:
        return dict(obs)
    with torch.no_grad():
        pt = obs["pt"]
        keep = pt != 0  # real objects only; never resurrect padding/ghosts
        if reco_pt_cut is not None:
            keep &= pt >= reco_pt_cut
        if abs_eta_cut is not None:
            keep &= obs["eta"].abs() <= abs_eta_cut
    return _zero_dropped_and_recompute(obs, keep)


def apply_chad_truncation(
    obs: dict[str, torch.Tensor],
    n_truth_chad: torch.Tensor,
    chad_pid: int = 211,
) -> dict[str, torch.Tensor]:
    """Per event, keep only the top-``n_truth_chad[i]`` charged hadrons by pt.

    Removes the reco charged hadrons the full-sim data contains but a truth-fed
    sim can never produce (K_S/Lambda decay-in-flight daughters, baryons absent
    from the truth record, GEANT4 material secondaries): the cap is the number
    of truth chads inside the reco acceptance (see ``n_truth_chad`` in
    :func:`_build_pflow_event_data`). Applied to BOTH pred and target for
    symmetry (the trainee is structurally at/below the ceiling, so it mostly
    bites the target). Other classes are untouched. Events with fewer chads
    than the cap keep all of them; a cap of 0 drops all chads.

    PRECONDITION: ``obs`` rows must be in the same event order as
    ``n_truth_chad`` -- in the fit loop that is guaranteed by passing
    ``event_ids=batch_event_ids(...)`` to :func:`restore_event_format`.
    Apply :func:`apply_reco_acceptance_cut` FIRST so the ranking only sees
    in-acceptance chads.
    """
    pt = obs["pt"]
    n_events, n_obj = pt.shape
    if n_obj == 0:
        return dict(obs)
    with torch.no_grad():
        pid = obs["pid"]
        is_chad = (pid.abs() == chad_pid) & (pt != 0)
        # Rank chads by pt within each event: non-chads score -inf so every real
        # chad outranks them; stable sort keeps pt-ties deterministic.
        scores = torch.where(is_chad, pt, torch.full_like(pt, float("-inf")))
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(
            1, order, torch.arange(n_obj, device=pt.device).expand(n_events, n_obj)
        )
        k = n_truth_chad.long().clamp(min=0).unsqueeze(1)  # (n_events, 1)
        keep = torch.where(rank < k, pt != 0, ~is_chad & (pt != 0))
    return _zero_dropped_and_recompute(obs, keep)


def batch_event_ids(truth_particles: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Global event numbers of a padded truth batch, in batch order.

    Reads the ``EVENT_NUMBER`` column of each event's first valid particle.
    Events with zero truth particles get a unique negative sentinel (they can
    never match a reconstructed object's event number, so their pred row stays
    all-zero). Pass the result as ``event_ids`` to :func:`restore_event_format`
    to align pred rows with the target batch rows.
    """
    n_events = mask.shape[0]
    sentinels = -(torch.arange(n_events, device=mask.device, dtype=torch.long) + 1)
    if mask.shape[1] == 0:
        return sentinels
    first = mask.long().argmax(dim=1)  # index of the first valid particle
    ids = truth_particles[
        torch.arange(n_events, device=mask.device), first, ColumnMap.EVENT_NUMBER
    ].long()
    return torch.where(mask.any(dim=1), ids, sentinels)


# =============================================================================
# Train validation splitting
# =============================================================================

def split_truth_objects(truth_tensor: torch.Tensor, train_fraction: float = 0.7, val_fraction: float = 0.2, ):
    """Split the truth tensor into train and validation parts along the event axis.

    Uses a contiguous event-axis split: the first ``train_fraction`` block goes
    to train and the tail goes to validation. ``seed`` is accepted for API
    compatibility but is intentionally unused.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        The train and validation truth tensors, each with shape
        ``(n_events_subset, max_n_particles, n_features)``.
    """
    n_events = truth_tensor.shape[0]
    train_tensor = truth_tensor[:int(train_fraction * n_events)]
    val_tensor = truth_tensor[int(train_fraction * n_events):int((train_fraction+val_fraction) * n_events)]
    test_tensor = truth_tensor[int((train_fraction+val_fraction) * n_events):]
    return train_tensor, val_tensor, test_tensor


def split_pflow_targets(target: dict[str, torch.Tensor], train_fraction: float = 0.7, val_fraction: float = 0.2, ):
    """Split the target dict into train and validation parts along the event axis.

    Uses a contiguous event-axis split matching :func:`split_truth_objects`.
    ``seed`` is accepted for API compatibility but is intentionally unused.

    Returns
    -------
    dict[str, torch.Tensor], dict[str, torch.Tensor]
        The train and validation target dicts, each with the same keys as the
        input and tensors with shape ``(n_events_subset, ...)``.
    """
    n_events = next(iter(target.values())).shape[0]  # number of events from any observable
    split_idx = int(train_fraction * n_events)
    train_target = {k: v[:int(train_fraction * n_events)] for k, v in target.items()}
    val_target = {k: v[int(train_fraction * n_events):int((train_fraction+val_fraction)*n_events)] for k, v in target.items()}
    test_target = {k: v[int((train_fraction+val_fraction)*n_events):] for k, v in target.items()}
    return train_target, val_target, test_target


def split_truth_objects_jagged(
    truth_ragged: list[torch.Tensor], train_fraction: float = 0.7, val_fraction: float = 0.2, 
):
    """Ragged counterpart of :func:`split_truth_objects`.

    Uses a contiguous event-axis split: the first ``train_fraction`` block goes
    to train and the tail goes to validation. ``seed`` is accepted for API
    compatibility but is intentionally unused.
    """
    n_events = len(truth_ragged)
    split_idx = int(train_fraction * n_events)
    train_tensor = truth_ragged[:int(train_fraction * n_events)]
    val_tensor = truth_ragged[int(train_fraction * n_events):int((train_fraction+val_fraction) * n_events)]
    test_tensor = truth_ragged[int((train_fraction+val_fraction) * n_events):]
    return train_tensor, val_tensor, test_tensor


def split_pflow_targets_jagged(
    target: dict, train_fraction: float = 0.7, val_fraction: float = 0.2, 
):
    """Ragged counterpart of :func:`split_pflow_targets`.

    Uses a contiguous event-axis split matching
    :func:`split_truth_objects_jagged`: per-particle list entries are sliced by
    event index and per-event tensors are sliced on dim 0. ``seed`` is accepted
    for API compatibility but is intentionally unused.
    """
    n_events = target["multiplicity"].shape[0]  # full event count (per-event tensor)
    split_idx = int(train_fraction * n_events)

    def _split_one(v, start: int, end: int):
        if isinstance(v, list):
            return v[start:end]
        return v[start:end]

    train_target = {k: _split_one(v, 0, int(train_fraction * n_events)) for k, v in target.items()}
    val_target = {k: _split_one(v, int(train_fraction * n_events), int((train_fraction+val_fraction) * n_events)) for k, v in target.items()}
    test_target = {k: _split_one(v, int((train_fraction+val_fraction) * n_events), n_events) for k, v in target.items()}
    return train_target, val_target, test_target
