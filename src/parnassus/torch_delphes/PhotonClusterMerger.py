"""Greedy seed-cone merging of the eflow photon stream (CMS supercluster scale).

CMS particle flow clusters EM deposits at a dR ~ 0.05 supercluster scale,
while the differentiable card emits one eflow photon per surviving ECal
tower (0.0174 x 1 deg) -- a ~1.7x photon multiplicity excess and a missing
merged-cluster population in dense jet cores. This module emulates the
missing clustering step on the flat eflow photon stream, between the ECal
and the EFlowMerger (design: torch_delphes/.claude/docs/
photon_merger_fraction_design.md).

Algorithm (per event): photons are ranked by descending pt (ties broken by
row order); the highest-ranked unclaimed photon seeds a cluster and absorbs
every unclaimed photon within dR < merge_radius of the SEED position; repeat
until every photon is claimed. The cone is anchored to the seed, so a
cluster never exceeds diameter 2R (no chain snowballing); a photon in reach
of two seeds goes to the higher-pt one; an isolated photon is a cluster of
one and its row passes through bit-identical, so the merge is the identity
on sparse events.

Cluster assignment is discrete and computed under no_grad. Merged
four-vectors are on-graph sums over cluster members, so upstream parameters
(resolutions, scales, fractions) keep their gradients through the merged
kinematics. ``merge_radius`` itself is a constant and receives NO gradient
here (milestone M0); per the design it is fit only through the recounted
photon count term (milestone M1+).
"""

import math

import torch
from torch import nn

from parnassus.data.particle_io import PT_MIN, ColumnMap
from parnassus.torch_delphes.SimpleCalorimeter import (
    calo_count_eta_edges,
    calo_count_region_masks,
)

# Events per padded assignment chunk: bounds the (chunk, m, m) pairwise-dR
# memory without changing the result (clustering is per-event).
_CHUNK_EVENTS = 128


class PhotonClusterMerger(nn.Module):
    """Merge nearby eflow photons into superclusters with a fixed seed cone.

    Parameters
    ----------
    merge_radius: float
        Cone radius dR = sqrt(deta^2 + dphi^2) around each seed. Init 0.045
        from the greedy truth-photon calibration against CMS PF counts
        (design doc section 3.1).
    """

    def __init__(self, merge_radius: float = 0.045) -> None:
        super().__init__()
        if not merge_radius > 0:
            raise ValueError(f"merge_radius must be positive, got {merge_radius}")
        self.merge_radius = float(merge_radius)

    def forward(self, photons: torch.Tensor) -> torch.Tensor:
        """Merge a flat (N, N_FEATURES) photon stream; returns (N', N_FEATURES).

        Surviving rows are the cluster seeds, in their original order.
        Rows of multi-photon clusters carry the summed four-vector (E, PX,
        PY, PZ on-graph; PT/ETA/PHI/MASS rederived from it); all other
        columns are the seed's. Single-photon clusters pass through
        untouched.
        """
        return self.forward_with_assignment(photons)[0]

    def forward_with_assignment(
        self, photons: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Like forward, but also returns the owner map: owner[i] is the input
        row index of photon i's cluster seed (owner[i] == i marks a seed).
        Merged output rows are the seeds in ascending input-row order."""
        if photons.shape[0] <= 1:
            owner = torch.arange(photons.shape[0], device=photons.device)
            return photons, owner
        owner = self._assign_clusters(photons)
        return self._merge(photons, owner), owner

    @torch.no_grad()
    def assign_to_seeds(
        self,
        eta: torch.Tensor,
        phi: torch.Tensor,
        event: torch.Tensor,
        seed_eta: torch.Tensor,
        seed_phi: torch.Tensor,
        seed_event: torch.Tensor,
    ) -> torch.Tensor:
        """Assign each (eta, phi, event) point to the nearest seed of the same
        event within merge_radius; returns indices into the seed arrays, -1
        when no seed is in reach. Used for the virtual (sub-threshold) towers
        of the merged-count composition."""
        n_pts = eta.shape[0]
        out = torch.full((n_pts,), -1, dtype=torch.long, device=eta.device)
        n_seeds = seed_eta.shape[0]
        if n_pts == 0 or n_seeds == 0:
            return out
        # Shared per-event indexing across both point sets.
        _, ev = torch.unique(torch.cat([event, seed_event]), return_inverse=True)
        p_ev, s_ev = ev[:n_pts], ev[n_pts:]
        n_events = int(ev.max()) + 1
        p_order = torch.argsort(p_ev, stable=True)
        s_order = torch.argsort(s_ev, stable=True)
        p_counts = torch.bincount(p_ev, minlength=n_events)
        s_counts = torch.bincount(s_ev, minlength=n_events)

        p_start = s_start = 0
        for ev0 in range(0, n_events, _CHUNK_EVENTS):
            pc = p_counts[ev0 : ev0 + _CHUNK_EVENTS]
            sc = s_counts[ev0 : ev0 + _CHUNK_EVENTS]
            np_, ns = int(pc.sum()), int(sc.sum())
            p_rows = p_order[p_start : p_start + np_]
            s_rows = s_order[s_start : s_start + ns]
            p_start += np_
            s_start += ns
            if np_ == 0 or ns == 0:
                continue
            out[p_rows] = self._nearest_in_chunk(
                eta[p_rows], phi[p_rows], pc,
                seed_eta[s_rows], seed_phi[s_rows], sc, s_rows,
            )
        return out

    def _nearest_in_chunk(self, p_eta, p_phi, p_counts, s_eta, s_phi, s_counts, s_rows):
        """Nearest same-event seed within merge_radius for one event chunk;
        both point sets arrive event-grouped. Returns global seed indices or -1."""
        device = p_eta.device
        b = p_counts.shape[0]
        mp, ms = int(p_counts.max()), int(s_counts.max())

        def layout(counts, n):
            ev_of = torch.repeat_interleave(torch.arange(b, device=device), counts)
            slot = torch.arange(n, device=device) - (torch.cumsum(counts, 0) - counts)[ev_of]
            return ev_of, slot

        p_ev, p_slot = layout(p_counts, p_eta.shape[0])
        s_ev, s_slot = layout(s_counts, s_eta.shape[0])

        def padded(ev_of, slot, m, values, fill):
            out = torch.full((b * m,), fill, dtype=values.dtype, device=device)
            out[ev_of * m + slot] = values
            return out.view(b, m)

        se = padded(s_ev, s_slot, ms, s_eta, torch.inf)  # inf -> never within radius
        sp = padded(s_ev, s_slot, ms, s_phi, 0.0)
        sr = padded(s_ev, s_slot, ms, s_rows.to(torch.float64), -1.0).long()

        deta = p_eta.view(-1, 1) - se[p_ev]                      # (n_pts, ms)
        dphi = p_phi.view(-1, 1) - sp[p_ev]
        dphi = torch.remainder(dphi + math.pi, 2.0 * math.pi) - math.pi
        dr2 = deta.square() + dphi.square()
        best = dr2.argmin(dim=1)
        within = dr2.gather(1, best.unsqueeze(1)).squeeze(1) < self.merge_radius**2
        return torch.where(within, sr[p_ev, best], torch.full_like(best, -1))

    # ------------------------------------------------------------------
    # discrete cluster assignment (off-graph)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _assign_clusters(self, photons: torch.Tensor) -> torch.Tensor:
        """Return owner row index per photon (owner[i] == i marks a seed)."""
        n = photons.shape[0]
        event_ids = photons[:, ColumnMap.EVENT_NUMBER]
        _, event_idx = torch.unique(event_ids, return_inverse=True)
        # Rows grouped by event, original order preserved within each event.
        order = torch.argsort(event_idx, stable=True)
        counts = torch.bincount(event_idx)

        owner = torch.empty(n, dtype=torch.long, device=photons.device)
        start = 0
        for chunk_counts in torch.split(counts, _CHUNK_EVENTS):
            n_rows = int(chunk_counts.sum())
            rows = order[start : start + n_rows]
            local_owner = self._assign_chunk(photons[rows], chunk_counts)
            owner[rows] = rows[local_owner]
            start += n_rows
        return owner

    def _assign_chunk(self, chunk: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Greedy seed-cone assignment for one event-grouped chunk.

        ``chunk`` is (n_rows, N_FEATURES) with rows sorted by event;
        ``counts`` is the per-event row count. Returns chunk-local owner
        indices. The parallel round scheme below is exactly equivalent to
        the sequential greedy walk: photons whose cones do not interact
        are just resolved in the same round instead of one at a time.
        """
        device = chunk.device
        n_rows = chunk.shape[0]
        b = counts.shape[0]
        m = int(counts.max())

        event_of_row = torch.repeat_interleave(torch.arange(b, device=device), counts)
        slot = torch.arange(n_rows, device=device)
        slot = slot - (torch.cumsum(counts, dim=0) - counts)[event_of_row]
        flat = event_of_row * m + slot

        def padded(col: int, fill: float) -> torch.Tensor:
            out = torch.full((b * m,), fill, dtype=torch.float64, device=device)
            out[flat] = chunk[:, col]
            return out.view(b, m)

        pt = padded(ColumnMap.PT, float("-inf"))  # padding ranks last
        eta = padded(ColumnMap.ETA, 0.0)
        phi = padded(ColumnMap.PHI, 0.0)
        valid = torch.zeros(b * m, dtype=torch.bool, device=device)
        valid[flat] = True
        valid = valid.view(b, m)

        # rank[b, i] = position of slot i in descending-pt order; stable sort
        # breaks pt ties by row order, so ranks are unique per event.
        rank = torch.argsort(torch.argsort(pt, dim=1, descending=True, stable=True), dim=1)

        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dphi = torch.remainder(dphi + math.pi, 2.0 * math.pi) - math.pi
        within = deta.square() + dphi.square() < self.merge_radius**2
        # can_claim[b, i, j]: i may absorb j (in cone, i outranks j, both real)
        can_claim = (
            within
            & (rank.unsqueeze(2) < rank.unsqueeze(1))
            & valid.unsqueeze(2)
            & valid.unsqueeze(1)
        )

        unclaimed = valid.clone()
        owner_slot = torch.arange(m, device=device).expand(b, m).clone()  # default: self
        rank_f = rank.to(torch.float64).unsqueeze(2)
        inf = torch.tensor(float("inf"), dtype=torch.float64, device=device)
        for _ in range(m):
            if not unclaimed.any():
                break
            # A photon seeds iff no unclaimed higher-pt photon has it in cone.
            blocked = (can_claim & unclaimed.unsqueeze(2)).any(dim=1)
            seeds = unclaimed & ~blocked
            grab = can_claim & seeds.unsqueeze(2) & unclaimed.unsqueeze(1)
            # A photon in reach of several seeds goes to the highest-pt one.
            grabbed = grab.any(dim=1)
            best_seed = torch.where(grab, rank_f, inf).argmin(dim=1)
            owner_slot = torch.where(grabbed, best_seed, owner_slot)
            unclaimed = unclaimed & ~(seeds | grabbed)
        assert not unclaimed.any(), "greedy cone assignment did not converge"

        row_of = torch.full((b * m,), -1, dtype=torch.long, device=device)
        row_of[flat] = torch.arange(n_rows, device=device)
        return row_of.view(b, m)[event_of_row, owner_slot[event_of_row, slot]]

    # ------------------------------------------------------------------
    # on-graph kinematic combination
    # ------------------------------------------------------------------

    def _merge(self, photons: torch.Tensor, owner: torch.Tensor) -> torch.Tensor:
        n = photons.shape[0]
        is_seed = owner == torch.arange(n, device=photons.device)
        size = torch.zeros(n, dtype=torch.long, device=photons.device)
        size.index_add_(0, owner, torch.ones_like(owner))

        four_cols = [ColumnMap.E, ColumnMap.PX, ColumnMap.PY, ColumnMap.PZ]
        four = photons[:, four_cols]
        summed = torch.zeros_like(four).index_add_(0, owner, four)

        merged = photons[is_seed].clone()
        multi = (size > 1)[is_seed]  # rewrite only true merges: singletons stay bit-identical
        if bool(multi.any()):
            e, px, py, pz = summed[is_seed][multi].unbind(dim=1)
            pt = torch.sqrt(px.square() + py.square()).clamp_min(PT_MIN)
            eta = torch.asinh(pz / pt)
            phi = torch.atan2(py, px)
            mass_sq = (e.square() - px.square() - py.square() - pz.square()).clamp_min(0.0)
            merged[multi, ColumnMap.E] = e
            merged[multi, ColumnMap.PX] = px
            merged[multi, ColumnMap.PY] = py
            merged[multi, ColumnMap.PZ] = pz
            merged[multi, ColumnMap.PT] = pt
            merged[multi, ColumnMap.ETA] = eta
            merged[multi, ColumnMap.PHI] = phi
            merged[multi, ColumnMap.MASS] = torch.sqrt(mass_sq)
            # Outer position mirrors the momentum direction, as the calo sets it.
            # Detached: non-fitted readout column (the tower_time rule).
            merged[multi, ColumnMap.ETA_OUTER] = eta.detach()
            merged[multi, ColumnMap.PHI_OUTER] = phi.detach()
        return merged


def compose_merged_photon_count(
    export: dict,
    owner: torch.Tensor,
    merged: torch.Tensor,
    merger: PhotonClusterMerger,
    calo,
) -> torch.Tensor:
    """Merged-photon replacement for the calo's per-region expected count.

    Generalizes compute_soft_count from towers to clusters: a merged photon
    exists iff at least one member tower survives, so per cluster c

        S_c = [1 - prod_{i in c} (1 - g_i)] * G_pt(c)

    with g_i the tower survival gates (live via sigma_after_c -- the c_E/c_S
    count-gradient channel this composition must preserve) and G_pt the pt
    acceptance moved from tower to cluster level (a merged cluster can pass
    pt >= count_pt_min when its towers individually do not). Members are the
    merger's emitted-photon clusters plus VIRTUAL members: every sub-threshold
    tower joins the nearest seed within merge_radius, or forms a virtual
    singleton cluster (pure gradient, zero forward -- today's sub-threshold
    role). Hard mirror: a real cluster counts iff its merged ROW passes the pt
    cut (exact vs the loss-side acceptance); virtual clusters never count.
    Straight-through pins the forward value to that hard count.

    Size-1 clusters take a fast path S = gate_nopt * pt_sigmoid that is
    arithmetically identical to the legacy per-tower gate, so merge_radius -> 0
    reproduces compute_soft_count bit-for-bit. Multi-member complements use
    the saturation-safe log_gate export (a float64 sigmoid saturates to exactly
    1.0, which would otherwise zero all cluster mates' gradients).

    Parameters: ``export`` = the calo's per-tower count export; ``owner`` = the
    merger's owner map over the pre-merge photon rows; ``merged`` = the merged
    photon stream (rows = seeds ascending); ``calo`` needs attributes
    ``count_pt_min, count_tau_rel, count_abs_eta_max, is_ecal``.
    """
    emitted = export["emitted"]
    device = emitted.device
    n_photons = owner.shape[0]
    tower_of_photon = emitted.nonzero(as_tuple=True)[0]
    assert tower_of_photon.shape[0] == n_photons, "export/photon stream misaligned"

    seed_rows = (owner == torch.arange(n_photons, device=device)).nonzero(as_tuple=True)[0]
    n_clusters = seed_rows.shape[0]
    assert merged.shape[0] == n_clusters, "merged rows must be the cluster seeds"
    seed_tower = tower_of_photon[seed_rows]

    # Virtual members: sub-threshold towers join the nearest seed within R.
    virt_tower = (~emitted).nonzero(as_tuple=True)[0]
    v_cluster = merger.assign_to_seeds(
        export["eta"][virt_tower], export["phi"][virt_tower], export["event"][virt_tower],
        export["eta"][seed_tower], export["phi"][seed_tower], export["event"][seed_tower],
    )
    attached = v_cluster >= 0
    v_single = virt_tower[~attached]
    n_total = n_clusters + v_single.shape[0]

    mem_tower = torch.cat([tower_of_photon, virt_tower[attached], v_single])
    mem_cid = torch.cat([
        torch.searchsorted(seed_rows, owner),
        v_cluster[attached],
        n_clusters + torch.arange(v_single.shape[0], device=device),
    ])

    # Cluster survival: singletons reuse the legacy gate arithmetic (bit-exact
    # R->0 reduction); multi-member clusters use saturation-safe complements.
    sizes = torch.bincount(mem_cid, minlength=n_total)
    gate_sum = torch.zeros(n_total, dtype=torch.float64, device=device).index_add_(
        0, mem_cid, export["gate_nopt"][mem_tower]
    )
    complements = (-torch.expm1(export["log_gate_nopt"][mem_tower])).clamp_min(1e-300)
    log_miss = torch.zeros(n_total, dtype=torch.float64, device=device).index_add_(
        0, mem_cid, torch.log(complements)
    )
    survival = torch.where(sizes == 1, gate_sum, 1.0 - torch.exp(log_miss))

    hard = torch.zeros(n_total, dtype=torch.bool, device=device)
    hard[:n_clusters] = True
    if calo.count_pt_min is not None:
        pt_sum = torch.zeros(n_total, dtype=torch.float64, device=device).index_add_(
            0, mem_cid, export["pt_soft"][mem_tower]
        )
        survival = survival * torch.sigmoid(
            (pt_sum - calo.count_pt_min) / (calo.count_tau_rel * calo.count_pt_min)
        )
        if n_clusters > 0:
            hard[:n_clusters] = merged[:, ColumnMap.PT].detach() >= calo.count_pt_min

    cluster_abs_eta = torch.cat([
        export["abs_eta_center"][seed_tower], export["abs_eta_center"][v_single]
    ])
    region_masks = calo_count_region_masks(
        cluster_abs_eta,
        calo_count_eta_edges(calo.is_ecal, calo.count_abs_eta_max),
        calo.count_abs_eta_max,
    )
    st = hard.to(survival.dtype).detach() + (survival - survival.detach())
    return torch.stack([
        export["anchor"] + (st * m.to(st.dtype)).sum() for m in region_masks
    ])
