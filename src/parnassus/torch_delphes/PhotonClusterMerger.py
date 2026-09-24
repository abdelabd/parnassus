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
kinematics. ``merge_radius`` itself is a constant and receives NO gradient.
"""

import math

import torch
from torch import nn

from parnassus.data.particle_io import PT_MIN, ColumnMap

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
        if photons.shape[0] <= 1:
            return photons
        return self._merge(photons, self._assign_clusters(photons))

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

