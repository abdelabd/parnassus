"""Differentiable distribution-matching losses for ``tune_cms_fullsim``.

This module is the package's self-contained loss layer. Two training losses share
the same per-pid object grouping (:func:`_group_objects_by_pid`) and the same
per-species expected-count terms (:func:`_count_terms`):

- :func:`per_event_wasserstein_loss` (``--loss wasserstein``, the default): a sliced
  Wasserstein-2 distance computed per particle type over the standardized
  ``[log_E, log_pt, eta]`` object clouds, plus a down-weighted event-level term on the
  ``log(HT)`` distribution, plus the count terms. It uses POT's
  ``ot.sliced_wasserstein_distance`` and stays differentiable w.r.t. the trainee
  card's predicted object kinematics.
- :func:`per_pid_soft_hist_loss` (``--loss soft_hist``): the same structure but with
  the optimal-transport shape objective replaced by a per-pid, per-observable
  soft-histogram MSE over ``[log_E, log_pt, eta]`` (plus the ``log(HT)`` and count
  terms). Directly optimizes histogram shape on a fixed shared bin grid.

- :func:`soft_histogram` and :func:`histogram_mse_loss` are fully-differentiable
  histogram primitives used BOTH by :func:`per_pid_soft_hist_loss` and as the
  diagnostic that :mod:`tune_cms_fullsim.intermediate_plots` uses to annotate each
  per-epoch plot with a soft-histogram MSE. They are an exact COPY of the same-named
  functions in :mod:`parnassus.torch_delphes.tuning` (keep the two in sync if you ever
  change the soft-histogram math).
"""

from __future__ import annotations

import os
from typing import Callable

import ot
import torch

from .config import CALO_COUNT_TERM_KEYS, COUNT_TERM_KEYS

# Weight of each differentiable per-species expected-count term relative to the
# per-pid sliced-Wasserstein terms. The count term is now a dimensionless,
# batch-invariant quantity (chi^2-sum / total-count; see below), so it sits on the
# same O(1) scale as the z-scored Wasserstein terms and this weight is a meaningful
# balance knob. It also feeds the calo RESOLUTION coefficients (which simultaneously
# receive the Wasserstein energy-shape gradient), so the balance genuinely matters
# there -- not only for the Adam-insensitive eff_logits. Overridable per-call via
# the CLI --count-weight.
COUNT_WEIGHT = 0.5

# Weight of the CALO-resolution count terms (CALO_COUNT_TERM_KEYS: ecal_photon,
# hcal_neutral_hadron) -- kept SEPARATE from COUNT_WEIGHT because these terms have a
# fundamentally different job. The tracking-efficiency count terms (COUNT_TERM_KEYS)
# only need to nudge Adam-insensitive eff_logits and have correctly-signed Wasserstein
# partners, so COUNT_WEIGHT ~ O(1) is right. The calo terms, by contrast, must OUT-VOTE
# a *wrong-signed* Wasserstein gradient on the forward resolution coefficients
# (forward_c_E/forward_c_S/common_c_E -- reparameterized autograd is blind to towers
# leaving the significance cut). Under Adam (step ~ sign(grad)) only their RELATIVE
# magnitude matters, so the calo count term must dominate -- hence a larger default. A
# single shared weight cannot do both: raising COUNT_WEIGHT enough to fix the calo sign
# also inflates the noisy lepton-efficiency count terms (observed to blow the val_loss
# up and trigger premature early-stopping). Overridable per-call via --calo-count-weight.
CALO_COUNT_WEIGHT = 10.0

# Weight of the per-event log(HT) sliced-Wasserstein term relative to the per-pid
# object terms. Overridable per-call via the CLI --event-weight.
EVENT_WEIGHT = 0.1

# =============================================================================
# Differentiable histogram primitives (diagnostic only; copied from tuning.py)
# =============================================================================


def soft_histogram(
    values: torch.Tensor,
    bin_edges: torch.Tensor,
    beta: float = 0.05,
) -> torch.Tensor:
    """Sigmoid-based soft histogram that is fully differentiable in
    ``values``.

    For each bin ``[lo, hi]`` and each value ``x``, the contribution is

    .. code-block::

        sigmoid((x - lo) / (beta * width)) - sigmoid((x - hi) / (beta * width))

    which approaches the hard indicator function as ``beta -> 0`` but
    retains a smooth gradient for any ``beta > 0``. The output is summed
    across all input values.

    Parameters
    ----------
    values : torch.Tensor
        1-D tensor of observable values, shape ``(N,)``. Values that
        fall outside ``[bin_edges[0], bin_edges[-1]]`` contribute very
        little to the histogram.
    bin_edges : torch.Tensor
        1-D tensor of ``n_bins + 1`` bin edges, strictly increasing.
    beta : float
        Relative softness, as a fraction of the per-bin width. Small
        values (0.01 to 0.05) give a near-hard histogram with still-
        well-conditioned gradients; larger values (0.1 to 0.3) give
        smoother gradients at the cost of bin overlap.

    Returns
    -------
    torch.Tensor
        A ``(n_bins,)`` tensor of soft counts. Not normalized; divide
        by ``.sum()`` to convert to a probability density.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {tuple(values.shape)}")
    if bin_edges.ndim != 1 or bin_edges.numel() < 2:
        raise ValueError("bin_edges must be 1-D with at least 2 entries")

    widths = (bin_edges[1:] - bin_edges[:-1]).clamp(min=1e-12)
    scale = beta * widths  # per-bin softness in the same units as `values`
    lo = bin_edges[:-1]  # (n_bins,)
    hi = bin_edges[1:]

    # values: (N,), lo/hi/scale: (n_bins,) -> broadcast to (N, n_bins)
    v = values.to(bin_edges.dtype).unsqueeze(1)  # (N, 1)
    scale_r = scale.unsqueeze(0)  # (1, n_bins)
    soft_lo = torch.sigmoid((v - lo.unsqueeze(0)) / scale_r)
    soft_hi = torch.sigmoid((v - hi.unsqueeze(0)) / scale_r)
    contrib = soft_lo - soft_hi  # (N, n_bins)
    return contrib.sum(dim=0)


def histogram_mse_loss(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    bin_edges: torch.Tensor,
    beta: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE between two soft histograms, both normalized to densities.

    Used by :mod:`tune_cms_fullsim.intermediate_plots` as a per-epoch
    distribution-mismatch diagnostic (it is not part of the training loss).

    Returns
    -------
    torch.Tensor
        Scalar MSE loss between the two normalized histograms.
    """
    pred_hist = soft_histogram(pred_values, bin_edges, beta=beta)
    target_hist = soft_histogram(target_values.detach(), bin_edges, beta=beta)
    pred_norm = pred_hist / (pred_hist.sum() + eps)
    target_norm = target_hist / (target_hist.sum() + eps)
    return ((pred_norm - target_norm) ** 2).mean()


# =============================================================================
# Shared object-grouping and count-term helpers (used by both training losses)
# =============================================================================

# Column order of the per-object feature stack used by the object-level losses; the
# trailing ``pid`` is the discrete grouping key (off the autograd graph). Shared by
# per_event_wasserstein_loss and per_pid_soft_hist_loss so the two agree on layout.
OBJECT_LEVEL_OBSERVABLES: list[str] = ["log_E", "log_pt", "eta", "pid"]

# Column index of each differentiable feature within a grouped ``(n, 3)`` object
# tensor (the first three columns of OBJECT_LEVEL_OBSERVABLES, in order).
_OBS_COL: dict[str, int] = {"log_E": 0, "log_pt": 1, "eta": 2}


def _group_objects_by_pid(input_tensor: torch.Tensor) -> dict[int, torch.Tensor]:
    """Flatten ``(n_events, max_n_particles, n_obs)`` -> ``(N, n_obs)``, drop
    padding / efficiency-killed slots (``pid == 0``), and group the valid particles
    by integer ``pid``.

    The last column is ``pid`` (discrete, off the autograd graph); columns ``0:3``
    are the differentiable features ``[log_E, log_pt, eta]``. Boolean-mask indexing
    is a differentiable gather, so on the pred side the returned feature tensors keep
    the gradient back to the learnable detector params.

    Returns ``dict`` pid -> ``(num_objects_with_that_pid, 3)`` feature tensor.
    """
    flat = input_tensor.reshape(-1, input_tensor.shape[-1])  # (N, n_obs)
    pid_col = flat[..., -1]  # (N,) pid, no grad
    valid = pid_col != 0  # padding/ghost -> pid == 0
    flat = flat[valid]  # differentiable gather
    pid_valid = pid_col[valid]

    groups: dict[int, torch.Tensor] = {}
    for pid in torch.unique(pid_valid).tolist():
        sel = pid_valid == pid
        groups[int(pid)] = flat[sel][:, :3]  # [log_E, log_pt, eta], keeps grad
    return groups


def _count_terms(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float,
    calo_count_weight: float,
) -> list[torch.Tensor]:
    """Differentiable per-species expected-count terms, shared by both training
    losses: the gradient source for the tracking-efficiency eff_logits and the calo
    resolution coefficients.

    For each species, ``pred[pred_key]`` is the trainee's differentiable expected
    reconstructed count per RECO bin (from its reco-bin <- pre-reco-region migration;
    see CMSEnergyFlowDefault._expected_reco_counts), and ``target[tgt_key]`` is the
    per-event reconstructed-DATA count of that species in the same reco bins. We match
    the batch totals; the fixed point is the true per-region efficiency (no truth
    information is used -- only the trainee's own reco and the reco data).

    Both forms are NORMALIZED chi^2 / Poisson (per region, ``(pred - tgt)^2 / (tgt + 1)``;
    the +1 floor keeps an empty bin finite), dimensionless and batch-size invariant
    (O(1), unlike the old bin-MEAN chi^2 that was extensive ~f^2*counts and blew up to
    ~1e4). But the two ROLES need different cross-region weighting:

     - Tracking-efficiency terms (COUNT_TERM_KEYS): *population-weighted* squared
       relative error, ``chi2.sum() / total = sum_b (O_b/sum O)*((pred_b-O_b)/O_b)^2``.
       Each region is weighted by its population fraction O_b/sum(O). Correct here:
       dense charged-hadron bins SHOULD dominate sparse lepton bins, and the eff_logit
       gradients are correctly signed (migration M is gradient-free; eff_logits get
       gradient ONLY here).

     - Calo-resolution terms (CALO_COUNT_TERM_KEYS): *per-region-FAIR* squared relative
       error, ``mean_b ((pred_b-O_b)/(O_b+1))^2`` -- every region weighted EQUALLY
       (1/n_reg), NOT by population. forward_c_E/forward_c_S act ONLY in the forward
       |eta| region, whose population fraction O_fwd/sum(O) is tiny; the population
       weighting above silently divides their (already wrong-sign-fighting) gradient by
       the central-dominated total and lets the wrong-signed shape gradient win (Adam
       follows sign, not magnitude). Per-region fairness keeps the term O(1) and
       batch-invariant ((O+1)^2 ~ n_events^2 matches the numerator) while restoring the
       forward region's full leverage; together with the larger CALO_COUNT_WEIGHT it
       re-establishes the calo count term's dominance over that wrong-signed gradient.

    Returns a list of scalar terms (one per present species); each keeps the gradient
    to ``pred[pred_key]`` and the target side is detached.
    """
    calo_pred_keys = {pred_key for _o, pred_key, _t in CALO_COUNT_TERM_KEYS}
    count_terms: list[torch.Tensor] = []
    for _out_key, pred_key, tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        pred_counts = pred.get(pred_key)
        if pred_counts is None or tgt_key not in target:
            continue
        pred_counts = pred_counts.reshape(-1)  # (n_regions,), differentiable
        tgt_counts = (
            target[tgt_key]
            .reshape(-1, pred_counts.shape[0])
            .sum(dim=0)
            .detach()
            .to(device=pred_counts.device, dtype=pred_counts.dtype)
        )
        if pred_key in calo_pred_keys:
            # per-region-fair squared relative error (equal weight per region)
            rel = (pred_counts - tgt_counts) ** 2 / (tgt_counts + 1.0) ** 2
            count_terms.append(calo_count_weight * rel.mean())
        else:
            # population-weighted squared relative error
            chi2 = (pred_counts - tgt_counts) ** 2 / (tgt_counts + 1.0)
            total = tgt_counts.sum().clamp_min(1.0)  # detached scale -> batch-invariant
            count_terms.append(count_weight * (chi2.sum() / total))
    return count_terms


# =============================================================================
# Active training loss: per-event sliced Wasserstein
# =============================================================================


def per_event_wasserstein_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    event_weight: float = EVENT_WEIGHT,
) -> torch.Tensor:
    """Sliced Wasserstein-2 loss between predicted and target observables.

    One SW_2 term per particle type (``pid``) over the standardized 3-D
    ``[log_E, log_pt, eta]`` object clouds, plus a down-weighted event-level
    SW_2 term on the per-event ``log(HT)`` distribution, plus the per-species
    expected-count terms. The target side is detached; gradients flow back to
    the trainee card through ``pred``.

    ``count_weight`` scales the tracking-efficiency count terms
    (:data:`COUNT_TERM_KEYS`); ``calo_count_weight`` scales the calo-resolution
    count terms (:data:`CALO_COUNT_TERM_KEYS`) -- kept separate because the calo
    terms must out-vote a wrong-signed Wasserstein gradient on the forward
    resolution coefficients and so need a larger weight (see
    :data:`CALO_COUNT_WEIGHT`). ``event_weight`` scales the per-event ``log(HT)``
    term. All three are relative to the unit-weighted per-pid object terms,
    default to the module constants, and are surfaced on the CLI as
    ``--count-weight`` / ``--calo-count-weight`` / ``--event-weight``.
    """

    pred_particles = torch.stack(
        [pred[k] for k in OBJECT_LEVEL_OBSERVABLES], dim=-1
    )  # (n_events, max_n_particles, n_observables)
    target_particles = torch.stack(
        [target[k] for k in OBJECT_LEVEL_OBSERVABLES], dim=-1
    )

    # Group both sides across the whole batch. Pred keeps its graph; the target is a
    # fixed reference, so detach it.
    pred_groups = _group_objects_by_pid(pred_particles)
    target_groups = _group_objects_by_pid(target_particles.detach())

    def sliced_sw2(
        x_pred: torch.Tensor, y_tgt: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """SW_2 between a differentiable pred cloud ``x_pred`` (n_pred, d) and a
        reference target cloud ``y_tgt`` (n_tgt, d), standardized by ``scale`` (d,).

        ``scale`` is a gradient-free per-feature constant (so no axis dominates the
        random 1-D projections) supplied by the caller rather than computed from
        ``y_tgt`` here: a per-pid batch std collapses to ~0 when a rare pid has only
        one target particle in the batch (e.g. a single electron), and a tiny clamp
        floor then inflates the distance by ~1/floor. Returns a scalar SW_2 that
        keeps the gradient back to ``x_pred``. Works for the (n, 3) object clouds
        and the (n, 1) per-event clouds alike; SW on 1-D data is exact.
        """
        y = y_tgt.detach().to(device=x_pred.device, dtype=x_pred.dtype)
        return ot.sliced_wasserstein_distance(
            x_pred / scale, y / scale, n_projections=100, p=2, seed=0
        )

    # Per-feature scale for the object terms: the std of each of the 3 features
    # [log_E, log_pt, eta] over ALL valid target particles, pooled across pids.
    # Pooling (rather than a per-pid std) keeps the scale well-conditioned even when
    # a rare pid has a single target particle in the batch -- a per-pid std would be
    # 0 there and explode that term through the clamp floor.
    target_flat = target_particles.detach().reshape(-1, target_particles.shape[-1])
    target_valid = target_flat[target_flat[..., -1] != 0]  # drop padding/ghosts (pid == 0)
    if target_valid.shape[0] > 0:
        object_scale = (
            target_valid[:, :3]
            .std(dim=0, unbiased=False)
            .clamp(min=1e-2)
            .to(device=pred_particles.device, dtype=pred_particles.dtype)
        )
    else:
        object_scale = torch.ones(
            3, device=pred_particles.device, dtype=pred_particles.dtype
        )

    # Object-level term: one SW_2 per pid present on BOTH sides, over the 3
    # standardized kinematic features [log_E, log_pt, eta], all on the shared pooled
    # scale above.
    object_wasserstein_distance: dict[int, torch.Tensor] = {}
    for pid in sorted(set(pred_groups) & set(target_groups)):
        x = pred_groups[pid]  # (n_pred, 3), differentiable
        y = target_groups[pid]  # (n_tgt, 3), reference (detached inside sliced_sw2)
        if x.shape[0] == 0 or y.shape[0] == 0:  # nothing to match on one side
            continue
        object_wasserstein_distance[int(pid)] = sliced_sw2(x, y, object_scale)

    # Event-level term: SW_2 on the per-event log(HT) distribution, (n_events,) ->
    # (n_events, 1). log_ht is differentiable (built from pt). multiplicity is
    # intentionally excluded: it is a hard pt != 0 count with no gradient, so an OT
    # term on it would change the loss value but not drive training.
    event_wasserstein_distance: dict[str, torch.Tensor] = {}
    for key in ("log_ht",):
        pv = pred[key].reshape(-1, 1)
        tv = target[key].reshape(-1, 1)
        if pv.shape[0] == 0 or tv.shape[0] == 0:
            continue
        # Per-event observable: (n_events,) values, never degenerate; standardize by
        # its own std with a 1e-2 floor as belt-and-suspenders.
        event_scale = (
            tv.detach()
            .std(dim=0, unbiased=False)
            .clamp(min=1e-2)
            .to(device=pv.device, dtype=pv.dtype)
        )
        event_wasserstein_distance[key] = sliced_sw2(pv, tv, event_scale)

    # Differentiable per-species expected-count terms (tracking-efficiency eff_logits
    # and calo-resolution coefficients). See :func:`_count_terms` for the
    # population-weighted (tracking) vs per-region-fair (calo) normalization.
    count_terms = _count_terms(
        pred, target, count_weight=count_weight, calo_count_weight=calo_count_weight
    )

    # Sum every term -> the scalar loss the training loop back-props. The per-event
    # log_ht term is down-weighted by ``event_weight`` relative to the per-pid object
    # terms (scalar * tensor keeps the gradient).
    terms = (
        list(object_wasserstein_distance.values())
        + [event_weight * d for d in event_wasserstein_distance.values()]
        + count_terms
    )
    if not terms:  # degenerate empty batch: keep a graph-connected zero
        return pred_particles.sum() * 0.0

    # Opt-in per-component breakdown (set MCGEN_LOSS_DEBUG=1) -- the durable
    # replacement for the old `embed()`: confirms the count terms now sit on the
    # same O(1) scale as the per-pid object terms instead of dominating.
    if os.environ.get("MCGEN_LOSS_DEBUG"):
        obj = {p: float(v) for p, v in object_wasserstein_distance.items()}
        evt = {k: float(event_weight * v) for k, v in event_wasserstein_distance.items()}
        cnt = [float(c) for c in count_terms]
        print(
            f"[loss] object_sw2={obj} event(w)={evt} count(w)={cnt} "
            f"total={float(torch.stack(terms).sum()):.4f}",
            flush=True,
        )
    return torch.stack(terms).sum()


# =============================================================================
# Alternative training loss: per-observable soft-histogram MSE
# =============================================================================

DEFAULT_SOFT_HIST_BIN_EDGES: dict[str, torch.Tensor] = {
    "pt": torch.linspace(0.0, 200.0, 41, dtype=torch.float64),
    "eta": torch.linspace(-5.0, 5.0, 41, dtype=torch.float64),
    "log_pt": torch.linspace(-1.0, 6.0, 41, dtype=torch.float64),
    "log_E": torch.linspace(-1.0, 7.0, 41, dtype=torch.float64),
    "multiplicity": torch.linspace(0.0, 400.0, 41, dtype=torch.float64),
    "ht": torch.linspace(0.0, 2000.0, 41, dtype=torch.float64),
    "log_ht": torch.linspace(0.0, 8.0, 41, dtype=torch.float64),
}

# Relative weight of each per-pid kinematic observable in the per-pid soft-histogram
# loss. Mirrors the convention of the (removed) pooled soft-hist loss: eta -- the axis
# that most directly constrains acceptance/shape -- at unit weight, the two correlated
# energy axes down-weighted. Each pid contributes the full obs-weighted set, and pids
# are unit-weighted relative to each other (as in the per-pid Wasserstein loss).
DEFAULT_PID_HIST_OBS_WEIGHTS: dict[str, float] = {
    "log_E": 0.5,
    "log_pt": 0.5,
    "eta": 1.0,
}


def per_pid_soft_hist_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    event_weight: float = EVENT_WEIGHT,
    beta: float = 0.15,
    bin_edges: dict[str, torch.Tensor] | None = None,
    obj_weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Per-PID, per-observable soft-histogram MSE loss (feature parity with
    :func:`per_event_wasserstein_loss`).

    One soft-histogram MSE term for every particle type ``pid`` present on BOTH sides
    and every kinematic observable in ``(log_E, log_pt, eta)``, over a fixed shared bin
    grid; plus a down-weighted per-event ``log(HT)`` soft-histogram term; plus the same
    per-species expected-count terms as the Wasserstein loss (scaled by ``count_weight``
    / ``calo_count_weight``). The target side is detached; gradients flow back to the
    trainee card through ``pred``.

    The per-``(pid, obs)`` histograms are NORMALIZED to densities inside
    :func:`histogram_mse_loss`, so each term matches *shape only* -- exactly like the
    standardized per-pid Wasserstein object terms. Absolute multiplicity / membership is
    carried by the count terms, which is precisely why they are included. Consequently
    this loss does NOT fix the calo ``c_E``/``c_S`` membership bias any better than the
    Wasserstein loss does: it swaps the optimal-transport shape objective for a
    histogram shape objective on the same kinematics while keeping the identical
    count-term gradient on ``eff_logits`` and the calo resolution coefficients.

    ``bin_edges`` defaults to :data:`DEFAULT_SOFT_HIST_BIN_EDGES` -- a single fixed grid
    shared across all pids, so pred and target always sit on the same axis and the
    objective is stationary across batches (rare pids merely under-populate bins).
    ``obj_weights`` defaults to :data:`DEFAULT_PID_HIST_OBS_WEIGHTS`; ``beta`` is the
    soft-histogram softness (small -> near-hard bins with flatter gradients, large ->
    smoother gradients with more bin bleed). All three are overridable.

    DDP note: this is a per-rank loss (matching :func:`per_event_wasserstein_loss`); the
    scalar is averaged across ranks by the training loop. A DDP-synced per-pid variant
    would have to all-gather the union of present pids before reducing each
    ``(pid, obs)`` histogram -- deferred until a multi-rank run actually needs it.
    """
    bin_edges = bin_edges if bin_edges is not None else DEFAULT_SOFT_HIST_BIN_EDGES
    obj_weights = (
        obj_weights if obj_weights is not None else DEFAULT_PID_HIST_OBS_WEIGHTS
    )

    pred_particles = torch.stack(
        [pred[k] for k in OBJECT_LEVEL_OBSERVABLES], dim=-1
    )  # (n_events, max_n_particles, n_observables)
    target_particles = torch.stack(
        [target[k] for k in OBJECT_LEVEL_OBSERVABLES], dim=-1
    )

    # Group both sides by pid; pred keeps its graph, the target is a detached reference.
    pred_groups = _group_objects_by_pid(pred_particles)
    target_groups = _group_objects_by_pid(target_particles.detach())

    # Per-pid, per-observable soft-histogram SHAPE terms (normalized densities).
    pid_obs_terms: dict[str, torch.Tensor] = {}
    for pid in sorted(set(pred_groups) & set(target_groups)):
        x = pred_groups[pid]  # (n_pred, 3), differentiable
        y = target_groups[pid]  # (n_tgt, 3), reference (detached inside histogram_mse_loss)
        if x.shape[0] == 0 or y.shape[0] == 0:  # nothing to match on one side
            continue
        for obs in ("log_E", "log_pt", "eta"):
            col = _OBS_COL[obs]
            edges = bin_edges[obs].to(device=x.device, dtype=x.dtype)
            pid_obs_terms[f"{pid}:{obs}"] = float(
                obj_weights.get(obs, 1.0)
            ) * histogram_mse_loss(x[:, col], y[:, col], edges, beta=beta)

    # Event-level term: soft-histogram MSE on the per-event log(HT) distribution,
    # down-weighted by event_weight (mirrors the Wasserstein event term; multiplicity is
    # intentionally excluded -- a hard pt != 0 count with no gradient).
    event_term: torch.Tensor | None = None
    pv = pred["log_ht"].reshape(-1)
    tv = target["log_ht"].reshape(-1)
    if pv.numel() and tv.numel():
        edges = bin_edges["log_ht"].to(device=pv.device, dtype=pv.dtype)
        event_term = event_weight * histogram_mse_loss(pv, tv, edges, beta=beta)

    # Same per-species expected-count terms as the Wasserstein loss: the gradient source
    # for eff_logits and the calo resolution coefficients (the histogram shape terms are
    # count-blind, so these carry the absolute multiplicity / membership signal).
    count_terms = _count_terms(
        pred, target, count_weight=count_weight, calo_count_weight=calo_count_weight
    )

    terms = (
        list(pid_obs_terms.values())
        + ([event_term] if event_term is not None else [])
        + count_terms
    )
    if not terms:  # degenerate empty batch: keep a graph-connected zero
        return pred_particles.sum() * 0.0

    # Opt-in per-component breakdown (set MCGEN_LOSS_DEBUG=1), mirroring the Wasserstein
    # loss: confirms the count terms sit on the same O(1) scale as the per-pid hist terms.
    if os.environ.get("MCGEN_LOSS_DEBUG"):
        pidh = {k: float(v) for k, v in pid_obs_terms.items()}
        evt = float(event_term) if event_term is not None else 0.0
        cnt = [float(c) for c in count_terms]
        print(
            f"[loss] pid_hist={pidh} event(w)={evt:.4f} count(w)={cnt} "
            f"total={float(torch.stack(terms).sum()):.4f}",
            flush=True,
        )
    return torch.stack(terms).sum()


# =============================================================================
# Loss dispatcher
# =============================================================================


LossFn = Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor
]

LOSS_CHOICES: tuple[str, ...] = ("wasserstein", "soft_hist")


def get_loss_fn(name: str) -> LossFn:
    """Return the training loss callable selected by ``name``."""
    if name == "wasserstein":
        return per_event_wasserstein_loss
    if name == "soft_hist":
        return per_pid_soft_hist_loss
    raise ValueError(
        f"Unknown loss {name!r}. Valid choices: {LOSS_CHOICES}."
    )
