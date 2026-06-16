"""Differentiable distribution-matching losses for ``tune_cms_fullsim``.

This module is the package's self-contained loss layer:

- :func:`per_event_wasserstein_loss` is the active training loss: a sliced
  Wasserstein-2 distance computed per particle type over the standardized
  ``[log_E, log_pt, eta]`` object clouds, plus a down-weighted event-level term
  on the ``log(HT)`` distribution. It uses POT's
  ``ot.sliced_wasserstein_distance`` and stays differentiable w.r.t. the
  trainee card's predicted object kinematics.
- :func:`soft_histogram` and :func:`histogram_mse_loss` are fully-differentiable
  histogram primitives. They are no longer part of the training loss; they
  remain only as the diagnostic that
  :mod:`tune_cms_fullsim.intermediate_plots` uses to annotate each per-epoch
  plot with a soft-histogram MSE. They are an exact COPY of the same-named
  functions in :mod:`parnassus.torch_delphes.tuning` (keep the two in sync if
  you ever change the soft-histogram math).
"""

from __future__ import annotations

import os

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
# Active training loss: per-event sliced Wasserstein
# =============================================================================


def per_event_wasserstein_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    event_weight: float = EVENT_WEIGHT,
) -> torch.Tensor:
    """Sliced Wasserstein-2 loss between predicted and target observables.

    One SW_2 term per particle type (``pid``) over the standardized 3-D
    ``[log_E, log_pt, eta]`` object clouds, plus a down-weighted event-level
    SW_2 term on the per-event ``log(HT)`` distribution, plus the per-species
    expected-count terms. The target side is detached; gradients flow back to
    the trainee card through ``pred``.

    ``count_weight`` scales every per-species count term and ``event_weight``
    scales the per-event ``log(HT)`` term, both relative to the unit-weighted
    per-pid object terms. They default to the module constants
    :data:`COUNT_WEIGHT` / :data:`EVENT_WEIGHT` and are surfaced on the CLI as
    ``--count-weight`` / ``--event-weight``.
    """

    object_level_observables = ["log_E", "log_pt", "eta", "pid"]

    pred_particles = torch.stack([pred[k] for k in object_level_observables], dim=-1) # shape (n_events, max_n_particles, n_observables)
    target_particles = torch.stack([target[k] for k in object_level_observables], dim=-1)

    def get_object_groups(input_tensor: torch.Tensor) -> dict[int, torch.Tensor]:
        """Flatten ``(n_events, max_n_particles, n_obs)`` -> ``(N, n_obs)``, drop
        padding / efficiency-killed slots (``pid == 0``), and group the valid
        particles by integer ``pid``.

        The last column is ``pid`` (discrete, off the autograd graph); columns
        ``0:3`` are the differentiable features ``[log_E, log_pt, eta]``. Boolean-
        mask indexing is a differentiable gather, so on the pred side the returned
        feature tensors keep the gradient back to the learnable detector params.

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

    # Group both sides across the whole batch. Pred keeps its graph; the target is a
    # fixed reference, so detach it.
    pred_groups = get_object_groups(pred_particles)
    target_groups = get_object_groups(target_particles.detach())

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

    # Differentiable per-species expected-count terms: the gradient source for the
    # tracking-efficiency eff_logits. For each species, pred[...] is the trainee's
    # differentiable expected reconstructed count per RECO bin (from its reco-bin <-
    # pre-reco-region migration; see CMSEnergyFlowDefault._expected_reco_counts), and
    # target[...] is the per-event reconstructed-DATA count of that species in the same
    # reco bins. We match the batch totals; the fixed point is the true per-region
    # efficiency (no truth information is used -- only the trainee's own reco and the
    # reco data).
    #
    # Weighting is a NORMALIZED chi^2 / Poisson: per region, (pred - tgt)^2 / (tgt + 1)
    # (the +1 floor keeps an empty bin finite), then the per-species term is the bin SUM
    # divided by that species' total target count. Dividing the chi^2 sum by the total
    # count makes each term a *population-weighted squared relative error*
    # (sum_b (O_b/sum O) * ((pred_b - O_b)/O_b)^2): it KEEPS the chi^2 dense/sparse
    # cross-bin balancing (dense charged-hadron bins still dominate sparse lepton bins by
    # population) but is dimensionless and batch-size invariant -- O(1) at init and on the
    # same scale as the z-scored Wasserstein terms, instead of the old bin-MEAN chi^2 that
    # was extensive (~f^2 * counts ∝ n_events) and blew up to ~1e4 early in training. The
    # tracking count terms are gradient-decoupled from every other parameter (migration M
    # is gradient-free, eff_logits get gradient ONLY here); the calo count terms also feed
    # the resolution coefficients, which is why their scale matters and is now controllable.
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
        chi2 = (pred_counts - tgt_counts) ** 2 / (tgt_counts + 1.0)
        total = tgt_counts.sum().clamp_min(1.0)  # detached scale -> batch-invariant
        count_terms.append(count_weight * (chi2.sum() / total))

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
