"""Differentiable distribution-matching losses for ``tune_cms_fullsim``.

This module is the package's self-contained loss layer. Three training losses share
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
- :func:`per_pid_wasserstein_1d_loss` (``--loss wasserstein_1d``): the same per-pid /
  per-observable scaffolding as ``soft_hist`` -- one term per ``(pid, obs)``, the
  ``log(HT)`` term and the count terms -- but each shape term is the exact 1D
  Wasserstein-``p`` distance between the two point clouds computed via quantile
  interpolation (:func:`quantile_wasserstein_distance`). It is BIN-FREE (no histogram,
  no fixed range, no softness) yet still DETERMINISTIC (no random projections, unlike the
  point-cloud sliced Wasserstein). The bin-free answer to "do not manually decide the
  bin".

  Both ``per_pid_*`` losses share their entire body via :func:`_per_pid_obs_loss`,
  differing only in the per-observable ``term_fn`` (a soft-hist MSE vs the quantile W).

- :func:`soft_histogram` and :func:`histogram_mse_loss` are fully-differentiable
  histogram primitives used BOTH by :func:`per_pid_soft_hist_loss` and as the diagnostic
  that :mod:`tune_cms_fullsim.intermediate_plots` uses to annotate each per-epoch plot
  with a soft-histogram MSE. They are an exact COPY of the same-named functions in
  :mod:`parnassus.torch_delphes.tuning` (keep the two in sync if you ever change the
  soft-histogram math). :func:`quantile_wasserstein_distance` is the bin-free 1D
  Wasserstein primitive used by :func:`per_pid_wasserstein_1d_loss`.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import ot
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather as diff_all_gather
from torch.distributed.nn.functional import all_reduce as diff_all_reduce

from .config import CALO_COUNT_TERM_KEYS, COUNT_TERM_KEYS
from .distributed import _is_dist

# Weight of each differentiable per-species expected-count term relative to the
# per-pid sliced-Wasserstein terms. The count term is now a dimensionless,
# batch-invariant quantity (chi^2-sum / total-count; see below), so it sits on the
# same O(1) scale as the z-scored Wasserstein terms and this weight is a meaningful
# balance knob. It also feeds the calo RESOLUTION coefficients (which simultaneously
# receive the Wasserstein energy-shape gradient), so the balance genuinely matters
# there -- not only for the Adam-insensitive eff_logits. Overridable per-call via
# the CLI --count-weight.
COUNT_WEIGHT = 0.1

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
CALO_COUNT_WEIGHT = 1.0

# Weight of the per-event log(HT) sliced-Wasserstein term relative to the per-pid
# object terms. Overridable per-call via the CLI --event-weight.
EVENT_WEIGHT = 0.1

# Per-event-RATE floor in the count-term Pearson denominators (replaces the old
# constant ``+1`` count floor). The count terms are evaluated on per-event RATES
# (counts divided by the batch's event count N); this floor is what makes them
# genuinely batch-size INVARIANT. The old ``+ 1.0`` count floor was a constant that
# did not scale with N, so its effective rate floor ``1/N`` SHRANK as the batch grew
# and any region whose data count was ~0 while the trainee still predicted a count
# (which is ~proportional to N) blew up like N (tracking, after /total) to N^2 (calo):
# the source of the observed batch-size growth. A FIXED rate floor ``f`` keeps every
# region O(1) and batch-invariant: an empty region contributes ``(pred_rate)^2 / f``
# (resp. ``/ f^2``) with a batch-invariant ``pred_rate``, while a well-populated region
# (rate >> f) is essentially unchanged from the old behavior. ``0.05`` floors only the
# genuinely sparse regions (lepton bins, forward |eta| HCal neutral hadrons; rate <~
# 0.05/event) and leaves dense charged-hadron / photon bins (rate >> 1) untouched.
# Overridable per-call via the CLI --count-rate-floor.
COUNT_RATE_FLOOR = 0.05

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
# DDP gather helpers for set-to-set losses
# =============================================================================


def _all_gather_varlen(
    tensor_1d: torch.Tensor,
    *,
    differentiable: bool = False,
) -> torch.Tensor:
    """All-gather a 1-D tensor whose length may differ across ranks.

    Under DDP + ``DistributedSampler`` each rank processes a different shard,
    so the number of valid particles / events in a batch can differ.  This
    helper pads every rank's local tensor to the global maximum length,
    gathers the padded tensors, and then slices out the valid parts so that
    the caller receives the concatenation of every rank's real data.

    When ``differentiable=True`` the gather uses
    ``torch.distributed.nn.functional.all_gather`` so that gradients flow
    back from the combined loss to each rank's local forward pass.  For
    target-side data pass ``differentiable=False`` **and** detach before
    calling.

    Parameters
    ----------
    tensor_1d : torch.Tensor
        1-D tensor of local data on this rank.
    differentiable : bool
        If True, the gather stays on the autograd graph.

    Returns
    -------
    torch.Tensor
        Concatenation of the valid parts from all ranks (rank 0, rank 1,
        …).  If every rank's tensor is empty the result is an empty tensor
        of the same dtype and device.
    """
    world_size = dist.get_world_size()

    # ---- gather sizes -------------------------------------------------------
    local_size = torch.tensor(
        [tensor_1d.numel()], dtype=torch.int64, device=tensor_1d.device
    )
    size_list = [
        torch.zeros(1, dtype=torch.int64, device=tensor_1d.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(sizes)

    if max_size == 0:
        return torch.zeros(0, dtype=tensor_1d.dtype, device=tensor_1d.device)

    # ---- pad to global max --------------------------------------------------
    if tensor_1d.numel() < max_size:
        pad = torch.zeros(
            max_size - tensor_1d.numel(),
            dtype=tensor_1d.dtype,
            device=tensor_1d.device,
        )
        padded = torch.cat([tensor_1d, pad])
    else:
        padded = tensor_1d

    # ---- gather padded tensors ----------------------------------------------
    if differentiable:
        # ``diff_all_gather`` returns a tuple/list of per-rank tensors in some
        # PyTorch versions; ``torch.cat`` handles both.
        gathered_flat = torch.cat(diff_all_gather(padded))
    else:
        gathered_list = [
            torch.zeros(max_size, dtype=tensor_1d.dtype, device=tensor_1d.device)
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_list, padded)
        gathered_flat = torch.cat(gathered_list)  # (world_size * max_size,)

    # ---- slice out valid parts ----------------------------------------------
    parts: list[torch.Tensor] = []
    for r, sz in enumerate(sizes):
        if sz > 0:
            parts.append(gathered_flat[r * max_size : r * max_size + sz])

    if parts:
        return torch.cat(parts)
    return torch.zeros(0, dtype=tensor_1d.dtype, device=tensor_1d.device)


def quantile_wasserstein_distance(
    pred_values: torch.Tensor,
    target_values: torch.Tensor,
    scale: torch.Tensor | None = None,
    n_quantiles: int = 100,
    p: int = 2,
) -> torch.Tensor:
    """Exact 1D Wasserstein-p distance between two point clouds, via quantile
    (inverse-CDF) interpolation. BIN-FREE: no histogram, no fixed range, no softness.

    For two 1D distributions the Wasserstein-p distance is
    ``W_p^p = integral_0^1 |F_pred^{-1}(u) - F_tgt^{-1}(u)|^p du``. We approximate the
    quadrature by sampling both inverse-CDFs (quantile functions) at ``n_quantiles``
    midpoint levels ``u_k = (k + 0.5) / n_quantiles`` and averaging
    ``|Q_pred(u_k) - Q_tgt(u_k)|^p``. ``n_quantiles`` is a QUADRATURE resolution (the
    result is insensitive to it once ~100), NOT a binning of the value axis -- there is
    no range assumption and no mass is clipped, unlike a histogram.

    Returns ``W_p^p`` (the p-th power, e.g. the SQUARED W2 for the default ``p=2``):
    that is smooth (a quadratic bowl, gradient proportional to the quantile mismatch)
    and matches the squared/MSE convention of :func:`histogram_mse_loss`, whereas the
    rooted distance has a 1/sqrt gradient singularity as the two distributions coincide.

    Versus the point-cloud sliced Wasserstein (:func:`per_event_wasserstein_loss`): this
    is DETERMINISTIC (no random projections -> none of that instability) and exact in 1D,
    at ``O(N log N)``. Fully differentiable in ``pred_values`` (``torch.quantile``
    linearly interpolates the order statistics); the target side is detached. Handles
    unequal ``n_pred != n_tgt`` natively.

    Parameters
    ----------
    pred_values, target_values : torch.Tensor
        1-D (or flattenable) clouds of observable values. ``target_values`` is detached.
    scale : torch.Tensor or None
        Optional gradient-free per-observable scale (e.g. the pooled target std) that
        both clouds are divided by before the distance, so observables with different
        native spreads are comparable (mirrors ``object_scale`` in
        :func:`per_event_wasserstein_loss`). ``W_p^p(x/s, y/s) = W_p^p(x, y) / s^p``.
    n_quantiles : int
        Number of midpoint quantile levels (quadrature resolution).
    p : int
        Wasserstein order. ``p=2`` (default) returns the squared W2.

    Returns
    -------
    torch.Tensor
        Scalar ``W_p^p`` between the two clouds (on the ``scale``-standardized values).
    """
    x = pred_values.reshape(-1)
    y = target_values.detach().reshape(-1).to(device=x.device, dtype=x.dtype)
    if x.numel() == 0 or y.numel() == 0:  # nothing to match: graph-connected zero
        return x.sum() * 0.0
    if scale is not None:
        s = scale.to(device=x.device, dtype=x.dtype)
        x = x / s
        y = y / s
    u = (torch.arange(n_quantiles, device=x.device, dtype=x.dtype) + 0.5) / n_quantiles
    q_pred = torch.quantile(x, u)  # differentiable inverse-CDF (interpolated order stats)
    q_tgt = torch.quantile(y, u)
    diff = (q_pred - q_tgt).abs()
    return diff.mean() if p == 1 else (diff**p).mean()


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
    count_rate_floor: float = COUNT_RATE_FLOOR,
) -> list[torch.Tensor]:
    """Differentiable per-species expected-count terms, shared by both training
    losses: the gradient source for the tracking-efficiency eff_logits and the calo
    resolution coefficients.

    For each species, ``pred[pred_key]`` is the trainee's differentiable expected
    reconstructed count per RECO bin (from its reco-bin <- pre-reco-region migration;
    see CMSEnergyFlowDefault._expected_reco_counts), and ``target[tgt_key]`` is the
    per-event reconstructed-DATA count of that species in the same reco bins. Both are
    batch TOTALS (sum over the batch's events), so both scale ~proportional to the event
    count N. We convert them to per-event RATES (divide by N) and match the rates; the
    fixed point is the true per-region efficiency (no truth information is used -- only
    the trainee's own reco and the reco data).

    Both forms are a NORMALIZED relative chi^2 on the rates with a fixed per-event-rate
    floor ``f = count_rate_floor`` (per region, ``(p - o)^2 / (o + f)`` with rates
    ``p = pred/N``, ``o = tgt/N``). The RATE form + FIXED floor is what makes these terms
    genuinely batch-size INVARIANT and O(1). The previous version matched COUNTS with a
    constant ``+1`` floor and CLAIMED batch-invariance, but the ``+1`` was only a
    div-by-zero guard, not an invariant normalization: its effective rate floor ``1/N``
    shrinks with N, so any region whose data count was ~0 while the trainee still
    predicted a count (~proportional to N) contributed ``pred^2/1`` ~ N^2, which after the
    ``/total`` (~N) grew like N for tracking and up to N^2 for the un-normalized calo mean
    -- the observed batch-size blow-up (e.g. the sparse forward |eta| HCal neutral-hadron
    region drove ``count_terms[4]`` from 320 at batch 512 to 3645 at batch 4096). With a
    fixed rate floor, an empty region contributes ``p^2/f`` (resp. ``p^2/f^2``) with a
    batch-invariant rate ``p``, and a well-populated region (``o >> f``) is essentially
    unchanged from the old behavior. The two ROLES still need different cross-region
    weighting:

     - Tracking-efficiency terms (COUNT_TERM_KEYS): *population-weighted* squared
       relative error, ``chi2.sum() / total = sum_b (o_b/sum o)*((p_b-o_b)/o_b)^2``.
       Each region is weighted by its population fraction o_b/sum(o). Correct here:
       dense charged-hadron bins SHOULD dominate sparse lepton bins, and the eff_logit
       gradients are correctly signed (migration M is gradient-free; eff_logits get
       gradient ONLY here).

     - Calo-resolution terms (CALO_COUNT_TERM_KEYS): *per-region-FAIR* squared relative
       error, ``mean_b ((p_b-o_b)/(o_b+f))^2`` -- every region weighted EQUALLY
       (1/n_reg), NOT by population. forward_c_E/forward_c_S act ONLY in the forward
       |eta| region, whose population fraction o_fwd/sum(o) is tiny; the population
       weighting above silently divides their (already wrong-sign-fighting) gradient by
       the central-dominated total and lets the wrong-signed shape gradient win (Adam
       follows sign, not magnitude). Per-region fairness restores the forward region's
       full leverage; together with the larger CALO_COUNT_WEIGHT it re-establishes the
       calo count term's dominance over that wrong-signed gradient.

    Returns a list of scalar terms (one per present species); each keeps the gradient
    to ``pred[pred_key]`` and the target side (counts, N and the floor) is detached.
    """
    calo_pred_keys = {pred_key for _o, pred_key, _t in CALO_COUNT_TERM_KEYS}
    count_terms: list[torch.Tensor] = []
    for _out_key, pred_key, tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        pred_counts = pred.get(pred_key)
        if pred_counts is None or tgt_key not in target:
            continue
        pred_counts = pred_counts.reshape(-1)  # (n_regions,), differentiable
        tgt_per_event = target[tgt_key].reshape(-1, pred_counts.shape[0])
        n_events = max(int(tgt_per_event.shape[0]), 1)  # events in batch -> rate denom
        tgt_counts = (
            tgt_per_event.sum(dim=0)
            .detach()
            .to(device=pred_counts.device, dtype=pred_counts.dtype)
        )
        # Per-event rates make the statistic intensive; the fixed floor keeps it so.
        pred_rate = pred_counts / n_events  # differentiable
        tgt_rate = tgt_counts / n_events  # detached
        if pred_key in calo_pred_keys:
            # per-region-fair squared relative error (equal weight per region)
            rel = (pred_rate - tgt_rate) ** 2 / (tgt_rate + count_rate_floor) ** 2
            count_terms.append(calo_count_weight * rel.mean())
        else:
            # population-weighted squared relative error
            chi2 = (pred_rate - tgt_rate) ** 2 / (tgt_rate + count_rate_floor)
            total = tgt_rate.sum().clamp_min(count_rate_floor)  # detached -> invariant
            count_terms.append(count_weight * (chi2.sum() / total))
    return count_terms


# Per-pid shape-term weighting modes (CLI --pid-weighting). "equal" is the default and a
# bit-exact no-op; the others down-weight rare species by their population fraction.
PID_WEIGHTING_CHOICES: tuple[str, ...] = ("equal", "fraction", "sqrt_fraction")


def _pid_population_weights(
    target_groups: dict[int, torch.Tensor],
    present_pids: list[int],
    mode: str = "equal",
    floor: float = 0.0,
) -> dict[int, float]:
    """Mean-1-normalized per-pid weights on the per-pid SHAPE terms, from DETACHED target
    counts.

    Every per-pid object loss currently weights each particle type (pid) EQUALLY in the
    shape objective, regardless of abundance, so when the card starts far from target the
    rare species (muon ~0.2%, electron ~0.5%) cost the optimizer as much as the abundant
    charged/neutral hadrons and photons that dominate the reconstructed event. This helper
    optionally redistributes the shape weight by population fraction:

    - ``"equal"``  -> every present pid gets ``1.0`` (an EXACT no-op; default).
    - ``"fraction"``      -> ``g_p = f_p`` (aggressive; rare species ~100-250x lighter).
    - ``"sqrt_fraction"`` -> ``g_p = sqrt(f_p)`` (gentle; rare species ~8-20x lighter,
      still learnable -- the right choice when the lepton momentum-smearing params are
      being trained, since the shape term is their ONLY gradient path).

    ``f_p = n_p / sum_q n_q`` is computed over ``present_pids`` only (``n_p =
    target_groups[p].shape[0]``, the count that actually emits terms), so the weights are
    self-consistent with the loop. They are then normalized to MEAN 1 over the present pids,
    ``w_p = g_p * P / sum_q g_q`` (``P = len(present_pids)``), so the *aggregate* shape
    weight is preserved -- only redistributed across species -- and the shape-vs-count
    balance is unchanged. An optional ``floor`` clamps ``w_p`` from below (protecting a
    rare species' gradient in a low-stat batch) and RE-normalizes to keep the mean-1
    invariant.

    The counts are Python ints (``.shape[0]``), so every returned weight is a plain Python
    float OFF the autograd graph: multiplying a graph tensor by it keeps the gradient to
    ``pred`` and puts nothing on the weight, and ``"equal"`` stays a bit-exact ``1.0``.
    """
    if not present_pids:
        return {}
    if mode == "equal":
        return {int(p): 1.0 for p in present_pids}

    counts = {int(p): float(target_groups[p].shape[0]) for p in present_pids}
    total = sum(counts.values())
    if total <= 0:  # degenerate (shouldn't happen: present groups are non-empty)
        return {int(p): 1.0 for p in present_pids}

    if mode == "fraction":
        g = {p: counts[p] / total for p in counts}
    elif mode == "sqrt_fraction":
        g = {p: math.sqrt(counts[p] / total) for p in counts}
    else:
        raise ValueError(
            f"Unknown pid_weighting {mode!r}. Valid choices: {PID_WEIGHTING_CHOICES}."
        )

    p_count = len(g)

    def _normalize(gv: dict[int, float]) -> dict[int, float]:
        denom = sum(gv.values())
        if denom <= 0:
            return {p: 1.0 for p in gv}
        return {p: gv[p] * p_count / denom for p in gv}

    w = _normalize(g)
    if floor > 0.0:
        w = _normalize({p: max(w[p], floor) for p in w})  # clamp, then re-mean-1
    return w


# =============================================================================
# Active training loss: per-event sliced Wasserstein
# =============================================================================


def per_event_wasserstein_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    count_rate_floor: float = COUNT_RATE_FLOOR,
    event_weight: float = EVENT_WEIGHT,
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
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
    :data:`CALO_COUNT_WEIGHT`). ``count_rate_floor`` is the per-event-rate floor in
    the count-term denominators that makes them batch-size invariant (see
    :data:`COUNT_RATE_FLOOR`). ``event_weight`` scales the per-event ``log(HT)``
    term. All are relative to the unit-weighted per-pid object terms, default to the
    module constants, and are surfaced on the CLI as ``--count-weight`` /
    ``--calo-count-weight`` / ``--count-rate-floor`` / ``--event-weight``.

    ``pid_weighting`` (:data:`PID_WEIGHTING_CHOICES`) optionally redistributes the
    per-pid OBJECT terms by population fraction (see :func:`_pid_population_weights`);
    ``"equal"`` (default) is a no-op. ``pid_weight_floor`` sets a lower clamp on the
    per-pid weight. Neither touches the ``log(HT)`` or count terms.
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
        n_proj = 100 // dist.get_world_size() if _is_dist() else 100
        return ot.sliced_wasserstein_distance(
            x_pred / scale, y / scale, n_projections=n_proj, p=2, seed=0
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
    # scale above. Each pid's term is scaled by an optional population weight (default
    # 1.0; see _pid_population_weights) so rare species need not dominate the shape match.
    present_pids = [
        int(pid)
        for pid in sorted(set(pred_groups) & set(target_groups))
        if pred_groups[pid].shape[0] > 0 and target_groups[pid].shape[0] > 0
    ]
    pid_weights = _pid_population_weights(
        target_groups, present_pids, mode=pid_weighting, floor=pid_weight_floor
    )
    object_wasserstein_distance: dict[int, torch.Tensor] = {}
    for pid in present_pids:
        x = pred_groups[pid]  # (n_pred, 3), differentiable
        y = target_groups[pid]  # (n_tgt, 3), reference (detached inside sliced_sw2)
        object_wasserstein_distance[pid] = pid_weights[pid] * sliced_sw2(
            x, y, object_scale
        )

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
        pred,
        target,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
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


def per_event_wasserstein_loss_distributed(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    count_rate_floor: float = COUNT_RATE_FLOOR,
    event_weight: float = EVENT_WEIGHT,
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
) -> torch.Tensor:
    """DDP-aware wrapper: gathers pred/target across ranks before computing loss.

    The sliced Wasserstein distance is a *set-to-set* loss —
    SW_2(A ∪ B, C ∪ D) ≠ SW_2(A,C) + SW_2(B,D).  When each DDP rank computes
    the loss on its local shard and the results are averaged, the mathematical
    value differs from the non-distributed loss computed on the full batch.
    This wrapper fixes that by gathering every rank's predicted and target
    observables (via :func:`_all_gather_varlen`) so the loss is always computed
    on the union of all shards, making the DDP and single-process loss
    **identical** for the same underlying data.
    """
    if not _is_dist():
        return per_event_wasserstein_loss(
            pred,
            target,
            count_weight=count_weight,
            calo_count_weight=calo_count_weight,
            count_rate_floor=count_rate_floor,
            event_weight=event_weight,
            pid_weighting=pid_weighting,
            pid_weight_floor=pid_weight_floor,
        )

    pred_gathered: dict[str, torch.Tensor] = {}
    target_gathered: dict[str, torch.Tensor] = {}

    # ---- per-particle observables ------------------------------------------
    # Extract valid (non-padding, non-efficiency-killed) particles from each
    # rank, gather them into a single cloud, and repack as (1, n_total) so the
    # loss sees "one event" with all particles — the event structure is
    # irrelevant since the loss immediately flattens and groups by pid.
    pred_pid = pred["pid"]
    tgt_pid = target["pid"].detach()
    pred_valid = (pred_pid != 0).reshape(-1)
    tgt_valid = (tgt_pid != 0).reshape(-1)

    for key in ("log_E", "log_pt", "eta", "pid"):
        pv = pred[key].reshape(-1)[pred_valid]
        tv = target[key].reshape(-1)[tgt_valid]

        pred_gathered[key] = _all_gather_varlen(pv, differentiable=True).unsqueeze(0)
        target_gathered[key] = _all_gather_varlen(tv, differentiable=False).unsqueeze(0)

    # ---- per-event observable: log(HT) -------------------------------------
    if "log_ht" in pred and "log_ht" in target:
        pred_gathered["log_ht"] = _all_gather_varlen(
            pred["log_ht"].reshape(-1), differentiable=True
        )
        target_gathered["log_ht"] = _all_gather_varlen(
            target["log_ht"].detach().reshape(-1), differentiable=False
        )

    # ---- differentiable expected-count terms --------------------------------
    # Pred counts come from the card already *batch-aggregated* as (n_regions,),
    # so we all-reduce (sum) them across ranks.  Target counts are per-event
    # (n_events_rank, n_regions) — gather and concatenate so the loss sums
    # them over the full combined batch, matching the aggregated pred.
    for _out_key, pred_key, tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        if pred_key not in pred or tgt_key not in target:
            continue
        pv = pred[pred_key]                     # (n_regions,) — batch-aggregated
        tv = target[tgt_key].detach()           # (n_events_rank, n_regions)

        # Pred: all-reduce sum (same shape on every rank, preserves autograd).
        pred_gathered[pred_key] = diff_all_reduce(pv, op=dist.ReduceOp.SUM)

        # Target: flatten, gather variable-length, reshape back.
        n_regions = tv.shape[1]
        tv_flat = tv.reshape(-1)                # (n_events_rank * n_regions,)
        target_gathered[tgt_key] = _all_gather_varlen(
            tv_flat, differentiable=False
        ).reshape(-1, n_regions)                # (total_events, n_regions)

    return per_event_wasserstein_loss(
        pred_gathered,
        target_gathered,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
        event_weight=event_weight,
        pid_weighting=pid_weighting,
        pid_weight_floor=pid_weight_floor,
    )


# =============================================================================
# Alternative per-pid, per-observable shape losses
#   - per_pid_soft_hist_loss      (--loss soft_hist):    per-bin soft-histogram MSE
#   - per_pid_wasserstein_1d_loss (--loss wasserstein_1d): bin-free 1D quantile W
# Both share the per-pid scaffolding in _per_pid_obs_loss.
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


ObsTermFn = Callable[[torch.Tensor, torch.Tensor, str], torch.Tensor]


def _per_pid_obs_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    term_fn: ObsTermFn,
    count_weight: float,
    calo_count_weight: float,
    count_rate_floor: float,
    event_weight: float,
    obj_weights: dict[str, float] | None,
    debug_label: str,
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
) -> torch.Tensor:
    """Shared body of the per-pid, per-observable shape losses
    (:func:`per_pid_soft_hist_loss` and :func:`per_pid_wasserstein_1d_loss`).

    Identical structure for both: one ``term_fn`` shape term per particle type ``pid``
    present on BOTH sides and per observable in ``(log_E, log_pt, eta)``; plus a
    down-weighted per-event ``log(HT)`` term; plus the same per-species expected-count
    terms. The ONLY thing that varies between the public losses is the per-observable
    ``term_fn(pred_values, target_values, obs_key) -> scalar`` -- a soft-histogram MSE
    (:func:`histogram_mse_loss`) for ``soft_hist`` vs the bin-free 1D quantile
    Wasserstein (:func:`quantile_wasserstein_distance`) for ``wasserstein_1d``. Each
    wrapper builds its own ``term_fn`` closure carrying whatever the metric needs (fixed
    bin edges / ``beta`` for the histogram; the pooled per-observable scale for the
    quantile W). The target side is detached and gradients flow back to the trainee card
    through ``pred``. ``debug_label`` tags the optional ``MCGEN_LOSS_DEBUG`` breakdown.

    ``pid_weighting`` / ``pid_weight_floor`` (:func:`_pid_population_weights`) optionally
    scale each pid's shape terms by its population fraction; ``"equal"`` (default) is a
    no-op. Only the per-pid shape terms are weighted -- the ``log(HT)`` and count terms
    are untouched.
    """
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

    # Optional per-pid population weight on the SHAPE terms (default 1.0 each).
    present_pids = [
        int(pid)
        for pid in sorted(set(pred_groups) & set(target_groups))
        if pred_groups[pid].shape[0] > 0 and target_groups[pid].shape[0] > 0
    ]
    pid_weights = _pid_population_weights(
        target_groups, present_pids, mode=pid_weighting, floor=pid_weight_floor
    )

    # Per-pid, per-observable SHAPE terms.
    pid_obs_terms: dict[str, torch.Tensor] = {}
    for pid in present_pids:
        x = pred_groups[pid]  # (n_pred, 3), differentiable
        y = target_groups[pid]  # (n_tgt, 3), reference (detached inside term_fn)
        for obs in ("log_E", "log_pt", "eta"):
            col = _OBS_COL[obs]
            pid_obs_terms[f"{pid}:{obs}"] = (
                pid_weights[pid]
                * float(obj_weights.get(obs, 1.0))
                * term_fn(x[:, col], y[:, col], obs)
            )

    # Event-level term: the same shape distance on the per-event log(HT) distribution,
    # down-weighted by event_weight (mirrors the Wasserstein event term; multiplicity is
    # intentionally excluded -- a hard pt != 0 count with no gradient).
    event_term: torch.Tensor | None = None
    pv = pred["log_ht"].reshape(-1)
    tv = target["log_ht"].reshape(-1)
    if pv.numel() and tv.numel():
        event_term = event_weight * term_fn(pv, tv, "log_ht")

    # Same per-species expected-count terms as the Wasserstein loss: the gradient source
    # for eff_logits and the calo resolution coefficients (the shape terms are
    # count-blind, so these carry the absolute multiplicity / membership signal).
    count_terms = _count_terms(
        pred,
        target,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
    )
    terms = (
        list(pid_obs_terms.values())
        + ([event_term] if event_term is not None else [])
        + count_terms
    )
    if not terms:  # degenerate empty batch: keep a graph-connected zero
        return pred_particles.sum() * 0.0
    # Opt-in per-component breakdown (set MCGEN_LOSS_DEBUG=1), mirroring the Wasserstein
    # loss: confirms the count terms sit on the same O(1) scale as the per-pid shape terms.
    if os.environ.get("MCGEN_LOSS_DEBUG"):
        pidh = {k: float(v) for k, v in pid_obs_terms.items()}
        evt = float(event_term) if event_term is not None else 0.0
        cnt = [float(c) for c in count_terms]
        print(
            f"[loss] {debug_label}={pidh} event(w)={evt:.4f} count(w)={cnt} "
            f"total={float(torch.stack(terms).sum()):.4f}",
            flush=True,
        )
    return torch.stack(terms).sum()


def per_pid_soft_hist_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    count_rate_floor: float = COUNT_RATE_FLOOR,
    event_weight: float = EVENT_WEIGHT,
    beta: float = 0.15,
    bin_edges: dict[str, torch.Tensor] | None = None,
    obj_weights: dict[str, float] | None = None,
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
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
    smoother gradients with more bin bleed). All three are overridable. If the manual
    bin grid is itself the concern, see :func:`per_pid_wasserstein_1d_loss` for a
    bin-free alternative.

    DDP note: this is a per-rank loss (matching :func:`per_event_wasserstein_loss`); the
    scalar is averaged across ranks by the training loop. A DDP-synced per-pid variant
    would have to all-gather the union of present pids before reducing each
    ``(pid, obs)`` histogram -- deferred until a multi-rank run actually needs it.
    """
    bin_edges = bin_edges if bin_edges is not None else DEFAULT_SOFT_HIST_BIN_EDGES

    def term_fn(
        pred_values: torch.Tensor, target_values: torch.Tensor, obs: str
    ) -> torch.Tensor:
        edges = bin_edges[obs].to(device=pred_values.device, dtype=pred_values.dtype)
        return histogram_mse_loss(pred_values, target_values, edges, beta=beta)

    return _per_pid_obs_loss(
        pred,
        target,
        term_fn=term_fn,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
        event_weight=event_weight,
        obj_weights=obj_weights,
        debug_label="pid_hist",
        pid_weighting=pid_weighting,
        pid_weight_floor=pid_weight_floor,
    )


def _pooled_target_obs_std(
    target: dict[str, torch.Tensor], *, eps_floor: float = 1e-2
) -> dict[str, torch.Tensor]:
    """Pooled-across-pid per-observable std of the target, used to standardize the
    bin-free 1D Wasserstein terms so the kinematic axes (and the ``log(HT)`` event axis)
    are comparable -- mirrors the ``object_scale`` / ``event_scale`` of
    :func:`per_event_wasserstein_loss`. Gradient-free; POOLING across pid avoids the
    degenerate ~0 std a per-pid scale would give a rare single-particle species (which a
    tiny clamp floor would then turn into a blown-up distance).
    """
    target_particles = torch.stack(
        [target[k] for k in OBJECT_LEVEL_OBSERVABLES], dim=-1
    ).detach()
    flat = target_particles.reshape(-1, target_particles.shape[-1])
    valid = flat[flat[..., -1] != 0]  # drop pid == 0 padding/ghosts
    scales: dict[str, torch.Tensor] = {}
    for obs in ("log_E", "log_pt", "eta"):
        col = _OBS_COL[obs]
        if valid.shape[0] > 0:
            scales[obs] = valid[:, col].std(unbiased=False).clamp(min=eps_floor)
        else:
            scales[obs] = torch.ones((), dtype=target_particles.dtype)
    tv = target["log_ht"].reshape(-1).detach()
    scales["log_ht"] = (
        tv.std(unbiased=False).clamp(min=eps_floor)
        if tv.numel()
        else torch.ones((), dtype=tv.dtype)
    )
    return scales


def per_pid_wasserstein_1d_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    count_rate_floor: float = COUNT_RATE_FLOOR,
    event_weight: float = EVENT_WEIGHT,
    obj_weights: dict[str, float] | None = None,
    n_quantiles: int = 100,
    p: int = 2,
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
) -> torch.Tensor:
    """Per-PID, per-observable BIN-FREE 1D Wasserstein loss.

    Same scaffolding as :func:`per_pid_soft_hist_loss` (shared via
    :func:`_per_pid_obs_loss`: one term per ``(pid, obs)`` in ``(log_E, log_pt, eta)``,
    plus a down-weighted per-event ``log(HT)`` term, plus the same per-species
    expected-count terms), but each shape term is the exact 1D Wasserstein-``p`` distance
    between the two point clouds computed via quantile interpolation
    (:func:`quantile_wasserstein_distance`) -- NO histogram, NO fixed bin grid, NO range,
    NO ``beta``. ``n_quantiles`` (default 100) is only a quadrature resolution, not a
    binning of the value axis.

    This is the bin-free answer to "do not manually decide the bin": it keeps the two
    desirable properties of ``soft_hist`` (DETERMINISTIC -- no random projections, so
    none of the point-cloud sliced-Wasserstein instability; and a direct, batch-stable
    metric between the two empirical distributions) while removing the bin grid entirely.
    Each per-``(pid, obs)`` cloud is standardized by the pooled per-observable target std
    (:func:`_pooled_target_obs_std`, the same convention as ``object_scale`` in
    :func:`per_event_wasserstein_loss`) so the three kinematic axes stay comparable and
    ``obj_weights`` keep their meaning. ``p=2`` returns the squared W2 per term (smooth,
    matching the MSE convention of ``soft_hist``).

    Scale note: as with the other shape objectives, this term is count-blind, so the
    absolute multiplicity / membership is carried by the count terms. The standardized
    squared-W2 shape term is O(1) but its balance against the count terms differs from
    ``soft_hist``'s MSE; the defaults are kept for parity, but re-validate the
    count/shape balance with ``MCGEN_LOSS_DEBUG=1`` before a production fit -- in
    particular the calo count term must still out-vote the wrong-signed forward-resolution
    gradient (see :data:`CALO_COUNT_WEIGHT`).
    """
    scale_by_obs = _pooled_target_obs_std(target)

    def term_fn(
        pred_values: torch.Tensor, target_values: torch.Tensor, obs: str
    ) -> torch.Tensor:
        return quantile_wasserstein_distance(
            pred_values,
            target_values,
            scale=scale_by_obs.get(obs),
            n_quantiles=n_quantiles,
            p=p,
        )

    return _per_pid_obs_loss(
        pred,
        target,
        term_fn=term_fn,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
        event_weight=event_weight,
        obj_weights=obj_weights,
        debug_label="pid_w1d",
        pid_weighting=pid_weighting,
        pid_weight_floor=pid_weight_floor,
    )


def per_pid_wasserstein_1d_loss_distributed(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    *,
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    count_rate_floor: float = COUNT_RATE_FLOOR,
    event_weight: float = EVENT_WEIGHT,
    obj_weights: dict[str, float] | None = None,
    n_quantiles: int = 100,
    p: int = 2,
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
) -> torch.Tensor:
    """DDP-aware wrapper: gathers pred/target across ranks before computing the
    bin-free 1D Wasserstein loss.

    Like the sliced Wasserstein, the per-pid 1D Wasserstein distance is a
    set-to-set loss that does not commute with summation over disjoint shards.
    This wrapper gathers every rank's predicted and target observables so the
    loss is computed on the union of all shards, making DDP and single-process
    results identical.
    """
    if not _is_dist():
        return per_pid_wasserstein_1d_loss(
            pred,
            target,
            count_weight=count_weight,
            calo_count_weight=calo_count_weight,
            count_rate_floor=count_rate_floor,
            event_weight=event_weight,
            obj_weights=obj_weights,
            n_quantiles=n_quantiles,
            p=p,
            pid_weighting=pid_weighting,
            pid_weight_floor=pid_weight_floor,
        )

    pred_gathered: dict[str, torch.Tensor] = {}
    target_gathered: dict[str, torch.Tensor] = {}

    # ---- per-particle observables ------------------------------------------
    pred_pid = pred["pid"]
    tgt_pid = target["pid"].detach()
    pred_valid = (pred_pid != 0).reshape(-1)
    tgt_valid = (tgt_pid != 0).reshape(-1)

    for key in ("log_E", "log_pt", "eta", "pid"):
        pv = pred[key].reshape(-1)[pred_valid]
        tv = target[key].reshape(-1)[tgt_valid]

        pred_gathered[key] = _all_gather_varlen(pv, differentiable=True).unsqueeze(0)
        target_gathered[key] = _all_gather_varlen(tv, differentiable=False).unsqueeze(0)

    # ---- per-event observable: log(HT) -------------------------------------
    if "log_ht" in pred and "log_ht" in target:
        pred_gathered["log_ht"] = _all_gather_varlen(
            pred["log_ht"].reshape(-1), differentiable=True
        )
        target_gathered["log_ht"] = _all_gather_varlen(
            target["log_ht"].detach().reshape(-1), differentiable=False
        )

    # ---- differentiable expected-count terms --------------------------------
    for _out_key, pred_key, tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        if pred_key not in pred or tgt_key not in target:
            continue
        pv = pred[pred_key]
        tv = target[tgt_key].detach()

        pred_gathered[pred_key] = diff_all_reduce(pv, op=dist.ReduceOp.SUM)

        n_regions = tv.shape[1]
        tv_flat = tv.reshape(-1)
        target_gathered[tgt_key] = _all_gather_varlen(
            tv_flat, differentiable=False
        ).reshape(-1, n_regions)

    return per_pid_wasserstein_1d_loss(
        pred_gathered,
        target_gathered,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
        event_weight=event_weight,
        obj_weights=obj_weights,
        n_quantiles=n_quantiles,
        p=p,
        pid_weighting=pid_weighting,
        pid_weight_floor=pid_weight_floor,
    )


# =============================================================================
# Loss dispatcher
# =============================================================================


LossFn = Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor
]

LOSS_CHOICES: tuple[str, ...] = ("wasserstein", "soft_hist", "wasserstein_1d")


def get_loss_fn(name: str) -> LossFn:
    """Return the training loss callable selected by ``name``."""
    if name == "wasserstein":
        return per_event_wasserstein_loss_distributed
    if name == "soft_hist":
        return per_pid_soft_hist_loss
    if name == "wasserstein_1d":
        return per_pid_wasserstein_1d_loss_distributed
    raise ValueError(
        f"Unknown loss {name!r}. Valid choices: {LOSS_CHOICES}."
    )
