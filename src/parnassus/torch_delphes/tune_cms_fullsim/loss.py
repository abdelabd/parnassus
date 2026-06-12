"""Differentiable distribution-matching losses for ``tune_cms_fullsim``.

This module is the package's self-contained loss layer. Two training losses
are available; the active one is picked at runtime via ``--loss`` (CLI) or the
``loss_name`` argument of :func:`tune_cms_fullsim.training.fit_card_to_fullsim`:

- :func:`per_event_wasserstein_loss` — a sliced Wasserstein-2 distance computed
  per particle type over the standardized ``[log_E, log_pt, eta]`` object
  clouds, plus a down-weighted event-level term on the ``log(HT)``
  distribution. Uses POT's ``ot.sliced_wasserstein_distance`` and is
  differentiable w.r.t. the trainee card's predicted object kinematics.
- :func:`multi_observable_soft_hist_loss` (DDP-aware via
  :func:`multi_observable_soft_hist_loss_distributed`) — the original
  soft-histogram MSE loss used in the pre-Wasserstein training harness
  (``5cac599`` and earlier). It is a weighted sum of per-observable soft-
  histogram MSEs computed with :func:`soft_histogram` /
  :func:`histogram_mse_loss`; under DDP the per-rank histograms are summed
  with a *differentiable* all-reduce so the global loss equals the loss
  computed on the union of every rank's events.

The histogram primitives :func:`soft_histogram` and :func:`histogram_mse_loss`
are also used as a per-epoch diagnostic by
:mod:`tune_cms_fullsim.intermediate_plots` (each plot is annotated with the
per-observable soft-hist MSE), so they stay here regardless of which training
loss is selected.

Selecting the active loss
-------------------------

Use :func:`get_loss_fn` (or pass ``loss_name`` to
:func:`tune_cms_fullsim.training.fit_card_to_fullsim`):

.. code-block:: python

    loss_fn = get_loss_fn("soft_hist")         # or "wasserstein"
    loss = loss_fn(pred_observables, target_observables)
"""

from __future__ import annotations

from typing import Callable

import ot
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce as diff_all_reduce

from .distributed import _is_dist

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
) -> torch.Tensor:
    """Sliced Wasserstein-2 loss between predicted and target observables.

    One SW_2 term per particle type (``pid``) over the standardized 3-D
    ``[log_E, log_pt, eta]`` object clouds, plus a down-weighted event-level
    SW_2 term on the per-event ``log(HT)`` distribution. The target side is
    detached; gradients flow back to the trainee card through ``pred``.
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

    def sliced_sw2(x_pred: torch.Tensor, y_tgt: torch.Tensor) -> torch.Tensor:
        """SW_2 between a differentiable pred cloud ``x_pred`` (n_pred, d) and a
        reference target cloud ``y_tgt`` (n_tgt, d).

        Each feature is standardized by the detached target std so no axis
        dominates the random 1-D projections (the scale is a gradient-free
        constant). Returns a scalar SW_2 distance that keeps the gradient back to
        ``x_pred``. Works for the (n, 3) object clouds and the (n, 1) per-event
        clouds alike; SW on 1-D data is exact.
        """
        y = y_tgt.detach().to(device=x_pred.device, dtype=x_pred.dtype)
        scale = y.std(dim=0, unbiased=False).clamp(min=1e-8)  # (d,)
        return ot.sliced_wasserstein_distance(
            x_pred / scale, y / scale, n_projections=100, p=2, seed=0
        )

    # Object-level term: one SW_2 per pid present on BOTH sides, over the 3
    # standardized kinematic features [log_E, log_pt, eta].
    object_wasserstein_distance: dict[int, torch.Tensor] = {}
    for pid in sorted(set(pred_groups) & set(target_groups)):
        x = pred_groups[pid]  # (n_pred, 3), differentiable
        y = target_groups[pid]  # (n_tgt, 3), reference (detached inside sliced_sw2)
        if x.shape[0] == 0 or y.shape[0] == 0:  # nothing to match on one side
            continue
        object_wasserstein_distance[int(pid)] = sliced_sw2(x, y)

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
        event_wasserstein_distance[key] = sliced_sw2(pv, tv)

    # Sum every term -> the scalar loss the training loop back-props. The per-event
    # log_ht term is down-weighted by EVENT_WEIGHT relative to the per-pid object
    # terms (scalar * tensor keeps the gradient).
    EVENT_WEIGHT = 0.1
    terms = list(object_wasserstein_distance.values()) + [
        EVENT_WEIGHT * d for d in event_wasserstein_distance.values()
    ]
    if not terms:  # degenerate empty batch: keep a graph-connected zero
        return pred_particles.sum() * 0.0
    return torch.stack(terms).sum()


# =============================================================================
# Alternative training loss: per-observable soft-histogram MSE
# =============================================================================
#
# 1. The current data pipeline carries per-particle observables as 2-D padded
#    tensors ``(n_events, max_n_objects)`` with ``pt == 0`` on padding /
#    efficiency-killed slots. We flatten and apply the ``pt != 0`` mask before
#    histogramming, matching the cut already used by :func:`per_event_wasserstein_loss`
#    and the intermediate-plot collector in :mod:`training`.
# 2. The default bin edges and weights cover the *current* observable set
#    (which adds ``log_E`` and ``log_ht`` to the old ``pt, eta, log_pt, ht``).


# Default bin edges for the observables emitted by
# :func:`tune_cms_fullsim.data.load_pflow_targets` /
# :func:`tune_cms_fullsim.data.load_pflow_targets_from_tensor`. Wide enough to
# cover the bulk of both target and trainee distributions on a QCD-jet sample;
# users can pass an alternative dict to :func:`multi_observable_soft_hist_loss`.
DEFAULT_SOFT_HIST_BIN_EDGES: dict[str, torch.Tensor] = {
    "pt": torch.linspace(0.0, 200.0, 41, dtype=torch.float64),
    "eta": torch.linspace(-5.0, 5.0, 41, dtype=torch.float64),
    "log_pt": torch.linspace(-1.0, 6.0, 41, dtype=torch.float64),
    "log_E": torch.linspace(-1.0, 7.0, 41, dtype=torch.float64),
    "multiplicity": torch.linspace(0.0, 400.0, 41, dtype=torch.float64),
    "ht": torch.linspace(0.0, 2000.0, 41, dtype=torch.float64),
    "log_ht": torch.linspace(0.0, 8.0, 41, dtype=torch.float64),
}

# Default per-observable weights
DEFAULT_SOFT_HIST_WEIGHTS: dict[str, float] = {
    "pt": 1.0,
    "eta": 1.0,
    "log_pt": 0.5,
    "log_E": 0.5,
    "multiplicity": 0.1,
    "ht": 0.1,
    "log_ht": 0.1,
}

_PARTICLE_OBSERVABLES: frozenset[str] = frozenset(
    {"pt", "eta", "phi", "E", "log_E", "log_pt"}
)


def _flatten_obs_for_hist(
    values: torch.Tensor,
    pt_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Flatten an observable tensor for :func:`soft_histogram` consumption.

    Per-particle observables come from
    :func:`tune_cms_fullsim.data.load_pflow_targets_from_tensor` as 2-D padded
    ``(n_events, max_n_objects)`` tensors with zeros on padding /
    efficiency-killed slots. This helper applies the same ``pt != 0`` cut the
    Wasserstein loss uses (boolean-mask indexing is a differentiable gather,
    so gradients on the pred side flow through unchanged).

    Per-event observables are already 1-D ``(n_events,)`` and are returned
    flattened (``reshape(-1)``) without masking.
    """
    if values.ndim >= 2:
        if pt_mask is None:
            # No pt to mask on -> fall back to flattening (rare; only happens
            # if a 2-D observable is supplied without a 2-D ``pt`` key).
            return values.reshape(-1)
        return values[pt_mask]
    return values.reshape(-1)


def multi_observable_soft_hist_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    bin_edges: dict[str, torch.Tensor] | None = None,
    beta: float = 0.15,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Sum of per-observable soft-histogram MSE losses, optionally weighted.

    Ported from the pre-Wasserstein harness (commit 5cac599). For every
    observable present in both ``pred`` and ``target`` and in ``bin_edges``,
    this function:

    1. Flattens the observable (2-D per-particle -> 1-D via :func:`_flatten_obs_for_hist`,
       with a ``pt != 0`` mask so padding and efficiency-killed slots are
       dropped on both sides).
    2. Builds a soft histogram of pred and target values on a common bin grid.
    3. Normalises both histograms to densities and computes their MSE.
    4. Adds the weighted per-observable MSE to the total loss.

    The result is a scalar that the training loop back-propagates through
    the trainee card. Gradients flow through pred values via :func:`soft_histogram`'s
    sigmoid path; the target side is detached inside :func:`histogram_mse_loss`.

    Parameters
    ----------
    bin_edges : dict[str, torch.Tensor], optional
        Per-observable bin edges. Defaults to :data:`DEFAULT_SOFT_HIST_BIN_EDGES`.
    beta : float
        Soft-histogram softness; see :func:`soft_histogram`. The old harness
        defaulted to 0.15 and we keep that here.
    weights : dict[str, float], optional
        Per-observable weights. Defaults to :data:`DEFAULT_SOFT_HIST_WEIGHTS`.
        Missing keys get weight 1.0.

    Returns
    -------
    torch.Tensor
        Scalar loss summed over every observable present in ``bin_edges``
        with finite values on both sides. Returns a graph-connected zero
        (``pred[...].sum() * 0.0``) on degenerate empty batches so the
        training loop's ``loss.backward()`` always has a graph to walk.
    """
    bin_edges = bin_edges if bin_edges is not None else DEFAULT_SOFT_HIST_BIN_EDGES
    weights = weights if weights is not None else DEFAULT_SOFT_HIST_WEIGHTS

    # Build the pt-mask once on each side: per-particle observables share the
    # same padding pattern, so we can mask all of them with a single boolean
    # tensor without recomputing.
    pred_pt = pred.get("pt")
    tgt_pt = target.get("pt")
    pred_mask = (pred_pt != 0) if (pred_pt is not None and pred_pt.ndim >= 2) else None
    tgt_mask = (tgt_pt != 0) if (tgt_pt is not None and tgt_pt.ndim >= 2) else None

    # Pick a reference tensor for the degenerate empty-batch return so the
    # backward pass always has a graph-connected zero to fall back on.
    any_pred = next(iter(pred.values())) if pred else None

    total: torch.Tensor | None = None
    for key, edges in bin_edges.items():
        if key not in pred or key not in target:
            continue
        if key not in _PARTICLE_OBSERVABLES and (pred[key].ndim >= 2 or target[key].ndim >= 2):
            # Be lenient: any 2-D observable gets the same padding cut.
            pred_vals = _flatten_obs_for_hist(pred[key], pred_mask)
            tgt_vals = _flatten_obs_for_hist(target[key], tgt_mask)
        elif key in _PARTICLE_OBSERVABLES:
            pred_vals = _flatten_obs_for_hist(pred[key], pred_mask)
            tgt_vals = _flatten_obs_for_hist(target[key], tgt_mask)
        else:
            pred_vals = pred[key].reshape(-1)
            tgt_vals = target[key].reshape(-1)

        if pred_vals.numel() == 0 or tgt_vals.numel() == 0:
            continue

        edges_dev = edges.to(device=pred_vals.device, dtype=pred_vals.dtype)
        w = float(weights.get(key, 1.0))
        term = w * histogram_mse_loss(pred_vals, tgt_vals, edges_dev, beta=beta)
        total = term if total is None else total + term

    if total is None:
        if any_pred is None:
            return torch.zeros((), dtype=torch.float64)
        return any_pred.sum() * 0.0
    return total


def multi_observable_soft_hist_loss_distributed(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    bin_edges: dict[str, torch.Tensor] | None = None,
    beta: float = 0.15,
    weights: dict[str, float] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """DDP-aware version of :func:`multi_observable_soft_hist_loss`.

    Each rank computes its local *unnormalized* soft histograms; we then
    all-reduce-SUM both histograms across ranks (differentiably on the pred
    side, via ``torch.distributed.nn.functional.all_reduce``), normalise, and
    compute the MSE. The result is mathematically identical to running
    :func:`multi_observable_soft_hist_loss` on the union of every rank's
    events.

    Falls back to :func:`multi_observable_soft_hist_loss` when no process
    group is initialized (single-process / plain ``python -m ...`` runs).

    Returns
    -------
    torch.Tensor
        Scalar loss; gradients propagate through the all-reduce back to the
        learnable parameters on the local rank.
    """
    if not _is_dist():
        return multi_observable_soft_hist_loss(
            pred, target, bin_edges=bin_edges, beta=beta, weights=weights
        )

    bin_edges = bin_edges if bin_edges is not None else DEFAULT_SOFT_HIST_BIN_EDGES
    weights = weights if weights is not None else DEFAULT_SOFT_HIST_WEIGHTS

    pred_pt = pred.get("pt")
    tgt_pt = target.get("pt")
    pred_mask = (pred_pt != 0) if (pred_pt is not None and pred_pt.ndim >= 2) else None
    tgt_mask = (tgt_pt != 0) if (tgt_pt is not None and tgt_pt.ndim >= 2) else None

    any_pred = next(iter(pred.values())) if pred else None

    # We must issue the same collective sequence on every rank or NCCL will
    # deadlock. Iterating over ``bin_edges`` (the same dict on every rank)
    # rather than ``pred.keys()`` guarantees that.
    total: torch.Tensor | None = None
    for key, edges in bin_edges.items():
        pred_vals_raw = pred.get(key)
        tgt_vals_raw = target.get(key)
        if pred_vals_raw is None or tgt_vals_raw is None:
            continue

        if pred_vals_raw.ndim >= 2:
            pred_vals = _flatten_obs_for_hist(pred_vals_raw, pred_mask)
        else:
            pred_vals = pred_vals_raw.reshape(-1)
        if tgt_vals_raw.ndim >= 2:
            tgt_vals = _flatten_obs_for_hist(tgt_vals_raw, tgt_mask)
        else:
            tgt_vals = tgt_vals_raw.reshape(-1)

        edges_dev = edges.to(
            device=pred_vals.device if pred_vals.numel() else (pred_vals_raw.device),
            dtype=pred_vals.dtype if pred_vals.numel() else (pred_vals_raw.dtype),
        )

        # Local histograms (zero-length tensors give zero histograms, exactly
        # what we want for ranks whose shard happens to be empty for this
        # observable).
        n_bins = edges_dev.numel() - 1
        if pred_vals.numel() > 0:
            pred_hist_local = soft_histogram(pred_vals, edges_dev, beta=beta)
        else:
            pred_hist_local = torch.zeros(
                n_bins, dtype=edges_dev.dtype, device=edges_dev.device
            )
        if tgt_vals.numel() > 0:
            tgt_hist_local = soft_histogram(tgt_vals.detach(), edges_dev, beta=beta)
        else:
            tgt_hist_local = torch.zeros(
                n_bins, dtype=edges_dev.dtype, device=edges_dev.device
            )

        # Differentiable all-reduce on the pred side keeps gradients connected;
        # the target side is detached so a plain (non-diff) all-reduce is fine.
        pred_hist = diff_all_reduce(pred_hist_local, op=dist.ReduceOp.SUM)
        tgt_hist = tgt_hist_local.clone()
        dist.all_reduce(tgt_hist, op=dist.ReduceOp.SUM)

        pred_norm = pred_hist / (pred_hist.sum() + eps)
        tgt_norm = tgt_hist / (tgt_hist.sum() + eps)
        loss_key = ((pred_norm - tgt_norm) ** 2).mean()

        w = float(weights.get(key, 1.0))
        term = w * loss_key
        total = term if total is None else total + term

    if total is None:
        if any_pred is None:
            return torch.zeros((), dtype=torch.float64)
        return any_pred.sum() * 0.0
    return total


# =============================================================================
# Loss dispatcher
# =============================================================================


LossFn = Callable[
    [dict[str, torch.Tensor], dict[str, torch.Tensor]], torch.Tensor
]

LOSS_CHOICES: tuple[str, ...] = ("wasserstein", "soft_hist")


def get_loss_fn(name: str) -> LossFn:
    """Return the training loss callable selected by ``name``.

    Recognised names (see :data:`LOSS_CHOICES`):

    - ``"wasserstein"`` -> :func:`per_event_wasserstein_loss` (the
      current active loss).
    - ``"soft_hist"`` -> :func:`multi_observable_soft_hist_loss_distributed`
      (the loss used in the pre-Wasserstein harness, ported verbatim).
      The DDP-aware variant is returned unconditionally; it falls back to
      :func:`multi_observable_soft_hist_loss` when no process group is
      initialized, so single-process runs work transparently.

    Both callables expect ``(pred, target)`` observable dicts produced by
    :func:`tune_cms_fullsim.data.load_pflow_targets_from_tensor` (pred side)
    and :func:`tune_cms_fullsim.data.load_pflow_targets` (target side) and
    return a scalar tensor that the training loop back-propagates.

    Raises
    ------
    ValueError
        If ``name`` is not one of :data:`LOSS_CHOICES`.
    """
    if name == "wasserstein":
        return per_event_wasserstein_loss
    if name == "soft_hist":
        return multi_observable_soft_hist_loss_distributed
    raise ValueError(
        f"Unknown loss {name!r}. Valid choices: {LOSS_CHOICES}."
    )

