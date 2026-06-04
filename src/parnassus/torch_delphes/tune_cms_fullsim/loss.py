"""Differentiable soft-histogram losses for ``tune_cms_fullsim``.

This module is the package's self-contained loss layer:

- :func:`soft_histogram` and :func:`histogram_mse_loss` are the
  fully-differentiable histogram primitives. They are an exact COPY of the
  same-named functions in :mod:`parnassus.torch_delphes.tuning` so that this
  package does not depend on ``tuning`` at all. (The ``tuning`` module keeps
  its own copies for its standalone demo / tests; keep the two in sync if you
  ever change the soft-histogram math.)
- :func:`multi_observable_loss` sums the per-observable histogram MSEs.
- :func:`multi_observable_loss_distributed` is the DDP-aware version that
  all-reduces the histograms across ranks; it falls back to
  :func:`multi_observable_loss` when no process group is initialized.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce as diff_all_reduce

from .distributed import _is_dist

# =============================================================================
# Differentiable histogram and loss primitives (copied verbatim from tuning.py)
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

    Suitable as a training loss when ``pred_values`` are produced by a
    differentiable detector simulation and ``target_values`` are
    detached (a fixed target distribution).

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
# Multi-observable losses
# =============================================================================


def multi_observable_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    bin_edges: dict[str, torch.Tensor],
    beta: float = 0.15,
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Sum of per-observable soft-histogram MSE losses, optionally weighted.

    Per-particle observables (``pt``, ``eta``, ``log_pt``, ``log_E``) arrive as
    2-D ``(n_events, max_n_particles)`` padded tensors; they are flattened to 1-D
    and stripped of padding / efficiency-ghost slots before histogramming. The
    validity mask is derived from the *same side's* ``pt`` key, because
    ``eta == 0`` and ``log_pt == log(1) == 0`` are valid real values and cannot
    be detected per-observable. Per-event observables (``multiplicity``,
    ``ht``, ``log_ht``) are already 1-D and pass through unchanged. ``pred`` and ``target``
    may have different ``max_n_particles``; each side is flattened and masked
    independently and ``histogram_mse_loss`` normalizes to a density, so
    unequal counts are handled correctly.

    Returns
    -------
    torch.Tensor
        Scalar loss summed over every observable for which both
        ``pred`` and ``target`` have at least one valid value.
    """

    def _flatten_valid(obs: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        v = obs[key]
        if v.ndim >= 2:  # per-particle, padded -> drop padding/ghost slots
            return v[obs["pt"] != 0]  # boolean-index READ is a differentiable gather
        return v.reshape(-1)  # per-event, already 1-D

    # Anchor the accumulator on bin_edges' dtype/device (bin_edges are float64
    # and moved to the compute device by the caller); a bare torch.zeros((), ...)
    # would sit on the CPU and break once DDP runs on GPU.
    any_edges = next(iter(bin_edges.values()))
    total = torch.zeros((), dtype=any_edges.dtype, device=any_edges.device)

    # The weights dict drives the active observable set: DEFAULT_OBS_WEIGHTS lists
    # the active observables (e.g. eta/log_pt/log_E/log_ht; linear pt/ht are kept
    # at weight 0), so iterating its keys honors those weights -- including the
    # weight-0 ones -- instead of defaulting every bin_edges key to weight 1.0.
    active_keys = weights.keys() if weights else bin_edges.keys()
    for key in active_keys:
        if key not in bin_edges or key not in pred or key not in target:
            continue
        pred_vals = _flatten_valid(pred, key)
        tgt_vals = _flatten_valid(target, key)
        if pred_vals.numel() == 0 or tgt_vals.numel() == 0:
            continue
        w = weights[key] if weights else 1.0
        total = total + w * histogram_mse_loss(pred_vals, tgt_vals, bin_edges[key], beta=beta)
    return total


def multi_observable_loss_distributed(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    bin_edges: dict[str, torch.Tensor],
    beta: float = 0.15,
    weights: dict[str, float] | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """DDP-aware multi-observable soft-histogram MSE loss.

    Each rank computes its local *unnormalized* soft histogram for both
    pred and target. We then all-reduce-SUM both histograms across ranks
    (differentiably for the pred side, via
    ``torch.distributed.nn.functional.all_reduce``), normalize, and
    compute the MSE. The result is mathematically identical to running
    :func:`multi_observable_loss` on the union of every rank's events.

    Falls back to :func:`multi_observable_loss` when no process group is
    initialized (single-process / plain ``python`` runs).

    Returns
    -------
    torch.Tensor
        Scalar loss; gradients propagate through the all-reduce back to
        the learnable parameters on the local rank.
    """
    if not _is_dist():
        return multi_observable_loss(pred, target, bin_edges, beta=beta, weights=weights)
    else:
        raise NotImplementedError("multi_observable_loss_distributed is not implemented yet")
