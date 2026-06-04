"""Differentiable distribution-matching losses for ``tune_cms_fullsim``.

This module is the package's self-contained loss layer:

- :func:`soft_histogram` and :func:`histogram_mse_loss` are the
  fully-differentiable histogram primitives. They are an exact COPY of the
  same-named functions in :mod:`parnassus.torch_delphes.tuning` so that this
  package does not depend on ``tuning`` at all. (The ``tuning`` module keeps
  its own copies for its standalone demo / tests; keep the two in sync if you
  ever change the soft-histogram math.) They are kept here as reference
  primitives; the active loss path now uses the Wasserstein distance below.
- :func:`wasserstein_1d_loss` is the exact 1-D Wasserstein (Earth-Mover)
  distance between two empirical distributions, implemented in pure torch so
  it stays fully differentiable w.r.t. the predicted sample positions and adds
  no external optimal-transport dependency.
- :func:`multi_observable_loss` sums the per-observable Wasserstein distances,
  each normalized by its target spread so the terms are comparable.
- :func:`multi_observable_loss_distributed` is the DDP-aware version; it falls
  back to :func:`multi_observable_loss` when no process group is initialized.
"""

from __future__ import annotations

import torch

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
# Exact 1-D Wasserstein (Earth-Mover) loss
# =============================================================================


def wasserstein_1d_loss(
    pred_values: torch.Tensor,  # shape (n_object_0,)
    target_values: torch.Tensor,  # shape (n_object_1,)
    p: int = 1,
) -> torch.Tensor:
    r"""Exact 1-D :math:`p`-Wasserstein distance between two empirical distributions.

    For two univariate empirical distributions with *uniform* sample weights,
    the :math:`p`-Wasserstein distance has the closed form

    .. math::

        W_p^p = \int_0^1 \bigl| F_\text{pred}^{-1}(q) - F_\text{tgt}^{-1}(q) \bigr|^p \, dq,

    i.e. the integral of the gap between the two inverse CDFs (quantile
    functions). It is evaluated exactly here by sorting each side, taking the
    union of the two empirical-CDF breakpoints, and summing the per-segment
    contribution. Unlike a histogram MSE, the gradient w.r.t. ``pred_values``
    does not vanish when the two distributions barely overlap, which gives Adam
    a useful signal early in a fit.

    This is implemented in pure torch (no ``ot``/POT dependency): gradients flow
    to ``pred_values`` through the ``sort`` and the inverse-CDF gather, while
    ``target_values`` is detached so only the trainee is updated. The two clouds
    may have different sizes (``n_object_0 != n_object_1``); "sliced" random
    projections are unnecessary in 1-D.

    Parameters
    ----------
    pred_values : torch.Tensor
        1-D tensor of predicted observable values, shape ``(n_object_0,)``.
        Carries gradients back to the learnable detector parameters.
    target_values : torch.Tensor
        1-D tensor of target observable values, shape ``(n_object_1,)``.
        Detached internally.
    p : int
        Order of the Wasserstein distance. ``p = 1`` is the Earth-Mover
        distance (robust to long tails); ``p = 2`` penalizes large quantile
        gaps more strongly.

    Returns
    -------
    torch.Tensor
        Scalar :math:`W_p^p` (``== W_1`` for the default ``p = 1``), on
        ``pred_values``' device and dtype.
    """
    pred = pred_values.reshape(-1)
    tgt = target_values.detach().reshape(-1).to(device=pred.device, dtype=pred.dtype)

    u, _ = torch.sort(pred)
    v, _ = torch.sort(tgt)
    n, m = u.numel(), v.numel()

    # Empirical-CDF levels: F jumps to k/N at the k-th smallest sample.
    u_cw = torch.arange(1, n + 1, device=u.device, dtype=u.dtype) / n
    v_cw = torch.arange(1, m + 1, device=v.device, dtype=v.dtype) / m

    # The union of both sets of CDF levels partitions the quantile axis [0, 1]
    # into segments on which both inverse CDFs are constant.
    qs, _ = torch.sort(torch.cat([u_cw, v_cw]))

    # Inverse CDF at each breakpoint: F^{-1}(q) = smallest sample whose CDF >= q,
    # i.e. searchsorted(cw, q, right=False). The clamp guards q == 1.0 (and any
    # float round-off) from indexing past the last sample. This is a plain gather
    # into the sorted values, so its gradient flows to pred (via `u`).
    u_q = u[torch.searchsorted(u_cw, qs).clamp(max=n - 1)]
    v_q = v[torch.searchsorted(v_cw, qs).clamp(max=m - 1)]

    # Segment widths in quantile space (depend only on the uniform weights, so
    # they are constants and carry no gradient).
    qs0 = torch.cat([qs.new_zeros(1), qs])
    delta = qs0[1:] - qs0[:-1]

    diff = (u_q - v_q).abs()
    return (delta * diff.pow(p)).sum()


# =============================================================================
# Multi-observable losses
# =============================================================================


def multi_observable_loss(
    pred: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    bin_edges: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
    p: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted sum of per-observable 1-D Wasserstein distances.

    Per-particle observables (``pt``, ``eta``, ``log_pt``, ``log_E``) arrive as
    2-D ``(n_events, max_n_particles)`` padded tensors; they are flattened to 1-D
    and stripped of padding / efficiency-ghost slots before the distance. The
    validity mask is derived from the *same side's* ``pt`` key, because
    ``eta == 0`` and ``log_pt == log(1) == 0`` are valid real values and cannot
    be detected per-observable. Per-event observables (``multiplicity``,
    ``ht``, ``log_ht``) are already 1-D and pass through unchanged. ``pred`` and ``target``
    may have different ``max_n_particles``; each side is flattened and masked
    independently and :func:`wasserstein_1d_loss` accepts unequal counts, so
    differing sizes are handled correctly.

    A Wasserstein distance is in the *units of its observable*, so the raw
    per-observable distances are not comparable (e.g. ``multiplicity`` spans
    0-400 while ``eta`` spans ~+-5). Each term is therefore divided by the
    detached target spread ``std(target)**p`` to make it dimensionless and
    O(1), which keeps the per-observable ``weights`` -- tuned against the old
    unit-free histogram MSE -- meaningful.

    ``bin_edges`` is no longer used to histogram; it is retained only to anchor
    the accumulator's dtype/device and (when ``weights`` is None) to enumerate
    the observable set. ``beta`` is unused on this Wasserstein path and kept only
    for call-signature compatibility with the soft-histogram primitives.

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
        # Normalize by the target spread (detached, so the scale carries no
        # gradient) raised to p, so W_p**p / std**p is dimensionless for any p.
        scale = tgt_vals.detach().std(unbiased=False).clamp(min=eps)
        total = total + w * wasserstein_1d_loss(pred_vals, tgt_vals, p=p) / scale.pow(p)
    return total
