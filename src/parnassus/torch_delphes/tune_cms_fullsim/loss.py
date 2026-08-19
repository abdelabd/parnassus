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
from typing import Callable, NamedTuple

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
COUNT_WEIGHT = 1.0

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


class LossComponent(NamedTuple):
    """One labeled contribution to the summed training loss, for the per-epoch
    debug breakdown built when a loss is called with ``return_breakdown=True``.

    ``weighted`` is exactly the scalar that was summed into the loss's ``terms``
    (``weighted == raw * weight`` for every term), so ``sum(c.weighted for c in
    components)`` equals the returned scalar loss. ``raw`` is the value BEFORE the
    weight; ``weight`` is the multiplicative factor applied. All three are plain
    Python floats (off the autograd graph) -- the breakdown is logging only.
    """

    category: str  # "pid_shape" | "event" | "count"
    label: str  # e.g. "211:eta", "log_ht", "ChargedHadron"
    raw: float  # value BEFORE the weight
    weight: float  # multiplicative weight applied
    weighted: float  # value AFTER the weight (== the term summed into ``terms``)


# Short human labels for the count terms, derived from the card output key (the
# first element of each COUNT_TERM_KEYS / CALO_COUNT_TERM_KEYS row) by dropping the
# shared ``ExpectedCounts`` suffix: ChargedHadron / Electron / Muon / EcalPhoton /
# HcalNeutralHadron. Used only to label the ``return_breakdown`` count entries.
_COUNT_LABELS: dict[str, str] = {
    out_key: out_key.removesuffix("ExpectedCounts")
    for out_key, _pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS)
}

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


# Stable integer per dtype so the parity check below can ride along on the int64
# size gather. Codes are arbitrary but must never be reused for another dtype.
_DTYPE_BY_CODE: dict[int, str] = {
    1: "float16", 2: "bfloat16", 3: "float32", 4: "float64",
    5: "int8", 6: "uint8", 7: "int16", 8: "int32", 9: "int64", 10: "bool",
}
_DTYPE_CODES: dict[torch.dtype, int] = {
    torch.float16: 1, torch.bfloat16: 2, torch.float32: 3, torch.float64: 4,
    torch.int8: 5, torch.uint8: 6, torch.int16: 7, torch.int32: 8,
    torch.int64: 9, torch.bool: 10,
}


def _dtype_code(dtype: torch.dtype) -> int:
    """Integer code of ``dtype``; 0 for anything not in :data:`_DTYPE_CODES`.

    An unlisted dtype codes as 0 on every rank, so it never triggers a false
    mismatch -- it only makes the check blind to that dtype.
    """
    return _DTYPE_CODES.get(dtype, 0)


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

    # ---- gather sizes (+ the two parity slots, see below) -------------------
    # The extra slots ride along on the size gather so the parity checks below
    # cost no extra collective.
    local_size = torch.tensor(
        [tensor_1d.numel(), int(tensor_1d.requires_grad), _dtype_code(tensor_1d.dtype)],
        dtype=torch.int64,
        device=tensor_1d.device,
    )
    size_list = [
        torch.zeros(3, dtype=torch.int64, device=tensor_1d.device)
        for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size)
    sizes = [int(s[0]) for s in size_list]
    grad_flags = [bool(s[1]) for s in size_list]
    dtype_codes = [int(s[2]) for s in size_list]
    max_size = max(sizes)

    if max_size == 0:
        # Nothing anywhere: every rank returns here, so no gather -- and hence no
        # backward node -- is created on any rank. Consistent, so no parity check.
        return torch.zeros(0, dtype=tensor_1d.dtype, device=tensor_1d.device)

    # ---- autograd-graph parity ---------------------------------------------
    # diff_all_gather registers its backward (REDUCE_SCATTER) node ONLY when the
    # input requires grad. If the ranks disagree, the FORWARD still matches and the
    # job deadlocks in BACKWARD -- surfacing ~10 min later as an opaque NCCL
    # watchdog timeout on a mismatched collective. Fail here instead, by name.
    # The usual cause is a caller substituting a freshly built torch.zeros(0) for a
    # rank that has no data; slice or mask a graph tensor to zero length instead,
    # which is empty but keeps grad_fn.
    if differentiable and len(set(grad_flags)) > 1:
        with_grad = [r for r, g in enumerate(grad_flags) if g]
        without = [r for r, g in enumerate(grad_flags) if not g]
        raise RuntimeError(
            "_all_gather_varlen(differentiable=True): ranks disagree on whether the "
            f"input is on the autograd graph -- requires_grad=True on ranks {with_grad}, "
            f"False on ranks {without} (local sizes {sizes}). The ranks without it would "
            "skip the backward REDUCE_SCATTER and deadlock the job. Build the empty "
            "case by slicing a graph tensor to zero length (e.g. pred['pt'].reshape(-1)"
            "[:0]) rather than with torch.zeros(0)."
        )

    # ---- dtype parity -------------------------------------------------------
    # NCCL matches collectives on ELEMENT COUNT, never on dtype, so ranks gathering
    # the same number of float32 and float64 values agree on every number NCCL
    # checks while disagreeing on the byte count. Nothing errors: the job simply
    # hangs until the watchdog reports a timeout on a collective whose sizes look
    # identical on every rank. Fail here instead, by name. The usual cause is an
    # upstream dtype that depends on the batch's contents (see EFlowMerger, where a
    # batch holding no calorimeter tower used to skip the float64 promotion).
    if len(set(dtype_codes)) > 1:
        by_dtype: dict[int, list[int]] = {}
        for r, c in enumerate(dtype_codes):
            by_dtype.setdefault(c, []).append(r)
        detail = ", ".join(
            f"{_DTYPE_BY_CODE.get(c, f'code {c}')} on ranks {rs}"
            for c, rs in sorted(by_dtype.items())
        )
        raise RuntimeError(
            f"_all_gather_varlen: ranks disagree on the input dtype -- {detail} "
            f"(local sizes {sizes}). The element counts match, so NCCL would not "
            "detect this and the job would hang in the collective. Make the producing "
            "code path return one dtype regardless of what the batch contains."
        )

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
    # Drop non-finite samples before the quantile: torch.quantile linearly interpolates
    # the order statistics, so a single -inf (e.g. log(0) from a pt == 0 object) makes an
    # interpolated quantile evaluate -inf + frac*(finite - (-inf)) = -inf + inf = NaN and
    # poisons the whole loss. The boolean gather keeps the surviving samples' gradient
    # (mirrors the isfinite filter used for the pooled scale in _pooled_target_obs_std).
    x = x[torch.isfinite(x)]
    y = y[torch.isfinite(y)]
    if x.numel() == 0 or y.numel() == 0:  # nothing to match: graph-connected zero
        return pred_values.sum() * 0.0
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

# ---------------------------------------------------------------------------
# |eta|-region split of the per-pid shape terms and the pair-mass terms
# ---------------------------------------------------------------------------
# Fixed detector regions = the tracker momentum-smearing regions of the CMS card
# (learnable.py LearnableTrackResolution boundaries; they contain the tracking-
# efficiency |eta| edges 1.5/2.5 and the ECal barrel/endcap edge). Bins:
# [0, 0.5], (0.5, 1.5], (1.5, 2.5], (2.5, inf) -- the overflow bin holds forward
# calorimeter objects (photons / neutral hadrons up to |eta| 5; empty for tracks).
SHAPE_ETA_EDGES: tuple[float, ...] = (0.5, 1.5, 2.5)
N_SHAPE_ETA_REGIONS: int = len(SHAPE_ETA_EDGES) + 1
# Observables whose per-pid shape term is split by region (``eta`` itself stays one
# pooled term: it is not affected by the smearing and a per-region eta term would be
# a near-tautology).
SHAPE_SPLIT_OBS: tuple[str, ...] = ("log_E", "log_pt")

# Pair-mass terms: per event, the two leading-pt reco objects of each of these pids
# form one pair; the mass hypothesis is the class mass (there is no K/pi/p separation
# in the reco on either side, so every charged hadron is a pion -- exact for K_S).
PAIR_MASS_HYPOTHESIS: dict[int, float] = {11: 0.000511, 13: 0.1056584, 211: 0.1395704}
PAIR_MASS_WEIGHT: float = 1.0
_PAIR_M_FLOOR: float = 1e-6  # GeV; guards ln(m) for collinear / degenerate pairs
# The pair-mass terms compare the per-event RESPONSE ``ln(m_reco / m_truth)`` -- the
# leading-2 reco pair of the class over the leading-2 TRUTH pair of the same class in
# the same event (truth is the shared input of both sides, so it is a label, not a fitted
# quantity) -- grouped by a fixed generic grid in ``ln m_truth`` of this width (a factor
# e^0.5 = 1.65 in mass; label ``mt{lo}-{hi}`` in GeV). Why not the reco mass itself: the
# leading-2 mass of a J/psi + Z gun sample is BIMODAL, and the quantile-W2 of two finite
# samples of the same bimodal distribution has a floor set by the binomial noise of the
# J/psi : Z fraction (~2 % of the pairs matched across Delta ln m ~ 3.4 -> W2^2 ~ 0.1 per
# category even at the truth) whose only reduction is WIDER peaks -- a rectified
# "increase a_raw / b_raw" gradient every batch (first M1 muon fit, 2026-08-17: a_raw ran
# to 6-27x truth while scale / eff converged; the all-truth 1-D scan had its minimum at
# a ~ 6x truth). Grouping by the common truth mass makes every term unimodal by
# construction (no resonance knowledge: any sample just populates whatever groups it
# occupies; a resonance straddling a grid edge splits identically on both sides), and
# the response removes the truth spread (Breit-Wigner, continuum) event by event, so its
# width IS the resolution and its center the scale -- the standardization by the target
# response std then sees a 17 % scale error as ~10 sigma instead of the ~1e-5 the pooled
# ln m std gave.
PAIR_TRUTH_LNM_WIDTH: float = 0.5


def _pair_truth_group(ln_m_truth: torch.Tensor) -> torch.Tensor:
    """Truth-mass group index ``floor(ln m_truth / PAIR_TRUTH_LNM_WIDTH)`` (long, gradient-free).

    Returns
    -------
    torch.Tensor
        Long tensor, same shape as ``ln_m_truth``.
    """
    return torch.floor(ln_m_truth.detach() / PAIR_TRUTH_LNM_WIDTH).long()


def _pair_truth_group_label(group: int) -> str:
    """Human label of a truth-mass group, e.g. ``"mt2.72-4.48"`` (GeV).

    Returns
    -------
    str
        ``"mt{lo}-{hi}"`` with the group's mass edges in ``%.3g``.
    """
    lo = math.exp(group * PAIR_TRUTH_LNM_WIDTH)
    hi = math.exp((group + 1) * PAIR_TRUTH_LNM_WIDTH)
    return f"mt{lo:.3g}-{hi:.3g}"


def _leading2_ln_mass(
    pt: torch.Tensor,
    eta: torch.Tensor,
    phi: torch.Tensor,
    sel: torch.Tensor,
    mass: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``ln m`` of the two highest-``pt`` selected objects per event.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(ln_m (n_pairs,), event_index (n_pairs,), eta2 (n_pairs, 2))`` for the events
        with at least two selected objects; ``ln_m`` is differentiable through ``pt``.
    """
    ev = (sel.sum(dim=1) >= 2).nonzero(as_tuple=True)[0]
    if ev.numel() == 0:
        empty = pt.new_zeros(0)
        return empty, ev, pt.new_zeros(0, 2)
    ranked = torch.where(sel, pt, torch.full_like(pt, float("-inf")))
    _vals, idx = ranked.topk(2, dim=1)  # gradient-free choice
    idx = idx[ev]
    rows = ev.unsqueeze(1)
    pt2, eta2, phi2 = pt[rows, idx], eta[rows, idx], phi[rows, idx]  # differentiable gathers
    px = pt2 * torch.cos(phi2)
    py = pt2 * torch.sin(phi2)
    pz = pt2 * torch.sinh(eta2)
    e = torch.sqrt(px * px + py * py + pz * pz + mass * mass)
    m2 = e.sum(dim=1) ** 2 - px.sum(dim=1) ** 2 - py.sum(dim=1) ** 2 - pz.sum(dim=1) ** 2
    return 0.5 * torch.log(m2.clamp(min=_PAIR_M_FLOOR**2)), ev, eta2


def attach_truth_pair_lnm(truth_particles: torch.Tensor, *obs: dict[str, torch.Tensor]) -> None:
    """Store the per-event truth leading-2 pair ``ln m`` of every pair class into each
    ``obs`` dict as ``"truth_pair_lnm:{pid}"`` (``(n_events,)``, NaN for events with fewer
    than two truth objects of the class). ``truth_particles`` is the padded
    ``(n_events, n_particles, N_FEATURES)`` batch tensor -- the SAME events, in the same
    order, as the pred and target dicts (``restore_event_format`` aligns them by event
    id). Class from the truth PDG id: |pid| 11 / 13 for e / mu, every other charged
    particle for 211 (the pion hypothesis on both sides, exact for K_S).
    """
    from parnassus.data.particle_io import ColumnMap  # noqa: PLC0415 (avoid a data<->loss cycle)

    tp = truth_particles.detach()
    pid = tp[..., ColumnMap.PID].abs()
    charged = tp[..., ColumnMap.CHARGE] != 0
    pt, eta, phi = tp[..., ColumnMap.PT], tp[..., ColumnMap.ETA], tp[..., ColumnMap.PHI]
    for p, mass in PAIR_MASS_HYPOTHESIS.items():
        sel = charged & ((pid == p) if p != 211 else ((pid != 11) & (pid != 13)))
        ln_m, ev, _eta2 = _leading2_ln_mass(pt, eta, phi, sel, mass)
        lab = torch.full((tp.shape[0],), float("nan"), dtype=pt.dtype, device=pt.device)
        lab[ev] = ln_m
        for o in obs:
            o[f"truth_pair_lnm:{p}"] = lab


def _eta_region_index(eta: torch.Tensor) -> torch.Tensor:
    """Region index in ``[0, N_SHAPE_ETA_REGIONS)`` of ``|eta|`` w.r.t.
    :data:`SHAPE_ETA_EDGES` (``|eta| <= 0.5`` -> 0, ``(0.5, 1.5]`` -> 1, ...; same
    ``searchsorted`` semantics as ``LearnableTrackResolution._region_index``).

    Returns
    -------
    torch.Tensor
        Long tensor of region indices, same shape as ``eta``; gradient-free.
    """
    edges = torch.tensor(SHAPE_ETA_EDGES, dtype=eta.dtype, device=eta.device)
    return torch.bucketize(eta.detach().abs(), edges, right=False)


def _pair_category(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Symmetric pair category of two region indices.

    Returns
    -------
    torch.Tensor
        ``min(r1, r2) * N_SHAPE_ETA_REGIONS + max(r1, r2)`` (label ``eta{r1}{r2}`` with
        ``r1 <= r2``, see :func:`_pair_category_label`).
    """
    lo = torch.minimum(r1, r2)
    hi = torch.maximum(r1, r2)
    return lo * N_SHAPE_ETA_REGIONS + hi


def _pair_category_label(cat: int) -> str:
    """Human label of a pair category (inverse of :func:`_pair_category`).

    Returns
    -------
    str
        ``"eta{r1}{r2}"`` with ``r1 <= r2``.
    """
    return f"eta{cat // N_SHAPE_ETA_REGIONS}{cat % N_SHAPE_ETA_REGIONS}"


def compute_pair_masses(
    obs: dict[str, torch.Tensor],
    hypotheses: dict[int, float] | None = None,
) -> dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Per-event leading-2 pair-mass RESPONSE per class.

    For every pid in ``hypotheses`` (default :data:`PAIR_MASS_HYPOTHESIS`): in each event
    take the two highest-``pt`` reco objects of that pid (``pt > 0``, so efficiency-
    killed / padding slots never pair); events with fewer than two contribute no pair.
    The invariant mass is built from ``(pt, eta, phi)`` with the class mass hypothesis
    on BOTH sides (no truth matching, no charge, no resonance knowledge -- the SAME rule
    on a J/psi / Z / K_S gun sample, where the leading pair IS the resonance, and on dijet /
    HZZ4l, where it is a combinatorial continuum identical on target and trainee). The
    response is ``ln m_reco - ln m_truth`` with the event's truth leading-2 pair mass of
    the same class from ``obs["truth_pair_lnm:{pid}"]`` (see :func:`attach_truth_pair_lnm`;
    REQUIRED -- a class with reco pairs but no truth label raises); pairs whose event has
    no truth pair (NaN label) are dropped.

    Differentiable through ``pt`` (the smeared / rescaled momentum) on the pred side;
    ``eta`` / ``phi`` carry no gradient in the trainee (tracks keep their truth angles).

    If ``obs`` already carries precomputed ``"pair_r:{pid}"`` / ``"pair_cat:{pid}"`` /
    ``"pair_grp:{pid}"`` entries (the DDP wrapper gathers those across ranks) they are
    returned as-is; if the per-object ``pt``/``eta``/``phi``/``pid`` keys are missing the
    result is empty.

    Returns
    -------
    dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ``{pid: (response (n_pairs,), category (n_pairs,) long, truth_group (n_pairs,)
        long)}`` -- ``category`` from :func:`_pair_category` on the two reco objects'
        :func:`_eta_region_index`, ``truth_group`` from :func:`_pair_truth_group` on the
        truth pair mass; pids without a single pair are omitted.
    """
    hyp = hypotheses if hypotheses is not None else PAIR_MASS_HYPOTHESIS
    out: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    pre = {int(k.split(":", 1)[1]) for k in obs if k.startswith("pair_r:")}
    if pre:
        for pid in sorted(pre):
            r = obs[f"pair_r:{pid}"].reshape(-1)
            if r.numel():
                out[pid] = (
                    r,
                    obs[f"pair_cat:{pid}"].reshape(-1).long(),
                    obs[f"pair_grp:{pid}"].reshape(-1).long(),
                )
        return out
    if any(k not in obs for k in ("pt", "eta", "phi", "pid")):
        return out
    pt, eta, phi, pid = obs["pt"], obs["eta"], obs["phi"], obs["pid"]
    if pt.dim() != 2 or pt.shape[1] < 2:
        return out
    for p, mass in hyp.items():
        ln_m, ev, eta2 = _leading2_ln_mass(pt, eta, phi, (pid == p) & (pt > 0), mass)
        if ev.numel() == 0:
            continue
        key = f"truth_pair_lnm:{p}"
        if key not in obs:
            raise KeyError(
                f"compute_pair_masses: {key!r} missing -- call attach_truth_pair_lnm("
                "truth_particles, pred, target) before the loss"
            )
        ln_mt = obs[key].reshape(-1)[ev]
        keep = torch.isfinite(ln_mt)
        if not bool(keep.any()):
            continue
        r = _eta_region_index(eta2[keep])
        out[int(p)] = (
            (ln_m - ln_mt)[keep],
            _pair_category(r[:, 0], r[:, 1]),
            _pair_truth_group(ln_mt[keep]),
        )
    return out


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
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
    out_components: list | None = None,
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

    ``pid_weighting`` / ``pid_weight_floor`` optionally apply the SAME per-species
    population weighting the shape terms use (:func:`_pid_population_weights`, CLI
    ``--pid-weighting``) ACROSS the count terms: each species' term is multiplied
    by a mean-1-normalized weight derived from its detached target population
    fraction in the batch (``"equal"`` = exact no-op). Rationale: each term is a
    scale-free RELATIVE chi^2, so without this a rare species with a structural
    (unfixable) count mismatch -- e.g. the decay-in-flight muon excess, 0.39
    reco/event vs a 0.165 truth ceiling -- contributes as much loss as the
    abundant species (the observed Muon ~0.18 vs everything else <0.01). The
    species population is read from the term's own target region counts, so no
    extra plumbing is needed. Note the per-species efficiency logits receive
    gradient ONLY from their own term, so a global rescale of that term mostly
    re-scales (not re-directs) their Adam step.

    When ``out_components`` is a list, each emitted term also appends a
    ``(label, raw_tensor, weight)`` tuple to it -- the human label
    (:data:`_COUNT_LABELS`), the UNWEIGHTED scalar, and the EFFECTIVE weight
    (``count_weight`` or ``calo_count_weight``, times the species population
    weight) -- aligned 1:1 with the returned ``count_terms`` (same order, same
    skips), for the per-epoch ``return_breakdown`` debug print. ``None``
    (default) makes this a byte-identical no-op.
    """
    calo_pred_keys = {pred_key for _o, pred_key, _t in CALO_COUNT_TERM_KEYS}
    # First pass: build every present species' raw term + its detached target
    # population (for the cross-species weighting below).
    entries: list[tuple[str, torch.Tensor, float, float]] = []
    for out_key, pred_key, tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
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
            raw, weight = rel.mean(), calo_count_weight
        else:
            # population-weighted squared relative error
            chi2 = (pred_rate - tgt_rate) ** 2 / (tgt_rate + count_rate_floor)
            total = tgt_rate.sum().clamp_min(count_rate_floor)  # detached -> invariant
            raw, weight = chi2.sum() / total, count_weight
        entries.append((out_key, raw, weight, float(tgt_counts.sum())))

    # Cross-species population weights (mean-1 over the present terms; "equal" is
    # a bit-exact 1.0 no-op). Detached plain floats, like the shape weights.
    species_w = _population_weights_from_counts(
        {out_key: tot for out_key, _raw, _w, tot in entries},
        mode=pid_weighting,
        floor=pid_weight_floor,
    )

    count_terms: list[torch.Tensor] = []
    for out_key, raw, weight, _tot in entries:
        eff_weight = weight * species_w[out_key]
        count_terms.append(eff_weight * raw)
        if out_components is not None:
            out_components.append((_COUNT_LABELS[out_key], raw, eff_weight))
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
    return _population_weights_from_counts(
        {int(p): float(target_groups[p].shape[0]) for p in present_pids},
        mode=mode,
        floor=floor,
    )


def _population_weights_from_counts(
    counts: dict, mode: str = "equal", floor: float = 0.0
) -> dict:
    """Mean-1-normalized population weights from a ``{key: detached count}`` map.

    The shared core of :func:`_pid_population_weights` (per-pid SHAPE weights)
    and the per-species COUNT-term weighting in :func:`_count_terms`; see the
    former's docstring for the mode semantics (``"equal"`` -> exact 1.0 no-op,
    ``"fraction"`` / ``"sqrt_fraction"`` down-weight low-population keys, mean-1
    normalized over the present keys, optional ``floor`` clamp + re-normalize).
    Keys are arbitrary hashables; every weight is a plain Python float.
    """
    if not counts:
        return {}
    if mode == "equal":
        return {p: 1.0 for p in counts}

    total = sum(counts.values())
    if total <= 0:  # degenerate (shouldn't happen: present groups are non-empty)
        return {p: 1.0 for p in counts}

    if mode == "fraction":
        g = {p: counts[p] / total for p in counts}
    elif mode == "sqrt_fraction":
        g = {p: math.sqrt(counts[p] / total) for p in counts}
    else:
        raise ValueError(
            f"Unknown pid_weighting {mode!r}. Valid choices: {PID_WEIGHTING_CHOICES}."
        )

    p_count = len(g)

    def _normalize(gv: dict) -> dict:
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
    return_breakdown: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[LossComponent]]:
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

    When ``return_breakdown`` is True the function returns ``(loss, components)``,
    where ``components`` is a list of :class:`LossComponent` (one per per-pid object
    term, the ``log_ht`` event term, and each per-class count term) carrying the raw
    (pre-weight) value, the weight applied, and the weighted (post-weight) value, so
    ``sum(c.weighted for c in components)`` equals ``loss``. ``False`` (default)
    returns the bare scalar exactly as before, with zero extra work.
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
    object_raw: dict[int, torch.Tensor] = {}  # pre-weight SW_2, for return_breakdown
    for pid in present_pids:
        x = pred_groups[pid]  # (n_pred, 3), differentiable
        y = target_groups[pid]  # (n_tgt, 3), reference (detached inside sliced_sw2)
        object_raw[pid] = sliced_sw2(x, y, object_scale)
        object_wasserstein_distance[pid] = pid_weights[pid] * object_raw[pid]

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
    # population-weighted (tracking) vs per-region-fair (calo) normalization and the
    # cross-species pid_weighting.
    count_components: list | None = [] if return_breakdown else None
    count_terms = _count_terms(
        pred,
        target,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
        pid_weighting=pid_weighting,
        pid_weight_floor=pid_weight_floor,
        out_components=count_components,
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
        zero = pred_particles.sum() * 0.0
        return (zero, []) if return_breakdown else zero

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
    total = torch.stack(terms).sum()
    if not return_breakdown:
        return total

    # Labeled raw/weight/weighted breakdown, assembled in the same order the terms
    # were summed. Raw and weighted scalars are stacked and converted in a SINGLE
    # device->host sync each (not per term) to keep the per-batch overhead negligible.
    cat_label: list[tuple[str, str]] = []
    weights: list[float] = []
    raw_tensors: list[torch.Tensor] = []
    wtd_tensors: list[torch.Tensor] = []
    for pid in present_pids:
        cat_label.append(("pid_shape", str(pid)))
        weights.append(float(pid_weights[pid]))
        raw_tensors.append(object_raw[pid])
        wtd_tensors.append(object_wasserstein_distance[pid])
    for key, raw in event_wasserstein_distance.items():
        cat_label.append(("event", key))
        weights.append(float(event_weight))
        raw_tensors.append(raw)
        wtd_tensors.append(event_weight * raw)
    for (label, raw, weight), wtd in zip(count_components or [], count_terms):
        cat_label.append(("count", label))
        weights.append(float(weight))
        raw_tensors.append(raw)
        wtd_tensors.append(wtd)
    raw_vals = torch.stack(raw_tensors).detach().cpu().tolist()
    wtd_vals = torch.stack(wtd_tensors).detach().cpu().tolist()
    components = [
        LossComponent(cat, lbl, rv, w, wv)
        for (cat, lbl), w, rv, wv in zip(cat_label, weights, raw_vals, wtd_vals)
    ]
    return total, components


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
    return_breakdown: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[LossComponent]]:
    """DDP-aware wrapper: gathers pred/target across ranks before computing loss.

    The sliced Wasserstein distance is a *set-to-set* loss —
    SW_2(A ∪ B, C ∪ D) ≠ SW_2(A,C) + SW_2(B,D).  When each DDP rank computes
    the loss on its local shard and the results are averaged, the mathematical
    value differs from the non-distributed loss computed on the full batch.
    This wrapper fixes that by gathering every rank's predicted and target
    observables (via :func:`_all_gather_varlen`) so the loss is always computed
    on the union of all shards, making the DDP and single-process loss
    **identical** for the same underlying data.

    ``return_breakdown`` is forwarded to the inner loss. Because the inner loss runs
    on the gathered UNION, the returned ``(loss, components)`` breakdown is identical
    on every rank, so the caller can print it from the main rank alone.
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
            return_breakdown=return_breakdown,
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
        return_breakdown=return_breakdown,
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
    # Leading-2 pair-mass response ln(m_reco / m_truth): 0.02-wide bins over +-0.8 (the
    # scale_raw range (0.7, 1.3) is +-0.36). NOTE: coarser than the 1-3 % resolutions
    # the terms resolve; the bin-free wasserstein_1d is the loss of choice for those --
    # this grid only keeps soft_hist functional.
    "pair_r": torch.linspace(-0.8, 0.8, 81, dtype=torch.float64),
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
    return_breakdown: bool = False,
    eta_split: bool = True,
    pair_mass: bool = True,
    pair_mass_weight: float = PAIR_MASS_WEIGHT,
) -> torch.Tensor | tuple[torch.Tensor, list[LossComponent]]:
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
    no-op. Only the per-pid shape terms are weighted -- the ``log(HT)``, pair-mass and
    count terms are untouched.

    ``eta_split`` (default True): the ``log_E`` / ``log_pt`` shape terms
    (:data:`SHAPE_SPLIT_OBS`) are split by the reco object's |eta| region
    (:data:`SHAPE_ETA_EDGES`), one term per ``(pid, obs, region)`` present on BOTH sides
    (key ``"{pid}:{obs}:eta{r}"``), weighted by ``pid_weight * obj_weight * f_r`` with
    ``f_r`` the TARGET population fraction of that pid in region r (sum_r f_r = 1, so the
    sum stays on the scale of the pooled term -- W2^2(mixture) <= sum_r f_r W2^2_r by
    convexity -- and sparse regions cannot dominate). ``eta`` stays one pooled term.
    Rationale: the momentum-smearing / calo parameters are per-region; pooled over eta a
    per-region scale can only be fitted as a mixture. ``False`` reproduces the pooled
    terms bit-for-bit.

    ``pair_mass`` (default True): adds the per-event pair-mass shape terms
    (:func:`compute_pair_masses`: the RESPONSE ``ln(m_reco / m_truth)`` of the two
    leading-pt reco objects of each of pids 11/13/211 under the class mass hypothesis
    over the event's truth leading-2 pair; needs the ``truth_pair_lnm:{pid}`` labels of
    :func:`attach_truth_pair_lnm` on both dicts), one term per ``(pid, truth-mass group,
    |eta|-region pair category)`` present on both sides (key
    ``"pair:{pid}:mt{lo}-{hi}:eta{r1}{r2}"``, category ``"pair"``; groups from the fixed
    ``ln m_truth`` grid :data:`PAIR_TRUTH_LNM_WIDTH` -- see the note there on why the
    reco-mass mixture must not be compared as a whole), weight ``pair_mass_weight * f``
    with ``f`` the fraction of ALL target pairs (every class) in that (pid, group,
    category) -- the pair terms sum to ``pair_mass_weight`` and a class with a handful of
    stray pairs fades; standardized by the target response std pooled over classes.
    Count-blind like every shape term. On a resonance-gun sample (one J/psi / Z / K_S per
    event) the response width IS the track resolution (``a_raw``/``b_raw``) and its
    center the ``scale_raw`` -- the only 1-D observable that carries the smearing width
    on a smooth spectrum; on other samples the leading pair is a combinatorial
    continuum, identical on both sides.

    When ``return_breakdown`` is True the function returns ``(loss, components)`` -- a
    list of :class:`LossComponent` (one per shape term, pair term, the ``log_ht`` event
    term, and each per-class count term) carrying the raw (pre-weight) value, the weight
    applied, and the weighted (post-weight) value, with ``sum(c.weighted) == loss``.
    ``False`` (default) returns the bare scalar exactly as before, with zero extra work.
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

    # Per-pid, per-observable SHAPE terms. Keep the pre-weight value and the weight
    # alongside each term for the optional return_breakdown.
    pid_obs_terms: dict[str, torch.Tensor] = {}
    pid_obs_raw: dict[str, torch.Tensor] = {}
    pid_obs_weight: dict[str, float] = {}
    for pid in present_pids:
        x = pred_groups[pid]  # (n_pred, 3), differentiable
        y = target_groups[pid]  # (n_tgt, 3), reference (detached inside term_fn)
        if eta_split:
            # Region of every object (reco |eta|, column 2), and the target population
            # fraction per region (detached plain floats).
            rx = _eta_region_index(x[:, _OBS_COL["eta"]])
            ry = _eta_region_index(y[:, _OBS_COL["eta"]])
            n_y = torch.bincount(ry, minlength=N_SHAPE_ETA_REGIONS).tolist()
            n_x = torch.bincount(rx, minlength=N_SHAPE_ETA_REGIONS).tolist()
            tot_y = float(sum(n_y))
        for obs in ("log_E", "log_pt", "eta"):
            col = _OBS_COL[obs]
            base_weight = pid_weights[pid] * float(obj_weights.get(obs, 1.0))
            if eta_split and obs in SHAPE_SPLIT_OBS:
                for r in range(N_SHAPE_ETA_REGIONS):
                    if n_x[r] == 0 or n_y[r] == 0:
                        continue  # region absent on one side: no term (sum f_r < 1)
                    key = f"{pid}:{obs}:eta{r}"
                    raw = term_fn(x[rx == r, col], y[ry == r, col], obs)
                    weight = base_weight * (n_y[r] / tot_y)
                    pid_obs_raw[key] = raw
                    pid_obs_weight[key] = weight
                    pid_obs_terms[key] = weight * raw
                continue
            key = f"{pid}:{obs}"
            raw = term_fn(x[:, col], y[:, col], obs)
            pid_obs_raw[key] = raw
            pid_obs_weight[key] = base_weight
            pid_obs_terms[key] = base_weight * raw

    # Per-event pair-mass terms (leading-2 objects of each class, per |eta|-region pair
    # category); count-blind; the pred side keeps its graph through pt.
    pair_terms: dict[str, torch.Tensor] = {}
    pair_raw: dict[str, torch.Tensor] = {}
    pair_weight: dict[str, float] = {}
    if pair_mass and abs(pair_mass_weight) > 0.0:
        pred_pairs = compute_pair_masses(pred)
        tgt_pairs = compute_pair_masses(target)
        tot_t = float(sum(r.numel() for r, _c, _g in tgt_pairs.values()))  # all target pairs
        for pid in sorted(set(pred_pairs) & set(tgt_pairs)):
            rp, cp, gp = pred_pairs[pid]
            rt, ct, gt = tgt_pairs[pid]
            rt = rt.detach()
            for g in torch.unique(torch.cat([gp, gt])).tolist():
                for cat in range(N_SHAPE_ETA_REGIONS**2):
                    sp = (gp == g) & (cp == cat)
                    st = (gt == g) & (ct == cat)
                    if not (bool(sp.any()) and bool(st.any())):
                        continue
                    key = f"pair:{pid}:{_pair_truth_group_label(g)}:{_pair_category_label(cat)}"
                    raw = term_fn(rp[sp], rt[st], "pair_r")
                    weight = pair_mass_weight * (float(st.sum()) / tot_t)
                    pair_raw[key] = raw
                    pair_weight[key] = weight
                    pair_terms[key] = weight * raw

    # Event-level term: the same shape distance on the per-event log(HT) distribution,
    # down-weighted by event_weight (mirrors the Wasserstein event term; multiplicity is
    # intentionally excluded -- a hard pt != 0 count with no gradient). Capture the RAW
    # term directly (not event_term / event_weight, which would divide by zero when
    # event_weight == 0).
    event_term: torch.Tensor | None = None
    event_raw: torch.Tensor | None = None
    pv = pred["log_ht"].reshape(-1)
    tv = target["log_ht"].reshape(-1)
    if pv.numel() and tv.numel():
        event_raw = term_fn(pv, tv, "log_ht")
        event_term = event_weight * event_raw

    # Same per-species expected-count terms as the Wasserstein loss: the gradient source
    # for eff_logits and the calo resolution coefficients (the shape terms are
    # count-blind, so these carry the absolute multiplicity / membership signal).
    # pid_weighting also redistributes ACROSS the count species (mean-1; "equal" no-op).
    count_components: list | None = [] if return_breakdown else None
    count_terms = _count_terms(
        pred,
        target,
        count_weight=count_weight,
        calo_count_weight=calo_count_weight,
        count_rate_floor=count_rate_floor,
        pid_weighting=pid_weighting,
        pid_weight_floor=pid_weight_floor,
        out_components=count_components,
    )
    terms = (
        list(pid_obs_terms.values())
        + list(pair_terms.values())
        + ([event_term] if event_term is not None else [])
        + count_terms
    )
    if not terms:  # degenerate empty batch: keep a graph-connected zero
        zero = pred_particles.sum() * 0.0
        return (zero, []) if return_breakdown else zero
    # Opt-in per-component breakdown (set MCGEN_LOSS_DEBUG=1), mirroring the Wasserstein
    # loss: confirms the count terms sit on the same O(1) scale as the per-pid shape terms.
    if os.environ.get("MCGEN_LOSS_DEBUG"):
        pidh = {k: float(v) for k, v in pid_obs_terms.items()}
        pairh = {k: float(v) for k, v in pair_terms.items()}
        evt = float(event_term) if event_term is not None else 0.0
        cnt = [float(c) for c in count_terms]
        print(
            f"[loss] {debug_label}={pidh} pair(w)={pairh} event(w)={evt:.4f} "
            f"count(w)={cnt} total={float(torch.stack(terms).sum()):.4f}",
            flush=True,
        )
    total = torch.stack(terms).sum()
    if not return_breakdown:
        return total

    # Labeled raw/weight/weighted breakdown, in the same order the terms were summed.
    # Raw and weighted scalars are stacked and converted in a SINGLE device->host sync
    # each (not per term) to keep the per-batch overhead negligible.
    cat_label: list[tuple[str, str]] = []
    weights: list[float] = []
    raw_tensors: list[torch.Tensor] = []
    wtd_tensors: list[torch.Tensor] = []
    for key in pid_obs_terms:  # pid-major, obs-minor insertion order
        cat_label.append(("pid_shape", key))
        weights.append(pid_obs_weight[key])
        raw_tensors.append(pid_obs_raw[key])
        wtd_tensors.append(pid_obs_terms[key])
    for key, wtd in pair_terms.items():
        cat_label.append(("pair", key))
        weights.append(pair_weight[key])
        raw_tensors.append(pair_raw[key])
        wtd_tensors.append(wtd)
    if event_term is not None:
        cat_label.append(("event", "log_ht"))
        weights.append(float(event_weight))
        raw_tensors.append(event_raw)
        wtd_tensors.append(event_term)
    for (label, raw, weight), wtd in zip(count_components or [], count_terms):
        cat_label.append(("count", label))
        weights.append(float(weight))
        raw_tensors.append(raw)
        wtd_tensors.append(wtd)
    raw_vals = torch.stack(raw_tensors).detach().cpu().tolist()
    wtd_vals = torch.stack(wtd_tensors).detach().cpu().tolist()
    components = [
        LossComponent(cat, lbl, rv, w, wv)
        for (cat, lbl), w, rv, wv in zip(cat_label, weights, raw_vals, wtd_vals)
    ]
    return total, components


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
    return_breakdown: bool = False,
    eta_split: bool = True,
    pair_mass: bool = True,
    pair_mass_weight: float = PAIR_MASS_WEIGHT,
) -> torch.Tensor | tuple[torch.Tensor, list[LossComponent]]:
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
        return_breakdown=return_breakdown,
        eta_split=eta_split,
        pair_mass=pair_mass,
        pair_mass_weight=pair_mass_weight,
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
    # remove any invalid values
    valid = valid[torch.isfinite(valid).all(dim=-1)]
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
    # Pair-mass terms: ONE scale on the target's response ln(m_reco/m_truth), pooled over
    # classes, truth groups and |eta| pair categories (the response is dimensionless and
    # centered alike everywhere, so the pooled std is the resolution; a per-class /
    # per-group std would collapse on a class with a handful of stray pairs).
    has_labels = any(k.startswith(("truth_pair_lnm:", "pair_r:")) for k in target)
    resp = [r.detach() for r, _c, _g in (compute_pair_masses(target) if has_labels else {}).values()]
    v = torch.cat(resp) if resp else torch.zeros(0, dtype=target_particles.dtype)
    v = v[torch.isfinite(v)]
    scales["pair_r"] = (
        v.std(unbiased=False).clamp(min=eps_floor)
        if v.numel() > 1
        else torch.ones((), dtype=target_particles.dtype)
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
    return_breakdown: bool = False,
    eta_split: bool = True,
    pair_mass: bool = True,
    pair_mass_weight: float = PAIR_MASS_WEIGHT,
) -> torch.Tensor | tuple[torch.Tensor, list[LossComponent]]:
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
        return_breakdown=return_breakdown,
        eta_split=eta_split,
        pair_mass=pair_mass,
        pair_mass_weight=pair_mass_weight,
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
    return_breakdown: bool = False,
    eta_split: bool = True,
    pair_mass: bool = True,
    pair_mass_weight: float = PAIR_MASS_WEIGHT,
) -> torch.Tensor | tuple[torch.Tensor, list[LossComponent]]:
    """DDP-aware wrapper: gathers pred/target across ranks before computing the
    bin-free 1D Wasserstein loss.

    Like the sliced Wasserstein, the per-pid 1D Wasserstein distance is a
    set-to-set loss that does not commute with summation over disjoint shards.
    This wrapper gathers every rank's predicted and target observables so the
    loss is computed on the union of all shards, making DDP and single-process
    results identical.

    ``return_breakdown`` is forwarded to the inner loss; because the inner loss runs
    on the gathered UNION, the returned ``(loss, components)`` breakdown is identical
    on every rank (printable from the main rank alone).
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
            return_breakdown=return_breakdown,
            eta_split=eta_split,
            pair_mass=pair_mass,
            pair_mass_weight=pair_mass_weight,
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

    # ---- per-event pair-mass responses ---------------------------------------
    # Computed per rank from the (per-event) pt/eta/phi/pid + truth labels, then gathered
    # as flat arrays; the inner loss picks the precomputed "pair_r/pair_cat/pair_grp:{pid}"
    # entries up (compute_pair_masses) instead of recomputing from per-event tensors,
    # which the gathered dicts no longer carry. Every rank must issue the same
    # collectives, so gather every hypothesis pid (possibly empty on this rank).
    if pair_mass:
        pred_pairs = compute_pair_masses(pred)
        tgt_pairs = compute_pair_masses(target)
        dt, dev = pred["pt"].dtype, pred["pt"].device
        empty_idx = torch.zeros(0, dtype=torch.long, device=dev)
        # compute_pair_masses OMITS a pid it found no pair for, and whether a rank
        # has a pair for a hypothesis depends on its own data shard. The pred-side
        # fallback must therefore stay ON the autograd graph even when empty:
        # all_gather registers its backward (REDUCE_SCATTER) node only when the
        # input requires grad, so a plain torch.zeros(0) makes THIS rank skip a
        # collective that every rank WITH a pair still issues. The forward still
        # matches, and the job then deadlocks in BACKWARD until the NCCL watchdog
        # kills it ~10 min later. Slicing a graph tensor to zero length is empty but
        # keeps grad_fn -- the same idiom the per-particle gathers above rely on
        # (`pred[key].reshape(-1)[pred_valid]` stays on the graph when it selects
        # nothing). _all_gather_varlen now also asserts this parity across ranks.
        # The target side is gathered with differentiable=False and needs no care.
        pred_empty = (pred["pt"].reshape(-1)[:0], empty_idx, empty_idx)
        tgt_empty = (torch.zeros(0, dtype=dt, device=dev), empty_idx, empty_idx)
        for pid in sorted(PAIR_MASS_HYPOTHESIS):
            rp, cp, gp = pred_pairs.get(pid, pred_empty)
            rt, ct, gt = tgt_pairs.get(pid, tgt_empty)
            pred_gathered[f"pair_r:{pid}"] = _all_gather_varlen(rp, differentiable=True)
            pred_gathered[f"pair_cat:{pid}"] = _all_gather_varlen(cp.to(dt), differentiable=False)
            pred_gathered[f"pair_grp:{pid}"] = _all_gather_varlen(gp.to(dt), differentiable=False)
            target_gathered[f"pair_r:{pid}"] = _all_gather_varlen(rt.detach(), differentiable=False)
            target_gathered[f"pair_cat:{pid}"] = _all_gather_varlen(ct.to(dt), differentiable=False)
            target_gathered[f"pair_grp:{pid}"] = _all_gather_varlen(gt.to(dt), differentiable=False)

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
        return_breakdown=return_breakdown,
        eta_split=eta_split,
        pair_mass=pair_mass,
        pair_mass_weight=pair_mass_weight,
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
