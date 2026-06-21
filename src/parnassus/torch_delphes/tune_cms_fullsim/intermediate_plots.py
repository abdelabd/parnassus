"""Per-epoch intermediate observable plots for ``tune_cms_fullsim`` training.

While :mod:`tune_cms_fullsim.plot_fit_results` makes the *offline* paper figures
once training has finished, this module renders a quick-look figure **after every
training epoch** so the fit can be watched as it converges.

:func:`save_intermediate_observable_plots` writes a single multi-page PDF per
epoch (``intermediate_epoch_<step>.pdf``). The first pages are the combined
(all-PID) observables, one per page; they are followed by one **per-PID** page
per particle type (charged hadron / electron / muon / neutral hadron / photon),
each a grid of the per-particle observables (``log_pt`` / ``log_E`` / ``eta`` /
``pt``) for that subgroup -- the per-epoch analogue of the per-PID figures
:mod:`tune_cms_fullsim.plot_fit_results` writes offline. Each panel overlays the
full-sim target, the current-epoch trainee prediction, and a faint epoch-0
reference, and shows that observable's soft-histogram MSE in the title as a quick
distribution-mismatch diagnostic (this is display-only; the training loss lives in
:mod:`tune_cms_fullsim.loss`). The bin edges are derived per panel from the pooled
target/prediction range (linear, ``_N_BINS`` bins); the same edges feed both the
title MSE and the plotted histogram, so the number always corresponds exactly to
the curves shown.

This module is imported lazily from :mod:`tune_cms_fullsim.training` (only on the
main rank, only when plotting is enabled) so matplotlib never enters the hot
training import path.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend for headless / srun runs

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

import torch

from .loss import histogram_mse_loss

# Order of pages in the per-epoch PDF: the particle-level observables first,
# then the per-event scalars. Keys missing from a run's obs dict are skipped.
_PANEL_ORDER: tuple[str, ...] = (
    "pt", "eta", "log_pt", "log_E", "ht", "log_ht", "multiplicity"
)

# Axis labels (the pt/eta/ht/multiplicity ones mirror plot_fit_results.main).
_XLABELS: dict[str, str] = {
    "pt": r"PF object $p_\mathrm{T}$ [GeV]",
    "eta": r"PF object $\eta$",
    "log_pt": r"PF object $\log\,p_\mathrm{T}$",
    "log_E": r"PF object $\log\,E$ [GeV]",
    "ht": r"PF scalar $H_\mathrm{T}$ [GeV]",
    "log_ht": r"PF scalar $\log\,H_\mathrm{T}$",
    "multiplicity": r"PF objects per event",
}

# Observables drawn on a log-y scale (wide dynamic range / steep tails).
_LOG_Y: frozenset[str] = frozenset({"pt"})

# Number of (linear) bins for the per-epoch histograms, derived from the data.
_N_BINS: int = 50

# Percentile used to clip the auto bin range (edges span the _CLIP_PCT-th to the
# (100 - _CLIP_PCT)-th percentile of the pooled values). Mirrors
# parnassus.torch_delphes.plotting._auto_bins so heavy tails (e.g. a few high-pt
# charged hadrons) don't squash the bulk of a per-PID distribution into one bin.
_CLIP_PCT: float = 1.0

# The observables the Wasserstein training loss actually optimizes. Panels for
# observables outside this set are still drawn (for reference) but annotated as
# not being part of the loss.
_LOSS_OBSERVABLES: frozenset[str] = frozenset({"log_E", "log_pt", "eta", "log_ht"})

# Softness of the diagnostic soft-histogram MSE shown in each panel title. This
# is a display-only diagnostic (the training loss is the Wasserstein distance),
# so it is fixed here rather than exposed as a CLI flag.
_DIAG_BETA: float = 0.15

# Per-PID pages: one page per particle type (mirrors
# plot_fit_results._FINAL_PID_GROUPS), each a grid of the per-particle observables
# for that |pid| subgroup. (name, |pid|, human label).
_PID_GROUPS: tuple[tuple[str, int, str], ...] = (
    ("211", 211, "charged hadron"),
    ("11", 11, "electron"),
    ("13", 13, "muon"),
    ("111", 111, "neutral hadron"),
    ("22", 22, "photon"),
)

# Per-particle observables drawn on each per-PID page (laid out as a 2-column
# grid). Per-event scalars (ht/log_ht/multiplicity) have no per-PID meaning and are
# intentionally omitted here.
_PER_PID_PANEL_ORDER: tuple[str, ...] = ("log_pt", "log_E", "eta", "pt")


def _histogram_counts(values: torch.Tensor, bin_edges: np.ndarray) -> np.ndarray:
    """Return plain histogram counts (not normalized) on ``bin_edges``.

    This helper keeps :mod:`intermediate_plots` self-contained and avoids
    importing private helpers from :mod:`plot_fit_results`.
    """
    x = values.detach().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.float64)
    counts, _ = np.histogram(x.cpu().numpy(), bins=bin_edges)
    return counts.astype(np.float64)


def _auto_bin_edges(
    value_tensors: list[torch.Tensor | None], n_bins: int = _N_BINS
) -> np.ndarray | None:
    """Percentile-clipped linear bin edges spanning the given 1-D tensors.

    The target, prediction and (optional) initial-reference values for ONE panel are
    pooled, then the edges run from the ``_CLIP_PCT``-th to the
    ``100 - _CLIP_PCT``-th percentile of the pooled finite values -- mirroring
    :func:`parnassus.torch_delphes.plotting._auto_bins` and the per-PID figures in
    :mod:`tune_cms_fullsim.plot_fit_results` -- so heavy tails (e.g. a handful of
    high-pt charged hadrons) don't squash the bulk into one bin. Because each panel is
    fed only its own subgroup's values (per observable, and per PID on the per-PID
    pages), the range is automatically **per-PID**. Non-finite entries are ignored.
    Returns ``None`` when no finite value is available (the caller then skips that page
    / blanks that grid cell), and widens a zero-width range (all values identical) into
    a small non-degenerate interval so the edges stay strictly increasing for both
    ``np.histogram`` and ``soft_histogram``.
    """
    finite: list[np.ndarray] = []
    for v in value_tensors:
        if v is None or v.numel() == 0:
            continue
        x = v.detach().reshape(-1)
        x = x[torch.isfinite(x)]
        if x.numel() > 0:
            finite.append(x.cpu().numpy())
    if not finite:
        return None
    pooled = np.concatenate(finite)
    lo = float(np.percentile(pooled, _CLIP_PCT))
    hi = float(np.percentile(pooled, 100.0 - _CLIP_PCT))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if hi <= lo:  # identical / degenerate clip -> widen to a strictly-increasing range
        pad = max(abs(lo), 1.0) * 1e-3
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_bins + 1)


def _draw_observable_panel(
    ax,
    key: str,
    tgt_vals: torch.Tensor,
    pred_vals: torch.Tensor,
    init_vals: torch.Tensor | None,
    step: int,
) -> bool:
    """Draw one observable's target/initial/current histograms onto ``ax``.

    Bins are derived from the pooled target/pred/init range (the same edges feed the
    title soft-hist MSE diagnostic and every histogram, so they stay consistent).
    Returns ``False`` without drawing when no side has any finite value (the caller
    then skips the page or blanks the subplot).
    """
    np_edges = _auto_bin_edges([tgt_vals, pred_vals, init_vals])
    if np_edges is None:  # no finite values on any side -> nothing to draw
        return False
    centers = 0.5 * (np_edges[1:] + np_edges[:-1])

    # Unweighted per-observable soft-hist MSE over these bins -- a distribution-
    # mismatch diagnostic. NaN when a side is empty.
    if pred_vals.numel() == 0 or tgt_vals.numel() == 0:
        mse = float("nan")
    else:
        edges_t = torch.as_tensor(np_edges, dtype=pred_vals.dtype)
        mse = float(histogram_mse_loss(pred_vals, tgt_vals, edges_t, beta=_DIAG_BETA))

    ax.step(
        centers,
        _histogram_counts(tgt_vals, np_edges),
        where="mid",
        color="black",
        label="target (full sim)",
    )
    if init_vals is not None and init_vals.numel() > 0:
        ax.step(
            centers,
            _histogram_counts(init_vals, np_edges),
            where="mid",
            color="tab:red",
            linestyle="--",
            alpha=0.4,
            label="trainee, initial",
        )
    ax.step(
        centers,
        _histogram_counts(pred_vals, np_edges),
        where="mid",
        color="tab:blue",
        label=f"trainee, epoch {step}",
    )

    ax.set_xlabel(_XLABELS.get(key, key))
    ax.set_ylabel("Counts")
    if key in _LOG_Y:
        ax.set_yscale("log")

    title = f"{key}: soft-hist MSE = {mse:.3e}"
    if key not in _LOSS_OBSERVABLES:
        title += " (not in loss)"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return True


def _page_footer(fig, step: int, val_loss: float | None) -> None:
    """Stamp the shared ``epoch N | val_loss`` footer on a page."""
    footer = f"epoch {step}"
    if val_loss is not None:
        footer += f"   |   val_loss = {val_loss:.4e}"
    fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=8, alpha=0.6)


def _render_per_pid_pages(
    pdf: PdfPages,
    pred_by_key: dict[str, torch.Tensor],
    target_by_key: dict[str, torch.Tensor],
    init_by_key: dict[str, torch.Tensor] | None,
    step: int,
    val_loss: float | None,
) -> None:
    """Append one page per PID subgroup, each a grid of per-particle observables.

    The collected ``pid`` array is element-aligned with every per-particle observable
    (the training loop strips them all with the same ``pt != 0`` cut, in the same
    order), so each PID subgroup is a boolean-mask slice ``pid.abs() == |pid|``.
    Silently does nothing when ``pid`` was not collected.
    """
    pid_pred = pred_by_key.get("pid")
    pid_tgt = target_by_key.get("pid")
    if pid_pred is None or pid_tgt is None:
        return
    pid_init = init_by_key.get("pid") if init_by_key is not None else None

    keys = [k for k in _PER_PID_PANEL_ORDER if k in pred_by_key and k in target_by_key]
    if not keys:
        return

    ncols = 2
    nrows = math.ceil(len(keys) / ncols)
    for pid_name, pid_abs, pid_label in _PID_GROUPS:
        m_pred = pid_pred.abs() == pid_abs
        m_tgt = pid_tgt.abs() == pid_abs
        m_init = (pid_init.abs() == pid_abs) if pid_init is not None else None
        if not (bool(m_pred.any()) or bool(m_tgt.any())):
            continue  # no objects of this type on either side -> skip the page

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False
        )
        axes_flat = axes.flatten()
        for i, key in enumerate(keys):
            tv = target_by_key[key][m_tgt]
            pv = pred_by_key[key][m_pred]
            iv = (
                init_by_key[key][m_init]
                if (init_by_key is not None and key in init_by_key and m_init is not None)
                else None
            )
            if not _draw_observable_panel(axes_flat[i], key, tv, pv, iv, step):
                axes_flat[i].set_axis_off()
                axes_flat[i].set_title(f"{key}: (no {pid_label})")
        for j in range(len(keys), len(axes_flat)):  # blank unused grid cells
            axes_flat[j].set_axis_off()

        fig.suptitle(f"per-PID {pid_label} (|pid| = {pid_abs})")
        _page_footer(fig, step, val_loss)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def save_intermediate_observable_plots(
    pred_by_key: dict[str, torch.Tensor],
    target_by_key: dict[str, torch.Tensor],
    observables: list[str],
    step: int,
    output_dir: str | Path,
    val_loss: float | None = None,
    init_by_key: dict[str, torch.Tensor] | None = None,
) -> Path:
    """Write one multi-page PDF (one observable per page) for a training epoch.

    Parameters
    ----------
    pred_by_key, target_by_key : dict[str, torch.Tensor]
        Per-observable **flattened, padding/ghost-stripped** 1-D values for the
        whole validation set, for the trainee prediction and the full-sim
        target respectively. Both dicts share the same observable keys.
    step : int
        Epoch index; controls the output filename and is shown on each page.
    output_dir : str | Path
        Directory to write ``intermediate_epoch_<step>.pdf`` into. Assumed to
        already exist (the caller creates it once).
    val_loss : float | None
        Optional total validation loss to annotate on each page.
    init_by_key : dict[str, torch.Tensor] | None
        Optional epoch-0 prediction, drawn as a faint dashed reference.

    Returns
    -------
    pathlib.Path
        The path of the written PDF.
    """
    out_path = Path(output_dir) / f"intermediate_epoch_{step:03d}.pdf"

    with PdfPages(out_path) as pdf:
        # Combined (all-PID) pages: one observable per page.
        for key in _PANEL_ORDER:
            if key not in observables or key not in pred_by_key or key not in target_by_key:
                continue

            pred_vals = pred_by_key[key]
            tgt_vals = target_by_key[key]
            init_vals = init_by_key.get(key) if init_by_key is not None else None

            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            if _draw_observable_panel(ax, key, tgt_vals, pred_vals, init_vals, step):
                _page_footer(fig, step, val_loss)
                fig.tight_layout()
                pdf.savefig(fig)
            plt.close(fig)

        # Per-PID pages: one page per particle type (22, 11, 13, 111, 211), a grid of
        # the per-particle observables (log_pt / log_E / eta / pt) for that subgroup.
        _render_per_pid_pages(pdf, pred_by_key, target_by_key, init_by_key, step, val_loss)

    return out_path
