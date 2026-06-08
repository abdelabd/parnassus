"""Per-epoch intermediate observable plots for ``tune_cms_fullsim`` training.

While :mod:`tune_cms_fullsim.plot_fit_results` makes the *offline* paper figures
once training has finished, this module renders a quick-look figure **after every
training epoch** so the fit can be watched as it converges.

:func:`save_intermediate_observable_plots` writes a single multi-page PDF per
epoch (``intermediate_epoch_<step>.pdf``), one observable per page. Each page
overlays the full-sim target, the current-epoch trainee prediction, and a faint
epoch-0 reference, and shows that observable's soft-histogram MSE in the title
as a quick distribution-mismatch diagnostic (this is display-only; the training
loss is the sliced-Wasserstein distance in :mod:`tune_cms_fullsim.loss`). The bin edges are
derived per observable from the pooled target/prediction range (linear,
``_N_BINS`` bins); the same edges feed both the title MSE and the plotted
histogram, so the number always corresponds exactly to the curves shown.

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

# Reuse the exact density-histogram helper (and, as an import side effect, the
# shared rcParams styling) from the offline plotting script so intermediate and
# final figures look identical.
from .loss import histogram_mse_loss
from .plot_fit_results import _density_histogram

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

# The observables the Wasserstein training loss actually optimizes. Panels for
# observables outside this set are still drawn (for reference) but annotated as
# not being part of the loss.
_LOSS_OBSERVABLES: frozenset[str] = frozenset({"log_E", "log_pt", "eta", "log_ht"})

# Softness of the diagnostic soft-histogram MSE shown in each panel title. This
# is a display-only diagnostic (the training loss is the Wasserstein distance),
# so it is fixed here rather than exposed as a CLI flag.
_DIAG_BETA: float = 0.15


def _auto_bin_edges(
    value_tensors: list[torch.Tensor | None], n_bins: int = _N_BINS
) -> np.ndarray | None:
    """Shared linear bin edges spanning the finite range of all given 1-D tensors.

    The target, prediction and (optional) initial-reference values for one
    observable are pooled so every overlaid histogram lands on the same axis.
    Non-finite entries are ignored. Returns ``None`` when no finite value is
    available (the caller then skips that page), and widens a zero-width range
    (all values identical) into a small non-degenerate interval so the edges
    stay strictly increasing for both ``np.histogram`` and ``soft_histogram``.
    """
    lo, hi = math.inf, -math.inf
    for v in value_tensors:
        if v is None or v.numel() == 0:
            continue
        x = v.detach().reshape(-1)
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            continue
        lo = min(lo, float(x.min()))
        hi = max(hi, float(x.max()))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    if hi <= lo:  # all values identical -> widen to a strictly-increasing range
        pad = max(abs(lo), 1.0) * 1e-3
        lo, hi = lo - pad, hi + pad
    return np.linspace(lo, hi, n_bins + 1)


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
        for key in _PANEL_ORDER:
            if key not in observables or key not in pred_by_key or key not in target_by_key:
                continue

            pred_vals = pred_by_key[key]
            tgt_vals = target_by_key[key]
            init_vals = init_by_key.get(key) if init_by_key is not None else None

            # Bins are derived per observable from the pooled target/pred/init
            # range (the loss no longer carries fixed bin edges). The same edges
            # feed the title MSE and every histogram, so they stay consistent.
            np_edges = _auto_bin_edges([tgt_vals, pred_vals, init_vals])
            if np_edges is None:  # no finite values on any side -> nothing to draw
                continue
            centers = 0.5 * (np_edges[1:] + np_edges[:-1])

            # Unweighted per-observable soft-hist MSE over these bins -- a
            # distribution-mismatch diagnostic. NaN when a side is empty.
            if pred_vals.numel() == 0 or tgt_vals.numel() == 0:
                mse = float("nan")
            else:
                edges_t = torch.as_tensor(np_edges, dtype=pred_vals.dtype)
                mse = float(histogram_mse_loss(pred_vals, tgt_vals, edges_t, beta=_DIAG_BETA))

            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.step(
                centers,
                _density_histogram(tgt_vals, np_edges),
                where="mid",
                color="black",
                label="target (full sim)",
            )
            if init_vals is not None and init_vals.numel() > 0:
                ax.step(
                    centers,
                    _density_histogram(init_vals, np_edges),
                    where="mid",
                    color="tab:red",
                    linestyle="--",
                    alpha=0.4,
                    label="trainee, initial",
                )
            ax.step(
                centers,
                _density_histogram(pred_vals, np_edges),
                where="mid",
                color="tab:blue",
                label=f"trainee, epoch {step}",
            )

            ax.set_xlabel(_XLABELS.get(key, key))
            ax.set_ylabel("normalised density")
            if key in _LOG_Y:
                ax.set_yscale("log")

            title = f"{key}: soft-hist MSE = {mse:.3e}"
            if key not in _LOSS_OBSERVABLES:
                title += " (not in loss)"
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            footer = f"epoch {step}"
            if val_loss is not None:
                footer += f"   |   val_loss = {val_loss:.4e}"
            fig.text(0.99, 0.01, footer, ha="right", va="bottom", fontsize=8, alpha=0.6)

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    return out_path
