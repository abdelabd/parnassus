"""Per-epoch intermediate observable plots for ``tune_cms_fullsim`` training.

While :mod:`tune_cms_fullsim.plot_fit_results` makes the *offline* paper figures
once training has finished, this module renders a quick-look figure **after every
training epoch** so the fit can be watched as it converges.

:func:`save_intermediate_observable_plots` writes a single multi-page PDF per
epoch (``intermediate_epoch_<step>.pdf``), one observable per page. Each page
overlays the full-sim target, the current-epoch trainee prediction, and a faint
epoch-0 reference, and shows that observable's *unweighted* soft-histogram MSE
in the title -- i.e. its loss contribution **before** the per-observable weight
(``DEFAULT_OBS_WEIGHTS``) is applied. The same bin edges and ``beta`` the loss
uses are passed in, so the number in the title corresponds exactly to the
histogram shown.

This module is imported lazily from :mod:`tune_cms_fullsim.training` (only on the
main rank, only when plotting is enabled) so matplotlib never enters the hot
training import path.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend for headless / srun runs

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import torch

# Reuse the exact density-histogram helper (and, as an import side effect, the
# shared rcParams styling) from the offline plotting script so intermediate and
# final figures look identical.
from .loss import histogram_mse_loss
from .plot_fit_results import _density_histogram

# Order of pages in the per-epoch PDF: the four loss-active particle-level
# observables first, then the two per-event scalars (weight 0 by default).
_PANEL_ORDER: tuple[str, ...] = ("pt", "eta", "E", "log_pt", "ht", "multiplicity")

# Axis labels (the pt/eta/ht/multiplicity ones mirror plot_fit_results.main).
_XLABELS: dict[str, str] = {
    "pt": r"PF object $p_\mathrm{T}$ [GeV]",
    "eta": r"PF object $\eta$",
    "E": r"PF object $E$ [GeV]",
    "log_pt": r"PF object $\log\,p_\mathrm{T}$",
    "ht": r"PF scalar $H_\mathrm{T}$ [GeV]",
    "multiplicity": r"PF objects per event",
}

# Observables drawn on a log-y scale (wide dynamic range / steep tails).
_LOG_Y: frozenset[str] = frozenset({"pt", "E"})


def save_intermediate_observable_plots(
    pred_by_key: dict[str, torch.Tensor],
    target_by_key: dict[str, torch.Tensor],
    edges: dict[str, torch.Tensor],
    weights: dict[str, float],
    beta: float,
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
        target respectively. Same keys as ``edges``.
    edges : dict[str, torch.Tensor]
        The loss bin edges (``DEFAULT_BIN_EDGES``); the displayed MSE is computed
        on these bins so it matches the histogram drawn.
    weights : dict[str, float]
        Per-observable loss weights; used only to flag weight-0 observables
        (shown but not part of the optimized loss).
    beta : float
        Soft-histogram softness used by the loss (so the title MSE matches).
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
            if key not in edges or key not in pred_by_key or key not in target_by_key:
                continue

            pred_vals = pred_by_key[key]
            tgt_vals = target_by_key[key]
            init_vals = init_by_key.get(key) if init_by_key is not None else None

            np_edges = edges[key].detach().cpu().numpy()
            centers = 0.5 * (np_edges[1:] + np_edges[:-1])

            # Unweighted per-observable soft-hist MSE -- the loss contribution
            # before multiplying by weights[key]. NaN when a side is empty.
            if pred_vals.numel() == 0 or tgt_vals.numel() == 0:
                mse = float("nan")
            else:
                mse = float(
                    histogram_mse_loss(
                        pred_vals, tgt_vals, edges[key].detach().cpu(), beta=beta
                    )
                )

            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            ax.step(
                centers,
                _density_histogram(tgt_vals, np_edges),
                where="mid",
                color="black",
                label="target (full sim)",
            )
            if init_vals is not None:
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
            if weights.get(key, 1.0) == 0:
                title += " (weight 0, not in loss)"
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
