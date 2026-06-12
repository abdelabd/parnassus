"""Shared plotting helpers for TorchDelphes validation and tuning diagnostics.

This module exists so that :mod:`validation.validate_torch_delphes`
(TorchDelphes-vs-C++Delphes) and
:mod:`tune_cms_fullsim.plot_fit_results` (trainee-vs-target) can produce
visually identical comparison plots:

- the same per-variable styling (axis labels, log-y rules, discrete-bar
  handling for PID / Charge / Status);
- the same histogram + ratio layout (``height_ratios=[3, 1]``, stepfilled
  reference, step overlays, ratio panel below);
- the same multi-panel stitching (``all.png`` combining four per-variable
  PNGs into one image with a centred multi-line title bar).

The two harnesses differ only in *what* they compare (C++ Delphes vs
TorchDelphes; or, target vs trainee-init vs trainee-fitted), not in *how*
the comparison is drawn — so all the matplotlib / PIL work lives here.

The constants (``AXIS_LABELS``, ``TITLE_LABELS``, ``DISCRETE_VARS``,
``LOG_Y_VARS``) are re-exported from
:mod:`tune_cms_fullsim.debug` for back-compat, but the canonical
definitions now live in this module.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from PIL import Image

# =============================================================================
# Per-variable styling constants
# =============================================================================

# Pretty (LaTeX) display labels for plot titles. Mirrors the convention in
# validate_torch_delphes.py: titles are bare symbols, axis labels include units.
TITLE_LABELS: dict[str, str] = {
    "Eta": r"$\eta$",
    "Phi": r"$\phi$",
    "PT": r"$p_T$",
    "ET": r"$E_T$",
    "P": r"$p$",
    "E": r"$E$",
}

# Pretty (LaTeX) axis labels per variable. Includes units where applicable.
AXIS_LABELS: dict[str, str] = {
    "Eta": r"$\eta$",
    "EtaOuter": r"$\eta_\mathrm{outer}$",
    "Phi": r"$\phi$",
    "PT": r"$p_T$ [GeV]",
    "ET": r"$E_T$ [GeV]",
    "P": r"$p$ [GeV]",
    "E": r"$E$ [GeV]",
    "PID": "PID",
    "Charge": "Charge",
    "Status": "Status",
    "Mass": "Mass [GeV]",
    "T": r"$t$",
    "X": r"$x$",
    "Y": r"$y$",
    "Z": r"$z$",
    "Px": r"$p_x$ [GeV]",
    "Py": r"$p_y$ [GeV]",
    "Pz": r"$p_z$ [GeV]",
}

# Variables that take discrete integer values; rendered as grouped bar charts
# rather than continuous histograms.
DISCRETE_VARS: frozenset[str] = frozenset({"PID", "Charge", "Status"})

# Variables drawn on a log-y scale (wide dynamic range / steep tails).
LOG_Y_VARS: frozenset[str] = frozenset({"PT", "ET", "P", "E"})

# Four kinematic variables we combine into the per-module ``all.png`` overview,
# selected based on which fields a given module's tensor carries. Mirrors the
# ``combined_vars`` selection in ``validate_torch_delphes.validate_against_benchmark``.
DEFAULT_COMBINED_VARS_TRACK: tuple[str, ...] = ("Eta", "Phi", "PT", "P")
DEFAULT_COMBINED_VARS_TOWER: tuple[str, ...] = ("Eta", "Phi", "E", "ET")
DEFAULT_COMBINED_VARS_PARTICLE: tuple[str, ...] = ("Eta", "Phi", "PT", "E")


def title_label(var: str) -> str:
    """Return a LaTeX-formatted display label for ``var`` (falls back to raw name)."""
    return TITLE_LABELS.get(var, var)


def axis_label(var: str) -> str:
    """Return a LaTeX-formatted axis label for ``var`` (falls back to raw name)."""
    return AXIS_LABELS.get(var, var)


def is_discrete(var: str) -> bool:
    """Whether ``var`` is a discrete integer variable (PID / Charge / Status)."""
    return var in DISCRETE_VARS


def log_y_for_var(var: str) -> bool:
    """Whether the variable should be plotted on a log-y scale by default."""
    return var in LOG_Y_VARS


def apply_axis_scale(ax: Axes, var: str) -> None:
    """Apply the per-variable y-scale (currently: log-y for PT / ET / P / E)."""
    if log_y_for_var(var):
        ax.set_yscale("log")


def combined_vars_for(kinematic_vars: Sequence[str]) -> tuple[str, ...]:
    """Pick the four-variable combo used in ``all.png`` for a given module.

    Mirrors the heuristic in :func:`validate_against_benchmark`:

    - tracks (have ``P`` but no ``E``)        → ``(Eta, Phi, PT, P)``
    - towers (have ``E`` but no ``P``)        → ``(Eta, Phi, E, ET)``
    - eflow-style (have both)                 → ``(Eta, Phi, PT, P)``
    - particle-style (rare)                   → ``(Eta, Phi, PT, E)``
    - fallback                                → first four entries of input
    """
    has_p = "P" in kinematic_vars
    has_e = "E" in kinematic_vars
    if has_p and not has_e:
        return DEFAULT_COMBINED_VARS_TRACK
    if has_e and not has_p:
        return DEFAULT_COMBINED_VARS_TOWER
    if has_p and has_e:
        # Modules that carry both prefer the track-style combo for
        # consistency with track branches.
        return DEFAULT_COMBINED_VARS_TRACK
    return tuple(kinematic_vars[:4])


# =============================================================================
# Histogram + ratio plotting (N-curve comparison)
# =============================================================================


def _auto_bins(values: Sequence[np.ndarray], n_bins: int) -> np.ndarray | int:
    """Return the percentile-clipped linear bin edges spanning ``values``.

    Mirrors the binning used by :mod:`validation.validate_torch_delphes`:
    edges run from the 1st to 99th percentile of the pooled values, so heavy
    tails (PT, P, E, …) don't squash the bulk into one bin. Returns
    ``n_bins`` as a fallback when no finite data is available, so callers can
    pass the result straight to :func:`numpy.histogram` / ``Axes.hist``.
    """
    if not values:
        return n_bins
    pooled = np.concatenate([np.asarray(v) for v in values if len(v) > 0])
    if pooled.size == 0:
        return n_bins
    pooled = pooled[np.isfinite(pooled)]
    if pooled.size == 0:
        return n_bins
    lo = float(np.percentile(pooled, 1.0))
    hi = float(np.percentile(pooled, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        center = float(np.mean(pooled)) if np.isfinite(pooled).any() else 0.0
        lo, hi = center - 0.5, center + 0.5
    return np.linspace(lo, hi, n_bins + 1)


def plot_comparison_with_ratio(
    distributions: Sequence[tuple[np.ndarray, str, str]],
    var: str,
    output_path: Path,
    *,
    n_bins: int = 50,
    bins: np.ndarray | Sequence[float] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Counts",
    ratio_ylabel: str | None = None,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
    legend_loc: str | tuple[float, float] = "best",
    log_y: bool | None = None,
) -> None:
    """Plot N distributions with a ratio panel underneath.

    The first entry in ``distributions`` is the **reference** — drawn as a
    stepfilled histogram (or stepfilled bars for discrete variables). The
    remaining entries are **overlays** drawn as step lines (or grouped bars).
    The lower panel shows ``counts(overlay_i) / counts(reference)`` for each
    overlay, in the overlay's color.

    Visual style is identical to
    :mod:`validation.validate_torch_delphes` (``height_ratios=[3, 1]``,
    ``hspace=0.05``, percentile-clipped bins, integer bar charts for
    PID / Charge / Status, log-y for PT / ET / P / E, ``Counts`` y-label,
    grid alpha=0.3) so trainee-vs-target and TorchDelphes-vs-C++ plots are
    directly comparable.

    Parameters
    ----------
    distributions : sequence of ``(values, label, color)``
        ``values`` is a 1-D numpy array; ``label`` is the legend entry
        (``len(values)`` is appended automatically); ``color`` is any
        matplotlib color spec. The first entry is treated as the reference.
    var : str
        Variable name; used to look up the axis label, the log-y rule and
        the discrete-bar handling.
    output_path : Path
        Where to write the figure (any format supported by matplotlib,
        e.g. ``.png`` or ``.pdf``).
    n_bins : int
        Number of histogram bins for continuous variables (ignored for
        discrete variables, which use one bin per unique integer value).
    bins : array-like, optional
        Explicit bin edges for continuous variables. If provided, overrides
        ``n_bins``/auto-binning. Ignored for discrete variables.
    title : str, optional
        Axes title for the upper panel (in addition to the per-variable
        LaTeX symbol). Defaults to just the symbol via :func:`title_label`.
    xlabel : str, optional
        X-axis label for the ratio panel. Defaults to :func:`axis_label(var)`.
    ylabel : str
        Y-axis label for the upper panel. Defaults to ``"Counts"``.
    ratio_ylabel : str, optional
        Y-label for the ratio panel. Defaults to ``"ratio / <ref label>"``.
    figsize, dpi : float / int
        Forwarded to :func:`matplotlib.pyplot.figure` / ``Figure.savefig``.
    legend_loc : str or (float, float)
        Forwarded to ``Axes.legend(loc=...)``.
    log_y : bool | None
        If ``True``, force log-y on the histogram panel. If ``False``, force
        linear-y. If ``None`` (default), defer to :func:`apply_axis_scale(var)`.
    """
    if len(distributions) < 2:
        raise ValueError(
            "plot_comparison_with_ratio needs at least 2 distributions "
            "(1 reference + 1 overlay)"
        )

    # Ensure 1-D numpy arrays once up front; drop NaNs so percentile-binning
    # doesn't return NaN edges.
    cleaned: list[tuple[np.ndarray, str, str]] = []
    for vals, lbl, clr in distributions:
        arr = np.asarray(vals).reshape(-1)
        arr = arr[np.isfinite(arr)] if arr.size else arr
        cleaned.append((arr, lbl, clr))
    distributions = cleaned

    ref_vals, ref_label, ref_color = distributions[0]
    overlays = distributions[1:]

    # If every distribution is empty there's nothing to draw -- skip rather
    # than emit a blank page.
    if all(len(d[0]) == 0 for d in distributions):
        return

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_hist = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_hist)

    if is_discrete(var):
        _draw_discrete(ax_hist, ax_ratio, distributions)
    else:
        if bins is None:
            bins_to_use: np.ndarray | int = _auto_bins(
                [d[0] for d in distributions], n_bins=n_bins
            )
        else:
            bins_to_use = np.asarray(bins)
        _draw_continuous(ax_hist, ax_ratio, distributions, bins=bins_to_use, var=var)

    # ---- Upper panel cosmetics
    ax_hist.set_ylabel(ylabel, fontsize=12)
    ax_hist.set_title(title if title is not None else title_label(var),
                      fontsize=14, fontweight="bold")
    ax_hist.legend(loc=legend_loc, fontsize=10)
    ax_hist.grid(True, alpha=0.3)
    ax_hist.tick_params(labelbottom=False)
    if not is_discrete(var):
        if log_y is None:
            apply_axis_scale(ax_hist, var)
        elif log_y:
            ax_hist.set_yscale("log")

    # ---- Lower (ratio) panel cosmetics
    if ratio_ylabel is None:
        ratio_ylabel = f"ratio / {ref_label}"
    ax_ratio.set_xlabel(xlabel if xlabel is not None else axis_label(var), fontsize=12)
    ax_ratio.set_ylabel(ratio_ylabel, fontsize=10)
    ax_ratio.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _draw_continuous(
    ax_hist: Axes,
    ax_ratio: Axes,
    distributions: Sequence[tuple[np.ndarray, str, str]],
    *,
    bins: np.ndarray | int,
    var: str,
) -> None:
    """Stepfilled reference + N-1 step overlays + ratio lines.

    Counts are computed via ``Axes.hist`` so the visible bars and the ratio
    use exactly the same histogramming.
    """
    del var  # unused (handled by caller)
    ref_vals, ref_label, ref_color = distributions[0]
    ref_hist = ax_hist.hist(
        ref_vals,
        bins=bins,
        histtype="stepfilled",
        color=ref_color,
        alpha=0.5,
        linewidth=2,
        label=f"{ref_label}: {len(ref_vals)} particles",
        density=False,
    )
    ref_counts = np.asarray(ref_hist[0])
    bin_edges = np.asarray(ref_hist[1])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Reference line at ratio=1 in the colour of the reference itself, so the
    # eye reads "this is the thing the other lines are compared to".
    ax_ratio.axhline(y=1.0, color=ref_color, linewidth=2)

    all_ratios: list[np.ndarray] = []
    for vals, label, color in distributions[1:]:
        h = ax_hist.hist(
            vals,
            bins=bin_edges,
            histtype="step",
            color=color,
            linewidth=2,
            label=f"{label}: {len(vals)} particles",
            density=False,
        )
        counts = np.asarray(h[0])
        ratio = np.divide(
            counts,
            ref_counts,
            out=np.ones_like(counts, dtype=float),
            where=ref_counts > 0,
        )
        ax_ratio.plot(bin_centers, ratio, color=color, markersize=4, linewidth=2)
        all_ratios.append(ratio)

    if all_ratios:
        pooled = np.concatenate(all_ratios)
        finite = pooled[np.isfinite(pooled)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            # Match validate_torch_delphes' ±10% padding around the observed range.
            ax_ratio.set_ylim(0.9 * lo, 1.1 * hi)


def _draw_discrete(
    ax_hist: Axes,
    ax_ratio: Axes,
    distributions: Sequence[tuple[np.ndarray, str, str]],
) -> None:
    """Grouped bars + ratio bars over the union of unique integer values."""
    pooled = np.concatenate([np.asarray(d[0]) for d in distributions if len(d[0]) > 0])
    if pooled.size == 0:
        return
    unique_vals = np.unique(pooled).astype(int)
    x = np.arange(len(unique_vals), dtype=float)

    # Grouped bars: width is divided across N curves so they sit side by side.
    n = len(distributions)
    bar_width = 0.8 / n

    all_counts: list[np.ndarray] = []
    for i, (vals, label, color) in enumerate(distributions):
        counts = np.array([np.sum(vals == v) for v in unique_vals], dtype=float)
        all_counts.append(counts)
        offset = (i - (n - 1) / 2) * bar_width
        ax_hist.bar(
            x + offset,
            counts,
            bar_width,
            label=f"{label}, {len(vals)} particles",
            color=color,
            alpha=0.7,
        )

    ax_hist.set_xticks(x)
    ax_hist.set_xticklabels([str(int(v)) for v in unique_vals], rotation=45, ha="right")
    ax_hist.tick_params(labelbottom=False)

    ref_counts = all_counts[0]
    ref_color = distributions[0][2]
    ax_ratio.axhline(y=1.0, color=ref_color, linewidth=2)

    all_ratios: list[np.ndarray] = []
    for i, (counts, (_, _, color)) in enumerate(zip(all_counts[1:], distributions[1:])):
        ratio = np.divide(
            counts, ref_counts,
            out=np.ones_like(counts, dtype=float),
            where=ref_counts > 0,
        )
        offset = ((i + 1) - (n - 1) / 2) * bar_width
        ax_ratio.bar(x + offset, ratio, bar_width, color=color, alpha=0.7)
        all_ratios.append(ratio)

    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels([str(int(v)) for v in unique_vals], rotation=45, ha="right")
    if all_ratios:
        pooled_r = np.concatenate(all_ratios)
        finite = pooled_r[np.isfinite(pooled_r)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            ax_ratio.set_ylim(0.9 * lo, 1.1 * hi)


# =============================================================================
# Multi-PNG stitching for ``all.png``
# =============================================================================


def stitch_pngs(
    image_paths: Sequence[Path],
    output_path: Path,
    title: str | None = None,
    title_height: int | None = None,
    title_fontsize: int = 22,
    title_line_spacing: float = 1.6,
    title_pad: int = 20,
    title_align: str = "center",
    background: tuple[int, int, int] = (255, 255, 255),
    ncols: int = 2,
) -> bool:
    """Combine multiple PNG files into a grid.

    Despite the legacy name, this function arranges inputs into a grid with
    ``ncols`` columns (default 2 → 2x2 for four inputs). Within each row, all
    images are vertically resized to a common height (preserving aspect
    ratio). Each row's width is independently determined; rows are padded to
    the widest row so the output is rectangular. An optional title bar is
    rendered on top.

    This is the same routine that backed ``validate_torch_delphes.py``'s
    ``all.png`` generation; consolidated here so both harnesses use the
    identical layout.

    Parameters
    ----------
    image_paths : sequence of Path
        Paths of PNGs to combine, in row-major order. Missing paths are
        silently skipped.
    output_path : Path
        Where to save the stitched PNG.
    title : str, optional
        Centered title rendered above the grid. Skipped if ``None``.
        May contain ``\\n`` for multi-line titles; the title bar will grow
        automatically unless ``title_height`` is explicitly provided.
    title_height : int, optional
        Pixel height of the title bar. If ``None`` (default), auto-sized
        from the number of lines in ``title``, ``title_fontsize``, and
        ``title_line_spacing``.
    title_fontsize : int
        Font size for the title.
    title_line_spacing : float
        Multiplier on ``title_fontsize`` (in points) used to estimate the
        rendered height of one line of title text. Only used when
        ``title_height`` is ``None``.
    title_pad : int
        Extra vertical padding (pixels) added above and below the title
        text when auto-sizing.
    title_align : str
        Horizontal alignment of each line of the title within the title
        bar. One of ``"left"``, ``"center"``, ``"right"``. Default
        ``"center"``.
    background : tuple of int
        RGB background colour.
    ncols : int
        Number of columns in the grid.

    Returns
    -------
    bool
        ``True`` if a stitched image was written, ``False`` if no input files
        existed.
    """
    existing = [Path(p) for p in image_paths if Path(p).exists()]
    if not existing:
        return False

    images = [Image.open(p).convert("RGB") for p in existing]

    # Chunk images into rows of `ncols`
    rows: list[list[Image.Image]] = [
        images[i : i + ncols] for i in range(0, len(images), ncols)
    ]

    # For each row, resize all images to the row's max height
    rendered_rows: list[Image.Image] = []
    for row in rows:
        row_max_h = max(im.height for im in row)
        resized = []
        for im in row:
            if im.height == row_max_h:
                resized.append(im)
            else:
                new_w = int(round(im.width * row_max_h / im.height))
                resized.append(im.resize((new_w, row_max_h), Image.LANCZOS))
        row_w = sum(im.width for im in resized)
        row_canvas = Image.new("RGB", (row_w, row_max_h), background)
        x = 0
        for im in resized:
            row_canvas.paste(im, (x, 0))
            x += im.width
        rendered_rows.append(row_canvas)

    total_w = max(r.width for r in rendered_rows)
    grid_h = sum(r.height for r in rendered_rows)

    # Auto-size the title bar to fit all lines if title_height wasn't given.
    # Matplotlib renders \n as a real line break, so we just need enough
    # vertical room. 1 pt ≈ 100/72 px at dpi=100, then scaled by line spacing.
    title_pad_bottom_px = 0
    if title:
        title = title.strip("\n")
        n_lines = title.count("\n") + 1
        line_px = int(round(title_fontsize * (100 / 72) * title_line_spacing))
        text_block_px = n_lines * line_px
        title_h = (text_block_px + title_pad + title_pad_bottom_px) if title_height is None else title_height
    else:
        title_h = 0

    canvas = Image.new("RGB", (total_w, grid_h + title_h), background)

    y = title_h
    for r in rendered_rows:
        x = (total_w - r.width) // 2
        canvas.paste(r, (x, y))
        y += r.height

    if title:
        # Render the title via matplotlib (so we don't need a system TTF font)
        fig = plt.figure(figsize=(total_w / 100, title_h / 100), dpi=100)
        fig.patch.set_facecolor(
            (background[0] / 255, background[1] / 255, background[2] / 255)
        )
        ax = fig.add_subplot(111)
        ax.set_position((0.0, 0.0, 1.0, 1.0))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        if title_align == "left":
            x_anchor, ha = 0.44, "left"
        elif title_align == "right":
            x_anchor, ha = 0.99, "right"
        else:
            x_anchor, ha = 0.5, "center"
        y_anchor = title_pad_bottom_px / title_h if title_h else 0.0
        ax.text(
            x_anchor,
            y_anchor,
            title,
            ha=ha,
            va="bottom",
            multialignment=title_align,
            fontsize=title_fontsize,
            fontweight="bold",
        )
        title_png = output_path.with_suffix(".title.png")
        fig.savefig(title_png, dpi=100, bbox_inches=None, pad_inches=0)
        plt.close(fig)

        title_img = Image.open(title_png).convert("RGB")
        if title_img.size != (total_w, title_h):
            title_img = title_img.resize((total_w, title_h), Image.LANCZOS)
        canvas.paste(title_img, (0, 0))
        title_png.unlink(missing_ok=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return True
