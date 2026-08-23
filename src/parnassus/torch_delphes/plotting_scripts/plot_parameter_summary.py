r"""Parameter-regression summary over a seed scan: one page per gun.

Reads ``<results>/<gun>_<seed>/round_0/history.json`` (the ``submit_param_regression.sh``
layout), takes each seed's early-stopping checkpoint ``best_result.parameters`` and the
gun's truth (partial generation YAML over the card defaults). Each page is one gun: one row
per trainable (and ``PINNED``) parameter, x = ``(theta_fit - theta_truth) / theta_truth``
averaged over the seeds, with the seed-to-seed std as a horizontal error bar, a dashed line
at 0 and the parameter name under its marker (colour = family, marker = subsystem). A mean
beyond ``--xlim`` is drawn as an arrow at the edge. Parameters the gun cannot constrain
(``UNCONSTRAINED`` quantities and the ``PINNED`` keys, which are fixed at the CMS default on
both sides and therefore sit at 0) are drawn in grey at the bottom of the page, below a
horizontal separator line.

    python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_summary \\
        --results /global/cfs/cdirs/m3246/diff_delphes/results [--output <pdf>] [--xlim 0.5]

A non-``.pdf`` ``--output`` (e.g. ``.png``) writes one file per page, ``<stem>_<gun>.<ext>``.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from parnassus.torch_delphes.plotting_scripts.plot_parameter_regression import (
    COLORS,
    DEFAULT_CONFIG,
    load_values,
)

# ---- layout knobs ---------------------------------------------------------------------------
FIG_SIZE = (4.5, 9)  # narrow and tall: four pages side by side in one LaTeX row
LABEL_SIZE = 36  # x axis label
TICK_SIZE = 22
TEXT_SIZE = 14  # parameter names (under the markers)
N_SEEDS = 5
TEXT_GAP = 0.22  # rows: marker at y, top of its name at y - TEXT_GAP
TEXT_PAD = 0.02  # minimum gap between a name and the axes edge, as a fraction of xlim
ARROW = 0.2  # length of the off-scale arrow, as a fraction of xlim
SEP_GAP = 0.5  # extra rows of space above the grey block; the separator line sits in it
plt.rcParams.update({
    "figure.figsize": FIG_SIZE, "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
})

# Page order: gun (the results-dir prefix) -> truth-config key.
GUNS = {"muongun": "muons", "ksgun": "chads", "electrongun": "electrons", "dijet": "dijets"}
# Family = subsystem (key prefix; marker) x quantity (last token; anything else, i.e. the
# calorimeter ``resolution_func`` coefficients, is a resolution). One colour per family from
# the Petroff palette; the tracker colours are reused on the dijet page, but no ECal group
# shares a colour with an HCal group. Families that are never ``constrained`` (resolution
# ``b``, rate) are drawn in ``GRAY`` and need no entry.
QUANTITIES = {
    "eff_logits": "efficiency", "scale_raw": "scale", "a_raw": "resolution",
    "b_raw": r"resolution $b$", "rate_raw": "rate",
}
SUBSYSTEMS = {"ECal": "s", "HCal": "^"}  # marker; anything else is the tracker ("o")
FAMILY_COLORS = {
    "Tracker efficiency": COLORS[0], "Tracker scale": COLORS[1],
    "Tracker resolution": COLORS[2],
    "ECal resolution": COLORS[2], "ECal scale": COLORS[1],
    "HCal resolution": COLORS[0], "HCal scale": COLORS[3],
}
GRAY = COLORS[4]
# Parameters the current samples cannot recover: drawn in grey at the bottom of the page.
# Matched on the tensor name (last key token, index stripped): the tracker resolution ``b``
# terms (a lever only from tracks with b*pT >~ a) and the calorimeter constant terms ``c_E``
# (the E^2 term is not identifiable at the dijet pT reach), plus the ``PINNED`` keys.
UNCONSTRAINED = ("b_raw", "c_E", "rate_raw")
# Per gun: parameters that are not trainable in the fit and fixed at the CMS default in the
# pseudo-data too (param_config_<key>.yaml), still shown (in grey, below the fitted ones) to
# list the full parameter set. Muon gun: the (1000, inf) pt efficiency bins, which the
# resonance gun does not reach, and the exponential roll-off rate of that efficiency
# (``eff = coeff * exp(0.5 - rate * pt)``), which has no gradient path.
PINNED = {
    "muongun": [
        "MuonTrackingEfficiency.eff_logits[2]", "MuonTrackingEfficiency.eff_logits[5]",
        "MuonTrackingEfficiency.rate_raw[0]", "MuonTrackingEfficiency.rate_raw[1]",
    ],
}


def constrained(key, pinned=()):
    """Whether the gun's sample can recover the flat parameter ``key``.

    Returns
    -------
    bool
        ``False`` for the ``UNCONSTRAINED`` quantities and the ``pinned`` keys.
    """
    name = key.split(".")[-1].split("[")[0]
    return key not in pinned and not name.endswith(UNCONSTRAINED)


def family(key, pinned=()):
    """Family label (row grouping), colour and marker of a flat parameter key.

    Returns
    -------
    tuple[str, str, str]
        ``(label, colour, marker)``; the colour is ``GRAY`` unless the key is ``constrained``.
    """
    quantity = QUANTITIES.get(key.split(".")[-1].split("[")[0], "resolution")
    subsystem = key.split(".")[0]
    label = f"{subsystem if subsystem in SUBSYSTEMS else 'Tracker'} {quantity}"
    color = FAMILY_COLORS[label] if constrained(key, pinned) else GRAY
    return label, color, SUBSYSTEMS.get(subsystem, "o")


def short_name(key):
    """Per-bar label: the tensor name, prefixed by the subsystem for calorimeter keys.

    Returns
    -------
    str
        e.g. ``eff_logits[1]``, ``a_raw[0]``, ``ECal common_c_E``, ``HCal scale_raw[0]``.
    """
    subsystem = key.split(".")[0]
    name = key.split(".")[-1]
    return f"{subsystem} {name}" if subsystem in SUBSYSTEMS else name


def load_runs(results, gun):
    """Load the per-seed ``history.json`` of ``<results>/<gun>_<seed>/round_0``.

    Returns
    -------
    list[dict]
        One history dict per seed found, in seed order (a warning is printed when
        the count differs from ``N_SEEDS``).
    """
    paths = sorted(results.glob(f"{gun}_*/round_0/history.json"))
    if len(paths) != N_SEEDS:
        print(f"[warn] {gun}: {len(paths)} seeds found, expected {N_SEEDS}")
    return [json.loads(p.read_text()) for p in paths]


def deviations(runs, truth, pinned=()):
    """Relative deviation of the best-epoch fit from the truth, per trainable/pinned parameter.

    Returns
    -------
    dict[str, np.ndarray]
        ``{key: fit/truth - 1 over the seeds}`` for the trainable parameters followed by the
        ``pinned`` ones (which must not be trainable).
    """
    fits = [run["best_result"]["parameters"] for run in runs]
    trainable = runs[0]["metadata"]["trainable_params"]
    assert not set(pinned) & set(trainable), f"pinned keys are trainable: {pinned}"
    keys = [*trainable, *pinned]
    return {k: np.array([fit[k] / truth[k] - 1 for fit in fits]) for k in keys}


def draw_page(ax, dev, xlim, pinned=()):
    """One gun: a row per parameter, names under the markers.

    Rows are grouped by family (first on top); the parameters the gun cannot ``constrain``
    come last, in grey below a separator line, with the ``pinned`` ones at the very bottom.
    """
    keys = sorted(dev, key=lambda k: (not constrained(k, pinned), k in pinned, family(k)[0]))
    ys = np.arange(len(keys), dtype=float)[::-1]
    first_grey = next((i for i, k in enumerate(keys) if not constrained(k, pinned)), len(keys))
    ys[first_grey:] -= SEP_GAP  # open the gap for the separator above the grey block
    pad, arrow = TEXT_PAD * xlim, ARROW * xlim
    renderer = ax.figure.canvas.get_renderer()
    texts = []
    for y, k in zip(ys, keys, strict=True):
        _, color, marker = family(k, pinned)
        mean = dev[k].mean()
        std = dev[k].std(ddof=1) if len(dev[k]) > 1 else 0.0
        if abs(mean) > xlim:  # off scale: outward arrow at the edge
            x = np.sign(mean) * xlim
            ax.annotate(
                "", xy=(x, y), xytext=(x - np.sign(mean) * arrow, y),
                arrowprops={"arrowstyle": "->", "color": color, "lw": 3, "mutation_scale": 30},
            )
        else:
            x = mean
            ax.errorbar(mean, y, xerr=std, fmt=marker, color=color, ms=8, capsize=3,
                        elinewidth=2)
        # Name under the marker, centred on it but kept inside the axes (width measured in
        # data units).
        text = ax.text(
            x, y - TEXT_GAP, short_name(k), ha="center", va="top", color=color,
            fontsize=TEXT_SIZE, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
        )
        half = text.get_window_extent(renderer).width / ax.bbox.width * xlim
        text.set_x(np.clip(x, -xlim + half + pad, xlim - half - pad))
        texts.append(text)

    if 0 < first_grey < len(keys):  # separator: midway between the last fitted name and the
        # first grey marker (the name's bottom edge measured in data units)
        bbox = texts[first_grey - 1].get_window_extent(renderer)
        bottom = ax.transData.inverted().transform((bbox.x0, bbox.y0))[1]
        ax.axhline((bottom + ys[first_grey]) / 2, color=GRAY, linewidth=1.5, zorder=0)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, zorder=0)
    ax.set_xlim(-xlim, xlim)
    ax.set_xticks(np.linspace(-0.8 * xlim, 0.8 * xlim, 5))  # few ticks, none on the edges
    ax.set_ylim(ys[-1] - 1.0, ys[0] + 0.4)  # room for the name under the bottom marker
    ax.set_yticks([])
    ax.tick_params(axis="y", which="minor", left=False, right=False)
    ax.set_xlabel(r"$(\theta_\mathrm{fit}-\theta_\mathrm{truth})\,/\,\theta_\mathrm{truth}$")


def main():
    """Draw one page per gun to ``--output`` (default ``<results>/plots/params_summary.pdf``)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--results", required=True, type=Path, help="dir containing <gun>_<seed>/round_0/"
    )
    ap.add_argument("--output", type=Path, help="default <results>/plots/params_summary.pdf")
    ap.add_argument("--xlim", type=float, default=0.5, help="x axis = +-XLIM (default 0.5)")
    args = ap.parse_args()

    figs = []
    for gun, key in GUNS.items():
        runs = load_runs(args.results, gun)
        if not runs:
            continue
        truth_yaml = DEFAULT_CONFIG.parent / f"param_config_{key}.yaml"  # partial, over the card
        truth = {**load_values(DEFAULT_CONFIG), **load_values(truth_yaml)}
        pinned = PINNED.get(gun, [])
        dev = deviations(runs, truth, pinned)
        print(f"{gun}: {len(runs)} seeds x {len(dev)} parameters ({len(pinned)} pinned)")
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.04, right=0.96, bottom=0.11, top=0.99)
        draw_page(ax, dev, args.xlim, pinned)
        figs.append((gun, fig))

    output = args.output or args.results / "plots" / "params_summary.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".pdf":
        with PdfPages(output) as pdf:
            for _, fig in figs:
                pdf.savefig(fig)
        print(f"Wrote {output}")
    else:  # one image per page
        for gun, fig in figs:
            page = output.with_name(f"{output.stem}_{gun}{output.suffix}")
            fig.savefig(page)
            print(f"Wrote {page}")


if __name__ == "__main__":
    main()
