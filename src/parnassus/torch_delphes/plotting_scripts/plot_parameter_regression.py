r"""Parameter convergence of one Optuna trial: fitted value vs epoch, with the truth.

One PDF page per parameter block. Solid line = fitted (physical) value, x=0 is the
initialisation from ``materialized_config.yaml`` and x=k+1 the snapshot after epoch k;
dashed line of the same colour = truth (partial generation YAML over the card defaults).

    python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \\
        --workspace doc/figure_pseudodata_all \\
        --truth-config src/parnassus/torch_delphes/param_configs/param_config_all.yaml
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.style.use(hep.style.ATLAS)
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["axes.titlesize"] = "x-large"  # block title (draw_block)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "param_configs" / "cms_target_default.yaml"

# Parameter i of every block uses COLORS[i] (Petroff 6-colour palette).
COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
HEADROOM = 0.4  # extra y range above the curves, as a fraction of their span, for the legend
MARGINS = {"left": 0.16, "right": 0.95, "bottom": 0.14, "top": 0.88}  # fixed axes box


def indexed(name, n):
    """``name[0]`` ... ``name[n-1]``.

    Returns
    -------
    list[str]
        The ``n`` indexed keys.
    """
    return [f"{name}[{i}]" for i in range(n)]


# One page per block, in this order. Only blocks with at least one trainable key are drawn.
BLOCKS = {
    "ChargedHadronTrackingEfficiency": indexed("ChargedHadronTrackingEfficiency.eff_logits", 4),
    "ElectronTrackingEfficiency": indexed("ElectronTrackingEfficiency.eff_logits", 6),
    "MuonTrackingEfficiency (efficiency)": indexed("MuonTrackingEfficiency.eff_logits", 6),
    "MuonTrackingEfficiency (rate)": indexed("MuonTrackingEfficiency.rate_raw", 2),
    **{
        f"{module}MomentumSmearing ({tensor})": indexed(
            f"{module}MomentumSmearing.resolution_module.{tensor}", 3
        )
        for module in ("ChargedHadron", "Electron", "Muon")
        for tensor in ("a_raw", "b_raw", "scale_raw")
    },
    "ECal scale": indexed("ECal.scale_module.scale_raw", 3),
    "ECal resolution (common)": [
        "ECal.resolution_func.common_c_E",
        "ECal.resolution_func.common_c_S",
        "ECal.resolution_func.common_c_N",
    ],
    "ECal resolution (barrel/endcap)": [
        "ECal.resolution_func.barrel_a",
        "ECal.resolution_func.barrel_b",
        "ECal.resolution_func.endcap_a",
        "ECal.resolution_func.endcap_b",
    ],
    "ECal resolution (forward)": [
        "ECal.resolution_func.forward_c_E",
        "ECal.resolution_func.forward_c_S",
    ],
    "HCal scale": indexed("HCal.scale_module.scale_raw", 2),
    "HCal resolution": [
        "HCal.resolution_func.central_c_E",
        "HCal.resolution_func.central_c_S",
        "HCal.resolution_func.forward_c_E",
        "HCal.resolution_func.forward_c_S",
    ],
    "HadronFractions": [
        "HadronFractions.chad_logit",
        "HadronFractions.k0s_logit",
        "HadronFractions.lambda_logit",
        "HadronFractions.photon_logit",
        "HadronFractions.k0l_logit",
    ],
}


def load_values(path):
    """Flat ``{key: {value: ...}}`` YAML -> ``{key: value}``.

    Returns
    -------
    dict[str, float]
        Parameter values keyed by flat parameter name.
    """
    return {k: v["value"] for k, v in yaml.safe_load(open(path)).items()}


def load_trial(trial_dir):
    """Parameter trajectories of one trial (``round_<trial>/``).

    Returns
    -------
    tuple[dict[str, list[float]], int, set[str]]
        ``({key: [init, after epoch 0, after epoch 1, ...]}, best_epoch, trainable keys)``;
        ``best_epoch`` is the x position of the early-stopping checkpoint (min val loss).
    """
    run = json.load(open(trial_dir / "history.json"))
    init = load_values(trial_dir / "materialized_config.yaml")
    snapshots = [epoch["parameters"] for epoch in run["history"].values()]
    curves = {k: [init[k]] + [snap[k] for snap in snapshots] for k in init}
    return curves, run["best_result"]["step"] + 1, set(run["metadata"]["trainable_params"])


def new_page(title, ylabel, *, margins=MARGINS):
    """A page in the house style: the fixed axes box, the epoch axis, ``ylabel`` and ``title``.

    ``title`` is drawn top-left; ``None`` (or empty) for no title.

    ``margins`` is the ``subplots_adjust`` axes box; text sizes follow the rcParams
    (``axes.labelsize``, ``axes.titlesize``, ``*tick.labelsize``, ``legend.fontsize``).

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The page and its axes: draw the curves on the axes, then :func:`finish_page`.
    """
    fig, ax = plt.subplots()
    fig.subplots_adjust(**margins)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    return fig, ax


def finish_page(ax, last_epoch, best_epoch, *, headroom=HEADROOM, extra_handles=()):
    """Best-epoch marker, epoch axis ``0..last_epoch``, legend band and legend.

    ``headroom`` is the legend band above the curves as a fraction of their span (of their
    log10 span on a log axis); the legend lists the labelled curves, then ``extra_handles``.
    """
    ax.axvline(best_epoch, color="black", linestyle=":", label="Best epoch")
    ax.set_xlim(0, last_epoch)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        lo, hi = math.log10(lo), math.log10(hi)
        ax.set_ylim(10**lo, 10 ** (hi + headroom * (hi - lo)))
    else:
        ax.set_ylim(lo, hi + headroom * (hi - lo))
    handles, _ = ax.get_legend_handles_labels()
    handles.extend(extra_handles)
    ax.legend(
        handles=handles,
        loc="upper right",
        ncol=2,
        frameon=True,
        framealpha=1.0,
        edgecolor="none",
    )


def draw_block(
    title, keys, curves, truth, best_epoch, *, headroom=HEADROOM, margins=MARGINS, ylabel=None
):
    """One block on a new page: fitted curves, truth lines, best-epoch marker and legend.

    ``ylabel`` defaults to "Parameter value".

    Returns
    -------
    matplotlib.figure.Figure
        The page; the caller saves and closes it.
    """
    fig, ax = new_page(title, ylabel or "Parameter value", margins=margins)
    epochs = range(len(curves[keys[0]]))
    for key, color in zip(keys, COLORS, strict=False):
        ax.plot(epochs, curves[key], color=color, label=key.split(".")[-1])
        ax.axhline(truth[key], color=color, linestyle="--")
    truth_handle = Line2D([], [], color="gray", linestyle="--", label="Truth")
    finish_page(ax, epochs[-1], best_epoch, headroom=headroom, extra_handles=[truth_handle])
    return fig


def main():
    """Plot one trial's parameter trajectories to ``<workspace>/plots/params_reg.pdf``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workspace", required=True, type=Path, help="dir containing round_<trial>/")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--truth-config", required=True, type=Path, help="generation (truth) YAML")
    args = ap.parse_args()

    curves, best_epoch, trainable = load_trial(args.workspace / f"round_{args.trial}")
    truth = {**load_values(DEFAULT_CONFIG), **load_values(args.truth_config)}

    output = args.workspace / "plots" / "params_reg.pdf"
    output.parent.mkdir(exist_ok=True)
    with PdfPages(output) as pdf:
        for title, keys in BLOCKS.items():
            if not trainable.intersection(keys):
                continue
            fig = draw_block(title, keys, curves, truth, best_epoch)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
