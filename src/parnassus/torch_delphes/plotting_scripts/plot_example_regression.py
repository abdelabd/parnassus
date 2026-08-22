r"""Two paper pages of one Optuna trial: the muon momentum scale convergence, and the loss.

The one-block version of ``plot_parameter_regression``: same inputs (one trial's
``round_<trial>/history.json`` + ``materialized_config.yaml``, and the partial generation YAML
over the card defaults as the truth) and the same page (``draw_block``: solid line = fitted
value, x=0 the initialisation and x=k+1 the snapshot after epoch k; dashed line of the same
colour = truth; dotted vertical line = early-stopping checkpoint), but only the
``MuonMomentumSmearing.resolution_module.scale_raw[0..2]`` block, i.e. the muon pT scale in
the three |eta| regions (barrel / mid / endcap) of ``LearnableMomentumResolution``. Page 2 is
the train / validation loss per epoch (same epoch axis, log y) with the same early-stopping
line. Paper pages: no titles, the block name is the y label of page 1 (``Y_LABEL``).

    python -m parnassus.torch_delphes.plotting_scripts.plot_example_regression \\
        --workspace /global/cfs/cdirs/m3246/diff_delphes/results/muongun_0 \\
        --truth-config src/parnassus/torch_delphes/param_configs/param_config_muons.yaml \\
        [--output doc/figure_pseudodata_muongun/plots/example_reg.pdf]

``--output`` (default ``<workspace>/plots/example_reg.pdf``): a ``.pdf`` gets both pages; any
other matplotlib extension writes one file per page, ``<stem>_params.<ext>`` and
``<stem>_loss.<ext>``.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from parnassus.torch_delphes.plotting_scripts.plot_parameter_regression import (
    BLOCKS,
    COLORS,
    DEFAULT_CONFIG,
    draw_block,
    finish_page,
    load_trial,
    load_values,
    new_page,
)

# The one block drawn: title and keys of plot_parameter_regression's page of the same name.
BLOCK = "MuonMomentumSmearing (scale_raw)"
KEYS = BLOCKS[BLOCK]  # MuonMomentumSmearing.resolution_module.scale_raw[0..2]
Y_LABEL = "MuonMomentumSmearing"  # page 1 y label (the pages have no title)
LOSS_LOG_Y = True  # the losses fall by more than a decade over a fit

# ---- layout knobs: paper-sized text (the reference's multi-page params_reg.pdf keeps its own) ----
FIG_SIZE = (8.6, 7.8)
LABEL_SIZE = 36  # axis labels
TICK_SIZE = 26
LEGEND_SIZE = 28  # 2 x 3 entries; 28 is the largest that still fits inside the axes
HEADROOM = 0.45  # legend band above the curves, as a fraction of their span (reference: 0.4)
HEADROOM_LOSS = 0.35  # same, loss page (2-row legend; log10 span)
LOSS_PAD = 1.5  # loss page y range = (min / LOSS_PAD, max * LOSS_PAD) before the legend band
MARGINS = {"left": 0.17, "right": 0.96, "bottom": 0.15, "top": 0.95}  # axes box (no title)
plt.rcParams.update({
    "figure.figsize": FIG_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    # Tight legend box (ATLAS style: handle 2.0, columns 2.0, border 1.0, pad 0.8) so the
    # two columns fit at LEGEND_SIZE.
    "legend.handlelength": 1.5,
    "legend.columnspacing": 1.0,
    "legend.borderpad": 0.4,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.3,
})


def load_losses(trial_dir):
    """Per-epoch train / validation loss of one trial (``round_<trial>/history.json``).

    Returns
    -------
    tuple[list[float], list[float]]
        ``(train, val)``; index k is the loss of epoch k, drawn at x = k + 1 (the parameter
        page's snapshot after epoch k).
    """
    run = json.load(open(trial_dir / "history.json"))
    history = list(run["history"].values())
    train = [epoch["train_loss"] for epoch in history]
    val = [epoch["val_loss"] for epoch in history]
    return train, val


def draw_loss(train, val, best_epoch):
    """Page 2: train / validation loss per epoch and the early-stopping (best) epoch.

    Returns
    -------
    matplotlib.figure.Figure
        The page; the caller saves and closes it.
    """
    fig, ax = new_page(None, "Loss", margins=MARGINS)
    if LOSS_LOG_Y:
        ax.set_yscale("log")
    epochs = range(1, len(train) + 1)
    ax.plot(epochs, train, color=COLORS[0], label="Train")
    ax.plot(epochs, val, color=COLORS[1], label="Validation")
    # Explicit range: the ATLAS style autoscales a log axis to whole decades, which wastes
    # most of the page on a ~1.5-decade drop.
    ax.set_ylim(min(train + val) / LOSS_PAD, max(train + val) * LOSS_PAD)
    finish_page(ax, epochs[-1], best_epoch, headroom=HEADROOM_LOSS)
    return fig


def main():
    """Plot the muon-scale trajectories and the loss of one trial to ``--output``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workspace", required=True, type=Path, help="dir containing round_<trial>/")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--truth-config", required=True, type=Path, help="generation (truth) YAML")
    ap.add_argument("--output", type=Path, help="default <workspace>/plots/example_reg.pdf")
    args = ap.parse_args()

    trial_dir = args.workspace / f"round_{args.trial}"
    curves, best_epoch, trainable = load_trial(trial_dir)
    if not trainable.intersection(KEYS):
        raise SystemExit(f"{trial_dir}: none of {KEYS} is trainable; nothing to plot")
    truth = {**load_values(DEFAULT_CONFIG), **load_values(args.truth_config)}
    train, val = load_losses(trial_dir)

    pages = {
        "params": draw_block(
            None,
            KEYS,
            curves,
            truth,
            best_epoch,
            headroom=HEADROOM,
            margins=MARGINS,
            ylabel=Y_LABEL,
        ),
        "loss": draw_loss(train, val, best_epoch),
    }
    output = args.output or args.workspace / "plots" / "example_reg.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".pdf":
        with PdfPages(output) as pdf:
            for fig in pages.values():
                pdf.savefig(fig)
        print(f"Wrote {output}")
    else:  # one image per page
        for name, fig in pages.items():
            page = output.with_name(f"{output.stem}_{name}{output.suffix}")
            fig.savefig(page)
            print(f"Wrote {page}")
    plt.close("all")


if __name__ == "__main__":
    main()
