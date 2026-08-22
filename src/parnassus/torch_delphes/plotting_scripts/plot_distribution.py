r"""Per-species feature distributions of one Optuna trial: target vs trainee.

The trainee card is run on the truth particles of ``--sample`` at the trial's initial
parameters (``materialized_config.yaml``) and at its best-epoch parameters
(``history.json`` ``best_result``); both are overlaid on the sample's pflow target for
log pT / log E / eta of charged hadron, neutral hadron, electron, muon and photon,
plus the leading-2 pair-mass response ln(m_reco / m_truth) of electrons, muons and
charged hadrons -- the loss's pair-mass observable (a peak at ln(scale) whose width is
the track resolution on the resonance-gun samples).
One PDF page per (species, observable): counts on log-y with a ratio-to-target panel.
Paper layout: no page title, and no x label unless ``X_LABEL`` (both are added in LaTeX --
the page order is printed); text sizes / legend band / margins are the knobs below
``plt.style.use``.

    python -m parnassus.torch_delphes.plotting_scripts.plot_distribution \\
        --workspace doc/figure_pseudodata_all \\
        --sample /global/cfs/cdirs/m3246/diff_delphes/pseudo_data_100k_param_config_all_dijet.root
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import yaml
from matplotlib.backends.backend_pdf import PdfPages

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.PhotonClusterMerger import PhotonClusterMerger
from parnassus.torch_delphes.tune_cms_fullsim.data import load_cms_flow_root
from parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results import (
    _build_val_dataloader,
    _set_trainee_from_snapshot,
    _trainee_observables,
)

plt.style.use(hep.style.ATLAS)  # after the plot_fit_results import, so this style wins

# ---- text / layout knobs (same recipe as plot_fullsim_comparison.py) ----------------------------
LABEL_SIZE = 30  # pt: y axis labels "Objects" / "Ratio" (ATLAS default 20)
TICK_SIZE = 20  # pt: tick labels
LEGEND_SIZE = 30  # pt: legend (2 short entries per row -> fits the axes width up to ~36)
FIG_SIZE = (8.8, 7.8)  # inches; the margins leave ~0.2 in beyond the outermost tick label
MARGINS = dict(left=0.18, right=0.96, bottom=0.06, top=0.95)  # axes box = 6.9 x 6.9 in (no x label)
# left: 0.2 in more than plot_fullsim_comparison -- log-y pages spanning < 1 decade get labelled
# minor ticks ("3 x 10^3"), wide enough to push the y label off an 8.6-in page.
HEIGHT_RATIOS = (8, 3)  # top panel : ratio panel (the ratio panel must fit the rotated "Ratio" label)
HEADROOM = 1.45  # log-y span above the floor = HEADROOM x the data's span; the band holds the legend
LEGEND_NCOL = 2  # entries per legend row
LABELS = {"tuned": "Fitted", "initial": "Initial", "target": "Target"}  # legend text per sample key
LEGEND_ORDER = ["Fitted", "Initial", "Target"]  # reading order, row by row: "Fitted Initial" / "Target"
X_LABEL = False  # True: draw the x label (centred, LABEL_SIZE) on a page grown by X_LABEL_BAND; False: LaTeX
X_LABEL_BAND = 0.8  # inches added below the axes for the x label (tick labels + labelpad + 30 pt text with sub/superscripts)
plt.rcParams.update({
    "font.size": LEGEND_SIZE, "axes.labelsize": LABEL_SIZE, "legend.fontsize": LEGEND_SIZE,
    "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE, "figure.figsize": FIG_SIZE,
    "legend.handlelength": 1.5, "legend.columnspacing": 1.0, "legend.handletextpad": 0.5, "legend.borderpad": 0.4,
    "axes.labelpad": 8,
})

SPECIES = {"Charged hadron": 211, "Neutral hadron": 111, "Electron": 11, "Muon": 13, "Photon": 22}
# Leading-2 pair-mass response pages (the loss's pair-mass observable,
# loss.compute_pair_masses): one page per class, truth-mass groups and |eta|-region pair
# categories pooled.
PAIR_SPECIES = {"Electron": 11, "Muon": 13, "Charged hadron": 211}
OBSERVABLES = {  # x labels (drawn only with X_LABEL)
    "log_pt": r"$\log(p_\mathrm{T}\,/\,\mathrm{GeV})$",
    "log_E": r"$\log(E\,/\,\mathrm{GeV})$",
    "eta": r"$\eta$",
}
PAIR_XLABEL = r"$\ln(m_{\mathrm{pair}}^{\mathrm{reco}}\,/\,m_{\mathrm{pair}}^{\mathrm{truth}})$"
N_BINS = 25
QUANTILES = (0.005, 0.995)  # bin range = pooled quantiles, so outliers cannot stretch the axis
COLORS = {"target": "#5790fc", "initial": "black", "tuned": "#e42536"}
# Draw order/widths: target at the back, thick red tuned, thinner black-dashed initial on
# top -- so both stay visible even where they coincide exactly.
STYLE = {
    "target": {"fill": True, "alpha": 0.4, "zorder": 1},
    "tuned": {"linewidth": 3.5, "zorder": 2},
    "initial": {"linewidth": 2, "linestyle": "--", "zorder": 3},
}
BATCH_SIZE = 2000
SEED = 0  # re-seeded before each pass: initial and tuned see the same smearing noise


def run_card(card, params, loader, meta):
    """Load ``params`` into ``card`` and run it over ``loader`` -> (pred, target) observables."""
    _set_trainee_from_snapshot(card, params)
    torch.manual_seed(SEED)
    return _trainee_observables(
        card, loader, reco_pt_cut=meta["reco_pt_cut"], abs_eta_cut=meta["eta_cut"]
    )


def values(obs, pid, key):
    """Flat numpy array of observable ``key`` for the objects with |pid| == ``pid``."""
    return obs[key][obs["pid"].abs() == pid].numpy()


def draw_page(pdf, samples, ylabel="Objects", xlabel=None):
    """One page: counts (log-y) of target / initial / tuned plus the ratio-to-target panel.

    ``xlabel`` is drawn only with ``X_LABEL``: the page then grows downwards by
    ``X_LABEL_BAND`` so the axes box keeps the same size and position.
    """
    pooled = np.concatenate(list(samples.values()))
    edges = np.linspace(*np.quantile(pooled, QUANTILES), N_BINS + 1)
    counts = {name: np.histogram(v, edges)[0] for name, v in samples.items()}

    width, height = FIG_SIZE
    band = X_LABEL_BAND if X_LABEL and xlabel else 0.0
    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, figsize=(width, height + band),
        gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.06},
    )
    fig.subplots_adjust(**{  # fixed axes box; the x-label band is taken below it
        **MARGINS,
        "bottom": (MARGINS["bottom"] * height + band) / (height + band),
        "top": 1 - (1 - MARGINS["top"]) * height / (height + band),
    })
    for name in ("target", "initial", "tuned"):
        ax.stairs(counts[name], edges, color=COLORS[name], label=LABELS[name], **STYLE[name])
    ax.set_yscale("log")
    nonzero = np.concatenate([c[c > 0] for c in counts.values()])
    lo, hi = np.log10(nonzero.min() / 2), np.log10(nonzero.max())
    ax.set_ylim(10**lo, 10 ** (lo + (hi - lo) * HEADROOM))  # the band above the data holds the legend
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: LEGEND_ORDER.index(labels[i]))  # row-major reading order
    n_rows = -(-len(order) // LEGEND_NCOL)  # matplotlib fills columns first -> transpose
    order = [order[r * LEGEND_NCOL + c] for c in range(LEGEND_NCOL) for r in range(n_rows) if r * LEGEND_NCOL + c < len(order)]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", ncol=LEGEND_NCOL)

    rax.axhline(1, color=COLORS["target"], linewidth=1, zorder=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = 1 / np.sqrt(counts["target"])  # target statistical uncertainty
        band = {**STYLE["target"], "alpha": 0.2}
        rax.stairs(1 + rel, edges, baseline=1 - rel, color=COLORS["target"], **band)
        for name in ("tuned", "initial"):
            rax.stairs(counts[name] / counts["target"], edges, color=COLORS[name], **STYLE[name])
    rax.set_ylim(0.7, 1.3)
    rax.set_yticks([0.8, 1.0, 1.2])
    rax.set_xlim(edges[0], edges[-1])
    rax.set_ylabel("Ratio")  # "Ratio to target" does not fit the ratio panel at LABEL_SIZE
    if X_LABEL and xlabel:
        rax.set_xlabel(xlabel, loc="center")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    """Plot the trial's distributions to ``<workspace>/plots/distributions.pdf``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workspace", required=True, type=Path, help="dir containing round_<trial>/")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--sample", required=True, type=Path, help="ROOT file (truth + pflow target)")
    ap.add_argument("--n-events", type=int, default=20000, help="first N events of --sample")
    args = ap.parse_args()

    trial_dir = args.workspace / f"round_{args.trial}"
    run = json.load(open(trial_dir / "history.json"))
    meta = run["metadata"]
    init_cfg = yaml.safe_load(open(trial_dir / "materialized_config.yaml"))
    init = {k: v["value"] for k, v in init_cfg.items()}
    best = run["best_result"]["parameters"]

    # Same acceptance cuts / photon merger the fit ran with (all off in --mode delphes).
    arrays = load_cms_flow_root(args.sample, n_events=args.n_events)
    loader = _build_val_dataloader(
        arrays, BATCH_SIZE, torch.device("cpu"), truth_pt_cut=meta["truth_pt_cut"],
        reco_pt_cut=meta["reco_pt_cut"], abs_eta_cut=meta["eta_cut"],
    )
    radius = meta["photon_merge_radius"]
    card = CMSEnergyFlowDefault(
        debug=False, learnable=True, photon_merger=PhotonClusterMerger(radius) if radius else None
    )
    initial, target = run_card(card, init, loader, meta)
    tuned, _ = run_card(card, best, loader, meta)

    output = args.workspace / "plots" / "distributions.pdf"
    output.parent.mkdir(exist_ok=True)
    page = 0
    with PdfPages(output) as pdf:
        for title, pid in SPECIES.items():
            for key, xlabel in OBSERVABLES.items():
                samples = {"target": target, "initial": initial, "tuned": tuned}
                arrays = {n: values(o, pid, key) for n, o in samples.items()}
                if not sum(len(v) for v in arrays.values()):
                    continue  # species absent from this sample (e.g. muon gun)
                draw_page(pdf, arrays, xlabel=xlabel)
                page += 1
                print(f"page {page}: {title} {key}")
        for title, pid in PAIR_SPECIES.items():
            key = f"pair_r:{pid}"
            samples = {"target": target, "initial": initial, "tuned": tuned}
            if not all(key in o for o in samples.values()):
                continue  # class has no pairs in this sample
            arrays = {n: o[key].numpy() for n, o in samples.items()}  # ln(m_reco/m_truth)
            draw_page(pdf, arrays, xlabel=PAIR_XLABEL)
            page += 1
            print(f"page {page}: {title} pair-mass response (leading 2)")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
