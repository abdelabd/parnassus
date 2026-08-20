r"""Per-species feature distributions of one Optuna trial: target vs trainee.

The trainee card is run on the truth particles of ``--sample`` at the trial's initial
parameters (``materialized_config.yaml``) and at its best-epoch parameters
(``history.json`` ``best_result``); both are overlaid on the sample's pflow target for
log pT / log E / eta of charged hadron, neutral hadron, electron, muon and photon,
plus the leading-2 pair-mass response ln(m_reco / m_truth) of electrons, muons and
charged hadrons -- the loss's pair-mass observable (a peak at ln(scale) whose width is
the track resolution on the resonance-gun samples).
One PDF page per (species, observable): counts on log-y with a ratio-to-target panel.

With ``--mode fullsim`` (the default) the PDF ends with leading-jet pages: each
sample's reco objects (after the trial's acceptance cuts) are clustered per event
with anti-kt R=0.5 and the leading jet's log pT / eta / log m / constituent
multiplicity are overlaid -- the same jet convention as the C++ evaluation
pipeline (cppDelphes leading_jet_filter: jet pt > 8 GeV, >= 2 constituents,
|eta| < 2.5). ``--mode delphes`` skips the jet pages.

    python -m parnassus.torch_delphes.plotting_scripts.plot_distribution \\
        --workspace doc/figure_cmsfull_pt5 \\
        --sample /work/hdd/bhvk/rli35/diffDelphes/samples/train_1000.root
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
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    apply_reco_acceptance_cut,
    load_cms_flow_root,
    load_pflow_targets_from_tensor,
    restore_event_format,
)
from parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results import (
    _build_val_dataloader,
    _set_trainee_from_snapshot,
    _trainee_observables,
)

plt.style.use(hep.style.ATLAS)  # after the plot_fit_results import, so this style wins

SPECIES = {"Charged hadron": 211, "Neutral hadron": 111, "Electron": 11, "Muon": 13, "Photon": 22}
# Leading-2 pair-mass response pages (the loss's pair-mass observable,
# loss.compute_pair_masses): one page per class, truth-mass groups and |eta|-region pair
# categories pooled.
PAIR_SPECIES = {"Electron": 11, "Muon": 13, "Charged hadron": 211}
OBSERVABLES = {
    "log_pt": r"$\log(p_\mathrm{T}\,/\,\mathrm{GeV})$",
    "log_E": r"$\log(E\,/\,\mathrm{GeV})$",
    "eta": r"$\eta$",
}
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

# Leading-jet pages (--mode fullsim): same convention as the C++ evaluation pipeline
# (cppDelphes/postprocessing data_utils.leading_jet_filter / cluster_jets defaults).
JET_R = 0.5
JET_PT_MIN = 8.0  # GeV, fastjet inclusive_jets threshold
JET_N_CONST_MIN = 2
JET_ABS_ETA_MAX = 2.5
JET_OBSERVABLES = {
    "jet_log_pt": r"$\log(p_\mathrm{T}^{\,\mathrm{jet}}\,/\,\mathrm{GeV})$",
    "jet_eta": r"$\eta^{\,\mathrm{jet}}$",
    "jet_log_m": r"$\log(m^{\,\mathrm{jet}}\,/\,\mathrm{GeV})$",
    "jet_nconst": "Jet constituent multiplicity",
}


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


def _accumulate_leading_jets(obs, feats):
    """Cluster one batch's padded per-event objects; append the leading jet's features.

    ``obs`` is a padded 2-D observable dict (``pt``/``eta``/``phi``/``log_E`` of shape
    ``(n_events, max_n)``, invalid slots have ``pt == 0``). Four-vectors use the same
    convention as the loss pipeline: ``E = exp(log_E)`` (class-mass hypothesis baked in
    by :func:`load_pflow_targets_from_tensor`). Events whose jets all fail the
    ``leading_jet_filter``-style selection are skipped (counted by the caller via
    array lengths).
    """
    try:
        import fastjet as fj
    except ImportError as err:  # pragma: no cover - environment guard
        raise RuntimeError(
            "--mode fullsim jet pages need the fastjet python package "
            "(available in the parnassus training env)."
        ) from err

    jet_def = fj.JetDefinition(fj.antikt_algorithm, JET_R)
    pt_all = obs["pt"]
    e_all = torch.exp(obs["log_E"])
    for i in range(pt_all.shape[0]):
        valid = pt_all[i] != 0
        if not bool(valid.any()):
            continue
        pt = pt_all[i, valid].numpy()
        eta = obs["eta"][i, valid].numpy()
        phi = obs["phi"][i, valid].numpy()
        e = e_all[i, valid].numpy()
        px, py, pz = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta)
        pjs = [fj.PseudoJet(px[k], py[k], pz[k], e[k]) for k in range(len(pt))]
        cs = fj.ClusterSequence(pjs, jet_def)  # keep alive while reading constituents
        jets = [
            j
            for j in fj.sorted_by_pt(cs.inclusive_jets(JET_PT_MIN))
            if len(j.constituents()) >= JET_N_CONST_MIN and abs(j.eta()) < JET_ABS_ETA_MAX
        ]
        if not jets:
            continue
        lead = jets[0]
        feats["jet_log_pt"].append(np.log(lead.pt()))
        feats["jet_eta"].append(lead.eta())
        feats["jet_log_m"].append(np.log(max(lead.m(), 1e-3)))
        feats["jet_nconst"].append(float(len(lead.constituents())))


def leading_jet_features(loader, card=None, params=None, meta=None):
    """Leading-jet feature arrays for one sample, preserving per-event structure.

    With ``card``/``params``/``meta`` given, re-runs the trainee batch-by-batch
    (mirroring ``_trainee_observables``: forward -> ``restore_event_format`` ->
    ``load_pflow_targets_from_tensor`` -> acceptance cut) and clusters its output;
    re-seeding with ``SEED`` reproduces the exact smearing noise of the object
    pages, so the jet pages describe the same reco sample. Without them, clusters
    the target objects straight from the loader batches (cuts already applied at
    loader build).
    """
    feats = {key: [] for key in JET_OBSERVABLES}
    if card is not None:
        _set_trainee_from_snapshot(card, params)
        torch.manual_seed(SEED)
    with torch.no_grad():
        for batch in loader:
            if card is None:
                _accumulate_leading_jets(batch, feats)
                continue
            truth_particles = batch["truth_particles"]
            mask = torch.any(truth_particles != 0, dim=-1)
            out = card(truth_particles[mask])
            pred = load_pflow_targets_from_tensor(restore_event_format(out["EFlowObject"], mask))
            pred = apply_reco_acceptance_cut(pred, meta["reco_pt_cut"], meta["eta_cut"])
            _accumulate_leading_jets(pred, feats)
    return {key: np.asarray(v) for key, v in feats.items()}


def draw_page(pdf, title, xlabel, samples, ylabel="Objects"):
    """One page: counts (log-y) of target / initial / tuned plus the ratio-to-target panel."""
    pooled = np.concatenate(list(samples.values()))
    edges = np.linspace(*np.quantile(pooled, QUANTILES), N_BINS + 1)
    counts = {name: np.histogram(v, edges)[0] for name, v in samples.items()}

    fig, (ax, rax) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06}
    )
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.12, top=0.90)  # fixed axes box
    for name in ("target", "initial", "tuned"):
        ax.stairs(counts[name], edges, color=COLORS[name], label=name, **STYLE[name])
    ax.set_yscale("log")
    nonzero = np.concatenate([c[c > 0] for c in counts.values()])
    ax.set_ylim(nonzero.min() / 2, nonzero.max() * 30)  # headroom for the legend
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize="x-large")
    ax.legend(loc="upper right")

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
    rax.set_ylabel("Ratio to target")
    rax.set_xlabel(xlabel)
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
    ap.add_argument("--n-events", type=int, default=50000, help="first N events of --sample")
    ap.add_argument(
        "--mode",
        choices=("fullsim", "delphes"),
        default="fullsim",
        help="fullsim (default): append leading-jet pages (anti-kt R=0.5 clustering of "
        "each sample's reco objects); delphes: object/pair pages only",
    )
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
    with PdfPages(output) as pdf:
        for title, pid in SPECIES.items():
            for key, xlabel in OBSERVABLES.items():
                samples = {"target": target, "initial": initial, "tuned": tuned}
                arrays = {n: values(o, pid, key) for n, o in samples.items()}
                if not sum(len(v) for v in arrays.values()):
                    continue  # species absent from this sample (e.g. muon gun)
                draw_page(pdf, title, xlabel, arrays)
        for title, pid in PAIR_SPECIES.items():
            key = f"pair_r:{pid}"
            samples = {"target": target, "initial": initial, "tuned": tuned}
            if not all(key in o for o in samples.values()):
                continue  # class has no pairs in this sample
            arrays = {n: o[key].numpy() for n, o in samples.items()}  # ln(m_reco/m_truth)
            draw_page(
                pdf,
                f"{title} pair-mass response (leading 2)",
                r"$\ln(m_{\mathrm{pair}}^{\mathrm{reco}}\,/\,m_{\mathrm{pair}}^{\mathrm{truth}})$",
                arrays,
            )
        if args.mode == "fullsim":
            # Leading-jet pages: one extra pass per card (same SEED -> same smearing
            # realization as the object pages above), target straight from the loader.
            jet_samples = {
                "target": leading_jet_features(loader),
                "initial": leading_jet_features(loader, card, init, meta),
                "tuned": leading_jet_features(loader, card, best, meta),
            }
            n_jets = {name: len(f["jet_log_pt"]) for name, f in jet_samples.items()}
            print(f"Leading jets passing selection: {n_jets}")
            for key, xlabel in JET_OBSERVABLES.items():
                arrays = {name: f[key] for name, f in jet_samples.items()}
                if not all(len(v) for v in arrays.values()):
                    continue
                draw_page(
                    pdf,
                    rf"Leading jet (anti-$k_t$, $R = {JET_R}$)",
                    xlabel,
                    arrays,
                    ylabel="Jets",
                )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
