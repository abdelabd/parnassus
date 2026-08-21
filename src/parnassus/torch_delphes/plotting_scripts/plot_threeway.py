r"""Three-way overlay of one fullsim fit: CMS full sim vs diff-Delphes vs C++ Delphes.

All legs use entries ``[--entry-start, --entry-start + --n-events)`` of ``--sample``
(disjoint from the fit's training range ``[0, n_events)``) and the same acceptance on every
object: ``pt >= reco_pt_cut`` (the fit's, or ``--reco-pt-cut``) and ``|eta| <= eta_cut``.
Pages: all / charged / neutral log pT and eta, then the leading jet's log pT, eta, log m,
constituent multiplicity, its pT / mass response (x_reco - x_truth) / x_truth and eta_reco - eta_truth
relative to the event's truth jet (anti-kt R=0.5 on massless constituents; jet pt > 8 GeV, >= 2 constituents,
|eta| < 2.5 -- the cppDelphes leading_jet_filter convention; the truth jet is clustered the
same way from ALL truth particles of the event -- the sample's truth_* branches for the full-sim
and diff-Delphes legs, the Delphes file's own truth_tree for the C++ leg). Raw counts
on linear y with a ratio-to-full-sim panel (band = full-sim sqrt(N)); fixed binning per page
(``BINS``); a metrics file lists the quantile-W1 distance to full sim and the yield ratio per page.

Legs and their structural differences (keep in mind when ranking):

- full sim: ``event_tree`` pflow objects of ``--sample`` (real pileup).
- diff-Delphes: the card at the trial's best-epoch parameters on the truth of the same
  entries (no pileup, truth pt >= 0.25 GeV). No chad truncation here, while the fit's
  target was truncated to the truth chad ceiling -> a charged yield ~5-8 % below full sim
  is the expected ceiling, not an efficiency failure.
- C++ Delphes: ``fastsim_tree`` of the sibling ``delphes_pu6p35_jet<bin>.root`` (PU mu=6.35
  on the unfiltered truth; ``fs_pt`` in MeV; ``fs_class`` 1 = track, 0 = neutral). Entries
  are the same events as ``test_470.root``; for jet1000 they are an independent sample of
  the same process (equal event count, so raw counts still compare).

Charged = has a track (|pid| in {211, 11, 13}; e/mu are 0.3 % of PF objects), which is
exactly the C++ binarized class, so the three pages mean the same thing on every leg.

    python -m parnassus.torch_delphes.plotting_scripts.plot_threeway \\
        --workspace doc/figure_fullsim/470_frac_pt5 \\
        --sample /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/test_470.root
"""

import argparse
import json
from pathlib import Path

import fastjet as fj
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
import uproot
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.PhotonClusterMerger import PhotonClusterMerger
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    apply_reco_acceptance_cut,
    batch_event_ids,
    load_cms_flow_root,
    load_pflow_targets_from_tensor,
    restore_event_format,
)
from parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results import (
    _build_val_dataloader,
    _set_trainee_from_snapshot,
)

plt.style.use(hep.style.ATLAS)  # after the plot_fit_results import, so this style wins

BATCH_SIZE = 2000
SEED = 0
CHARGED_PIDS = (211, 11, 13)  # "has a track" == C++ Delphes fs_class 1
JET_DEF = fj.JetDefinition(fj.antikt_algorithm, 0.5)
JET_PT_MIN, JET_N_CONST_MIN, JET_ABS_ETA_MAX = 8.0, 2, 2.5
PAGES = {"All objects": None, "Charged": True, "Neutral": False}
OBS = {"log_pt": r"$\log(p_\mathrm{T}\,/\,\mathrm{GeV})$", "eta": r"$\eta$"}
JET_OBS = {
    "jet_log_pt": r"$\log(p_\mathrm{T}^{\,\mathrm{jet}}\,/\,\mathrm{GeV})$",
    "jet_eta": r"$\eta^{\,\mathrm{jet}}$",
    "jet_log_m": r"$\log(m^{\,\mathrm{jet}}\,/\,\mathrm{GeV})$",
    "jet_nconst": "Jet constituent multiplicity",
    "jet_pt_resp": r"$(p_\mathrm{T}^{\,\mathrm{jet}} - p_\mathrm{T}^{\,\mathrm{truth}})\,/\,p_\mathrm{T}^{\,\mathrm{truth}}$",
    "jet_m_resp": r"$(m^{\,\mathrm{jet}} - m^{\,\mathrm{truth}})\,/\,m^{\,\mathrm{truth}}$",
    "jet_eta_resp": r"$\eta^{\,\mathrm{jet}} - \eta^{\,\mathrm{truth}}$",
}
# (n_bins, low, high) per page; None -> pooled 0.5 % / 99.5 % quantile. The log-pT pages start
# at the floor (set in main) and take their upper edge from the data (it depends on the sample).
BINS = {
    "log_pt": (30, None, None),
    "eta": (20, -2.0, 2.0),
    "jet_log_pt": (30, None, None),
    "jet_eta": (25, -2.5, 2.5),
    "jet_log_m": (30, 0.5, 5.5),
    "jet_nconst": (20, -0.5, 39.5),
    "jet_pt_resp": (30, -0.4, 0.2),
    "jet_m_resp": (30, -1.0, 1.0),
    "jet_eta_resp": (30, -0.06, 0.06),
}
STYLE = {
    "full sim": dict(color="#5790fc", fill=True, alpha=0.4, zorder=1),
    "diff-Delphes": dict(color="#e42536", linewidth=3, zorder=2),
    "C++ Delphes": dict(color="black", linewidth=2, linestyle="--", zorder=3),
}


def rows(obs):
    """Padded observable dict -> per event (pt, eta, phi, charged) numpy arrays."""
    for i in range(obs["pt"].shape[0]):
        m = obs["pt"][i] != 0
        pt, eta, phi, pid = (obs[k][i, m].numpy() for k in ("pt", "eta", "phi", "pid"))
        yield pt, eta, phi, np.isin(np.abs(pid), CHARGED_PIDS)


def card_rows(card, params, loader, cut, eta_cut):
    _set_trainee_from_snapshot(card, params)
    torch.manual_seed(SEED)
    with torch.no_grad():
        for batch in loader:
            truth = batch["truth_particles"]
            mask = torch.any(truth != 0, dim=-1)
            out = restore_event_format(card(truth[mask])["EFlowObject"], mask, event_ids=batch_event_ids(truth, mask))
            yield from rows(apply_reco_acceptance_cut(load_pflow_targets_from_tensor(out), cut, eta_cut))


def cpp_rows(path, start, n, cut, eta_cut):
    a = uproot.open(path)["fastsim_tree"].arrays(
        ["fs_pt", "fs_eta", "fs_phi", "fs_class"], entry_start=start, entry_stop=start + n, library="np"
    )
    for pt, eta, phi, cls in zip(a["fs_pt"], a["fs_eta"], a["fs_phi"], a["fs_class"]):
        pt = pt / 1000.0
        m = (pt >= cut) & (np.abs(eta) <= eta_cut)
        yield pt[m], eta[m], phi[m], cls[m] == 1


def leading_jet(pt, eta, phi):
    """(pt, eta, m, nconst) of the leading selected anti-kt jet of massless (pt, eta, phi), or None."""
    px, py, pz, e = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    cs = fj.ClusterSequence([fj.PseudoJet(*v) for v in zip(px, py, pz, e)], JET_DEF)
    jets = [
        j for j in fj.sorted_by_pt(cs.inclusive_jets(JET_PT_MIN))
        if len(j.constituents()) >= JET_N_CONST_MIN and abs(j.eta()) < JET_ABS_ETA_MAX
    ]
    return (jets[0].pt(), jets[0].eta(), max(jets[0].m(), 1e-3), len(jets[0].constituents())) if jets else None


def truth_jets(pt, eta, phi):
    """Per-event leading truth jet (all truth particles, same clustering/selection as the reco legs)."""
    return [leading_jet(p, e, f) for p, e, f in zip(pt, eta, phi)]


def collect(events, truths):
    """Flat per-object log_pt / eta / charged plus leading-jet observables and responses to the truth jet."""
    acc = {k: [] for k in ("log_pt", "eta", "charged", *JET_OBS)}
    for (pt, eta, phi, charged), tj in zip(events, truths):
        acc["log_pt"].append(np.log(pt))
        acc["eta"].append(eta)
        acc["charged"].append(charged)
        jet = leading_jet(pt, eta, phi)
        if jet is None:
            continue
        jpt, jeta, jm, jn = jet
        acc["jet_log_pt"].append(np.log(jpt))
        acc["jet_eta"].append(jeta)
        acc["jet_log_m"].append(np.log(jm))
        acc["jet_nconst"].append(jn)
        if tj is None:
            continue
        acc["jet_pt_resp"].append(jpt / tj[0] - 1)
        acc["jet_m_resp"].append(jm / tj[2] - 1)
        acc["jet_eta_resp"].append(jeta - tj[1])
    return {k: np.concatenate(v) if k in ("log_pt", "eta", "charged") else np.asarray(v, float)
            for k, v in acc.items()}


def qwass(a, b, n_q=200):
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def draw_page(pdf, title, xlabel, samples, ylabel, key):
    """Linear-y raw counts + ratio to full sim (band = full-sim sqrt(N)); returns (W1, N / N_fullsim) per leg."""
    n_bins, lo, hi = BINS[key]
    qlo, qhi = np.quantile(np.concatenate(list(samples.values())), (0.005, 0.995))
    edges = np.linspace(qlo if lo is None else lo, qhi if hi is None else hi, n_bins + 1)
    counts = {n: np.histogram(v, edges)[0].astype(float) for n, v in samples.items()}

    fig, (ax, rax) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    fig.subplots_adjust(left=0.16, right=0.95, bottom=0.17, top=0.90)
    for n, c in counts.items():
        ax.stairs(c, edges, label=n, **STYLE[n])
    ax.set_ylim(0, max(c.max() for c in counts.values()) * 1.4)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize="large")
    ax.legend(loc="upper right", fontsize="small")

    fs = counts["full sim"]
    ok = fs > 0  # empty full-sim bins: no band (an inf corner breaks the fill polygon), no ratio
    rel = np.divide(1, np.sqrt(fs), out=np.zeros_like(fs), where=ok)  # full-sim sqrt(N) on the ratio
    rax.stairs(1 + rel, edges, baseline=1 - rel, color=STYLE["full sim"]["color"], fill=True, alpha=0.3)
    for n in ("diff-Delphes", "C++ Delphes"):
        rax.stairs(np.divide(counts[n], fs, out=np.full_like(fs, np.nan), where=ok), edges, **STYLE[n])
    rax.axhline(1, color=STYLE["full sim"]["color"], linewidth=1)
    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    rax.set_xlim(edges[0], edges[-1])
    rax.xaxis.set_major_formatter(FormatStrFormatter("%g"))  # no "x10^-2" offset text under the label
    rax.set_ylabel("Ratio")
    rax.set_xlabel(xlabel)
    pdf.savefig(fig)
    plt.close(fig)
    fs = samples["full sim"]
    return {n: (qwass(v, fs), len(v) / len(fs)) for n, v in samples.items() if n != "full sim"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, type=Path, help="fit output-base (uses round_0/)")
    ap.add_argument("--sample", required=True, type=Path, help="full-sim file (test_470.root / train_1000.root)")
    ap.add_argument("--delphes", type=Path, help="C++ Delphes file; default: sibling delphes_pu6p35_jet<bin>.root")
    ap.add_argument("--n-events", type=int, default=20000)
    ap.add_argument("--entry-start", type=int, default=100000, help="first entry (fit trained on [0, 100k))")
    ap.add_argument("--reco-pt-cut", type=float, help="constituent pt floor for all legs; default: the fit's")
    args = ap.parse_args()

    run = json.load(open(args.workspace / "round_0" / "history.json"))
    meta, best = run["metadata"], run["best_result"]["parameters"]
    cut, eta_cut = args.reco_pt_cut or meta["reco_pt_cut"], meta["eta_cut"]
    BINS["log_pt"] = (30, np.log(cut), None)
    delphes = args.delphes or args.sample.with_name(f"delphes_pu6p35_jet{args.sample.stem.split('_')[1]}.root")

    arrays = load_cms_flow_root(args.sample, args.n_events, entry_start=args.entry_start)
    loader = _build_val_dataloader(
        arrays, BATCH_SIZE, torch.device("cpu"),
        truth_pt_cut=meta["truth_pt_cut"], reco_pt_cut=cut, abs_eta_cut=eta_cut,
    )
    radius = meta["photon_merge_radius"]
    card = CMSEnergyFlowDefault(
        debug=False, learnable=True, photon_merger=PhotonClusterMerger(radius) if radius else None
    )
    tj_sample = truth_jets(arrays["truth_pt"], arrays["truth_eta"], arrays["truth_phi"])
    tr = uproot.open(delphes)["truth_tree"].arrays(
        ["tr_pt", "tr_eta", "tr_phi"], entry_start=args.entry_start, entry_stop=args.entry_start + args.n_events, library="np"
    )
    tj_cpp = truth_jets([p / 1000.0 for p in tr["tr_pt"]], tr["tr_eta"], tr["tr_phi"])
    legs = {
        "full sim": collect((r for b in loader for r in rows(b)), tj_sample),
        "diff-Delphes": collect(card_rows(card, best, loader, cut, eta_cut), tj_sample),
        "C++ Delphes": collect(cpp_rows(delphes, args.entry_start, args.n_events, cut, eta_cut), tj_cpp),
    }

    out = args.workspace / "plots"
    out.mkdir(exist_ok=True)
    tag = f"threeway_pt{cut:g}"
    lines = [
        f"{args.workspace.name}: {args.n_events} events/leg from entry {args.entry_start}, "
        f"pt >= {cut:g} GeV, |eta| <= {eta_cut}, best {run['best_result']['epoch']}. "
        "C++ Delphes: PU mu=6.35, unfiltered truth; diff-Delphes: no PU, truth pt>=0.25, untruncated.",
        f"{'page':26s} {'W1 diff':>9s} {'W1 C++':>9s} {'N diff/fs':>10s} {'N C++/fs':>10s}",
    ]

    def row(name, m):
        (wd, nd), (wc, nc) = m["diff-Delphes"], m["C++ Delphes"]
        lines.append(f"{name:26s} {wd:9.4f} {wc:9.4f} {nd:10.3f} {nc:10.3f}")

    with PdfPages(out / f"{tag}.pdf") as pdf:
        for page, charged in PAGES.items():
            for key, xlabel in OBS.items():
                samples = {n: f[key] if charged is None else f[key][f["charged"] == charged]
                           for n, f in legs.items()}
                title = rf"{page}   ($p_\mathrm{{T}} \geq {cut:g}$ GeV)"
                row(f"{page} {key}", draw_page(pdf, title, xlabel, samples, "Objects", key))
        for key, xlabel in JET_OBS.items():
            samples = {n: f[key] for n, f in legs.items()}
            title = rf"Leading jet, anti-$k_t$ $R=0.5$   (constituent $p_\mathrm{{T}} \geq {cut:g}$ GeV)"
            row(key, draw_page(pdf, title, xlabel, samples, "Jets", key))

    text = "\n".join(lines)
    (out / f"{tag}_metrics.txt").write_text(text + "\n")
    print(text)
    print(f"Wrote {out / tag}.pdf")


if __name__ == "__main__":
    main()
