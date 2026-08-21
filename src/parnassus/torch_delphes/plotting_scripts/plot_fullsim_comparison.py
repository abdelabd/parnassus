r"""Overlay of one fullsim fit: CMS full sim vs diff-Delphes vs C++ Delphes vs Parnassus.

Legend labels: "CMS" = full sim, "Parnassus-P" = the diff-Delphes card, "Delphes" = C++ Delphes,
"Parnassus-F" = the generative Parnassus fast sim (the prose below uses the long names).

All legs use entries ``[--entry-start, --entry-start + --n-events)`` of ``--sample``
(disjoint from the fit's training range ``[0, n_events)``): the SAME entry range and the same
number of jets on every leg (a file too short for the range is an error, never a silent
truncation), and the same acceptance on every object: ``pt >= reco_pt_cut`` (the fit's, or
``--reco-pt-cut``) and ``|eta| <= eta_cut``.
Pages: all / charged / neutral log pT and eta, then the leading jet's log pT, eta, log m,
constituent multiplicity, its pT / mass response (x_reco - x_truth) / x_truth and eta_reco - eta_truth
relative to the entry's truth jet (anti-kt R=0.5 on massless constituents; jet pt > 8 GeV, >= 2 constituents,
|eta| < 2.5 -- the cppDelphes leading_jet_filter convention; the truth jet is clustered the
same way from ALL truth particles of the entry -- the sample's truth_* branches for the full-sim
and diff-Delphes legs, the Delphes / Parnassus file's own truth_tree for those legs). Raw counts
on linear y with a ratio-to-full-sim panel (band = full-sim sqrt(N)); fixed binning per page
(``BINS``); a metrics file lists the quantile-W1 distance to full sim and the yield ratio per page.

Every entry of every file is ONE anti-kt R=0.5 jet of the CMS open-simulation sample (its own
constituents only; a dijet event contributes two consecutive entries), so the object pages are
in-jet constituent spectra and the "leading jet" of an entry is that jet.

Legs and their structural differences (keep in mind when ranking):

- full sim: ``event_tree`` pflow constituents of ``--sample`` (real pileup, pt >= 1 GeV).
- diff-Delphes: the card at the trial's best-epoch parameters on the truth of the same
  entries (no pileup, truth pt >= 0.25 GeV). No chad truncation here, while the fit's
  target was truncated to the truth chad ceiling -> a charged yield ~5-8 % below full sim
  is the expected ceiling, not an efficiency failure.
- C++ Delphes: ``fastsim_tree`` of the sibling ``delphes_pu6p35_jet<bin>.root`` (PU mu=6.35
  on the unfiltered truth; ``fs_pt`` in MeV; ``fs_class`` 1 = track, 0 = neutral). Entries
  are the same events as ``test_470.root`` / ``test_800.root``; for jet1000 they are an
  independent sample of the same process (equal event count, so raw counts still compare).
- Parnassus: ``fastsim_tree`` of the generative fast sim (flow matching, PNDM sampler, sampled
  multiplicity) in ``PARNASSUS_FILES[<bin>]``, auto-picked from the sample's pT-hat bin (800 and
  1000 exist; any other bin skips the leg). Fixed 201-slot arrays: ``fs_ind`` 1 marks a generated
  particle and the other slots hold NON-zero padding that must be masked; ``fs_pt`` in MeV. The
  model generates only (pt, eta, phi) -- ``fs_class`` is a copy of ``fs_ind`` -- so Parnassus is
  drawn on the "All objects" and jet pages only. Its 200k jets are the first 200k entries of the
  Delphes file of the same bin (same events as the C++ leg; same events as full sim for
  test_800, an independent sample of the same process for train_1000 -- the metrics header says
  which). ~0.3 % of its jets carry a phi-wrapped stray constituent far from the axis and ~20 %
  put some soft particles beyond R=0.5; the reclustering sheds both into separate jets (median
  leading-jet pT loss 0.2 %).

Charged = has a track (|pid| in {211, 11, 13}; e/mu are 0.3 % of PF objects), which is
exactly the C++ binarized class, so the three pages mean the same thing on every leg.

    python -m parnassus.torch_delphes.plotting_scripts.plot_fullsim_comparison \\
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
PAGES = {"All objects": None} #, "Charged": True, "Neutral": False}
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
# Ranges end where the data ends and leave the upper-right legend over a tail (eta, jet_log_m,
# jet_pt_resp), so the 1.4x headroom below never has to grow.
BINS = {
    "log_pt": (30, None, None),
    "eta": (27, -2.7, 2.7),
    "jet_log_pt": (30, None, None),
    "jet_eta": (25, -2.5, 2.5),
    "jet_log_m": (30, 1.5, 6.2),
    "jet_nconst": (20, -0.5, 39.5),
    "jet_pt_resp": (35, -0.4, 0.3),
    "jet_m_resp": (30, -1.0, 1.0),
    "jet_eta_resp": (30, -0.06, 0.06),
}
LEGEND_FONTSIZE = 15.5  # points; between "medium" (14) and "large" (16.8) of the ATLAS style
# Fixed jet log-pT axis per pT-hat bin (the "<bin>" of a "<split>_<bin>.root" sample stem);
# bins not listed keep the pooled-quantile range.
JET_LOG_PT_RANGE = {"1000": (6.5, 7.4)}
# Parnassus output per pT-hat bin.
PARNASSUS_DIR = Path("/global/cfs/cdirs/m3246/diff_delphes/parnassua_data")
PARNASSUS_FILES = {
    "800": PARNASSUS_DIR / "Jet800_fm_pos_cms_pow_v4_glob_npf_J800_1000_49_25_pndm_0.0.root",
    "1000": PARNASSUS_DIR / "Jet1000_fm_pos_cms_pow_v4_glob_npf_J800_1000_49_25_pndm_0.0.root",
}
# Legend labels of the legs (also the keys of STYLE / legs / the metrics columns).
CMS, PARNASSUS_P, DELPHES, PARNASSUS_F = "CMS", "Parnassus-P", "Delphes", "Parnassus-F"
LEGEND_ORDER = [PARNASSUS_P, PARNASSUS_F, DELPHES, CMS]  # legend rows, top to bottom
STYLE = {
    CMS: dict(color="#5790fc", fill=True, alpha=0.4, zorder=1),
    PARNASSUS_P: dict(color="#e42536", linewidth=3, zorder=2),
    DELPHES: dict(color="black", linewidth=2, linestyle="--", zorder=3),
    PARNASSUS_F: dict(color="#f89c20", linewidth=2, zorder=4),
}
SHORT = {PARNASSUS_P: "ParnP", DELPHES: "Delph", PARNASSUS_F: "ParnF"}  # metrics columns


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


def parnassus_rows(path, start, n, cut, eta_cut):
    """Parnassus ``fastsim_tree``: 201 fixed slots, ``fs_ind == 1`` = generated particle (the
    other slots are NON-zero padding), ``fs_pt`` in MeV, no class information (charged = None)."""
    a = uproot.open(path)["fastsim_tree"].arrays(
        ["fs_pt", "fs_eta", "fs_phi", "fs_ind"], entry_start=start, entry_stop=start + n, library="np"
    )
    for pt, eta, phi, ind in zip(a["fs_pt"], a["fs_eta"], a["fs_phi"], a["fs_ind"]):
        real = ind == 1
        pt, eta, phi = pt[real] / 1000.0, eta[real], phi[real]
        m = (pt >= cut) & (np.abs(eta) <= eta_cut)
        yield pt[m], eta[m], phi[m], None


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


def parnassus_truth_jets(path, start, n):
    """Leading truth jet per entry of a Parnassus-format ``truth_tree`` (``tr_ind`` mask, MeV)."""
    tr = uproot.open(path)["truth_tree"].arrays(
        ["tr_pt", "tr_eta", "tr_phi", "tr_ind"], entry_start=start, entry_stop=start + n, library="np"
    )
    real = tr["tr_ind"] == 1
    return truth_jets(
        [p[k] / 1000.0 for p, k in zip(tr["tr_pt"], real)],
        [e[k] for e, k in zip(tr["tr_eta"], real)],
        [f[k] for f, k in zip(tr["tr_phi"], real)],
    )


def check_entries(path, tree, start, n):
    """Every leg must see the same ``n`` jets: refuse a file that cannot cover the entry range."""
    total = uproot.open(path)[tree].num_entries
    if start + n > total:
        raise ValueError(
            f"{path}: entries [{start}, {start + n}) requested but {tree} has only {total}; "
            "lower --n-events / --entry-start so every leg sees the same jets"
        )


def same_events(path_a, path_b, start, n):
    """True when both files' ``jet_tree`` hold the same jets (pt, eta) on the entry range."""
    a, b = (
        uproot.open(p)["jet_tree"].arrays(["jet_pt", "jet_eta"], entry_start=start, entry_stop=start + n, library="np")
        for p in (path_a, path_b)
    )
    return bool(np.allclose(a["jet_pt"], b["jet_pt"], rtol=1e-6) and np.allclose(a["jet_eta"], b["jet_eta"], atol=1e-6))


def collect(events, truths):
    """Flat per-object log_pt / eta / charged plus leading-jet observables and responses to the truth jet.

    ``charged`` is None for a leg without class information (Parnassus); ``n_events`` counts the
    entries seen so the legs can be checked against each other.
    """
    acc = {k: [] for k in ("log_pt", "eta", "charged", *JET_OBS)}
    n_events, has_class = 0, True
    for (pt, eta, phi, charged), tj in zip(events, truths):
        n_events += 1
        acc["log_pt"].append(np.log(pt))
        acc["eta"].append(eta)
        if charged is None:
            has_class = False
        else:
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
    out = {k: np.concatenate(acc[k]) for k in ("log_pt", "eta")}
    out["charged"] = np.concatenate(acc["charged"]) if has_class else None
    out.update({k: np.asarray(acc[k], float) for k in JET_OBS})
    out["n_events"] = n_events
    return out


def qwass(a, b, n_q=200):
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def draw_page(pdf, title, xlabel, samples, ylabel, key, note=None):
    """Linear-y raw counts + ratio to CMS full sim (band = its sqrt(N)); returns (W1, N / N_CMS) per leg."""
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
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: LEGEND_ORDER.index(labels[i]))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper right", fontsize=LEGEND_FONTSIZE)
    if note:
        ax.text(0.97, 0.58, note, transform=ax.transAxes, ha="right", va="top", fontsize="x-small", color="gray")

    fs = counts[CMS]
    ok = fs > 0  # empty full-sim bins: no band (an inf corner breaks the fill polygon), no ratio
    rel = np.divide(1, np.sqrt(fs), out=np.zeros_like(fs), where=ok)  # full-sim sqrt(N) on the ratio
    rax.stairs(1 + rel, edges, baseline=1 - rel, color=STYLE[CMS]["color"], fill=True, alpha=0.3)
    for n in counts:
        if n != CMS:
            rax.stairs(np.divide(counts[n], fs, out=np.full_like(fs, np.nan), where=ok), edges, **STYLE[n])
    rax.axhline(1, color=STYLE[CMS]["color"], linewidth=1)
    rax.set_ylim(0.5, 1.5)
    rax.set_yticks([0.6, 0.8, 1.0, 1.2, 1.4])
    rax.set_xlim(edges[0], edges[-1])
    rax.xaxis.set_major_formatter(FormatStrFormatter("%g"))  # no "x10^-2" offset text under the label
    rax.set_ylabel("Ratio")
    rax.set_xlabel(xlabel)
    pdf.savefig(fig)
    plt.close(fig)
    fs = samples[CMS]
    return {n: (qwass(v, fs), len(v) / len(fs)) for n, v in samples.items() if n != CMS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, type=Path, help="fit output-base (uses round_0/)")
    ap.add_argument("--sample", required=True, type=Path, help="full-sim file (test_470.root / train_1000.root)")
    ap.add_argument("--delphes", type=Path, help="C++ Delphes file; default: sibling delphes_pu6p35_jet<bin>.root")
    ap.add_argument(
        "--parnassus", type=Path,
        help="Parnassus file; default: PARNASSUS_FILES[<bin>] of the sample's pT-hat bin (leg skipped when the bin has none)",
    )
    ap.add_argument("--no-parnassus", action="store_true", help="draw without the Parnassus leg")
    ap.add_argument("--n-events", type=int, default=20000)
    ap.add_argument("--entry-start", type=int, default=100000, help="first entry (fit trained on [0, 100k))")
    ap.add_argument("--reco-pt-cut", type=float, help="constituent pt floor for all legs; default: the fit's")
    args = ap.parse_args()

    run = json.load(open(args.workspace / "round_0" / "history.json"))
    meta, best = run["metadata"], run["best_result"]["parameters"]
    cut, eta_cut = args.reco_pt_cut or meta["reco_pt_cut"], meta["eta_cut"]
    BINS["log_pt"] = (30, np.log(cut), None)
    start, n = args.entry_start, args.n_events
    pt_bin = args.sample.stem.split("_")[1]
    if pt_bin in JET_LOG_PT_RANGE:
        BINS["jet_log_pt"] = (30, *JET_LOG_PT_RANGE[pt_bin])
    delphes = args.delphes or args.sample.with_name(f"delphes_pu6p35_jet{pt_bin}.root")
    parnassus = None if args.no_parnassus else args.parnassus or PARNASSUS_FILES.get(pt_bin)
    if parnassus is None and not args.no_parnassus:
        print(f"No Parnassus sample for pT-hat bin {pt_bin} (have {sorted(PARNASSUS_FILES)}); leg skipped.")
    check_entries(args.sample, "event_tree", start, n)
    check_entries(delphes, "fastsim_tree", start, n)
    if parnassus:
        check_entries(parnassus, "fastsim_tree", start, n)

    arrays = load_cms_flow_root(args.sample, n, entry_start=start)
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
        ["tr_pt", "tr_eta", "tr_phi"], entry_start=start, entry_stop=start + n, library="np"
    )
    tj_cpp = truth_jets([p / 1000.0 for p in tr["tr_pt"]], tr["tr_eta"], tr["tr_phi"])
    legs = {
        CMS: collect((r for b in loader for r in rows(b)), tj_sample),
        PARNASSUS_P: collect(card_rows(card, best, loader, cut, eta_cut), tj_sample),
        DELPHES: collect(cpp_rows(delphes, start, n, cut, eta_cut), tj_cpp),
    }
    if parnassus:
        legs[PARNASSUS_F] = collect(parnassus_rows(parnassus, start, n, cut, eta_cut), parnassus_truth_jets(parnassus, start, n))
    seen = {name: leg["n_events"] for name, leg in legs.items()}
    if any(c != n for c in seen.values()):
        raise RuntimeError(f"legs saw different numbers of jets: {seen} (expected {n} each)")

    out = args.workspace / "plots"
    out.mkdir(exist_ok=True)
    tag = f"fullsim_comparison_pt{cut:g}"
    comp = [name for name in legs if name != CMS]
    lines = [
        f"{args.workspace.name}: {n} jets/leg from entry {start}, "
        f"pt >= {cut:g} GeV, |eta| <= {eta_cut}, best {run['best_result']['epoch']}. "
        f"{DELPHES}: PU mu=6.35, unfiltered truth; {PARNASSUS_P}: no PU, truth pt>=0.25, untruncated.",
    ]
    if parnassus:
        lines.append(
            f"{PARNASSUS_F}: {parnassus.name}; same jets as {DELPHES}: {same_events(parnassus, delphes, start, n)}, "
            f"as {CMS}: {same_events(parnassus, args.sample, start, n)}; no class info -> All objects + jet pages only."
        )
    lines.append(
        f"{'page':26s}" + "".join(f" {'W1 ' + SHORT[c]:>9s}" for c in comp)
        + "".join(f" {'N ' + SHORT[c] + '/CMS':>12s}" for c in comp)
    )

    def row(name, m):
        lines.append(
            f"{name:26s}" + "".join(f" {m[c][0]:9.4f}" if c in m else f" {'-':>9s}" for c in comp)
            + "".join(f" {m[c][1]:12.3f}" if c in m else f" {'-':>12s}" for c in comp)
        )

    with PdfPages(out / f"{tag}.pdf") as pdf:
        for page, charged in PAGES.items():
            for key, xlabel in OBS.items():
                # A leg without class information (Parnassus) is drawn on the "All objects" page only.
                samples = {name: f[key] if charged is None else f[key][f["charged"] == charged]
                           for name, f in legs.items() if charged is None or f["charged"] is not None}
                title = rf"{page}   ($p_\mathrm{{T}} \geq {cut:g}$ GeV)"
                note = f"{PARNASSUS_F}: no class info" if PARNASSUS_F in legs and PARNASSUS_F not in samples else None
                row(f"{page} {key}", draw_page(pdf, title, xlabel, samples, "Objects", key, note))
        for key, xlabel in JET_OBS.items():
            samples = {name: f[key] for name, f in legs.items()}
            title = rf"Leading jet, anti-$k_t$ $R=0.5$   (constituent $p_\mathrm{{T}} \geq {cut:g}$ GeV)"
            row(key, draw_page(pdf, title, xlabel, samples, "Jets", key))

    text = "\n".join(lines)
    (out / f"{tag}_metrics.txt").write_text(text + "\n")
    print(text)
    print(f"Wrote {out / tag}.pdf")


if __name__ == "__main__":
    main()
