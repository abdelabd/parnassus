r"""Overlay of one fullsim fit: CMS full sim vs diff-Delphes vs C++ Delphes vs Parnassus.

Legend labels: "CMS" = full sim, "Parnassus-P" = the diff-Delphes card, "Delphes" = C++ Delphes,
"Parnassus-F" = the generative Parnassus fast sim (the prose below uses the long names).

All legs use entries ``[--entry-start, --entry-start + --n-events)`` of ``--sample``
(disjoint from the fit's training range ``[0, n_events)``): the SAME entry range and the same
number of jets on every leg (a file too short for the range is an error, never a silent
truncation), and the same acceptance on every object: ``pt >= reco_pt_cut`` (the fit's, or
``--reco-pt-cut``) and ``|eta| <= eta_cut``.
Pages: the leading jet's pT and mass response (x_reco - x_truth) / x_truth, then eta_reco - eta_truth
and phi_reco - phi_truth (wrapped to [-pi, pi)) relative to the entry's truth jet (anti-kt R=0.5 on
massless constituents; jet pt > 8 GeV, >= 2 constituents, |eta| < 2.5 -- the cppDelphes
leading_jet_filter convention; the truth jet is clustered the same way from ALL truth particles of
the entry -- the sample's truth_* branches for the full-sim and diff-Delphes legs, the Delphes /
Parnassus file's own truth_tree for those legs). Raw counts on linear y with a ratio-to-full-sim
panel (band = full-sim sqrt(N)); fixed binning per page (``BINS``); a metrics file lists the
quantile-W1 distance to full sim and the yield ratio per page.

Every entry of every file is ONE anti-kt R=0.5 jet of the CMS open-simulation sample (its own
constituents only; a dijet event contributes two consecutive entries), so the "leading jet" of an
entry is that jet.

Legs and their structural differences (keep in mind when ranking):

- full sim: ``event_tree`` pflow constituents of ``--sample`` (real pileup, pt >= 1 GeV).
- diff-Delphes: the card at the trial's best-epoch parameters on the truth of the same
  entries (no pileup, truth pt >= 0.25 GeV; no chad truncation, while the fit's target was
  truncated to the truth chad ceiling).
- C++ Delphes: ``fastsim_tree`` of the sibling ``delphes_pu6p35_jet<bin>.root`` (PU mu=6.35
  on the unfiltered truth; ``fs_pt`` in MeV). Entries are the same events as ``test_470.root`` /
  ``test_800.root``; for jet1000 they are an independent sample of the same process (equal
  event count, so raw counts still compare).
- Parnassus: ``fastsim_tree`` of the generative fast sim (flow matching, PNDM sampler, sampled
  multiplicity) in ``PARNASSUS_FILES[<bin>]``, auto-picked from the sample's pT-hat bin (800 and
  1000 exist; any other bin skips the leg). Fixed 201-slot arrays: ``fs_ind`` 1 marks a generated
  particle and the other slots hold NON-zero padding that must be masked; ``fs_pt`` in MeV. The
  model generates only (pt, eta, phi), which is all the response pages need. Its 200k jets are
  the first 200k entries of the Delphes file of the same bin (same events as the C++ leg; same
  events as full sim for test_800, an independent sample of the same process for train_1000 --
  the metrics header says which). ~0.3 % of its jets carry a phi-wrapped stray constituent far
  from the axis and ~20 % put some soft particles beyond R=0.5; the reclustering sheds both into
  separate jets (median leading-jet pT loss 0.2 %).

    python -m parnassus.torch_delphes.plotting_scripts.plot_fullsim_comparison \\
        --workspace doc/figure_fullsim/470_frac_pt5 \\
        --sample /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/test_470.root
"""

import argparse
import json
from pathlib import Path
from typing import NamedTuple

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

# ---- text / layout knobs -----------------------------------------------------------------------
LABEL_SIZE = 30  # pt: x/y axis labels and "Ratio" (ATLAS default 20)
TICK_SIZE = 20  # pt: tick labels
LEGEND_SIZE = 24  # pt: legend; a 2-column legend is ~15 x this wide and must fit inside the axes width
FIG_SIZE = (8.6, 7.8)  # inches; sized so the margins leave ~0.2 in beyond the outermost tick label / "x10^4" offset text
MARGINS = dict(left=0.16, right=0.96, bottom=0.06, top=0.95)  # axes box = 6.9 x 6.9 in (no x label: it is added in LaTeX)
HEIGHT_RATIOS = (8, 3)  # top panel : ratio panel (the ratio panel must fit the rotated "Ratio" label)
RATIO_YLIM = (0.0, 2.0)  # ratio panel (was 0.5-1.5; the tails clipped)
RATIO_YTICKS = (0.0, 0.5, 1.0, 1.5)  # no tick at the top edge: its label would collide with the upper panel's "0"
HEADROOM = 1.45  # y-limit = HEADROOM x tallest bin; the band above the data holds the 2-row legend
LEGEND_NCOL = 2  # entries per legend row (4 legs -> 2 rows, read row by row in LEGEND_ORDER)
plt.rcParams.update({
    "font.size": LEGEND_SIZE, "axes.labelsize": LABEL_SIZE, "legend.fontsize": LEGEND_SIZE,
    "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE, "figure.figsize": FIG_SIZE,
    "legend.handlelength": 1.5, "legend.columnspacing": 1.0, "legend.handletextpad": 0.5, "legend.borderpad": 0.4,
    "axes.labelpad": 8,
})

BATCH_SIZE = 2000
SEED = 0
JET_DEF = fj.JetDefinition(fj.antikt_algorithm, 0.5)
JET_PT_MIN, JET_N_CONST_MIN, JET_ABS_ETA_MAX = 8.0, 2, 2.5
# Pages in order: x label per leading-jet response (also the keys of BINS, collect() and the metrics rows).
JET_OBS = {
    "jet_pt_resp": "Leading jet\n" + r"$(p_\mathrm{T} - p_\mathrm{T}^{\,\mathrm{truth}})\,/\,p_\mathrm{T}^{\,\mathrm{truth}}$",
    "jet_m_resp": "Leading jet\n" + r"$(m - m^{\,\mathrm{truth}})\,/\,m^{\,\mathrm{truth}}$",
    "jet_eta_resp": "Leading jet\n" + r"$\eta - \eta^{\,\mathrm{truth}}$",
    "jet_phi_resp": "Leading jet\n" + r"$\phi - \phi^{\,\mathrm{truth}}$",
}
# (n_bins, low, high) per page; None -> pooled 0.5 % / 99.5 % quantile.
# Ranges end where the data ends on both sides (the legend lives in its own band above the data).
BINS = {
    "jet_pt_resp": (26, -0.4, 0.25),
    "jet_m_resp": (30, -1.0, 1.0),
    "jet_eta_resp": (20, -0.02, 0.02),
    "jet_phi_resp": (20, -0.02, 0.02),  # same spread as eta on every leg (sigma ~0.007-0.01)
}
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
    """Padded observable dict -> per event (pt, eta, phi) numpy arrays."""
    for i in range(obs["pt"].shape[0]):
        m = obs["pt"][i] != 0
        pt, eta, phi = (obs[k][i, m].numpy() for k in ("pt", "eta", "phi"))
        yield pt, eta, phi


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
        ["fs_pt", "fs_eta", "fs_phi"], entry_start=start, entry_stop=start + n, library="np"
    )
    for pt, eta, phi in zip(a["fs_pt"], a["fs_eta"], a["fs_phi"]):
        pt = pt / 1000.0
        m = (pt >= cut) & (np.abs(eta) <= eta_cut)
        yield pt[m], eta[m], phi[m]


def parnassus_rows(path, start, n, cut, eta_cut):
    """Parnassus ``fastsim_tree``: 201 fixed slots, ``fs_ind == 1`` = generated particle (the
    other slots are NON-zero padding), ``fs_pt`` in MeV."""
    a = uproot.open(path)["fastsim_tree"].arrays(
        ["fs_pt", "fs_eta", "fs_phi", "fs_ind"], entry_start=start, entry_stop=start + n, library="np"
    )
    for pt, eta, phi, ind in zip(a["fs_pt"], a["fs_eta"], a["fs_phi"], a["fs_ind"]):
        real = ind == 1
        pt, eta, phi = pt[real] / 1000.0, eta[real], phi[real]
        m = (pt >= cut) & (np.abs(eta) <= eta_cut)
        yield pt[m], eta[m], phi[m]


class Jet(NamedTuple):
    pt: float
    eta: float
    phi: float  # PseudoJet.phi_std(): [-pi, pi)
    m: float  # floored at 1e-3


def leading_jet(pt, eta, phi):
    """Leading selected anti-kt jet of massless (pt, eta, phi) as a ``Jet``, or None."""
    px, py, pz, e = pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), pt * np.cosh(eta)
    cs = fj.ClusterSequence([fj.PseudoJet(*v) for v in zip(px, py, pz, e)], JET_DEF)
    jets = [
        j for j in fj.sorted_by_pt(cs.inclusive_jets(JET_PT_MIN))
        if len(j.constituents()) >= JET_N_CONST_MIN and abs(j.eta()) < JET_ABS_ETA_MAX
    ]
    if not jets:
        return None
    j = jets[0]
    return Jet(j.pt(), j.eta(), j.phi_std(), max(j.m(), 1e-3))


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


def dphi(a, b):
    """a - b wrapped to [-pi, pi) (both from ``phi_std()``, so at most one wrap)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def collect(events, truths):
    """Leading-jet responses to the entry's truth jet: pt and m as x_reco / x_truth - 1, eta as the
    difference, phi as the difference wrapped to [-pi, pi). Entries without a selected reco or
    truth jet contribute nothing; ``n_events`` counts every entry seen so the legs can be checked
    against each other.
    """
    acc = {k: [] for k in JET_OBS}
    n_events = 0
    for (pt, eta, phi), tj in zip(events, truths):
        n_events += 1
        jet = leading_jet(pt, eta, phi)
        if jet is None or tj is None:
            continue
        acc["jet_pt_resp"].append(jet.pt / tj.pt - 1)
        acc["jet_m_resp"].append(jet.m / tj.m - 1)
        acc["jet_eta_resp"].append(jet.eta - tj.eta)
        acc["jet_phi_resp"].append(dphi(jet.phi, tj.phi))
    out = {k: np.asarray(acc[k], float) for k in JET_OBS}
    out["n_events"] = n_events
    return out


def qwass(a, b, n_q=200):
    q = (np.arange(n_q) + 0.5) / n_q
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def draw_page(pdf, xlabel, samples, ylabel, key):
    """Linear-y raw counts + ratio to CMS full sim (band = its sqrt(N)); returns (W1, N / N_CMS) per leg.

    The legend sits in the band above the data (HEADROOM), LEGEND_NCOL entries per row in
    LEGEND_ORDER; the x label names the page and is centred.
    """
    n_bins, lo, hi = BINS[key]
    qlo, qhi = np.quantile(np.concatenate(list(samples.values())), (0.005, 0.995))
    edges = np.linspace(qlo if lo is None else lo, qhi if hi is None else hi, n_bins + 1)
    counts = {n: np.histogram(v, edges)[0].astype(float) for n, v in samples.items()}

    fig, (ax, rax) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": 0.06})
    fig.subplots_adjust(**MARGINS)
    for n, c in counts.items():
        ax.stairs(c, edges, label=n, **STYLE[n])
    ax.set_ylim(0, max(c.max() for c in counts.values()) * HEADROOM)
    ax.set_ylabel(ylabel)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: LEGEND_ORDER.index(labels[i]))  # row-major intent
    n_rows = -(-len(order) // LEGEND_NCOL)  # matplotlib fills columns first -> transpose
    order = [order[r * LEGEND_NCOL + c] for c in range(LEGEND_NCOL) for r in range(n_rows) if r * LEGEND_NCOL + c < len(order)]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left", ncol=LEGEND_NCOL)

    fs = counts[CMS]
    ok = fs > 0  # empty full-sim bins: no band (an inf corner breaks the fill polygon), no ratio
    rel = np.divide(1, np.sqrt(fs), out=np.zeros_like(fs), where=ok)  # full-sim sqrt(N) on the ratio
    rax.stairs(1 + rel, edges, baseline=1 - rel, color=STYLE[CMS]["color"], fill=True, alpha=0.3)
    for n in counts:
        if n != CMS:
            rax.stairs(np.divide(counts[n], fs, out=np.full_like(fs, np.nan), where=ok), edges, **STYLE[n])
    rax.axhline(1, color=STYLE[CMS]["color"], linewidth=1)
    rax.set_ylim(*RATIO_YLIM)
    rax.set_yticks(RATIO_YTICKS)
    rax.set_xlim(edges[0], edges[-1])
    rax.xaxis.set_major_formatter(FormatStrFormatter("%g"))  # no "x10^-2" offset text under the label
    rax.set_ylabel("Ratio")
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
    start, n = args.entry_start, args.n_events
    pt_bin = args.sample.stem.split("_")[1]
    delphes =args.delphes or args.sample.with_name(f"delphes_pu6p35_jet{pt_bin}.root")
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
            f"as {CMS}: {same_events(parnassus, args.sample, start, n)}."
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
        for key, xlabel in JET_OBS.items():
            row(key, draw_page(pdf, xlabel, {name: f[key] for name, f in legs.items()}, "Jets", key))

    text = "\n".join(lines)
    (out / f"{tag}_metrics.txt").write_text(text + "\n")
    print(text)
    print(f"Wrote {out / tag}.pdf")


if __name__ == "__main__":
    main()
