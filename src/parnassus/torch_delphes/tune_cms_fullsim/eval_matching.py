"""Diagnostics for the truth<->reco survival matcher behind ``--eff-loss bce``.

Runs :func:`data.match_event` on a plain sample exactly as training does (both
lists cut BEFORE matching, same rule and gate) and reports, per charged species:

1. the fraction of reco objects with no truth match (fakes / unmodeled sources);
2. the matched survival fraction vs truth pt in fine bins;
3. deltaR from each truth particle to its nearest matchable reco object (any
   charged class, or the same class with --match-within-species), split by pt
   band, with the gate marked (is the gate cutting a real tail?);
4. the matched survival fraction per tracking-efficiency region -- the number the
   BCE converges to in each bin;
5. ``scans/``: heatmaps of the matched fraction over a (truth-pt cut, reco-pt cut)
   grid at several gates -- ``truth_survival/dR_<gate>.png`` (fraction of truth
   particles matched) and ``reco_survival/dR_<gate>.png`` (fraction of reco
   objects matched), one panel per species.

Usage:
    python -m parnassus.torch_delphes.tune_cms_fullsim.eval_matching \\
        --root-file /global/cfs/cdirs/m3246/diff_delphes/cms_opendata_zenodo/train_1000.root \\
        --n-events 20000 --matching hungarian --max-dr 0.05 --reco-pt-cut 5 \\
        --output-dir doc/matching_diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from parnassus.torch_delphes.learnable import EFF_BINNING_SPECS  # noqa: E402

from .config import (  # noqa: E402
    DEFAULT_ABS_ETA_CUT,
    DEFAULT_RECO_PT_CUT,
    DEFAULT_TRUTH_PT_CUT,
    MATCHING_CHOICES,
)
from .data import (  # noqa: E402
    MATCH_MAX_DR,
    _delta_phi,
    _in_acceptance,
    load_cms_flow_root,
    match_event,
    match_groups,
)

SPECIES = {0: "charged_hadron", 1: "electron", 2: "muon"}
PT_BANDS = ((0.0, 1.0), (1.0, 5.0), (5.0, 20.0), (20.0, np.inf))
SCAN_PT_CUTS = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
SCAN_MAX_DR = (0.01, 0.05, 0.1)


def _events(arrays):
    """Per-event numpy dicts (truth/pflow pt, eta, phi, class), parsed once."""
    keys = ("pt", "eta", "phi", "class")
    return [
        ({k: np.asarray(arrays[f"truth_{k}"][i], dtype=np.float64) for k in keys},
         {k: np.asarray(arrays[f"pflow_{k}"][i], dtype=np.float64) for k in keys})
        for i in range(len(arrays["truth_pt"]))
    ]


def _cut(t, r, truth_pt_cut, reco_pt_cut, abs_eta_cut):
    tk = _in_acceptance(t["pt"], t["eta"], truth_pt_cut, abs_eta_cut)
    rk = _in_acceptance(r["pt"], r["eta"], reco_pt_cut, abs_eta_cut)
    return {k: v[tk] for k, v in t.items()}, {k: v[rk] for k, v in r.items()}


def scan(events, truth_cuts, reco_cuts, max_dr, abs_eta_cut, matching, within_species):
    """Matched fractions over the (truth cut, reco cut) grid at one gate:
    ``(truth_frac, reco_frac)``, each ``(n_species, n_truth_cuts, n_reco_cuts)``."""
    n_t = np.zeros((len(SPECIES), len(truth_cuts), len(reco_cuts), 2))  # [..., (matched, total)]
    n_r = np.zeros_like(n_t)
    for t0, r0 in events:
        for a, tc in enumerate(truth_cuts):
            for b, rc in enumerate(reco_cuts):
                t, r = _cut(t0, r0, tc or None, rc or None, abs_eta_cut)
                tcls, rcls = t["class"].astype(np.int64), r["class"].astype(np.int64)
                surv, match = match_event(t["eta"], t["phi"], tcls, r["eta"], r["phi"], rcls, max_dr, matching, within_species)
                for c, cls in enumerate(SPECIES):
                    n_t[c, a, b] += (surv[tcls == cls].sum(), (tcls == cls).sum())
                    n_r[c, a, b] += (match[rcls == cls].sum(), (rcls == cls).sum())
    with np.errstate(invalid="ignore"):
        return n_t[..., 0] / n_t[..., 1], n_r[..., 0] / n_r[..., 1]


def _heatmap(frac, truth_cuts, reco_cuts, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    order = (1, 2, 0)  # electron, muon, charged hadron
    for ax, c in zip(axes, order):
        im = ax.imshow(frac[c].T, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for a in range(len(truth_cuts)):
            for b in range(len(reco_cuts)):
                v = frac[c, a, b]
                if np.isfinite(v):
                    ax.text(a, b, f"{v:.2f}", ha="center", va="center", fontsize=7,
                            color="white" if v < 0.5 else "black")
        ax.set(xticks=range(len(truth_cuts)), yticks=range(len(reco_cuts)),
               xlabel="truth pt cut [GeV]", ylabel="reco pt cut [GeV]", title=SPECIES[c])
        ax.set_xticklabels([f"{v:g}" for v in truth_cuts]); ax.set_yticklabels([f"{v:g}" for v in reco_cuts])
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="matched fraction")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def collect(arrays, truth_pt_cut, reco_pt_cut, abs_eta_cut, matching, max_dr, within_species):
    """Flat per-particle arrays over all events, after the training-time cuts:
    truth (pt, eta, class, survived, dr_nearest) and reco (class, matched)."""
    t_pt, t_eta, t_cls, t_surv, t_dr, r_cls, r_match = ([] for _ in range(7))
    for i in range(len(arrays["truth_pt"])):
        t = {k: np.asarray(arrays[f"truth_{k}"][i], dtype=np.float64) for k in ("pt", "eta", "phi", "class")}
        r = {k: np.asarray(arrays[f"pflow_{k}"][i], dtype=np.float64) for k in ("pt", "eta", "phi", "class")}
        tk = _in_acceptance(t["pt"], t["eta"], truth_pt_cut, abs_eta_cut)
        rk = _in_acceptance(r["pt"], r["eta"], reco_pt_cut, abs_eta_cut)
        t = {k: v[tk] for k, v in t.items()}
        r = {k: v[rk] for k, v in r.items()}
        tc, rc = t["class"].astype(np.int64), r["class"].astype(np.int64)
        surv, match = match_event(t["eta"], t["phi"], tc, r["eta"], r["phi"], rc, max_dr, matching, within_species)
        # deltaR to the nearest reco object the rule may pair with (inf when none)
        dr = np.full(tc.shape[0], np.inf)
        for group in match_groups(within_species):
            ti, ri = np.flatnonzero(np.isin(tc, group)), np.flatnonzero(np.isin(rc, group))
            if ti.size and ri.size:
                deta = t["eta"][ti][:, None] - r["eta"][ri][None, :]
                dphi = _delta_phi(t["phi"][ti], r["phi"][ri])
                dr[ti] = np.sqrt(deta**2 + dphi**2).min(axis=1)
        t_pt.append(t["pt"]); t_eta.append(t["eta"]); t_cls.append(tc)
        t_surv.append(surv); t_dr.append(dr); r_cls.append(rc); r_match.append(match)
    cat = np.concatenate
    return dict(t_pt=cat(t_pt), t_eta=cat(t_eta), t_cls=cat(t_cls), t_surv=cat(t_surv),
                t_dr=cat(t_dr), r_cls=cat(r_cls), r_match=cat(r_match))


def _frac(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-file", type=Path, required=True)
    parser.add_argument("--n-events", type=int, default=20000)
    parser.add_argument("--matching", type=str, default="hungarian", choices=list(MATCHING_CHOICES))
    parser.add_argument("--max-dr", type=float, default=MATCH_MAX_DR, help="deltaR gate")
    parser.add_argument("--match-within-species", action="store_true",
                        help="pair only same-species truth/reco (default: across all charged species)")
    parser.add_argument("--truth-pt-cut", type=float, default=DEFAULT_TRUTH_PT_CUT, help="<= 0 disables")
    parser.add_argument("--reco-pt-cut", type=float, default=DEFAULT_RECO_PT_CUT, help="<= 0 disables")
    parser.add_argument("--eta-cut", type=float, default=DEFAULT_ABS_ETA_CUT, help="<= 0 disables")
    parser.add_argument("--eff-binning", type=str, default="ptbins12", choices=list(EFF_BINNING_SPECS))
    parser.add_argument("--output-dir", type=Path, default=Path("doc/matching_diagnostics"))
    parser.add_argument("--scan-pt-cuts", type=float, nargs="+", default=list(SCAN_PT_CUTS),
                        help="grid of truth AND reco pt cuts for the scans/ heatmaps (0 = no cut)")
    parser.add_argument("--scan-max-dr", type=float, nargs="+", default=list(SCAN_MAX_DR),
                        help="gates for the scans/ heatmaps (one file each)")
    args = parser.parse_args()
    cut = lambda v: v if v > 0 else None
    truth_pt_cut, reco_pt_cut, abs_eta_cut = cut(args.truth_pt_cut), cut(args.reco_pt_cut), cut(args.eta_cut)

    arrays = load_cms_flow_root(args.root_file, n_events=args.n_events)
    d = collect(arrays, truth_pt_cut, reco_pt_cut, abs_eta_cut, args.matching, args.max_dr, args.match_within_species)
    rule = f"{args.matching}, {'within' if args.match_within_species else 'across'} species"
    print(f"[eval_matching] {args.root_file.name}: {len(arrays['truth_pt'])} events, matching={rule}, "
          f"max_dr={args.max_dr}, cuts: truth pt>={truth_pt_cut}, reco pt>={reco_pt_cut}, |eta|<={abs_eta_cut}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"matching": args.matching, "within_species": args.match_within_species,
                     "max_dr": args.max_dr, "per_species": {}}
    pt_edges = np.geomspace(max(truth_pt_cut or 0.1, 0.1), 1000.0, 31)
    dr_edges = np.geomspace(1e-4, 1.0, 41)
    with PdfPages(args.output_dir / "matching_diagnostics.pdf") as pdf:
        fig_s, ax_s = plt.subplots(1, 3, figsize=(15, 4))
        fig_d, ax_d = plt.subplots(1, 3, figsize=(15, 4))
        fig_r, ax_r = plt.subplots(1, 3, figsize=(15, 4))
        for col, (cls, name) in enumerate(SPECIES.items()):
            t = d["t_cls"] == cls
            r = d["r_cls"] == cls
            pt, surv, dr = d["t_pt"][t], d["t_surv"][t], d["t_dr"][t]
            s = {
                "n_truth": int(t.sum()), "n_reco": int(r.sum()),
                "truth_survival_fraction": _frac(surv),
                "reco_unmatched_fraction": _frac(~d["r_match"][r]),
                "beyond_gate_fraction_by_pt_band": {
                    f"[{lo},{hi})": _frac(dr[(pt >= lo) & (pt < hi)] > args.max_dr) for lo, hi in PT_BANDS
                },
            }
            # 2. survival vs pt
            idx = np.digitize(pt, pt_edges) - 1
            frac = np.array([_frac(surv[idx == b]) for b in range(len(pt_edges) - 1)])
            n_b = np.array([(idx == b).sum() for b in range(len(pt_edges) - 1)])
            err = np.sqrt(np.clip(frac * (1 - frac), 0, None) / np.maximum(n_b, 1))
            ctr = np.sqrt(pt_edges[1:] * pt_edges[:-1])
            ax_s[col].errorbar(ctr, frac, yerr=err, fmt="o-", ms=3, lw=1)
            ax_s[col].set(xscale="log", ylim=(-0.02, 1.02), xlabel="truth pt [GeV]",
                          ylabel="matched survival fraction", title=f"{name} (n={t.sum()})")
            ax_s[col].grid(alpha=0.3)
            # 3. deltaR to the nearest same-class reco object, by pt band
            for lo, hi in PT_BANDS:
                sel = (pt >= lo) & (pt < hi) & np.isfinite(dr)
                ax_d[col].hist(np.clip(dr[sel], dr_edges[0], dr_edges[-1]), bins=dr_edges, histtype="step",
                               lw=1.5, label=f"pt in [{lo:g}, {hi:g}) n={sel.sum()}")
            ax_d[col].axvline(args.max_dr, color="k", ls="--", lw=1, label=f"gate {args.max_dr}")
            ax_d[col].set(xscale="log", yscale="log", xlabel="deltaR to nearest matchable reco",
                          title=f"{name}: unmatched reco {s['reco_unmatched_fraction']:.3f}")
            ax_d[col].legend(fontsize=7)
            # 4. survival per efficiency region
            spec = EFF_BINNING_SPECS[args.eff_binning][name]
            masks = spec.region_masks(pt, np.abs(d["t_eta"][t]))
            reg = [(_frac(surv[m]), int(m.sum())) for m in masks]
            s["survival_by_region"] = [f for f, _n in reg]
            s["n_by_region"] = [n for _f, n in reg]
            ax_r[col].bar(range(len(reg)), [f for f, _n in reg], color="C0")
            for k, (f, n) in enumerate(reg):
                ax_r[col].text(k, f + 0.01, f"n={n}", ha="center", fontsize=6)
            labels = [f"e{k // spec.n_pt}p{k % spec.n_pt}" for k in range(len(reg))]  # e<eta bin>p<pt bin>
            ax_r[col].set(xticks=range(len(reg)), ylim=(0, 1.1), ylabel="matched survival fraction",
                          xlabel="region (e = |eta| bin, p = pt bin; eta-major order)",
                          title=f"{name}: per efficiency region ({args.eff_binning})")
            ax_r[col].set_xticklabels(labels, fontsize=7)
            summary["per_species"][name] = s
            print(f"  {name:15s} truth n={s['n_truth']:7d} matched={s['truth_survival_fraction']:.4f} | "
                  f"reco n={s['n_reco']:7d} matched={(1-s['reco_unmatched_fraction']):.4f} | "
                  f"beyond gate by pt band: " + ", ".join(f"{k}: {v:.3f}" for k, v in s["beyond_gate_fraction_by_pt_band"].items()))
        for fig, slug, title in ((fig_s, "survival_vs_pt", "matched survival fraction vs truth pt"),
                                 (fig_d, "deltaR_nearest_reco", "deltaR to the nearest matchable reco object"),
                                 (fig_r, "survival_per_region", "matched survival fraction per efficiency region")):
            fig.suptitle(f"{title}  [{rule}, dR <= {args.max_dr}, reco pt >= {reco_pt_cut}]")
            fig.tight_layout()
            pdf.savefig(fig)
            fig.savefig(args.output_dir / f"{slug}.png", dpi=110)
            plt.close(fig)
    (args.output_dir / "matching_diagnostics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[eval_matching] wrote {args.output_dir / 'matching_diagnostics.pdf'} (+ per-page PNGs) and .json")

    # 5. (truth cut, reco cut) scans at each gate
    events = _events(arrays)
    cuts = args.scan_pt_cuts
    for max_dr in args.scan_max_dr:
        truth_frac, reco_frac = scan(events, cuts, cuts, max_dr, abs_eta_cut, args.matching, args.match_within_species)
        name = f"dR_{max_dr:g}.png".replace(".", "p").replace("ppng", ".png")
        for sub_dir, frac, what in (("truth_survival", truth_frac, "fraction of truth particles matched"),
                                    ("reco_survival", reco_frac, "fraction of reco objects matched")):
            d = args.output_dir / "scans" / sub_dir
            d.mkdir(parents=True, exist_ok=True)
            _heatmap(frac, cuts, cuts, f"{what}  [{rule}, dR <= {max_dr:g}, |eta| <= {abs_eta_cut}]", d / name)
        print(f"[eval_matching] scans at dR <= {max_dr:g} -> {args.output_dir / 'scans'}/*/{name}")


if __name__ == "__main__":
    main()
