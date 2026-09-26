"""Compare the in-training truth<->reco survival matching against the
preprocessed-matched sample set.

For each sequential-closure process (muongun, ksgun, dijet, electrongun):

1. load the PLAIN sample (``pseudo_data_200k_param_config_all_{p}.root``);
2. load the PREPROCESSED-matched sample
   (``..._{p}_hungarian_matched_survival.root``), whose ``truth_survived``
   branch holds the preprocessing-era Hungarian labels (plus, when present,
   ``truth_survived_generator`` — the generator-hook ground truth — and
   ``truth_eff_region``);
3. run the IN-TRAINING matcher (:func:`tune_cms_fullsim.data.match_event`,
   per-class Hungarian, deltaR <= 0.05) on the plain sample;
4. compare the two label sets per charged class (agreement, false
   positives/negatives, survival fractions), and each against the generator
   labels where available.

The preprocessed files carry no match indices, only labels, so the comparison
is label-level (as is the loss's consumption of them). Events are verified to
be THE SAME between the two files (multiplicity + truth pt, bitwise) before
any comparison; mismatched events would make label comparison meaningless.

Usage (login node is fine; ~1 min per 20k events per sample):

    python -m parnassus.torch_delphes.full_phasespace_tuning.compare_matching \\
        --n-events 20000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import uproot

from parnassus.torch_delphes.tune_cms_fullsim.data import MATCH_MAX_DR, match_event

PROCESSES: tuple[str, ...] = ("muongun", "ksgun", "dijet", "electrongun")
CLASS_NAMES: dict[int, str] = {0: "chad", 1: "electron", 2: "muon"}

PLAIN_BRANCHES = (
    "truth_pt", "truth_eta", "truth_phi", "truth_class",
    "pflow_eta", "pflow_phi", "pflow_class",
)
LABEL_BRANCHES = ("truth_survived", "truth_survived_generator", "truth_eff_region")


def _read(path: Path, branches: tuple[str, ...], n_events: int) -> dict:
    with uproot.open(f"{path}:event_tree") as tree:
        have = [b for b in branches if b in tree]
        stop = None if n_events < 0 else n_events
        return tree.arrays(have, library="np", entry_stop=stop)


def _flat(arrays: dict, key: str, dtype) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=dtype) for a in arrays[key]])


def _assert_same_events(plain: dict, labeled: dict, proc: str) -> None:
    n_p, n_l = len(plain["truth_pt"]), len(labeled["truth_pt"])
    assert n_p == n_l, f"{proc}: event counts differ ({n_p} vs {n_l})"
    mult_p = np.array([len(a) for a in plain["truth_pt"]])
    mult_l = np.array([len(a) for a in labeled["truth_pt"]])
    assert (mult_p == mult_l).all(), f"{proc}: per-event truth multiplicities differ"
    pt_p = _flat(plain, "truth_pt", np.float64)
    pt_l = _flat(labeled, "truth_pt", np.float64)
    assert np.array_equal(pt_p, pt_l), f"{proc}: truth pt content differs"


def _jit_labels(plain: dict, max_dr: float) -> np.ndarray:
    out = []
    for i in range(len(plain["truth_pt"])):
        out.append(match_event(
            np.asarray(plain["truth_eta"][i], np.float64),
            np.asarray(plain["truth_phi"][i], np.float64),
            np.asarray(plain["truth_class"][i], np.int64),
            np.asarray(plain["pflow_eta"][i], np.float64),
            np.asarray(plain["pflow_phi"][i], np.float64),
            np.asarray(plain["pflow_class"][i], np.int64),
            max_dr=max_dr,
        ))
    return np.concatenate(out)


def _report(name: str, a: np.ndarray, b: np.ndarray, m: np.ndarray) -> str:
    """One comparison line for labels ``a`` vs ``b`` on mask ``m``."""
    n = int(m.sum())
    if n == 0:
        return f"      {name:<34} n=0"
    agree = float((a[m] == b[m]).mean())
    fp = int((a[m] & ~b[m]).sum())
    fn = int((~a[m] & b[m]).sum())
    return (f"      {name:<34} n={n:<8} agree={agree:.6f}  "
            f"a=1,b=0: {fp:<6} a=0,b=1: {fn:<6} "
            f"frac(a)={a[m].mean():.4f} frac(b)={b[m].mean():.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--samples-dir", type=Path,
        default=Path("/global/cfs/cdirs/m3246/diff_delphes/allsamples"),
    )
    ap.add_argument("--n-events", type=int, default=20000, help="-1 = all")
    ap.add_argument("--max-dr", type=float, default=MATCH_MAX_DR)
    ap.add_argument("--processes", nargs="+", default=list(PROCESSES))
    args = ap.parse_args()

    any_disagreement = False
    for proc in args.processes:
        plain_f = args.samples_dir / f"pseudo_data_200k_param_config_all_{proc}.root"
        lab_f = args.samples_dir / (
            f"pseudo_data_200k_param_config_all_{proc}_hungarian_matched_survival.root"
        )
        plain = _read(plain_f, PLAIN_BRANCHES, args.n_events)
        labeled = _read(lab_f, PLAIN_BRANCHES + LABEL_BRANCHES, args.n_events)
        _assert_same_events(plain, labeled, proc)

        jit = _jit_labels(plain, args.max_dr).astype(bool)
        pre = _flat(labeled, "truth_survived", np.float64).astype(bool)
        cls = _flat(plain, "truth_class", np.int64)
        gen = (
            _flat(labeled, "truth_survived_generator", np.float64).astype(bool)
            if "truth_survived_generator" in labeled else None
        )
        reg = (
            _flat(labeled, "truth_eff_region", np.int64)
            if "truth_eff_region" in labeled else None
        )

        print(f"== {proc}  ({len(plain['truth_pt'])} events, "
              f"{cls.size} truth rows, max_dr={args.max_dr})")
        for c, cname in CLASS_NAMES.items():
            m = cls == c
            if not m.any():
                continue
            print(f"   {cname}:")
            print(_report("in-training vs preprocessed", jit, pre, m))
            any_disagreement |= bool((jit[m] != pre[m]).any())
            if gen is not None:
                print(_report("in-training vs generator", jit, gen, m))
                print(_report("preprocessed vs generator", pre, gen, m))
            if reg is not None:
                print(_report(
                    "in-training vs preprocessed (region>0)", jit, pre, m & (reg > 0)
                ))

    print("\nRESULT:",
          "labels DISAGREE somewhere (see lines above)" if any_disagreement
          else "in-training and preprocessed labels are IDENTICAL on every "
               "compared row")


if __name__ == "__main__":
    main()
