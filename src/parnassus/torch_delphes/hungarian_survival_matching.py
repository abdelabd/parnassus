"""Build truth-survival labels by Hungarian truth<->reco matching (EFF_LOSS_PLAN.md
Phase-2 rehearsal / step 5).

Reads a cms-flow ROOT file and writes a copy in which the ``truth_survived`` branch
is REPLACED by a matching-derived label: per event and per charged species
(class 0 = charged hadron, 1 = electron, 2 = muon), truth particles and reco
(pflow) objects of the same class are assigned one-to-one by the Hungarian
algorithm on a deltaR^2 cost with a max-deltaR gate; a truth particle is labeled
survived iff it received a within-gate reco match. Unmatched reco objects are
fakes for this purpose and are ignored. Neutral species are never matched (no
per-particle correspondence exists — EFF_LOSS_MOTIV.md section 2).

When the input already carries generator-truth labels (a ``*_truth_matched_survival``
sample), they are preserved under ``truth_survived_generator`` and a per-species
confusion matrix (matched label vs generator truth, on the ``truth_eff_region > 0``
support the BCE loss uses) is printed and written to a ``.confusion.json`` sidecar
— the direct measurement of the matcher's label noise, free on pseudodata.

``truth_in_tracker`` / ``truth_eff_region`` are copied through unchanged when
present. (On real fullsim data those would come from running the propagator in this
preprocessing step — not needed for the pseudodata rehearsal, where the generator
already wrote them.)

Usage:
    python -m parnassus.torch_delphes.hungarian_survival_matching \\
        --input  .../pseudo_data_200k_param_config_all_muongun_truth_matched_survival.root \\
        --output .../pseudo_data_200k_param_config_all_muongun_hungarian_matched_survival.root
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

TREE = "event_tree"
CHARGED_CLASSES = (0, 1, 2)  # chad, electron, muon (pid_to_class convention)
CLASS_NAMES = {0: "charged_hadron", 1: "electron", 2: "muon"}
# Disallowed-pair cost: any pair above the deltaR gate gets this, and an
# assignment at this cost is treated as unmatched.
_BIG = 1.0e9


def _delta_phi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a[:, None] - b[None, :]
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def match_event(
    t_eta: np.ndarray,
    t_phi: np.ndarray,
    t_cls: np.ndarray,
    r_eta: np.ndarray,
    r_phi: np.ndarray,
    r_cls: np.ndarray,
    max_dr: float,
) -> np.ndarray:
    """Per-event survival labels: True where a truth particle got a same-class
    reco match within ``max_dr`` under the per-class Hungarian assignment."""
    survived = np.zeros(t_eta.shape[0], dtype=bool)
    max_dr2 = max_dr * max_dr
    for cls in CHARGED_CLASSES:
        ti = np.flatnonzero(t_cls == cls)
        ri = np.flatnonzero(r_cls == cls)
        if ti.size == 0 or ri.size == 0:
            continue
        deta = t_eta[ti][:, None] - r_eta[ri][None, :]
        dphi = _delta_phi(t_phi[ti], r_phi[ri])
        cost = deta * deta + dphi * dphi
        cost = np.where(cost <= max_dr2, cost, _BIG)
        rows, cols = linear_sum_assignment(cost)
        ok = cost[rows, cols] < _BIG
        survived[ti[rows[ok]]] = True
    return survived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-dr",
        type=float,
        default=0.05,
        help=(
            "deltaR gate on truth-reco pairs (default 0.05: diff-Delphes momentum "
            "smearing preserves the track direction, so genuine matches sit at "
            "deltaR ~ 0; the gate mainly rejects cross-particle coincidences)."
        ),
    )
    args = parser.parse_args()

    with uproot.open(str(args.input)) as f:
        tree = f[TREE]
        branches = tree.keys(filter_typename="*[]")  # jagged only; n* regenerated
        rec = tree.arrays(branches)

    n_events = len(rec)
    has_generator_labels = "truth_survived" in rec.fields
    print(
        f"[hungarian] {args.input.name}: {n_events} events, max_dr={args.max_dr}, "
        f"generator labels: {'present' if has_generator_labels else 'absent'}"
    )

    t_eta = ak.to_list(rec["truth_eta"])
    t_phi = ak.to_list(rec["truth_phi"])
    t_cls = ak.to_list(rec["truth_class"])
    r_eta = ak.to_list(rec["pflow_eta"])
    r_phi = ak.to_list(rec["pflow_phi"])
    r_cls = ak.to_list(rec["pflow_class"])

    matched: list[np.ndarray] = []
    for i in tqdm(range(n_events), unit="evt", desc="[hungarian] matching"):
        matched.append(
            match_event(
                np.asarray(t_eta[i], dtype=np.float64),
                np.asarray(t_phi[i], dtype=np.float64),
                np.asarray(t_cls[i], dtype=np.int64),
                np.asarray(r_eta[i], dtype=np.float64),
                np.asarray(r_phi[i], dtype=np.float64),
                np.asarray(r_cls[i], dtype=np.int64),
                args.max_dr,
            )
        )

    payload = {field: rec[field] for field in rec.fields if field != "truth_survived"}
    payload["truth_survived"] = ak.Array(matched)
    if has_generator_labels:
        payload["truth_survived_generator"] = rec["truth_survived"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(str(args.output)) as fout:
        fout[TREE] = payload
    print(f"[hungarian] wrote {n_events} events -> {args.output}")

    # ---- Confusion matrix vs the generator truth, on the BCE support ----------
    if has_generator_labels and "truth_eff_region" in rec.fields:
        gen = np.concatenate([np.asarray(v, dtype=bool) for v in ak.to_list(rec["truth_survived"])])
        hun = np.concatenate(matched).astype(bool)
        region = np.concatenate(
            [np.asarray(v, dtype=np.int64) for v in ak.to_list(rec["truth_eff_region"])]
        )
        cls = np.concatenate([np.asarray(v, dtype=np.int64) for v in ak.to_list(rec["truth_class"])])
        support = region > 0
        report: dict[str, dict[str, float | int]] = {}
        for c in CHARGED_CLASSES:
            m = support & (cls == c)
            n = int(m.sum())
            if n == 0:
                continue
            tp = int((hun[m] & gen[m]).sum())
            tn = int((~hun[m] & ~gen[m]).sum())
            fp = int((hun[m] & ~gen[m]).sum())  # matcher says survived, coin said killed
            fn = int((~hun[m] & gen[m]).sum())  # coin said survived, matcher found no match
            report[CLASS_NAMES[c]] = {
                "n": n,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "label_agreement": (tp + tn) / n,
                "gen_survival": float(gen[m].mean()),
                "hungarian_survival": float(hun[m].mean()),
            }
            print(
                f"[hungarian] {CLASS_NAMES[c]:>14}: n={n}  agreement={(tp+tn)/n:.5f}  "
                f"fp={fp} fn={fn}  survival gen={gen[m].mean():.4f} vs "
                f"matched={hun[m].mean():.4f}"
            )
        sidecar = args.output.with_name(args.output.name + ".confusion.json")
        sidecar.write_text(json.dumps({"max_dr": args.max_dr, "per_species": report}, indent=2) + "\n")
        print(f"[hungarian] confusion matrix -> {sidecar}")


if __name__ == "__main__":
    main()
