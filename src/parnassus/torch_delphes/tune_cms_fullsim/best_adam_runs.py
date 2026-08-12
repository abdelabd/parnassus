#!/usr/bin/env python3
"""Report the best Adam fine-tune runs in an optuna_gebo_adam scan root.

Reads each round's ``gebo/adam/gebo_summary.json`` straight off disk -- stdlib
only, no optuna / torch import, so it runs under the system python3 (3.6) as
well as the venv's 3.12.

Rounds are grouped by training loss (soft_hist vs wasserstein_1d), because an
Adam trial's score is that loss's best VALIDATION loss and the two losses are
NOT comparable with each other. Within a group the trust region / grad mode
variants ARE comparable, so they are ranked together.

Usage:
    python best_adam_runs.py                 # best run per loss
    python best_adam_runs.py --top 5         # top 5 per loss
    python best_adam_runs.py --root <dir>    # a different scan root
    python best_adam_runs.py --json          # machine-readable
"""

import argparse
import json
from pathlib import Path

DEFAULT_ROOT = Path("/pscratch/sd/a/aelabd/parnassus/src/runs/gebo_adam_scans")
LOSSES = ("soft_hist", "wasserstein_1d")


def collect(root):
    """One record per round whose Adam stage finished (wrote a summary)."""
    runs = []
    for path in sorted(root.glob("*/round_*/gebo/adam/gebo_summary.json")):
        round_dir = path.parents[2]          # .../<scan>/round_N
        scan = round_dir.parent.name
        loss = next((l for l in LOSSES if scan.startswith(l)), None)
        if loss is None:
            continue
        try:
            with open(path) as f:
                adam = json.load(f)
            best_loss = float(adam["best_loss"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue                          # partial/corrupt write -- skip
        if best_loss != best_loss:            # NaN
            continue

        # GEBO's own best_loss is on a different scale (full-dataset training
        # objective, not a val loss), so it is context only, never the ranking key.
        gebo_best = None
        try:
            with open(round_dir / "gebo" / "gebo_summary.json") as f:
                gebo_best = float(json.load(f)["best_loss"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            pass

        initial = adam.get("initial_loss")
        runs.append({
            "loss": loss,
            "scan": scan,
            "round": round_dir.name,
            "val_loss": best_loss,
            "initial_loss": initial,
            # <0 means Adam actually improved on its own first epoch; 0.0 means
            # the fit never beat epoch 0 and early stopping ended it.
            "delta": None if initial is None else best_loss - float(initial),
            "best_epoch": adam.get("best_epoch"),
            "epochs_run": adam.get("n_total_points"),
            "gebo_best_loss": gebo_best,
            "config": str(round_dir / "gebo" / "adam" / "best_config.yaml"),
            "round_dir": str(round_dir),
        })
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"scan root holding <scan>/round_*/ (default {DEFAULT_ROOT})")
    ap.add_argument("--top", type=int, default=1, help="rows per loss (default 1)")
    ap.add_argument("--json", action="store_true", help="dump the ranked records as JSON")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"--root {args.root} is not a directory")

    runs = collect(args.root)
    if not runs:
        raise SystemExit(f"no completed Adam runs found under {args.root}")

    ranked = {
        loss: sorted((r for r in runs if r["loss"] == loss), key=lambda r: r["val_loss"])
        for loss in LOSSES
    }

    if args.json:
        print(json.dumps({l: rs[:args.top] for l, rs in ranked.items()}, indent=2))
        return

    for loss in LOSSES:
        rows = ranked[loss][:args.top]
        print(f"\n=== {loss} — best {len(rows)} of {len(ranked[loss])} completed Adam runs ===")
        if not rows:
            print("  (none)")
            continue
        print(f"  {'val_loss':>11s} {'delta':>10s} {'best_ep':>7s} {'epochs':>6s} "
              f"{'gebo_best':>10s}  run")
        for r in rows:
            delta = "   n/a" if r["delta"] is None else f"{r['delta']:+.2e}"
            gebo = "  n/a" if r["gebo_best_loss"] is None else f"{r['gebo_best_loss']:.3e}"
            stall = "  <- no gain over epoch 0" if r["delta"] == 0.0 else ""
            print(f"  {r['val_loss']:11.5e} {delta:>10s} {r['best_epoch']:7} "
                  f"{r['epochs_run']:6} {gebo:>10s}  {r['scan']}/{r['round']}{stall}")
        print(f"  best config -> {rows[0]['config']}")


if __name__ == "__main__":
    main()
