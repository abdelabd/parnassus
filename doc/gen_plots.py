#!/usr/bin/env python3
"""Generate the six manuscript figures used by doc/differentiable_delphes.tex.

This is a thin wrapper around ``parnassus.torch_delphes.plot_fit_results``
with defaults matching the small-data tuning workflow in ``doc/README.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.generate_pseudodata import (
    TARGET_CHAD_EFF_BARREL_LOWPT,
    TARGET_CHAD_RES_A_BARREL_FACTOR,
    TARGET_CHAD_SCALE,
    TARGET_ECAL_SCALE,
    TARGET_HCAL_SCALE,
    TARGET_K0S_ECAL_FRAC,
)
from parnassus.torch_delphes.plot_fit_results import (
    _set_trainee_from_snapshot,
    plot_loss,
    plot_observable,
    plot_param_drift,
)
from parnassus.torch_delphes.tune_cms_fullsim import (
    load_cms_flow_root,
    pflow_target_observables,
    trainee_observables,
    truth_to_particle_tensor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("./fit_results/all66_smalldata_history.json"),
        help="Training-history JSON from tune_cms_fullsim.py",
    )
    parser.add_argument(
        "--root-file",
        type=Path,
        default=Path("../src/parnassus/tests/benchmark_data/cms_pseudodata.root"),
        help="Pseudo/full-data ROOT file used for target observables",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./figures"),
        help="Directory for output PDF figures",
    )
    parser.add_argument(
        "--n-events-for-plots",
        type=int,
        default=400,
        help="How many events to use when recomputing observable overlays",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-only",
        action="store_true",
        help="Only generate loss/parameter drift plots (skip observable overlays).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.history.exists():
        raise FileNotFoundError(f"History file not found: {args.history}")

    with args.history.open() as f:
        history = json.load(f)

    print(f"[gen_plots] Writing figures to {args.output_dir}")

    # 1) Loss trajectory
    plot_loss(history, args.output_dir / "loss_trajectory.png")
    print("  wrote loss_trajectory.png")

    # 2) Scale parameter drift
    scale_members: dict[str, list[tuple[str, float]]] = {
        "charged-hadron pT scale": [
            (f"ChargedHadronMomentumSmearing.resolution_module.scale_raw[{i}]", TARGET_CHAD_SCALE)
            for i in range(3)
        ],
        "ECal energy scale": [
            (f"ECal.scale_module.scale_raw[{i}]", TARGET_ECAL_SCALE) for i in range(3)
        ],
        "HCal energy scale": [
            (f"HCal.scale_module.scale_raw[{i}]", TARGET_HCAL_SCALE) for i in range(2)
        ],
    }
    plot_param_drift(
        history,
        scale_members,
        args.output_dir / "param_drift_scales.png",
        title="Scale-parameter drift during Adam fit",
    )
    print("  wrote param_drift_scales.png")

    # 3) Other perturbed parameters
    other_members: dict[str, list[tuple[str, float]]] = {
        "chad res. a (barrel)": [
            (
                "ChargedHadronMomentumSmearing.resolution_module.a_raw[0]",
                0.06 * TARGET_CHAD_RES_A_BARREL_FACTOR,
            ),
        ],
        "chad eff. (barrel, low-pT)": [
            (
                "ChargedHadronTrackingEfficiency.eff_logits[0]",
                TARGET_CHAD_EFF_BARREL_LOWPT,
            ),
        ],
        "K0-short ECal fraction": [
            ("HadronFractions.k0s_logit", TARGET_K0S_ECAL_FRAC),
        ],
    }
    plot_param_drift(
        history,
        other_members,
        args.output_dir / "param_drift_other.png",
        title="Resolution / efficiency / fraction drift during Adam fit",
    )
    print("  wrote param_drift_other.png")

    if args.history_only:
        print("[gen_plots] --history-only requested; skipped observable plots.")
        return

    if not args.root_file.exists():
        raise FileNotFoundError(
            f"ROOT file not found: {args.root_file}. "
            "Use --history-only or pass a valid --root-file."
        )

    # 4-6) Observable overlays: target vs trainee init vs trainee final
    arrays = load_cms_flow_root(args.root_file, n_events=args.n_events_for_plots)
    available_events = len(arrays["truth_pt"])
    n_events = min(args.n_events_for_plots, available_events)
    if n_events < args.n_events_for_plots:
        print(
            "[gen_plots] Requested "
            f"--n-events-for-plots={args.n_events_for_plots}, "
            f"but ROOT file has {available_events}; using {n_events}."
        )

    truth_tensor = truth_to_particle_tensor(arrays, n_events=n_events)
    target = pflow_target_observables(arrays, n_events=n_events)

    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True)
    with torch.no_grad():
        out_init = trainee(truth_tensor)
    pred_init = trainee_observables(out_init, n_events=n_events)

    _set_trainee_from_snapshot(trainee, history["parameters"][-1])
    torch.manual_seed(args.seed)
    with torch.no_grad():
        out_final = trainee(truth_tensor)
    pred_final = trainee_observables(out_final, n_events=n_events)

    plot_observable(
        target["pt"],
        pred_init["pt"],
        pred_final["pt"],
        edges=np.linspace(0.0, 100.0, 51),
        xlabel=r"PF object $p_\mathrm{T}$ [GeV]",
        output_path=args.output_dir / "observable_pt.png",
        log_y=True,
    )
    print("  wrote observable_pt.png")

    plot_observable(
        target["eta"],
        pred_init["eta"],
        pred_final["eta"],
        edges=np.linspace(-5.0, 5.0, 51),
        xlabel=r"PF object $\eta$",
        output_path=args.output_dir / "observable_eta.png",
    )
    print("  wrote observable_eta.png")

    plot_observable(
        target["phi"],
        pred_init["phi"],
        pred_final["phi"],
        edges=np.linspace(-np.pi, np.pi, 51),
        xlabel=r"PF object $\phi",
        output_path=args.output_dir / "observable_phi.png",
    )
    print("  wrote observable_phi.png")

    plot_observable(
        target["E"],
        pred_init["E"],
        pred_final["E"],
        edges=np.linspace(0.0, 400.0, 41),
        xlabel=r"PF object $E$ [GeV]",
        output_path=args.output_dir / "observable_E.png",
    )
    print("  wrote observable_E.png")

    plot_observable(
        target["log_pt"],
        pred_init["log_pt"],
        pred_final["log_pt"],
        edges=np.linspace(-1.0, 6.0, 41),
        xlabel=r"PF object $\log p_T$ [GeV]",
        output_path=args.output_dir / "observable_log_pt.png",
    )
    print("  wrote observable_log_pt.png")

    # plot_observable(
    #     target["multiplicity"],
    #     pred_init["multiplicity"],
    #     pred_final["multiplicity"],
    #     edges=np.linspace(0.0, 600.0, 61),
    #     xlabel=r"PF objects per event",
    #     output_path=args.output_dir / "observable_multiplicity.png",
    # )
    # print("  wrote observable_multiplicity.png")

    n_png = len(list(args.output_dir.glob("*.png")))
    print(f"[gen_plots] Done. Found {n_png} PNG(s) in {args.output_dir}.")


if __name__ == "__main__":
    main()
