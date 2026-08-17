r"""Parameter convergence of one Optuna trial: fitted value vs epoch, with the truth.

One PDF page per parameter block. Solid line = fitted (physical) value, x=0 is the
initialisation from ``materialized_config.yaml`` and x=k+1 the snapshot after epoch k;
dashed line of the same colour = truth (partial generation YAML over the card defaults).

    python -m parnassus.torch_delphes.plotting_scripts.plot_parameter_regression \\
        --workspace doc/figure_pseudodata_all \\
        --truth-config src/parnassus/torch_delphes/param_configs/param_config_all.yaml
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

plt.style.use(hep.style.ATLAS)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "param_configs" / "cms_target_default.yaml"

# Parameter i of every block uses COLORS[i] (Petroff 6-colour palette).
COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]


def indexed(name, n):
    """``name[0]`` ... ``name[n-1]``."""
    return [f"{name}[{i}]" for i in range(n)]


# One page per block, in this order. Only blocks with at least one trainable key are drawn.
BLOCKS = {
    "ChargedHadronTrackingEfficiency": indexed("ChargedHadronTrackingEfficiency.eff_logits", 4),
    "ElectronTrackingEfficiency": indexed("ElectronTrackingEfficiency.eff_logits", 6),
    "MuonTrackingEfficiency (efficiency)": indexed("MuonTrackingEfficiency.eff_logits", 6),
    "MuonTrackingEfficiency (rate)": indexed("MuonTrackingEfficiency.rate_raw", 2),
    **{
        f"{module}MomentumSmearing ({tensor})": indexed(
            f"{module}MomentumSmearing.resolution_module.{tensor}", 3
        )
        for module in ("ChargedHadron", "Electron", "Muon")
        for tensor in ("a_raw", "b_raw", "scale_raw")
    },
    "ECal scale": indexed("ECal.scale_module.scale_raw", 3),
    "ECal resolution (common)": [
        "ECal.resolution_func.common_c_E",
        "ECal.resolution_func.common_c_S",
        "ECal.resolution_func.common_c_N",
    ],
    "ECal resolution (barrel/endcap)": [
        "ECal.resolution_func.barrel_a",
        "ECal.resolution_func.barrel_b",
        "ECal.resolution_func.endcap_a",
        "ECal.resolution_func.endcap_b",
    ],
    "ECal resolution (forward)": [
        "ECal.resolution_func.forward_c_E",
        "ECal.resolution_func.forward_c_S",
    ],
    "HCal scale": indexed("HCal.scale_module.scale_raw", 2),
    "HCal resolution": [
        "HCal.resolution_func.central_c_E",
        "HCal.resolution_func.central_c_S",
        "HCal.resolution_func.forward_c_E",
        "HCal.resolution_func.forward_c_S",
    ],
    "HadronFractions": [
        "HadronFractions.chad_logit",
        "HadronFractions.k0s_logit",
        "HadronFractions.lambda_logit",
        "HadronFractions.photon_logit",
        "HadronFractions.k0l_logit",
    ],
}


def load_values(path):
    """Flat ``{key: {value: ...}}`` YAML -> ``{key: value}``."""
    return {k: v["value"] for k, v in yaml.safe_load(open(path)).items()}


def main():
    """Plot one trial's parameter trajectories to ``<workspace>/plots/params_reg.pdf``."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workspace", required=True, type=Path, help="dir containing round_<trial>/")
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--truth-config", required=True, type=Path, help="generation (truth) YAML")
    args = ap.parse_args()

    trial_dir = args.workspace / f"round_{args.trial}"
    run = json.load(open(trial_dir / "history.json"))
    init = load_values(trial_dir / "materialized_config.yaml")
    truth = {**load_values(DEFAULT_CONFIG), **load_values(args.truth_config)}
    trainable = set(run["metadata"]["trainable_params"])

    snapshots = [epoch["parameters"] for epoch in run["history"].values()]
    curves = {k: [init[k]] + [snap[k] for snap in snapshots] for k in init}
    epochs = range(len(snapshots) + 1)
    best_epoch = run["best_result"]["step"] + 1  # min val loss (early-stopping checkpoint)

    output = args.workspace / "plots" / "params_reg.pdf"
    output.parent.mkdir(exist_ok=True)
    with PdfPages(output) as pdf:
        for title, keys in BLOCKS.items():
            if not trainable.intersection(keys):
                continue
            fig, ax = plt.subplots()
            fig.subplots_adjust(left=0.16, right=0.95, bottom=0.14, top=0.88)  # fixed axes box
            for key, color in zip(keys, COLORS, strict=False):
                ax.plot(epochs, curves[key], color=color, label=key.split(".")[-1])
                ax.axhline(truth[key], color=color, linestyle="--")
            ax.axvline(best_epoch, color="black", linestyle=":", label="Best epoch")
            ax.set_xlim(0, epochs[-1])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi + 0.4 * (hi - lo))  # headroom for the legend
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Parameter value")
            ax.set_title(title, loc="left", fontsize="x-large")
            handles, _ = ax.get_legend_handles_labels()
            handles.append(Line2D([], [], color="gray", linestyle="--", label="Truth"))
            ax.legend(
                handles=handles, loc="upper right", ncol=2,
                frameon=True, framealpha=1.0, edgecolor="none",
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
