"""Generate figures for the Differentiable TorchDelphes JINST-style paper.

This script is a small, self-contained matplotlib driver that reads a
training-history JSON file (written by
``tune_cms_fullsim.py --history-path``) plus the committed Pythia-
generated pseudodata ROOT file, and writes a set of PDF figures under
``doc/figures/`` for inclusion in the paper.

The intended figures are:

- ``loss_trajectory.pdf`` : loss vs Adam step on log-y scale.
- ``param_drift_scales.pdf`` : chad-pT, ECal, HCal scale trajectories
  vs step, with the known ground-truth values as horizontal dashed
  lines.
- ``param_drift_other.pdf`` : trajectories of the non-scale
  perturbed parameters (resolution barrel ``a``, efficiency barrel
  low-pT logit, K0S hadron fraction) with ground-truth lines.
- ``observable_pt.pdf`` : PF pT histogram for target (full-sim) and
  trainee at init and after training.
- ``observable_eta.pdf`` : same for pseudorapidity.
- ``observable_multiplicity.pdf`` : per-event PF multiplicity.

The script is driven entirely from the training history dict: it re-
runs the trainee card at init and at the final parameter values to
recompute the histograms, so nothing is stored in the history itself
beyond the loss trajectory and the per-parameter snapshots.

Usage
-----
.. code-block:: shell

    uv run python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results \
        --history doc/fit_results/all66_history.json \
        --root-file src/parnassus/tests/benchmark_data/cms_pseudodata.root \
        --output-dir doc/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend for CI / headless runs

import matplotlib.pyplot as plt
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

from .data import (
    load_cms_flow_root,
    load_pflow_targets,
    load_pflow_targets_from_tensor,
    load_truth_events,
    restore_event_format,
)

# Mild styling so the figures are legible in both light and dark themes.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
})


def plot_loss(history: dict, output_path: Path) -> None:
    """Plot the loss trajectory on a log-y axis."""
    steps = history["step"]
    loss = history["loss"]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(steps, loss, color="tab:blue", label="training loss")
    ax.set_xlabel("Adam step")
    ax.set_ylabel("soft-histogram MSE loss")
    ax.set_title("Loss trajectory")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_param_drift(
    history: dict,
    param_groups: dict[str, list[tuple[str, float]]],
    output_path: Path,
    title: str,
) -> None:
    """Plot per-parameter trajectories with horizontal ground-truth lines.

    ``param_groups`` maps a group label to a list of
    ``(history_key, target_value)`` pairs. One matplotlib axis is used
    per group (stacked vertically).
    """
    if not history.get("parameters"):
        raise ValueError("history dict has no 'parameters' snapshots")
    snapshots = history["parameters"]
    steps = history["step"]

    n_groups = len(param_groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(6.0, 2.6 * n_groups), sharex=True, squeeze=False)
    for ax, (group_label, members) in zip(axes[:, 0], param_groups.items(), strict=True):
        for key, target in members:
            if key not in snapshots[0]:
                continue
            trajectory = [snap[key] for snap in snapshots]
            (line,) = ax.plot(steps, trajectory, label=key)
            ax.axhline(target, color=line.get_color(), linestyle="--", alpha=0.4)
        ax.set_ylabel(group_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    axes[-1, 0].set_xlabel("Adam step")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_top_param_drift(
    history: dict,
    output_path: Path,
    n_top: int = 5,
    eps_denom: float = 1e-3,
    title: str = "Top parameters by relative drift",
) -> list[tuple[str, float]]:
    """Plot the top-``n_top`` parameters ranked by relative drift.

    Relative drift for a key with snapshot trajectory ``v[0..T]`` is
    ``|v[-1] - v[0]| / max(|v[0]|, eps_denom)``. The ``eps_denom`` floor
    prevents parameters whose initial value is essentially zero (e.g. the
    charged-hadron ECal fraction at ``sigmoid(logit(1e-6)) ~ 1e-6``) from
    dominating the ranking with a meaningless ratio.

    The dashed horizontal line for each trajectory is the **initial** value
    (not a ground-truth target), since this plot is intended for training
    runs against real data where no per-parameter target exists.

    Returns
    -------
    list[tuple[str, float]]
        The ``(key, rel_change)`` pairs that were plotted, in descending
        order of relative change. Useful for logging.
    """
    if not history.get("parameters"):
        raise ValueError("history dict has no 'parameters' snapshots")
    snapshots = history["parameters"]
    steps = history["step"]
    if len(snapshots) < 2:
        raise ValueError("Need at least 2 parameter snapshots to compute drift")

    initial = snapshots[0]
    final = snapshots[-1]

    rel_changes: list[tuple[str, float]] = []
    for key, v0 in initial.items():
        if key not in final:
            continue
        v1 = final[key]
        denom = max(abs(v0), eps_denom)
        rel_changes.append((key, abs(v1 - v0) / denom))

    rel_changes.sort(key=lambda kv: kv[1], reverse=True)
    top = rel_changes[:n_top]

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for key, rel in top:
        trajectory = [snap[key] for snap in snapshots]
        v0 = trajectory[0]
        (line,) = ax.plot(steps, trajectory, label=f"{key}  (Δ/|init|={rel:.2g})")
        ax.axhline(v0, color=line.get_color(), linestyle="--", alpha=0.4)
    ax.set_xlabel("Adam step")
    ax.set_ylabel("post-transform parameter value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)

    ax.set_yscale("log")
    fig.savefig(output_path.with_stem(output_path.stem + "_logy"))
    plt.close(fig)
    return top


# ---------------------------------------------------------------------------
# Observable histograms: target vs trainee-init vs trainee-final
# ---------------------------------------------------------------------------


def _set_trainee_from_snapshot(card: CMSEnergyFlowDefault, snapshot: dict[str, float]) -> None:
    """Restore a trainee card's learnable parameters from a snapshot.

    The snapshot stores *post-transform* values (e.g. the actual scale
    = 1 + 0.3*tanh(raw), or the efficiency probability after sigmoid),
    so we invert the transform to recover the raw parameter. This is
    the exact reverse of the ``_snapshot`` helper in
    :mod:`tune_cms_fullsim`.
    """
    with torch.no_grad():
        for name, p in card.named_parameters():
            # Per-component keys like "foo.scale_raw[2]"; rebuild vector.
            keys = [k for k in snapshot if k.startswith(name)]
            if not keys:
                continue
            if len(keys) == 1 and not keys[0].endswith("]"):
                vals = torch.tensor([snapshot[keys[0]]], dtype=p.dtype)
            else:
                vals = torch.zeros(p.numel(), dtype=p.dtype)
                for k in keys:
                    i = int(k[k.rfind("[") + 1 : k.rfind("]")])
                    vals[i] = snapshot[k]
            # Invert the relevant transform.
            if name.endswith(".scale_raw"):
                y = vals.clamp(0.7 + 1e-6, 1.3 - 1e-6)
                raw = torch.atanh((y - 1.0) / 0.3)
            elif name.endswith((".eff_logits", "_logit")):
                y = vals.clamp(1e-6, 1.0 - 1e-6)
                raw = torch.log(y / (1.0 - y))
            elif name.endswith((".rate_raw", ".a_raw", ".b_raw")) or name.startswith((
                "ECal.resolution_func",
                "HCal.resolution_func",
            )):
                raw = torch.log(torch.expm1(vals.clamp(min=1e-12)))
            else:
                raw = vals
            p.copy_(raw.reshape(p.shape).to(p.dtype))


def _trainee_observables(card: CMSEnergyFlowDefault, truth_tensor: torch.Tensor) -> dict:
    """Run the trainee card on padded truth events -> predicted observable dict.

    Mirrors the fit loop in :mod:`tune_cms_fullsim.training`: drop the padded
    truth particles, run the card (which expects a flat
    ``(n_particles, n_features)`` tensor), regroup the flat ``EFlowObject``
    output back into per-event format, then extract the observable dict. The
    caller seeds the RNG before calling, since the card's momentum smearing /
    Gumbel-ST efficiency is stochastic.
    """
    mask = torch.any(truth_tensor != 0, dim=-1)
    with torch.no_grad():
        out = card(truth_tensor[mask])
    eflow_restored = restore_event_format(out["EFlowObject"], mask)
    return load_pflow_targets_from_tensor(eflow_restored)


def _obs_values(obs: dict, key: str) -> torch.Tensor:
    """Flatten an observable to 1-D for histogramming, dropping padding/ghosts.

    Per-particle observables are 2-D ``(n_events, max_n_objects)`` zero-padded;
    the same ``pt != 0`` cut the loss uses removes padding and efficiency-killed
    ghost slots (``eta == 0`` and ``log_pt == 0`` are valid values, so they can
    not be masked per-observable). Per-event observables (1-D) pass through.
    """
    v = obs[key]
    if v.ndim >= 2:
        return v[obs["pt"] != 0]
    return v.reshape(-1)


def _density_histogram(values: torch.Tensor, edges: np.ndarray) -> np.ndarray:
    """Normalised histogram counts (density, summing to 1).

    Returns
    -------
    numpy.ndarray
        Non-negative array of length ``len(edges) - 1`` summing to 1
        (or to 0 when ``values`` is empty).
    """
    counts, _ = np.histogram(values.detach().cpu().numpy(), bins=edges)
    total = counts.sum()
    if total == 0:
        return counts.astype(np.float64)
    return counts.astype(np.float64) / total


def plot_observable(
    target_vals: torch.Tensor,
    init_vals: torch.Tensor,
    final_vals: torch.Tensor,
    edges: np.ndarray,
    xlabel: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    """Overlay target / trainee-init / trainee-final on one axis."""
    h_tgt = _density_histogram(target_vals, edges)
    h_init = _density_histogram(init_vals, edges)
    h_final = _density_histogram(final_vals, edges)
    centers = 0.5 * (edges[1:] + edges[:-1])

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.step(centers, h_tgt, where="mid", color="black", label="target (full sim)")
    ax.step(centers, h_init, where="mid", color="tab:red", label="trainee, initial")
    ax.step(centers, h_final, where="mid", color="tab:blue", label="trainee, fitted")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalised density")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main: load history, build figures
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument(
        "--root-file",
        type=Path,
        default=Path("src/parnassus/tests/benchmark_data/cms_pseudodata.root"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("doc/figures"))
    parser.add_argument("--n-events-for-plots", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.history.open() as f:
        history = json.load(f)

    print(f"Writing figures to {args.output_dir}")

    # ----- 1. Loss trajectory -----
    plot_loss(history, args.output_dir / "loss_trajectory.pdf")
    print("  wrote loss_trajectory.pdf")

    # ----- 2. Scale parameter drift -----
    # The committed pseudodata has uniform targets across eta regions
    # for the three scale types (chad pT, ECal E, HCal E).
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
        args.output_dir / "param_drift_scales.pdf",
        title="Scale-parameter drift during Adam fit",
    )
    print("  wrote param_drift_scales.pdf")

    # ----- 3. Other perturbed parameters -----
    # Default values: chad a barrel 0.06, chad barrel low-pt eff 0.70,
    # K0S ECal fraction 0.30.
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
        args.output_dir / "param_drift_other.pdf",
        title="Resolution / efficiency / fraction drift during Adam fit",
    )
    print("  wrote param_drift_other.pdf")

    # ----- 4. Observable histograms (target vs init vs final) -----
    arrays = load_cms_flow_root(args.root_file, n_events=args.n_events_for_plots)
    truth_tensor = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True)
    pred_init = _trainee_observables(trainee, truth_tensor)

    # Restore the *final* parameter snapshot and re-run.
    _set_trainee_from_snapshot(trainee, history["parameters"][-1])
    torch.manual_seed(args.seed)
    pred_final = _trainee_observables(trainee, truth_tensor)

    plot_observable(
        _obs_values(target, "pt"),
        _obs_values(pred_init, "pt"),
        _obs_values(pred_final, "pt"),
        edges=np.linspace(0.0, 100.0, 51),
        xlabel=r"PF object $p_\mathrm{T}$ [GeV]",
        output_path=args.output_dir / "observable_pt.pdf",
        log_y=True,
    )
    print("  wrote observable_pt.pdf")

    plot_observable(
        _obs_values(target, "eta"),
        _obs_values(pred_init, "eta"),
        _obs_values(pred_final, "eta"),
        edges=np.linspace(-5.0, 5.0, 51),
        xlabel=r"PF object $\eta$",
        output_path=args.output_dir / "observable_eta.pdf",
    )
    print("  wrote observable_eta.pdf")

    # plot_observable(
    #     _obs_values(target, "multiplicity"),
    #     _obs_values(pred_init, "multiplicity"),
    #     _obs_values(pred_final, "multiplicity"),
    #     edges=np.linspace(0.0, 600.0, 61),
    #     xlabel=r"PF objects per event",
    #     output_path=args.output_dir / "observable_multiplicity.pdf",
    # )
    # print("  wrote observable_multiplicity.pdf")

    print(f"Done. {len(list(args.output_dir.glob('*.pdf')))} figures in {args.output_dir}.")


if __name__ == "__main__":
    main()
