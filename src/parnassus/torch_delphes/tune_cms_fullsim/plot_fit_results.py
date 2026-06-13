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

With ``--debug``, the script additionally writes per-module
intermediate-output overlays (``intermediate/<ModuleName>/<Var>.pdf``)
for every entry in the same debug branch list used by
:mod:`parnassus.torch_delphes.validation.validate_torch_delphes`
(``ParticleAfterProp``, ``ChargedHadronEfficiency``, ``ECalTower``, …),
so the trainee's per-module response can be inspected against the
target the same way TorchDelphes is inspected against C++ Delphes.
This requires the input ROOT file to have been produced with
``generate_pseudodata.py --debug``.

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

    # With per-module intermediate-output overlays (ROOT file must have been
    # generated with `generate_pseudodata.py --debug`):
    uv run python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results \
        --history doc/fit_results/all66_history.json \
        --root-file src/parnassus/tests/benchmark_data/cms_pseudodata_debug.root \
        --output-dir doc/figures \
        --debug
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
import uproot
from PIL import Image

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .data import (
    load_cms_flow_root,
    load_pflow_targets,
    load_pflow_targets_from_tensor,
    load_truth_events,
    restore_event_format,
)
from .debug import (
    INTERMEDIATE_BRANCHES,
    debug_branch_name,
    extract_variable,
    filter_valid_rows,
)
from .training import PLOT_FRACTION, TRAIN_FRACTION, contiguous_event_partitions
from parnassus.torch_delphes.plotting import (
    combined_vars_for,
    plot_comparison_with_ratio,
    stitch_pngs,
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


def _load_history(path: Path) -> dict:
    """Load a training-history JSON and normalize it for the plot helpers.

    Accepts ONLY the current nested schema written by
    ``tune_cms_fullsim`` (``{"metadata", "history", "best_result"}``).
    Old flat-format files (top-level ``"loss"``/``"step"``/``"parameters"``)
    are rejected with a clear error because they lack the per-epoch
    validation loss needed to pick the best (min-val-loss) epoch.

    Returns a dict exposing the parallel-list keys the existing plot
    helpers consume — ``"step"``, ``"loss"`` (train loss), ``"parameters"``
    (per-epoch snapshots) — plus ``"val_loss"``, ``"metadata"`` and
    ``"best_result"``. Epochs are returned in ascending ``step`` order.
    """
    with path.open() as f:
        raw = json.load(f)

    if "history" not in raw or "metadata" not in raw:
        raise SystemExit(
            f"{path} is in the old flat history format. Re-run training with "
            "the updated --history-path to produce the new "
            "{metadata, history, best_result} schema (the old format has no "
            "per-epoch validation loss, so the best epoch cannot be selected)."
        )

    entries = sorted(raw["history"].values(), key=lambda e: e["step"])
    return {
        "step": [e["step"] for e in entries],
        "loss": [e["train_loss"] for e in entries],
        "val_loss": [e.get("val_loss") for e in entries],
        "parameters": [e.get("parameters", {}) for e in entries],
        "metadata": raw["metadata"],
        "best_result": raw.get("best_result", {}),
    }


def plot_loss(history: dict, output_path: Path) -> None:
    """Plot the train/val loss trajectory on a log-y axis.

    Marks the best (min-validation-loss) epoch with a vertical line so it
    is visually clear why it can differ from the last epoch (early stopping
    keeps training for ``patience`` extra steps after the best).
    """
    steps = history["step"]
    loss = history["loss"]
    val_loss = history.get("val_loss") or []
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.semilogy(steps, loss, color="tab:blue", label="training loss")
    if any(v is not None for v in val_loss):
        ax.semilogy(steps, val_loss, color="tab:orange", label="validation loss")
    best_step = history.get("best_result", {}).get("step")
    if best_step is not None:
        ax.axvline(
            best_step,
            color="tab:green",
            linestyle="--",
            alpha=0.6,
            label=f"best (min val) @ step {best_step}",
        )
    ax.set_xlabel("Adam step")
    ax.set_ylabel("loss")
    ax.set_title("Loss trajectory")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_param_drift(
    history: dict,
    param_groups: dict[str, list[tuple[str, float | None]]],
    output_path: Path,
    title: str,
) -> None:
    """Plot per-parameter trajectories with horizontal ground-truth lines.

    ``param_groups`` maps a group label to a list of
    ``(history_key, target_value_or_none)`` pairs. If ``target_value`` is
    ``None``, no dashed reference line is drawn for that parameter. One
    matplotlib axis is used per group (stacked vertically).
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
            if target is not None:
                ax.axhline(
                    target,
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.4,
                    label=f"{key} target",
                )
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
# Full parameter atlas (all 66) -- dict-driven grouped PNG structure
# ---------------------------------------------------------------------------


AxisSpec = dict[str, object]
PlotSpec = dict[str, list[AxisSpec]]


PARAMETER_PLOT_LAYOUT: dict[str, dict[str, object]] = {
    "TrackingEfficiency": {
        "plots": {
            "ChargedHadronTrackingEfficiency.png": [
                {
                    "title": "ChargedHadronTrackingEfficiency.eff_logits[0..3]",
                    "keys": [
                        f"ChargedHadronTrackingEfficiency.eff_logits[{i}]" for i in range(4)
                    ],
                    "ylabel": "efficiency",
                }
            ],
            "ElectronTrackingEfficiency.png": [
                {
                    "title": "ElectronTrackingEfficiency.eff_logits[0..5]",
                    "keys": [
                        f"ElectronTrackingEfficiency.eff_logits[{i}]" for i in range(6)
                    ],
                    "ylabel": "efficiency",
                }
            ],
            "MuonTrackingEfficiency.png": [
                {
                    "title": "MuonTrackingEfficiency.eff_logits[0..5]",
                    "keys": [
                        f"MuonTrackingEfficiency.eff_logits[{i}]" for i in range(6)
                    ],
                    "ylabel": "efficiency",
                },
                {
                    "title": "MuonTrackingEfficiency.rate_raw[0..1]",
                    "keys": [
                        "MuonTrackingEfficiency.rate_raw[0]",
                        "MuonTrackingEfficiency.rate_raw[1]",
                    ],
                    "ylabel": "rate",
                },
            ],
        },
        "stacked_pngs": {
            "TrackingEfficiency.png": [
                "ChargedHadronTrackingEfficiency.png",
                "ElectronTrackingEfficiency.png",
                "MuonTrackingEfficiency.png",
            ]
        },
    },
    "MomentumSmearing": {
        "plots": {
            "ChargedHadronMomentumSmearing.png": [
                {
                    "title": "ChargedHadronMomentumSmearing.resolution_module.a_raw[0..2]",
                    "keys": [
                        f"ChargedHadronMomentumSmearing.resolution_module.a_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "a_raw",
                },
                {
                    "title": "ChargedHadronMomentumSmearing.resolution_module.b_raw[0..2]",
                    "keys": [
                        f"ChargedHadronMomentumSmearing.resolution_module.b_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "b_raw",
                },
                {
                    "title": "ChargedHadronMomentumSmearing.resolution_module.scale_raw[0..2]",
                    "keys": [
                        f"ChargedHadronMomentumSmearing.resolution_module.scale_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "scale_raw",
                },
            ],
            "ElectronMomentumSmearing.png": [
                {
                    "title": "ElectronMomentumSmearing.resolution_module.a_raw[0..2]",
                    "keys": [
                        f"ElectronMomentumSmearing.resolution_module.a_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "a_raw",
                },
                {
                    "title": "ElectronMomentumSmearing.resolution_module.b_raw[0..2]",
                    "keys": [
                        f"ElectronMomentumSmearing.resolution_module.b_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "b_raw",
                },
                {
                    "title": "ElectronMomentumSmearing.resolution_module.scale_raw[0..2]",
                    "keys": [
                        f"ElectronMomentumSmearing.resolution_module.scale_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "scale_raw",
                },
            ],
            "MuonMomentumSmearing.png": [
                {
                    "title": "MuonMomentumSmearing.resolution_module.a_raw[0..2]",
                    "keys": [
                        f"MuonMomentumSmearing.resolution_module.a_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "a_raw",
                },
                {
                    "title": "MuonMomentumSmearing.resolution_module.b_raw[0..2]",
                    "keys": [
                        f"MuonMomentumSmearing.resolution_module.b_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "b_raw",
                },
                {
                    "title": "MuonMomentumSmearing.resolution_module.scale_raw[0..2]",
                    "keys": [
                        f"MuonMomentumSmearing.resolution_module.scale_raw[{i}]" for i in range(3)
                    ],
                    "ylabel": "scale_raw",
                },
            ],
        }
    },
    "_root": {
        "plots": {
            "HadronFractions.png": [
                {
                    "title": "HadronFractions logits",
                    "keys": [
                        "HadronFractions.chad_logit",
                        "HadronFractions.k0s_logit",
                        "HadronFractions.lambda_logit",
                    ],
                    "ylabel": "fraction",
                }
            ]
        }
    },
    "ECal": {
        "plots": {
            "ECal_scale.png": [
                {
                    "title": "ECal.scale_module.scale_raw[0..2]",
                    "keys": [f"ECal.scale_module.scale_raw[{i}]" for i in range(3)],
                    "ylabel": "scale_raw",
                }
            ],
            "ECal_resolution.png": [
                {
                    "title": "ECal.resolution_func.common_c_*",
                    "keys": [
                        "ECal.resolution_func.common_c_E",
                        "ECal.resolution_func.common_c_S",
                        "ECal.resolution_func.common_c_N",
                    ],
                    "ylabel": "common",
                },
                {
                    "title": "ECal.resolution_func.barrel_{a,b}",
                    "keys": [
                        "ECal.resolution_func.barrel_a",
                        "ECal.resolution_func.barrel_b",
                    ],
                    "ylabel": "barrel",
                },
                {
                    "title": "ECal.resolution_func.endcap_{a,b}",
                    "keys": [
                        "ECal.resolution_func.endcap_a",
                        "ECal.resolution_func.endcap_b",
                    ],
                    "ylabel": "endcap",
                },
                {
                    "title": "ECal.resolution_func.forward_c_{E,S}",
                    "keys": [
                        "ECal.resolution_func.forward_c_E",
                        "ECal.resolution_func.forward_c_S",
                    ],
                    "ylabel": "forward",
                },
            ],
        }
    },
    "HCal": {
        "plots": {
            "HCal_scale.png": [
                {
                    "title": "HCal.scale_module.scale_raw[0..1]",
                    "keys": [f"HCal.scale_module.scale_raw[{i}]" for i in range(2)],
                    "ylabel": "scale_raw",
                }
            ],
            # HCal has fewer resolution coefficients than ECal in the current
            # card/config. We keep the same top-level naming convention.
            "HCal_resolution.png": [
                {
                    "title": "HCal.resolution_func.central_c_{E,S}",
                    "keys": [
                        "HCal.resolution_func.central_c_E",
                        "HCal.resolution_func.central_c_S",
                    ],
                    "ylabel": "central",
                },
                {
                    "title": "HCal.resolution_func.forward_c_{E,S}",
                    "keys": [
                        "HCal.resolution_func.forward_c_E",
                        "HCal.resolution_func.forward_c_S",
                    ],
                    "ylabel": "forward",
                },
            ],
        }
    },
}


def _stack_pngs_vertical(image_paths: list[Path], output_path: Path) -> bool:
    """Stack existing PNG files vertically by copy-pasting their pixel data."""
    existing = [p for p in image_paths if p.exists()]
    if not existing:
        return False

    imgs = [Image.open(p).convert("RGBA") for p in existing]
    try:
        width = max(img.width for img in imgs)
        height = sum(img.height for img in imgs)
        canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        y = 0
        for img in imgs:
            x = (width - img.width) // 2
            canvas.paste(img, (x, y))
            y += img.height
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.convert("RGB").save(output_path)
        return True
    finally:
        for img in imgs:
            img.close()


def _plot_parameter_panel(
    history: dict,
    truth: dict[str, float],
    axis_specs: list[AxisSpec],
    output_path: Path,
) -> int:
    """Render one PNG panel containing one or more vertically-stacked axes.

    Returns
    -------
    int
        Number of parameter trajectories actually drawn.
    """
    if not history.get("parameters"):
        raise ValueError("history dict has no 'parameters' snapshots")

    snapshots = history["parameters"]
    steps = history["step"]

    n_axes = max(1, len(axis_specs))
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(7.6, 2.7 * n_axes),
        sharex=True,
        squeeze=False,
    )

    n_drawn = 0
    for ax, spec in zip(axes[:, 0], axis_specs, strict=True):
        keys = list(spec.get("keys", []))
        axis_title = str(spec.get("title", ""))
        ylabel = str(spec.get("ylabel", "value"))

        n_axis_lines = 0
        for key in keys:
            if key not in snapshots[0]:
                continue
            traj = [snap.get(key, np.nan) for snap in snapshots]
            line_label = key.split(".")[-1]
            (line,) = ax.plot(steps, traj, label=line_label)
            if key in truth:
                ax.axhline(
                    truth[key],
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.4,
                    label=f"{line_label} target",
                )
            n_axis_lines += 1
            n_drawn += 1

        ax.set_ylabel(ylabel)
        if axis_title:
            ax.set_title(axis_title, fontsize=10, loc="left")
        ax.grid(True, alpha=0.3)
        if n_axis_lines > 0:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "no matching parameters in history", ha="center", va="center")

    axes[-1, 0].set_xlabel("optimizer step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return n_drawn


def plot_all_parameter_groups(
    history: dict,
    truth: dict[str, float],
    output_dir: Path,
) -> tuple[int, int]:
    """Render the dict-defined full parameter atlas (all grouped PNGs).

    Parameters
    ----------
    output_dir : Path
        Target root directory, e.g. ``<fig_dir>_parameters``.

    Returns
    -------
    (int, int)
        ``(n_pngs_written, n_parameter_trajectories_drawn)``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_pngs_written = 0
    n_params_drawn = 0
    for section, section_spec in PARAMETER_PLOT_LAYOUT.items():
        section_dir = output_dir if section == "_root" else (output_dir / section)
        plots: PlotSpec = section_spec.get("plots", {})  # type: ignore[assignment]
        for filename, axis_specs in plots.items():
            out_png = section_dir / filename
            n_params_drawn += _plot_parameter_panel(history, truth, axis_specs, out_png)
            n_pngs_written += 1

        stacked = section_spec.get("stacked_pngs", {})
        for stacked_name, members in stacked.items():
            src = [section_dir / name for name in members]
            if _stack_pngs_vertical(src, section_dir / stacked_name):
                n_pngs_written += 1

    return n_pngs_written, n_params_drawn


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


def plot_observable(
    var: str,
    target_vals: torch.Tensor,
    init_vals: torch.Tensor,
    final_vals: torch.Tensor,
    edges: np.ndarray,
    xlabel: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    """Overlay target / trainee-init / trainee-final with a ratio panel.

    This now uses the same shared helper as the ``intermediate/`` plots, so
    final observable PDFs (``observable_*.pdf``) also include the lower ratio
    axis with ``init/target`` and ``fitted/target``.
    """
    plot_comparison_with_ratio(
        distributions=[
            (target_vals.detach().cpu().numpy(), "target", _TARGET_COLOR),
            (init_vals.detach().cpu().numpy(), "trainee, initial", _INIT_COLOR),
            (final_vals.detach().cpu().numpy(), "trainee, fitted", _FINAL_COLOR),
        ],
        var=var,
        output_path=output_path,
        bins=edges,
        xlabel=xlabel,
        ylabel="Counts",
        ratio_ylabel="ratio / target",
        figsize=(5.5, 4.8),
        legend_loc="best",
        log_y=log_y,
    )


# ---------------------------------------------------------------------------
# Intermediate (per-module) observables  --  enabled by --debug
# ---------------------------------------------------------------------------


def _trainee_intermediate_outputs(
    card: CMSEnergyFlowDefault, truth_tensor: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Run a debug-mode trainee card and return its full per-module output dict.

    The card MUST have been built with ``debug=True`` so the forward pass
    returns ``ParticleAfterProp``, ``ChargedHadronEfficiency``, ``ECalTower``,
    ``Track``, etc., in addition to the final ``EFlowObject``. Each value in
    the returned dict is a flat ``(N_objects_for_module, N_FEATURES)`` tensor
    -- the same layout used by the target side written to ROOT by
    :mod:`parnassus.torch_delphes.generate_pseudodata`. The caller seeds the
    RNG before calling, since the card's momentum smearing / Gumbel-ST
    efficiency is stochastic.
    """
    mask = torch.any(truth_tensor != 0, dim=-1)
    with torch.no_grad():
        out = card(truth_tensor[mask])
    return out


def _load_target_intermediate_values(
    root_file: Path,
    module_name: str,
    var: str,
    n_events: int,
) -> torch.Tensor | None:
    """Read one debug branch from the pseudodata ROOT file as a flat 1-D tensor.

    Returns ``None`` if the requested branch isn't in the file (e.g. the
    pseudodata was generated without ``--debug``, or the module wasn't
    populated for some reason). Flattens the jagged per-event arrays into a
    single 1-D ``torch.Tensor`` ready to feed into
    :func:`plot_observable`.
    """
    import uproot  # local import: keeps the no-debug code path uproot-free

    branch = debug_branch_name(module_name, var)
    with uproot.open(str(root_file)) as f:
        tree = f["event_tree"]
        if branch not in tree:
            return None
        arr = tree[branch].array(  # pyright: ignore[reportAttributeAccessIssue]
            entry_stop=n_events
        )
    # Awkward -> numpy via flatten -> torch. Empty events become zero-length
    # subarrays and disappear under flatten, which is what we want.
    import awkward as ak

    flat = ak.flatten(arr)
    return torch.from_numpy(np.asarray(flat).astype(np.float64))


def _module_values_from_outputs(
    outputs: dict[str, torch.Tensor], module_name: str, var: str
) -> torch.Tensor | None:
    """Extract a 1-D value tensor for ``(module, var)`` from a debug output dict.

    Applies the same ghost-row filter
    (:func:`parnassus.torch_delphes.tune_cms_fullsim.debug.filter_valid_rows`)
    used on the target side at write time, so trainee and target distributions
    are filtered identically before histogramming.
    """
    tensor = outputs.get(module_name)
    if tensor is None or tensor.numel() == 0:
        return None
    tensor = filter_valid_rows(tensor, module_name)
    if tensor.numel() == 0:
        return None
    return extract_variable(tensor, var).detach().cpu()


# Colour scheme for the intermediate-plot overlays. Target matches the yellow
# "C++ Delphes" stepfilled style of validate_torch_delphes; trainee curves
# stay red (initial) / blue (fitted) consistent with the headline observable
# plots above.
_TARGET_COLOR: str = "gold"
_INIT_COLOR: str = "tab:red"
_FINAL_COLOR: str = "tab:blue"


def _intermediate_suptitle(module_name: str, n_events: int) -> str:
    """Two-line title for the per-module ``all.png`` overview.

    Mirrors the short ``_get_suptitle`` body in
    :mod:`validation.validate_torch_delphes`: identify the module then the
    event count.
    """
    pretty_name = "PropagatedParticle" if module_name == "ParticleAfterProp" else module_name
    return f"Output: {pretty_name}\n{n_events} events"


def plot_intermediate_observables(
    root_file: Path,
    trainee_init_outputs: dict[str, torch.Tensor],
    trainee_final_outputs: dict[str, torch.Tensor],
    output_dir: Path,
    n_events: int,
) -> None:
    """Generate target / init / fitted overlays for every intermediate output.

    Visually identical to the per-branch comparison plots produced by
    :mod:`parnassus.torch_delphes.validation.validate_torch_delphes`:

    - Per-variable PNGs (``<output_dir>/<ModuleName>/<Var>.png``) drawn via
      :func:`parnassus.torch_delphes.plotting.plot_comparison_with_ratio` --
      stepfilled yellow target (the reference), step-red trainee-initial and
      step-blue trainee-fitted overlays, with a ratio panel below showing
      ``init/target`` and ``fitted/target``. Counts are NOT normalised;
      log-y / discrete-bar handling is identical to the validation harness.
    - A combined ``<output_dir>/<ModuleName>/all.png`` per module stitching
      the (Eta, Phi, PT, P) / (Eta, Phi, E, ET) / (Eta, Phi, PT, E) panel
      group, picked via
      :func:`parnassus.torch_delphes.plotting.combined_vars_for` (the same
      heuristic used by the validation harness).

    Branches absent from the ROOT file (e.g. file produced without
    ``--debug``) are silently skipped with a summary count at the end.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Writing intermediate plots to {output_dir}")

    n_plots_written = 0
    n_branches_missing = 0

    for module_name, variables in INTERMEDIATE_BRANCHES:
        module_dir = output_dir / module_name
        module_has_any = False
        for var in variables:
            target_vals = _load_target_intermediate_values(
                root_file, module_name, var, n_events=n_events
            )
            if target_vals is None:
                n_branches_missing += 1
                continue

            init_vals = _module_values_from_outputs(
                trainee_init_outputs, module_name, var
            )
            final_vals = _module_values_from_outputs(
                trainee_final_outputs, module_name, var
            )

            # Pool inputs as plain numpy arrays for the shared helper. Empty
            # overlays are passed as length-0 arrays so the legend still
            # carries them and the layout stays consistent across modules.
            target_np = target_vals.numpy() if target_vals is not None else np.empty(0)
            init_np = init_vals.numpy() if init_vals is not None else np.empty(0)
            final_np = final_vals.numpy() if final_vals is not None else np.empty(0)

            if target_np.size == 0 and init_np.size == 0 and final_np.size == 0:
                # No finite data anywhere -- skip the page rather than emit a
                # blank plot.
                continue

            if not module_has_any:
                module_dir.mkdir(parents=True, exist_ok=True)
                module_has_any = True

            plot_comparison_with_ratio(
                distributions=[
                    (target_np, "target", _TARGET_COLOR),
                    (init_np, "trainee, initial", _INIT_COLOR),
                    (final_np, "trainee, fitted", _FINAL_COLOR),
                ],
                var=var,
                output_path=module_dir / f"{var}.png",
                ratio_ylabel="ratio / target",
                # Match the legend placement used by validate_torch_delphes
                # for Eta plots (anchored above the axes) so the per-module
                # `all.png` stitch shows the legend without overlap.
                legend_loc=("upper left" if var != "Eta" else "best"),
            )
            n_plots_written += 1

        # Per-module ``all.png``: stitch the (Eta, Phi, PT/P, E/ET) group
        # picked by the same heuristic validate_torch_delphes uses. Skip
        # cleanly when the module wrote no per-var PNGs (e.g. branches
        # missing from the ROOT file).
        if module_has_any:
            combo = combined_vars_for(variables)
            per_var_pngs = [module_dir / f"{var}.png" for var in combo]
            all_png = module_dir / "all.png"
            wrote = stitch_pngs(
                image_paths=per_var_pngs,
                output_path=all_png,
                title=_intermediate_suptitle(module_name, n_events=n_events),
                title_align="left",
            )
            n_var_pngs = len(list(module_dir.glob("*.png"))) - (1 if wrote else 0)
            extra = f" + all.png ({', '.join(combo)})" if wrote else ""
            print(f"    {module_name}: wrote {n_var_pngs} per-var PNGs{extra}")

    if n_branches_missing > 0:
        print(
            f"  NOTE: {n_branches_missing} debug branch(es) were absent from "
            f"{root_file}. If you didn't intend this, regenerate the ROOT "
            "file with `generate_pseudodata.py --debug`."
        )
    print(f"  Wrote {n_plots_written} intermediate plots in total.")


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
    parser.add_argument(
        "--n-events-for-plots",
        type=int,
        default=-1,
        help=(
            "Maximum number of events to use from the plotting window. The "
            "window is the NEXT 20% block right after the training 70% block. "
            "Use -1 (default) to use the full plotting window."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--truth-config",
        type=Path,
        default=None,
        help=(
            "Optional path to the GENERATION/truth config that made the ROOT "
            "file. When provided, its physical 'value' fields are drawn as the "
            "dashed target-reference lines on parameter plots. Leave unset for "
            "real fullsim data (unknown truth): no target lines are plotted."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "In addition to the headline observable plots, generate target / "
            "trainee-init / trainee-fitted overlays for every intermediate "
            "per-module output (ParticleAfterProp, ChargedHadronEfficiency, "
            "ECalTower, ...). Mirrors the --debug branch list of "
            "validation/validate_torch_delphes.py. Plots land in "
            "<output-dir>/intermediate/<ModuleName>/<Var>.pdf. "
            "REQUIRES the input ROOT file to have been written with "
            "`generate_pseudodata.py --debug` (so the per-module target "
            "branches are present); missing branches are silently skipped."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    history = _load_history(args.history)

    # Ground-truth physical values keyed in the same ``name[i]`` form as the
    # history snapshots. Optional: for real fullsim data the truth is unknown,
    # so we omit dashed target lines entirely.
    truth: dict[str, float] = {}
    if args.truth_config is not None:
        flat_truth_cfg = pc.load_param_config(args.truth_config)
        n_trainable = sum(1 for spec in flat_truth_cfg.values() if spec["trainable"])
        truth = {k: spec["value"] for k, spec in flat_truth_cfg.items()}
    else:
        print(
            "No --truth-config provided: skipping dashed target-reference lines "
            "on parameter plots."
        )

    print(f"Writing figures to {args.output_dir}")

    # ----- 1. Loss trajectory -----
    plot_loss(history, args.output_dir / "loss_trajectory.pdf")
    print("  wrote loss_trajectory.pdf")

    # ----- 2. Parameter values versus optimizer step -----

    param_atlas_dir = Path(args.output_dir) / "parameters"
    param_atlas_dir.mkdir(parents=True, exist_ok=True)
    n_param_pngs, n_param_traces = plot_all_parameter_groups(
        history=history,
        truth=truth,
        output_dir=param_atlas_dir,
    )
    print(
        "  wrote "
        f"{n_param_pngs} grouped parameter PNGs ({n_param_traces} trajectories) "
        f"to {param_atlas_dir}"
    )

    # ----- 3. Final observable histograms (target vs init vs final) -----
    final_obs_output_dir = args.output_dir / "observables" / "final"
    final_obs_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot on the contiguous block immediately after the training block:
    # train=[0, 70%), plot=[70%, 90%), tail=[90%, 100%).
    with uproot.open(str(args.root_file)) as f:
        total_in_file = int(f["event_tree"].num_entries)

    # Respect the same event cap training used (history metadata["n_events"]).
    # ``-1`` means "all events in file".
    meta_n_events = int(history.get("metadata", {}).get("n_events", -1))
    total_considered = total_in_file if meta_n_events < 0 else min(total_in_file, meta_n_events)
    train_end, plot_end = contiguous_event_partitions(total_considered)
    plot_start = train_end
    window_size = max(0, plot_end - plot_start)
    if window_size <= 0:
        raise SystemExit(
            "Plot window is empty after applying the 70/20 split policy. "
            f"total_considered={total_considered}, train_end={train_end}, plot_end={plot_end}."
        )
    n_plot_events = window_size if args.n_events_for_plots < 0 else min(args.n_events_for_plots, window_size)
    if n_plot_events <= 0:
        raise SystemExit("No events selected for plotting. Increase --n-events-for-plots.")

    print(
        "Plot event window: "
        f"train=[0:{train_end}) ({TRAIN_FRACTION:.0%}), "
        f"plot=[{plot_start}:{plot_end}) ({PLOT_FRACTION:.0%}), "
        f"using first {n_plot_events} event(s) from plot window."
    )

    arrays = load_cms_flow_root(
        args.root_file,
        n_events=n_plot_events,
        entry_start=plot_start,
    )
    truth_tensor = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True)
    pred_init = _trainee_observables(trainee, truth_tensor)

    # Restore the *best* (min-validation-loss) parameter snapshot and re-run.
    # Early stopping keeps training past the best epoch, so the last snapshot
    # is over-trained; best_result holds the snapshot we actually want to show.
    best_params = history["best_result"].get("parameters") or (
        history["parameters"][-1] if history.get("parameters") else None
    )
    if not best_params:
        raise SystemExit(
            "History has no parameter snapshots, cannot plot trainee observables. "
            "Re-run training with --history-path (snapshots are enabled there)."
        )
    _set_trainee_from_snapshot(trainee, best_params)
    torch.manual_seed(args.seed)
    pred_final = _trainee_observables(trainee, truth_tensor)

    plot_observable(
        "PT",
        _obs_values(target, "pt"),
        _obs_values(pred_init, "pt"),
        _obs_values(pred_final, "pt"),
        edges=np.linspace(0.0, 100.0, 51),
        xlabel=r"PF object $p_\mathrm{T}$ [GeV]",
        output_path=final_obs_output_dir / "observable_pt.pdf",
        log_y=True,
    )
    print("  wrote observable_pt.pdf")

    plot_observable(
        "Eta",
        _obs_values(target, "eta"),
        _obs_values(pred_init, "eta"),
        _obs_values(pred_final, "eta"),
        edges=np.linspace(-5.0, 5.0, 51),
        xlabel=r"PF object $\eta$",
        output_path=final_obs_output_dir / "observable_eta.pdf",
    )
    print("  wrote observable_eta.pdf")

    plot_observable(
        "ht",
        _obs_values(target, "ht"),
        _obs_values(pred_init, "ht"),
        _obs_values(pred_final, "ht"),
        edges=np.linspace(0.0, 1000.0, 51),
        xlabel=r"PF scalar $H_\mathrm{T}$ [GeV]",
        output_path=final_obs_output_dir / "observable_ht.pdf",
    )
    print("  wrote observable_ht.pdf")

    # log(HT) -- the per-event scalar actually in the loss. Keep these edges in
    # sync with DEFAULT_BIN_EDGES["log_ht"] in config.py.
    plot_observable(
        "log_ht",
        _obs_values(target, "log_ht"),
        _obs_values(pred_init, "log_ht"),
        _obs_values(pred_final, "log_ht"),
        edges=np.linspace(4.5, 7.5, 51),
        xlabel=r"PF scalar $\log\,H_\mathrm{T}$",
        output_path=final_obs_output_dir / "observable_log_ht.pdf",
    )
    print("  wrote observable_log_ht.pdf")

    plot_observable(
        "multiplicity",
        _obs_values(target, "multiplicity"),
        _obs_values(pred_init, "multiplicity"),
        _obs_values(pred_final, "multiplicity"),
        edges=np.linspace(0.0, 300.0, 61),
        xlabel=r"PF objects per event",
        output_path=final_obs_output_dir / "observable_multiplicity.pdf",
    )
    print("  wrote observable_multiplicity.pdf")

    # ----- 4. Optional: per-module intermediate observables (--debug) -----
    # Mirrors the --debug branch list of validate_torch_delphes.py: for every
    # post-module output (ParticleAfterProp, ChargedHadronEfficiency,
    # ECalTower, ...), overlay target / trainee-init / trainee-fitted on the
    # same axes. Requires the ROOT file to have been written with
    # `generate_pseudodata.py --debug` -- otherwise the target branches are
    # missing and each pair of plots is silently skipped (with a summary
    # warning at the end). The trainee plots are obtained by re-running the
    # init and best-fit cards in debug mode, reusing the same RNG seed as the
    # headline EFlowObject plots above so the stochastic noise is consistent.
    if args.debug:
        print("\n  --debug: rendering per-module intermediate-output overlays...")
        torch.manual_seed(args.seed)
        trainee_dbg = CMSEnergyFlowDefault(debug=True, learnable=True)
        init_outputs = _trainee_intermediate_outputs(trainee_dbg, truth_tensor)

        _set_trainee_from_snapshot(trainee_dbg, best_params)
        torch.manual_seed(args.seed)
        final_outputs = _trainee_intermediate_outputs(trainee_dbg, truth_tensor)

        plot_intermediate_observables(
            root_file=args.root_file,
            trainee_init_outputs=init_outputs,
            trainee_final_outputs=final_outputs,
            output_dir=args.output_dir / "observables" / "intermediate",
            n_events=n_plot_events,
        )

    print(f"Done. {len(list(args.output_dir.glob('*.pdf')))} figures in {args.output_dir}.")


if __name__ == "__main__":
    main()
