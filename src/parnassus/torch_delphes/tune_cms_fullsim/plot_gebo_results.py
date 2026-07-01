r"""Generate figures for GEBO (gradient-enhanced Bayesian optimisation) runs.

This is the GEBO counterpart of :mod:`tune_cms_fullsim.plot_fit_results`.
Instead of reading an Adam training-history JSON, it reads the outputs of
:mod:`tune_cms_fullsim.gebo_search`:

- ``gebo_summary.json``  – best parameters + per-iteration loss history
- ``gebo_data.pt``        – all queried (X, Y) points (required for parameter
  drift plots; optional otherwise)

It generates the **same headline observable figures** as the Adam plot script
(target vs trainee-init vs trainee-best), plus GEBO-specific diagnostics:

- ``loss_trajectory.pdf`` — best_loss (and candidate_loss) vs iteration
- ``param_drift_all.pdf`` — every parameter vs query index, with truth lines
- ``observable_pt.pdf`` — PF pT: target / trainee-init / trainee-best
- ``observable_eta.pdf`` — PF η
- ``observable_ht.pdf`` — PF HT
- ``observable_log_ht.pdf`` — PF log(HT)
- ``observable_multiplicity.pdf`` — PF multiplicity per event
- ``PID/`` — per-PID observable overlays
- ``acquisition_history.pdf`` — acq_value vs iteration (if ``gebo_data.pt``)
- ``loss_scatter.pdf`` — all queried points (if ``gebo_data.pt``)

Usage
-----
.. code-block:: shell

    python -m parnassus.torch_delphes.tune_cms_fullsim.plot_gebo_results \
        --summary doc/figures/w1d_pseudodata_botorch/gebo_summary.json \
        --root-file /path/to/data.root \
        --truth-config src/parnassus/torch_delphes/param_configs/cms_target_default.yaml \
        --param-config src/parnassus/torch_delphes/param_configs/optuna_config.yaml \
        --output-dir doc/figures/w1d_pseudodata_botorch
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .config import OBSERVABLES
from .data import (
    load_cms_flow_root,
    load_pflow_targets_from_tensor,
    load_pflow_targets_ragged,
    load_truth_events_ragged,
    restore_event_format,
)
from .dataloader import DelphesDataLoader, DelphesDataSet
from parnassus.torch_delphes.plotting import plot_comparison_with_ratio

# ---- Styling (matches plot_fit_results.py) ----
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

# Colour scheme: matches plot_fit_results.py
_TARGET_COLOR: str = "gold"
_INIT_COLOR: str = "tab:red"
_FINAL_COLOR: str = "tab:blue"

# Default truth config (matches plot_fit_results.py).
_DEFAULT_TRUTH_CONFIG = (
    Path(pc.__file__).resolve().parent / "param_configs" / "cms_target_default.yaml"
)


# =============================================================================
# Data loading
# =============================================================================


def _load_gebo_summary(path: Path) -> dict:
    """Load ``gebo_summary.json``."""
    with path.open() as f:
        return json.load(f)


def _load_gebo_data(path: Path) -> dict | None:
    """Load ``gebo_data.pt`` if it exists; return None otherwise."""
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


# =============================================================================
# Helper: build a card from raw parameter dict
# =============================================================================


def _build_card_from_raw_params(
    raw_params: dict[str, float], device: torch.device
) -> CMSEnergyFlowDefault:
    """Build a card and set its parameters from a raw-parameter dict.

    ``raw_params`` maps ``"param_name[i]"`` → raw (pre-transform) value,
    exactly as stored in ``gebo_summary.json["best_raw_params"]``.
    """
    card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    with torch.no_grad():
        for name, p in card.named_parameters():
            keys = [k for k in raw_params if k.startswith(name)]
            if not keys:
                continue
            if len(keys) == 1 and not keys[0].endswith("]"):
                raw_val = torch.tensor(raw_params[keys[0]], dtype=p.dtype, device=device)
                p.copy_(raw_val.reshape(p.shape))
            else:
                vals = torch.zeros(p.numel(), dtype=p.dtype, device=device)
                for k in keys:
                    i = int(k[k.rfind("[") + 1:k.rfind("]")])
                    vals[i] = raw_params[k]
                p.copy_(vals.reshape(p.shape))
    return card


# =============================================================================
# Helper: build a card from a physical param config
# =============================================================================


def _set_trainee_from_snapshot(
    card: CMSEnergyFlowDefault, snapshot: dict[str, float]
) -> None:
    """Restore a card from a physical-value snapshot (in-place).

    Mirrors :func:`plot_fit_results._set_trainee_from_snapshot`.
    """
    with torch.no_grad():
        for name, p in card.named_parameters():
            keys = [k for k in snapshot if k.startswith(name)]
            if not keys:
                continue
            if len(keys) == 1 and not keys[0].endswith("]"):
                vals = torch.tensor([snapshot[keys[0]]], dtype=p.dtype)
            else:
                vals = torch.zeros(p.numel(), dtype=p.dtype)
                for k in keys:
                    i = int(k[k.rfind("[") + 1:k.rfind("]")])
                    vals[i] = snapshot[k]
            if name.endswith(".scale_raw"):
                y = vals.clamp(0.7 + 1e-6, 1.3 - 1e-6)
                raw = torch.atanh((y - 1.0) / 0.3)
            elif name.endswith((".eff_logits", "_logit")):
                y = vals.clamp(1e-6, 1.0 - 1e-6)
                raw = torch.log(y / (1.0 - y))
            elif name.endswith((".rate_raw", ".a_raw", ".b_raw")) or name.startswith((
                "ECal.resolution_func", "HCal.resolution_func",
            )):
                raw = torch.log(torch.expm1(vals.clamp(min=1e-12)))
            else:
                raw = vals
            p.copy_(raw.reshape(p.shape).to(p.dtype))


def _load_init_snapshot(param_config: Path | None) -> tuple[dict[str, float] | None, str]:
    """Load the starting (before-fit) parameter snapshot from a YAML config.

    Handles two config formats:

    * **Standard param config** (``{param: {value, trainable, lr_scale}}``) —
      uses ``value`` directly.
    * **Optuna search config** (``{search: ..., parameters: {param: {low, high}}}``) —
      uses the midpoint ``(low + high) / 2`` for each trainable parameter and
      ``value`` for pinned parameters.

    Returns ``(snapshot, source_description)``.  ``snapshot`` is None if no
    config is provided (caller falls back to constructor defaults).
    """
    if param_config is None:
        return None, "constructor defaults (no --param-config)"

    pkg_configs = Path(pc.__file__).resolve().parent / "param_configs"
    for cand in (param_config, pkg_configs / param_config.name):
        if not cand.exists():
            continue
        with open(cand) as f:
            raw = yaml.safe_load(f)

        # Detect format: optuna config has a top-level "search" key.
        if isinstance(raw, dict) and "search" in raw:
            param_specs = raw.get("parameters", {})
            snap: dict[str, float] = {}
            for key, spec in param_specs.items():
                if "value" in spec:
                    snap[key] = float(spec["value"])
                elif "low" in spec and "high" in spec:
                    snap[key] = 0.5 * (float(spec["low"]) + float(spec["high"]))
                else:
                    snap[key] = 0.0
            return snap, f"{cand} (optuna format, midpoints)"
        else:
            # Standard param config format.
            snap = {k: spec["value"] for k, spec in pc.load_param_config(cand).items()}
            return snap, str(cand)

    return None, f"constructor defaults ({param_config} not found)"


# =============================================================================
# Plot: loss trajectory
# =============================================================================


def plot_loss_trajectory(summary: dict, output_path: Path) -> None:
    """Plot best_loss (and candidate_loss) vs iteration."""
    history = summary.get("history", [])
    if not history:
        print("  [skip] loss_trajectory: no history in summary")
        return

    iters = [h["iteration"] for h in history]
    best = [h["best_loss"] for h in history]
    cand = [h["candidate_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(iters, best, color="tab:blue", label="best loss so far", linewidth=2)
    ax.scatter(iters, cand, color="tab:orange", s=18, alpha=0.7,
               label="candidate loss", zorder=5)
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"GEBO loss trajectory  (best = {min(best):.4e})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  wrote {output_path.name}")


# =============================================================================
# Plot: parameter drift (requires gebo_data.pt)
# =============================================================================

_TENSOR_TITLE_ABBREV: tuple[tuple[str, str], ...] = (
    ("MomentumSmearing.resolution_module", "MS"),
    ("TrackingEfficiency", "TrkEff"),
    ("HadronFractions", "HadFrac"),
    ("scale_module.", ""),
    ("resolution_func.", ""),
)


def _abbrev_tensor(base: str) -> str:
    for long, short in _TENSOR_TITLE_ABBREV:
        base = base.replace(long, short)
    return base


def plot_param_drift(
    data: dict,
    truth: dict[str, float],
    n_initial: int,
    output_path: Path,
    ncols: int = 4,
    smooth_window: int = 5,
) -> None:
    """Plot every parameter's physical value vs query index, with truth lines.

    ``data`` is the dict from ``gebo_data.pt``.  ``train_X`` (n_total, dim)
    contains raw parameter vectors; each is converted to physical space via
    :func:`param_config.to_physical`.  The first ``n_initial`` points are the
    Sobol initialisation; the rest are BO iterations.
    """
    train_X = data["train_X"]  # (n_total, dim) in raw space
    param_names = data["param_names"]  # list of "name[i]" strings
    n_total, dim = train_X.shape

    # Convert every raw vector to physical.
    phys_trajectories: dict[str, list[float]] = {name: [] for name in param_names}
    for i_row in range(n_total):
        vec = train_X[i_row]  # (dim,)
        for j, name in enumerate(param_names):
            base = name.rsplit("[", 1)[0] if name.endswith("]") else name
            phys_val = float(pc.to_physical(base, vec[j].unsqueeze(0)))
            phys_trajectories[name].append(phys_val)

    # Group by tensor base name.
    def _idx(key: str) -> int:
        return int(key[key.rindex("[") + 1:-1]) if key.endswith("]") else -1

    groups: dict[str, list[str]] = {}
    for name in param_names:
        base = name.rsplit("[", 1)[0] if name.endswith("]") else name
        groups.setdefault(base, []).append(name)

    n = len(groups)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 2.6 * nrows), squeeze=False
    )

    def _sliding_mean(values: list[float], window: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if window <= 1 or arr.size == 0:
            return arr
        window = max(1, int(window))
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        padded = np.pad(arr, (pad_left, pad_right), mode="edge")
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(padded, kernel, mode="valid")

    queries = list(range(n_total))
    flat_axes = axes.flatten()
    for ax, (base, keys) in zip(flat_axes, groups.items()):
        for key in sorted(keys, key=_idx):
            trajectory = _sliding_mean(phys_trajectories[key], smooth_window)
            label = f"[{_idx(key)}]" if key.endswith("]") else "value"
            (line,) = ax.plot(queries, trajectory, label=label, linewidth=1.2)
            if key in truth:
                ax.axhline(truth[key], color=line.get_color(),
                           linestyle="--", alpha=0.4)
        # Mark the end of initialisation with a vertical line.
        if n_initial < n_total:
            ax.axvline(n_initial - 1, color="gray", linestyle=":", alpha=0.5,
                       linewidth=0.8)
        ax.set_title(_abbrev_tensor(base), fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Iteration", fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=5, ncol=2 if len(keys) > 3 else 1)

    for ax in flat_axes[n:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Parameter trajectories during GEBO  ({len(param_names)} parameters, "
        f"{n_initial} initial + {n_total - n_initial} BO), \n mean-smoothed over {smooth_window} iterations",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  wrote {output_path.name}")


# =============================================================================
# Helpers: batched trainee forward
# =============================================================================


def _build_val_dataloader(
    arrays: dict, batch_size: int, device: torch.device,
) -> DelphesDataLoader:
    truth_ragged = load_truth_events_ragged(arrays)
    target_ragged = load_pflow_targets_ragged(arrays)
    dataset = DelphesDataSet(truth_ragged, target_ragged, device=device)
    return DelphesDataLoader(dataset, batch_size=batch_size, shuffle=False)


def _trainee_observables(
    card: CMSEnergyFlowDefault, dataloader: DelphesDataLoader
) -> tuple[dict, dict]:
    acc_pred: dict[str, list[torch.Tensor]] = {}
    acc_tgt: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in dataloader:
            truth_particles = batch["truth_particles"]
            mask = torch.any(truth_particles != 0, dim=-1)
            out = card(truth_particles[mask])
            eflow_restored = restore_event_format(out["EFlowObject"], mask)
            pred = load_pflow_targets_from_tensor(eflow_restored)
            target = {k: batch[k] for k in batch if k != "truth_particles"}
            for key in OBSERVABLES:
                if key not in pred or key not in target:
                    continue
                pv, tv = pred[key], target[key]
                if pv.ndim >= 2:
                    pv = pv[pred["pt"] != 0]
                    tv = tv[target["pt"] != 0]
                else:
                    pv, tv = pv.reshape(-1), tv.reshape(-1)
                acc_pred.setdefault(key, []).append(pv.detach().cpu())
                acc_tgt.setdefault(key, []).append(tv.detach().cpu())
    pred_obs = {k: torch.cat(v) for k, v in acc_pred.items()}
    target_obs = {k: torch.cat(v) for k, v in acc_tgt.items()}
    return pred_obs, target_obs


def _obs_values(obs: dict, key: str) -> torch.Tensor:
    v = obs[key]
    if v.ndim >= 2:
        return v[obs["pt"] != 0]
    return v.reshape(-1)


# =============================================================================
# Plot: observable overlay (target / init / fitted)
# =============================================================================


def plot_observable(
    var: str,
    target_vals: torch.Tensor,
    init_vals: torch.Tensor,
    fitted_vals: torch.Tensor,
    edges: np.ndarray,
    xlabel: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    """Overlay target / trainee-init / trainee-best with a ratio panel."""
    # Order: target (step-filled gold), init (red), fitted (blue).
    # plot_comparison_with_ratio treats the first distribution as the
    # reference for the ratio panel and step-fills it.
    target_np = target_vals.detach().cpu().numpy()
    init_np = init_vals.detach().cpu().numpy()
    fitted_np = fitted_vals.detach().cpu().numpy()

    fig, (ax_top, ax_ratio) = plt.subplots(
        2, 1, figsize=(5.5, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    # Top panel: step-filled target, line histograms for init/fitted.
    h_target, _ = np.histogram(target_np, bins=edges)
    h_init, _ = np.histogram(init_np, bins=edges)
    h_fitted, _ = np.histogram(fitted_np, bins=edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    ax_top.stairs(h_target, edges, fill=True, color=_TARGET_COLOR, alpha=0.5,
                  label="target (full-sim)", linewidth=1.2)
    ax_top.stairs(h_init, edges, fill=False, color=_INIT_COLOR, linewidth=1.6,
                  label="trainee, initial")
    ax_top.stairs(h_fitted, edges, fill=False, color=_FINAL_COLOR, linewidth=1.6,
                  label="trainee, GEBO best")
    if log_y:
        ax_top.set_yscale("log")
    ax_top.set_ylabel("Counts")
    ax_top.legend(loc="best", fontsize=9)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(var)

    # Ratio panel: init/target and fitted/target.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_init = np.where(h_target > 0, h_init / h_target, np.nan)
        ratio_fitted = np.where(h_target > 0, h_fitted / h_target, np.nan)

    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax_ratio.step(bin_centers, ratio_init, where="mid", color=_INIT_COLOR,
                  linewidth=1.4, label="init / target")
    ax_ratio.step(bin_centers, ratio_fitted, where="mid", color=_FINAL_COLOR,
                  linewidth=1.4, label="GEBO best / target")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel("ratio / target")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# =============================================================================
# Plot: per-PID final observables
# =============================================================================

_FINAL_PID_GROUPS: tuple[tuple[str, int, str], ...] = (
    ("211", 211, "charged hadron"),
    ("11", 11, "electron"),
    ("13", 13, "muon"),
    ("111", 111, "neutral hadron"),
    ("22", 22, "photon"),
)
_FINAL_PID_VARS: tuple[str, ...] = ("Eta", "PT", "P", "E")


def _final_pid_values(obs: dict[str, torch.Tensor], pid_abs: int, var: str) -> np.ndarray:
    if "pid" not in obs or "pt" not in obs:
        return np.empty(0, dtype=np.float64)
    pid = obs["pid"]
    pt = obs["pt"]
    eta = obs["eta"]
    mask = (pt != 0) & (pid.abs() == pid_abs)
    if not torch.any(mask):
        return np.empty(0, dtype=np.float64)
    if var == "Eta":
        vals = eta[mask]
    elif var == "PT":
        vals = pt[mask]
    elif var == "P":
        vals = pt[mask] * torch.cosh(eta[mask])
    elif var == "E":
        vals = torch.exp(obs["log_E"][mask])
    else:
        return np.empty(0, dtype=np.float64)
    return vals.detach().cpu().numpy().astype(np.float64)


def plot_final_pid_observables(
    target: dict[str, torch.Tensor],
    pred_init: dict[str, torch.Tensor],
    pred_final: dict[str, torch.Tensor],
    output_dir: Path,
    n_events: int,
) -> None:
    """Render per-PID observable overlays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Writing per-PID plots to {output_dir}")

    from parnassus.torch_delphes.plotting import combined_vars_for, stitch_pngs

    for pid_name, pid_abs, pid_label in _FINAL_PID_GROUPS:
        pid_dir = output_dir / pid_name
        pid_dir.mkdir(parents=True, exist_ok=True)

        for var in _FINAL_PID_VARS:
            target_np = _final_pid_values(target, pid_abs=pid_abs, var=var)
            init_np = _final_pid_values(pred_init, pid_abs=pid_abs, var=var)
            final_np = _final_pid_values(pred_final, pid_abs=pid_abs, var=var)

            if target_np.size == 0 and init_np.size == 0 and final_np.size == 0:
                continue

            fig, (ax_top, ax_ratio) = plt.subplots(
                2, 1, figsize=(5.5, 4.8), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )

            edges = np.linspace(
                min(
                    target_np.min() if target_np.size else 0,
                    init_np.min() if init_np.size else 0,
                    final_np.min() if final_np.size else 0,
                ),
                max(
                    target_np.max() if target_np.size else 1,
                    init_np.max() if init_np.size else 1,
                    final_np.max() if final_np.size else 1,
                ),
                51,
            )

            h_target, _ = np.histogram(target_np, bins=edges)
            h_init, _ = np.histogram(init_np, bins=edges)
            h_final, _ = np.histogram(final_np, bins=edges)
            bin_centers = 0.5 * (edges[:-1] + edges[1:])

            ax_top.stairs(h_target, edges, fill=True, color=_TARGET_COLOR,
                          alpha=0.5, label="target (full-sim)", linewidth=1.2)
            ax_top.stairs(h_init, edges, fill=False, color=_INIT_COLOR,
                          linewidth=1.6, label="trainee, initial")
            ax_top.stairs(h_final, edges, fill=False, color=_FINAL_COLOR,
                          linewidth=1.6, label="trainee, GEBO best")
            ax_top.set_ylabel("Counts")
            ax_top.legend(loc=("upper left" if var != "Eta" else "best"), fontsize=8)
            ax_top.grid(True, alpha=0.3)
            ax_top.set_title(f"{pid_label}  {var}")
            if var in ("PT", "P", "E"):
                ax_top.set_yscale("log")

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_init = np.where(h_target > 0, h_init / h_target, np.nan)
                ratio_final = np.where(h_target > 0, h_final / h_target, np.nan)
            ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            ax_ratio.step(bin_centers, ratio_init, where="mid", color=_INIT_COLOR,
                          linewidth=1.4)
            ax_ratio.step(bin_centers, ratio_final, where="mid", color=_FINAL_COLOR,
                          linewidth=1.4)
            ax_ratio.set_xlabel(var)
            ax_ratio.set_ylabel("ratio / target")
            ax_ratio.set_ylim(0.5, 1.5)
            ax_ratio.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.savefig(pid_dir / f"{var}.png")
            plt.close(fig)

        combo = combined_vars_for(_FINAL_PID_VARS)
        per_var_pngs = [pid_dir / f"{v}.png" for v in combo]
        all_png = pid_dir / "all.png"
        wrote = stitch_pngs(
            image_paths=per_var_pngs,
            output_path=all_png,
            title=f"Final EFlowObject: {pid_label}\n{n_events} events",
            title_align="left",
        )
        n_var = len(list(pid_dir.glob("*.png"))) - (1 if wrote else 0)
        extra = f" + all.png ({', '.join(combo)})" if wrote else ""
        print(f"    {pid_name} ({pid_label}): wrote {n_var} per-var PNGs{extra}")


# =============================================================================
# Plot: acquisition history + loss scatter
# =============================================================================


def plot_acquisition_history(summary: dict, output_path: Path) -> None:
    history = summary.get("history", [])
    acq_vals = [h.get("acq_value") for h in history if "acq_value" in h]
    if not acq_vals:
        return
    iters = [h["iteration"] for h in history if "acq_value" in h]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.plot(iters, acq_vals, color="tab:green", marker="o", markersize=4,
            linewidth=1.5)
    ax.set_xlabel("BO iteration")
    ax.set_ylabel("acquisition value")
    ax.set_title("Acquisition function value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  wrote {output_path.name}")


def plot_loss_scatter(data: dict, n_initial: int, output_path: Path) -> None:
    if data is None:
        return
    losses = data["train_Y"][:, 0].numpy()
    n = len(losses)
    best_idx = int(np.argmin(losses))
    fig, ax = plt.subplots(figsize=(7, 4.0))
    colors = [
        "tab:red" if i == best_idx else
        "tab:gray" if i < n_initial else "tab:blue"
        for i in range(n)
    ]
    ax.scatter(range(n), losses, c=colors, s=20, alpha=0.7, edgecolors="none")
    ax.axhline(losses[best_idx], color="tab:red", linestyle="--", alpha=0.5,
               label=f"best = {losses[best_idx]:.4e}")
    if n_initial < n:
        ax.axvline(n_initial - 0.5, color="gray", linestyle=":", alpha=0.5,
                   label=f"initial ({n_initial} Sobol)")
    ax.set_xlabel("query index")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.set_title(f"All {n} queried points  (best @ idx {best_idx})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"  wrote {output_path.name}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True,
                        help="Path to gebo_summary.json.")
    parser.add_argument("--root-file", type=Path, required=True,
                        help="CMS full-simulation ROOT file used for the fit.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same dir as --summary).")
    parser.add_argument("--n-events-for-plots", type=int, default=20_000,
                        help="Number of validation events to plot (default 2000).")
    parser.add_argument("--plot-batch-size", type=int, default=2000,
                        help="Events per trainee forward batch.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--truth-config", type=Path, default=_DEFAULT_TRUTH_CONFIG,
        help="Generation/truth config YAML for reference lines "
             "(default: cms_target_default.yaml).",
    )
    parser.add_argument(
        "--param-config", type=Path, default=None,
        help="Starting (before-fit) config YAML for the 'initial' trainee "
             "curves.  If omitted, falls back to constructor defaults.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.summary.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_gebo_summary(args.summary)
    gebo_data = _load_gebo_data(args.summary.parent / "gebo_data.pt")

    best_loss = summary.get("best_loss", float("nan"))
    dimension = summary.get("dimension", "?")
    n_total = summary.get("n_total_points", "?")
    n_initial = args.n_initial if hasattr(args, "n_initial") else summary.get("args", {}).get("n_initial", 30)
    # Try to get n_initial from the summary args.
    _sargs = summary.get("args", {})
    if isinstance(_sargs, dict):
        n_initial = _sargs.get("n_initial", 40)

    print(f"GEBO run: dim={dimension}, {n_total} points, best_loss={best_loss:.4e}")
    print(f"Writing figures to {output_dir}")

    # ---- Load truth config ----
    flat_truth = pc.load_param_config(args.truth_config)
    truth = {k: spec["value"] for k, spec in flat_truth.items()}
    print(f"  truth config: {args.truth_config}")

    # ---- Load init snapshot ----
    init_snapshot, init_source = _load_init_snapshot(args.param_config)
    print(f"  trainee 'initial' (before-fit) params: {init_source}")

    # ---- 1. Loss trajectory ----
    plot_loss_trajectory(summary, output_dir / "loss_trajectory.pdf")

    # ---- 2. Parameter drift (requires gebo_data.pt) ----
    if gebo_data is not None:
        plot_param_drift(gebo_data, truth, n_initial,
                         output_dir / "param_drift_all.pdf")
    else:
        print("  [skip] param_drift_all: gebo_data.pt not found")

    # ---- 3. Acquisition history ----
    plot_acquisition_history(summary, output_dir / "acquisition_history.pdf")

    # ---- 4. Loss scatter ----
    if gebo_data is not None:
        plot_loss_scatter(gebo_data, n_initial, output_dir / "loss_scatter.pdf")

    # ---- 5. Observable histograms ----
    import uproot
    with uproot.open(str(args.root_file)) as f:
        n_total_events = int(f["event_tree"].num_entries)
    val_entry_start = int(0.7 * n_total_events)
    n_val_events = int(0.9 * n_total_events) - val_entry_start
    n_plot_events = min(args.n_events_for_plots, n_val_events)

    arrays = load_cms_flow_root(
        args.root_file, n_events=n_plot_events, entry_start=val_entry_start,
    )
    device = torch.device("cpu")
    val_loader = _build_val_dataloader(arrays, args.plot_batch_size, device)
    del arrays
    gc.collect()

    print(
        f"  plotting on validation split: "
        f"{n_plot_events} events "
        f"(entries {val_entry_start}:{val_entry_start + n_plot_events})"
    )

    # --- Initial trainee (from --param-config or defaults) ---
    torch.manual_seed(args.seed)
    trainee_init = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    if init_snapshot is not None:
        _set_trainee_from_snapshot(trainee_init, init_snapshot)
    pred_init, target = _trainee_observables(trainee_init, val_loader)

    # --- Fitted trainee (from GEBO best) ---
    best_raw = summary.get("best_raw_params", {})
    if not best_raw:
        raise SystemExit("gebo_summary.json has no 'best_raw_params'")
    trainee_final = _build_card_from_raw_params(best_raw, device)
    torch.manual_seed(args.seed)
    pred_final, _ = _trainee_observables(trainee_final, val_loader)

    # Observable plots.
    plot_observable(
        "PT",
        _obs_values(target, "pt"),
        _obs_values(pred_init, "pt"),
        _obs_values(pred_final, "pt"),
        edges=np.linspace(0.0, 100.0, 51),
        xlabel=r"PF object $p_\mathrm{T}$ [GeV]",
        output_path=output_dir / "observable_pt.pdf",
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
        output_path=output_dir / "observable_eta.pdf",
    )
    print("  wrote observable_eta.pdf")

    plot_observable(
        "ht",
        _obs_values(target, "ht"),
        _obs_values(pred_init, "ht"),
        _obs_values(pred_final, "ht"),
        edges=np.linspace(0.0, 1000.0, 51),
        xlabel=r"PF scalar $H_\mathrm{T}$ [GeV]",
        output_path=output_dir / "observable_ht.pdf",
    )
    print("  wrote observable_ht.pdf")

    plot_observable(
        "log_ht",
        _obs_values(target, "log_ht"),
        _obs_values(pred_init, "log_ht"),
        _obs_values(pred_final, "log_ht"),
        edges=np.linspace(4.5, 7.5, 51),
        xlabel=r"PF scalar $\log\,H_\mathrm{T}$",
        output_path=output_dir / "observable_log_ht.pdf",
    )
    print("  wrote observable_log_ht.pdf")

    plot_observable(
        "multiplicity",
        _obs_values(target, "multiplicity"),
        _obs_values(pred_init, "multiplicity"),
        _obs_values(pred_final, "multiplicity"),
        edges=np.linspace(0.0, 300.0, 61),
        xlabel=r"PF objects per event",
        output_path=output_dir / "observable_multiplicity.pdf",
    )
    print("  wrote observable_multiplicity.pdf")

    # Per-PID observables.
    plot_final_pid_observables(
        target=target,
        pred_init=pred_init,
        pred_final=pred_final,
        output_dir=output_dir / "PID",
        n_events=n_plot_events,
    )

    n_pdfs = len(list(output_dir.glob("*.pdf")))
    print(f"\nDone.  {n_pdfs} PDF figures in {output_dir}/")


if __name__ == "__main__":
    main()
