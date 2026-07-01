"""Shared validation + plotting helpers for GEBO.

Imported by both :mod:`gebo_search` (for intermediate per-best plots during
the BO loop) and :mod:`plot_gebo_results` (for final post-hoc plots).
"""

from __future__ import annotations

import gc
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

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

# ---- Styling ----
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

TARGET_COLOR: str = "gold"
INIT_COLOR: str = "tab:red"
FINAL_COLOR: str = "tab:blue"


# =============================================================================
# Card builder
# =============================================================================


def build_card_from_raw_params(
    raw_params: dict[str, float], device: torch.device
) -> CMSEnergyFlowDefault:
    """Build a card and set its parameters from a raw-parameter dict.

    ``raw_params`` maps ``"param_name[i]"`` → raw (pre-transform) value.
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
# Validation dataloader
# =============================================================================


def build_val_dataloader(
    root_file: Path,
    n_plot_events: int,
    batch_size: int,
    device: torch.device,
) -> tuple[DelphesDataLoader, int, int]:
    """Build a ragged validation dataloader from a CMS full-sim ROOT file.

    Returns ``(loader, val_entry_start, n_plot_events)``.
    """
    import uproot
    with uproot.open(str(root_file)) as f:
        n_total_events = int(f["event_tree"].num_entries)
    val_entry_start = int(0.7 * n_total_events)
    n_val_events = int(0.9 * n_total_events) - val_entry_start
    n_plot = min(n_plot_events, n_val_events)

    arrays = load_cms_flow_root(root_file, n_events=n_plot, entry_start=val_entry_start)
    truth_ragged = load_truth_events_ragged(arrays)
    target_ragged = load_pflow_targets_ragged(arrays)
    dataset = DelphesDataSet(truth_ragged, target_ragged, device=device)
    del arrays
    gc.collect()
    return DelphesDataLoader(dataset, batch_size=batch_size, shuffle=False), val_entry_start, n_plot


# =============================================================================
# Trainee forward pass
# =============================================================================


def trainee_observables(
    card: CMSEnergyFlowDefault, dataloader: DelphesDataLoader
) -> tuple[dict, dict]:
    """Run trainee batch-by-batch → (pred, target) flattened observables."""
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


def obs_values(obs: dict, key: str) -> torch.Tensor:
    """Flatten an observable, dropping padding."""
    v = obs[key]
    if v.ndim >= 2:
        return v[obs["pt"] != 0]
    return v.reshape(-1)


# =============================================================================
# Plot: single observable (target vs best-only, no init baseline)
# =============================================================================


def plot_observable_best_only(
    var: str,
    target_vals: torch.Tensor,
    fitted_vals: torch.Tensor,
    edges: np.ndarray,
    xlabel: str,
    output_path: Path,
    log_y: bool = False,
    iteration: int | None = None,
    init_vals: torch.Tensor | None = None,
) -> None:
    """Target (gold step-filled) vs trainee-best (blue) with ratio panel.

    If ``init_vals`` is provided, also plots the initial trainee in red.
    """
    target_np = target_vals.detach().cpu().numpy()
    fitted_np = fitted_vals.detach().cpu().numpy()
    init_np = init_vals.detach().cpu().numpy() if init_vals is not None else None

    fig, (ax_top, ax_ratio) = plt.subplots(
        2, 1, figsize=(5.5, 4.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )

    h_target, _ = np.histogram(target_np, bins=edges)
    h_fitted, _ = np.histogram(fitted_np, bins=edges)
    if init_np is not None:
        h_init, _ = np.histogram(init_np, bins=edges)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    ax_top.stairs(h_target, edges, fill=True, color=TARGET_COLOR, alpha=0.5,
                  label="target (full-sim)", linewidth=1.2)
    if init_np is not None:
        ax_top.stairs(h_init, edges, fill=False, color=INIT_COLOR, linewidth=1.6,
                      label="trainee, initial")
    ax_top.stairs(h_fitted, edges, fill=False, color=FINAL_COLOR, linewidth=1.6,
                  label="trainee, current best")
    if log_y:
        ax_top.set_yscale("log")
    ax_top.set_ylabel("Counts")
    ax_top.legend(loc="best", fontsize=9)
    ax_top.grid(True, alpha=0.3)
    title = var
    if iteration is not None:
        title += f"  (iter {iteration})"
    ax_top.set_title(title)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_fitted = np.where(h_target > 0, h_fitted / h_target, np.nan)
        ratio_init = np.where(h_target > 0, h_init / h_target, np.nan) if init_np is not None else None

    ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    if ratio_init is not None:
        ax_ratio.step(bin_centers, ratio_init, where="mid", color=INIT_COLOR,
                      linewidth=1.4, label="init / target")
    ax_ratio.step(bin_centers, ratio_fitted, where="mid", color=FINAL_COLOR,
                  linewidth=1.4, label="best / target")
    ax_ratio.set_xlabel(xlabel)
    ax_ratio.set_ylabel("ratio / target")
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# =============================================================================
# Per-PID observable plots
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


def plot_pid_observables(
    target: dict[str, torch.Tensor],
    pred_final: dict[str, torch.Tensor],
    output_dir: Path,
    n_events: int,
    iteration: int | None = None,
    pred_init: dict[str, torch.Tensor] | None = None,
) -> None:
    """Render per-PID observable overlays (target vs best-only)."""
    from parnassus.torch_delphes.plotting import combined_vars_for, stitch_pngs

    output_dir.mkdir(parents=True, exist_ok=True)

    for pid_name, pid_abs, pid_label in _FINAL_PID_GROUPS:
        pid_dir = output_dir / pid_name
        pid_dir.mkdir(parents=True, exist_ok=True)

        for var in _FINAL_PID_VARS:
            target_np = _final_pid_values(target, pid_abs=pid_abs, var=var)
            final_np = _final_pid_values(pred_final, pid_abs=pid_abs, var=var)
            init_np = _final_pid_values(pred_init, pid_abs=pid_abs, var=var) if pred_init is not None else None

            if target_np.size == 0 and final_np.size == 0:
                continue

            fig, (ax_top, ax_ratio) = plt.subplots(
                2, 1, figsize=(5.5, 4.8), sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )

            if target_np.size > 0 or final_np.size > 0:
                lo = min(
                    target_np.min() if target_np.size else final_np.min(),
                    final_np.min() if final_np.size else target_np.min(),
                )
                hi = max(
                    target_np.max() if target_np.size else final_np.max(),
                    final_np.max() if final_np.size else target_np.max(),
                )
            else:
                lo, hi = 0, 1
            edges = np.linspace(lo, hi, 51)

            h_target, _ = np.histogram(target_np, bins=edges)
            h_final, _ = np.histogram(final_np, bins=edges)
            h_init = np.histogram(init_np, bins=edges)[0] if init_np is not None else None
            bin_centers = 0.5 * (edges[:-1] + edges[1:])

            ax_top.stairs(h_target, edges, fill=True, color=TARGET_COLOR,
                          alpha=0.5, label="target (full-sim)", linewidth=1.2)
            if init_np is not None:
                ax_top.stairs(h_init, edges, fill=False, color=INIT_COLOR,
                              linewidth=1.6, label="trainee, initial")
            ax_top.stairs(h_final, edges, fill=False, color=FINAL_COLOR,
                          linewidth=1.6, label="trainee, best")
            ax_top.set_ylabel("Counts")
            ax_top.legend(loc=("upper left" if var != "Eta" else "best"), fontsize=8)
            ax_top.grid(True, alpha=0.3)
            title = f"{pid_label}  {var}"
            if iteration is not None:
                title += f"  (iter {iteration})"
            ax_top.set_title(title)
            if var in ("PT", "P", "E"):
                ax_top.set_yscale("log")

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio_final = np.where(h_target > 0, h_final / h_target, np.nan)
                ratio_init = np.where(h_target > 0, h_init / h_target, np.nan) if h_init is not None else None
            ax_ratio.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
            if ratio_init is not None:
                ax_ratio.step(bin_centers, ratio_init, where="mid", color=INIT_COLOR,
                              linewidth=1.4)
            ax_ratio.step(bin_centers, ratio_final, where="mid", color=FINAL_COLOR,
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
        stitch_pngs(
            image_paths=per_var_pngs,
            output_path=all_png,
            title=f"EFlowObject: {pid_label}\n{n_events} events",
            title_align="left",
        )
