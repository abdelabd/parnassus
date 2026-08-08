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
    load_pflow_targets_ragged,
    load_truth_events_ragged,
)
from .dataloader import DelphesDataLoader, DelphesDataSet
from .gebo_plotting import (
    build_card_from_raw_params,
    build_val_dataloader,
    obs_values,
    plot_observable_best_only,
    plot_pid_observables,
    trainee_observables,
)
from parnassus.torch_delphes.plotting import (
    plot_comparison_with_ratio,
    stitch_pngs,
    combined_vars_for,
)

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
        return values
        # arr = np.asarray(values, dtype=float)
        # if window <= 1 or arr.size == 0:
        #     return arr
        # window = max(1, int(window))
        # pad_left = window // 2
        # pad_right = window - 1 - pad_left
        # padded = np.pad(arr, (pad_left, pad_right), mode="edge")
        # kernel = np.ones(window, dtype=float) / float(window)
        # return np.convolve(padded, kernel, mode="valid")

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


def _load_saved_init_pred(
    summary_dir: Path,
    val_loader: DelphesDataLoader,
    seed: int,
) -> dict | None:
    """Rebuild the initial-trainee observables from the raw params saved by
    :mod:`gebo_search` (``init_trainee_raw.pt``).

    That file stores the best Sobol point captured at the start of the run.
    Evaluating it against the *current* ``val_loader`` reproduces exactly the
    initial-trainee overlay that ``gebo_search`` draws in its intermediate
    plots -- and is robust to the validation split changing since the run.
    Returns ``None`` for older runs that predate the saved file (their plots
    simply omit the initial-trainee curves).
    """
    init_raw_path = summary_dir / "init_trainee_raw.pt"
    if not init_raw_path.exists():
        return None
    ckpt = torch.load(init_raw_path, map_location="cpu", weights_only=False)
    raw_params = ckpt["raw_params"]
    param_names = ckpt["param_names"]
    best_sobol_raw = {
        param_names[j]: float(raw_params[j]) for j in range(len(param_names))
    }
    torch.manual_seed(seed)
    init_card = build_card_from_raw_params(best_sobol_raw, torch.device("cpu"))
    pred_init, _ = trainee_observables(init_card, val_loader)
    return pred_init


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
    # GEBO runs save gebo_data.pt; the two fine-tuning stages save lbfgs_data.pt
    # (lbfgs_finetune.py) / adam_data.pt (adam_finetune.py) with the same
    # train_X / train_Y / param_names schema. Fall back through them so
    # param_drift_all.pdf / loss_scatter.pdf render for all three.
    gebo_data = _load_gebo_data(args.summary.parent / "gebo_data.pt")
    if gebo_data is None:
        gebo_data = _load_gebo_data(args.summary.parent / "lbfgs_data.pt")
    if gebo_data is None:
        gebo_data = _load_gebo_data(args.summary.parent / "adam_data.pt")

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

    # ---- Load init snapshot (legacy fallback for the 'initial' trainee) ----
    init_snapshot, init_source = _load_init_snapshot(args.param_config)

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
    truth_ragged = load_truth_events_ragged(arrays)
    target_ragged = load_pflow_targets_ragged(arrays)
    dataset = DelphesDataSet(truth_ragged, target_ragged, device=device)
    val_loader = DelphesDataLoader(dataset, batch_size=args.plot_batch_size, shuffle=False)
    del arrays
    gc.collect()

    print(
        f"  plotting on validation split: "
        f"{n_plot_events} events "
        f"(entries {val_entry_start}:{val_entry_start + n_plot_events})"
    )

    # --- Fitted trainee (from GEBO best) ---
    best_raw = summary.get("best_raw_params", {})
    if not best_raw:
        raise SystemExit("gebo_summary.json has no 'best_raw_params'")
    trainee_final = build_card_from_raw_params(best_raw, device)
    torch.manual_seed(args.seed)
    pred_final, target = trainee_observables(trainee_final, val_loader)

    # --- Initial trainee ---
    # Prefer the raw params saved by gebo_search (init_trainee_raw.pt: the best
    # Sobol point), re-evaluated against the CURRENT validation split so the
    # overlay matches gebo_search's intermediate plots exactly. Fall back to
    # --param-config (legacy) if the file is absent; otherwise omit init curves.
    pred_init = _load_saved_init_pred(args.summary.parent, val_loader, args.seed)
    if pred_init is not None:
        print(f"  trainee 'initial': {args.summary.parent / 'init_trainee_raw.pt'} "
              "(saved best-Sobol point)")
    elif init_snapshot is not None:
        torch.manual_seed(args.seed)
        trainee_init = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
        _set_trainee_from_snapshot(trainee_init, init_snapshot)
        pred_init, _ = trainee_observables(trainee_init, val_loader)
        print(f"  trainee 'initial' (fallback --param-config): {init_source}")
    else:
        print("  trainee 'initial': none (no init_trainee_raw.pt, no --param-config); "
              "initial-trainee curves omitted")

    # Observable plots.
    plot_observable_best_only(
        "PT",
        obs_values(target, "pt"),
        obs_values(pred_final, "pt"),
        edges=np.linspace(0.0, 100.0, 51),
        xlabel=r"PF object $p_\mathrm{T}$ [GeV]",
        output_path=output_dir / "observable_pt.pdf",
        log_y=True,
        init_vals=obs_values(pred_init, "pt") if pred_init is not None else None,
    )
    print("  wrote observable_pt.pdf")

    plot_observable_best_only(
        "Eta",
        obs_values(target, "eta"),
        obs_values(pred_final, "eta"),
        edges=np.linspace(-5.0, 5.0, 51),
        xlabel=r"PF object $\eta$",
        output_path=output_dir / "observable_eta.pdf",
        init_vals=obs_values(pred_init, "eta") if pred_init is not None else None,
    )
    print("  wrote observable_eta.pdf")

    plot_observable_best_only(
        "ht",
        obs_values(target, "ht"),
        obs_values(pred_final, "ht"),
        edges=np.linspace(0.0, 1000.0, 51),
        xlabel=r"PF scalar $H_\mathrm{T}$ [GeV]",
        output_path=output_dir / "observable_ht.pdf",
        init_vals=obs_values(pred_init, "ht") if pred_init is not None else None,
    )
    print("  wrote observable_ht.pdf")

    plot_observable_best_only(
        "log_ht",
        obs_values(target, "log_ht"),
        obs_values(pred_final, "log_ht"),
        edges=np.linspace(4.5, 7.5, 51),
        xlabel=r"PF scalar $\log\,H_\mathrm{T}$",
        output_path=output_dir / "observable_log_ht.pdf",
        init_vals=obs_values(pred_init, "log_ht") if pred_init is not None else None,
    )
    print("  wrote observable_log_ht.pdf")

    plot_observable_best_only(
        "multiplicity",
        obs_values(target, "multiplicity"),
        obs_values(pred_final, "multiplicity"),
        edges=np.linspace(0.0, 300.0, 61),
        xlabel=r"PF objects per event",
        output_path=output_dir / "observable_multiplicity.pdf",
        init_vals=obs_values(pred_init, "multiplicity") if pred_init is not None else None,
    )
    print("  wrote observable_multiplicity.pdf")

    # Per-PID observables.
    plot_pid_observables(
        target=target,
        pred_final=pred_final,
        output_dir=output_dir / "PID",
        n_events=n_plot_events,
        pred_init=pred_init,
    )

    n_pdfs = len(list(output_dir.glob("*.pdf")))
    print(f"\nDone.  {n_pdfs} PDF figures in {output_dir}/")


if __name__ == "__main__":
    main()
