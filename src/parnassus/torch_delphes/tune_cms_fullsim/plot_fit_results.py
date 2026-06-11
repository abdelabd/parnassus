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

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

# The truth reference lines on the parameter-drift plots come from the same
# param config used to generate the sample (its physical ``value`` fields).
_DEFAULT_PARAM_CONFIG = (
    Path(pc.__file__).resolve().parent / "param_configs" / "cms_target_default.yaml"
)

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
    ax.step(centers, h_tgt, where="mid", color="black", label="target")
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
    parser.add_argument("--n-events-for-plots", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--truth-config",
        type=Path,
        default=_DEFAULT_PARAM_CONFIG,
        help=(
            "The GENERATION/truth config that made the ROOT file -- its physical "
            "'value' fields are drawn as the truth reference lines on the "
            "parameter-drift plots. Do NOT pass a training config: a trained "
            "parameter's 'value' there is its off-truth STARTING point, not its "
            "truth, so the reference line would be wrong. Defaults to "
            "cms_target_default.yaml."
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

    # Ground-truth physical value of every scalar, keyed by the same name[i]
    # form the history snapshots use. These come from the GENERATION config; a
    # training config would give a trained param's start value, not its truth.
    flat_truth_cfg = pc.load_param_config(args.truth_config)
    n_trainable = sum(1 for spec in flat_truth_cfg.values() if spec["trainable"])
    if n_trainable:
        print(
            f"WARNING: {args.truth_config} marks {n_trainable} parameter(s) trainable -- "
            "it looks like a TRAINING config, not the generation/truth config. Truth "
            "reference lines for those params will show their START value, not the truth. "
            "Pass the generation config (e.g. param_configs/cms_target_default.yaml) "
            "to --truth-config instead."
        )
    truth = {k: spec["value"] for k, spec in flat_truth_cfg.items()}

    print(f"Writing figures to {args.output_dir}")

    # ----- 1. Loss trajectory -----
    plot_loss(history, args.output_dir / "loss_trajectory.pdf")
    print("  wrote loss_trajectory.pdf")

    # ----- 2. Scale parameter drift -----
    # Truth values are read from the param config (per eta region).
    scale_members: dict[str, list[tuple[str, float]]] = {
        "charged-hadron pT scale": [
            (k, truth[k])
            for k in (
                f"ChargedHadronMomentumSmearing.resolution_module.scale_raw[{i}]"
                for i in range(3)
            )
        ],
        "ECal energy scale": [
            (f"ECal.scale_module.scale_raw[{i}]", truth[f"ECal.scale_module.scale_raw[{i}]"])
            for i in range(3)
        ],
        "HCal energy scale": [
            (f"HCal.scale_module.scale_raw[{i}]", truth[f"HCal.scale_module.scale_raw[{i}]"])
            for i in range(2)
        ],
    }
    plot_param_drift(
        history,
        scale_members,
        args.output_dir / "param_drift_scales.pdf",
        title="Scale-parameter drift during Adam fit",
    )
    print("  wrote param_drift_scales.pdf")

    # ----- 3. Other representative parameters (truth from the config) -----
    other_members: dict[str, list[tuple[str, float]]] = {
        "chad res. a (barrel)": [
            (k := "ChargedHadronMomentumSmearing.resolution_module.a_raw[0]", truth[k]),
        ],
        "chad eff. (barrel, low-pT)": [
            (k := "ChargedHadronTrackingEfficiency.eff_logits[0]", truth[k]),
        ],
        "K0-short ECal fraction": [
            (k := "HadronFractions.k0s_logit", truth[k]),
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

    plot_observable(
        _obs_values(target, "ht"),
        _obs_values(pred_init, "ht"),
        _obs_values(pred_final, "ht"),
        edges=np.linspace(0.0, 1000.0, 51),
        xlabel=r"PF scalar $H_\mathrm{T}$ [GeV]",
        output_path=args.output_dir / "observable_ht.pdf",
    )
    print("  wrote observable_ht.pdf")

    # log(HT) -- the per-event scalar actually in the loss. Keep these edges in
    # sync with DEFAULT_BIN_EDGES["log_ht"] in config.py.
    plot_observable(
        _obs_values(target, "log_ht"),
        _obs_values(pred_init, "log_ht"),
        _obs_values(pred_final, "log_ht"),
        edges=np.linspace(4.5, 7.5, 51),
        xlabel=r"PF scalar $\log\,H_\mathrm{T}$",
        output_path=args.output_dir / "observable_log_ht.pdf",
    )
    print("  wrote observable_log_ht.pdf")

    plot_observable(
        _obs_values(target, "multiplicity"),
        _obs_values(pred_init, "multiplicity"),
        _obs_values(pred_final, "multiplicity"),
        edges=np.linspace(0.0, 600.0, 61),
        xlabel=r"PF objects per event",
        output_path=args.output_dir / "observable_multiplicity.pdf",
    )
    print("  wrote observable_multiplicity.pdf")

    # ----- 5. Optional: per-module intermediate observables (--debug) -----
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
            output_dir=args.output_dir / "intermediate",
            n_events=args.n_events_for_plots,
        )

    print(f"Done. {len(list(args.output_dir.glob('*.pdf')))} figures in {args.output_dir}.")


if __name__ == "__main__":
    main()
