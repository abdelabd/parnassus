"""Generate figures for the Differentiable TorchDelphes JINST-style paper.

This script is a small, self-contained matplotlib driver that reads a
training-history JSON file (written by
``tune_cms_fullsim.py --history-path``) plus the committed Pythia-
generated pseudodata ROOT file, and writes a set of PDF figures under
``doc/figures/`` for inclusion in the paper.

The intended figures are:

- ``loss_trajectory.pdf`` : loss vs Adam step on log-y scale.
- ``param_drift_all.pdf`` : trajectories of **every** parameter vs step
  (one subplot per tensor), with the known ground-truth values as
  horizontal dashed lines.
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
import gc
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

from .config import (
    DEFAULT_ABS_ETA_CUT,
    DEFAULT_RECO_PT_CUT,
    DEFAULT_TRUTH_PT_CUT,
    OBSERVABLES,
)
from .data import (
    apply_reco_acceptance_cut,
    load_cms_flow_root,
    load_pflow_targets_from_tensor,
    load_pflow_targets_ragged,
    load_truth_events_ragged,
    restore_event_format,
)
from .dataloader import DelphesDataLoader, DelphesDataSet
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
    ax.set_xlabel("Epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss trajectory")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# Predictable abbreviations so the per-tensor subplot titles stay short but
# recognizable in the all-parameter grid.
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


def plot_all_param_drift(
    history: dict,
    truth: dict[str, float],
    output_path: Path,
    ncols: int = 4,
    title: str = "All parameter trajectories during Adam fit",
) -> None:
    """Plot EVERY parameter in the history snapshots vs Adam step.

    One subplot per tensor (its scalar elements ``name[i]`` share an axis, so
    they keep a common y-scale); scalar parameters with no index get their own
    single-line subplot. Subplots are laid out in an ``ncols``-wide grid.
    Dashed horizontal lines are the ground-truth values from ``truth`` (the
    generation config), skipped for any key absent from it.

    This covers the full parameter set, so the whole fit can be inspected at a
    glance.
    """
    if not history.get("parameters"):
        raise ValueError("history dict has no 'parameters' snapshots")
    snapshots = history["parameters"]
    steps = history["step"]

    # Group scalar keys by their tensor name: "Module.tensor[i]" -> "Module.tensor".
    # Bare scalars ("...barrel_a") form their own single-member group. First-seen
    # order is preserved so related tensors stay adjacent in the grid.
    def _idx(key: str) -> int:
        return int(key[key.rindex("[") + 1 : -1]) if key.endswith("]") else -1

    groups: dict[str, list[str]] = {}
    for key in snapshots[0]:
        base = key.rsplit("[", 1)[0] if key.endswith("]") else key
        groups.setdefault(base, []).append(key)

    n = len(groups)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 2.6 * nrows), squeeze=False
    )
    flat_axes = axes.flatten()
    for ax, (base, keys) in zip(flat_axes, groups.items()):
        for key in sorted(keys, key=_idx):
            trajectory = [snap[key] for snap in snapshots]
            label = f"[{_idx(key)}]" if key.endswith("]") else "value"
            (line,) = ax.plot(steps, trajectory, label=label, linewidth=1.2)
            if key in truth:
                ax.axhline(truth[key], color=line.get_color(), linestyle="--", alpha=0.4)
        ax.set_title(_abbrev_tensor(base), fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Epoch", fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=5, ncol=2 if len(keys) > 3 else 1)
    # Hide the trailing empty grid cells.
    for ax in flat_axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{title}  ({len(snapshots[0])} parameters)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
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
    ax.set_xlabel("Epoch")
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


def _load_init_snapshot(
    param_config: Path | None, history: dict
) -> tuple[dict[str, float] | None, str]:
    """Resolve the trainee-init parameter snapshot for the "before fit" curves.

    The honest "before fit" baseline is the perturbed STARTING config the fit was
    launched from -- NOT the card's hardcoded constructor defaults (which are
    neither the start nor the truth). The config path is taken from
    ``param_config`` if given, else from the history metadata (``param_config``,
    recorded by the tuning CLI). Config ``value`` fields are physical, the same
    convention :func:`_set_trainee_from_snapshot` inverts, so the returned dict can
    be fed straight to it.

    Returns ``(snapshot, source)``. ``snapshot`` is ``None`` when no config can be
    resolved -- the caller then falls back to constructor defaults; ``source`` is a
    human-readable description for logging.
    """
    cfg = param_config
    if cfg is None:
        meta_cfg = history.get("metadata", {}).get("param_config")
        cfg = Path(meta_cfg) if meta_cfg else None
    if cfg is None:
        return None, "constructor defaults (no --param-config and none in history metadata)"
    # Resolve: as given (CWD-relative), else by basename next to the packaged configs.
    pkg_configs = Path(pc.__file__).resolve().parent / "param_configs"
    for cand in (cfg, pkg_configs / cfg.name):
        if cand.exists():
            snap = {k: spec["value"] for k, spec in pc.load_param_config(cand).items()}
            return snap, str(cand)
    return None, f"constructor defaults ({cfg} not found)"


def _build_val_dataloader(
    arrays: dict,
    batch_size: int,
    device: torch.device,
    truth_pt_cut: float | None = None,
    reco_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
) -> DelphesDataLoader:
    """Build the ragged validation dataloader the plot forwards iterate over.

    Mirrors the tuning path (:mod:`tune_cms_fullsim.cli`): keep the truth
    particles and per-particle targets *ragged* (per-event lists, no global
    padding) and let :func:`dataloader.delphes_collate_fn` pad each batch to its
    own max multiplicity. This avoids densifying the whole validation split to
    the global max multiplicity -- the allocation that, together with the old
    single-shot forward, OOM-killed this script on a memory-limited node.

    ``shuffle=False`` so the batch order is fixed: re-seeding the RNG before each
    full pass then gives the init and fitted cards the *same* per-batch noise, so
    their difference reflects the parameters, not the stochastic smearing.

    ``truth_pt_cut`` / ``reco_pt_cut`` / ``abs_eta_cut``: the same acceptance
    cuts the fit applied (read from the history metadata by the caller), so the
    plotted target matches what the loss saw. NO chad truncation here -- the
    final plots deliberately show the full (untruncated) spectra.
    """
    truth_ragged = load_truth_events_ragged(
        arrays, truth_pt_cut=truth_pt_cut, abs_eta_cut=abs_eta_cut
    )
    target_ragged = load_pflow_targets_ragged(
        arrays, reco_pt_cut=reco_pt_cut, abs_eta_cut=abs_eta_cut
    )
    dataset = DelphesDataSet(truth_ragged, target_ragged, device=device)
    return DelphesDataLoader(dataset, batch_size=batch_size, shuffle=False)


def _trainee_observables(
    card: CMSEnergyFlowDefault,
    dataloader: DelphesDataLoader,
    reco_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
) -> tuple[dict, dict]:
    """Run the trainee batch-by-batch -> (pred, target) flattened observables.

    Mirrors the validation loop in :mod:`tune_cms_fullsim.training`: for each
    batch drop the padded truth particles, run the card (which expects a flat
    ``(n_particles, n_features)`` tensor), regroup the flat ``EFlowObject``
    output back into per-event format, then extract the observable dict. The
    caller seeds the RNG before calling, since the card's momentum smearing /
    Gumbel-ST efficiency is stochastic.

    Memory note: unlike the old single-shot forward over the whole validation
    split, this iterates batches so peak memory is bounded by one batch's
    intermediate detector-sim tensors. The result is therefore NOT bit-identical
    to the single-shot output (the card draws data-sized randoms per forward, so
    batching re-chunks the RNG stream), but the pooled histograms are unchanged
    and init-vs-fitted stays comparable (same seed + fixed batch order).

    Both ``pred`` and ``target`` are returned already flattened to 1-D and
    co-indexed: per-particle observables (pt/eta/log_E/log_pt/pid) are stripped
    of padding/ghosts with the same ``pt != 0`` cut the loss uses (so the per-PID
    plots' element-aligned selection keeps working), and per-event observables
    pass through. ``*_region_counts`` keys are skipped (no predicted counterpart
    and never plotted).
    """
    acc_pred: dict[str, list[torch.Tensor]] = {}
    acc_tgt: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in dataloader:
            truth_particles = batch["truth_particles"]
            mask = torch.any(truth_particles != 0, dim=-1)
            out = card(truth_particles[mask])
            eflow_restored = restore_event_format(out["EFlowObject"], mask)
            pred = load_pflow_targets_from_tensor(eflow_restored)
            # Same acceptance cut the fit applied to the trainee output, so the
            # plotted trainee matches the target's acceptance (which carries the
            # cut from the loader). No-op when both are None.
            if reco_pt_cut is not None or abs_eta_cut is not None:
                pred = apply_reco_acceptance_cut(pred, reco_pt_cut, abs_eta_cut)
            target = {k: batch[k] for k in batch if k != "truth_particles"}
            for key in OBSERVABLES:
                if key not in pred or key not in target:
                    continue  # skips *_region_counts (no predicted counterpart)
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

    This uses the same shared helper as the intermediate plots so final
    observable PDFs also include the lower ratio axis with
    ``init/target`` and ``fitted/target``.
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
    card: CMSEnergyFlowDefault, dataloader: DelphesDataLoader
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

    Batched (one pass over ``dataloader``) to bound peak memory: each module's
    flat output is concatenated across batches, which is exactly the single-shot
    flat output because the downstream consumers (``filter_valid_rows`` /
    ``extract_variable``) are row-wise. The per-batch ``EVENT_NUMBER`` column is
    no longer globally unique after concatenation, but it is never read on this
    path, so that is harmless.
    """
    acc: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in dataloader:
            truth_particles = batch["truth_particles"]
            mask = torch.any(truth_particles != 0, dim=-1)
            out = card(truth_particles[mask])
            for module_name, tensor in out.items():
                acc.setdefault(module_name, []).append(tensor.detach().cpu())
    return {k: torch.cat(v, dim=0) for k, v in acc.items()}


def _load_target_intermediate_values(
    root_file: Path,
    module_name: str,
    var: str,
    n_events: int,
    entry_start: int = 0,
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
            entry_start=entry_start,
            entry_stop=entry_start + n_events,
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


# Final EFlowObject-by-PID plotting configuration.
_FINAL_PID_GROUPS: tuple[tuple[str, int, str], ...] = (
    ("211", 211, "charged hadron"),
    ("11", 11, "electron"),
    ("13", 13, "muon"),
    ("111", 111, "neutral hadron"),
    ("22", 22, "photon"),
)
_FINAL_PID_VARS: tuple[str, ...] = ("Eta", "PT", "P", "E")


def _final_pid_values(obs: dict[str, torch.Tensor], pid_abs: int, var: str) -> np.ndarray:
    """Extract one per-particle final observable for a given PID subgroup.

    ``obs`` is the dict returned by :func:`load_pflow_targets` or
    :func:`load_pflow_targets_from_tensor`.
    """
    if "pid" not in obs or "pt" not in obs:
        return np.empty(0, dtype=np.float64)

    pid = obs["pid"]
    pt = obs["pt"]
    eta = obs["eta"]
    # Keep only real reconstructed objects (drop zero-padding / ghosts) and
    # select one PID subgroup.
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


def _final_pid_suptitle(pid_label: str, n_events: int) -> str:
    """Two-line title for ``observables/final/<pid>/all.png``."""
    return f"Final EFlowObject: {pid_label}\n{n_events} events"


def plot_final_pid_observables(
    target: dict[str, torch.Tensor],
    pred_init: dict[str, torch.Tensor],
    pred_final: dict[str, torch.Tensor],
    output_dir: Path,
    n_events: int,
) -> None:
    """Render final observable overlays per PID subgroup.

    For each PID subgroup, writes one PNG per available variable in
    ``_FINAL_PID_VARS`` and a stitched ``all.png`` under
    ``<output_dir>/<pid_name>/``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Writing final per-PID plots to {output_dir}")

    n_total_var_pngs = 0
    n_total_all_pngs = 0

    for pid_name, pid_abs, pid_label in _FINAL_PID_GROUPS:
        pid_dir = output_dir / pid_name
        pid_dir.mkdir(parents=True, exist_ok=True)
        n_written_this_pid = 0

        for var in _FINAL_PID_VARS:
            target_np = _final_pid_values(target, pid_abs=pid_abs, var=var)
            init_np = _final_pid_values(pred_init, pid_abs=pid_abs, var=var)
            final_np = _final_pid_values(pred_final, pid_abs=pid_abs, var=var)

            if target_np.size == 0 and init_np.size == 0 and final_np.size == 0:
                continue

            plot_comparison_with_ratio(
                distributions=[
                    (target_np, "target", _TARGET_COLOR),
                    (init_np, "trainee, initial", _INIT_COLOR),
                    (final_np, "trainee, fitted", _FINAL_COLOR),
                ],
                var=var,
                output_path=pid_dir / f"{var}.png",
                ratio_ylabel="ratio / target",
                legend_loc=("upper left" if var != "Eta" else "best"),
            )
            n_written_this_pid += 1
            n_total_var_pngs += 1

        # Stitch the canonical 4-variable view for this PID subgroup.
        per_var_pngs = [pid_dir / f"{var}.png" for var in _FINAL_PID_VARS]
        if any(p.exists() for p in per_var_pngs):
            wrote_all = stitch_pngs(
                image_paths=per_var_pngs,
                output_path=pid_dir / "all.png",
                title=_final_pid_suptitle(pid_label=pid_label, n_events=n_events),
                title_align="left",
            )
            if wrote_all:
                n_total_all_pngs += 1

        print(
            f"    {pid_name}: wrote {n_written_this_pid} per-var PNGs"
            f"{' + all.png' if (pid_dir / 'all.png').exists() else ''}"
        )

    print(
        f"  Wrote {n_total_var_pngs} per-var final PID plots and "
        f"{n_total_all_pngs} stitched all.png pages."
    )


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
    entry_start: int = 0,
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
                root_file,
                module_name,
                var,
                n_events=n_events,
                entry_start=entry_start,
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
        default=None,
        help=(
            "Optional cap on plotted validation events. If provided and smaller "
            "than the validation split size, only the first N validation events "
            "are used for all plots."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--plot-batch-size",
        type=int,
        default=2000,
        help=(
            "Events per trainee forward batch when building the plot histograms. "
            "Lower it on a memory-tight node (e.g. a login node), raise it for "
            "speed on a compute node. Does not affect results -- the histograms "
            "pool all batches."
        ),
    )
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
        "--param-config",
        type=Path,
        default=None,
        help=(
            "The perturbed STARTING config the fit was launched from. Its 'value' "
            "fields define the honest 'trainee, initial' (before-fit) curves -- "
            "WITHOUT this the init trainee uses the card's hardcoded constructor "
            "defaults, which are neither your perturbed start nor the truth, so the "
            "'before' curve is misleading. If omitted, the path recorded in the "
            "history metadata ('param_config') is used; if that is missing/unfound, "
            "falls back to constructor defaults with a warning."
        ),
    )
    parser.add_argument(
        "--truth-pt-cut",
        type=float,
        default=None,
        help=(
            "Truth-input acceptance cut (pt >= this, |eta| <= --eta-cut). "
            "Default: the value recorded in the history metadata (falling back "
            f"to {DEFAULT_TRUTH_PT_CUT}), so plots match the fit. <= 0 disables."
        ),
    )
    parser.add_argument(
        "--reco-pt-cut",
        type=float,
        default=None,
        help=(
            "Reco acceptance cut applied to BOTH the target (at load) and the "
            "trainee output (pt >= this, |eta| <= --eta-cut). Default: the "
            "history metadata value (falling back to "
            f"{DEFAULT_RECO_PT_CUT}). <= 0 disables. The chad truncation is "
            "NEVER applied here -- final plots show untruncated spectra."
        ),
    )
    parser.add_argument(
        "--eta-cut",
        type=float,
        default=None,
        help=(
            "|eta| acceptance bound for both cuts. Default: the history "
            f"metadata value (falling back to {DEFAULT_ABS_ETA_CUT}). "
            "<= 0 disables."
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

    # Honest "before fit" baseline: the perturbed starting config the fit began
    # from (NOT the card's constructor defaults). None => fall back to defaults.
    init_snapshot, init_source = _load_init_snapshot(args.param_config, history)
    print(f"  trainee 'initial' (before-fit) params: {init_source}")

    print(f"Writing figures to {args.output_dir}")

    # ----- 1. Loss trajectory -----
    plot_loss(history, args.output_dir / "loss_trajectory.pdf")
    print("  wrote loss_trajectory.pdf")

    # ----- 2. All parameter trajectories (one subplot per tensor) -----
    plot_all_param_drift(
        history,
        truth,
        args.output_dir / "param_drift_all.pdf",
    )
    print("  wrote param_drift_all.pdf")

    # ----- 3. Observable histograms (target vs init vs final) -----
    # Fast path: compute the default contiguous validation block boundaries,
    # then read ONLY that block from ROOT. This avoids loading and densifying
    # the full dataset just to split it afterward.
    import uproot

    with uproot.open(str(args.root_file)) as f:
        n_total_events = int(f["event_tree"].num_entries)
    val_entry_start = int(0.7 * n_total_events)  # default train_fraction
    n_val_events = int(0.9*n_total_events) - val_entry_start
    n_plot_events = n_val_events
    if (
        args.n_events_for_plots is not None
        and args.n_events_for_plots > 0
        and args.n_events_for_plots < n_val_events
    ):
        n_plot_events = int(args.n_events_for_plots)

    arrays = load_cms_flow_root(
        args.root_file,
        n_events=n_plot_events,
        entry_start=val_entry_start,
    )
    # Ragged loaders + a batched dataloader (same memory-light path as the tuning
    # loop): the truth/target are never densified to the global max multiplicity,
    # and each trainee forward below runs one batch at a time. The big ``arrays``
    # dict is the largest remaining transient -- free it once the dataloader owns
    # the ragged tensors.
    device = torch.device("cpu")

    # Acceptance cuts: CLI value if given, else the value the FIT ran with
    # (history metadata), else the package default -- so plots match the fit
    # without extra flags. <= 0 on the CLI disables the respective cut.
    meta = history.get("metadata", {})

    def _resolve_cut(cli_value: float | None, meta_key: str, default: float) -> float | None:
        if cli_value is not None:
            return cli_value if cli_value > 0 else None
        if meta_key in meta:
            return meta[meta_key]  # may legitimately be None (= disabled)
        return default
    truth_pt_cut = _resolve_cut(args.truth_pt_cut, "truth_pt_cut", DEFAULT_TRUTH_PT_CUT)
    reco_pt_cut = _resolve_cut(args.reco_pt_cut, "reco_pt_cut", DEFAULT_RECO_PT_CUT)
    abs_eta_cut = _resolve_cut(args.eta_cut, "eta_cut", DEFAULT_ABS_ETA_CUT)
    print(
        f"  [filter] truth cut: pt >= {truth_pt_cut}, |eta| <= {abs_eta_cut} | "
        f"reco cut (target+trainee): pt >= {reco_pt_cut}, |eta| <= {abs_eta_cut} "
        "| no chad truncation in plots"
    )

    val_loader = _build_val_dataloader(
        arrays,
        args.plot_batch_size,
        device,
        truth_pt_cut=truth_pt_cut,
        reco_pt_cut=reco_pt_cut,
        abs_eta_cut=abs_eta_cut,
    )
    del arrays
    gc.collect()

    print(
        "  plotting on validation split: "
        f"{n_plot_events} events (entries {val_entry_start}:{val_entry_start + n_plot_events})"
    )

    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    # Set the init trainee to the perturbed STARTING config (the honest before-fit
    # baseline). Without this it would keep the card's constructor defaults.
    if init_snapshot is not None:
        _set_trainee_from_snapshot(trainee, init_snapshot)
    # The target observables come out of the same batched pass as pred_init (they
    # are read from each batch's target keys), so they are flattened/co-indexed
    # identically to the predictions.
    pred_init, target = _trainee_observables(
        trainee, val_loader, reco_pt_cut=reco_pt_cut, abs_eta_cut=abs_eta_cut
    )

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
    # Re-run the fitted card over the same fixed-order batches; the target is
    # identical to the first pass, so discard it.
    pred_final, _ = _trainee_observables(
        trainee, val_loader, reco_pt_cut=reco_pt_cut, abs_eta_cut=abs_eta_cut
    )

    plot_observable(
        "PT",
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
        "Eta",
        _obs_values(target, "eta"),
        _obs_values(pred_init, "eta"),
        _obs_values(pred_final, "eta"),
        edges=np.linspace(-5.0, 5.0, 51),
        xlabel=r"PF object $\eta$",
        output_path=args.output_dir / "observable_eta.pdf",
    )
    print("  wrote observable_eta.pdf")

    plot_observable(
        "ht",
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
        "log_ht",
        _obs_values(target, "log_ht"),
        _obs_values(pred_init, "log_ht"),
        _obs_values(pred_final, "log_ht"),
        edges=np.linspace(4.5, 7.5, 51),
        xlabel=r"PF scalar $\log\,H_\mathrm{T}$",
        output_path=args.output_dir / "observable_log_ht.pdf",
    )
    print("  wrote observable_log_ht.pdf")

    plot_observable(
        "multiplicity",
        _obs_values(target, "multiplicity"),
        _obs_values(pred_init, "multiplicity"),
        _obs_values(pred_final, "multiplicity"),
        edges=np.linspace(0.0, 300.0, 61),
        xlabel=r"PF objects per event",
        output_path=args.output_dir / "observable_multiplicity.pdf",
    )
    print("  wrote observable_multiplicity.pdf")

    per_pid_output_dir = args.output_dir / "PID"
    plot_final_pid_observables(
        target=target,
        pred_init=pred_init,
        pred_final=pred_final,
        output_dir=per_pid_output_dir,
        n_events=n_plot_events,
    )

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
        trainee_dbg = CMSEnergyFlowDefault(debug=True, learnable=True).to(device)
        if init_snapshot is not None:
            _set_trainee_from_snapshot(trainee_dbg, init_snapshot)
        init_outputs = _trainee_intermediate_outputs(trainee_dbg, val_loader)

        _set_trainee_from_snapshot(trainee_dbg, best_params)
        torch.manual_seed(args.seed)
        final_outputs = _trainee_intermediate_outputs(trainee_dbg, val_loader)

        plot_intermediate_observables(
            root_file=args.root_file,
            trainee_init_outputs=init_outputs,
            trainee_final_outputs=final_outputs,
            output_dir=args.output_dir / "intermediate",
            n_events=n_plot_events,
            entry_start=val_entry_start,
        )

    print(f"Done. {len(list(args.output_dir.glob('*.pdf')))} figures in {args.output_dir}.")


if __name__ == "__main__":
    main()
