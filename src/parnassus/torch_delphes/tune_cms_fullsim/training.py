"""Optimizer setup and the Adam fit loop for ``tune_cms_fullsim``.

This module holds the training machinery proper:

- :func:`_classify_parameter` / :func:`build_parameter_groups` bucket the 66
  learnable parameters into four Adam groups (scales / efficiency / fractions
  / resolution) with independent learning rates (the matchers and default LRs
  live in :mod:`.config`).
- :func:`fit_card_to_fullsim` is the Adam optimization loop that fits the
  trainee card to a fixed target observable dict.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .config import (
    DEFAULT_BIN_EDGES,
    DEFAULT_OBS_WEIGHTS,
    _DEFAULT_LR,
    _DEFAULT_LR_EFFICIENCY,
    _DEFAULT_LR_FRACTIONS,
    _DEFAULT_LR_RESOLUTION,
    _DEFAULT_LR_SCALES,
    _EFFICIENCY_SUFFIXES,
    _FRACTION_FRAGMENTS,
    _SCALE_SUFFIXES,
)
from .data import restore_event_format, load_pflow_targets_from_tensor
from .distributed import _is_dist, _is_main
from .loss import multi_observable_loss_distributed

# =============================================================================
# Parameter groups for per-group learning rates
# =============================================================================
#
# The 66 learnable parameters live in parameter spaces with very different
# natural scales, so a single global Adam learning rate is a poor fit. We
# instead set each group's Adam learning rate to ``base_lr * ratio_group``,
# where ``base_lr`` is the global ``--lr`` magnitude and ``ratio_group`` is a
# dimensionless per-group ratio (``--lr-<group>``). The default ratios are
# ``scales : efficiency : fractions : resolution = 1 : 1 : 1 : 0.1`` because:
#
# - Resolution coefficients ``a_raw, b_raw, c_E, c_S, c_N`` are softplus-
#   wrapped positive scalars whose post-softplus values range from ~1e-3 to
#   ~3. A raw Adam step can blow up a tiny coefficient by orders of magnitude
#   in one step, so they step ~10x slower (ratio 0.1).
# - Region scales ``scale_raw`` sit at 0 in the raw space and are bounded via
#   ``tanh``; a raw step moves the bounded output only mildly, which is safe
#   and responsive (ratio 1).
# - Efficiency logits ``eff_logits`` map to probabilities via sigmoid and sit
#   at finite values like ``logit(0.95)=2.94``; a raw step moves the
#   probability only slightly, the right magnitude for PF efficiency fits
#   (ratio 1).
# - Hadron-fraction logits behave like efficiency logits (ratio 1).
#
# The function below maps each named parameter in a learnable card to one
# of the four groups so that :func:`fit_card_to_fullsim` can build a
# ``torch.optim.Adam`` with per-group learning rates ``base_lr * ratio_group``.
# (The suffix/fragment matchers, ``base_lr`` and the default ratios live in
# ``config.py``.)


def _classify_parameter(name: str) -> str:
    """Return the parameter-group key for a given fully-qualified name.

    Returns
    -------
    str
        One of ``"scales"``, ``"efficiency"``, ``"fractions"``,
        ``"resolution"``.
    """
    if any(name.endswith(s) for s in _SCALE_SUFFIXES):
        return "scales"
    if any(frag in name for frag in _FRACTION_FRAGMENTS):
        return "fractions"
    if any(name.endswith(s) for s in _EFFICIENCY_SUFFIXES):
        return "efficiency"
    return "resolution"


def build_parameter_groups(
    card: CMSEnergyFlowDefault,
    lr: float = _DEFAULT_LR,
    lr_scales: float = _DEFAULT_LR_SCALES,
    lr_resolution: float = _DEFAULT_LR_RESOLUTION,
    lr_efficiency: float = _DEFAULT_LR_EFFICIENCY,
    lr_fractions: float = _DEFAULT_LR_FRACTIONS,
) -> list[dict]:
    """Split ``card.parameters()`` into Adam parameter groups by type.

    The effective Adam learning rate of each group is ``lr * ratio_group``:
    ``lr`` is the global magnitude (``--lr``) and the four ``lr_<group>``
    arguments are dimensionless per-group *ratios* (``--lr-<group>``). With
    the default ratios (1 / 1 / 1 / 0.1) and ``lr = 1e-3``, the effective LRs
    are scales/efficiency/fractions = 1e-3 and resolution = 1e-4.

    Returns
    -------
    list[dict]
        A list of four ``{"params": [...], "lr": float, "name": str}``
        dicts, suitable to pass directly to ``torch.optim.Adam``.
        Empty groups are filtered out so the optimizer doesn't complain
        about an empty ``params`` list.
    """
    buckets: dict[str, list[nn.Parameter]] = {
        "scales": [],
        "resolution": [],
        "efficiency": [],
        "fractions": [],
    }
    for name, p in card.named_parameters():
        buckets[_classify_parameter(name)].append(p)
    ratios = {
        "scales": lr_scales,
        "resolution": lr_resolution,
        "efficiency": lr_efficiency,
        "fractions": lr_fractions,
    }
    groups: list[dict] = []
    for key, params in buckets.items():
        if not params:
            continue
        groups.append({"params": params, "lr": lr * ratios[key], "name": key})
    return groups


# =============================================================================
# Fit loop
# =============================================================================


def fit_card_to_fullsim(
    card: CMSEnergyFlowDefault | DDP,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    n_steps: int = 100,
    lr: float | None = None,
    beta: float = 0.15,
    log_every: int = 10,
    parameters_to_train: list[nn.Parameter] | None = None,
    bin_edges: dict[str, torch.Tensor] | None = None,
    observable_weights: dict[str, float] | None = None,
    lr_scales: float = _DEFAULT_LR_SCALES,
    lr_resolution: float = _DEFAULT_LR_RESOLUTION,
    lr_efficiency: float = _DEFAULT_LR_EFFICIENCY,
    lr_fractions: float = _DEFAULT_LR_FRACTIONS,
    snapshot_parameters: bool = False,
    rank: int = 0,
    device: torch.device = torch.device("cpu"),
    intermediate_plot_dir: str | Path | None = None,
    plot_every: int = 1,
) -> dict[str, list[float]]:
    """Run Adam on ``card`` to match ``target_observables``.

    Each step runs the trainee once over every training batch and steps
    Adam per batch. The target observables are read from the ROOT file
    once (into the dataloaders) and re-used on every step.

    When ``parameters_to_train`` is ``None`` this function automatically
    builds four parameter groups (scales, resolution, efficiency,
    fractions) whose effective learning rates are ``lr * ratio_group``
    (the global magnitude ``lr`` times the per-group ratio ``lr_*``),
    because the four groups have very different natural step sizes. Pass a
    non-None ``parameters_to_train`` list to get a single group at the
    global ``lr`` value (backwards-compatible with earlier callers that
    fit a small focused subset).

    Parameters
    ----------
    lr : float | None
        Global learning-rate magnitude. When ``parameters_to_train`` is
        ``None`` the four parameter groups get ``lr * lr_<group>``; when a
        ``parameters_to_train`` list is given, the single group uses ``lr``
        directly (the caller is expected to have folded in any per-group
        ratio). Falls back to ``_DEFAULT_LR`` when ``None``.
    snapshot_parameters : bool
        If True, the history dict will additionally contain a
        ``"parameters"`` list whose i-th entry is a ``{name: float}``
        dict recording every learnable parameter value after step i.
        Off by default because it is O(n_steps * 66) in memory and
        only needed for plotting parameter-drift trajectories.
    intermediate_plot_dir : str | Path | None
        If set (and not ``""``), write a multi-page PDF per epoch
        (``intermediate_epoch_<step>.pdf``, one observable per page)
        comparing the trainee prediction to the full-sim target on the
        validation set, with each observable's *unweighted* soft-hist MSE
        in the page title. Only the main rank plots. ``None``/``""``
        disables it. See :mod:`tune_cms_fullsim.intermediate_plots`.
    plot_every : int
        Save intermediate plots every ``plot_every`` epochs (default 1 =
        every epoch). The final / early-stopped epoch is always plotted.

    Returns
    -------
    dict[str, list]
        ``"step"``, ``"loss"`` (mean train loss) and ``"val_loss"``
        always present, one entry per epoch and index-aligned. If
        ``snapshot_parameters`` is True, also contains
        ``"parameters"`` (a list of ``dict[str, float]``) for offline
        plotting of the per-parameter trajectory.
    """
    base_lr = lr if lr is not None else _DEFAULT_LR
    if parameters_to_train is not None:
        param_groups: list[dict] = [
            {
                "params": list(parameters_to_train),
                "lr": base_lr,
                "name": "user",
            }
        ]
    else:
        # Always group on the underlying (unwrapped) card so the
        # parameter-name suffix matching works regardless of DDP wrapping.
        underlying = card.module if isinstance(card, DDP) else card
        param_groups = build_parameter_groups(
            underlying,
            lr=base_lr,
            lr_scales=lr_scales,
            lr_resolution=lr_resolution,
            lr_efficiency=lr_efficiency,
            lr_fractions=lr_fractions,
        )

    opt = torch.optim.Adam(param_groups)
    if _is_main(rank):
        for g in param_groups:
            print(
                f"  param group {g['name']!r}: effective lr = {g['lr']:.3e} "
                f"({len(g['params'])} tensors)"
            )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2)

    edges = bin_edges if bin_edges is not None else DEFAULT_BIN_EDGES
    weights = observable_weights if observable_weights is not None else DEFAULT_OBS_WEIGHTS

    # Per-epoch intermediate plots: resolve the output dir (None/"" disables),
    # create it once on the main rank, and remember the epoch-0 prediction so
    # later epochs can draw it as a faint "initial" reference.
    plot_dir = (
        Path(intermediate_plot_dir) if intermediate_plot_dir not in (None, "") else None
    )
    plot_every = max(1, plot_every)
    if plot_dir is not None and _is_main(rank):
        plot_dir.mkdir(parents=True, exist_ok=True)
    init_pred_by_key: dict[str, torch.Tensor] | None = None

    def _render_intermediate(
        acc_pred: dict[str, list[torch.Tensor]],
        acc_tgt: dict[str, list[torch.Tensor]],
        step: int,
        val_loss: float,
    ) -> None:
        """Concatenate the accumulated val observables and write one PDF."""
        nonlocal init_pred_by_key
        pred_by_key = {k: torch.cat(v) for k, v in acc_pred.items() if v}
        target_by_key = {k: torch.cat(v) for k, v in acc_tgt.items() if v}
        if init_pred_by_key is None:
            init_pred_by_key = {k: t.clone() for k, t in pred_by_key.items()}
        # Lazy import keeps matplotlib out of the training import path.
        from .intermediate_plots import save_intermediate_observable_plots

        save_intermediate_observable_plots(
            pred_by_key,
            target_by_key,
            edges,
            weights,
            beta,
            step,
            plot_dir,
            val_loss=val_loss,
            init_by_key=init_pred_by_key,
        )

    history: dict[str, list] = {"step": [], "loss": [], "val_loss": []}
    if snapshot_parameters:
        history["parameters"] = []

    underlying_for_snap = card.module if isinstance(card, DDP) else card

    def _snapshot() -> dict[str, float]:
        """Record the current post-transform value of every parameter.

        We store the interpretable (post-softplus / post-tanh /
        post-sigmoid) value rather than the raw parameter, because
        that's what the plots and the JINST paper tables will show.

        Returns
        -------
        dict[str, float]
            ``{name_or_name[i]: value}`` mapping, one entry per scalar
            component of every ``nn.Parameter`` on the card.
        """
        snap: dict[str, float] = {}
        for name, p in underlying_for_snap.named_parameters():
            if name.endswith(".scale_raw"):
                val = 1.0 + 0.3 * torch.tanh(p)
            elif name.endswith((".eff_logits", "_logit")):
                val = torch.sigmoid(p)
            elif name.endswith((".rate_raw", ".a_raw", ".b_raw")) or name.startswith((
                "ECal.resolution_func",
                "HCal.resolution_func",
            )):
                val = torch.nn.functional.softplus(p)
            else:
                val = p
            vflat = val.detach().flatten().tolist() if val.ndim else [float(val.detach())]
            for i, vv in enumerate(vflat):
                snap[f"{name}[{i}]" if val.ndim else name] = float(vv)
        return snap

    pbar = tqdm(
        range(n_steps),
        disable=not _is_main(rank),
        desc="tune",
        unit="step",
        dynamic_ncols=True,
    )

    min_loss = float("inf")
    patience_counter = 0
    patience = 10  # Number of steps to wait for improvement before early stopping

    for step in pbar:
        # Re-enter train mode each step: the validation block below leaves the
        # card in eval(). The current card has no train/eval-dependent layers,
        # but this keeps the train forward correct if one is ever added.
        card.train()
        loss_acc = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        for batch in train_dataloader:

            opt.zero_grad()
            truth_particles = batch["truth_particles"] # shape is (batch_size, n_particles, n_features)
            # remove the padded particles where all features are zero
            mask = torch.any(truth_particles != 0, dim=-1) # shape is (batch_size, n_particles)
            truth_particles_nonpadded = truth_particles[mask]
            
            # out["EFlowObject"] has shape (all objects, 20 features)
            out = card(truth_particles_nonpadded)

            # first restore the (events, objects, features) shape by grouping with event number
            eflow_objects = out["EFlowObject"]
            eflow_objects_restored = restore_event_format(eflow_objects, mask)
            # Then extract the observables from predicted objects
            pred_observables = load_pflow_targets_from_tensor(eflow_objects_restored)

            # get the target from batch
            target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}
            # pred = trainee_observables(out)
            loss = multi_observable_loss_distributed(
                pred_observables, target_observables, edges, beta=beta, weights=weights
            )

            loss.backward()
            opt.step()

            loss_acc += loss.detach()

        loss_acc /= len(train_dataloader)

        # Record the per-step MEAN over all train batches (mirrors the val
        # side's averaged val_loss_acc), not the noisy last-batch loss.
        print_loss = float(loss_acc)
        history["step"].append(step)
        history["loss"].append(print_loss)
        if snapshot_parameters:
            history["parameters"].append(_snapshot())
        if _is_main(rank):
            pbar.set_postfix(loss=f"{print_loss:.4e}", refresh=False)
        if _is_main(rank) and log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            tqdm.write(f"  step {step:3d}/{n_steps}  loss = {print_loss:.4e}")

        # Per-epoch intermediate plots: collect the full validation set's
        # observables on the main rank so we can render below. We collect on
        # every plot-enabled epoch (not just scheduled ones) so the early-
        # stopped epoch can always be rendered before the break.
        collect_obs = plot_dir is not None and _is_main(rank)
        plot_this_epoch = collect_obs and (step % plot_every == 0 or step == n_steps - 1)
        acc_pred: dict[str, list[torch.Tensor]] = {}
        acc_tgt: dict[str, list[torch.Tensor]] = {}

        # validation loop --- no grad, no step, just logging
        with torch.no_grad():
            card.eval()
            val_loss_acc = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            for batch in val_dataloader:
                truth_particles = batch["truth_particles"] # shape is (batch_size, n_particles, n_features)
                # remove the padded particles where all features are zero
                mask = torch.any(truth_particles != 0, dim=-1) # shape is (batch_size, n_particles)
                truth_particles_nonpadded = truth_particles[mask]
                
                out = card(truth_particles_nonpadded)

                eflow_objects = out["EFlowObject"]
                eflow_objects_restored = restore_event_format(eflow_objects, mask)
                pred_observables = load_pflow_targets_from_tensor(eflow_objects_restored)

                target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}
                val_loss = multi_observable_loss_distributed(
                    pred_observables, target_observables, edges, beta=beta, weights=weights
                )
                val_loss_acc += val_loss.detach()

                # Accumulate the flattened, padding/ghost-stripped values for
                # each observable (same cut the loss uses: pt != 0 for 2-D
                # per-particle obs; 1-D per-event obs pass through). Detached to
                # CPU so memory stays flat and concatenation across batches with
                # different max_n_objects is safe.
                if collect_obs:
                    for key in edges.keys():
                        if key not in pred_observables or key not in target_observables:
                            continue
                        pv, tv = pred_observables[key], target_observables[key]
                        if pv.ndim >= 2:
                            pv = pv[pred_observables["pt"] != 0]
                            tv = tv[target_observables["pt"] != 0]
                        else:
                            pv, tv = pv.reshape(-1), tv.reshape(-1)
                        acc_pred.setdefault(key, []).append(pv.detach().cpu())
                        acc_tgt.setdefault(key, []).append(tv.detach().cpu())
            val_loss_acc /= len(val_dataloader)
            print_val_loss = float(val_loss_acc)
            tqdm.write(f"  step {step:3d}/{n_steps}  val_loss = {print_val_loss:.4e}")

        # Record the per-step val loss aligned with step/loss/parameters above
        # (all appended before any early-stopping break, so index i refers to
        # the same epoch across every list).
        history["val_loss"].append(print_val_loss)

        # lr scheduler step
        lr_scheduler.step(val_loss_acc)

        # Save the per-epoch intermediate observable plots (scheduled epochs).
        rendered = False
        if plot_this_epoch:
            _render_intermediate(acc_pred, acc_tgt, step, print_val_loss)
            rendered = True

        # early stopping check
        if val_loss_acc < min_loss:
            min_loss = val_loss_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # Always render the final (early-stopped) epoch, even when it is
                # not a scheduled plot_every epoch.
                if collect_obs and not rendered:
                    _render_intermediate(acc_pred, acc_tgt, step, print_val_loss)
                if _is_main(rank):
                    tqdm.write(f"Early stopping at step {step} with val_loss {print_val_loss:.4e}")
                break

    return history
