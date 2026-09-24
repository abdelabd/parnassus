"""Optimizer setup and the Adam fit loop for ``tune_cms_fullsim``.

This module holds the training machinery proper:

- :func:`fit_card_to_fullsim` is the Adam optimization loop that fits the
  trainee card to a fixed target observable dict, using the per-parameter Adam
  groups built by
  :func:`parnassus.torch_delphes.param_config.select_trainable`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.param_config import to_physical

from .config import OBSERVABLES
from .data import (
    batch_event_ids,
    load_pflow_targets_from_tensor,
    restore_event_format,
)
from .distributed import _is_dist, _is_main

# =============================================================================
# Fit loop
# =============================================================================


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average ``value`` across ranks in-place; no-op when not under DDP."""
    if _is_dist():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


def _placeholder_loss(
    pred: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Stand-in for the training loss between critic_loss_plan.md steps 5a and 5c.

    Step 5a deleted the count terms and the per-pid shape losses; the critic loss
    (``critic.py``, step 5b) is wired in here in step 5c, together with the
    pred-side acceptance cut + chad truncation (fullsim mode) and the validation
    monitor. Until then the fit loop cannot run.
    """
    raise NotImplementedError(
        "training loss removed in critic_loss_plan.md step 5a; the critic loss "
        "arrives in step 5c"
    )


def fit_card_to_fullsim(
    card: CMSEnergyFlowDefault | DDP,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    param_groups: list[dict],
    n_steps: int = 100,
    log_every: int = 10,
    snapshot_parameters: bool = False,
    rank: int = 0,
    device: torch.device = torch.device("cpu"),
    intermediate_plot_dir: str | Path | None = None,
    plot_every: int = 1,
    early_stopping_patience: int | None = 10,
    lr_scheduler_patience: int | None = 4,
    lr_scheduler_factor: float = 0.5,
    reco_pt_cut: float | None = None,
    reco_abs_eta_cut: float | None = None,
    truncate_chads: bool = False,
    epoch_callback: Callable[[int, float], bool] | None = None,
    comet_exp: "object | None" = None,
) -> dict[str, list[float]]:
    """Run Adam on ``card`` to match the target observables.

    Each step runs the trainee once over every training batch and steps
    Adam per batch. The target observables are read from the ROOT file
    once (into the dataloaders) and re-used on every step.

    ``param_groups`` are the ready-made ``torch.optim.Adam`` groups (with each
    parameter's effective learning rate already folded in); build them with
    :func:`parnassus.torch_delphes.param_config.select_trainable`.

    Parameters
    ----------
    snapshot_parameters : bool
        If True, the history dict will additionally contain a
        ``"parameters"`` list whose i-th entry is a ``{name: float}``
        dict recording every learnable parameter value after step i.
        Off by default because it is O(n_steps * 68) in memory and
        only needed for plotting parameter-drift trajectories.
    intermediate_plot_dir : str | Path | None
        If set (and not ``""``), write a multi-page PDF per epoch
        (``intermediate_epoch_<step>.pdf``, one observable per page)
        comparing the trainee prediction to the full-sim target on the
        validation set, with each observable's histogram MSE in the page
        title. Only the main rank plots. ``None``/``""`` disables it. See
        :mod:`tune_cms_fullsim.intermediate_plots`.
    plot_every : int
        Save intermediate plots every ``plot_every`` epochs (default 1 =
        every epoch). The final / early-stopped epoch is always plotted.
    early_stopping_patience : int | None
        Number of epochs with no improvement in ``val_loss`` after which
        training is stopped. Set to ``None`` (or any value ``<= 0``) to
        disable early stopping entirely; the loop will then always run
        the full ``n_steps``. Default is 10.
    lr_scheduler_patience : int | None
        Patience (epochs of no ``val_loss`` improvement) for the
        ``ReduceLROnPlateau`` learning-rate decay. Set to ``None`` (or any
        value ``<= 0``) to disable LR decay entirely and train at a constant
        lr -- recommended for single-parameter closure fits, where the
        stochastic (resampled) loss otherwise triggers premature lr collapse.
        Default is 4.
    lr_scheduler_factor : float
        Multiplicative factor applied to the lr on each plateau reduction
        (only used when ``lr_scheduler_patience`` is enabled). Default is 0.5.
    reco_pt_cut, reco_abs_eta_cut : float | None
        Reco acceptance cut applied to the PRED side only, right before the loss
        (``apply_reco_acceptance_cut``): the TARGET already carries the same cut
        statically from the loaders (``load_pflow_targets_ragged(reco_pt_cut=...,
        abs_eta_cut=...)``), so both sides of every shape term live in the same
        acceptance. Intermediate plots collect the filtered dicts, so they show
        the post-cut view on both sides. ``None`` (default) disables. Surfaced on
        the CLI as ``--reco-pt-cut`` / ``--eta-cut``. NOTE: losses are not
        comparable across different cut settings.
    truncate_chads : bool
        Per event, keep only the top-``n_truth_chad`` charged hadrons by pt on
        the PRED side (``apply_chad_truncation``; the target was truncated
        statically in the loader with ``truncate_chads=True``). The cap comes
        from the batch's ``n_truth_chad`` target key. Requires the acceptance
        cut to run first (wired below) and the ``event_ids``-aligned
        ``restore_event_format`` (also wired below). Default False.
    epoch_callback : Callable[[int, float], bool] | None
        Optional per-epoch hook called as ``epoch_callback(step, val_loss)`` right
        after the validation loss for that epoch is computed (and after any
        intermediate plot is rendered). If it returns ``True`` the fit loop breaks
        immediately -- the same clean exit as early stopping, so the partial
        ``history`` accumulated so far is still returned intact. Used by the Optuna
        search (:mod:`.optuna_search`) to report the val-loss trajectory to a trial
        and stop pruned trials early; ``None`` (default) is a no-op.
    comet_exp : comet_ml.Experiment | None
        Optional live Comet experiment (build one with
        :func:`.comet_utils.init_comet_experiment`). When set, each epoch logs the
        train/val loss and the per-group effective learning rates at
        ``step=<epoch>``; ``snapshot_parameters=True`` additionally logs every
        card parameter's physical value as ``param/<name>``. The caller owns the
        experiment's lifetime (call :func:`.comet_utils.end_comet_experiment`
        when the fit returns). Only the main rank logs, so passing the same
        experiment on every rank is safe -- but callers only build it on rank 0.
        ``None`` (default) disables logging entirely.

    Returns
    -------
    dict[str, list]
        ``"step"``, ``"loss"`` (mean train loss) and ``"val_loss"``
        always present, one entry per epoch and index-aligned. If
        ``snapshot_parameters`` is True, also contains
        ``"parameters"`` (a list of ``dict[str, float]``) for offline
        plotting of the per-parameter trajectory.
    """
    opt = torch.optim.Adam(param_groups)
    if _is_main(rank):
        for g in param_groups:
            print(
                f"  param group {g['name']!r}: effective lr = {g['lr']:.3e} "
                f"({len(g['params'])} tensors)"
            )
    # ReduceLROnPlateau halves the lr after `lr_scheduler_patience` stagnant
    # epochs. On a stochastic (resampled) loss this can collapse the lr long
    # before convergence (see the per-param closure diagnosis), so it is
    # optional: pass `lr_scheduler_patience=None` (or <= 0) to disable LR decay
    # and train at a constant lr.
    lr_scheduler = None
    if lr_scheduler_patience is not None and lr_scheduler_patience > 0:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=lr_scheduler_factor, patience=lr_scheduler_patience
        )

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
        observables = OBSERVABLES
        save_intermediate_observable_plots(
            pred_by_key,
            target_by_key,
            observables,
            step,
            plot_dir,
            val_loss=val_loss,
            init_by_key=init_pred_by_key,
        )

    def _render_with_gather(
        acc_pred: dict[str, list[torch.Tensor]],
        acc_tgt: dict[str, list[torch.Tensor]],
        step: int,
        val_loss: float,
    ) -> None:
        """Render the intermediate plot using the FULL validation set under DDP.

        Each rank only iterates its ``DistributedSampler`` shard, so its
        ``acc_pred``/``acc_tgt`` cover ~1/world_size of the validation data. To
        plot the full distribution we ``all_gather_object`` every rank's
        (CPU-tensor) shard lists and concatenate them on the main rank before
        rendering. This is a collective, so all ranks must reach it on the same
        epoch (they do: the plot-epoch / early-stop / callback breaks are decided
        from the all-reduced ``val_loss``, identically on every rank). Outside DDP
        it just renders the single-process observables directly.
        """
        if _is_dist() and dist.get_world_size() > 1:
            gathered: list = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, (acc_pred, acc_tgt))
            if not _is_main(rank):
                return
            merged_pred: dict[str, list[torch.Tensor]] = {}
            merged_tgt: dict[str, list[torch.Tensor]] = {}
            for gp, gt in gathered:
                for k, v in gp.items():
                    merged_pred.setdefault(k, []).extend(v)
                for k, v in gt.items():
                    merged_tgt.setdefault(k, []).extend(v)
            _render_intermediate(merged_pred, merged_tgt, step, val_loss)
        elif _is_main(rank):
            _render_intermediate(acc_pred, acc_tgt, step, val_loss)

    history: dict[str, list] = {"step": [], "loss": [], "val_loss": []}
    if snapshot_parameters:
        history["parameters"] = []

    underlying_for_snap = card.module if isinstance(card, DDP) else card
    trainable_params = [p for p in underlying_for_snap.parameters() if p.requires_grad]

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
            # Shared with param_config so snapshots, configs and plots all use
            # the same interpretable (post-transform) values.
            val = to_physical(name, p)
            vflat = val.detach().flatten().tolist() if val.ndim else [float(val.detach())]
            for i, vv in enumerate(vflat):
                snap[f"{name}[{i}]" if val.ndim else name] = float(vv)
        return snap

    def _comet_metric_name(*parts: str) -> str:
        """Join ``parts`` into a Comet-safe ``/``-separated metric name.

        Component labels carry characters that are awkward in a metric key
        (``211:eta``, ``ECal.resolution_func.barrel_a``, ``eff_logits[0]``); map
        anything outside ``[A-Za-z0-9_.-/]`` to ``_`` so the Comet UI groups them
        cleanly instead of silently mangling or rejecting the name.
        """
        safe = [
            "".join(ch if (ch.isalnum() or ch in "_.-") else "_" for ch in str(p))
            for p in parts
        ]
        return "/".join(s for s in safe if s)

    def _log_comet_epoch(
        step: int,
        train_loss: float,
        val_loss: float,
        snapshot: dict[str, float] | None,
    ) -> None:
        """Log one epoch's metrics to Comet (no-op when logging is off).

        Wrapped in a broad try/except on purpose: Comet is telemetry, and a
        transient network/API failure must never abort a fit that is otherwise
        progressing (these runs are hours long and hold multi-node allocations).
        """
        if comet_exp is None or not _is_main(rank):
            return
        try:
            metrics: dict[str, float] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
            # Effective per-group lr actually used this epoch (logged before the
            # ReduceLROnPlateau step below, which applies to the NEXT epoch).
            for i, g in enumerate(opt.param_groups):
                metrics[_comet_metric_name("lr", g.get("name", f"group{i}"))] = float(
                    g["lr"]
                )
            # Physical (post-transform) parameter values, so parameter drift is
            # visible in Comet without post-processing the history JSON.
            if snapshot:
                for pname, pval in snapshot.items():
                    metrics[_comet_metric_name("param", pname)] = pval
            comet_exp.log_metrics(metrics, step=step)
        except Exception as e:  # noqa: BLE001 - telemetry must not break the fit
            tqdm.write(f"  [comet] warning: metric logging failed at step {step}: {e}")

    pbar = tqdm(
        range(n_steps),
        disable=not _is_main(rank),
        desc="tune",
        unit="step",
        dynamic_ncols=True,
    )

    min_loss = float("inf")
    patience_counter = 0
    # ``early_stopping_patience`` is the number of epochs with no val_loss
    # improvement before we break. ``None`` (or any non-positive value)
    # disables early stopping entirely.
    early_stopping_enabled = (
        early_stopping_patience is not None and early_stopping_patience > 0
    )

    for step in pbar:
        # Re-enter train mode each step: the validation block below leaves the
        # card in eval(). The current card has no train/eval-dependent layers,
        # but this keeps the train forward correct if one is ever added.
        card.train()

        # Re-shuffle the per-rank shard each epoch.
        train_sampler = getattr(train_dataloader, "sampler", None)
        if isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(step)

        loss_acc = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        for batch in train_dataloader:

            opt.zero_grad()
            truth_particles = batch["truth_particles"] # shape is (batch_size, n_particles, n_features)
            # remove the padded particles where all features are zero
            mask = torch.any(truth_particles != 0, dim=-1) # shape is (batch_size, n_particles)
            truth_particles_nonpadded = truth_particles[mask]
            
            # out["EFlowObject"] has shape (all objects, N_FEATURES)
            out = card(truth_particles_nonpadded)

            # first restore the (events, objects, features) shape by grouping with
            # event number; event_ids aligns pred row i with target batch row i
            # (required by the per-event chad truncation below, harmless otherwise).
            eflow_objects = out["EFlowObject"]
            eflow_objects_restored = restore_event_format(
                eflow_objects, mask, event_ids=batch_event_ids(truth_particles, mask)
            )
            # Then extract the observables from predicted objects
            pred_observables = load_pflow_targets_from_tensor(eflow_objects_restored)

            # get the target from batch
            target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}

            loss = _placeholder_loss(pred_observables, target_observables)
            loss.backward()
            opt.step()

            loss_acc += loss.detach()

        loss_acc /= len(train_dataloader)
        loss_acc = _all_reduce_mean(loss_acc)

        # Record the per-step MEAN over all train batches (mirrors the val
        # side's averaged val_loss_acc), not the noisy last-batch loss.
        print_loss = float(loss_acc)
        history["step"].append(step)
        history["loss"].append(print_loss)
        if snapshot_parameters:
            history["parameters"].append(_snapshot())
        # Per-epoch intermediate plots: collect this rank's validation-shard
        # observables so we can render below. Under DDP every rank collects its
        # shard and `_render_with_gather` all-gathers them to the main rank, so the
        # plot uses the FULL validation set (not just rank 0's 1/world_size shard).
        # We collect on every plot-enabled epoch (not just scheduled ones) so the
        # early-stopped / pruned epoch can always be rendered before the break.
        collect_obs = plot_dir is not None
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
                eflow_objects_restored = restore_event_format(
                    eflow_objects, mask, event_ids=batch_event_ids(truth_particles, mask)
                )
                pred_observables = load_pflow_targets_from_tensor(eflow_objects_restored)
                # get the target from batch
                target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}

                val_loss = _placeholder_loss(pred_observables, target_observables)
                val_loss_acc += val_loss.detach()

            val_loss_acc /= len(val_dataloader)
            val_loss_acc = _all_reduce_mean(val_loss_acc)
            print_val_loss = float(val_loss_acc)
            if _is_main(rank):
                tqdm.write(f"  step {step:3d}/{n_steps}  val_loss = {print_val_loss:.4e}")

        # Record the per-step val loss aligned with step/loss/parameters above
        # (all appended before any early-stopping break, so index i refers to
        # the same epoch across every list).
        history["val_loss"].append(print_val_loss)

        # Comet: one point per epoch (loss curves, per-group lr and -- when
        # snapshotting -- every parameter's physical value).
        _log_comet_epoch(
            step,
            print_loss,
            print_val_loss,
            history["parameters"][-1] if snapshot_parameters else None,
        )

        # lr scheduler step (only when LR decay is enabled)
        if lr_scheduler is not None:
            lr_scheduler.step(val_loss_acc)

        # Save the per-epoch intermediate observable plots (scheduled epochs).
        rendered = False
        if plot_this_epoch:
            _render_with_gather(acc_pred, acc_tgt, step, print_val_loss)
            rendered = True

        # Per-epoch hook (e.g. Optuna trial report + median pruning). Returning
        # True breaks the loop with the same clean exit as early stopping, so the
        # partial `history` is returned intact. Render this (final) epoch first if
        # it was not a scheduled plot epoch, mirroring the early-stopping path.
        if epoch_callback is not None and epoch_callback(step, print_val_loss):
            if collect_obs and not rendered:
                _render_with_gather(acc_pred, acc_tgt, step, print_val_loss)
            if _is_main(rank):
                tqdm.write(
                    f"Stopping at step {step} via epoch_callback "
                    f"(val_loss {print_val_loss:.4e})"
                )
            break

        # early stopping check
        if val_loss_acc < min_loss:
            min_loss = val_loss_acc
            patience_counter = 0
        elif early_stopping_enabled:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                # Always render the final (early-stopped) epoch, even when it is
                # not a scheduled plot_every epoch.
                if collect_obs and not rendered:
                    _render_with_gather(acc_pred, acc_tgt, step, print_val_loss)
                if _is_main(rank):
                    tqdm.write(f"Early stopping at step {step} with val_loss {print_val_loss:.4e}")
                break

    return history
