"""Optimizer setup and the Adam fit loop for ``tune_cms_fullsim``.

This module holds the training machinery proper:

- :func:`fit_card_to_fullsim` is the Adam optimization loop that fits the
  trainee card to a fixed target observable dict, using the per-parameter Adam
  groups built by
  :func:`parnassus.torch_delphes.param_config.select_trainable`. The training
  loss is selected at call time via the ``loss_name`` argument (one of
  :data:`tune_cms_fullsim.loss.LOSS_CHOICES`); the dispatcher in
  :mod:`tune_cms_fullsim.loss` resolves the name to either the sliced
  Wasserstein loss or the soft-histogram MSE loss.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.param_config import to_physical

from .data import restore_event_format, load_pflow_targets_from_tensor
from .distributed import _is_dist, _is_main
from .loss import LOSS_CHOICES, get_loss_fn

# =============================================================================
# Fit loop
# =============================================================================

# Shared dataset-partition policy used by both the training CLI and
# plot_fit_results.py:
#   - train on the first 70% of events,
#   - reserve the next 20% for plotting/holdout studies,
#   - leave the tail (last 10%) unused by this workflow.
TRAIN_FRACTION: float = 0.70
PLOT_FRACTION: float = 0.20


def contiguous_event_partitions(
    n_events: int,
    train_fraction: float = TRAIN_FRACTION,
    plot_fraction: float = PLOT_FRACTION,
) -> tuple[int, int]:
    """Return contiguous split boundaries for train and plot windows.

    Parameters
    ----------
    n_events : int
        Number of events in the dataset being partitioned.
    train_fraction : float
        Fraction assigned to the leading training block.
    plot_fraction : float
        Fraction assigned to the block immediately after training.

    Returns
    -------
    (train_end, plot_end) : tuple[int, int]
        Event-index boundaries such that:

        - training window is ``[0, train_end)``
        - plotting window is ``[train_end, plot_end)``

        with ``0 <= train_end <= plot_end <= n_events``.
    """
    if n_events < 0:
        raise ValueError(f"n_events must be >= 0, got {n_events}")
    if not (0.0 <= train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in [0,1], got {train_fraction}")
    if not (0.0 <= plot_fraction <= 1.0):
        raise ValueError(f"plot_fraction must be in [0,1], got {plot_fraction}")
    if train_fraction + plot_fraction > 1.0 + 1e-12:
        raise ValueError(
            "train_fraction + plot_fraction must be <= 1.0, got "
            f"{train_fraction + plot_fraction:.3f}"
        )

    train_end = int(train_fraction * n_events)
    plot_end = min(n_events, train_end + int(plot_fraction * n_events))
    return train_end, plot_end


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average ``value`` across ranks in-place; no-op when not under DDP.
    """
    if _is_dist():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= dist.get_world_size()
    return value


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
    early_stopping_patience: int | None = 40,
    loss_name: str = "wasserstein",
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
        Off by default because it is O(n_steps * 66) in memory and
        only needed for plotting parameter-drift trajectories.
    early_stopping_patience : int | None
        Number of epochs with no improvement in ``val_loss`` after which
        training is stopped. Set to ``None`` (or any value ``<= 0``) to
        disable early stopping entirely; the loop will then always run
        the full ``n_steps``. Default is 10.
    loss_name : str
        Selects the training loss. One of
        :data:`tune_cms_fullsim.loss.LOSS_CHOICES`:

        - ``"wasserstein"`` (default): per-pid sliced Wasserstein-2 over
          ``[log_E, log_pt, eta]`` + a down-weighted ``log(HT)`` term, via
          :func:`tune_cms_fullsim.loss.per_event_wasserstein_loss`.
        - ``"soft_hist"``: the original soft-histogram MSE loss (ported
          from commit 5cac599) summed across observables, via
          :func:`tune_cms_fullsim.loss.multi_observable_soft_hist_loss_distributed`.
          DDP-aware: per-rank histograms are summed with a differentiable
          all-reduce.

    Returns
    -------
    dict[str, list]
        ``"step"``, ``"loss"`` (mean train loss) and ``"val_loss"``
        always present, one entry per epoch and index-aligned. If
        ``snapshot_parameters`` is True, also contains
        ``"parameters"`` (a list of ``dict[str, float]``) for offline
        plotting of the per-parameter trajectory.
    """
    # Resolve the training-loss callable once, up front. Raises ValueError
    # with the valid choices if ``loss_name`` is invalid -- fail fast rather
    # than per-step.
    loss_fn = get_loss_fn(loss_name)
    if _is_main(rank):
        print(
            f"  training loss: {loss_name!r} "
            f"(choices: {list(LOSS_CHOICES)})"
        )
    opt = torch.optim.Adam(param_groups)
    if _is_main(rank):
        for g in param_groups:
            print(
                f"  param group {g['name']!r}: effective lr = {g['lr']:.3e} "
                f"({len(g['params'])} tensors)"
            )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

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
            # Shared with param_config so snapshots, configs and plots all use
            # the same interpretable (post-transform) values.
            val = to_physical(name, p)
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
            
            # out["EFlowObject"] has shape (all objects, 20 features)
            out = card(truth_particles_nonpadded)

            # first restore the (events, objects, features) shape by grouping with event number
            eflow_objects = out["EFlowObject"]
            eflow_objects_restored = restore_event_format(eflow_objects, mask)
            # Then extract the observables from predicted objects
            pred_observables = load_pflow_targets_from_tensor(eflow_objects_restored)

            # get the target from batch
            target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}
            loss = loss_fn(
                pred_observables, target_observables
            )

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
        if _is_main(rank):
            pbar.set_postfix(loss=f"{print_loss:.4e}", refresh=False)
        if _is_main(rank) and log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            tqdm.write(f"  step {step:3d}/{n_steps}  loss = {print_loss:.4e}")

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
                val_loss = loss_fn(
                    pred_observables, target_observables
                )
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

        # lr scheduler step
        lr_scheduler.step(val_loss_acc)

        # early stopping check
        if val_loss_acc < min_loss:
            min_loss = val_loss_acc
            patience_counter = 0
        elif early_stopping_enabled:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                if _is_main(rank):
                    tqdm.write(f"Early stopping at step {step} with val_loss {print_val_loss:.4e}")
                break

    return history
