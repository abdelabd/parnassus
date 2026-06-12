"""Optimizer setup and the Adam fit loop for ``tune_cms_fullsim``.

This module holds the training machinery proper:

- :func:`fit_card_to_fullsim` is the Adam optimization loop that fits the
  trainee card to a fixed target observable dict, using the per-parameter Adam
  groups built by
  :func:`parnassus.torch_delphes.param_config.select_trainable`.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.param_config import to_physical

from .config import OBSERVABLES
from .data import restore_event_format, load_pflow_targets_from_tensor
from .distributed import _is_main
from .loss import per_event_wasserstein_loss

# =============================================================================
# Fit loop
# =============================================================================


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
    intermediate_plot_dir : str | Path | None
        If set (and not ``""``), write a multi-page PDF per epoch
        (``intermediate_epoch_<step>.pdf``, one observable per page)
        comparing the trainee prediction to the full-sim target on the
        validation set, with each observable's soft-hist MSE in the page
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
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)

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
            # Differentiable per-region expected charged-hadron count -- the honest
            # gradient signal for ChargedHadronTrackingEfficiency.eff_logits.
            pred_observables["chad_expected_counts"] = out["ChargedHadronExpectedCounts"]

            # get the target from batch
            target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}
            loss = per_event_wasserstein_loss(
                pred_observables, target_observables
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
                pred_observables["chad_expected_counts"] = out["ChargedHadronExpectedCounts"]

                target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}
                val_loss = per_event_wasserstein_loss(
                    pred_observables, target_observables
                )
                val_loss_acc += val_loss.detach()

                # Accumulate the flattened, padding/ghost-stripped values for
                # each observable (same cut the loss uses: pt != 0 for 2-D
                # per-particle obs; 1-D per-event obs pass through). Detached to
                # CPU so memory stays flat and concatenation across batches with
                # different max_n_objects is safe.
                if collect_obs:
                    for key in OBSERVABLES:
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
        elif early_stopping_enabled:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                # Always render the final (early-stopped) epoch, even when it is
                # not a scheduled plot_every epoch.
                if collect_obs and not rendered:
                    _render_intermediate(acc_pred, acc_tgt, step, print_val_loss)
                if _is_main(rank):
                    tqdm.write(f"Early stopping at step {step} with val_loss {print_val_loss:.4e}")
                break

    return history
