"""Optimizer setup and the Adam fit loop for ``tune_cms_fullsim``.

This module holds the training machinery proper:

- :func:`fit_card_to_fullsim` is the Adam optimization loop that fits the
  trainee card to a fixed target observable dict, using the per-parameter Adam
  groups built by
  :func:`parnassus.torch_delphes.param_config.select_trainable`.
"""

from __future__ import annotations

import math


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
from .critics import (
    WassersteinCritic,
    build_critic_input,
    build_event_response,
    card_loss,
    cat_critic_inputs,
    critic_step,
    held_out_w1,
)
from .data import (
    batch_event_ids,
    load_pflow_targets_from_tensor,
    restore_event_format,
)
from .distributed import _is_dist, _is_main

# =============================================================================
# Fit loop
# =============================================================================


# Every fit ends with a cosine lr tail of this many epochs (see the scheduler in
# ``fit_card_to_fullsim``): it starts when the validation monitor has not improved
# for ``early_stopping_patience`` epochs, or at ``n_steps`` at the latest.
LR_TAIL_EPOCHS: int = 10


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average ``value`` across ranks in-place; no-op when not under DDP."""
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
    intermediate_plot_dir: str | Path | None = None,
    plot_every: int = 1,
    early_stopping_patience: int | None = 10,
    critic_bound: float = 1.0,
    n_critic: int = 5,
    critic_lr: float = 1e-3,
    critic_hidden: int = 128,
    critic_warmup: int = 100,
    monitor_steps: int = 300,
    truth_values: dict[str, float] | None = None,
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
        Number of epochs with no improvement in ``val_loss`` (the held-out
        Wasserstein monitor, see ``monitor_steps``) after which the cosine lr
        tail starts (``LR_TAIL_EPOCHS`` epochs, then the fit stops). Set to
        ``None`` (or any value ``<= 0``) to start the tail at ``n_steps`` instead.
        Either way every fit ends annealed, after at most ``n_steps +
        LR_TAIL_EPOCHS`` epochs; ``n_steps`` is the cap on the full-lr phase.
        Default is 10.
    critic_bound, n_critic, critic_lr, critic_hidden, critic_warmup
        The Wasserstein critic (:mod:`.critics`, critic_loss_plan.md section 3):
        ``critic_bound`` is ``R``, the norm bound on the per-object embedding = the
        EMD mass-creation penalty in standardised feature units; ``critic_hidden`` is
        the MLP width; ``critic_lr`` its Adam lr (betas (0.5, 0.9)). Before every card
        step the critic takes ``n_critic`` Adam steps on the detached copies of the
        card's own batch; ``critic_warmup`` extra steps run before the very first
        card step.
    monitor_steps : int
        Step budget of the validation monitor (:func:`.critics.held_out_w1`): after
        every epoch a FRESH critic is fitted for this many steps on half of the
        validation events and scored on the other half; its distance estimate is
        the epoch's ``val_loss`` (comparable across epochs: fixed seed, split and
        budget). Default 300.
    truth_values : dict[str, float] | None
        Physical truth value per fitted scalar (same keys as the parameter
        snapshots, e.g. ``MuonTrackingEfficiency.eff_logits[0]``), typically from
        the pseudodata generation config. When given, every epoch prints each of
        these parameters next to its truth; otherwise every scalar of the tensors
        that require grad is printed without a truth column.
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
        ``"step"``, ``"loss"`` (mean train loss) and ``"val_loss"`` (held-out
        Wasserstein monitor) always present, one entry per epoch and index-aligned. If
        ``snapshot_parameters`` is True, also contains
        ``"parameters"`` (a list of ``dict[str, float]``) for offline
        plotting of the per-parameter trajectory.
    """
    opt = torch.optim.Adam(param_groups)

    # Learning-rate schedule: every group keeps its initial lr until the cosine
    # tail starts (plateau of the validation monitor, or n_steps), then follows a
    # half cosine to ~0 over LR_TAIL_EPOCHS epochs, after which the fit stops. The
    # monitor makes one decision (the start of the tail) and the tail is fixed, so
    # its noise floor can neither collapse the lr early (the ReduceLROnPlateau
    # failure of the per-parameter closures) nor cut the annealing short (early
    # stopping under a full-run cosine). Stepped once per epoch.
    tail_start: int | None = None  # first epoch of the tail, set once below

    def lr_factor(epoch: int) -> float:
        if tail_start is None or epoch < tail_start:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * (epoch - tail_start + 1) / (LR_TAIL_EPOCHS + 1)))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    n_total = n_steps + LR_TAIL_EPOCHS

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

    # ----------------- Wasserstein critic (critics.py; critic_loss_plan.md M2) -----------------
    # Standardisation constants of the critic features: log_pt mean / std over ALL
    # target objects of the training split, fixed for the whole fit. They define the
    # ground metric that the Lipschitz bound and ``critic_bound`` refer to, so they
    # are neither per batch nor taken from the (moving) reco side. The ragged
    # dataset lists hold real objects only; under a DistributedSampler every rank
    # holds the full dataset, so the constants agree across ranks.
    log_pt_all = torch.cat(list(train_dataloader.dataset.log_pt))
    log_pt_mean = float(log_pt_all.mean())
    log_pt_std = max(float(log_pt_all.std()), 1e-6)
    # One persistent critic with its own Adam for the whole fit; float64 like the
    # card; stays in train() mode (each forward runs one spectral-norm power
    # iteration in place). Not DDP-wrapped yet (single process).
    critic = WassersteinCritic(hidden=critic_hidden, bound=critic_bound).to(
        device=device, dtype=log_pt_all.dtype
    )
    critic_opt = torch.optim.Adam(critic.parameters(), lr=critic_lr, betas=(0.5, 0.9))
    history["critic_loss"] = []  # per epoch: mean over batches of the last critic-step loss
    history["critic_stats"] = [log_pt_mean, log_pt_std]
    if _is_main(rank):
        print(
            f"  critic: R={critic_bound} hidden={critic_hidden} n_critic={n_critic} "
            f"lr={critic_lr:.1e} warmup={critic_warmup}; "
            f"log_pt standardised with mean={log_pt_mean:.4f} std={log_pt_std:.4f} "
            f"({log_pt_all.numel()} target objects)"
        )
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
        range(n_total),
        disable=not _is_main(rank),
        desc="tune",
        unit="step",
        dynamic_ncols=True,
    )

    min_loss = float("inf")
    patience_counter = 0
    # ``early_stopping_patience`` epochs without monitor improvement start the
    # cosine tail early; ``None`` (or any non-positive value) leaves the start of
    # the tail at ``n_steps``.
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

        loss_acc = torch.zeros((), dtype=torch.float64, device=device)
        critic_loss_acc = 0.0
        for batch_index, batch in enumerate(train_dataloader):
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

            # Critic inputs: live on the reco side (smearing through x, efficiency
            # coins and threshold gates through w = exp(log_w)), plain on the target.
            x_r, w_r = build_critic_input(pred_observables, log_pt_mean, log_pt_std)
            x_t, w_t = build_critic_input(target_observables, log_pt_mean, log_pt_std)
            # Per-event summed-pt response of each side against the same truth particles.
            h_r = build_event_response(pred_observables, truth_particles)
            h_t = build_event_response(target_observables, truth_particles)

            # Critic updates on the detached copies of this batch, before the card
            # step. The very first batch of the fit first warms the critic up, so the
            # first card gradient does not come from an untrained critic, and records
            # every step for the inspection below.
            first_batch = step == 0 and batch_index == 0
            if first_batch:
                warm = [
                    critic_step(critic, critic_opt, x_r, w_r, h_r, x_t, w_t, h_t, n_steps=1)
                    for _ in range(critic_warmup)
                ]
                c_steps = [
                    critic_step(critic, critic_opt, x_r, w_r, h_r, x_t, w_t, h_t, n_steps=1)
                    for _ in range(n_critic)
                ]
                c_loss = c_steps[-1]
            else:
                c_loss = critic_step(critic, critic_opt, x_r, w_r, h_r, x_t, w_t, h_t, n_steps=n_critic)
            critic_loss_acc += c_loss

            # Card step on the LIVE features and weights. The zero-weight anchor keeps
            # every trainable parameter in the graph (DDP unused-parameter guard).
            opt.zero_grad()
            loss = card_loss(critic, x_r, w_r, h_r) + 0.0 * sum(p.sum() for p in trainable_params)
            loss.backward()
            opt.step()

            loss_acc += loss.detach()

            if first_batch:
                if _is_main(rank):
                    def _fmt(vals):
                        return " ".join(f"{v:.3e}" for v in vals)
                    tqdm.write(
                        f"  [critic] warm-up ({critic_warmup} steps), loss every 10th: "
                        f"{_fmt(warm[::10])} ... last {warm[-1]:.3e}"
                    )
                    tqdm.write(f"  [critic] {n_critic} steps on the first batch: {_fmt(c_steps)}")
                    tqdm.write(f"  [card] loss = {float(loss):.4e}; |grad| per trainable parameter:")
                    for name, p in underlying_for_snap.named_parameters():
                        if p.requires_grad and p.grad is not None:
                            tqdm.write(f"      {name:<60} {float(p.grad.norm()):.3e}")

        loss_acc /= len(train_dataloader)
        loss_acc = _all_reduce_mean(loss_acc)

        # Record the per-step MEAN over all train batches (mirrors the val
        # side's averaged val_loss_acc), not the noisy last-batch loss.
        print_loss = float(loss_acc)
        history["step"].append(step)
        history["loss"].append(print_loss)
        history["critic_loss"].append(critic_loss_acc / max(1, len(train_dataloader)))
        if snapshot_parameters:
            history["parameters"].append(_snapshot())
        # Per-epoch log: card loss, critic loss, and the fitted parameters next to
        # their truth values (M3 run: no validation metric yet, see M4).
        if _is_main(rank):
            tqdm.write(
                f"  step {step:3d}/{n_total}  train loss = {print_loss:.4e}  "
                f"critic loss = {history['critic_loss'][-1]:.4e}  lr x{lr_scheduler.get_last_lr()[0] / lr_scheduler.base_lrs[0]:.3f}"
            )
            snap = history["parameters"][-1] if snapshot_parameters else _snapshot()
            fitted_tensors = {n for n, p in underlying_for_snap.named_parameters() if p.requires_grad}
            keys = list(truth_values) if truth_values else [
                k for k in snap if k.split("[")[0] in fitted_tensors
            ]
            tqdm.write(f"  step {step:3d}/{n_total}  parameter{'':<53} current |      truth | rel. dev.")
            for k in keys:
                cur = snap[k]
                if truth_values and k in truth_values:
                    tv = truth_values[k]
                    dev = f"{cur / tv - 1.0:+.3%}" if tv != 0 else f"{cur - tv:+.5f} (abs)"
                    tqdm.write(f"      {k:<60} {cur:10.5f} | {tv:10.5f} | {dev}")
                else:
                    tqdm.write(f"      {k:<60} {cur:10.5f} |        n/a |")


        # ---- validation: held-out Wasserstein monitor ----
        # Card in eval, no grad: forward the validation set once, collect the critic
        # inputs of both sides and hand them to a FRESH critic (critics.held_out_w1)
        # fitted on half of the events and scored on the other half. That distance
        # estimate is the epoch's val_loss: comparable across epochs (fixed seed,
        # split and step budget), it drives early stopping and the scheduler.
        # Under DDP every rank monitors its own shard and the values are averaged.
        # The same pass collects the flattened observables for the intermediate
        # plots (every plot-enabled epoch, so the final epoch can always be drawn).
        collect_obs = plot_dir is not None
        plot_this_epoch = collect_obs and (step % plot_every == 0 or step == n_total - 1)
        acc_pred: dict[str, list[torch.Tensor]] = {}
        acc_tgt: dict[str, list[torch.Tensor]] = {}
        card.eval()
        reco_inputs, target_inputs = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                truth_particles = batch["truth_particles"]
                mask = torch.any(truth_particles != 0, dim=-1)
                objects = restore_event_format(
                    card(truth_particles[mask])["EFlowObject"], mask, event_ids=batch_event_ids(truth_particles, mask)
                )
                pred_observables = load_pflow_targets_from_tensor(objects)
                target_observables = {k: v for k, v in batch.items() if k != "truth_particles"}
                for obs, inputs in ((pred_observables, reco_inputs), (target_observables, target_inputs)):
                    x, w = build_critic_input(obs, log_pt_mean, log_pt_std)
                    inputs.append((x, w, build_event_response(obs, truth_particles)))
                if collect_obs:
                    # Padding/ghost-stripped 1-D values per observable, on CPU so
                    # batches with different max_n_objects concatenate safely.
                    for key in OBSERVABLES:
                        if key not in pred_observables or key not in target_observables:
                            continue
                        pv, tv = pred_observables[key], target_observables[key]
                        if pv.ndim >= 2:
                            pv = pv[pred_observables["pt"] != 0]
                            tv = tv[target_observables["pt"] != 0]
                        else:
                            pv, tv = pv.reshape(-1), tv.reshape(-1)
                        acc_pred.setdefault(key, []).append(pv.cpu())
                        acc_tgt.setdefault(key, []).append(tv.cpu())
        val_w1 = held_out_w1(
            *cat_critic_inputs(reco_inputs),
            *cat_critic_inputs(target_inputs),
            hidden=critic_hidden,
            bound=critic_bound,
            lr=critic_lr,
            n_steps=monitor_steps,
            batch_size=val_dataloader.batch_size,
        )
        val_loss_acc = _all_reduce_mean(torch.tensor(val_w1, dtype=torch.float64, device=device))
        print_val_loss = float(val_loss_acc)
        if _is_main(rank):
            tqdm.write(f"  step {step:3d}/{n_total}  val W1 (held-out monitor) = {print_val_loss:.4e}")

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

        # Plateau detection on the held-out monitor: it decides (once) when the
        # cosine tail starts; the tail then runs its fixed length and ends the fit.
        if val_loss_acc < min_loss:
            min_loss = val_loss_acc
            patience_counter = 0
        else:
            patience_counter += 1
        plateau = early_stopping_enabled and patience_counter >= early_stopping_patience
        if tail_start is None and (plateau or step + 1 >= n_steps):
            tail_start = step + 1
            if _is_main(rank):
                why = (
                    f"no monitor improvement for {early_stopping_patience} epochs"
                    if plateau else f"n_steps = {n_steps} reached"
                )
                tqdm.write(f"  cosine tail: {LR_TAIL_EPOCHS} epochs from step {tail_start} ({why})")
        lr_scheduler.step()  # next epoch's multiplier (sees tail_start)

        # Per-epoch intermediate observable plots (scheduled epochs).
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
                tqdm.write(f"Stopping at step {step} via epoch_callback")
            break

        # End of the fit: the cosine tail has run its length. Always plot this
        # final epoch, whether or not it was a scheduled one.
        if tail_start is not None and step + 1 >= tail_start + LR_TAIL_EPOCHS:
            if collect_obs and not rendered:
                _render_with_gather(acc_pred, acc_tgt, step, print_val_loss)
            if _is_main(rank):
                tqdm.write(f"Done at step {step}: cosine tail finished, val_loss {print_val_loss:.4e}")
            break

    return history
