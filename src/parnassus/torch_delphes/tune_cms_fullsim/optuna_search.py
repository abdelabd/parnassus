r"""Optuna hyperparameter search for the CMS TorchDelphes card fit.

The single-fit CLI (:mod:`.cli`) needs a hand-picked starting config, global
learning rate, and batch size, and the converged result is sensitive to all
three. This module wraps the *same* Adam fit in an Optuna study: each trial
samples

- one INIT value per learnable scalar (within the range given in the
  ``optuna_config.yaml`` ``parameters:`` block),
- one ``lr_scale`` per parameter GROUP (resolution / scale / efficiency),
- the global learning rate, and
- the batch size,

materializes a normal ``{value, trainable, lr_scale}`` param config, runs the
fit (:func:`.training.fit_card_to_fullsim`), and is scored by its best (minimum)
validation loss. A TPE (Bayesian) sampler proposes the trials and a median
pruner stops clearly-unpromising ones early via the per-epoch val-loss
trajectory.

Each trial writes a self-contained ``round_<n>/`` directory under
``--output-base`` (default ``doc/fit_results``)::

    doc/fit_results/round_0/materialized_config.yaml   # the concrete config used
    doc/fit_results/round_0/history.json               # {metadata, history, best_result}
    doc/fit_results/round_0/intermediate_plots/...      # per-epoch observable PDFs

After the study, the best trial's ``history.json`` is copied to ``--history-path``
(default ``doc/fit_results/all_v2.json``) so the existing validation pipeline
works unchanged::

    python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results \
        --history doc/fit_results/all_v2.json --root-file ... --output-dir doc/figures \
        --truth-config .../cms_target_default.yaml --debug

The copied history keeps ``metadata.param_config`` pointing at the best round's
``materialized_config.yaml``, so ``plot_fit_results`` draws the honest
"before-fit" baseline from the values that trial actually started from.

Resume / add more trials
------------------------
Pass ``--storage sqlite:///<path>/study.db --study-name <name>`` to make the study
**persistent**: re-running with the same storage + name loads the existing trials
and runs ``--n-trials`` MORE (so a re-run resumes an interrupted study from where
it stopped, and lets you add trials later). New trials keep numbering, so the
``round_<n>/`` dirs continue. Without ``--storage`` the study is in-memory (a fresh
run every time). To start over, delete the ``study.db`` or change ``--study-name``.

Multiple GPUs (one study, N-GPU fit per trial)
----------------------------------------------
Launch with ``torchrun`` on a node with N GPUs (single-node; forks N ranks with no
SLURM job step, so it works inside an allocation that only permits one srun task).
All N ranks join one process group (via :func:`.distributed._init_distributed`,
which also supports ``srun`` for multi-node). **Rank 0** owns the single Optuna
study and runs ``study.optimize``; for each trial it broadcasts the sampled config
to the other ranks, and all N ranks fit it **data-parallel (DDP)** in lockstep (the
already-DDP-aware :func:`.training.fit_card_to_fullsim`). So the dataset is loaded
once per rank and reused, trials run sequentially (TPE sees each result before
proposing the next), there is a single sampler (no duplicate configs), and the
study is in-memory (no shared DB). Median pruning is kept and synchronized: rank
0's prune decision is broadcast every epoch so all ranks break together (no
collective deadlock). Example for 4 GPUs::

    torchrun --standalone --nproc-per-node=4 -m \
        parnassus.torch_delphes.tune_cms_fullsim.optuna_search \
        --root-file ... --n-events -1 --n-steps 200 --n-trials 50 \
        --output-base doc/pseudodata_results \
        --history-path doc/pseudodata_results/all_optuna.json

A plain ``python -m ...optuna_search`` (no launcher) runs single-process on one
GPU and works identically. The per-trial peak memory is the calorimeter forward
(float64), which scales with the per-rank ``batch_size``; keep the ``batch_size``
choices to what fits one GPU (≤ 4096 on a 40 GB card).

Run with
``python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search ...``.
"""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import time
from pathlib import Path

import optuna
import torch
import torch.distributed as dist
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .comet_utils import end_comet_experiment, init_comet_experiment
from .dataloader import DelphesDataLoader
from .distributed import _cleanup_distributed, _init_distributed
from .loss import (
    CALO_COUNT_WEIGHT,
    COUNT_RATE_FLOOR,
    COUNT_WEIGHT,
    EVENT_WEIGHT,
    LOSS_CHOICES,
    PID_WEIGHTING_CHOICES,
)
from .runner import load_split_datasets, write_history_json
from .training import fit_card_to_fullsim

# The three lr_scale groups, mirroring param_config.default_lr_scale: resolution
# coefficients (a/b/c softplus) step slower than the scale (tanh) and efficiency
# (sigmoid + rate_raw) parameters.
LR_SCALE_GROUPS: tuple[str, ...] = ("resolution", "scale", "efficiency")

# The open interval a *trainable* logit (efficiency / fraction) init must start
# inside (a saturated sigmoid init has a vanishing gradient). Aliased from
# param_config so the search-space validator rejects out-of-window ranges with the
# exact bound param_config.load_param_config enforces (single source of truth).
LOGIT_INIT_MIN = pc._TRAINABLE_LOGIT_MIN  # noqa: SLF001
LOGIT_INIT_MAX = pc._TRAINABLE_LOGIT_MAX  # noqa: SLF001


def _group_of(base: str) -> str:
    """Return the lr_scale group of a parameter (by its base name).

    Mirrors :func:`param_config.default_lr_scale`: ``scale`` (tanh scales),
    ``efficiency`` (sigmoid efficiencies/fractions and the softplus ``rate_raw``),
    or ``resolution`` (every other softplus coefficient: a/b/c_*).
    """
    kind = pc.param_transform_kind(base)
    if kind == "scale":
        return "scale"
    if kind == "logit" or base.endswith("rate_raw"):
        return "efficiency"
    return "resolution"


def _reclaim_gpu(device: torch.device) -> None:
    """Return the previous trial's freed GPU blocks to the caching allocator.

    Called at the START of each trial: by then the previous trial's objective
    frame has returned, so its trainee / optimizer / param_groups / dataloaders
    are unreferenced and ``gc.collect`` can drop them. ``empty_cache`` then hands
    those blocks back to the allocator pool. Without this, trials that sample
    DIFFERENT batch sizes leave the pool fragmented (the freed blocks are reserved
    but not coalesced), and a later larger-batch trial can hit a spurious OOM even
    though the same batch fits in a fresh process.
    """
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# Config parsing / validation
# =============================================================================


def _parse_range(name: str, spec: dict) -> tuple[float, float, bool]:
    """Validate and unpack a ``{low, high, log}`` range spec."""
    if "low" not in spec or "high" not in spec:
        raise ValueError(f"{name}: range spec must have 'low' and 'high' (got {spec}).")
    low, high = float(spec["low"]), float(spec["high"])
    log = bool(spec.get("log"))
    if not low < high:
        raise ValueError(f"{name}: range 'low' ({low}) must be < 'high' ({high}).")
    if log and low <= 0:
        raise ValueError(f"{name}: log-scale range needs 'low' > 0 (got {low}).")
    return low, high, log


def load_search_config(path: str | Path) -> tuple[dict, dict]:
    """Load and validate an ``optuna_config.yaml``.

    Fails fast (raising before the expensive data load) if the file is malformed,
    a range is invalid, an init range falls outside a ``param_config`` guard, or
    the ``parameters`` set does not exactly cover the card's learnable scalars.

    Returns
    -------
    (search, parameters)
        ``search`` is the global study/hyperparameter block; ``parameters`` maps
        each scalar key to its init spec (either ``{low, high, log}`` for a
        sampled/trainable param or ``{value}`` for a pinned one).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "parameters" not in raw or "search" not in raw:
        raise SystemExit(
            f"{path}: expected a mapping with top-level 'search' and 'parameters' "
            "sections (see param_configs/optuna_config.yaml for the format)."
        )
    search = raw["search"] or {}
    parameters = raw["parameters"] or {}

    # Validate the global-hyperparameter search space.
    _parse_range("search.lr", search.get("lr", {}))
    bs = search.get("batch_size", {})
    if "choices" not in bs or not bs["choices"]:
        raise SystemExit(f"{path}: search.batch_size must give a non-empty 'choices' list.")
    lr_scale = search.get("lr_scale", {})
    for g in LR_SCALE_GROUPS:
        if g not in lr_scale:
            raise SystemExit(f"{path}: search.lr_scale is missing the '{g}' group.")
        _parse_range(f"search.lr_scale.{g}", lr_scale[g])

    # Coverage check: the parameters block must cover the card's scalars exactly.
    # A midpoint config applied to a probe card fails fast on a missing/extra key
    # before the expensive data load.
    probe = CMSEnergyFlowDefault(debug=False, learnable=True)
    midpoint_cfg = {
        key: {"value": _midpoint(key, spec), "trainable": "value" not in spec, "lr_scale": 1.0}
        for key, spec in parameters.items()
    }
    try:
        pc.apply_param_config(probe, midpoint_cfg)  # checks coverage (key set) + to_raw guards
    except ValueError as e:
        raise SystemExit(f"{path}: invalid 'parameters' block -- {e}") from e

    # Per-parameter init-range guards. A sampled init can land on either endpoint,
    # so BOTH must satisfy the same range guards param_config enforces -- crucially
    # the trainable-logit (0.1, 0.9) window, which lives in load_param_config (not
    # to_raw), so the midpoint apply above does NOT catch it. A range that strays
    # outside it would produce a saturated, gradient-less init AND break
    # plot_fit_results when it reloads the materialized config.
    for key, spec in parameters.items():
        base = key.split("[", 1)[0]
        kind = pc.param_transform_kind(base)
        if "value" in spec:
            try:
                pc.to_raw(base, float(spec["value"]))
            except ValueError as e:
                raise SystemExit(f"{path}: {key!r} pinned value invalid -- {e}") from e
            continue
        lo, hi, _log = _parse_range(key, spec)
        for end in (lo, hi):
            if kind == "logit":
                if not (LOGIT_INIT_MIN < end < LOGIT_INIT_MAX):
                    raise SystemExit(
                        f"{path}: {key!r} is a trainable logit; its init range "
                        f"[{lo}, {hi}] must lie strictly inside the open interval "
                        f"({LOGIT_INIT_MIN}, {LOGIT_INIT_MAX}) -- a saturated sigmoid "
                        "init has a vanishing gradient. Pin the boundary with "
                        "{value: ...} instead."
                    )
            else:
                # scale boundary (0.7, 1.3) and softplus > 0 are enforced by to_raw.
                try:
                    pc.to_raw(base, end)
                except ValueError as e:
                    raise SystemExit(
                        f"{path}: {key!r} init-range endpoint {end} invalid -- {e}"
                    ) from e

    return search, parameters


def _midpoint(key: str, spec: dict) -> float:
    """Representative in-range value of a param spec (for fail-fast validation)."""
    if "value" in spec:
        return float(spec["value"])
    low, high, log = _parse_range(key, spec)
    return (low * high) ** 0.5 if log else 0.5 * (low + high)


# =============================================================================
# Per-trial sampling
# =============================================================================


def sample_trial(
    trial: optuna.Trial, search: dict, parameters: dict
) -> tuple[dict[str, dict], float, int, dict[str, float]]:
    """Sample one full hyperparameter point for a trial.

    Returns
    -------
    (flat_cfg, lr, batch_size, group_lr_scale)
        ``flat_cfg`` is the materialized ``{key: {value, trainable, lr_scale}}``
        config (the exact format :func:`param_config.load_param_config` returns),
        ready for :func:`param_config.apply_param_config` /
        :func:`param_config.select_trainable`.
    """
    lr_lo, lr_hi, lr_log = _parse_range("search.lr", search["lr"])
    lr = trial.suggest_float("lr", lr_lo, lr_hi, log=lr_log)
    batch_size = int(trial.suggest_categorical("batch_size", list(search["batch_size"]["choices"])))

    group_lr_scale: dict[str, float] = {}
    for g in LR_SCALE_GROUPS:
        lo, hi, log = _parse_range(f"search.lr_scale.{g}", search["lr_scale"][g])
        group_lr_scale[g] = trial.suggest_float(f"lr_scale[{g}]", lo, hi, log=log)

    flat_cfg: dict[str, dict] = {}
    for key, spec in parameters.items():
        base = key.split("[", 1)[0]
        if "value" in spec:
            value = float(spec["value"])
            trainable = False
        else:
            lo, hi, log = _parse_range(key, spec)
            value = trial.suggest_float(key, lo, hi, log=log)
            trainable = True
        flat_cfg[key] = {
            "value": value,
            "trainable": trainable,
            "lr_scale": group_lr_scale[_group_of(base)],
        }
    return flat_cfg, lr, batch_size, group_lr_scale


def _dump_flat_config(flat_cfg: dict[str, dict], path: Path) -> None:
    """Write a materialized ``{value, trainable, lr_scale}`` config to YAML.

    Round-trips through :func:`param_config.load_param_config` (used by
    ``plot_fit_results`` to recover the honest before-fit baseline).
    """
    # Plain python scalars so yaml.safe_dump emits a clean, human-readable file.
    out = {
        key: {
            "value": float(spec["value"]),
            "trainable": bool(spec["trainable"]),
            "lr_scale": float(spec["lr_scale"]),
        }
        for key, spec in flat_cfg.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Entry point for ``python -m ...tune_cms_fullsim.optuna_search``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-file",
        type=Path,
        required=True,
        help="CMS full-simulation ROOT file (cms-flow schema), loaded ONCE and reused across trials.",  # noqa: E501
    )
    parser.add_argument(
        "--optuna-config",
        type=Path,
        default=Path(pc.__file__).resolve().parent / "param_configs" / "optuna_config.yaml",
        help="YAML search space (search: + parameters: sections). See param_configs/optuna_config.yaml.",  # noqa: E501
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Number of Optuna trials (overrides search.n_trials in the config; default 50).",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("doc/fit_results"),
        help="Base dir; each trial writes <base>/round_<n>/{materialized_config.yaml,history.json,intermediate_plots/}.",  # noqa: E501
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=Path("doc/fit_results/all_v2.json"),
        help="Where the BEST trial's history.json is copied (the file plot_fit_results consumes).",
    )
    parser.add_argument("--n-events", type=int, default=-1)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="torch RNG seed (fixed across trials, so trials differ only by hyperparameters).",
    )
    parser.add_argument(
        "--shuffle-split",
        action="store_true",
        help=(
            "Randomly permute the event order BEFORE the 70/20/10 train/val/test "
            "split. The split is contiguous, so by default train/val/test are "
            "blocks of the ROOT file in file order -- which biases the val score "
            "whenever the file is ordered (per-seed generation parts concatenated "
            "by the merge, a multi-gun merged sample, an HT-sorted production). "
            "Off by default so existing runs reproduce exactly."
        ),
    )
    parser.add_argument(
        "--shuffle-split-seed",
        type=int,
        default=0,
        help=(
            "Seed for --shuffle-split's permutation. Kept separate from --seed so "
            "the train/val split can be held fixed while the training RNG varies "
            "(and vice versa). Drawn from its own generator, so every DDP rank "
            "derives the same split and the training RNG stream is untouched."
        ),
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=10,
        help=(
            "Save per-round intermediate plots every N epochs (the final / "
            "early-stopped / pruned epoch is always plotted). Default 10 "
            "(per-epoch plotting x many trials is slow)."
        ),
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "Optional Optuna storage URL (e.g. sqlite:///doc/fit_results/optuna.db) "
            "for a resumable/inspectable study. Default: in-memory."
        ),
    )
    parser.add_argument("--study-name", type=str, default="tune_cms_fullsim")
    parser.add_argument(
        "--comet-name",
        type=str,
        default="optuna_adam",
        help=(
            "Base name for the per-trial Comet experiments; each trial logs as "
            "'<comet-name>_<trial number>' (default 'optuna_adam' -> "
            "'optuna_adam_0', 'optuna_adam_1', ...). One experiment per trial, so "
            "trials stay separable in the Comet UI. Requires COMET_API_KEY "
            "(workspace from COMET_WORKSPACE); without it the search runs unlogged."
        ),
    )
    parser.add_argument(
        "--comet-disabled",
        action="store_true",
        help="Disable Comet logging even when COMET_API_KEY is set.",
    )
    # Fit knobs passed straight through (defaults match the tuning CLI).
    parser.add_argument("--loss", type=str, default="wasserstein_1d", choices=list(LOSS_CHOICES))
    parser.add_argument(
        "--pid-weighting",
        type=str,
        default="sqrt_fraction",
        choices=list(PID_WEIGHTING_CHOICES),
    )
    parser.add_argument("--pid-weight-floor", type=float, default=0.0)
    parser.add_argument("--count-weight", type=float, default=COUNT_WEIGHT)
    parser.add_argument("--calo-count-weight", type=float, default=CALO_COUNT_WEIGHT)
    parser.add_argument("--count-rate-floor", type=float, default=COUNT_RATE_FLOOR)
    parser.add_argument("--event-weight", type=float, default=EVENT_WEIGHT)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Per-trial early stopping (<=0 disables, the default -- pruning handles early termination).",  # noqa: E501
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=4,
        help="Per-trial ReduceLROnPlateau patience (<=0 disables, the default -- search varies lr).",  # noqa: E501
    )
    args = parser.parse_args()

    # One process group from the launcher (`srun -n N`). A plain `python -m ...`
    # run has no launcher rank env -> world_size == 1 (single GPU, no DDP).
    rank, world_size, local_rank, device = _init_distributed()
    if rank == 0 and hasattr(sys.stdout, "reconfigure"):
        # Line-buffer stdout so progress streams live when redirected to a log file
        # (skip when stdout is a non-reconfigurable stream, e.g. pytest capture).
        sys.stdout.reconfigure(line_buffering=True)

    def log(msg: str) -> None:
        """Print on rank 0 only."""
        if rank == 0:
            print(msg)

    # Guard against a multi-process launch where torch.distributed silently failed
    # to init: without it each process would run its OWN full study -> duplicated
    # work. Covers both torchrun (WORLD_SIZE) and srun (SLURM_NTASKS).
    launcher_n = max(
        int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("SLURM_NTASKS", "1"))
    )
    if launcher_n > 1 and world_size == 1:
        raise SystemExit(
            "Launched with multiple processes but torch.distributed did not "
            "initialize. Use `torchrun --standalone --nproc-per-node=N` (single node) "
            "or `srun` (multi-node); a single process is fine for single-GPU."
        )

    if not args.root_file.exists():
        raise SystemExit(
            f"--root-file {args.root_file} does not exist. Provide a CMS full-simulation "
            "ROOT file (cms-flow schema)."
        )

    search, parameters = load_search_config(args.optuna_config)
    n_trials = args.n_trials if args.n_trials is not None else int(search.get("n_trials", 50))
    sampler_seed = int(search.get("seed", 0))

    log(
        f"[optuna] world_size={world_size} device={device} n_trials={n_trials} "
        f"loss={args.loss!r} pid_weighting={args.pid_weighting!r}"
    )

    # Load the data ONCE per rank (reused across all trials; under DDP a
    # DistributedSampler shards it per epoch).
    log(f"[optuna] loading events from {args.root_file} ...")
    t0 = time.perf_counter()
    # Loaded ONCE, outside the trial loop: every trial is therefore scored on the
    # same train/val split, shuffled or not.
    train_dataset, val_dataset = load_split_datasets(
        args.root_file,
        n_events=args.n_events,
        device=device,
        shuffle_seed=args.shuffle_split_seed if args.shuffle_split else None,
    )
    log(
        f"[optuna] loaded {len(train_dataset)} train / {len(val_dataset)} val events "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    def _run_fit_collective(
        cfg: dict,
        lr: float,
        batch_size: int,
        round_dir: Path,
        trial: optuna.Trial | None,
        comet_exp: object | None = None,
    ) -> tuple[dict, bool]:
        """Fit one trial's config; run by EVERY rank in lockstep (DDP if world_size>1).

        Returns ``(history, pruned)``. Rank 0 uses the result; the other ranks
        discard it. Every cross-rank collective (gradient all-reduce, the
        intermediate-plot all-gather, and the per-epoch prune broadcast below) is
        matched across ranks because all ranks run the same epochs.

        ``comet_exp`` is the trial's Comet experiment (rank 0 only; the mirror-loop
        ranks pass ``None``), forwarded to the fit so each epoch of the trial is
        logged under that trial's own experiment.
        """
        _reclaim_gpu(device)
        torch.manual_seed(args.seed)  # identical init + smearing on every rank
        trainee = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
        pc.apply_param_config(trainee, cfg)
        params_to_train, param_groups = pc.select_trainable(trainee, cfg, global_lr=lr)
        if not params_to_train:
            raise SystemExit("optuna_config marks no parameter trainable; nothing to optimize.")

        if world_size > 1:
            train_sampler: DistributedSampler | None = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank,
                shuffle=True, seed=args.seed, drop_last=False,
            )
            val_sampler: DistributedSampler | None = DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank,
                shuffle=False, drop_last=False,
            )
            ddp_kwargs: dict = {"find_unused_parameters": False}
            if device.type == "cuda":
                ddp_kwargs["device_ids"] = [local_rank]
                ddp_kwargs["output_device"] = local_rank
            trainee = DDP(trainee, **ddp_kwargs)
        else:
            train_sampler = None
            val_sampler = None
        train_dl = DelphesDataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler
        )
        val_dl = DelphesDataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler
        )

        state = {"pruned": False}

        def epoch_callback(step: int, val_loss: float) -> bool:
            # Rank 0 owns the pruning decision; broadcast it so ALL ranks break the
            # fit loop on the same epoch (a rank that kept going would deadlock on
            # the next collective).
            stop = False
            if rank == 0 and trial is not None:
                trial.report(val_loss, step)
                stop = trial.should_prune()
            if world_size > 1:
                flag = torch.tensor([1 if stop else 0], device=device)
                dist.broadcast(flag, src=0)
                stop = bool(flag.item())
            if stop:
                state["pruned"] = True
            return stop

        history = fit_card_to_fullsim(
            trainee,
            train_dl,
            val_dl,
            param_groups=param_groups,
            n_steps=args.n_steps,
            log_every=max(1, args.n_steps // 10),
            snapshot_parameters=True,  # plot_fit_results needs best_result.parameters
            rank=rank,
            device=device,
            intermediate_plot_dir=str(round_dir / "intermediate_plots"),
            plot_every=args.plot_every,
            early_stopping_patience=(
                args.early_stopping_patience if args.early_stopping_patience > 0 else None
            ),
            lr_scheduler_patience=(
                args.lr_scheduler_patience if args.lr_scheduler_patience > 0 else None
            ),
            count_weight=args.count_weight,
            calo_count_weight=args.calo_count_weight,
            count_rate_floor=args.count_rate_floor,
            event_weight=args.event_weight,
            loss_name=args.loss,
            pid_weighting=args.pid_weighting,
            pid_weight_floor=args.pid_weight_floor,
            epoch_callback=epoch_callback,
            comet_exp=comet_exp,
        )
        return history, state["pruned"]

    # Non-zero ranks mirror rank 0: receive each trial's config and fit it
    # collectively, until rank 0 broadcasts the stop sentinel.
    if rank != 0:
        while True:
            box: list = [None]
            dist.broadcast_object_list(box, src=0)
            payload = box[0]
            if payload.get("stop"):
                break
            _run_fit_collective(
                payload["cfg"], payload["lr"], payload["batch_size"],
                Path(payload["round_dir"]), trial=None,
            )
        _cleanup_distributed()
        return

    # Rank 0 owns the (in-memory) study and drives the search.
    args.output_base.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        flat_cfg, lr, batch_size, group_lr_scale = sample_trial(trial, search, parameters)
        round_dir = args.output_base / f"round_{trial.number}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _dump_flat_config(flat_cfg, round_dir / "materialized_config.yaml")
        log(
            f"[optuna] trial {trial.number}: lr={lr:.3e} batch={batch_size} "
            f"lr_scale={{{', '.join(f'{g}={v:.2g}' for g, v in group_lr_scale.items())}}}"
        )
        # Hand this trial's config to the other ranks before fitting it together.
        if world_size > 1:
            dist.broadcast_object_list(
                [{"stop": False, "cfg": flat_cfg, "lr": lr,
                  "batch_size": batch_size, "round_dir": str(round_dir)}],
                src=0,
            )
        # One Comet experiment PER TRIAL, named "<--comet-name>_<trial number>"
        # (default "optuna_adam_0", "optuna_adam_1", ...). Created here rather than
        # once for the study so each trial's loss curves stay separable, matching
        # how the GEBO scans name their per-round experiments.
        comet_exp = init_comet_experiment(
            name=f"{args.comet_name}_{trial.number}",
            params={
                **trial.params,  # the sampled point: lr, batch_size, lr_scale[*], inits
                "trial_number": trial.number,
                "study_name": args.study_name,
                "n_steps": args.n_steps,
                "n_events": args.n_events,
                "loss": args.loss,
                "pid_weighting": args.pid_weighting,
                "world_size": world_size,
                "round_dir": str(round_dir),
            },
            disabled=args.comet_disabled,
            rank=rank,
            log_prefix="[optuna]",
            quiet=True,  # one URL line per trial would drown the search log
        )

        try:
            history, pruned = _run_fit_collective(
                flat_cfg, lr, batch_size, round_dir, trial, comet_exp=comet_exp
            )

            val_losses = [v for v in history.get("val_loss", []) if v is not None]
            best_val = min(val_losses) if val_losses else float("inf")
            if comet_exp is not None:
                comet_exp.log_metric("best_val_loss", best_val)
                comet_exp.log_other("pruned", pruned)
                comet_exp.log_other("epochs_run", len(history.get("step", [])))
        finally:
            # Close the trial's experiment on every exit path -- including the
            # TrialPruned raised below and any mid-fit exception -- so a long study
            # never accumulates dangling live experiments.
            end_comet_experiment(comet_exp)
        metadata = {
            "n_events": args.n_events,
            "n_steps": args.n_steps,
            "lr": lr,
            "batch_size": batch_size,
            "lr_scale_groups": group_lr_scale,
            # plot_fit_results reads this to draw the honest before-fit baseline.
            "param_config": str(round_dir / "materialized_config.yaml"),
            "param_group_lrs": sorted({lr * v for v in group_lr_scale.values()}),
            "trainable_params": sorted(k for k, spec in flat_cfg.items() if spec["trainable"]),
            "trial_number": trial.number,
            "optuna_config": str(args.optuna_config),
            "world_size": world_size,
            "early_stopping_patience": max(0, args.early_stopping_patience),
            "lr_scheduler_patience": max(0, args.lr_scheduler_patience),
            # Which events landed in train vs val -- None = historic file-order split.
            "shuffle_split_seed": args.shuffle_split_seed if args.shuffle_split else None,
        }
        write_history_json(round_dir / "history.json", history, metadata)

        # Record the round dir + score so _select_best_trial can recover it even
        # for a pruned trial (study.best_trial only sees COMPLETE ones).
        trial.set_user_attr("round_dir", str(round_dir))
        trial.set_user_attr("best_val_loss", best_val)
        log(f"[optuna] trial {trial.number}: best val_loss = {best_val:.4e}  -> {round_dir}")

        if pruned:
            raise optuna.TrialPruned
        return best_val

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=1)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, group=True, n_startup_trials=10, seed=sampler_seed),
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.storage is not None,
    )
    # On RESUME (a persistent study that already has trials), re-seed the sampler by
    # the number of existing trials. Otherwise a fresh ``TPESampler(seed)`` would
    # replay the same random startup draws as the previous run and duplicate its
    # first configs while the study is still in the random-startup phase.
    n_existing = len(study.trials)
    if n_existing > 0:
        log(f"[optuna] resuming study {args.study_name!r}: {n_existing} existing trials")
        study.sampler = TPESampler(
            multivariate=True, group=True, n_startup_trials=10, seed=sampler_seed + n_existing
        )
    study.optimize(objective, n_trials=n_trials)

    # Release the mirror-loop ranks.
    if world_size > 1:
        dist.broadcast_object_list([{"stop": True}], src=0)

    # Pick the best round (prefer COMPLETE trials; fall back to the lowest recorded
    # best_val_loss so an all-pruned study still yields a usable result).
    best = _select_best_trial(study)
    if best is None:
        raise SystemExit("[optuna] no trial produced a finite validation loss; nothing to copy.")

    best_round = Path(best.user_attrs["round_dir"])
    best_val = best.user_attrs.get("best_val_loss")
    args.history_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best_round / "history.json", args.history_path)

    log("")
    log(f"[optuna] best trial: {best.number}  best val_loss = {best_val:.4e}")
    log(f"[optuna] best round dir: {best_round}")
    log(f"[optuna] copied best history -> {args.history_path}")
    log("[optuna] validate with:")
    log(
        f"  python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results "
        f"--history {args.history_path} --root-file {args.root_file} "
        f"--output-dir doc/figures --debug"
    )
    _cleanup_distributed()


def _select_best_trial(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    """Return the trial with the lowest validation loss, pruned trials included.

    Prefers the study's COMPLETE-trial best; if none completed (all pruned), falls
    back to the trial with the smallest recorded ``best_val_loss`` user attr.
    """
    complete = [
        t for t in study.trials if t.value is not None and t.user_attrs.get("round_dir")
    ]
    if complete:
        return min(complete, key=lambda t: t.value)
    scored = [
        t
        for t in study.trials
        if t.user_attrs.get("round_dir")
        and t.user_attrs.get("best_val_loss") not in {None, float("inf")}
    ]
    if scored:
        return min(scored, key=lambda t: t.user_attrs["best_val_loss"])
    return None


if __name__ == "__main__":
    main()
