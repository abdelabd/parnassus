r"""Optuna hyperparameter search for the CMS TorchDelphes card fit.

The single-fit CLI (:mod:`.cli`) needs a hand-picked starting config and per-group
learning rates, and the converged result is sensitive to both. This module wraps
the *same* Adam fit in an Optuna study: each trial samples

- one ABSOLUTE learning rate per parameter GROUP (resolution / scale /
  efficiency) -- ``lr`` and ``lr_scale`` are exactly degenerate (Adam only ever
  sees their product), so the search parameterizes the product directly,
- the ``photon_merge_radius`` (continuous; its 0 limit is the merger off;
  ``--mode fullsim`` only -- ``--mode delphes`` forces the merger OFF and does
  not search it, Delphes having no supercluster-scale merging), and
- one value per entry of the ``optuna_config.yaml`` ``constants:`` block.

Every scalar NOT listed in ``constants:`` starts at its card constructor default
and is fitted by Adam -- unless the optional ``parameters:`` block (a per-scalar
``{trainable: bool}`` mask, unlisted = true) marks it ``trainable: false``, which
pins it at that default. Every scalar that IS listed in ``constants:`` is a
per-trial CONSTANT (``trainable: false``), either pinned (``{value}``) or
TPE-sampled (``{low, high, init}``). The shipped block lists the five ``HadronFractions``
logits: their count-level effects are invisible to Adam (the soft-count gate
reads a DETACHED energy), but they are visible to the study, which is scored on
the val-loss VALUE -- count terms included.

Each trial materializes a normal full-cover ``{value, trainable, lr_scale}``
param config (with ``lr_scale`` holding the group's absolute lr, i.e.
``global_lr = 1``), runs the fit (:func:`.training.fit_card_to_fullsim`), and is
scored by its best (minimum) validation loss. A TPE (Bayesian) sampler proposes
the trials and a median pruner stops clearly-unpromising ones early via the
per-epoch val-loss trajectory.

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

Seed trial
----------
The per-entry ``init:`` fields (on every ``search.lr`` group, on
``photon_merge_radius``, and on each sampled entry of ``constants:``) define
**one seed trial**, enqueued via ``study.enqueue_trial``: trial 0 runs the
known-good learning rates on the believed-truth constants at the calibrated
radius. It is fully pinned, so it is an exactly reproducible baseline rather
than the truth physics crossed with an arbitrary optimizer draw, and it needs no
external file -- with the shipped ``init:`` values that trial is exactly the
CMS-default baseline card. On resume the seed is not
re-enqueued while a waiting or evaluated seed trial exists; a seed that crashed
mid-fit (FAIL / zombie RUNNING) is enqueued again. Note the seed trial COUNTS
toward ``--n-trials``.

``--init-config <param config .yaml>`` is an optional OVERRIDE for the fitted
(unlisted) scalars: values found in that file replace the card defaults those
parameters start from, which is how you refine from a previously converged card.
It never affects the constants block.

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
(float64), which scales with the per-rank ``search.batch_size``; keep it to what
fits one GPU (≤ 4096 on a 40 GB card).

Run with
``python -m parnassus.torch_delphes.tune_cms_fullsim.optuna_search ...``.
"""

from __future__ import annotations

import argparse
import gc
import math
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
from optuna.trial import TrialState
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.PhotonClusterMerger import PhotonClusterMerger

from .comet_utils import end_comet_experiment, init_comet_experiment
from .config import (
    DEFAULT_ABS_ETA_CUT,
    DEFAULT_MODE,
    DEFAULT_RECO_PT_CUT,
    DEFAULT_TRUTH_PT_CUT,
    MODE_CHOICES,
)
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
from .runner import load_split_datasets, resolve_acceptance_cuts, write_history_json
from .training import fit_card_to_fullsim

# The three lr groups, mirroring param_config.default_lr_scale: resolution
# coefficients (a/b/c softplus), scale (tanh) and efficiency (sigmoid +
# rate_raw) parameters each get their own absolute learning rate.
LR_GROUPS: tuple[str, ...] = ("resolution", "scale", "efficiency")

# The open interval a *trainable* logit (efficiency / fraction) must start inside
# (a saturated sigmoid init has a vanishing gradient). Aliased from param_config
# so the --init-config override validator uses the exact bound
# load_param_config enforces (single source of truth). Constants are exempt:
# trainable: false parameters never move, so a boundary value is fine.
LOGIT_INIT_MIN = pc._TRAINABLE_LOGIT_MIN  # noqa: SLF001
LOGIT_INIT_MAX = pc._TRAINABLE_LOGIT_MAX  # noqa: SLF001

# SimpleCalorimeter bypasses the ECal tower/rescale path for hadron fractions
# below _CHAD_BYPASS_MAX (C++ parity: the CMS card's 0.0 routes charged-hadron
# tracks straight to EFlowTrack). Between it and _CHAD_ACTIVE_MIN the ECal path
# is ON but the tower is always sub-threshold, which imprints spurious pT spikes
# at ~2-3 GeV plus a 4-9 GeV depletion that C++ Delphes does not produce.
_CHAD_BYPASS_MAX = 1.0e-4
_CHAD_ACTIVE_MIN = 5.0e-3


def _group_of(base: str) -> str:
    """Return the lr group of a parameter (by its base name).

    Mirrors :func:`param_config.default_lr_scale`: ``scale`` (tanh scales),
    ``efficiency`` (sigmoid efficiencies/fractions and the softplus ``rate_raw``),
    or ``resolution`` (every other softplus coefficient: a/b/c_*).

    Raises
    ------
    ValueError
        If ``base`` matches no known transform. A new card parameter whose name
        misses every :func:`param_config.param_transform_kind` suffix would
        otherwise be silently filed under ``resolution`` -- and, worse, treated
        as ``identity`` by ``to_physical`` / ``to_raw``, so its configs would
        store the RAW value while claiming to be physical.
    """
    kind = pc.param_transform_kind(base)
    if kind == "scale":
        return "scale"
    if kind == "logit" or base.endswith("rate_raw"):
        return "efficiency"
    if kind == "softplus":
        return "resolution"
    raise ValueError(
        f"{base!r} matches no known parameter transform (kind={kind!r}), so it "
        f"cannot be assigned an lr group. Register its name suffix in "
        f"param_config.param_transform_kind first."
    )


def _short(key: str) -> str:
    """Compact log label for a scalar key.

    Returns
    -------
    str
        The key's last dotted component.
    """
    return key.rsplit(".", 1)[-1]


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


def _parse_sampled(name: str, spec: dict) -> tuple[float, float, bool, float]:
    """Validate a ``{low, high, log, init}`` sampled-constant spec.

    ``init`` is REQUIRED: it is the value the seed trial runs, so a spec without
    one would silently leave trial 0 to the sampler.

    Returns
    -------
    (low, high, log, init)
    """
    low, high, log = _parse_range(name, spec)
    if "init" not in spec:
        raise SystemExit(
            f"{name}: a sampled entry needs an 'init' value (the seed trial runs "
            f"it). Add e.g. 'init: {0.5 * (low + high):g}', or pin the parameter "
            "with {value: ...}."
        )
    init = float(spec["init"])
    if not low <= init <= high:
        raise SystemExit(f"{name}: init {init} is outside its range [{low}, {high}].")
    return low, high, log, init


def load_search_config(path: str | Path) -> tuple[dict, dict]:
    """Load and validate an ``optuna_config.yaml``.

    Fails fast (raising before the expensive data load) if the file is malformed,
    a range is invalid, or the ``constants`` block names a key the card does not
    have. Validation applies the MERGED config (card defaults for every unlisted
    scalar, a representative in-range value for every listed one) to a probe
    card, so the full-cover invariant and the ``to_raw`` guards are exercised at
    load time.

    The optional ``parameters:`` block is a per-scalar trainable mask
    (``{key: {trainable: bool}}``; unlisted keys default to ``true``). A key
    marked ``trainable: false`` that is not in ``constants`` is folded in here as
    a constant pinned at its card default, so every consumer sees ONE frozen set.

    Returns
    -------
    (search, constants)
        ``search`` is the study/hyperparameter block; ``constants`` maps each
        frozen scalar key to its spec (``{value}`` pinned or ``{low, high, log,
        init}`` TPE-sampled). Both are per-trial CONSTANTS; every scalar absent
        from ``constants`` is fitted, starting from its card default.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "constants" not in raw or "search" not in raw:
        raise SystemExit(
            f"{path}: expected a mapping with top-level 'search' and 'constants' "
            "sections plus an optional 'parameters' trainable mask (see "
            "param_configs/optuna_config.yaml). Listed 'constants' entries are "
            "per-trial constants; everything else is fitted from the card defaults "
            "unless 'parameters' marks it trainable: false."
        )
    search = raw["search"] or {}
    constants = raw["constants"] or {}
    probe = CMSEnergyFlowDefault(debug=False, learnable=True)
    defaults = pc.card_default_config(probe)
    frozen = _parse_trainable_mask(path, raw.get("parameters") or {}, constants, set(defaults))
    constants = {**{k: {"value": defaults[k]["value"]} for k in frozen}, **constants}
    try:
        _validate_search_config(path, search, constants, probe)
    except ValueError as e:
        # _parse_range's message names the offending key but not the file.
        raise SystemExit(f"{path}: {e}") from e
    return search, constants


def _parse_trainable_mask(
    path: str | Path, parameters: dict, constants: dict, card_keys: set[str]
) -> set[str]:
    """Keys the ``parameters:`` mask freezes (``trainable: false``) beyond ``constants``.

    Returns
    -------
    set[str]
        Scalar keys to pin at their card default. Unknown keys and a
        ``trainable: true`` on a ``constants`` entry are contradictions and fail.
    """
    if not isinstance(parameters, dict):
        raise SystemExit(f"{path}: 'parameters' must be a mapping of scalar_key -> {{trainable}}.")
    unknown = sorted(set(parameters) - card_keys)
    if unknown:
        raise SystemExit(f"{path}: 'parameters' names keys the card does not have: {unknown}")
    frozen: set[str] = set()
    for key, spec in parameters.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("trainable"), bool):
            raise SystemExit(f"{path}: parameters.{key} must be {{trainable: true|false}}.")
        if spec["trainable"] and key in constants:
            raise SystemExit(
                f"{path}: parameters.{key} is trainable: true but also listed in "
                "'constants' (a per-trial constant is never trainable)."
            )
        if not spec["trainable"] and key not in constants:
            frozen.add(key)
    return frozen


def _validate_search_config(
    path: str | Path, search: dict, constants: dict, probe: CMSEnergyFlowDefault
) -> None:
    """Fail fast on a malformed ``search`` / ``constants`` pair (see caller)."""
    # Absolute per-group Adam learning rates. `lr` and the old `lr_scale` were
    # exactly degenerate (select_trainable only ever uses their product), so the
    # search parameterizes the product and materialized configs carry it with
    # global_lr = 1.
    lr = search.get("lr", {})
    for g in LR_GROUPS:
        if g not in lr:
            raise SystemExit(f"{path}: search.lr is missing the '{g}' group.")
        # `init` is required here for the same reason as on a sampled constant:
        # it is what the seed trial runs. Without it trial 0 drew random lrs, so
        # the "baseline" was the believed-truth physics crossed with an arbitrary
        # optimizer -- in the 18-trial mee round that landed 11th of 18.
        _parse_sampled(f"search.lr.{g}", lr[g])

    if "batch_size" in search:
        raise SystemExit(
            f"{path}: 'search.batch_size' was renamed to 'search.global_batch_size' and "
            "its meaning CHANGED: it is now the batch summed over all ranks, and each "
            "rank uses global_batch_size // world_size. The old per-rank reading made a "
            "study silently depend on the GPU count -- 4 ranks x 2048 meant a global "
            "batch of 8192 and ~10x fewer Adam updates than the same config on 1 GPU."
        )
    bs = search.get("global_batch_size")
    if not isinstance(bs, int) or isinstance(bs, bool) or bs <= 0:
        raise SystemExit(
            f"{path}: search.global_batch_size must be a positive integer (got {bs!r})."
        )

    # PhotonClusterMerger radius: a continuous per-trial constant whose R -> 0
    # limit IS the merger off (an isolated photon is a cluster of one and passes
    # through bit-identical, so small R is the identity).
    pmr = search.get("photon_merge_radius")
    if not isinstance(pmr, dict):
        raise SystemExit(
            f"{path}: search.photon_merge_radius must be a "
            "{low, high, init} range, e.g. {low: 0.0, high: 0.1, init: 0.045}."
        )
    r_lo, r_hi, _log, _init = _parse_sampled("search.photon_merge_radius", pmr)
    if r_lo < 0 or not math.isfinite(r_hi):
        raise SystemExit(
            f"{path}: search.photon_merge_radius must be a finite range with "
            f"low >= 0 (got [{r_lo}, {r_hi}])."
        )

    card_keys = set(pc.card_default_config(probe))
    unknown = sorted(set(constants) - card_keys)
    if unknown:
        raise SystemExit(f"{path}: 'constants' names keys the card does not have: {unknown}")
    if set(constants) == card_keys:
        raise SystemExit(
            f"{path}: 'constants' (plus the 'parameters' trainable: false mask) covers "
            "every card scalar, so nothing would be fitted. List only the parameters "
            "you want frozen at a per-trial value."
        )

    # Per-constant guards, then a merged probe apply (coverage + to_raw).
    for key, spec in constants.items():
        base = key.split("[", 1)[0]
        values = (
            [float(spec["value"])]
            if "value" in spec
            else list(_parse_sampled(key, spec)[:2])  # both endpoints must be representable
        )
        for v in values:
            try:
                pc.to_raw(base, v)
            except ValueError as e:
                raise SystemExit(f"{path}: {key!r} value {v} invalid -- {e}") from e
        # The whole INTERVAL must miss the artifact zone, not just its endpoints:
        # a range straddling it (1e-5 .. 0.5) would sample straight into it.
        if base.endswith("chad_logit") and (
            min(values) < _CHAD_ACTIVE_MIN and max(values) > _CHAD_BYPASS_MAX
        ):
            raise SystemExit(
                f"{path}: {key!r} spans ({_CHAD_BYPASS_MAX}, {_CHAD_ACTIVE_MIN}), "
                "where the ECal rescale path is ON but the tower is always "
                "sub-threshold -- that imprints spurious ~2-3 GeV pT spikes and a "
                f"4-9 GeV depletion. Stay at/below {_CHAD_BYPASS_MAX} (the C++ "
                f"bypass) or at/above {_CHAD_ACTIVE_MIN}."
            )

    merged = pc.card_default_config(probe)
    for key, spec in constants.items():
        merged[key] = {**merged[key], "value": _representative(key, spec), "trainable": False}
    try:
        pc.apply_param_config(probe, merged)
    except ValueError as e:
        raise SystemExit(f"{path}: invalid 'constants' block -- {e}") from e


def coupled_photon_constants(constants: dict) -> list[str]:
    """Sampled constants that fight ``photon_merge_radius`` over photon yield.

    ``photon_logit`` and ``k0l_logit`` both move the photon<->NH balance, as the
    radius does, so their optima are correlated. They are NOT degenerate -- only
    the fractions move energy into the HCal stream, and merging HARDENS the
    photon spectrum (merged = sum) where a leak SOFTENS it, so the per-pid shape
    and ``hcal_nh`` count terms can separate them. Expect a ridge and judge the
    study on the response surface, not on the single best trial.

    Returns
    -------
    list[str]
        The sampled (not pinned) coupled keys; empty when none are searched.
    """
    return [
        k
        for k in ("HadronFractions.photon_logit", "HadronFractions.k0l_logit")
        if k in constants and "value" not in constants[k]
    ]


def _representative(key: str, spec: dict) -> float:
    """In-range value of a constant spec, for fail-fast validation.

    Returns
    -------
    float
        The pinned value, or the range's (geometric) midpoint.
    """
    if "value" in spec:
        return float(spec["value"])
    low, high, log, _init = _parse_sampled(key, spec)
    return (low * high) ** 0.5 if log else 0.5 * (low + high)


def apply_init_overrides(defaults: dict[str, dict], init_cfg: dict, source: str) -> list[str]:
    """Override fitted-scalar start values from an ``--init-config`` param config.

    ``defaults`` (from :func:`param_config.card_default_config`) is mutated in
    place; only keys present in BOTH are touched, so a partial file is fine and
    the constants block is untouched (the caller passes the fitted subset).
    Overridden logits are re-checked against the trainable window, because unlike
    a constant these parameters DO have to move.

    Returns
    -------
    list[str]
        The keys that were overridden.
    """
    changed = []
    for key, spec in init_cfg.items():
        if key not in defaults:
            continue
        base = key.split("[", 1)[0]
        value = float(spec["value"])
        if pc.param_transform_kind(base) == "logit":
            blo, bhi = pc.logit_bounds(base)
            vmin = blo + LOGIT_INIT_MIN * (bhi - blo)
            vmax = blo + LOGIT_INIT_MAX * (bhi - blo)
            if not vmin < value < vmax:
                raise SystemExit(
                    f"{source}: {key!r} = {value} is outside the trainable-logit "
                    f"window ({vmin}, {vmax}); it is a FITTED parameter here, so a "
                    "saturated init would have no gradient. Freeze it in the "
                    "constants block instead."
                )
        defaults[key]["value"] = value
        changed.append(key)
    return changed


def seed_trial_pending(trials) -> bool:
    """True when the study still needs its ``init:``-valued seed trial enqueued.

    A seed trial (marked with the ``warm_start_config`` user attr at enqueue
    time) counts as delivered when it is WAITING (it will run this time) or
    COMPLETE/PRUNED (it was evaluated). A FAIL seed -- or a zombie RUNNING one
    left behind by a killed driver; only rank 0 runs trials, so at start-up no
    trial can genuinely be running -- was never evaluated, and the seed must be
    re-enqueued. This is also why ``enqueue_trial(skip_if_exists=True)`` cannot
    replace this check: optuna matches the fixed params against trials in ANY
    state, including FAIL, and would silently drop the retry.

    Returns
    -------
    bool
        ``True`` if no delivered seed trial exists in ``trials``.
    """
    delivered = (TrialState.WAITING, TrialState.COMPLETE, TrialState.PRUNED)
    return not any(
        "warm_start_config" in t.user_attrs and t.state in delivered for t in trials
    )


# =============================================================================
# Per-trial sampling
# =============================================================================


def sample_trial(
    trial: optuna.Trial, search: dict, constants: dict, defaults: dict[str, dict]
) -> tuple[dict[str, dict], dict[str, float]]:
    """Sample one trial's learning rates and constants into a full param config.

    ``defaults`` is :func:`param_config.card_default_config` (optionally
    ``--init-config``-overridden): every scalar it holds that the ``constants``
    block does not claim is FITTED, starting from that value.

    Returns
    -------
    (flat_cfg, group_lr)
        ``flat_cfg`` is the full-cover materialized ``{key: {value, trainable,
        lr_scale}}`` config -- the exact format
        :func:`param_config.load_param_config` returns, ready for
        :func:`param_config.apply_param_config` /
        :func:`param_config.select_trainable` (with ``global_lr=1``, since
        ``lr_scale`` already holds the group's absolute lr).
    """
    group_lr = {}
    for g in LR_GROUPS:
        lo, hi, log, _init = _parse_sampled(f"search.lr.{g}", search["lr"][g])
        group_lr[g] = trial.suggest_float(f"lr[{g}]", lo, hi, log=log)

    flat_cfg: dict[str, dict] = {}
    for key, default in defaults.items():
        spec = constants.get(key)
        if spec is None:
            value, trainable = default["value"], True
        elif "value" in spec:
            value, trainable = float(spec["value"]), False
        else:
            lo, hi, log, _init = _parse_sampled(key, spec)
            value, trainable = trial.suggest_float(key, lo, hi, log=log), False
        flat_cfg[key] = {
            "value": value,
            "trainable": trainable,
            "lr_scale": group_lr[_group_of(key.split("[", 1)[0])],
        }
    return flat_cfg, group_lr


def sample_photon_merge_radius(trial: optuna.Trial, search: dict) -> float | None:
    """Per-trial PhotonClusterMerger radius; ``None`` means merger OFF.

    Sampled continuously from ``search.photon_merge_radius``. The R -> 0 limit
    IS the merger off (an isolated photon is a cluster of one and passes through
    bit-identical), so the range needs no off-sentinel; ``None`` is returned only
    for an exact 0, which ``PhotonClusterMerger`` would reject.

    R is deliberately a per-trial CONSTANT, not a gradient-fitted parameter:
    within a trial all physics params converge self-consistently at that R, and
    trials are compared on converged validation loss -- which sidesteps the
    count-channel degeneracy a learnable R would face at unconverged
    resolutions (design doc section 6, M3).
    """
    lo, hi, log = _parse_range("search.photon_merge_radius", search["photon_merge_radius"])
    radius = trial.suggest_float("photon_merge_radius", lo, hi, log=log)
    return radius if radius > 0 else None


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
        "--init-config",
        type=Path,
        default=None,
        help=(
            "Optional param-config YAML ({value, trainable, lr_scale} per scalar, e.g. a "
            "previous round's materialized_config.yaml). OVERRIDES the card-default "
            "start value of every FITTED scalar it names, so a study can refine from a "
            "converged card; keys in the constants block are ignored. Default: start "
            "every fitted scalar at its card constructor default."
        ),
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help=(
            "Number of Optuna trials (overrides search.n_trials in the config). The "
            "enqueued seed trial COUNTS toward this number."
        ),
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
    # v2: the constants restructure renamed every optuna param (lr/lr_scale[g] ->
    # lr[g], categorical radius -> continuous), so a pre-restructure study cannot
    # be resumed meaningfully -- TPE would file its trials under a different
    # parameter group and silently ignore them.
    parser.add_argument("--study-name", type=str, default="tune_cms_fullsim_v2")
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
        "--mode",
        choices=MODE_CHOICES,
        default=DEFAULT_MODE,
        help=(
            "Target flavour: fullsim (default) applies the acceptance cuts + chad "
            "truncation below and searches the photon merge radius; delphes turns "
            "the cuts/truncation off (ignoring those flags) and forces the photon "
            "merger OFF -- the radius is not searched (search.photon_merge_radius "
            "is validated but ignored). See the tune_cms_fullsim CLI help."
        ),
    )
    parser.add_argument(
        "--truth-pt-cut",
        type=float,
        default=DEFAULT_TRUTH_PT_CUT,
        help=(
            "Truth-input acceptance: pt >= this and |eta| <= --eta-cut, all "
            f"species. Default {DEFAULT_TRUTH_PT_CUT} (no-op on the _selected "
            "files). <= 0 disables the pt part."
        ),
    )
    parser.add_argument(
        "--reco-pt-cut",
        type=float,
        default=DEFAULT_RECO_PT_CUT,
        help=(
            "Reco acceptance on BOTH target and trainee (all classes): pt >= "
            "this and |eta| <= --eta-cut; also gates the count terms and sets "
            "the n_truth_chad truncation-ceiling floor. Default "
            f"{DEFAULT_RECO_PT_CUT}. <= 0 disables the pt part. Losses are not "
            "comparable across different settings."
        ),
    )
    parser.add_argument(
        "--eta-cut",
        type=float,
        default=DEFAULT_ABS_ETA_CUT,
        help=(
            "|eta| acceptance bound shared by the truth and reco cuts (and the "
            f"calo count regions). Default {DEFAULT_ABS_ETA_CUT}. <= 0 disables."
        ),
    )
    parser.add_argument(
        "--no-chad-truncation",
        action="store_true",
        help=(
            "Disable the per-event truth-ceiling charged-hadron truncation "
            "(on by default; see the tune_cms_fullsim CLI help)."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Per-trial early stopping; epochs of no val improvement before stopping (<=0 disables). Default 10.",  # noqa: E501
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=0,
        help=(
            "Per-trial ReduceLROnPlateau patience; <=0 DISABLES lr decay, the default "
            "here. The study searches the per-group lr, so decaying it mid-fit would "
            "make the sampled value a mere starting point -- and the decay only goes "
            "down, which silently rescues too-high lrs and biases the search."
        ),
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

    search, constants = load_search_config(args.optuna_config)
    n_trials = args.n_trials if args.n_trials is not None else int(search.get("n_trials", 40))
    sampler_seed = int(search.get("seed", 0))

    # The config fixes the GLOBAL batch; each rank takes an equal slice. Adam
    # steps once per global batch, so this is what keeps the number of updates
    # per epoch -- and therefore the fit itself -- identical on 1 GPU and on N.
    global_batch_size = int(search["global_batch_size"])
    if global_batch_size % world_size:
        raise SystemExit(
            f"search.global_batch_size ({global_batch_size}) must be divisible by the "
            f"number of ranks ({world_size}); otherwise the ranks would take unequal "
            "batches. Pick a multiple, or launch a different --nproc-per-node."
        )
    batch_size = global_batch_size // world_size

    # Start values of the FITTED scalars: card constructor defaults, optionally
    # overridden from --init-config. Built once and reused by every trial.
    defaults = pc.card_default_config(CMSEnergyFlowDefault(debug=False, learnable=True))
    if args.init_config is not None:
        if not args.init_config.exists():
            raise SystemExit(f"--init-config {args.init_config} does not exist.")
        # This view SHARES its spec dicts with `defaults`, so the override lands
        # there directly; filtering exists only to keep the constants out of reach.
        fitted = {k: v for k, v in defaults.items() if k not in constants}
        overridden = apply_init_overrides(
            fitted, pc.load_param_config(args.init_config), str(args.init_config)
        )

    # Resolve --mode + the acceptance-cut args (shared with the tuning CLI via runner).
    truth_pt_cut, reco_pt_cut, abs_eta_cut, truncate_chads = resolve_acceptance_cuts(args)
    # delphes: Delphes has no supercluster-scale photon merging -> merger OFF and
    # the radius is NOT searched (search.photon_merge_radius stays validated by
    # load_search_config but is ignored).
    search_radius = args.mode != "delphes"

    r_lo, r_hi, _r_log, r_init = _parse_sampled(
        "search.photon_merge_radius", search["photon_merge_radius"]
    )
    n_sampled = sum(1 for s in constants.values() if "value" not in s)
    log(
        f"[optuna] world_size={world_size} device={device} n_trials={n_trials} "
        f"batch={global_batch_size} global ({batch_size}/rank) loss={args.loss!r} "
        f"pid_weighting={args.pid_weighting!r} mode={args.mode} "
        f"truth_pt_cut={truth_pt_cut} reco_pt_cut={reco_pt_cut} "
        f"eta_cut={abs_eta_cut} chad_truncation={'ON' if truncate_chads else 'OFF'} "
        f"photon_merge_radius={f'[{r_lo}, {r_hi}]' if search_radius else 'OFF (delphes)'} "
        f"lr_decay={'ON' if args.lr_scheduler_patience > 0 else 'OFF'}"
    )
    log(
        f"[optuna] search space: 3 group lrs + {'radius + ' if search_radius else ''}"
        f"{n_sampled} sampled constant(s); {len(defaults) - len(constants)} scalars "
        f"fitted from {'--init-config-overridden ' if args.init_config else ''}card "
        f"defaults, {len(constants)} frozen"
    )
    if args.init_config is not None:
        log(f"[optuna] init override: {len(overridden)} fitted scalars from {args.init_config}")
    if search_radius and r_lo < r_hi:
        log(
            "[optuna] NOTE: the radius is searched -- median pruning pools "
            "val-loss across radii, so a slow-adapting radius can be pruned "
            "before it converges. Judge on COMPLETE trials' best values."
        )
    coupled = coupled_photon_constants(constants)
    if coupled and search_radius:
        log(
            "[optuna] NOTE: sampling " + ", ".join(coupled) + " together with "
            "photon_merge_radius -- all move the photon<->NH balance, so their "
            "optima are correlated (separable via the photon shape + hcal_nh "
            "count terms, but it costs trials). Judge on the response surface "
            "(the importances printed at the end), not on the single best trial."
        )

    # Load the data ONCE per rank (reused across all trials; under DDP a
    # DistributedSampler shards it per epoch).
    log(f"[optuna] loading events from {args.root_file} ...")
    t0 = time.perf_counter()
    train_dataset, val_dataset = load_split_datasets(
        args.root_file,
        n_events=args.n_events,
        device=device,
        truth_pt_cut=truth_pt_cut,
        reco_pt_cut=reco_pt_cut,
        abs_eta_cut=abs_eta_cut,
        truncate_chads=truncate_chads,
    )
    log(
        f"[optuna] loaded {len(train_dataset)} train / {len(val_dataset)} val events "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    if truncate_chads:
        log(
            f"[filter] mean n_truth_chad (train split) = "
            f"{float(train_dataset.n_truth_chad.mean()):.2f} -- the per-event "
            "ceiling the reco chads are truncated to"
        )
    # Adam steps once per global batch, so this -- not the epoch count -- is how
    # much fitting a trial actually gets. Logged because a too-large global batch
    # starves the fit in a way the loss curve alone makes hard to spot.
    updates_per_epoch = math.ceil(len(train_dataset) / world_size / batch_size)
    log(
        f"[optuna] {updates_per_epoch} Adam updates/epoch "
        f"(~{updates_per_epoch * args.n_steps} per trial at --n-steps {args.n_steps}); "
        f"halve search.global_batch_size to double them"
    )

    def _run_fit_collective(
        cfg: dict,
        round_dir: Path,
        trial: optuna.Trial | None,
        merge_radius: float | None,
        comet_exp: object | None = None,
    ) -> tuple[dict, bool]:
        """Fit one trial's config; run by EVERY rank in lockstep (DDP if world_size>1).

        Returns ``(history, pruned)``. Rank 0 uses the result; the other ranks
        discard it. Every cross-rank collective (gradient all-reduce, the
        intermediate-plot all-gather, and the per-epoch prune broadcast below) is
        matched across ranks because all ranks run the same epochs.

        ``merge_radius`` is the trial's (possibly sampled) PhotonClusterMerger
        radius -- an explicit argument, and part of the DDP payload, so every
        rank builds the identical card. ``comet_exp`` is the trial's Comet
        experiment (rank 0 only; the mirror-loop ranks pass ``None``), forwarded
        to the fit so each epoch of the trial is logged under that trial's own
        experiment.
        """
        _reclaim_gpu(device)
        torch.manual_seed(args.seed)  # identical init + smearing on every rank
        trainee = CMSEnergyFlowDefault(
            debug=False,
            learnable=True,
            # Harmonize the differentiable count terms with the reco acceptance
            # cut (tracking expected counts + calo soft counts).
            count_pt_min=reco_pt_cut,
            count_abs_eta_max=abs_eta_cut,
            # Supercluster-scale photon merging (per-trial constant; the
            # ecal_photon count term is recomputed from the merged clusters
            # inside the card).
            photon_merger=(
                PhotonClusterMerger(merge_radius) if merge_radius is not None else None
            ),
        ).to(device)
        pc.apply_param_config(trainee, cfg)
        # cfg's lr_scale already holds each group's ABSOLUTE lr, so global_lr = 1.
        params_to_train, param_groups = pc.select_trainable(trainee, cfg, global_lr=1.0)
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
            reco_pt_cut=reco_pt_cut,
            reco_abs_eta_cut=abs_eta_cut,
            truncate_chads=truncate_chads,
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
                payload["cfg"], Path(payload["round_dir"]), trial=None,
                merge_radius=payload["photon_merge_radius"],
            )
        _cleanup_distributed()
        return

    # Rank 0 owns the (in-memory) study and drives the search.
    args.output_base.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        flat_cfg, group_lr = sample_trial(trial, search, constants, defaults)
        trial_merge_radius = sample_photon_merge_radius(trial, search) if search_radius else None
        round_dir = args.output_base / f"round_{trial.number}"
        round_dir.mkdir(parents=True, exist_ok=True)
        _dump_flat_config(flat_cfg, round_dir / "materialized_config.yaml")
        lrs = ", ".join(f"{g}={v:.2e}" for g, v in group_lr.items())
        sampled = ", ".join(
            f"{_short(k)}={flat_cfg[k]['value']:.3g}"
            for k, s in constants.items()
            if "value" not in s
        )
        log(
            f"[optuna] trial {trial.number}: lr={{{lrs}}} "
            f"merge_radius={trial_merge_radius} constants={{{sampled}}}"
        )
        # Hand this trial's config to the other ranks before fitting it together.
        if world_size > 1:
            dist.broadcast_object_list(
                [{"stop": False, "cfg": flat_cfg, "round_dir": str(round_dir),
                  "photon_merge_radius": trial_merge_radius}],
                src=0,
            )
        # One Comet experiment PER TRIAL, named "<--comet-name>_<trial number>"
        # (default "optuna_adam_0", "optuna_adam_1", ...). Created here rather
        # than once for the study so each trial's loss curves stay separable.
        comet_exp = init_comet_experiment(
            name=f"{args.comet_name}_{trial.number}",
            params={
                **trial.params,  # the sampled point: lr[*], radius, constants
                "trial_number": trial.number,
                "study_name": args.study_name,
                "batch_size": batch_size,
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
                flat_cfg, round_dir, trial,
                merge_radius=trial_merge_radius, comet_exp=comet_exp,
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
            # global == the Adam batch; batch_size is the per-rank slice. Two runs
            # are only comparable at equal global_batch_size (it sets the number of
            # optimizer updates per epoch, and so the amount of fitting done).
            "global_batch_size": global_batch_size,
            "batch_size": batch_size,
            "updates_per_epoch": updates_per_epoch,
            # Absolute per-group Adam lrs; the materialized config carries them in
            # its lr_scale field, so global_lr is 1 by construction.
            "lr_groups": group_lr,
            "global_lr": 1.0,
            # plot_fit_results reads this to draw the honest before-fit baseline.
            "param_config": str(round_dir / "materialized_config.yaml"),
            "constants": {k: flat_cfg[k]["value"] for k in constants},
            "trainable_params": sorted(k for k, spec in flat_cfg.items() if spec["trainable"]),
            "trial_number": trial.number,
            "optuna_config": str(args.optuna_config),
            "world_size": world_size,
            "early_stopping_patience": max(0, args.early_stopping_patience),
            "lr_scheduler_patience": max(0, args.lr_scheduler_patience),
            # --mode + acceptance cuts + truncation (resolved values; None =
            # disabled). Losses are NOT comparable across different settings.
            "mode": args.mode,
            "truth_pt_cut": truth_pt_cut,
            "reco_pt_cut": reco_pt_cut,
            "eta_cut": abs_eta_cut,
            "chad_truncation": truncate_chads,
            # THIS trial's radius (sampled when the scan block is active; None =
            # merger off). plot_fit_results resolves the per-round merger from it.
            "photon_merge_radius": trial_merge_radius,
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
    # One study == one --mode: the mode fixes the selection AND whether the radius
    # is a search dimension, so cross-mode trials are incomparable (and the old
    # seed would suppress the new one). Studies predating the attr were fullsim.
    study_mode = study.user_attrs.get("mode", "fullsim" if n_existing else args.mode)
    if study_mode != args.mode:
        if world_size > 1:  # release the mirror-loop ranks before bailing out
            dist.broadcast_object_list([{"stop": True}], src=0)
        raise SystemExit(
            f"[optuna] study {args.study_name!r} was run with --mode {study_mode}; "
            f"use a different --study-name for --mode {args.mode}."
        )
    study.set_user_attr("mode", args.mode)
    # Seed trial (rank 0 only -- ranks != 0 returned into the mirror loop above):
    # the config's `init:` values, i.e. the believed-truth constants at the
    # calibrated radius. (Re-)enqueued whenever the study has no
    # WAITING/COMPLETE/PRUNED seed yet -- on a fresh study, and on one whose
    # previous seed crashed mid-fit (FAIL, or a zombie RUNNING trial after a hard
    # kill; optuna never re-runs those).
    if not seed_trial_pending(study.trials):
        log("[optuna] seed trial already waiting or evaluated; NOT re-enqueueing")
    else:
        # Trial 0 is fully determined: known-good lrs, believed-truth constants,
        # calibrated radius (fullsim). That makes it an exactly reproducible
        # baseline rather than the truth physics plus an arbitrary optimizer draw.
        enqueue_params = {f"lr[{g}]": float(search["lr"][g]["init"]) for g in LR_GROUPS}
        enqueue_params.update(
            {key: float(spec["init"]) for key, spec in constants.items() if "value" not in spec}
        )
        if search_radius:
            enqueue_params["photon_merge_radius"] = r_init
        study.enqueue_trial(
            enqueue_params,
            user_attrs={"warm_start_config": f"{args.optuna_config}#init"},
        )
        seeded = ", ".join(f"{_short(k)}={v:g}" for k, v in enqueue_params.items())
        log(f"[optuna] seed: trial {n_existing} enqueued with {seeded} (fully pinned)")
    try:
        study.optimize(objective, n_trials=n_trials)
    finally:
        # ALWAYS release the mirror-loop ranks -- an exception escaping the
        # study would otherwise leave every non-zero rank blocked in
        # broadcast_object_list until the collective timeout kills the job.
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
    # The constants are correlated by construction (photon fraction, k0l and the
    # merge radius all move the photon<->NH balance), so the single best trial
    # can sit anywhere along a ridge. The importances say which dimensions the
    # study could actually resolve -- read them before trusting one trial.
    _log_param_importances(study, log)
    log("[optuna] validate with:")
    log(
        f"  python -m parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results "
        f"--history {args.history_path} --root-file {args.root_file} "
        f"--output-dir doc/figures --debug"
    )
    _cleanup_distributed()


def _log_param_importances(study: optuna.Study, log) -> None:
    """Print the parameter importances, or why they are unavailable.

    Prefers optuna's default fANOVA evaluator and falls back to PedAnova, which
    needs no scikit-learn (an optional dependency here). Needs >= 2 COMPLETE
    trials with differing params; never fatal -- the study's results are already
    written by the time this runs.
    """
    for label, kwargs in (
        ("fANOVA", {}),
        ("PedAnova", {"evaluator": optuna.importance.PedAnovaImportanceEvaluator()}),
    ):
        try:
            importances = optuna.importance.get_param_importances(study, **kwargs)
        except (ImportError, RuntimeError, ValueError) as e:
            last = e
            continue
        if not importances or not any(importances.values()):
            # Too few COMPLETE trials to separate the dimensions (PedAnova scores
            # against a top-quantile baseline, so a handful of trials yields
            # nothing) -- say so rather than print a bare header.
            log(
                f"[optuna] parameter importances ({label}): not resolvable from "
                f"{len([t for t in study.trials if t.value is not None])} completed "
                "trial(s); run more."
            )
            return
        log(f"[optuna] parameter importances ({label}, fraction of val-loss variance):")
        for name, value in importances.items():
            log(f"    {name:44s} {value:.3f}")
        return
    log(f"[optuna] parameter importances unavailable: {last}")


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
