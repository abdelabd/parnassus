"""Helpers shared by the two GEBO fine-tuning stages.

:mod:`.lbfgs_finetune` and :mod:`.adam_finetune` are alternative *second stages*
of the same pipeline: both start from a GEBO run's best point (identified by its
``gebo_summary.json``) and both have to recover the settings that run used --
which live in three places of decreasing reliability (an explicit CLI override,
the summary's own ``args`` block, and the config GEBO archived under
``<run_dir>/configs/``). This module holds that recovery logic, plus the Comet
reconnection both stages use to log into the round's EXISTING GEBO experiment
rather than opening a second one.

Nothing here runs an optimizer; see the two ``*_finetune`` modules for that.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable

import yaml

try:
    import comet_ml

    _HAS_COMET = True
except ImportError:  # optional dependency, exactly as in gebo_search / comet_utils
    _HAS_COMET = False


def trainable_bases(optuna_config: Path) -> set[str]:
    """Base names GEBO optimizes (any param with a ``{low, high}`` spec).

    A parameter pinned with ``{value: ...}`` is NOT in the search space, so it is
    excluded here -- the fine-tune stage must optimize exactly the dimensions
    GEBO did, or its starting vector will not line up with the run's bounds.
    """
    with open(optuna_config) as f:
        raw = yaml.safe_load(f)
    specs = raw.get("parameters", {}) if isinstance(raw, dict) else {}
    return {k.split("[", 1)[0] for k, spec in specs.items() if "value" not in spec}


def load_archived_config(run_dir: Path):
    """Load the most recent archived run config (``configs/config_<N>.yaml``).

    GEBO snapshots the run config into ``<output_dir>/configs/``; recovering it
    lets a fine-tune stage rebuild the exact run settings even from an
    INTERMEDIATE ``gebo_summary.json`` (which carries no ``args``). Returns
    ``None`` if no snapshot is found or it fails to parse.
    """
    configs_dir = run_dir / "configs"
    if not configs_dir.is_dir():
        return None

    def _idx(p: Path) -> int:
        m = re.fullmatch(r"config_(\d+)", p.stem)
        return int(m.group(1)) if m else -1

    snaps = [p for p in configs_dir.glob("config_*.yaml") if _idx(p) >= 0]
    if not snaps:
        return None
    from .gebo_search import load_gebo_config

    try:
        return load_gebo_config(max(snaps, key=_idx))
    except SystemExit:
        return None


def make_setting_resolver(run_args: dict, cfg_ns: Any | None) -> Callable[..., Any]:
    """Build the ``_cfg(key, default)`` lookup both fine-tuners use.

    Resolution order is summary ``args`` -> archived config -> caller default;
    the CLI override sits above all of it and is applied by the caller (an
    explicitly passed flag must beat whatever the run recorded).
    """

    def _cfg(key: str, default: Any = None) -> Any:
        sv = run_args.get(key)
        if sv is not None:
            return sv
        if cfg_ns is not None:
            cv = getattr(cfg_ns, key, None)
            if cv is not None:
                return cv
        return default

    return _cfg


def reconnect_comet(run_dir: Path, *, disabled: bool = False, log_prefix: str = "[finetune]"):
    """Reattach to the Comet experiment the round's GEBO stage created.

    ``gebo_search.py`` stores its experiment key in ``<run_dir>/gebo_state.pt``
    (and reuses it across its own resumes); picking the same key up here means
    the fine-tune stage's metrics land in the SAME experiment as the GEBO
    iterations that produced its starting point, so one Comet run shows the whole
    round end to end. Callers pass a metric-name suffix (e.g. ``"_adam"``) to
    keep the two stages' curves distinct within that experiment.

    Returns ``None`` -- meaning "logging off", never an error -- when Comet is
    disabled, not installed, unkeyed, or when the GEBO stage itself did not log
    (no state file, or no key in it).
    """
    if disabled or not _HAS_COMET or not os.environ.get("COMET_API_KEY"):
        return None

    state_path = run_dir / "gebo_state.pt"
    if not state_path.exists():
        return None
    try:
        import torch

        state = torch.load(state_path, map_location="cpu", weights_only=True)
        comet_key = state.get("comet_key")
    except Exception as e:  # noqa: BLE001 - telemetry must not break the fine-tune
        print(f"{log_prefix} could not read a Comet key from {state_path}: {e}")
        return None
    if not comet_key:
        return None

    try:
        experiment = comet_ml.ExistingExperiment(
            api_key=os.environ["COMET_API_KEY"],
            previous_experiment=comet_key,
        )
    except Exception as e:  # noqa: BLE001
        print(f"{log_prefix} could not reconnect to Comet experiment {comet_key}: {e}")
        return None
    print(f"{log_prefix} logging into the round's GEBO Comet experiment: {experiment.url}")
    return experiment
