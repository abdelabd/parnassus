"""Comet experiment setup shared by the Adam fit paths.

Shared by the plain Adam CLI (:mod:`.cli`) and the Optuna+Adam search
(:mod:`.optuna_search`), so there is one copy of the "is Comet available /
enabled / keyed?" dance rather than one per entry point.

The rules:

- ``comet_ml`` is an OPTIONAL dependency; if it is not installed, everything here
  degrades to ``None`` and the callers keep running.
- Logging additionally requires ``COMET_API_KEY`` in the environment (the
  workspace comes from ``COMET_WORKSPACE``), so a run on a node without
  credentials is not an error -- it just does not log.
- Under DDP only **rank 0** creates an experiment. Every rank runs the same fit,
  so letting all of them log would create ``world_size`` duplicate experiments
  per run.

The returned object is a real ``comet_ml.Experiment`` or ``None``; callers treat
``None`` as "logging disabled" and skip their logging calls.
"""

from __future__ import annotations

import os
from typing import Any

try:
    import comet_ml

    _HAS_COMET = True
except ImportError:  # optional dependency
    _HAS_COMET = False

# One Comet project for every fit path, so Adam and Optuna+Adam runs are
# comparable in one place.
COMET_PROJECT_NAME = "DiffDelphes"


def comet_available() -> bool:
    """True if ``comet_ml`` is importable AND an API key is in the environment."""
    return _HAS_COMET and bool(os.environ.get("COMET_API_KEY"))


def init_comet_experiment(
    name: str | None = None,
    params: dict[str, Any] | None = None,
    *,
    disabled: bool = False,
    rank: int = 0,
    project_name: str = COMET_PROJECT_NAME,
    log_prefix: str = "[comet]",
    quiet: bool = False,
) -> "comet_ml.Experiment | None":
    """Create a Comet experiment, or return ``None`` when logging is off.

    Parameters
    ----------
    name : str | None
        Experiment name shown in the Comet UI (``set_name``). ``None`` leaves
        Comet's auto-generated name.
    params : dict | None
        Hyperparameters to record (``log_parameters``). Values that Comet cannot
        serialize are stringified by Comet itself; pass ``vars(args)`` freely.
    disabled : bool
        Caller's ``--comet-disabled`` flag; short-circuits to ``None``.
    rank : int
        DDP rank. Only rank 0 ever creates an experiment (see module docstring).
    project_name : str
        Comet project; defaults to the shared :data:`COMET_PROJECT_NAME`.
    log_prefix : str
        Prefix for the one-line status message (e.g. ``"[optuna]"``).
    quiet : bool
        Suppress the status message (used for per-trial experiments, where one
        line per trial would drown the log).

    Returns
    -------
    comet_ml.Experiment | None
        The live experiment on rank 0 when enabled and keyed; ``None`` otherwise.
    """
    if rank != 0:
        return None
    if disabled:
        if not quiet:
            print(f"{log_prefix} Comet logging disabled via --comet-disabled.")
        return None
    if not _HAS_COMET:
        if not quiet:
            print(f"{log_prefix} Comet logging disabled (comet_ml not installed).")
        return None
    if not os.environ.get("COMET_API_KEY"):
        if not quiet:
            print(f"{log_prefix} Comet logging disabled (COMET_API_KEY not set).")
        return None

    experiment = comet_ml.Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name=project_name,
        workspace=os.environ.get("COMET_WORKSPACE", ""),
    )
    if params:
        experiment.log_parameters(params)
    if name:
        experiment.set_name(name)
    if not quiet:
        print(f"{log_prefix} Comet experiment {name or '(auto-named)'}: {experiment.url}")
    return experiment


def end_comet_experiment(experiment: "comet_ml.Experiment | None") -> None:
    """Flush and close ``experiment``; a no-op when it is ``None``.

    Never raises: a Comet-side failure at teardown must not take down a fit that
    has already produced its results.
    """
    if experiment is None:
        return
    try:
        experiment.end()
    except Exception as e:  # noqa: BLE001 - telemetry must not break the run
        print(f"[comet] warning: failed to close experiment cleanly: {e}")
