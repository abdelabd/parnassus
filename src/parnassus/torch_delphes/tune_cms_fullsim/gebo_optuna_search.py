r"""Optuna hyperparameter search over GEBO + fine-tune *run settings*, for ONE
(loss, trust-region, gradient-mode) combination.

This is the ``optuna_search.py``-style outer loop, but for the
:mod:`.gebo_search` + fine-tune pipeline instead of the Adam fit in
:mod:`.training`. It does NOT reimplement any of them: each Optuna trial
materializes a normal ``gebo_search`` YAML run config (the same schema
:func:`.gebo_search.load_gebo_config` reads) and invokes

.. code-block:: shell

    python -m parnassus.torch_delphes.tune_cms_fullsim.gebo_search   <round_dir>/gebo_config.yaml
    python -m parnassus.torch_delphes.tune_cms_fullsim.<finetune>_finetune --gebo-summary <round_dir>/gebo/gebo_summary.json --n-steps ...

as **subprocesses** -- one GPU, one process at a time, exactly as if you had
run them by hand. Every artifact those two scripts normally produce (Comet
experiment, ``gebo_state.pt``/``gebo_debug.yaml``, per-new-best intermediate
plots, ``gebo_summary.json``, ``best_config.yaml``, the fine-tune stage's own
outputs) is produced unchanged, one full set per trial, under
``<round_dir>/gebo/`` and ``<round_dir>/gebo/<finetune>/``.

Two fine-tuners (``--finetune``), and they optimize genuinely different things:

``lbfgs`` (default, :mod:`.lbfgs_finetune`)
    Bounded L-BFGS-B on GEBO's OWN objective -- the whole dataset in one
    deterministic evaluation returning loss plus an analytic 66-D gradient. A
    local polish of exactly what the BO loop minimized. Trial score = that
    objective's final value. Launched by ``gebo_scans_interactive.sh``.

``adam`` (:mod:`.adam_finetune`)
    The real ``tune_cms_fullsim`` training loop, warm-started from GEBO's best
    point: stochastic mini-batches, a 70/20 train/val split, per-group Adam
    learning rates, ``ReduceLROnPlateau`` and early stopping. This makes the
    pipeline ``run_optuna.sh``'s Optuna+Adam with **GEBO** rather than Optuna
    choosing the 66 parameter initializations; the outer Optuna keeps sampling
    Adam's ``lr`` / ``batch_size`` / per-group ``lr_scale`` from the ``adam:``
    section of the meta-search config, over the same ranges
    ``param_configs/optuna_config.yaml`` uses. Trial score = the fit's best
    VALIDATION loss. Launched by ``optuna_gebo_adam.sh``.

Those two scores are not comparable, so the two pipelines default to separate
study roots (``doc/gebo_optuna_scans/`` vs ``doc/gebo_adam_scans/``) -- never
point them at the same ``--output-base``.

What is held CONSTANT across every trial (of every one of the 12 scans this
is meant to be run as, see ``gebo_scans_interactive.sh`` /
``gebo_scans_submit.sh``): the acquisition function (``LogEI``) and its
optimizer (``acquisition.num_restarts=20``, ``raw_samples=4096``), the
wall-clock cap on the GEBO stage (``--gebo-time-limit-hours``, default 2.0 --
``gebo_search.py`` stops its BO loop early once this elapses, even short of
the trial's own iteration TARGET below, and the trial still proceeds to
L-BFGS with whatever GEBO found so far), the L-BFGS-B iteration budget
(``--lbfgs-n-steps``, default 200) evaluated against ``--lbfgs-n-events``
events (default 20000, overriding whatever ``n_events`` GEBO's trial happened
to sample) with a batch size of the trial's GEBO-sampled ``batch_size`` times
``--lbfgs-batch-size-multiplier`` (default 12, clamped to
``--lbfgs-batch-size-max``, default 4096) -- scaled UP rather than reused
directly, since GEBO's sampled ``batch_size`` can be as small as 64 and would
otherwise force hundreds of mini-batches per L-BFGS objective/gradient call at
20k events, the 66-physical-parameter search space (``--optuna-config``), the
root file, and the RNG seed (so trials differ only by the sampled
run-setting hyperparameters, same convention as :mod:`.optuna_search`). What
is SCANNED per trial is read from ``--meta-search-config`` (default
``configs/gebo_meta_search.yaml``), including ``n_iterations_gebo`` -- this
trial's GEBO BO-iteration TARGET (replacing what used to be a fixed
``--gebo-n-iterations``); see that file for the exact ranges. A trial is
scored by its L-BFGS **final** loss (the true end of the GEBO -> L-BFGS
pipeline) -- UNLESS ``--gebo-time-limit-hours`` fires before GEBO completes
even its first BO iteration (checkpointed progress stays at 0), in which case
the trial is scored ``inf`` directly and L-BFGS is skipped (see the objective
function): running L-BFGS on a point BO never actually refined would waste
budget and hide how bad the sampled run settings are.

Resuming
--------
Two independent levels of resume, both required because this is meant to run
inside a walltime-limited SLURM job:

1. **Study-level** (an interrupted scan, i.e. "run more trials / pick up where
   the last job left off"): ``--storage`` is a persistent ``sqlite:///...``
   URL created with ``load_if_exists=True``, exactly like :mod:`.optuna_search`.

2. **Trial-level** (a trial killed mid-GEBO or mid-L-BFGS when the job hits its
   walltime): the storage is configured with Optuna's heartbeat mechanism
   (``heartbeat_interval`` / ``grace_period``) and a
   ``RetryFailedTrialCallback``. On startup, :func:`optuna.storages.fail_stale_trials`
   marks any trial whose heartbeat went stale (its owning process was killed
   without a clean failure) as FAILED, which the callback automatically
   re-enqueues as a fresh trial with the SAME sampled hyperparameters. That
   retried trial is mapped back to the ORIGINAL round directory (via
   ``RetryFailedTrialCallback.retried_trial_number``), so it does not start a
   new GEBO run from scratch -- it re-invokes ``gebo_search.py`` pointed at the
   same ``output_dir``, and ``gebo_search.py``'s OWN ``gebo_state.pt``
   checkpoint (written every BO iteration) resumes it. To make this land on
   exactly the trial's sampled ``n_iterations_gebo`` total (not "n_done +
   n_iterations more", which is ``gebo_search.py``'s own human-driven resume
   semantics), this script always requests ``n_iterations = target - n_done``
   REMAINING iterations, which ``gebo_search.py``'s ``n_iterations += n_done``
   resume logic then lands back at exactly ``target``. Optuna's retry
   mechanism re-enqueues the retried trial with the SAME sampled parameter
   values as the original (including ``n_iterations_gebo``), so ``target``
   is identical across the retry. A trial whose GEBO run is
   already complete (``n_done >= target``) is not re-invoked at all; a trial
   whose L-BFGS step already produced ``lbfgs/gebo_summary.json`` is not
   re-invoked either (L-BFGS has no iteration checkpoint of its own -- a job
   killed mid-L-BFGS just redoes the full, comparatively cheap L-BFGS run from
   the GEBO best point on resume).

Usage
-----
.. code-block:: shell

    python -m parnassus.torch_delphes.tune_cms_fullsim.gebo_optuna_search \
        --loss wasserstein_1d --trust-region cosine --grad-mode grad \
        --n-trials 40 --time-budget-hours 23
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import optuna
import torch
import yaml

from parnassus.torch_delphes import param_config as pc

# The two loss functions this scan pipeline supports (wasserstein_1d and
# soft_hist -- both deterministic; plain "wasserstein" uses random sliced
# projections, which would make the GEBO objective non-deterministic across
# evaluations and is deliberately excluded here).
SCAN_LOSS_CHOICES: tuple[str, ...] = ("wasserstein_1d", "soft_hist")

TRUST_REGION_CHOICES: tuple[str, ...] = ("cosine", "adaptive", "none")
TRUST_REGION_CLASS_PATHS: dict[str, str] = {
    "cosine": "parnassus.torch_delphes.tune_cms_fullsim.trust_region.CosineTrustRegion",
    "adaptive": "parnassus.torch_delphes.tune_cms_fullsim.trust_region.AdaptiveTrustRegion",
    "none": "parnassus.torch_delphes.tune_cms_fullsim.trust_region.NoTrustRegion",
}

GRAD_MODE_CHOICES: tuple[str, ...] = ("grad", "no_grad")

# Which optimizer polishes GEBO's best point. "lbfgs" runs bounded L-BFGS-B on
# GEBO's own full-dataset objective (a local polish of exactly what BO
# minimized); "adam" runs the real tune_cms_fullsim training loop -- stochastic
# mini-batches, train/val split, early stopping -- warm-started from that point,
# and is scored by its best VALIDATION loss. See the two *_finetune modules.
FINETUNE_CHOICES: tuple[str, ...] = ("lbfgs", "adam")
FINETUNE_MODULES: dict[str, str] = {
    "lbfgs": "parnassus.torch_delphes.tune_cms_fullsim.lbfgs_finetune",
    "adam": "parnassus.torch_delphes.tune_cms_fullsim.adam_finetune",
}

# gebo_search.py's own defaults (see configs/gebo_w1_cosine_grad.yaml), used
# to pin whichever of max_train_points / max_train_points_no_grad this scan's
# --grad-mode does NOT exercise, so it is never sampled for nothing.
DEFAULT_GP_MAX_TRAIN_POINTS = 80
DEFAULT_GP_MAX_TRAIN_POINTS_NO_GRAD = 500

_DEFAULT_ROOT_FILE = Path("/global/cfs/cdirs/m3246/diff_delphes/cms_pseudodata_100k.root")
_DEFAULT_OPTUNA_CONFIG = Path(pc.__file__).resolve().parent / "param_configs" / "optuna_config.yaml"
_DEFAULT_TRUTH_CONFIG = Path(pc.__file__).resolve().parent / "param_configs" / "cms_target_default.yaml"
_DEFAULT_META_SEARCH_CONFIG = Path(__file__).resolve().parent / "configs" / "gebo_meta_search.yaml"


# =============================================================================
# Meta search-space config
# =============================================================================


def load_meta_search_config(path: Path) -> dict:
    """Load and lightly validate ``gebo_meta_search.yaml``."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "common" not in raw or "trust_region" not in raw:
        raise SystemExit(
            f"{path}: expected a mapping with top-level 'common' and 'trust_region' "
            "sections (see configs/gebo_meta_search.yaml)."
        )
    return raw


def _suggest(trial: optuna.Trial, key: str, spec: dict) -> float | int | str:
    """Sample one value from a ``{low, high, log, type}`` or ``{choices}`` spec."""
    if "choices" in spec:
        return trial.suggest_categorical(key, list(spec["choices"]))
    low, high, log = spec["low"], spec["high"], bool(spec.get("log", False))
    if spec.get("type") == "int":
        return trial.suggest_int(key, int(low), int(high), log=log)
    return trial.suggest_float(key, float(low), float(high), log=log)


# =============================================================================
# Per-trial sampling -> materialized gebo_search config
# =============================================================================


def sample_trial(trial: optuna.Trial, meta: dict, args: argparse.Namespace) -> dict:
    """Sample one full run-setting point for a trial.

    Returns a flat dict of physical values (not yet shaped into the nested
    gebo_search YAML schema; see :func:`build_gebo_config`).
    """
    common = meta["common"]
    p: dict = {
        "n_events": _suggest(trial, "n_events", common["n_events"]),
        "n_initial": _suggest(trial, "n_initial", common["n_initial"]),
        "n_iterations_gebo": _suggest(trial, "n_iterations_gebo", common["n_iterations_gebo"]),
        "batch_size": _suggest(trial, "batch_size", common["batch_size"]),
    }

    lw = common["loss_weights"]
    p["count_weight"] = _suggest(trial, "count_weight", lw["count_weight"])
    p["calo_count_weight"] = _suggest(trial, "calo_count_weight", lw["calo_count_weight"])
    p["event_weight"] = _suggest(trial, "event_weight", lw["event_weight"])
    p["count_rate_floor"] = _suggest(trial, "count_rate_floor", lw["count_rate_floor"])
    p["pid_weighting"] = _suggest(trial, "pid_weighting", lw["pid_weighting"])
    p["pid_weight_floor"] = _suggest(trial, "pid_weight_floor", lw["pid_weight_floor"])

    gp = common["gp"]
    if args.grad_mode == "grad":
        p["max_train_points"] = _suggest(trial, "max_train_points", gp["max_train_points"])
        p["max_train_points_no_grad"] = DEFAULT_GP_MAX_TRAIN_POINTS_NO_GRAD
    else:
        p["max_train_points"] = DEFAULT_GP_MAX_TRAIN_POINTS
        p["max_train_points_no_grad"] = _suggest(
            trial, "max_train_points_no_grad", gp["max_train_points_no_grad"]
        )
    p["lengthscale_prior"] = [
        _suggest(trial, "lengthscale_prior_concentration", gp["lengthscale_prior_concentration"]),
        _suggest(trial, "lengthscale_prior_rate", gp["lengthscale_prior_rate"]),
    ]
    p["outputscale_prior"] = [
        _suggest(trial, "outputscale_prior_concentration", gp["outputscale_prior_concentration"]),
        _suggest(trial, "outputscale_prior_rate", gp["outputscale_prior_rate"]),
    ]
    p["noise_floor"] = _suggest(trial, "noise_floor", gp["noise_floor"])
    p["init_lengthscale_frac"] = _suggest(trial, "init_lengthscale_frac", gp["init_lengthscale_frac"])

    tr_spec = meta["trust_region"].get(args.trust_region) or {}
    p["trust_region_init_args"] = {
        key: _suggest(trial, f"tr_{key}", spec) for key, spec in tr_spec.items()
    }

    # Adam fine-tuning knobs, sampled ONLY for --finetune=adam so the lbfgs
    # scans keep the exact search space they have always had (an extra sampled
    # dimension would change TPE's behavior and make the two incomparable).
    if args.finetune == "adam":
        adam = meta.get("adam")
        if adam is None:
            raise SystemExit(
                f"{args.meta_search_config}: --finetune=adam needs a top-level "
                "'adam' section (lr / batch_size / lr_scale)."
            )
        p["adam_lr"] = _suggest(trial, "adam_lr", adam["lr"])
        p["adam_batch_size"] = _suggest(trial, "adam_batch_size", adam["batch_size"])
        p["adam_lr_scale"] = {
            g: _suggest(trial, f"adam_lr_scale_{g}", spec)
            for g, spec in adam["lr_scale"].items()
        }
    return p


def build_gebo_config(params: dict, args: argparse.Namespace, round_dir: Path, n_iterations: int, round_id: int) -> dict:
    """Shape a sampled point into the nested YAML schema :func:`.gebo_search.load_gebo_config` reads."""
    return {
        "root_file": str(args.root_file),
        "n_events": params["n_events"],
        "n_iterations": n_iterations,
        "time_limit_hours": args.gebo_time_limit_hours,
        "n_initial": params["n_initial"],
        "batch_size": params["batch_size"],
        "seed": args.seed,
        "loss": args.loss,
        "acq": "LogEI",
        "no_grad": args.grad_mode == "no_grad",
        "output_dir": str(round_dir / "gebo"),
        "machine_debug": True,
        "comet_disabled": args.comet_disabled,
        "comet_name": f"{args.loss}__{args.trust_region}__{args.grad_mode}__{round_id}",
        "optuna_config": str(args.optuna_config),
        "param_config": None,
        "trust_region": {
            "class_path": TRUST_REGION_CLASS_PATHS[args.trust_region],
            "init_args": params["trust_region_init_args"],
        },
        "loss_weights": {
            "count_weight": params["count_weight"],
            "calo_count_weight": params["calo_count_weight"],
            "count_rate_floor": params["count_rate_floor"],
            "event_weight": params["event_weight"],
            "pid_weighting": params["pid_weighting"],
            "pid_weight_floor": params["pid_weight_floor"],
        },
        "plotting": {
            "plot_every_best": True,
            "n_plot_events": args.n_plot_events,
            "plot_batch_size": args.plot_batch_size,
            "truth_config": str(args.truth_config),
        },
        "gp": {
            "max_train_points": params["max_train_points"],
            "max_train_points_no_grad": params["max_train_points_no_grad"],
            "lengthscale_prior": params["lengthscale_prior"],
            "outputscale_prior": params["outputscale_prior"],
            "noise_floor": params["noise_floor"],
            "init_lengthscale_frac": params["init_lengthscale_frac"],
        },
        "acquisition": {
            "num_restarts": args.acqf_num_restarts,
            "raw_samples": args.acqf_raw_samples,
        },
    }


# =============================================================================
# Subprocess plumbing + idempotent resume helpers
# =============================================================================


def _run_subprocess(cmd: list[str], log_path: Path) -> None:
    """Run ``cmd``, appending its combined stdout/stderr to ``log_path``.

    Appends (not truncates) so a resumed run's log keeps the prior attempt's
    output. Raises ``RuntimeError`` (which Optuna treats as a trial failure,
    triggering the retry-callback machinery) on a non-zero exit.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)} =====\n")
        f.flush()
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (exit {result.returncode}): {' '.join(cmd)}  -- see {log_path}"
        )


def _gebo_n_done(round_dir: Path) -> int:
    """Iterations already completed for this round's GEBO run (0 if none yet)."""
    state_path = round_dir / "gebo" / "gebo_state.pt"
    if not state_path.exists():
        return 0
    ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
    return int(ckpt.get("iteration", 0))


# =============================================================================
# Optuna objective
# =============================================================================


def make_objective(args: argparse.Namespace, meta: dict):
    def objective(trial: optuna.Trial) -> float:
        # A retried trial (from a stale/failed original) gets a NEW trial
        # number but must land on the SAME round_dir as the original attempt,
        # so gebo_search.py's own gebo_state.pt resume picks up mid-run
        # instead of restarting GEBO from Sobol iteration 0.
        retried_from = optuna.storages.RetryFailedTrialCallback.retried_trial_number(trial)
        round_id = retried_from if retried_from is not None else trial.number
        round_dir = args.output_base / f"round_{round_id}"
        round_dir.mkdir(parents=True, exist_ok=True)

        params = sample_trial(trial, meta, args)
        trial.set_user_attr("round_dir", str(round_dir))
        trial.set_user_attr("round_id", round_id)

        try:
            n_iterations_target = params["n_iterations_gebo"]
            n_done = _gebo_n_done(round_dir)
            remaining = max(n_iterations_target - n_done, 0)
            gebo_cfg = build_gebo_config(params, args, round_dir, n_iterations=remaining, round_id=round_id)
            gebo_cfg_path = round_dir / "gebo_config.yaml"
            with open(gebo_cfg_path, "w") as f:
                yaml.safe_dump(gebo_cfg, f, sort_keys=False, default_flow_style=False)

            # Gate on gebo_summary.json existing, NOT on remaining > 0: gebo_search.py
            # writes a valid summary and hands off to L-BFGS as soon as it gracefully
            # stops for ANY reason (full iteration count, its own --time-limit-hours,
            # or a model-fit error) -- even if that's far short of n_iterations_target.
            # A resumed trial should treat that the same way a first-time-through run
            # does (accept it, move to L-BFGS), not redo GEBO from the checkpoint all
            # the way out to the full target before ever touching L-BFGS again.
            gebo_summary_path = round_dir / "gebo" / "gebo_summary.json"
            if not gebo_summary_path.exists():
                print(
                    f"[scan] trial {trial.number} (round {round_id}): "
                    f"running GEBO ({remaining} of {n_iterations_target} iterations remaining) ...",
                    flush=True,
                )
                _run_subprocess(
                    [sys.executable, "-m", "parnassus.torch_delphes.tune_cms_fullsim.gebo_search",
                     str(gebo_cfg_path)],
                    log_path=round_dir / "gebo_run.log",
                )
            else:
                print(f"[scan] trial {trial.number} (round {round_id}): GEBO already complete, skipping", flush=True)

            with open(gebo_summary_path) as f:
                gebo_summary = json.load(f)
            gebo_best_loss = float(gebo_summary["best_loss"])

            # n_iterations_gebo is always sampled >= 20 (see gebo_meta_search.yaml),
            # so the ONLY way this round's checkpoint shows zero completed BO
            # iterations is --gebo-time-limit-hours firing before iteration 1 (the
            # initial Sobol evaluation itself ate the whole budget). gebo_search.py
            # still writes a "valid" gebo_summary.json in that case -- best_loss
            # from the random Sobol init alone, with NO Bayesian refinement -- so
            # running the comparatively expensive L-BFGS stage on it would waste
            # budget polishing a point BO never actually got to touch. Treat it the
            # same as the other bad-point cases below: a legitimately bad, COMPLETE
            # trial (score inf), skipping the fine-tune stage entirely.
            if _gebo_n_done(round_dir) == 0:
                print(
                    f"[scan] trial {trial.number} (round {round_id}): GEBO timed out before "
                    f"completing a single BO iteration (Sobol-only best_loss = {gebo_best_loss:.4e}) "
                    "-- scoring as a bad, complete trial and skipping the fine-tune stage.",
                    flush=True,
                )
                trial.set_user_attr("gebo_best_loss", gebo_best_loss)
                trial.set_user_attr(
                    "failure_reason", "GEBO timed out before completing a single BO iteration"
                )
                return float("inf")

            # A bad sampled GP-prior combination can make botorch's GP hyperparameter
            # fit numerically fail mid-run (see gebo_search.py's ModelFittingError
            # handling); gebo_search.py stops the BO loop right there rather than
            # crashing, and flags it here. This is only fatal to the trial if the
            # BO stage never actually found anything -- i.e. its best loss never
            # beat the best of the n_initial Sobol points (best_initial_loss,
            # reached here since the earlier _gebo_n_done(round_dir) == 0 check
            # already handles the "not even one BO iteration completed" case). If
            # at least one BO iteration DID improve on the Sobol init, that
            # progress is real and worth polishing with L-BFGS despite the later
            # GP-fit error, so fall through to the normal L-BFGS stage instead of
            # skipping it.
            if gebo_summary.get("model_fit_error", False):
                best_initial_loss = gebo_summary.get("best_initial_loss")
                bo_improved_on_sobol = (
                    best_initial_loss is not None and gebo_best_loss < best_initial_loss - 1e-8
                )
                if not bo_improved_on_sobol:
                    print(
                        f"[scan] trial {trial.number} (round {round_id}): GEBO hit a GP model-fit "
                        f"error (best_loss so far = {gebo_best_loss:.4e}) and never beat the best "
                        f"Sobol init ({best_initial_loss}) -- scoring as a bad, complete trial and "
                        "skipping the fine-tune stage.",
                        flush=True,
                    )
                    trial.set_user_attr("gebo_best_loss", gebo_best_loss)
                    trial.set_user_attr("failure_reason", "GEBO hit a GP model-fit error")
                    return float("inf")
                print(
                    f"[scan] trial {trial.number} (round {round_id}): GEBO hit a GP model-fit "
                    f"error but still improved on the best Sobol init ({best_initial_loss:.4e} -> "
                    f"{gebo_best_loss:.4e}) -- proceeding to the fine-tune stage anyway.",
                    flush=True,
                )

            # The fine-tune stage writes into <round>/gebo/<finetune>/, so the
            # two fine-tuners never share a directory (or a resume gate).
            finetune_summary_path = (
                round_dir / "gebo" / args.finetune / "gebo_summary.json"
            )
            if not finetune_summary_path.exists():
                print(f"[scan] trial {trial.number} (round {round_id}): running "
                      f"{args.finetune.upper()} fine-tuning "
                      f"({args.finetune_n_events} events) ...", flush=True)
                cmd = [
                    sys.executable, "-m", FINETUNE_MODULES[args.finetune],
                    "--gebo-summary", str(gebo_summary_path),
                    "--n-steps", str(args.finetune_n_steps),
                    "--n-events", str(args.finetune_n_events),
                ]
                if args.finetune == "lbfgs":
                    # Scale UP from the trial's GEBO batch_size rather than reusing it
                    # directly: --finetune-n-events is much larger than what GEBO
                    # evaluated, and GEBO's sampled batch_size can be as small as 64
                    # (see gebo_meta_search.yaml), which at 20k events would mean
                    # hundreds of mini-batches per L-BFGS objective/gradient call
                    # (this is what made every L-BFGS stage across an early scan look
                    # hung -- it wasn't hung, just running ~300x more mini-batches per
                    # eval than GEBO did).
                    lbfgs_batch_size = min(
                        round(params["batch_size"] * args.lbfgs_batch_size_multiplier),
                        args.lbfgs_batch_size_max,
                    )
                    cmd += ["--batch-size", str(lbfgs_batch_size)]
                    if args.lbfgs_max_fun is not None:
                        cmd += ["--max-fun", str(args.lbfgs_max_fun)]
                else:
                    # Adam's batch size / lr / per-group lr_scale are SAMPLED by this
                    # trial (meta['adam']), not derived from GEBO's -- the outer
                    # Optuna tunes the training the same way optuna_search.py does.
                    cmd += ["--batch-size", str(params["adam_batch_size"]),
                            "--lr", repr(params["adam_lr"])]
                    for group, scale in params["adam_lr_scale"].items():
                        cmd += [f"--lr-scale-{group}", repr(scale)]
                _run_subprocess(
                    cmd, log_path=round_dir / "gebo" / f"{args.finetune}_run.log"
                )
            else:
                print(f"[scan] trial {trial.number} (round {round_id}): "
                      f"{args.finetune.upper()} already complete, skipping", flush=True)

            with open(finetune_summary_path) as f:
                finetune_summary = json.load(f)
            finetune_best_loss = float(finetune_summary["best_loss"])
        except (RuntimeError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            # A subprocess exit failure here means THIS sampled hyperparameter
            # point is bad (e.g. botorch.exceptions.ModelFittingError from an
            # extreme GP-prior sample, or a CUDA OOM from an extreme
            # batch_size/n_events draw) -- deterministic given the same params,
            # so letting it propagate would mark the trial FAILED and the
            # heartbeat RetryFailedTrialCallback (see main()) would re-enqueue
            # the SAME round_id/hyperparameters, which would fail identically
            # forever and burn the whole trial budget on one bad point. Score
            # it as a legitimately bad, COMPLETE trial instead: TPE learns to
            # avoid the region and the scan moves on to a new sample.
            #
            # A trial killed by an external SLURM timeout/preemption never
            # reaches this except block at all -- the whole process dies
            # mid-call, with no exception raised in THIS process. That case is
            # what the heartbeat + RetryFailedTrialCallback machinery in
            # main() is for, and is unaffected by this except block.
            print(f"[scan] trial {trial.number} (round {round_id}): FAILED -- {exc}", flush=True)
            trial.set_user_attr("failure_reason", str(exc))
            return float("inf")

        trial.set_user_attr("gebo_best_loss", gebo_best_loss)
        trial.set_user_attr("finetune_best_loss", finetune_best_loss)
        # Kept under its historical key too, so tooling and notebooks written
        # against the L-BFGS-only scans keep reading the studies they already have.
        if args.finetune == "lbfgs":
            trial.set_user_attr("lbfgs_best_loss", finetune_best_loss)
        print(
            f"[scan] trial {trial.number} (round {round_id}): "
            f"gebo_best={gebo_best_loss:.4e}  "
            f"{args.finetune}_best={finetune_best_loss:.4e}",
            flush=True,
        )
        return finetune_best_loss

    return objective


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--loss", required=True, choices=SCAN_LOSS_CHOICES)
    parser.add_argument("--trust-region", required=True, choices=TRUST_REGION_CHOICES)
    parser.add_argument("--grad-mode", required=True, choices=GRAD_MODE_CHOICES,
                         help="'grad' -> gebo_search no_grad: false (gradient-enhanced GP); "
                              "'no_grad' -> no_grad: true (loss-only GP).")

    parser.add_argument("--root-file", type=Path, default=_DEFAULT_ROOT_FILE)
    parser.add_argument("--optuna-config", type=Path, default=_DEFAULT_OPTUNA_CONFIG,
                         help="Fixed 66-physical-parameter search space (unchanged across every trial).")
    parser.add_argument("--truth-config", type=Path, default=_DEFAULT_TRUTH_CONFIG)
    parser.add_argument("--meta-search-config", type=Path, default=_DEFAULT_META_SEARCH_CONFIG,
                         help="The scanned run-setting hyperparameter ranges (configs/gebo_meta_search.yaml).")

    parser.add_argument("--output-base", type=Path, default=None,
                         help="Default: doc/gebo_optuna_scans/<loss>__<trust-region>__<grad-mode>.")
    parser.add_argument("--study-name", type=str, default=None, help="Default: same as --output-base's basename.")
    parser.add_argument("--storage", type=str, default=None,
                         help="Optuna storage URL. Default: sqlite:///<output-base>/optuna_study.db "
                              "(REQUIRED to be persistent for resume -- do not pass 'None'/in-memory).")

    parser.add_argument("--n-trials", type=int, default=40,
                         help="Trials to run THIS invocation (on top of any already in the resumed study).")
    parser.add_argument("--time-budget-hours", type=float, default=23.0,
                         help="Stop starting new trials after this many wall-clock hours "
                              "(checked between trials, not mid-trial). <=0 disables.")

    parser.add_argument("--gebo-time-limit-hours", type=float, default=2.0,
                         help="Wall-clock cap on the GEBO stage of each trial: gebo_search.py stops its "
                              "BO loop once this elapses, even short of the trial's sampled "
                              "n_iterations_gebo target (see --meta-search-config), and the trial "
                              "proceeds to the fine-tune stage with whatever GEBO found so far "
                              "(<=0 disables; held constant across the whole scan).")
    parser.add_argument("--finetune", choices=FINETUNE_CHOICES, default="lbfgs",
                         help="Optimizer that polishes GEBO's best point. 'lbfgs' (default) runs "
                              "bounded L-BFGS-B on GEBO's own full-dataset objective. 'adam' runs the "
                              "real tune_cms_fullsim training loop (stochastic mini-batches, train/val "
                              "split, ReduceLROnPlateau, early stopping) warm-started from that point, "
                              "with its lr / batch_size / per-group lr_scale SAMPLED per trial from the "
                              "meta-search config's 'adam' section -- i.e. Optuna+GEBO+Adam, where GEBO "
                              "supplies the parameter inits that optuna_search.py would otherwise "
                              "sample. Trials are then scored by the fit's best VALIDATION loss, which "
                              "is NOT comparable to an L-BFGS scan's score: keep the two in separate "
                              "studies (see --output-base / optuna_gebo_adam.sh).")
    # Shared fine-tune-stage budget. The --lbfgs-* spellings are kept as aliases
    # so existing scripts and command lines keep working unchanged.
    parser.add_argument("--finetune-n-steps", "--lbfgs-n-steps", dest="finetune_n_steps",
                         type=int, default=200,
                         help="Fine-tune iteration budget per trial: L-BFGS-B maxiter, or Adam epochs "
                              "(held constant across the whole scan).")
    parser.add_argument("--finetune-n-events", "--lbfgs-n-events", dest="finetune_n_events",
                         type=int, default=20_000,
                         help="Events the fine-tune stage runs on, overriding whatever n_events "
                              "GEBO's trial happened to sample (held constant across the whole scan). "
                              "For --finetune=adam this is split 70/20 into train/val by "
                              "runner.load_split_datasets, as in every other Adam fit.")
    # L-BFGS-only knobs (Adam's batch size is sampled per trial instead).
    parser.add_argument("--lbfgs-max-fun", type=int, default=None)
    parser.add_argument("--lbfgs-batch-size-multiplier", type=float, default=12.0,
                         help="L-BFGS's forward-pass batch size = the trial's GEBO-sampled batch_size "
                              "times this factor (clamped to --lbfgs-batch-size-max). A bigger batch "
                              "amortizes --lbfgs-n-events's extra data over far fewer mini-batches per "
                              "objective/gradient evaluation than reusing GEBO's (often tiny) sampled "
                              "batch_size directly would -- held constant across the whole scan.")
    parser.add_argument("--lbfgs-batch-size-max", type=int, default=4096,
                         help="Hard ceiling on the scaled L-BFGS batch size: the calorimeter forward "
                              "runs in float64, and 4096 is the repo's own documented safe operating "
                              "point (~11GB/GPU; 8192 OOMs a 40GB card) -- held constant across the "
                              "whole scan.")

    parser.add_argument("--acqf-num-restarts", type=int, default=20)
    parser.add_argument("--acqf-raw-samples", type=int, default=4096)
    parser.add_argument("--n-plot-events", type=int, default=20000)
    parser.add_argument("--plot-batch-size", type=int, default=2000)

    parser.add_argument("--seed", type=int, default=0,
                         help="Fixed across trials (and passed straight to every GEBO run) so trials "
                              "differ only by the sampled run-setting hyperparameters.")
    parser.add_argument("--comet-disabled", action="store_true")

    args = parser.parse_args()

    # Line-buffer stdout: this process is normally launched with stdout
    # redirected to logs/<scan_name>.log (see gebo_scans_run_on_node.sh), and
    # Python fully-buffers a non-tty stdout by default -- so without this, the
    # "[scan] ..." lines below (resuming a study, catching a stale trial,
    # starting/finishing each trial's GEBO/L-BFGS stage) don't reach the log
    # until the process exits, which is exactly what you'd want to watch to
    # confirm a resumed run is making progress.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if not args.root_file.exists():
        raise SystemExit(f"--root-file {args.root_file} does not exist.")
    if not args.optuna_config.exists():
        raise SystemExit(f"--optuna-config {args.optuna_config} does not exist.")
    if not args.meta_search_config.exists():
        raise SystemExit(f"--meta-search-config {args.meta_search_config} does not exist.")

    scan_name = f"{args.loss}__{args.trust_region}__{args.grad_mode}"
    if args.output_base is None:
        # Separate default roots per fine-tuner. An Adam trial's score is a
        # VALIDATION loss and an L-BFGS trial's is a full-dataset training
        # objective, so pointing both at doc/gebo_optuna_scans/<scan_name> would
        # silently resume one study with the other's incomparable trial values.
        default_root = (
            "doc/gebo_optuna_scans" if args.finetune == "lbfgs" else "doc/gebo_adam_scans"
        )
        args.output_base = Path(default_root) / scan_name
    if args.study_name is None:
        args.study_name = scan_name
    args.output_base.mkdir(parents=True, exist_ok=True)
    if args.storage is None:
        args.storage = f"sqlite:///{(args.output_base / 'optuna_study.db').resolve()}"

    meta = load_meta_search_config(args.meta_search_config)

    print(f"[scan] {scan_name}  (finetune={args.finetune})")
    print(f"[scan] output_base = {args.output_base}")
    print(f"[scan] storage     = {args.storage}")

    # Heartbeat + retry callback: the trial-level resume mechanism described
    # in the module docstring. grace_period must exceed the longest gap
    # between heartbeats a live trial can have; the heartbeat thread ticks
    # independently of the blocking subprocess calls in the objective, so a
    # few minutes of slack is generous.
    storage = optuna.storages.RDBStorage(
        url=args.storage,
        engine_kwargs={"connect_args": {"timeout": 30}},
        heartbeat_interval=60,
        grace_period=600,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
    )
    if args.storage.startswith("sqlite:"):
        # SQLite's default rollback-journal mode takes a whole-database
        # exclusive lock for the duration of any write; the heartbeat
        # thread's periodic writes then collide with the main thread's own
        # trial/study writes and raise "database is locked" (observed
        # crashing a live scan). WAL mode lets a writer and readers coexist;
        # the 30s connect timeout above is the fallback if a write really
        # does have to wait. Applied per-connection since pragmas are
        # per-connection and the heartbeat thread uses its own connection.
        from sqlalchemy import event

        @event.listens_for(storage.engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _rec) -> None:
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA busy_timeout=30000")
            cur.close()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True, n_startup_trials=10, seed=args.seed),
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )
    n_before = len(study.trials)
    if n_before > 0:
        print(f"[scan] resuming study {args.study_name!r}: {n_before} existing trials")
        failed_trial_numbers = optuna.storages.fail_stale_trials(study)
        if failed_trial_numbers:
            print(f"[scan] marked {len(failed_trial_numbers)} stale (heartbeat-timed-out) trial(s) FAILED "
                  f"-> re-enqueued for retry at their original round_dir")

    timeout = args.time_budget_hours * 3600 if args.time_budget_hours and args.time_budget_hours > 0 else None
    study.optimize(make_objective(args, meta), n_trials=args.n_trials, timeout=timeout)

    completed = [t for t in study.trials if t.value is not None]
    if completed:
        best = min(completed, key=lambda t: t.value)
        print(
            f"\n[scan] {scan_name}: best trial {best.number}  "
            f"lbfgs_loss={best.value:.6e}  round_dir={best.user_attrs.get('round_dir')}"
        )
    else:
        print(f"\n[scan] {scan_name}: no trial completed successfully yet.")


if __name__ == "__main__":
    main()
