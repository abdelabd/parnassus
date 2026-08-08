r"""Adam fine-tuning of the best model from a GEBO run.

This is the alternative second stage to :mod:`.lbfgs_finetune`, and the two are
deliberately NOT the same optimizer applied to the same objective:

- :mod:`.lbfgs_finetune` runs bounded L-BFGS-B on GEBO's *own* objective -- the
  whole dataset in one deterministic, fixed-seed evaluation returning loss plus
  an analytic 66-D gradient. It is a local polish of the exact quantity the BO
  loop minimized.
- **This module runs the real ``tune_cms_fullsim`` training loop**
  (:func:`.training.fit_card_to_fullsim`): a train/val split, shuffled
  stochastic mini-batches, per-parameter-group Adam learning rates,
  ``ReduceLROnPlateau`` decay and early stopping on the validation loss. It is
  the same fit ``python -m parnassus.torch_delphes.tune_cms_fullsim`` performs,
  only started from GEBO's best point instead of a hand-written param config.

That is the whole point of the ``--finetune adam`` pipeline: it replaces the
Optuna-sampled *initialization* in ``run_optuna.sh`` (Optuna+Adam) with a
GEBO-searched one, keeping the Adam training itself untouched. The trial is
scored by the fit's best (minimum) **validation** loss, exactly as the
Optuna+Adam search scores its trials -- which is a different quantity from the
L-BFGS stage's full-dataset training objective, so numbers from the two
fine-tuners are not comparable trial-for-trial.

How GEBO's best point becomes an Adam initialization: GEBO reports
``best_physical_params`` (the post-transform values of the dimensions it
searched). Those are written into a full ``{value, trainable, lr_scale}`` param
config -- the searched dimensions marked ``trainable: true`` with this run's
per-group ``lr_scale``, every other card scalar pinned at its default -- which
is then loaded through the standard :mod:`~parnassus.torch_delphes.param_config`
path, so the same range guards and gradient masking apply as in any other fit.

Outputs (under ``--output-dir``, default ``<gebo-dir>/adam``):

- ``init_config.yaml``   -- the materialized starting config (GEBO's best point)
- ``best_config.yaml``   -- the fine-tuned card as a standard param config
- ``gebo_summary.json``  -- summary in the schema :mod:`.plot_gebo_results` and
  :mod:`.gebo_optuna_search` consume (``best_loss`` = best validation loss)
- ``history.json``       -- the per-epoch fit history, in the schema
  :mod:`.plot_fit_results` consumes
- ``adam_data.pt``       -- parameter trajectory + losses, mirroring
  ``gebo_data.pt`` / ``lbfgs_data.pt``

Usage
-----
.. code-block:: shell

    python -m parnassus.torch_delphes.tune_cms_fullsim.adam_finetune \
        --gebo-summary doc/gebo_adam_scans/<scan>/round_0/gebo/gebo_summary.json \
        --n-steps 200 --lr 3e-3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .comet_utils import end_comet_experiment
from .config import _DEFAULT_LR
from .dataloader import DelphesDataLoader
from .finetune_utils import (
    load_archived_config,
    make_setting_resolver,
    reconnect_comet,
    trainable_bases,
)
from .gebo_search import ParamVectorizer
from .runner import load_split_datasets, write_history_json
from .training import fit_card_to_fullsim

# Metric-name suffix for everything this stage logs into the round's Comet
# experiment, so the Adam curves never collide with the GEBO stage's own.
COMET_SUFFIX = "_adam"

# lr_scale groups, mirroring optuna_search.LR_SCALE_GROUPS. Kept as a plain
# tuple here rather than imported so this module does not pull in Optuna.
LR_SCALE_GROUPS: tuple[str, ...] = ("resolution", "scale", "efficiency")


def _group_of(base: str) -> str:
    """Return the lr_scale group of a parameter (by its base name).

    Identical to :func:`.optuna_search._group_of` -- ``scale`` (tanh scales),
    ``efficiency`` (sigmoid efficiencies/fractions and the softplus ``rate_raw``),
    ``resolution`` (every other softplus coefficient).
    """
    kind = pc.param_transform_kind(base)
    if kind == "scale":
        return "scale"
    if kind == "logit" or base.endswith("rate_raw"):
        return "efficiency"
    return "resolution"


def build_init_config(
    best_physical: dict[str, float],
    trainable: set[str],
    group_lr_scale: dict[str, float],
) -> dict[str, dict]:
    """Materialize GEBO's best point as a full ``{value, trainable, lr_scale}`` config.

    Every card scalar gets an entry (``apply_param_config`` requires exact
    coverage): the dimensions GEBO searched take its best physical value and are
    marked trainable with their group's ``lr_scale``; the rest are pinned at the
    card's own default.

    Trainable *logit* values are nudged strictly inside
    ``(_TRAINABLE_LOGIT_MIN, _TRAINABLE_LOGIT_MAX)``. GEBO's raw-space bounds
    come from the same ``optuna_config``, so its best point is already in range
    -- but a point that landed exactly ON a bound would round-trip to the closed
    endpoint and be rejected by ``load_param_config``'s saturated-init guard,
    failing the trial for a value that is numerically fine.
    """
    probe = CMSEnergyFlowDefault(debug=False, learnable=True)
    eps = 1e-6
    flat_cfg: dict[str, dict] = {}
    for key, default_value in _card_defaults(probe).items():
        base = key.split("[", 1)[0]
        is_trainable = base in trainable and key in best_physical
        value = float(best_physical[key]) if is_trainable else float(default_value)
        if is_trainable and pc.param_transform_kind(base) == "logit":
            value = min(
                max(value, pc._TRAINABLE_LOGIT_MIN + eps),  # noqa: SLF001
                pc._TRAINABLE_LOGIT_MAX - eps,  # noqa: SLF001
            )
        flat_cfg[key] = {
            "value": value,
            "trainable": is_trainable,
            "lr_scale": group_lr_scale[_group_of(base)] if is_trainable else 1.0,
        }
    return flat_cfg


def _card_defaults(card: CMSEnergyFlowDefault) -> dict[str, float]:
    """Flat ``{scalar_key: physical value}`` for every parameter of ``card``."""
    out: dict[str, float] = {}
    for name, p in card.named_parameters():
        vals = pc.to_physical(name, p.data.flatten()).tolist()
        for key, v in zip(pc._scalar_keys(name, p), vals):  # noqa: SLF001
            out[key] = float(v)
    return out


def _dump_flat_config(flat_cfg: dict[str, dict], path: Path) -> None:
    """Write a materialized config to YAML (same format as ``optuna_search``)."""
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


def main() -> None:
    """Entry point for ``python -m ...tune_cms_fullsim.adam_finetune``."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--gebo-summary",
        type=Path,
        required=True,
        help="Path to a GEBO run's gebo_summary.json (its best point is the Adam init).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write results (default: <gebo-summary dir>/adam).",
    )
    # --- Adam hyperparameters (sampled per trial by gebo_optuna_search) -------
    parser.add_argument("--n-steps", type=int, default=200,
                        help="Max training epochs (default 200).")
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR,
                        help=("Global Adam lr magnitude; each parameter's effective lr is "
                              f"this times its group lr_scale (default {_DEFAULT_LR})."))
    for group in LR_SCALE_GROUPS:
        parser.add_argument(f"--lr-scale-{group}", type=float, default=1.0,
                            help=f"lr_scale for the '{group}' parameter group (default 1.0).")
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Train/val batch size (default 4096, the tune_cms_fullsim default).")
    parser.add_argument("--early-stopping-patience", type=int, default=10,
                        help="Epochs of no val_loss improvement before stopping (<=0 disables).")
    parser.add_argument("--lr-scheduler-patience", type=int, default=4,
                        help="ReduceLROnPlateau patience (<=0 disables lr decay).")
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.5,
                        help="Multiplicative lr decay on each plateau (default 0.5).")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Epochs between the stdout loss-breakdown tables (default 10).")
    parser.add_argument("--intermediate-plot-dir", type=str, default="",
                        help=("Directory for per-epoch observable PDFs. Empty (default) "
                              "disables them -- 12 concurrent scans plotting every epoch is "
                              "a lot of I/O for little value mid-scan."))
    parser.add_argument("--plot-every", type=int, default=1,
                        help="Save intermediate plots every N epochs (default 1).")
    # --- overrides for settings otherwise reused from the GEBO run -----------
    parser.add_argument("--n-events", type=int, default=None,
                        help="Override the run's event count (default: reuse).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the run's RNG seed (default: reuse).")
    parser.add_argument("--root-file", type=Path, default=None,
                        help="Override the run's ROOT file (default: reuse).")
    parser.add_argument("--optuna-config", type=Path, default=None,
                        help="Override the run's optuna_config (which dims are trainable).")
    parser.add_argument("--comet-disabled", action="store_true",
                        help="Do not log into the round's GEBO Comet experiment.")
    args = parser.parse_args()

    # Line-buffer stdout: this process is normally launched with stdout
    # redirected to a log file (see gebo_optuna_search.py's _run_subprocess),
    # and Python fully-buffers a non-tty stdout by default -- so without this
    # none of the progress below reaches the log until the process exits,
    # making a slow fit indistinguishable from a hung one.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if not args.gebo_summary.exists():
        raise SystemExit(f"--gebo-summary {args.gebo_summary} does not exist.")
    with open(args.gebo_summary) as f:
        summary = json.load(f)

    best_physical = summary.get("best_physical_params", {})
    if not best_physical:
        raise SystemExit(f"{args.gebo_summary} has no 'best_physical_params'.")

    # Recover run settings from the archived config (reliable even for an
    # intermediate summary that has no 'args'). The run dir is the summary's
    # parent, or its grandparent when the summary lives under 'intermediate/'.
    run_dir = args.gebo_summary.parent
    if run_dir.name == "intermediate":
        run_dir = run_dir.parent
    cfg_ns = load_archived_config(run_dir)
    if cfg_ns is not None:
        print(f"[adam] recovered run settings from {run_dir}/configs/")
    _cfg = make_setting_resolver(summary.get("args", {}) or {}, cfg_ns)

    root_file = args.root_file or _cfg("root_file")
    if root_file is None:
        raise SystemExit("could not resolve 'root_file' from the summary or an "
                         "archived config; pass --root-file explicitly.")
    root_file = Path(root_file)
    n_events = args.n_events if args.n_events is not None else int(_cfg("n_events", -1))
    seed = args.seed if args.seed is not None else int(_cfg("seed", 0))
    optuna_config = args.optuna_config or _cfg("optuna_config")
    if optuna_config is None:
        raise SystemExit("could not resolve 'optuna_config' from the summary or an "
                         "archived config; pass --optuna-config explicitly.")
    optuna_config = Path(optuna_config)
    loss_name = _cfg("loss", "wasserstein_1d")

    # The loss weights this round SAMPLED for GEBO. Reusing them means the Adam
    # stage minimizes the same loss the BO stage did -- only the optimizer and
    # the batching change -- so the starting point is actually a good one for it.
    loss_kwargs = dict(
        count_weight=float(_cfg("count_weight", 0.1)),
        calo_count_weight=float(_cfg("calo_count_weight", 1.0)),
        count_rate_floor=float(_cfg("count_rate_floor", 0.05)),
        event_weight=float(_cfg("event_weight", 0.1)),
        pid_weighting=_cfg("pid_weighting", "sqrt_fraction"),
        pid_weight_floor=float(_cfg("pid_weight_floor", 0.0)),
    )

    output_dir = args.output_dir or (args.gebo_summary.parent / "adam")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[adam] device = {device}")
    print(f"[adam] source run: {args.gebo_summary}  (GEBO best_loss={summary.get('best_loss')})")
    print(f"[adam] loss={loss_name!r}  lr={args.lr:.3e}  batch_size={args.batch_size}  "
          f"n_steps={args.n_steps}  n_events={n_events}")

    # --- starting config: GEBO's best point ---------------------------------
    trainable = trainable_bases(optuna_config)
    group_lr_scale = {g: float(getattr(args, f"lr_scale_{g}")) for g in LR_SCALE_GROUPS}
    print("[adam] lr_scale: "
          + ", ".join(f"{g}={v:.3g}" for g, v in group_lr_scale.items()))
    init_cfg = build_init_config(best_physical, trainable, group_lr_scale)
    init_config_path = output_dir / "init_config.yaml"
    _dump_flat_config(init_cfg, init_config_path)
    # Round-trip through load_param_config so the standard range guards run on
    # the materialized file (the same validation any hand-written config gets).
    flat_cfg = pc.load_param_config(init_config_path)

    # --- data ----------------------------------------------------------------
    print(f"[adam] loading {n_events} events from {root_file} ...")
    t0 = time.perf_counter()
    train_dataset, val_dataset = load_split_datasets(root_file, n_events=n_events, device=device)
    train_dataloader = DelphesDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DelphesDataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"[adam] loaded in {time.perf_counter() - t0:.1f}s "
          f"({len(train_dataset)} train / {len(val_dataset)} val events)")

    # --- trainee -------------------------------------------------------------
    torch.manual_seed(seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    pc.apply_param_config(trainee, flat_cfg)
    params_to_train, param_groups = pc.select_trainable(trainee, flat_cfg, global_lr=args.lr)
    if not params_to_train:
        raise SystemExit(
            f"{optuna_config} marks no parameter as trainable; nothing to fine-tune."
        )
    print(f"[adam] {sum(int(spec['trainable']) for spec in flat_cfg.values())} trainable scalars "
          f"in {len(param_groups)} lr group(s)")

    # --- Comet: log into the round's existing GEBO experiment ----------------
    comet_exp = reconnect_comet(run_dir, disabled=args.comet_disabled, log_prefix="[adam]")

    # --- fit -----------------------------------------------------------------
    t0 = time.perf_counter()
    history = fit_card_to_fullsim(
        trainee,
        train_dataloader,
        val_dataloader,
        param_groups,
        n_steps=args.n_steps,
        log_every=args.log_every,
        snapshot_parameters=True,
        rank=0,
        device=device,
        intermediate_plot_dir=args.intermediate_plot_dir or None,
        plot_every=args.plot_every,
        early_stopping_patience=args.early_stopping_patience,
        lr_scheduler_patience=args.lr_scheduler_patience,
        lr_scheduler_factor=args.lr_scheduler_factor,
        loss_name=loss_name,
        comet_exp=comet_exp,
        comet_suffix=COMET_SUFFIX,
        **loss_kwargs,
    )
    elapsed = time.perf_counter() - t0

    # --- results -------------------------------------------------------------
    val_losses = [v for v in history.get("val_loss", []) if v is not None]
    if not val_losses:
        end_comet_experiment(comet_exp)
        raise SystemExit("[adam] the fit recorded no validation loss; nothing to report.")
    best_i = min(range(len(val_losses)), key=lambda i: val_losses[i])
    best_val = float(val_losses[best_i])
    init_val = float(val_losses[0])
    print(f"[adam] done in {elapsed:.1f}s ({len(history['step'])} epochs)  "
          f"best val_loss = {best_val:.6e} at epoch {history['step'][best_i]}  "
          f"(init {init_val:.6e}, delta = {best_val - init_val:+.3e})")

    metadata = {
        "source_summary": str(args.gebo_summary),
        "root_file": str(root_file),
        "n_events": n_events,
        "batch_size": args.batch_size,
        "seed": seed,
        "loss": loss_name,
        "lr": args.lr,
        "lr_scale": group_lr_scale,
        "n_steps": args.n_steps,
        "optuna_config": str(optuna_config),
        **loss_kwargs,
    }
    write_history_json(output_dir / "history.json", history, metadata)

    # The best epoch's parameters, as a standard param config. history's
    # "parameters" snapshots are physical values, so they map straight back into
    # a config -- the trainee itself holds the LAST epoch, which early stopping
    # means is generally not the best one.
    snapshots = history.get("parameters", [])
    best_physical_out = dict(snapshots[best_i]) if best_i < len(snapshots) else {}
    best_cfg = {
        key: {
            "value": float(best_physical_out.get(key, spec["value"])),
            "trainable": bool(spec["trainable"]),
            "lr_scale": float(spec["lr_scale"]),
        }
        for key, spec in flat_cfg.items()
    }
    best_config_path = output_dir / "best_config.yaml"
    _dump_flat_config(best_cfg, best_config_path)

    # gebo_summary.json in the shared schema, so gebo_optuna_search's resume gate
    # and plot_gebo_results treat this stage exactly like the L-BFGS one.
    # best_raw_params is reconstructed from the best epoch's physical values via
    # the standard transforms, keeping the key set identical to GEBO's.
    vectorizer = ParamVectorizer(
        CMSEnergyFlowDefault(debug=False, learnable=True), trainable_keys=trainable
    )
    param_names = vectorizer.param_names()
    best_raw_out = {
        n: float(pc.to_raw(n.split("[", 1)[0], best_physical_out[n]))
        for n in param_names
        if n in best_physical_out
    }
    out_summary = {
        "method": "Adam",
        "source_summary": str(args.gebo_summary),
        "initial_loss": init_val,
        "best_loss": best_val,
        "best_idx": best_i,
        "best_epoch": history["step"][best_i],
        "best_train_loss": float(history["loss"][best_i]),
        "best_raw_params": best_raw_out,
        "best_physical_params": {k: float(v) for k, v in best_physical_out.items()},
        "best_config_path": str(best_config_path),
        "init_config_path": str(init_config_path),
        "n_total_points": len(history["step"]),
        "dimension": vectorizer.dim,
        "history": [
            {"iteration": i + 1, "loss": v, "best_loss": min(val_losses[: i + 1]),
             "candidate_loss": v}
            for i, v in enumerate(val_losses)
        ],
        "elapsed_s": elapsed,
        "args": metadata,
    }
    summary_path = output_dir / "gebo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2, default=str)

    # Parameter + loss trajectory in gebo_data.pt's schema, so
    # plot_gebo_results can render param_drift_all.pdf / loss_scatter.pdf.
    if snapshots:
        traj_X = np.stack([
            [float(pc.to_raw(n.split("[", 1)[0], snap[n])) for n in param_names if n in snap]
            for snap in snapshots
        ])
        torch.save(
            {
                "train_X": torch.tensor(traj_X, dtype=torch.float64),
                "train_Y": torch.tensor(val_losses, dtype=torch.float64).unsqueeze(-1),
                "best_raw": torch.tensor(
                    [best_raw_out[n] for n in param_names if n in best_raw_out],
                    dtype=torch.float64,
                ),
                "param_names": [n for n in param_names if n in best_raw_out],
            },
            output_dir / "adam_data.pt",
        )

    if comet_exp is not None:
        try:
            comet_exp.log_metric(f"best_val_loss{COMET_SUFFIX}", best_val)
            comet_exp.log_other(f"epochs_run{COMET_SUFFIX}", len(history["step"]))
        except Exception as e:  # noqa: BLE001 - telemetry must not break the run
            print(f"[adam] warning: final Comet logging failed: {e}")
    end_comet_experiment(comet_exp)

    print(f"\n[adam] results saved to {output_dir}/")
    print(f"[adam] best config -> {best_config_path}")
    print(f"[adam] summary     -> {summary_path}")


if __name__ == "__main__":
    main()
