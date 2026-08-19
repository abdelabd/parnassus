"""Build the per-stage ``--param-config`` of the sequential full-phase-space fit.

Every stage of the chain runs the plain tuning CLI (``python -m
parnassus.torch_delphes.tune_cms_fullsim --param-config <this file> --lr 1``) on one
sample with one block of parameters trainable. The config written here is a FULL
``{value, trainable, lr_scale}`` param config (:mod:`parnassus.torch_delphes.param_config`
format, every card scalar present):

- ``value``: the card constructor defaults for the first stage, or -- with
  ``--from-history`` -- every parameter's value at the end of the previous stage, read
  from that stage's ``history.json`` (``--pick last``: the last epoch; ``--pick best``:
  the ``best_result`` = min val-loss epoch). Frozen and fitted parameters alike are
  carried, so a block fitted earlier stays at its fitted value.
- ``trainable``: True exactly for the keys listed under ``trainable:`` in the stage
  YAML (``--stage``); everything else is frozen. Without ``--stage`` every parameter is
  frozen -- that is the final fitted card after the last stage.
- ``lr_scale``: the ABSOLUTE per-group Adam learning rate (``--lr-resolution`` /
  ``--lr-scale`` / ``--lr-efficiency``; defaults = the optuna seed-trial values), to be
  used with ``--lr 1`` exactly like a materialized optuna config.

Stage YAML schema (see the ``stage*_*.yaml`` files next to this module)::

    name: stage2_chads          # output sub-directory
    process: ksgun              # sample: <SAMPLE_DIR>/<pattern % process>.root
    trainable: [<scalar key>, ...]
    extra_args: [--no-pair-mass]   # appended to the tuning CLI call (optional)

``--show <field>`` prints one field of the stage YAML (used by run_sequential.sh).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults.CMSDefault import CMSEnergyFlowDefault
from parnassus.torch_delphes.tune_cms_fullsim.optuna_search import LR_GROUPS, _group_of

# Absolute per-group Adam learning rates (used with ``--lr 1``): the seed-trial values
# of the optuna configs, i.e. what every M1-M3 / dijet reference fit ran with.
DEFAULT_GROUP_LR: dict[str, float] = {
    "resolution": 4.9e-3,
    "scale": 1.9e-3,
    "efficiency": 1.5e-2,
}
PICK_CHOICES: tuple[str, ...] = ("last", "best")


def load_stage(path: str | Path) -> dict:
    """Parse and validate a stage YAML.

    Returns
    -------
    dict
        ``{name, process, trainable (list[str]), extra_args (list[str])}``.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or "trainable" not in raw:
        raise ValueError(f"{path}: expected a mapping with at least a 'trainable' list.")
    trainable = raw["trainable"] or []
    if not isinstance(trainable, list) or not all(isinstance(k, str) for k in trainable):
        raise ValueError(f"{path}: 'trainable' must be a list of scalar keys.")
    extra = raw.get("extra_args") or []
    if not isinstance(extra, list) or not all(isinstance(a, str) for a in extra):
        raise ValueError(f"{path}: 'extra_args' must be a list of strings.")
    return {
        "name": str(raw.get("name") or Path(path).stem),
        "process": str(raw.get("process") or ""),
        "trainable": trainable,
        "extra_args": extra,
    }


def history_parameters(path: str | Path, pick: str = "last") -> dict[str, float]:
    """All parameter values (physical units, ``{scalar key: value}``) recorded in a
    ``history.json`` written by the tuning CLI / optuna_search.

    Returns
    -------
    dict[str, float]
        ``best_result.parameters`` for ``pick == "best"``; the parameters of the last
        epoch in ``history`` for ``pick == "last"``.
    """
    if pick not in PICK_CHOICES:
        raise ValueError(f"pick must be one of {PICK_CHOICES}, got {pick!r}")
    with open(path) as f:
        payload = json.load(f)
    if pick == "best":
        params = (payload.get("best_result") or {}).get("parameters")
    else:
        epochs = payload.get("history") or {}
        if not epochs:
            raise ValueError(f"{path}: empty 'history' -- was the fit run with --history-path?")
        last = max(epochs, key=lambda k: int(k.rsplit("_", 1)[1]))
        params = epochs[last].get("parameters")
    if not params:
        raise ValueError(
            f"{path}: no parameter snapshot found ({pick}); the tuning CLI records them "
            "only when --history-path is given."
        )
    return {str(k): float(v) for k, v in params.items()}


def build_stage_config(
    trainable: list[str],
    values: dict[str, float] | None = None,
    group_lr: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Assemble the full flat param config of one stage.

    Parameters
    ----------
    trainable : list[str]
        Scalar keys to fit in this stage; must all be card keys.
    values : dict[str, float] or None
        Start values (physical units) for every scalar; ``None`` = card defaults.
        Must cover every card key when given (a previous stage's full snapshot).
    group_lr : dict[str, float] or None
        Absolute per-group learning rate; ``None`` = :data:`DEFAULT_GROUP_LR`.

    Returns
    -------
    dict[str, dict]
        ``{key: {value, trainable, lr_scale}}`` over every card scalar, in card order.
    """
    lr = {**DEFAULT_GROUP_LR, **(group_lr or {})}
    missing_groups = [g for g in LR_GROUPS if g not in lr]
    if missing_groups:
        raise ValueError(f"group_lr misses {missing_groups}")
    defaults = pc.card_default_config(CMSEnergyFlowDefault(debug=False, learnable=True))
    unknown = sorted(set(trainable) - set(defaults))
    if unknown:
        raise ValueError(f"trainable keys are not card scalars: {unknown}")
    if values is not None:
        missing = sorted(set(defaults) - set(values))
        if missing:
            raise ValueError(f"start values miss {len(missing)} card scalars, e.g. {missing[:3]}")
    train_set = set(trainable)
    cfg: dict[str, dict] = {}
    for key, spec in defaults.items():
        value = float(values[key]) if values is not None else float(spec["value"])
        cfg[key] = {
            "value": float(f"{value:.8g}"),
            "trainable": key in train_set,
            "lr_scale": float(lr[_group_of(key.split("[", 1)[0])]),
        }
    return cfg


def write_config(cfg: dict[str, dict], path: str | Path) -> None:
    """Write ``cfg`` as a param-config YAML and re-read it through
    :func:`param_config.load_param_config` (validates e.g. trainable-logit windows).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    pc.load_param_config(path)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point (see the module docstring)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--stage", type=Path, default=None, help="Stage YAML (trainable mask).")
    parser.add_argument(
        "--from-history",
        type=Path,
        default=None,
        help="Previous stage's history.json: start every parameter from its value there.",
    )
    parser.add_argument("--pick", choices=PICK_CHOICES, default="last")
    parser.add_argument("--out", type=Path, default=None, help="Output param-config YAML.")
    parser.add_argument("--show", type=str, default=None, help="Print one stage field and exit.")
    for g in LR_GROUPS:
        parser.add_argument(f"--lr-{g}", type=float, default=DEFAULT_GROUP_LR[g])
    args = parser.parse_args(argv)

    stage = load_stage(args.stage) if args.stage is not None else None
    if args.show is not None:
        if stage is None:
            raise SystemExit("--show needs --stage")
        val = stage[args.show]
        print(" ".join(val) if isinstance(val, list) else val)
        return
    if args.out is None:
        raise SystemExit("--out is required (unless --show)")

    values = history_parameters(args.from_history, args.pick) if args.from_history else None
    trainable = stage["trainable"] if stage else []
    cfg = build_stage_config(
        trainable,
        values,
        {g: getattr(args, f"lr_{g}") for g in LR_GROUPS},
    )
    write_config(cfg, args.out)
    n_train = sum(1 for s in cfg.values() if s["trainable"])
    src = f"{args.from_history} ({args.pick})" if args.from_history else "card defaults"
    print(
        f"[stage-config] wrote {args.out}: {len(cfg)} scalars, {n_train} trainable, "
        f"start values from {src}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
