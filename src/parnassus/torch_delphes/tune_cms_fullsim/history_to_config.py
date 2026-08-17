"""Turn a finished fit's ``history.json`` into a param config to warm-start from.

The tuning CLI (:mod:`.cli`) has no ``--resume`` flag and never checkpoints
optimizer state. What it *does* support is starting from arbitrary parameter
values via ``--param-config``, and a fit's ``history.json`` already records
every parameter's converged **physical** value under
``best_result.parameters`` (written by ``training._snapshot`` through the same
:func:`param_config.to_physical` transforms a param config uses). So the two
formats line up key-for-key, and "resume" is a format conversion:

    history.json  best_result.parameters   ->   {key: {value, trainable, lr_scale}}

``trainable`` / ``lr_scale`` are carried over from the config the original fit
ran with (``--source-config``, defaulting to the one recorded in the history's
``metadata.param_config``), so the warm-started fit optimizes the same subset
with the same per-group learning rates.

This is exactly what :mod:`.adam_finetune` does when it writes
``best_config.yaml`` after fine-tuning GEBO's best point; this module just makes
it available for any history.json produced by the plain CLI or the Optuna
search.

SATURATED PARAMETERS
--------------------
Converged fits routinely leave an efficiency / fraction outside the open
interval ``(0.1, 0.9)`` that :func:`param_config.load_param_config` enforces on
*trainable* logits by default. **This module writes the converged value anyway,
exactly, and leaves it trainable.**

That guard is a conditioning heuristic on the STARTING value only -- nothing
constrains a logit during training. Adam walks them far outside the window and
keeps moving them there (measured on a real 84-epoch run: a logit sitting at a
sigmoid Jacobian of 0.002, 45x below the guard boundary, still took 0.036-sized
steps), because Adam normalizes by the gradient's RMS. The card's own
constructor defaults (0.99, 0.98, ...) and several truth values also lie outside
the window. Clamping a converged value back inside would move it AWAY from the
answer the fit had already found, for no benefit; pinning it would freeze a
parameter that is still perfectly fittable.

So the output of this module needs ``--allow-saturated-init`` when fed to the
tuning CLI. Read-only consumers (``plot_fit_results``) already disable the guard
themselves, since they never initialize anything for optimization.

Run with
``python -m parnassus.torch_delphes.tune_cms_fullsim.history_to_config ...``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from parnassus.torch_delphes import param_config as pc

def history_to_flat_config(
    history_path: Path,
    source_config: Path | None = None,
) -> tuple[dict[str, dict], list[tuple[str, float]]]:
    """Build a param config from a history's best epoch.

    Values are copied verbatim from the best epoch -- never clamped, never
    pinned (see SATURATED PARAMETERS in the module docstring).

    Returns
    -------
    (flat_cfg, saturated)
        ``flat_cfg`` is the ``{key: {value, trainable, lr_scale}}`` mapping;
        ``saturated`` lists ``(key, value)`` for every trainable logit that
        landed outside the guard window -- reported so the caller knows the
        config needs ``--allow-saturated-init``, NOT because anything was
        changed.
    """
    with open(history_path) as f:
        payload = json.load(f)

    best = (payload.get("best_result") or {}).get("parameters") or {}
    if not best:
        raise SystemExit(
            f"{history_path}: best_result.parameters is empty. The fit must be run with "
            "snapshot_parameters=True (the CLI and optuna_search both do this)."
        )

    if source_config is None:
        recorded = (payload.get("metadata") or {}).get("param_config")
        if not recorded:
            raise SystemExit(
                f"{history_path}: no metadata.param_config recorded; pass --source-config "
                "with the config the original fit ran with (it supplies trainable/lr_scale)."
            )
        source_config = Path(recorded)
    if not Path(source_config).exists():
        raise SystemExit(f"--source-config {source_config} does not exist.")

    src = pc.load_param_config(source_config)
    if set(src) != set(best):
        missing = sorted(set(src) - set(best))[:5]
        extra = sorted(set(best) - set(src))[:5]
        raise SystemExit(
            f"key mismatch between {history_path} and {source_config}: "
            f"{len(missing)} missing (e.g. {missing}), {len(extra)} extra (e.g. {extra})."
        )

    flat_cfg: dict[str, dict] = {}
    saturated: list[tuple[str, float]] = []
    for key, spec in src.items():
        value = float(best[key])
        trainable = bool(spec["trainable"])
        base = key.split("[", 1)[0]
        if trainable and pc.param_transform_kind(base) == "logit":
            if not (pc._TRAINABLE_LOGIT_MIN < value < pc._TRAINABLE_LOGIT_MAX):  # noqa: SLF001
                saturated.append((key, value))
        flat_cfg[key] = {
            "value": value,
            "trainable": trainable,
            "lr_scale": float(spec["lr_scale"]),
        }
    return flat_cfg, saturated


def main() -> None:
    """Entry point for ``python -m ...tune_cms_fullsim.history_to_config``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True, help="Finished fit's history.json.")
    parser.add_argument("--output", type=Path, required=True, help="Param config YAML to write.")
    parser.add_argument(
        "--source-config",
        type=Path,
        default=None,
        help="Config the original fit ran with (supplies trainable/lr_scale). "
        "Default: the history's metadata.param_config.",
    )
    args = parser.parse_args()

    flat_cfg, saturated = history_to_flat_config(args.history, args.source_config)

    out = {
        key: {
            "value": float(spec["value"]),
            "trainable": bool(spec["trainable"]),
            "lr_scale": float(spec["lr_scale"]),
        }
        for key, spec in flat_cfg.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)

    # Reload through the real loader so a bad file fails HERE, not 20 minutes
    # into the warm-started fit. The saturated guard is off for the same reason
    # it is off for the fit itself: these values came from a converged run.
    pc.load_param_config(args.output, enforce_saturated_guard=False)

    n_train = sum(1 for s in flat_cfg.values() if s["trainable"])
    print(f"[history_to_config] wrote {args.output} ({len(flat_cfg)} params, {n_train} trainable)")
    print("[history_to_config] verified: reloads cleanly via param_config.load_param_config")
    if saturated:
        print(
            f"[history_to_config] {len(saturated)} trainable logit(s) sit outside the "
            f"({pc._TRAINABLE_LOGIT_MIN}, {pc._TRAINABLE_LOGIT_MAX}) init window "  # noqa: SLF001
            "-- kept EXACTLY as converged, not clamped:"
        )
        for key, value in saturated:
            print(f"    {key:<52} {value:.6g}")
        print(
            "[history_to_config] pass --allow-saturated-init to "
            "`python -m parnassus.torch_delphes.tune_cms_fullsim` when using this config."
        )


if __name__ == "__main__":
    main()
