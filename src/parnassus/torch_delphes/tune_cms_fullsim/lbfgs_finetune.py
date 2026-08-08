r"""L-BFGS fine-tuning of the best model from a GEBO run.

Gradient-Enhanced Bayesian Optimization (:mod:`.gebo_search`) is a *global*
search: it finds a good basin in the 66-D detector-parameter space but stops at
the best point it happened to query. This script takes that best point and
polishes it with a *local* quasi-Newton refinement -- **bounded L-BFGS-B**
(SciPy) driven by the exact same differentiable objective GEBO used (loss **and**
its analytic 66-D gradient, from :func:`.gebo_search.make_objective`). Because the
objective already returns the gradient, L-BFGS-B needs no finite differencing,
and the raw-space bounds from the run's ``optuna_config`` keep every parameter
in range.

It reuses the run's own settings (root file, event count, loss, per-species
weights, batch size, seed, trainable set) straight from ``gebo_summary.json`` so
the refinement optimizes exactly what the search did -- only the optimizer
changes. Single-device (CPU/one GPU); no DDP.

Outputs (under ``--output-dir``, default ``<gebo-dir>/lbfgs``):

- ``best_config.yaml``   -- the fine-tuned card as a standard param config
- ``gebo_summary.json``  -- summary in the schema :mod:`.plot_gebo_results`
  consumes, so the same plotting script renders the fine-tuned observables
- ``lbfgs_data.pt``      -- the final raw parameter vector + names

Usage
-----
.. code-block:: shell

    python -m parnassus.torch_delphes.tune_cms_fullsim.lbfgs_finetune \
        --gebo-summary doc/figures/gebo_w1_cosine/gebo_summary.json \
        --n-steps 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import minimize

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .data import (
    load_cms_flow_root,
    load_pflow_targets_ragged,
    load_truth_events_ragged,
)
from .finetune_utils import (
    load_archived_config,
    make_setting_resolver,
    trainable_bases,
)
from .gebo_plotting import build_card_from_raw_params
from .gebo_search import (
    ParamVectorizer,
    load_bounds_from_optuna_config,
    make_objective,
)


def main() -> None:
    """Entry point for ``python -m ...tune_cms_fullsim.lbfgs_finetune``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gebo-summary",
        type=Path,
        required=True,
        help="Path to a GEBO run's gebo_summary.json (its best point is the start).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write results (default: <gebo-summary dir>/lbfgs).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="Max L-BFGS-B iterations (scipy maxiter, default 100).",
    )
    parser.add_argument(
        "--max-fun",
        type=int,
        default=None,
        help="Max objective evaluations (scipy maxfun; default: scipy default).",
    )
    # Overrides for the reconstructed objective (default: reuse the run's values).
    parser.add_argument("--n-events", type=int, default=None,
                        help="Override the run's event count (default: reuse).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override the run's forward-pass batch size (default: reuse).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override the run's RNG seed (default: reuse).")
    parser.add_argument("--root-file", type=Path, default=None,
                        help="Override the run's ROOT file (default: reuse).")
    parser.add_argument("--optuna-config", type=Path, default=None,
                        help="Override the run's optuna_config (bounds/trainable set).")
    args = parser.parse_args()

    # Line-buffer stdout: this process is normally launched with stdout
    # redirected to a log file (see gebo_optuna_search.py's _run_subprocess),
    # and Python fully-buffers a non-tty stdout by default -- so without this,
    # none of the [lbfgs] progress prints below (or the per-iteration
    # _callback prints) reach the log until the process exits, making a slow
    # run indistinguishable from a hung one.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    if not args.gebo_summary.exists():
        raise SystemExit(f"--gebo-summary {args.gebo_summary} does not exist.")
    with open(args.gebo_summary) as f:
        summary = json.load(f)

    run_args = summary.get("args", {}) or {}
    best_raw_params = summary.get("best_raw_params", {})
    if not best_raw_params:
        raise SystemExit(f"{args.gebo_summary} has no 'best_raw_params'.")

    # Recover run settings from the archived config (reliable even for an
    # intermediate summary that has no 'args'). The run dir is the summary's
    # parent, or its grandparent when the summary lives under 'intermediate/'.
    run_dir = args.gebo_summary.parent
    if run_dir.name == "intermediate":
        run_dir = run_dir.parent
    cfg_ns = load_archived_config(run_dir)
    if cfg_ns is not None:
        print(f"[lbfgs] recovered run settings from {run_dir}/configs/")

    # Resolve each setting: CLI override -> summary 'args' -> archived config.
    _cfg = make_setting_resolver(run_args, cfg_ns)

    root_file = args.root_file or _cfg("root_file")
    if root_file is None:
        raise SystemExit("could not resolve 'root_file' from the summary or an "
                         "archived config; pass --root-file explicitly.")
    root_file = Path(root_file)
    n_events = args.n_events if args.n_events is not None else int(_cfg("n_events", -1))
    batch_size = args.batch_size if args.batch_size is not None else int(_cfg("batch_size", 256))
    seed = args.seed if args.seed is not None else int(_cfg("seed", 0))
    optuna_config = args.optuna_config or _cfg("optuna_config")
    if optuna_config is None:
        raise SystemExit("could not resolve 'optuna_config' from the summary or an "
                         "archived config; pass --optuna-config explicitly.")
    optuna_config = Path(optuna_config)
    loss_name = _cfg("loss", "wasserstein_1d")

    output_dir = args.output_dir or (args.gebo_summary.parent / "lbfgs")
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "lbfgs_state.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lbfgs] device = {device}")
    print(f"[lbfgs] source run: {args.gebo_summary}  (best_loss={summary.get('best_loss')})")

    # --- data ----------------------------------------------------------------
    print(f"[lbfgs] loading {n_events} events from {root_file} ...")
    t0 = time.perf_counter()
    arrays = load_cms_flow_root(root_file, n_events=n_events)
    truth_ragged = load_truth_events_ragged(arrays)
    target_ragged = load_pflow_targets_ragged(arrays)
    del arrays
    print(f"[lbfgs] loaded {len(truth_ragged)} events in {time.perf_counter() - t0:.1f}s")

    # --- vectorizer + bounds (same trainable set the run used) ---------------
    trainable = trainable_bases(optuna_config)
    probe = CMSEnergyFlowDefault(debug=False, learnable=True)
    vectorizer = ParamVectorizer(probe, trainable_keys=trainable)
    param_names = vectorizer.param_names()
    dim = vectorizer.dim
    bounds_t = load_bounds_from_optuna_config(optuna_config, vectorizer)  # (dim, 2)
    scipy_bounds = [(float(bounds_t[i, 0]), float(bounds_t[i, 1])) for i in range(dim)]
    print(f"[lbfgs] search space dimension = {dim}")

    # --- objective (loss + analytic gradient), identical to the GEBO run -----
    objective = make_objective(
        vectorizer=vectorizer,
        truth_ragged=truth_ragged,
        target_ragged=target_ragged,
        device=device,
        loss_name=loss_name,
        count_weight=float(_cfg("count_weight", 0.1)),
        calo_count_weight=float(_cfg("calo_count_weight", 1.0)),
        count_rate_floor=float(_cfg("count_rate_floor", 0.05)),
        event_weight=float(_cfg("event_weight", 0.1)),
        pid_weighting=_cfg("pid_weighting", "sqrt_fraction"),
        pid_weight_floor=float(_cfg("pid_weight_floor", 0.0)),
        batch_size=batch_size,
        seed=seed,
    )

    # --- starting point: the run's best raw vector ---------------------------
    missing = [n for n in param_names if n not in best_raw_params]
    if missing:
        raise SystemExit(
            f"best_raw_params is missing {len(missing)} trainable keys "
            f"(e.g. {missing[:3]}); does --optuna-config match the run?"
        )
    orig_x0 = np.array([float(best_raw_params[n]) for n in param_names], dtype=np.float64)
    x0 = orig_x0

    # --- resume from a checkpoint, if this round's L-BFGS died mid-run --------
    # Unlike GEBO, a single scipy.optimize.minimize() call is one blocking
    # operation with no native checkpoint/resume support, so we save our own
    # state (current point + full history) after every callback iteration --
    # mirroring gebo_search.py's per-iteration gebo_state.pt -- and resume from
    # the last accepted point instead of restarting from GEBO's best every time
    # this process gets relaunched.
    resumed = False
    history: list[dict] = []
    traj: list[np.ndarray] = []  # accepted parameter vector after each iteration
    true_init_loss: float | None = None
    if state_path.exists():
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
        n_done = int(ckpt.get("iteration", 0))
        if n_done > 0:
            resumed = True
            x0 = ckpt["x"].numpy().astype(np.float64)
            history = list(ckpt.get("history", []))
            traj = [row.numpy().astype(np.float64) for row in ckpt.get("traj", [])]
            true_init_loss = ckpt.get("true_init_loss")

    _state: dict = {"loss": None}

    def fun_and_jac(x: np.ndarray) -> tuple[float, np.ndarray]:
        theta = torch.tensor(x, dtype=torch.float64).unsqueeze(0)  # (1, dim)
        out = objective(theta)  # (1, dim+1) = [loss, grad...]
        loss = float(out[0, 0])
        grad = out[0, 1:].detach().cpu().numpy().astype(np.float64)
        _state["loss"] = loss
        return loss, grad

    current_loss, _ = fun_and_jac(x0)
    if true_init_loss is None:
        true_init_loss = current_loss
    if resumed:
        print(
            f"[lbfgs] RESUMING from {state_path}: {len(history)} iterations already "
            f"done (of {args.n_steps} max), current loss = {current_loss:.6e}"
        )
    else:
        print(f"[lbfgs] initial loss (GEBO best) = {current_loss:.6e}")

    def _callback(xk: np.ndarray) -> None:
        history.append({
            "iteration": len(history) + 1,
            "loss": _state["loss"],
            # duplicated under the GEBO keys so plot_gebo_results can render it
            "best_loss": _state["loss"],
            "candidate_loss": _state["loss"],
        })
        traj.append(np.array(xk, dtype=np.float64))
        print(f"[lbfgs] iter {len(history):3d}  loss={_state['loss']:.6e}")
        torch.save(
            {
                "iteration": len(history),
                "x": torch.tensor(xk, dtype=torch.float64),
                "history": history,
                "traj": torch.stack([torch.tensor(t, dtype=torch.float64) for t in traj]),
                "true_init_loss": true_init_loss,
            },
            state_path,
        )

    remaining_steps = max(args.n_steps - len(history), 0)
    t0 = time.perf_counter()
    if remaining_steps == 0:
        # The checkpoint already reached the iteration budget, but the process
        # died before this script could write its final output -- nothing left
        # to optimize, just finalize with what the checkpoint already has.
        print(
            f"[lbfgs] resume point already reached the {args.n_steps}-iteration "
            "budget; finalizing without further optimization."
        )
        res = SimpleNamespace(
            fun=current_loss, x=x0, success=True, status=0,
            message="resume point already reached the iteration budget",
            nit=len(history), nfev=len(history),
        )
    else:
        options = {"maxiter": remaining_steps}
        if args.max_fun is not None:
            options["maxfun"] = args.max_fun
        res = minimize(
            fun_and_jac,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=scipy_bounds,
            callback=_callback,
            options=options,
        )
    elapsed = time.perf_counter() - t0
    final_loss = float(res.fun)
    print(
        f"[lbfgs] done in {elapsed:.1f}s  ({res.nit} iters, {res.nfev} evals)  "
        f"final loss = {final_loss:.6e}  (init {true_init_loss:.6e}, "
        f"Δ = {final_loss - true_init_loss:+.3e})"
    )
    print(f"[lbfgs] scipy: success={res.success} status={res.status} — {res.message}")

    # --- materialize + save --------------------------------------------------
    best_raw = torch.tensor(res.x, dtype=torch.float64)
    best_raw_out = {n: float(best_raw[i]) for i, n in enumerate(param_names)}
    best_physical = {
        n: float(pc.to_physical(n.split("[", 1)[0], best_raw[i].unsqueeze(0)))
        for i, n in enumerate(param_names)
    }

    # Dump a standard param config (trainable subset set to the fine-tuned
    # values; pinned params at their card defaults, matching the GEBO run).
    best_card = build_card_from_raw_params(best_raw_out, torch.device("cpu"))
    best_config_path = output_dir / "best_config.yaml"
    pc.dump_param_config(best_card, best_config_path)

    out_summary = {
        "method": "L-BFGS-B",
        "source_summary": str(args.gebo_summary),
        "initial_loss": true_init_loss,
        "best_loss": final_loss,
        "best_idx": 0,
        "best_raw_params": best_raw_out,
        "best_physical_params": best_physical,
        "best_config_path": str(best_config_path),
        "n_total_points": len(history),
        "dimension": dim,
        "bounds": {
            "low": [float(bounds_t[i, 0]) for i in range(dim)],
            "high": [float(bounds_t[i, 1]) for i in range(dim)],
        },
        "history": history,
        "scipy_result": {
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "n_iterations": int(res.nit),
            "n_evaluations": int(res.nfev),
            "elapsed_s": elapsed,
        },
        "args": {
            "root_file": str(root_file),
            "n_events": n_events,
            "batch_size": batch_size,
            "seed": seed,
            "loss": loss_name,
            "optuna_config": str(optuna_config),
            "n_steps": args.n_steps,
            "n_initial": 0,
        },
    }
    summary_path = output_dir / "gebo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out_summary, f, indent=2, default=str)

    # Parameter + loss trajectory (starting from the GEBO best = orig_x0), so
    # plot_gebo_results can render param_drift_all.pdf / loss_scatter.pdf. The
    # ``train_X`` / ``train_Y`` keys mirror gebo_data.pt's schema (train_Y here
    # is just the (n, 1) loss column the plots read). ``traj``/``history`` span
    # every iteration across all resumes, so this covers the full run, not just
    # whatever's left after the most recent restart.
    traj_X = np.stack([orig_x0, *traj]) if traj else orig_x0[None, :]
    traj_losses = [true_init_loss] + [h["loss"] for h in history]
    torch.save(
        {
            "train_X": torch.tensor(traj_X, dtype=torch.float64),
            "train_Y": torch.tensor(traj_losses, dtype=torch.float64).unsqueeze(-1),
            "best_raw": best_raw,
            "param_names": param_names,
        },
        output_dir / "lbfgs_data.pt",
    )

    print(f"\n[lbfgs] results saved to {output_dir}/")
    print(f"[lbfgs] best config -> {best_config_path}")
    print(f"[lbfgs] summary     -> {summary_path}")
    print(
        "[lbfgs] plot with:\n"
        f"  python -m parnassus.torch_delphes.tune_cms_fullsim.plot_gebo_results "
        f"--summary {summary_path} --root-file {root_file} "
        f"--truth-config {_cfg('truth_config', '<truth_config>')}"
    )


if __name__ == "__main__":
    main()
