r"""Tune the learnable CMS TorchDelphes card against CMS full-simulation reco.

This script fits the 66 learnable parameters of
:class:`parnassus.torch_delphes.defaults.CMSEnergyFlowDefault` so that its
reconstructed observable distributions match a CMS full-simulation sample.
It is designed to read the ROOT-file format used by the Parnassus paper
(arXiv:2406.01620) and distributed on
`Zenodo record 11389651 <https://zenodo.org/records/11389651>`_, which the
``parnassus-hep/cms-flow`` repository also consumes.

ROOT file schema
----------------
The script expects a ROOT file with a TTree named ``event_tree`` that
contains (at least) the following branches, each as a jagged per-event
array:

- ``truth_pt`` : GeV, generator-level stable-particle transverse momentum
- ``truth_eta`` : generator-level stable-particle pseudorapidity
- ``truth_phi`` : generator-level stable-particle azimuth
- ``truth_class`` : integer particle-type label
  (``0=charged hadron, 1=electron, 2=muon, 3=neutral hadron, 4=photon``)
- ``pflow_pt`` / ``pflow_eta`` / ``pflow_phi`` / ``pflow_class`` : the
  corresponding CMS particle-flow reconstructed objects that serve as
  the fitting target

Class-to-PDG mapping is done with
:func:`parnassus.utils.class_to_pid_vectorized`.

Usage
-----
.. code-block:: shell

    uv run python -m parnassus.torch_delphes.tune_cms_fullsim \
        --root-file /path/to/train_800_1000_filter.root \
        --n-events 2000 \
        --n-steps 100

If ``--root-file`` is omitted (or the path doesn't exist), the script
generates a small synthetic fixture with the same schema, fits against
it, and reports the loss trajectory. This is useful for sandbox
validation and for CI. The real sample lives on Zenodo and is too large
(and the host is often blocked in sandboxed environments) to download
inline.

The fit loop is exactly the one in :mod:`parnassus.torch_delphes.tuning`
adapted to a fixed *external* target: the reco observables are computed
once from the ROOT file and then re-used on every step, and each step
only re-runs the trainee on the truth input.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .config import (
    DEFAULT_BIN_EDGES,
    DEFAULT_OBS_WEIGHTS,
    _DEFAULT_LR_EFFICIENCY,
    _DEFAULT_LR_FRACTIONS,
    _DEFAULT_LR_RESOLUTION,
    _DEFAULT_LR_SCALES,
)
from .data import (
    load_cms_flow_root,
    pflow_target_observables,
    trainee_observables,
    truth_to_particle_tensor,
)
from .distributed import (
    _barrier,
    _cleanup_distributed,
    _init_distributed,
    _is_dist,
    _is_main,
)
from .fixture import write_synthetic_fixture
from .loss import multi_observable_loss_distributed
from .training import fit_card_to_fullsim

# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Entry point for ``python -m parnassus.torch_delphes.tune_cms_fullsim``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-file",
        type=Path,
        default=None,
        help=(
            "Path to a CMS full-simulation ROOT file with an event_tree "
            "following the cms-flow schema. If not provided, a synthetic "
            "fixture is generated under /tmp."
        ),
    )
    parser.add_argument("--n-events", type=int, default=200)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help=(
            "Global learning rate. Only used when --train-what is a strict "
            "subset (scale_only, resolution_only). With --train-what=all the "
            "four --lr-* flags are used instead, because the four parameter "
            "groups have very different natural step sizes."
        ),
    )
    parser.add_argument("--lr-scales", type=float, default=_DEFAULT_LR_SCALES)
    parser.add_argument("--lr-resolution", type=float, default=_DEFAULT_LR_RESOLUTION)
    parser.add_argument("--lr-efficiency", type=float, default=_DEFAULT_LR_EFFICIENCY)
    parser.add_argument("--lr-fractions", type=float, default=_DEFAULT_LR_FRACTIONS)
    parser.add_argument("--n-passes-per-step", type=int, default=2)
    parser.add_argument("--beta", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--history-path",
        type=Path,
        default=None,
        help=(
            "If set, write the full training history (loss trajectory and "
            "per-step parameter snapshots) to this path as JSON. Used by "
            "the plotting scripts and by the JINST-paper figures."
        ),
    )
    parser.add_argument(
        "--train-what",
        choices=("all", "scale_only", "resolution_only"),
        default="all",
        help="Which parameter subset to optimize.",
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=Path("/tmp/tune_cms_fullsim_fixture.root"),
        help="Where to write (and load) the synthetic fallback fixture.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # DDP bootstrap (no-op for plain ``python -m ...`` invocations).
    # ------------------------------------------------------------------
    rank, world_size, local_rank, device = _init_distributed()

    def log(msg: str) -> None:
        """Print only on rank 0 to keep stdout readable under srun."""
        if _is_main(rank):
            print(msg)

    if world_size > 1:
        log(
            f"[DDP] running on {world_size} ranks "
            f"(local_rank={local_rank} on host {socket.gethostname()})"
        )

    # Resolve the input file: use the real file if it exists, otherwise
    # generate a synthetic fixture so the demo is always runnable.
    # Only rank 0 writes the fixture; everyone else waits at a barrier.
    root_file: Path
    if args.root_file is not None and args.root_file.exists():
        root_file = args.root_file
        log(f"Loading full-simulation events from {root_file}")
    else:
        if args.root_file is not None:
            log(f"WARNING: {args.root_file} does not exist; falling back to fixture.")
        log(
            "No real ROOT file provided. Generating synthetic fixture at "
            f"{args.fixture_path} (same schema as Zenodo record 11389651)..."
        )
        if _is_main(rank):
            write_synthetic_fixture(
                args.fixture_path,
                n_events=args.n_events,
                particles_per_event=60,
                seed=args.seed,
            )
        _barrier()
        root_file = args.fixture_path
        log(
            "To fit against the real sample, download e.g. "
            "https://zenodo.org/records/11389651 and rerun with --root-file."
        )

    # ------------------------------------------------------------------
    # Per-rank data sharding.
    # ``args.n_events`` is the *global* event budget; we split it
    # contiguously across ranks. Each rank reads only its own slice
    # from the ROOT file. The "local" event count is what gets passed
    # to the trainee/target observable builders so that scatter-add
    # buffers are sized correctly per rank.
    # ------------------------------------------------------------------
    base = args.n_events // world_size
    extra = args.n_events % world_size
    local_n_events = base + (1 if rank < extra else 0)
    entry_start = rank * base + min(rank, extra)
    log(
        f"[DDP] global n_events={args.n_events}; rank {rank} reads "
        f"events [{entry_start}, {entry_start + local_n_events})"
    )

    arrays = load_cms_flow_root(
        root_file, n_events=local_n_events, entry_start=entry_start
    )
    truth_tensor = truth_to_particle_tensor(arrays, n_events=local_n_events)
    target = pflow_target_observables(arrays, n_events=local_n_events)

    truth_tensor = truth_tensor.to(device)
    target = {k: v.to(device) for k, v in target.items()}
    bin_edges = {k: v.to(device) for k, v in DEFAULT_BIN_EDGES.items()}

    # Use a *globally* reduced count of target particles for the print.
    n_target_local = target["pt"].numel()
    if _is_dist():
        tmp = torch.tensor([n_target_local], dtype=torch.long, device=device)
        dist.all_reduce(tmp, op=dist.ReduceOp.SUM)
        n_target_global = int(tmp.item())
    else:
        n_target_global = n_target_local

    log(
        f"Loaded {args.n_events} events globally ({local_n_events} on rank {rank}), "
        f"{truth_tensor.shape[0]} truth particles (local), "
        f"{n_target_global} PFlow target particles (global)."
    )

    # All ranks must use the *same* initial parameters for the manual-
    # gradient-sync scheme to keep them in sync; ``torch.manual_seed``
    # with the user seed (not seed+rank!) handles that.
    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)

    # Pick the parameter subset to train.
    if args.train_what == "all":
        params_to_train: list[nn.Parameter] | None = None
    else:
        chad_res = trainee.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
        ecal_scale = trainee.ECal.scale_module  # type: ignore[union-attr]
        if args.train_what == "scale_only":
            params_to_train = [chad_res.scale_raw, ecal_scale.scale_raw]
        else:  # resolution_only
            params_to_train = [chad_res.a_raw, chad_res.b_raw]

    print_msg = (
        f"Training {'all 66' if params_to_train is None else len(params_to_train)} "
        f"learnable params for {args.n_steps} Adam steps..."
    )
    log(print_msg)

    def _averaged_loss(n: int = 6) -> float:
        """Average the soft-histogram loss over ``n`` fresh forward passes.

        Single-pass losses fluctuate by several tens of percent due to the
        log-normal smearing and the Gumbel-ST sampling, so a fair
        "before vs after" comparison has to average over samples.

        Under DDP the loss is computed via the differentiable all-reduce
        helper, so the returned value is the *global* loss (identical
        on every rank).

        Returns
        -------
        float
            Mean loss value over the ``n`` passes.
        """
        acc = 0.0
        with torch.no_grad():
            for _ in range(n):
                out = trainee(truth_tensor)
                pred = trainee_observables(out, n_events=local_n_events)
                acc += float(
                    multi_observable_loss_distributed(
                        pred,
                        target,
                        bin_edges,
                        beta=args.beta,
                        weights=DEFAULT_OBS_WEIGHTS,
                    )
                )
        return acc / n

    loss_before = _averaged_loss()
    log(f"Averaged loss at init (6 passes): {loss_before:.4e}")

    history = fit_card_to_fullsim(
        trainee,
        truth_tensor,
        target,
        n_events=local_n_events,
        n_steps=args.n_steps,
        lr=args.lr,
        n_passes_per_step=args.n_passes_per_step,
        beta=args.beta,
        log_every=max(1, args.n_steps // 10),
        parameters_to_train=params_to_train,
        bin_edges=bin_edges,
        lr_scales=args.lr_scales,
        lr_resolution=args.lr_resolution,
        lr_efficiency=args.lr_efficiency,
        lr_fractions=args.lr_fractions,
        snapshot_parameters=args.history_path is not None,
        rank=rank,
    )

    if args.history_path is not None and _is_main(rank):
        args.history_path.parent.mkdir(parents=True, exist_ok=True)
        with args.history_path.open("w") as f:
            json.dump(
                {
                    "loss": history["loss"],
                    "step": history["step"],
                    "parameters": history.get("parameters", []),
                    "n_events": args.n_events,
                    "n_steps": args.n_steps,
                    "lr_scales": args.lr_scales,
                    "lr_resolution": args.lr_resolution,
                    "lr_efficiency": args.lr_efficiency,
                    "lr_fractions": args.lr_fractions,
                    "train_what": args.train_what,
                    "world_size": world_size,
                },
                f,
                indent=2,
            )
        log(f"Wrote training history to {args.history_path}")

    loss_after = _averaged_loss()
    log(f"Averaged loss after training (6 passes): {loss_after:.4e}")
    rel = 100.0 * (loss_before - loss_after) / max(loss_before, 1e-30)
    log(f"  relative improvement: {rel:+.1f}%")

    # Print the learned charged-hadron scale and ECal scale for a quick
    # sanity check. On the synthetic fixture the expected target is
    # chad_scale=1.2 and ecal_scale=1.1 in every region.
    chad_res = trainee.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
    chad_scales = (1.0 + 0.3 * torch.tanh(chad_res.scale_raw)).detach().tolist()
    ecal_scale_vals = (
        (
            1.0
            + 0.3
            * torch.tanh(
                trainee.ECal.scale_module.scale_raw  # type: ignore[union-attr]
            )
        )
        .detach()
        .tolist()
    )
    log("")
    log(f"Final charged-hadron scale (3 eta regions): {chad_scales}")
    log(f"Final ECal scale            (3 eta regions): {ecal_scale_vals}")
    if root_file == args.fixture_path:
        log("(on synthetic fixture: target values are 1.25 and 1.20 respectively)")

    _cleanup_distributed()
