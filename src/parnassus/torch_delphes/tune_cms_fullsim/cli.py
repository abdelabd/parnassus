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
    _DEFAULT_LR,
    _DEFAULT_LR_EFFICIENCY,
    _DEFAULT_LR_FRACTIONS,
    _DEFAULT_LR_RESOLUTION,
    _DEFAULT_LR_SCALES,
)
from .data import (
    load_cms_flow_root, load_truth_events, load_pflow_targets, split_truth_objects, split_pflow_targets,
)    

from .dataloader import DelphesDataSet, DelphesDataLoader

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
        default=_DEFAULT_LR,
        help=(
            "Global learning-rate magnitude. Each parameter group's effective "
            "Adam learning rate is --lr times its --lr-<group> ratio, so this "
            "is the single knob for sweeping the overall step size. For a "
            "strict --train-what subset it is multiplied by that subset's "
            "ratio (--lr-scales or --lr-resolution)."
        ),
    )
    parser.add_argument(
        "--lr-scales", type=float, default=_DEFAULT_LR_SCALES,
        help="Relative LR ratio for the scales group (effective lr = --lr * this).",
    )
    parser.add_argument(
        "--lr-resolution", type=float, default=_DEFAULT_LR_RESOLUTION,
        help="Relative LR ratio for the resolution group (effective lr = --lr * this).",
    )
    parser.add_argument(
        "--lr-efficiency", type=float, default=_DEFAULT_LR_EFFICIENCY,
        help="Relative LR ratio for the efficiency group (effective lr = --lr * this).",
    )
    parser.add_argument(
        "--lr-fractions", type=float, default=_DEFAULT_LR_FRACTIONS,
        help="Relative LR ratio for the fractions group (effective lr = --lr * this).",
    )
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
    truth_tensor = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    train_truth_tensor, val_truth_tensor = split_truth_objects(truth_tensor, train_fraction=0.8, seed=args.seed)
    train_target, val_target = split_pflow_targets(target, train_fraction=0.8, seed=args.seed)

    train_dataset = DelphesDataSet(train_truth_tensor, train_target, device=device)
    val_dataset = DelphesDataSet(val_truth_tensor, val_target, device=device)

    train_dataloader = DelphesDataLoader(train_dataset, batch_size=512, shuffle=True)
    val_dataloader = DelphesDataLoader(val_dataset, batch_size=512, shuffle=False)

    # truth_tensor = truth_tensor.to(device)
    # target = {k: v.to(device) for k, v in target.items()}
    bin_edges = {k: v.to(device) for k, v in DEFAULT_BIN_EDGES.items()}

    # All ranks must use the *same* initial parameters for the manual-
    # gradient-sync scheme to keep them in sync; ``torch.manual_seed``
    # with the user seed (not seed+rank!) handles that.
    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)

    # Pick the parameter subset to train. ``lr_for_fit`` is the value handed to
    # fit_card_to_fullsim's ``lr`` arg: for --train-what=all it is the base
    # magnitude (build_parameter_groups applies the per-group ratios); for a
    # strict subset we fold in that group's ratio here so the subset steps at
    # the same effective rate it would inside the "all" run.
    if args.train_what == "all":
        params_to_train: list[nn.Parameter] | None = None
        lr_for_fit = args.lr
    else:
        chad_res = trainee.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
        ecal_scale = trainee.ECal.scale_module  # type: ignore[union-attr]
        if args.train_what == "scale_only":
            params_to_train = [chad_res.scale_raw, ecal_scale.scale_raw]
            lr_for_fit = args.lr * args.lr_scales
        else:  # resolution_only
            params_to_train = [chad_res.a_raw, chad_res.b_raw]
            lr_for_fit = args.lr * args.lr_resolution

    print_msg = (
        f"Training {'all 66' if params_to_train is None else len(params_to_train)} "
        f"learnable params for {args.n_steps} Adam steps..."
    )
    log(print_msg)


    history = fit_card_to_fullsim(
        trainee,
        train_dataloader,
        val_dataloader,
        # n_events=local_n_events,
        n_steps=args.n_steps,
        lr=lr_for_fit,
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
        device=device,
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
                    # --lr is the global magnitude; lr_* are per-group ratios;
                    # effective_lr[group] = lr * lr_<group> is what Adam used.
                    "lr": args.lr,
                    "lr_scales": args.lr_scales,
                    "lr_resolution": args.lr_resolution,
                    "lr_efficiency": args.lr_efficiency,
                    "lr_fractions": args.lr_fractions,
                    "effective_lr": {
                        "scales": args.lr * args.lr_scales,
                        "resolution": args.lr * args.lr_resolution,
                        "efficiency": args.lr * args.lr_efficiency,
                        "fractions": args.lr * args.lr_fractions,
                    },
                    "train_what": args.train_what,
                    "world_size": world_size,
                },
                f,
                indent=2,
            )
        log(f"Wrote training history to {args.history_path}")

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
