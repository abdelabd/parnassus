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

``--root-file`` is required and must point at an existing CMS
full-simulation ROOT file with the cms-flow schema. Generate one with
:mod:`parnassus.torch_delphes.generate_pseudodata`, or download the real
sample from Zenodo record 11389651.

The fit loop is exactly the one in :mod:`parnassus.torch_delphes.tuning`
adapted to a fixed *external* target: the reco observables are computed
once from the ROOT file and then re-used on every step, and each step
only re-runs the trainee on the truth input.
"""

from __future__ import annotations

import argparse
import gc
import json
import socket
from pathlib import Path

import torch

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .config import _DEFAULT_LR
from .data import (
    load_cms_flow_root,
    load_truth_events_ragged,
    load_pflow_targets_ragged,
    split_truth_objects_ragged,
    split_pflow_targets_ragged,
)

from .dataloader import DelphesDataSet, DelphesDataLoader

from .distributed import (
    _cleanup_distributed,
    _init_distributed,
    _is_main,
)
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
        required=True,
        help=(
            "Path to a CMS full-simulation ROOT file with an event_tree "
            "following the cms-flow schema. Required: generate one with "
            "parnassus.torch_delphes.generate_pseudodata, or download the real "
            "sample from Zenodo record 11389651."
        ),
    )
    parser.add_argument("--n-events", type=int, default=-1)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument(
        "--lr",
        type=float,
        default=_DEFAULT_LR,
        help=(
            "Global learning-rate magnitude. Each parameter's effective Adam "
            "learning rate is --lr times its per-parameter 'lr_scale' from the "
            "--param-config file, so this is the single knob for sweeping the "
            "overall step size."
        ),
    )
    parser.add_argument(
        "--param-config",
        type=Path,
        required=True,
        help=(
            "Path to a YAML parameter config (see "
            "parnassus.torch_delphes.param_config). Its 'value' fields "
            "initialize every learnable parameter, 'trainable' selects the "
            "optimized subset (per-element), and 'lr_scale' sets each "
            "parameter's effective Adam lr = --lr * lr_scale. Shipped configs "
            "live under parnassus/torch_delphes/param_configs/."
        ),
    )
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
        "--intermediate-plot-dir",
        type=str,
        default="doc/figures/intermediate_plots",
        help=(
            "Directory for per-epoch intermediate observable plots: one "
            "multi-page PDF per epoch (intermediate_epoch_<step>.pdf, one "
            "observable per page) comparing the trainee prediction to the "
            "full-sim target, with each observable's soft-hist MSE in the "
            "page title as a distribution-mismatch diagnostic. Pass an empty "
            "string to disable. Only the main rank plots."
        ),
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help=(
            "Save intermediate plots every N epochs (default 1 = every epoch). "
            "The final / early-stopped epoch is always plotted."
        ),
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

    # Require a real input file -- no synthetic fallback. Point --root-file at a
    # CMS full-simulation ROOT file (cms-flow schema).
    root_file = args.root_file
    if not root_file.exists():
        raise SystemExit(
            f"--root-file {root_file} does not exist. Provide a CMS full-simulation "
            "ROOT file (cms-flow schema): generate one with "
            "parnassus.torch_delphes.generate_pseudodata, or download the real sample "
            "from Zenodo record 11389651."
        )
    log(f"Loading full-simulation events from {root_file}")

    arrays = load_cms_flow_root(
        root_file, n_events=args.n_events
    )
    # Ragged (no global padding): truth particles are kept as a per-event list and
    # each batch is padded to its own max in delphes_collate_fn. Padding every event
    # to the GLOBAL max multiplicity here would allocate ~50 GB at 100k events -- and
    # the fit loop un-pads it on the very next line anyway.
    truth_ragged = load_truth_events_ragged(arrays)
    # ``target`` carries the per-reco-bin per-species counts (chad/electron/muon
    # _region_counts) the differentiable count terms match the trainee's reco-bin
    # migration against.
    target = load_pflow_targets_ragged(arrays)

    # The uproot arrays dict is the largest remaining transient; free it before the
    # train/val split so peak RSS stays low.
    del arrays
    gc.collect()

    train_truth_tensor, val_truth_tensor = split_truth_objects_ragged(truth_ragged, train_fraction=0.8, seed=args.seed)
    train_target, val_target = split_pflow_targets_ragged(target, train_fraction=0.8, seed=args.seed)

    train_dataset = DelphesDataSet(train_truth_tensor, train_target, device=device)
    val_dataset = DelphesDataSet(val_truth_tensor, val_target, device=device)

    train_dataloader = DelphesDataLoader(train_dataset, batch_size=512, shuffle=True)
    # drop_last on val so a tiny tail batch (e.g. 20000 % 512 = 32 events) is not
    # weighted equally with full batches in the per-batch val-loss mean.
    val_dataloader = DelphesDataLoader(
        val_dataset, batch_size=512, shuffle=False, drop_last=True
    )

    # All ranks must use the *same* initial parameters for the manual-
    # gradient-sync scheme to keep them in sync; ``torch.manual_seed``
    # with the user seed (not seed+rank!) handles that.
    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)

    # The param config drives everything: ``value`` initializes every learnable
    # parameter, ``trainable`` selects the optimized subset (per element, with a
    # gradient mask for partially-trainable tensors), and ``lr_scale`` sets each
    # parameter's effective Adam lr = ``args.lr * lr_scale`` (used to build the
    # optimizer's parameter groups).
    param_cfg = pc.load_param_config(args.param_config)
    pc.apply_param_config(trainee, param_cfg)
    params_to_train, param_groups = pc.select_trainable(trainee, param_cfg, global_lr=args.lr)
    if not params_to_train:
        raise SystemExit(
            f"--param-config {args.param_config} marks no parameter as trainable; "
            "nothing to optimize."
        )

    n_trainable_scalars = sum(1 for spec in param_cfg.values() if spec["trainable"])
    log(
        f"Training {n_trainable_scalars} scalar(s) across {len(params_to_train)} "
        f"tensor(s) from {args.param_config} for {args.n_steps} Adam steps..."
    )

    history = fit_card_to_fullsim(
        trainee,
        train_dataloader,
        val_dataloader,
        param_groups=param_groups,
        n_steps=args.n_steps,
        log_every=max(1, args.n_steps // 10),
        snapshot_parameters=args.history_path is not None,
        rank=rank,
        device=device,
        intermediate_plot_dir=args.intermediate_plot_dir,
        plot_every=args.plot_every,
    )

    if args.history_path is not None and _is_main(rank):
        args.history_path.parent.mkdir(parents=True, exist_ok=True)

        # Pull the index-aligned per-epoch lists returned by the fit loop.
        steps = history["step"]
        losses = history["loss"]
        val_losses = history.get("val_loss", [])
        params_list = history.get("parameters", [])

        # Metadata: the run-level scalars. --lr is the global magnitude; each
        # parameter's effective Adam lr is --lr * its config lr_scale, recorded
        # as the distinct optimizer-group lrs that were actually used.
        metadata = {
            "n_events": args.n_events,
            "n_steps": args.n_steps,
            "lr": args.lr,
            "param_config": str(args.param_config),
            "param_group_lrs": sorted({g["lr"] for g in param_groups}),
            "trainable_params": sorted(k for k, spec in param_cfg.items() if spec["trainable"]),
            "world_size": world_size,
        }

        # Per-epoch history keyed "epoch_{step}". Each step here is a full
        # pass over the train dataloader, so "epoch" is accurate.
        history_dict = {
            f"epoch_{steps[i]}": {
                "step": steps[i],
                "train_loss": losses[i],
                "val_loss": val_losses[i] if i < len(val_losses) else None,
                "parameters": params_list[i] if i < len(params_list) else {},
            }
            for i in range(len(steps))
        }

        # Best epoch = minimum validation loss; fall back to the last epoch
        # when no val loss was recorded.
        if val_losses:
            best_i = min(range(len(val_losses)), key=lambda i: val_losses[i])
        elif steps:
            best_i = len(steps) - 1
        else:
            best_i = None

        if best_i is None:
            best_result: dict = {}
        else:
            best_result = {
                "epoch": f"epoch_{steps[best_i]}",
                "step": steps[best_i],
                "train_loss": losses[best_i],
                "val_loss": val_losses[best_i] if best_i < len(val_losses) else None,
                "parameters": params_list[best_i] if best_i < len(params_list) else {},
            }

        with args.history_path.open("w") as f:
            json.dump(
                {
                    "metadata": metadata,
                    "history": history_dict,
                    "best_result": best_result,
                },
                f,
                indent=2,
            )
        log(f"Wrote training history to {args.history_path}")

    # Print the learned charged-hadron / ECal / HCal scales for a quick sanity
    # check against the generate_pseudodata.py TARGET_*_SCALE values.
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
    hcal_scale_vals = (
        (
            1.0
            + 0.3
            * torch.tanh(
                trainee.HCal.scale_module.scale_raw  # type: ignore[union-attr]
            )
        )
        .detach()
        .tolist()
    )
    log("")
    log(f"Final charged-hadron scale (3 eta regions): {chad_scales}")
    log(f"Final ECal scale            (3 eta regions): {ecal_scale_vals}")
    log(f"Final HCal scale            (2 eta regions): {hcal_scale_vals}")

    _cleanup_distributed()
