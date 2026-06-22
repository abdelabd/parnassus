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
- ``truth_pdgid`` : full PDG ID of the truth particle (130 for K_L0, 2112
    for neutron, 321 for charged kaon, 2212 for proton, ...). REQUIRED -- it is
    used directly as the trainee card's ``ColumnMap.PID`` so the ECal/HCal
    energy-fraction LUT routes long-lived neutrals to HCal instead of
    mis-mapping them to pi0 (which would push them through the photon stream).
    Loading raises if it is absent.
- ``pflow_pt`` / ``pflow_eta`` / ``pflow_phi`` / ``pflow_class`` : the
  corresponding CMS particle-flow reconstructed objects that serve as
  the fitting target

Class-to-PDG mapping (used on the ``pflow_class`` target side) is done with
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .config import _DEFAULT_LR
from .data import (
    load_cms_flow_root,
    load_truth_events_ragged,
    load_pflow_targets_ragged,
    split_truth_objects_jagged,
    split_pflow_targets_jagged,
)

from .dataloader import DelphesDataSet, DelphesDataLoader

from .loss import CALO_COUNT_WEIGHT, COUNT_WEIGHT, EVENT_WEIGHT, LOSS_CHOICES
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
        "--loss",
        type=str,
        default="wasserstein",
        choices=list(LOSS_CHOICES),
        help=(
            "Training loss. 'wasserstein' (default) is the per-pid sliced "
            "Wasserstein-2 over [log_E, log_pt, eta] plus a down-weighted "
            "log(HT) term and expected-count terms. 'soft_hist' is the "
            "soft-histogram MSE loss summed across observables; DDP-aware "
            "via differentiable all-reduce on per-rank histograms."
        ),
    )
    parser.add_argument(
        "--count-weight",
        type=float,
        default=COUNT_WEIGHT,
        help=(
            "Weight on every per-species expected-count term, relative to the "
            "unit-weighted per-pid object Wasserstein terms. The count term is a "
            "dimensionless, batch-invariant normalized chi^2 (~O(1)), so this is a "
            f"meaningful balance knob. Default {COUNT_WEIGHT}. Set 0 to disable the "
            "tracking-efficiency count terms (drops the eff_logits count gradient)."
        ),
    )
    parser.add_argument(
        "--calo-count-weight",
        type=float,
        default=CALO_COUNT_WEIGHT,
        help=(
            "Weight on the CALO-resolution expected-count terms (ecal_photon, "
            "hcal_neutral_hadron), kept SEPARATE from --count-weight. These must "
            "out-vote a wrong-signed Wasserstein gradient on the forward resolution "
            "coefficients (forward_c_E/forward_c_S/common_c_E), so they need a larger "
            f"weight and a per-region-fair normalization. Default {CALO_COUNT_WEIGHT}. "
            "Set 0 to disable the calo-resolution count gradient."
        ),
    )
    parser.add_argument(
        "--event-weight",
        type=float,
        default=EVENT_WEIGHT,
        help=(
            "Weight on the per-event log(HT) Wasserstein term, relative to the "
            f"per-pid object terms. Default {EVENT_WEIGHT}."
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
        default="doc/fit_results/intermediate_plots",
        help=(
            "Directory for per-epoch intermediate observable plots: one "
            "multi-page PDF per epoch (intermediate_epoch_<step>.pdf, one "
            "observable per page) comparing the trainee prediction to the "
            "full-sim target, with each observable's soft-hist MSE in the "
            "page title as a distribution-mismatch diagnostic. Pass an empty "
            "string to disable (default: doc/fit_results/intermediate_plots). "
            "Only the main rank plots."
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
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help=(
            "Stop after this many epochs with no val_loss improvement. Pass a "
            "value <= 0 to disable early stopping (always run the full "
            "--n-steps). Default 10."
        ),
    )
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=4,
        help=(
            "Patience for the ReduceLROnPlateau lr decay (epochs of no val_loss "
            "improvement before the lr is halved). Pass a value <= 0 to disable "
            "lr decay entirely and train at a constant lr -- recommended for "
            "single-parameter closure fits, where the stochastic loss otherwise "
            "collapses the lr before convergence. Default 4."
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

    train_truth_tensor, val_truth_tensor, _ = split_truth_objects_jagged(truth_ragged, train_fraction=0.7, val_fraction=0.2)
    train_target, val_target, _ = split_pflow_targets_jagged(target, train_fraction=0.7, val_fraction=0.2)

    train_dataset = DelphesDataSet(train_truth_tensor, train_target, device=device)
    val_dataset = DelphesDataSet(val_truth_tensor, val_target, device=device)

    # If DDP then each rank sees disjoint shard -- keep the jagged split
    # output as-is and only shard at the DataLoader level.
    if world_size > 1:
        train_sampler: DistributedSampler | None = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        val_sampler: DistributedSampler | None = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
    else:
        train_sampler = None
        val_sampler = None

    # Per-rank batch size: divide by world_size so that after all_gather the
    # combined batch has the same size as the single-process case.
    per_rank_batch_size = max(1, 512 // world_size)

    train_dataloader = DelphesDataLoader(
        train_dataset, batch_size=per_rank_batch_size, shuffle=True, sampler=train_sampler
    )
    val_dataloader = DelphesDataLoader(
        val_dataset, batch_size=per_rank_batch_size, shuffle=False, sampler=val_sampler
    )

    # Same initial parameters for DDP
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

    # Wrap in DDP
    if world_size > 1:
        ddp_kwargs: dict = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        # Keep the fast path: we enforce stable graph usage in training so we
        # can run with find_unused_parameters disabled.
        ddp_kwargs["find_unused_parameters"] = False
        trainee = DDP(trainee, **ddp_kwargs)

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
        # <= 0 on the CLI means "disable" (None) for both knobs.
        early_stopping_patience=(
            args.early_stopping_patience if args.early_stopping_patience > 0 else None
        ),
        lr_scheduler_patience=(
            args.lr_scheduler_patience if args.lr_scheduler_patience > 0 else None
        ),
        count_weight=args.count_weight,
        calo_count_weight=args.calo_count_weight,
        event_weight=args.event_weight,
        loss_name=args.loss,
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
            # Optimizer-schedule knobs (recorded so runs are reproducible and the
            # notebook cache key can detect changes). 0 = disabled.
            "early_stopping_patience": max(0, args.early_stopping_patience),
            "lr_scheduler_patience": max(0, args.lr_scheduler_patience),
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
    # ``trainee`` may be DDP-wrapped under srun; unwrap with ``.module`` so
    # the attribute lookups below see the real card.
    trainee_card = trainee.module if isinstance(trainee, DDP) else trainee
    chad_res = trainee_card.ChargedHadronMomentumSmearing.resolution_module  # type: ignore[union-attr]
    chad_scales = (1.0 + 0.3 * torch.tanh(chad_res.scale_raw)).detach().tolist()
    ecal_scale_vals = (
        (
            1.0
            + 0.3
            * torch.tanh(
                trainee_card.ECal.scale_module.scale_raw  # type: ignore[union-attr]
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
                trainee_card.HCal.scale_module.scale_raw  # type: ignore[union-attr]
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
