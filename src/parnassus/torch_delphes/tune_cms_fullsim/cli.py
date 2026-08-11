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
import socket
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .comet_utils import end_comet_experiment, init_comet_experiment
from .config import (
    _DEFAULT_LR,
    DEFAULT_ABS_ETA_CUT,
    DEFAULT_RECO_PT_CUT,
    DEFAULT_TRUTH_PT_CUT,
)

from .dataloader import DelphesDataLoader

from .runner import load_split_datasets, write_history_json

from .loss import (
    CALO_COUNT_WEIGHT,
    COUNT_RATE_FLOOR,
    COUNT_WEIGHT,
    EVENT_WEIGHT,
    LOSS_CHOICES,
    PID_WEIGHTING_CHOICES,
)
from .distributed import (
    _cleanup_distributed,
    _init_distributed,
    _is_dist,
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
            "log(HT) term and expected-count terms. 'soft_hist' is the same "
            "structure with a per-pid, per-observable soft-histogram MSE over "
            "[log_E, log_pt, eta] in place of the optimal-transport term, plus "
            "the same log(HT) and expected-count terms (it directly optimizes "
            "histogram shape on a fixed bin grid). 'wasserstein_1d' keeps the same "
            "per-pid/per-observable scaffolding but matches each axis with the exact "
            "BIN-FREE 1D Wasserstein distance via quantiles (no histogram, no bin grid, "
            "no range, no softness; deterministic, with no random projections -- so it "
            "avoids both the manual binning of soft_hist and the instability of the "
            "point-cloud sliced Wasserstein); same log(HT) and count terms. NOTE: its "
            "standardized shape-term scale differs from soft_hist's MSE, so re-check the "
            "count/shape balance with MCGEN_LOSS_DEBUG=1 before a production fit. All "
            "three honor --count-weight/--calo-count-weight/--event-weight."
        ),
    )
    parser.add_argument(
        "--count-weight",
        type=float,
        default=COUNT_WEIGHT,
        help=(
            "Weight on the tracking-efficiency per-species expected-count terms, "
            "relative to the unit-weighted per-pid object Wasserstein terms. The count "
            "term is a normalized relative chi^2 on per-event rates with a fixed rate "
            "floor (--count-rate-floor), making it dimensionless and batch-size invariant "
            f"(~O(1)), so this is a meaningful balance knob. Default {COUNT_WEIGHT}. Set 0 "
            "to disable the tracking-efficiency count terms (drops the eff_logits count "
            "gradient)."
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
        "--count-rate-floor",
        type=float,
        default=COUNT_RATE_FLOOR,
        help=(
            "Per-event-RATE floor in the count-term Pearson denominators (shared by the "
            "tracking and calo count terms). The count terms are evaluated on per-event "
            "rates (counts / batch event count); this fixed floor is what makes them "
            "batch-size INVARIANT. The old constant '+1' count floor had an effective "
            "rate floor 1/N that shrank with batch size N, so sparse data regions (lepton "
            "bins, forward |eta| HCal neutral hadrons) where the trainee still predicted a "
            "count grew with N. A region with rate << this floor is regularized; a region "
            f"with rate >> it is unchanged. Default {COUNT_RATE_FLOOR}. Re-validate the "
            "count/shape balance with MCGEN_LOSS_DEBUG=1 if you change it."
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
        "--pid-weighting",
        type=str,
        default="equal",
        choices=list(PID_WEIGHTING_CHOICES),
        help=(
            "Per-pid population weighting of the per-species SHAPE terms (count and "
            "log(HT) terms are untouched). 'equal' (default) weights every particle type "
            "the same -- so rare species (muon ~0.2%%, electron ~0.5%%) cost the optimizer "
            "as much as the abundant charged/neutral hadrons and photons. 'fraction' "
            "down-weights each pid by its population fraction (aggressive: rare species "
            "~100-250x lighter, which effectively FREEZES their momentum-smearing params). "
            "'sqrt_fraction' down-weights by sqrt(fraction) (gentle: rare species ~8-20x "
            "lighter but still learnable -- the recommended mode when training "
            "muon/electron smearing). Weights are mean-1 normalized, so only the "
            "cross-species balance changes, not the overall shape-vs-count balance."
        ),
    )
    parser.add_argument(
        "--pid-weight-floor",
        type=float,
        default=0.0,
        help=(
            "Lower clamp on the per-pid shape weight (default 0.0 = off), re-normalized to "
            "keep the mean-1 invariant. A small floor (e.g. 0.1) protects a rare species' "
            "gradient in a low-statistics batch. Only meaningful with --pid-weighting "
            "fraction/sqrt_fraction."
        ),
    )
    parser.add_argument(
        "--truth-pt-cut",
        type=float,
        default=DEFAULT_TRUTH_PT_CUT,
        help=(
            "Truth-input acceptance: keep truth particles (all species) with "
            "pt >= this and |eta| <= --eta-cut before feeding the trainee. "
            f"Default {DEFAULT_TRUTH_PT_CUT} (a no-op on the externally "
            "preprocessed _selected files, made explicit here). <= 0 disables "
            "the pt part."
        ),
    )
    parser.add_argument(
        "--reco-pt-cut",
        type=float,
        default=DEFAULT_RECO_PT_CUT,
        help=(
            "Reco acceptance: keep reco objects (ALL classes) with pt >= this "
            "and |eta| <= --eta-cut, applied to BOTH the pflow target (at load "
            "time; no-op on pre-cut files) and the trainee output (at loss "
            "time; a real cut: sub-GeV photons, forward NH). Also gates the "
            "differentiable count terms and sets the floor for the "
            f"n_truth_chad truncation ceiling. Default {DEFAULT_RECO_PT_CUT}. "
            "<= 0 disables the pt part. NOTE: losses are not comparable "
            "across different cut settings."
        ),
    )
    parser.add_argument(
        "--eta-cut",
        type=float,
        default=DEFAULT_ABS_ETA_CUT,
        help=(
            "|eta| acceptance bound shared by the truth and reco cuts (and the "
            f"calo count regions). Default {DEFAULT_ABS_ETA_CUT}, matching the "
            "preprocessing of the _selected files. <= 0 disables."
        ),
    )
    parser.add_argument(
        "--no-chad-truncation",
        action="store_true",
        help=(
            "Disable the per-event truth-ceiling charged-hadron truncation "
            "(top-n_truth_chad by pt on both target and trainee; on by "
            "default). The truncation removes the reco chads the full-sim data "
            "contains but a truth-fed sim can never produce (~22/event on the "
            "pt-hat 2500-3000 sample: K_S/Lambda decay daughters, baryons "
            "missing from the truth record, GEANT4 material secondaries)."
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
            "multi-page PDF per epoch (intermediate_epoch_<step>.pdf) comparing "
            "the trainee prediction to the full-sim target -- combined (all-PID) "
            "observables one per page, then one per-PID page per particle type "
            "(charged hadron/electron/muon/neutral hadron/photon) gridding "
            "log_pt/log_E/eta/pt -- with each panel's soft-hist MSE in the title "
            "as a distribution-mismatch diagnostic. Pass an empty string to "
            "disable (default: doc/fit_results/intermediate_plots). Only the main "
            "rank plots."
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
    parser.add_argument(
        "--comet-name",
        type=str,
        default=None,
        help=(
            "Comet experiment name. Default: 'adam_<param-config stem>'. Logging "
            "additionally requires COMET_API_KEY in the environment (workspace "
            "from COMET_WORKSPACE); without it the run proceeds unlogged."
        ),
    )
    parser.add_argument(
        "--comet-disabled",
        action="store_true",
        help="Disable Comet logging even when COMET_API_KEY is set.",
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

    # Resolve the acceptance-cut args (<= 0 disables the respective part).
    truth_pt_cut = args.truth_pt_cut if args.truth_pt_cut > 0 else None
    reco_pt_cut = args.reco_pt_cut if args.reco_pt_cut > 0 else None
    abs_eta_cut = args.eta_cut if args.eta_cut > 0 else None
    truncate_chads = not args.no_chad_truncation
    log(
        f"[filter] truth cut: pt >= {truth_pt_cut}, |eta| <= {abs_eta_cut} | "
        f"reco cut (target+trainee, all classes): pt >= {reco_pt_cut}, "
        f"|eta| <= {abs_eta_cut} | chad truncation at n_truth_chad: "
        f"{'ON' if truncate_chads else 'OFF'}"
    )

    # Ragged (no global padding): truth particles are kept as a per-event list and
    # each batch is padded to its own max in delphes_collate_fn. Padding every event
    # to the GLOBAL max multiplicity here would allocate ~50 GB at 100k events. The
    # target carries the per-reco-bin per-species counts (chad/electron/muon
    # _region_counts) the differentiable count terms match against. Shared with the
    # Optuna search via tune_cms_fullsim.runner.
    train_dataset, val_dataset = load_split_datasets(
        root_file,
        n_events=args.n_events,
        device=device,
        truth_pt_cut=truth_pt_cut,
        reco_pt_cut=reco_pt_cut,
        abs_eta_cut=abs_eta_cut,
        truncate_chads=truncate_chads,
    )
    if truncate_chads:
        n_t = train_dataset.n_truth_chad
        log(
            f"[filter] mean n_truth_chad (train split) = {float(n_t.mean()):.2f} "
            f"-- the per-event ceiling the reco chads are truncated to"
        )

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

    # Batch size: for wasserstein / wasserstein_1d, divide by world_size so
    # that after all_gather the combined batch matches the single-process
    # case. For soft_hist, use the full batch size — the loss does not need
    # all_gather and benefits from larger per-rank batches.
    if args.loss in ("wasserstein", "wasserstein_1d"):
        batch_size = max(1, 4096 // world_size)
    else:
        batch_size = 4096

    train_dataloader = DelphesDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, sampler=train_sampler
    )
    val_dataloader = DelphesDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, sampler=val_sampler
    )

    # Same initial parameters for DDP
    torch.manual_seed(args.seed)
    trainee = CMSEnergyFlowDefault(
        debug=False,
        learnable=True,
        # Harmonize the differentiable count terms with the reco acceptance cut
        # (tracking expected counts + calo soft counts; object creation untouched).
        count_pt_min=reco_pt_cut,
        count_abs_eta_max=abs_eta_cut,
    ).to(device)

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
        # For Wasserstein+DDP, use separate RNGs to efficiently sample unit-vectors
        if args.loss in ("wasserstein", "wasserstein_1d") and _is_dist():
            torch.manual_seed(args.seed + rank)
            import numpy as np
            np.random.seed(args.seed + rank)

    n_trainable_scalars = sum(1 for spec in param_cfg.values() if spec["trainable"])
    log(
        f"Training {n_trainable_scalars} scalar(s) across {len(params_to_train)} "
        f"tensor(s) from {args.param_config} for {args.n_steps} Adam steps..."
    )

    # Comet (rank 0 only; None when comet_ml is missing, unkeyed, or disabled).
    # The name defaults to the param config being fitted, which is the single
    # most useful discriminator between otherwise-identical Adam runs.
    comet_exp = init_comet_experiment(
        name=args.comet_name or f"adam_{Path(args.param_config).stem}",
        params={
            **vars(args),
            "world_size": world_size,
            "batch_size": batch_size,
            "n_trainable_scalars": n_trainable_scalars,
            "param_group_lrs": sorted({g["lr"] for g in param_groups}),
        },
        disabled=args.comet_disabled,
        rank=rank,
        log_prefix="[adam]",
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
        count_rate_floor=args.count_rate_floor,
        event_weight=args.event_weight,
        loss_name=args.loss,
        pid_weighting=args.pid_weighting,
        pid_weight_floor=args.pid_weight_floor,
        reco_pt_cut=reco_pt_cut,
        reco_abs_eta_cut=abs_eta_cut,
        truncate_chads=truncate_chads,
        comet_exp=comet_exp,
    )

    # Summary metrics on the experiment, then close it. Done before the history
    # write so a Comet hiccup cannot cost us the fit results on disk.
    if comet_exp is not None:
        _val = [v for v in history.get("val_loss", []) if v is not None]
        if _val:
            comet_exp.log_metric("best_val_loss", min(_val))
        comet_exp.log_other("epochs_run", len(history["step"]))
    end_comet_experiment(comet_exp)

    if args.history_path is not None and _is_main(rank):
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
            # Acceptance cuts + truncation (resolved values; None = disabled).
            # Losses are NOT comparable across different settings of these.
            "truth_pt_cut": truth_pt_cut,
            "reco_pt_cut": reco_pt_cut,
            "eta_cut": abs_eta_cut,
            "chad_truncation": truncate_chads,
        }
        # The {metadata, history, best_result} schema (best = min val loss) is the
        # single source of truth shared with the Optuna search and consumed by
        # plot_fit_results; see tune_cms_fullsim.runner.
        write_history_json(args.history_path, history, metadata)
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
