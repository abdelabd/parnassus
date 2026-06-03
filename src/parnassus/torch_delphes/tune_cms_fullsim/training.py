"""Optimizer setup and the Adam fit loop for ``tune_cms_fullsim``.

This module holds the training machinery proper:

- :func:`_classify_parameter` / :func:`build_parameter_groups` bucket the 66
  learnable parameters into four Adam groups (scales / efficiency / fractions
  / resolution) with independent learning rates (the matchers and default LRs
  live in :mod:`.config`).
- :func:`fit_card_to_fullsim` is the Adam optimization loop that fits the
  trainee card to a fixed target observable dict.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .config import (
    DEFAULT_BIN_EDGES,
    DEFAULT_OBS_WEIGHTS,
    _DEFAULT_LR_EFFICIENCY,
    _DEFAULT_LR_FRACTIONS,
    _DEFAULT_LR_RESOLUTION,
    _DEFAULT_LR_SCALES,
    _EFFICIENCY_SUFFIXES,
    _FRACTION_FRAGMENTS,
    _SCALE_SUFFIXES,
)
from .data import restore_event_format, load_pflow_targets_from_tensor
from .distributed import _is_dist, _is_main
from .loss import multi_observable_loss_distributed

# =============================================================================
# Parameter groups for per-group learning rates
# =============================================================================
#
# The 66 learnable parameters live in parameter spaces with very different
# natural scales, so a single global Adam learning rate is a poor fit:
#
# - Resolution coefficients ``a_raw, b_raw, c_E, c_S, c_N`` are softplus-
#   wrapped positive scalars whose post-softplus values range from ~1e-3 to
#   ~3. An Adam step of 5e-2 on the raw parameter can blow up a tiny
#   coefficient by orders of magnitude in one step.
# - Region scales ``scale_raw`` sit at 0 in the raw space and are bounded
#   to [-1, +1] via ``tanh``; an Adam step of 5e-2 moves the bounded
#   output by at most ~1.5%, which is safe and responsive.
# - Efficiency logits ``eff_logits`` map to probabilities via sigmoid and
#   sit at finite values like ``logit(0.95)=2.94``. Stepping the logit by
#   5e-2 moves the probability by less than 0.01, which is the right
#   magnitude for PF efficiency fits.
# - Hadron-fraction logits behave like efficiency logits.
#
# The function below maps each named parameter in a learnable card to one
# of the four groups so that :func:`fit_card_to_fullsim` can build a
# ``torch.optim.Adam`` with per-group learning rates.
# (The suffix/fragment matchers and the default LRs live in ``config.py``.)


def _classify_parameter(name: str) -> str:
    """Return the parameter-group key for a given fully-qualified name.

    Returns
    -------
    str
        One of ``"scales"``, ``"efficiency"``, ``"fractions"``,
        ``"resolution"``.
    """
    if any(name.endswith(s) for s in _SCALE_SUFFIXES):
        return "scales"
    if any(frag in name for frag in _FRACTION_FRAGMENTS):
        return "fractions"
    if any(name.endswith(s) for s in _EFFICIENCY_SUFFIXES):
        return "efficiency"
    return "resolution"


def build_parameter_groups(
    card: CMSEnergyFlowDefault,
    lr_scales: float = _DEFAULT_LR_SCALES,
    lr_resolution: float = _DEFAULT_LR_RESOLUTION,
    lr_efficiency: float = _DEFAULT_LR_EFFICIENCY,
    lr_fractions: float = _DEFAULT_LR_FRACTIONS,
) -> list[dict]:
    """Split ``card.parameters()`` into Adam parameter groups by type.

    Returns
    -------
    list[dict]
        A list of four ``{"params": [...], "lr": float, "name": str}``
        dicts, suitable to pass directly to ``torch.optim.Adam``.
        Empty groups are filtered out so the optimizer doesn't complain
        about an empty ``params`` list.
    """
    buckets: dict[str, list[nn.Parameter]] = {
        "scales": [],
        "resolution": [],
        "efficiency": [],
        "fractions": [],
    }
    for name, p in card.named_parameters():
        buckets[_classify_parameter(name)].append(p)
    lrs = {
        "scales": lr_scales,
        "resolution": lr_resolution,
        "efficiency": lr_efficiency,
        "fractions": lr_fractions,
    }
    groups: list[dict] = []
    for key, params in buckets.items():
        if not params:
            continue
        groups.append({"params": params, "lr": lrs[key], "name": key})
    return groups


# =============================================================================
# Fit loop
# =============================================================================


def fit_card_to_fullsim(
    card: CMSEnergyFlowDefault | DDP,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    n_steps: int = 100,
    lr: float | None = None,
    beta: float = 0.15,
    log_every: int = 10,
    parameters_to_train: list[nn.Parameter] | None = None,
    bin_edges: dict[str, torch.Tensor] | None = None,
    observable_weights: dict[str, float] | None = None,
    lr_scales: float = _DEFAULT_LR_SCALES,
    lr_resolution: float = _DEFAULT_LR_RESOLUTION,
    lr_efficiency: float = _DEFAULT_LR_EFFICIENCY,
    lr_fractions: float = _DEFAULT_LR_FRACTIONS,
    snapshot_parameters: bool = False,
    rank: int = 0,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """Run Adam on ``card`` to match ``target_observables``.

    The trainee is run ``n_passes_per_step`` times per step, with losses
    averaged, to beat down the stochastic-smearing / Gumbel-ST noise in
    the trainee observables. The target observables are read from the
    ROOT file once and re-used on every step.

    When ``parameters_to_train`` is ``None`` this function automatically
    builds four parameter groups (scales, resolution, efficiency,
    fractions) with independent learning rates, because the four
    groups have very different natural step sizes. Pass a non-None
    ``parameters_to_train`` list to get a single group at the global
    ``lr`` value (backwards-compatible with earlier callers that
    fit a small focused subset).

    Parameters
    ----------
    lr : float | None
        Only used when ``parameters_to_train`` is provided (single
        group). If ``None`` in that case, falls back to
        ``_DEFAULT_LR_SCALES``. When ``parameters_to_train`` is
        ``None``, the four ``lr_*`` arguments are used instead.
    snapshot_parameters : bool
        If True, the history dict will additionally contain a
        ``"parameters"`` list whose i-th entry is a ``{name: float}``
        dict recording every learnable parameter value after step i.
        Off by default because it is O(n_steps * 66) in memory and
        only needed for plotting parameter-drift trajectories.

    Returns
    -------
    dict[str, list]
        ``"step"`` and ``"loss"`` always present. If
        ``snapshot_parameters`` is True, also contains
        ``"parameters"`` (a list of ``dict[str, float]``) for offline
        plotting of the per-parameter trajectory.
    """
    if parameters_to_train is not None:
        param_groups: list[dict] = [
            {
                "params": list(parameters_to_train),
                "lr": lr if lr is not None else _DEFAULT_LR_SCALES,
                "name": "user",
            }
        ]
    else:
        # Always group on the underlying (unwrapped) card so the
        # parameter-name suffix matching works regardless of DDP wrapping.
        underlying = card.module if isinstance(card, DDP) else card
        param_groups = build_parameter_groups(
            underlying,
            lr_scales=lr_scales,
            lr_resolution=lr_resolution,
            lr_efficiency=lr_efficiency,
            lr_fractions=lr_fractions,
        )

    opt = torch.optim.Adam(param_groups)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5, verbose=True)

    edges = bin_edges if bin_edges is not None else DEFAULT_BIN_EDGES
    weights = observable_weights if observable_weights is not None else DEFAULT_OBS_WEIGHTS
    history: dict[str, list] = {"step": [], "loss": []}
    if snapshot_parameters:
        history["parameters"] = []

    underlying_for_snap = card.module if isinstance(card, DDP) else card

    def _snapshot() -> dict[str, float]:
        """Record the current post-transform value of every parameter.

        We store the interpretable (post-softplus / post-tanh /
        post-sigmoid) value rather than the raw parameter, because
        that's what the plots and the JINST paper tables will show.

        Returns
        -------
        dict[str, float]
            ``{name_or_name[i]: value}`` mapping, one entry per scalar
            component of every ``nn.Parameter`` on the card.
        """
        snap: dict[str, float] = {}
        for name, p in underlying_for_snap.named_parameters():
            if name.endswith(".scale_raw"):
                val = 1.0 + 0.3 * torch.tanh(p)
            elif name.endswith((".eff_logits", "_logit")):
                val = torch.sigmoid(p)
            elif name.endswith((".rate_raw", ".a_raw", ".b_raw")) or name.startswith((
                "ECal.resolution_func",
                "HCal.resolution_func",
            )):
                val = torch.nn.functional.softplus(p)
            else:
                val = p
            vflat = val.detach().flatten().tolist() if val.ndim else [float(val.detach())]
            for i, vv in enumerate(vflat):
                snap[f"{name}[{i}]" if val.ndim else name] = float(vv)
        return snap

    pbar = tqdm(
        range(n_steps),
        disable=not _is_main(rank),
        desc="tune",
        unit="step",
        dynamic_ncols=True,
    )

    min_loss = float("inf")
    patience_counter = 0
    patience = 10  # Number of steps to wait for improvement before early stopping

    for step in pbar:
        loss_acc = torch.zeros(
            (), dtype=torch.float64, device=device
        )
        for batch in train_dataloader:

            opt.zero_grad()
            truth_particles = batch["truth_particles"] # shape is (batch_size, n_particles, n_features)
            # remove the padded particles where all features are zero
            mask = torch.any(truth_particles != 0, dim=-1) # shape is (batch_size, n_particles)
            truth_particles_nonpadded = truth_particles[mask]
            
            # out["EFlowObject"] has shape (all objects, 20 features)
            out = card(truth_particles_nonpadded)

            # first restore the (events, objects, features) shape by grouping with event number
            eflow_objects = out["EFlowObject"]
            eflow_objects_restored = restore_event_format(eflow_objects, mask)
            # Then extract the observables from predicted objects
            pred_observables = load_pflow_targets_from_tensor(eflow_objects_restored)

            # get the target from batch
            target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}
            # pred = trainee_observables(out)
            loss = multi_observable_loss_distributed(
                pred_observables, target_observables, edges, beta=beta, weights=weights
            )

            loss.backward()
            opt.step()

            loss_acc += loss.detach()

        loss_acc /= len(train_dataloader)

        print_loss = float(loss.detach())
        history["step"].append(step)
        history["loss"].append(print_loss)
        if snapshot_parameters:
            history["parameters"].append(_snapshot())
        if _is_main(rank):
            pbar.set_postfix(loss=f"{print_loss:.4e}", refresh=False)
        if _is_main(rank) and log_every > 0 and (step % log_every == 0 or step == n_steps - 1):
            tqdm.write(f"  step {step:3d}/{n_steps}  loss = {print_loss:.4e}")

        # validation loop --- no grad, no step, just logging
        with torch.no_grad():
            card.eval()
            val_loss_acc = torch.zeros(
                (), dtype=torch.float64, device=device
            )
            for batch in val_dataloader:
                truth_particles = batch["truth_particles"] # shape is (batch_size, n_particles, n_features)
                # remove the padded particles where all features are zero
                mask = torch.any(truth_particles != 0, dim=-1) # shape is (batch_size, n_particles)
                truth_particles_nonpadded = truth_particles[mask]
                
                out = card(truth_particles_nonpadded)

                eflow_objects = out["EFlowObject"]
                eflow_objects_restored = restore_event_format(eflow_objects, mask)
                pred_observables = load_pflow_targets_from_tensor(eflow_objects_restored)

                target_observables = {k: batch[k] for k in batch.keys() if k != "truth_particles"}
                val_loss = multi_observable_loss_distributed(
                    pred_observables, target_observables, edges, beta=beta, weights=weights
                )
                val_loss_acc += val_loss.detach()
            val_loss_acc /= len(val_dataloader)
            print_val_loss = float(val_loss_acc)
            tqdm.write(f"  step {step:3d}/{n_steps}  val_loss = {print_val_loss:.4e}")

        # lr scheduler step
        lr_scheduler.step(val_loss_acc)
        # early stopping check
        if val_loss_acc < min_loss:
            min_loss = val_loss_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if _is_main(rank):
                    tqdm.write(f"Early stopping at step {step} with val_loss {print_val_loss:.4e}")
                break

    return history
