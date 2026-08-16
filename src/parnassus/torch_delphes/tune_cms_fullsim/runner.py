"""Shared run helpers for ``tune_cms_fullsim`` (data loading + history I/O).

This module factors the two pieces that the single-fit CLI (:mod:`.cli`) and the
Optuna hyperparameter search (:mod:`.optuna_search`) both need, so they stay a
single source of truth:

- :func:`resolve_acceptance_cuts` — turn the ``--mode`` / ``--*-cut`` /
  ``--no-chad-truncation`` CLI args into the resolved cut values (``None`` /
  ``False`` = disabled) both entrypoints feed to the loaders, the card and the fit.
- :func:`load_split_datasets` — read a CMS full-simulation ROOT file and build
  the train/val :class:`~.dataloader.DelphesDataSet` pair (ragged, memory-light),
  exactly as the tuning CLI does. The caller wraps these in dataloaders (and, for
  DDP, adds the per-rank samplers).
- :func:`write_history_json` — serialize a fit's per-epoch history to the
  ``{metadata, history, best_result}`` JSON schema the plotting pipeline
  (:mod:`.plot_fit_results`) consumes, selecting the best epoch by **minimum
  validation loss**.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import NamedTuple

import torch

from .data import (
    load_cms_flow_root,
    load_pflow_targets_ragged,
    load_truth_events_ragged,
    split_pflow_targets_jagged,
    split_truth_objects_jagged,
)
from .dataloader import DelphesDataSet


class AcceptanceCuts(NamedTuple):
    """Resolved ``--mode`` / ``--*-cut`` CLI args (``None`` / ``False`` = disabled)."""

    truth_pt_cut: float | None
    reco_pt_cut: float | None
    abs_eta_cut: float | None
    truncate_chads: bool


def resolve_acceptance_cuts(args) -> AcceptanceCuts:
    """Resolve ``--mode`` + the acceptance-cut args (shared by cli and optuna_search).

    ``fullsim``: ``<= 0`` disables the respective part and ``--no-chad-truncation``
    turns the truncation off. ``delphes``: everything off -- diff-Delphes has to
    reproduce Delphes without any CMS selection, so the cut flags are ignored.

    Returns
    -------
    AcceptanceCuts
        ``(truth_pt_cut, reco_pt_cut, abs_eta_cut, truncate_chads)``; ``None`` /
        ``False`` = disabled, ready to unpack into the loader / card / fit kwargs.
    """
    if args.mode == "delphes":
        return AcceptanceCuts(None, None, None, truncate_chads=False)
    return AcceptanceCuts(
        args.truth_pt_cut if args.truth_pt_cut > 0 else None,
        args.reco_pt_cut if args.reco_pt_cut > 0 else None,
        args.eta_cut if args.eta_cut > 0 else None,
        truncate_chads=not args.no_chad_truncation,
    )


def load_split_datasets(
    root_file: Path,
    n_events: int,
    device: torch.device,
    train_fraction: float = 0.7,
    val_fraction: float = 0.2,
    truth_pt_cut: float | None = None,
    reco_pt_cut: float | None = None,
    abs_eta_cut: float | None = None,
    truncate_chads: bool = False,
) -> tuple[DelphesDataSet, DelphesDataSet]:
    """Load a CMS full-sim ROOT file and build the train/val dataset pair.

    Mirrors the data path in :mod:`.cli`: the truth particles and per-particle
    targets are kept *ragged* (per-event lists, no global padding) and each batch
    is padded to its own max multiplicity by the collate fn -- so a 100k-event
    sample stays well under the OOM threshold. The big transient ``arrays`` dict
    is freed before the split to keep peak RSS low.

    Parameters
    ----------
    root_file : Path
        CMS full-simulation ROOT file (cms-flow schema).
    n_events : int
        Number of events to read (``-1`` for all).
    device : torch.device
        Device the :class:`DelphesDataSet` tensors live on.
    train_fraction, val_fraction : float
        Fractions of the sample used for the train and validation splits (the
        remainder is held out). Defaults match the CLI (0.7 / 0.2).
    truth_pt_cut, reco_pt_cut, abs_eta_cut : float | None
        Acceptance cuts: truth input (``pt >= truth_pt_cut``, ``|eta| <=
        abs_eta_cut``, all species) and reco target (``pt >= reco_pt_cut``,
        ``|eta| <= abs_eta_cut``, all classes). ``None`` disables (legacy
        behavior). See :func:`.data._build_truth_rows` /
        :func:`.data._build_pflow_event_data`.
    truncate_chads : bool
        Truncate the target's reco charged hadrons at the per-event
        ``n_truth_chad`` ceiling (see :func:`.data._build_pflow_event_data`).

    Returns
    -------
    (train_dataset, val_dataset)
        The two :class:`DelphesDataSet` objects, ready to wrap in a
        :class:`~.dataloader.DelphesDataLoader`.
    """
    arrays = load_cms_flow_root(root_file, n_events=n_events)
    truth_ragged = load_truth_events_ragged(
        arrays, truth_pt_cut=truth_pt_cut, abs_eta_cut=abs_eta_cut
    )
    target = load_pflow_targets_ragged(
        arrays,
        reco_pt_cut=reco_pt_cut,
        abs_eta_cut=abs_eta_cut,
        truncate_chads=truncate_chads,
    )

    # The uproot arrays dict is the largest remaining transient; free it before
    # the split so peak RSS stays low.
    del arrays
    gc.collect()

    train_truth_tensor, val_truth_tensor, _ = split_truth_objects_jagged(
        truth_ragged, train_fraction=train_fraction, val_fraction=val_fraction
    )
    train_target, val_target, _ = split_pflow_targets_jagged(
        target, train_fraction=train_fraction, val_fraction=val_fraction
    )

    train_dataset = DelphesDataSet(train_truth_tensor, train_target, device=device)
    val_dataset = DelphesDataSet(val_truth_tensor, val_target, device=device)
    return train_dataset, val_dataset


def build_history_payload(history: dict[str, list], metadata: dict) -> dict:
    """Assemble the ``{metadata, history, best_result}`` payload from a fit.

    The per-epoch ``history`` lists (``step`` / ``loss`` / ``val_loss`` /
    optional ``parameters``) are the index-aligned lists returned by
    :func:`.training.fit_card_to_fullsim`. The best epoch is the one with the
    minimum recorded validation loss; when no val loss was recorded it falls back
    to the last epoch (matching the historic CLI behavior).

    Returns
    -------
    dict
        ``{"metadata": ..., "history": {epoch_i: {...}}, "best_result": {...}}``.
    """
    steps = history["step"]
    losses = history["loss"]
    val_losses = history.get("val_loss", [])
    params_list = history.get("parameters", [])

    # Per-epoch history keyed "epoch_{step}". Each step here is a full pass over
    # the train dataloader, so "epoch" is accurate.
    history_dict = {
        f"epoch_{steps[i]}": {
            "step": steps[i],
            "train_loss": losses[i],
            "val_loss": val_losses[i] if i < len(val_losses) else None,
            "parameters": params_list[i] if i < len(params_list) else {},
        }
        for i in range(len(steps))
    }

    # Best epoch = minimum validation loss; fall back to the last epoch when no
    # val loss was recorded.
    if val_losses:
        best_i: int | None = min(range(len(val_losses)), key=lambda i: val_losses[i])
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

    return {"metadata": metadata, "history": history_dict, "best_result": best_result}


def write_history_json(
    history_path: Path, history: dict[str, list], metadata: dict
) -> dict:
    """Write the fit history to ``history_path`` in the nested schema.

    Creates the parent directory, serializes
    :func:`build_history_payload`'s output with ``indent=2``, and returns the
    payload (so callers can reuse the computed ``best_result`` without
    re-reading the file).
    """
    payload = build_history_payload(history, metadata)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w") as f:
        json.dump(payload, f, indent=2)
    return payload
