"""Shared definitions for intermediate-output (``--debug``) comparisons.

This module mirrors the structure of
:mod:`parnassus.torch_delphes.validation.validate_torch_delphes`: it exposes
the list of per-module post-pipeline outputs we want to inspect alongside the
final ``EFlowObject`` (e.g. ``ParticleAfterProp``, ``ChargedHadronEfficiency``,
``ECalTower`` …), the kinematic variables we extract from each, and the
ROOT-branch naming convention shared by the producer and consumer.

It is shared between
:mod:`parnassus.torch_delphes.generate_pseudodata` (``--debug`` writes these
intermediate outputs into the pseudodata ROOT file) and
:mod:`parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results` (``--debug``
reads them back and overlays trainee-vs-target distributions, one PDF per
module / variable).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch

from parnassus.data.particle_io import ColumnMap

# =============================================================================
# Branch / variable definitions
# =============================================================================

# Kinematic variables extracted per intermediate-module output. The names are
# the C++ Delphes / TorchDelphes ROOT-branch convention used by
# :mod:`validation.validate_torch_delphes`, so the plots are directly
# comparable across the two harnesses.
#
# We omit C++ Delphes' ``Eem`` / ``Ehad`` columns: those are not stored in our
# tensor representation (they are computed at ROOT-write time from PID, see
# the TODO at the top of ``SimpleCalorimeter.py``), so they would not be
# meaningful here.
TRACK_KINEMATIC_VARS: tuple[str, ...] = (
    "PID",
    "Charge",
    "P",
    "PT",
    "Eta",
    "EtaOuter",
    "Phi",
    "T",
    "X",
    "Y",
    "Z",
)
TOWER_KINEMATIC_VARS: tuple[str, ...] = ("E", "ET", "Eta", "Phi", "T")
EFLOW_KINEMATIC_VARS: tuple[str, ...] = (
    "PID",
    "Charge",
    "E",
    "P",
    "PT",
    "Eta",
    "Phi",
    "T",
    "X",
    "Y",
    "Z",
)

# The full per-module list, matching the ``--debug`` branch list in
# :mod:`validation.validate_torch_delphes`. The order follows the CMS card's
# data flow so the plots scroll naturally from input to final particle flow.
#
# We omit the C++ Delphes ``Particle`` branch (raw GenParticle): the
# ``truth_*`` branches already cover the truth side, and ``ParticleBeforeProp``
# below is its TorchDelphes-side counterpart.
INTERMEDIATE_BRANCHES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ParticleBeforeProp", TRACK_KINEMATIC_VARS),
    ("ParticleAfterProp", TRACK_KINEMATIC_VARS),
    ("ChargedHadron", TRACK_KINEMATIC_VARS),
    ("Electron", TRACK_KINEMATIC_VARS),
    ("Muon", TRACK_KINEMATIC_VARS),
    ("NeutralParticle", TRACK_KINEMATIC_VARS),
    ("ChargedHadronEfficiency", TRACK_KINEMATIC_VARS),
    ("ElectronEfficiency", TRACK_KINEMATIC_VARS),
    ("MuonEfficiency", TRACK_KINEMATIC_VARS),
    ("ChargedHadronSmeared", TRACK_KINEMATIC_VARS),
    ("ElectronSmeared", TRACK_KINEMATIC_VARS),
    ("MuonSmeared", TRACK_KINEMATIC_VARS),
    ("Track", TRACK_KINEMATIC_VARS),
    ("ECalTower", TOWER_KINEMATIC_VARS),
    ("ECal_EFlowTrack", TRACK_KINEMATIC_VARS),
    ("EFlowPhoton", TOWER_KINEMATIC_VARS),
    ("HCalTower", TOWER_KINEMATIC_VARS),
    ("EFlowTrack", TRACK_KINEMATIC_VARS),
    ("EFlowNeutralHadron", TOWER_KINEMATIC_VARS),
    ("Tower", TOWER_KINEMATIC_VARS),
    ("EFlowObject", EFLOW_KINEMATIC_VARS),
)

# Tower-like branches: their PT column is computed as E/cosh(eta) and can be
# tiny by construction, so we filter ghost rows on E instead of PT.
_TOWER_BRANCHES: frozenset[str] = frozenset({
    "ECalTower",
    "EFlowPhoton",
    "HCalTower",
    "EFlowNeutralHadron",
    "Tower",
})


# =============================================================================
# ROOT-branch naming
# =============================================================================


def debug_branch_name(module_name: str, var: str) -> str:
    """Return the ROOT-branch name used for a ``(module, var)`` debug pair.

    We use the ``{ModuleName}.{Var}`` convention from the Parnassus
    TorchDelphes ROOT tree (see :mod:`validation.validate_torch_delphes`),
    e.g. ``"Track.PT"``, ``"ECalTower.E"``. uproot writes the ``.`` literally
    into the branch name. The existing ``truth_*`` / ``pflow_*`` branches keep
    their original snake_case names; debug branches use the
    validation-script convention so the two harnesses can share the same
    plotting / inspection code.
    """
    return f"{module_name}.{var}"


def all_debug_branch_names() -> list[str]:
    """List every ROOT-branch name written in ``--debug`` mode.

    Returned in the same module-then-variable order as
    :data:`INTERMEDIATE_BRANCHES` so callers can iterate predictably.
    """
    out: list[str] = []
    for module_name, variables in INTERMEDIATE_BRANCHES:
        for var in variables:
            out.append(debug_branch_name(module_name, var))
    return out


# =============================================================================
# Per-event jagged extraction from a flat (N, N_FEATURES) tensor
# =============================================================================

# Direct ``ColumnMap``-backed variables. ``P`` and ``ET`` are derived
# (handled in :func:`extract_variable` below).
_VAR_TO_COLUMN: dict[str, int] = {
    "PID": int(ColumnMap.PID),
    "Status": int(ColumnMap.STATUS),
    "Charge": int(ColumnMap.CHARGE),
    "E": int(ColumnMap.E),
    "Px": int(ColumnMap.PX),
    "Py": int(ColumnMap.PY),
    "Pz": int(ColumnMap.PZ),
    "PT": int(ColumnMap.PT),
    "Eta": int(ColumnMap.ETA),
    "Phi": int(ColumnMap.PHI),
    "T": int(ColumnMap.T),
    "X": int(ColumnMap.X),
    "Y": int(ColumnMap.Y),
    "Z": int(ColumnMap.Z),
    "Mass": int(ColumnMap.MASS),
    "EtaOuter": int(ColumnMap.ETA_OUTER),
}


def extract_variable(tensor: torch.Tensor, var: str) -> torch.Tensor:
    """Return a 1-D tensor of ``var`` values for every row in ``tensor``.

    ``tensor`` is a flat ``(N, N_FEATURES)`` per-module output. Most variables
    are direct :class:`~parnassus.data.particle_io.ColumnMap` columns; ``P``
    and ``ET`` are derived (momentum magnitude and transverse energy).
    """
    if var in _VAR_TO_COLUMN:
        return tensor[:, _VAR_TO_COLUMN[var]]
    if var == "P":
        px = tensor[:, ColumnMap.PX]
        py = tensor[:, ColumnMap.PY]
        pz = tensor[:, ColumnMap.PZ]
        return torch.sqrt(px * px + py * py + pz * pz)
    if var == "ET":
        # ET = E / cosh(eta) -- equivalent to E*sin(theta) for massless towers,
        # matching what C++ Delphes writes into the ``Tower.ET`` branch.
        e = tensor[:, ColumnMap.E]
        eta = tensor[:, ColumnMap.ETA]
        return e / torch.cosh(eta)
    raise KeyError(f"Unknown debug variable {var!r}")


def filter_valid_rows(tensor: torch.Tensor, module_name: str) -> torch.Tensor:
    """Drop ghost / empty rows that should not enter the histograms.

    In *learnable* card outputs, ``LearnableEfficiency`` leaves
    efficiency-killed rows in place with ``pt = 0`` (a Gumbel-ST 0/1 mask
    zeroes PT/PX/PY/PZ/E). We drop those here so the saved arrays match what a
    C++ Delphes-style filter would produce, matching the existing
    :func:`parnassus.torch_delphes.generate_pseudodata.eflow_to_class_arrays`
    convention.

    Tower-like branches (towers, EFlowPhoton, EFlowNeutralHadron, Tower) have
    ``pt`` defined as ``E/cosh(eta)`` and so are filtered on ``E`` instead.
    """
    if tensor.numel() == 0:
        return tensor
    if module_name in _TOWER_BRANCHES:
        keep = tensor[:, ColumnMap.E] > 1e-6
    else:
        keep = tensor[:, ColumnMap.PT] > 1e-6
    return tensor[keep]


def tensor_to_per_event_arrays(
    tensor: torch.Tensor,
    module_name: str,
    variables: Iterable[str],
    n_events: int,
    *,
    apply_filter: bool = True,
) -> dict[str, list[np.ndarray]]:
    """Split a flat per-module output tensor into per-event jagged arrays.

    Generic analogue of
    :func:`parnassus.torch_delphes.generate_pseudodata.eflow_to_class_arrays`:
    builds a ``{var_name: [event0_arr, event1_arr, ...]}`` mapping, one
    1-D numpy array per event, ready to feed into ``ak.Array`` for ROOT
    writing.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape ``(N, N_FEATURES)``. May be empty.
    module_name : str
        Module key as used in :data:`INTERMEDIATE_BRANCHES`. Picks the right
        ghost-row filter (PT for tracks, E for towers).
    variables : Iterable[str]
        Variable names from the kinematic-var lists above.
    n_events : int
        Number of events in the batch (some events may produce zero rows).
    apply_filter : bool
        If True (default), drop ghost rows via :func:`filter_valid_rows`.
    """
    variables = list(variables)
    if tensor.numel() == 0:
        empty = [np.empty(0, dtype=np.float32) for _ in range(n_events)]
        return {var: list(empty) for var in variables}

    if apply_filter:
        tensor = filter_valid_rows(tensor, module_name)

    if tensor.numel() == 0:
        empty = [np.empty(0, dtype=np.float32) for _ in range(n_events)]
        return {var: list(empty) for var in variables}

    ev_np = tensor[:, ColumnMap.EVENT_NUMBER].detach().cpu().numpy().astype(np.int64)
    out: dict[str, list[np.ndarray]] = {}
    for var in variables:
        vals = extract_variable(tensor, var).detach().cpu().numpy().astype(np.float32)
        out[var] = [vals[ev_np == i] for i in range(n_events)]
    return out


# =============================================================================
# Plot styling: axis labels / log-y rules / display titles
# =============================================================================
#
# The canonical definitions now live in :mod:`parnassus.torch_delphes.plotting`
# (shared with :mod:`validation.validate_torch_delphes`). We re-export them
# here so existing callers keep working without modification.

from parnassus.torch_delphes.plotting import (  # noqa: E402
    AXIS_LABELS,
    DISCRETE_VARS,
    axis_label,
    log_y_for_var,
)

__all__ = [
    "AXIS_LABELS",
    "DISCRETE_VARS",
    "INTERMEDIATE_BRANCHES",
    "axis_label",
    "debug_branch_name",
    "extract_variable",
    "filter_valid_rows",
    "log_y_for_var",
    "tensor_to_per_event_arrays",
]
