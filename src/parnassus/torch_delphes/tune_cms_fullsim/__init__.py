"""Fit the learnable CMS TorchDelphes card to CMS full-simulation reco.

This package fits the 66 learnable parameters of
:class:`parnassus.torch_delphes.defaults.CMSEnergyFlowDefault` so that its
reconstructed observable distributions match a CMS full-simulation sample.
It used to be a single ``tune_cms_fullsim.py`` module; it is now split by
concern across the submodules below, but the public API is unchanged — every
name that used to be importable from
``parnassus.torch_delphes.tune_cms_fullsim`` is re-exported here.

Submodules
----------
- :mod:`.config`      — all constants (branch names, bin edges, weights, LRs).
- :mod:`.distributed` — DDP / rank helpers.
- :mod:`.data`        — ROOT I/O and observable construction.
- :mod:`.loss`        — differentiable soft-histogram losses (self-contained).
- :mod:`.training`    — parameter groups and the Adam fit loop.
- :mod:`.fixture`     — synthetic fallback ROOT writer.
- :mod:`.cli`         — the ``main()`` entry point (see also :mod:`.__main__`).

Run with ``python -m parnassus.torch_delphes.tune_cms_fullsim ...``.
"""

from __future__ import annotations

from .cli import main
from .config import DEFAULT_BIN_EDGES, DEFAULT_OBS_WEIGHTS, PFLOW_BRANCHES, TRUTH_BRANCHES
from .data import (
    load_cms_flow_root,
)
from .fixture import write_synthetic_fixture
from .loss import multi_observable_loss, multi_observable_loss_distributed
from .training import build_parameter_groups, fit_card_to_fullsim

__all__ = [
    "DEFAULT_BIN_EDGES",
    "DEFAULT_OBS_WEIGHTS",
    "PFLOW_BRANCHES",
    "TRUTH_BRANCHES",
    "build_parameter_groups",
    "fit_card_to_fullsim",
    "load_cms_flow_root",
    "main",
    "multi_observable_loss",
    "multi_observable_loss_distributed",
    "pflow_target_observables",
    "trainee_observables",
    "truth_to_particle_tensor",
    "write_synthetic_fixture",
]
