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
- :mod:`.config`      — constants (branch names, observable keys, default LR).
- :mod:`.distributed` — DDP / rank helpers.
- :mod:`.data`        — ROOT I/O and observable construction.
- :mod:`.loss`        — the sliced-Wasserstein training loss (self-contained).
- :mod:`.training`    — the Adam fit loop.
- :mod:`.cli`         — the ``main()`` entry point (see also :mod:`.__main__`).

Run with ``python -m parnassus.torch_delphes.tune_cms_fullsim ...``.
"""

from __future__ import annotations

from .cli import main
from .config import OBSERVABLES, PFLOW_BRANCHES, TRUTH_BRANCHES
from .data import (
    load_cms_flow_root,
)
from .training import fit_card_to_fullsim

__all__ = [
    "OBSERVABLES",
    "PFLOW_BRANCHES",
    "TRUTH_BRANCHES",
    "fit_card_to_fullsim",
    "load_cms_flow_root",
    "main",
]
