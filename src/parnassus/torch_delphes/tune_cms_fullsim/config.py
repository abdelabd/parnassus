"""Constants for the ``tune_cms_fullsim`` package.

This module centralizes the constants used across the package:

- the ROOT branch names we consume (:data:`TRUTH_BRANCHES`, :data:`PFLOW_BRANCHES`);
- the observable keys built from them (:data:`OBSERVABLES`);
- the default global Adam learning rate (:data:`_DEFAULT_LR`).

Nothing here consumes the global RNG, so importing this module has no effect on
reproducibility.
"""

from __future__ import annotations


# =============================================================================
# ROOT branch names
# =============================================================================

# Exact branch names we consume. Keeping them centralized here makes it trivial
# to retarget the script at a different reco format (e.g. a Delphes output
# tree) by passing an alternative schema dict.
TRUTH_BRANCHES: tuple[str, ...] = ("truth_pt", "truth_eta", "truth_phi", "truth_class")
PFLOW_BRANCHES: tuple[str, ...] = ("pflow_pt", "pflow_eta", "pflow_phi", "pflow_class")


# =============================================================================
# Observables
# =============================================================================

OBSERVABLES: list[str] = ["pt", "eta", "log_E", "log_pt", "multiplicity", "ht", "log_ht", "pid"]


# =============================================================================
# Learning rate
# =============================================================================
#
# Global Adam learning-rate magnitude (the ``--lr`` default). Each parameter's
# effective learning rate is ``--lr * lr_scale``, where the per-parameter
# ``lr_scale`` comes from the YAML param config and the Adam parameter groups
# are built by :func:`parnassus.torch_delphes.param_config.select_trainable`.
_DEFAULT_LR: float = 1e-2
