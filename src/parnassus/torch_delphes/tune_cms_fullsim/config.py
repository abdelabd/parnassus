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

# NOTE: "chad_region_counts" is a per-event (n_events, 4) target carried for the
# differentiable expected-count loss term; it is NOT a plottable 1-D/2-D observable
# and has no prediction-side counterpart in load_pflow_targets_from_tensor, so the
# intermediate-plot loop skips it (it is absent from the pred dict).
OBSERVABLES: list[str] = [
    "pt", "eta", "log_E", "log_pt", "multiplicity", "ht", "log_ht", "pid", "chad_region_counts",
]


# =============================================================================
# Learning rate
# =============================================================================
#
# Global Adam learning-rate magnitude (the ``--lr`` default). Each parameter's
# effective learning rate is ``--lr * lr_scale``, where the per-parameter
# ``lr_scale`` comes from the YAML param config and the Adam parameter groups
# are built by :func:`parnassus.torch_delphes.param_config.select_trainable`.
_DEFAULT_LR: float = 1e-2
