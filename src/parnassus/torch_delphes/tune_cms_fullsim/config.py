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

# NOTE: the "*_region_counts" keys are per-event (n_events, n_regions) targets carried
# for the differentiable expected-count loss terms (one per track species); they are
# NOT plottable 1-D/2-D observables and have no prediction-side counterpart in
# load_pflow_targets_from_tensor, so the intermediate-plot loop skips them (they are
# absent from the pred dict).
OBSERVABLES: list[str] = [
    "pt", "eta", "log_E", "log_pt", "multiplicity", "ht", "log_ht", "pid",
    "chad_region_counts", "electron_region_counts", "muon_region_counts",
    # Calorimeter object-count targets (per-|eta|-region; ECal photons, HCal
    # neutral hadrons) for the differentiable resolution-param count term. Same
    # non-plottable, prediction-less status as the *_region_counts above.
    "ecal_photon_region_counts", "hcal_nh_region_counts",
]


# =============================================================================
# Count-term wiring
# =============================================================================
#
# One row per differentiable per-species count term, shared by training.py (which
# injects the card's expected-count output into the pred dict) and loss.py (which
# matches it against the data target). Tuple = (card forward-output key, pred-dict
# key, target-dict key). Keeping this in one place ensures the three files agree on
# the per-species names.
COUNT_TERM_KEYS: tuple[tuple[str, str, str], ...] = (
    ("ChargedHadronExpectedCounts", "chad_expected_counts", "chad_region_counts"),
    ("ElectronExpectedCounts", "electron_expected_counts", "electron_region_counts"),
    ("MuonExpectedCounts", "muon_expected_counts", "muon_region_counts"),
)

# Calorimeter resolution-param count terms. Same (card-key, pred-key, target-key)
# shape as COUNT_TERM_KEYS, wired identically in training.py and loss.py, but the
# expected count comes from the differentiable soft significance gate in
# SimpleCalorimeter (per |eta| region) and supplies the resolution params the
# correctly-signed d(membership)/d(theta) gradient the hard cuts drop. ECal -> the
# EFlowPhoton (pid 22) clouds; HCal -> the neutral-hadron (pid 111) clouds.
CALO_COUNT_TERM_KEYS: tuple[tuple[str, str, str], ...] = (
    ("EcalPhotonExpectedCounts", "ecal_photon_expected_counts", "ecal_photon_region_counts"),
    ("HcalNeutralHadronExpectedCounts", "hcal_neutral_hadron_expected_counts", "hcal_nh_region_counts"),
)


# =============================================================================
# Learning rate
# =============================================================================
#
# Global Adam learning-rate magnitude (the ``--lr`` default). Each parameter's
# effective learning rate is ``--lr * lr_scale``, where the per-parameter
# ``lr_scale`` comes from the YAML param config and the Adam parameter groups
# are built by :func:`parnassus.torch_delphes.param_config.select_trainable`.
_DEFAULT_LR: float = 1e-2
