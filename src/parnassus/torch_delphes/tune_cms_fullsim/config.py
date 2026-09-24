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
#
# ``truth_pdgid`` carries the *real* PDG IDs of the truth particles
# (130 for K_L0, 2112 for neutron, 321 for charged kaon, 2212 for proton, ...).
# We pass these through unchanged to the trainee card so the ECal/HCal
# energy-fraction LUT routes long-lived neutral hadrons (K_L0, n) to HCal
# instead of mis-mapping them to a single ``class_to_pid`` bucket value
# (``111`` = pi0) that would route everything via ECal as photons.
TRUTH_BRANCHES: tuple[str, ...] = (
    "truth_pt",
    "truth_eta",
    "truth_phi",
    "truth_class",
    "truth_pdgid",
)
PFLOW_BRANCHES: tuple[str, ...] = ("pflow_pt", "pflow_eta", "pflow_phi", "pflow_class")


# =============================================================================
# Observables
# =============================================================================

OBSERVABLES: list[str] = [
    "pt", "eta", "phi", "log_E", "log_pt", "multiplicity", "ht", "log_ht", "pid",
    # Per-event count of truth charged hadrons inside the reco acceptance
    # (pt >= reco_pt_cut, |eta| <= eta_cut) -- the per-event cap used by
    # apply_chad_truncation. A (n_events,) target-side scalar with no
    # prediction-side counterpart, so plots skip it.
    "n_truth_chad",
]


# =============================================================================
# Acceptance-cut defaults
# =============================================================================
#
# Defaults for the --truth-pt-cut / --reco-pt-cut / --eta-cut CLI args. The truth
# cut harmonizes the trainee INPUT with the externally preprocessed files (which
# already apply truth pt >= 0.25, |eta| <= 2.7); the reco cut is applied to BOTH
# the pflow target and the trainee output (the target files already carry
# pflow pt >= 1, |eta| <= 2.7, so it is a no-op there at the defaults, but it is
# a real cut on the trainee: sub-GeV photons and |eta| > 2.7 forward objects).
DEFAULT_TRUTH_PT_CUT: float = 0.25
DEFAULT_RECO_PT_CUT: float = 1.0
DEFAULT_ABS_ETA_CUT: float = 2.7

# --mode for the tune entrypoints. fullsim = the acceptance cuts + chad truncation
# above and the photon merger below (default); delphes = all of them off
# (diff-Delphes must reproduce Delphes: no CMS selection, no supercluster-scale
# merging), the --*-cut / --no-chad-truncation / --photon-merge-radius flags (and
# the optuna radius search) are ignored.
MODE_CHOICES: tuple[str, ...] = ("fullsim", "delphes")
DEFAULT_MODE: str = "fullsim"

# Default PhotonClusterMerger seed-cone radius for the tune entrypoints (M2:
# frozen constant; <= 0 disables; forced off in --mode delphes). Calibrated
# against CMS PF photon counts on the dijet sample
# (docs/photon_merger_fraction_design.md sec 3.1 + M0/M1).
DEFAULT_PHOTON_MERGE_RADIUS: float = 0.045


# =============================================================================
# Learning rate
# =============================================================================
#
# Global Adam learning-rate magnitude (the ``--lr`` default). Each parameter's
# effective learning rate is ``--lr * lr_scale``, where the per-parameter
# ``lr_scale`` comes from the YAML param config and the Adam parameter groups
# are built by :func:`parnassus.torch_delphes.param_config.select_trainable`.
_DEFAULT_LR: float = 1e-2
