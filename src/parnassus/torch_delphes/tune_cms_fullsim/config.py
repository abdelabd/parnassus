"""Constants for the ``tune_cms_fullsim`` package.

This module centralizes every tunable constant used across the package so
they live in exactly one place:

- the ROOT branch names we consume (:data:`TRUTH_BRANCHES`, :data:`PFLOW_BRANCHES`);
- the default histogram bin edges and per-observable loss weights
  (:data:`DEFAULT_BIN_EDGES`, :data:`DEFAULT_OBS_WEIGHTS`);
- the parameter-name matchers used to bucket the 66 learnable params into
  four optimizer groups (:data:`_SCALE_SUFFIXES`, :data:`_EFFICIENCY_SUFFIXES`,
  :data:`_FRACTION_FRAGMENTS`) and the per-group default learning rates
  (:data:`_DEFAULT_LR_SCALES` etc.).

Nothing here consumes the global RNG (``torch.linspace`` is deterministic),
so importing this module has no effect on reproducibility.
"""

from __future__ import annotations

import torch

# =============================================================================
# ROOT branch names
# =============================================================================

# Exact branch names we consume. Keeping them centralized here makes it trivial
# to retarget the script at a different reco format (e.g. a Delphes output
# tree) by passing an alternative schema dict.
TRUTH_BRANCHES: tuple[str, ...] = ("truth_pt", "truth_eta", "truth_phi", "truth_class")
PFLOW_BRANCHES: tuple[str, ...] = ("pflow_pt", "pflow_eta", "pflow_phi", "pflow_class")


# =============================================================================
# Observable histogram bins and loss weights
# =============================================================================

# Default bin edges for the four observables. Wide-enough to cover the bulk
# of both target and trainee distributions on a QCD-jet sample; users can
# override these via --bin-config when needed.
DEFAULT_BIN_EDGES: dict[str, torch.Tensor] = {
    "pt": torch.linspace(0.0, 200.0, 41, dtype=torch.float64),
    "eta": torch.linspace(-5.0, 5.0, 41, dtype=torch.float64),
    # "phi": torch.linspace(-np.pi, np.pi, 41, dtype=torch.float64),
    # E ranges higher than pt since E = pt * cosh(eta) >= pt for the
    # reconstructed particles; widen the upper edge accordingly.
    "E": torch.linspace(0.0, 400.0, 41, dtype=torch.float64),
    # log(pt) bin spanning ~0.37 GeV (e^-1) to ~403 GeV (e^6); linear in
    # log space gives Adam much better gradient signal in the high-pT
    # tail than the linear pt histogram does.
    "log_pt": torch.linspace(-1.0, 6.0, 41, dtype=torch.float64),
    "multiplicity": torch.linspace(0.0, 400.0, 41, dtype=torch.float64),
    "ht": torch.linspace(0.0, 2000.0, 41, dtype=torch.float64),
}

# Default per-observable weights. The particle-level observables (pt, eta)
# are far less noisy than the per-event scalars (multiplicity, ht) because
# they have O(N_particles) rather than O(N_events) samples, so we upweight
# them and put a small tie-breaking weight on the per-event pair.
#
# E and log_pt are also particle-level. log_pt is highly correlated with pt
# (it's a monotone reparametrization), so we down-weight it slightly to
# avoid double-counting; its main role is to give Adam a non-vanishing
# gradient on the high-pT tail where the linear-pt histogram bins are
# nearly empty. E adds genuinely new information through the eta-dependent
# pt -> p_total mapping (it probes the forward calo scales).
DEFAULT_OBS_WEIGHTS: dict[str, float] = {
    "pt": 1.0,
    "eta": 1.0,
    # "phi": 1.0,
    "E": 1.0,
    "log_pt": 0.5,
    "multiplicity": 0.5,
    "ht": 0.5,
}


# =============================================================================
# Parameter-group classification and per-group learning rates
# =============================================================================
#
# The 66 learnable parameters live in parameter spaces with very different
# natural scales, so a single global Adam learning rate is a poor fit. The
# name matchers below let ``training.build_parameter_groups`` bucket each
# named parameter into one of four groups (scales / efficiency / fractions /
# resolution), each with its own learning rate. See the rationale comment in
# ``training.py`` (above ``_classify_parameter``) for why the four groups
# need different step sizes.

_SCALE_SUFFIXES = ("scale_raw",)
_EFFICIENCY_SUFFIXES = ("eff_logits", "rate_raw")
_FRACTION_FRAGMENTS = ("HadronFractions",)
# Everything else on a learnable card that has an ``nn.Parameter`` is a
# resolution coefficient (softplus-wrapped positive number).

# Learning-rate model: a single global magnitude (``_DEFAULT_LR``) times a
# per-group *relative ratio*. The effective Adam learning rate of a group is
#
#     effective_lr = lr * lr_<group>
#
# where ``lr`` is the global ``--lr`` and ``lr_<group>`` is the dimensionless
# ``--lr-<group>`` ratio below. This lets the user sweep the overall step size
# with one knob (``--lr``) while keeping the physically-motivated ratios
# between groups fixed. The default ratios encode "resolution coefficients
# step 10x slower than everything else" (1 : 1 : 1 : 0.1), because the
# softplus-wrapped resolution coefficients are far more sensitive to a raw
# Adam step than the tanh/sigmoid-wrapped scale/efficiency/fraction params
# (see the rationale comment in ``training.py`` above ``_classify_parameter``).
_DEFAULT_LR: float = 1e-3

_DEFAULT_LR_SCALES: float = 1.0
_DEFAULT_LR_EFFICIENCY: float = 1.0
_DEFAULT_LR_FRACTIONS: float = 1.0
_DEFAULT_LR_RESOLUTION: float = 0.1
