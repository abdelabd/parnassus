"""Tests for the per-truth-particle survival labels written by generate_pseudodata.

These labels are the training target of the BCE efficiency loss (see
``torch_delphes/EFF_LOSS_PLAN.md`` / ``EFF_LOSS_MOTIV.md``). The tests verify:

1. **Alignment & consistency**: the three label branches are per-event aligned
   with the truth arrays; ``truth_survived => truth_in_tracker``;
   ``truth_eff_region >= 0`` exactly on the ``truth_in_tracker`` support;
   neutral species are never in-tracker.
2. **The delphes-mode invariant** is enforced inside ``truth_arrays_to_pflow``
   itself (a raise, exercised on every call): survival-to-output equals the
   efficiency-stage Bernoulli mask on the support. Here we only check the call
   completes, which means the invariant held.
3. **Closed form**: per-region survival fractions match ``sigmoid(eff_logits)``
   of the generating card within binomial error — the Gumbel-sigmoid hard
   sample is exactly Bernoulli(eff) in distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from parnassus.torch_delphes.generate_pseudodata import (
    make_target_card,
    truth_arrays_to_pflow,
)
from parnassus.torch_delphes.learnable import CMS_EFF_REGION_SPECS

RNG_SEED = 1234


def _make_truth_arrays(
    n_events: int,
    per_event: list[tuple[int, float, float]],
    rng: np.random.Generator,
) -> dict[str, list[np.ndarray]]:
    """Build class-based truth arrays: ``per_event`` lists (pdgid, pt_lo, pt_hi)
    specs; each event gets one particle per spec with uniform pt in (lo, hi] and
    uniform eta in (-2.4, 2.4)."""
    from parnassus.utils import pid_to_class

    pts, etas, phis, clss, pids = [], [], [], [], []
    for _ in range(n_events):
        pt = np.array([rng.uniform(lo, hi) for _, lo, hi in per_event], dtype=np.float32)
        eta = rng.uniform(-2.4, 2.4, size=len(per_event)).astype(np.float32)
        phi = rng.uniform(-np.pi, np.pi, size=len(per_event)).astype(np.float32)
        pid = np.array([p for p, _, _ in per_event], dtype=np.int64)
        cls = np.array([pid_to_class(int(p)) for p in pid], dtype=np.int32)
        pts.append(pt)
        etas.append(eta)
        phis.append(phi)
        clss.append(cls)
        pids.append(pid)
    return {
        "truth_pt": pts,
        "truth_eta": etas,
        "truth_phi": phis,
        "truth_class": clss,
        "truth_pdgid": pids,
    }


@pytest.fixture(scope="module")
def labeled_sample():
    """A ~30k-particle mixed sample run through the default target card."""
    torch.manual_seed(RNG_SEED)
    rng = np.random.default_rng(RNG_SEED)
    # Per event: 4 charged pions across the chad pt bins, 2 muons and 2
    # electrons in their mid-pt bins, plus neutrals (photon, K_L).
    per_event = [
        (211, 0.2, 0.9),
        (-211, 0.2, 0.9),
        (211, 2.0, 50.0),
        (-211, 2.0, 50.0),
        (13, 5.0, 100.0),
        (-13, 5.0, 100.0),
        (11, 5.0, 90.0),
        (-11, 5.0, 90.0),
        (22, 1.0, 20.0),
        (130, 1.0, 20.0),
    ]
    n_events = 3000
    truth = _make_truth_arrays(n_events, per_event, rng)
    card = make_target_card(device="cpu")  # CMS defaults
    branches = truth_arrays_to_pflow(truth, card, batch_size=512, device="cpu")
    return truth, branches, card


def test_label_alignment_and_consistency(labeled_sample):
    truth, branches, _ = labeled_sample
    n_events = len(truth["truth_pt"])
    for key in ("truth_in_tracker", "truth_eff_region", "truth_survived"):
        assert key in branches
        assert len(branches[key]) == n_events
        for i in range(n_events):
            assert len(branches[key][i]) == len(truth["truth_pt"][i])

    in_tracker = np.concatenate(branches["truth_in_tracker"])
    region = np.concatenate(branches["truth_eff_region"])
    survived = np.concatenate(branches["truth_survived"])
    cls = np.concatenate(truth["truth_class"])

    # survived => in_tracker; region >= 0 exactly on the support.
    assert not np.any(survived & ~in_tracker)
    assert np.array_equal(region >= 0, in_tracker)
    # Neutral species (class 3 = neutral hadron, 4 = photon) never reach a
    # tracking-efficiency module.
    assert not np.any(in_tracker & np.isin(cls, [3, 4]))
    # The charged samples overwhelmingly reach the tracker at these pt.
    assert in_tracker[np.isin(cls, [0, 1, 2])].mean() > 0.5
    # Both outcomes occur.
    assert 0 < survived[in_tracker].mean() < 1


def test_regions_match_specs(labeled_sample):
    truth, branches, _ = labeled_sample
    region = np.concatenate(branches["truth_eff_region"])
    cls = np.concatenate(truth["truth_class"])
    in_tracker = np.concatenate(branches["truth_in_tracker"])

    spec_ranges = {
        0: CMS_EFF_REGION_SPECS["charged_hadron"],
        1: CMS_EFF_REGION_SPECS["electron"],
        2: CMS_EFF_REGION_SPECS["muon"],
    }
    for c, spec in spec_ranges.items():
        sel = in_tracker & (cls == c) & (region > 0)
        assert sel.any(), f"no in-region particles for class {c}"
        labels = region[sel]
        lo, hi = spec.label_offset + 1, spec.label_offset + spec.n_regions
        assert labels.min() >= lo and labels.max() <= hi, (
            f"class {c} labels outside [{lo}, {hi}]"
        )


def test_survival_fraction_matches_card_efficiency(labeled_sample):
    """Closed form: on the support, P(survived | region r) = sigmoid(eff_logits[r])
    (for constant bins; the muon high-pt exponential bins are not populated here)."""
    _, branches, card = labeled_sample
    region = np.concatenate(branches["truth_eff_region"])
    survived = np.concatenate(branches["truth_survived"])

    modules = {
        "charged_hadron": card.ChargedHadronTrackingEfficiency,
        "electron": card.ElectronTrackingEfficiency,
        "muon": card.MuonTrackingEfficiency,
    }
    checked = 0
    for name, module in modules.items():
        spec = CMS_EFF_REGION_SPECS[name]
        effs = torch.sigmoid(module.eff_logits).detach().numpy()
        for r in range(spec.n_regions):
            if name == "muon" and r % spec.n_pt == spec.n_pt - 1:
                continue  # exponential roll-off bins (pt > 1 TeV): not sampled
            label = spec.label_offset + r + 1
            sel = region == label
            n = int(sel.sum())
            if n < 200:
                continue  # not enough stats for a meaningful check
            frac = survived[sel].mean()
            sigma = np.sqrt(effs[r] * (1 - effs[r]) / n)
            assert abs(frac - effs[r]) < max(5 * sigma, 0.02), (
                f"{name} region {r} (label {label}): survival fraction {frac:.4f} "
                f"vs card efficiency {effs[r]:.4f} with n={n}"
            )
            checked += 1
    assert checked >= 8, f"only {checked} regions had enough statistics"
