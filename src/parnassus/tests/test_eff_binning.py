"""Tests for the mode-dependent tracking-efficiency binning
(CONSOLIDATE_MODES_PLAN.md phase 2): cms4 = the legacy delphes layout,
ptbins12 = the fullsim chad refinement ported from
diff_delphes_runze_cmssinglejet."""

import pytest
import torch

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.learnable import (
    CMS_EFF_REGION_SPECS,
    CMS_EFF_REGION_SPECS_PTBINS,
    CMSChargedHadronLearnableEfficiency,
)


def test_cms4_layout_unchanged():
    card = CMSEnergyFlowDefault(learnable=True)  # default binning
    assert card.eff_binning == "cms4"
    assert card.ChargedHadronTrackingEfficiency.eff_logits.numel() == 4
    assert card.ElectronTrackingEfficiency.eff_logits.numel() == 6
    assert card.MuonTrackingEfficiency.eff_logits.numel() == 6
    assert CMS_EFF_REGION_SPECS["electron"].label_offset == 4
    assert CMS_EFF_REGION_SPECS["muon"].label_offset == 10


def test_ptbins12_layout():
    card = CMSEnergyFlowDefault(learnable=True, eff_binning="ptbins12")
    assert card.ChargedHadronTrackingEfficiency.eff_logits.numel() == 12
    assert card.ElectronTrackingEfficiency.eff_logits.numel() == 6
    assert card.MuonTrackingEfficiency.eff_logits.numel() == 6
    # Disjoint global label ranges: chad 1-12, electron 13-18, muon 19-24.
    assert CMS_EFF_REGION_SPECS_PTBINS["charged_hadron"].label_offset == 0
    assert CMS_EFF_REGION_SPECS_PTBINS["electron"].label_offset == 12
    assert CMS_EFF_REGION_SPECS_PTBINS["muon"].label_offset == 18


def test_ptbins12_defaults_replicate_legacy_values():
    """Every above-1-GeV sub-bin defaults to its containing legacy bin's value,
    so a freshly constructed ptbins12 card evaluates the same efficiency as the
    legacy card at any (pt, eta)."""
    cms4 = CMSChargedHadronLearnableEfficiency(binning="cms4")
    pt12 = CMSChargedHadronLearnableEfficiency(binning="ptbins12")
    pt = torch.tensor([0.5, 2.0, 15.0, 30.0, 70.0, 500.0] * 2, dtype=torch.float64)
    eta = torch.tensor([0.5] * 6 + [2.0] * 6, dtype=torch.float64)
    assert torch.allclose(
        cms4.compute_efficiency(pt, eta), pt12.compute_efficiency(pt, eta)
    )


def test_invalid_binning_rejected():
    with pytest.raises((ValueError, KeyError)):
        CMSEnergyFlowDefault(learnable=True, eff_binning="cms99")
