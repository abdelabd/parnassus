"""Tests for the --existence umbrella (CONSOLIDATE_MODES_PLAN.md phase 1):
the shared bundle resolver, and the tower_bce=False export gating."""

import pytest
import torch

from parnassus.torch_delphes.tune_cms_fullsim.config import resolve_existence_bundle
from parnassus.torch_delphes.tune_cms_fullsim.loss import CALO_COUNT_WEIGHT

from test_torch_delphes_learnable import _make_batch  # noqa: F401 (same test dir)
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault


def _resolve(**kw):
    base = dict(
        mode="delphes",
        existence=None,
        eff_loss=None,
        calo_bce=None,
        calo_count_weight=None,
        default_calo_count_weight=CALO_COUNT_WEIGHT,
    )
    base.update(kw)
    return resolve_existence_bundle(**base)


def test_no_umbrella_keeps_legacy_defaults():
    assert _resolve() == ("bce", False, CALO_COUNT_WEIGHT)
    assert _resolve(mode="fullsim") == ("counts", False, CALO_COUNT_WEIGHT)


def test_bundles():
    assert _resolve(existence="counts") == ("counts", False, CALO_COUNT_WEIGHT)
    assert _resolve(existence="bce") == ("bce", True, 0.0)


def test_explicit_flags_win_over_bundle():
    assert _resolve(existence="bce", calo_count_weight=0.5) == ("bce", True, 0.5)
    assert _resolve(existence="bce", calo_bce=False) == ("bce", False, 0.0)
    assert _resolve(existence="counts", eff_loss="bce") == (
        "bce", False, CALO_COUNT_WEIGHT,
    )


def test_bce_requires_delphes_mode():
    with pytest.raises(SystemExit):
        _resolve(mode="fullsim", existence="bce")
    # counts is fine in fullsim (it IS fullsim behavior)
    assert _resolve(mode="fullsim", existence="counts") == (
        "counts", False, CALO_COUNT_WEIGHT,
    )


def test_invalid_existence_value():
    with pytest.raises(ValueError):
        _resolve(existence="maybe")


def test_tower_bce_false_skips_exports_and_draw_parity():
    """tower_bce=False must (a) export no bce_*/track_cond keys and (b) leave
    the reconstruction draws byte-identical (the gated block consumes no RNG)."""
    batch = _make_batch(n=400, seed=3)
    outs = {}
    for flag in (True, False):
        card = CMSEnergyFlowDefault(debug=False, learnable=True, tower_bce=flag)
        torch.manual_seed(7)
        outs[flag] = card(batch.clone())  # clone: forward mutates its input
    for key in ("EcalCountExport", "HcalCountExport"):
        on, off = outs[True][key], outs[False][key]
        assert any(k.startswith("bce_") for k in on)
        assert not any(k.startswith("bce_") for k in off)
        assert "track_cond_out" not in off
    for key in ("EFlowTrack", "EFlowPhoton", "EFlowNeutralHadron"):
        assert torch.equal(outs[True][key], outs[False][key]), key
