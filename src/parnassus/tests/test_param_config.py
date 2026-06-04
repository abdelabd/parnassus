"""Tests for the declarative parameter-config loader.

Covers the contract the generation + tuning entrypoints rely on:

- round-trip: ``dump_param_config`` -> ``load_param_config`` ->
  ``apply_param_config`` reproduces the original physical values;
- range guards: an out-of-range physical value (e.g. a momentum scale of
  1.3, on the open boundary of the ``1 + 0.3*tanh`` parameterization) raises
  rather than silently producing NaN;
- partial trainability: marking one element of a vector parameter trainable
  freezes its siblings exactly across optimizer steps (the gradient mask);
- the shipped configs load and apply onto a fresh card.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

_CHAD_SCALE = "ChargedHadronMomentumSmearing.resolution_module.scale_raw"
_PARAM_CONFIG_DIR = Path(pc.__file__).resolve().parent / "param_configs"


def _fresh_card() -> CMSEnergyFlowDefault:
    torch.manual_seed(0)
    return CMSEnergyFlowDefault(debug=False, learnable=True)


def test_dump_load_apply_roundtrip(tmp_path: Path) -> None:
    """A dumped config reapplied to a fresh card reproduces physical values."""
    src = _fresh_card()
    cfg_path = tmp_path / "dump.yaml"
    pc.dump_param_config(src, cfg_path)

    flat = pc.load_param_config(cfg_path)
    assert len(flat) == 66  # one entry per learnable scalar

    dst = _fresh_card()
    # Perturb dst so apply has to actually do something.
    with torch.no_grad():
        for p in dst.parameters():
            p.add_(0.1)
    pc.apply_param_config(dst, flat)

    for name, p in dst.named_parameters():
        phys = pc.to_physical(name, p.detach()).flatten().tolist()
        for i, v in enumerate(phys):
            key = name if p.ndim == 0 else f"{name}[{i}]"
            assert v == pytest.approx(flat[key]["value"], abs=1e-6)


def test_to_raw_rejects_scale_on_boundary() -> None:
    """A scale of 1.3 is on the open boundary and must raise, not NaN."""
    with pytest.raises(ValueError, match="open interval"):
        pc.to_raw(_CHAD_SCALE, [1.3, 1.3, 1.3])
    # Reachable values are fine and round-trip.
    raw = pc.to_raw(_CHAD_SCALE, [1.25, 1.2, 0.9])
    back = pc.to_physical(_CHAD_SCALE, raw).tolist()
    assert back == pytest.approx([1.25, 1.2, 0.9], abs=1e-9)


def test_to_raw_rejects_bad_softplus_and_logit() -> None:
    """Non-positive coefficients and out-of-[0,1] fractions raise."""
    with pytest.raises(ValueError):
        pc.to_raw("ECal.resolution_func.common_c_E", -1.0)
    with pytest.raises(ValueError):
        pc.to_raw("HadronFractions.k0s_logit", 1.5)


def test_apply_validates_coverage(tmp_path: Path) -> None:
    """A config missing an entry is rejected against the card."""
    card = _fresh_card()
    cfg_path = tmp_path / "dump.yaml"
    pc.dump_param_config(card, cfg_path)
    flat = pc.load_param_config(cfg_path)
    flat.pop(f"{_CHAD_SCALE}[0]")
    with pytest.raises(ValueError, match="missing"):
        pc.apply_param_config(_fresh_card(), flat)


def test_partial_trainability_freezes_siblings(tmp_path: Path) -> None:
    """Only the trainable element of a vector moves; siblings stay fixed."""
    card = _fresh_card()
    cfg_path = tmp_path / "dump.yaml"
    pc.dump_param_config(card, cfg_path)
    flat = pc.load_param_config(cfg_path)

    # Train only CHAD scale element 0; pin 1 and 2.
    flat[f"{_CHAD_SCALE}[0]"]["trainable"] = True
    flat[f"{_CHAD_SCALE}[1]"]["trainable"] = False
    flat[f"{_CHAD_SCALE}[2]"]["trainable"] = False

    pc.apply_param_config(card, flat)
    params, groups = pc.select_trainable(card, flat, global_lr=0.1)

    chad = dict(card.named_parameters())[_CHAD_SCALE]
    assert params == [chad]  # the one tensor with any trainable element
    assert len(groups) == 1 and groups[0]["lr"] == pytest.approx(0.1)

    frozen_before = chad.detach()[[1, 2]].clone()
    moved_before = chad.detach()[0].item()
    opt = torch.optim.Adam(groups)
    for _ in range(5):
        opt.zero_grad()
        # Pulls every element toward 5 (nonzero gradient at the start value);
        # the mask must keep elements 1 and 2 from moving.
        loss = ((chad - 5.0) ** 2).sum()
        loss.backward()
        opt.step()

    assert torch.equal(chad.detach()[[1, 2]], frozen_before)  # exactly fixed
    assert chad.detach()[0].item() != moved_before  # element 0 moved


def test_select_trainable_rejects_mixed_lr_scale(tmp_path: Path) -> None:
    """Trainable elements of one tensor must share a single lr_scale."""
    card = _fresh_card()
    cfg_path = tmp_path / "dump.yaml"
    pc.dump_param_config(card, cfg_path)
    flat = pc.load_param_config(cfg_path)
    flat[f"{_CHAD_SCALE}[0]"].update(trainable=True, lr_scale=1.0)
    flat[f"{_CHAD_SCALE}[1]"].update(trainable=True, lr_scale=0.5)
    with pytest.raises(ValueError, match="lr_scale"):
        pc.select_trainable(card, flat, global_lr=0.1)


@pytest.mark.parametrize(
    "name", ["cms_target_default.yaml", "debug_train_chad_scale_barrel.yaml"]
)
def test_shipped_configs_load_and_apply(name: str) -> None:
    """The shipped configs cover the card exactly and apply cleanly."""
    flat = pc.load_param_config(_PARAM_CONFIG_DIR / name)
    card = _fresh_card()
    pc.apply_param_config(card, flat)  # raises if coverage/shape/range is wrong
    params, groups = pc.select_trainable(card, flat, global_lr=0.05)
    if name == "debug_train_chad_scale_barrel.yaml":
        chad = dict(card.named_parameters())[_CHAD_SCALE]
        assert params == [chad]
        # Barrel starts off-truth at 1.0; the pinned siblings are at 1.25.
        scales = pc.to_physical(_CHAD_SCALE, chad.detach()).tolist()
        assert scales[0] == pytest.approx(1.0, abs=1e-9)
        assert scales[1] == pytest.approx(1.25, abs=1e-9)
