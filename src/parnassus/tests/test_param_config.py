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


def test_card_default_config_is_what_dump_writes(tmp_path: Path) -> None:
    """``card_default_config`` is the in-memory form of ``dump_param_config``.

    The Optuna search initializes every FITTED scalar from this dict rather than
    from a file, so the two must not drift apart.
    """
    card = _fresh_card()
    defaults = pc.card_default_config(card)
    assert len(defaults) == 68
    assert all(set(spec) == {"value", "trainable", "lr_scale"} for spec in defaults.values())
    assert not any(spec["trainable"] for spec in defaults.values())

    cfg_path = tmp_path / "dump.yaml"
    pc.dump_param_config(card, cfg_path)
    assert pc.load_param_config(cfg_path) == defaults

    # Every default is representable, so it is a legal starting point. (The
    # stricter trainable-logit window check lives in test_optuna_search, which
    # owns the fitted/frozen split.)
    for key, spec in defaults.items():
        pc.to_raw(key.split("[", 1)[0], spec["value"])  # raises if out of range


def test_dump_load_apply_roundtrip(tmp_path: Path) -> None:
    """A dumped config reapplied to a fresh card reproduces physical values."""
    src = _fresh_card()
    cfg_path = tmp_path / "dump.yaml"
    pc.dump_param_config(src, cfg_path)

    flat = pc.load_param_config(cfg_path)
    assert len(flat) == 68  # one entry per learnable scalar

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


def test_windowed_logit_k0s() -> None:
    """k0s_logit maps through the hardcoded (0.1, 0.5) physical window."""
    key = "HadronFractions.k0s_logit"
    assert pc.logit_bounds(key) == (0.1, 0.5)
    assert float(pc.to_raw(key, 0.3)) == pytest.approx(0.0, abs=1e-12)
    assert float(pc.to_physical(key, torch.tensor(0.0))) == pytest.approx(0.3)
    for bad in (0.05, 0.6, -0.1, 1.0):
        with pytest.raises(ValueError):
            pc.to_raw(key, bad)
    for raw in (-40.0, 40.0):
        f = float(pc.to_physical(key, torch.tensor(raw)))
        assert 0.1 <= f <= 0.5
    # Round-trip at a non-central value; other logits keep the (0, 1) window.
    assert float(pc.to_physical(key, pc.to_raw(key, 0.42))) == pytest.approx(0.42)
    assert pc.logit_bounds("HadronFractions.chad_logit") == (0.0, 1.0)


def test_windowed_logit_photon_k0l() -> None:
    """photon_logit / k0l_logit map through their hardcoded physical windows."""
    for key, lo, hi, pin in (
        ("HadronFractions.photon_logit", 0.8, 1.0, 1.0),
        ("HadronFractions.k0l_logit", 0.0, 0.4, 0.0),
    ):
        assert pc.logit_bounds(key) == (lo, hi)
        # Round-trip at an interior value.
        mid = (lo + hi) / 2
        assert float(pc.to_physical(key, pc.to_raw(key, mid))) == pytest.approx(mid)
        # Out-of-window values raise.
        for bad in (lo - 0.05, hi + 0.05):
            with pytest.raises(ValueError):
                pc.to_raw(key, bad)
        # Saturated raws stay inside the window (float64: the card/param dtype;
        # a float32 tensor rounds 0.4*sigmoid(40) to 0.400000006 > hi).
        for raw in (-40.0, 40.0):
            f = float(pc.to_physical(key, torch.tensor(raw, dtype=torch.float64)))
            assert lo <= f <= hi
        # The boundary pin loads via the _safe_logit clamp: physical value
        # lands within 1e-6 of the bound in normalized (window-width) units.
        raw = pc.to_raw(key, pin)
        assert abs(float(pc.to_physical(key, raw)) - pin) <= 1.5e-6 * (hi - lo)


@pytest.mark.parametrize(
    ("key", "bad_vals", "good_val"),
    [
        # Scaled guard window (0.005, 0.995) x window width:
        ("HadronFractions.k0s_logit", (0.1, 0.101, 0.499, 0.5), 0.45),  # (0.102, 0.498)
        ("HadronFractions.photon_logit", (0.8, 0.8005, 0.9995, 1.0), 0.97),  # (0.801, 0.999)
        ("HadronFractions.k0l_logit", (0.0, 0.001, 0.399, 0.4), 0.2),  # (0.002, 0.398)
    ],
)
def test_windowed_logit_trainable_guard(
    tmp_path: Path, key: str, bad_vals: tuple, good_val: float
) -> None:
    """Trainable windowed-logit inits must sit strictly inside the scaled
    guard window; pinned boundary values always load."""
    import yaml

    def _write(value: float, trainable: bool) -> Path:
        p = tmp_path / "one.yaml"
        with p.open("w") as f:
            yaml.safe_dump({key: {"value": value, "trainable": trainable}}, f)
        return p

    for bad in bad_vals:
        with pytest.raises(ValueError, match="trainable logit"):
            pc.load_param_config(_write(bad, trainable=True))
    assert pc.load_param_config(_write(good_val, trainable=True))[key]["value"] == good_val
    # Pinned boundary values load (clamped raw, never move).
    hi = pc.logit_bounds(key)[1]
    assert pc.load_param_config(_write(hi, trainable=False))[key]["value"] == hi


def test_dump_snaps_windowed_boundary_pins(tmp_path: Path) -> None:
    """``dump_param_config`` writes the exact window bound for pinned
    photon/k0l (not the clamp values 0.99999980 / 4e-07), so a regenerated
    YAML matches the hand-maintained pins."""
    p = tmp_path / "dump.yaml"
    pc.dump_param_config(_fresh_card(), p)
    flat = pc.load_param_config(p)
    assert flat["HadronFractions.photon_logit"]["value"] == 1.0
    assert flat["HadronFractions.k0l_logit"]["value"] == 0.0
    # Plain (0, 1) logits keep their historical representation.
    assert flat["HadronFractions.chad_logit"]["value"] == pytest.approx(1e-6)


def test_load_rejects_trainable_logit_at_dead_gradient_tails(tmp_path: Path) -> None:
    """A *trainable* efficiency/fraction logit initialized at/near 0 or 1 maps to a
    saturated raw logit (vanishing sigmoid gradient) and must be rejected; fixed
    logits anywhere in [0, 1] and interior trainable logits load fine."""
    import yaml

    key = "ChargedHadronTrackingEfficiency.eff_logits[0]"

    def _write(value: float, trainable: bool) -> Path:
        p = tmp_path / "one.yaml"
        with p.open("w") as f:
            yaml.safe_dump({key: {"value": value, "trainable": trainable}}, f)
        return p

    # Trainable logit outside the open (0.005, 0.995) interval -> reject.
    for bad in (0.0, 1e-6, 0.005, 0.995, 0.999, 1.0):
        with pytest.raises(ValueError, match="trainable logit"):
            pc.load_param_config(_write(bad, trainable=True))

    # Interior trainable logit -> ok (incl. believed-truth efficiencies at 0.99).
    assert pc.load_param_config(_write(0.85, trainable=True))[key]["value"] == 0.85
    assert pc.load_param_config(_write(0.99, trainable=True))[key]["value"] == 0.99

    # Fixed logit anywhere in [0, 1] -> ok (it never moves; e.g. a 0.99 truth).
    assert pc.load_param_config(_write(0.99, trainable=False))[key]["value"] == 0.99

    # The shipped truth config (many fixed logits at 0.95/0.99/1e-6) still loads.
    pc.load_param_config(_PARAM_CONFIG_DIR / "cms_target_default.yaml")


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


@pytest.mark.parametrize("name", ["cms_target_default.yaml"])
def test_shipped_configs_load_and_apply(name: str) -> None:
    """The shipped full configs cover the card exactly, apply, and select cleanly.

    Asserted at the *mechanism* level (apply reproduces every config value;
    select_trainable returns exactly the tensors with a trainable element), so
    the test is robust to editing which parameter a config trains.
    """
    flat = pc.load_param_config(_PARAM_CONFIG_DIR / name)
    card = _fresh_card()
    pc.apply_param_config(card, flat)  # raises if coverage/shape/range is wrong

    def keys_for(pname: str, p: torch.Tensor) -> list[str]:
        return [pname] if p.ndim == 0 else [f"{pname}[{i}]" for i in range(p.numel())]

    # apply set every scalar to its config physical value.
    for pname, p in card.named_parameters():
        phys = pc.to_physical(pname, p.detach()).flatten().tolist()
        for key, v in zip(keys_for(pname, p), phys):
            assert v == pytest.approx(flat[key]["value"], abs=1e-6)

    # select_trainable returns exactly the tensors with >=1 trainable element.
    params, groups = pc.select_trainable(card, flat, global_lr=0.05)
    param_ids = {id(p) for p in params}
    expected = {
        pname
        for pname, p in card.named_parameters()
        if any(flat[k]["trainable"] for k in keys_for(pname, p))
    }
    got = {pname for pname, p in card.named_parameters() if id(p) in param_ids}
    assert got == expected
    if expected:
        assert groups  # at least one optimizer group when something trains


def test_partial_config_over_defaults(tmp_path: Path) -> None:
    """A PARTIAL generation config overrides only the scalars it lists.

    ``load_param_config_over_defaults`` lays the file over
    ``card_default_config``: listed keys take the file's value, everything
    else stays at the card default (frozen); unknown keys still fail apply.
    """
    partial_path = _PARAM_CONFIG_DIR / "param_config_chadtrkeff.yaml"
    partial = pc.load_param_config(partial_path)
    card = _fresh_card()
    defaults = pc.card_default_config(card)
    merged = pc.load_param_config_over_defaults(partial_path, card)

    assert set(merged) == set(defaults)
    assert set(partial) < set(defaults)
    for key, spec in merged.items():
        assert spec == (partial[key] if key in partial else defaults[key])
    assert not any(spec["trainable"] for spec in merged.values())
    pc.apply_param_config(card, merged)  # full cover -> applies cleanly
    applied = pc.card_default_config(card)
    for key, spec in merged.items():
        assert applied[key]["value"] == pytest.approx(spec["value"], abs=1e-6)

    bad = tmp_path / "bad.yaml"
    bad.write_text("NoSuchModule.scale_raw[0]:\n  value: 1.1\n")
    with pytest.raises(ValueError, match="unknown entries"):
        pc.apply_param_config(_fresh_card(), pc.load_param_config_over_defaults(bad, card))
