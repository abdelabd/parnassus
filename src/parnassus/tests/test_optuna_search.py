"""Tests for the Optuna hyperparameter search (``tune_cms_fullsim.optuna_search``).

The pure-logic tests (config parsing, group classification, sampling, the
materialized-config round-trip, and the bad-config guards) need no data and no
Optuna study to fit. The two integration tests (the ``epoch_callback`` break and
the full study end-to-end) **skip** when the committed Pythia pseudodata
(``benchmark_data/cms_pseudodata.root``) is not present, mirroring
``test_tune_cms_fullsim.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

optuna = pytest.importorskip("optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.tune_cms_fullsim import optuna_search as osearch
from parnassus.torch_delphes.tune_cms_fullsim.training import fit_card_to_fullsim
from parnassus.torch_delphes.tune_cms_fullsim.runner import load_split_datasets
from parnassus.torch_delphes.tune_cms_fullsim.dataloader import DelphesDataLoader

PARAM_CONFIGS = Path(pc.__file__).resolve().parent / "param_configs"
SHIPPED_OPTUNA_CONFIG = PARAM_CONFIGS / "optuna_config.yaml"
PSEUDODATA_PATH = Path(__file__).parent / "benchmark_data" / "cms_pseudodata.root"

FRACTION_KEYS = {
    "HadronFractions.chad_logit",
    "HadronFractions.k0s_logit",
    "HadronFractions.lambda_logit",
    "HadronFractions.photon_logit",
    "HadronFractions.k0l_logit",
}

GOOD_SEARCH = {
    "global_batch_size": 256,
    "lr": {
        g: {"low": 1e-4, "high": 1e-2, "log": True, "init": 1e-3}
        for g in osearch.LR_GROUPS
    },
    "photon_merge_radius": {"low": 0.0, "high": 0.1, "init": 0.045},
}


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    """The committed pseudodata; skip when it isn't present in the tree."""
    if not PSEUDODATA_PATH.exists():
        pytest.skip("committed pseudodata file not available")
    return PSEUDODATA_PATH


@pytest.fixture
def card_defaults() -> dict[str, dict]:
    """Fresh card-default start values (the fitted scalars' starting point)."""
    return pc.card_default_config(CMSEnergyFlowDefault(debug=False, learnable=True))


def _write_cfg(tmp_path: Path, search: dict, constants: dict, name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    with p.open("w") as f:
        yaml.safe_dump({"search": search, "constants": constants}, f)
    return p


# ---------------------------------------------------------------------------
# Config parsing / coverage (no data)
# ---------------------------------------------------------------------------


def test_shipped_config_parses_and_merges_to_full_cover(card_defaults):
    """The shipped config freezes exactly the 5 fractions; the merge covers 68."""
    search, constants = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)
    assert {"lr", "global_batch_size", "photon_merge_radius"} <= set(search)
    # Every lr group carries an init (required): it is what the seed trial runs.
    for g in osearch.LR_GROUPS:
        assert g in search["lr"]
        spec = search["lr"][g]
        assert {"low", "high", "init"} <= set(spec), g
        assert spec["low"] <= spec["init"] <= spec["high"], g
    assert search["global_batch_size"] == 2048
    assert search["photon_merge_radius"] == {"low": 0.0, "high": 0.1, "init": 0.045}

    # constants: a SUBSET (the fractions), not a full cover.
    assert set(constants) == FRACTION_KEYS
    assert set(constants) < set(card_defaults)
    assert len(card_defaults) == 68

    # k0l/lambda/chad pinned this round; k0s/photon TPE-sampled with an init.
    assert constants["HadronFractions.chad_logit"] == {"value": 0.0}
    assert constants["HadronFractions.lambda_logit"] == {"value": 0.3}
    assert constants["HadronFractions.k0l_logit"] == {"value": 0.0}
    for key in ("HadronFractions.k0s_logit", "HadronFractions.photon_logit"):
        assert {"low", "high", "init"} <= set(constants[key])


def test_group_classification():
    """`_group_of` mirrors param_config.default_lr_scale's grouping."""
    cases = {
        "ECal.scale_module.scale_raw": "scale",
        "ChargedHadronMomentumSmearing.resolution_module.scale_raw": "scale",
        "ChargedHadronMomentumSmearing.resolution_module.a_raw": "resolution",
        "ECal.resolution_func.barrel_a": "resolution",
        "HCal.resolution_func.central_c_S": "resolution",
        "ChargedHadronTrackingEfficiency.eff_logits": "efficiency",
        "HadronFractions.chad_logit": "efficiency",
        "HadronFractions.photon_logit": "efficiency",
        "HadronFractions.k0l_logit": "efficiency",
        "MuonTrackingEfficiency.rate_raw": "efficiency",
    }
    for base, expected in cases.items():
        assert osearch._group_of(base) == expected, base


def test_group_of_rejects_unknown_parameter():
    """An unrecognized parameter name raises instead of silently becoming
    'resolution' (which would also make to_physical treat it as identity)."""
    with pytest.raises(ValueError, match="no known parameter transform"):
        osearch._group_of("ECal.threshold_module.energy_min")


def test_group_of_covers_every_card_scalar(card_defaults):
    """Every scalar on the card maps to one of the three lr groups."""
    groups = {osearch._group_of(k.split("[", 1)[0]) for k in card_defaults}
    assert groups == set(osearch.LR_GROUPS)


def test_load_search_config_rejects_bad(tmp_path: Path):
    """Malformed configs fail fast with a clear error."""
    base = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)[1]

    # Missing top-level sections -- including a pre-restructure 'parameters:'
    # file, whose {low, high} entries meant something else entirely.
    for raw in ({"search": GOOD_SEARCH}, {"constants": base},
                {"search": GOOD_SEARCH, "parameters": base}):
        p = tmp_path / "raw.yaml"
        with p.open("w") as f:
            yaml.safe_dump(raw, f)
        with pytest.raises(SystemExit, match="constants"):
            osearch.load_search_config(p)

    # A full-cover constants block would leave nothing to fit.
    full = {k: {"value": v["value"]} for k, v in pc.card_default_config(
        CMSEnergyFlowDefault(debug=False, learnable=True)
    ).items()}
    with pytest.raises(SystemExit, match="nothing would be fitted"):
        osearch.load_search_config(_write_cfg(tmp_path, GOOD_SEARCH, full))

    # A key the card does not have.
    with pytest.raises(SystemExit, match="does not have"):
        osearch.load_search_config(
            _write_cfg(tmp_path, GOOD_SEARCH, {**base, "Nope.not_a_param": {"value": 1.0}})
        )

    # Inverted range on a constant.
    with pytest.raises((SystemExit, ValueError)):
        osearch.load_search_config(
            _write_cfg(
                tmp_path,
                GOOD_SEARCH,
                {**base, "HadronFractions.k0s_logit": {"low": 0.4, "high": 0.2, "init": 0.3}},
            )
        )

    # Sampled entry without an init, and with an out-of-range init.
    with pytest.raises(SystemExit, match="init"):
        osearch.load_search_config(
            _write_cfg(
                tmp_path,
                GOOD_SEARCH,
                {**base, "HadronFractions.k0s_logit": {"low": 0.2, "high": 0.4}},
            )
        )
    with pytest.raises(SystemExit, match="outside its range"):
        osearch.load_search_config(
            _write_cfg(
                tmp_path,
                GOOD_SEARCH,
                {**base, "HadronFractions.k0s_logit": {"low": 0.2, "high": 0.4, "init": 0.9}},
            )
        )

    # Values outside a physical window fail the to_raw guard -- range and pin,
    # each of the three windowed logits.
    for key, spec in (
        ("HadronFractions.k0s_logit", {"low": 0.15, "high": 0.85, "init": 0.3}),  # window 0.1-0.5
        ("HadronFractions.photon_logit", {"low": 0.5, "high": 0.9, "init": 0.85}),  # 0.8-1.0
        ("HadronFractions.k0l_logit", {"low": 0.1, "high": 0.5, "init": 0.2}),  # 0.0-0.4
        ("HadronFractions.photon_logit", {"value": 0.5}),
    ):
        with pytest.raises(SystemExit, match="invalid"):
            osearch.load_search_config(_write_cfg(tmp_path, GOOD_SEARCH, {**base, key: spec}))

    # An lr group without `init` is rejected: it is what the seed trial runs, so
    # omitting it silently hands trial 0 back to the sampler.
    no_init = {g: {"low": 1e-4, "high": 1e-2, "log": True} for g in osearch.LR_GROUPS}
    with pytest.raises(SystemExit, match="init"):
        osearch.load_search_config(
            _write_cfg(tmp_path, {**GOOD_SEARCH, "lr": no_init}, base)
        )

    # Bad search block: missing lr group, non-integer batch, radius without init.
    for search in (
        {**GOOD_SEARCH, "lr": {"scale": {"low": 1e-4, "high": 1e-2, "init": 1e-3}}},
        {**GOOD_SEARCH, "global_batch_size": 0},
        {**GOOD_SEARCH, "global_batch_size": "big"},
        {**GOOD_SEARCH, "photon_merge_radius": {"low": 0.0, "high": 0.1}},
        {**GOOD_SEARCH, "photon_merge_radius": {"low": -0.1, "high": 0.1, "init": 0.0}},
        {**GOOD_SEARCH, "photon_merge_radius": {"choices": [0.045]}},
    ):
        with pytest.raises(SystemExit):
            osearch.load_search_config(_write_cfg(tmp_path, search, base))


def test_global_batch_size_semantics(tmp_path: Path):
    """The batch is GLOBAL, so the per-rank slice shrinks with the GPU count and
    the number of Adam updates per epoch stays fixed.

    Regression: read as per-rank, `batch_size: 2048` silently became a global
    batch of 8192 on 4 GPUs -- ~10x fewer optimizer updates per epoch than the
    same config on one GPU, which cost the mee round 0.27 -> 0.43 in val loss.
    """
    base = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)[1]

    # An old per-rank config fails loudly rather than changing meaning silently.
    stale = {k: v for k, v in GOOD_SEARCH.items() if k != "global_batch_size"}
    stale["batch_size"] = 2048
    with pytest.raises(SystemExit, match="renamed"):
        osearch.load_search_config(_write_cfg(tmp_path, stale, base))

    # The invariant the fix buys: updates/epoch is independent of world_size.
    n_train, global_batch = 70000, 512
    per_epoch = {
        w: math.ceil(n_train / w / (global_batch // w)) for w in (1, 2, 4, 8)
    }
    assert len(set(per_epoch.values())) == 1, per_epoch

    # Indivisible splits are rejected -- unequal per-rank batches would desync
    # the ranks' update counts.
    assert 512 % 3 != 0


def test_chad_bypass_zone_guard(tmp_path: Path):
    """chad_logit may not touch the ECal-rescale artifact zone (1e-4, 5e-3).

    The old trainable-logit window guard blocked it incidentally; constants are
    exempt from that window, so the zone needs its own rule.
    """
    base = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)[1]
    key = "HadronFractions.chad_logit"
    for bad in (
        {"value": 1.0e-3},  # pinned inside the zone
        {"low": 1.0e-3, "high": 0.5, "init": 0.1},  # range starting inside it
        {"low": 1.0e-5, "high": 0.5, "init": 0.1},  # range STRADDLING it
    ):
        with pytest.raises(SystemExit, match="sub-threshold"):
            osearch.load_search_config(_write_cfg(tmp_path, GOOD_SEARCH, {**base, key: bad}))
    # Legal: below the bypass cutoff, or entirely in the active region.
    for ok in ({"value": 0.0}, {"value": 1.0e-4}, {"low": 5.0e-3, "high": 0.5, "init": 0.1}):
        osearch.load_search_config(_write_cfg(tmp_path, GOOD_SEARCH, {**base, key: ok}))


def test_boundary_constants_are_legal(tmp_path: Path):
    """A constant may sit ON a window boundary (photon 1.0, k0l 0.0, k0s 0.5):
    the trainable-logit guard protects gradient flow, and constants never move.

    Regression: the pre-restructure validator rejected these outright, and the
    materialized config must still round-trip through load_param_config.
    """
    base = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)[1]
    constants = {
        **base,
        "HadronFractions.photon_logit": {"value": 1.0},
        "HadronFractions.k0l_logit": {"low": 0.0, "high": 0.4, "init": 0.0},
        "HadronFractions.k0s_logit": {"low": 0.1, "high": 0.5, "init": 0.5},
    }
    search, parsed = osearch.load_search_config(_write_cfg(tmp_path, GOOD_SEARCH, constants))
    defaults = pc.card_default_config(CMSEnergyFlowDefault(debug=False, learnable=True))

    trial = optuna.trial.FixedTrial(
        {
            **{f"lr[{g}]": 1e-3 for g in osearch.LR_GROUPS},
            "HadronFractions.k0l_logit": 0.0,
            "HadronFractions.k0s_logit": 0.5,
        }
    )
    flat_cfg, _ = osearch.sample_trial(trial, search, parsed, defaults)
    assert flat_cfg["HadronFractions.photon_logit"]["value"] == 1.0
    assert flat_cfg["HadronFractions.k0l_logit"]["value"] == 0.0

    mat = tmp_path / "materialized_config.yaml"
    osearch._dump_flat_config(flat_cfg, mat)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    pc.apply_param_config(card, pc.load_param_config(mat))  # no guard fires


# ---------------------------------------------------------------------------
# Sampling + materialized-config round-trip (no data)
# ---------------------------------------------------------------------------


def test_sample_trial_produces_valid_config(tmp_path: Path, card_defaults):
    """Sampled configs cover the card, freeze exactly the constants, and yield
    one Adam group per lr group at the sampled ABSOLUTE learning rate."""
    search, constants = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)

    captured: dict = {}

    def objective(trial: optuna.Trial) -> float:
        flat_cfg, group_lr = osearch.sample_trial(trial, search, constants, card_defaults)
        # Full cover, and listed <-> frozen exactly.
        assert set(flat_cfg) == set(card_defaults)
        assert {k for k, s in flat_cfg.items() if not s["trainable"]} == set(constants)
        assert set(group_lr) == set(osearch.LR_GROUPS)

        # Unlisted params start at their card default and carry their group's lr.
        for key, spec in flat_cfg.items():
            if key in constants:
                continue
            assert spec["value"] == pytest.approx(card_defaults[key]["value"]), key
            assert spec["lr_scale"] == group_lr[osearch._group_of(key.split("[", 1)[0])], key

        card = CMSEnergyFlowDefault(debug=False, learnable=True)
        pc.apply_param_config(card, flat_cfg)  # raises on a guard/coverage error
        # lr_scale holds the absolute lr, so global_lr is 1.
        params_to_train, param_groups = pc.select_trainable(card, flat_cfg, global_lr=1.0)
        assert params_to_train
        assert {g["lr"] for g in param_groups} == set(group_lr.values())

        captured["flat_cfg"] = flat_cfg
        return 0.0

    # Multiple trials exercise many sampled points -> no to_raw guard violation.
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=5)

    # Materialized config round-trips through the normal loader.
    mat = tmp_path / "materialized_config.yaml"
    osearch._dump_flat_config(captured["flat_cfg"], mat)
    reloaded = pc.load_param_config(mat)
    assert set(reloaded) == set(captured["flat_cfg"])
    for key, spec in reloaded.items():
        assert spec["value"] == pytest.approx(captured["flat_cfg"][key]["value"])
        assert spec["trainable"] == captured["flat_cfg"][key]["trainable"]
        assert spec["lr_scale"] == pytest.approx(captured["flat_cfg"][key]["lr_scale"])


def test_sample_photon_merge_radius():
    """Continuous per-trial draw; an exact 0 means merger OFF (None radius)."""
    search = {"photon_merge_radius": {"low": 0.0, "high": 0.1, "init": 0.045}}
    t = optuna.trial.FixedTrial({"photon_merge_radius": 0.045})
    assert osearch.sample_photon_merge_radius(t, search) == 0.045
    assert t.params["photon_merge_radius"] == 0.045  # registered as a trial param
    # PhotonClusterMerger rejects a non-positive radius, so 0 maps to "no merger".
    t0 = optuna.trial.FixedTrial({"photon_merge_radius": 0.0})
    assert osearch.sample_photon_merge_radius(t0, search) is None


# ---------------------------------------------------------------------------
# Seed trial + --init-config overrides (no data)
# ---------------------------------------------------------------------------


def test_seed_trial_pending_tracks_seed_trial_state():
    """The re-enqueue decision follows the seed trial's lifecycle: pending on a
    fresh study, delivered while the seed is WAITING or once it COMPLETEs, and
    pending again when the seed FAILed mid-fit (optuna never re-runs FAIL trials,
    so without this the init values would silently never be evaluated)."""
    attrs = {"warm_start_config": "optuna_config.yaml#init"}

    # Fresh study -> pending.
    study = optuna.create_study()
    assert osearch.seed_trial_pending(study.trials)

    # Enqueued (WAITING) -> delivered; evaluated (COMPLETE) -> still delivered.
    study.enqueue_trial({"x": 0.7}, user_attrs=attrs)
    assert not osearch.seed_trial_pending(study.trials)
    study.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert not osearch.seed_trial_pending(study.trials)

    # Seed crashed mid-fit (FAIL) -> pending again.
    crashed = optuna.create_study()
    crashed.enqueue_trial({"x": 0.7}, user_attrs=attrs)

    def boom(trial: optuna.Trial) -> float:
        trial.suggest_float("x", 0.0, 1.0)
        raise RuntimeError("simulated mid-fit crash")

    with pytest.raises(RuntimeError):
        crashed.optimize(boom, n_trials=1)
    assert crashed.trials[0].state == optuna.trial.TrialState.FAIL
    assert osearch.seed_trial_pending(crashed.trials)

    # Sampler-driven trials without the marker never count as a seed.
    unmarked = optuna.create_study()
    unmarked.optimize(lambda t: t.suggest_float("x", 0.0, 1.0), n_trials=1)
    assert osearch.seed_trial_pending(unmarked.trials)


def test_seed_trial_uses_init_values(card_defaults):
    """The enqueued seed is FULLY pinned -- `init:` lrs, constants and radius --
    so it is an exactly reproducible baseline; the next trial is sampler-driven.

    Regression: the lrs used to be sampler-filled, which made trial 0 the
    believed-truth card crossed with an arbitrary optimizer draw (11th of 18 in
    the mee round) rather than a baseline worth comparing against.
    """
    search, constants = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)
    enqueue = {f"lr[{g}]": float(search["lr"][g]["init"]) for g in osearch.LR_GROUPS}
    enqueue |= {k: float(s["init"]) for k, s in constants.items() if "value" not in s}
    enqueue["photon_merge_radius"] = float(search["photon_merge_radius"]["init"])

    configs: list[dict] = []
    radii: list[float | None] = []
    lrs: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        cfg, group_lr = osearch.sample_trial(trial, search, constants, card_defaults)
        configs.append(cfg)
        lrs.append(group_lr)
        radii.append(osearch.sample_photon_merge_radius(trial, search))
        return 0.0

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(enqueue)
    study.optimize(objective, n_trials=2)

    for key, value in enqueue.items():
        assert study.trials[0].params[key] == pytest.approx(value), key
    assert radii[0] == pytest.approx(search["photon_merge_radius"]["init"])
    # The lrs reach the fit, not just the trial params.
    for g in osearch.LR_GROUPS:
        assert lrs[0][g] == pytest.approx(search["lr"][g]["init"]), g
    # ...and trial 1's are sampler-driven, so the pin is the seed's alone.
    assert any(lrs[1][g] != pytest.approx(lrs[0][g]) for g in osearch.LR_GROUPS)

    # Trial 0 IS the CMS-default baseline card: every fitted scalar at its
    # constructor default, every constant at its init.
    for key, spec in configs[0].items():
        if key in constants and "value" not in constants[key]:
            assert spec["value"] == pytest.approx(constants[key]["init"]), key
        elif key in constants:
            assert spec["value"] == pytest.approx(constants[key]["value"]), key
        else:
            assert spec["value"] == pytest.approx(card_defaults[key]["value"]), key

    # Trial 1 is sampler-driven, not a copy of the seed.
    assert any(
        abs(configs[1][k]["value"] - configs[0][k]["value"]) > 1e-12
        for k in enqueue
        if k in configs[0]
    )


def test_apply_init_overrides(card_defaults):
    """--init-config replaces the START value of fitted scalars only, and rejects
    a value that would leave a FITTED logit with no gradient."""
    fitted = {k: v for k, v in card_defaults.items() if not k.startswith("HadronFractions.")}
    key = "ChargedHadronTrackingEfficiency.eff_logits[0]"
    assert fitted[key]["value"] == pytest.approx(0.7)

    changed = osearch.apply_init_overrides(
        fitted,
        {
            key: {"value": 0.42, "trainable": False, "lr_scale": 1.0},
            "HadronFractions.k0s_logit": {"value": 0.45, "trainable": False, "lr_scale": 1.0},
            "Not.on.the.card": {"value": 1.0, "trainable": False, "lr_scale": 1.0},
        },
        "init.yaml",
    )
    assert changed == [key]  # the constant and the unknown key are ignored
    assert fitted[key]["value"] == pytest.approx(0.42)

    # A saturated init for a FITTED logit is rejected (it could never move).
    with pytest.raises(SystemExit, match="trainable-logit window"):
        osearch.apply_init_overrides(
            fitted, {key: {"value": 0.999, "trainable": False, "lr_scale": 1.0}}, "init.yaml"
        )


# ---------------------------------------------------------------------------
# epoch_callback break (needs data)
# ---------------------------------------------------------------------------


def test_epoch_callback_breaks_loop(fixture_root: Path):
    """`epoch_callback` returning True breaks the fit cleanly with history intact."""
    device = torch.device("cpu")
    train_ds, val_ds = load_split_datasets(fixture_root, n_events=24, device=device)
    train_dl = DelphesDataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DelphesDataLoader(val_ds, batch_size=8, shuffle=False)

    torch.manual_seed(0)
    card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    cfg = pc.card_default_config(card)
    cfg["ChargedHadronMomentumSmearing.resolution_module.scale_raw[0]"]["trainable"] = True
    pc.apply_param_config(card, cfg)
    _, param_groups = pc.select_trainable(card, cfg, global_lr=1e-2)

    seen: list[int] = []

    def stop_after_one(step: int, val_loss: float) -> bool:
        seen.append(step)
        return step >= 1  # break once step 1 has been processed

    history = fit_card_to_fullsim(
        card,
        train_dl,
        val_dl,
        param_groups=param_groups,
        n_steps=10,
        log_every=0,
        early_stopping_patience=None,
        lr_scheduler_patience=None,
        epoch_callback=stop_after_one,
    )
    # Broke at step 1 -> epochs 0 and 1 only, and every list stays index-aligned.
    assert history["step"] == [0, 1]
    assert len(history["loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert seen == [0, 1]


# ---------------------------------------------------------------------------
# End-to-end study (needs data)
# ---------------------------------------------------------------------------


def test_optuna_search_end_to_end(fixture_root: Path, tmp_path: Path, monkeypatch):
    """A tiny study writes per-round artifacts and a best history the plot pipeline accepts."""
    from parnassus.torch_delphes.tune_cms_fullsim.plot_fit_results import (
        _load_history,
        _load_init_snapshot,
    )

    out_base = tmp_path / "fit_results"
    history_path = out_base / "all_v2.json"
    argv = [
        "optuna_search",
        "--root-file", str(fixture_root),
        "--optuna-config", str(SHIPPED_OPTUNA_CONFIG),
        "--n-trials", "2",
        "--n-steps", "2",
        "--n-events", "80",
        "--plot-every", "1",
        "--output-base", str(out_base),
        "--history-path", str(history_path),
        "--loss", "wasserstein_1d",
        "--pid-weighting", "sqrt_fraction",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    osearch.main()

    # Per-round artifacts (requirement 4).
    for i in range(2):
        rd = out_base / f"round_{i}"
        assert (rd / "materialized_config.yaml").exists(), rd
        assert (rd / "history.json").exists(), rd
        plots = list((rd / "intermediate_plots").glob("*.pdf"))
        assert plots, f"no intermediate plots in {rd}"
        # The materialized config stays a full-cover, loadable param config --
        # plot_fit_results needs it for the honest before-fit baseline.
        assert len(pc.load_param_config(rd / "materialized_config.yaml")) == 68

    # Trial 0 is the seed: constants at their `init:`, fitted scalars at the card
    # defaults, and every constant frozen.
    round0 = pc.load_param_config(out_base / "round_0" / "materialized_config.yaml")
    defaults = pc.card_default_config(CMSEnergyFlowDefault(debug=False, learnable=True))
    _, constants = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)
    assert round0["ChargedHadronTrackingEfficiency.eff_logits[0]"]["value"] == pytest.approx(0.7)
    assert round0["ChargedHadronTrackingEfficiency.eff_logits[0]"]["trainable"] is True
    for key in constants:
        assert round0[key]["trainable"] is False, key
    assert round0["HadronFractions.chad_logit"]["value"] == pytest.approx(0.0)
    assert round0["HadronFractions.photon_logit"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert round0["HadronFractions.k0l_logit"]["value"] == pytest.approx(0.0, abs=1e-6)
    assert round0["HadronFractions.k0s_logit"]["value"] == pytest.approx(0.3)
    # Fitted scalars all start at their card default.
    for key, spec in round0.items():
        if key not in constants:
            assert spec["value"] == pytest.approx(defaults[key]["value"], rel=1e-6), key

    # Canonical best history exists and is accepted by the plot pipeline.
    assert history_path.exists()
    history = _load_history(history_path)
    assert history["best_result"].get("parameters"), "best_result has no parameter snapshot"
    assert history["metadata"].get("trial_number") in (0, 1)
    # Absolute per-group lrs replace the old lr / lr_scale pair.
    assert set(history["metadata"]["lr_groups"]) == set(osearch.LR_GROUPS)
    assert history["metadata"]["global_lr"] == 1.0
    # Single-rank run: the per-rank batch IS the global batch, and the recorded
    # update count is what makes two studies comparable.
    assert history["metadata"]["global_batch_size"] == 2048
    assert history["metadata"]["batch_size"] == 2048
    assert history["metadata"]["updates_per_epoch"] >= 1
    # No --mode given -> fullsim with the acceptance cuts on (pins the default).
    assert history["metadata"]["mode"] == "fullsim"
    assert history["metadata"]["reco_pt_cut"] == pytest.approx(1.0)
    assert history["metadata"]["chad_truncation"] is True

    # The before-fit baseline resolves to the best round's materialized config
    # (NOT constructor defaults) -- this is what makes plot_fit_results honest.
    snapshot, source = _load_init_snapshot(None, history)
    assert snapshot is not None and "materialized_config.yaml" in source

    # Best score equals the min val_loss over the best round's epochs.
    best_val = history["best_result"]["val_loss"]
    assert best_val == pytest.approx(min(history["val_loss"]))


def test_optuna_search_radius_is_per_trial_and_reaches_the_card(
    fixture_root: Path, tmp_path: Path, monkeypatch
):
    """Each trial's sampled radius lands in its round metadata AND is exactly the
    radius its card was BUILT with (guards against the sampled value reaching the
    metadata but not the physics, or vice versa). The seed trial runs `init`."""
    import json

    with open(SHIPPED_OPTUNA_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg["search"]["photon_merge_radius"] = {"low": 0.02, "high": 0.08, "init": 0.045}

    # Spy on merger construction: one entry per trial whose card got a merger.
    built: list[float] = []
    real_merger = osearch.PhotonClusterMerger

    class SpyMerger(real_merger):
        def __init__(self, merge_radius):
            built.append(merge_radius)
            super().__init__(merge_radius)

    monkeypatch.setattr(osearch, "PhotonClusterMerger", SpyMerger)
    cfg_path = tmp_path / "optuna_config_radius.yaml"
    with cfg_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    out_base = tmp_path / "fit_results"
    argv = [
        "optuna_search",
        "--root-file", str(fixture_root),
        "--optuna-config", str(cfg_path),
        "--n-trials", "3",
        "--n-steps", "2",
        "--n-events", "80",
        "--plot-every", "1",
        "--output-base", str(out_base),
        "--history-path", str(out_base / "all.json"),
        "--loss", "wasserstein_1d",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    osearch.main()

    radii = []
    for i in range(3):
        with (out_base / f"round_{i}" / "history.json").open() as f:
            meta = json.load(f)["metadata"]
        assert 0.02 <= meta["photon_merge_radius"] <= 0.08, meta["photon_merge_radius"]
        radii.append(meta["photon_merge_radius"])
    # The seed trial runs the init radius; later trials are sampler-driven.
    assert radii[0] == pytest.approx(0.045)
    assert len(set(radii)) > 1, "every trial drew the same radius"
    # The card each trial actually fitted used exactly the radius its metadata
    # records: one merger per trial, in trial order, same values.
    assert built == pytest.approx(radii)


def test_optuna_search_delphes_mode_disables_merger(
    fixture_root: Path, tmp_path: Path, monkeypatch
):
    """--mode delphes: no PhotonClusterMerger is ever built (seed trial included),
    the radius is not a search dimension, every round records mode/cuts/radius
    off, and a study cannot be resumed under the other mode.
    """
    import json

    built: list[float] = []
    real_merger = osearch.PhotonClusterMerger

    class SpyMerger(real_merger):
        def __init__(self, merge_radius):
            built.append(merge_radius)
            super().__init__(merge_radius)

    monkeypatch.setattr(osearch, "PhotonClusterMerger", SpyMerger)

    out_base = tmp_path / "fit_results"
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    argv = [
        "optuna_search",
        "--root-file", str(fixture_root),
        "--optuna-config", str(SHIPPED_OPTUNA_CONFIG),
        "--mode", "delphes",
        "--n-trials", "2",
        "--n-steps", "2",
        "--n-events", "80",
        "--plot-every", "1",
        "--output-base", str(out_base),
        "--history-path", str(out_base / "all.json"),
        "--storage", storage,
        "--study-name", "delphes_t",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    osearch.main()

    assert built == [], f"merger built in delphes mode with radii {built}"
    for i in range(2):
        with (out_base / f"round_{i}" / "history.json").open() as f:
            meta = json.load(f)["metadata"]
        assert meta["mode"] == "delphes"
        assert meta["photon_merge_radius"] is None
        assert meta["truth_pt_cut"] is None
        assert meta["reco_pt_cut"] is None
        assert meta["eta_cut"] is None
        assert meta["chad_truncation"] is False

    # The radius is not a TPE dimension: absent from every trial, and the pinned
    # seed trial carries only the group lrs + the sampled constants.
    study = optuna.load_study(study_name="delphes_t", storage=storage)
    assert study.user_attrs["mode"] == "delphes"
    assert all("photon_merge_radius" not in t.params for t in study.trials)
    _search, constants = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)
    sampled = {k for k, s in constants.items() if "value" not in s}
    assert set(study.trials[0].params) == {f"lr[{g}]" for g in osearch.LR_GROUPS} | sampled

    # One study == one mode: resuming it as fullsim is refused.
    monkeypatch.setattr(sys, "argv", [a if a != "delphes" else "fullsim" for a in argv])
    with pytest.raises(SystemExit, match="--mode delphes"):
        osearch.main()
