"""Tests for the Optuna hyperparameter search (``tune_cms_fullsim.optuna_search``).

The pure-logic tests (config parsing, group classification, sampling, the
materialized-config round-trip, and the bad-config guards) need no data and no
Optuna study to fit. The two integration tests (the ``epoch_callback`` break and
the full study end-to-end) **skip** when the committed Pythia pseudodata
(``benchmark_data/cms_pseudodata.root``) is not present, mirroring
``test_tune_cms_fullsim.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

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


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    """The committed pseudodata; skip when it isn't present in the tree."""
    if not PSEUDODATA_PATH.exists():
        pytest.skip("committed pseudodata file not available")
    return PSEUDODATA_PATH


# ---------------------------------------------------------------------------
# Config parsing / coverage (no data)
# ---------------------------------------------------------------------------


def test_shipped_config_parses_and_covers_card():
    """The shipped optuna_config.yaml parses and exactly covers the 70 card scalars."""
    search, parameters = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)
    assert {"lr", "batch_size", "lr_scale"} <= set(search)
    for g in osearch.LR_SCALE_GROUPS:
        assert g in search["lr_scale"]

    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    card_keys = set()
    for name, p in card.named_parameters():
        card_keys |= set(pc._scalar_keys(name, p))
    assert set(parameters) == card_keys
    assert len(parameters) == 70


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
        "MuonTrackingEfficiency.rate_raw": "efficiency",
        "ECal.threshold_module.energy_min_raw": "efficiency",
        "HCal.threshold_module.energy_sig_min_raw": "efficiency",
    }
    for base, expected in cases.items():
        assert osearch._group_of(base) == expected, base


def test_load_search_config_rejects_bad(tmp_path: Path):
    """Malformed configs fail fast with a clear error."""
    import yaml

    base = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)[1]

    def _write(cfg: dict) -> Path:
        p = tmp_path / "bad.yaml"
        with p.open("w") as f:
            yaml.safe_dump(cfg, f)
        return p

    # Missing top-level sections.
    with pytest.raises(SystemExit):
        osearch.load_search_config(_write({"parameters": base}))

    good_search = {
        "lr": {"low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"choices": [256]},
        "lr_scale": {g: {"low": 0.1, "high": 10.0, "log": True} for g in osearch.LR_SCALE_GROUPS},
    }

    # Inverted range (low >= high) on a parameter.
    bad_params = dict(base)
    bad_params["ECal.scale_module.scale_raw[0]"] = {"low": 1.2, "high": 0.8}
    with pytest.raises((SystemExit, ValueError)):
        osearch.load_search_config(_write({"search": good_search, "parameters": bad_params}))

    # Init range outside the trainable-logit guard (0.1, 0.9).
    bad_params2 = dict(base)
    bad_params2["ChargedHadronTrackingEfficiency.eff_logits[0]"] = {"low": 0.92, "high": 0.99}
    with pytest.raises(SystemExit):
        osearch.load_search_config(_write({"search": good_search, "parameters": bad_params2}))


# ---------------------------------------------------------------------------
# Sampling + materialized-config round-trip (no data)
# ---------------------------------------------------------------------------


def test_sample_trial_produces_valid_config(tmp_path: Path):
    """Sampled configs apply to the card and yield exactly one Adam group per lr_scale group."""
    search, parameters = osearch.load_search_config(SHIPPED_OPTUNA_CONFIG)

    captured: dict = {}

    def objective(trial: optuna.Trial) -> float:
        flat_cfg, lr, batch_size, group_lr_scale = osearch.sample_trial(trial, search, parameters)
        assert set(flat_cfg) == set(parameters)
        # A param is trainable iff its search spec is a range (not a {value: ...} pin,
        # e.g. the calo energy_sig_min thresholds, which are pinned at truth).
        for key, spec in flat_cfg.items():
            assert spec["trainable"] == ("value" not in parameters[key]), key
        assert batch_size in search["batch_size"]["choices"]
        assert set(group_lr_scale) == set(osearch.LR_SCALE_GROUPS)

        card = CMSEnergyFlowDefault(debug=False, learnable=True)
        pc.apply_param_config(card, flat_cfg)  # raises on a guard/coverage error
        params_to_train, param_groups = pc.select_trainable(card, flat_cfg, global_lr=lr)
        assert params_to_train
        # One distinct effective lr per group (lr * the 3 group lr_scales).
        assert len({g["lr"] for g in param_groups}) == len(osearch.LR_SCALE_GROUPS)

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
    cfg = pc.load_param_config(PARAM_CONFIGS / "debug_all_params_v2.yaml")
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
        # The materialized config is a normal, loadable param config.
        assert len(pc.load_param_config(rd / "materialized_config.yaml")) == 70

    # Canonical best history exists and is accepted by the plot pipeline.
    assert history_path.exists()
    history = _load_history(history_path)
    assert history["best_result"].get("parameters"), "best_result has no parameter snapshot"
    assert history["metadata"].get("trial_number") in (0, 1)

    # The before-fit baseline resolves to the best round's materialized config
    # (NOT constructor defaults) -- this is what makes plot_fit_results honest.
    snapshot, source = _load_init_snapshot(None, history)
    assert snapshot is not None and "materialized_config.yaml" in source

    # Best score equals the min val_loss over the best round's epochs.
    best_val = history["best_result"]["val_loss"]
    assert best_val == pytest.approx(min(history["val_loss"]))
