"""Tests for the ``tune_cms_fullsim`` harness.

These tests cover the pipeline end-to-end using the synthetic fixture writer
(the real Zenodo sample is not in the test tree): generate a cms-flow-format
ROOT file, load it, build the padded truth particle tensor and the target
observable dict, run a forward + backward step on the learnable CMS card, and
run the Adam fit loop for a few steps.

Real-sample smoke tests are the user's responsibility (download the file and
run ``python -m parnassus.torch_delphes.tune_cms_fullsim --root-file ...``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from parnassus.data.particle_io import N_FEATURES, ColumnMap
from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.tune_cms_fullsim import (
    OBSERVABLES,
    PFLOW_BRANCHES,
    TRUTH_BRANCHES,
    fit_card_to_fullsim,
    load_cms_flow_root,
    write_synthetic_fixture,
)
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    load_pflow_targets,
    load_pflow_targets_from_tensor,
    load_truth_events,
    restore_event_format,
    split_pflow_targets,
    split_truth_objects,
)
from parnassus.torch_delphes.tune_cms_fullsim.dataloader import (
    DelphesDataLoader,
    DelphesDataSet,
)
from parnassus.torch_delphes.tune_cms_fullsim.config import COUNT_TERM_KEYS
from parnassus.torch_delphes.tune_cms_fullsim.loss import per_event_wasserstein_loss


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped synthetic ROOT fixture so we write it once per run."""
    path = tmp_path_factory.mktemp("fullsim_fixture") / "fixture.root"
    write_synthetic_fixture(path, n_events=30, particles_per_event=50, seed=0)
    return path


def _make_dataloaders(
    arrays: dict, device: torch.device, batch_size: int = 8, seed: int = 0
) -> tuple[DelphesDataLoader, DelphesDataLoader]:
    """Build train/val dataloaders from loaded ROOT arrays (mirrors the CLI)."""
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)
    tr_truth, va_truth = split_truth_objects(truth, train_fraction=0.8, seed=seed)
    tr_tgt, va_tgt = split_pflow_targets(target, train_fraction=0.8, seed=seed)
    tr_ds = DelphesDataSet(tr_truth, tr_tgt, device=device)
    va_ds = DelphesDataSet(va_truth, va_tgt, device=device)
    return (
        DelphesDataLoader(tr_ds, batch_size=batch_size, shuffle=True),
        DelphesDataLoader(va_ds, batch_size=batch_size, shuffle=False),
    )


def _trainable_config(card: CMSEnergyFlowDefault, tmp_path: Path, prefixes: list[str]) -> dict:
    """Dump ``card``'s defaults to a config, mark the matching scalars trainable,
    apply it back, and return the flat config (for ``select_trainable``)."""
    cfg_path = tmp_path / "cfg.yaml"
    pc.dump_param_config(card, cfg_path)
    cfg = pc.load_param_config(cfg_path)
    matched = 0
    for key, spec in cfg.items():
        if any(key.startswith(p) for p in prefixes):
            spec["trainable"] = True
            matched += 1
    assert matched > 0, f"no config keys matched {prefixes}"
    pc.apply_param_config(card, cfg)
    return cfg


# ---------------------------------------------------------------------------
# ROOT I/O and observable construction
# ---------------------------------------------------------------------------


def test_fixture_has_expected_branches(fixture_root: Path):
    """The fixture is readable and contains every branch the script reads."""
    import uproot

    with uproot.open(str(fixture_root)) as f:
        keys = set(f["event_tree"].keys())
    for branch in TRUTH_BRANCHES + PFLOW_BRANCHES:
        assert branch in keys, f"missing branch {branch} in fixture"


def test_load_cms_flow_root_roundtrip(fixture_root: Path):
    """``load_cms_flow_root`` returns dense per-event numpy arrays."""
    arrays = load_cms_flow_root(fixture_root, n_events=30)
    for branch in TRUTH_BRANCHES + PFLOW_BRANCHES:
        assert branch in arrays
        assert len(arrays[branch]) == 30
    assert arrays["truth_pt"][0].ndim == 1
    assert arrays["pflow_pt"][0].ndim == 1


def test_load_truth_events_shapes(fixture_root: Path):
    """``load_truth_events`` pads to ``(n_events, max_n_particles, N_FEATURES)``."""
    arrays = load_cms_flow_root(fixture_root, n_events=10)
    truth = load_truth_events(arrays)
    assert truth.ndim == 3
    assert truth.shape[0] == 10
    assert truth.shape[2] == N_FEATURES
    # Real (non-padded) rows have positive mass, finite pt, and E^2 ~ p^2 + m^2.
    real = truth[torch.any(truth != 0, dim=-1)]
    assert real.shape[0] > 0
    assert float(real[:, ColumnMap.MASS].min()) > 0
    assert torch.isfinite(real[:, ColumnMap.PT]).all()
    p_sq = real[:, ColumnMap.PX] ** 2 + real[:, ColumnMap.PY] ** 2 + real[:, ColumnMap.PZ] ** 2
    e_sq = real[:, ColumnMap.E] ** 2
    m_sq = real[:, ColumnMap.MASS] ** 2
    assert torch.allclose(e_sq, p_sq + m_sq, atol=1e-6)


def test_load_pflow_targets_shapes(fixture_root: Path):
    """Target observables have the right shapes and are all finite."""
    arrays = load_cms_flow_root(fixture_root, n_events=15)
    tgt = load_pflow_targets(arrays)
    for key in OBSERVABLES:
        assert key in tgt, f"missing observable {key}"
    # Per-particle observables are 2-D (n_events, max_n_particles).
    assert tgt["pt"].ndim == 2 and tgt["pt"].shape[0] == 15
    # Per-event observables are 1-D, length n_events.
    assert tgt["multiplicity"].shape == (15,)
    assert tgt["ht"].shape == (15,)
    # Per-species region-count targets: (n_events, n_regions), non-negative integers.
    assert tgt["chad_region_counts"].shape == (15, 4)
    assert tgt["electron_region_counts"].shape == (15, 6)
    assert tgt["muon_region_counts"].shape == (15, 6)
    for key in ("chad_region_counts", "electron_region_counts", "muon_region_counts"):
        assert (tgt[key] >= 0).all()
    for name, v in tgt.items():
        assert torch.isfinite(v).all(), f"non-finite in target '{name}'"


def test_load_pflow_targets_region_counts_cross_check(fixture_root: Path):
    """One reco bin's electron count, recomputed by hand from the raw pflow branches,
    matches the loader's ``electron_region_counts`` (barrel low-pt = region 0)."""
    arrays = load_cms_flow_root(fixture_root, n_events=15)
    tgt = load_pflow_targets(arrays)
    from parnassus.utils import class_to_pid_vectorized

    total_manual = 0
    for i in range(15):
        pt = np.asarray(arrays["pflow_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["pflow_eta"][i], dtype=np.float64)
        cls = np.asarray(arrays["pflow_class"][i], dtype=np.int64)
        if pt.size == 0:
            continue
        abs_pid = np.abs(class_to_pid_vectorized(cls))
        is_e = abs_pid == 11
        barrel_low = (np.abs(eta) <= 1.5) & (pt > 0.1) & (pt <= 1.0)
        total_manual += int(np.sum(is_e & barrel_low))
    assert int(tgt["electron_region_counts"][:, 0].sum()) == total_manual


# ---------------------------------------------------------------------------
# Loss and fit loop
# ---------------------------------------------------------------------------


def test_one_step_gradient_is_finite(fixture_root: Path):
    """A single forward + backward step produces a finite loss and finite
    gradients on every parameter that the loss reaches."""
    arrays = load_cms_flow_root(fixture_root, n_events=20)
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    torch.manual_seed(7)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)

    mask = torch.any(truth != 0, dim=-1)
    out = card(truth[mask])
    eflow = restore_event_format(out["EFlowObject"], mask)
    pred = load_pflow_targets_from_tensor(eflow)
    # Inject the per-species expected counts so the loss exercises the count terms
    # (the eff_logits' only gradient path), exactly as training.py does.
    for out_key, pred_key, _tgt_key in COUNT_TERM_KEYS:
        pred[pred_key] = out[out_key]

    loss = per_event_wasserstein_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in card.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    for name, p in card.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"

    # The count terms must reach the tracking-efficiency logits (their only path).
    # The synthetic fixture's uniform class mix populates the lepton low/mid bins.
    params = dict(card.named_parameters())
    for mod in (
        "ChargedHadronTrackingEfficiency",
        "ElectronTrackingEfficiency",
        "MuonTrackingEfficiency",
    ):
        g = params[f"{mod}.eff_logits"].grad
        assert g is not None and float(g.abs().sum()) > 0, f"{mod}.eff_logits got no count grad"


def test_fit_card_to_fullsim_runs(fixture_root: Path, tmp_path: Path):
    """The fit loop runs a handful of steps without errors and returns a
    history dict of the right shape. We do NOT assert convergence on the
    synthetic fixture (the loss is dominated by stochastic smearing/Gumbel
    noise at tiny batch sizes)."""
    arrays = load_cms_flow_root(fixture_root, n_events=24)
    device = torch.device("cpu")
    train_dl, val_dl = _make_dataloaders(arrays, device, batch_size=8, seed=0)

    torch.manual_seed(3)
    card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    cfg = _trainable_config(
        card, tmp_path, ["ChargedHadronMomentumSmearing.resolution_module.scale_raw"]
    )
    _, param_groups = pc.select_trainable(card, cfg, global_lr=1e-1)

    history = fit_card_to_fullsim(
        card, train_dl, val_dl, param_groups=param_groups, n_steps=3, log_every=0
    )
    assert len(history["step"]) == 3
    assert len(history["loss"]) == 3
    assert len(history["val_loss"]) == 3
    for loss in history["loss"]:
        assert loss == loss  # not NaN


# ---------------------------------------------------------------------------
# Real-Pythia pseudodata end-to-end test
# ---------------------------------------------------------------------------

PSEUDODATA_PATH = Path(__file__).parent / "benchmark_data" / "cms_pseudodata.root"


@pytest.mark.skipif(
    not PSEUDODATA_PATH.exists(),
    reason="committed pseudodata file not available",
)
def test_fit_against_committed_pseudodata(tmp_path: Path):
    """End-to-end fit against the committed Pythia-generated pseudodata.

    The pseudodata was generated by
    :mod:`parnassus.torch_delphes.generate_pseudodata` with a deliberately
    perturbed CMS card (charged-hadron pT scale 1.25, ECal energy scale 1.20).
    Fitting the ECal + chad scale parameters of a fresh learnable card should
    drive the ECal scale meaningfully away from its 1.0 default toward 1.20. We
    do NOT assert full convergence -- that needs longer runs.
    """
    arrays = load_cms_flow_root(PSEUDODATA_PATH, n_events=150)
    device = torch.device("cpu")
    train_dl, val_dl = _make_dataloaders(arrays, device, batch_size=64, seed=11)

    torch.manual_seed(11)
    card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    cfg = _trainable_config(
        card,
        tmp_path,
        [
            "ECal.scale_module.scale_raw",
            "ChargedHadronMomentumSmearing.resolution_module.scale_raw",
        ],
    )
    _, param_groups = pc.select_trainable(card, cfg, global_lr=5e-2)

    ecal_scale = card.ECal.scale_module.scale_raw  # type: ignore[union-attr]
    before_ecal = (1.0 + 0.3 * torch.tanh(ecal_scale)).detach().clone()
    assert torch.allclose(before_ecal, torch.ones_like(before_ecal)), (
        "ECal scale was expected to start at 1.0 before the fit"
    )

    fit_card_to_fullsim(
        card, train_dl, val_dl, param_groups=param_groups, n_steps=20, log_every=0
    )

    after_ecal = (1.0 + 0.3 * torch.tanh(ecal_scale)).detach().clone()
    max_shift = float((after_ecal - 1.0).abs().max())
    assert max_shift > 0.01, f"ECal scale barely moved from 1.0: after={after_ecal.tolist()}"
    # None should have diverged the wrong way.
    assert float(after_ecal.min()) > 0.95, f"ECal scale drifted below 0.95: {after_ecal.tolist()}"
