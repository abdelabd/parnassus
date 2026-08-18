"""Tests for the ``tune_cms_fullsim`` harness.

These tests cover the pipeline end-to-end against the committed Pythia-generated
pseudodata (``benchmark_data/cms_pseudodata.root``): load it, build the padded
truth particle tensor and the target observable dict, run a forward + backward
step on the learnable CMS card, and run the Adam fit loop for a few steps. They
**skip** when that file is not present in the tree -- proper data is a
prerequisite for the fit, so there is no synthetic stand-in. Generate the file
with :mod:`parnassus.torch_delphes.generate_pseudodata` (or download the real
Zenodo sample) and rerun.
"""

from __future__ import annotations

import argparse
import math
import re
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
)
from parnassus.torch_delphes.SimpleCalorimeter import (
    calo_count_eta_edges,
    calo_count_region_masks,
)
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    apply_chad_truncation,
    apply_reco_acceptance_cut,
    batch_event_ids,
    load_pflow_targets,
    load_pflow_targets_from_tensor,
    load_pflow_targets_ragged,
    load_truth_events,
    load_truth_events_ragged,
    restore_event_format,
    split_pflow_targets_jagged,
    split_truth_objects_jagged,
)
from parnassus.torch_delphes.tune_cms_fullsim.dataloader import (
    DelphesDataLoader,
    DelphesDataSet,
    delphes_collate_fn,
)
from parnassus.torch_delphes.tune_cms_fullsim.config import (
    CALO_COUNT_TERM_KEYS,
    COUNT_TERM_KEYS,
    DEFAULT_MODE,
)
from parnassus.torch_delphes.tune_cms_fullsim.loss import (
    LOSS_CHOICES,
    N_SHAPE_ETA_REGIONS,
    PID_WEIGHTING_CHOICES,
    SHAPE_ETA_EDGES,
    _count_terms,
    _pid_population_weights,
    attach_truth_pair_lnm,
    compute_pair_masses,
    get_loss_fn,
    per_event_wasserstein_loss,
    per_pid_soft_hist_loss,
    per_pid_wasserstein_1d_loss,
    quantile_wasserstein_distance,
)
from parnassus.torch_delphes.tune_cms_fullsim.runner import (
    AcceptanceCuts,
    resolve_acceptance_cuts,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PSEUDODATA_PATH = Path(__file__).parent / "benchmark_data" / "cms_pseudodata.root"


@pytest.fixture(scope="module")
def fixture_root() -> Path:
    """The committed Pythia pseudodata; skip when it isn't present in the tree.

    There is no synthetic stand-in -- proper data is a prerequisite for the fit.
    Generate the file with ``parnassus.torch_delphes.generate_pseudodata`` (or
    download the real Zenodo sample) and rerun.
    """
    if not PSEUDODATA_PATH.exists():
        pytest.skip("committed pseudodata file not available")
    return PSEUDODATA_PATH


def _make_dataloaders(
    arrays: dict, device: torch.device, batch_size: int = 8,
) -> tuple[DelphesDataLoader, DelphesDataLoader]:
    """Build train/val dataloaders from loaded ROOT arrays (mirrors the CLI)."""
    truth = load_truth_events_ragged(arrays)
    target = load_pflow_targets_ragged(arrays)
    tr_truth, va_truth, _ = split_truth_objects_jagged(truth, train_fraction=0.7, val_fraction=0.2)
    tr_tgt, va_tgt, _ = split_pflow_targets_jagged(target, train_fraction=0.7, val_fraction=0.2)
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
    """The pseudodata file is readable and contains every branch the script reads."""
    import uproot

    with uproot.open(str(fixture_root)) as f:
        keys = set(f["event_tree"].keys())
    for branch in TRUTH_BRANCHES + PFLOW_BRANCHES:
        assert branch in keys, f"missing branch {branch} in pseudodata"


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
    # Real (non-padded) rows have non-negative mass, finite pt, and E^2 ~ p^2 + m^2.
    real = truth[torch.any(truth != 0, dim=-1)]
    assert real.shape[0] > 0
    assert float(real[:, ColumnMap.MASS].min()) >= 0
    assert float(real[:, ColumnMap.MASS].max()) > 0
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


def test_load_truth_events_ragged_matches_dense(fixture_root: Path):
    """The ragged truth loader holds exactly the dense loader's non-padded rows.

    The dense ``(n_events, max_n_particles, N_FEATURES)`` tensor with its zero
    padding removed must equal the concatenation of the ragged per-event tensors,
    in the same order -- i.e. per-batch padding loses nothing.
    """
    arrays = load_cms_flow_root(fixture_root, n_events=20)
    dense = load_truth_events(arrays)            # (n, max_n, N_FEATURES)
    ragged = load_truth_events_ragged(arrays)    # list of (n_i, N_FEATURES)

    assert len(ragged) == dense.shape[0]
    # Every real truth row has STATUS=1, so it is never all-zero; the mask cleanly
    # separates real rows from padding.
    dense_nonpad = dense[torch.any(dense != 0, dim=-1)]
    ragged_cat = torch.cat(ragged, dim=0) if ragged else dense_nonpad
    assert ragged_cat.shape == dense_nonpad.shape
    assert torch.equal(ragged_cat, dense_nonpad)
    # Per-event multiplicities line up too.
    dense_counts = torch.any(dense != 0, dim=-1).sum(dim=1)
    ragged_counts = torch.tensor([t.shape[0] for t in ragged])
    assert torch.equal(ragged_counts, dense_counts)


def test_delphes_collate_reproduces_dense_batch(fixture_root: Path):
    """Per-batch padding reproduces the dense global-padding loaders bit-for-bit.

    Collating the whole (unshuffled) event set pads each entry to the same global
    max, so the collated batch must equal the dense tensors element-for-element --
    truth card input and every target observable. A sub-batch that omits the
    busiest event pads to a strictly smaller width (the memory win).
    """
    arrays = load_cms_flow_root(fixture_root, n_events=12)
    device = torch.device("cpu")

    dense_truth = load_truth_events(arrays)
    dense_tgt = load_pflow_targets(arrays)

    ragged = load_truth_events_ragged(arrays)
    target = load_pflow_targets_ragged(arrays)
    ds = DelphesDataSet(ragged, target, device=device)
    batch = delphes_collate_fn([ds[i] for i in range(len(ds))])

    # Truth: full set -> per-batch max == global max -> identical to the dense tensor.
    assert torch.equal(batch["truth_particles"], dense_truth)
    # The un-padded card input (the way training.py builds it) is identical.
    dense_input = dense_truth[torch.any(dense_truth != 0, dim=-1)]
    ragged_input = batch["truth_particles"][torch.any(batch["truth_particles"] != 0, dim=-1)]
    assert torch.equal(ragged_input, dense_input)

    # Targets: the truth loader keeps empty events as zero-row entries, so the
    # dataset's truth-count always matches the full target count and the dense
    # and collated targets are directly comparable.
    if len(ragged) == dense_tgt["multiplicity"].shape[0]:
        for key in OBSERVABLES:
            assert torch.equal(batch[key], dense_tgt[key]), f"mismatch in target '{key}'"

    # Omitting the unique busiest event pads to a strictly smaller width.
    lengths = [t.shape[0] for t in ragged]
    if lengths.count(max(lengths)) == 1:
        busiest = lengths.index(max(lengths))
        sub = delphes_collate_fn([ds[i] for i in range(len(ds)) if i != busiest])
        assert sub["truth_particles"].shape[1] < dense_truth.shape[1]


# ---------------------------------------------------------------------------
# Loss and fit loop
# ---------------------------------------------------------------------------


def test_one_step_gradient_is_finite(fixture_root: Path):
    """A single forward + backward step produces a finite loss and finite
    gradients on every parameter that the loss reaches."""
    # Use a generous slice so the sparse lepton low/mid reco bins are reliably
    # populated (only a few leptons per ~20 events), which the count-term
    # gradient assertion on the muon/electron eff_logits below relies on.
    arrays = load_cms_flow_root(fixture_root, n_events=300)
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
    # The pseudodata populates the lepton low/mid bins, so each species' eff_logits
    # vector receives a nonzero count-term gradient.
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
    history dict of the right shape. We do NOT assert convergence on this small
    slice (the loss is dominated by stochastic smearing/Gumbel noise at tiny
    batch sizes)."""
    arrays = load_cms_flow_root(fixture_root, n_events=24)
    device = torch.device("cpu")
    train_dl, val_dl = _make_dataloaders(arrays, device, batch_size=8)

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
# Per-pid soft-histogram loss (--loss soft_hist)
# ---------------------------------------------------------------------------


def test_get_loss_fn_dispatch():
    """The dispatcher maps each loss name to the right callable (no data needed).

    Both Wasserstein losses dispatch to their ``*_distributed`` wrappers (a
    no-op passthrough outside DDP); ``soft_hist`` stays non-distributed.
    """
    from parnassus.torch_delphes.tune_cms_fullsim.loss import (
        per_event_wasserstein_loss_distributed,
        per_pid_wasserstein_1d_loss_distributed,
    )

    assert "wasserstein" in LOSS_CHOICES and "soft_hist" in LOSS_CHOICES
    assert "wasserstein_1d" in LOSS_CHOICES
    assert get_loss_fn("wasserstein") is per_event_wasserstein_loss_distributed
    assert get_loss_fn("soft_hist") is per_pid_soft_hist_loss
    assert get_loss_fn("wasserstein_1d") is per_pid_wasserstein_1d_loss_distributed
    with pytest.raises(ValueError):
        get_loss_fn("not_a_loss")


def test_quantile_wasserstein_distance_math():
    """Pure-math properties of the bin-free 1D quantile-Wasserstein primitive (no data).

    With default ``p=2`` the distance is the SQUARED W2: identical clouds give ~0; a rigid
    shift by ``delta`` gives ``delta**2``; it is symmetric in its two arguments; it handles
    unequal sample sizes; and it is differentiable in the pred (first) argument."""
    torch.manual_seed(0)
    base = torch.randn(500, dtype=torch.float64)
    delta = 1.3
    shifted = base + delta

    # Identical clouds -> ~0.
    assert float(quantile_wasserstein_distance(base, base)) == pytest.approx(0.0, abs=1e-9)

    # Rigid shift by delta -> squared-W2 == delta**2 (no scale standardization here).
    d = quantile_wasserstein_distance(base, shifted)
    assert float(d) == pytest.approx(delta**2, rel=1e-3)

    # p=1 -> W1 == |delta|.
    d1 = quantile_wasserstein_distance(base, shifted, p=1)
    assert float(d1) == pytest.approx(abs(delta), rel=1e-3)

    # Symmetric in the two clouds.
    rev = quantile_wasserstein_distance(shifted, base)
    assert float(d) == pytest.approx(float(rev), rel=1e-9)

    # Handles unequal sample sizes (no binning / common axis required).
    other = torch.randn(317, dtype=torch.float64) + delta
    assert torch.isfinite(quantile_wasserstein_distance(base, other))

    # Differentiable in the pred (first) argument; target detached.
    x = base.clone().requires_grad_(True)
    quantile_wasserstein_distance(x, shifted).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and float(x.grad.abs().sum()) > 0


def test_wasserstein_1d_one_step_gradient_is_finite(fixture_root: Path):
    """A forward + backward step under the per-pid bin-free 1D-Wasserstein loss gives a
    finite loss and finite gradients, and the shared count terms still reach the
    tracking-efficiency logits (their only gradient path)."""
    arrays = load_cms_flow_root(fixture_root, n_events=300)
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    torch.manual_seed(7)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)

    mask = torch.any(truth != 0, dim=-1)
    out = card(truth[mask])
    eflow = restore_event_format(out["EFlowObject"], mask)
    pred = load_pflow_targets_from_tensor(eflow)
    attach_truth_pair_lnm(truth, pred, target)
    for out_key, pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        pred[pred_key] = out[out_key]

    loss = per_pid_wasserstein_1d_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in card.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    for name, p in card.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"

    # The shared count terms must reach the tracking-efficiency logits.
    params = dict(card.named_parameters())
    for mod in (
        "ChargedHadronTrackingEfficiency",
        "ElectronTrackingEfficiency",
        "MuonTrackingEfficiency",
    ):
        g = params[f"{mod}.eff_logits"].grad
        assert g is not None and float(g.abs().sum()) > 0, f"{mod}.eff_logits got no count grad"


def test_soft_hist_one_step_gradient_is_finite(fixture_root: Path):
    """A forward + backward step under the per-pid soft-histogram loss gives a finite
    loss and finite gradients, and the shared count terms still reach the
    tracking-efficiency logits (their only gradient path)."""
    arrays = load_cms_flow_root(fixture_root, n_events=300)
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    torch.manual_seed(7)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)

    mask = torch.any(truth != 0, dim=-1)
    out = card(truth[mask])
    eflow = restore_event_format(out["EFlowObject"], mask)
    pred = load_pflow_targets_from_tensor(eflow)
    attach_truth_pair_lnm(truth, pred, target)
    # Inject BOTH the tracking and calo expected counts so every count term fires,
    # exactly as training.py does.
    for out_key, pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        pred[pred_key] = out[out_key]

    loss = per_pid_soft_hist_loss(pred, target)
    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in card.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    for name, p in card.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"

    # The shared count terms must reach the tracking-efficiency logits.
    params = dict(card.named_parameters())
    for mod in (
        "ChargedHadronTrackingEfficiency",
        "ElectronTrackingEfficiency",
        "MuonTrackingEfficiency",
    ):
        g = params[f"{mod}.eff_logits"].grad
        assert g is not None and float(g.abs().sum()) > 0, f"{mod}.eff_logits got no count grad"


def test_pid_population_weights_math():
    """Pure-math properties of the per-pid population weighting helper (no data).

    target_groups uses .shape[0] as the per-pid count; weights are mean-1 normalized over
    the present pids and down-weight rare species, with 'equal' an exact no-op."""
    # Abundant 211 (1000), rarer 11 (10), rarest 13 (4). Only shape[0] matters.
    tg = {211: torch.zeros(1000, 3), 11: torch.zeros(10, 3), 13: torch.zeros(4, 3)}
    present = [211, 11, 13]
    P = len(present)

    # 'equal' -> all exactly 1.0.
    w_eq = _pid_population_weights(tg, present, mode="equal")
    assert w_eq == {211: 1.0, 11: 1.0, 13: 1.0}

    # Every mode: positive, and mean-1 (sum == P) over present pids.
    for mode in PID_WEIGHTING_CHOICES:
        w = _pid_population_weights(tg, present, mode=mode)
        assert all(v > 0 for v in w.values())
        assert sum(w.values()) == pytest.approx(P, rel=1e-9)

    # Non-equal modes: abundant pid up-weighted, rare pid down-weighted.
    w_frac = _pid_population_weights(tg, present, mode="fraction")
    w_sqrt = _pid_population_weights(tg, present, mode="sqrt_fraction")
    for w in (w_frac, w_sqrt):
        assert w[211] > 1.0 > w[11] > w[13]  # rarer -> smaller

    # 'fraction' suppresses the rare pid MORE than 'sqrt_fraction' (monotone in strength).
    assert w_frac[13] < w_sqrt[13] < 1.0
    assert w_frac[11] < w_sqrt[11] < 1.0

    # A floor lifts the rare pids (re-normalized, so still mean-1).
    w_floor = _pid_population_weights(tg, present, mode="sqrt_fraction", floor=0.3)
    assert w_floor[13] > w_sqrt[13] and w_floor[11] > w_sqrt[11]
    assert sum(w_floor.values()) == pytest.approx(P, rel=1e-9)

    # Degenerate: empty present set -> empty dict (no divide-by-zero).
    assert _pid_population_weights(tg, [], mode="fraction") == {}


@pytest.mark.parametrize(
    "loss_fn",
    [per_event_wasserstein_loss, per_pid_soft_hist_loss, per_pid_wasserstein_1d_loss],
)
def test_pid_weighting_wired_and_equal_default(fixture_root: Path, loss_fn):
    """For every loss: pid_weighting='equal' equals the default (the default IS equal, and
    the equal weights are literal 1.0 -- see test_pid_population_weights_math, so it is a
    no-op), while 'sqrt_fraction' actually CHANGES the real-data loss (the knob is wired
    through to the shape terms). Each loss is deterministic for a fixed pred (Wasserstein
    uses a fixed SW seed; the hist/quantile losses have no RNG)."""
    arrays = load_cms_flow_root(fixture_root, n_events=300)
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    torch.manual_seed(7)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    mask = torch.any(truth != 0, dim=-1)
    out = card(truth[mask])
    pred = load_pflow_targets_from_tensor(restore_event_format(out["EFlowObject"], mask))
    attach_truth_pair_lnm(truth, pred, target)
    for out_key, pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        pred[pred_key] = out[out_key]

    default = float(loss_fn(pred, target))
    explicit_equal = float(loss_fn(pred, target, pid_weighting="equal"))
    sqrt_frac = float(loss_fn(pred, target, pid_weighting="sqrt_fraction"))
    assert default == explicit_equal  # default is 'equal'
    assert sqrt_frac != explicit_equal  # the knob reaches the shape terms


@pytest.mark.parametrize(
    "loss_fn",
    [per_event_wasserstein_loss, per_pid_soft_hist_loss, per_pid_wasserstein_1d_loss],
)
def test_pid_weighting_downweights_rare_pid(loss_fn):
    """When only a RARE pid's pred shape is mismatched (the abundant pid matches exactly),
    down-weighting that rare pid lowers the loss: fraction < sqrt_fraction < equal. Uses a
    synthetic 2-pid batch and zeroes the count/event terms to isolate the shape sum."""
    torch.manual_seed(0)
    n_abundant, n_rare = 2000, 8

    def make_particles(shift_rare: float):
        # One event, (1, n_abundant + n_rare, 4) = [log_E, log_pt, eta, pid].
        ab = torch.randn(n_abundant, 3, dtype=torch.float64)
        ra = torch.randn(n_rare, 3, dtype=torch.float64)
        ra = ra + shift_rare  # shift the rare species' kinematics
        ab = torch.cat([ab, torch.full((n_abundant, 1), 211.0, dtype=torch.float64)], dim=1)
        ra = torch.cat([ra, torch.full((n_rare, 1), 13.0, dtype=torch.float64)], dim=1)
        return torch.cat([ab, ra], dim=0).unsqueeze(0)

    # Shared target; pred matches the abundant pid exactly and shifts only the rare pid.
    torch.manual_seed(1)
    tgt_parts = make_particles(0.0)
    pred_parts = tgt_parts.clone()
    pred_parts[0, n_abundant:, :3] += 1.0  # shift only the rare (pid 13) cloud in pred

    def to_obs(parts):
        d = {
            "log_E": parts[..., 0],
            "log_pt": parts[..., 1],
            "eta": parts[..., 2],
            "pid": parts[..., 3],
        }
        # per-event log_ht: matched on both sides so the event term is ~0 anyway.
        d["log_ht"] = parts[..., 1].sum(dim=1)
        return d

    pred, target = to_obs(pred_parts), to_obs(tgt_parts)
    kw = dict(count_weight=0.0, calo_count_weight=0.0, event_weight=0.0)

    eq = float(loss_fn(pred, target, pid_weighting="equal", **kw))
    sq = float(loss_fn(pred, target, pid_weighting="sqrt_fraction", **kw))
    fr = float(loss_fn(pred, target, pid_weighting="fraction", **kw))
    assert fr < sq < eq  # the rare-pid mismatch is progressively down-weighted


def test_pid_weighting_keeps_leptons_learnable(fixture_root: Path):
    """Under 'sqrt_fraction' the muon/electron momentum-smearing params still receive a
    finite, nonzero gradient (their only gradient path here is the per-pid shape term:
    the pair-mass terms -- a second, un-weighted path -- are switched off to isolate the
    per-pid weighting), and under the aggressive 'fraction' mode that same gradient is
    SMALLER -- confirming the per-pid weight actually reaches the parameter gradient."""
    arrays = load_cms_flow_root(fixture_root, n_events=300)
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    smearing_prefixes = ("MuonMomentumSmearing", "ElectronMomentumSmearing")

    def lepton_smearing_grad_norm(mode: str) -> float:
        torch.manual_seed(7)
        card = CMSEnergyFlowDefault(debug=False, learnable=True)
        mask = torch.any(truth != 0, dim=-1)
        out = card(truth[mask])
        pred = load_pflow_targets_from_tensor(restore_event_format(out["EFlowObject"], mask))
        for out_key, pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
            pred[pred_key] = out[out_key]
        loss = per_pid_wasserstein_1d_loss(pred, target, pid_weighting=mode, pair_mass=False)
        assert torch.isfinite(loss)
        loss.backward()
        total = 0.0
        for name, p in card.named_parameters():
            if any(name.startswith(pre) for pre in smearing_prefixes) and p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"{name} non-finite grad"
                total += float(p.grad.abs().sum())
        return total

    g_sqrt = lepton_smearing_grad_norm("sqrt_fraction")
    g_equal = lepton_smearing_grad_norm("equal")
    g_fraction = lepton_smearing_grad_norm("fraction")

    # sqrt_fraction keeps the lepton smearing params learnable (nonzero gradient)...
    assert g_sqrt > 0
    # ...but down-weighted vs equal, and 'fraction' suppresses them even further.
    assert g_fraction < g_sqrt < g_equal


def test_count_terms_shared_between_losses(fixture_root: Path):
    """Both training losses add the SAME shared ``_count_terms`` contribution -- the
    additive count-term sub-sum equals ``_count_terms(...).sum()`` for each loss,
    locking in that the helper extraction is shared (not duplicated/divergent)."""
    arrays = load_cms_flow_root(fixture_root, n_events=300)
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)

    torch.manual_seed(7)
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    mask = torch.any(truth != 0, dim=-1)
    out = card(truth[mask])
    eflow = restore_event_format(out["EFlowObject"], mask)
    pred = load_pflow_targets_from_tensor(eflow)
    attach_truth_pair_lnm(truth, pred, target)
    for out_key, pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        pred[pred_key] = out[out_key]

    cw, ccw = 0.5, 10.0
    shared = float(
        torch.stack(
            _count_terms(pred, target, count_weight=cw, calo_count_weight=ccw)
        ).sum()
    )
    assert shared > 0  # the pseudodata populates several reco bins

    # Each loss is deterministic for a fixed pred (Wasserstein uses a fixed SW seed;
    # the histogram losses have no RNG), so loss(with counts) - loss(counts zeroed) is
    # exactly the count contribution for that loss.
    w_with = float(
        per_event_wasserstein_loss(pred, target, count_weight=cw, calo_count_weight=ccw)
    )
    w_without = float(
        per_event_wasserstein_loss(pred, target, count_weight=0.0, calo_count_weight=0.0)
    )
    h_with = float(
        per_pid_soft_hist_loss(pred, target, count_weight=cw, calo_count_weight=ccw)
    )
    h_without = float(
        per_pid_soft_hist_loss(pred, target, count_weight=0.0, calo_count_weight=0.0)
    )
    wh_with = float(
        per_pid_wasserstein_1d_loss(
            pred, target, count_weight=cw, calo_count_weight=ccw
        )
    )
    wh_without = float(
        per_pid_wasserstein_1d_loss(
            pred, target, count_weight=0.0, calo_count_weight=0.0
        )
    )

    assert (w_with - w_without) == pytest.approx(shared, rel=1e-6)
    assert (h_with - h_without) == pytest.approx(shared, rel=1e-6)
    assert (wh_with - wh_without) == pytest.approx(shared, rel=1e-6)


def test_count_terms_batch_size_invariant():
    """Every per-species count term is batch-size INVARIANT: evaluating ``_count_terms``
    on the same per-event count distribution replicated to a larger batch returns the same
    per-term values. This locks the regression where the old constant ``+1`` Pearson floor
    (effective rate floor ``1/N``) let sparse/empty data regions grow ~linearly (tracking,
    after ``/total``) to ~quadratically (calo) with batch size N.

    Fully synthetic (no fixture): ``target[tgt]`` is the realistic ``(n_events, n_regions)``
    per-event count tensor and ``pred[pred]`` is the ``(n_regions,)`` batch SUM (so it scales
    with the event count, exactly like ``_expected_reco_counts`` / the calo soft count).
    Region 0 is deliberately a STRUCTURALLY-EMPTY data region (target == 0) while the model
    still predicts a nonzero count there -- the exact batch-growth trigger -- which the
    per-event-rate floor must keep finite and invariant. The two batch sizes are built by
    tiling the same base events, so each term is expected to match (near-)exactly.
    """
    torch.manual_seed(0)
    cw, ccw = 0.5, 10.0
    n_base, n_reg = 50, 4

    # Per-key base per-event target and a per-event predicted rate (model batch sum / N).
    base_target: dict[str, torch.Tensor] = {}
    per_event_pred_rate: dict[str, torch.Tensor] = {}
    for _out_key, pred_key, tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
        counts = torch.randint(0, 5, (n_base, n_reg), dtype=torch.float64)
        counts[:, 0] = 0.0  # region 0: empty in the data, but the model still predicts there
        base_target[tgt_key] = counts
        per_event_pred_rate[pred_key] = torch.rand(n_reg, dtype=torch.float64) + 0.1

    def terms_at(n_tiles: int) -> list[float]:
        n_events = n_base * n_tiles
        target = {k: v.repeat(n_tiles, 1) for k, v in base_target.items()}
        pred = {k: rate * n_events for k, rate in per_event_pred_rate.items()}
        return [
            float(t)
            for t in _count_terms(pred, target, count_weight=cw, calo_count_weight=ccw)
        ]

    small = terms_at(1)
    large = terms_at(8)  # 8x the events, identical per-event distribution
    assert len(small) == len(large) == len(COUNT_TERM_KEYS) + len(CALO_COUNT_TERM_KEYS)
    for s, l in zip(small, large):
        assert s > 0  # empty-region + populated-region mismatch both contribute
        assert s == pytest.approx(l, rel=1e-9), (s, l)


def test_fit_soft_hist_runs(fixture_root: Path, tmp_path: Path):
    """The fit loop runs a few steps under ``--loss soft_hist`` without errors and
    returns a finite history (exercises the weight-injecting closure + the graph
    anchor end-to-end)."""
    arrays = load_cms_flow_root(fixture_root, n_events=24)
    device = torch.device("cpu")
    train_dl, val_dl = _make_dataloaders(arrays, device, batch_size=8)

    torch.manual_seed(3)
    card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    cfg = _trainable_config(
        card, tmp_path, ["ChargedHadronMomentumSmearing.resolution_module.scale_raw"]
    )
    _, param_groups = pc.select_trainable(card, cfg, global_lr=1e-1)

    history = fit_card_to_fullsim(
        card,
        train_dl,
        val_dl,
        param_groups=param_groups,
        n_steps=3,
        log_every=0,
        loss_name="soft_hist",
    )
    assert len(history["step"]) == 3
    for loss in history["loss"]:
        assert loss == loss  # not NaN
    for vloss in history["val_loss"]:
        assert vloss == vloss


def test_fit_wasserstein_1d_runs(fixture_root: Path, tmp_path: Path):
    """The fit loop runs a few steps under ``--loss wasserstein_1d`` without errors and
    returns a finite history (exercises the weight-injecting closure + the shared graph
    anchor branch end-to-end, mirroring ``test_fit_soft_hist_runs``)."""
    arrays = load_cms_flow_root(fixture_root, n_events=24)
    device = torch.device("cpu")
    train_dl, val_dl = _make_dataloaders(arrays, device, batch_size=8)

    torch.manual_seed(3)
    card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
    cfg = _trainable_config(
        card, tmp_path, ["ChargedHadronMomentumSmearing.resolution_module.scale_raw"]
    )
    _, param_groups = pc.select_trainable(card, cfg, global_lr=1e-1)

    history = fit_card_to_fullsim(
        card,
        train_dl,
        val_dl,
        param_groups=param_groups,
        n_steps=3,
        log_every=0,
        loss_name="wasserstein_1d",
    )
    assert len(history["step"]) == 3
    for loss in history["loss"]:
        assert loss == loss  # not NaN
    for vloss in history["val_loss"]:
        assert vloss == vloss


# ---------------------------------------------------------------------------
# Acceptance cuts + truth-ceiling chad truncation
# ---------------------------------------------------------------------------


def _mk_padded_obs(pt_rows: list[list[float]], pid_rows: list[list[int]],
                   eta_rows: list[list[float]] | None = None) -> dict:
    """Padded synthetic observables dict for the loss-side filter tests."""
    n = len(pt_rows)
    width = max(len(r) for r in pt_rows)
    pt = torch.zeros((n, width), dtype=torch.float64)
    pid = torch.zeros((n, width), dtype=torch.float64)
    eta = torch.zeros((n, width), dtype=torch.float64)
    for i, (pr, ir) in enumerate(zip(pt_rows, pid_rows)):
        pt[i, : len(pr)] = torch.tensor(pr, dtype=torch.float64)
        pid[i, : len(ir)] = torch.tensor(ir, dtype=torch.float64)
        if eta_rows is not None:
            eta[i, : len(eta_rows[i])] = torch.tensor(eta_rows[i], dtype=torch.float64)
    valid = pt != 0
    obs = {
        "pt": pt,
        "eta": eta,
        "phi": torch.zeros_like(pt),
        "log_pt": torch.where(valid, torch.log(pt.clamp(min=1e-6)), torch.zeros_like(pt)),
        "log_E": torch.where(valid, torch.log(pt.clamp(min=1e-6)), torch.zeros_like(pt)),
        "pid": pid,
        "multiplicity": valid.sum(dim=1).to(pt.dtype),
        "ht": pt.sum(dim=1),
        "log_ht": torch.log(pt.sum(dim=1).clamp(min=1e-6)),
    }
    return obs


def test_truth_acceptance_cut_matches_hand_selection(fixture_root: Path):
    """The truth loader's acceptance cut equals a hand selection on the raw arrays."""
    arrays = load_cms_flow_root(fixture_root, n_events=10)
    cut_rows = load_truth_events_ragged(arrays, truth_pt_cut=1.0, abs_eta_cut=2.0)
    assert len(cut_rows) == len(arrays["truth_pt"])
    for i, row in enumerate(cut_rows):
        pt = np.asarray(arrays["truth_pt"][i], dtype=np.float64)
        eta = np.asarray(arrays["truth_eta"][i], dtype=np.float64)
        sel = (pt >= 1.0) & (np.abs(eta) <= 2.0)
        assert row.shape[0] == int(sel.sum())
        if row.shape[0]:
            np.testing.assert_allclose(row[:, ColumnMap.PT].numpy(), pt[sel])
            assert float(row[:, ColumnMap.PT].min()) >= 1.0
            assert float(row[:, ColumnMap.ETA].abs().max()) <= 2.0


def test_truth_loader_keeps_empty_events():
    """An event emptied (or born empty) stays in the list as a (0, N_FEATURES) row,
    keeping the truth list aligned with the (all-events) pflow target loader."""
    obj = np.empty(3, dtype=object)
    arrays = {
        "truth_pt": obj.copy(), "truth_eta": obj.copy(),
        "truth_phi": obj.copy(), "truth_pdgid": obj.copy(),
    }
    vals = {
        "truth_pt": [np.array([5.0, 0.4]), np.array([]), np.array([0.3])],
        "truth_eta": [np.array([0.1, 0.2]), np.array([]), np.array([1.0])],
        "truth_phi": [np.array([0.0, 1.0]), np.array([]), np.array([2.0])],
        "truth_pdgid": [np.array([211, 22]), np.array([]), np.array([211])],
    }
    for k in arrays:
        for i in range(3):
            arrays[k][i] = vals[k][i]
    rows = load_truth_events_ragged(arrays, truth_pt_cut=1.0, abs_eta_cut=2.7)
    assert len(rows) == 3  # one entry per event, empties kept
    assert rows[0].shape == (1, N_FEATURES)  # 0.4 GeV particle cut away
    assert rows[1].shape == (0, N_FEATURES)  # born empty
    assert rows[2].shape == (0, N_FEATURES)  # emptied by the cut


def test_reco_acceptance_cut_target_loader(fixture_root: Path):
    """The target loader's reco cut applies to every class, and mult/ht/region
    counts are built from the cut set (sub-cut region bins go to zero)."""
    arrays = load_cms_flow_root(fixture_root, n_events=16)
    plain = load_pflow_targets_ragged(arrays)
    cut = load_pflow_targets_ragged(arrays, reco_pt_cut=1.0, abs_eta_cut=2.0)
    for i, pt in enumerate(cut["pt"]):
        if pt.numel():
            assert float(pt.min()) >= 1.0
            assert float(cut["eta"][i].abs().max()) <= 2.0
        # mult/ht recomputed from the kept objects
        assert float(cut["multiplicity"][i]) == pt.numel()
        assert torch.isclose(cut["ht"][i], pt.sum().double(), atol=1e-9)
    # Cut never adds objects
    assert float(cut["multiplicity"].sum()) <= float(plain["multiplicity"].sum())
    # Chad (pt, |eta|) region-count targets: rebuilt from the cut set -- every
    # sub-1-GeV pt region must be empty. Region 0/1 of the chad spec are the
    # pt <= 1 bins (CMS_EFF_REGION_SPECS: pt edges (0.1, 1.0)).
    from parnassus.torch_delphes.learnable import CMS_EFF_REGION_SPECS
    spec = CMS_EFF_REGION_SPECS["charged_hadron"]
    # Identify sub-cut regions by probing the masks with a 0.5 GeV chad at eta 0.
    probe_pt = np.array([0.5])
    probe_eta = np.array([0.0])
    for b, m in enumerate(spec.region_masks(probe_pt, probe_eta)):
        if m[0]:
            assert float(cut["chad_region_counts"][:, b].sum()) == 0.0
    # Calo count-target widths follow the bounded region layout.
    assert cut["ecal_photon_region_counts"].shape[1] == len(calo_count_eta_edges(True, 2.0)) + 1
    assert cut["hcal_nh_region_counts"].shape[1] == len(calo_count_eta_edges(False, 2.0)) + 1


def test_n_truth_chad_matches_hand_count(fixture_root: Path):
    """``n_truth_chad`` equals a hand count of truth charged hadrons inside the
    reco acceptance, and survives split + dataset + collate as a (batch,) tensor."""
    from parnassus.utils import pid_to_class_vectorized

    arrays = load_cms_flow_root(fixture_root, n_events=12)
    target = load_pflow_targets_ragged(arrays, reco_pt_cut=1.0, abs_eta_cut=2.7)
    for i in range(len(arrays["truth_pt"])):
        t_pt = np.asarray(arrays["truth_pt"][i], dtype=np.float64)
        if t_pt.size:
            t_eta = np.asarray(arrays["truth_eta"][i], dtype=np.float64)
            t_pid = np.asarray(arrays["truth_pdgid"][i], dtype=np.int64)
            hand = int(np.sum(
                (pid_to_class_vectorized(t_pid) == 0) & (t_pt >= 1.0) & (np.abs(t_eta) <= 2.7)
            ))
        else:
            hand = 0
        assert int(target["n_truth_chad"][i]) == hand
    # Rides the standard split/dataset/collate machinery like the region counts.
    truth = load_truth_events_ragged(arrays)
    ds = DelphesDataSet(truth, target, device=torch.device("cpu"))
    batch = delphes_collate_fn([ds[i] for i in range(min(4, len(ds)))])
    assert batch["n_truth_chad"].shape == (min(4, len(ds)),)
    assert torch.equal(batch["n_truth_chad"], target["n_truth_chad"][: min(4, len(ds))])


def test_target_chad_truncation(fixture_root: Path):
    """Loader-side truncation keeps exactly min(n_chad, n_truth_chad) chads per
    event, keeps the TOP-pt subset, and leaves other classes untouched."""
    arrays = load_cms_flow_root(fixture_root, n_events=16)
    plain = load_pflow_targets_ragged(arrays, reco_pt_cut=1.0, abs_eta_cut=2.7)
    trunc = load_pflow_targets_ragged(
        arrays, reco_pt_cut=1.0, abs_eta_cut=2.7, truncate_chads=True
    )
    for i in range(len(plain["pt"])):
        p_pid = plain["pid"][i]
        t_pid = trunc["pid"][i]
        n_chad_plain = int((p_pid.abs() == 211).sum())
        n_chad_trunc = int((t_pid.abs() == 211).sum())
        k = int(plain["n_truth_chad"][i])
        assert n_chad_trunc == min(n_chad_plain, k)
        # Top-pt subset: the kept chads are the k hardest of the plain set.
        plain_chad_pt = plain["pt"][i][p_pid.abs() == 211]
        trunc_chad_pt = trunc["pt"][i][t_pid.abs() == 211]
        if n_chad_trunc:
            expected = torch.sort(plain_chad_pt, descending=True).values[:n_chad_trunc]
            assert torch.allclose(
                torch.sort(trunc_chad_pt, descending=True).values, expected
            )
        # Other classes byte-identical.
        for cls_pid in (11, 13, 22, 111):
            assert int((t_pid.abs() == cls_pid).sum()) == int((p_pid.abs() == cls_pid).sum())


def test_apply_reco_acceptance_cut_synthetic():
    """Exact keep mask, zeroing, recompute, empty batch, gradient, no mutation."""
    obs = _mk_padded_obs(
        pt_rows=[[5.0, 0.5, 2.0], [3.0]],
        pid_rows=[[211, 22, 22], [111]],
        eta_rows=[[0.1, 0.2, 3.0], [1.0]],
    )
    obs["pt"] = obs["pt"].requires_grad_(True)
    before = {k: v.detach().clone() for k, v in obs.items()}
    out = apply_reco_acceptance_cut(obs, 1.0, 2.7)
    # Event 0: 0.5 GeV photon fails pt, 2.0 GeV photon at eta 3.0 fails eta.
    assert out["multiplicity"].tolist() == [1.0, 1.0]
    assert out["pt"][0].detach().tolist() == [5.0, 0.0, 0.0]
    assert out["pid"][0].tolist() == [211.0, 0.0, 0.0]
    assert torch.isclose(out["ht"][0], torch.tensor(5.0, dtype=torch.float64))
    # Gradient flows only through kept slots.
    out["ht"].sum().backward()
    assert obs["pt"].grad is not None
    assert obs["pt"].grad[0].tolist() == [1.0, 0.0, 0.0]
    # No mutation of the input dict's tensors.
    for k, v in before.items():
        assert torch.equal(obs[k].detach(), v), f"input {k} mutated"
    # None thresholds and empty batches pass through.
    same = apply_reco_acceptance_cut(obs, None, None)
    assert torch.equal(same["pt"].detach(), obs["pt"].detach())
    empty = {k: v[:, :0] if v.ndim == 2 else v for k, v in _mk_padded_obs([[1.0]], [[211]]).items()}
    assert apply_reco_acceptance_cut(empty, 1.0, 2.7)["pt"].shape[1] == 0


def test_apply_chad_truncation_synthetic():
    """Per-event k (incl. k=0 and k>n), pt ties, non-chads untouched, gradient."""
    obs = _mk_padded_obs(
        pt_rows=[[5.0, 3.0, 3.0, 2.0, 10.0], [4.0, 1.5, 0.0, 0.0, 0.0]],
        pid_rows=[[211, 211, 211, 22, 111], [211, 211, 0, 0, 0]],
    )
    obs["pt"] = obs["pt"].requires_grad_(True)
    n_t = torch.tensor([2.0, 0.0], dtype=torch.float64)
    out = apply_chad_truncation(obs, n_t)
    # Event 0: keep top-2 chads (5.0 and one of the tied 3.0s), photon + NH untouched.
    assert int((out["pid"][0].abs() == 211).sum()) == 2
    kept0 = out["pt"][0].detach()
    assert 5.0 in kept0.tolist() and 10.0 in kept0.tolist() and 2.0 in kept0.tolist()
    assert out["multiplicity"][0] == 4.0  # 2 chads + photon + NH
    # Event 1: k=0 drops ALL chads.
    assert int((out["pid"][1].abs() == 211).sum()) == 0
    assert out["multiplicity"][1] == 0.0
    # k > n keeps everything.
    out_all = apply_chad_truncation(obs, torch.tensor([99.0, 99.0], dtype=torch.float64))
    assert torch.equal(out_all["pt"].detach(), obs["pt"].detach())
    # Gradient flows through kept slots only (event 1 fully dropped chads).
    out["ht"].sum().backward()
    assert obs["pt"].grad is not None
    assert obs["pt"].grad[1].tolist() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_resolve_acceptance_cuts_modes():
    """Mode fullsim honours the cut flags (<= 0 disables); delphes turns everything off."""

    def ns(**kw) -> argparse.Namespace:
        base = {
            "mode": DEFAULT_MODE,
            "truth_pt_cut": 0.25,
            "reco_pt_cut": 1.0,
            "eta_cut": 2.7,
            "no_chad_truncation": False,
        }
        return argparse.Namespace(**{**base, **kw})

    assert DEFAULT_MODE == "fullsim"
    assert resolve_acceptance_cuts(ns()) == AcceptanceCuts(0.25, 1.0, 2.7, truncate_chads=True)
    assert resolve_acceptance_cuts(
        ns(reco_pt_cut=0.0, eta_cut=-1.0, no_chad_truncation=True)
    ) == (0.25, None, None, False)
    # delphes: cuts + truncation off regardless of the (ignored) cut flags.
    assert resolve_acceptance_cuts(ns(mode="delphes", reco_pt_cut=10.0, eta_cut=1.0)) == (
        None, None, None, False,
    )


def test_count_terms_pid_weighting():
    """--pid-weighting also redistributes ACROSS the count-term species:
    'equal' is a bit-exact no-op; 'sqrt_fraction' down-weights the rare species
    and up-weights the abundant one, mean-1 normalized."""
    n_events = 10
    # Two species: abundant chads (90/event across 4 regions) and rare muons
    # (0.4/event across 6 regions), with a fixed 20% pred deficit on both.
    chad_tgt = torch.full((n_events, 4), 22.5, dtype=torch.float64)  # 90/event
    muon_tgt = torch.full((n_events, 6), 0.4 / 6, dtype=torch.float64)
    target = {"chad_region_counts": chad_tgt, "muon_region_counts": muon_tgt}
    pred = {
        "chad_expected_counts": chad_tgt.sum(dim=0) * 0.8,
        "muon_expected_counts": muon_tgt.sum(dim=0) * 0.8,
    }

    def _terms(mode):
        return _count_terms(
            pred, target, count_weight=1.0, calo_count_weight=1.0,
            pid_weighting=mode,
        )

    equal = [float(t) for t in _terms("equal")]
    sqrt = [float(t) for t in _terms("sqrt_fraction")]
    # equal: identical to the unweighted computation (weights exactly 1.0).
    raw = [float(t) for t in _terms("equal")]
    assert equal == raw
    # The relative chi^2 raws are scale-free, so with equal weighting the two
    # species cost nearly the same despite a 225x population gap.
    assert equal[0] > 0 and equal[1] > 0
    # sqrt_fraction: chad (abundant) up-weighted, muon (rare) down-weighted,
    # mean preserved (mean-1 normalization over the two species).
    assert sqrt[0] > equal[0]
    assert sqrt[1] < equal[1]
    w_chad = sqrt[0] / equal[0]
    w_muon = sqrt[1] / equal[1]
    assert abs((w_chad + w_muon) / 2 - 1.0) < 1e-9  # mean-1
    # Expected ratio: sqrt(f_chad)/sqrt(f_muon) = sqrt(90/0.4)
    assert abs(w_chad / w_muon - math.sqrt(90 / 0.4)) < 1e-6


def test_restore_event_format_event_ids_alignment():
    """With ``event_ids`` pred rows land at their batch positions (shuffled,
    with an objectless event); ``None`` keeps the legacy sorted layout."""
    n_features = N_FEATURES
    # Batch of 3 events with global ids [7, 2, 5]; event 2 (id 5) has no objects.
    ids = torch.tensor([7, 2, 5])
    objs = []
    for ev_id, pt in [(2, 1.0), (7, 3.0), (2, 2.0)]:
        row = torch.zeros(n_features, dtype=torch.float64)
        row[ColumnMap.PT] = pt
        row[ColumnMap.EVENT_NUMBER] = ev_id
        objs.append(row)
    eflow = torch.stack(objs)
    mask = torch.ones((3, 4), dtype=torch.bool)
    out = restore_event_format(eflow, mask, event_ids=ids)
    assert out.shape[0] == 3
    assert sorted(out[0, :, ColumnMap.PT].tolist())[-1] == 3.0  # id 7 -> row 0
    assert sorted(out[1, :, ColumnMap.PT].tolist())[-2:] == [1.0, 2.0]  # id 2 -> row 1
    assert out[2].abs().sum() == 0.0  # id 5 produced nothing
    # Legacy layout: ascending event number (2 -> row 0, 7 -> row 1).
    legacy = restore_event_format(eflow, mask)
    assert sorted(legacy[0, :, ColumnMap.PT].tolist())[-2:] == [1.0, 2.0]
    assert sorted(legacy[1, :, ColumnMap.PT].tolist())[-1] == 3.0
    # batch_event_ids reads ids from the truth tensor (objectless events get
    # sentinels that never match).
    truth = torch.zeros((3, 2, n_features), dtype=torch.float64)
    for i, ev_id in enumerate([7, 2, 5]):
        truth[i, 0, ColumnMap.PT] = 1.0
        truth[i, 0, ColumnMap.EVENT_NUMBER] = ev_id
    t_mask = torch.any(truth != 0, dim=-1)
    assert batch_event_ids(truth, t_mask).tolist() == [7, 2, 5]


def test_calo_count_region_masks_bounded():
    """Bounded region layout: edges beyond the bound drop; the final region is
    capped; unbounded reproduces the legacy masks."""
    eta = np.array([0.5, 1.6, 2.6, 2.9, 3.5])
    # Legacy (unbounded) ECal: <=1.5 / (1.5, 2.5] / > 2.5.
    m = calo_count_region_masks(eta, calo_count_eta_edges(True, None), None)
    assert len(m) == 3
    assert m[2].tolist() == [False, False, True, True, True]
    # Bounded at 2.7: final region (2.5, 2.7]; 2.9 and 3.5 fall nowhere.
    m = calo_count_region_masks(eta, calo_count_eta_edges(True, 2.7), 2.7)
    assert len(m) == 3
    assert m[2].tolist() == [False, False, True, False, False]
    # HCal bounded at 2.7: the 3.0 edge drops -> single <= 2.7 region.
    assert calo_count_eta_edges(False, 2.7) == ()
    m = calo_count_region_masks(eta, (), 2.7)
    assert len(m) == 1
    assert m[0].tolist() == [True, True, True, False, False]
    # Torch tensors work identically (duck-typed).
    m_t = calo_count_region_masks(torch.from_numpy(eta), (), 2.7)
    assert m_t[0].tolist() == [True, True, True, False, False]


def test_soft_count_pt_gate_monotone_and_shapes(fixture_root: Path):
    """The card's expected-count outputs follow the bounded region layout, are
    monotonically non-increasing in the pt gate, and keep resolution-param
    gradients; the tracking counts are gated too."""
    arrays = load_cms_flow_root(fixture_root, n_events=8)
    truth = load_truth_events(arrays)
    flat = truth[torch.any(truth != 0, dim=-1)]

    def _forward(count_pt_min, count_abs_eta_max):
        torch.manual_seed(11)
        card = CMSEnergyFlowDefault(
            debug=False, learnable=True,
            count_pt_min=count_pt_min, count_abs_eta_max=count_abs_eta_max,
        )
        out = card(flat)
        return card, out

    _, legacy = _forward(None, None)
    assert legacy["EcalPhotonExpectedCounts"].shape == (3,)
    assert legacy["HcalNeutralHadronExpectedCounts"].shape == (2,)

    card, bounded = _forward(1.0, 2.7)
    assert bounded["EcalPhotonExpectedCounts"].shape == (3,)
    assert bounded["HcalNeutralHadronExpectedCounts"].shape == (1,)
    # Gate + bound can only remove counted objects (same seed => same smears).
    assert float(bounded["EcalPhotonExpectedCounts"].sum()) <= float(
        legacy["EcalPhotonExpectedCounts"].sum()
    ) + 1e-9
    assert float(bounded["ChargedHadronExpectedCounts"].sum()) <= float(
        legacy["ChargedHadronExpectedCounts"].sum()
    ) + 1e-9

    _, tighter = _forward(3.0, 2.7)
    assert float(tighter["EcalPhotonExpectedCounts"].sum()) <= float(
        bounded["EcalPhotonExpectedCounts"].sum()
    ) + 1e-9

    # Resolution params still receive gradient through the soft gate.
    (bounded["EcalPhotonExpectedCounts"].sum()
     + bounded["HcalNeutralHadronExpectedCounts"].sum()).backward()
    grads = [
        p.grad for n, p in card.named_parameters()
        if "resolution" in n and p.grad is not None and float(p.grad.abs().sum()) > 0
    ]
    assert grads, "no resolution parameter received gradient from the gated counts"


def test_fit_runs_with_acceptance_cuts(fixture_root: Path, tmp_path: Path):
    """End-to-end: loaders with cuts + truncation, gated card, fit hooks on --
    a short wasserstein_1d fit stays finite."""
    arrays = load_cms_flow_root(fixture_root, n_events=24)
    device = torch.device("cpu")
    truth = load_truth_events_ragged(arrays, truth_pt_cut=0.25, abs_eta_cut=2.7)
    target = load_pflow_targets_ragged(
        arrays, reco_pt_cut=1.0, abs_eta_cut=2.7, truncate_chads=True
    )
    tr_truth, va_truth, _ = split_truth_objects_jagged(truth, 0.7, 0.2)
    tr_tgt, va_tgt, _ = split_pflow_targets_jagged(target, 0.7, 0.2)
    train_dl = DelphesDataLoader(
        DelphesDataSet(tr_truth, tr_tgt, device=device), batch_size=8, shuffle=True
    )
    val_dl = DelphesDataLoader(
        DelphesDataSet(va_truth, va_tgt, device=device), batch_size=8, shuffle=False
    )

    torch.manual_seed(3)
    card = CMSEnergyFlowDefault(
        debug=False, learnable=True, count_pt_min=1.0, count_abs_eta_max=2.7
    ).to(device)
    cfg = _trainable_config(
        card, tmp_path, ["ChargedHadronMomentumSmearing.resolution_module.scale_raw"]
    )
    _, param_groups = pc.select_trainable(card, cfg, global_lr=1e-1)

    history = fit_card_to_fullsim(
        card,
        train_dl,
        val_dl,
        param_groups=param_groups,
        n_steps=2,
        log_every=0,
        loss_name="wasserstein_1d",
        reco_pt_cut=1.0,
        reco_abs_eta_cut=2.7,
        truncate_chads=True,
    )
    assert len(history["step"]) == 2
    for loss in history["loss"] + history["val_loss"]:
        assert loss == loss and loss not in (float("inf"), float("-inf"))


def test_intermediate_plots_include_per_pid_pages(tmp_path: Path):
    """The per-epoch intermediate PDF gains one per-PID page per particle type when
    the aligned ``pid`` array is present, and degrades gracefully (combined pages
    only) when it is not. Uses synthetic aligned arrays -- no ROOT data needed."""
    import re

    from parnassus.torch_delphes.tune_cms_fullsim import OBSERVABLES
    from parnassus.torch_delphes.tune_cms_fullsim.intermediate_plots import (
        _PID_GROUPS,
        save_intermediate_observable_plots,
    )

    def _page_count(pdf_path: Path) -> int:
        # Count page objects in the matplotlib PDF (page dicts are uncompressed).
        return len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf_path.read_bytes()))

    torch.manual_seed(0)
    n = 4000
    pids = torch.tensor([g[1] for g in _PID_GROUPS], dtype=torch.float64)
    pid = pids[torch.randint(0, len(pids), (n,))]
    pt = torch.rand(n, dtype=torch.float64) * 50 + 1.0
    eta = (torch.rand(n, dtype=torch.float64) - 0.5) * 6.0
    log_pt = torch.log(pt)
    log_E = torch.log(pt * torch.cosh(eta) + 0.5)
    log_ht = torch.rand(200, dtype=torch.float64) * 3 + 3  # per-event

    def mk(shift: float) -> dict:
        return {
            "pt": pt + shift,
            "eta": eta,
            "log_pt": log_pt + 0.01 * shift,
            "log_E": log_E + 0.01 * shift,
            "pid": pid,
            "log_ht": log_ht + 0.01 * shift,
        }

    target, pred, init = mk(0.0), mk(0.5), mk(1.0)

    p_pid = save_intermediate_observable_plots(
        pred, target, OBSERVABLES, step=3, output_dir=tmp_path, val_loss=1.0, init_by_key=init
    )
    # Same data minus the pid column -> combined pages only.
    drop = lambda d: {k: v for k, v in d.items() if k != "pid"}
    p_nopid = save_intermediate_observable_plots(
        drop(pred), drop(target), OBSERVABLES, step=4, output_dir=tmp_path, init_by_key=drop(init)
    )

    assert p_pid.exists() and p_nopid.exists()
    # Every PID group is populated, so we get exactly one extra page per group.
    assert _page_count(p_pid) - _page_count(p_nopid) == len(_PID_GROUPS)

    # Pair-mass pages: one extra page per class whose "pair_r:{pid}" / "pair_cat:{pid}"
    # arrays are present (here muons and charged hadrons, not electrons).
    def with_pairs(d: dict, shift: float) -> dict:
        d = dict(d)
        for pid_abs in (13, 211):
            d[f"pair_r:{pid_abs}"] = torch.randn(300, dtype=torch.float64) * 0.02 + shift
            d[f"pair_cat:{pid_abs}"] = torch.randint(0, 6, (300,))
            d[f"pair_grp:{pid_abs}"] = torch.randint(2, 4, (300,))
        return d

    p_pair = save_intermediate_observable_plots(
        with_pairs(pred, 0.01),
        with_pairs(target, 0.0),
        OBSERVABLES,
        step=5,
        output_dir=tmp_path,
        init_by_key=with_pairs(init, 0.02),
    )
    assert _page_count(p_pair) - _page_count(p_pid) == 2


# ---------------------------------------------------------------------------
# Real-Pythia pseudodata end-to-end test
# ---------------------------------------------------------------------------


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
    train_dl, val_dl = _make_dataloaders(arrays, device, batch_size=64)

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


# ---------------------------------------------------------------------------
# |eta|-split shape terms + leading-2 pair-mass terms
# ---------------------------------------------------------------------------


def _make_ref_obs(seed: int, shift: float = 0.0) -> dict[str, torch.Tensor]:
    """Deterministic synthetic padded observables: 64 events x up to 12 objects of pids
    211/11/13/22, pt log-uniform ~0.5-200 GeV, |eta| < 3, phi uniform; log_E from
    pt cosh(eta). Used to pin the OFF/OFF loss values to the pre-split implementation.
    """
    g = torch.Generator().manual_seed(seed)
    n_ev, width = 64, 12
    n_obj = torch.randint(2, width + 1, (n_ev,), generator=g)
    pt = torch.zeros(n_ev, width, dtype=torch.float64)
    eta = torch.zeros_like(pt)
    phi = torch.zeros_like(pt)
    pid = torch.zeros_like(pt)
    pids = torch.tensor([211.0, 211.0, 11.0, 13.0, 22.0], dtype=torch.float64)
    for i in range(n_ev):
        k = int(n_obj[i])
        pt[i, :k] = torch.exp(torch.rand(k, generator=g, dtype=torch.float64) * 6.0 - 0.7) * (
            1.0 + shift
        )
        eta[i, :k] = torch.rand(k, generator=g, dtype=torch.float64) * 6.0 - 3.0
        phi[i, :k] = torch.rand(k, generator=g, dtype=torch.float64) * math.tau
        pid[i, :k] = pids[torch.randint(0, 5, (k,), generator=g)]
    valid = pt != 0
    e = pt * torch.cosh(eta)
    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "pid": pid,
        "log_pt": torch.where(valid, torch.log(pt.clamp(min=1e-6)), torch.zeros_like(pt)),
        "log_E": torch.where(valid, torch.log(e.clamp(min=1e-6)), torch.zeros_like(pt)),
        "multiplicity": valid.sum(dim=1).to(pt.dtype),
        "ht": pt.sum(dim=1),
        "log_ht": torch.log(pt.sum(dim=1).clamp(min=1e-6)),
    }


def test_eta_split_and_pair_mass_off_reproduce_pooled_loss_exactly():
    """``eta_split=False, pair_mass=False`` reproduces the pre-split per-pid losses
    bit-for-bit (values frozen from the implementation before the split was added).
    """
    pred, tgt = _make_ref_obs(1, shift=0.05), _make_ref_obs(2)
    w, wc = per_pid_wasserstein_1d_loss(
        pred, tgt, eta_split=False, pair_mass=False, return_breakdown=True
    )
    h, hc = per_pid_soft_hist_loss(
        pred, tgt, eta_split=False, pair_mass=False, return_breakdown=True
    )
    assert float(w) == pytest.approx(3.583771651462262e-01, rel=1e-12, abs=0.0)
    assert float(h) == pytest.approx(2.522655773210395e-03, rel=1e-12, abs=0.0)
    assert len(wc) == len(hc) == 13  # 4 pids x 3 obs + log_ht, no region / pair keys
    assert not any(re.search(r":eta\d", c.label) or c.category == "pair" for c in wc)


def test_eta_split_keys_and_weights():
    """With the split ON: ``log_E``/``log_pt`` come as one term per populated |eta| region
    (``{pid}:{obs}:eta{r}``), ``eta`` stays a single pooled term, and the region weights of
    a pid/obs sum to the pooled pid weight x obj weight (target-fraction weights, all four
    regions populated here).
    """
    pred, tgt = _make_ref_obs(1, shift=0.05), _make_ref_obs(2)
    _, comps = per_pid_wasserstein_1d_loss(
        pred, tgt, eta_split=True, pair_mass=False, return_breakdown=True
    )
    labels = {c.label for c in comps}
    for r in range(N_SHAPE_ETA_REGIONS):
        assert f"211:log_pt:eta{r}" in labels
        assert f"211:log_E:eta{r}" in labels
    assert "211:eta" in labels
    assert "211:log_pt" not in labels
    w_split = sum(c.weight for c in comps if c.label.startswith("211:log_pt:eta"))
    assert w_split == pytest.approx(0.5, abs=1e-12)  # obj weight 0.5 x pid weight 1 (equal)
    assert not any(c.category == "pair" for c in comps)


def test_eta_split_attributes_a_shift_to_its_region():
    """Shifting ONLY the region-1 objects of a pid on the pred side changes ONLY that
    pid's ``:eta1`` terms (the other regions' terms are bit-identical), which is exactly
    the attribution the pooled term cannot provide.
    """
    tgt = _make_ref_obs(3)
    pred = {k: v.clone() for k, v in tgt.items()}
    lo, hi = SHAPE_ETA_EDGES[0], SHAPE_ETA_EDGES[1]
    in_r1 = (pred["pid"] == 211) & (pred["eta"].abs() > lo) & (pred["eta"].abs() <= hi)
    pred["log_pt"] = torch.where(in_r1, pred["log_pt"] + 0.3, pred["log_pt"])
    pred["log_E"] = torch.where(in_r1, pred["log_E"] + 0.3, pred["log_E"])
    _, base = per_pid_wasserstein_1d_loss(tgt, tgt, pair_mass=False, return_breakdown=True)
    _, comps = per_pid_wasserstein_1d_loss(pred, tgt, pair_mass=False, return_breakdown=True)
    base_by = {c.label: c.raw for c in base}
    for c in comps:
        if c.label.startswith("211:log_pt:eta") or c.label.startswith("211:log_E:eta"):
            if c.label.endswith(":eta1"):
                assert c.raw > base_by[c.label] + 1e-6, c.label
            else:
                assert c.raw == pytest.approx(base_by[c.label], abs=1e-12), c.label


def test_pair_masses_analytic_and_categories():
    """Leading-2 pair mass: analytic mass of a hand-built muon pair (third, softer muon
    ignored), category from the two objects' |eta| regions, and no pair for events with
    fewer than two objects of the class.
    """
    obs = _mk_padded_obs(
        pt_rows=[[10.0, 10.0, 3.0], [5.0], [2.0, 2.0]],
        pid_rows=[[13, 13, 13], [13], [211, 13]],
        eta_rows=[[0.0, 0.3, 1.0], [0.0], [0.0, 2.0]],
    )
    obs["phi"] = torch.tensor(
        [[0.0, math.pi, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    # Truth pair mass label 1 GeV (ln = 0) -> the response IS ln m_reco; the analytic
    # check below therefore reads the reco mass off the response.
    obs["truth_pair_lnm:13"] = torch.zeros(3, dtype=torch.float64)
    obs["truth_pair_lnm:211"] = torch.zeros(3, dtype=torch.float64)
    pairs = compute_pair_masses(obs)
    assert set(pairs) == {13}  # event 1: one muon; event 2: one muon + one chad -> no pair
    ln_m, cat, grp = pairs[13]
    assert ln_m.shape == cat.shape == grp.shape == (1,)
    assert int(grp[0]) == 0  # ln m_truth = 0 -> group 0 (mass 1-1.65 GeV)
    m_mu = 0.1056584
    e1 = math.sqrt(100.0 + m_mu**2)
    e2 = math.sqrt(100.0 * math.cosh(0.3) ** 2 + m_mu**2)
    pz2 = 10.0 * math.sinh(0.3)
    m_hand = math.sqrt((e1 + e2) ** 2 - pz2**2)  # px, py cancel (back-to-back)
    assert math.exp(float(ln_m[0])) == pytest.approx(m_hand, rel=1e-12)
    assert int(cat[0]) == 0  # both objects in region 0 -> category eta00
    # A pair straddling regions 0 and 2 -> sorted category (0, 2).
    obs2 = _mk_padded_obs([[4.0, 4.0]], [[211, 211]], [[0.2, -2.0]])
    obs2["phi"] = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    obs2["truth_pair_lnm:211"] = torch.tensor([math.log(91.0)], dtype=torch.float64)
    _, cat2, grp2 = compute_pair_masses(obs2)[211]
    assert int(cat2[0]) == 0 * N_SHAPE_ETA_REGIONS + 2
    assert int(grp2[0]) == 9  # ln 91 = 4.51 -> group [4.5, 5.0) = 90-148 GeV
    # A missing truth label for a class with reco pairs is an error, not a silent skip;
    # an event without a truth pair (NaN label) drops that pair.
    del obs2["truth_pair_lnm:211"]
    with pytest.raises(KeyError):
        compute_pair_masses(obs2)
    obs2["truth_pair_lnm:211"] = torch.tensor([float("nan")], dtype=torch.float64)
    assert compute_pair_masses(obs2) == {}


def _truth_from_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    """A padded ``(n_events, width, N_FEATURES)`` truth tensor whose PID / CHARGE / PT /
    ETA / PHI columns mirror ``obs`` (211/11/13 charged, 22 neutral), for
    :func:`attach_truth_pair_lnm` in the synthetic tests.
    """
    pt = obs["pt"]
    t = torch.zeros(*pt.shape, N_FEATURES, dtype=pt.dtype)
    t[..., ColumnMap.PID] = obs["pid"]
    t[..., ColumnMap.CHARGE] = (obs["pid"] != 22) & (obs["pid"] != 0)
    t[..., ColumnMap.PT] = pt
    t[..., ColumnMap.ETA] = obs["eta"]
    t[..., ColumnMap.PHI] = obs["phi"]
    return t


def _make_gun_obs(
    pts: list[float], seed: int = 0, truth_pts: list[float] | None = None
) -> dict[str, torch.Tensor]:
    """One back-to-back muon pair per event at eta = 0 (category eta00), both legs with
    the given reco pt, so the pair mass is ``sqrt(4 pt^2 + 2 m_mu^2)`` -- pt 1.55 ->
    J/psi-like 3.1 GeV, 45.5 -> Z-like 91 GeV. The truth pair label
    (``truth_pair_lnm:13``) is the same formula on ``truth_pts`` (default = ``pts``, i.e.
    response 0). Deterministic.
    """
    g = torch.Generator().manual_seed(seed)
    n = len(pts)
    pt = torch.zeros(n, 3, dtype=torch.float64)
    pt[:, 0] = torch.tensor(pts, dtype=torch.float64)
    pt[:, 1] = pt[:, 0]
    phi0 = torch.rand(n, generator=g, dtype=torch.float64) * math.tau
    phi = torch.zeros_like(pt)
    phi[:, 0] = phi0
    phi[:, 1] = torch.remainder(phi0 + math.pi, math.tau)
    eta = torch.zeros_like(pt)
    pid = torch.zeros_like(pt)
    pid[:, :2] = 13.0
    valid = pt != 0
    e = pt * torch.cosh(eta)
    tpt = torch.tensor(truth_pts if truth_pts is not None else pts, dtype=torch.float64)
    m_mu = 0.1056584
    return {
        "pt": pt,
        "eta": eta,
        "phi": phi,
        "pid": pid,
        "log_pt": torch.where(valid, torch.log(pt.clamp(min=1e-6)), torch.zeros_like(pt)),
        "log_E": torch.where(valid, torch.log(e.clamp(min=1e-6)), torch.zeros_like(pt)),
        "multiplicity": valid.sum(dim=1).to(pt.dtype),
        "ht": pt.sum(dim=1),
        "log_ht": torch.log(pt.sum(dim=1).clamp(min=1e-6)),
        "truth_pair_lnm:13": 0.5 * torch.log(4.0 * tpt**2 + 2.0 * m_mu**2),
    }


def test_pair_mass_terms_are_responses_grouped_by_truth_mass():
    """Pair terms compare the response ln(m_reco/m_truth), one term per (pid, truth-mass
    group, category): a J/psi-like + Z-like sample gives ``pair:13:mt2.72-4.48:eta00`` and
    ``pair:13:mt90-148:eta00``; the term weight is the target fraction of all pairs in the
    (pid, group, category); an identical response on both sides gives a ZERO term regardless of
    the J/psi : Z mixture fraction (the bimodal reco-mass mixture that widened ``a_raw``
    in the first M1 fit cannot enter); a momentum-scale shift lights both group terms up
    with the response centered at ln(scale).
    """
    tgt = _make_gun_obs([1.55] * 30 + [45.5] * 50, seed=1)  # 80 pairs
    pred = _make_gun_obs([1.55] * 50 + [45.5] * 10, seed=2)  # 60 pairs, other mixture
    _, comps = per_pid_wasserstein_1d_loss(
        pred, tgt, eta_split=False, pair_mass=True, pair_mass_weight=0.7, return_breakdown=True
    )
    pair = {c.label: c for c in comps if c.category == "pair"}
    assert set(pair) == {"pair:13:mt2.72-4.48:eta00", "pair:13:mt90-148:eta00"}
    assert pair["pair:13:mt2.72-4.48:eta00"].weight == pytest.approx(0.7 * 30 / 80)
    assert pair["pair:13:mt90-148:eta00"].weight == pytest.approx(0.7 * 50 / 80)
    assert all(abs(c.raw) < 1e-12 for c in pair.values())  # response 0 on both sides
    truth = [1.55] * 50 + [45.5] * 10
    shifted = _make_gun_obs([1.1 * t for t in truth], seed=2, truth_pts=truth)
    resp, _cat, grp = compute_pair_masses(shifted)[13]
    # not exactly ln 1.1: the muon-mass term in m^2 = 4 pt^2 + 2 m_mu^2 does not scale
    assert torch.allclose(resp, torch.full_like(resp, math.log(1.1)), atol=2e-3)
    assert set(grp.tolist()) == {2, 9}
    _, comps_s = per_pid_wasserstein_1d_loss(
        shifted, tgt, eta_split=False, pair_mass=True, pair_mass_weight=0.7, return_breakdown=True
    )
    pair_s = {c.label: c for c in comps_s if c.category == "pair"}
    assert set(pair_s) == set(pair)
    assert all(c.raw > 1e-3 for c in pair_s.values())


def test_pair_mass_terms_keys_weights_and_gradient():
    """Pair terms appear per (pid, window, category) present on both sides with weights
    summing to at most ``pair_mass_weight`` per pid, are absent with ``pair_mass=False``
    or weight 0, and carry a finite, non-zero gradient to the pred pt.
    """
    pred, tgt = _make_ref_obs(1, shift=0.05), _make_ref_obs(2)
    attach_truth_pair_lnm(_truth_from_obs(tgt), pred, tgt)  # shared truth = the target's
    pred["pt"] = pred["pt"].clone().requires_grad_(True)
    loss, comps = per_pid_wasserstein_1d_loss(
        pred, tgt, eta_split=False, pair_mass=True, pair_mass_weight=0.7, return_breakdown=True
    )
    pair = [c for c in comps if c.category == "pair"]
    assert pair, "no pair terms"
    # weights = target fractions over ALL pairs (every class); (group, category) cells
    # present on BOTH sides only, so the sum is <= pair_mass_weight
    assert 0.0 < sum(c.weight for c in pair) <= 0.7 + 1e-12
    assert all(re.fullmatch(r"pair:\d+:mt[\d.e+-]+-[\d.e+-]+:eta\d\d", c.label) for c in pair)
    loss.backward()
    g = pred["pt"].grad
    assert g is not None
    assert torch.isfinite(g).all()
    assert float(g.abs().sum()) > 0
    _, comps_off = per_pid_wasserstein_1d_loss(
        pred, tgt, eta_split=False, pair_mass=False, return_breakdown=True
    )
    assert not any(c.category == "pair" for c in comps_off)
    _, comps_w0 = per_pid_wasserstein_1d_loss(
        pred, tgt, eta_split=False, pair_mass=True, pair_mass_weight=0.0, return_breakdown=True
    )
    assert not any(c.category == "pair" for c in comps_w0)


def test_pair_mass_and_eta_split_one_step_gradient_is_finite(fixture_root: Path):
    """One real forward/backward through the trainee card with the full new loss
    (eta_split + pair_mass ON, both per-pid losses): finite loss, finite grads.
    """
    arrays = load_cms_flow_root(fixture_root, n_events=200)
    truth = load_truth_events(arrays)
    target = load_pflow_targets(arrays)
    for loss_fn in (per_pid_wasserstein_1d_loss, per_pid_soft_hist_loss):
        torch.manual_seed(3)
        card = CMSEnergyFlowDefault(debug=False, learnable=True)
        mask = torch.any(truth != 0, dim=-1)
        out = card(truth[mask])
        pred = load_pflow_targets_from_tensor(restore_event_format(out["EFlowObject"], mask))
        for out_key, pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
            pred[pred_key] = out[out_key]
        attach_truth_pair_lnm(truth, pred, target)
        loss, comps = loss_fn(pred, target, eta_split=True, pair_mass=True, return_breakdown=True)
        assert torch.isfinite(loss)
        assert any(":eta" in c.label for c in comps)
        assert any(c.category == "pair" for c in comps)
        loss.backward()
        for name, p in card.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
