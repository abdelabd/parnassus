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
from parnassus.torch_delphes.tune_cms_fullsim.data import (
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
)
from parnassus.torch_delphes.tune_cms_fullsim.loss import (
    LOSS_CHOICES,
    PID_WEIGHTING_CHOICES,
    _count_terms,
    _pid_population_weights,
    get_loss_fn,
    per_event_wasserstein_loss,
    per_pid_soft_hist_loss,
    per_pid_wasserstein_1d_loss,
    quantile_wasserstein_distance,
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

    # Targets: only directly comparable when no truth event was skipped (empty
    # truth events would desync the dataset's truth-count from the full target
    # count -- a pre-existing behavior we deliberately preserve).
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
    """The dispatcher maps each loss name to the right callable (no data needed)."""
    assert "wasserstein" in LOSS_CHOICES and "soft_hist" in LOSS_CHOICES
    assert "wasserstein_1d" in LOSS_CHOICES
    assert get_loss_fn("wasserstein") is per_event_wasserstein_loss
    assert get_loss_fn("soft_hist") is per_pid_soft_hist_loss
    assert get_loss_fn("wasserstein_1d") is per_pid_wasserstein_1d_loss
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
    finite, nonzero gradient (their only gradient path is the shape term), and under the
    aggressive 'fraction' mode that same gradient is SMALLER -- confirming the per-pid
    weight actually reaches the parameter gradient."""
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
        loss = per_pid_wasserstein_1d_loss(pred, target, pid_weighting=mode)
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
