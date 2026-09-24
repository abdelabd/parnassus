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
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    apply_chad_truncation,
    apply_reco_acceptance_cut,
    batch_event_ids,
    load_pflow_targets,
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
from parnassus.torch_delphes.tune_cms_fullsim.config import DEFAULT_MODE
from parnassus.torch_delphes.tune_cms_fullsim.runner import (
    AcceptanceCuts,
    resolve_acceptance_cuts,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

PSEUDODATA_PATH = Path(__file__).parent / "benchmark_data" / "cms_pseudodata.root"

# Fit-loop tests wait for the critic loss (critic_loss_plan.md step 5c).
_STEP5A_SKIP = "training loss removed in critic_loss_plan.md step 5a; re-enabled in step 5c"


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
    for name, v in tgt.items():
        assert torch.isfinite(v).all(), f"non-finite in target '{name}'"


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


@pytest.mark.skip(reason=_STEP5A_SKIP)
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
    """The target loader's reco cut applies to every class, and mult/ht are
    built from the cut set."""
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


@pytest.mark.skip(reason=_STEP5A_SKIP)
def test_fit_runs_with_acceptance_cuts(fixture_root: Path, tmp_path: Path):
    """End-to-end: loaders with cuts + truncation, fit hooks on -- a short fit
    stays finite."""
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
        n_steps=2,
        log_every=0,
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



# ---------------------------------------------------------------------------
# Real-Pythia pseudodata end-to-end test
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason=_STEP5A_SKIP)
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
