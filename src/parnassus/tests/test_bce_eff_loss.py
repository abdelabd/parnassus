"""Tests for the BCE survival efficiency loss (--eff-loss bce; EFF_LOSS_PLAN.md).

Covers, without any ROOT I/O:

1. **Closed form**: minimizing the BCE terms alone recovers the per-region
   empirical survival fraction (the Bernoulli MLE) when eps is a per-region
   sigmoid.
2. **Pooled vs per_species weighting**: same minimizer, different scale; the
   pooled sum equals the flat per-particle BCE over all tracks.
3. **Gradient routing**: BCE gradients land on eps and nothing else; an empty
   species is a graph-connected zero.
4. **Loss assembly**: with eff_loss="bce" the wasserstein_1d loss drops the three
   tracking count terms, keeps the calo count terms, and adds the bce components;
   with eff_loss="counts" it is bit-identical to the pre-change behavior.
5. **Labels and export**: the Hungarian matcher's per-event labels; the label
   builder is row-aligned with the truth rows under the acceptance cuts; the
   learnable card's ``TrackSurvivalExport`` is keyed by input row, carries
   gradient to the efficiency parameters, and ``_inject_track_bce`` pairs it with
   the row-aligned labels.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault
from parnassus.torch_delphes.learnable import CMS_EFF_REGION_SPECS
from parnassus.torch_delphes.tune_cms_fullsim.data import (
    _build_survival_labels,
    _build_truth_rows,
    match_event,
)
from parnassus.torch_delphes.tune_cms_fullsim.loss import BCE_SPECIES, _bce_eff_terms
from parnassus.torch_delphes.tune_cms_fullsim.training import _inject_track_bce

RNG = np.random.default_rng(7)
_SPEC_OF = {"chad": "charged_hadron", "electron": "electron", "muon": "muon"}


def _make_tracks(n: int = 20000) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict]:
    """Random per-species (region, survived) tracks with known per-region survival
    probabilities. Returns (region_by_species, x_by_species, prob_by_species)."""
    regions, xs, probs = {}, {}, {}
    for key in BCE_SPECIES:
        n_r = CMS_EFF_REGION_SPECS[_SPEC_OF[key]].n_regions
        probs[key] = torch.from_numpy(RNG.uniform(0.2, 0.95, size=n_r))
        regions[key] = torch.from_numpy(RNG.integers(0, n_r, size=n // 3)).long()
        xs[key] = (torch.rand(n // 3, dtype=torch.float64) < probs[key][regions[key]]).double()
    return regions, xs, probs


def _fresh_logits() -> dict[str, torch.Tensor]:
    return {
        key: torch.zeros(
            CMS_EFF_REGION_SPECS[_SPEC_OF[key]].n_regions, dtype=torch.float64, requires_grad=True
        )
        for key in BCE_SPECIES
    }


def _dicts(logits, regions, xs) -> tuple[dict, dict]:
    """pred["bce_eps:k"] = sigmoid(logits_k)[region], target["bce_x:k"] = x."""
    pred = {f"bce_eps:{k}": torch.sigmoid(logits[k])[regions[k]] for k in BCE_SPECIES}
    target = {f"bce_x:{k}": xs[k] for k in BCE_SPECIES}
    return pred, target


def test_bce_recovers_empirical_survival_fractions():
    regions, xs, _ = _make_tracks()
    logits = _fresh_logits()
    opt = torch.optim.Adam(list(logits.values()), lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        pred, target = _dicts(logits, regions, xs)
        loss = torch.stack(_bce_eff_terms(pred, target, bce_weight=1.0)).sum()
        loss.backward()
        opt.step()
    for key in BCE_SPECIES:
        eff = torch.sigmoid(logits[key]).detach()
        for r in range(eff.numel()):
            sel = regions[key] == r
            if not sel.any():
                continue
            frac = float(xs[key][sel].mean())
            assert abs(float(eff[r]) - frac) < 0.01, (
                f"{key} region {r}: fitted {float(eff[r]):.4f} vs empirical {frac:.4f}"
            )


def test_pooled_sum_equals_flat_bce_and_per_species_differs():
    regions, xs, _ = _make_tracks(5000)
    logits = _fresh_logits()
    for t in logits.values():
        with torch.no_grad():
            t.normal_(0.3, 0.5)
    pred, target = _dicts(logits, regions, xs)

    pooled = torch.stack(
        _bce_eff_terms(pred, target, bce_weight=1.0, bce_weighting="pooled")
    ).sum()
    flat = F.binary_cross_entropy(
        torch.cat([pred[f"bce_eps:{k}"] for k in BCE_SPECIES]),
        torch.cat([target[f"bce_x:{k}"] for k in BCE_SPECIES]),
        reduction="mean",
    )
    assert torch.allclose(pooled, flat, rtol=1e-12)

    per_species = torch.stack(
        _bce_eff_terms(pred, target, bce_weight=1.0, bce_weighting="per_species")
    ).sum()
    assert not torch.allclose(pooled, per_species)  # different combination rule

    with pytest.raises(ValueError):
        _bce_eff_terms(pred, target, bce_weight=1.0, bce_weighting="nope")


def test_gradients_only_on_eps_and_empty_species_is_graph_connected():
    regions, xs, _ = _make_tracks(2000)
    # Only chads carry tracks: electron/muon get the graph-connected zero.
    for key in ("electron", "muon"):
        regions[key], xs[key] = regions[key][:0], xs[key][:0]
    logits = _fresh_logits()
    pred, target = _dicts(logits, regions, xs)
    other = torch.randn(5, dtype=torch.float64, requires_grad=True)
    pred["not_bce"] = other

    loss = torch.stack(_bce_eff_terms(pred, target, bce_weight=1.0)).sum()
    loss.backward()
    assert logits["chad"].grad is not None and logits["chad"].grad.abs().sum() > 0
    # Empty species: zero grad but graph-connected (grad tensor exists).
    assert logits["muon"].grad is not None
    assert float(logits["muon"].grad.abs().sum()) == 0.0
    assert other.grad is None


def _toy_loss_dicts(eff_loss_labels: bool):
    """Minimal pred/target dicts accepted by per_pid_wasserstein_1d_loss."""
    n_ev, n_p = 8, 6
    g = torch.Generator().manual_seed(3)
    mk = lambda: torch.rand((n_ev, n_p), dtype=torch.float64, generator=g) + 0.5

    def side(requires_grad: bool):
        pt = mk()
        if requires_grad:
            pt = pt.clone().requires_grad_(True)
        d = {
            "pt": pt,
            "eta": mk() - 1.0,
            "phi": mk() - 1.0,
            "log_pt": mk(),
            "log_E": mk(),
            "pid": torch.full((n_ev, n_p), 211.0, dtype=torch.float64),
            "log_ht": mk()[:, 0],
            "multiplicity": torch.full((n_ev,), float(n_p), dtype=torch.float64),
            "ht": mk()[:, 0] * n_p,
        }
        return d

    pred = side(True)
    target = side(False)
    # Tracking + calo count-term inputs.
    for spec_key, pred_key, tgt_key in (
        ("charged_hadron", "chad_expected_counts", "chad_region_counts"),
        ("electron", "electron_expected_counts", "electron_region_counts"),
        ("muon", "muon_expected_counts", "muon_region_counts"),
    ):
        n_r = CMS_EFF_REGION_SPECS[spec_key].n_regions
        pred[pred_key] = (mk()[0, :n_r]).clone().requires_grad_(True)
        target[tgt_key] = mk()[:, :n_r]
    for pred_key, tgt_key, n_r in (
        ("ecal_photon_expected_counts", "ecal_photon_region_counts", 3),
        ("hcal_neutral_hadron_expected_counts", "hcal_nh_region_counts", 2),
    ):
        pred[pred_key] = (mk()[0, :n_r]).clone().requires_grad_(True)
        target[tgt_key] = mk()[:, :n_r]
    if eff_loss_labels:
        regions, xs, _ = _make_tracks(500)
        p, t = _dicts(_fresh_logits(), regions, xs)
        pred.update(p)
        target.update(t)
    return pred, target


def test_assembly_bce_drops_tracking_counts_keeps_calo():
    from parnassus.torch_delphes.tune_cms_fullsim.loss import per_pid_wasserstein_1d_loss

    pred, target = _toy_loss_dicts(eff_loss_labels=True)
    _total, comps = per_pid_wasserstein_1d_loss(
        pred, target, eff_loss="bce", pair_mass=False, return_breakdown=True
    )
    labels_by_cat: dict[str, list[str]] = {}
    for c in comps:
        labels_by_cat.setdefault(c.category, []).append(c.label)
    assert sorted(labels_by_cat["bce"]) == ["BceChargedHadron", "BceElectron", "BceMuon"]
    counts = labels_by_cat["count"]
    assert all("Calo" in l or "Ecal" in l or "Hcal" in l for l in counts), counts
    assert len(counts) == 2  # calo only; tracking count terms dropped

    # eff_loss="counts" on the same dicts: 5 count terms, no bce category.
    pred2, target2 = _toy_loss_dicts(eff_loss_labels=True)
    _t2, comps2 = per_pid_wasserstein_1d_loss(
        pred2, target2, eff_loss="counts", pair_mass=False, return_breakdown=True
    )
    cats2 = {c.category for c in comps2}
    assert "bce" not in cats2
    assert sum(1 for c in comps2 if c.category == "count") == 5


# ---------------------------------------------------------------------------
# 5. Hungarian labels + the card's per-track survival export
# ---------------------------------------------------------------------------


def test_match_event_within_gate_and_species_toggle():
    # truth: chad at (0,0), chad at (1,1), muon at (2,2); reco: chad near (0,0), chad far,
    # chad near (2,2) [wrong species for the muon].
    t_eta = np.array([0.0, 1.0, 2.0]); t_phi = np.array([0.0, 1.0, 2.0]); t_cls = np.array([0, 0, 2])
    r_eta = np.array([0.01, 1.5, 2.0]); r_phi = np.array([0.0, 1.0, 2.0]); r_cls = np.array([0, 0, 0])
    for matching in ("hungarian", "nn"):
        # default: across species -> the muon is matched to the reco chad on top of it
        got, reco = match_event(t_eta, t_phi, t_cls, r_eta, r_phi, r_cls, 0.05, matching)
        assert got.tolist() == [True, False, True] and reco.tolist() == [True, False, True]
        # within species: the muon has no reco muon -> lost; that reco chad is a fake
        got, reco = match_event(t_eta, t_phi, t_cls, r_eta, r_phi, r_cls, 0.05, matching, True)
        assert got.tolist() == [True, False, False] and reco.tolist() == [True, False, False]
    # One-to-one: two truth chads at the same spot, one reco -> exactly one survives.
    got, _ = match_event(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.array([0, 0]),
        np.array([0.0]), np.array([0.0]), np.array([0]), max_dr=0.05,
    )
    assert int(got.sum()) == 1
    # Two truth chads A (0,0), B (0.05,0); two reco chads r1 at 0.01 and r2 at 0.02
    # from A (0.03 from B, inside the gate). Hungarian assigns r2 to B (both
    # survive); nn lets both reco objects claim A (B does not survive).
    args = (np.array([0.0, 0.05]), np.zeros(2), np.zeros(2, dtype=int),
            np.array([0.01, 0.02]), np.zeros(2), np.zeros(2, dtype=int))
    assert match_event(*args, matching="hungarian")[0].tolist() == [True, True]
    assert match_event(*args, matching="nn")[0].tolist() == [True, False]


def _toy_arrays(n_events: int = 6, seed: int = 0) -> dict[str, np.ndarray]:
    """Per-event ragged truth/pflow arrays in the uproot ``library="np"`` layout:
    charged hadrons, electrons, muons and photons; reco = a subset of the truth
    tracks at their truth direction (plus one fake)."""
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ("truth_pt", "truth_eta", "truth_phi", "truth_class", "truth_pdgid",
                            "pflow_pt", "pflow_eta", "pflow_phi", "pflow_class")}
    pdg_of = {0: 211, 1: 11, 2: 13, 3: 22}
    for _ in range(n_events):
        n = int(rng.integers(3, 12))
        cls = rng.integers(0, 4, size=n)
        pt = rng.uniform(0.2, 50.0, size=n); eta = rng.uniform(-2.8, 2.8, size=n)
        phi = rng.uniform(-np.pi, np.pi, size=n)
        keep = rng.random(n) < 0.7
        for k, v in (("truth_pt", pt), ("truth_eta", eta), ("truth_phi", phi), ("truth_class", cls),
                     ("truth_pdgid", np.array([pdg_of[int(c)] for c in cls])),
                     ("pflow_pt", np.append(pt[keep], 3.0)), ("pflow_eta", np.append(eta[keep], 0.3)),
                     ("pflow_phi", np.append(phi[keep], 0.3)), ("pflow_class", np.append(cls[keep], 0))):
            cols[k].append(np.asarray(v))
    return {k: np.array(v, dtype=object) for k, v in cols.items()}


def test_survival_labels_align_with_truth_rows_under_cuts():
    arrays = _toy_arrays()
    for cuts in ({}, {"truth_pt_cut": 1.0, "abs_eta_cut": 2.5}):
        rows = _build_truth_rows(arrays, **cuts)
        labels = _build_survival_labels(arrays, **cuts)
        assert [r.shape[0] for r in rows] == [x.shape[0] for x in labels]
    # No cuts: a matched truth track is exactly one that has a same-class reco object at
    # its position (the toy reco is a subset of the truth tracks + one fake). With a
    # reco cut, the reco list is cut BEFORE matching: a track reconstructed below the
    # cut does not count as survived (the toy reco pt equals the truth pt).
    for reco_pt_cut in (None, 10.0):
        labels = _build_survival_labels(arrays, reco_pt_cut=reco_pt_cut)
        for i, x in enumerate(labels):
            charged = arrays["truth_class"][i] < 3
            in_reco = np.isin(arrays["truth_eta"][i], arrays["pflow_eta"][i])
            above = arrays["truth_pt"][i] >= (reco_pt_cut or 0.0)
            assert np.array_equal(x.numpy().astype(bool), charged & in_reco & above)


def test_track_survival_export_is_row_keyed_and_differentiable():
    """The card's TrackSurvivalExport: one eps per charged track that reached the
    tracker, keyed by its input-row UID; eps in [0, 1] with gradient to the
    efficiency parameters (including the muon roll-off rate); _inject_track_bce
    pairs it with the row-aligned survival labels."""
    torch.manual_seed(0)
    arrays = _toy_arrays(n_events=20, seed=1)
    rows = torch.cat([torch.from_numpy(r) for r in _build_truth_rows(arrays)])
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    out = card(rows)
    exp = out["TrackSurvivalExport"]
    uids = torch.cat([exp[f"uid:{k}"] for k in BCE_SPECIES]).long()
    assert uids.unique().numel() == uids.numel()  # one export per input row at most
    assert uids.min() >= 0 and uids.max() < rows.shape[0]
    pids = rows[uids, 0].abs()  # ColumnMap.PID == 0
    assert set(pids.long().tolist()) <= {211, 11, 13}  # only charged tracks
    for key, pid in (("chad", 211), ("electron", 11), ("muon", 13)):
        assert torch.all(rows[exp[f"uid:{key}"].long(), 0].abs() == pid)
        eps = exp[f"eps:{key}"]
        assert torch.all((eps >= 0) & (eps <= 1))
    torch.cat([exp[f"eps:{k}"] for k in BCE_SPECIES]).sum().backward()
    assert card.ChargedHadronTrackingEfficiency.eff_logits.grad.abs().sum() > 0
    assert card.MuonTrackingEfficiency.eff_logits.grad.abs().sum() > 0

    # Pairing: labels are per input row; the injector gathers them by UID.
    batch = {"bce_x": torch.arange(rows.shape[0], dtype=torch.float64).unsqueeze(0)}
    mask = torch.ones((1, rows.shape[0]), dtype=torch.bool)
    pred, target = {}, {}
    _inject_track_bce(pred, target, out, batch, mask)
    for key in BCE_SPECIES:
        assert torch.equal(target[f"bce_x:{key}"], exp[f"uid:{key}"].to(torch.float64))
        assert pred[f"bce_eps:{key}"] is exp[f"eps:{key}"]


def test_tower_bce_terms_region_fair_and_gradients():
    """--calo-bce term: per-region-fair combination, log-space stability at the
    q -> 0 and q -> 1 extremes (incl. log_q == -0.0 exactly), gradient flow."""
    from parnassus.torch_delphes.tune_cms_fullsim.loss import _log1mexp, _tower_bce_terms

    # log1mexp edge cases: exact within float64 across the whole range.
    lq = torch.log(torch.tensor([0.3, 0.9, 1e-40, 1 - 1e-12], dtype=torch.float64))
    assert torch.allclose(
        torch.exp(_log1mexp(lq)),
        torch.tensor([0.7, 0.1, 1.0, 1e-12], dtype=torch.float64),
        rtol=1e-6,
    )
    # log_q == -0.0 exactly (saturated log_ndtr): finite value, finite gradient.
    lq0 = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    out = _log1mexp(lq0 * 1.0)
    out.sum().backward()
    assert torch.isfinite(out).all() and torch.isfinite(lq0.grad).all()

    # Region-fair combination: two regions with different tower counts must
    # weigh equally. Region 0: 3 towers; region 1: 1 tower.
    base = torch.tensor([-0.5, -1.0, -1.5, -2.0], dtype=torch.float64, requires_grad=True)
    pred = {
        "tower_logq:ecal": base,
        "tower_region:ecal": torch.tensor([0, 0, 0, 1]),
    }
    target = {"tower_x:ecal": torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)}
    (term,) = _tower_bce_terms(pred, target, calo_bce_weight=2.0)
    per = -(target["tower_x:ecal"] * base.detach() + (1 - target["tower_x:ecal"]) * _log1mexp(base.detach()))
    expected = 2.0 * 0.5 * (per[:3].mean() + per[3])
    assert torch.allclose(term, expected, rtol=1e-12)
    term.backward()
    assert torch.isfinite(base.grad).all() and base.grad.abs().sum() > 0

    # Empty calo: graph-connected zero.
    pred_e = {
        "tower_logq:hcal": base[:0],
        "tower_region:hcal": torch.zeros(0, dtype=torch.long),
    }
    target_e = {"tower_x:hcal": torch.zeros(0, dtype=torch.float64)}
    (zt,) = _tower_bce_terms(pred_e, target_e, calo_bce_weight=1.0)
    assert float(zt) == 0.0 and zt.grad_fn is not None
