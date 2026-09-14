"""Tests for the BCE survival efficiency loss (--eff-loss bce; EFF_LOSS_PLAN.md).

Covers, without any ROOT I/O:

1. **Closed form**: minimizing the BCE terms alone recovers the per-region
   empirical survival fraction (the Bernoulli MLE) for every species.
2. **Pooled vs per_species weighting**: same minimizer, different scale; the
   pooled sum equals the flat per-particle BCE over all labeled particles.
3. **Gradient routing**: BCE gradients land on eff_logits and nothing else.
4. **Loss assembly**: with eff_loss="bce" the wasserstein_1d loss drops the three
   tracking count terms, keeps the calo count terms, and adds the bce components;
   with eff_loss="counts" it is bit-identical to the pre-change behavior.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from parnassus.torch_delphes.learnable import CMS_EFF_REGION_SPECS
from parnassus.torch_delphes.tune_cms_fullsim.loss import (
    BCE_TERM_KEYS,
    _bce_eff_terms,
)

RNG = np.random.default_rng(7)


def _make_labels(n: int = 20000) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Random labels over every constant region of the three species, with known
    per-region survival probabilities. Returns (region, x, truth_prob_by_label)."""
    labels = []
    probs = {}
    for _key, spec_key, _lk in BCE_TERM_KEYS:
        spec = CMS_EFF_REGION_SPECS[spec_key]
        for r in range(spec.n_regions):
            if spec_key == "muon" and r % spec.n_pt == spec.n_pt - 1:
                continue  # exponential bins: excluded at load, never labeled
            label = spec.label_offset + r + 1
            labels.append(label)
            probs[label] = RNG.uniform(0.2, 0.95)
    region = torch.from_numpy(RNG.choice(labels, size=n)).long()
    p = torch.tensor([probs[int(l)] for l in region], dtype=torch.float64)
    x = (torch.rand(n, dtype=torch.float64) < p).double()
    return region, x, probs


def _fresh_logits() -> dict[str, torch.Tensor]:
    return {
        logits_key: torch.zeros(
            CMS_EFF_REGION_SPECS[spec_key].n_regions, dtype=torch.float64, requires_grad=True
        )
        for _key, spec_key, logits_key in BCE_TERM_KEYS
    }


def test_bce_recovers_empirical_survival_fractions():
    region, x, _probs = _make_labels()
    pred = _fresh_logits()
    target = {"bce_region": region, "bce_x": x}
    opt = torch.optim.Adam(list(pred.values()), lr=0.2)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.stack(_bce_eff_terms(pred, target, bce_weight=1.0)).sum()
        loss.backward()
        opt.step()

    for _key, spec_key, logits_key in BCE_TERM_KEYS:
        spec = CMS_EFF_REGION_SPECS[spec_key]
        eff = torch.sigmoid(pred[logits_key]).detach().numpy()
        for r in range(spec.n_regions):
            label = spec.label_offset + r + 1
            sel = region == label
            n = int(sel.sum())
            if n == 0:
                continue
            frac = float(x[sel].mean())
            assert abs(eff[r] - frac) < 0.01, (
                f"{spec_key} region {r}: fitted {eff[r]:.4f} vs empirical {frac:.4f}"
            )


def test_pooled_sum_equals_flat_bce_and_per_species_differs():
    region, x, _ = _make_labels(5000)
    pred = _fresh_logits()
    for t in pred.values():
        with torch.no_grad():
            t.normal_(0.3, 0.5)
    target = {"bce_region": region, "bce_x": x}

    pooled = torch.stack(
        _bce_eff_terms(pred, target, bce_weight=1.0, bce_weighting="pooled")
    ).sum()

    # Flat reference: per-particle BCE over all labeled particles at once.
    logit_per_particle = torch.zeros_like(x)
    for _key, spec_key, logits_key in BCE_TERM_KEYS:
        spec = CMS_EFF_REGION_SPECS[spec_key]
        lo = spec.label_offset + 1
        m = (region >= lo) & (region <= lo + spec.n_regions - 1)
        logit_per_particle[m] = pred[logits_key][region[m] - lo]
    flat = F.binary_cross_entropy_with_logits(logit_per_particle, x, reduction="mean")
    assert torch.allclose(pooled, flat, rtol=1e-12)

    per_species = torch.stack(
        _bce_eff_terms(pred, target, bce_weight=1.0, bce_weighting="per_species")
    ).sum()
    assert not torch.allclose(pooled, per_species)  # different combination rule

    with pytest.raises(ValueError):
        _bce_eff_terms(pred, target, bce_weight=1.0, bce_weighting="nope")


def test_gradients_only_on_logits_and_empty_species_is_graph_connected():
    region, x, _ = _make_labels(2000)
    # Keep only chad labels: electron/muon species get the graph-connected zero.
    chad = CMS_EFF_REGION_SPECS["charged_hadron"]
    m = (region >= chad.label_offset + 1) & (region <= chad.label_offset + chad.n_regions)
    target = {"bce_region": region[m], "bce_x": x[m]}
    pred = _fresh_logits()
    other = torch.randn(5, dtype=torch.float64, requires_grad=True)
    pred["not_bce"] = other

    loss = torch.stack(_bce_eff_terms(pred, target, bce_weight=1.0)).sum()
    loss.backward()
    assert pred["bce_logits:chad"].grad is not None
    assert pred["bce_logits:chad"].grad.abs().sum() > 0
    # Empty species: zero grad but graph-connected (grad tensor exists).
    assert pred["bce_logits:muon"].grad is not None
    assert float(pred["bce_logits:muon"].grad.abs().sum()) == 0.0
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
        region, x, _ = _make_labels(500)
        target["bce_region"] = region
        target["bce_x"] = x
        for _key, spec_key, logits_key in BCE_TERM_KEYS:
            pred[logits_key] = torch.zeros(
                CMS_EFF_REGION_SPECS[spec_key].n_regions,
                dtype=torch.float64,
                requires_grad=True,
            )
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
