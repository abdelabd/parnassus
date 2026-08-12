"""PhotonClusterMerger unit tests (milestone M0 of the photon-merger design).

Covers the M0 checklist from torch_delphes/.claude/docs/
photon_merger_fraction_design.md: equivalence with a sequential numpy
reference (including pt ties), four-momentum conservation, phi wraparound,
identity on isolated photons, cross-event isolation, chain no-snowballing,
higher-pt-seed ownership, gradient flow through merged kinematics, and
card-level consistency with the merger-off baseline.
"""

import math
import types

import numpy as np
import pytest
import torch

from parnassus.data.particle_io import ColumnMap, N_FEATURES
from parnassus.torch_delphes.defaults.CMSDefault import CMSEnergyFlowDefault
from parnassus.torch_delphes.PhotonClusterMerger import (
    PhotonClusterMerger,
    compose_merged_photon_count,
)

RADIUS = 0.045
FOUR_COLS = [ColumnMap.E, ColumnMap.PX, ColumnMap.PY, ColumnMap.PZ]


def make_photons(specs) -> torch.Tensor:
    """Build a flat photon tensor from (event, pt, eta, phi) rows (massless)."""
    out = torch.zeros(len(specs), N_FEATURES, dtype=torch.float64)
    for r, (ev, pt, eta, phi) in enumerate(specs):
        out[r, ColumnMap.PID] = 22.0
        out[r, ColumnMap.STATUS] = 1.0
        out[r, ColumnMap.E] = pt * math.cosh(eta)
        out[r, ColumnMap.PX] = pt * math.cos(phi)
        out[r, ColumnMap.PY] = pt * math.sin(phi)
        out[r, ColumnMap.PZ] = pt * math.sinh(eta)
        out[r, ColumnMap.PT] = pt
        out[r, ColumnMap.ETA] = eta
        out[r, ColumnMap.PHI] = phi
        out[r, ColumnMap.ETA_OUTER] = eta
        out[r, ColumnMap.PHI_OUTER] = phi
        out[r, ColumnMap.EVENT_NUMBER] = ev
    return out


def reference_owner(photons: torch.Tensor, radius: float = RADIUS) -> np.ndarray:
    """Sequential greedy seed-cone reference (per event), returns owner rows."""
    ev = photons[:, ColumnMap.EVENT_NUMBER].numpy()
    pt = photons[:, ColumnMap.PT].numpy()
    eta = photons[:, ColumnMap.ETA].numpy()
    phi = photons[:, ColumnMap.PHI].numpy()
    owner = np.full(len(pt), -1)
    for e in np.unique(ev):
        rows = np.flatnonzero(ev == e)
        claimed = np.zeros(len(rows), dtype=bool)
        for j in np.argsort(-pt[rows], kind="stable"):
            if claimed[j]:
                continue
            dphi = np.abs(phi[rows] - phi[rows[j]])
            dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
            dr2 = (eta[rows] - eta[rows[j]]) ** 2 + dphi**2
            members = (dr2 < radius**2) & ~claimed
            members[j] = True
            owner[rows[members]] = rows[j]
            claimed |= members
    return owner


def clusters_from_owner(owner) -> set[frozenset]:
    groups: dict[int, set[int]] = {}
    for i, o in enumerate(np.asarray(owner)):
        groups.setdefault(int(o), set()).add(i)
    return {frozenset(v) for v in groups.values()}


def test_matches_sequential_reference_random():
    """Vectorized round-based assignment == sequential greedy, on dense blobs."""
    rng = np.random.default_rng(42)
    for trial in range(10):
        specs = []
        for ev in range(rng.integers(1, 6)):
            for _ in range(rng.integers(0, 8)):  # blob per event
                ceta = rng.uniform(-2.0, 2.0)
                cphi = rng.uniform(-np.pi, np.pi)
                for _ in range(rng.integers(1, 9)):
                    specs.append((
                        ev,
                        float(rng.uniform(1.0, 100.0)),
                        ceta + rng.normal(0, 0.03),
                        cphi + rng.normal(0, 0.03),
                    ))
        if not specs:
            continue
        photons = make_photons(specs)
        merger = PhotonClusterMerger(RADIUS)
        owner = merger._assign_clusters(photons).numpy()
        ref = reference_owner(photons)
        assert clusters_from_owner(owner) == clusters_from_owner(ref), f"trial {trial}"
        assert np.array_equal(owner, ref), f"trial {trial}: seed choice differs"


def test_pt_ties_break_by_row_order():
    """Equal-pt photons within one cone: the earlier row seeds."""
    photons = make_photons([(0, 10.0, 0.0, 0.0), (0, 10.0, 0.02, 0.0)])
    owner = PhotonClusterMerger(RADIUS)._assign_clusters(photons)
    assert owner.tolist() == [0, 0]


def test_four_momentum_conservation_per_event():
    rng = np.random.default_rng(7)
    specs = [
        (ev, float(rng.uniform(1, 50)), float(rng.uniform(-2, 2)), float(rng.uniform(-np.pi, np.pi)))
        for ev in range(4)
        for _ in range(30)
    ]
    photons = make_photons(specs)
    merged = PhotonClusterMerger(RADIUS)(photons)
    for ev in range(4):
        before = photons[photons[:, ColumnMap.EVENT_NUMBER] == ev][:, FOUR_COLS].sum(dim=0)
        after = merged[merged[:, ColumnMap.EVENT_NUMBER] == ev][:, FOUR_COLS].sum(dim=0)
        assert torch.allclose(before, after, rtol=1e-12, atol=1e-12)


def test_phi_wraparound_merges_across_boundary():
    photons = make_photons([(0, 20.0, 1.0, np.pi - 0.01), (0, 5.0, 1.0, -np.pi + 0.01)])
    merged = PhotonClusterMerger(RADIUS)(photons)
    assert merged.shape[0] == 1
    assert abs(abs(float(merged[0, ColumnMap.PHI])) - np.pi) < 0.02
    assert torch.allclose(merged[0, FOUR_COLS], photons[:, FOUR_COLS].sum(dim=0))


def test_isolated_photons_pass_through_bit_identical():
    photons = make_photons([(0, 5.0, -1.0, 0.0), (0, 8.0, 0.0, 2.0), (1, 3.0, 1.5, -2.0)])
    merged = PhotonClusterMerger(RADIUS)(photons)
    assert torch.equal(merged, photons)


def test_cross_event_isolation():
    """Same positions in different events never merge."""
    specs = [(ev, pt, 0.5, 0.5) for ev in (0, 1, 2) for pt in (10.0, 4.0)]
    merged = PhotonClusterMerger(RADIUS)(make_photons(specs))
    assert merged.shape[0] == 3  # one in-event merge each, none across
    assert sorted(merged[:, ColumnMap.EVENT_NUMBER].tolist()) == [0.0, 1.0, 2.0]


def test_chain_does_not_snowball():
    """A-B close, B-C close, A-C far: cone anchored to seed A keeps C out."""
    photons = make_photons([(0, 30.0, 0.0, 0.0), (0, 20.0, 0.04, 0.0), (0, 10.0, 0.08, 0.0)])
    owner = PhotonClusterMerger(RADIUS)._assign_clusters(photons)
    assert owner.tolist() == [0, 0, 2]


def test_overlap_goes_to_higher_pt_seed():
    """A member in reach of two seeds joins the higher-pt one."""
    photons = make_photons([
        (0, 100.0, 0.00, 0.0),  # seed 1
        (0, 50.0, 0.06, 0.0),   # seed 2 (out of seed 1's cone)
        (0, 1.0, 0.03, 0.0),    # within R of both
    ])
    owner = PhotonClusterMerger(RADIUS)._assign_clusters(photons)
    assert owner.tolist() == [0, 1, 0]


def test_empty_and_single_row_passthrough():
    merger = PhotonClusterMerger(RADIUS)
    empty = torch.zeros(0, N_FEATURES, dtype=torch.float64)
    assert merger(empty) is empty
    one = make_photons([(0, 5.0, 0.0, 0.0)])
    assert merger(one) is one


def test_gradient_flows_through_merged_kinematics():
    """Upstream params (here: a global energy scale) get gradients through
    merged four-vectors; conservation makes dE_sum/dscale the unscaled total."""
    base = make_photons([
        (0, 10.0, 0.0, 0.0), (0, 5.0, 0.02, 0.0), (0, 3.0, 1.0, 1.0),
        (1, 7.0, -0.5, 0.1), (1, 2.0, -0.51, 0.12),
    ])
    scale = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    photons = base.clone()
    photons[:, FOUR_COLS] = base[:, FOUR_COLS] * scale
    merged = PhotonClusterMerger(RADIUS)(photons)
    assert merged.shape[0] == 3
    loss = merged[:, ColumnMap.E].sum()
    loss.backward()
    assert torch.isfinite(merged).all()
    assert scale.grad is not None and torch.isfinite(scale.grad)
    assert torch.allclose(scale.grad, base[:, ColumnMap.E].sum())


# ---------------------------------------------------------------------------
# Card integration
# ---------------------------------------------------------------------------


def _make_truth_particles(n_events: int = 3) -> torch.Tensor:
    """Synthetic truth: per event, a tight photon 'jet core' (forces merging)
    plus scattered photons and charged pions."""
    rng = np.random.default_rng(5)
    rows = []
    for ev in range(n_events):
        ceta, cphi = rng.uniform(-1.0, 1.0), rng.uniform(-2.0, 2.0)
        for _ in range(8):  # dense core photons
            rows.append((22, 0.0, 0.0, float(rng.uniform(5, 30)),
                         ceta + rng.normal(0, 0.02), cphi + rng.normal(0, 0.02), ev))
        for _ in range(4):  # isolated photons
            rows.append((22, 0.0, 0.0, float(rng.uniform(2, 10)),
                         float(rng.uniform(-2, 2)), float(rng.uniform(-3, 3)), ev))
        for _ in range(6):  # charged pions
            rows.append((211 * int(rng.choice([-1, 1])), 1.0, 0.13957,
                         float(rng.uniform(1, 20)),
                         float(rng.uniform(-2, 2)), float(rng.uniform(-3, 3)), ev))
    out = torch.zeros(len(rows), N_FEATURES, dtype=torch.float64)
    for r, (pid, charge, mass, pt, eta, phi) in enumerate([(p, c, m, pt, e, ph) for p, c, m, pt, e, ph, _ in rows]):
        out[r, ColumnMap.PID] = pid
        out[r, ColumnMap.STATUS] = 1.0
        out[r, ColumnMap.CHARGE] = math.copysign(charge, pid) if charge else 0.0
        out[r, ColumnMap.PT] = pt
        out[r, ColumnMap.ETA] = eta
        out[r, ColumnMap.PHI] = phi
        out[r, ColumnMap.PX] = pt * math.cos(phi)
        out[r, ColumnMap.PY] = pt * math.sin(phi)
        out[r, ColumnMap.PZ] = pt * math.sinh(eta)
        p2 = float(out[r, ColumnMap.PX] ** 2 + out[r, ColumnMap.PY] ** 2 + out[r, ColumnMap.PZ] ** 2)
        out[r, ColumnMap.E] = math.sqrt(p2 + mass**2)
        out[r, ColumnMap.MASS] = mass
    out[:, ColumnMap.EVENT_NUMBER] = torch.tensor([row[-1] for row in rows], dtype=torch.float64)
    return out


def test_card_default_has_no_merger():
    card = CMSEnergyFlowDefault(debug=False, learnable=True)
    assert card.photon_merger is None


def test_card_with_merger_matches_standalone_merge():
    """Same seed: non-photon streams are bit-identical with and without the
    merger; the merged photon stream equals the standalone merge of the
    baseline photons (the merger consumes no RNG)."""
    truth = _make_truth_particles()
    plain = CMSEnergyFlowDefault(debug=False, learnable=True)
    merged_card = CMSEnergyFlowDefault(
        debug=False, learnable=True, photon_merger=PhotonClusterMerger(RADIUS)
    )
    # Clone per call: the card forward writes into its input tensor in place
    # (pre-existing behavior), so reuse would corrupt the second run.
    torch.manual_seed(11)
    out_plain = plain(truth.clone())
    torch.manual_seed(11)
    out_merged = merged_card(truth.clone())

    assert torch.equal(out_plain["EFlowTrack"], out_merged["EFlowTrack"])
    assert torch.equal(out_plain["EFlowNeutralHadron"], out_merged["EFlowNeutralHadron"])

    n_plain = out_plain["EFlowPhoton"].shape[0]
    assert n_plain > 5, "fixture produced too few eflow photons to test merging"
    expected = PhotonClusterMerger(RADIUS)(out_plain["EFlowPhoton"])
    assert expected.shape[0] < n_plain, "dense core should force at least one merge"
    assert torch.equal(expected, out_merged["EFlowPhoton"])
    assert torch.isfinite(out_merged["EFlowObject"]).all()


# ---------------------------------------------------------------------------
# M1: merged-count composition (compose_merged_photon_count)
# ---------------------------------------------------------------------------


def _cut_cards(radius: float | None):
    """Learnable cards with the production count acceptance; one plain, one
    with a merger of the given radius (None = no merger)."""
    kwargs = dict(debug=False, learnable=True, count_pt_min=1.0, count_abs_eta_max=2.7)
    plain = CMSEnergyFlowDefault(**kwargs)
    merged = CMSEnergyFlowDefault(
        **kwargs, photon_merger=None if radius is None else PhotonClusterMerger(radius)
    )
    return plain, merged


def test_composed_count_r_zero_matches_legacy():
    """R -> 0: every cluster is a singleton, so the composed count must equal
    the legacy compute_soft_count output exactly (values) and in gradient."""
    truth = _make_truth_particles(4)
    plain, merged = _cut_cards(radius=1e-12)
    torch.manual_seed(23)
    out_plain = plain(truth.clone())
    torch.manual_seed(23)
    out_merged = merged(truth.clone())

    legacy = out_plain["EcalPhotonExpectedCounts"]
    composed = out_merged["EcalPhotonExpectedCounts"]
    assert torch.equal(legacy.detach(), composed.detach())

    legacy.sum().backward()
    composed.sum().backward()
    checked = 0
    for (name, p_a), (_, p_b) in zip(
        plain.named_parameters(), merged.named_parameters(), strict=True
    ):
        if (p_a.grad is None) != (p_b.grad is None):
            raise AssertionError(f"gradient presence differs for {name}")
        if p_a.grad is not None:
            assert torch.allclose(p_a.grad, p_b.grad, rtol=1e-10, atol=1e-14), name
            checked += 1
    assert checked > 0, "no parameter received a count gradient"


def test_composed_count_forward_is_hard_merged_count():
    """With real merging the composed forward value is an exact integer count,
    bounded by the merged photon multiplicity, and smaller than the legacy
    per-tower count (merging reduces photon counts on the dense fixture)."""
    truth = _make_truth_particles(4)
    plain, merged = _cut_cards(radius=RADIUS)
    torch.manual_seed(29)
    out_plain = plain(truth.clone())
    torch.manual_seed(29)
    out_merged = merged(truth.clone())

    composed = out_merged["EcalPhotonExpectedCounts"].detach()
    assert torch.equal(composed, composed.round()), "ST pin must give integer forward values"
    n_rows = out_merged["EFlowPhoton"].shape[0]
    assert composed.sum() <= n_rows
    legacy_total = out_plain["EcalPhotonExpectedCounts"].detach().sum()
    assert composed.sum() < legacy_total, "dense fixture must lose photons to merging"

    # Gradient reaches the calo resolution coefficients through the composition.
    out_merged["EcalPhotonExpectedCounts"].sum().backward()
    params = dict(merged.named_parameters())
    g = params["ECal.resolution_func.common_c_E"].grad
    assert g is not None and torch.isfinite(g).all() and float(g.abs()) > 0


def _reference_compose(args, pt_soft, members, hard_pt, pt_min, tau):
    """Loop reference for cluster survival: members = list of per-cluster lists
    of tower indices; args = per-tower gate logits (single-factor gates)."""
    gates = torch.sigmoid(args)
    survivals, hards = [], []
    for mem, hpt in zip(members, hard_pt, strict=True):
        miss = torch.ones((), dtype=torch.float64)
        for i in mem:
            miss = miss * (1.0 - gates[i])
        s = gates[mem[0]] if len(mem) == 1 else 1.0 - miss
        s = s * torch.sigmoid((pt_soft[mem].sum() - pt_min) / (tau * pt_min))
        survivals.append(s)
        hards.append(hpt)
    st = [float(h) + (s - s.detach()) for s, h in zip(survivals, hards, strict=True)]
    return torch.stack(st)


def test_compose_synthetic_matches_reference():
    """Hand-built export: cluster of 2 emitted + 1 attached virtual, a real
    singleton, and a virtual singleton. Values and gradients must match a
    plain-loop reference implementation of S = [1 - prod(1-g)] * G_pt."""
    # towers: 0 seed A, 1 absorbed into A, 2 singleton B, 3 virtual near A,
    #         4 virtual isolated C
    args = torch.tensor([2.0, 1.0, 0.5, -1.5, -1.0], dtype=torch.float64, requires_grad=True)
    # virtual tower 4 gets pt above count_pt_min so its pt gate is ~1 and the
    # attached-vs-isolated gradient comparison below isolates the mate term
    pt_soft = torch.tensor([5.0, 3.0, 2.0, 0.5, 2.0], dtype=torch.float64)
    eta = torch.tensor([0.0, 0.02, 1.6, 0.01, 2.6], dtype=torch.float64)
    phi = torch.zeros(5, dtype=torch.float64)
    export = {
        "gate_nopt": torch.sigmoid(args),
        "log_gate_nopt": -torch.nn.functional.softplus(-args),
        "pt_soft": pt_soft,
        "emitted": torch.tensor([True, True, True, False, False]),
        "abs_eta_center": eta.abs().detach(),
        "eta": eta,
        "phi": phi,
        "event": torch.zeros(5, dtype=torch.float64),
        "anchor": torch.zeros((), dtype=torch.float64),
    }
    owner = torch.tensor([0, 0, 2])  # photons = towers 0,1,2; rows 0,1 cluster; 2 alone
    merged = torch.zeros(2, N_FEATURES, dtype=torch.float64)
    merged[0, ColumnMap.PT] = 7.9
    merged[1, ColumnMap.PT] = 2.0
    calo = types.SimpleNamespace(
        count_pt_min=1.0, count_tau_rel=0.05, count_abs_eta_max=2.7, is_ecal=True
    )
    merger = PhotonClusterMerger(RADIUS)

    counts = compose_merged_photon_count(export, owner, merged, merger, calo)
    assert counts.shape == (3,)  # ECal regions (0,1.5), (1.5,2.5), (2.5,2.7)
    # Forward: A (region 0) and B (region 1) are real and pass pt; C is virtual.
    assert torch.equal(counts.detach(), torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64))

    # Reference: A = towers {0,1,3} (virtual 3 attaches to seed 0), B = {2}, C = {4}.
    ref = _reference_compose(
        args, pt_soft, members=[[0, 1, 3], [2], [4]], hard_pt=[1.0, 1.0, 0.0],
        pt_min=1.0, tau=0.05,
    )
    grad_actual = torch.autograd.grad(counts.sum(), args, retain_graph=True)[0]
    grad_ref = torch.autograd.grad(ref.sum(), args)[0]
    assert torch.allclose(grad_actual, grad_ref, rtol=1e-9, atol=1e-12)
    # Attached virtual (3) must be suppressed by its robust mates relative to
    # the isolated virtual (4) despite a larger gate slope at its logit.
    assert grad_actual[3].abs() < grad_actual[4].abs()


def test_compose_handles_empty_photon_stream():
    """No emitted photons: every tower is a virtual singleton, forward is zero,
    gradient still flows (today's sub-threshold role)."""
    args = torch.tensor([-2.0, -3.0], dtype=torch.float64, requires_grad=True)
    export = {
        "gate_nopt": torch.sigmoid(args),
        "log_gate_nopt": -torch.nn.functional.softplus(-args),
        "pt_soft": torch.tensor([2.0, 3.0], dtype=torch.float64),
        "emitted": torch.tensor([False, False]),
        "abs_eta_center": torch.tensor([0.5, 1.9], dtype=torch.float64),
        "eta": torch.tensor([0.5, 1.9], dtype=torch.float64),
        "phi": torch.zeros(2, dtype=torch.float64),
        "event": torch.zeros(2, dtype=torch.float64),
        "anchor": torch.zeros((), dtype=torch.float64),
    }
    owner = torch.zeros(0, dtype=torch.long)
    merged = torch.zeros(0, N_FEATURES, dtype=torch.float64)
    calo = types.SimpleNamespace(
        count_pt_min=1.0, count_tau_rel=0.05, count_abs_eta_max=2.7, is_ecal=True
    )
    counts = compose_merged_photon_count(export, owner, merged, PhotonClusterMerger(RADIUS), calo)
    assert torch.equal(counts.detach(), torch.zeros(3, dtype=torch.float64))
    grad = torch.autograd.grad(counts.sum(), args)[0]
    assert torch.isfinite(grad).all() and (grad.abs() > 0).all()
