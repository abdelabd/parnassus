"""Unit tests for ``tune_cms_fullsim.critics`` (critic_loss_plan.md milestone M1).

Synthetic padded observable dicts only: no ROOT data, no detector card. Everything
runs in float64 to match the fit loop.
"""

from __future__ import annotations

import math

import pytest
import torch

from parnassus.torch_delphes.tune_cms_fullsim.critics import (
    CRITIC_CLASSES,
    ETA_SCALE,
    N_CRITIC_FEATURES,
    WassersteinCritic,
    build_critic_input,
    card_loss,
    critic_loss,
    critic_step,
)

R = 4.0


def _make_obs(
    n_events: int, width: int, seed: int, log_pt_shift: float = 0.0, n_valid: int | None = None
) -> dict[str, torch.Tensor]:
    """Padded observables: every event has ``n_valid`` real objects (default: a
    random 1..width), the rest are pads with pt == 0 and zero features."""
    g = torch.Generator().manual_seed(seed)
    pt = torch.zeros(n_events, width, dtype=torch.float64)
    eta = torch.zeros_like(pt)
    phi = torch.zeros_like(pt)
    pid = torch.zeros_like(pt)
    classes = torch.tensor(CRITIC_CLASSES, dtype=torch.float64)
    for i in range(n_events):
        k = n_valid if n_valid is not None else int(torch.randint(1, width + 1, (1,), generator=g))
        pt[i, :k] = torch.exp(torch.randn(k, generator=g, dtype=torch.float64) + log_pt_shift + 1.0)
        eta[i, :k] = (torch.rand(k, generator=g, dtype=torch.float64) - 0.5) * 5.0
        phi[i, :k] = (torch.rand(k, generator=g, dtype=torch.float64) - 0.5) * 2.0 * math.pi
        pid[i, :k] = classes[torch.randint(0, len(classes), (k,), generator=g)]
    valid = pt != 0
    return {
        "pt": pt,
        "log_pt": torch.where(valid, torch.log(pt.clamp_min(1e-6)), torch.zeros_like(pt)),
        "eta": eta,
        "phi": phi,
        "pid": pid,
    }


def _critic(seed: int = 0, warm: int = 30, **kw) -> WassersteinCritic:
    """A float64 critic whose spectral-norm power iterations have converged
    (``warm`` training-mode forwards), so the Lipschitz bounds hold to ~1e-3.
    Returned in eval mode: in training mode every forward runs one more power
    iteration in place, so two consecutive forwards would not use identical weights
    and the exactness checks below could not compare them bit for bit."""
    torch.manual_seed(seed)
    critic = WassersteinCritic(bound=R, **kw).double()
    x, w = build_critic_input(_make_obs(16, 8, seed=seed + 100))
    with torch.no_grad():
        for _ in range(warm):
            critic(x, w)
    return critic.eval()


# ---------------------------------------------------------------------------
# input builder + padding invariance
# ---------------------------------------------------------------------------


def test_input_builder_masks_pads_and_encodes_classes():
    obs = _make_obs(5, 7, seed=1)
    x, w = build_critic_input(obs, log_pt_mean=0.5, log_pt_std=2.0)
    valid = obs["pt"] != 0
    assert x.shape == (5, 7, N_CRITIC_FEATURES) and w.shape == (5, 7)
    # Pads: weight 0, features 0. Real objects: weight exactly 1 without log_w.
    assert torch.equal(w, valid.to(torch.float64))
    assert torch.equal(x[~valid], torch.zeros_like(x[~valid]))
    # Standardised log_pt, scaled eta, (cos, sin) phi, one-hot class.
    assert torch.allclose(x[valid][:, 0], (obs["log_pt"][valid] - 0.5) / 2.0)
    assert torch.allclose(x[valid][:, 1], obs["eta"][valid] / ETA_SCALE)
    assert torch.allclose(x[valid][:, 2] ** 2 + x[valid][:, 3] ** 2, torch.ones(int(valid.sum()), dtype=torch.float64))
    onehot = x[valid][:, 4:]
    assert torch.equal(onehot.sum(dim=1), torch.ones(int(valid.sum()), dtype=torch.float64))
    for j, c in enumerate(CRITIC_CLASSES):
        assert torch.equal(onehot[:, j] == 1, obs["pid"][valid] == c)
    # log_w -> weight exp(log_w), still 0 on pads.
    obs["log_w"] = torch.where(valid, torch.full_like(obs["pt"], math.log(2.0)), torch.zeros_like(obs["pt"]))
    _, w2 = build_critic_input(obs)
    assert torch.allclose(w2, 2.0 * valid.to(torch.float64))


def test_critic_is_invariant_to_padding_width():
    critic = _critic(seed=2)
    obs = _make_obs(6, 5, seed=3)
    x, w = build_critic_input(obs)
    padded = {k: torch.cat([v, torch.zeros(6, 4, dtype=v.dtype)], dim=1) for k, v in obs.items()}
    xp, wp = build_critic_input(padded)
    with torch.no_grad():
        assert torch.allclose(critic(x, w), critic(xp, wp), atol=1e-12)


# ---------------------------------------------------------------------------
# Lipschitz / mass bound
# ---------------------------------------------------------------------------


def test_embedding_norm_is_bounded_and_one_object_moves_f_by_at_most_2R():
    critic = _critic(seed=4)
    obs = _make_obs(8, 6, seed=5, n_valid=3)
    x, w = build_critic_input(obs)
    # Extreme features: the embedding norm still sits under R.
    x_far = torch.cat([x, 50.0 * torch.randn(8, 6, N_CRITIC_FEATURES, dtype=torch.float64)], dim=1)
    with torch.no_grad():
        assert float(critic.embed(x_far).norm(dim=-1).max()) <= R * (1 + 1e-9)
        base = critic(x, w)
        # Add one extreme object (slot 3 is a pad in every event) with weight 1.
        x_add = x.clone()
        x_add[:, 3] = 50.0 * torch.randn(8, N_CRITIC_FEATURES, dtype=torch.float64)
        w_add = w.clone()
        w_add[:, 3] = 1.0
        assert float((critic(x_add, w_add) - base).abs().max()) <= 2.0 * R * 1.01
        # Move one real object by a feature distance d: |df| <= 2 d.
        delta = torch.randn(8, N_CRITIC_FEATURES, dtype=torch.float64)
        delta = 0.3 * delta / delta.norm(dim=-1, keepdim=True)
        x_mv = x.clone()
        x_mv[:, 0] = x[:, 0] + delta
        assert float((critic(x_mv, w) - base).abs().max()) <= 2.0 * 0.3 * 1.01


# ---------------------------------------------------------------------------
# coin gradient: exact through the linear branch, absent from the nonlinear one
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alive", [True, False])
def test_coin_gradient_is_the_linear_branch_finite_difference(alive: bool):
    """Object 0 of each event carries a coin with efficiency ``sigmoid(theta)``:
    ``log_w = log eff - sg(log eff)`` (alive) or ``log1p(-eff) - sg(...)`` (dead, as
    inherited by a neutral object). The card-loss gradient w.r.t. theta must equal
    ``-(lin(with object) - lin(without object)) * dlog p/dtheta`` exactly, and the
    nonlinear branch must contribute nothing."""
    critic = _critic(seed=6)
    obs = _make_obs(5, 4, seed=7, n_valid=3)
    theta = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    eff = torch.sigmoid(theta)
    log_p = torch.log(eff) if alive else torch.log1p(-eff)
    log_w = torch.zeros_like(obs["pt"])
    log_w[:, 0] = log_p - log_p.detach()
    obs["log_w"] = log_w
    x, w = build_critic_input(obs)
    assert torch.equal(w.detach(), (obs["pt"] != 0).to(torch.float64))  # value exactly 1

    lin, nonlin = critic.branches(x, w)
    (g_lin,) = torch.autograd.grad(lin.sum(), theta, retain_graph=True)
    g_nonlin = torch.autograd.grad(nonlin.sum(), theta, retain_graph=True, allow_unused=True)[0]
    assert g_nonlin is None or float(g_nonlin) == 0.0
    (g_card,) = torch.autograd.grad(card_loss(critic, x, w) * 5, theta)  # sum over events
    assert torch.allclose(g_card, -g_lin, rtol=0, atol=1e-12)

    # Closed form: finite difference of the linear branch times dlog p / dtheta.
    with torch.no_grad():
        w_without = w.detach().clone()
        w_without[:, 0] = 0.0
        lin_with, _ = critic.branches(x.detach(), w.detach())
        lin_without, _ = critic.branches(x.detach(), w_without)
    dlogp = (1.0 - eff) if alive else -eff
    expected = ((lin_with - lin_without).sum() * dlogp).detach()
    assert torch.allclose(g_lin, expected, rtol=1e-10, atol=1e-12)
    assert float(expected.abs()) > 0.0


# ---------------------------------------------------------------------------
# training: the critic separates two shifted sets
# ---------------------------------------------------------------------------


def test_critic_step_separates_shifted_sets():
    critic = _critic(seed=8, warm=0).train()
    opt = torch.optim.Adam(critic.parameters(), lr=1e-3, betas=(0.5, 0.9))
    x_r, w_r = build_critic_input(_make_obs(64, 6, seed=9))
    x_t, w_t = build_critic_input(_make_obs(64, 6, seed=10, log_pt_shift=1.0))
    with torch.no_grad():
        before = float(critic_loss(critic, x_r, w_r, x_t, w_t))
    last = critic_step(critic, opt, x_r, w_r, x_t, w_t, n_steps=200)
    # The critic minimises mean f(reco) - mean f(target): it must go clearly negative.
    assert last < before and last < -0.5, (before, last)
    # Its negative, the distance estimate, is bounded by 2R x the mean object count.
    assert -last <= 2.0 * R * float(w_t.sum(dim=1).mean()) * 1.01
