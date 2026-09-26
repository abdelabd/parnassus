"""Wasserstein critic for ``tune_cms_fullsim`` (critic_loss_plan.md section 3, milestone M1).

The training loss is a WGAN-style dual estimate of the Wasserstein distance between
the trainee's reconstructed events and the target events. Each event is a weighted
set of objects; the critic is a Deep Sets network that is Lipschitz with respect to
the unbalanced earth-mover distance between such sets (moving one unit of mass a
feature distance ``d`` costs ``d``, creating or destroying one unit costs ``R``).

Per event, with per-object features ``x_o``, weights ``w_o`` and event-level
response features ``h``::

    e_o   = bound_R(phi(x_o))                 |e_o| <= R, phi 1-Lipschitz
    S     = sum_o w_o e_o                     live weights
    S_det = sum_o sg(w_o) e_o                 same value, weights detached
    u     = bound_R(psi(h))                   |u| <= R, psi 1-Lipschitz
    f     = a . S + rho(S_det, u)             |a| <= 1, rho 1-Lipschitz

Every reconstructed object carries ``w_o = exp(log_w)``: exactly 1 in value, with the
likelihood ratio of its efficiency coins and calorimeter threshold gate in the
gradient (``ColumnMap.LOG_OBJ_WEIGHT``). The two-branch head makes that gradient
exact: the coin and threshold gradients flow only through the LINEAR branch, where
``d f / d w_o = a . e_o`` equals the finite difference of adding or removing the
object; the nonlinear branch sees the weights detached, so it supplies per-event
structure to the smearing gradients, which are pathwise and exact for any ``rho``,
without biasing the coin gradients. Both branches share the same fixed point at the
truth.

Lipschitz control: spectral normalisation on every linear map and on the pair
table, ReLU activations, and the Euclidean projection of ``phi`` onto the ball of
radius ``R`` (a per-component bound would cap the norm at ``R sqrt(latent)``
instead). ``f`` is then 2-Lipschitz with respect to the unbalanced EMD: one object
contributes at most ``2R``. (The first layer of ``phi`` is the sum of two unit-norm
maps, so ``phi`` is 1-Lipschitz in the continuous features and in the pair one-hot
separately, at most ``sqrt 2`` in both together.) The
projection, not a smooth squash: a radial ``tanh`` freezes every object it saturates
(gradient factor ~ 1e-3), so a critic saturated at ``-R`` on a large excess could
never re-assign the potential of a small deficit sitting in the same corner of
feature space (measured on the muon gun: region-0 muons stayed at ``-R`` for any
training budget). The projection keeps the tangential gradient alive at the bound.

Class-cell table: besides the continuous features, each object carries the index of
its (class, cell) pair, the cell being its (pt, |eta|) bin on the grid formed by the
union of the card's tracking-efficiency region edges
(``learnable.CMS_EFF_REGION_SPECS``), computed identically on both sides. The first
layer of ``phi`` adds one spectral-normed ``nn.Embedding`` row per pair to the linear
map of the continuous features: exactly a one-hot of the pair through a linear
layer, without materialising the one-hot. Two reasons. A 1-Lipschitz potential in
continuous (log_pt, eta) cannot follow the hard region edges (a step at pt = 1 GeV
between a small deficit and a large excess would need a ramp narrower than the data
allow); the cell gives the critic that step. And the pair rather than the class and
the cell separately: a killed electron becomes a photon in the same cell, so its
efficiency signal is a change of the class composition inside a cell at fixed total,
which an additive class-plus-cell encoding cannot express (electron gun: the rare
cells inherited the bulk's potential and two of three rare efficiencies were pushed
the wrong way; with the pair table they turn).

Event response: the marginal spectra hide scales and resolutions behind the truth
spectrum (a 10x wider muon resolution was worth 0.0075/event to the critic, its
noise floor). Both sides of an event share its truth particles, so each event also
carries, per class and per |eta| band of the card's momentum resolution, the log
ratio of the summed reco pt to the summed truth pt in units of
:data:`RESPONSE_SCALE`, plus a presence flag (:func:`build_event_response`, same rule
on both sides, no reco-to-truth matching). For a two-muon event that is the muon
response with width ``sigma / sqrt(2)``: a resolution error is a first-order change
of a peak a few units wide, which a ReLU critic can feel through the kinks inside
it. The band split is what identifies the per-band scales: pooled over bands, the
response distribution has one mode per band and a fit can settle with the bands
permuted (fifth muon-gun run: bands 0 and 2 landed on the band-1 truth and vice
versa). The response enters only the nonlinear branch, so it feeds the smearing
gradients pathwise (through the live reco pt) and never the coin gradients.

Objective (``real`` = target, ``fake`` = reconstructed): the critic minimises
``mean f(reco) - mean f(target)`` on detached reco inputs; the card minimises
``-mean f(reco)`` with live inputs. The critic value ``mean f(target) - mean f(reco)``
estimates the distance and is a non-stationary diagnostic, not a validation metric.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from parnassus.data.particle_io import ColumnMap
from parnassus.torch_delphes.learnable import CMS_EFF_REGION_SPECS, LearnableMomentumResolution
from parnassus.utils import class_to_pid_vectorized, pid_to_class_vectorized

# Canonical reco classes carried by ``pid`` on both sides (data.py: 211/11/13/111/22).
CRITIC_CLASSES: tuple[int, ...] = (211, 11, 13, 111, 22)
ETA_SCALE: float = 2.5
# (pt, |eta|) cell grid = union of the tracking-efficiency region edges of all species.
REGION_PT_EDGES: tuple[float, ...] = tuple(
    sorted({e for s in CMS_EFF_REGION_SPECS.values() for e in s.pt_edges})
)
REGION_ETA_EDGES: tuple[float, ...] = tuple(
    sorted({e for s in CMS_EFF_REGION_SPECS.values() for e in s.abs_eta_edges})
)
N_REGION_CELLS: int = (len(REGION_PT_EDGES) + 1) * (len(REGION_ETA_EDGES) + 1)
# (class, cell) pairs: one embedding row each.
N_PAIRS: int = len(CRITIC_CLASSES) * N_REGION_CELLS
# Features per object: standardised log_pt, eta / ETA_SCALE, cos phi, sin phi, and
# the (class, cell) pair index (an integer stored in the last column).
N_CONTINUOUS_FEATURES: int = 4
N_CRITIC_FEATURES: int = N_CONTINUOUS_FEATURES + 1
# Event response: one unit = RESPONSE_SCALE in log pt (2 %, a typical track
# resolution); per class and |eta| band a response and a presence flag. The bands are
# the card's momentum-resolution regions (inner boundaries; the last one is the
# acceptance edge).
RESPONSE_SCALE: float = 0.02
RESPONSE_ETA_EDGES: tuple[float, ...] = tuple(
    inspect.signature(LearnableMomentumResolution.__init__).parameters["boundaries"].default[:-1]
)
N_RESPONSE_BANDS: int = len(RESPONSE_ETA_EDGES) + 1
N_EVENT_FEATURES: int = 2 * len(CRITIC_CLASSES) * N_RESPONSE_BANDS


def region_cell(pt: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Cell index in ``[0, N_REGION_CELLS)`` of each object on the (pt, |eta|) grid
    (gradient-free; ``torch.bucketize`` with right-closed bins like the region specs)."""
    pt_bin = torch.bucketize(pt.detach(), torch.tensor(REGION_PT_EDGES, dtype=pt.dtype, device=pt.device))
    eta_bin = torch.bucketize(
        eta.detach().abs(), torch.tensor(REGION_ETA_EDGES, dtype=eta.dtype, device=eta.device)
    )
    return pt_bin * (len(REGION_ETA_EDGES) + 1) + eta_bin


def pair_index(pid: torch.Tensor, pt: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    """Index in ``[0, N_PAIRS)`` of each object's (class, cell) pair,
    ``class * N_REGION_CELLS + cell`` with the class in :data:`CRITIC_CLASSES` order
    (pads and unknown pids fall in class 0; they carry weight 0)."""
    abs_pid = pid.abs()
    cls = torch.zeros_like(abs_pid, dtype=torch.long)
    for i, c in enumerate(CRITIC_CLASSES):
        cls = torch.where(abs_pid == c, torch.full_like(cls, i), cls)
    return cls * N_REGION_CELLS + region_cell(pt, eta)


def build_critic_input(
    obs: dict[str, torch.Tensor],
    log_pt_mean: float = 0.0,
    log_pt_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-object critic features and weights from a padded observable dict.

    ``obs`` is a pred or target dict (``pt``, ``log_pt``, ``eta``, ``phi``, ``pid`` of
    shape ``(n_events, max_n_objects)``; ``log_w`` on the pred side). Padded and
    efficiency-killed slots have ``pt == 0`` and get weight 0 and zero features, so
    the critic never sees the padding width. Real objects get weight
    ``exp(log_w)`` (exactly 1 in value, gradient = coin likelihood ratio) on the pred
    side and 1 on the target side. ``log_pt_mean`` / ``log_pt_std`` are the target
    statistics (computed once by the caller); ``eta`` is scaled by
    :data:`ETA_SCALE`; ``phi`` enters as ``(cos, sin)`` so the wrap at pi is invisible;
    the last column is the (class, cell) pair index (:func:`pair_index`), an integer
    the critic looks up in its embedding table.

    Returns
    -------
    (x, w)
        ``x`` of shape ``(n_events, max_n_objects, N_CRITIC_FEATURES)`` and ``w`` of
        shape ``(n_events, max_n_objects)``, both in the dtype of ``obs["pt"]``.
    """
    pt = obs["pt"]
    valid = pt != 0
    valid_f = valid.to(pt.dtype)
    columns = [
        (obs["log_pt"] - log_pt_mean) / log_pt_std,
        obs["eta"] / ETA_SCALE,
        torch.cos(obs["phi"]),
        torch.sin(obs["phi"]),
        pair_index(obs["pid"], pt, obs["eta"]).to(pt.dtype),
    ]
    x = torch.stack(columns, dim=-1) * valid_f.unsqueeze(-1)
    w = valid_f
    if "log_w" in obs:
        w = w * torch.exp(obs["log_w"])
    return x, w


def response_band(eta: torch.Tensor) -> torch.Tensor:
    """Band index in ``[0, N_RESPONSE_BANDS)`` of each object on the |eta| grid of the
    card's momentum resolution (gradient-free)."""
    return torch.bucketize(eta.detach().abs(), torch.tensor(RESPONSE_ETA_EDGES, dtype=eta.dtype, device=eta.device))


def build_event_response(obs: dict[str, torch.Tensor], truth_particles: torch.Tensor) -> torch.Tensor:
    """Per-event response features: the summed-pt response and a presence flag per
    class and |eta| band.

    ``obs`` is a pred or target dict as in :func:`build_critic_input`;
    ``truth_particles`` the same events' ``(n_events, max_n_particles, N_FEATURES)``
    truth batch in ``ColumnMap`` layout (padded rows all zero), whose PDG codes are
    folded onto the canonical classes exactly as ``data.py`` does for the reco side.
    For class ``c`` and band ``b`` (reco objects by their reco eta, truth particles by
    their truth eta) the response is ``(log sum_reco pt - log sum_truth pt) /
    RESPONSE_SCALE`` where both sums are nonzero and 0 otherwise, and the flag is 1
    where both are nonzero. The reco sums are live (pathwise smearing gradient);
    killed objects have ``pt == 0`` and simply drop out.

    Returns
    -------
    torch.Tensor
        ``(n_events, N_EVENT_FEATURES)``: the responses (class-major, band-minor),
        then the flags in the same order.
    """
    pt, abs_pid = obs["pt"], obs["pid"].abs()
    band = response_band(obs["eta"])
    pt_t = truth_particles[..., ColumnMap.PT]
    pid_np = truth_particles[..., ColumnMap.PID].detach().cpu().numpy().astype(np.int64)
    abs_pid_t = torch.from_numpy(class_to_pid_vectorized(pid_to_class_vectorized(pid_np))).to(pt.device).abs()
    valid_t = torch.any(truth_particles != 0, dim=-1)
    band_t = response_band(truth_particles[..., ColumnMap.ETA])
    responses, flags = [], []
    for c in CRITIC_CLASSES:
        for b in range(N_RESPONSE_BANDS):
            sum_reco = (pt * ((abs_pid == c) & (band == b))).sum(dim=1)
            sum_truth = (pt_t * ((abs_pid_t == c) & (band_t == b) & valid_t)).sum(dim=1)
            present = (sum_reco > 0) & (sum_truth > 0)
            response = (torch.log(sum_reco.clamp_min(1e-6)) - torch.log(sum_truth.clamp_min(1e-6))) / RESPONSE_SCALE
            responses.append(torch.where(present, response, torch.zeros_like(response)))
            flags.append(present.to(pt.dtype))
    return torch.stack(responses + flags, dim=-1)


class WassersteinCritic(nn.Module):
    """Lipschitz Deep Sets critic with a linear branch on the live object weights and
    a nonlinear branch on the detached weights and the event response (module
    docstring).

    Parameters
    ----------
    num_features : int
        Per-object input width (:data:`N_CRITIC_FEATURES`): the continuous features
        plus the pair-index column.
    num_event_features : int
        Event-response input width (:data:`N_EVENT_FEATURES`).
    hidden : int
        Width of the hidden layers of ``phi``, ``psi`` and ``rho``.
    latent : int
        Dimension of the per-object embedding ``e_o`` and of the event-response
        embedding ``u`` (free, thanks to the norm bound).
    bound : float
        ``R``: the norm bound on ``e_o`` and ``u`` = the EMD mass-creation penalty,
        in standardised feature units.
    """

    def __init__(
        self,
        num_features: int = N_CRITIC_FEATURES,
        num_event_features: int = N_EVENT_FEATURES,
        hidden: int = 128,
        latent: int = 32,
        bound: float = 4.0,
    ) -> None:
        super().__init__()
        self.bound = float(bound)
        # First layer of phi: linear in the continuous features plus one embedding
        # row per (class, cell) pair (= a linear layer on the pair one-hot; the
        # spectral norm of the table keeps any two rows at most sqrt 2 apart).
        self.phi_in = spectral_norm(nn.Linear(num_features - 1, hidden))
        self.pair = spectral_norm(nn.Embedding(N_PAIRS, hidden))
        self.phi = nn.Sequential(
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden, latent)),
        )
        # |a| <= 1 (the spectral norm of a 1 x latent matrix is its 2-norm); no bias,
        # a constant cancels in every mean difference.
        self.readout = spectral_norm(nn.Linear(latent, 1, bias=False))
        # Event-response embedding, bounded like an object so one event's response
        # is worth at most R.
        self.psi = nn.Sequential(
            spectral_norm(nn.Linear(num_event_features, hidden)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden, latent)),
        )
        self.rho = nn.Sequential(
            spectral_norm(nn.Linear(2 * latent, hidden)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden, 1)),
        )

    def _project(self, v: torch.Tensor) -> torch.Tensor:
        """Euclidean projection onto the ball of radius ``R``: identity inside,
        ``R v / |v|`` outside (1-Lipschitz, the ball is convex). Outside the ball only
        the tangential gradient survives, which is what lets the critic re-assign a
        saturated object's potential; a radial ``tanh`` squash has no such gradient
        once saturated (module docstring). Epsilon inside the norm keeps ``v = 0``
        finite."""
        norm = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + 1e-12)
        return v * torch.clamp(self.bound / norm, max=1.0)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Bounded per-object embedding ``e = proj_R(phi(x))``, ``|e| <= R``; the
        last column of ``x`` is the (class, cell) pair index."""
        first = self.phi_in(x[..., :-1]) + self.pair(x[..., -1].long())
        return self._project(self.phi(first))

    def branches(
        self, x: torch.Tensor, w: torch.Tensor, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The two scalar branches per event: ``(a . S, rho(S_det, u))``.

        ``x``: ``(n_events, n_objects, num_features)``; ``w``: ``(n_events, n_objects)``;
        ``h``: ``(n_events, num_event_features)``.
        """
        e = self.embed(x)
        s_live = (e * w.unsqueeze(-1)).sum(dim=1)
        s_det = (e * w.detach().unsqueeze(-1)).sum(dim=1)
        u = self._project(self.psi(h))
        return self.readout(s_live).squeeze(-1), self.rho(torch.cat([s_det, u], dim=-1)).squeeze(-1)

    def forward(self, x: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Critic value per event, shape ``(n_events,)``."""
        lin, nonlin = self.branches(x, w, h)
        return lin + nonlin


def critic_loss(
    critic: WassersteinCritic,
    x_reco: torch.Tensor,
    w_reco: torch.Tensor,
    h_reco: torch.Tensor,
    x_target: torch.Tensor,
    w_target: torch.Tensor,
    h_target: torch.Tensor,
) -> torch.Tensor:
    """``mean f(reco) - mean f(target)``: what the CRITIC minimises (its negative is
    the Wasserstein estimate). Pure; detaching is the caller's job."""
    return critic(x_reco, w_reco, h_reco).mean() - critic(x_target, w_target, h_target).mean()


def card_loss(
    critic: WassersteinCritic, x_reco: torch.Tensor, w_reco: torch.Tensor, h_reco: torch.Tensor
) -> torch.Tensor:
    """``-mean f(reco)`` on LIVE features, weights and response: what the CARD
    minimises. The target term is a constant for the card and is left out."""
    return -critic(x_reco, w_reco, h_reco).mean()


def critic_step(
    critic: WassersteinCritic,
    optimizer: torch.optim.Optimizer,
    x_reco: torch.Tensor,
    w_reco: torch.Tensor,
    h_reco: torch.Tensor,
    x_target: torch.Tensor,
    w_target: torch.Tensor,
    h_target: torch.Tensor,
    n_steps: int = 5,
) -> float:
    """Run ``n_steps`` critic updates on one batch and return the last critic loss.

    Every input is detached here, so the critic's backward never walks into the card
    graph (which must survive for the card step that follows) and never deposits
    gradient in the card parameters. The weights become plain 1/0 masks.
    """
    x_r, w_r, h_r = x_reco.detach(), w_reco.detach(), h_reco.detach()
    x_t, w_t, h_t = x_target.detach(), w_target.detach(), h_target.detach()
    loss = torch.zeros((), dtype=x_r.dtype, device=x_r.device)
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = critic_loss(critic, x_r, w_r, h_r, x_t, w_t, h_t)
        loss.backward()
        optimizer.step()
    return float(loss.detach())


def cat_critic_inputs(
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate per-batch ``(x, w, h)`` critic inputs along the event axis.

    Every batch is padded to its own widest event, so ``x`` and ``w`` are first
    zero-padded on the object axis to the widest batch; the extra slots have zero
    features and zero weight and the critic ignores them (padding invariance).
    """
    width = max(x.shape[1] for x, _, _ in batches)
    xs = [torch.nn.functional.pad(x, (0, 0, 0, width - x.shape[1])) for x, _, _ in batches]
    ws = [torch.nn.functional.pad(w, (0, width - w.shape[1])) for _, w, _ in batches]
    return torch.cat(xs), torch.cat(ws), torch.cat([h for _, _, h in batches])


def held_out_w1(
    x_reco: torch.Tensor,
    w_reco: torch.Tensor,
    h_reco: torch.Tensor,
    x_target: torch.Tensor,
    w_target: torch.Tensor,
    h_target: torch.Tensor,
    *,
    hidden: int = 128,
    bound: float = 1.0,
    lr: float = 1e-3,
    n_steps: int = 300,
    batch_size: int = 2048,
    seed: int = 0,
) -> float:
    """Stationary Wasserstein monitor: the distance estimate of a FRESH critic.

    A new critic (same architecture and bound as the training critic, initialised
    from ``seed``) is fitted for ``n_steps`` Adam steps on mini-batches of
    ``batch_size`` events drawn from a random half of the events, then scored on
    the other half. The value returned is ``mean f(target) - mean f(reco)`` on the
    held-out half: a lower bound on the distance that fluctuates around zero at the
    truth. Fitting on one half and scoring on the other removes the in-sample bias
    of a critic that has memorised its events (Danihelka et al. 2017, "independent
    critic"); the fixed seed, split and step budget make epochs comparable. The
    card is never touched: inputs are detached in :func:`critic_step`, and the split
    and the initialisation use their own generator, so the fit's random stream is
    not perturbed.

    ``x_*``: ``(n_events, n_objects, N_CRITIC_FEATURES)``; ``w_*``:
    ``(n_events, n_objects)``; ``h_*``: ``(n_events, N_EVENT_FEATURES)``; reco and
    target hold the same events in the same order.
    """
    device, dtype = x_reco.device, x_reco.dtype
    n_events = x_reco.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_events, generator=gen).to(device)
    fit, held = perm[: n_events // 2], perm[n_events // 2 :]
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        critic = WassersteinCritic(hidden=hidden, bound=bound).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.5, 0.9))
    for _ in range(n_steps):
        i = fit[torch.randperm(fit.numel(), generator=gen)[:batch_size].to(device)]
        critic_step(
            critic, opt, x_reco[i], w_reco[i], h_reco[i], x_target[i], w_target[i], h_target[i], n_steps=1
        )
    critic.eval()  # freeze the spectral-norm power iterations for scoring
    with torch.no_grad():
        f_reco = torch.cat([critic(x_reco[i], w_reco[i], h_reco[i]) for i in held.split(batch_size)])
        f_target = torch.cat([critic(x_target[i], w_target[i], h_target[i]) for i in held.split(batch_size)])
    return float(f_target.mean() - f_reco.mean())
