r"""Gradient-Enhanced Bayesian Optimization (GEBO) for CMS TorchDelphes.

This module uses BoTorch + GPyTorch's ``RBFKernelGrad`` to perform gradient-
enhanced Bayesian optimization over the ~66 learnable detector parameters of
:class:`~parnassus.torch_delphes.defaults.CMSEnergyFlowDefault`.  At each
evaluated point the objective returns **both** the scalar distribution-matching
loss **and** its exact 66-dimensional gradient w.r.t. the raw parameters,
giving the Gaussian Process surrogate 67 pieces of structural information per
function evaluation instead of 1.  This breaks the curse of dimensionality and
lets BO scale to the full 66-D parameter space.

The script:

1. Loads a CMS full-simulation ROOT file once.
2. Builds a flat *raw-parameter* vector from the card's trainable scalars.
3. Runs the standard GEBO loop:
   * evaluate ``[loss, ∇loss]`` at a handful of random initial points,
   * fit a ``GPyTorch`` model with ``RBFKernelGrad``,
   * use Expected Improvement (or LCB) to pick the next candidate,
   * repeat.
4. Saves the full history (queried points, losses, best parameters) to JSON.

It does **not** run any Adam fine-tuning afterwards — the best point found by
BO is the final output.

Usage
-----
.. code-block:: shell

    python -m parnassus.torch_delphes.tune_cms_fullsim.gebo_search \
        --root-file /path/to/data.root \
        --n-events 2000 \
        --n-iterations 50 \
        --output-dir doc/gebo_results
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import gpytorch
import torch
import yaml
from botorch.acquisition import (
    ExpectedImprovement,
    LogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from botorch.fit import fit_gpytorch_mll
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

try:
    import comet_ml
    _HAS_COMET = True
except ImportError:
    _HAS_COMET = False

from parnassus.torch_delphes import param_config as pc
from parnassus.torch_delphes.defaults import CMSEnergyFlowDefault

from .config import (
    CALO_COUNT_TERM_KEYS,
    COUNT_TERM_KEYS,
)
from .data import (
    load_cms_flow_root,
    load_pflow_targets_ragged,
    load_truth_events_ragged,
    restore_event_format,
    load_pflow_targets_from_tensor,
)
from .loss import (
    CALO_COUNT_WEIGHT,
    COUNT_RATE_FLOOR,
    COUNT_WEIGHT,
    EVENT_WEIGHT,
    LOSS_CHOICES,
    PID_WEIGHTING_CHOICES,
    get_loss_fn,
)

# =============================================================================
# Parameter vectorizer – maps between a flat raw-parameter tensor and a card
# =============================================================================


class ParamVectorizer:
    """Maps between a flat 1-D raw-parameter vector and a
    ``CMSEnergyFlowDefault`` card's ``nn.Parameter`` tensors.

    The *raw* space is the internal parameterisation (logit, arctanh, …) stored
    in ``param.data``; the card's forward pass applies the physical transforms
    (sigmoid, tanh, softplus) automatically.  Working in raw space keeps every
    dimension unconstrained and GP-friendly.
    """

    def __init__(self, card: CMSEnergyFlowDefault, trainable_keys: set[str] | None = None):
        """
        Parameters
        ----------
        card : CMSEnergyFlowDefault
            A ``learnable=True`` card whose ``named_parameters()`` defines the
            ordering and shapes.
        trainable_keys : set[str] | None
            If given, only parameters whose *base name* (without ``[i]``) is in
            this set are included.  ``None`` means all parameters.
        """
        self._entries: list[tuple[str, int, int]] = []  # (name, start, length)
        self._shapes: dict[str, torch.Size] = {}
        offset = 0
        for name, p in card.named_parameters():
            base = name.split("[", 1)[0]
            if trainable_keys is not None and base not in trainable_keys:
                continue
            n = p.numel()
            self._entries.append((name, offset, n))
            self._shapes[name] = p.shape
            offset += n
        self._dim = offset

    @property
    def dim(self) -> int:
        """Total number of raw scalar parameters."""
        return self._dim

    def get_vector(self, card: CMSEnergyFlowDefault) -> torch.Tensor:
        """Extract the current raw parameter values as a flat ``(dim,)`` tensor."""
        vec = torch.empty(self._dim, dtype=torch.float64)
        for name, start, n in self._entries:
            p = dict(card.named_parameters())[name]
            vec[start : start + n] = p.data.flatten().to(torch.float64)
        return vec

    def set_vector(self, card: CMSEnergyFlowDefault, vector: torch.Tensor) -> None:
        """Copy a flat raw vector into the card's parameters (in-place)."""
        if vector.shape != (self._dim,):
            raise ValueError(
                f"Expected vector of shape ({self._dim},), got {tuple(vector.shape)}"
            )
        with torch.no_grad():
            for name, start, n in self._entries:
                p = dict(card.named_parameters())[name]
                p.data.copy_(vector[start : start + n].reshape(p.shape).to(p.dtype))

    def param_names(self) -> list[str]:
        """Ordered list of parameter names (with ``[i]`` suffixes for elements)."""
        names: list[str] = []
        for name, start, n in self._entries:
            if n == 1:
                names.append(name)
            else:
                for i in range(n):
                    names.append(f"{name}[{i}]")
        return names


# =============================================================================
# Bounds helpers – derive raw-space bounds from an optuna-style config
# =============================================================================


def _physical_to_raw_bounds(
    key: str,
    low: float,
    high: float,
) -> tuple[float, float]:
    """Convert a physical-range ``[low, high]`` to raw-space bounds.

    Uses the same inverse transforms as :func:`parnassus.torch_delphes.param_config.to_raw`.
    """
    base = key.split("[", 1)[0]
    kind = pc.param_transform_kind(base)
    # Add a small margin so the bounds are strictly inside any singularities.
    eps = 1e-4
    if kind == "scale":
        lo = pc.to_raw(base, max(low, pc._SCALE_MIN + eps))
        hi = pc.to_raw(base, min(high, pc._SCALE_MAX - eps))
    elif kind == "logit":
        lo = pc.to_raw(base, max(low, 1e-6))
        hi = pc.to_raw(base, min(high, 1.0 - 1e-6))
    elif kind == "softplus":
        lo = pc.to_raw(base, max(low, 1e-12))
        hi = pc.to_raw(base, high)
    else:  # identity
        lo, hi = low, high
    return float(lo), float(hi)


def _default_bounds_for_param(key: str) -> tuple[float, float]:
    """Heuristic wide raw-space bounds when no config range is available."""
    base = key.split("[", 1)[0]
    kind = pc.param_transform_kind(base)
    if kind == "scale":
        return pc.to_raw(base, pc._SCALE_MIN + 1e-4), pc.to_raw(base, pc._SCALE_MAX - 1e-4)
    if kind == "logit":
        return pc.to_raw(base, 1e-6), pc.to_raw(base, 1.0 - 1e-6)
    if kind == "softplus":
        return -9.0, 5.0  # softplus(-9) ≈ 1.2e-4, softplus(5) ≈ 5.0
    return -10.0, 10.0


def load_bounds_from_optuna_config(
    config_path: str | Path,
    vectorizer: ParamVectorizer,
) -> torch.Tensor:
    """Read an ``optuna_config.yaml`` and build a ``(dim, 2)`` bounds tensor.

    For each parameter named in the config, derives raw-space ``[low, high]``
    from its ``{low, high}`` (or ``{value}``) spec.  Parameters NOT in the
    config get wide default bounds.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)
    param_specs = raw.get("parameters", {}) if isinstance(raw, dict) else {}

    bounds_list: list[tuple[float, float]] = []
    for name in vectorizer.param_names():
        if name in param_specs:
            spec = param_specs[name]
            if "value" in spec:  # pinned param – narrow bounds around its value
                v = float(spec["value"])
                r = pc.to_raw(name.split("[", 1)[0], v)
                margin = 0.1
                bounds_list.append((float(r) - margin, float(r) + margin))
            else:
                low = float(spec.get("low", 0.0))
                high = float(spec.get("high", 1.0))
                bounds_list.append(_physical_to_raw_bounds(name, low, high))
        else:
            bounds_list.append(_default_bounds_for_param(name))

    bounds_t = torch.tensor(bounds_list, dtype=torch.float64)
    return bounds_t


# =============================================================================
# Objective function – evaluates loss + gradient for a given parameter vector
# =============================================================================


def make_objective(
    vectorizer: ParamVectorizer,
    truth_ragged: list[torch.Tensor],
    target_ragged: dict,
    device: torch.device,
    loss_name: str = "wasserstein_1d",
    count_weight: float = COUNT_WEIGHT,
    calo_count_weight: float = CALO_COUNT_WEIGHT,
    count_rate_floor: float = COUNT_RATE_FLOOR,
    event_weight: float = EVENT_WEIGHT,
    pid_weighting: str = "equal",
    pid_weight_floor: float = 0.0,
    batch_size: int = 256,
    seed: int = 0,
) -> callable:
    """Build a closure ``obj(theta_1d) -> Tensor[loss, grad_1...grad_d]``.

    The closure is meant to be called as ``evaluate_objective(theta_batch)``
    where ``theta_batch`` has shape ``(q, d)`` and returns ``(q, d+1)``.

    Internally it processes the static dataset in mini-batches, sums the
    per-batch losses, and calls ``autograd.grad`` once to get the total
    gradient.

    A fixed ``torch.manual_seed(seed)`` is set before each evaluation so the
    Gumbel noise in the efficiency module is deterministic and the objective
    is a pure function of *θ*.
    """
    base_loss_fn = get_loss_fn(loss_name)

    def loss_fn(pred: dict, target: dict) -> torch.Tensor:
        return base_loss_fn(
            pred,
            target,
            count_weight=count_weight,
            calo_count_weight=calo_count_weight,
            count_rate_floor=count_rate_floor,
            event_weight=event_weight,
            pid_weighting=pid_weighting,
            pid_weight_floor=pid_weight_floor,
        )

    # Pre-build per-event target dicts.  The target side is static; we only
    # vary the card parameters.  Each entry in ``_targets`` is a dict with the
    # same keys the training loop's batch dict carries.
    n_events = len(truth_ragged)
    _targets: list[dict[str, torch.Tensor]] = []
    for i in range(n_events):
        tgt_i: dict[str, torch.Tensor] = {}
        for k, v in target_ragged.items():
            tgt_i[k] = v[i] if isinstance(v, list) else v[i : i + 1]
        _targets.append(tgt_i)

    def evaluate_objective(theta: torch.Tensor) -> torch.Tensor:
        """Evaluate loss + gradient at one or more parameter vectors.

        Parameters
        ----------
        theta : torch.Tensor
            Shape ``(q, d)`` – one or more query points in raw parameter space.

        Returns
        -------
        torch.Tensor
            Shape ``(q, d+1)`` – each row is ``[loss, ∂loss/∂θ₁, …, ∂loss/∂θ_d]``.
        """
        if theta.ndim == 1:
            theta = theta.unsqueeze(0)
        q, d = theta.shape
        if d != vectorizer.dim:
            raise ValueError(
                f"Expected theta of shape (q, {vectorizer.dim}), got (q, {d})"
            )

        results = []
        for i_q in range(q):
            theta_i = theta[i_q]  # (d,) on CPU

            # Build a fresh card for this query point.
            torch.manual_seed(seed)
            card = CMSEnergyFlowDefault(debug=False, learnable=True).to(device)
            vectorizer.set_vector(card, theta_i)

            # Zero gradients on all card parameters (they accumulate across
            # mini-batches via retain_graph=False add).
            for p in card.parameters():
                if p.requires_grad:
                    p.grad = None

            total_loss = torch.zeros((), dtype=torch.float64, device=device)

            # Mini-batch over the static dataset.
            for start in range(0, n_events, batch_size):
                end = min(start + batch_size, n_events)
                batch_truth = _build_batch_truth(
                    truth_ragged[start:end], device
                )
                if batch_truth.shape[0] == 0:
                    continue

                # Run the card on the unpadded truth particles.
                mask = torch.any(batch_truth != 0, dim=-1)
                truth_nonpadded = batch_truth[mask]

                out = card(truth_nonpadded)
                eflow = out["EFlowObject"]
                eflow_restored = restore_event_format(eflow, mask)
                pred = load_pflow_targets_from_tensor(eflow_restored)
                for out_key, pred_key, _tgt_key in (*COUNT_TERM_KEYS, *CALO_COUNT_TERM_KEYS):
                    pred[pred_key] = out[out_key]

                # Assemble target dict for this mini-batch.
                target = _assemble_target_dict(_targets[start:end], device)
                batch_loss = loss_fn(pred, target)
                total_loss = total_loss + batch_loss.detach()
                batch_loss.backward()

            # Gather gradients from the card's parameters into a flat vector
            # in the same order as the vectorizer.
            grad_flat = torch.empty(vectorizer.dim, dtype=torch.float64, device=device)
            for name, start, n in vectorizer._entries:
                p = dict(card.named_parameters())[name]
                if p.grad is not None:
                    grad_flat[start : start + n] = p.grad.flatten().to(torch.float64)
                else:
                    grad_flat[start : start + n] = 0.0

            results.append(
                torch.cat([total_loss.unsqueeze(0), grad_flat])
            )

        return torch.stack(results)  # (q, d+1)

    return evaluate_objective


def _build_batch_truth(
    events: list[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Pad a list of per-event truth tensors to the batch's max multiplicity."""
    if not events:
        return torch.zeros((0, 0), dtype=torch.float64, device=device)
    max_n = max(e.shape[0] for e in events)
    n_feat = events[0].shape[1]
    out = torch.zeros((len(events), max_n, n_feat), dtype=torch.float64, device=device)
    for i, ev in enumerate(events):
        n = ev.shape[0]
        if n > 0:
            out[i, :n] = ev.to(device)
    return out


def _assemble_target_dict(
    target_list: list[dict[str, torch.Tensor]], device: torch.device
) -> dict[str, torch.Tensor]:
    """Stack per-event target dicts into a single batch dict.

    Per-particle observables (pt, eta, log_E, log_pt, pid) are padded to the
    batch's max multiplicity; per-event observables are stacked.

    Re-implements the subset of ``delphes_collate_fn`` needed for target-only
    dicts (which lack ``truth_particles``).
    """
    from torch.nn.utils.rnn import pad_sequence

    RAGGED = {"pt", "eta", "log_E", "log_pt", "pid"}
    out: dict[str, torch.Tensor] = {}
    for key in target_list[0]:
        vals = [d[key].to(device) for d in target_list]
        if key in RAGGED:
            if all(v.numel() == 0 for v in vals):
                ref = vals[0]
                out[key] = torch.zeros((len(vals), 0), dtype=ref.dtype, device=device)
            else:
                out[key] = pad_sequence(vals, batch_first=True, padding_value=0.0)
        else:
            out[key] = torch.stack(vals).to(device)
    return out


# =============================================================================
# Gradient-enhanced GP model (GPyTorch + BoTorch)
# =============================================================================


class GradientGPModel(gpytorch.models.ExactGP, GPyTorchModel):
    """A GP model that jointly fits function *values* and *gradients*.

    Uses ``RBFKernelGrad`` so the kernel captures the covariance between
    ``f(x)`` and ``∂f/∂x``.  The ``_num_outputs = 1`` tells BoTorch this is a
    single-output model (the gradients are supporting information, not
    separate outputs).
    """

    _num_outputs = 1  # conceptually one output (the loss)

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        # Single shared lengthscale (no ARD).  In 66-D standardized space the
        # typical pairwise distance is sqrt(2*d) ≈ 11.5, so a lengthscale of
        # ~8 keeps kernel values in the 0.3–0.9 range between typical points.
        # ARD with 66 per-dimension lengthscales from 40 points is hopelessly
        # underdetermined — the GP collapses to the prior (lengthscale=1),
        # which gives exp(-66) ≈ 0 kernel values for ALL point pairs.
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernelGrad(
                lengthscale_prior=gpytorch.priors.GammaPrior(3.0, 0.3),
            ),
            outputscale_prior=gpytorch.priors.GammaPrior(2.0, 0.5),
        )
        # Initialise the lengthscale to sqrt(dim) ≈ 8.1 so the kernel sees
        # correlation from the start instead of starting at 1.0 (near-zero
        # correlation in high-d).
        d = train_x.size(-1)
        init_lengthscale = d ** 0.5
        self.covar_module.base_kernel.lengthscale = init_lengthscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


# =============================================================================
# Diagnostics helpers (Comet-logged GP introspection)
# =============================================================================

_N_RANDOM_DIAG_POINTS = 512


def _extract_likelihood_noise(
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
) -> torch.Tensor | None:
    """Extract the per-task noise tensor from a MultitaskGaussianLikelihood.

    For ``rank=0`` the per-task noises are stored as raw (unconstrained)
    parameters in ``raw_task_noises`` and transformed through
    ``raw_task_noises_constraint``.  Falls back to ``likelihood.noise``
    (the global noise scalar) if per-task noise is not enabled.

    Returns ``None`` if no noise path works.
    """
    with torch.no_grad():
        # rank=0 or rank>0 with has_task_noise → per-task raw parameters
        if hasattr(likelihood, "has_task_noise") and likelihood.has_task_noise:
            if hasattr(likelihood, "raw_task_noises") and likelihood.raw_task_noises is not None:
                raw = likelihood.raw_task_noises
                constraint = getattr(likelihood, "raw_task_noises_constraint", None)
                if constraint is not None:
                    return constraint.transform(raw).detach()
                return raw.detach()
        # Fallback: global noise (scalar, shared across all tasks)
        if hasattr(likelihood, "noise") and likelihood.noise is not None:
            n = likelihood.noise
            if isinstance(n, torch.Tensor):
                return n.detach()
    return None


def _collect_gp_diagnostics(
    model: GradientGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_X_stdz: torch.Tensor,
    train_Y_stdz: torch.Tensor,
    mll: gpytorch.mlls.ExactMarginalLogLikelihood,
) -> dict[str, float]:
    """Collect GP hyperparameters into a dict (always, for YAML + Comet)."""
    diag: dict[str, float] = {}
    with torch.no_grad():
        ls = model.covar_module.base_kernel.lengthscale.detach().flatten()
        diag["gp_lengthscale_min"] = ls.min().item()
        diag["gp_lengthscale_median"] = ls.median().item()
        diag["gp_lengthscale_max"] = ls.max().item()

        diag["gp_outputscale"] = model.covar_module.outputscale.detach().item()

        noise = _extract_likelihood_noise(likelihood)
        if noise is not None:
            if noise.numel() > 1:
                diag["gp_noise_loss_task"] = noise[0].item()
                diag["gp_noise_grad_median"] = noise[1:].median().item()
                diag["gp_noise_grad_max"] = noise[1:].max().item()
            else:
                diag["gp_noise"] = noise.item()

        model.train()
        diag["gp_mll"] = mll(model(train_X_stdz), train_Y_stdz).detach().item()
        model.eval()
    return diag


def _collect_predictive_diagnostics(
    model: GradientGPModel,
    candidate_stdz: torch.Tensor,
    bounds_stdz: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Collect predictive variance diagnostics into a dict."""
    diag: dict[str, float] = {}
    with torch.no_grad():
        posterior_cand = model.posterior(candidate_stdz.unsqueeze(0))
        diag["pred_var_candidate"] = posterior_cand.variance[..., 0].item()

        rand_X = (
            torch.rand(_N_RANDOM_DIAG_POINTS, dim, device=device, dtype=dtype)
            * (bounds_stdz[:, 1] - bounds_stdz[:, 0])
            + bounds_stdz[:, 0]
        )
        posterior_rand = model.posterior(rand_X)
        diag["pred_var_random_mean"] = posterior_rand.variance[..., 0].mean().item()
        diag["pred_var_random_max"] = posterior_rand.variance[..., 0].max().item()
    return diag


def _log_gp_diagnostics(
    model: GradientGPModel,
    likelihood: gpytorch.likelihoods.MultitaskGaussianLikelihood,
    train_X_stdz: torch.Tensor,
    train_Y_stdz: torch.Tensor,
    bounds_stdz: torch.Tensor,
    dim: int,
    device: torch.device,
    iteration: int,
    mll: gpytorch.mlls.ExactMarginalLogLikelihood,
    comet_exp: "comet_ml.Experiment | None" = None,
) -> dict[str, float]:
    """Log GP hyperparameters to Comet; always returns the diagnostic dict."""
    diag = _collect_gp_diagnostics(model, likelihood, train_X_stdz, train_Y_stdz, mll)
    if comet_exp is not None:
        for key, val in diag.items():
            comet_exp.log_metric(key, val, step=iteration)
    return diag


def _log_predictive_diagnostics(
    model: GradientGPModel,
    candidate_stdz: torch.Tensor,
    bounds_stdz: torch.Tensor,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    iteration: int,
    comet_exp: "comet_ml.Experiment | None" = None,
) -> dict[str, float]:
    """Log predictive variance to Comet; always returns the diagnostic dict."""
    diag = _collect_predictive_diagnostics(model, candidate_stdz, bounds_stdz, dim, device, dtype)
    if comet_exp is not None:
        for key, val in diag.items():
            comet_exp.log_metric(key, val, step=iteration)
    return diag


# =============================================================================
# Machine-debug YAML
# =============================================================================

def _append_machine_debug_entry(yaml_path: Path, entry: dict) -> None:
    """Append a diagnostic dict as a new document to a YAML file.

    Uses YAML document separator (``---``) so the file is a stream of
    per-iteration records that any YAML parser can read as a list.
    """
    import yaml as _yaml

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "a") as f:
        _yaml.dump([entry], f, default_flow_style=False, allow_unicode=True,
                    sort_keys=False)


# =============================================================================
# GEBO loop
# =============================================================================


def run_gebo(
    objective: callable,
    dim: int,
    bounds: torch.Tensor,
    n_initial: int = 10,
    n_iterations: int = 50,
    acq_name: str = "EI",
    seed: int = 0,
    verbose: bool = True,
    comet_exp: "comet_ml.Experiment | None" = None,
    machine_debug_path: Path | None = None,
    state_path: Path | None = None,
) -> dict:
    """Run the gradient-enhanced Bayesian optimization loop.

    Parameters
    ----------
    objective : callable
        ``objective(theta_batch) -> (q, d+1)`` tensor.
    dim : int
        Dimensionality of the search space.
    bounds : torch.Tensor
        ``(dim, 2)`` lower and upper bounds.
    n_initial : int
        Number of random initial points (sampled via Sobol).
    n_iterations : int
        Number of BO iterations (TOTAL, including any already-done ones when
        resuming).
    acq_name : str
        Acquisition function: ``"EI"`` or ``"LogEI"``.
    seed : int
        RNG seed for reproducibility.
    verbose : bool
        Whether to print progress.
    state_path : Path | None
        If set, save a checkpoint (``train_X``, ``train_Y``, ``history``,
        ``iteration``) to this file after every BO iteration.  On startup,
        if the file exists and contains a valid state with fewer than
        ``n_iterations`` completed, resume from the saved state instead of
        generating fresh initial points.

    Returns
    -------
    dict
        With keys ``train_X`` (n_points, dim), ``train_Y`` (n_points, d+1),
        ``best_idx``, ``best_loss``, ``best_params``, ``history`` (list of
        per-iteration summaries).
    """
    torch.manual_seed(seed)

    # --- try to resume from a saved state ------------------------------------
    resumed = False
    start_iteration = 1
    train_X: torch.Tensor | None = None
    train_Y: torch.Tensor | None = None
    history: list[dict] = []

    if state_path is not None and state_path.exists():
        ckpt = torch.load(state_path, map_location="cpu", weights_only=True)
        n_done = ckpt.get("iteration", 0)
        if 0 < n_done < n_iterations:
            train_X = ckpt["train_X"].to(dtype=torch.float64)
            train_Y = ckpt["train_Y"].to(dtype=torch.float64)
            history = ckpt.get("history", [])
            start_iteration = n_done + 1
            resumed = True
            if verbose:
                print(
                    f"[gebo] RESUMING from {state_path}: "
                    f"{train_X.shape[0]} points, {n_done}/{n_iterations} iterations done"
                )
            # Re-derive prev_candidate from the last BO point.
            if train_X.shape[0] > n_initial:
                prev_candidate = train_X[-1].clone()
            else:
                prev_candidate = None
        else:
            if verbose and n_done >= n_iterations:
                print(
                    f"[gebo] state file has {n_done} iterations (>= {n_iterations}); "
                    "starting fresh."
                )

    if not resumed:
        # --- generate initial points via Sobol -------------------------------
        init_X = draw_sobol_samples(
            bounds=bounds.T, n=n_initial, q=1, seed=seed
        ).squeeze(1).to(dtype=torch.float64)  # (n_initial, dim)

        if verbose:
            print(f"[gebo] evaluating {n_initial} initial points ...")
        t0 = time.perf_counter()
        init_Y = objective(init_X)  # (n_initial, d+1)
        if verbose:
            print(f"[gebo] initial evaluation took {time.perf_counter() - t0:.1f}s")
            print(f"[gebo] initial best loss: {init_Y[:, 0].min().item():.6e}")

        train_X = init_X.clone()
        train_Y = init_Y.clone()

        # --- machine-debug: initial diagnostics ------------------------------
        if machine_debug_path is not None:
            init_losses = init_Y[:, 0]
            init_grad_norms = init_Y[:, 1:].norm(dim=1)
            _append_machine_debug_entry(machine_debug_path, {
                "iteration": 0,
                "phase": "initial",
            "n_points": n_initial,
            "loss_min": init_losses.min().item(),
            "loss_max": init_losses.max().item(),
            "loss_median": init_losses.median().item(),
            "loss_mean": init_losses.mean().item(),
            "loss_std": init_losses.std().item(),
            "grad_norm_min": init_grad_norms.min().item(),
            "grad_norm_median": init_grad_norms.median().item(),
            "grad_norm_max": init_grad_norms.max().item(),
            "grad_norm_mean": init_grad_norms.mean().item(),
            "all_losses": [float(v) for v in init_losses],
        })

        prev_candidate = None

    # --- common setup for both fresh and resumed paths -----------------------
    best_idx = int(train_Y[:, 0].argmin().item())

    device = train_X.device
    dtype = torch.float64
    train_X = train_X.to(dtype=torch.float64)
    train_Y = train_Y.to(device=device, dtype=torch.float64)

    for iteration in range(start_iteration, n_iterations + 1):
        t_iter = time.perf_counter()

        # --- normalization -------------------------------------------------------
        # Standardize inputs to zero-mean unit-variance per dimension so the ARD
        # kernel sees every dimension on equal footing.
        X_mean = train_X.mean(dim=0, keepdim=True).to(device=device, dtype=torch.float64)
        X_std = train_X.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8).to(device=device, dtype=torch.float64)
        train_X_stdz = (train_X - X_mean) / X_std

        # Standardize the loss (column 0) to zero-mean unit-variance.
        Y_loss = train_Y[:, 0]
        Y_mean = Y_loss.mean().to(device=device, dtype=torch.float64)
        Y_std = Y_loss.std(unbiased=False).clamp(min=1e-8).to(device=device, dtype=torch.float64)
        train_Y_stdz = train_Y.clone()
        train_Y_stdz[:, 0] = (Y_loss - Y_mean) / Y_std

        # Standardize each gradient dimension independently to zero-mean
        # unit-variance so all 67 tasks (1 loss + 66 grads) live on the
        # same scale.  The old chain-rule scaling (grad * X_std / Y_std)
        # shrinks gradients when Y_std is large, causing the GP to dump
        # all loss variation into per-task noise and collapse outputscale.
        grad_cols = train_Y[:, 1:]  # (n, d)
        grad_mean = grad_cols.mean(dim=0, keepdim=True).to(device=device, dtype=torch.float64)
        grad_std = grad_cols.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-8).to(device=device, dtype=torch.float64)
        train_Y_stdz[:, 1:] = (grad_cols - grad_mean) / grad_std

        bounds_dev = bounds.to(device=device, dtype=torch.float64)
        bounds_stdz = torch.stack([
            (bounds_dev[:, 0] - X_mean.squeeze(0)) / X_std.squeeze(0),
            (bounds_dev[:, 1] - X_mean.squeeze(0)) / X_std.squeeze(0),
        ], dim=-1)  # (d, 2)


        # --- fit the gradient-enhanced GP (on standardized data) -------------
        num_tasks = dim + 1
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=num_tasks,
            rank=0,
            has_task_noise=False,  # single shared noise — prevents GP from
                                   # decoupling loss from gradients
            noise_constraint=gpytorch.constraints.GreaterThan(5e-2),
        ).to(device=device, dtype=torch.float64)
        model = GradientGPModel(train_X_stdz, train_Y_stdz, likelihood).to(
            device=device, dtype=torch.float64
        )

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_mll(mll)

        # --- diagnostics: GP hyperparameters ---------------------------------
        gp_diag = _log_gp_diagnostics(
            model=model,
            likelihood=likelihood,
            train_X_stdz=train_X_stdz,
            train_Y_stdz=train_Y_stdz,
            bounds_stdz=bounds_stdz,
            dim=dim,
            device=device,
            iteration=iteration,
            mll=mll,
            comet_exp=comet_exp,
        )

        # --- acquisition function (works in standardized space) ---------------
        best_f = train_Y_stdz[:, 0].min()

        select_loss = ScalarizedPosteriorTransform(
            weights=torch.cat([
                torch.ones(1, device=device, dtype=torch.float64),
                torch.zeros(dim, device=device, dtype=torch.float64),
            ])
        )

        if acq_name == "LogEI":
            acq_func = LogExpectedImprovement(
                model,
                best_f=best_f,
                posterior_transform=select_loss,
                maximize=False,
            )
        elif acq_name == "qLNEI":
            acq_func = qLogNoisyExpectedImprovement(
                model=model,
                X_baseline=train_X_stdz,
                posterior_transform=select_loss,
                prune_baseline=True,
            )
        else:
            acq_func = ExpectedImprovement(
                model,
                best_f=best_f,
                posterior_transform=select_loss,
                maximize=False,
            )

        # --- optimize acquisition in standardized space -----------------------
        candidate_stdz, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds_stdz.T.to(dtype=torch.float64),
            q=1,
            num_restarts=20,
            raw_samples=4096,
        )

        # --- diagnostics: predictive variance at candidate vs random ----------
        pred_diag = _log_predictive_diagnostics(
            model=model,
            candidate_stdz=candidate_stdz,
            bounds_stdz=bounds_stdz,
            dim=dim,
            device=device,
            dtype=dtype,
            iteration=iteration,
            comet_exp=comet_exp,
        )

        # --- un-standardize candidate for objective evaluation ----------------
        candidate = (candidate_stdz.to(dtype=torch.float64) * X_std + X_mean).to(device=device, dtype=torch.float64)

        # --- evaluate the new point (original space) --------------------------
        new_Y = objective(candidate).to(device=device, dtype=torch.float64)  # (1, d+1), in original units

        train_X = torch.cat([train_X, candidate])
        train_Y = torch.cat([train_Y, new_Y])

        best_idx = train_Y[:, 0].argmin().item()
        best_loss = float(train_Y[best_idx, 0])

        # --- diagnostics: candidate distance from previous -------------------
        cand_dist = None
        if prev_candidate is not None:
            cand_dist = (candidate - prev_candidate).norm().item()
        prev_candidate = candidate.clone()

        # --- diagnostics: unique loss count ----------------------------------
        losses = train_Y[:, 0]
        n_unique = len(torch.unique(losses.round(decimals=6)))

        elapsed = time.perf_counter() - t_iter
        if verbose:
            dist_str = f"dist={cand_dist:.3f}  " if cand_dist is not None else ""
            print(
                f"[gebo] iter {iteration:3d}/{n_iterations}  "
                f"new_loss={float(new_Y[0, 0]):.6e}  "
                f"best_loss={best_loss:.6e}  "
                f"acq_val={float(acq_value):.4f}  "
                f"{dist_str}"
                f"uniq={n_unique}  "
                f"({elapsed:.1f}s)"
            )

        # --- log per-iteration metrics to Comet ------------------------------
        if comet_exp is not None:
            comet_exp.log_metric("new_loss", float(new_Y[0, 0]), step=iteration)
            comet_exp.log_metric("best_loss", best_loss, step=iteration)
            comet_exp.log_metric("acq_value", float(acq_value), step=iteration)
            comet_exp.log_metric("n_unique_losses", n_unique, step=iteration)
            if cand_dist is not None:
                comet_exp.log_metric("candidate_distance", cand_dist, step=iteration)

        # --- machine-debug: append per-iteration YAML -------------------------
        if machine_debug_path is not None:
            debug_entry: dict = {
                "iteration": iteration,
                "phase": "bo",
                "new_loss": float(new_Y[0, 0]),
                "best_loss": best_loss,
                "acq_value": float(acq_value),
                "n_unique_losses": n_unique,
                "n_total_points": int(train_X.shape[0]),
                "elapsed_s": elapsed,
            }
            if cand_dist is not None:
                debug_entry["candidate_distance"] = cand_dist
            debug_entry.update(gp_diag)
            debug_entry.update(pred_diag)
            _append_machine_debug_entry(machine_debug_path, debug_entry)

        history.append({
            "iteration": iteration,
            "candidate_loss": float(new_Y[0, 0]),
            "best_loss": best_loss,
            "best_idx": best_idx,
            "acq_value": float(acq_value),
        })

        # --- save checkpoint after every iteration ---------------------------
        if state_path is not None:
            torch.save({
                "train_X": train_X.cpu(),
                "train_Y": train_Y.cpu(),
                "history": history,
                "iteration": iteration,
                "n_initial": n_initial,
            }, state_path)

    return {
        "train_X": train_X,       # (n_total, dim)
        "train_Y": train_Y,       # (n_total, d+1)
        "best_idx": best_idx,
        "best_loss": float(train_Y[best_idx, 0]),
        "best_params": train_X[best_idx],
        "history": history,
    }


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    """Entry point for ``python -m ...tune_cms_fullsim.gebo_search``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-file",
        type=Path,
        required=True,
        help="CMS full-simulation ROOT file (cms-flow schema).",
    )
    parser.add_argument(
        "--n-events",
        type=int,
        default=2000,
        help="Number of events to load for the static evaluation dataset "
             "(default 2000; -1 = all).",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=50,
        help="Number of BO iterations after the initial points.",
    )
    parser.add_argument(
        "--n-initial",
        type=int,
        default=30,
        help="Number of random initial points (Sobol).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for the forward pass during objective evaluation.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="wasserstein_1d",
        choices=list(LOSS_CHOICES),
        help="Loss function (default wasserstein_1d: deterministic, bin-free).",
    )
    parser.add_argument(
        "--acq",
        type=str,
        default="EI",
        choices=["EI", "LogEI", "qLNEI"],
        help="Acquisition function.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("doc/gebo_results"),
        help="Directory to save results.",
    )
    parser.add_argument(
        "--optuna-config",
        type=Path,
        default=Path(pc.__file__).resolve().parent
        / "param_configs"
        / "optuna_config.yaml",
        help="YAML config for parameter bounds (optuna_config format).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed.",
    )
    # Loss weight knobs (passed straight through).
    parser.add_argument("--count-weight", type=float, default=COUNT_WEIGHT)
    parser.add_argument("--calo-count-weight", type=float, default=CALO_COUNT_WEIGHT)
    parser.add_argument("--count-rate-floor", type=float, default=COUNT_RATE_FLOOR)
    parser.add_argument("--event-weight", type=float, default=EVENT_WEIGHT)
    parser.add_argument(
        "--pid-weighting",
        type=str,
        default="sqrt_fraction",
        choices=list(PID_WEIGHTING_CHOICES),
    )
    parser.add_argument("--pid-weight-floor", type=float, default=0.0)
    parser.add_argument(
        "--comet-disabled",
        action="store_true",
        default=False,
        help="Disable Comet ML logging.",
    )
    parser.add_argument(
        "--machine-debug",
        action="store_true",
        default=False,
        help="Write per-iteration diagnostic YAML to --output-dir/gebo_debug.yaml.",
    )
    args = parser.parse_args()

    # --- Comet setup ---------------------------------------------------------
    comet_exp = None
    if not args.comet_disabled and _HAS_COMET and os.environ.get("COMET_API_KEY"):
        comet_exp = comet_ml.Experiment(
            api_key=os.environ["COMET_API_KEY"],
            project_name="DiffDelphes",
            workspace=os.environ.get("COMET_WORKSPACE", ""),
        )
        comet_exp.log_parameters(vars(args))
        print(f"[gebo] Comet experiment: {comet_exp.url}")
    elif args.comet_disabled:
        print("[gebo] Comet logging disabled via --comet-disabled.")
    else:
        print("[gebo] Comet logging disabled (no API key or comet_ml not installed).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gebo] device = {device}")

    # --- load data -----------------------------------------------------------
    print(f"[gebo] loading {args.n_events} events from {args.root_file} ...")
    t0 = time.perf_counter()
    arrays = load_cms_flow_root(args.root_file, n_events=args.n_events)
    truth_ragged = load_truth_events_ragged(arrays)
    target_ragged = load_pflow_targets_ragged(arrays)
    del arrays
    n_loaded = len(truth_ragged)
    print(
        f"[gebo] loaded {n_loaded} events in {time.perf_counter() - t0:.1f}s"
    )

    # --- build vectorizer ----------------------------------------------------
    # Determine trainable parameter keys from the optuna config (any param
    # with a {low, high} spec is trainable; {value} only = pinned).
    with open(args.optuna_config) as f:
        optuna_raw = yaml.safe_load(f)
    param_specs = optuna_raw.get("parameters", {}) if isinstance(optuna_raw, dict) else {}
    trainable_bases: set[str] = set()
    for key, spec in param_specs.items():
        if "value" not in spec:
            trainable_bases.add(key.split("[", 1)[0])

    # Build a probe card to enumerate parameters.
    probe = CMSEnergyFlowDefault(debug=False, learnable=True)
    vectorizer = ParamVectorizer(probe, trainable_keys=trainable_bases)
    print(f"[gebo] search space dimension = {vectorizer.dim}")

    # Build bounds from optuna config.
    bounds = load_bounds_from_optuna_config(args.optuna_config, vectorizer)
    print(
        f"[gebo] bounds range: "
        f"[{bounds[:, 0].min().item():.2f}, {bounds[:, 1].max().item():.2f}]"
    )

    # --- build objective -----------------------------------------------------
    objective = make_objective(
        vectorizer=vectorizer,
        truth_ragged=truth_ragged,
        target_ragged=target_ragged,
        device=device,
        loss_name=args.loss,
        count_weight=args.count_weight,
        calo_count_weight=args.calo_count_weight,
        count_rate_floor=args.count_rate_floor,
        event_weight=args.event_weight,
        pid_weighting=args.pid_weighting,
        pid_weight_floor=args.pid_weight_floor,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # --- run GEBO ------------------------------------------------------------
    print(f"[gebo] starting BO: {args.n_initial} initial + {args.n_iterations} iterations")
    os.makedirs(args.output_dir, exist_ok=True)

    machine_debug_path = args.output_dir / "gebo_debug.yaml" if args.machine_debug else None
    state_path = args.output_dir / "gebo_state.pt"  # always save/load

    if machine_debug_path is not None:
        is_resuming = state_path.exists()
        if not is_resuming:
            machine_debug_path.write_text("")  # fresh run: truncate
        print(f"[gebo] machine-debug YAML -> {machine_debug_path}" +
              (" (appending to existing)" if is_resuming else ""))
    if state_path.exists():
        print(f"[gebo] found checkpoint; will resume from {state_path}")
    else:
        print(f"[gebo] no checkpoint yet — starting fresh")

    result = run_gebo(
        objective=objective,
        dim=vectorizer.dim,
        bounds=bounds,
        n_initial=args.n_initial,
        n_iterations=args.n_iterations,
        acq_name=args.acq,
        seed=args.seed,
        verbose=True,
        comet_exp=comet_exp,
        machine_debug_path=machine_debug_path,
        state_path=state_path,
    )

    # --- save results --------------------------------------------------------
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Best parameters (raw + physical).
    best_raw = result["best_params"]  # (dim,) tensor
    param_names = vectorizer.param_names()

    best_physical: dict[str, float] = {}
    for i, name in enumerate(param_names):
        base = name.split("[", 1)[0]
        best_physical[name] = float(pc.to_physical(base, best_raw[i].unsqueeze(0)))

    # Materialize the best card and dump a standard param config so
    # plot_fit_results (and any downstream consumer) can reload it.
    best_card = CMSEnergyFlowDefault(debug=False, learnable=True)
    vectorizer.set_vector(best_card, best_raw)
    best_config_path = output_dir / "best_config.yaml"
    pc.dump_param_config(best_card, best_config_path)

    # Save summary JSON.
    summary = {
        "best_loss": result["best_loss"],
        "best_idx": result["best_idx"],
        "best_raw_params": {name: float(best_raw[i]) for i, name in enumerate(param_names)},
        "best_physical_params": best_physical,
        "best_config_path": str(best_config_path),
        "n_total_points": int(result["train_X"].shape[0]),
        "dimension": vectorizer.dim,
        "bounds": {
            "low": [float(bounds[i, 0]) for i in range(vectorizer.dim)],
            "high": [float(bounds[i, 1]) for i in range(vectorizer.dim)],
        },
        "history": result["history"],
        "args": vars(args),
    }
    summary_path = output_dir / "gebo_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save the full (X, Y) dataset and parameter names.
    torch.save(
        {
            "train_X": result["train_X"],
            "train_Y": result["train_Y"],
            "param_names": param_names,
        },
        output_dir / "gebo_data.pt",
    )

    print(f"\n[gebo] done.  best_loss = {result['best_loss']:.6e}")
    print(f"[gebo] results saved to {output_dir}/")
    print(f"[gebo] best config   -> {best_config_path}")
    print(f"[gebo] summary       -> {summary_path}")

    if comet_exp is not None:
        comet_exp.end()


if __name__ == "__main__":
    main()
