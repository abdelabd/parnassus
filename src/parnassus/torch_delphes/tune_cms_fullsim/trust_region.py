"""Trust-region schedules for the GEBO acquisition step.

A *trust region* constrains how far the Bayesian-optimization acquisition
optimizer may move from the current best point each iteration -- effectively a
per-dimension "learning rate" for BO. The concrete schedule is selected in the
run config via a ``{class_path, init_args}`` spec (LightningCLI/jsonargparse
style) and built by :func:`instantiate_trust_region`, so only the
hyperparameters a given schedule actually uses need be supplied (e.g. the
adaptive schedule takes ``patience`` but not ``t0``).

Schedules live in this standalone module (rather than in ``gebo_search``) so the
``class_path`` import and the ``isinstance`` check resolve to the SAME class
object even when ``gebo_search`` is run as ``__main__`` (``python -m ...``),
which would otherwise import the schedule classes twice under two module names.
"""

from __future__ import annotations

import importlib

import numpy as np
import torch


class TrustRegion:
    """Base class for a GEBO trust-region schedule.

    :meth:`setup` derives the per-dimension radii from the search-space geometry,
    :meth:`get_bounds` returns the acquisition box for one iteration, and
    :meth:`update` evolves the radius after each evaluation. Subclasses override
    :meth:`update` (and optionally :meth:`setup`).

    Radii are per-dimension so a param-config ``lr_scale`` (``tr_scales``) can
    shrink or grow individual axes independently.
    """

    radius: torch.Tensor       # current per-dim radius (dim,)
    radius_max: torch.Tensor   # per-dim maximum (dim,)
    radius_min: torch.Tensor   # per-dim minimum (dim,)

    def setup(
        self,
        dim: int,
        bounds: torch.Tensor,
        tr_scales: torch.Tensor | None,
        n_iterations: int,
    ) -> None:
        """Initialize the per-dimension radii from the search-space geometry.

        The initial (max) radius spans half the search-space diameter; each
        dimension is then scaled by ``tr_scales`` (default all-ones).
        """
        radius_init = 0.5 * (bounds[:, 1] - bounds[:, 0]).norm().item()
        scales = (
            tr_scales.to(dtype=torch.float64)
            if tr_scales is not None
            else torch.ones(dim, dtype=torch.float64)
        )
        self._dim = dim
        self._n_iterations = n_iterations
        self._radius_init = radius_init
        self._scales = scales
        self.radius_max = radius_init * scales
        self.radius_min = torch.zeros(dim, dtype=torch.float64)
        self.radius = self.radius_max.clone()

    def get_bounds(
        self, center_stdz: torch.Tensor, bounds_stdz: torch.Tensor
    ) -> torch.Tensor:
        """Return the ``(dim, 2)`` acquisition box: the per-dim radius
        hyperrectangle around ``center_stdz`` intersected with the global
        standardized bounds."""
        r = self.radius.to(device=bounds_stdz.device, dtype=bounds_stdz.dtype)
        lower = torch.clamp(center_stdz - r, min=bounds_stdz[:, 0])
        upper = torch.clamp(center_stdz + r, max=bounds_stdz[:, 1])
        return torch.stack([lower, upper], dim=-1)

    def update(self, improved: bool, verbose: bool = False) -> None:
        """Advance the schedule after one BO iteration (base: no-op)."""


class CosineTrustRegion(TrustRegion):
    """Cosine annealing WITH warm restarts (SGDR-style).

    Each cycle the per-dimension radius decays from its max down to
    ``radius_min_frac`` of the initial radius over ``t0`` iterations, then resets
    to max; the cycle length doubles after every restart. The periodic
    re-expansion forces re-exploration to escape local minima.
    """

    def __init__(self, radius_min_frac: float = 0.01, t0: int | None = None):
        self.radius_min_frac = radius_min_frac
        self.t0 = t0

    def setup(self, dim, bounds, tr_scales, n_iterations):
        super().setup(dim, bounds, tr_scales, n_iterations)
        self.radius_min = self.radius_min_frac * self._radius_init * self._scales
        self._cycle_len = self.t0 if self.t0 is not None else max(n_iterations // 4, 5)
        self._step_in_cycle = 0

    def update(self, improved, verbose=False):
        self._step_in_cycle += 1
        if self._step_in_cycle >= self._cycle_len:
            self._step_in_cycle = 0
            self._cycle_len *= 2
            if verbose:
                print(
                    f"[gebo]         cosine restart → radius reset to per-dim max, "
                    f"next cycle={self._cycle_len} iters"
                )
        progress = self._step_in_cycle / max(self._cycle_len, 1)
        r = 0.5 * (1.0 + np.cos(np.pi * progress))  # in [0, 1]
        self.radius = self.radius_min + (self.radius_max - self.radius_min) * r


class AdaptiveTrustRegion(TrustRegion):
    """TuRBO-style adaptive radius (Eriksson et al. 2019).

    Global success/failure measured by whether each new point improves the best
    loss: after ``patience`` consecutive successes ALL dimensions' radii double;
    after ``patience`` consecutive failures ALL halve. Radii are clamped to
    ``[radius_min, radius_max]``, preserving the per-dimension ``tr_scales``.
    """

    def __init__(self, radius_min_frac: float = 0.01, patience: int = 15):
        self.radius_min_frac = radius_min_frac
        self.patience = patience

    def setup(self, dim, bounds, tr_scales, n_iterations):
        super().setup(dim, bounds, tr_scales, n_iterations)
        self.radius_min = self.radius_min_frac * self._radius_init * self._scales
        self._success = 0
        self._failure = 0

    def update(self, improved, verbose=False):
        if improved:
            self._success += 1
            self._failure = 0
            if self._success >= self.patience:
                self.radius = torch.clamp(self.radius * 2.0, max=self.radius_max)
                self._success = 0
                if verbose:
                    print(f"[gebo]         TR expanded → med={self.radius.median().item():.1f}")
        else:
            self._success = 0
            self._failure += 1
            if self._failure >= self.patience:
                self.radius = torch.clamp(self.radius / 2.0, min=self.radius_min)
                self._failure = 0
                if verbose:
                    print(f"[gebo]         TR shrunk  → med={self.radius.median().item():.1f}")


class NoTrustRegion(TrustRegion):
    """No trust region: the acquisition optimizer searches the full global
    bounds (vanilla BO). The radius stays fixed at its max, purely so the
    per-iteration radius diagnostics still have something to report."""

    def get_bounds(self, center_stdz, bounds_stdz):
        return bounds_stdz  # full global bounds, unchanged


def instantiate_trust_region(spec: dict) -> TrustRegion:
    """Build a :class:`TrustRegion` from a ``{class_path, init_args}`` spec.

    Mirrors the LightningCLI/jsonargparse convention: ``class_path`` is a
    fully-qualified class name and ``init_args`` (optional) its constructor
    kwargs. Only the hyperparameters the chosen schedule actually uses need be
    given -- e.g. the adaptive schedule takes ``patience`` but not ``t0``.
    """
    if not isinstance(spec, dict) or "class_path" not in spec:
        raise SystemExit(
            "trust_region must be a mapping with a 'class_path' (and optional "
            "'init_args'); see the example configs in tune_cms_fullsim/configs/."
        )
    class_path = spec["class_path"]
    init_args = spec.get("init_args") or {}
    module_path, _, cls_name = class_path.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, cls_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise SystemExit(f"trust_region: cannot import class_path {class_path!r} -- {e}")
    try:
        obj = cls(**init_args)
    except TypeError as e:
        raise SystemExit(f"trust_region: bad init_args for {class_path} -- {e}")
    if not isinstance(obj, TrustRegion):
        raise SystemExit(f"trust_region: {class_path} is not a TrustRegion subclass.")
    return obj
