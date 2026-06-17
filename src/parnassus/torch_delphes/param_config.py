"""Declarative parameter configuration for the learnable CMS TorchDelphes card.

A YAML config lists every learnable scalar of a
:class:`~parnassus.torch_delphes.defaults.CMSEnergyFlowDefault` card -- one
flat entry per scalar, keyed by the ``named_parameters()`` name with an
``[i]`` index suffix for elements of a vector parameter (matching the
``tune_cms_fullsim`` history-snapshot convention) and the bare name for true
scalars. Each entry carries:

- ``value``: the *physical* (post-transform) value to initialize the
  parameter with (e.g. a momentum scale of ``1.25``, a resolution ``a`` of
  ``0.06``, a tracking efficiency of ``0.7``), in interpretable units;
- ``trainable``: whether Adam should optimize that scalar (per-element, so
  one element of a vector can be trained while its siblings are pinned);
- ``lr_scale``: a per-parameter learning-rate ratio; the effective Adam
  learning rate is ``global_lr * lr_scale``.

The same format drives **both** pseudodata generation (use ``value`` as the
ground truth) and tuning (use ``value`` to initialize and ``trainable`` to
select the optimized subset). :func:`to_raw` converts physical values to the
raw parameter space through *range-guarded* inverse transforms, so an
out-of-range target -- e.g. a momentum scale of ``1.3``, which sits on the
open boundary of the ``1 + 0.3*tanh`` parameterization -- raises a clear
:class:`ValueError` instead of silently producing ``NaN`` (which previously
nuked every charged hadron from the generated sample).

This module deliberately imports only from
:mod:`parnassus.torch_delphes.defaults` / :mod:`~parnassus.torch_delphes.learnable`
(not from the ``tune_cms_fullsim`` package) so it never participates in an
import cycle with the tuning CLI.
"""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from parnassus.torch_delphes.learnable import _safe_logit, _softplus_inv

# Scale parameterization: ``scale = _SCALE_CENTER + _SCALE_HALF_WIDTH*tanh(raw)``,
# bounded to the OPEN interval ``(0.7, 1.3)``. The boundary is unreachable
# (``arctanh(±1) = ±inf``), so :func:`to_raw` rejects values on/outside it.
_SCALE_CENTER = 1.0
_SCALE_HALF_WIDTH = 0.3
_SCALE_MIN = _SCALE_CENTER - _SCALE_HALF_WIDTH  # 0.7
_SCALE_MAX = _SCALE_CENTER + _SCALE_HALF_WIDTH  # 1.3

# Allowed initial range for *trainable* ``logit`` parameters. A trainable
# efficiency/fraction initialized at (or near) 0/1 maps to a raw logit with a
# vanishing sigmoid Jacobian -- ``d eff / d logit = eff*(1-eff)`` -- so Adam gets
# essentially no gradient and the parameter is stuck. Initializing a *fitted*
# logit inside (0.1, 0.9) keeps that Jacobian >= 0.09 (well-conditioned). Fixed
# (``trainable: false``) logit params are unconstrained in [0, 1] (e.g. a truth
# efficiency of 0.99 or a 1e-6 fraction is fine -- they never move).
_TRAINABLE_LOGIT_MIN = 0.1
_TRAINABLE_LOGIT_MAX = 0.9

_DTYPE = torch.float64


# =============================================================================
# Per-parameter transforms (single source of truth, shared with _snapshot)
# =============================================================================


def param_transform_kind(name: str) -> str:
    """Return the transform family of a parameter, inferred from its name.

    Mirrors the post-transform logic in
    ``tune_cms_fullsim.training._snapshot`` so the config stores the same
    interpretable values the history/plots show.

    Returns
    -------
    str
        One of ``"scale"``, ``"logit"``, ``"softplus"``, ``"identity"``.
    """
    if name.endswith(".scale_raw") or name == "scale_raw":
        return "scale"
    if name.endswith((".eff_logits", "_logit")):
        return "logit"
    if name.endswith((".rate_raw", ".a_raw", ".b_raw")) or name.startswith(
        ("ECal.resolution_func", "HCal.resolution_func")
    ):
        return "softplus"
    return "identity"


def to_physical(name: str, raw: torch.Tensor) -> torch.Tensor:
    """Map a raw parameter tensor to its interpretable physical value.

    Returns
    -------
    torch.Tensor
        ``1 + 0.3*tanh(raw)`` for scales, ``sigmoid(raw)`` for logits,
        ``softplus(raw)`` for positive coefficients, ``raw`` otherwise.
    """
    kind = param_transform_kind(name)
    if kind == "scale":
        return _SCALE_CENTER + _SCALE_HALF_WIDTH * torch.tanh(raw)
    if kind == "logit":
        return torch.sigmoid(raw)
    if kind == "softplus":
        return F.softplus(raw)
    return raw


def to_raw(name: str, value: float | list[float] | torch.Tensor) -> torch.Tensor:
    """Map a physical value (scalar or 1-D) to the raw parameter space.

    Range-guarded: scales must be in the open ``(0.7, 1.3)``,
    efficiencies/fractions in ``[0, 1]`` (clipped away from the ends), and
    softplus-wrapped coefficients ``> 0``. Out-of-range input raises
    :class:`ValueError` rather than producing ``inf``/``NaN``.

    Returns
    -------
    torch.Tensor
        Raw values (``float64``) whose post-transform equals ``value``,
        same shape as the input.
    """
    kind = param_transform_kind(name)
    v = torch.as_tensor(value, dtype=_DTYPE)
    if kind == "scale":
        if bool(torch.any(v <= _SCALE_MIN)) or bool(torch.any(v >= _SCALE_MAX)):
            raise ValueError(
                f"{name}: scale value(s) {v.flatten().tolist()} must be strictly "
                f"inside the open interval ({_SCALE_MIN}, {_SCALE_MAX}); the "
                f"1 + 0.3*tanh parameterization cannot represent the boundary "
                f"(e.g. 1.3 -> arctanh(1) = inf/NaN). Use e.g. 1.25 instead."
            )
        arg = ((v - _SCALE_CENTER) / _SCALE_HALF_WIDTH).clamp(-1 + 1e-12, 1 - 1e-12)
        return torch.atanh(arg)
    if kind == "logit":
        if bool(torch.any(v < 0.0)) or bool(torch.any(v > 1.0)):
            raise ValueError(
                f"{name}: probability/fraction {v.flatten().tolist()} must be in [0, 1]."
            )
        flat = [_safe_logit(float(x)) for x in v.flatten().tolist()]
        return torch.tensor(flat, dtype=_DTYPE).reshape(v.shape)
    if kind == "softplus":
        if bool(torch.any(v <= 0.0)):
            raise ValueError(
                f"{name}: softplus-wrapped coefficient {v.flatten().tolist()} must be > 0."
            )
        return _softplus_inv(v)
    return v


def default_lr_scale(name: str) -> float:
    """Default per-group learning-rate ratio for a parameter.

    Mirrors ``tune_cms_fullsim.config`` / ``training._classify_parameter``:
    resolution coefficients step 10x slower (``0.1``); scales, efficiency
    (incl. ``rate_raw``) and hadron fractions are ``1.0``.

    Returns
    -------
    float
        The ``lr_scale`` written into a freshly dumped config.
    """
    if name.endswith("scale_raw"):
        return 1.0
    if name.endswith(("eff_logits", "rate_raw")):
        return 1.0
    if "HadronFractions" in name:
        return 1.0
    return 0.1  # resolution: a_raw, b_raw, ECal/HCal resolution_func.*


# =============================================================================
# Config keys: flat per-scalar names
# =============================================================================


def _scalar_keys(name: str, param: torch.Tensor) -> list[str]:
    """Flat config keys for a parameter tensor.

    Returns
    -------
    list[str]
        ``[name]`` for a 0-D scalar; ``[f"{name}[i]" ...]`` for a vector
        (matching the history-snapshot key convention).
    """
    if param.ndim == 0:
        return [name]
    return [f"{name}[{i}]" for i in range(param.numel())]


# =============================================================================
# Load / reassemble / validate
# =============================================================================


def load_param_config(path: str | Path) -> dict[str, dict]:
    """Parse a YAML param config into a flat ``{scalar_key: spec}`` dict.

    Each ``spec`` is normalized to ``{"value": float, "trainable": bool,
    "lr_scale": float}``. ``trainable`` defaults to ``False`` and
    ``lr_scale`` to :func:`default_lr_scale` when omitted.

    Returns
    -------
    dict[str, dict]
        The flat per-scalar configuration (no card validation yet; use
        :func:`apply_param_config` / :func:`select_trainable` which validate
        against a concrete card).
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping of scalar_key -> spec.")
    out: dict[str, dict] = {}
    for key, spec in raw.items():
        if not isinstance(spec, dict) or "value" not in spec:
            raise ValueError(f"{path}: entry {key!r} must be a mapping with a 'value' field.")
        # The base name (strip a trailing ``[i]``) drives the transform kind.
        base = key.split("[", 1)[0]
        value = float(spec["value"])
        trainable = bool(spec.get("trainable", False))
        # Guard trainable logit params away from the dead-gradient sigmoid tails.
        # A non-trainable logit may sit anywhere in [0, 1] (it never moves), but a
        # *fitted* one initialized outside (0.1, 0.9) starts where the sigmoid
        # Jacobian ~ 0, so Adam cannot move it (e.g. value 1.0 -> raw logit ~ +13.8).
        if trainable and param_transform_kind(base) == "logit":
            if not (_TRAINABLE_LOGIT_MIN < value < _TRAINABLE_LOGIT_MAX):
                raise ValueError(
                    f"{path}: {key!r} is a trainable logit parameter with value {value}, "
                    f"which must be strictly inside the open interval "
                    f"({_TRAINABLE_LOGIT_MIN}, {_TRAINABLE_LOGIT_MAX}). Initializing a "
                    f"fitted efficiency/fraction at the 0/1 tails maps to a saturated raw "
                    f"logit with a vanishing gradient, so the fit cannot move it. Use an "
                    f"interior value (e.g. 0.85); only pin the boundary with trainable: false."
                )
        out[str(key)] = {
            "value": value,
            "trainable": trainable,
            "lr_scale": float(spec.get("lr_scale", default_lr_scale(base))),
        }
    return out


def _reassemble(card: nn.Module, flat_cfg: dict[str, dict]) -> dict[str, dict]:
    """Group a flat per-scalar config into per-tensor specs, validating cover.

    Returns
    -------
    dict[str, dict]
        ``{tensor_name: {"value": list, "trainable": list, "lr_scale": list}}``
        in index order, one entry per ``nn.Parameter`` on the card.

    Raises
    ------
    ValueError
        If any scalar key is missing, duplicated, or left over (i.e. the
        config does not match the card's parameter set exactly).
    """
    consumed: set[str] = set()
    per_tensor: dict[str, dict] = {}
    for name, p in card.named_parameters():
        keys = _scalar_keys(name, p)
        missing = [k for k in keys if k not in flat_cfg]
        if missing:
            raise ValueError(
                f"param config is missing entries for {missing} "
                f"(parameter {name!r}, shape {tuple(p.shape)})."
            )
        per_tensor[name] = {
            "value": [flat_cfg[k]["value"] for k in keys],
            "trainable": [flat_cfg[k]["trainable"] for k in keys],
            "lr_scale": [flat_cfg[k]["lr_scale"] for k in keys],
        }
        consumed.update(keys)
    extra = sorted(set(flat_cfg) - consumed)
    if extra:
        raise ValueError(f"param config has unknown entries not on the card: {extra}")
    return per_tensor


# =============================================================================
# Apply / select
# =============================================================================


def apply_param_config(card: nn.Module, flat_cfg: dict[str, dict]) -> None:
    """Initialize every parameter of ``card`` from a flat param config.

    Sets ``param.data`` (in place, under ``no_grad``) to the raw value whose
    post-transform equals the config ``value``. Used by both pseudodata
    generation (the truth card) and tuning (the trainee init).
    """
    per_tensor = _reassemble(card, flat_cfg)
    with torch.no_grad():
        for name, p in card.named_parameters():
            raw = to_raw(name, per_tensor[name]["value"]).reshape(p.shape).to(p.dtype)
            p.copy_(raw)


def select_trainable(
    card: nn.Module,
    flat_cfg: dict[str, dict],
    global_lr: float,
) -> tuple[list[nn.Parameter], list[dict]]:
    """Set ``requires_grad`` per the config and build Adam parameter groups.

    Per tensor: fully-frozen tensors get ``requires_grad_(False)`` and are
    skipped; any-trainable tensors are included with ``requires_grad=True``;
    *partially* trainable tensors additionally get a gradient hook that zeroes
    the frozen elements (plain Adam with no weight decay never moves a
    zero-gradient element, so the pinned siblings stay exactly fixed).

    Adam groups are keyed by the distinct effective learning rate
    ``global_lr * lr_scale``; the trainable elements of a tensor must share a
    single ``lr_scale``.

    Returns
    -------
    (params, param_groups)
        ``params`` is the flat list of optimized tensors; ``param_groups`` is
        the ``torch.optim.Adam`` group list (``{"params", "lr", "name"}``).
    """
    per_tensor = _reassemble(card, flat_cfg)
    params: list[nn.Parameter] = []
    groups_by_lr: dict[float, list[nn.Parameter]] = {}
    for name, p in card.named_parameters():
        spec = per_tensor[name]
        trainable = spec["trainable"]
        if not any(trainable):
            p.requires_grad_(False)
            continue
        p.requires_grad_(True)
        params.append(p)
        if not all(trainable):
            mask = torch.tensor(
                [1.0 if t else 0.0 for t in trainable], dtype=p.dtype, device=p.device
            ).reshape(p.shape)
            p.register_hook(lambda g, m=mask: (g * m) if g is not None else None)
        lr_scales = {ls for ls, t in zip(spec["lr_scale"], trainable) if t}
        if len(lr_scales) != 1:
            raise ValueError(
                f"{name}: the trainable elements must share one lr_scale, got {lr_scales}."
            )
        eff_lr = global_lr * lr_scales.pop()
        groups_by_lr.setdefault(eff_lr, []).append(p)
    param_groups = [
        {"params": ps, "lr": lr, "name": f"lr={lr:g}"} for lr, ps in groups_by_lr.items()
    ]
    return params, param_groups


# =============================================================================
# Dump (seed / regenerate configs)
# =============================================================================


def dump_param_config(card: nn.Module, path: str | Path) -> None:
    """Write ``card``'s current physical values to a flat per-scalar YAML.

    Every entry is written with ``trainable: false`` and ``lr_scale`` seeded
    from :func:`default_lr_scale`. Intended as a starting point: copy the file
    and flip the ``value`` / ``trainable`` fields you want.
    """
    out: dict[str, dict] = {}
    for name, p in card.named_parameters():
        phys = to_physical(name, p.detach()).flatten().tolist()
        lr_scale = default_lr_scale(name)
        for key, val in zip(_scalar_keys(name, p), phys):
            # Round to 8 significant figures to keep the YAML human-readable
            # (drops float-repr noise like 0.9499999999999998) while staying
            # well within round-trip tolerance for these O(1)-O(1e-3) values.
            out[key] = {"value": float(f"{val:.8g}"), "trainable": False, "lr_scale": lr_scale}
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, default_flow_style=False)
