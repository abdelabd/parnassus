from dataclasses import dataclass, field
from typing import final

import torch
from torch import Tensor


@dataclass(kw_only=True)
class VarTransformConfig:
    name: str
    transform_type: str = "std"
    transform_fn: str | None = None
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None

    shift: float = field(init=False)
    scale: float = field(init=False)

    def __post_init__(self):
        if self.transform_fn not in {"log", "log1p", None}:
            raise ValueError(
                f"Expected transform_fn for var {self.name} "
                f"be in [None, 'log', 'log1p'], got {self.transform_fn}"
            )
        if self.transform_type not in {"std", "minmax"}:
            raise ValueError(
                f"Expected transform_type for var {self.name} "
                f"be in ['std', 'minmax'], got {self.transform_type}"
            )
        if self.transform_type == "std":
            if self.mean is None or self.std is None:
                raise ValueError(
                    f"For var {self.name} and 'std' transform_type mean and std values "
                    f"should be provided, got mean={self.mean}, std={self.std}"
                )
            self.shift = self.mean
            self.scale = self.std
        if self.transform_type == "minmax":
            if self.min is None or self.max is None:
                raise ValueError(
                    f"For var {self.name} and 'minmax' transform_type min and max values "
                    f"should be provided, got min={self.min}, max={self.max}"
                )
            self.shift = self.min
            self.scale = self.max - self.min


@final
class VarTransform:
    def __init__(self, cfg: VarTransformConfig):
        self.cfg = cfg

    def calculate(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if self.cfg.transform_type == "std":
            mean = x.mean()
            if len(x) < 2:
                if self.cfg.name in {"eta", "phi"}:
                    std = torch.tensor(0.1).float()
                else:
                    std = torch.tensor(1).float()
            else:
                std = x.std()
                if std == 0:
                    std = (
                        torch.tensor(0.1).float()
                        if self.cfg.name in {"eta", "phi"}
                        else torch.tensor(1).float()
                    )
            return mean, std
        min_, max_ = x.min(), x.max()
        if min_ == max_:
            min_, max_ = min_ - 1, max_ + 1
        return min_, max_ - min_

    def transform(
        self, x: Tensor, shift: float | Tensor | None = None, scale: float | Tensor | None = None
    ) -> Tensor:
        if shift is None:
            shift = self.cfg.shift
        if scale is None:
            scale = self.cfg.scale
        if self.cfg.transform_fn == "log":
            x = torch.log(x)
        elif self.cfg.transform_fn == "log1p":
            x = torch.log1p(x)

        return (x - shift) / scale

    def inverse_transform(
        self, x: Tensor, shift: float | Tensor | None = None, scale: float | Tensor | None = None
    ):
        if shift is None:
            shift = self.cfg.shift
        if scale is None:
            scale = self.cfg.scale
        x = x * scale + shift
        if self.cfg.transform_fn == "log":
            return torch.exp(x)
        if self.cfg.transform_fn == "log1p":
            return torch.expm1(x)
        return x
