import itertools
from abc import ABC, abstractmethod
from typing import final

import torch
from torch import Tensor, nn
from torch.distributions import Normal
from tqdm import tqdm
from typing_extensions import override

default_device = torch.device("cpu")


class Sampler(ABC):
    def __init__(
        self,
        n_steps: int,
        zero_init_padded: bool = True,
        random_seed: int | None = None,
    ):
        """Base class for sampler.
        Args:
        n_steps: Number of steps to take
        zero_init_padded: Whether to zero out the padded values
        random_seed: Random seed for reproducibility.
        """
        self.n_steps: int = n_steps
        self.zero_init_padded: bool = zero_init_padded

        t_steps = torch.linspace(0, 1, n_steps)
        self.t_steps: Tensor = torch.cat([t_steps, torch.ones_like(t_steps)[:1]])

        if random_seed is not None:
            _ = torch.manual_seed(random_seed)
        self.random_seed: int | None = random_seed

        self.distribution: Normal = Normal(0, 1)

    def reset(self, random_seed: int | None = None):
        if random_seed is None:
            assert self.random_seed is not None, "Random seed not provided"
            random_seed = self.random_seed
        _ = torch.manual_seed(random_seed)

    def _init_fastsim(
        self,
        shape: list[int],
        mask: Tensor | None = None,
        device: torch.device = default_device,
    ):
        fastsim = self.distribution.sample(shape).to(device)
        if mask is not None and self.zero_init_padded and mask.shape[-1] == 2:
            fastsim[~mask[..., 1]] = 0
        return fastsim

    @torch.no_grad()
    def _transfer(self, x_curr: Tensor, d_curr: Tensor, dt: Tensor) -> Tensor:
        """Do the integration step.

        Args:
            x_curr: Current state
            d_curr: Current derivative
            dt: Time step

        Returns
        -------
            x_next: Next state
        """
        return x_curr + d_curr * dt

    @torch.no_grad()
    def _step(
        self,
        model: nn.Module,
        truth: Tensor,
        fastsim: Tensor,
        mask: Tensor,
        global_data: Tensor,
        timestep: Tensor,
        dt: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Do a single (euler) step of the model.

        Returns
        -------
            fastsim_next: The next state of the fastsim
            deriv: Model output at the current timestep (dx/dt)

        """
        deriv = model.forward(
            fastsim,
            truth,
            mask,
            timestep=timestep.expand(fastsim.shape[0]),
            global_data=global_data,
        )
        fastsim_next = self._transfer(fastsim, deriv, dt)
        return fastsim_next, deriv

    @abstractmethod
    def sample(
        self,
        model: nn.Module,
        truth: Tensor,
        pflow_shape: list[int],
        mask: Tensor,
        global_data: Tensor,
        save_seq: bool = False,
        init_fastsim: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        pass


@final
class EulerSampler(Sampler):
    @override
    def sample(
        self,
        model: nn.Module,
        truth: Tensor,
        pflow_shape: list[int],
        mask: Tensor,
        global_data: Tensor,
        save_seq: bool = False,
        init_fastsim: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        device = truth.device
        if init_fastsim is None:
            fastsim = self._init_fastsim(pflow_shape, mask, device)
        else:
            fastsim = init_fastsim
        t_steps = self.t_steps.to(device)
        seq = [fastsim.cpu()]
        for t_cur, t_next in tqdm(itertools.pairwise(t_steps), total=self.n_steps):
            fastsim, _ = self._step(
                model,
                truth,
                fastsim,
                mask,
                global_data,
                timestep=t_cur,
                dt=t_next - t_cur,
            )
            if save_seq:
                seq.append(fastsim.cpu())
        if save_seq:
            seq = torch.stack(seq)
            return fastsim.cpu(), seq
        return fastsim.cpu()
