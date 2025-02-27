from pathlib import Path
from typing import final

from torch import Tensor, nn
from torch.export import load


@final
class ParticleModelWrapper(nn.Module):
    def __init__(self, file_path: str | Path):
        super(nn.Module, self).__init__()
        self.net: nn.Module = load(f=file_path).module()

    def forward(
        self, fs_data: Tensor, tr_data: Tensor, mask: Tensor, timestep: Tensor, global_data: Tensor
    ) -> Tensor:
        return self.net(fs_data, tr_data, mask, timestep, global_data)


@final
class EventModelWrapper(nn.Module):
    def __init__(self, file_path: str | Path):
        super(nn.Module, self).__init__()
        self.net: nn.Module = load(f=file_path).module()

    def forward(
        self, fs_data: Tensor, tr_data: Tensor, mask: Tensor, timestep: Tensor, global_data: Tensor
    ) -> Tensor:
        return self.net(fs_data, tr_data, mask, global_data, timestep)
