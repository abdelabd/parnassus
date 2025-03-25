from pathlib import Path
from typing import final

from torch import Tensor, nn
from torch.export import load


@final
class ModelWrapper(nn.Module):
    """A wrapper class for a neural network.

    This class loads a pre-trained neural network model from a specified file path
    and provides a forward method to pass input data through the model.

    """

    def __init__(self, file_path: str | Path):
        super().__init__()
        self.net: nn.Module = load(f=file_path).module()

    def forward(
        self, fs_data: Tensor, tr_data: Tensor, mask: Tensor, timestep: Tensor, global_data: Tensor
    ) -> Tensor:
        return self.net(fs_data, tr_data, mask, timestep, global_data)
