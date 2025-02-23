from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


def do_padding(tensor: Tensor, max_len: int):
    shape = tensor.shape
    new_shape = (max_len,) + shape[1:]
    x = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    x[: shape[0]] = tensor
    return x


@dataclass(kw_only=True)
class DatasetConfig:
    # Data loading configs
    file_path: str | Path
    num_events: int = 1
    entry_start: int = 0

    batch_loading: bool = False
    batch_size: int | None = None

    # Data preprocessing configs
    truth_variables: list[str] = field(
        default_factory=lambda: ["pt", "eta", "phi", "vx", "vy", "vz", "class"]
    )
    max_particles: int = 400
    zero_neutral_vtx: bool = True

    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        if self.batch_loading and self.batch_size is None:
            raise ValueError("Asked for batch_loading, but batch_size is not provided")

        if not Path(self.file_path).exists():
            raise FileNotFoundError(f"No file exist in {self.file_path} location")


class BaseDataset(Dataset[dict[str, Tensor]]):
    @abstractmethod
    def load_data(self):
        pass
