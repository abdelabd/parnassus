from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from parnassus.configs.data import DatasetConfig
from parnassus.utils.logger import ProgressBar
from parnassus.utils.transform import VarTransform


def do_padding(tensor: Tensor, max_len: int):
    shape = tensor.shape
    new_shape = (max_len,) + shape[1:]
    x = torch.zeros(new_shape, dtype=tensor.dtype, device=tensor.device)
    x[: shape[0]] = tensor
    return x


class BaseDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform]):
        super().__init__()
        self.cfg: DatasetConfig = cfg

        self.var_transform_dict: dict[str, VarTransform] = var_transform_dict

        self.truth_variables: list[str] = []
        self.full_data_array: dict[str, npt.NDArray[np.float32]] = {}

        self.entry_start: int
        self.entry_stop: int

        self.n_particle_mask: npt.NDArray[np.bool]
        self.n_truth_particles: npt.NDArray[np.int32]
        self.truth_cumsum: npt.NDArray[np.int64]

        if not Path(self.cfg.file_path).exists():
            raise FileNotFoundError(f"Trying to load file {self.cfg.file_path}, no file exist!")
        self.load_data()

    def _preprocess_data(self, mask_events: bool = True):
        with ProgressBar() as progress:
            task = progress.add_task(
                "[green]Preprocessing data", total=len(self.full_data_array.keys())
            )
            for var in self.full_data_array:
                value = self.full_data_array[var]
                if mask_events:
                    value = value[self.n_particle_mask]
                if var not in {"ht", "eventNumber", "met_x", "met_y"} and value.dtype == object:
                    value = np.concatenate(value)
                if "eta" in var:
                    value = np.clip(value, -3, 3)
                elif "phi" in var:
                    value = np.atan2(np.sin(value), np.cos(value))
                self.full_data_array[var] = value
                progress.update(task, advance=1)

    def _get_truth_data(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Returns the truth particle data for the given index.

        Args:
            idx (int): index of the event

        Returns
        -------
            truth_data (dict): dictionary containing the truth data
            truth_mask (torch.Tensor): mask for the truth data
            global_data (torch.Tensor): tensor containing the global data
                                        (shift and scale from the truth data)
        """
        n_truth_particles = self.n_truth_particles[idx]
        truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx + 1]

        truth_vars = {
            key: torch.tensor(self.full_data_array[key][truth_start:truth_end])
            for key in self.truth_variables
        }

        truth_data: dict[str, torch.Tensor] = {}
        global_data: dict[str, torch.Tensor] = {}

        truth_idx = torch.argsort(truth_vars["ptrel"], descending=True)

        for var_name in self.truth_variables:
            var_data = truth_vars[var_name][truth_idx]
            if var_name == "class":
                truth_data[var_name] = F.one_hot(var_data.long(), 5).float()
                continue
            var_transform = self.var_transform_dict[var_name]
            shift, scale = var_transform.calculate(var_data)
            global_data[var_name + "_shift"] = shift
            global_data[var_name + "_scale"] = scale
            truth_data[var_name] = (
                var_transform.transform(truth_vars[var_name][truth_idx]).float().unsqueeze(-1)
            )

        truth_mask = torch.zeros(self.cfg.max_particles)
        truth_mask[:n_truth_particles] = 1

        return truth_data, truth_mask, global_data

    def __len__(self):
        return len(self.n_truth_particles)

    def __getitem__(self, idx: Any) -> dict[str, torch.Tensor]:  # pyright: ignore[reportImplicitOverride]
        n_truth_particles = self.n_truth_particles[idx]
        truth_data_dict, truth_mask, event_data_dict = self._get_truth_data(idx)
        truth_data = torch.cat([truth_data_dict[key] for key in self.truth_variables], -1)
        truth_data = do_padding(truth_data, max_len=self.cfg.max_particles)

        event_data = torch.cat([
            torch.stack(list(event_data_dict.values()), -1).to(torch.float32),
            self.var_transform_dict["met_x"].transform(
                torch.tensor(self.full_data_array["met_x"][idx], dtype=torch.float32).unsqueeze(-1)
            ),
            self.var_transform_dict["met_y"].transform(
                torch.tensor(self.full_data_array["met_y"][idx], dtype=torch.float32).unsqueeze(-1)
            ),
            self.var_transform_dict["npart"]
            .transform(torch.tensor(n_truth_particles, dtype=torch.float32).unsqueeze(-1))
            .float(),
            self.var_transform_dict["ht"]
            .transform(
                torch.tensor(self.full_data_array["ht"][idx], dtype=torch.float32).unsqueeze(-1)
            )
            .float(),
        ])

        return {"truth_data": truth_data, "truth_mask": truth_mask, "event_data": event_data}

    @abstractmethod
    def load_data(self):
        pass
