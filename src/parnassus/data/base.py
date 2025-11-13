from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from parnassus.configs.data import DatasetConfig
from parnassus.utils.logger import ProgressBar
from parnassus.utils.transform import VarTransform
from parnassus.utils.typing import FloatArray, VarNameTuple

if TYPE_CHECKING:
    from parnassus.utils.typing import BoolArray, IntArray, LongArray


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

        self.truth_vars_to_load: VarNameTuple = cfg.truth_vars_to_load
        self.ctxt_vars: list[str] = [var.replace("truth_", "") for var in cfg.ctxt_vars]
        self.ctxt_global_vars: list[str] = [
            var.replace("truth_", "") for var in cfg.ctxt_global_vars
        ]

        self.full_data_array: dict[str, FloatArray] = {}

        self.n_particle_mask: BoolArray
        self.n_truth_particles: IntArray
        self.truth_cumsum: LongArray
        self.eventNumber: LongArray

        if not Path(self.cfg.file_path).exists():
            raise FileNotFoundError(f"Trying to load file {self.cfg.file_path}, no file exist!")
        self.load_data()
        self._validate_required_attributes()
        self._preprocess_data()

        self.n_events = len(self.n_truth_particles)
        self.scaled_ctxt_global_data: Tensor = self._prepare_ctxt_global_data()

    def _validate_required_attributes(self) -> None:
        """Validate that all required attributes are set by load_data().

        Raises
        ------
        AttributeError
            If any required attribute is not set or is None.
        """
        required_attrs = {
            "n_truth_particles": "IntArray",
            "truth_cumsum": "LongArray",
            "eventNumber": "IntArray",
        }

        for attr_name, expected_type in required_attrs.items():
            if not hasattr(self, attr_name):
                raise AttributeError(
                    f"'{attr_name}' not set in load_data(). Expected type: {expected_type}"
                )
            attr_value = getattr(self, attr_name)
            if attr_value is None:
                raise AttributeError(
                    f"'{attr_name}' is None after load_data(). Expected type: {expected_type}"
                )

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

    def _calculate_means(self) -> FloatArray:
        n_vars = len([el for el in self.ctxt_vars if "class" not in el])
        means = np.zeros((self.n_events, n_vars), dtype=np.float32)
        with ProgressBar() as progress:
            task = progress.add_task("[green]Calculating means", total=len(self.ctxt_vars))
            for i, var in enumerate(self.ctxt_vars):
                if "class" not in var:
                    means[:, i] = (
                        np.add.reduceat(
                            self.full_data_array[var],
                            self.truth_cumsum[:-1],
                        )
                        / self.n_truth_particles
                    )
                progress.update(task, advance=1)
        return means

    def _prepare_ctxt_global_data(self) -> Tensor:
        scaled_ctxt_global_data_list: list[Tensor] = []
        means = self._calculate_means()
        for var in self.ctxt_global_vars:
            if var == "means":
                scaled_ctxt_global_data_list.append(torch.tensor(means, dtype=torch.float32))
            elif var.startswith("ntruth"):
                var_transform = self.var_transform_dict["npart"]
                scaled_ctxt_global_data_list.append(
                    var_transform.transform(
                        torch.tensor(
                            self.n_truth_particles,
                            dtype=torch.float32,
                        ).view(-1, 1)
                    )
                )
            else:
                var_transform = self.var_transform_dict[var]
                scaled_ctxt_global_data_list.append(
                    var_transform.transform(
                        torch.tensor(
                            self.full_data_array[var],
                            dtype=torch.float32,
                        ).view(-1, 1)
                    )
                )
        return torch.cat(scaled_ctxt_global_data_list, dim=-1)

    def _get_data(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the context data for the given index.

        Args:
            idx (int): index of the event

        Returns
        -------
        ctxt_data (torch.Tensor): context data for the event
        ctxt_global_data (torch.Tensor): global context data for the event
        mask (torch.Tensor): mask for the event
        """
        n_truth_particles = self.n_truth_particles[idx]
        truth_start, truth_end = self.truth_cumsum[idx], self.truth_cumsum[idx + 1]

        truth_idx = np.argsort(self.full_data_array["ptrel"][truth_start:truth_end], axis=0)

        ctxt_data_list = []
        for var in self.ctxt_vars:
            x = torch.tensor(self.full_data_array[var][truth_start:truth_end][truth_idx]).view(
                -1, 1
            )
            if var == "phi":
                ctxt_data_list.extend([
                    torch.sin(x).float(),
                    torch.cos(x).float(),
                ])
            elif var == "class":
                ctxt_data_list.append(F.one_hot(x.long().squeeze(-1), num_classes=5).float())
            else:
                var_transform = self.var_transform_dict[var]
                ctxt_data_list.append(var_transform.transform(x).float())

        ctxt_data = torch.cat(ctxt_data_list, dim=-1)
        ctxt_data = do_padding(ctxt_data, self.cfg.max_particles)
        ctxt_global_data = self.scaled_ctxt_global_data[idx]
        mask = torch.zeros((self.cfg.max_particles,), dtype=torch.bool)
        mask[:n_truth_particles] = 1

        return ctxt_data, ctxt_global_data, mask

    def __len__(self):
        return len(self.n_truth_particles)

    def __getitem__(self, idx: Any) -> dict[str, Tensor]:  # pyright: ignore[reportImplicitOverride]
        ctxt_data, ctxt_global_data, mask = self._get_data(idx)
        event_number = torch.tensor(self.eventNumber[idx], dtype=torch.long).unsqueeze(-1)
        return {
            "ctxt_data": ctxt_data,
            "ctxt_global_data": ctxt_global_data,
            "mask": mask,
            "event_number": event_number,
        }

    @abstractmethod
    def load_data(self):
        pass
