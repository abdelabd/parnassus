from typing import Any, final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
import uproot
from tqdm import tqdm
from typing_extensions import override

from parnassus.utils.transform import VarTransform

from .base import BaseDataset, DatasetConfig, do_padding


@final
class RootDataset(BaseDataset):
    def __init__(self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform]):
        self.cfg = cfg

        self.var_transform_dict = var_transform_dict

        self.truth_variables: list[str] = []
        self.full_data_array: dict[str, npt.NDArray[np.float32]] = {}

        self.entry_start: int
        self.entry_stop: int

        self.n_particle_mask: npt.NDArray[np.bool]
        self.n_truth_particles: npt.NDArray[np.int32]
        self.truth_cumsum: npt.NDArray[np.int64]

    def _update_num_events(self, tree: Any):
        tree_num_entries = tree.num_entries
        if self.cfg.entry_start > tree_num_entries:
            raise ValueError(
                f"Requested entry_start exceeds number of events in the file {self.cfg.file_path}"
            )
        if self.cfg.num_events > (tree_num_entries - self.cfg.entry_start):
            raise ValueError(
                f"""Requested num_events ({self.cfg.num_events}) exceeds number of events
                    in the file {self.cfg.file_path}"""
            )
        self.entry_start = self.cfg.entry_start
        self.entry_stop = self.cfg.entry_start + self.cfg.num_events

    def _preprocess_data(self, mask_events: bool = True):
        for var in tqdm(self.full_data_array.keys()):
            value = self.full_data_array[var]
            if mask_events:
                value = value[self.n_particle_mask]
            if var not in {"ht", "eventNumber", "met_x", "met_y"}:
                value = np.concatenate(value)
            if "eta" in var:
                value = np.clip(value, -3, 3)
            elif "phi" in var:
                value = np.atan2(np.sin(value), np.cos(value))
            self.full_data_array[var] = value

    def _load_truth(self, tree: Any):
        self.n_truth_particles = tree["ntruth"].array(
            library="np",
            entry_stop=self.entry_stop,
            entry_start=self.entry_start,
        )
        self.n_particle_mask = self.n_truth_particles < self.cfg.max_particles

        for var in tqdm(self.cfg.truth_variables):
            self.full_data_array[var] = tree[f"truth_{var}"].array(
                library="np",
                entry_stop=self.entry_stop,
                entry_start=self.entry_start,
            )
            self.truth_variables.append(var if var != "pt" else "ptrel")

        self.full_data_array["ht"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_x"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_y"] = np.zeros(self.cfg.num_events, dtype=np.float32)

        for j in range(self.cfg.num_events):
            self.full_data_array["ht"][j] = self.full_data_array["pt"][j].sum()
            self.full_data_array["met_x"][j] = (
                self.full_data_array["pt"][j] * np.cos(self.full_data_array["phi"][j])
            ).sum()
            self.full_data_array["met_y"][j] = (
                self.full_data_array["pt"][j] * np.sin(self.full_data_array["phi"][j])
            ).sum()

        self.full_data_array["ptrel"] = np.array(
            [x / x.sum() for x in self.full_data_array["pt"]], dtype=object
        )

        _ = self.full_data_array.pop("pt")

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

    @override
    def load_data(self):
        with uproot.open(self.cfg.file_path) as f:  # pyright: ignore  # noqa: PGH003
            tree = f["evt_tree"]
            self._update_num_events(tree)
            self._load_truth(tree)
            self.full_data_array["eventNumber"] = tree["eventNumber"].array(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
                library="np", entry_start=self.entry_start, entry_stop=self.entry_stop
            )
            self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])
            self._preprocess_data()

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
