from typing import final

import numpy as np
import pyhepmc
from typing_extensions import override

from parnassus.utils import pid_to_class
from parnassus.utils.logger import ProgressBar
from parnassus.utils.transform import VarTransform

from .base import BaseDataset, DatasetConfig


@final
class HepMCDataset(BaseDataset):
    def __init__(self, cfg: DatasetConfig, var_transform_dict: dict[str, VarTransform]):
        super().__init__(cfg=cfg, var_transform_dict=var_transform_dict)

    @override
    def load_data(self):
        self.n_truth_particles = np.zeros(self.cfg.num_events, dtype=np.int32)

        for var in ["ptrel", *self.cfg.truth_variables]:
            self.full_data_array[var] = np.zeros(
                self.cfg.num_events * self.cfg.max_particles, dtype=np.float32
            )
            if var != "pt":
                self.truth_variables.append(var)

        self.full_data_array["eventNumber"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["ht"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_x"] = np.zeros(self.cfg.num_events, dtype=np.float32)
        self.full_data_array["met_y"] = np.zeros(self.cfg.num_events, dtype=np.float32)

        curr_event_idx = 0
        curr_particle_idx = 0
        with pyhepmc.open(self.cfg.file_path, "r") as f:
            evt: pyhepmc.GenEvent
            with ProgressBar() as progress:
                task = progress.add_task(
                    "[green]Loading data from HepMC file", total=self.cfg.num_events
                )
                for evt in f:
                    if curr_event_idx == self.cfg.num_events:
                        break
                    event_start_particle_idx = curr_particle_idx
                    num_particles = 0
                    for vtx in evt.vertices:
                        for part in vtx.particles_out:
                            pid = pid_to_class(part.pid)
                            if (
                                part.status != 1
                                or np.abs(part.momentum.eta()) >= 2.7
                                or part.momentum.pt() <= 0.25
                                or abs(pid) in {12, 14, 16}
                            ):
                                continue
                            self.full_data_array["pt"][curr_particle_idx] = part.momentum.pt()
                            self.full_data_array["eta"][curr_particle_idx] = part.momentum.eta()
                            self.full_data_array["phi"][curr_particle_idx] = part.momentum.phi()
                            self.full_data_array["class"][curr_particle_idx] = float(pid)
                            self.full_data_array["vx"][curr_particle_idx] = vtx.position.x
                            self.full_data_array["vy"][curr_particle_idx] = vtx.position.y
                            self.full_data_array["vz"][curr_particle_idx] = vtx.position.z

                            num_particles += 1
                            curr_particle_idx += 1
                    if num_particles >= self.cfg.max_particles:
                        curr_particle_idx -= num_particles
                        continue

                    self.full_data_array["ht"][curr_event_idx] = self.full_data_array["pt"][
                        event_start_particle_idx:curr_particle_idx
                    ].sum()
                    self.full_data_array["ptrel"][event_start_particle_idx:curr_particle_idx] = (
                        self.full_data_array["pt"][event_start_particle_idx:curr_particle_idx]
                        / self.full_data_array["ht"][curr_event_idx]
                    )
                    self.full_data_array["met_x"][curr_event_idx] = (
                        self.full_data_array["pt"][event_start_particle_idx:curr_particle_idx]
                        * np.cos(
                            self.full_data_array["phi"][event_start_particle_idx:curr_particle_idx]
                        )
                    ).sum()
                    self.full_data_array["met_y"][curr_event_idx] = (
                        self.full_data_array["pt"][event_start_particle_idx:curr_particle_idx]
                        * np.sin(
                            self.full_data_array["phi"][event_start_particle_idx:curr_particle_idx]
                        )
                    ).sum()
                    self.n_truth_particles[curr_event_idx] = num_particles
                    self.full_data_array["eventNumber"][curr_event_idx] = float(evt.event_number)
                    curr_event_idx += 1
                    progress.update(task, advance=1)
                if self.cfg.num_events > curr_event_idx:
                    print("Requested more events than in file")
        _ = self.full_data_array.pop("pt")
        for key in self.truth_variables:
            self.full_data_array[key] = self.full_data_array[key][:curr_particle_idx]
        for key in ["ht", "met_x", "met_y"]:
            self.full_data_array[key] = self.full_data_array[key][:curr_event_idx]
        self.n_truth_particles = self.n_truth_particles[:curr_event_idx]
        self.truth_cumsum = np.cumsum([0, *list(self.n_truth_particles)])
