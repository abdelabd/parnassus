"""
This is the dataloader for delphes model tuning.
"""
import torch
from torch.utils.data import DataLoader, Dataset

from .config import OBSERVABLES


class DelphesDataSet(Dataset):
    """Dataset for Delphes model tuning."""
    def __init__(self, truth_particles: torch.Tensor, target_objects: dict[str, torch.Tensor], device: torch.device) -> None:
        self.truth_particles = truth_particles.to(device)
        for obs in OBSERVABLES:
            self.__setattr__(f"{obs}", target_objects[obs].to(device))

    def __len__(self) -> int:
        return len(self.truth_particles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {"truth_particles": self.truth_particles[idx]}
        for obs in OBSERVABLES:
            item[f"{obs}"] = self.__getattribute__(f"{obs}")[idx]
        return item
    
    
class DelphesDataLoader(DataLoader):
    """DataLoader for Delphes model tuning."""
    def __init__(self, dataset: DelphesDataSet, batch_size: int, shuffle: bool = True) -> None:
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle)
        

