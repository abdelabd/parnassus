"""
This is the dataloader for delphes model tuning.
"""
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

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
    """DataLoader for Delphes model tuning.

    Accepts an optional ``sampler`` so the CLI can plug in a
    ``torch.utils.data.distributed.DistributedSampler`` under ``srun`` /
    DDP. When a sampler is provided, ``shuffle`` must be False (PyTorch
    requires this; the sampler controls the per-epoch order via
    ``set_epoch``).
    """
    def __init__(
        self,
        dataset: DelphesDataSet,
        batch_size: int,
        shuffle: bool = True,
        sampler: Sampler | None = None,
    ) -> None:
        if sampler is not None:
            # ``DataLoader`` raises a ValueError if both ``shuffle=True`` and a
            # custom sampler are passed; the sampler is responsible for the
            # ordering, so silently turn shuffle off here for caller
            # convenience.
            shuffle = False
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
        

