"""
This is the dataloader for delphes model tuning.

The dataset stores the truth particles and the per-particle target observables
*ragged* (a list of per-event tensors, no padding) and pads each batch only to
its own max multiplicity in :func:`delphes_collate_fn`. This is numerically
identical to padding every event to the GLOBAL max multiplicity (the padded
slots are all-zero, so the fit loop's ``any(... != 0)`` truth mask and the loss's
``pid == 0`` drop behave the same) but uses an order of magnitude less memory --
the dense global padding allocates ~100k x ~3000 x 21 x 8 B ~ 50 GB up front,
which the fit loop then immediately un-pads.
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import OBSERVABLES

# The per-particle (variable-length) observables: padded per batch. Every other
# key in OBSERVABLES is per-event (fixed shape) and is stacked instead.
RAGGED_OBSERVABLES: tuple[str, ...] = ("pt", "eta", "log_E", "log_pt", "pid")


def _to_device(value, device):
    """Move a ragged per-event list OR a dense per-event tensor to ``device``."""
    if isinstance(value, list):
        return [t.to(device) for t in value]
    return value.to(device)


class DelphesDataSet(Dataset):
    """Dataset for Delphes model tuning (ragged; per-batch padding in collate)."""
    def __init__(self, truth_particles: list[torch.Tensor], target_objects: dict, device: torch.device) -> None:
        # truth_particles: list of (n_particles_i, N_FEATURES) tensors, one per event.
        self.truth_particles = [t.to(device) for t in truth_particles]
        for obs in OBSERVABLES:
            # Per-particle obs are ragged lists; per-event obs are dense tensors.
            self.__setattr__(f"{obs}", _to_device(target_objects[obs], device))

    def __len__(self) -> int:
        return len(self.truth_particles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # ``[idx]`` selects one event whether the attribute is a list (ragged
        # per-particle obs -> a 1-D tensor) or a tensor (per-event obs -> a row).
        item = {"truth_particles": self.truth_particles[idx]}
        for obs in OBSERVABLES:
            item[f"{obs}"] = self.__getattribute__(f"{obs}")[idx]
        return item


def delphes_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate by padding each batch to its OWN max multiplicity.

    ``pad_sequence`` pads dim 0 of each variable-length tensor with ``0.0``,
    matching the old dense zero-padding exactly. Per-event observables are
    stacked. The result is the same batch dict the fit loop consumes, just with a
    per-batch (not global) padded width.
    """
    out: dict[str, torch.Tensor] = {}

    truth = [b["truth_particles"] for b in batch]  # each (n_i, N_FEATURES)
    if all(t.shape[0] == 0 for t in truth):  # whole batch empty (mirrors restore_event_format guard)
        ref = truth[0]
        out["truth_particles"] = torch.zeros(
            (len(truth), 0, ref.shape[-1]), dtype=ref.dtype, device=ref.device
        )
    else:
        out["truth_particles"] = pad_sequence(truth, batch_first=True, padding_value=0.0)

    for obs in OBSERVABLES:
        vals = [b[obs] for b in batch]
        if obs in RAGGED_OBSERVABLES:
            if all(v.shape[0] == 0 for v in vals):  # no real particles anywhere in the batch
                ref = vals[0]
                out[obs] = torch.zeros((len(vals), 0), dtype=ref.dtype, device=ref.device)
            else:
                out[obs] = pad_sequence(vals, batch_first=True, padding_value=0.0)
        else:
            out[obs] = torch.stack(vals)
    return out


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
        drop_last: bool = False,
        sampler: Sampler | None = None,
    ) -> None:
        if sampler is not None:
            # DataLoader forbids setting both shuffle=True and a custom
            # sampler; let the sampler own ordering.
            shuffle = False
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=delphes_collate_fn,
            # The dataset holds tensors already on ``device``; keep loading in the
            # main process so we never fork/pickle them (CUDA tensors can't cross a
            # worker fork). The per-batch padding work is trivial.
            num_workers=0,
        )
