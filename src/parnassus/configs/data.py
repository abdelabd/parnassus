from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from parnassus.utils.typing import VarNameTuple


@dataclass(kw_only=True)
class DatasetConfig:
    # Data loading configs
    file_path: Path | str
    num_events: int = 1
    entry_start: int = 0

    batch_loading: bool = False
    batch_size: int | None = None

    # Data preprocessing configs
    max_particles: int = 400

    truth_vars_to_load: VarNameTuple = field(init=False)
    ctxt_vars: VarNameTuple = field(init=False)
    ctxt_global_vars: VarNameTuple = field(init=False)

    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

        if self.batch_loading and self.batch_size is None:
            raise ValueError("Asked for batch_loading, but batch_size is not provided")

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls(**config)
