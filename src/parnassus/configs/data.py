from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self


@dataclass(kw_only=True)
class DatasetConfig:
    # Data loading configs
    file_path: Path | str
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

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls(**config)
