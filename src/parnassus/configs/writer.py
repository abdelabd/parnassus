from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

from parnassus.configs.accessors import AccessorStore


@dataclass(kw_only=True)
class WriterConfig:
    # Data writer configs
    file_path: Path | str

    format: str = "default"
    accessor_store: AccessorStore = field(default_factory=AccessorStore)

    def __post_init__(self):
        if isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls(**config)
