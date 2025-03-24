from dataclasses import dataclass, field
from itertools import starmap
from pathlib import Path
from typing import Any, Self

import yaml

from .data import DatasetConfig
from .model import MODELS_DICT, ModelConfig
from .pipeline import GenPipelineConfig, get_pipeline_config
from .writer import WriterConfig

DEFAULT_MODEL = "cms_2011_flow_v00"


@dataclass(slots=True)
class Config:
    # Writer config
    writer_config: WriterConfig

    # Pipeline configs
    pipeline_configs: list[GenPipelineConfig]

    # Dataset config
    dataset_config: DatasetConfig

    # Generation properties
    model: ModelConfig = field(init=False)
    model_name: str = DEFAULT_MODEL
    num_steps: int = 40
    batch_size: int = 2000

    device: str = "mps"
    gpu_id: int = 0

    def __post_init__(self):
        assert self.model_name == DEFAULT_MODEL, f"Currently only {DEFAULT_MODEL} is supported."

        self.model = MODELS_DICT[self.model_name]

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        output_config_dict = config_dict["output"]
        pipeline_config_dict = config_dict["pipelines"]
        dataset_config_dict = config_dict["dataset"]
        model_config_dict = config_dict["model"]
        return cls(
            pipeline_configs=list(starmap(get_pipeline_config, pipeline_config_dict.items())),
            dataset_config=DatasetConfig.from_dict(dataset_config_dict),
            model_name=model_config_dict["name"],
            num_steps=model_config_dict["num_steps"],
            batch_size=model_config_dict["batch_size"],
            writer_config=WriterConfig.from_dict(output_config_dict),
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> Self:
        if isinstance(config_path, str):
            config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} doesn't exist!")

        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
