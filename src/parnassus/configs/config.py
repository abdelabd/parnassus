from dataclasses import dataclass, field
from itertools import starmap
from pathlib import Path
from typing import Any, Self

import yaml

from .data import DatasetConfig
from .model import MODELS_DICT, GenerativeModelConfig
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
    model: GenerativeModelConfig = field(init=False)
    model_name: str = DEFAULT_MODEL
    num_steps: int = 40
    batch_size: int = 2000

    device: str = "mps"
    gpu_id: int = 0

    def __post_init__(self):
        # Validate model name
        if self.model_name not in MODELS_DICT:
            available = ", ".join(MODELS_DICT.keys())
            raise ValueError(
                f"Unknown model '{self.model_name}'. Available models: {available}. "
                f"Currently only {DEFAULT_MODEL} is fully supported."
            )

        self.model = MODELS_DICT[self.model_name]

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> Self:
        """Create Config from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Dictionary containing configuration sections (output, pipelines, dataset, model).

        Returns
        -------
        Config
            A new Config instance with properly initialized components.

        Raises
        ------
        ValueError
            If the specified model name is not found in MODELS_DICT.
        """
        output_config_dict = config_dict["output"]
        pipeline_config_dict = config_dict["pipelines"]
        dataset_config_dict = config_dict["dataset"]
        model_config_dict = config_dict["model"]

        # Get model config first to extract variable requirements
        model_name = model_config_dict["name"]
        if model_name not in MODELS_DICT:
            available = ", ".join(MODELS_DICT.keys())
            raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

        model_config = MODELS_DICT[model_name]

        # Create dataset config with variables and max_particles from model
        dataset_config = DatasetConfig.from_dict_and_model(dataset_config_dict, model_config)

        return cls(
            pipeline_configs=list(starmap(get_pipeline_config, pipeline_config_dict.items())),
            dataset_config=dataset_config,
            model_name=model_name,
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
