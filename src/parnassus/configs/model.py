from dataclasses import dataclass, field
from pathlib import Path

import yaml

from parnassus.utils import VarTransformConfig


@dataclass
class ModelConfig:
    name: str

    transform_config_path: Path

    event_model_path: Path
    particle_model_path: Path

    var_transform_dict: dict[str, VarTransformConfig] = field(init=False)
    metadata: str | None = None

    def __post_init__(self):
        with open(self.transform_config_path) as f:
            transform_config = yaml.safe_load(f)
            self.var_transform_dict = {
                key: VarTransformConfig(
                    name=key,
                    min=value["min"],
                    max=value["max"],
                    mean=value["mean"],
                    std=value["std"],
                )
                for key, value in transform_config.items()
            }


MODELS_DICT = {
    "cms_2011_flow_v00": ModelConfig(
        name="CMS 2011 Flow v00",
        transform_config_path=Path(__file__)
        .cwd()
        .joinpath("src/parnassus/configs/var_transform_cms.yaml"),
        event_model_path=Path(__file__)
        .cwd()
        .joinpath("src/parnassus/pretrained_models/exported_evt_model.pt2"),
        particle_model_path=Path(__file__)
        .cwd()
        .joinpath("src/parnassus/pretrained_models/exported_part_model.pt2"),
    ),
}
