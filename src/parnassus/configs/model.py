from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

import yaml

from parnassus.utils import VarTransformConfig
from parnassus.utils.typing import VarNameTuple


@dataclass(slots=True)
class VariablesConfig:
    truth_vars_to_load: VarNameTuple
    fs_vars: VarNameTuple
    ctxt_vars: VarNameTuple
    ctxt_global_vars: VarNameTuple


@dataclass(slots=True)
class SamplerConfig:
    type: Literal["euler"] = "euler"
    num_steps: int = 50
    reverse_time: bool = False


@dataclass(slots=True)
class ModelConfig:
    name: str
    file_path: Path
    variables_config: VariablesConfig
    sampler_config: SamplerConfig = field(default_factory=SamplerConfig)


@dataclass(slots=True)
class GenerativeModelConfig:
    name: str

    max_particles: int
    transform_config_path: Path
    var_transform_dict: dict[str, VarTransformConfig] = field(init=False)

    truth_vars_to_load: VarNameTuple

    event_model_config: ModelConfig
    particle_model_config: ModelConfig
    impact_model_config: ModelConfig | None = None

    tr_output_vars: list[str] = field(init=False)
    pf_output_vars: list[str] = field(init=False)

    def __post_init__(self):
        with open(self.transform_config_path) as f:
            transform_config = yaml.safe_load(f)
            self.var_transform_dict = {
                key: VarTransformConfig(name=key, **value)
                for key, value in transform_config.items()
            }

        self.pf_output_vars = [
            var.replace("pflow_", "")
            for var in (
                *self.particle_model_config.variables_config.fs_vars,
                *(
                    self.impact_model_config.variables_config.fs_vars
                    if self.impact_model_config
                    else []
                ),
            )
        ]
        self.tr_output_vars = [
            var.replace("truth_", "")
            for var in self.event_model_config.variables_config.truth_vars_to_load
        ]

    @classmethod
    def load_from_metadata(cls, metadata_path: Path) -> Self:
        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)
        top_path = metadata_path.parent.absolute()

        event_model_config: ModelConfig | None = None
        particle_model_config: ModelConfig | None = None
        impact_model_config: ModelConfig | None = None

        for key, value in metadata["models"].items():
            variables_config = VariablesConfig(**value["variables"])
            sampler_config = SamplerConfig(**value.get("sampler", {}))
            if key == "event":
                event_model_config = ModelConfig(
                    name="event",
                    file_path=top_path / value["file_name"],
                    variables_config=variables_config,
                    sampler_config=sampler_config,
                )
            elif key == "particle":
                particle_model_config = ModelConfig(
                    name="particle",
                    file_path=top_path / value["file_name"],
                    variables_config=variables_config,
                    sampler_config=sampler_config,
                )
            elif key == "impact":
                impact_model_config = ModelConfig(
                    name="impact",
                    file_path=top_path / value["file_name"],
                    variables_config=variables_config,
                    sampler_config=sampler_config,
                )
            else:
                raise ValueError(f"Unknown model type: {key}")

        if event_model_config is None or particle_model_config is None:
            raise ValueError("Both event and particle models must be defined in metadata.")

        return cls(
            name=metadata["name"],
            max_particles=metadata["max_particles"],
            transform_config_path=top_path / "var_transform.yaml",
            truth_vars_to_load=event_model_config.variables_config.truth_vars_to_load,
            event_model_config=event_model_config,
            particle_model_config=particle_model_config,
            impact_model_config=impact_model_config,
        )


MODELS_DICT = {
    "cms_2011_flow_v00": GenerativeModelConfig.load_from_metadata(
        Path(__file__).parent.parent / "pretrained_models/cms_2011/metadata.yaml"
    )
}
