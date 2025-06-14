from dataclasses import dataclass, field
from typing import Any, Self

from fastjet import JetDefinition, antikt_algorithm, ee_genkt_algorithm


@dataclass(slots=True)
class GenPipelineConfig:
    name: str

    @classmethod
    def from_dict(cls, name: str, config: dict[str, Any]) -> Self:
        return cls(name, **{field: config[field] for field in cls.__slots__ if field in config})


@dataclass(slots=True)
class JetClusteringConfig(GenPipelineConfig):
    algorithm: str = "antikt"
    collection: str = "pflow"
    dr: float = 0.5
    nconst_min: int = 2
    min_pt: float = 0
    num_processes: int = 1

    jet_definition: JetDefinition = field(init=False)

    def __post_init__(self):
        if self.algorithm == "genkt":
            self.jet_definition = JetDefinition(ee_genkt_algorithm, self.dr, -1.0)
        elif self.algorithm == "antikt":
            self.jet_definition = JetDefinition(antikt_algorithm, self.dr)
        else:
            raise NotImplementedError(f"Jet algorithm {self.algorithm} is not supported!")
        if self.collection not in {"pflow", "truth"}:
            raise ValueError(
                f'Requested clustering {self.collection}, only "pflow" and "truth" are supported.'
            )


@dataclass(slots=True)
class IsolationConfig(GenPipelineConfig):
    collection: str = "electrons"
    dr: float = 0.4
    num_processes: int = 1

    def __post_init__(self):
        if self.collection not in {"electrons", "muons", "all"}:
            raise ValueError(
                f"Requested isolation for {self.collection}, "
                'only "electrons" and "muons", and "all" (both of them) are supported.'
            )
        if self.dr <= 0:
            raise ValueError(f'Only positive values of "dR" are supported, asked for {self.dr}')


def get_pipeline_config(name: str, config: dict[str, Any]) -> GenPipelineConfig:
    match config["type"]:
        case "cluster":
            return JetClusteringConfig.from_dict(name=name, config=config)
        case "isolation":
            return IsolationConfig.from_dict(name=name, config=config)
        case _:
            return GenPipelineConfig(name=name)
