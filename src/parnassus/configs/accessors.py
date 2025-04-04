from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, final

from typing_extensions import override

from .scheme import GenEvent


class Accessor(ABC):
    def __init__(
        self, name: str, collection: str, output_name: str | None = None, dtype: str | None = None
    ):
        self._name: str = name
        self._collection: str = collection
        self._output_name: str = output_name or name
        self._dtype: str = dtype or "float32"

    @abstractmethod
    def get(self, event: GenEvent):
        pass

    @property
    def name(self) -> str:
        return self._name

    @property
    def output_name(self) -> str:
        return self._output_name

    @property
    def dtype(self) -> str:
        return self._dtype

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Accessor):
            return NotImplemented
        return self._name == other._name and self._collection == other._collection

    @override
    def __hash__(self) -> int:
        return hash(self._name + self._collection + self._output_name + self._dtype)

    @override
    def __repr__(self) -> str:
        return f"{self._collection} '{self.name}' accessor"


@final
class ParticleAccessor(Accessor):
    @override
    def get(self, event: GenEvent):
        return getattr(getattr(event, self._collection), self._name)


@final
class JetAccessor(Accessor):
    @override
    def get(self, event: GenEvent):
        return getattr(event.jets[self._collection], self._name)


class AccessorStore:
    def __init__(self):
        self.accessors_dict: dict[str, list[Accessor]] = {}

    def update_from_dict(self, data: Mapping[str, Sequence[Accessor]]):
        for key, accessors in data.items():
            if key not in self.accessors_dict:
                self.accessors_dict[key] = list(accessors)
            for accessor in accessors:
                if accessor in self.accessors_dict[key]:
                    pass
                self.accessors_dict[key].append(accessor)

    def get_branch_types(self) -> dict[str, str]:
        return {
            name: "var * {"
            + ", ".join([f'"{accessor.output_name}" : {accessor.dtype}' for accessor in accessors])
            + "}"
            for name, accessors in self.accessors_dict.items()
        }

    def init_data_dict(self) -> dict[str, dict[str, Any]]:
        return {
            name: {accessor.output_name: [] for accessor in accessors}
            for name, accessors in self.accessors_dict.items()
        }

    def update_data_dict(self, event: GenEvent, data_dict: dict[str, dict[str, Any]]):
        for name, accessors in self.accessors_dict.items():
            for accessor in accessors:
                data_dict[name][accessor.output_name].append(accessor.get(event))
