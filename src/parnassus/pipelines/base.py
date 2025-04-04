from abc import ABC, abstractmethod

from parnassus.configs.accessors import Accessor
from parnassus.configs.scheme import GenEvent


class GenPipeline(ABC):
    @abstractmethod
    def process(self, events: list[GenEvent]):
        pass

    @abstractmethod
    def get_accessors(self) -> dict[str, list[Accessor]]:
        pass
