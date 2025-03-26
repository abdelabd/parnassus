from abc import ABC, abstractmethod

from parnassus.configs.scheme import GenEvent


class GenPipeline(ABC):
    @abstractmethod
    def process(self, events: list[GenEvent]):
        pass
