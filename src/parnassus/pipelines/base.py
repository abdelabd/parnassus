from abc import ABC, abstractmethod

from parnassus.data.scheme import GenEvent


class GenPipeline(ABC):
    @abstractmethod
    def process(self, events: list[GenEvent]):
        pass
