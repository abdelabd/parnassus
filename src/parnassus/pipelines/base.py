from abc import ABC, abstractmethod

from parnassus.data.scheme import GenEvent


class GenPipeline(ABC):
    def __init__(self, name: str):
        self.name: str = name

    @abstractmethod
    def process(self, events: list[GenEvent]):
        pass
