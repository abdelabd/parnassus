from abc import ABC, abstractmethod

from parnassus.configs.scheme import GenEvent
from parnassus.configs.writer import WriterConfig


class BaseWriter(ABC):
    def __init__(self, config: WriterConfig):
        super().__init__()
        self.config: WriterConfig = config

    @abstractmethod
    def write(self, events: list[GenEvent]):
        pass
