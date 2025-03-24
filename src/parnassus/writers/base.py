from abc import ABC, abstractmethod

from parnassus.configs.writer import WriterConfig
from parnassus.data.scheme import GenEvent


class BaseWriter(ABC):
    def __init__(self, config: WriterConfig):
        super().__init__()
        self.config: WriterConfig = config

    @abstractmethod
    def write(self, events: list[GenEvent]):
        pass
