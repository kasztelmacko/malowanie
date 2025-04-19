from abc import ABC, abstractmethod

class ProcessBase(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs):
        pass