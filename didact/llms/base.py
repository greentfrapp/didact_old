from abc import ABC, abstractmethod
import numpy as np


class BaseLLM(ABC):
    @abstractmethod
    async def aembed(self, value: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed(self, value: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    async def aprompt(self, value: str) -> str:
        raise NotImplementedError
    
    @abstractmethod
    def prompt(self, value: str) -> str:
        raise NotImplementedError
