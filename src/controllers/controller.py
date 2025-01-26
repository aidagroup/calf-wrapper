from abc import ABC, abstractmethod
import numpy as np


class Controller(ABC):
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        pass
