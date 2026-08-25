from abc import ABC, abstractmethod

import numpy as np


class FallbackPolicy(ABC):
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        pass
