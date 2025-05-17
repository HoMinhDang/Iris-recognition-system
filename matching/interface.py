from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class IrisMatcher(ABC):
    @abstractmethod
    def match(self, feature1: np.ndarray | Any, feature2: np.ndarray | Any) -> float:
        """Compare two iris features and return similarity score."""
        pass