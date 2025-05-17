from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, normalized_iris: np.ndarray) -> np.ndarray | Any:
        """Extract features from normalized iris image."""
        pass