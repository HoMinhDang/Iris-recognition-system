from abc import ABC, abstractmethod
import numpy as np
from typing import Any

class IrisNormalizer(ABC):
    @abstractmethod
    def normalize(self, image: np.ndarray, segmentation_output: Any) -> np.ndarray | Any:
        """Normalize iris to a fixed-size rectangular representation."""
        pass