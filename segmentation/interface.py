from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class IrisSegmentator(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | any:
        """Segment iris and pupil boundaries from input image.
        Returns: (iris_mask, pupil_mask)"""
        pass
    
    def get_masks(self, image: np.ndarray) -> list[Tuple[str, np.ndarray]]:
        """Get segmentation masks for the given image."""
        pass