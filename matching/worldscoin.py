import numpy as np
from .interface import IrisMatcher
import iris
from typing import Any

class WorldCoinsMatcher(IrisMatcher):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.matcher = iris.HammingDistanceMatcher()
        
    def match(self, feature1: np.ndarray | Any, feature2: np.ndarray | Any) -> float:
        # make sure the feature are the IrisTemplate objects
        if not isinstance(feature1, iris.IrisTemplate):
            feature1 = iris.IrisTemplate(feature1)
        if not isinstance(feature2, iris.IrisTemplate):
            feature2 = iris.IrisTemplate(feature2)
        return self.matcher.run(feature1, feature2)