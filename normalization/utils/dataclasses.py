import numpy as np
from typing import Any, List, Tuple, Dict, Callable, Literal
import math

# Reference from iris library

class SegmentationMap:
    def __init__(self, predictions: np.ndarray, index2class: Dict[int, str]) -> None:
        self.predictions = predictions
        self.index2class = index2class

    def index_of(self, class_name: str) -> int:
        """Get the class index given the class name."""
        for idx, name in self.index2class.items():
            if name == class_name:
                return idx
        raise ValueError(f"Class '{class_name}' not found in index2class mapping")
    
    @property
    def height(self) -> int:
        return self.predictions.shape[0]

    @property
    def width(self) -> int:
        return self.predictions.shape[1]

    @property
    def nb_classes(self) -> int:
        return self.predictions.shape[2]
    


class Callback:
    """Base class of the Callback API."""

    def on_execute_start(self, *args: Any, **kwargs: Any) -> None:
        """Execute this method called before node execute method."""
        pass

    def on_execute_end(self, result: Any) -> None:
        """Execute this method called after node execute method.

        Args:
            result (Any): execute method output.
        """
        pass

class GeometryMask:
    def __init__(
        self,
        pupil_mask: np.ndarray,
        iris_mask: np.ndarray,
        eyeball_mask: np.ndarray
    ) -> None:
        for mask, name in zip([pupil_mask, iris_mask, eyeball_mask], ["pupil", "iris", "eyeball"]):
            if mask.ndim != 2:
                raise ValueError(f"{name}_mask must be a 2D array")
            if not np.array_equal(mask, mask.astype(bool)):
                raise ValueError(f"{name}_mask must be binary (0 or 1 values)")

        self.pupil_mask = pupil_mask
        self.iris_mask = iris_mask
        self.eyeball_mask = eyeball_mask

    @property
    def filled_eyeball_mask(self) -> np.ndarray:
        """Iris mask with pupil filled in."""
        binary_maps = np.zeros(self.eyeball_mask.shape[:2], dtype=np.uint8)

        binary_maps += self.pupil_mask
        binary_maps += self.iris_mask
        binary_maps += self.eyeball_mask

        return binary_maps.astype(bool)
    
    @property
    def filled_iris_mask(self) -> np.ndarray:
        """Fill iris mask.

        Returns:
            np.ndarray: Iris mask with filled pupil "holes".
        """
        binary_maps = np.zeros(self.iris_mask.shape[:2], dtype=np.uint8)

        binary_maps += self.pupil_mask
        binary_maps += self.iris_mask

        return binary_maps.astype(bool)

    def serialize(self) -> Dict[str, Any]:
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'GeometryMask':
        return GeometryMask(
            **data
        )
    

class NoiseMask:
    """Simple container for noise mask."""

    def __init__(self, mask: np.ndarray):
        if mask.ndim != 2:
            raise ValueError("Noise mask must be a 2D array")
        if not np.array_equal(mask, mask.astype(bool)):
            raise ValueError("Noise mask must be binary (0 or 1 values)")

        self.mask = mask.astype(bool)

    def serialize(self) -> Dict[str, Any]:
        return {"mask": self.mask.astype(np.uint8)}

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> 'NoiseMask':
        return NoiseMask(mask=np.array(data["mask"], dtype=bool))
    

def estimate_diameter(polygon: np.ndarray) -> float:
    return float(np.linalg.norm(polygon[:, None, :] - polygon[None, :, :], axis=-1).max())

class GeometryPolygons:
    def __init__(
        self,
        pupil_array: np.ndarray,
        iris_array: np.ndarray,
        eyeball_array: np.ndarray
    ) -> None:
        self.pupil_array = self._validate_array(pupil_array, "pupil_array")
        self.iris_array = self._validate_array(iris_array, "iris_array")
        self.eyeball_array = self._validate_array(eyeball_array, "eyeball_array")

        self._pupil_diameter = None
        self._iris_diameter = None

    def _validate_array(self, arr: np.ndarray, name: str) -> np.ndarray:
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy array")
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"{name} must be of shape (N, 2), got {arr.shape}")
        return arr.astype(np.float32)

    @property
    def pupil_diameter(self) -> float:
        if self._pupil_diameter is None:
            self._pupil_diameter = estimate_diameter(self.pupil_array)
        return self._pupil_diameter

    @property
    def iris_diameter(self) -> float:
        if self._iris_diameter is None:
            self._iris_diameter = estimate_diameter(self.iris_array)
        return self._iris_diameter

    def serialize(self) -> Dict[str, np.ndarray]:
        """Serialize GeometryPolygons object."""
        return {
            "pupil": self.pupil_array,
            "iris": self.iris_array,
            "eyeball": self.eyeball_array
        }

    @staticmethod
    def deserialize(data: Dict[str, np.ndarray]) -> "GeometryPolygons":
        """Deserialize GeometryPolygons object."""
        return GeometryPolygons(
            pupil_array=data["pupil"],
            iris_array=data["iris"],
            eyeball_array=data["eyeball"]
        )
    
class IRImage:
    def __init__(
        self,
        img_data: np.ndarray,
        eye_size: Literal["left", "right"]
    ) -> None:
        self.img_data = img_data
        self.eye_size = eye_size

    @property
    def height(self) -> int:
        """Return IR image's height.

        Return:
            int: image height.
        """
        return self.img_data.shape[0]

    @property
    def width(self) -> int:
        """Return IR image's width.

        Return:
            int: image width.
        """
        return self.img_data.shape[1]

    def serialize(self) -> Dict[str, Any]:
        """Serialize IRImage object.

        Returns:
            Dict[str, Any]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "IRImage":
        """Deserialize IRImage object.

        Args:
            data (Dict[str, Any]): Serialized object to dict.

        Returns:
            IRImage: Deserialized object.
        """
        return IRImage(**data)
    

class EyeOrientation:
    """Data holder for the eye orientation. The angle must be comprised between -pi/2 (included) and pi/2 (excluded)."""
    def __init__(self, angle: float):
        self.angle = angle

    def serialize(self) -> float:
        """Serialize EyeOrientation object.

        Returns:
            float: Serialized object.
        """
        return self.angle

    @staticmethod
    def deserialize(data: float) -> "EyeOrientation":
        """Deserialize EyeOrientation object.

        Args:
            data (float): Serialized object to float.

        Returns:
            EyeOrientation: Deserialized object.
        """
        return EyeOrientation(angle=data)
    

class NormalizedIris:
    """Data holder for the normalized iris images."""
    def __init__(self, normalized_img: np.ndarray, normalized_mask: np.ndarray):
        self._validate_array_2d(normalized_img, "normalized_image")
        self._validate_array_2d(normalized_mask, "normalized_mask")
        self._validate_uint8(normalized_img)
        self._validate_binary(normalized_mask)
        self._validate_same_shape(normalized_img, normalized_mask)

        self.normalized_image = normalized_img
        self.normalized_mask = normalized_mask

    def serialize(self) -> Dict[str, np.ndarray]:
        """Serialize NormalizedIris object.

        Returns:
            Dict[str, np.ndarray]: Serialized object.
        """
        return self.dict(by_alias=True)

    @staticmethod
    def deserialize(data: Dict[str, np.ndarray]) -> "NormalizedIris":
        """Deserialize NormalizedIris object.

        Args:
            data (Dict[str, np.ndarray]): Serialized object to dict.

        Returns:
            NormalizedIris: Deserialized object.
        """
        return NormalizedIris(**data)
    
    @staticmethod
    def _validate_array_2d(arr: np.ndarray, name: str):
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D numpy array.")

    @staticmethod
    def _validate_uint8(arr: np.ndarray):
        if arr.dtype != np.uint8:
            raise ValueError("normalized_image must be of type uint8.")

    @staticmethod
    def _validate_binary(arr: np.ndarray):
        unique_vals = np.unique(arr)
        if not np.array_equal(unique_vals, [0]) and not np.array_equal(unique_vals, [1]) and not np.array_equal(unique_vals, [0, 1]):
            raise ValueError("normalized_mask must be a binary array containing only 0 and 1.")

    @staticmethod
    def _validate_same_shape(arr1: np.ndarray, arr2: np.ndarray):
        if arr1.shape != arr2.shape:
            raise ValueError("normalized_image and normalized_mask must have the same shape.")