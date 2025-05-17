import numpy as np
from .interface import IrisSegmentator
from typing import Tuple
from iris.nodes.segmentation.onnx_multilabel_segmentation import ONNXMultilabelSegmentation
import iris

class WorldCoinSegmentator(IrisSegmentator):
    def __init__(self):
        super().__init__()
        self.segmentation_model = ONNXMultilabelSegmentation.create_from_hugging_face(
            model_name="iris_semseg_upp_scse_mobilenetv2.onnx",
            input_resolution=(640, 480),  # Default model resolution
            input_num_channels=3  # Use 1 for grayscale, 3 for RGB
        )
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray] | any:
        """Segment iris and pupil boundaries from input image."""
        # assert the size of the image is 640x480
        if image.shape[0] != 480 or image.shape[1] != 640:
            raise ValueError("Image size must be 640x480")
        
        # Load the image and convert to IRImage format
        ir_image = iris.IRImage(img_data=image, eye_side="left")
        # Run segmentation
        segmentation_output = self.segmentation_model.run(ir_image)
        return segmentation_output
    
    def get_masks(self, image: np.ndarray) -> list[Tuple[str, np.ndarray]]:
        """Get segmentation masks for the given image."""
        segmentation_output = self.segment(image)
        # print(f"Segmentation output shape: {segmentation_output.predictions.shape}")
        masks = [
            ("eye_mask", segmentation_output.predictions[:, :, 0]),
            ("iris_mask", segmentation_output.predictions[:, :, 1]),
            ("pupil_mask", segmentation_output.predictions[:, :, 2]),
            ("eyelashes_mask", segmentation_output.predictions[:, :, 3]),
        ]
        return masks