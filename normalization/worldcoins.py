from abc import ABC, abstractmethod
import numpy as np
import iris
from typing import Any

class WorldCoinsNormalizer(ABC):
    def __init__(self):
        super().__init__()
        self.segmentation_binarization = iris.MultilabelSegmentationBinarization()
        self.vectorization = iris.ContouringAlgorithm()
        self.specular_reflection_detection = iris.SpecularReflectionDetection()
        self.interpolation = iris.ContourInterpolation()
        self.distance_filter = iris.ContourPointNoiseEyeballDistanceFilter()
        self.eye_orientation = iris.MomentOfArea()
        self.eye_center_estimation = iris.BisectorsMethod()
        self.smoothing = iris.Smoothing()
        self.geometry_estimation = iris.FusionExtrapolation(
            circle_extrapolation=iris.LinearExtrapolation(dphi=0.703125),
            ellipse_fit=iris.LSQEllipseFitWithRefinement(dphi=0.703125),
            algorithm_switch_std_threshold=3.5
        )
        self.noise_masks_aggregation = iris.NoiseMaskUnion()
        self.normalization = iris.LinearNormalization()
        
    def normalize(self, image: np.ndarray, segmentation_output: Any) -> np.ndarray | Any:
        """Normalize iris to a fixed-size rectangular representation."""
        # Binarize segmentation map
        ir_image = iris.IRImage(img_data=image, eye_side="left")
        
        segmentation_binarization_output = self.segmentation_binarization.run(
            segmentation_map=segmentation_output
        )
        
        # Vectorize the iris mask
        vectorization_output = self.vectorization.run(
            geometry_mask=segmentation_binarization_output[0],
        )
        
        # Detect specular reflections
        specular_reflection_detection_output = self.specular_reflection_detection.run(
            ir_image=ir_image
        )
        
        # Interpolate the contours
        interpolation_output = self.interpolation.run(
            polygons=vectorization_output
        )
        
        # Filter noise based on eyeball distance
        distance_filter_output = self.distance_filter.run(
            polygons=interpolation_output,
            geometry_mask=segmentation_binarization_output[1]
        )
        
        # Estimate eye orientation
        eye_orientation_output = self.eye_orientation.run(
            geometries=distance_filter_output
        )
        
        # Estimate eye center using bisectors method
        eye_center_estimation_output = self.eye_center_estimation.run(
            geometries=distance_filter_output
        )
        
        # Smooth the contours
        smoothing_output = self.smoothing.run(
            polygons=distance_filter_output,
            eye_centers=eye_center_estimation_output
        )
        
        # Estimate geometry using fusion extrapolation
        geometry_estimation_output = self.geometry_estimation.run(
            input_polygons=smoothing_output,
            eye_center=eye_center_estimation_output    
        )
        
        # Aggregate noise masks
        noise_masks_aggregation_output = self.noise_masks_aggregation.run(
            elements=[
                segmentation_binarization_output[1],
                specular_reflection_detection_output
            ]
        )
        
        # Normalize the image using the aggregated noise mask and estimated geometry
        normalization_output = self.normalization.run(
            image=ir_image,
            noise_mask=noise_masks_aggregation_output,
            extrapolated_contours=geometry_estimation_output,
            eye_orientation=eye_orientation_output
        )

        return normalization_output
