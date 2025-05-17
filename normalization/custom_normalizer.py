import numpy as np
from typing import Any, List, Tuple, Dict, Callable
from abc import ABC, abstractmethod
import cv2
import iris
from .utils.dataclasses import SegmentationMap, Callback, GeometryMask, NoiseMask, GeometryPolygons, IRImage, EyeOrientation, NormalizedIris


class MultilabelSegmentationBinarization:
    """Implementation of a binarization algorithm for multilabel segmentation. Algorithm performs thresholding of each prediction's channel separately to create rasters based on specified by the user classes' thresholds."""
    def __init__(
        self,
        eyeball_threshold: float = 0.5,
        iris_threshold: float = 0.5,
        pupil_threshold: float = 0.5,
        eyelashes_threshold: float = 0.5,
        callbacks: List[Callback] = [],
    ) -> None:
        self.eyeball_threshold = eyeball_threshold
        self.iris_threshold = iris_threshold
        self.pupil_threshold = pupil_threshold
        self.eyelashes_threshold = eyelashes_threshold
        self.callbacks = callbacks

    def run(self, segmentation_map: SegmentationMap) -> Tuple[GeometryMask, NoiseMask]:
        """Perform segmentation binarization.

        Args:
            segmentation_map (SegmentationMap): Predictions.

        Returns:
            Tuple[GeometryMask, NoiseMask]: Binarized geometry mask and noise mask.
        """
        eyeball_preds = segmentation_map.predictions[..., segmentation_map.index_of("eyeball")]
        iris_preds = segmentation_map.predictions[..., segmentation_map.index_of("iris")]
        pupil_preds = segmentation_map.predictions[..., segmentation_map.index_of("pupil")]
        eyelashes_preds = segmentation_map.predictions[..., segmentation_map.index_of("eyelashes")]

        # Apply thresholds
        eyeball_mask = eyeball_preds >= self.eyeball_threshold
        iris_mask = iris_preds >= self.iris_threshold
        pupil_mask = pupil_preds >= self.pupil_threshold
        eyelashes_mask = eyelashes_preds >= self.eyelashes_threshold

        return GeometryMask(pupil_mask=pupil_mask, iris_mask=iris_mask, eyeball_mask=eyeball_mask), NoiseMask(
            mask=eyelashes_mask
        )



def filter_polygon_areas(
    polygons: List[np.ndarray],
    rel_tr: float = 0.03,
    abs_tr: float = 0.0
) -> List[np.ndarray]:
    """Filter out polygons whose area is below either an absolute threshold or a fraction of the largest area."""
    if not polygons:
        return []
    
    areas = []
    for p in polygons:
        if len(p) >= 3:
            area = float(cv2.contourArea(p))
        else:
            area = 0.0
        areas.append(area)
    max_area = max(areas)

    if max_area == 0.0:
        return []

    area_factors = np.array(areas) / max_area

    filtered_polygons = [
        polygon for a, af, polygon in zip(areas, area_factors, polygons) if a > abs_tr and af > rel_tr
    ]

    return filtered_polygons


class ContouringAlgorithm:
    """
    Implementation of a vectorization process through contouring raster image.
    """
    def __init__(
        self,
        contour_filters: List[Callable[[List[np.ndarray]], List[np.ndarray]]] = [filter_polygon_areas]
    )-> None:
        self.contour_filters = contour_filters

    def run(self, geometry_mask: GeometryMask) -> GeometryPolygons:
        if not np.any(geometry_mask.iris_mask):
            raise ValueError("Geometry raster verification failed.")

        geometry_contours = self._find_contours(geometry_mask)

        return geometry_contours

    def _find_contours(self, mask) -> GeometryPolygons:
        """
        Extract contours for each part of the eye
        """

        eyeball_array = self._find_class_contours(mask.filled_eyeball_mask.astype(np.uint8))
        iris_array = self._find_class_contours(mask.filled_iris_mask.astype(np.uint8))
        pupil_array = self._find_class_contours(mask.pupil_mask.astype(np.uint8))

        return GeometryPolygons(
            pupil_array=pupil_array,
            iris_array=iris_array,
            eyeball_array=eyeball_array
        )

    def _find_class_contours(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Find the outer contour of a single-class binary mask.
        """
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None or len(contours) == 0:
            raise ValueError("No contours found in binary mask.")

        hierarchy = hierarchy[0]
        parent_indices = [i for i, h in enumerate(hierarchy) if h[3] == -1]
        filtered = [np.squeeze(contours[i]) for i in parent_indices if contours[i].ndim >= 2]

        filtered = self._filter_contours(filtered)

        if len(filtered) != 1:
            raise ValueError(f"Expected 1 main contour, but got {len(filtered)}.")

        return filtered[0]

    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply all contour filters sequentially.
        """
        for f in self.contour_filters:
            contours = f(contours)
        return contours

    @staticmethod
    def filter_polygon_areas(contours: List[np.ndarray], min_area: float = 100.0) -> List[np.ndarray]:
        """
        Example filter: removes small contours by area.
        """
        return [c for c in contours if cv2.contourArea(c.astype(np.float32)) >= min_area]


class SpecularReflectionDetection:
    """Apply a threshold to the IR Image to detect specular reflections."""
    def __init__(self, reflection_threshold: int = 254):
        self.reflection_threshold = reflection_threshold

    def run(self, ir_image: IRImage):
        _, reflection_segmap = cv2.threshold(
            ir_image.img_data, self.reflection_threshold, 255, cv2.THRESH_BINARY
        )
        reflection_segmap = (reflection_segmap / 255.0).astype(bool)

        return NoiseMask(mask=reflection_segmap)


class ContourInterpolation:
    """
    Implementation of contour interpolation algorithm conditioned by given NoiseMask.
    """
    def __init__(
        self,
        max_distance_between_boundary_points: float = 0.01
    ) -> None:
        if not (0.0 < max_distance_between_boundary_points < 1.0):
            raise ValueError("max_distance_between_boundary_points must be in (0.0, 1.0)")
        self.max_distance_between_boundary_points = max_distance_between_boundary_points

    def run(self, polygons: GeometryPolygons) -> GeometryPolygons:
        max_dist_px = self.max_distance_between_boundary_points * polygons.iris_diameter

        refined_pupil_array = self._interpolate_polygon_points(polygons.pupil_array, max_dist_px)
        refined_iris_array = self._interpolate_polygon_points(polygons.iris_array, max_dist_px)
        refined_eyeball_array = self._interpolate_polygon_points(polygons.eyeball_array, max_dist_px)

        return GeometryPolygons(
            pupil_array=refined_pupil_array,
            iris_array=refined_iris_array,
            eyeball_array=refined_eyeball_array,
        )
    
    def _interpolate_polygon_points(self, polygon: np.ndarray, max_dist_px: float) -> np.ndarray:
        if polygon.ndim != 2 or polygon.shape[1] != 2:
            raise ValueError("Polygon must be of shape (N, 2)")
        
        # Wrap around for closed contour
        previous_points = np.roll(polygon, shift=1, axis=0)
        segment_lengths = np.linalg.norm(polygon - previous_points, axis=1)
        num_segments = np.ceil(segment_lengths / max_dist_px).astype(int)

        interpolated_x = []
        interpolated_y = []

        for (x0, y0), (x1, y1), n_pts in zip(previous_points, polygon, num_segments):
            interpolated_x.append(np.linspace(x0, x1, n_pts, endpoint=False))
            interpolated_y.append(np.linspace(y0, y1, n_pts, endpoint=False))
        
        all_points = np.stack([np.concatenate(interpolated_x), np.concatenate(interpolated_y)], axis=1)

        # Remove duplicate points
        _, unique_indices = np.unique(all_points, axis=0, return_index=True)

        return all_points[np.sort(unique_indices)]
    

class ContourPointNoiseEyeballDistanceFilter:
    """Implementation of point filtering algorithm that removes points which are to close to eyeball or noise.

    The role of this algorithm is to create a buffer around the pupil and iris polygons. This accounts for
    potential segmentation imprecisions, making the overall pipeline more robust against edge cases and out-of-distribution images.

    The buffer width is computed relatively to the iris diameter: `min_distance_to_noise_and_eyeball * iris_diameter`
    The trigger for this buffer are the eyeball boundary and the noise (e.g. eyelashes, specular reflection, etc.).
    """
    def __init__(self, min_dist_to_noise_and_eyeball: float = 0.005) -> None:
        self.min_dist_to_noise_and_eyeball = min_dist_to_noise_and_eyeball

    def run(self, polygons: GeometryPolygons, geometry_mask: NoiseMask) -> GeometryPolygons:
        """Perform polygon refinement by filtering out those iris/pupil polygons points which are to close to eyeball or noise."""
        noise_and_eyeball_polygon_points_mask = geometry_mask.mask.copy()

        for eyeball_point in np.round(polygons.eyeball_array).astype(int):
            x, y = eyeball_point
            noise_and_eyeball_polygon_points_mask[y, x] = True

        min_dist_to_noise_and_eyeball_in_px = round(
            self.min_dist_to_noise_and_eyeball * polygons.iris_diameter
        )

        forbidden_touch_map = cv2.blur(
            noise_and_eyeball_polygon_points_mask.astype(float),
            ksize=(
                2 * min_dist_to_noise_and_eyeball_in_px + 1,
                2 * min_dist_to_noise_and_eyeball_in_px + 1,
            ),
        )
        forbidden_touch_map = forbidden_touch_map.astype(bool)

        return GeometryPolygons(
            pupil_array=self._filter_polygon_points(forbidden_touch_map, polygons.pupil_array),
            iris_array=self._filter_polygon_points(forbidden_touch_map, polygons.iris_array),
            eyeball_array=polygons.eyeball_array,
        )
    
    def _filter_polygon_points(self, forbidden_touch_map: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
        """Filter polygon's points.

        Returns:
            np.ndarray: Filtered polygon's points.
        """
        valid_points = [not forbidden_touch_map[y, x] for x, y in np.round(polygon_points).astype(int)]
        if not any(valid_points):
            raise ValueError("No valid points after filtering polygon points!")

        return polygon_points[valid_points]
    
class NoiseMaskUnion:
    """Aggregate several NoiseMask into one by computing their union. I.E. For every bit of the NoiseMask, the output is an OR of the same bit across all NoiseMasks."""
    def run(self, elements: List[NoiseMask]) -> NoiseMask:
        """Compute the union of a list of NoiseMask."""
        if not all([mask.mask.shape == elements[0].mask.shape for mask in elements]):
            raise ValueError(f"Every NoiseMask.mask must have the same shape to be aggregated. Received {[mask.mask.shape for mask in elements]}")
        
        noise_union = np.sum([mask.mask for mask in elements], axis=0) > 0 
        return NoiseMask(mask=noise_union)
    
class LinearNormalization:
    """
    Implementation of a normalization algorithm which uses linear transformation to map image pixels.
    """
    def __init__(self, res_in_r: int = 128, oversat_threshold: int = 254):
        self.res_in_r = res_in_r
        self.oversat_threshold = oversat_threshold

    def run(
        self,
        image: IRImage,
        noise_mask: NoiseMask,
        extrapolated_contours: GeometryPolygons,
        eye_orientation: EyeOrientation
    ) -> NormalizedIris:
        """
        Normalize iris using linear transformation when sampling points from cartisian to polar coordinates.
        
        Args:
            image (IRImage): Input image to normalize.
            noise_mask (NoiseMask): Noise mask.
            extrapolated_contours (GeometryPolygons): Extrapolated contours.
            eye_orientation (EyeOrientation): Eye orientation angle.
            
        Returns:
            NormalizedIris: NormalizedIris object containing normalized image and iris mask.
        """
        # Lấy các đường viền
        pupil = extrapolated_contours.pupil_array
        iris = extrapolated_contours.iris_array

        # Xoay điểm theo góc mắt
        shift = -round(np.degrees(eye_orientation.angle) * len(pupil) / 360.0)
        pupil = np.roll(pupil, shift, axis=0)
        iris = np.roll(iris, shift, axis=0)

        # Tạo mặt nạ vùng mống mắt
        h, w = noise_mask.mask.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [extrapolated_contours.iris_array.astype(np.int32)], 255)

        # Loại bỏ vùng nhiễu và điểm quá sáng
        iris_mask = (mask > 0) & (~noise_mask.mask)
        iris_mask[image.img_data >= self.oversat_threshold] = False

        # Tạo các điểm ánh xạ tuyến tính từ đồng tử đến mống mắt
        res_r = self.res_in_r
        t_values = np.linspace(0, 1, res_r)
        all_points = []
        for t in t_values:
            row = pupil + t * (iris - pupil)
            all_points.append(np.round(row).astype(int))
        all_points = np.array(all_points)

        # Làm phẳng mảng và kiểm tra điểm ngoài ảnh
        flat_points = np.vstack(all_points)
        flat_points[flat_points[:, 0] >= image.img_data.shape[1], 0] = -1
        flat_points[flat_points[:, 1] >= image.img_data.shape[0], 1] = -1

        # Lấy giá trị ảnh và mask tại các điểm ánh xạ
        norm_img, norm_mask = [], []
        for x, y in flat_points:
            if x >= 0 and y >= 0:
                norm_img.append(image.img_data[y, x] / 255.0)
                norm_mask.append(iris_mask[y, x])
            else:
                norm_img.append(0)
                norm_mask.append(False)

        h, w = all_points.shape[:2]
        norm_img = np.clip(np.round(np.array(norm_img).reshape(h, w) * 255), 0, 255).astype(np.uint8)
        norm_mask = np.array(norm_mask).reshape(h, w)

        return NormalizedIris(normalized_img=norm_img, normalized_mask=norm_mask)
    

class CustomNormalizer(ABC):
    def __init__(self):
        super().__init__()
        # Manual implementation
        self.segmentation_binarization = MultilabelSegmentationBinarization()
        self.vectorization = ContouringAlgorithm()
        self.specular_reflection_detection = SpecularReflectionDetection()
        self.interpolation = ContourInterpolation()
        self.distance_filter = ContourPointNoiseEyeballDistanceFilter()

        # Using SOTA from iris library to implement the eye orientation by using Moment and smoothen the contour by extrapolate into circles or ellipses
        self.eye_orientation = iris.MomentOfArea()
        self.eye_center_estimation = iris.BisectorsMethod()
        self.smoothing = iris.Smoothing()
        self.geometry_estimation = iris.FusionExtrapolation(
            circle_extrapolation=iris.LinearExtrapolation(dphi=0.703125),
            ellipse_fit=iris.LSQEllipseFitWithRefinement(dphi=0.703125),
            algorithm_switch_std_threshold=3.5
        )

        # Manual implementation
        self.noise_masks_aggregation = NoiseMaskUnion()
        self.normalization = LinearNormalization()
        
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
