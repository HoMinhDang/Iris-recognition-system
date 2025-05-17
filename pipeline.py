import numpy as np
from segmentation.interface import IrisSegmentator
from normalization.interface import IrisNormalizer
from feature_extraction.interface import FeatureExtractor
from matching.interface import IrisMatcher
import cv2, os
from typing import Tuple
import matplotlib.pyplot as plt
import uuid

class IrisRecognitionPipeline:
    def __init__(self, segmentator: IrisSegmentator, normalizer: IrisNormalizer,
                 feature_extractor: FeatureExtractor, matcher: IrisMatcher):
        self.segmentator = segmentator
        self.normalizer = normalizer
        self.feature_extractor = feature_extractor
        self.matcher = matcher

        self.current_image_path = None
        self.image = None
        self.segmentation_output = None
        self.segmentation_masks = None
        self.normalization_output = None
        self.feature_extraction_output = None
        self.threshold = 0.33

    def set_image(self, image_path: str):
        """Set and cache the image for the pipeline."""
        if self.current_image_path != image_path:
            self.current_image_path = image_path
            self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.image = cv2.resize(self.image, (640, 480))
            # Reset cached results when new image is set
            self.segmentation_output = None
            self.segmentation_masks = None
            self.normalization_output = None
            self.feature_extraction_output = None
        self.run(image_path)

    def run(self, image_path: str = None):
        """Process an image to extract and return iris features for enrollment."""
        if image_path is None:
            if self.current_image_path is None:
                raise ValueError("Image path must be provided for segmentation.")
        else:
            if image_path != self.current_image_path or self.current_image_path is None:
                self.set_image(image_path)

        # Only run processing if we don't have cached results
        if self.feature_extraction_output is None:
            self.segmentation_output = self.segmentator.segment(self.image)
            self.segmentation_masks = self.segmentator.get_masks(self.image)
            self.normalization_output = self.normalizer.normalize(self.image, self.segmentation_output)
            self.feature_extraction_output = self.feature_extractor.extract(self.normalization_output)
        
        return self.feature_extraction_output

    def getScore(self, image_path1: str, image_path2: str) -> float:
        """Compare two iris images and return similarity score."""
        features1 = self.run(image_path1)
        features2 = self.run(image_path2)
        score = self.matcher.match(features1, features2)
        return score

    def get_segmentation_images(self, image_path: str = None) -> list[Tuple[str, str]]:
        """Get segmentation output for the given image. Returns list of tuples with mask name and output image path."""
        if image_path is None:
            if self.current_image_path is None:
                raise ValueError("Image path must be provided for segmentation.")
        else:
            self.run(image_path)

        output = []
        for (mask_name, mask) in self.segmentation_masks:
            mask = np.array(mask)
            mask_norm = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            image_bgr = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
            color_mask = np.zeros_like(image_bgr)
            color_mask[:, :, 2] = mask_norm  # Red channel

            overlay = cv2.addWeighted(image_bgr, 1.0, color_mask, 0.5, 0)
            
            # Save overlay image
            output_path = f"tmp/segmentation_{mask_name}_{uuid.uuid4().hex}.png"
            output_path = self.save_image(overlay, output_path)
            output.append((mask_name, output_path))

        return output

    def save_image(self, image: np.ndarray, filename: str) -> str:
        """Save the image to the specified filename."""
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(filename, image)
        return filename
    
    def get_normalization_image(self, image_path: str = None) -> str:
        """Get normalization output for the given image and return the saved image path."""
        if image_path is None:
            if self.current_image_path is None:
                raise ValueError("Image path must be provided for segmentation.")
        else:
            self.run(image_path)
        
        normalization_image = self.normalization_output.normalized_image
        output_path = f"tmp/normalization_output_{uuid.uuid4().hex}.png"
        return self.save_image(normalization_image, output_path)
    
    def get_iris_code_image(self, image_path: str = None) -> str:
        """Get feature extraction output for the given image and return the saved image path."""
        if image_path is None:
            if self.current_image_path is None:
                raise ValueError("Image path must be provided for segmentation.")
        else:
            self.run(image_path)
            
        save_path = f"tmp/iris_code_output_{uuid.uuid4().hex}.png"
        codes = self.feature_extraction_output.iris_codes
        n_wavelets = len(codes)

        # Reduce the height multiplier in figsize to make the figure more compact
        fig, axes = plt.subplots(n_wavelets * 2, 1, figsize=(6, 1 * n_wavelets * 2))
        axes = axes.ravel()

        for i in range(n_wavelets):
            code = codes[i]
            for j in range(2):
                idx = i * 2 + j
                axes[idx].imshow(code[:, :, j], cmap='gray')
                axes[idx].set_title(f'Wavelet {i} — code bit {j}', fontsize=6, pad=2)  # Smaller font and minimal padding
                axes[idx].axis('off')

        # Minimize vertical spacing and margins
        plt.subplots_adjust(hspace=0.02, top=0.98, bottom=0.02)  # Tighter vertical spacing and margins
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return save_path

    def verify(self, image_path1: str, image_path2: str, threshold: float = 0.37) -> bool:
        """Verify if two iris images belong to the same individual."""
        score = self.getScore(image_path1, image_path2)
        return score < threshold
    
    def get_threshold(self) -> float:
        """Get the current threshold value."""
        return self.threshold
    
    def set_threshold(self, threshold: float):
        """Set a new threshold value."""
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.threshold = threshold