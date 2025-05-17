import os
import cv2
from pipeline import IrisRecognitionPipeline
from segmentation import WorldCoinSegmentator
from normalization.custom_normalizer import CustomNormalizer
from feature_extraction.custom_extractor import CustomFeatureExtractor
from matching.custom_matcher import CustomMatcher

# Ensure some requirements are met
assert os.path.exists("Data"), "Data folder does not exist. Please create a Data folder with the required images."

def main():
    # Initialize pipeline components
    segmentator = WorldCoinSegmentator()
    normalizer = CustomNormalizer()
    feature_extractor = CustomFeatureExtractor()
    matcher = CustomMatcher()

    # Create pipeline
    pipeline1 = IrisRecognitionPipeline(
        segmentator=segmentator,
        normalizer=normalizer,
        feature_extractor=feature_extractor,
        matcher=matcher
    )

    # Test images
    image1 = "Data/zulaikahr1.bmp"
    image2 = "Data/zulaikahr2.bmp"

    # Process first image
    pipeline1.set_image(image1)
    print("Pipeline 1 - Image 1 processed.")

    # Compare images
    print(f"Similarity score: {pipeline1.getScore(image1, image2)}")

if __name__ == "__main__":
    main() 