import os
import cv2
from pipeline import IrisRecognitionPipeline
from segmentation import WorldCoinSegmentator
from normalization import WorldCoinsNormalizer
from feature_extraction import WorldCoinsFeatureExtractor
from matching import WorldCoinsMatcher

# Ensure some requirements are met
assert os.path.exists("Data"), "Data folder does not exist. Please create a Data folder with the required images."

def main():
    # Create sample images (placeholder)
    image1 = "Data/zulaikahr1.bmp"
    image2 = "Data/zulaikahr2.bmp"

    # Create pipeline with one combination of components
    pipeline1 = IrisRecognitionPipeline(
        segmentator=WorldCoinSegmentator(),
        normalizer=WorldCoinsNormalizer(),
        feature_extractor=WorldCoinsFeatureExtractor(),
        matcher=WorldCoinsMatcher()
    )

    pipeline1.set_image(image1)
    print("Pipeline 1 - Image 1 processed.")
    print(f"Similarity score: {pipeline1.getScore(image1, image2)}")

if __name__ == "__main__":
    main()