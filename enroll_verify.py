import os
import re
from pipeline import IrisRecognitionPipeline
from database import IrisDatabase
from pipeline import IrisRecognitionPipeline
from segmentation import WorldCoinSegmentator
from normalization import WorldCoinsNormalizer
from feature_extraction import WorldCoinsFeatureExtractor
from feature_extraction import CustomFeatureExtractor
from matching import WorldCoinsMatcher


def parse_filename(filename: str) -> tuple[str, str]:
    """Parse filename to extract person's name and eye type.
    Returns: (name, eye_type) where eye_type is 'left' or 'right'."""
    # Extract name and eye part (e.g., 'aeval1' -> name='aeva', eye='l1')
    base = os.path.splitext(filename)[0]  # Remove .bmp
    match = re.match(r"([a-zA-Z]+)([lr]\d)", base)
    if not match:
        raise ValueError(f"Invalid filename format: {filename}")
    name, eye = match.groups()
    eye_type = 'left' if eye.startswith('l') else 'right'
    return name, eye_type

def enroll_images_from_folder(pipeline, data_folder: str = "Data", storage_path: str = "Data/iris_db"):
    """Enroll all iris images from the specified folder into the database."""
    db = IrisDatabase(pipeline, storage_path=storage_path)
    
    # Check if database already has enrolled subjects
    if not db.is_database_empty():
        print(f"Database already contains {db.get_database_size()} iris entries. Skipping enrollment.")
        return db
    
    # Scan Data/ folder
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"Data folder {data_folder} does not exist.")
    
    for filename in os.listdir(data_folder):
        if filename.endswith('.bmp'):
            try:
                name, eye_type = parse_filename(filename)
                image_path = os.path.join(data_folder, filename)
                print(f"Enrolling {name}'s {eye_type} eye from {filename}...")
                subject_id = db.enroll(image_path, name, eye_type)
                print(f"Enrolled with subject ID: {subject_id}")
            except ValueError as e:
                print(f"Skipping {filename}: {e}")
    
    print(f"Enrollment complete. Database contains {db.get_database_size()} iris entries.")
    return db

def verify_iris(pipeline, test_image_path: str, storage_path: str = "Data/iris_db"):
    """Verify a test iris image against the database."""
    db = IrisDatabase(pipeline, storage_path=storage_path)
    
    if db.is_database_empty():
        print("Database is empty. Please enroll subjects first.")
        return
    
    # Identify the test image
    subject_id, name, score, eye = db.identify(test_image_path)
   
    if subject_id:
        print(f"Match found: {name} ({eye} eye, Score: {score:.4f})")
    else:
        print(f"No match found (Best score: {score:.4f})")

def main():
    # Initialize your pipeline (replace with your actual initialization)
    pipeline = IrisRecognitionPipeline(
        segmentator=WorldCoinSegmentator(),
        normalizer=WorldCoinsNormalizer(),
        feature_extractor=WorldCoinsFeatureExtractor(),
        matcher=WorldCoinsMatcher()
    )
    
    # Enroll images from Data/ folder
    data_folder = "Data"
    storage_path = "Data/iris_db"
    db = enroll_images_from_folder(pipeline, data_folder, storage_path)
    
    # Verify a test image
    test_image = "Data/aeval1.bmp"
    if os.path.exists(test_image):
        print(f"\nVerifying test image: {test_image}")
        verify_iris(pipeline, test_image, storage_path)
    else:
        print(f"Test image {test_image} not found.")

if __name__ == "__main__":
    main()