import os
import pickle
import numpy as np
from typing import Optional, Dict, Tuple, List
from pipeline import IrisRecognitionPipeline  # Assuming the previous code is in this module
import uuid

class IrisDatabase:
    def __init__(self, pipeline: IrisRecognitionPipeline, storage_path: str = "Data/iris_db"):
        """Initialize the iris database with a recognition pipeline and persistent storage location."""
        self.pipeline = pipeline
        self.storage_path = storage_path
        self.database: Dict[str, Dict] = {}  # {subject_id: {'features': array, 'name': str, 'image_path': str, 'eye': str}}
        self._ensure_storage_directory()
        self._load_database()

    def _ensure_storage_directory(self):
        """Create storage directory if it doesn't exist."""
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def _get_db_file_path(self) -> str:
        """Get the path to the database file."""
        return os.path.join(self.storage_path, "iris_database.pkl")

    def _load_database(self):
        """Load existing database from file if it exists."""
        db_file = self._get_db_file_path()
        if os.path.exists(db_file):
            with open(db_file, 'rb') as f:
                self.database = pickle.load(f)

    def _save_database(self):
        """Save the database to file."""
        with open(self._get_db_file_path(), 'wb') as f:
            pickle.dump(self.database, f)

    def enroll(self, image_path: str, name: str, eye: str) -> str:
        """Enroll a new iris image into the database with eye information."""
        # Check if image is already enrolled to avoid duplicates
        for subject_id, data in self.database.items():
            if data['image_path'] == image_path:
                return subject_id

        # Extract features using the pipeline
        features = self.pipeline.run(image_path)
        
        # Generate unique subject ID
        subject_id = str(uuid.uuid4())
        
        # Store in database with eye information
        self.database[subject_id] = {
            'features': features,
            'name': name,
            'image_path': image_path,
            'eye': eye  # 'left' or 'right'
        }
        
        # Save updated database
        self._save_database()
        
        return subject_id

    def identify(self, image_path: str, threshold: Optional[float] = None) -> Tuple[Optional[str], Optional[str], float, Optional[str]]:
        """Identify an iris image against the database, checking both eyes.
        Returns: (subject_id, name, score, eye) or (None, None, 0.0, None) if no match found."""
        if not self.database:
            return None, None, 0.0, None

        # Extract features from input image
        input_features = self.pipeline.run(image_path)
        
        # Use pipeline's threshold if none provided
        threshold = threshold if threshold is not None else self.pipeline.get_threshold()
        
        best_score = float('inf')  # Lower score is better match
        best_subject_id = None
        best_name = None
        best_eye = None
        
        # Compare against all database entries
        for subject_id, data in self.database.items():
            score = self.pipeline.matcher.match(input_features, data['features'])
            if score < best_score:
                best_score = score
                best_subject_id = subject_id
                best_name = data['name']
                best_eye = data['eye']
        
        # Check if best match is below threshold
        if best_score < threshold:
            return best_subject_id, best_name, best_score, best_eye
        return None, None, best_score, None

    def get_subject_info(self, subject_id: str) -> Optional[Dict]:
        """Get information about a specific subject."""
        return self.database.get(subject_id)

    def remove_subject(self, subject_id: str) -> bool:
        """Remove a subject from the database and update persistent storage."""
        if subject_id in self.database:
            del self.database[subject_id]
            self._save_database()
            return True
        return False

    def get_database_size(self) -> int:
        """Return the number of enrolled subjects."""
        return len(self.database)

    def change_storage_path(self, new_path: str):
        """Change the storage location and move the database."""
        old_db_file = self._get_db_file_path()
        self.storage_path = new_path
        self._ensure_storage_directory()
        
        # Move database file if it exists
        if os.path.exists(old_db_file):
            new_db_file = self._get_db_file_path()
            os.rename(old_db_file, new_db_file)
            # Remove old directory if empty
            old_dir = os.path.dirname(old_db_file)
            if os.path.exists(old_dir) and not os.listdir(old_dir):
                os.rmdir(old_dir)
        self._save_database()

    def is_database_empty(self) -> bool:
        """Check if the database is empty."""
        return len(self.database) == 0

    def get_subjects_by_name(self, name: str) -> List[Dict]:
        """Get all enrolled irises for a given person."""
        return [data for data in self.database.values() if data['name'] == name]