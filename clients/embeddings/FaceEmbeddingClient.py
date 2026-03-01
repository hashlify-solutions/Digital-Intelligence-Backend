import cv2
import os
import numpy as np
from typing import Optional, List, Tuple
import logging
import warnings

# Configure for macOS compatibility before importing face_recognition
# import platform
# if platform.system() == 'Darwin':  # macOS
#     # face_recognition uses dlib which can have multiprocessing issues on macOS
#     os.environ.setdefault('JOBLIB_MULTIPROCESSING', '0')
#     os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Suppress pkg_resources deprecation warning from face_recognition_models
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

import face_recognition

logger = logging.getLogger(__name__)


class FaceEmbeddingClient:
    """
    Face embedding client using face_recognition library with local models.
    Uses dlib's state-of-the-art face recognition built with deep learning.
    """

    def __init__(self, model="large", num_jitters=1, tolerance=0.6):
        """
        Initialize the face embedding client.
        
        Args:
            model (str): Model to use - 'small' (faster) or 'large' (more accurate)
            num_jitters (int): Number of times to re-sample the face when calculating encoding
            tolerance (float): How much distance between faces to consider it a match
        """
        self.model = model
        self.num_jitters = num_jitters
        self.tolerance = tolerance
        self.embedding_size = 128  # face_recognition produces 128-dimensional embeddings

    def extract_face_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[np.ndarray]: 128-dimensional face embedding or None if no face found
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None

            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(
                image, 
                model="hog"  # Use HOG model for faster processing, can use "cnn" for better accuracy
            )
            
            if not face_locations:
                logger.warning(f"No faces found in image: {image_path}")
                return None
                
            if len(face_locations) > 1:
                logger.warning(f"Multiple faces found in image: {image_path}, using the first one")
            
            # Extract face encodings (embeddings)
            face_encodings = face_recognition.face_encodings(
                image, 
                face_locations, 
                num_jitters=self.num_jitters,
                model=self.model
            )
            
            if not face_encodings:
                logger.error(f"Failed to extract face encoding from: {image_path}")
                return None
                
            # Return the first face encoding
            embedding = face_encodings[0]
            logger.debug(f"Extracted face embedding of shape {embedding.shape} from {image_path}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting face embedding from {image_path}: {str(e)}")
            return None

    def extract_face_embedding_from_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a numpy image array.
        
        Args:
            image_array (np.ndarray): Image as numpy array (RGB format)
            
        Returns:
            Optional[np.ndarray]: 128-dimensional face embedding or None if no face found
        """
        try:
            # Find face locations
            face_locations = face_recognition.face_locations(image_array, model="hog")
            
            if not face_locations:
                logger.warning("No faces found in image array")
                return None
                
            if len(face_locations) > 1:
                logger.warning("Multiple faces found in image array, using the first one")
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(
                image_array, 
                face_locations, 
                num_jitters=self.num_jitters,
                model=self.model
            )
            
            if not face_encodings:
                logger.error("Failed to extract face encoding from image array")
                return None
                
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Error extracting face embedding from image array: {str(e)}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
            
        Returns:
            float: Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            # Convert to 0-1 range (cosine similarity ranges from -1 to 1)
            similarity = (similarity + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing faces similarity: {str(e)}")
            return 0.0

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two face embeddings.
        Lower distance means more similar faces.
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
            
        Returns:
            float: Distance between embeddings
        """
        try:
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
        except Exception as e:
            logger.error(f"Error computing distance: {str(e)}")
            return float('inf')

    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: Optional[float] = None) -> bool:
        """
        Determine if two face embeddings represent the same person.
        
        Args:
            embedding1 (np.ndarray): First face embedding
            embedding2 (np.ndarray): Second face embedding
            threshold (Optional[float]): Custom threshold, uses default if None
            
        Returns:
            bool: True if faces match, False otherwise
        """
        if threshold is None:
            threshold = self.tolerance
            
        distance = self.compute_distance(embedding1, embedding2)
        return distance <= threshold

    def batch_extract_embeddings(self, image_paths: List[str]) -> List[Tuple[str, Optional[np.ndarray]]]:
        """
        Extract embeddings from multiple images.
        
        Args:
            image_paths (List[str]): List of image file paths
            
        Returns:
            List[Tuple[str, Optional[np.ndarray]]]: List of (path, embedding) tuples
        """
        results = []
        
        for image_path in image_paths:
            embedding = self.extract_face_embedding(image_path)
            results.append((image_path, embedding))
            
        return results

    def validate_face_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Validate that an image contains exactly one face.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            if not os.path.exists(image_path):
                return False, "Image file not found"
                
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Find face locations
            face_locations = face_recognition.face_locations(image, model="hog")
            
            if len(face_locations) == 0:
                return False, "No faces detected in image"
            elif len(face_locations) > 1:
                return False, f"Multiple faces detected ({len(face_locations)}). Please use an image with exactly one face."
            else:
                return True, "Valid face image"
                
        except Exception as e:
            return False, f"Error validating image: {str(e)}"

    def get_embedding_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_name": "face_recognition (dlib)",
            "embedding_size": self.embedding_size,
            "model_type": self.model,
            "num_jitters": self.num_jitters,
            "tolerance": self.tolerance,
            "description": "Face recognition using dlib's state-of-the-art face recognition built with deep learning"
        }


# if __name__ == "__main__":
#     # Test the face embedding client
#     client = FaceEmbeddingClient()
#     print("Face Embedding Client Info:")
#     print(client.get_embedding_info())
    
#     # Example usage (uncomment to test with actual images)
#     # embedding = client.extract_face_embedding("path/to/face/image.jpg")
#     # if embedding is not None:
#     #     print(f"Extracted embedding shape: {embedding.shape}")
