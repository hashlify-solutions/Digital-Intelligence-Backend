import cv2
import os
import numpy as np
from typing import Optional, List, Tuple
import logging
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

# Set joblib backend to threading on macOS to avoid fork() issues
# import platform
# if platform.system() == 'Darwin':  # macOS
#     os.environ['JOBLIB_MULTIPROCESSING'] = '0'  # Disable multiprocessing in joblib
#     os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable parallelism in tokenizers
#     os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads

from sentence_transformers import SentenceTransformer
from utils.helpers import monitor_gpu_memory

logger = logging.getLogger(__name__)


class ObjectEmbeddingClient:
    """
    Object embedding client using local models for object similarity matching.
    Supports both CLIP-based and ResNet-based embeddings.
    """

    def __init__(self, model_type, device):
        """
        Initialize the object embedding client.
        
        Args:
            model_type (str): Model to use - 'clip' (recommended) or 'resnet'
            device (str): Device to use - 'auto', 'cpu', 'cuda'
        """
        self.model_type = model_type
        
        # Setup device with robust CUDA detection
        self.device = device
            
        logger.info(f"Initializing ObjectEmbeddingClient with {model_type} on {self.device}")
        
        # Initialize model
        self._load_model()

    # def _detect_available_device(self, device_preference):
    #     """
    #     Detect available device with robust CUDA testing.
        
    #     Args:
    #         device_preference (str): Preferred device - 'auto', 'cpu', 'cuda'
            
    #     Returns:
    #         str: Available device ('cpu' or 'cuda')
    #     """
    #     if device_preference == "cpu":
    #         return "cpu"
    #     elif device_preference == "cuda":
    #         # Force CUDA even if it might fail (for explicit user choice)
    #         return "cuda"
    #     else:  # device_preference == "auto"
    #         if not torch.cuda.is_available():
    #             logger.info("CUDA not available, using CPU")
    #             return "cpu"
            
    #         # Test CUDA initialization to catch driver issues early
    #         try:
    #             # Try to create a small tensor on GPU to test CUDA
    #             test_tensor = torch.tensor([1.0], device='cuda')
    #             test_tensor.cpu()  # Move back to CPU
    #             logger.info("CUDA available and working, using GPU")
    #             return "cuda"
    #         except Exception as e:
    #             logger.warning(f"CUDA available but failed initialization test: {str(e)}")
    #             logger.info("Falling back to CPU device")
    #             return "cpu"

    def _load_model(self):
        """Load the selected model."""
        try:
            if self.model_type == "clip":
                self._load_clip_model()
            elif self.model_type == "resnet":
                self._load_resnet_model()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load {self.model_type} model: {str(e)}")
            
            # Try fallback strategies
            if self.model_type == "clip":
                logger.info("Falling back to ResNet model")
                self.model_type = "resnet"
                try:
                    self._load_resnet_model()
                except Exception as fallback_e:
                    logger.error(f"ResNet fallback also failed: {str(fallback_e)}")
                    # Force CPU and try ResNet again
                    if self.device != "cpu":
                        logger.info("Forcing CPU device and retrying ResNet")
                        self.device = "cpu"
                        self._load_resnet_model()
                    else:
                        raise
            else:
                # For ResNet failures, try CPU device if not already using it
                if self.device != "cpu":
                    logger.info("Retrying with CPU device")
                    self.device = "cpu"
                    self._load_resnet_model()
                else:
                    raise

    def _load_clip_model(self):
        """Load CLIP model for object embeddings."""
        try:
            # Use sentence-transformers CLIP model
            logger.info(f"Loading CLIP model on device: {self.device}")
            
            # Set cache directory to avoid path issues
            import os
            cache_folder = os.path.expanduser("~/.cache/torch/sentence_transformers")
            os.makedirs(cache_folder, exist_ok=True)
            
            # PyTorch 2.8+ compat: use device context to avoid meta tensor initialization
            with torch.device('cpu'):
                self.model = SentenceTransformer(
                    'clip-ViT-B-32', 
                    device='cpu',
                    cache_folder=cache_folder
                )
            # Move to target device after successful materialization
            if self.device != "cpu":
                self.model = self.model.to(self.device)
            self.embedding_size = 512  # CLIP ViT-B-32 embedding size
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model on {self.device}: {str(e)}")
            # Check if this is a CUDA-related error and device is not CPU
            if self.device != "cpu" and ("NVML" in str(e) or "CUDA" in str(e) or "cuda" in str(e)):
                logger.info("CUDA error detected, retrying CLIP with CPU")
                self.device = "cpu"
                
                # Set cache directory to avoid path issues
                import os
                cache_folder = os.path.expanduser("~/.cache/torch/sentence_transformers")
                os.makedirs(cache_folder, exist_ok=True)
                
                with torch.device('cpu'):
                    self.model = SentenceTransformer(
                        'clip-ViT-B-32', 
                        device=self.device,
                        cache_folder=cache_folder
                    )
                self.embedding_size = 512
                logger.info("CLIP model loaded successfully on CPU")
            else:
                raise

    def _load_resnet_model(self):
        """Load ResNet model for object embeddings."""
        try:
            # Load pre-trained ResNet50
            # PyTorch 2.8+ compat: avoid meta tensor issues by loading weights manually with assign=True
            logger.info(f"Loading ResNet model on device: {self.device}")
            self.model = models.resnet50(weights=None)
            state_dict = models.ResNet50_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
            self.model.load_state_dict(state_dict, assign=True)
            
            # Remove the final classification layer to get embeddings
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            
            self.model.eval()
            
            # Try to move model to device with error handling
            try:
                self.model.to(self.device)
                logger.info(f"ResNet model moved to {self.device} successfully")
            except Exception as device_e:
                logger.error(f"Failed to move ResNet model to {self.device}: {str(device_e)}")
                if self.device != "cpu" and ("NVML" in str(device_e) or "CUDA" in str(device_e) or "cuda" in str(device_e)):
                    logger.info("CUDA error detected, moving ResNet to CPU")
                    self.device = "cpu"
                    self.model.to(self.device)
                    logger.info("ResNet model moved to CPU successfully")
                else:
                    raise device_e
            
            self.embedding_size = 2048  # ResNet50 feature size
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("ResNet model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load ResNet model: {str(e)}")
            raise

    def extract_object_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract object embedding from an image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Optional[np.ndarray]: Object embedding or None if extraction failed
        """
        try:
            # Validate image path
            if not os.path.exists(image_path):
                logger.error(f"Image file not found: {image_path}")
                return None

            if self.model_type == "clip":
                return self._extract_clip_embedding(image_path)
            elif self.model_type == "resnet":
                return self._extract_resnet_embedding(image_path)
                
        except Exception as e:
            logger.error(f"Error extracting object embedding from {image_path}: {str(e)}")
            return None

    def _extract_clip_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embedding using CLIP model."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # CLIP can handle images directly
            with monitor_gpu_memory("clip_embedding"):
                embedding = self.model.encode([image], convert_to_numpy=True)[0]
            
            logger.debug(f"Extracted CLIP embedding of shape {embedding.shape} from {image_path}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting CLIP embedding: {str(e)}")
            return None

    def _extract_resnet_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """Extract embedding using ResNet model."""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.model(image_tensor)
                embedding = features.squeeze().cpu().numpy()
            
            logger.debug(f"Extracted ResNet embedding of shape {embedding.shape} from {image_path}")
            return embedding
            
        except Exception as e:
            logger.error(f"Error extracting ResNet embedding: {str(e)}")
            return None

    def extract_object_embedding_from_array(self, image_array: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract object embedding from a numpy image array.
        
        Args:
            image_array (np.ndarray): Image as numpy array (RGB format)
            
        Returns:
            Optional[np.ndarray]: Object embedding or None if extraction failed
        """
        try:
            # Convert numpy array to PIL Image
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
                
            image = Image.fromarray(image_array)
            
            if self.model_type == "clip":
                embedding = self.model.encode([image], convert_to_numpy=True)[0]
                return embedding
            elif self.model_type == "resnet":
                image_tensor = self.transform(image).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model(image_tensor)
                    embedding = features.squeeze().cpu().numpy()
                return embedding
                
        except Exception as e:
            logger.error(f"Error extracting object embedding from image array: {str(e)}")
            return None

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two object embeddings.
        
        Args:
            embedding1 (np.ndarray): First object embedding
            embedding2 (np.ndarray): Second object embedding
            
        Returns:
            float: Similarity score between 0 and 1 (1 = identical, 0 = completely different)
        """
        try:
            # Normalize embeddings
            embedding1_norm = embedding1 / np.linalg.norm(embedding1)
            embedding2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(embedding1_norm, embedding2_norm)
            
            # Ensure similarity is in 0-1 range
            similarity = max(0, min(1, (similarity + 1) / 2))
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing objects similarity: {str(e)}")
            return 0.0

    def compute_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two object embeddings.
        Lower distance means more similar objects.
        
        Args:
            embedding1 (np.ndarray): First object embedding
            embedding2 (np.ndarray): Second object embedding
            
        Returns:
            float: Distance between embeddings
        """
        try:
            distance = np.linalg.norm(embedding1 - embedding2)
            return float(distance)
        except Exception as e:
            logger.error(f"Error computing distance: {str(e)}")
            return float('inf')

    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray, threshold: float = 0.7) -> bool:
        """
        Determine if two object embeddings represent similar objects.
        
        Args:
            embedding1 (np.ndarray): First object embedding
            embedding2 (np.ndarray): Second object embedding
            threshold (float): Similarity threshold (0-1)
            
        Returns:
            bool: True if objects are similar, False otherwise
        """
        similarity = self.compute_similarity(embedding1, embedding2)
        return similarity >= threshold

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
            embedding = self.extract_object_embedding(image_path)
            results.append((image_path, embedding))
            
        return results

    def validate_object_image(self, image_path: str) -> Tuple[bool, str]:
        """
        Validate that an image is suitable for object detection.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple[bool, str]: (is_valid, message)
        """
        try:
            if not os.path.exists(image_path):
                return False, "Image file not found"
                
            # Load image to check if it's readable
            image = Image.open(image_path).convert('RGB')
            
            # Check image dimensions
            width, height = image.size
            if width < 32 or height < 32:
                return False, "Image too small (minimum 32x32 pixels)"
                
            if width > 4096 or height > 4096:
                return False, "Image too large (maximum 4096x4096 pixels)"
                
            return True, "Valid object image"
                
        except Exception as e:
            return False, f"Error validating image: {str(e)}"

    def get_embedding_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            dict: Model information
        """
        model_descriptions = {
            "clip": "CLIP (Contrastive Language-Image Pre-training) - Excellent for general object similarity",
            "resnet": "ResNet50 - Convolutional neural network for visual feature extraction"
        }
        
        return {
            "model_name": f"{self.model_type.upper()}",
            "embedding_size": self.embedding_size,
            "device": self.device,
            "description": model_descriptions.get(self.model_type, "Unknown model"),
            "model_type": self.model_type
        }


# if __name__ == "__main__":
    # Test the object embedding client
    # client = ObjectEmbeddingClient(model_type="clip", device="cpu")
    # print("Object Embedding Client Info:")
    # print(client.get_embedding_info())
    
    # Example usage (uncomment to test with actual images)
    # embedding = client.extract_object_embedding("path/to/object/image.jpg")
    # if embedding is not None:
    #     print(f"Extracted embedding shape: {embedding.shape}")
