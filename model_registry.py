"""
Model Registry - Singleton for managing shared AI model instances.

This module provides a centralized registry for AI models to avoid
reloading models for each task and enable efficient multi-GPU utilization.

Usage:
    from model_registry import ModelRegistry
    
    # Get a model (loads only once, cached thereafter)
    classifier = ModelRegistry.get_model("classifier")
    
    # Get model batch size from centralized config
    batch_size = ModelRegistry.get_batch_size("classifier")
"""

import threading
import logging
from typing import Dict, Any, Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for managing AI model instances.
    
    Features:
    - Thread-safe model loading with double-checked locking
    - Multi-GPU support with device assignment
    - Centralized batch size configuration
    - Lazy loading - models loaded only when first requested
    """
    
    _instance = None
    _lock = threading.Lock()
    _models: Dict[str, Any] = {}
    _model_locks: Dict[str, threading.Lock] = {}
    _initialized = False
    
    # Model type to loader mapping
    _model_loaders = {
        "classifier": "_load_classifier",
        "toxic": "_load_toxic",
        "emotion": "_load_emotion",
        "embeddings": "_load_embeddings",
        "face_embeddings": "_load_face_embeddings",
        "object_embeddings": "_load_object_embeddings",
        "face_detector": "_load_face_detector",
        "object_detector": "_load_object_detector",
        "nsfw_detector": "_load_nsfw_detector",
        "transcriber": "_load_transcriber",
        "llava": "_load_llava",
        "llama": "_load_llama",
        "async_llama": "_load_async_llama",
    }
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_registry()
        return cls._instance
    
    def _init_registry(self):
        """Initialize the registry with configuration."""
        if self._initialized:
            return
            
        self._compute_config = settings.compute_config
        logger.info(f"ModelRegistry initialized with config: "
                   f"gpu_count={self._compute_config['gpu_count']}, "
                   f"gpu_batch_size={self._compute_config['gpu_batch_size']}")
        self._initialized = True
    
    @classmethod
    def get_model(cls, model_type: str, device_id: int = 0, **kwargs) -> Any:
        """
        Get a model instance, loading it if not already cached.
        
        Args:
            model_type: Type of model (e.g., "classifier", "toxic", "emotion")
            device_id: GPU device ID to use (default: 0)
            **kwargs: Additional arguments for model initialization
            
        Returns:
            The model instance
            
        Raises:
            ValueError: If model_type is not recognized
        """
        instance = cls()
        key = f"{model_type}_{device_id}"
        
        # Fast path - check if model exists
        if key in cls._models:
            return cls._models[key]
        
        # Get or create lock for this model type
        if key not in cls._model_locks:
            with cls._lock:
                if key not in cls._model_locks:
                    cls._model_locks[key] = threading.Lock()
        
        # Double-checked locking for thread safety
        with cls._model_locks[key]:
            if key not in cls._models:
                model = instance._load_model(model_type, device_id, **kwargs)
                cls._models[key] = model
                logger.info(f"Model '{model_type}' loaded on device {device_id}")
        
        return cls._models[key]
    
    @classmethod
    def get_batch_size(cls, model_type: str) -> int:
        """
        Get the optimal batch size for a model type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Optimal batch size from centralized configuration
        """
        compute = settings.compute_config
        
        if model_type == "embeddings":
            return compute["embedding_batch_size"]
        elif model_type in ["classifier", "toxic", "emotion", "ner"]:
            return compute["gpu_batch_size"]
        else:
            return compute["batch_size"]
    
    @classmethod
    def get_device(cls) -> str:
        """Get the appropriate device string based on configuration."""
        compute = settings.compute_config
        if compute["gpu_count"] > 0:
            return "cuda"
        return "cpu"
    
    @classmethod
    def clear_cache(cls, model_type: Optional[str] = None):
        """
        Clear cached models.
        
        Args:
            model_type: If provided, only clear models of this type.
                       If None, clear all models.
        """
        with cls._lock:
            if model_type:
                keys_to_remove = [k for k in cls._models if k.startswith(model_type)]
                for key in keys_to_remove:
                    del cls._models[key]
                    logger.info(f"Cleared model cache: {key}")
            else:
                cls._models.clear()
                logger.info("Cleared all model caches")
    
    @classmethod
    def list_loaded_models(cls) -> list:
        """Get list of currently loaded models."""
        return list(cls._models.keys())
    
    def _load_model(self, model_type: str, device_id: int, **kwargs) -> Any:
        """Load a model based on type."""
        if model_type not in self._model_loaders:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available types: {list(self._model_loaders.keys())}")
        
        loader_name = self._model_loaders[model_type]
        loader = getattr(self, loader_name)
        return loader(device_id, **kwargs)
    
    def _load_classifier(self, device_id: int, **kwargs):
        """Load the classifier model."""
        from clients.classifier.index import get_classifier_client
        model_name = kwargs.get("model_name")
        return get_classifier_client(model_name)
    
    def _load_toxic(self, device_id: int, **kwargs):
        """Load the toxicity detection model."""
        from clients.toxic.index import get_toxic_client
        model_name = kwargs.get("model_name")
        return get_toxic_client(model_name)
    
    def _load_emotion(self, device_id: int, **kwargs):
        """Load the emotion analysis model."""
        from clients.emotion.index import get_emotion_client
        model_name = kwargs.get("model_name")
        return get_emotion_client(model_name)
    
    def _load_embeddings(self, device_id: int, **kwargs):
        """Load the text embeddings model."""
        from clients.embeddings.index import get_embeddings_client
        model_name = kwargs.get("model_name")
        return get_embeddings_client(model_name)
    
    def _load_face_embeddings(self, device_id: int, **kwargs):
        """Load the face embeddings model."""
        from clients.embeddings.index import get_face_embedding_client
        model_name = kwargs.get("model_name", "face_recognition")
        return get_face_embedding_client(model_name)
    
    def _load_object_embeddings(self, device_id: int, **kwargs):
        """Load the object embeddings model."""
        from clients.embeddings.index import get_object_embedding_client
        model_name = kwargs.get("model_name", "clip")
        return get_object_embedding_client(model_name)
    
    def _load_face_detector(self, device_id: int, **kwargs):
        """Load the face detection model."""
        from clients.face_detector.index import get_face_detector_client
        model_name = kwargs.get("model_name", "dnn")
        return get_face_detector_client(model_name)
    
    def _load_object_detector(self, device_id: int, **kwargs):
        """Load the object detection model."""
        from clients.object_detector.index import get_object_detector_client
        model_name = kwargs.get("model_name", "yolo")
        return get_object_detector_client(model_name)
    
    def _load_nsfw_detector(self, device_id: int, **kwargs):
        """Load the NSFW detection model."""
        from clients.nsfw_detector.index import get_nsfw_detector_client
        return get_nsfw_detector_client()
    
    def _load_transcriber(self, device_id: int, **kwargs):
        """Load the audio transcription model."""
        from clients.transcriber.index import get_transcriber_client
        return get_transcriber_client()
    
    def _load_llava(self, device_id: int, **kwargs):
        """Load the LLaVA model for image description."""
        from clients.llava.index import get_llava_client
        return get_llava_client()
    
    def _load_llama(self, device_id: int, **kwargs):
        """Load the LLaMA model for text generation."""
        from clients.llama.index import get_llama_client
        return get_llama_client()
    
    def _load_async_llama(self, device_id: int, **kwargs):
        """Load the AsyncLlamaClient for concurrent batch LLM processing."""
        from clients.llama.async_llama_client import AsyncLlamaClient
        return AsyncLlamaClient(
            basic_params=kwargs.get("basic_params"),
            advanced_params=kwargs.get("advanced_params"),
            prompt_engineering=kwargs.get("prompt_engineering"),
        )


def preload_models(model_types: Optional[list] = None):
    """
    Preload specified models into memory.
    
    Useful for warming up workers on startup to avoid
    first-request latency.
    
    Args:
        model_types: List of model types to preload.
                    If None, preloads common models.
    """
    if model_types is None:
        # Default models to preload for text analysis
        model_types = ["classifier", "toxic", "emotion", "embeddings"]
    
    logger.info(f"Preloading models: {model_types}")
    
    for model_type in model_types:
        try:
            ModelRegistry.get_model(model_type)
            logger.info(f"Preloaded: {model_type}")
        except Exception as e:
            logger.error(f"Failed to preload {model_type}: {e}")
    
    logger.info("Model preloading complete")
