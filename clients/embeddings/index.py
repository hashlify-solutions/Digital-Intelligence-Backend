from .MiniLML12v2 import MiniLML12V2Client
from .FaceEmbeddingClient import FaceEmbeddingClient
from .ObjectEmbeddingClient import ObjectEmbeddingClient
from utils.helpers import get_optimal_device

embeddingClientsMapping = {
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": MiniLML12V2Client
}

face_embedding_client_mapping = {
    "face_recognition": FaceEmbeddingClient,
}

object_embedding_client_mapping = {
    "clip": ObjectEmbeddingClient,
    "resnet": ObjectEmbeddingClient,
}

def get_embeddings_client(model_name):
    """Get text embedding client with VRAM-aware device selection."""
    device = get_optimal_device(min_vram_gb=0.5)
    return embeddingClientsMapping.get(model_name, MiniLML12V2Client)(device=device)

def get_face_embedding_client(
    model_name="face_recognition"
) -> FaceEmbeddingClient:
    """
    Get face embedding client instance.

    Uses dlib/face_recognition which is CPU-only — no GPU device needed.

    Args:
        model_name (str): Model name (currently only 'face_recognition' supported)
    Returns:
        FaceEmbeddingClient: Face embedding client instance
    """
    FaceEmbeddingClientInstance = face_embedding_client_mapping.get(model_name, FaceEmbeddingClient)
    return FaceEmbeddingClientInstance()


def get_object_embedding_client(model_name="clip") -> ObjectEmbeddingClient:
    """
    Get object embedding client instance.
    Args:
        model_name (str): Model name ('clip' or 'resnet')

    Returns:
        ObjectEmbeddingClient: Object embedding client instance
    """
    min_vram = 1.0 if model_name == "clip" else 0.5
    device = get_optimal_device(min_vram_gb=min_vram)
    ObjectEmbeddingClientInstance = object_embedding_client_mapping.get(model_name, ObjectEmbeddingClient)
    return ObjectEmbeddingClientInstance(model_type=model_name, device=device)