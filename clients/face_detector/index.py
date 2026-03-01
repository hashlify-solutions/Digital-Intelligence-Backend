from .HaarFaceDetector import HaarFaceDetector
from .DNNFaceDetector import DNNFaceDetector
from utils.helpers import get_optimal_device

faceDetectorClientMapping = {
    "haar": HaarFaceDetector,
    "dnn": DNNFaceDetector,
}


def get_face_detector_client(model_name="dnn") -> HaarFaceDetector | DNNFaceDetector:
    if model_name == "dnn":
        device = get_optimal_device(min_vram_gb=0.2)
        return DNNFaceDetector(device=device)
    return faceDetectorClientMapping.get(model_name, DNNFaceDetector)()
