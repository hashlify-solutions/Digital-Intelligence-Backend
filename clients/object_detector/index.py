from .YoloObjectDetector import YoloObjectDetector
from utils.helpers import get_optimal_device

objectDetectorClientMapping = {
    "yolo": YoloObjectDetector,
}

def get_object_detector_client(model_name="yolo")->YoloObjectDetector:
    device = get_optimal_device(min_vram_gb=3.0)
    return objectDetectorClientMapping.get(model_name, YoloObjectDetector)(device=device)