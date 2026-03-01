from .NudeNetClassifier import NudeNetClassifier

nsfwDetectorClientMapping = {
    "nude": NudeNetClassifier,
}

def get_nsfw_detector_client(model_name="nude") -> NudeNetClassifier:
    return nsfwDetectorClientMapping.get(model_name, NudeNetClassifier)()