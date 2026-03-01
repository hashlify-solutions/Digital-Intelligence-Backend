# Import module
from nudenet import NudeClassifier
import logging

logger = logging.getLogger(__name__)

class NudeNetClassifier:
    def __init__(self):
        self.model = NudeClassifier()

    def is_nsfw_image(self, input_image_path):
        try:
            nsfw_detection_score = self.model.classify(input_image_path)
            if nsfw_detection_score[input_image_path]['unsafe'] > 0.8:
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error in detecting for nsfw: {str(e)}")
            return False 