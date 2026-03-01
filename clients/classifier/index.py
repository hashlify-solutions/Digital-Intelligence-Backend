from .XlmRobertaLargeXnli import XlmRobertaLargeXnliClient
from utils.helpers import get_optimal_device

classifierClientMapping = {
    "joeddav/xlm-roberta-large-xnli": XlmRobertaLargeXnliClient
}

def get_classifier_client(model_name):
    device = get_optimal_device(min_vram_gb=2.0)
    return classifierClientMapping.get(model_name, XlmRobertaLargeXnliClient)(device)