from .TwitterXlmRobertaBaseSentiment import TwitterXlmRobertaBaseSentimentClient
from utils.helpers import get_optimal_device

emotionClientMapping = {
    "cardiffnlp/twitter-xlm-roberta-base-sentiment": TwitterXlmRobertaBaseSentimentClient
}

def get_emotion_client(model_name):
    device = get_optimal_device(min_vram_gb=1.0)
    return emotionClientMapping.get(model_name, TwitterXlmRobertaBaseSentimentClient)(device)