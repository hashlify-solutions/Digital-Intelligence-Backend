from .AkhooliXLMLargeArabicToxic import AkhooliXLMLargeArabicToxicClient
from utils.helpers import get_optimal_device

toxicClientMapping = {
    "akhooli/xlm-r-large-arabic-toxic": AkhooliXLMLargeArabicToxicClient
}

def get_toxic_client(model_name):
    device = get_optimal_device(min_vram_gb=2.0)
    return toxicClientMapping.get(model_name, AkhooliXLMLargeArabicToxicClient)(device)