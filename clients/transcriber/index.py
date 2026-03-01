from .Whisper import Whisper
from utils.helpers import get_optimal_device

transcriberClientMapping = {"whisper": Whisper}


_WHISPER_VRAM_GB = {
    "tiny": 0.5,
    "base": 0.5,
    "small": 1.5,
    "medium": 3.0,
    "large": 5.0,
}

def get_transcriber_client(model_name="whisper", model_size="small") -> Whisper:
    """
    Get transcriber client instance.

    Args:
        model_name (str): Transcriber model name (defaults to "whisper")
        model_size (str): Whisper model size - "tiny", "base", "small", "medium", "large"
        
    Returns:
        Whisper: Whisper transcriber client instance
    """
    min_vram = _WHISPER_VRAM_GB.get(model_size, 1.5)
    device = get_optimal_device(min_vram_gb=min_vram)
    WhisperInstance = transcriberClientMapping.get(model_name, Whisper)
    return WhisperInstance(model_size=model_size, device=device)
