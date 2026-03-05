from .Llama import Llama
from config.settings import settings

llamaClientMapping = {"llama": Llama}

def get_llama_client() -> Llama:
    """
    Get llama client instance.
        
    Returns:
        Llama: Llama client instance
    """
    return Llama(timeout=settings.effective_llama_timeout)