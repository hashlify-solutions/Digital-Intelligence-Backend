from .Llava import Llava
from config.settings import settings

llavaClientMapping = {"llava": Llava}

def get_llava_client() -> Llava:
    """
    Get llava client instance.
        
    Returns:
        Llava: Llava client instance
    """
    return Llava(timeout=settings.effective_llava_timeout)