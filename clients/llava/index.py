from .Llava import Llava

llavaClientMapping = {"llava": Llava}

def get_llava_client() -> Llava:
    """
    Get llava client instance.
        
    Returns:
        Llava: Llava client instance
    """
    return Llava()