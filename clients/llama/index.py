from .Llama import Llama

llamaClientMapping = {"llama": Llama}

def get_llama_client() -> Llama:
    """
    Get llama client instance.
        
    Returns:
        Llama: Llama client instance
    """
    return Llama()