"""
LLM Configuration for Arabic Social Analyzer
"""

from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class LLMConfig:
    """Configuration for LLM models and parameters"""
    
    # Model priority list (fastest to slowest)
    MODEL_PRIORITY = [
        "llama3.1:8b",    # Good balance - best for classification
        # "llama3:8b",      # Alternative
        # "llama3.1:1b",    # Fastest, smallest - best for entity extraction
        # "llama2:7b",      # Fallback
        # "llama2:13b"      # Largest, slowest
    ]
    
    # Fast processing parameters for entity extraction
    FAST_PARAMS = {
        "temperature": 0.1,      # Very low temperature for deterministic responses
        "num_ctx": 512,          # Minimal context for faster processing
        "num_tokens": 128,       # Very short responses for entities
        "num_thread": 8,         # Maximum threads
        "num_gpu": 1,            # Use GPU
        "num_gqa": 8,            # Group query attention
        "rope_freq_base": 10000,
        "rope_freq_scale": 0.5,
        "repeat_penalty": 1.0,   # No repetition penalty for speed
        "top_k": 10,             # Very restrictive for speed
        "top_p": 0.8,            # Lower for speed
        "stream": False,         # No streaming
        "num_predict": 128,      # Very short predictions
        "cache_prompt": True,    # Cache everything
        "prompt_cache_all": True,
    }
    
    # Standard processing parameters for classification
    STANDARD_PARAMS = {
        "temperature": 0.3,      # Low temperature for consistent responses
        "num_ctx": 1024,         # Moderate context
        "num_tokens": 256,       # Moderate response length
        "num_thread": 4,         # Moderate threads
        "num_gpu": 1,            # Use GPU
        "num_gqa": 8,            # Group query attention
        "rope_freq_base": 10000,
        "rope_freq_scale": 0.5,
        "repeat_penalty": 1.1,   # Slight repetition penalty
        "top_k": 40,             # Moderate restriction
        "top_p": 0.9,            # Moderate sampling
        "stream": False,         # No streaming
        "num_predict": 256,      # Moderate predictions
        "cache_prompt": True,    # Cache everything
        "prompt_cache_all": True,
    }
    
    # Entity extraction prompt template
    ENTITY_EXTRACTION_PROMPT = """Extract entities (names, locations, organizations, and other key information) from the following text:

{preview_text}

Return your response as a JSON string in exactly this format:
[
    'entity1',
    'entity2',
    'entity3',
    ...
]
Return ONLY a valid JSON array with no additional text, comments, or explanations.
If no entities are found, return "[]".
"""
    
    # Entity classification prompt template
    ENTITY_CLASSIFICATION_PROMPT = """Classify the given entities into one of the following categories:

Entities:
{entities}

Categories:
{categories}

Return your response as a JSON string in exactly this format:

{{
    "category1": ["entity1", "entity2", "entity3"],
    "category2": ["entity4", "entity5", "entity6"],
    ...
}}

Return ONLY a valid JSON object with no additional text, comments, or explanations.
"""
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available models (to be implemented with Ollama check)"""
        # This should be implemented to check actual available models
        # For now, return the priority list
        return cls.MODEL_PRIORITY
    
    @classmethod
    def get_best_model_for_task(cls, task_type: str = "entity_extraction") -> str:
        """Get the best available model for a specific task"""
        available_models = cls.get_available_models()
        
        if task_type == "entity_extraction":
            # Prefer faster models for entity extraction
            priority_models = ["llama3.1:1b", "llama3.1:8b", "llama3:8b", "llama2:7b", "llama2:13b"]
        else:
            # Prefer larger models for classification
            priority_models = ["llama3.1:8b", "llama3:8b", "llama2:13b", "llama2:7b", "llama3.1:1b"]
        
        for model in priority_models:
            if model in available_models:
                logger.info(f"Selected model for {task_type}: {model}")
                return model
        
        # Fallback to first available model
        if available_models:
            logger.warning(f"No preferred model available for {task_type}, using: {available_models[0]}")
            return available_models[0]
        
        logger.error("No models available")
        return ""
    
    @classmethod
    def get_params_for_task(cls, task_type: str = "entity_extraction") -> Dict:
        """Get parameters for a specific task"""
        if task_type == "entity_extraction":
            return cls.FAST_PARAMS.copy()
        else:
            return cls.STANDARD_PARAMS.copy()
    
    @classmethod
    def get_prompt_for_task(cls, task_type: str) -> str:
        """Get prompt template for a specific task"""
        if task_type == "entity_extraction":
            return cls.ENTITY_EXTRACTION_PROMPT
        elif task_type == "entity_classification":
            return cls.ENTITY_CLASSIFICATION_PROMPT
        else:
            raise ValueError(f"Unknown task type: {task_type}")

# Global configuration instance
llm_config = LLMConfig() 