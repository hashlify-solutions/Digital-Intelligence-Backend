from transformers import pipeline 
import logging
import torch
from typing import Optional, Dict, Any, List
from config.settings import settings

logger = logging.getLogger(__name__)


class AkhooliXLMLargeArabicToxicClient:
    def __init__(self, device="cpu"):
        model_path = f"./models/toxic"
        self.device = device
        
        # Get batch size from centralized configuration
        compute = settings.compute_config
        self.batch_size = compute["gpu_batch_size"]
        
        # Use FP16 on GPU for faster inference
        use_fp16 = device == 'cuda' and torch.cuda.is_available()
        dtype_kwarg = {"torch_dtype": torch.float16} if use_fp16 else {}
        
        try:
            self.toxicity_analyzer = pipeline(
                "sentiment-analysis", 
                model=model_path, 
                device=0 if device == 'cuda' else -1, 
                truncation=True,
                **dtype_kwarg
            )
            logger.info(f"Akhooli XLM toxicity analyzer loaded on {device}, batch_size={self.batch_size}")
        except RuntimeError as e:
            if "NVML" in str(e) or "CUDA" in str(e):
                logger.error(f"NVML/CUDA error detected in Akhooli XLM large Arabic toxicity analyzer, falling back to CPU: {e}")
                self.device = 'cpu'
                self.toxicity_analyzer = pipeline("sentiment-analysis", model=model_path, device=-1, truncation=True)
                logger.info("Akhooli XLM large Arabic toxicity analyzer loaded successfully on CPU (fallback)")

    def _map_label(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Map internal labels to human-readable labels."""
        label_mapping = {
            "LABEL_1": "toxic",
            "LABEL_0": "non-toxic"
        }
        result["label"] = label_mapping.get(result["label"], result["label"])
        return result

    def analyze_toxicity(self, sequence_to_analyze: str) -> Optional[Dict[str, Any]]:
        """Analyze toxicity of a single text sequence."""
        try:
            result = self.toxicity_analyzer(sequence_to_analyze, truncation=True, max_length=512)[0]
            return self._map_label(result)
        except RuntimeError as e:
            logger.error(f"Error in toxicity analysis: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in toxicity analysis: {e}")
            return None
    
    def analyze_toxicity_batch(
        self, 
        sequences: List[str], 
        batch_size: int = None
    ) -> List[Optional[Dict[str, Any]]]:
        """
        Analyze toxicity for multiple sequences efficiently in batches.
        
        Args:
            sequences: List of text sequences to analyze
            batch_size: Override batch size (uses config default if not provided)
            
        Returns:
            List of toxicity results for each sequence
        """
        if not sequences:
            return []
        
        batch_size = batch_size or self.batch_size
        results = []
        total = len(sequences)
        
        logger.info(f"Starting batch toxicity analysis: {total} sequences, batch_size={batch_size}")
        
        for i in range(0, total, batch_size):
            batch = sequences[i:i + batch_size]
            try:
                # Pipeline handles batching internally
                batch_results = self.toxicity_analyzer(
                    batch, 
                    truncation=True, 
                    max_length=512
                )
                
                # Map labels for each result
                for result in batch_results:
                    results.append(self._map_label(result))
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Toxicity analysis progress: {min(i + batch_size, total)}/{total}")
                    
            except Exception as e:
                logger.error(f"Error in toxicity batch {i//batch_size}: {e}")
                # Fall back to individual processing
                for seq in batch:
                    result = self.analyze_toxicity(seq)
                    results.append(result)
        
        logger.info(f"Batch toxicity analysis complete: {len(results)} results")
        return results
    
# if __name__ == "__main__":
#     akhooliXLMLargeArabicToxicClientObj = AkhooliXLMLargeArabicToxicClient()
#     sequence_to_analyze = "مرحبًا، كيف حالك؟ سأعطيك الكثير من المخدرات اليوم وسأجعلك مريضًا اليوم يا حبيبي."
#     result = akhooliXLMLargeArabicToxicClientObj.analyze_toxicity(sequence_to_analyze)
#     print(result)