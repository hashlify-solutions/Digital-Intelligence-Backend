from transformers import pipeline
import logging
import torch
from typing import List, Dict, Any, Union
from config.settings import settings

logger = logging.getLogger(__name__)


class XlmRobertaLargeXnliClient:
    def __init__(self, device):
        model_path = "./models/classifier"
        self.device = device
        
        # Get batch size from centralized configuration
        compute = settings.compute_config
        self.batch_size = compute["gpu_batch_size"]
        
        # Use FP16 on GPU for faster inference (RTX 5090 has excellent FP16 throughput)
        use_fp16 = device == 'cuda' and torch.cuda.is_available()
        dtype_kwarg = {"torch_dtype": torch.float16} if use_fp16 else {}
        
        try:
            # Try to create pipeline with specified device
            self.classifier = pipeline(
                "zero-shot-classification", 
                model=model_path, 
                device=0 if device == 'cuda' else -1,
                **dtype_kwarg
            )
            logger.info(f"XLM-RoBERTa large XNLI classifier loaded successfully on {device}, "
                       f"batch_size={self.batch_size}")
        except RuntimeError as e:
            if "NVML" in str(e) or "CUDA" in str(e):
                logger.error(f"NVML/CUDA error detected in XLM-RoBERTa large XNLI classifier, falling back to CPU: {e}")
                self.device = 'cpu'
                self.classifier = pipeline(
                    "zero-shot-classification", 
                    model=model_path, 
                    device=-1
                )
                logger.info("XLM-RoBERTa large XNLI classifier loaded successfully on CPU (fallback)")
            else:
                raise

    def classify(self, sequence_to_classify: str, candidate_labels: List[str]) -> Dict[str, Any]:
        """Classify a single sequence."""
        return self.classifier(sequence_to_classify, candidate_labels, multi_label=True)
    
    def classify_batch(
        self, 
        sequences: List[str], 
        candidate_labels: List[str],
        batch_size: int = None
    ) -> List[Dict[str, Any]]:
        """
        Classify multiple sequences efficiently in batches.
        
        Args:
            sequences: List of text sequences to classify
            candidate_labels: List of labels for classification
            batch_size: Override batch size (uses config default if not provided)
            
        Returns:
            List of classification results for each sequence
        """
        if not sequences:
            return []
        
        batch_size = batch_size or self.batch_size
        results = []
        total = len(sequences)
        
        logger.info(f"Starting batch classification: {total} sequences, batch_size={batch_size}")
        
        for i in range(0, total, batch_size):
            batch = sequences[i:i + batch_size]
            try:
                # The transformers pipeline handles batching internally
                batch_results = self.classifier(
                    batch, 
                    candidate_labels, 
                    multi_label=True
                )
                
                # Ensure results are always a list
                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
                    
                results.extend(batch_results)
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Classification progress: {min(i + batch_size, total)}/{total}")
                    
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size}: {e}")
                # Fall back to individual processing for this batch
                for seq in batch:
                    try:
                        result = self.classify(seq, candidate_labels)
                        results.append(result)
                    except Exception as inner_e:
                        logger.error(f"Error classifying sequence: {inner_e}")
                        results.append({"error": str(inner_e), "sequence": seq[:100]})
        
        logger.info(f"Batch classification complete: {len(results)} results")
        return results

# if __name__ == "__main__":
#     xlmRobertaLargeXnliClientObject = XlmRobertaLargeXnliClient()
#     sequence_to_classify = "مرحبًا، كيف حالك؟ سأعطيك الكثير من المخدرات اليوم وسأجعلك مريضًا اليوم يا حبيبي."
#     candidate_labels = ["sports", "politics", "general"]
#     classification_result = xlmRobertaLargeXnliClientObject.classify(sequence_to_classify, candidate_labels)
#     print(classification_result)