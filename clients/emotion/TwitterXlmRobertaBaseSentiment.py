from transformers import pipeline
import logging
import torch
from typing import List, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class TwitterXlmRobertaBaseSentimentClient:
    def __init__(self, device='cpu'):
        model_path = "./models/emotion"
        self.device = device
        
        # Get batch size from centralized configuration
        compute = settings.compute_config
        self.batch_size = compute["gpu_batch_size"]
        
        # Use FP16 on GPU for faster inference
        use_fp16 = device == 'cuda' and torch.cuda.is_available()
        dtype_kwarg = {"torch_dtype": torch.float16} if use_fp16 else {}
        
        try:
            # Try to create pipeline with specified device
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis", 
                model=model_path, 
                tokenizer=model_path, 
                device=0 if device == 'cuda' else -1,
                **dtype_kwarg
            )
            logger.info(f"Twitter XLM-RoBERTa sentiment analyzer loaded on {device}, batch_size={self.batch_size}")
            
        except RuntimeError as e:
            if "NVML" in str(e) or "CUDA" in str(e):
                logger.error(f"NVML/CUDA error detected in sentiment analyzer, falling back to CPU: {e}")
                self.device = 'cpu'
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model=model_path, 
                    tokenizer=model_path, 
                    device=-1
                )
                logger.info("Twitter XLM-RoBERTa sentiment analyzer loaded successfully on CPU (fallback)")
            else:
                raise

    def analyze_sentiment(self, sequence_to_analyze: str) -> Dict[str, Any]: 
        """Analyze sentiment of a single text sequence."""
        return self.sentiment_analyzer(sequence_to_analyze)[0]
    
    def analyze_sentiment_batch(
        self, 
        sequences: List[str], 
        batch_size: int = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze sentiment for multiple sequences efficiently in batches.
        
        Args:
            sequences: List of text sequences to analyze
            batch_size: Override batch size (uses config default if not provided)
            
        Returns:
            List of sentiment results for each sequence
        """
        if not sequences:
            return []
        
        batch_size = batch_size or self.batch_size
        results = []
        total = len(sequences)
        
        logger.info(f"Starting batch sentiment analysis: {total} sequences, batch_size={batch_size}")
        
        for i in range(0, total, batch_size):
            batch = sequences[i:i + batch_size]
            try:
                # Pipeline handles batching internally
                batch_results = self.sentiment_analyzer(batch)
                results.extend(batch_results)
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Sentiment analysis progress: {min(i + batch_size, total)}/{total}")
                    
            except Exception as e:
                logger.error(f"Error in sentiment batch {i//batch_size}: {e}")
                # Fall back to individual processing
                for seq in batch:
                    try:
                        result = self.analyze_sentiment(seq)
                        results.append(result)
                    except Exception as inner_e:
                        logger.error(f"Error analyzing sentiment: {inner_e}")
                        results.append({"label": "unknown", "score": 0.0, "error": str(inner_e)})
        
        logger.info(f"Batch sentiment analysis complete: {len(results)} results")
        return results
    

# if __name__ == "__main__":
#     twitterXlmRobertaBaseSentimentObject = TwitterXlmRobertaBaseSentimentClient()
    # sequenceToAnalyze = "سأساعدك وأساندك في كل مرحلة من مراحل حياتك." # Positive
    # sequenceToAnalyze = "سأقتلك" # Negative
    # sequenceToAnalyze = "مرحبًا، كيف حالك؟ سأعطيك الكثير من المخدرات اليوم وسأجعلك مريضًا اليوم يا حبيبي." # Neutral
    # result = twitterXlmRobertaBaseSentimentObject.analyze_sentiment(sequenceToAnalyze)
    # print(result)