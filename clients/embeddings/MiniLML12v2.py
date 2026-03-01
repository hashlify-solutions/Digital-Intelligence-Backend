from transformers import AutoModel, AutoTokenizer
from torch.nn import functional as F
import torch
import logging
from typing import List, Union
from config.settings import settings

logger = logging.getLogger(__name__)


class MiniLML12V2Client:
    def __init__(self, device: str = "cpu"):
        model_path = "./models/embeddings"
        
        compute = settings.compute_config
        self.batch_size = compute["embedding_batch_size"]
        self.device = device
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        with torch.device('cpu'):
            self.model = AutoModel.from_pretrained(model_path)
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()
            logger.info(f"MiniLML12V2 embeddings model loaded on GPU, batch_size={self.batch_size}")
        else:
            self.device = "cpu"
            logger.info(f"MiniLML12V2 embeddings model loaded on CPU, batch_size={self.batch_size}")
        
        self.model.eval()

    def create_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Create embeddings for one or more texts.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Move inputs to device
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Use inference_mode (faster than no_grad) for inference
        with torch.inference_mode():
            outputs = self.model(**inputs)
        
        embeddings = self.mean_pooling(outputs, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.detach().cpu().numpy().tolist()

    def create_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = None
    ) -> List[List[float]]:
        """
        Create embeddings for a large list of texts in optimized batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Override batch size (uses config default if not provided)
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        batch_size = batch_size or self.batch_size
        all_embeddings = []
        total = len(texts)
        
        logger.info(f"Starting batch embedding: {total} texts, batch_size={batch_size}")
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            try:
                batch_embeddings = self.create_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                
                if (i + batch_size) % (batch_size * 10) == 0:
                    logger.info(f"Embedding progress: {min(i + batch_size, total)}/{total}")
                    
            except Exception as e:
                logger.error(f"Error in embedding batch {i//batch_size}: {e}")
                # Fall back to individual processing
                for text in batch:
                    try:
                        emb = self.create_embeddings([text])[0]
                        all_embeddings.append(emb)
                    except Exception as inner_e:
                        logger.error(f"Error embedding text: {inner_e}")
                        # Return zero vector as fallback
                        all_embeddings.append([0.0] * 384)
        
        logger.info(f"Batch embedding complete: {len(all_embeddings)} embeddings")
        return all_embeddings

    def mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# if __name__ == "__main__":
#     MiniLML12V2ClientObj = MiniLML12V2Client()
#     sentences = [
#         "I love you."
#     ]
#     embeddings = MiniLML12V2ClientObj.create_embeddings(sentences)
#     print(f"Embeddings are: ")
#     print(embeddings)