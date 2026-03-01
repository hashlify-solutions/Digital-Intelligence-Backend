import asyncio
import aiohttp
import json
import logging
from typing import Optional, Dict, Any, List
from setup import setup_logging
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time
from config.llm_config import llm_config
from config.settings import settings
from concurrent.futures import ThreadPoolExecutor

logger = setup_logging()


class AsyncLlamaClient:
    """Async LLM client for concurrent processing with connection pooling"""
    
    # Default basic parameters
    DEFAULT_BASIC_PARAMS = {
        "temperature": 0.6,
        "num_ctx": 2048,
        "num_tokens": 512
    }

    # Default advanced parameters
    DEFAULT_ADVANCED_PARAMS = {
        "num_layers": 12,
        "hidden_size": 768,
        "activation_function": "GELU",
        "dropout_rate": 0.1,
        "attention_heads": 12,
        "vocab_size": 30522,
        "max_sequence_length": 128,
        "learning_rate": 3e-5,
        "optimizer": "AdamW",
        "batch_size": 32,
        "epochs": 5,
        "loss_function": "CrossEntropy",
        "weight_decay": 0.01,
        "gradient_clipping": 1.0,
        "early_stopping": True,
        "validation_split": 0.2,
        "learning_rate_scheduler": "linear_decay",
        "seed": 42,
        "log_interval": 50,
        "save_checkpoint": "every_epoch"
    }
    
    # Shared connection pool and executor across instances
    _session: Optional[aiohttp.ClientSession] = None
    _executor: Optional[ThreadPoolExecutor] = None
    _lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None

    def __init__(
        self, 
        basic_params: Optional[Dict[str, Any]] = None,
        advanced_params: Optional[Dict[str, Any]] = None,
        prompt_engineering: Optional[str] = None,
        max_concurrent_requests: int = None,
        timeout: int = None
    ) -> None:
        self.basic_params = {**self.DEFAULT_BASIC_PARAMS, **(basic_params or {})}
        self.advanced_params = {**self.DEFAULT_ADVANCED_PARAMS, **(advanced_params or {})}
        self.prompt_engineering = prompt_engineering
        
        # Get configuration from centralized settings
        compute = settings.compute_config
        
        # Use settings-based concurrency if not explicitly provided
        self.max_concurrent_requests = max_concurrent_requests or compute["max_concurrent_llm"]
        self.timeout = timeout or compute["timeout_seconds"]
        self.max_retries = compute["max_retries"]
        
        # Create semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        logger.info(
            f"AsyncLlamaClient initialized: max_concurrent={self.max_concurrent_requests}, "
            f"timeout={self.timeout}s, retries={self.max_retries}"
        )
    
    @classmethod
    async def get_session(cls) -> aiohttp.ClientSession:
        """Get or create a shared aiohttp session with connection pooling."""
        if cls._session is None or cls._session.closed:
            # Configure connection pool
            connector = aiohttp.TCPConnector(
                limit=100,  # Max connections
                limit_per_host=50,  # Max connections per host
                ttl_dns_cache=300,  # DNS cache TTL
                keepalive_timeout=30,  # Keep connections alive
            )
            timeout = aiohttp.ClientTimeout(
                total=settings.compute_config["timeout_seconds"],
                connect=30,
                sock_read=120
            )
            cls._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            logger.info("Created shared aiohttp session with connection pooling")
        return cls._session
    
    @classmethod
    def get_executor(cls) -> ThreadPoolExecutor:
        """Get or create a shared thread pool executor."""
        if cls._executor is None:
            compute = settings.compute_config
            # Use half of max workers for LLM tasks to leave room for other operations
            max_workers = max(4, compute["max_workers"] // 2)
            cls._executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Created shared ThreadPoolExecutor with {max_workers} workers")
        return cls._executor
    
    @classmethod
    async def close(cls):
        """Close shared resources."""
        if cls._session and not cls._session.closed:
            await cls._session.close()
            cls._session = None
            logger.info("Closed shared aiohttp session")
        if cls._executor:
            cls._executor.shutdown(wait=False)
            cls._executor = None
            logger.info("Shut down shared ThreadPoolExecutor")
        
    async def chat_async(self, prompt: str, variables: dict) -> str:
        """Async version of chat method with retry logic and connection pooling"""
        async with self.semaphore:
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    logger.debug(f"Starting async LLM request (attempt {attempt + 1}/{self.max_retries})")
                    
                    # Combine prompt engineering if provided
                    final_prompt = prompt
                    if self.prompt_engineering and self.prompt_engineering.strip():
                        final_prompt = self.prompt_engineering.strip() + "\n" + prompt
                    
                    prompt_template = PromptTemplate(
                        template=final_prompt,
                        input_variables=list(variables.keys()),
                    )
                    
                    # Get GPU count for optimization
                    compute = settings.compute_config
                    num_gpu = compute["gpu_count"]
                    num_thread = min(compute["cpu_count"], 8)  # Cap threads
                    
                    # Optimize model parameters for faster inference
                    optimized_params = {
                        "temperature": 0.3,  # Lower temperature for faster, more deterministic responses
                        "num_ctx": 1024,     # Reduce context length for faster processing
                        "num_tokens": 256,   # Reduce max tokens for faster responses
                        "num_thread": num_thread,  # Dynamic based on CPU
                        "num_gpu": num_gpu,  # Dynamic based on GPU availability
                        "num_gqa": 8,        # Group query attention for efficiency
                        "rope_freq_base": 10000,
                        "rope_freq_scale": 0.5,
                        "repeat_penalty": 1.1,
                        "top_k": 40,
                        "top_p": 0.9,
                        "tfs_z": 1,
                        "typical_p": 1,
                        "mirostat": 0,
                        "mirostat_tau": 5,
                        "mirostat_eta": 0.1,
                        "seed": -1,
                        "num_keep": 0,
                        "stop": [],
                        "stream": False,     # Disable streaming for faster batch processing
                        "logit_bias": {},
                        "num_predict": 256,  # Limit prediction length
                        "grammar": "",
                        "temperature_last": False,
                        "cache_prompt": True, # Cache prompts for faster processing
                        "slot_id": -1,
                        "prompt_cache_all": True,
                        "prompt_cache_ro": False,
                        "image_data": [],
                        "system": "",
                        "template": "",
                        "assistant": "",
                        "raw": False,
                        "options": {
                            "num_thread": num_thread,
                            "num_gpu": num_gpu,
                            "num_gqa": 8,
                            "rope_freq_base": 10000,
                            "rope_freq_scale": 0.5,
                            "repeat_penalty": 1.1,
                            "top_k": 40,
                            "top_p": 0.9,
                            "tfs_z": 1,
                            "typical_p": 1,
                            "mirostat": 0,
                            "mirostat_tau": 5,
                            "mirostat_eta": 0.1,
                            "seed": -1,
                            "num_keep": 0,
                            "stop": [],
                            "stream": False,
                            "logit_bias": {},
                            "num_predict": 256,
                            "grammar": "",
                            "temperature_last": False,
                            "cache_prompt": True,
                            "slot_id": -1,
                            "prompt_cache_all": True,
                            "prompt_cache_ro": False,
                            "image_data": [],
                            "system": "",
                            "template": "",
                            "assistant": "",
                            "raw": False
                        }
                    }
                    
                    # Override with user-provided parameters
                    model_params = {**optimized_params, **self.basic_params, **self.advanced_params}
                    
                    llm = ChatOllama(
                        model="llama3.1:8b",
                        **model_params
                    )
                    
                    llm_chain = prompt_template | llm | StrOutputParser()
                    
                    # Run in shared thread pool to avoid blocking
                    executor = self.get_executor()
                    loop = asyncio.get_event_loop()
                    start_time = time.time()
                    answer = await loop.run_in_executor(
                        executor, 
                        lambda: llm_chain.invoke(variables)
                    )
                    processing_time = time.time() - start_time
                    
                    logger.debug(f"Async LLM request completed in {processing_time:.2f}s")
                    
                    if not answer or len(answer.strip()) == 0:
                        logger.warning("Empty response from LLM")
                        return "عذراً، لم أتمكن من توليد رد مناسب. هل يمكنك إعادة صياغة سؤالك؟"
                        
                    return answer.strip()
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"LLM request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                    if attempt < self.max_retries - 1:
                        # Exponential backoff
                        await asyncio.sleep(2 ** attempt)
            
            logger.error(f"All LLM retry attempts failed: {last_error}")
            return "عذراً، حدث خطأ في معالجة طلبك. يرجى المحاولة مرة أخرى."

    async def chat_async_fast(self, prompt: str, variables: dict) -> str:
        """Fast version using optimized parameters and smaller model for entity extraction"""
        async with self.semaphore:
            # Get fast parameters for entity extraction
            fast_params = llm_config.get_params_for_task("entity_extraction")
            # Get model priority list
            model_priority = llm_config.MODEL_PRIORITY
            last_error = None
            for model_name in model_priority:
                try:
                    logger.info(f"Trying model for entity_extraction: {model_name}")
                    llm = ChatOllama(
                        model=model_name,
                        **fast_params
                    )
                    llm_chain = PromptTemplate(
                        template=prompt,
                        input_variables=list(variables.keys()),
                    ) | llm | StrOutputParser()
                    # Run in thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    start_time = time.time()
                    answer = await loop.run_in_executor(
                        None, 
                        lambda: llm_chain.invoke(variables)
                    )
                    processing_time = time.time() - start_time
                    logger.info(f"Model {model_name} succeeded for entity_extraction in {processing_time:.2f}s")
                    if not answer or len(answer.strip()) == 0:
                        logger.warning(f"Empty response from {model_name}")
                        continue
                    return answer.strip()
                except Exception as e:
                    last_error = e
                    # Check for 404 or model not found in error message
                    error_str = str(e).lower()
                    if ("not found" in error_str or "404" in error_str or "pulling it first" in error_str):
                        logger.warning(f"Model {model_name} not available (404), trying next. Error: {e}")
                        continue
                    logger.error(f"Error using model {model_name}: {e}")
                    continue
            logger.error(f"All models failed for entity_extraction. Last error: {last_error}")
            return "[]"  # Return empty JSON array instead of error message

    async def extract_entities_async(self, preview_text: str) -> List[str]:
        """Async entity extraction using fast model"""
        prompt = llm_config.get_prompt_for_task("entity_extraction")
        
        try:
            logger.debug(f"Starting fast async entity extraction for text: {preview_text[:50]}...")
            response = await self.chat_async_fast(prompt, {"preview_text": preview_text})
            
            # Clean the response - remove any non-JSON text
            response = response.strip()
            
            # If response is "None" or contains "none", return empty list
            if not response or "none" in response.lower():
                logger.debug("No entities found in response")
                return []
            
            # Try to parse as JSON
            try:
                entities = json.loads(response)
                if isinstance(entities, list):
                    logger.debug(f"Successfully extracted {len(entities)} entities: {entities}")
                    return entities
                else:
                    logger.warning(f"Entity response is not a list: {response}")
                    return []
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse entity JSON: {response}, error: {e}")
                # Fallback: try to extract entities by splitting on commas
                try:
                    # Remove brackets and quotes, then split
                    cleaned_response = response.replace('[', '').replace(']', '').replace('"', '').replace("'", '')
                    entities = [entity.strip() for entity in cleaned_response.split(',') if entity.strip()]
                    logger.debug(f"Fallback extraction found {len(entities)} entities: {entities}")
                    return entities
                except Exception as fallback_error:
                    logger.error(f"Fallback entity extraction also failed: {fallback_error}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error in fast async entity extraction: {e}")
            return []

    async def classify_entities_async(self, entities: List[str], categories: List[str]) -> Dict[str, List[str]]:
        """Async entity classification"""
        prompt = llm_config.get_prompt_for_task("entity_classification")
        
        try:
            logger.debug(f"Starting async entity classification for {len(entities)} entities")
            response = await self.chat_async(prompt, {"entities": entities, "categories": categories})
            
            # Clean the response
            response = response.strip()
            
            if not response:
                return {}
            
            try:
                result = json.loads(response)
                logger.debug(f"Successfully classified entities: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse entity classification JSON: {response}, error: {e}")
                return {}
                
        except Exception as e:
            logger.error(f"Error in async entity classification: {e}")
            return {}

    async def process_batch_entities_async(self, texts: List[tuple]) -> List[tuple]:
        """Process entity extraction for a batch of texts with optimized concurrency"""
        logger.info(f"Starting concurrent entity extraction for {len(texts)} documents")
        start_time = time.time()
        
        # Use async 
        # cio.gather for true parallel execution
        async def extract_with_id(doc_id: str, text: str) -> tuple:
            try:
                entities = await self.extract_entities_async(text)
                return (doc_id, entities)
            except Exception as e:
                logger.error(f"Error processing entities for document {doc_id}: {e}")
                return (doc_id, [])
        
        # Create tasks for all documents
        tasks = [extract_with_id(doc_id, text) for doc_id, text in texts]
        
        # Execute all tasks concurrently with gather
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that slipped through
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in batch entity extraction: {result}")
                doc_id = texts[i][0] if i < len(texts) else f"unknown_{i}"
                final_results.append((doc_id, []))
            else:
                final_results.append(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Concurrent entity extraction completed in {processing_time:.2f}s for {len(texts)} documents")
        
        return final_results

    async def process_batch_entity_classification_async(self, entity_results: List[tuple], categories: List[str]) -> List[tuple]:
        """Process entity classification for a batch of entities with optimized concurrency"""
        logger.info(f"Starting concurrent entity classification for {len(entity_results)} documents")
        start_time = time.time()
        
        # Filter out empty entity lists
        valid_entities = [(doc_id, entities) for doc_id, entities in entity_results if entities]
        
        if not valid_entities:
            logger.info("No entities to classify, returning empty results")
            return [(doc_id, {}) for doc_id, _ in entity_results]
        
        # Use asyncio.gather for true parallel execution
        async def classify_with_id(doc_id: str, entities: List[str]) -> tuple:
            try:
                classification = await self.classify_entities_async(entities, categories)
                return (doc_id, classification)
            except Exception as e:
                logger.error(f"Error processing entity classification for document {doc_id}: {e}")
                return (doc_id, {})
        
        # Create tasks for all valid entities
        tasks = [classify_with_id(doc_id, entities) for doc_id, entities in valid_entities]
        
        # Execute all tasks concurrently with gather
        classification_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Build results dict
        results_dict = {}
        for i, result in enumerate(classification_results):
            if isinstance(result, Exception):
                logger.error(f"Exception in batch classification: {result}")
                doc_id = valid_entities[i][0] if i < len(valid_entities) else f"unknown_{i}"
                results_dict[doc_id] = {}
            else:
                results_dict[result[0]] = result[1]
        
        # Build final results list, maintaining order and adding empty results for docs without entities
        results = []
        for doc_id, _ in entity_results:
            if doc_id in results_dict:
                results.append((doc_id, results_dict[doc_id]))
            else:
                results.append((doc_id, {}))
        
        processing_time = time.time() - start_time
        logger.info(f"Concurrent entity classification completed in {processing_time:.2f}s for {len(valid_entities)} documents")
        
        return results