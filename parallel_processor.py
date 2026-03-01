import asyncio
import concurrent.futures
import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from pymongo import UpdateOne
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import time
from config.settings import settings
from performance_monitor import create_performance_monitor
# multi_gpu_opt functions replaced by settings.compute_config
from clients.llama.async_llama_client import AsyncLlamaClient
from model_registry import ModelRegistry
if TYPE_CHECKING:
    from analyzer_v1 import ArabicSocialAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ProcessingBatch:
    """Represents a batch of documents to be processed"""
    documents: List[Dict[str, Any]]
    batch_id: str
    case_id: str
    alert_config: Optional[Dict] = None

@dataclass
class ModelResult:
    """Represents the result from a single model inference"""
    document_id: str
    model_type: str  # 'classifier', 'toxic', 'emotion', 'entities', 'entity_classification'
    result: Dict[str, Any]
    confidence: float
    processing_time: float

@dataclass
class ProcessingStats:
    """Statistics for processing performance"""
    total_documents: int
    processed_documents: int
    failed_documents: int
    processing_time_seconds: float
    documents_per_second: float
    batch_count: int
    average_batch_time: float
    retry_count: int
    error_count: int

@dataclass
class AnalyzerData:
    """Pickle-safe data extracted from analyzer for multiprocessing"""
    content_categories: Dict[str, List[str]]
    model_configs: Dict[str, Any]

class ParallelProcessor:
    def __init__(self, 
                 max_workers: int = None,
                 batch_size: int = None,
                 use_multiprocessing: bool = None,
                 enable_monitoring: bool = True,
                 enable_auto_optimization: bool = True):
        """
        Initialize the parallel processor
        
        Args:
            max_workers: Maximum number of worker processes/threads
            batch_size: Number of documents to process in each batch
            use_multiprocessing: Whether to use multiprocessing (True) or threading (False)
            enable_monitoring: Whether to enable performance monitoring
            enable_auto_optimization: Whether to enable automatic optimization
        """
        # Get configuration from centralized settings
        compute = settings.compute_config
        
        self.max_workers = max_workers or compute["max_workers"]
        self.batch_size = batch_size or compute["batch_size"]
        self.gpu_batch_size = compute["gpu_batch_size"]
        self.embedding_batch_size = compute["embedding_batch_size"]
        self.use_multiprocessing = use_multiprocessing if use_multiprocessing is not None else compute["use_multiprocessing"]
        self.enable_monitoring = enable_monitoring and compute["enable_monitoring"]
        self.enable_auto_optimization = enable_auto_optimization and compute["enable_auto_optimization"]
        self.max_retries = compute["max_retries"]
        self.timeout_seconds = compute["timeout_seconds"]
        self.max_concurrent_llm = compute["max_concurrent_llm"]
        
        # Log configuration
        logger.info(f"ParallelProcessor initialized with: workers={self.max_workers}, "
                   f"batch_size={self.batch_size}, gpu_batch={self.gpu_batch_size}, "
                   f"embedding_batch={self.embedding_batch_size}")
        
        # Initialize executor based on configuration
        if self.use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
        # Performance monitoring
        self.performance_monitor = None
        self.processing_stats = ProcessingStats(
            total_documents=0,
            processed_documents=0,
            failed_documents=0,
            processing_time_seconds=0.0,
            documents_per_second=0.0,
            batch_count=0,
            average_batch_time=0.0,
            retry_count=0,
            error_count=0
        )
        
        # Initialize async LLM client for concurrent entity processing
        self.async_llama_client = None

    def _extract_analyzer_data(self, analyzer: 'ArabicSocialAnalyzer') -> AnalyzerData:
        """Extract pickle-safe data from analyzer"""
        return AnalyzerData(
            content_categories=analyzer.content_categories,
            model_configs=analyzer.model_profile
        )

    def _get_model_clients(self, analyzer: 'ArabicSocialAnalyzer'):
        """Get model clients from analyzer"""
        return {
            'classifier': analyzer.classifier_client,
            'toxic': analyzer.toxic_client,
            'emotion': analyzer.emotion_client
        }

    async def process_case_parallel(self, 
                                  documents: List[Dict[str, Any]],
                                  analyzer: 'ArabicSocialAnalyzer',
                                  collection: AsyncIOMotorCollection,
                                  alert_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process documents in parallel with optimized batch processing
        """
        start_time = time.time()
        
        # Initialize async LLM client only if explicitly enabled in model_profile
        llama_settings = analyzer.model_profile.get('llama', {})
        use_llm_entities = llama_settings.get('enable_llm_entities', False)
        if use_llm_entities:
            if self.async_llama_client is None:
                logger.info("Initializing async LLM client...")
                self.async_llama_client = ModelRegistry.get_model(
                    "async_llama",
                    basic_params=llama_settings.get('basic_params'),
                    advanced_params=llama_settings.get('advanced_params'),
                    prompt_engineering=llama_settings.get('prompt_engineering'),
                )
                logger.info(f"Async LLM client initialized successfully with max_concurrent_requests={self.async_llama_client.max_concurrent_requests}")
            else:
                logger.info(f"Async LLM client already initialized with max_concurrent_requests={self.async_llama_client.max_concurrent_requests}")
        
        # Initialize performance monitoring
        if self.enable_monitoring:
            self.performance_monitor = create_performance_monitor(analyzer.case_id)
            await self.performance_monitor.start_monitoring()
        
        try:
            # Auto-optimize configuration if enabled
            if self.enable_auto_optimization:
                await self._auto_optimize_configuration()
            
            # Create batches
            batches = self._create_batches(documents, analyzer.case_id, alert_config)
            self.processing_stats.batch_count = len(batches)
            self.processing_stats.total_documents = len(documents)
            
            logger.info(f"Created {len(batches)} batches for {len(documents)} documents")
            
            # Process batches with retry mechanism
            batch_results = await self._process_batches_with_retry(batches, analyzer)
            
            # Update database with results
            await self._bulk_update_database(collection, batch_results)
            
            # Calculate final statistics
            end_time = time.time()
            self.processing_stats.processing_time_seconds = end_time - start_time
            self.processing_stats.processed_documents = len(documents)
            self.processing_stats.documents_per_second = len(documents) / self.processing_stats.processing_time_seconds
            self.processing_stats.average_batch_time = self.processing_stats.processing_time_seconds / len(batches)
            
            # Stop performance monitoring
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()
            
            return self.get_processing_stats().__dict__
            
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
            self.processing_stats.error_count += 1
            
            # Stop performance monitoring on error
            if self.performance_monitor:
                await self.performance_monitor.stop_monitoring()
            
            raise

    async def _auto_optimize_configuration(self):
        """Auto-optimize configuration based on centralized settings.compute_config"""
        try:
            compute = settings.compute_config
            
            # ENHANCED: Lock configuration once optimized to prevent overrides
            if not hasattr(self, '_config_locked'):
                self._config_locked = False
            
            # Only apply optimization if not already locked or if significant improvement
            old_batch_size = self.batch_size
            old_max_workers = self.max_workers
            
            new_batch_size = compute["batch_size"]
            new_max_workers = compute["max_workers"]
            
            if not self._config_locked or new_batch_size > old_batch_size * 1.5:
                self.batch_size = new_batch_size
                self.max_workers = new_max_workers
                self._config_locked = True  # Lock configuration
                
                # Log optimization details
                logger.info(f"DYNAMIC OPTIMIZATION LOCKED:")
                logger.info(f"  Batch size: {old_batch_size} -> {self.batch_size} (LOCKED)")
                logger.info(f"  Max workers: {old_max_workers} -> {self.max_workers} (LOCKED)")
                logger.info(f"  GPU: {compute['gpu_name']} ({compute['gpu_memory_gb']}GB VRAM)")
                logger.info(f"  Configuration LOCKED to prevent overrides")
            else:
                logger.info(f"Configuration already locked: batch_size={self.batch_size}, max_workers={self.max_workers}")
            
            # Recreate executor if worker count changed
            if old_max_workers != self.max_workers:
                if hasattr(self, 'executor'):
                    self.executor.shutdown(wait=True)
                
                if self.use_multiprocessing:
                    self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
                else:
                    self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
                
                logger.info(f"Recreated executor with {self.max_workers} workers")
                
        except Exception as e:
            logger.warning(f"Auto-optimization failed: {e}")
            # Fallback to conservative values but don't override if already locked
            if not hasattr(self, '_config_locked') or not self._config_locked:
                self.batch_size = max(8, self.batch_size // 2)
                self.max_workers = max(2, self.max_workers // 2)
                logger.info(f"Fallback to conservative values: batch_size={self.batch_size}, max_workers={self.max_workers}")

    async def _adjust_batch_size_during_processing(self, current_memory_usage_percent: float):
        """Enhanced dynamic batch size adjustment with configuration locking"""
        try:
            # ENHANCED: Don't reduce batch size if configuration is locked
            if hasattr(self, '_config_locked') and self._config_locked:
                if current_memory_usage_percent > 95:
                    logger.warning(f"🔒 Configuration locked - memory pressure {current_memory_usage_percent:.1f}% but maintaining batch_size={self.batch_size}")
                return
            
            if current_memory_usage_percent > 90:
                # Critical memory pressure - reduce batch size immediately
                new_batch_size = max(4, self.batch_size // 4)
                if new_batch_size != self.batch_size:
                    logger.warning(f"Critical memory pressure ({current_memory_usage_percent:.1f}%), "
                                 f"reducing batch size from {self.batch_size} to {new_batch_size}")
                    self.batch_size = new_batch_size
                    
            elif current_memory_usage_percent > 80:
                # High memory pressure - reduce batch size
                new_batch_size = max(8, self.batch_size // 2)
                if new_batch_size != self.batch_size:
                    logger.info(f"High memory pressure ({current_memory_usage_percent:.1f}%), "
                               f"reducing batch size from {self.batch_size} to {new_batch_size}")
                    self.batch_size = new_batch_size
                    
            elif current_memory_usage_percent < 50:
                # Low memory usage - can increase batch size (but respect lock)
                if not hasattr(self, '_config_locked') or not self._config_locked:
                    new_batch_size = min(256, self.batch_size * 2)
                    if new_batch_size != self.batch_size:
                        logger.info(f"Low memory pressure ({current_memory_usage_percent:.1f}%), "
                                   f"increasing batch size from {self.batch_size} to {new_batch_size}")
                        self.batch_size = new_batch_size
                        
        except Exception as e:
            logger.warning(f"Dynamic batch size adjustment failed: {e}")

    async def _process_batches_with_retry(self, batches: List[ProcessingBatch], analyzer: 'ArabicSocialAnalyzer') -> List[List[Dict]]:
        """Process batches with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                return await self._process_batches_parallel(batches, analyzer)
            except Exception as e:
                self.processing_stats.retry_count += 1
                logger.warning(f"Batch processing attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    async def _process_single_batch(self, batch: ProcessingBatch, analyzer: 'ArabicSocialAnalyzer') -> List[Dict]:
        """Process a single batch of documents with enhanced error handling and consistent batch sizing"""
        logger.info(f"Processing batch {batch.batch_id} with {len(batch.documents)} documents")
        
        # ENHANCED: Validate and preprocess documents
        valid_docs = []
        for doc in batch.documents:
            if doc and isinstance(doc, dict):
                text = doc.get("Preview Text", "")
                if text and isinstance(text, str) and text.strip():
                    valid_docs.append(doc)
                else:
                    logger.debug(f"Skipping document with invalid text in batch {batch.batch_id}")
            else:
                logger.debug(f"Skipping invalid document type {type(doc)} in batch {batch.batch_id}")
        
        if not valid_docs:
            logger.warning(f"Batch {batch.batch_id} has no valid documents after validation")
            return []
        
        # ENHANCED: Build best-available text per Message Type with null safety
        texts = []
        doc_ids = []
        message_types = []
        classification_diagnostics: Dict[str, Dict[str, Any]] = {}
        for doc in valid_docs:
            mtype = str(doc.get("Message Type", "")).strip().lower()
            preview = str(doc.get("Preview Text", "") or "").strip()
            combined_text = preview
            if mtype == 'note':
                title = str(doc.get("Title", "") or doc.get("title", "") or "").strip()
                body = str(doc.get("Body", "") or doc.get("body", "") or "").strip()
                summary = str(doc.get("Summary", "") or doc.get("summary", "") or "").strip()
                parts = [p for p in [title, summary, body, preview] if p]
                combined_text = " \n".join(parts) if parts else preview
            elif mtype == 'browsing_history':
                url = str(doc.get("url", "") or doc.get("URL", "") or "").strip()
                title = str(doc.get("title", "") or doc.get("Title", "") or "").strip()
                search_value = str(doc.get("search_value", "") or "").strip()
                domain = ''
                try:
                    meta = doc.get("url_metadata") or {}
                    domain = str(meta.get("domain", "")).strip()
                except Exception:
                    domain = ''
                parts = [p for p in [title, search_value, domain, url, preview] if p]
                combined_text = " \n".join(parts) if parts else preview

            doc_id = doc.get("_id", str(len(texts)))
            if combined_text and isinstance(combined_text, str) and combined_text.strip():
                texts.append(combined_text.strip())
                doc_ids.append(str(doc_id))
                message_types.append(mtype)
            else:
                logger.debug(f"Skipping document with empty text after combine: {doc.get('_id')}")
                if mtype in ('note', 'browsing_history'):
                    classification_diagnostics[str(doc_id)] = {"type": mtype, "reason": "empty_text"}
        
        if not texts:
            logger.warning(f"Batch {batch.batch_id} has no valid texts after extraction")
            return []
        
        # ENHANCED: Create text tuples with validation
        valid_texts = list(zip(doc_ids, texts))

        # ENHANCED: Compute custom classifications for notes and browsing_history using batched calls
        custom_class_map: Dict[str, Dict[str, Any]] = {}
        try:
            note_categories = analyzer.content_categories.get("note_classification", []) or []
            browsing_categories = analyzer.content_categories.get("browsing_history_classification", []) or []

            # Prepare per-type text lists
            note_pairs = [(did, txt) for did, txt, mt in zip(doc_ids, texts, message_types) if mt == 'note']
            browsing_pairs = [(did, txt) for did, txt, mt in zip(doc_ids, texts, message_types) if mt == 'browsing_history']

            # Notes batch classification
            if note_pairs and note_categories:
                note_ids = [did for did, _ in note_pairs]
                note_texts_only = [txt for _, txt in note_pairs]
                try:
                    note_cls_batch = analyzer.classifier_client.classify(note_texts_only, note_categories)
                except Exception as e:
                    logger.warning(f"Note batch classification failed: {e}")
                    note_cls_batch = []

                # Normalize to list of dicts
                def _normalize_note(x):
                    if isinstance(x, dict) and 'labels' in x and 'scores' in x:
                        return x
                    return {"labels": [], "scores": []}
                note_cls_batch = [
                    _normalize_note(x) for x in (note_cls_batch if isinstance(note_cls_batch, list) else [note_cls_batch])
                ]
                for idx, did in enumerate(note_ids):
                    if idx < len(note_cls_batch):
                        labels = note_cls_batch[idx].get('labels') or []
                        scores = note_cls_batch[idx].get('scores') or []
                        top_label = labels[0] if labels else 'others'
                        top_score = scores[0] if scores else 0.0
                        low_conf = not (labels and scores and top_score >= 0.5)
                        custom_class_map.setdefault(did, {})['note_classification'] = {
                            'label': top_label,
                            'low_confidence': bool(low_conf),
                            'top_score': float(top_score)
                        }
                        if low_conf:
                            classification_diagnostics.setdefault(did, {"type": "note"})
                            classification_diagnostics[did].update({
                                "reason": "low_confidence" if labels and scores else "no_result",
                                "top_score": float(top_score)
                            })
            # elif note_pairs and not note_categories:
            #     for did, _ in note_pairs:
            #         custom_class_map.setdefault(did, {})['note_classification'] = {
            #             'label': 'others',
            #             'low_confidence': True,
            #             'top_score': 0.0
            #         }
            #         classification_diagnostics.setdefault(did, {"type": "note", "reason": "no_categories"})

            # Browsing batch classification
            if browsing_pairs and browsing_categories:
                browsing_ids = [did for did, _ in browsing_pairs]
                browsing_texts_only = [txt for _, txt in browsing_pairs]
                try:
                    browsing_cls_batch = analyzer.classifier_client.classify(browsing_texts_only, browsing_categories)
                except Exception as e:
                    logger.warning(f"Browsing batch classification failed: {e}")
                    browsing_cls_batch = []

                def _normalize_browse(x):
                    if isinstance(x, dict) and 'labels' in x and 'scores' in x:
                        return x
                    return {"labels": [], "scores": []}
                browsing_cls_batch = [
                    _normalize_browse(x) for x in (browsing_cls_batch if isinstance(browsing_cls_batch, list) else [browsing_cls_batch])
                ]
                for idx, did in enumerate(browsing_ids):
                    if idx < len(browsing_cls_batch):
                        labels = browsing_cls_batch[idx].get('labels') or []
                        scores = browsing_cls_batch[idx].get('scores') or []
                        top_label = labels[0] if labels else 'others'
                        top_score = scores[0] if scores else 0.0
                        low_conf = not (labels and scores and top_score >= 0.5)
                        custom_class_map.setdefault(did, {})['browsing_history_classification'] = {
                            'label': top_label,
                            'low_confidence': bool(low_conf),
                            'top_score': float(top_score)
                        }
                        if low_conf:
                            classification_diagnostics.setdefault(did, {"type": "browsing_history"})
                            classification_diagnostics[did].update({
                                "reason": "low_confidence" if labels and scores else "no_result",
                                "top_score": float(top_score)
                            })
            # elif browsing_pairs and not browsing_categories:
            #     for did, _ in browsing_pairs:
            #         custom_class_map.setdefault(did, {})['browsing_history_classification'] = {
            #             'label': 'others',
            #             'low_confidence': True,
            #             'top_score': 0.0
            #         }
            #         classification_diagnostics.setdefault(did, {"type": "browsing_history", "reason": "no_categories"})
        except Exception as e:
            logger.warning(f"Custom classification batching failed: {e}")
        
        # Process models in parallel with enhanced error handling
        model_tasks = [
            self._process_classifier_batch_safe(valid_texts, analyzer),
            self._process_toxic_batch_safe(valid_texts, analyzer),
            self._process_emotion_batch_safe(valid_texts, analyzer),
            self._process_entities_batch_ner_safe(valid_texts, analyzer)
        ]
        
        # Wait for all models to complete with enhanced error handling
        model_results = await asyncio.gather(*model_tasks, return_exceptions=True)
        
        # ENHANCED: Process results with better error handling
        valid_results = []
        for i, result in enumerate(model_results):
            if isinstance(result, Exception):
                logger.error(f"Model {i} failed with error: {result}")
                self.processing_stats.error_count += 1
                # Return empty list for failed model but continue processing
                valid_results.append([])
            else:
                valid_results.append(result)
        
        # Combine results and create final analysis with enhanced error handling
        try:
            return self._combine_model_results_enhanced(valid_docs, valid_results, batch.alert_config, custom_class_map, classification_diagnostics)
        except Exception as e:
            logger.error(f"Error combining model results for batch {batch.batch_id}: {e}")
            # Return safe fallback results
            return self._create_fallback_results(valid_docs)

    async def _process_entities_batch_ner(self, texts: List[tuple], analyzer: 'ArabicSocialAnalyzer') -> List[ModelResult]:
        """Fast entity extraction using batched NER client; falls back to async LLM if unavailable."""
        try:
            start_time = time.time()
            doc_ids = [str(doc_id) for doc_id, _ in texts]
            text_content = [text for _, text in texts]

            results: List[List[str]] = []
            labeled_results: List[List[tuple]] = []
            if getattr(analyzer, 'ner_client', None) is not None:
                # Batched NER
                # Get labels to improve downstream categorization
                labeled_results = analyzer.ner_client.analyze_batch_with_labels(text_content)
                # Also keep plain strings for backward compatibility
                results = [[ent for ent, _ in per_text] for per_text in labeled_results]
            else:
                # Fallback: async LLM entities
                entity_pairs = await self.async_llama_client.process_batch_entities_async(texts)
                # Align with doc order
                mapping = {str(doc_id): ents for doc_id, ents in entity_pairs}
                results = [mapping.get(doc_id, []) for doc_id in doc_ids]
                labeled_results = [[(ent, "MISC") for ent in ents] for ents in results]

            processing_time = time.time() - start_time

            model_results: List[ModelResult] = []
            for i, doc_id in enumerate(doc_ids):
                ents = results[i] if i < len(results) else []
                # Simple, fast categorization into the app's entitiesClasses
                categories = analyzer.content_categories.get("entitiesClasses", [])
                labeled = labeled_results[i] if i < len(labeled_results) else []
                classified = self._classify_entities_simple(labeled, categories, text_content[i])

                model_results.append(ModelResult(
                    document_id=str(doc_id),
                    model_type='entities',
                    result={'entities': ents, 'entities_classification': classified},
                    confidence=1.0 if ents else 0.0,
                    processing_time=processing_time
                ))

            if self.performance_monitor:
                self.performance_monitor.record_model_inference("entities_ner", processing_time)

            return model_results
        except Exception as e:
            logger.error(f"Error in NER entities batch processing: {e}")
            raise

    def _classify_entities_simple(self, entity_pairs: List[tuple], categories: List[str], full_text: str) -> Dict[str, List[str]]:
        """
        Map (entity_text, entity_group) pairs into app-specific entity classes without LLM.
        - entity_group typically in {PER, ORG, LOC, MISC}
        - Use regex heuristics for date/time/number/currency/hashtag
        - For classes not covered, return empty lists
        """
        import re
        classified: Dict[str, List[str]] = {c: [] for c in categories}

        # Build helper sets for quick membership
        want_person = 'person' in classified
        want_org = 'organization' in classified or 'brand' in classified
        want_brand = 'brand' in classified
        want_location = 'location' in classified or 'city' in classified or 'country' in classified
        want_date = 'date' in classified
        want_time = 'time' in classified
        want_number = 'number' in classified
        want_currency = 'currency' in classified
        want_hashtag = 'hashtag' in classified

        # Regexes
        date_re = re.compile(r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b")
        time_re = re.compile(r"\b(\d{1,2}:\d{2}(?::\d{2})?\s?(?:AM|PM|am|pm)?)\b")
        number_re = re.compile(r"\b\d+[\d,\.]*\b")
        currency_re = re.compile(r"[\$€£¥]|\b(?:USD|EUR|GBP|SAR|AED|EGP|KWD|QAR|OMR)\b", re.IGNORECASE)
        hashtag_re = re.compile(r"#\w+", re.UNICODE)

        # From entity groups (allow creating keys to be inclusive like LLM output)
        for text, group in entity_pairs:
            g = (group or '').upper()
            if g == 'PER':
                classified.setdefault('person', []).append(text)
            if g == 'ORG':
                classified.setdefault('organization', []).append(text)
            if g == 'LOC':
                classified.setdefault('location', []).append(text)

        # Heuristics from raw text
        # Expand with regex-derived categories even if not predeclared
        dates = [m.group(0) for m in date_re.finditer(full_text or '')]
        times = [m.group(0) for m in time_re.finditer(full_text or '')]
        numbers = [m.group(0) for m in number_re.finditer(full_text or '')]
        currencies = [m.group(0) for m in currency_re.finditer(full_text or '')]
        hashtags = [m.group(0) for m in hashtag_re.finditer(full_text or '')]
        if dates:
            classified.setdefault('date', []).extend(dates)
        if times:
            classified.setdefault('time', []).extend(times)
        if numbers:
            classified.setdefault('number', []).extend(numbers)
        if currencies:
            classified.setdefault('currency', []).extend(currencies)
        if hashtags:
            classified.setdefault('hashtag', []).extend(hashtags)

        # De-duplicate lists
        for k, v in classified.items():
            if isinstance(v, list):
                seen = set()
                dedup = []
                for x in v:
                    if x not in seen:
                        seen.add(x)
                        dedup.append(x)
                classified[k] = dedup

        # Remove empty classes to keep storage compact
        return {k: v for k, v in classified.items() if v}

    async def _process_classifier_batch_safe(self, texts: List[tuple], analyzer: 'ArabicSocialAnalyzer') -> List[ModelResult]:
        """Enhanced classifier batch processing with better error handling"""
        try:
            start_time = time.time()
            
            # Extract just the text content with validation
            text_content = []
            valid_indices = []
            for i, (doc_id, text) in enumerate(texts):
                if text and isinstance(text, str) and text.strip():
                    text_content.append(text)
                    valid_indices.append(i)
                else:
                    logger.debug(f"Skipping invalid text for document {doc_id}")
            
            if not text_content:
                logger.warning("No valid texts for classifier processing")
                return []
            
            # Extract pickle-safe data and model clients
            analyzer_data = self._extract_analyzer_data(analyzer)
            model_clients = self._get_model_clients(analyzer)
            
            # Run classification with enhanced error handling
            results = self._run_classifier_inference_enhanced(
                text_content,
                analyzer_data,
                model_clients
            )
            
            processing_time = time.time() - start_time
            
            # Convert to ModelResult objects with proper mapping
            model_results = []
            for i, valid_idx in enumerate(valid_indices):
                if i < len(results):
                    doc_id, _ = texts[valid_idx]
                    model_results.append(ModelResult(
                        document_id=str(doc_id),
                        model_type='classifier',
                        result=results[i],
                        confidence=results[i].get('confidence', 0.0),
                        processing_time=processing_time
                    ))
            
            if self.performance_monitor:
                self.performance_monitor.record_model_inference("classifier", processing_time)
            
            return model_results
            
        except Exception as e:
            logger.error(f"Error in enhanced classifier batch processing: {e}")
            # Return empty results instead of raising
            return []

    async def _process_toxic_batch_safe(self, texts: List[tuple], analyzer: 'ArabicSocialAnalyzer') -> List[ModelResult]:
        """Enhanced toxic batch processing with better error handling"""
        try:
            start_time = time.time()
            
            # Extract just the text content with validation
            text_content = []
            valid_indices = []
            for i, (doc_id, text) in enumerate(texts):
                if text and isinstance(text, str) and text.strip():
                    text_content.append(text)
                    valid_indices.append(i)
                else:
                    logger.debug(f"Skipping invalid text for toxic analysis: {doc_id}")
            
            if not text_content:
                logger.warning("No valid texts for toxic processing")
                return []
            
            # Extract pickle-safe data and model clients
            analyzer_data = self._extract_analyzer_data(analyzer)
            model_clients = self._get_model_clients(analyzer)
            
            # Run toxicity analysis with enhanced error handling
            results = self._run_toxic_inference_enhanced(
                text_content,
                analyzer_data,
                model_clients
            )
            
            processing_time = time.time() - start_time
            
            # Convert to ModelResult objects with proper mapping
            model_results = []
            for i, valid_idx in enumerate(valid_indices):
                if i < len(results):
                    doc_id, _ = texts[valid_idx]
                    model_results.append(ModelResult(
                        document_id=str(doc_id),
                        model_type='toxic',
                        result=results[i],
                        confidence=results[i].get('confidence', 0.0),
                        processing_time=processing_time
                    ))
            
            if self.performance_monitor:
                self.performance_monitor.record_model_inference("toxic", processing_time)
            
            return model_results
            
        except Exception as e:
            logger.error(f"Error in enhanced toxic batch processing: {e}")
            # Return empty results instead of raising
            return []

    async def _process_emotion_batch_safe(self, texts: List[tuple], analyzer: 'ArabicSocialAnalyzer') -> List[ModelResult]:
        """Enhanced emotion batch processing with better error handling"""
        try:
            start_time = time.time()
            
            # Extract just the text content with validation
            text_content = []
            valid_indices = []
            for i, (doc_id, text) in enumerate(texts):
                if text and isinstance(text, str) and text.strip():
                    text_content.append(text)
                    valid_indices.append(i)
                else:
                    logger.debug(f"Skipping invalid text for emotion analysis: {doc_id}")
            
            if not text_content:
                logger.warning("No valid texts for emotion processing")
                return []
            
            # Extract pickle-safe data and model clients
            analyzer_data = self._extract_analyzer_data(analyzer)
            model_clients = self._get_model_clients(analyzer)
            
            # Run emotion analysis with enhanced error handling
            results = self._run_emotion_inference_enhanced(
                text_content,
                analyzer_data,
                model_clients
            )
            
            processing_time = time.time() - start_time
            
            # Convert to ModelResult objects with proper mapping
            model_results = []
            for i, valid_idx in enumerate(valid_indices):
                if i < len(results):
                    doc_id, _ = texts[valid_idx]
                    model_results.append(ModelResult(
                        document_id=str(doc_id),
                        model_type='emotion',
                        result=results[i],
                        confidence=results[i].get('confidence', 0.0),
                        processing_time=processing_time
                    ))
            
            if self.performance_monitor:
                self.performance_monitor.record_model_inference("emotion", processing_time)
            
            return model_results
            
        except Exception as e:
            logger.error(f"Error in enhanced emotion batch processing: {e}")
            # Return empty results instead of raising
            return []

    async def _process_entities_batch_ner_safe(self, texts: List[tuple], analyzer: 'ArabicSocialAnalyzer') -> List[ModelResult]:
        """Fast entity extraction using batched NER client; falls back to async LLM if unavailable."""
        try:
            start_time = time.time()
            doc_ids = [str(doc_id) for doc_id, _ in texts]
            text_content = [text for _, text in texts]

            results: List[List[str]] = []
            labeled_results: List[List[tuple]] = []
            if getattr(analyzer, 'ner_client', None) is not None:
                # Batched NER
                # Get labels to improve downstream categorization
                labeled_results = analyzer.ner_client.analyze_batch_with_labels(text_content)
                # Also keep plain strings for backward compatibility
                results = [[ent for ent, _ in per_text] for per_text in labeled_results]
            else:
                # Fallback: async LLM entities
                entity_pairs = await self.async_llama_client.process_batch_entities_async(texts)
                # Align with doc order
                mapping = {str(doc_id): ents for doc_id, ents in entity_pairs}
                results = [mapping.get(doc_id, []) for doc_id in doc_ids]
                labeled_results = [[(ent, "MISC") for ent in ents] for ents in results]

            processing_time = time.time() - start_time

            model_results: List[ModelResult] = []
            for i, doc_id in enumerate(doc_ids):
                ents = results[i] if i < len(results) else []
                # Simple, fast categorization into the app's entitiesClasses
                categories = analyzer.content_categories.get("entitiesClasses", [])
                labeled = labeled_results[i] if i < len(labeled_results) else []
                classified = self._classify_entities_simple(labeled, categories, text_content[i])

                model_results.append(ModelResult(
                    document_id=str(doc_id),
                    model_type='entities',
                    result={'entities': ents, 'entities_classification': classified},
                    confidence=1.0 if ents else 0.0,
                    processing_time=processing_time
                ))

            if self.performance_monitor:
                self.performance_monitor.record_model_inference("entities_ner", processing_time)

            return model_results
        except Exception as e:
            logger.error(f"Error in NER entities batch processing: {e}")
            raise

    def _run_classifier_inference(self, texts: List[str], analyzer_data: AnalyzerData, model_clients: Dict) -> List[Dict]:
        """Run classifier inference on a batch of texts using single batched calls per category"""
        try:
            classifier_client = model_clients['classifier']
            
            # Debug: Log what we're about to classify
            logger.info(f"Classifying {len(texts)} texts with topics: {analyzer_data.content_categories['topic'][:3]}...")
            
            # Single batched call per category
            topics_batch = classifier_client.classify(
                texts,
                analyzer_data.content_categories["topic"],
            )
            
            # Debug: Log what topics_batch looks like
            logger.info(f"Topics batch type: {type(topics_batch)}, length: {len(topics_batch) if isinstance(topics_batch, list) else 'N/A'}")
            if topics_batch and len(topics_batch) > 0:
                logger.info(f"First topic result: {topics_batch[0] if isinstance(topics_batch, list) else topics_batch}")
            
            interaction_batch = classifier_client.classify(
                texts,
                analyzer_data.content_categories["interaction_type"],
            )
            sentiment_batch = classifier_client.classify(
                texts,
                analyzer_data.content_categories["sentiment_aspects"],
            )

            # Normalize outputs to consistent dicts with 'labels' and 'scores'
            def _ensure_list(x):
                return x if isinstance(x, list) else [x]

            def _normalize(d):
                if not isinstance(d, dict):
                    return {"labels": [], "scores": []}
                # Handle the actual output structure from zero-shot classifier
                if 'labels' in d and 'scores' in d:
                    return d
                # Convert single label/score shape to lists if encountered
                if 'label' in d and 'score' in d:
                    return {"labels": [d['label']], "scores": [d['score']]}
                # If it's a list of labels/scores, wrap in dict
                if isinstance(d, list):
                    return {"labels": d, "scores": [1.0] * len(d)}
                return {"labels": [], "scores": []}

            # Ensure we have lists and normalize each result
            topics_batch = [_normalize(x) for x in _ensure_list(topics_batch)]
            interaction_batch = [_normalize(x) for x in _ensure_list(interaction_batch)]
            sentiment_batch = [_normalize(x) for x in _ensure_list(sentiment_batch)]

            results: List[Dict] = []
            for i in range(len(texts)):
                topic_result = topics_batch[i] if i < len(topics_batch) else {"labels": [], "scores": []}
                interaction_result = interaction_batch[i] if i < len(interaction_batch) else {"labels": [], "scores": []}
                sentiment_result = sentiment_batch[i] if i < len(sentiment_batch) else {"labels": [], "scores": []}
                
                # Ensure we have valid scores
                topic_score = max(topic_result.get('scores', [0.0]))
                interaction_score = max(interaction_result.get('scores', [0.0]))
                sentiment_score = max(sentiment_result.get('scores', [0.0]))
                
                results.append({
                    'topics': topic_result,
                    'interaction': interaction_result,
                    'sentiment_aspects': sentiment_result,
                    'confidence': max(topic_score, interaction_score, sentiment_score)
                })

            return results
        except Exception as e:
            logger.error(f"Error in classifier inference: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Debug: Log the actual data that caused the error
            try:
                logger.error(f"Texts being classified: {texts[:2] if texts else 'None'}...")
                logger.error(f"Content categories: {analyzer_data.content_categories}")
                logger.error(f"Classifier client type: {type(classifier_client)}")
            except Exception as debug_e:
                logger.error(f"Error during debug logging: {debug_e}")
            
            raise

    def _run_toxic_inference(self, texts: List[str], analyzer_data: AnalyzerData, model_clients: Dict) -> List[Dict]:
        """Run toxicity inference on a batch of texts with one batched call"""
        try:
            toxic_client = model_clients['toxic']
            results = toxic_client.analyze_toxicity(texts)
            return results
        except Exception as e:
            logger.error(f"Error in toxic inference: {e}")
            raise

    def _run_emotion_inference(self, texts: List[str], analyzer_data: AnalyzerData, model_clients: Dict) -> List[Dict]:
        """Run emotion inference on a batch of texts with one batched call"""
        try:
            emotion_client = model_clients['emotion']
            results = emotion_client.analyze_sentiment(texts)
            return results
        except Exception as e:
            logger.error(f"Error in emotion inference: {e}")
            raise

    def _combine_model_results(self, documents: List[Dict], model_results: List, alert_config: Optional[Dict]) -> List[Dict]:
        """Combine results from all models into final analysis"""
        try:
            # Extract results by model type
            classifier_results = {r.document_id: r.result for r in model_results[0] if isinstance(r, ModelResult)}
            toxic_results = {r.document_id: r.result for r in model_results[1] if isinstance(r, ModelResult)}
            emotion_results = {r.document_id: r.result for r in model_results[2] if isinstance(r, ModelResult)}
            entities_results = {r.document_id: r.result for r in model_results[3] if isinstance(r, ModelResult)}
            
            combined_results = []
            
            for doc in documents:
                doc_id = str(doc.get('_id'))
                
                # Get results for this document
                classifier_result = classifier_results.get(doc_id, {})
                toxic_result = toxic_results.get(doc_id, {})
                emotion_result = emotion_results.get(doc_id, {})
                entities_result = entities_results.get(doc_id, {})
                
                # Create analysis summary
                analysis_summary = self._create_analysis_summary(
                    classifier_result, toxic_result, emotion_result, entities_result
                )
                
                # Check alert conditions
                alert = self._check_alert_conditions(analysis_summary, alert_config)
                
                # Create final result
                result = {
                    'document_id': doc_id,
                    'analysis_summary': analysis_summary,
                    'processed': True,
                    'alert': alert
                }
                
                combined_results.append(result)
            
            return combined_results
        except Exception as e:
            logger.error(f"Error combining model results: {e}")
            raise

    def _create_analysis_summary(self, classifier_result: Dict, toxic_result: Dict, 
                               emotion_result: Dict, entities_result: Dict) -> Dict:
        """Create analysis summary from model results"""
        try:
            # Extract topics
            topics = classifier_result.get('topics', {})
            top_topic = topics.get('labels', ['others'])[0] if topics.get('labels') else 'others'
            
            # Extract interaction type
            interaction = classifier_result.get('interaction', {})
            interaction_type = interaction.get('labels', ['others'])[0] if interaction.get('labels') else 'others'
            
            # Extract sentiment aspects
            sentiment_aspects = classifier_result.get('sentiment_aspects', {})
            sentiment_aspect = sentiment_aspects.get('labels', ['others'])[0] if sentiment_aspects.get('labels') else 'others'
            
            # Extract toxicity
            toxicity_score = toxic_result.get('score', 0) * 100 if toxic_result.get('score') else 10.0
            toxicity_label = toxic_result.get('label', 'non-toxic')
            
            # Determine risk level
            if toxicity_label == "toxic":
                risk_level = "high" if toxicity_score >= 80 else "medium" if toxicity_score >= 50 else "low"
            else:
                risk_level = "low"
            
            # Extract emotion
            emotion = emotion_result.get('label', 'neutral')
            
            # Extract entities
            entities = entities_result.get('entities', [])
            entities_classification = entities_result.get('entities_classification', {})
            
            # Filter entities_classification keys to allowed categories only, to keep UI consistent
            allowed_categories = set()
            try:
                # Try to access analyzer categories if possible (passed via closure in caller), else infer from result
                # In this context, we don't have analyzer. We'll restrict later when combining results.
                pass
            except Exception:
                pass

            return {
                "top_topic": top_topic,
                "toxicity_score": toxicity_score,
                "risk_level": risk_level,
                "sentiment_aspect": sentiment_aspect,
                "emotion": emotion,
                "language": "arabic",  # Default for now
                "interaction_type": interaction_type,
                "entities": entities,
                "entities_classification": entities_classification
            }
        except Exception as e:
            logger.error(f"Error creating analysis summary: {e}")
            return {
                "top_topic": "others",
                "toxicity_score": 0.0,
                "risk_level": "low",
                "sentiment_aspect": "others",
                "emotion": "neutral",
                "language": "arabic",
                "interaction_type": "others",
                "entities": [],
                "entities_classification": {}
            }

    def _check_alert_conditions(self, analysis_summary: Dict, alert_config: Optional[Dict]) -> bool:
        """Check if analysis summary meets alert conditions"""
        if not alert_config:
            return False
        
        try:
            keys_matched = 0
            keys_to_check = len(alert_config)
            
            for key, value in alert_config.items():
                if key == "toxicity_score":
                    if analysis_summary.get(key, 0) >= value:
                        keys_matched += 1
                elif key == "entities":
                    if any(entity in analysis_summary.get("entities", []) for entity in value):
                        keys_matched += 1
                elif analysis_summary.get(key) in value:
                    keys_matched += 1
            
            return keys_matched == keys_to_check
        except Exception as e:
            logger.error(f"Error checking alert conditions: {e}")
            return False

    async def _bulk_update_database(self, collection: AsyncIOMotorCollection, batch_results: List[List[Dict]]):
        """Bulk update database with processing results"""
        try:
            # Flatten batch results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
            
            # Prepare bulk operations using proper MongoDB format
            bulk_operations = []
            for result in all_results:
                doc_id = result.get('document_id')
                if doc_id:
                    # Clean and validate the analysis_summary before storing
                    analysis_summary = result.get('analysis_summary', {})
                    
                    # Ensure entities are properly formatted
                    if 'entities' in analysis_summary and isinstance(analysis_summary['entities'], list):
                        # Filter out error messages and invalid entities
                        analysis_summary['entities'] = [
                            entity for entity in analysis_summary['entities'] 
                            if entity and isinstance(entity, str) and 
                            not any(error_msg in entity.lower() for error_msg in [
                                'عذراً', 'error', 'خطأ', 'حدث خطأ', 'try pulling it first'
                            ])
                        ]
                    
                    # Ensure entities_classification is properly formatted
                    if 'entities_classification' in analysis_summary:
                        cleaned_classification = {}
                        for category, entities in analysis_summary['entities_classification'].items():
                            if isinstance(entities, list):
                                cleaned_entities = [
                                    entity for entity in entities 
                                    if entity and isinstance(entity, str) and 
                                    not any(error_msg in entity.lower() for error_msg in [
                                        'عذراً', 'error', 'خطأ', 'حدث خطأ', 'try pulling it first'
                                    ])
                                ]
                                if cleaned_entities:
                                    cleaned_classification[category] = cleaned_entities
                        analysis_summary['entities_classification'] = cleaned_classification
                    
                    # Use proper MongoDB UpdateOne operation
                    bulk_operations.append(
                        UpdateOne(
                            {'_id': ObjectId(doc_id)},
                            {
                                '$set': {
                                    'analysis_summary': analysis_summary,
                                    'processed': result.get('processed', True),
                                    'alert': result.get('alert', False)
                                }
                            }
                        )
                    )
            
            if bulk_operations:
                # Execute bulk operation
                result = await collection.bulk_write(bulk_operations)
                logger.info(f"Bulk update completed: {result.modified_count} documents updated")
            
        except Exception as e:
            logger.error(f"Error in bulk database update: {e}")
            # Fallback to individual updates
            await self._fallback_individual_updates(collection, batch_results)

    async def _fallback_individual_updates(self, collection: AsyncIOMotorCollection, batch_results: List[List[Dict]]):
        """Fallback to individual document updates if bulk update fails"""
        try:
            logger.info("Falling back to individual document updates")
            
            # Flatten batch results
            all_results = []
            for batch in batch_results:
                all_results.extend(batch)
            
            # Update documents individually
            for result in all_results:
                doc_id = result.get('document_id')
                if doc_id:
                    try:
                        # Clean and validate the analysis_summary before storing
                        analysis_summary = result.get('analysis_summary', {})
                        
                        # Ensure entities are properly formatted
                        if 'entities' in analysis_summary and isinstance(analysis_summary['entities'], list):
                            # Filter out error messages and invalid entities
                            analysis_summary['entities'] = [
                                entity for entity in analysis_summary['entities'] 
                                if entity and isinstance(entity, str) and 
                                not any(error_msg in entity.lower() for error_msg in [
                                    'عذراً', 'error', 'خطأ', 'حدث خطأ', 'try pulling it first'
                                ])
                            ]
                        
                        # Ensure entities_classification is properly formatted
                        if 'entities_classification' in analysis_summary:
                            cleaned_classification = {}
                            for category, entities in analysis_summary['entities_classification'].items():
                                if isinstance(entities, list):
                                    cleaned_entities = [
                                        entity for entity in entities 
                                        if entity and isinstance(entity, str) and 
                                        not any(error_msg in entity.lower() for error_msg in [
                                            'عذراً', 'error', 'خطأ', 'حدث خطأ', 'try pulling it first'
                                        ])
                                    ]
                                    if cleaned_entities:
                                        cleaned_classification[category] = cleaned_entities
                            analysis_summary['entities_classification'] = cleaned_classification
                        
                        await collection.update_one(
                            {'_id': ObjectId(doc_id)},
                            {
                                '$set': {
                                    'analysis_summary': analysis_summary,
                                    'processed': result.get('processed', True),
                                    'alert': result.get('alert', False)
                                }
                            }
                        )
                    except Exception as e:
                        logger.error(f"Error updating document {doc_id}: {e}")
                        self.processing_stats.error_count += 1
            
        except Exception as e:
            logger.error(f"Error in fallback individual updates: {e}")

    def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics"""
        return self.processing_stats

    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 

    def _run_classifier_inference_enhanced(self, texts: List[str], analyzer_data: AnalyzerData, model_clients: Dict) -> List[Dict]:
        """Enhanced classifier inference with better error handling and tensor safety"""
        try:
            classifier_client = model_clients['classifier']
            
            # Debug: Log what we're about to classify
            logger.info(f"Classifying {len(texts)} texts with topics: {analyzer_data.content_categories['topic'][:3]}...")
            
            # ENHANCED: Validate texts before classification
            valid_texts = []
            for text in texts:
                if text and isinstance(text, str) and text.strip():
                    # Truncate to safe length
                    safe_text = text[:512] if len(text) > 512 else text
                    valid_texts.append(safe_text)
                else:
                    valid_texts.append("empty_text")
            
            # Single batched call per category with enhanced error handling
            try:
                topics_batch = classifier_client.classify(
                    valid_texts,
                    analyzer_data.content_categories["topic"],
                )
            except Exception as e:
                logger.error(f"Topics classification failed: {e}")
                topics_batch = [{"labels": ["others"], "scores": [0.0]} for _ in valid_texts]
            
            try:
                interaction_batch = classifier_client.classify(
                    valid_texts,
                    analyzer_data.content_categories["interaction_type"],
                )
            except Exception as e:
                logger.error(f"Interaction classification failed: {e}")
                interaction_batch = [{"labels": ["others"], "scores": [0.0]} for _ in valid_texts]
            
            try:
                sentiment_batch = classifier_client.classify(
                    valid_texts,
                    analyzer_data.content_categories["sentiment_aspects"],
                )
            except Exception as e:
                logger.error(f"Sentiment classification failed: {e}")
                sentiment_batch = [{"labels": ["others"], "scores": [0.0]} for _ in valid_texts]

            # Normalize outputs to consistent dicts with 'labels' and 'scores'
            def _ensure_list(x):
                return x if isinstance(x, list) else [x]

            def _normalize(d):
                if not isinstance(d, dict):
                    return {"labels": [], "scores": []}
                # Handle the actual output structure from zero-shot classifier
                if 'labels' in d and 'scores' in d:
                    return d
                # Convert single label/score shape to lists if encountered
                if 'label' in d and 'score' in d:
                    return {"labels": [d['label']], "scores": [d['score']]}
                # If it's a list of labels/scores, wrap in dict
                if isinstance(d, list):
                    return {"labels": d, "scores": [1.0] * len(d)}
                return {"labels": [], "scores": []}

            # Ensure we have lists and normalize each result
            topics_batch = [_normalize(x) for x in _ensure_list(topics_batch)]
            interaction_batch = [_normalize(x) for x in _ensure_list(interaction_batch)]
            sentiment_batch = [_normalize(x) for x in _ensure_list(sentiment_batch)]

            results: List[Dict] = []
            for i in range(len(valid_texts)):
                topic_result = topics_batch[i] if i < len(topics_batch) else {"labels": [], "scores": []}
                interaction_result = interaction_batch[i] if i < len(interaction_batch) else {"labels": [], "scores": []}
                sentiment_result = sentiment_batch[i] if i < len(sentiment_batch) else {"labels": [], "scores": []}
                
                # Ensure we have valid scores
                topic_score = max(topic_result.get('scores', [0.0]))
                interaction_score = max(interaction_result.get('scores', [0.0]))
                sentiment_score = max(sentiment_result.get('scores', [0.0]))
                
                results.append({
                    'topics': topic_result,
                    'interaction': interaction_result,
                    'sentiment_aspects': sentiment_result,
                    'confidence': max(topic_score, interaction_score, sentiment_score)
                })

            return results
        except Exception as e:
            logger.error(f"Error in enhanced classifier inference: {e}")
            # Return safe fallback results
            return [{
                'topics': {"labels": ["others"], "scores": [0.0]},
                'interaction': {"labels": ["others"], "scores": [0.0]},
                'sentiment_aspects': {"labels": ["others"], "scores": [0.0]},
                'confidence': 0.0
            } for _ in texts]

    def _run_toxic_inference_enhanced(self, texts: List[str], analyzer_data: AnalyzerData, model_clients: Dict) -> List[Dict]:
        """Enhanced toxic inference with better error handling"""
        try:
            toxic_client = model_clients['toxic']
            # ENHANCED: Use the new safe toxic analysis method
            results = toxic_client.analyze_toxicity(texts)
            return results
        except Exception as e:
            logger.error(f"Error in enhanced toxic inference: {e}")
            # Return safe fallback results
            return [{'label': 'non-toxic', 'score': 0.0, 'confidence': 0.0} for _ in texts]

    def _run_emotion_inference_enhanced(self, texts: List[str], analyzer_data: AnalyzerData, model_clients: Dict) -> List[Dict]:
        """Enhanced emotion inference with better error handling"""
        try:
            emotion_client = model_clients['emotion']
            # ENHANCED: Use the new safe emotion analysis method
            results = emotion_client.analyze_sentiment(texts)
            return results
        except Exception as e:
            logger.error(f"Error in enhanced emotion inference: {e}")
            # Return safe fallback results
            return [{'label': 'neutral', 'score': 0.0, 'confidence': 0.0} for _ in texts]

    def _combine_model_results_enhanced(self, documents: List[Dict], model_results: List, alert_config: Optional[Dict], custom_class_map: Optional[Dict[str, Dict[str, str]]] = None, classification_diagnostics: Optional[Dict[str, Dict[str, Any]]] = None) -> List[Dict]:
        """Enhanced model results combination with better error handling"""
        try:
            # Extract results by model type with enhanced error handling
            classifier_results = {}
            toxic_results = {}
            emotion_results = {}
            entities_results = {}
            
            if len(model_results) >= 1 and isinstance(model_results[0], list):
                classifier_results = {r.document_id: r.result for r in model_results[0] if isinstance(r, ModelResult)}
            if len(model_results) >= 2 and isinstance(model_results[1], list):
                toxic_results = {r.document_id: r.result for r in model_results[1] if isinstance(r, ModelResult)}
            if len(model_results) >= 3 and isinstance(model_results[2], list):
                emotion_results = {r.document_id: r.result for r in model_results[2] if isinstance(r, ModelResult)}
            if len(model_results) >= 4 and isinstance(model_results[3], list):
                entities_results = {r.document_id: r.result for r in model_results[3] if isinstance(r, ModelResult)}
            
            combined_results = []
            total_target = 0
            classified_count = 0
            reasons: Dict[str, int] = {}
            
            for doc in documents:
                doc_id = str(doc.get('_id'))
                
                # Get results for this document with safe defaults
                classifier_result = classifier_results.get(doc_id, {})
                toxic_result = toxic_results.get(doc_id, {})
                emotion_result = emotion_results.get(doc_id, {})
                entities_result = entities_results.get(doc_id, {})
                
                # Create analysis summary with enhanced error handling
                analysis_summary = self._create_analysis_summary_enhanced(
                    classifier_result, toxic_result, emotion_result, entities_result
                )

                # Inject custom classifications for notes and browsing_history if available
                try:
                    if custom_class_map and doc_id in custom_class_map:
                        mtype = str(doc.get("Message Type", "")).strip().lower()
                        if mtype == 'note' and 'note_classification' in custom_class_map[doc_id]:
                            nl = custom_class_map[doc_id]['note_classification']
                            # Always classify; store label and flags
                            analysis_summary['note_classification'] = nl['label'] if isinstance(nl, dict) else nl
                            if isinstance(nl, dict):
                                analysis_summary['note_classification_low_confidence'] = nl.get('low_confidence', False)
                                analysis_summary['note_classification_top_score'] = nl.get('top_score', 0.0)
                            classified_count += 1
                        if mtype == 'browsing_history' and 'browsing_history_classification' in custom_class_map[doc_id]:
                            bl = custom_class_map[doc_id]['browsing_history_classification']
                            analysis_summary['browsing_history_classification'] = bl['label'] if isinstance(bl, dict) else bl
                            if isinstance(bl, dict):
                                analysis_summary['browsing_history_classification_low_confidence'] = bl.get('low_confidence', False)
                                analysis_summary['browsing_history_classification_top_score'] = bl.get('top_score', 0.0)
                            classified_count += 1
                except Exception as e:
                    logger.warning(f"Failed to inject custom classifications: {e}")
                
                # Diagnostics tracking
                mtype_here = str(doc.get("Message Type", "")).strip().lower()
                if mtype_here in ('note', 'browsing_history'):
                    total_target += 1
                    missing_note = mtype_here == 'note' and 'note_classification' not in analysis_summary
                    missing_browse = mtype_here == 'browsing_history' and 'browsing_history_classification' not in analysis_summary
                    if (missing_note or missing_browse) and classification_diagnostics and doc_id in classification_diagnostics:
                        reason = classification_diagnostics[doc_id].get('reason', 'unknown')
                        reasons[reason] = reasons.get(reason, 0) + 1

                # Check alert conditions
                alert = self._check_alert_conditions(analysis_summary, alert_config)
                
                # Create final result
                result = {
                    'document_id': doc_id,
                    'analysis_summary': analysis_summary,
                    'processed': True,
                    'alert': alert
                }
                
                combined_results.append(result)
            
            # Emit coverage summary for target types
            if total_target > 0:
                coverage = (classified_count / total_target) * 100.0
                logger.info(f"Note/Browsing classification coverage: {coverage:.1f}% ({classified_count}/{total_target})")
                if reasons:
                    logger.info(f"Unclassified reasons: {reasons}")
            
            return combined_results
        except Exception as e:
            logger.error(f"Error in enhanced model results combination: {e}")
            # Return safe fallback results
            return self._create_fallback_results(documents)

    def _create_analysis_summary_enhanced(self, classifier_result: Dict, toxic_result: Dict, 
                                       emotion_result: Dict, entities_result: Dict) -> Dict:
        """Enhanced analysis summary creation with better error handling"""
        try:
            # Extract topics with safe defaults
            topics = classifier_result.get('topics', {})
            top_topic = topics.get('labels', ['others'])[0] if topics.get('labels') else 'others'
            
            # Extract interaction type with safe defaults
            interaction = classifier_result.get('interaction', {})
            interaction_type = interaction.get('labels', ['others'])[0] if interaction.get('labels') else 'others'
            
            # Extract sentiment aspects with safe defaults
            sentiment_aspects = classifier_result.get('sentiment_aspects', {})
            sentiment_aspect = sentiment_aspects.get('labels', ['others'])[0] if sentiment_aspects.get('labels') else 'others'
            
            # Extract toxicity with safe defaults
            toxicity_score = toxic_result.get('score', 0) * 100 if toxic_result.get('score') else 0.0
            toxicity_label = toxic_result.get('label', 'non-toxic')
            
            # Determine risk level
            if toxicity_label == "toxic":
                risk_level = "high" if toxicity_score >= 80 else "medium" if toxicity_score >= 50 else "low"
            else:
                risk_level = "low"
            
            # Extract emotion with safe defaults
            emotion = emotion_result.get('label', 'neutral')
            
            # Extract entities with safe defaults
            entities = entities_result.get('entities', [])
            entities_classification = entities_result.get('entities_classification', {})
            
            return {
                "top_topic": top_topic,
                "toxicity_score": toxicity_score,
                "risk_level": risk_level,
                "sentiment_aspect": sentiment_aspect,
                "emotion": emotion,
                "language": "arabic",  # Default for now
                "interaction_type": interaction_type,
                "entities": entities,
                "entities_classification": entities_classification
            }
        except Exception as e:
            logger.error(f"Error creating enhanced analysis summary: {e}")
            return {
                "top_topic": "others",
                "toxicity_score": 10.0,
                "risk_level": "low",
                "sentiment_aspect": "others",
                "emotion": "neutral",
                "language": "arabic",
                "interaction_type": "others",
                "entities": [],
                "entities_classification": {}
            } 

    def _create_batches(self, documents: List[Dict], case_id: str, alert_config: Optional[Dict]) -> List[ProcessingBatch]:
        """Create batches with enhanced error handling, null checks, and balanced sizing"""
        if not documents:
            return []
        
        # ENHANCED: Validate all documents first and collect valid ones
        valid_docs = []
        for doc in documents:
            if doc and isinstance(doc, dict):
                # Check for required fields and valid text
                text = doc.get("Preview Text", "")
                if text and isinstance(text, str) and text.strip():
                    valid_docs.append(doc)
                else:
                    logger.debug(f"Skipping document with invalid text: {doc.get('_id', 'unknown')}")
            else:
                logger.debug(f"Skipping invalid document: {type(doc)}")
        
        if not valid_docs:
            logger.warning("No valid documents found for processing")
            return []
        
        # ENHANCED: Create balanced batches to avoid tiny last batch
        total_docs = len(valid_docs)
        if total_docs <= self.batch_size:
            # Single batch case
            batches = [ProcessingBatch(
                documents=valid_docs,
                batch_id=f"{case_id}_batch_0",
                case_id=case_id,
                alert_config=alert_config
            )]
        else:
            # Calculate optimal batch distribution
            num_batches = max(2, (total_docs + self.batch_size - 1) // self.batch_size)
            docs_per_batch = total_docs // num_batches
            remainder = total_docs % num_batches
            
            batches = []
            start_idx = 0
            for i in range(num_batches):
                # Distribute remainder evenly across first few batches
                current_batch_size = docs_per_batch + (1 if i < remainder else 0)
                end_idx = start_idx + current_batch_size
                batch_docs = valid_docs[start_idx:end_idx]
                
                batch = ProcessingBatch(
                    documents=batch_docs,
                    batch_id=f"{case_id}_batch_{i}",
                    case_id=case_id,
                    alert_config=alert_config
                )
                batches.append(batch)
                start_idx = end_idx
        
        logger.info(f"Created {len(batches)} balanced batches from {len(documents)} documents")
        for i, batch in enumerate(batches):
            logger.info(f"  Batch {i}: {len(batch.documents)} documents")
        
        return batches

    async def _process_batches_parallel(self, batches: List[ProcessingBatch], analyzer: 'ArabicSocialAnalyzer') -> List[List[Dict]]:
        """Process multiple batches in parallel with enhanced error handling"""
        # Create tasks for each batch
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_single_batch(batch, analyzer))
            tasks.append(task)
        
        try:
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # ENHANCED: Process results with better error handling
            valid_results = []
            for i, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch {i} failed with error: {result}")
                    self.processing_stats.error_count += 1
                    # Return empty list for failed batch but continue processing
                    valid_results.append([])
                else:
                    valid_results.append(result)
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Error in parallel batch processing: {e}")
            raise

    async def _monitor_memory_during_processing(self):
        """Monitor memory usage during processing with enhanced configuration locking"""
        try:
            while True:
                # Check memory every 10 seconds
                await asyncio.sleep(10)
                
                # Get current memory usage from psutil directly
                try:
                    import psutil
                    memory_percent = psutil.virtual_memory().percent
                except ImportError:
                    memory_percent = 50
                
                # ENHANCED: Only adjust if configuration is not locked
                if not hasattr(self, '_config_locked') or not self._config_locked:
                    await self._adjust_batch_size_during_processing(memory_percent)
                
                # Log memory status
                if memory_percent > 80:
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
                elif memory_percent < 30:
                    logger.info(f"Low memory usage: {memory_percent:.1f}%")
                    
        except asyncio.CancelledError:
            logger.debug("Memory monitoring cancelled")
        except Exception as e:
            logger.warning(f"Memory monitoring error: {e}")

    async def _process_single_batch_with_retry(self, batch: ProcessingBatch, analyzer: 'ArabicSocialAnalyzer') -> List[Dict]:
        """Process single batch with retry mechanism and enhanced error handling"""
        for attempt in range(self.max_retries):
            try:
                return await self._process_single_batch(batch, analyzer)
            except Exception as e:
                logger.warning(f"Batch {batch.batch_id} attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Last attempt failed, return fallback results
                    logger.error(f"Batch {batch.batch_id} failed after {self.max_retries} attempts, using fallback")
                    return self._create_fallback_results(batch.documents)
                await asyncio.sleep(2 ** attempt)