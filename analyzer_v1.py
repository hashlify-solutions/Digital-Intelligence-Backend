import datetime
import re
import torch
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from setup import setup_logging
from bson import ObjectId
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from motor.motor_asyncio import AsyncIOMotorCollection
from config.db import alerts_collection, with_db_retry
import asyncio
from utils.translationmap import (
    sentiment_translation_map,
    interaction_translation_map,
    topic_translation_map,
    sentiments_default,
    topics_default,
    interactions_default,
    topics_arabic,
    sentiments_arabic,
    interactions_arabic,
    entitiesClasses_arabic,
    entitiesClasses_default,
)
import numpy as np
from clients.llama.llama_v1 import LlamaClient

import json
from typing import Dict, Any, List, Optional
from parallel_processor import ParallelProcessor
from config.settings import settings
from pymongo import UpdateOne
from model_registry import ModelRegistry

session = requests.Session()
retry = Retry(connect=3, backoff_factor=3)
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)

logger = setup_logging()


# Arabic Social Analyzer
class ArabicSocialAnalyzer:
    def __init__(
        self,
        mongo_collection_case: AsyncIOMotorCollection,
        mongo_collection__all_cases: AsyncIOMotorCollection,
        case_id: str,
        alert_id: str = None,
        topics: list = None,
        sentiments: list = None,
        interactions: list = None,
        entitiesClasses: list = None,
        model_profile: Dict = None,
        use_parallel_processing: bool = False,
        note_classifications: list = None,
        browsing_history_classifications: list = None,
        is_llama_validation_enabled: bool = False,
    ):
        logger.info("Initializing ArabicSocialAnalyzer...")
        classifier_obj = model_profile.get("classifier", {})
        toxic_obj = model_profile.get("toxic", {})
        emotion_obj = model_profile.get("emotion", {})
        # embeddings_obj = model_profile.get("embeddings", {})  # If needed elsewhere
        # llama_obj = model_profile.get("llama", {})
        # Validate model profile
        if not all([classifier_obj, toxic_obj, emotion_obj]):
            raise ValueError("Missing required model configurations")

        self.collection_case = mongo_collection_case
        self.collection__all_cases = mongo_collection__all_cases
        self.case_id = case_id
        self.alert_id = alert_id
        self.model_profile = model_profile
        self.use_parallel_processing = use_parallel_processing
        self.retrying_time = 10
        self.headers = {"Content-Type": "application/json"}
        self.device = 0 if torch.cuda.is_available() else -1
        self.note_classifications = note_classifications if note_classifications else []
        self.browsing_history_classifications = (
            browsing_history_classifications if browsing_history_classifications else []
        )
        self.is_llama_validation_enabled = is_llama_validation_enabled
        # Initialize content categories with validation
        # Use English labels for classifier (model expects English), keep Arabic for display
        self.content_categories = {
            "topic": (
                topics if topics and len(topics) > 0 else topics_default
            ),  # Use English topics
            "interaction_type": (
                interactions
                if interactions and len(interactions) > 0
                else interactions_default
            ),  # Use English interactions
            "sentiment_aspects": (
                sentiments if sentiments and len(sentiments) > 0 else sentiments_default
            ),  # Use English sentiments
            # Use broad English defaults for LLM-friendly classification if none provided
            "entitiesClasses": (
                entitiesClasses
                if entitiesClasses and len(entitiesClasses) > 0
                else entitiesClasses_default
            ),
            # Optional targeted classifications (must be present here for classifier usage in all pipelines)
            "note_classification": note_classifications if note_classifications else [],
            "browsing_history_classification": (
                browsing_history_classifications
                if browsing_history_classifications
                else []
            ),
        }

        # Store Arabic versions for display purposes
        self.arabic_categories = {
            "topic": topics if topics and len(topics) > 0 else topics_arabic,
            "interaction_type": (
                interactions
                if interactions and len(interactions) > 0
                else interactions_arabic
            ),
            "sentiment_aspects": (
                sentiments if sentiments and len(sentiments) > 0 else sentiments_arabic
            ),
            "entitiesClasses": (
                entitiesClasses
                if entitiesClasses and len(entitiesClasses) > 0
                else entitiesClasses_arabic
            ),
            # New optional classification categories
            "note_classification": note_classifications if note_classifications else [],
            "browsing_history_classification": (
                browsing_history_classifications
                if browsing_history_classifications
                else []
            ),
        }

        # Initialize model clients via ModelRegistry (cached singleton, no re-loading)
        try:
            self.classifier_client = ModelRegistry.get_model(
                "classifier", model_name=classifier_obj.get("name")
            )
            self.toxic_client = ModelRegistry.get_model(
                "toxic", model_name=toxic_obj.get("name")
            )
            self.emotion_client = ModelRegistry.get_model(
                "emotion", model_name=emotion_obj.get("name")
            )
            logger.info("Model clients loaded via ModelRegistry (cached)")
        except Exception as e:
            logger.error(f"Error initializing model clients: {e}")
            raise

        # self.llama_basic_params = llama_obj.get('basic_params', {})
        # self.llama_advanced_params = llama_obj.get('advanced_params', {})

        # Initialize parallel processor if enabled
        if self.use_parallel_processing:
            # Get configuration from centralized settings
            compute = settings.compute_config
            self.parallel_processor = ParallelProcessor(
                max_workers=compute["max_workers"],
                batch_size=compute["batch_size"],
                use_multiprocessing=compute["use_multiprocessing"],
            )
            logger.info(
                f"Parallel processor initialized: workers={compute['max_workers']}, "
                f"batch_size={compute['batch_size']}"
            )
        else:
            self.parallel_processor = None

        # Initialize fast NER client to replace slow LLM entities by default
        # try:
        #     self.ner_client = ArabicNERClient()
        #     logger.info("ArabicNERClient initialized for fast entity extraction")
        # except Exception as e:
        #     logger.error(f"Failed to initialize ArabicNERClient: {e}")
        #     self.ner_client = None

    def detect_language_details(self, text):
        """Detect specific language features."""
        has_arabic = bool(re.search(r"[\u0600-\u06FF]", text))
        has_english = bool(re.search(r"[a-zA-Z]", text))
        has_emojis = bool(re.search(r"[\U0001F300-\U0001F9FF]", text))

        return {
            "primary_language": "arabic" if has_arabic else "english",
            "is_multilingual": has_arabic and has_english,
            "has_emojis": has_emojis,
        }

    def preprocess_text(self, text):
        """Normalize and clean text content."""
        # Remove URLs and special characters
        text = re.sub(r"http\S+|www\.\S+", "", text)
        # Normalize Arabic characters
        text = re.sub(r"[إأآا]", "ا", text)
        # Normalize Hebrew characters
        text = re.sub(r"ة", "ه", text)
        # Remove diacritics
        text = re.sub(r"[\u064B-\u0652]", "", text)
        # Remove repeated characters
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        # Remove special characters and normalize spaces
        text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @with_db_retry(max_retries=5, delay=2)
    async def _safe_update_case_status(self, status_update: Dict[str, Any]):
        await self.collection__all_cases.find_one_and_update(
            {"_id": ObjectId(self.case_id)}, {"$set": status_update}
        )

    @with_db_retry(max_retries=5, delay=2)
    async def _safe_fetch_alert(self):
        return await alerts_collection.find_one({"_id": ObjectId(self.alert_id)})

    @with_db_retry(max_retries=5, delay=2)
    async def _safe_find_unprocessed_documents(self):
        return await self.collection_case.find({"processed": {"$ne": True}}).to_list(
            length=None
        )

    @with_db_retry(max_retries=5, delay=2)
    async def _safe_count_documents(self, query: Dict[str, Any]):
        return await self.collection_case.count_documents(query)

    async def validate_analysis_summary_via_llama(
        self, text: str, analysis_summary: Dict[str, Any]
    ):

        llama_client = LlamaClient()
        top_topic_validation_score = llama_client.validate_classification_result(
            text, self.content_categories["topic"], analysis_summary["top_topic"]
        )
        sentiment_aspect_validation_score = llama_client.validate_classification_result(
            text,
            self.content_categories["sentiment_aspects"],
            analysis_summary["sentiment_aspect"],
        )
        interaction_type_validation_score = llama_client.validate_classification_result(
            text,
            self.content_categories["interaction_type"],
            analysis_summary["interaction_type"],
        )
        toxicity_score_validation_score = llama_client.validate_toxicity_result(
            text, analysis_summary["toxicity_score"]
        )
        emotion_validation_score = llama_client.validate_emotion_result(
            text, 
            ["positive", "negative", "neutral"],
            analysis_summary["emotion"]
        )

        return {
            "top_topic_validation_score": top_topic_validation_score,
            "sentiment_aspect_validation_score": sentiment_aspect_validation_score,
            "interaction_type_validation_score": interaction_type_validation_score,
            "toxicity_score_validation_score": toxicity_score_validation_score,
            "emotion_validation_score": emotion_validation_score,
        }

    async def extract_entities(self, preview_text: str):
        prompt = """Extract entities (names, locations, organizations, and other key information) from the following text:
        
        {preview_text}

        Return your response as a JSON string in exactly this format:
        [
            'entity1',
            'entity2',
            'entity3',
            ...
        ]
        Return ONLY a valid JSON array with no additional text, comments, or explanations.
        If no entities are found, return "None".       
        """

        try:
            client = LlamaClient(
                prompt=prompt,
                variables={"preview_text": preview_text},
                prompt_engineering=self.model_profile.get("llama", {}).get(
                    "prompt_engineering", ""
                ),
            )
            response = client.chat()

            # Clean the response - remove any non-JSON text
            response = response.strip()

            # If response is "None" or contains "none", return empty list
            if not response or "none" in response.lower():
                return []

            # Try to parse as JSON
            try:
                entities = json.loads(response)
                if isinstance(entities, list):
                    return entities
                else:
                    logger.warning(f"Entity response is not a list: {response}")
                    return []
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse entity JSON: {response}, error: {e}")
                # Fallback: try to extract entities by splitting on commas
                try:
                    # Remove brackets and quotes, then split
                    cleaned_response = (
                        response.replace("[", "")
                        .replace("]", "")
                        .replace('"', "")
                        .replace("'", "")
                    )
                    entities = [
                        entity.strip()
                        for entity in cleaned_response.split(",")
                        if entity.strip()
                    ]
                    return entities
                except Exception as fallback_error:
                    logger.error(
                        f"Fallback entity extraction also failed: {fallback_error}"
                    )
                    return []

        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            return []

    async def classify_entities(self, entities, categories):
        prompt = """Classify the given entities into one of the following categories:

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

            Return ONLY a valid JSON array with no additional text, comments, or explanations.
            """

        try:
            client = LlamaClient(
                prompt=prompt,
                variables={"entities": entities, "categories": categories},
                prompt_engineering=self.model_profile.get("llama", {}).get(
                    "prompt_engineering", ""
                ),
            )
            response = client.chat()

            # Clean the response
            response = response.strip()

            if not response:
                return {}

            try:
                parsed = json.loads(response)
                # Accept whatever categories the model returns (no limiting)
                return parsed
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse entity classification JSON: {response}, error: {e}"
                )
                return {}

        except Exception as e:
            logger.error(f"Error in entity classification: {e}")
            return {}

    def emotion_classifer(self, text, url):
        max_retries = 3
        payload = {
            "inputs": text,
        }
        logger.debug("Starting core classification")
        for attempt in range(max_retries):
            try:
                response = session.post(
                    url, headers=self.headers, json=payload, timeout=60
                )
                if response.status_code == 200:
                    classification_result = response.json()
                    return classification_result[0]
                elif response.status_code == 503:
                    logger.warning(
                        f"Model is loading. Retrying in {self.retrying_time} seconds... (Attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(self.retrying_time * 2)
                else:
                    logger.error(f"Failed to generate response: {response.text}")
                    break
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                break

    def process_analyze_content_response(self, analysis: Dict[str, Any]):
        top_topic = (
            analysis["content_analysis"]["topics"][0]["category"]
            if analysis["content_analysis"]["topics"]
            and analysis["content_analysis"]["topics"][0]["confidence"] >= 65
            else "others"
        )

        sentiment_aspect = (
            analysis["content_analysis"]["sentiment_aspects"][0]["aspect"]
            if analysis["content_analysis"]["sentiment_aspects"]
            and analysis["content_analysis"]["sentiment_aspects"][0]["strength"] >= 65
            else "others"
        )

        interaction_type = (
            analysis["content_analysis"]["interaction"][0]["type"]
            if analysis["content_analysis"]["interaction"]
            and analysis["content_analysis"]["interaction"][0]["confidence"] >= 65
            else "others"
        )

        toxicity_label = analysis["toxicity"]["toxicity_label"]
        toxicity_score = (
            analysis["toxicity"]["toxicity_score"]
            if toxicity_label == "toxic"
            else 10.00
        )

        if toxicity_label == "toxic":
            risk_level = (
                "high"
                if toxicity_score >= 80
                else "medium" if toxicity_score >= 50 else "low"
            )
        else:
            risk_level = "low"

        emotion = analysis["sentiment_metrics"]["emotion"]

        entities = analysis["entity"] if analysis["entity"] else None

        entities_classification = analysis["entities_classification"]

        language = (
            analysis["text_metadata"]["language_info"]["primary_language"]
            if analysis["text_metadata"]["language_info"]
            else "both" if analysis["text_metadata"]["is_multilingual"] else None
        )

        analysis_summary = {
            "top_topic": top_topic,
            "sentiment_aspect": sentiment_aspect,
            "interaction_type": interaction_type,
            "toxicity_score": toxicity_score,
            "emotion": emotion,
            "risk_level": risk_level,
            "language": language,
            "entities": entities,
            "entities_classification": entities_classification,
        }

        return analysis_summary

    async def analyze_content(self, text: str):
        """Analyze text content using various machine learning models."""
        try:
            if not text or not isinstance(text, str):
                raise ValueError("Invalid input text")

            logger.debug(f"Starting analysis for text: {text[:100]}...")
            cleaned_text = self.preprocess_text(text)

            if not cleaned_text:
                raise ValueError("Text is empty after preprocessing")

            logger.debug(f"Cleaned text: {cleaned_text[:100]}")
            lang_info = self.detect_language_details(cleaned_text)
            logger.debug(f"Language details detected: {lang_info}")

            # Validate model clients
            if not all(
                [self.classifier_client, self.toxic_client, self.emotion_client]
            ):
                raise ValueError("One or more model clients are not initialized")

            # Classify with error handling
            try:
                topic_classification = self.classifier_client.classify(
                    cleaned_text, self.content_categories["topic"]
                )
                interaction_type = self.classifier_client.classify(
                    cleaned_text, self.content_categories["interaction_type"]
                )
                sentiment_aspects = self.classifier_client.classify(
                    cleaned_text, self.content_categories["sentiment_aspects"]
                )
                # Optional custom classifications
                note_cls_raw = None
                browsing_cls_raw = None
                if self.content_categories.get("note_classification"):
                    try:
                        if (
                            len(self.content_categories.get("note_classification", []))
                            > 0
                        ):
                            note_cls_raw = self.classifier_client.classify(
                                cleaned_text,
                                self.content_categories["note_classification"],
                            )
                    except Exception as e:
                        logger.warning(f"Note classification failed: {e}")
                if self.content_categories.get("browsing_history_classification"):
                    try:
                        if (
                            len(
                                self.content_categories.get(
                                    "browsing_history_classification", []
                                )
                            )
                            > 0
                        ):
                            browsing_cls_raw = self.classifier_client.classify(
                                cleaned_text,
                                self.content_categories[
                                    "browsing_history_classification"
                                ],
                            )
                    except Exception as e:
                        logger.warning(f"Browsing history classification failed: {e}")
            except Exception as e:
                logger.error(f"Error in classification: {e}")
                raise

            # Analyze toxicity with error handling
            try:
                toxicity_classification = self.toxic_client.analyze_toxicity(
                    cleaned_text
                )
                if toxicity_classification is None:
                    logger.warning("Toxicity analysis returned None, using default values")
                    toxicity_classification = {"score": 0.0, "label": "non-toxic"}
            except Exception as e:
                logger.error(f"Error in toxicity analysis: {e}")
                toxicity_classification = {"score": 0.0, "label": "non-toxic"}

            # Analyze emotion with error handling
            try:
                emotion = self.emotion_client.analyze_sentiment(cleaned_text)
            except Exception as e:
                logger.error(f"Error in emotion analysis: {e}")
                emotion = {"score": 0.0, "label": "neutral"}

            # Extract and classify entities
            entities_classification = {}
            try:
                entities = await self.extract_entities(cleaned_text)
                logger.info(f"Entities extracted: {entities}")

                if entities and len(entities) > 0:
                    entities_classification = await self.classify_entities(
                        entities, self.content_categories["entitiesClasses"]
                    )
                    logger.info(f"Entities classified: {entities_classification}")
                else:
                    entities = []
            except Exception as e:
                logger.error(
                    f"NER entity extraction failed, falling back to legacy LLM: {e}"
                )
                try:
                    entities = await self.extract_entities(cleaned_text)
                except Exception as _:
                    entities = []

            # Validate and format results
            analysis = {
                "text_metadata": {
                    "content": cleaned_text[:500],
                    "length": len(cleaned_text),
                    "language_info": lang_info,
                    "timestamp": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                },
                "content_analysis": {
                    "topics": [
                        {
                            "category": topic_translation_map.get(label, label),
                            "confidence": round(score * 100, 2),
                        }
                        for label, score in zip(
                            topic_classification["labels"][:3],
                            topic_classification["scores"][:3],
                        )
                    ],
                    "interaction": [
                        {
                            "type": interaction_translation_map.get(label, label),
                            "confidence": round(score * 100, 2),
                        }
                        for label, score in zip(
                            interaction_type["labels"][:3],
                            interaction_type["scores"][:3],
                        )
                    ],
                    "sentiment_aspects": [
                        {
                            "aspect": sentiment_translation_map.get(label, label),
                            "strength": round(score * 100, 2),
                        }
                        for label, score in zip(
                            sentiment_aspects["labels"][:3],
                            sentiment_aspects["scores"][:3],
                        )
                    ],
                },
                "sentiment_metrics": {
                    "emotion": emotion["label"],
                    "emotion_confidence": round(emotion["score"] * 100, 2),
                },
                "toxicity": {
                    "toxicity_score": round(toxicity_classification["score"] * 100, 2),
                    "toxicity_label": toxicity_classification["label"],
                },
                "entity": entities,
                "entities_classification": entities_classification,
            }

            # Attach optional custom classifications if available
            try:
                if "content_analysis" in analysis:
                    if "note_cls_raw" in locals() and note_cls_raw:
                        analysis["content_analysis"]["note_classification"] = [
                            {"category": label, "confidence": round(score * 100, 2)}
                            for label, score in zip(
                                note_cls_raw["labels"][:3], note_cls_raw["scores"][:3]
                            )
                        ]
                    if "browsing_cls_raw" in locals() and browsing_cls_raw:
                        analysis["content_analysis"][
                            "browsing_history_classification"
                        ] = [
                            {"category": label, "confidence": round(score * 100, 2)}
                            for label, score in zip(
                                browsing_cls_raw["labels"][:3],
                                browsing_cls_raw["scores"][:3],
                            )
                        ]
            except Exception as e:
                logger.warning(f"Failed attaching custom classifications: {e}")

            # Validate final analysis
            if not all(
                [
                    analysis["content_analysis"]["topics"],
                    analysis["content_analysis"]["interaction"],
                    analysis["content_analysis"]["sentiment_aspects"],
                    analysis["sentiment_metrics"]["emotion"],
                    analysis["toxicity"]["toxicity_score"] is not None,
                ]
            ):
                raise ValueError("Incomplete analysis results")

            logger.debug(f"Analysis complete: {analysis}")
            return analysis

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}\nText: {text[:200]}")
            raise

    async def process_single_document_notes_browsinghistories(
        self,
        doc,
        apply_single_doc_processing_notes,
        apply_single_doc_processing_browsinghistories,
        alert_item=None,
    ):
        """Process a single document."""
        try:
            # Build analysis text depending on message type to maximize context
            mtype = str(doc.get("Message Type", "")).strip().lower()
            text = str(doc.get("Preview Text", "")).strip()
            # Only process notes and browsing_history documents in this method
            if mtype not in ("note", "browsing_history"):
                logger.info(
                    f"Skipping document ID {doc.get('_id')} with message type '{mtype}' - not a note or browsing_history"
                )
                return
            if mtype == "note" and not apply_single_doc_processing_notes:
                logger.info(
                    f"Skipping note document ID {doc.get('_id')} - single document processing for notes is disabled"
                )
                return
            if (
                mtype == "browsing_history"
                and not apply_single_doc_processing_browsinghistories
            ):
                logger.info(
                    f"Skipping browsing history document ID {doc.get('_id')} - single document processing for browsing histories is disabled"
                )
                return

            logger.info(f"Processing document type: {mtype}, text length: {len(text)}")
            logger.info(
                f"Available custom classifications - note: {self.content_categories.get('note_classification', [])}, browsing: {self.content_categories.get('browsing_history_classification', [])}"
            )

            if mtype == "note":
                title = str(doc.get("Title", "") or doc.get("title", "")).strip()
                body = str(doc.get("Body", "") or doc.get("body", "")).strip()
                summary = str(doc.get("Summary", "") or doc.get("summary", "")).strip()
                join_parts = [p for p in [title, summary, body, text] if p]
                text = " \n".join(join_parts) if join_parts else text
                logger.info(
                    f"Note document - title: {title}, body: {body[:100]}, summary: {summary}, combined text length: {len(text)}"
                )
            elif mtype == "browsing_history":
                url = str(doc.get("url", "") or doc.get("URL", "")).strip()
                title = str(doc.get("title", "") or doc.get("Title", "")).strip()
                search_value = str(doc.get("search_value", "")).strip()
                preview = str(doc.get("Preview Text", "")).strip()
                join_parts = [p for p in [title, search_value, url, preview] if p]
                text = " \n".join(join_parts) if join_parts else text
                logger.info(
                    f"Browsing history document - url: {url}, title: {title}, search: {search_value}, combined text length: {len(text)}"
                )

            if not text:
                logger.warning(f"No text content found for document {doc.get('_id')}")
                return

            # logger.info(f"Processing document ID: {doc.get('_id')}")
            analysis = await self.analyze_content(text)

            analysis_summary = {}

            # Add custom classification to summary by message type if available
            try:
                custom_classifications = analysis.get("content_analysis", {})
                logger.info(
                    f"Custom classifications from analysis: {custom_classifications}"
                )

                # For note type, add note classification if available
                if mtype == "note":
                    logger.info(
                        f"Processing note classification for document {doc.get('_id')}"
                    )
                    if custom_classifications.get("note_classification"):
                        analysis_summary["note_classification"] = (
                            custom_classifications["note_classification"][0]["category"]
                        )
                        logger.info(
                            f"Added note classification from analysis: {analysis_summary['note_classification']}"
                        )
                    elif self.content_categories.get("note_classification"):
                        logger.info(
                            f"Attempting to classify note with categories: {self.content_categories['note_classification']}"
                        )
                        # If no classification result but categories exist, try to classify
                        try:
                            note_cls = self.classifier_client.classify(
                                text, self.content_categories["note_classification"]
                            )
                            logger.info(f"Note classification result: {note_cls}")
                            if (
                                note_cls
                                and note_cls.get("labels")
                                and note_cls.get("scores")
                            ):
                                top_label = note_cls["labels"][0]
                                top_score = note_cls["scores"][0]
                                if (
                                    top_score >= 0.5
                                ):  # Only use if confidence is reasonable
                                    analysis_summary["note_classification"] = top_label
                                    logger.info(
                                        f"Added note classification: {top_label} with score {top_score}"
                                    )
                                else:
                                    logger.info(
                                        f"Note classification score too low: {top_score}"
                                    )
                            else:
                                logger.warning(
                                    "Note classification returned invalid result"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to classify note: {e}")
                    else:
                        logger.info("No note classification categories configured")

                # For browsing_history type, add browsing classification if available
                if mtype == "browsing_history":
                    logger.info(
                        f"Processing browsing history classification for document {doc.get('_id')}"
                    )
                    if custom_classifications.get("browsing_history_classification"):
                        analysis_summary["browsing_history_classification"] = (
                            custom_classifications["browsing_history_classification"][
                                0
                            ]["category"]
                        )
                        logger.info(
                            f"Added browsing classification from analysis: {analysis_summary['browsing_history_classification']}"
                        )
                    elif self.content_categories.get("browsing_history_classification"):
                        logger.info(
                            f"Attempting to classify browsing history with categories: {self.content_categories['browsing_history_classification']}"
                        )
                        # If no classification result but categories exist, try to classify
                        try:
                            browsing_cls = self.classifier_client.classify(
                                text,
                                self.content_categories[
                                    "browsing_history_classification"
                                ],
                            )
                            logger.info(
                                f"Browsing history classification result: {browsing_cls}"
                            )
                            if (
                                browsing_cls
                                and browsing_cls.get("labels")
                                and browsing_cls.get("scores")
                            ):
                                top_label = browsing_cls["labels"][0]
                                top_score = browsing_cls["scores"][0]
                                if (
                                    top_score >= 0.5
                                ):  # Only use if confidence is reasonable
                                    analysis_summary[
                                        "browsing_history_classification"
                                    ] = top_label
                                    logger.info(
                                        f"Added browsing classification: {top_label} with score {top_score}"
                                    )
                                else:
                                    logger.info(
                                        f"Browsing history classification score too low: {top_score}"
                                    )
                            else:
                                logger.warning(
                                    "Browsing history classification returned invalid result"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to classify browsing history: {e}")
                    else:
                        logger.info(
                            "No browsing history classification categories configured"
                        )
            except Exception as e:
                logger.warning(f"Failed to set custom classification in summary: {e}")

            # Update MongoDB
            await self.collection_case.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "social_media_analysis": analysis,
                        "processed": True,
                        "analysis_summary": analysis_summary,
                    }
                },
            )

            logger.info(f"Processed document ID: {doc.get('_id')} successfully.")

        except Exception as e:
            logger.error(f"Error processing document {doc.get('_id')}: {str(e)}")

    async def process_single_document(self, doc, alert_item=None):
        """Process a single document."""
        try:
            # Build analysis text depending on message type to maximize context
            mtype = str(doc.get("Message Type", "")).strip().lower()
            text = str(doc.get("Preview Text", "")).strip()

            if mtype == "note":
                title = str(doc.get("Title", "") or doc.get("title", "")).strip()
                body = str(doc.get("Body", "") or doc.get("body", "")).strip()
                summary = str(doc.get("Summary", "") or doc.get("summary", "")).strip()
                join_parts = [p for p in [title, summary, body, text] if p]
                text = " \n".join(join_parts) if join_parts else text
                logger.info(
                    f"Note document - title: {title}, body: {body[:100]}, summary: {summary}, combined text length: {len(text)}"
                )
            elif mtype == "browsing_history":
                url = str(doc.get("url", "") or doc.get("URL", "")).strip()
                title = str(doc.get("title", "") or doc.get("Title", "")).strip()
                search_value = str(doc.get("search_value", "")).strip()
                preview = str(doc.get("Preview Text", "")).strip()
                join_parts = [p for p in [title, search_value, url, preview] if p]
                text = " \n".join(join_parts) if join_parts else text
                logger.info(
                    f"Browsing history document - url: {url}, title: {title}, search: {search_value}, combined text length: {len(text)}"
                )

            if not text:
                logger.warning(f"No text content found for document {doc.get('_id')}")
                return

            logger.debug(f"alert_item {alert_item}")
            # alert_item may be a boolean (True/False) or other non-dict in some flows;
            # ensure we treat only dicts as alert definitions.
            if not isinstance(alert_item, dict):
                logger.debug("alert_item is not a dict; treating as empty alert")
                alert_item = {}
            keys_to_check = list(alert_item.keys())
            logger.debug(f"keys to check {keys_to_check} {alert_item}")

            logger.info(f"Processing document ID: {doc.get('_id')}")
            analysis = await self.analyze_content(text)
            analysis_summary = self.process_analyze_content_response(analysis)

            # Add custom classification to summary by message type if available
            try:
                custom_classifications = analysis.get("content_analysis", {})
                logger.info(
                    f"Custom classifications from analysis: {custom_classifications}"
                )

                # For note type, add note classification if available
                if mtype == "note":
                    logger.info(
                        f"Processing note classification for document {doc.get('_id')}"
                    )
                    if custom_classifications.get("note_classification"):
                        analysis_summary["note_classification"] = (
                            custom_classifications["note_classification"][0]["category"]
                        )
                        logger.info(
                            f"Added note classification from analysis: {analysis_summary['note_classification']}"
                        )
                    elif self.content_categories.get("note_classification"):
                        logger.info(
                            f"Attempting to classify note with categories: {self.content_categories['note_classification']}"
                        )
                        # If no classification result but categories exist, try to classify
                        try:
                            note_cls = self.classifier_client.classify(
                                text, self.content_categories["note_classification"]
                            )
                            logger.info(f"Note classification result: {note_cls}")
                            if (
                                note_cls
                                and note_cls.get("labels")
                                and note_cls.get("scores")
                            ):
                                top_label = note_cls["labels"][0]
                                top_score = note_cls["scores"][0]
                                if (
                                    top_score >= 0.5
                                ):  # Only use if confidence is reasonable
                                    analysis_summary["note_classification"] = top_label
                                    logger.info(
                                        f"Added note classification: {top_label} with score {top_score}"
                                    )
                                else:
                                    logger.info(
                                        f"Note classification score too low: {top_score}"
                                    )
                            else:
                                logger.warning(
                                    "Note classification returned invalid result"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to classify note: {e}")
                    else:
                        logger.info("No note classification categories configured")

                # For browsing_history type, add browsing classification if available
                if mtype == "browsing_history":
                    logger.info(
                        f"Processing browsing history classification for document {doc.get('_id')}"
                    )
                    if custom_classifications.get("browsing_history_classification"):
                        analysis_summary["browsing_history_classification"] = (
                            custom_classifications["browsing_history_classification"][
                                0
                            ]["category"]
                        )
                        logger.info(
                            f"Added browsing classification from analysis: {analysis_summary['browsing_history_classification']}"
                        )
                    elif self.content_categories.get("browsing_history_classification"):
                        logger.info(
                            f"Attempting to classify browsing history with categories: {self.content_categories['browsing_history_classification']}"
                        )
                        # If no classification result but categories exist, try to classify
                        try:
                            browsing_cls = self.classifier_client.classify(
                                text,
                                self.content_categories[
                                    "browsing_history_classification"
                                ],
                            )
                            logger.info(
                                f"Browsing history classification result: {browsing_cls}"
                            )
                            if (
                                browsing_cls
                                and browsing_cls.get("labels")
                                and browsing_cls.get("scores")
                            ):
                                top_label = browsing_cls["labels"][0]
                                top_score = browsing_cls["scores"][0]
                                if (
                                    top_score >= 0.5
                                ):  # Only use if confidence is reasonable
                                    analysis_summary[
                                        "browsing_history_classification"
                                    ] = top_label
                                    logger.info(
                                        f"Added browsing classification: {top_label} with score {top_score}"
                                    )
                                else:
                                    logger.info(
                                        f"Browsing history classification score too low: {top_score}"
                                    )
                            else:
                                logger.warning(
                                    "Browsing history classification returned invalid result"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to classify browsing history: {e}")
                    else:
                        logger.info(
                            "No browsing history classification categories configured"
                        )
            except Exception as e:
                logger.warning(f"Failed to set custom classification in summary: {e}")

            alert = None
            if isinstance(alert_item, dict):
                keys_to_check = list(alert_item.keys())
            else:
                keys_to_check = []
            analysis_summary_keys = list(analysis_summary.keys())
            keys_matched = 0
            if isinstance(alert_item, dict) and alert_item:
                for key, value in alert_item.items():
                    logger.info(f"{key}: {analysis_summary[key]} in {value}")
                    if key == "toxicity_score":
                        if (
                            key in analysis_summary_keys
                            and analysis_summary[key] >= value
                        ):
                            keys_matched += 1
                            logger.info(
                                f"Toxicity matched with {key}: {analysis_summary[key]} >= {value}"
                            )
                    elif key == "entities":
                        if key in analysis_summary_keys and any(
                            entity in analysis_summary["entities"] for entity in value
                        ):
                            logger.info(f"{key}: {analysis_summary[key]} === {value}")
                            keys_matched += 1
                    elif (
                        key in analysis_summary_keys and analysis_summary[key] in value
                    ):
                        logger.info(f"{key}: {analysis_summary[key]} === {value}")
                        keys_matched += 1
                logger.info(
                    f"keys_matched: {keys_matched} keys_to_check: {len(keys_to_check)}"
                )
                if keys_matched == len(keys_to_check):
                    alert = True
                else:
                    alert = False

            collection_update_doc = {
                "social_media_analysis": analysis,
                "processed": True,
                "analysis_summary": analysis_summary,
                "alert": alert,
            }

            if self.is_llama_validation_enabled:
                llama_validation_summary = (
                    await self.validate_analysis_summary_via_llama(
                        text, analysis_summary
                    )
                )
                collection_update_doc["llama_validation_summary"] = llama_validation_summary

            # Update MongoDB
            await self.collection_case.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": collection_update_doc
                },
            )

            logger.info(f"Processed document ID: {doc.get('_id')} successfully.")

        except Exception as e:
            logger.error(f"Error processing document {doc.get('_id')}: {str(e)}")

    def _prepare_doc_text(self, doc) -> str:
        """Extract and prepare text from a document based on its type."""
        mtype = str(doc.get("Message Type", "")).strip().lower()
        text = str(doc.get("Preview Text", "")).strip()
        
        if mtype == "note":
            title = str(doc.get("Title", "") or doc.get("title", "")).strip()
            body = str(doc.get("Body", "") or doc.get("body", "")).strip()
            summary = str(doc.get("Summary", "") or doc.get("summary", "")).strip()
            join_parts = [p for p in [title, summary, body, text] if p]
            text = " \n".join(join_parts) if join_parts else text
        elif mtype == "browsing_history":
            url = str(doc.get("url", "") or doc.get("URL", "")).strip()
            title = str(doc.get("title", "") or doc.get("Title", "")).strip()
            search_value = str(doc.get("search_value", "")).strip()
            preview = str(doc.get("Preview Text", "")).strip()
            join_parts = [p for p in [title, search_value, url, preview] if p]
            text = " \n".join(join_parts) if join_parts else text
        
        return text

    def _check_alert(self, analysis_summary: Dict, alert_item: Dict) -> Optional[bool]:
        """Check if an analysis summary triggers an alert."""
        if not isinstance(alert_item, dict) or not alert_item:
            return None
        
        keys_to_check = list(alert_item.keys())
        analysis_summary_keys = list(analysis_summary.keys())
        keys_matched = 0
        
        for key, value in alert_item.items():
            if key == "toxicity_score":
                if key in analysis_summary_keys and analysis_summary[key] >= value:
                    keys_matched += 1
            elif key == "entities":
                if key in analysis_summary_keys and analysis_summary.get("entities") and any(
                    entity in analysis_summary["entities"] for entity in value
                ):
                    keys_matched += 1
            elif key in analysis_summary_keys and analysis_summary[key] in value:
                keys_matched += 1
        
        return keys_matched == len(keys_to_check) if keys_to_check else None

    async def process_documents_batched(self, documents: List[Dict], alert_config: Dict):
        """
        Process ALL documents using GPU-optimal batched inference.
        
        Instead of per-doc sequential model calls (7N calls for N docs),
        this method extracts all texts first, runs batched GPU inference
        (~7 total batched calls), then bulk-writes results to MongoDB.
        """
        compute = settings.compute_config
        gpu_batch_size = compute["gpu_batch_size"]
        
        logger.info(f"Starting BATCHED analysis for {len(documents)} documents "
                    f"(gpu_batch_size={gpu_batch_size})")
        import time as _time
        batch_start = _time.time()
        
        # Step 1: Extract and preprocess all texts
        doc_texts = []
        valid_docs = []
        for doc in documents:
            text = self._prepare_doc_text(doc)
            if not text:
                logger.warning(f"No text content for document {doc.get('_id')}, skipping")
                continue
            cleaned = self.preprocess_text(text)
            if not cleaned:
                logger.warning(f"Empty text after preprocessing for document {doc.get('_id')}, skipping")
                continue
            doc_texts.append(cleaned)
            valid_docs.append(doc)
        
        if not valid_docs:
            logger.info("No valid documents to process after text extraction")
            return
        
        n = len(valid_docs)
        logger.info(f"Extracted {n} valid texts, starting batched GPU inference...")
        
        # Step 2: Batched GPU inference for all classification models
        topic_labels = self.content_categories["topic"]
        interaction_labels = self.content_categories["interaction_type"]
        sentiment_labels = self.content_categories["sentiment_aspects"]
        
        try:
            topic_results = self.classifier_client.classify_batch(doc_texts, topic_labels)
            logger.info(f"Topic classification complete: {len(topic_results)} results")
        except Exception as e:
            logger.error(f"Batch topic classification failed: {e}")
            topic_results = [{"labels": [], "scores": []} for _ in doc_texts]
        
        try:
            interaction_results = self.classifier_client.classify_batch(doc_texts, interaction_labels)
            logger.info(f"Interaction classification complete: {len(interaction_results)} results")
        except Exception as e:
            logger.error(f"Batch interaction classification failed: {e}")
            interaction_results = [{"labels": [], "scores": []} for _ in doc_texts]
        
        try:
            sentiment_results = self.classifier_client.classify_batch(doc_texts, sentiment_labels)
            logger.info(f"Sentiment classification complete: {len(sentiment_results)} results")
        except Exception as e:
            logger.error(f"Batch sentiment classification failed: {e}")
            sentiment_results = [{"labels": [], "scores": []} for _ in doc_texts]
        
        try:
            toxicity_results = self.toxic_client.analyze_toxicity_batch(doc_texts)
            logger.info(f"Toxicity analysis complete: {len(toxicity_results)} results")
        except Exception as e:
            logger.error(f"Batch toxicity analysis failed: {e}")
            toxicity_results = [{"score": 0.0, "label": "non-toxic"} for _ in doc_texts]
        
        try:
            emotion_results = self.emotion_client.analyze_sentiment_batch(doc_texts)
            logger.info(f"Emotion analysis complete: {len(emotion_results)} results")
        except Exception as e:
            logger.error(f"Batch emotion analysis failed: {e}")
            emotion_results = [{"score": 0.0, "label": "neutral"} for _ in doc_texts]
        
        # Step 3: Batch entity extraction via AsyncLlamaClient (loaded from ModelRegistry)
        llama_params = self.model_profile.get("llama", {})
        async_llama = ModelRegistry.get_model(
            "async_llama",
            basic_params=llama_params.get("basic_params"),
            advanced_params=llama_params.get("advanced_params"),
            prompt_engineering=llama_params.get("prompt_engineering", ""),
        )
        
        # Prepare text tuples for batch entity extraction: (doc_id, text)
        text_tuples = [(str(doc.get("_id")), text) for doc, text in zip(valid_docs, doc_texts)]
        
        try:
            entity_results = await async_llama.process_batch_entities_async(text_tuples)
            logger.info(f"Batch entity extraction complete: {len(entity_results)} results")
        except Exception as e:
            logger.error(f"Batch entity extraction failed: {e}")
            entity_results = [(str(doc.get("_id")), []) for doc in valid_docs]
        
        # Build a lookup dict: doc_id -> entities
        entity_map = {doc_id: entities for doc_id, entities in entity_results}
        
        # Batch entity classification
        entity_categories = self.content_categories["entitiesClasses"]
        try:
            classification_results = await async_llama.process_batch_entity_classification_async(
                entity_results, entity_categories
            )
            logger.info(f"Batch entity classification complete: {len(classification_results)} results")
        except Exception as e:
            logger.error(f"Batch entity classification failed: {e}")
            classification_results = [(str(doc.get("_id")), {}) for doc in valid_docs]
        
        # Build lookup dict: doc_id -> classification
        classification_map = {doc_id: cls for doc_id, cls in classification_results}
        
        gpu_time = _time.time() - batch_start
        logger.info(f"All GPU + LLM inference complete in {gpu_time:.2f}s for {n} documents")
        
        # Clear GPU memory between inference and assembly phases
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Step 4: Assemble results and bulk-write to MongoDB
        bulk_ops = []
        for i, doc in enumerate(valid_docs):
            try:
                doc_id = str(doc.get("_id"))
                cleaned_text = doc_texts[i]
                lang_info = self.detect_language_details(cleaned_text)
                
                # Safely get results with defaults
                topic_cls = topic_results[i] if i < len(topic_results) else {"labels": [], "scores": []}
                interact_cls = interaction_results[i] if i < len(interaction_results) else {"labels": [], "scores": []}
                sent_cls = sentiment_results[i] if i < len(sentiment_results) else {"labels": [], "scores": []}
                tox_cls = toxicity_results[i] if i < len(toxicity_results) else {"score": 0.0, "label": "non-toxic"}
                emo_cls = emotion_results[i] if i < len(emotion_results) else {"score": 0.0, "label": "neutral"}
                
                if tox_cls is None:
                    tox_cls = {"score": 0.0, "label": "non-toxic"}
                
                entities = entity_map.get(doc_id, [])
                entities_classification = classification_map.get(doc_id, {})
                
                # Build full analysis object (same structure as analyze_content)
                analysis = {
                    "text_metadata": {
                        "content": cleaned_text[:500],
                        "length": len(cleaned_text),
                        "language_info": lang_info,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    },
                    "content_analysis": {
                        "topics": [
                            {
                                "category": topic_translation_map.get(label, label),
                                "confidence": round(score * 100, 2),
                            }
                            for label, score in zip(
                                (topic_cls.get("labels") or [])[:3],
                                (topic_cls.get("scores") or [])[:3],
                            )
                        ],
                        "interaction": [
                            {
                                "type": interaction_translation_map.get(label, label),
                                "confidence": round(score * 100, 2),
                            }
                            for label, score in zip(
                                (interact_cls.get("labels") or [])[:3],
                                (interact_cls.get("scores") or [])[:3],
                            )
                        ],
                        "sentiment_aspects": [
                            {
                                "aspect": sentiment_translation_map.get(label, label),
                                "strength": round(score * 100, 2),
                            }
                            for label, score in zip(
                                (sent_cls.get("labels") or [])[:3],
                                (sent_cls.get("scores") or [])[:3],
                            )
                        ],
                    },
                    "sentiment_metrics": {
                        "emotion": emo_cls.get("label", "neutral"),
                        "emotion_confidence": round(emo_cls.get("score", 0.0) * 100, 2),
                    },
                    "toxicity": {
                        "toxicity_score": round(tox_cls.get("score", 0.0) * 100, 2),
                        "toxicity_label": tox_cls.get("label", "non-toxic"),
                    },
                    "entity": entities,
                    "entities_classification": entities_classification,
                }
                
                analysis_summary = self.process_analyze_content_response(analysis)
                
                # Check alert
                alert = self._check_alert(analysis_summary, alert_config)
                
                collection_update_doc = {
                    "social_media_analysis": analysis,
                    "processed": True,
                    "analysis_summary": analysis_summary,
                    "alert": alert,
                }
                
                bulk_ops.append(
                    UpdateOne(
                        {"_id": doc["_id"]},
                        {"$set": collection_update_doc}
                    )
                )
                
            except Exception as e:
                logger.error(f"Error assembling result for document {doc.get('_id')}: {e}")
                continue
        
        # Step 5: Bulk write to MongoDB
        if bulk_ops:
            mongo_batch_size = 500
            for i in range(0, len(bulk_ops), mongo_batch_size):
                batch = bulk_ops[i:i + mongo_batch_size]
                try:
                    result = await self.collection_case.bulk_write(batch, ordered=False)
                    logger.info(f"MongoDB bulk write batch {i//mongo_batch_size + 1}: "
                              f"modified={result.modified_count}, upserted={result.upserted_count}")
                except Exception as e:
                    logger.error(f"MongoDB bulk write failed for batch {i//mongo_batch_size + 1}: {e}")
                    # Fallback: try individual updates
                    for op in batch:
                        try:
                            await self.collection_case.update_one(
                                op._filter, op._doc
                            )
                        except Exception as inner_e:
                            logger.error(f"Individual update fallback also failed: {inner_e}")
        
        total_time = _time.time() - batch_start
        logger.info(
            f"BATCHED processing complete: {len(bulk_ops)}/{n} documents "
            f"in {total_time:.2f}s ({n/total_time:.1f} docs/sec)"
        )

    async def process_documents(self):
        """Process all unprocessed documents in MongoDB with batched GPU inference."""
        await self._safe_update_case_status(
            {
                "status": "processing",
                "analysis_started_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
        )

        # Get alert configuration
        filtered_alert = {}
        if self.alert_id and self.alert_id != None:
            alert = await self._safe_fetch_alert()
            if alert:
                filtered_alert = {
                    k: v
                    for k, v in alert.items()
                    if v is not None
                    and k not in ["_id", "user_id", "name", "description", "created_at"]
                }
                logger.info(f"Alert found; details: {filtered_alert}")

        # Get unprocessed documents
        documents = await self._safe_find_unprocessed_documents()

        if not documents:
            logger.info("No unprocessed documents found.")
            return

        logger.info(f"Found {len(documents)} documents to process.")

        # Use batched GPU inference for maximum throughput
        try:
            await self.process_documents_batched(documents, filtered_alert)
        except Exception as e:
            logger.error(f"Batched processing failed, falling back to sequential: {e}")
            # Fallback: process documents individually
            await asyncio.gather(
                *(self.process_single_document(doc, filtered_alert) for doc in documents)
            )

        logger.info("All documents processed successfully.")
        await self._safe_update_case_status(
            {
                "status": "completed",
                "analysis_completed_at": datetime.datetime.now(
                    datetime.timezone.utc
                ).isoformat(),
            }
        )

    async def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics and system resources"""
        try:
            # Get system resources from centralized settings
            compute = settings.compute_config

            # Get document counts
            total_documents = await self.collection_case.count_documents({})
            processed_documents = await self.collection_case.count_documents(
                {"processed": True}
            )
            unprocessed_documents = await self.collection_case.count_documents(
                {"processed": {"$ne": True}}
            )
            alert_documents = await self.collection_case.count_documents(
                {"alert": True}
            )

            stats = {
                "case_id": self.case_id,
                "total_documents": total_documents,
                "processed_documents": processed_documents,
                "unprocessed_documents": unprocessed_documents,
                "alert_documents": alert_documents,
                "processing_progress": (
                    (processed_documents / total_documents * 100)
                    if total_documents > 0
                    else 0
                ),
                "system_resources": {
                    "cpu_count": compute["cpu_count"],
                    "ram_gb": compute["ram_gb"],
                    "gpu_count": compute["gpu_count"],
                    "gpu_memory_gb": compute["gpu_memory_gb"],
                    "gpu_name": compute["gpu_name"],
                },
                "parallel_processing_enabled": self.use_parallel_processing,
                "model_clients_initialized": all(
                    [
                        self.classifier_client is not None,
                        self.toxic_client is not None,
                        self.emotion_client is not None,
                    ]
                ),
            }

            return stats
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {"error": str(e)}
