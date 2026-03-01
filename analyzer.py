import datetime
import re
from setup import setup_logging
from bson import ObjectId
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from motor.motor_asyncio import AsyncIOMotorCollection
from config.db import alerts_collection
from utils.translationmap import (
    sentiment_translation_map,
    interaction_translation_map,
    topic_translation_map,
    topics_arabic,
    sentiments_arabic,
    interactions_arabic,
    entitiesClasses_arabic,
)
from typing import Dict
from model_registry import ModelRegistry
from utils.helpers import get_optimal_device, cleanup_gpu_memory, monitor_gpu_memory

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
        mongo_collection_case: AsyncIOMotorCollection = None,
        mongo_collection__all_cases: AsyncIOMotorCollection = None,
        case_id: str = None,
        alert_id: str = None,
        topics: list = None,
        sentiments: list = None,
        interactions: list = None,
        entitiesClasses: list = None,
        model_profile: Dict = None,
    ):
        logger.info("Initializing ArabicSocialAnalyzer...")
        classifier_obj = model_profile.get("classifier", {})
        toxic_obj = model_profile.get("toxic", {})
        emotion_obj = model_profile.get("emotion", {})
        self.collection_case = mongo_collection_case
        self.collection__all_cases = mongo_collection__all_cases
        self.case_id = case_id
        self.alert_id = alert_id
        self.model_profile = model_profile
        self.retrying_time = 10
        self.headers = {"Content-Type": "application/json"}

        # Intelligent device selection with memory management
        optimal_device = get_optimal_device()
        # optimal_device = "cpu"
        self.device = 0 if optimal_device == "cuda" else -1
        self.device_name = optimal_device
        logger.info(f"ArabicSocialAnalyzer initialized with device: {self.device_name}")
        self.content_categories = {
            "topic": None,
            "interaction_type": None,
            "sentiment_aspects": None,
        }
        self.content_categories["topic"] = (
            topics if topics and len(topics) > 0 else topics_arabic
        )
        self.content_categories["interaction_type"] = (
            interactions
            if interactions and len(interactions) > 0
            else interactions_arabic
        )
        self.content_categories["sentiment_aspects"] = (
            sentiments if sentiments and len(sentiments) > 0 else sentiments_arabic
        )
        self.content_categories["entitiesClasses"] = (
            entitiesClasses
            if entitiesClasses and len(entitiesClasses) > 0
            else entitiesClasses_arabic
        )
        # model clients
        self.classifier_client = ModelRegistry.get_model("classifier", model_name=classifier_obj.get("name"))
        self.toxic_client = ModelRegistry.get_model("toxic", model_name=toxic_obj.get("name"))
        self.emotion_client = ModelRegistry.get_model("emotion", model_name=emotion_obj.get("name"))
        self.llamaClient = ModelRegistry.get_model("llama")

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
        """Normalize and clean social media text."""
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[إأآا]", "ا", text)
        text = re.sub(r"ة", "ه", text)
        text = re.sub(r"[\u064B-\u0652]", "", text)
        text = re.sub(r"(.)\1{2,}", r"\1", text)
        return text

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

    def analyze_content(self, text):
        """Analyze social media content with optimized GPU memory management."""
        try:
            logger.debug(f"Starting analysis for text: {text[:100]}...")
            cleaned_text = self.preprocess_text(text)
            logger.debug(f"Cleaned text: {cleaned_text[:100]}")
            lang_info = self.detect_language_details(cleaned_text)
            logger.debug(f"Language details detected: {lang_info}")
            emotion_obj = self.model_profile.get("emotion", {})

            # classifier model consumption via classifier client with memory monitoring
            with monitor_gpu_memory("topic_classification"):
                topic_classification = self.classifier_client.classify(
                    cleaned_text, self.content_categories["topic"]
                )

            with monitor_gpu_memory("interaction_classification"):
                interaction_type = self.classifier_client.classify(
                    cleaned_text, self.content_categories["interaction_type"]
                )

            with monitor_gpu_memory("sentiment_classification"):
                sentiment_aspects = self.classifier_client.classify(
                    cleaned_text, self.content_categories["sentiment_aspects"]
                )

            # toxic model consumption via toxic client
            with monitor_gpu_memory("toxicity_analysis"):
                toxicity_classification = self.toxic_client.analyze_toxicity(
                    cleaned_text
                )

            # emotion model consumption via emotion client
            with monitor_gpu_memory("emotion_analysis"):
                emotion = self.emotion_client.analyze_sentiment(cleaned_text)

            # Clean up GPU memory after all model inferences
            cleanup_gpu_memory()

            # extracting entities from the text via Llama3.1:8b
            entities = self.llamaClient.extract_entities(cleaned_text)
            logger.info(f"entity extracted with LLM model: {entities}")

            # classifying entities into categories
            if entities and len(entities) > 0:
                try:
                    entities_classification = self.llamaClient.classify_entities(
                        entities, self.content_categories["entitiesClasses"]
                    )
                    logger.info(
                        f"entities classified with LLM model: {entities_classification}"
                    )
                except Exception as e:
                    logger.error(f"Error classifying entities in analyzer: {e}")
                    entities_classification = {}
            else:
                entities_classification = None

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

            logger.debug(f"Analysis complete: {analysis}")
            return analysis

        except Exception as e:
            # Clean up GPU memory on any error
            cleanup_gpu_memory()

            # Check if it's a CUDA memory error
            if "CUDA" in str(e) and (
                "memory" in str(e).lower() or "allocation" in str(e).lower()
            ):
                logger.error(f"CUDA memory error during analysis: {str(e)}")
                logger.error(
                    "Consider setting DI_DEVICE=cpu environment variable or reducing batch size"
                )
            else:
                logger.error(f"Analysis error: {str(e)}\nText: {text[:200]}")
            raise

    async def process_single_document(self, doc, alert_item=None):
        """Process a single document."""
        try:
            text = str(doc.get("Preview Text", "")).strip()
            if not text:
                return

            logger.debug(f"alert_item {alert_item}")
            keys_to_check = list(alert_item.keys())
            logger.debug(f"keys to check {keys_to_check} {alert_item}")

            logger.info(f"Processing document ID: {doc.get('_id')}")
            analysis = self.analyze_content(text)

            top_topic = (
                analysis["content_analysis"]["topics"][0]["category"]
                if analysis["content_analysis"]["topics"]
                and analysis["content_analysis"]["topics"][0]["confidence"] >= 65
                else "others"
            )

            sentiment_aspect = (
                analysis["content_analysis"]["sentiment_aspects"][0]["aspect"]
                if analysis["content_analysis"]["sentiment_aspects"]
                and analysis["content_analysis"]["sentiment_aspects"][0]["strength"]
                >= 65
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
                "toxicity_score": toxicity_score,
                "risk_level": risk_level,
                "sentiment_aspect": sentiment_aspect,
                "emotion": emotion,
                "language": language,
                "interaction_type": interaction_type,
                "entities": entities,
                "entities_classification": entities_classification,
            }

            alert = None
            keys_to_check = list(alert_item.keys())
            analysis_summary_keys = list(analysis_summary.keys())
            keys_matched = 0
            if alert_item:
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

            # Update MongoDB
            await self.collection_case.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "social_media_analysis": analysis,
                        "processed": True,
                        "analysis_summary": analysis_summary,
                        "alert": alert,
                    }
                },
            )

            logger.info(f"Processed document ID: {doc.get('_id')} successfully.")

        except Exception as e:
            logger.error(f"Error processing document {doc.get('_id')}: {str(e)}")

    async def process_documents(self):
        """Process all unprocessed documents in MongoDB."""
        await self.collection__all_cases.find_one_and_update(
            {"_id": ObjectId(self.case_id)},
            {
                "$set": {
                    "status": "processing",
                    "analysis_started_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            },
        )
        filtered_alert = {}
        if self.alert_id and self.alert_id != None:
            alert = await alerts_collection.find_one({"_id": ObjectId(self.alert_id)})
            if alert:
                filtered_alert = {
                    k: v
                    for k, v in alert.items()
                    if v is not None
                    and k not in ["_id", "user_id", "name", "description", "created_at"]
                }
                logger.info(f"Alert found; details: {filtered_alert}")

        documents = await self.collection_case.find(
            {"processed": {"$ne": True}}
        ).to_list(length=None)
        for doc in documents:
            await self.process_single_document(doc, filtered_alert)

        logger.info("All documents processed successfully.")

        await self.collection__all_cases.find_one_and_update(
            {"_id": ObjectId(self.case_id)},
            {
                "$set": {
                    "status": "completed",
                    "analysis_completed_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                }
            },
        )
