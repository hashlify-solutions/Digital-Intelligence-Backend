import math
from datetime import datetime, timezone
from bson import ObjectId
import logging
import torch
import gc
from contextlib import contextmanager
from config.settings import settings
from qdrant_client.models import Distance, VectorParams
from config.db import qdrant_client, with_db_retry, collection_case, db
import time
from typing import List, Any, Dict
from qdrant_client.http.exceptions import ResponseHandlingException
import httpx
import re
import asyncio
from clients.llama.async_llama_client import AsyncLlamaClient
from model_registry import ModelRegistry

logger = logging.getLogger(__name__)

def sanitize_nan_values(data: dict) -> dict:
    """
    Recursively replace NaN, Infinity, and -Infinity values with None in a dictionary.
    """
    for key, value in data.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            data[key] = None
        elif isinstance(value, dict):
            data[key] = sanitize_nan_values(value)
        elif isinstance(value, list):
            data[key] = [
                sanitize_nan_values(v) if isinstance(v, dict) else v for v in value
            ]
    return data


def serialize_mongodb_document(doc):
    """
    Convert MongoDB document with ObjectId fields to JSON-serializable format.
    Recursively converts ObjectId instances to strings.
    """
    if doc is None:
        return None

    if isinstance(doc, ObjectId):
        return str(doc)
    elif isinstance(doc, dict):
        return {key: serialize_mongodb_document(value) for key, value in doc.items()}
    elif isinstance(doc, list):
        return [serialize_mongodb_document(item) for item in doc]
    elif isinstance(doc, datetime):
        return doc.isoformat()
    else:
        return doc


def get_optimal_device(min_vram_gb: float = 0.5):
    """
    Determine the optimal device for an ML model based on CUDA availability
    and whether enough *free* VRAM remains for the model.

    Args:
        min_vram_gb: Minimum free GPU VRAM (in GB) required to place this
                     model on CUDA.  Callers should pass an estimate that
                     covers model weights **plus** a reasonable inference
                     buffer.  Defaults to 0.5 GB (suitable for small models
                     like MiniLM, ResNet-50, etc.).

    Returns:
        ``'cuda'`` when a GPU is present and free VRAM >= *min_vram_gb*,
        ``'cpu'`` otherwise.
    """

    forced_device = settings.di_device.lower()
    if forced_device in ["cpu", "cuda"]:
        logger.info(f"Using forced device from environment: {forced_device}")
        return forced_device

    if not torch.cuda.is_available():
        logger.info("CUDA not available, using CPU")
        return "cpu"

    try:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)

        torch.cuda.empty_cache()
        allocated_memory = torch.cuda.memory_allocated(0)
        cached_memory = torch.cuda.memory_reserved(0)
        free_memory = gpu_memory - max(allocated_memory, cached_memory)
        free_memory_gb = free_memory / (1024**3)

        logger.info(
            f"GPU Memory — Total: {gpu_memory_gb:.2f} GB, "
            f"Free: {free_memory_gb:.2f} GB, "
            f"Required: {min_vram_gb:.2f} GB"
        )

        if free_memory_gb >= min_vram_gb:
            logger.info(
                f"Sufficient VRAM ({free_memory_gb:.2f} GB free >= "
                f"{min_vram_gb:.2f} GB required) — using CUDA"
            )
            return "cuda"
        else:
            logger.warning(
                f"Insufficient VRAM ({free_memory_gb:.2f} GB free < "
                f"{min_vram_gb:.2f} GB required) — falling back to CPU"
            )
            return "cpu"

    except Exception as e:
        logger.warning(f"Error checking CUDA memory, falling back to CPU: {e}")
        return "cpu"


def cleanup_gpu_memory():
    """
    Clean up GPU memory and perform garbage collection.
    """

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Force garbage collection
    gc.collect()


def monitor_gpu_memory(operation_name="operation"):
    """
    Context manager to monitor GPU memory usage during inference operations.
    """

    @contextmanager
    def _monitor():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated(0)
            peak_memory = torch.cuda.max_memory_allocated(0)
            logger.debug(
                f"GPU memory before {operation_name}: {initial_memory / (1024**2):.2f}MB"
            )
            try:
                yield
            finally:
                final_memory = torch.cuda.memory_allocated(0)
                peak_memory = torch.cuda.max_memory_allocated(0)
                logger.debug(
                    f"GPU memory after {operation_name}: {final_memory / (1024**2):.2f}MB"
                )
                logger.debug(
                    f"Peak GPU memory during {operation_name}: {peak_memory / (1024**2):.2f}MB"
                )
                # Clean up
                torch.cuda.empty_cache()
        else:
            yield

    return _monitor()


def sanitize_nan_values_recursive(data):
    """Recursively replaces NaN values with None in dictionaries and lists."""
    if isinstance(data, dict):
        return {k: sanitize_nan_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_nan_values(v) for v in data]
    elif isinstance(data, float) and math.isnan(data):  # Check if value is NaN
        return None  # Or return "Unknown" if preferred
    else:
        return data


def create_quadrant_collection_if_not_exists(collection_name, vector_size, distance):
    """
    Create a quadrant collection if it doesn't exist.
    Enhanced with robust retry logic for better reliability.
    """
    logger.info(f"Checking if quadrant collection {collection_name} exists or not.")
    
    # Check if collection exists with retry logic
    collection_exists = robust_qdrant_collection_exists(collection_name)
    
    if not collection_exists:
        logger.info(
            f"Quadrant collection {collection_name} doesn't exist, hence creating it."
        )
        vectors_config = VectorParams(size=vector_size, distance=distance)
        success = robust_qdrant_create_collection(collection_name, vectors_config)
        
        if not success:
            error_msg = f"Failed to create quadrant collection {collection_name} after multiple attempts"
            logger.error(error_msg)
            raise Exception(error_msg)
    else:
        logger.info(f"Quadrant collection {collection_name} already exists.")


def robust_qdrant_upsert(collection_name: str, points: List[Any], max_retries: int = 3, base_delay: float = 1.0) -> bool:
    """
    Performs a robust upsert to Qdrant with retry logic and exponential backoff.
    
    Args:
        collection_name: Name of the Qdrant collection
        points: List of points to upsert
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds for exponential backoff
        
    Returns:
        bool: True if upsert was successful, False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Attempting to upsert {len(points)} points to collection {collection_name} (attempt {attempt + 1}/{max_retries + 1})")
            
            result = qdrant_client.upsert(
                collection_name=collection_name,
                points=points,
            )
            
            logger.info(f"Successfully upserted {len(points)} points to collection {collection_name}")
            return True
            
        except (ResponseHandlingException, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logger.warning(
                    f"Timeout/connection error during upsert to {collection_name} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(
                    f"Failed to upsert to {collection_name} after {max_retries + 1} attempts. Final error: {e}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Unexpected error during upsert to {collection_name}: {e}")
            return False
    
    return False


def robust_qdrant_collection_exists(collection_name: str, max_retries: int = 3) -> bool:
    """
    Checks if a Qdrant collection exists with retry logic.
    
    Args:
        collection_name: Name of the collection to check
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if collection exists, False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            return qdrant_client.collection_exists(collection_name=collection_name)
        except (ResponseHandlingException, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt < max_retries:
                delay = 1.0 * (2 ** attempt)
                logger.warning(
                    f"Timeout/connection error checking collection {collection_name} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to check collection {collection_name} after {max_retries + 1} attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error checking collection {collection_name}: {e}")
            return False
    
    return False


def robust_qdrant_create_collection(collection_name: str, vectors_config: VectorParams, max_retries: int = 3) -> bool:
    """
    Creates a Qdrant collection with retry logic.
    
    Args:
        collection_name: Name of the collection to create
        vectors_config: Vector configuration for the collection
        max_retries: Maximum number of retry attempts
        
    Returns:
        bool: True if collection was created successfully, False otherwise
    """
    for attempt in range(max_retries + 1):
        try:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )
            logger.info(f"Successfully created collection {collection_name}")
            return True
        except (ResponseHandlingException, httpx.ReadTimeout, httpx.ConnectTimeout, httpx.TimeoutException) as e:
            if attempt < max_retries:
                delay = 1.0 * (2 ** attempt)
                logger.warning(
                    f"Timeout/connection error creating collection {collection_name} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay} seconds..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Failed to create collection {collection_name} after {max_retries + 1} attempts: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error creating collection {collection_name}: {e}")
            return False
    
    return False


async def _precompute_and_save_geolocations(case_id: str, case_name: str, models_profile: dict):
    """
    Precompute geolocations for a case and persist them to a dedicated collection.
    This implementation:
    - Persists direct location messages that already contain latitude/longitude.
    - Uses only pre-classified entities from analysis_summary.entities_classification (e.g., city, country, address) to infer coordinates.
    It does NOT perform any on-demand NER or entity classification; it only resolves coordinates for the already-classified entities.
    """
    try:
        messages_collection = db[f"{case_name}_{case_id}"]
        geo_collection = db[f"{case_name}_{case_id}_geolocations"]

        # Clear previous precomputed results to avoid duplicates on re-run
        try:
            await geo_collection.delete_many({})
        except Exception:
            pass

        # Gather direct location messages that have coordinates
        query = {
            "Message Type": "location",
            "latitude": {"$exists": True, "$ne": None},
            "longitude": {"$exists": True, "$ne": None}
        }
        messages = await messages_collection.find(query).to_list(None)

        geolocation_docs = []
        for message in messages:
            try:
                message_id = str(message.get("_id"))
                analysis_summary = message.get("analysis_summary", {})
                toxicity_score = analysis_summary.get("toxicity_score")
                risk_level = analysis_summary.get("risk_level")

                # Map toxicity_score -> toxicity_level for display consistency
                # Note: analyzer stores toxicity_score as percentage (0-100), so use percentage thresholds
                toxicity_level = "low"
                if toxicity_score is not None:
                    try:
                        score = float(toxicity_score)
                        if score >= 70:
                            toxicity_level = "high"
                        elif score >= 40:
                            toxicity_level = "medium"
                    except Exception:
                        pass
                elif isinstance(risk_level, str):
                    rl = risk_level.lower()
                    if "high" in rl or "critical" in rl:
                        toxicity_level = "high"
                    elif "medium" in rl or "moderate" in rl:
                        toxicity_level = "medium"

                location_name = (
                    message.get("location_name")
                    or message.get("Name")
                    or message.get("Preview Text")
                    or "Unknown Location"
                )

                geolocation_docs.append({
                    "id": message_id,
                    "type": "direct_location",
                    "coordinates": {
                        "latitude": message.get("latitude"),
                        "longitude": message.get("longitude")
                    },
                    "location_name": location_name,
                    "location_type": message.get("location_type", "Unknown"),
                    "toxicity_level": toxicity_level,
                    "toxicity_score": toxicity_score,
                    "risk_level": risk_level,
                    "source": message.get("Source", "UFDR"),
                    "application": message.get("Application", "Location Services"),
                    "timestamp": message.get("Date") or message.get("timestamp"),
                    "message_type": "location",
                    "ai_confidence": None,
                    "ai_location_type": None,
                    "coordinate_source": "direct",
                    "extracted_entities": []
                })
            except Exception as build_err:
                logger.warning(f"Skipping message in geolocation precompute due to build error: {build_err}")
                continue

        # 2) Collect messages that have pre-classified location entities only (no fresh extraction)
        entity_query = {
            "analysis_summary.entities_classification": {"$exists": True, "$ne": {}}
        }
        try:
            candidate_messages = await messages_collection.find(entity_query).to_list(None)
        except Exception as e:
            logger.warning(f"Failed to query candidate messages for entity-based geolocation: {e}")
            candidate_messages = []

        # Prepare Async LLM client (loaded from ModelRegistry for caching)
        try:
            async_llm_client = ModelRegistry.get_model("async_llama")
        except Exception as e:
            logger.error(f"Failed to initialize AsyncLlamaClient, will skip entity-based geolocations: {e}")
            async_llm_client = None

        # 2a) Build message map only (no fresh NER extraction)
        import json
        message_map: Dict[str, Dict[str, Any]] = {}
        for m in candidate_messages:
            mid = str(m.get("_id"))
            message_map[mid] = m

        # 2b) Initialize per_doc_entities (use pre-classified entities only)
        per_doc_entities: Dict[str, Dict[str, str]] = {}

        # 2c) Merge with pre-classified entities in analysis_summary
        for m in candidate_messages:
            mid = str(m.get("_id"))
            analysis_summary = m.get("analysis_summary", {})
            entities_classification = analysis_summary.get("entities_classification", {}) or {}
            existing_map = per_doc_entities.get(mid, {})
            for entity_type, ents in entities_classification.items():
                etype = str(entity_type).lower()
                if etype not in ["location", "city", "country", "address", "unknown"]:
                    continue
                if not ents:
                    continue
                for ent in ents:
                    if not ent:
                        continue
                    prev = existing_map.get(ent)
                    if prev in [None, "unknown", "location", "address"] or etype in ["city", "country"]:
                        existing_map[ent] = etype
            if existing_map:
                per_doc_entities[mid] = existing_map

        # 3) Build global unique entity set with strongest type hints
        def _normalize_entity(e: str) -> str:
            return re.sub(r"\s+", " ", e or "").strip().lower()

        strongest_type = {"city": 3, "country": 3, "location": 2, "address": 2, "unknown": 1}
        unique_entities: Dict[str, Dict[str, Any]] = {}
        for _mid, ent_map in per_doc_entities.items():
            for ent, etype in ent_map.items():
                norm = _normalize_entity(ent)
                if not norm:
                    continue
                current = unique_entities.get(norm)
                if (current is None) or (strongest_type.get(etype, 1) > strongest_type.get(current["hint"], 1)):
                    unique_entities[norm] = {"text": ent, "hint": etype}

        # 4) Resolve coordinates for unique entities concurrently
        resolved_coords: Dict[str, Dict[str, Any]] = {}
        if async_llm_client and unique_entities:
            async def _resolve(entity_text: str, hint: str) -> Dict[str, Any]:
                prompt = (
                    "Extract geographical coordinates (latitude, longitude) for the following Arabic location entity: {entity}\n"
                    "Expected type (hint): {expected_type}\n"
                    "If expected type is city or country, use the city's center or the country's centroid/capital.\n"
                    "Return ONLY valid JSON with keys latitude, longitude, confidence, location_type.\n"
                    'Example: {{"latitude": 24.7136, "longitude": 46.6753, "confidence": 0.9, "location_type": "city"}}.\n'
                    "If ambiguous or unknown, set latitude and longitude to null and confidence to 0.0."
                )
                try:
                    raw = await async_llm_client.chat_async(
                        prompt,
                        {"entity": entity_text, "expected_type": hint or "country_or_city"}
                    )
                    data = None
                    try:
                        data = json.loads(raw.strip()) if isinstance(raw, str) else None
                    except Exception:
                        if isinstance(raw, str):
                            s = raw.find("{")
                            e = raw.rfind("}")
                            if s != -1 and e != -1 and e > s:
                                try:
                                    data = json.loads(raw[s:e+1])
                                except Exception:
                                    data = None
                    if not isinstance(data, dict):
                        return {"lat": None, "lng": None, "confidence": 0.0, "ai_type": "unknown"}
                    lat = data.get("latitude")
                    lng = data.get("longitude")
                    conf = data.get("confidence", 0.0)
                    ai_type = data.get("location_type") or (hint if hint in ["city", "country"] else "unknown")
                    try:
                        lat = float(lat) if lat is not None else None
                        lng = float(lng) if lng is not None else None
                        conf = float(conf) if conf is not None else 0.0
                    except Exception:
                        lat, lng, conf = None, None, 0.0
                    return {"lat": lat, "lng": lng, "confidence": conf, "ai_type": ai_type}
                except Exception as e:
                    logger.warning(f"Failed resolving coords for '{entity_text}': {e}")
                    return {"lat": None, "lng": None, "confidence": 0.0, "ai_type": "unknown"}

            tasks = []
            idx_map = []
            for norm, info in unique_entities.items():
                tasks.append(_resolve(info["text"], info["hint"]))
                idx_map.append(norm)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, norm in enumerate(idx_map):
                res = results[i]
                if isinstance(res, dict):
                    resolved_coords[norm] = res
                else:
                    resolved_coords[norm] = {"lat": None, "lng": None, "confidence": 0.0, "ai_type": "unknown"}

        # 5) Materialize per-message geolocation docs from entities
        for mid, ent_map in per_doc_entities.items():
            m = message_map.get(mid) or await messages_collection.find_one({"_id": ObjectId(mid)})
            if not m:
                continue

            analysis_summary = m.get("analysis_summary", {})
            toxicity_score = analysis_summary.get("toxicity_score")
            risk_level = analysis_summary.get("risk_level")
            toxicity_level = "low"
            # analyzer/parallel processing uses toxicity_score as a percentage (0-100)
            if toxicity_score is not None:
                try:
                    s = float(toxicity_score)
                    if s >= 70:
                        toxicity_level = "high"
                    elif s >= 40:
                        toxicity_level = "medium"
                except Exception:
                    pass
            elif isinstance(risk_level, str):
                rl = risk_level.lower()
                if "high" in rl or "critical" in rl:
                    toxicity_level = "high"
                elif "medium" in rl or "moderate" in rl:
                    toxicity_level = "medium"

            for ent, etype in ent_map.items():
                norm = (re.sub(r"\s+", " ", ent or "").strip().lower())
                resolved = resolved_coords.get(norm, {"lat": None, "lng": None, "confidence": 0.0, "ai_type": etype or "unknown"})
                lat, lng = resolved.get("lat"), resolved.get("lng")
                ai_conf = resolved.get("confidence", 0.0)
                ai_loc_type = resolved.get("ai_type") or (etype if etype in ["city", "country"] else "unknown")
                coordinate_source = "ai_classification" if (lat is not None and lng is not None) else None

                geolocation_docs.append({
                    "id": f"{mid}_{ai_loc_type}_{abs(hash(ent)) % 100000}",
                    "type": "classified_entity",
                    "coordinates": {"latitude": lat, "longitude": lng} if (lat is not None and lng is not None) else None,
                    "location_name": ent,
                    "location_type": ai_loc_type,
                    "toxicity_level": toxicity_level,
                    "toxicity_score": toxicity_score,
                    "risk_level": risk_level,
                    "source": m.get("Source", "Message Analysis"),
                    "application": m.get("Application", "Entity Classification"),
                    "timestamp": m.get("Date") or m.get("timestamp"),
                    "message_type": m.get("Message Type", "unknown"),
                    "ai_confidence": ai_conf,
                    "ai_location_type": ai_loc_type,
                    "coordinate_source": coordinate_source,
                    "extracted_entities": [{
                        "entity_type": ai_loc_type,
                        "entity_value": ent,
                        "confidence": ai_conf,
                        "original_message": (m.get("Preview Text", "")[:200] + "...") if m.get("Preview Text") else ""
                    }]
                })

        # 6) Persist and update case flags
        try:
            if geolocation_docs:
                await geo_collection.insert_many(geolocation_docs)
            await collection_case.update_one(
                {"_id": ObjectId(case_id)},
                {"$set": {"geolocations_ready": True, "geolocations_count": len(geolocation_docs)}}
            )
        except Exception as e:
            logger.error(f"Failed persisting precomputed geolocations or updating case flags: {e}")

        logger.info(f"Precomputed {len(geolocation_docs)} geolocations (direct + entity-based) for case {case_id}")
    except Exception as e:
        logger.error(f"_precompute_and_save_geolocations failed for case {case_id}: {e}")


@with_db_retry(max_retries=5, delay=2)
async def _safe_update_case_processing_status(case_id: str):
    await collection_case.find_one_and_update(
        {"_id": ObjectId(case_id)},
        {
            "$set": {
                "status": "processing",
                "analysis_started_at": datetime.now(timezone.utc).isoformat()
            }
        }
    )

@with_db_retry(max_retries=5, delay=2)
async def _safe_update_case_total_messages(case_id: str, total_messages: int):
    await collection_case.find_one_and_update(
        {"_id": ObjectId(case_id)},
        {
            "$set": {
                "total_messages": total_messages
            }
        }
    )

@with_db_retry(max_retries=5, delay=2)
async def _safe_mark_case_failed(case_id: str, error_message: str):
    await collection_case.find_one_and_update(
        {"_id": ObjectId(case_id)},
        {
            "$set": {
                "status": "failed",
                "error": error_message
            }
        }
    )


@with_db_retry(max_retries=5, delay=2)
async def _finalize_case_processing(case_id: str, cases_collection=None):
    """Record processing_completed_at and total_processing_time on the case."""
    col = cases_collection or collection_case
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()

    case = await col.find_one(
        {"_id": ObjectId(case_id)}, {"processing_started_at": 1}
    )

    update_fields = {"processing_completed_at": now_iso}

    if case and case.get("processing_started_at"):
        try:
            started = datetime.fromisoformat(case["processing_started_at"])
            total_seconds = (now - started).total_seconds()

            hours, remainder = divmod(int(total_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                human_readable = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                human_readable = f"{minutes}m {seconds}s"
            else:
                human_readable = f"{seconds}s"

            update_fields["total_processing_time"] = human_readable
            update_fields["total_processing_time_seconds"] = round(total_seconds, 2)
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not parse processing_started_at for case {case_id}: {e}")

    await col.find_one_and_update(
        {"_id": ObjectId(case_id)},
        {"$set": update_fields},
    )
    
def convert_datetime_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_datetime_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime_to_str(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj