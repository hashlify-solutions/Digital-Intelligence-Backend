# Standard 3rd party dependencies imports
from fastapi import APIRouter, Depends, HTTPException
from fastapi import UploadFile, BackgroundTasks, Form, HTTPException, Query, File, Body
from bson import ObjectId
import os
from pathlib import Path
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import math
from fastapi.encoders import jsonable_encoder
from datetime import datetime, timezone
import re
import json
import shutil

# Internal local modules imports
from schemas.case import GetMessagesByIdsRequest, RAGQueryAnalyticsRequest
from utils.auth import get_current_user
from config.db import (
    db,
    collection_case,
    users_collection,
    models_master_collection,
)
from setup import setup_logging
from clients.llama.llama_v1 import LlamaClient
from analyzer_v1 import ArabicSocialAnalyzer
from rag_v1 import ArabicRagAnalyzer
from utils import pipelines_v1
from config.settings import settings
from tasks.celery_tasks import process_csv_upload_v1
from utils.helpers import convert_datetime_to_str
from config.db import ufdr_files_collection

# Constants
UPLOAD_DIR = settings.upload_dir
ARCHIVE_DIR = "archived_data"
Path(ARCHIVE_DIR).mkdir(exist_ok=True)

# Router and Logger Setup
router = APIRouter()
logger = setup_logging()


# Helper functions
def sanitize_nan_values_recursive(data):
    """Recursively replaces NaN values with None in dictionaries and lists."""
    if isinstance(data, dict):
        return {k: sanitize_nan_values_recursive(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_nan_values_recursive(v) for v in data]
    elif isinstance(data, float) and math.isnan(data):  # Check if value is NaN
        return None  # Or return "Unknown" if preferred
    else:
        return data


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


# Routes
@router.post("/upload-data", response_model=dict)
async def upload_case_data(
    file: Optional[UploadFile] = File(None),
    topics: list[str] = Form(...),
    sentiments: list[str] = Form(...),
    interactions: list[str] = Form(...),
    entitiesClasses: list[str] = Form(...),
    noteClassifications: Optional[list[str]] = Form(None),
    browsingHistoryClassifications: Optional[list[str]] = Form(None),
    caseName: str = Form(...),
    category: str = Form(...),
    alert_id: Optional[str] = Form(None),
    is_rag: bool = Form(),
    models_profile_id: Optional[str] = Form(None),
    llama_basic_params: Optional[str] = Form(None),
    llama_advanced_params: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
    areaZone: Optional[str] = Form(None),
    is_llama_validation_enabled: Optional[bool] = Form(False),
    use_local_file_path: bool = Form(False),
    local_file_path: Optional[str] = Form(None),
    local_folder_path: Optional[str] = Form(None),
):
    try:
        # Validate inputs based on whether we use a local path or an uploaded file
        if use_local_file_path:
            if not local_file_path or not local_folder_path:
                raise HTTPException(
                    status_code=400,
                    detail="local_file_path and local_folder_path are required when use_local_file_path is true.",
                )
            if not os.path.isfile(local_file_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"local_file_path does not exist: {local_file_path}",
                )
            if not os.path.isdir(local_folder_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"local_folder_path does not exist: {local_folder_path}",
                )
            file_extension = local_file_path.rsplit(".", 1)[-1].lower()
            file_name = os.path.basename(local_file_path)
        else:
            if file is None:
                raise HTTPException(
                    status_code=400,
                    detail="file is required when use_local_file_path is false.",
                )
            file_extension = file.filename.split(".")[-1].lower()
            file_name = file.filename

        if file_extension not in ["csv", "ufdr"]:
            raise HTTPException(
                status_code=400, detail="Only CSV or UFDR files are allowed."
            )

        if not caseName:
            raise HTTPException(
                status_code=400, detail="Case name is required for creating a new case."
            )

        if is_rag:
            if len(topics) == 0 or len(sentiments) == 0 or len(interactions) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Topics, Sentiments, and Interactions are required for RAG data.",
                )

        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        logger.info(f"Processing file: {file_name} (local={use_local_file_path})")

        topics = topics[0].split(",") if topics and topics[0].split(",") != "" else []
        sentiments = (
            sentiments[0].split(",") if sentiments and sentiments[0] != "" else []
        )
        interactions = (
            interactions[0].split(",") if interactions and interactions[0] != "" else []
        )
        entitiesClasses = (
            entitiesClasses[0].split(",")
            if entitiesClasses and entitiesClasses[0] != ""
            else []
        )
        noteClassifications = (
            noteClassifications[0].split(",")
            if noteClassifications and noteClassifications[0] != ""
            else []
        )
        browsingHistoryClassifications = (
            browsingHistoryClassifications[0].split(",")
            if browsingHistoryClassifications
            and browsingHistoryClassifications[0] != ""
            else []
        )

        if models_profile_id:
            models_profile = await models_master_collection.find_one(
                {"_id": ObjectId(models_profile_id)}
            )
            if not models_profile:
                raise HTTPException(status_code=404, detail="Model Profile not found.")
        else:
            models_profile = await models_master_collection.find().to_list(None)
            models_profile = models_profile[0]
            logger.info("Using the default model profile from models_master collection")

        case_data = {
            "name": caseName,
            "status": "pending",
            "user_id": ObjectId(user_id),
            "topics": topics,
            "sentiments": sentiments,
            "interactions": interactions,
            "is_rag": is_rag,
            "model_profile": models_profile,
            "entitiesClasses": entitiesClasses,
            "category": category,
            "areaZone": areaZone,
            "noteClassifications": noteClassifications,
            "browsingHistoryClassifications": browsingHistoryClassifications,
            "alert_id": ObjectId(alert_id) if alert_id else None,
            "processing_started_at": datetime.now(timezone.utc).isoformat(),
        }
        case = await collection_case.insert_one(case_data)
        case_object_id = case.inserted_id

        ufdr_file_id = None
        if file_extension == "ufdr":
            ufdr_file_data = {
                "name": file_name,
                "caseId": case_object_id,
                "file_size": 0,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
            }
            ufdr_file = await ufdr_files_collection.insert_one(ufdr_file_data)
            ufdr_file_id = ufdr_file.inserted_id

        if use_local_file_path:
            folder_path = local_folder_path
            file_path = local_file_path
            local_file_size = os.path.getsize(file_path)
            logger.info(
                f"Using local file path: {file_path} (size: {local_file_size / (1024**3):.2f} GB)"
            )
            if file_extension == "ufdr" and ufdr_file_id:
                await ufdr_files_collection.update_one(
                    {"_id": ufdr_file_id},
                    {"$set": {"file_size": local_file_size, "updated_at": datetime.now()}},
                )
        else:
            if file_extension == "csv":
                folder_path = os.path.join(f"{UPLOAD_DIR}/{caseName}_{case_object_id}/csv")
            else:
                folder_path = os.path.join(
                    f"{UPLOAD_DIR}/{caseName}_{case_object_id}/ufdr/{ufdr_file.inserted_id}"
                )
            Path(folder_path).mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(folder_path, file.filename)
            chunk_size = 1024 * 1024
            total_bytes_written = 0
            with open(file_path, "wb") as f:
                while chunk := await file.read(chunk_size):
                    f.write(chunk)
                    total_bytes_written += len(chunk)
                    if total_bytes_written % (1024 * 1024 * 1024) == 0:
                        logger.info(
                            f"Written {total_bytes_written / (1024**3):.2f} GB of {file_extension.upper()} file: {file.filename}"
                        )
            logger.info(
                f"Successfully saved {file_extension.upper()} file to: {file_path} (Total size: {total_bytes_written / (1024**3):.2f} GB)"
            )
            if file_extension == "ufdr":
                await ufdr_files_collection.update_one(
                    {"_id": ufdr_file.inserted_id},
                    {
                        "$set": {
                            "file_size": total_bytes_written,
                            "updated_at": datetime.now(),
                        }
                    },
                )

        # Parse user-provided params if present
        user_basic_params = json.loads(llama_basic_params) if llama_basic_params else {}
        user_advanced_params = (
            json.loads(llama_advanced_params) if llama_advanced_params else {}
        )
        # Get defaults
        default_basic = getattr(
            LlamaClient,
            "DEFAULT_BASIC_PARAMS",
            {"temperature": 0.6, "num_ctx": 2048, "num_tokens": 512},
        )
        default_advanced = getattr(LlamaClient, "DEFAULT_ADVANCED_PARAMS", {})
        # Get from profile if present
        profile_llama = models_profile.get("llama", {})
        profile_basic = profile_llama.get("basic_params", {})
        profile_advanced = profile_llama.get("advanced_params", {})
        # Merge: default < profile < user
        basic_params = {**default_basic, **profile_basic, **user_basic_params}
        advanced_params = {
            **default_advanced,
            **profile_advanced,
            **user_advanced_params,
        }

        # Converting models_profile to a serializable format
        models_profile_serializable = {
            "_id": str(models_profile.get("_id", "")),
            "classifier": models_profile.get("classifier", {}),
            "toxic": models_profile.get("toxic", {}),
            "emotion": models_profile.get("emotion", {}),
            "embeddings": models_profile.get("embeddings", {}),
            "llama": models_profile.get("llama", {}),
        }

        # Executing the celery task
        process_csv_upload_v1.delay(
            folder_path=folder_path,
            file_path=file_path,
            file_extension=file_extension,
            case_id=str(case_object_id),
            ufdr_file_id=str(ufdr_file_id),
            case_name=caseName,
            alert_id=str(alert_id) if alert_id else None,
            topics=topics,
            sentiments=sentiments,
            interactions=interactions,
            entities_classes=entitiesClasses,
            is_rag=is_rag,
            models_profile=models_profile_serializable,
            note_classifications=noteClassifications,
            browsing_history_classifications=browsingHistoryClassifications,
            is_llama_validation_enabled=True,
        )

        return JSONResponse(
            content={
                "message": "File uploaded successfully. Processing started.",
                "case_id": str(case.inserted_id),
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Error uploading CSV/UFDR file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading CSV/UFDR file: {str(e)}"
        )


@router.get("/cases-all")
async def get_all_cases(user_id: str = Depends(get_current_user)):
    cases = []
    items = await collection_case.find({"user_id": ObjectId(user_id)}).to_list(None)
    for case in items:
        collection = db[f"{case['name']}_{case['_id']}"]
        alert_count = await collection.count_documents({"alert": True})

        def convert_objectid_to_str(obj):
            if isinstance(obj, dict):
                return {k: convert_objectid_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_objectid_to_str(item) for item in obj]
            elif isinstance(obj, ObjectId):
                return str(obj)
            return obj

        case = convert_objectid_to_str(case)
        case["alert_count"] = alert_count

        # Calculate analysis duration if both start and completion times exist
        if case.get("analysis_started_at") and case.get("analysis_completed_at"):
            try:
                start_time = datetime.fromisoformat(case["analysis_started_at"])
                end_time = datetime.fromisoformat(case["analysis_completed_at"])
                # Convert both to UTC if they have tzinfo, else treat as naive
                if start_time.tzinfo and end_time.tzinfo:
                    start_time = start_time.astimezone(timezone.utc)
                    end_time = end_time.astimezone(timezone.utc)
                duration = end_time - start_time
                # Ensure non-negative duration
                if duration.total_seconds() < 0:
                    duration = abs(duration)
                minutes = int(duration.total_seconds() // 60)
                seconds = int(duration.total_seconds() % 60)
                case["analysis_duration"] = f"{minutes}m {seconds}s"
            except Exception as e:
                case["analysis_duration"] = "0m 0s"
        else:
            case["analysis_duration"] = "0m 0s"

        # Ensure total_messages is included in the response
        if "total_messages" not in case:
            case["total_messages"] = 0

        cases.append(case)

    return JSONResponse(
        content=convert_datetime_to_str(cases),
        status_code=200,
    )


@router.get("/{case_id}/classification-options")
async def get_case_classification_options(
    case_id: str, _: str = Depends(get_current_user)
):
    """Get configured classification options for notes and browsing history for a case."""
    logger.info(f"Attempting to get classification options for case_id: {case_id}")
    logger.info(f"Current user: {_}")

    try:
        # Validate ObjectId format first
        try:
            object_id = ObjectId(case_id)
            logger.info(f"Valid ObjectId format: {object_id}")
        except Exception as e:
            logger.error(f"Invalid ObjectId format for case_id '{case_id}': {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid ObjectId format: {case_id}"
            )

        # Check if case exists
        case = await collection_case.find_one({"_id": object_id})
        if not case:
            logger.error(f"Case not found with _id: {object_id}")
            # Let's also check if there are any cases at all
            total_cases = await collection_case.count_documents({})
            logger.info(f"Total cases in collection: {total_cases}")

            # Check if there are any cases with similar IDs
            similar_cases = await collection_case.find({}).limit(5).to_list(5)
            logger.info(f"Sample cases: {[str(c.get('_id')) for c in similar_cases]}")

            raise HTTPException(
                status_code=404, detail=f"Case not found with ID: {case_id}"
            )

        logger.info(f"Case found: {case.get('name')} with ID: {str(case['_id'])}")

        note_classes = case.get("noteClassifications", []) or []
        browsing_classes = case.get("browsingHistoryClassifications", []) or []

        logger.info(f"Note classifications: {note_classes}")
        logger.info(f"Browsing history classifications: {browsing_classes}")

        return JSONResponse(
            content={
                "noteClassifications": note_classes,
                "browsingHistoryClassifications": browsing_classes,
            },
            status_code=200,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_case_classification_options: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{case_id}/classified-data")
async def get_case_classified_data(
    case_id: str,
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(10, alias="limit", le=100),
    message_type: Optional[str] = Query(
        None, description="Filter by message type: 'note' or 'browsing_history'"
    ),
    classification: Optional[str] = Query(
        None, description="Filter by specific classification category"
    ),
    _: str = Depends(get_current_user),
):
    """Get the actual classified data for notes and browsing history messages."""
    logger.info(
        f"Getting classified data for case_id: {case_id}, message_type: {message_type}, classification: {classification}"
    )

    try:
        # Validate ObjectId format first
        try:
            object_id = ObjectId(case_id)
            logger.info(f"Valid ObjectId format: {object_id}")
        except Exception as e:
            logger.error(f"Invalid ObjectId format for case_id '{case_id}': {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid ObjectId format: {case_id}"
            )

        # Check if case exists
        case = await collection_case.find_one({"_id": object_id})
        if not case:
            logger.error(f"Case not found with _id: {object_id}")
            raise HTTPException(
                status_code=404, detail=f"Case not found with ID: {case_id}"
            )

        logger.info(f"Case found: {case.get('name')} with ID: {str(case['_id'])}")

        # Get the collection for this case
        collection = db[f"{case['name']}_{case_id}"]

        # Build query based on filters
        query = {}

        # Filter by message type if specified
        if message_type:
            if message_type.lower() not in ["note", "browsing_history"]:
                raise HTTPException(
                    status_code=400,
                    detail="message_type must be 'note' or 'browsing_history'",
                )
            query["Message Type"] = message_type.lower()
        else:
            # If no message type specified, get both
            query["Message Type"] = {"$in": ["note", "browsing_history"]}

        # Filter by classification if specified
        if classification:
            # Check if the classification exists in the case configuration
            note_classes = case.get("noteClassifications", [])
            browsing_classes = case.get("browsingHistoryClassifications", [])
            all_classes = note_classes + browsing_classes

            if classification not in all_classes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Classification '{classification}' not found in case configuration. Available: {all_classes}",
                )

            # Add classification filter based on message type
            if message_type == "note" or (
                not message_type and classification in note_classes
            ):
                query["analysis_summary.note_classification"] = classification
            elif message_type == "browsing_history" or (
                not message_type and classification in browsing_classes
            ):
                query["analysis_summary.browsing_history_classification"] = (
                    classification
                )

        # Get total count for pagination
        total_count = await collection.count_documents(query)
        logger.info(f"Found {total_count} messages matching query: {query}")

        if total_count == 0:
            return JSONResponse(
                content={
                    "data": [],
                    "pagination": {
                        "total": 0,
                        "page": page,
                        "limit": limit,
                        "total_pages": 0,
                    },
                    "message": "No classified messages found matching the criteria",
                },
                status_code=200,
            )

        # Apply pagination
        skip = (page - 1) * limit

        # Get the classified messages
        messages_cursor = collection.find(query).skip(skip).limit(limit).sort("_id", -1)
        messages = await messages_cursor.to_list(limit)

        # Process the messages
        classified_data = []
        for message in messages:
            # Convert ObjectId to string
            message["_id"] = str(message["_id"])
            if "case_id" in message:
                message["case_id"] = str(message["case_id"])

            # Extract classification information
            analysis_summary = message.get("analysis_summary", {})

            # Get the relevant classification based on message type
            msg_type = message.get("Message Type", "").lower()
            if msg_type == "note":
                classification_result = analysis_summary.get("note_classification")
                classification_type = "note_classification"
            elif msg_type == "browsing_history":
                classification_result = analysis_summary.get(
                    "browsing_history_classification"
                )
                classification_type = "browsing_history_classification"
            else:
                classification_result = None
                classification_type = None

            # Create a clean message object with classification info
            clean_message = {
                "_id": message["_id"],
                "Message Type": message.get("Message Type"),
                "timestamp": message.get("timestamp"),
                "classification_type": classification_type,
                "classification_result": classification_result,
                "confidence_score": None,  # Will be filled if available
                "text_content": (
                    message.get("Preview Text", "")[:200] + "..."
                    if message.get("Preview Text")
                    else ""
                ),
                "additional_fields": {},
            }

            # Add message type specific fields
            if msg_type == "note":
                clean_message["additional_fields"] = {
                    "title": message.get("Title", ""),
                    "body": message.get("Body", ""),
                    "summary": message.get("Summary", ""),
                }
            elif msg_type == "browsing_history":
                clean_message["additional_fields"] = {
                    "url": message.get("url", ""),
                    "title": message.get("title", ""),
                    "search_value": message.get("search_value", ""),
                }

            # Try to get confidence score if available
            if classification_result and "content_analysis" in analysis_summary:
                content_analysis = analysis_summary["content_analysis"]
                if classification_type in content_analysis:
                    classifications = content_analysis[classification_type]
                    if isinstance(classifications, list) and len(classifications) > 0:
                        clean_message["confidence_score"] = classifications[0].get(
                            "confidence", None
                        )

            classified_data.append(clean_message)

        # Sanitize any NaN values
        classified_data = sanitize_nan_values_recursive(classified_data)

        return JSONResponse(
            content={
                "data": classified_data,
                "pagination": {
                    "total": total_count,
                    "page": page,
                    "limit": limit,
                    "total_pages": math.ceil(total_count / limit),
                },
                "filters_applied": {
                    "message_type": message_type,
                    "classification": classification,
                },
                "case_info": {
                    "case_id": str(case["_id"]),
                    "case_name": case["name"],
                    "noteClassifications": case.get("noteClassifications", []),
                    "browsingHistoryClassifications": case.get(
                        "browsingHistoryClassifications", []
                    ),
                },
            },
            status_code=200,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_case_classified_data: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/case/{case_id}/debug-classifications")
async def debug_case_classifications(case_id: str, _: str = Depends(get_current_user)):
    """Debug endpoint to check case classifications and processed documents."""
    try:
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    # Get case info
    case_info = {
        "case_id": str(case["_id"]),
        "name": case["name"],
        "noteClassifications": case.get("noteClassifications", []),
        "browsingHistoryClassifications": case.get(
            "browsingHistoryClassifications", []
        ),
        "topics": case.get("topics", []),
        "sentiments": case.get("sentiments", []),
        "interactions": case.get("interactions", []),
        "entitiesClasses": case.get("entitiesClasses", []),
    }

    # Get collection and check for note/browsing_history documents
    collection = db[f"{case['name']}_{case_id}"]

    # Count documents by Message Type
    message_types = await collection.aggregate(
        [{"$group": {"_id": "$Message Type", "count": {"$sum": 1}}}]
    ).to_list(None)

    # Get sample note and browsing_history documents
    note_sample = await collection.find({"Message Type": "note"}).limit(1).to_list(1)
    browsing_sample = (
        await collection.find({"Message Type": "browsing_history"}).limit(1).to_list(1)
    )

    # Check processed documents
    processed_count = await collection.count_documents({"processed": True})
    total_count = await collection.count_documents({})

    debug_info = {
        "case_info": case_info,
        "message_type_counts": message_types,
        "processed_stats": {
            "processed": processed_count,
            "total": total_count,
            "unprocessed": total_count - processed_count,
        },
        "note_sample": note_sample[0] if note_sample else None,
        "browsing_sample": browsing_sample[0] if browsing_sample else None,
    }

    return JSONResponse(
        content=debug_info,
        status_code=200,
    )


@router.get("/awake-all")
async def get_all_cases(background_tasks: BackgroundTasks):
    text = "How are you?"
    analyzer = ArabicSocialAnalyzer("dawdaw", collection_case, "wdawdawd", "wdawdad")
    background_tasks.add_task(analyzer.analyze_content, text)
    return JSONResponse(
        content="Success",
        status_code=200,
    )


@router.get("/case/{case_id}")
async def get_case(
    case_id: str,
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(10, alias="limit", le=100),
    _: str = Depends(get_current_user),
):
    try:
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    try:
        collection = db[f"{case['name']}_{case_id}"]

        total_count = await collection.count_documents(
            {"case_id": ObjectId(case_id)}
        )  # Total items
        skip = (page - 1) * limit

        case_data_cursor = (
            collection.find({"case_id": ObjectId(case_id)}).skip(skip).limit(limit)
        )

        case_data = []
        async for data in case_data_cursor:
            data["_id"] = str(data["_id"])
            data["case_id"] = str(data["case_id"])
            data = sanitize_nan_values(data)
            case_data.append(data)

    except Exception:
        raise HTTPException(status_code=404, detail="Collection not found")

    return JSONResponse(
        content={
            "data": case_data,
            "pagination": {
                "total": total_count,
                "page": page,
                "limit": limit,
                "total_pages": (total_count + limit - 1) // limit,  # Ceiling division
            },
        },
        status_code=200,
    )


@router.get("/case-paginated/{case_id}")
async def get_case(
    case_id: str,
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(10, alias="limit", le=100),
    _: str = Depends(get_current_user),
):
    try:
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    try:
        collection = db[f"{case['name']}_{case_id}"]

        total_count = await collection.count_documents(
            {"case_id": ObjectId(case_id)}
        )  # Total items
        skip = (page - 1) * limit

        case_data_cursor = (
            collection.find({"case_id": ObjectId(case_id)}).skip(skip).limit(limit)
        )

        case_data = []
        async for data in case_data_cursor:
            data["_id"] = str(data["_id"])
            data["case_id"] = str(data["case_id"])
            data = sanitize_nan_values(data)
            case_data.append(data)

    except Exception:
        raise HTTPException(status_code=404, detail="Collection not found")

    return JSONResponse(
        content={
            "data": case_data,
            "pagination": {
                "total": total_count,
                "page": page,
                "limit": limit,
                "total_pages": (total_count + limit - 1) // limit,  # Ceiling division
            },
        },
        status_code=200,
    )


@router.get("/alert-messages/{case_id}")
async def get_alert_messages(case_id: str, _: str = Depends(get_current_user)):
    try:
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")
    logger.info(f"Case found: {case}")
    try:
        collection = db[f"{case['name']}_{case_id}"]
        print(f"Querying collection: {case['name']}_{case_id}")
        logger.info(f"Querying collection: {case['name']}_{case_id}")
        case_data_cursor = await collection.find(
            {"case_id": ObjectId(case_id), "alert": True}
        ).to_list(None)
        case_data = []
        for data in case_data_cursor:
            data["_id"] = str(data["_id"])
            data["case_id"] = str(data["case_id"])
            data = sanitize_nan_values(data)
            case_data.append(data)
    except Exception:
        raise HTTPException(status_code=404, detail="Collection not found")

    return JSONResponse(
        content=case_data,
        status_code=200,
    )


@router.get("/message-filter/{case_id}")
async def get_filtered_messages(
    case_id: str,
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(3, alias="limit", le=100),
    sematic_search: Optional[str] = Query(None),
    top_topic: Optional[List[str]] = Query(None),
    toxicity_score: Optional[int] = Query(None),
    sentiment_aspect: Optional[List[str]] = Query(None),
    emotion: Optional[List[str]] = Query(None),
    language: Optional[List[str]] = Query(None),
    risk_level: Optional[List[str]] = Query(None),
    application_type: Optional[List[str]] = Query(None),
    applications: Optional[List[str]] = Query(None),
    interaction_type: Optional[List[str]] = Query(None),
    entities: Optional[List[str]] = Query(None),
    entities_classes: Optional[List[str]] = Query(None),
    alert: Optional[bool] = Query(None),
    to: Optional[List[str]] = Query(None),
    from_: Optional[List[str]] = Query(None, alias="from"),
    _: str = Depends(get_current_user),
):
    try:
        # Validate ObjectId
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    query = {}
    if sematic_search and sematic_search.strip() != "":
        model_profile = case.get("model_profile", {})

        if not isinstance(model_profile, dict):
            try:
                if isinstance(model_profile, str):
                    models_profile = await models_master_collection.find_one(
                        {"_id": ObjectId(model_profile)}
                    )
                    if not models_profile:
                        raise HTTPException(
                            status_code=404, detail="Model Profile not found."
                        )
                else:
                    models_profile = model_profile
            except Exception as e:
                models_profile = await models_master_collection.find().to_list(None)
                if len(models_profile) > 1:
                    models_profile = models_profile[1]
                else:
                    raise HTTPException(
                        status_code=404, detail="No model profiles available."
                    )
        else:
            models_profile = model_profile

        collection = db[f"{case['name']}_{case_id}"]
        rag = ArabicRagAnalyzer(collection, collection_case, case_id, models_profile)
        items = rag.semantic_search(sematic_search)
        if not items or len(items) == 0:
            raise HTTPException(
                status_code=404, detail="No messages found with the given filters"
            )
        logger.info(f"Items found: {items}")
        mongo_ids = [item.payload.get("mongo_id") for item in items]
        query["_id"] = {"$in": [ObjectId(id) for id in mongo_ids]}
        logger.info(f"Query after semantic search: {query}")

    # Process filter parameters
    top_topic = (
        top_topic[0].split(",") if top_topic and top_topic[0].split(",") != "" else None
    )
    toxicity_score = toxicity_score if toxicity_score else None
    sentiment_aspect = (
        sentiment_aspect[0].split(",")
        if sentiment_aspect and sentiment_aspect[0] != ""
        else None
    )
    emotion = emotion[0].split(",") if emotion and emotion[0] != "" else None
    language = language[0].split(",") if language and language[0] != "" else None
    risk_level = (
        risk_level[0].split(",") if risk_level and risk_level[0] != "" else None
    )
    application_type = (
        application_type[0].split(",")
        if application_type and application_type[0] != ""
        else None
    )
    applications = (
        applications[0].split(",") if applications and applications[0] != "" else None
    )
    interaction_type = (
        interaction_type[0].split(",")
        if interaction_type and interaction_type[0] != ""
        else None
    )
    entities = entities if entities and len(entities) > 0 else None
    entities_classes = (
        entities_classes if entities_classes and len(entities_classes) > 0 else None
    )
    alert = alert if alert else None
    to_filter = to[0].split(",") if to and to[0] != "" else None
    from_filter = from_[0].split(",") if from_ and from_[0] != "" else None

    # Build query filters
    if top_topic:
        query["analysis_summary.top_topic"] = {"$in": top_topic}
    if toxicity_score:
        query["analysis_summary.toxicity_score"] = {"$gte": int(toxicity_score)}
    if sentiment_aspect:
        query["analysis_summary.sentiment_aspect"] = {"$in": sentiment_aspect}
    if emotion:
        query["analysis_summary.emotion"] = {"$in": emotion}
    if language:
        query["analysis_summary.language"] = {"$in": language}
    if risk_level:
        query["analysis_summary.risk_level"] = {"$in": risk_level}

    # Handle application_type and applications filters
    if application_type or applications:
        app_filters = []

        # Handle application_type filter
        if application_type:
            application_mapping = {
                "browsing_history": [
                    "WebBookmark",
                    "VisitedPage",
                    "BrowserHistory",
                    "Search",
                ],
                "location": ["Location Services"],
                "email": ["Email"],
                "sms/mms": ["SMS/MMS"],
                "message": ["SMS/MMS"],
                "chat": ["Chat"],
                "contact": ["Contact"],
                "call": ["call"],
                "note": ["Note"],
            }

            # Special handling for calls
            if "call" in [at.lower() for at in application_type]:
                query["Message Type"] = {"$regex": "^call$", "$options": "i"}
                application_type = [
                    app for app in application_type if app.lower() != "call"
                ]

            expanded_applications = []
            for app_type in application_type:
                if app_type.lower() in application_mapping:
                    expanded_applications.extend(application_mapping[app_type.lower()])
                else:
                    expanded_applications.append(app_type)

            # Special handling for 'message' type
            if "message" in [a.lower() for a in application_type]:
                app_filters.append(
                    {"Message Type": {"$regex": "^message$", "$options": "i"}}
                )

            if expanded_applications:
                app_filters.append(
                    {
                        "$or": [
                            {"Application": {"$regex": f"^{app}$", "$options": "i"}}
                            for app in expanded_applications
                        ]
                    }
                )

        # Handle applications filter
        if applications:
            if not app_filters:
                # If no application_type filter, just filter by applications
                app_filters.append(
                    {
                        "$or": [
                            {"Application": {"$regex": f"^{app}$", "$options": "i"}}
                            for app in applications
                        ]
                    }
                )
            else:
                # For combined filters, we need to intersect the applications
                app_regex_list = [
                    {"$regex": f"^{app}$", "$options": "i"} for app in applications
                ]

                # Find and modify the application filter
                for i, filt in enumerate(app_filters):
                    if "$or" in filt:  # This is our application filter
                        # Keep only the applications that match our filter
                        filtered_apps = [
                            app_cond
                            for app_cond in filt["$or"]
                            if any(
                                app_cond["Application"]["$regex"].lower()
                                == f"^{app}$".lower()
                                for app in applications
                            )
                        ]
                        app_filters[i]["$or"] = filtered_apps

        # Combine all application filters
        if app_filters:
            if application_type and applications:
                # We want messages that match BOTH application and type
                query["$and"] = [
                    {
                        "$or": [
                            {"Application": {"$regex": f"^{app}$", "$options": "i"}}
                            for app in applications
                        ]
                    },
                    {"Message Type": {"$regex": "^message$", "$options": "i"}},
                ]
            elif application_type:
                # When only application_type is provided, use OR logic between type and apps
                query["$or"] = app_filters
            else:
                # When only applications is provided
                query.update(app_filters[0])

    if interaction_type:
        query["analysis_summary.interaction_type"] = {"$in": interaction_type}
    if entities:
        query["analysis_summary.entities"] = {"$in": entities}
    if entities_classes:
        and_conditions = []
        for class_name in entities_classes:
            and_conditions.append(
                {
                    f"analysis_summary.entities_classification.{class_name}": {
                        "$exists": True,
                        "$ne": [],
                    }
                }
            )
        query["$and"] = and_conditions

    if alert is not None:
        query["alert"] = alert

    if to_filter:
        query["To"] = {"$in": to_filter}

    if from_filter:
        query["From"] = {"$in": from_filter}

    casesItems = []
    try:
        collection = db[f"{case['name']}_{case_id}"]
        logger.info(f"Querying collection: {case['name']}_{case_id} with query {query}")

        total_count = await collection.count_documents(query)
        if total_count == 0:
            return JSONResponse(
                content={
                    "data": [],
                    "pagination": {
                        "total": 0,
                        "page": page,
                        "limit": limit,
                        "total_pages": 0,
                    },
                },
                status_code=200,
            )

        skip = (page - 1) * limit
        case_data_cursor = (
            await collection.find(query).skip(skip).limit(limit).to_list(None)
        )
        logger.info(f"Found {len(case_data_cursor)} messages for page {page}")

        for data in case_data_cursor:
            data["_id"] = str(data["_id"])
            data["case_id"] = str(data["case_id"])
            data = sanitize_nan_values(data)
            casesItems.append(data)
    except Exception as e:
        logger.error(f"Error querying collection: {e}")
        raise HTTPException(status_code=404, detail="Collection not found")

    return JSONResponse(
        content={
            "data": casesItems,
            "pagination": {
                "total": total_count,
                "page": page,
                "limit": limit,
                "total_pages": math.ceil(total_count / limit),
            },
        },
        status_code=200,
    )


def _build_message_filter_query(params: Dict[str, Any]) -> Dict:
    """Build query dict from parsed params (compatible with get_filtered_messages parsing)."""
    query: Dict[str, Any] = params.get("base_query", {}) or {}

    top_topic = params.get("top_topic")
    toxicity_score = params.get("toxicity_score")
    sentiment_aspect = params.get("sentiment_aspect")
    emotion = params.get("emotion")
    language = params.get("language")
    risk_level = params.get("risk_level")
    application_type = params.get("application_type")
    applications = params.get("applications")
    interaction_type = params.get("interaction_type")
    entities = params.get("entities")
    entities_classes = params.get("entities_classes")
    alert = params.get("alert")
    to_filter = params.get("to")
    from_filter = params.get("from")

    if top_topic:
        query["analysis_summary.top_topic"] = {"$in": top_topic}
    if toxicity_score:
        query["analysis_summary.toxicity_score"] = {"$gte": int(toxicity_score)}
    if sentiment_aspect:
        query["analysis_summary.sentiment_aspect"] = {"$in": sentiment_aspect}
    if emotion:
        query["analysis_summary.emotion"] = {"$in": emotion}
    if language:
        query["analysis_summary.language"] = {"$in": language}
    if risk_level:
        query["analysis_summary.risk_level"] = {"$in": risk_level}

    if application_type or applications:
        app_filters = []
        if application_type:
            application_mapping = {
                "browsing_history": [
                    "WebBookmark",
                    "VisitedPage",
                    "BrowserHistory",
                    "Search",
                ],
                "location": ["Location Services"],
                "email": ["Email"],
                "sms/mms": ["SMS/MMS"],
                "message": ["SMS/MMS"],
                "chat": ["Chat"],
                "contact": ["Contact"],
                "call": ["call"],
                "note": ["Note"],
            }
            if any(at.lower() == "call" for at in application_type):
                query["Message Type"] = {"$regex": "^call$", "$options": "i"}
                application_type = [
                    app for app in application_type if app.lower() != "call"
                ]
            expanded_applications = []
            for app_type in application_type:
                if app_type.lower() in application_mapping:
                    expanded_applications.extend(application_mapping[app_type.lower()])
                else:
                    expanded_applications.append(app_type)
            if "message" in [a.lower() for a in application_type]:
                app_filters.append(
                    {"Message Type": {"$regex": "^message$", "$options": "i"}}
                )
            if expanded_applications:
                app_filters.append(
                    {
                        "$or": [
                            {"Application": {"$regex": f"^{app}$", "$options": "i"}}
                            for app in expanded_applications
                        ]
                    }
                )

        if applications:
            if not app_filters:
                app_filters.append(
                    {
                        "$or": [
                            {"Application": {"$regex": f"^{app}$", "$options": "i"}}
                            for app in applications
                        ]
                    }
                )
            else:
                app_filters = app_filters + [
                    {
                        "$or": [
                            {"Application": {"$regex": f"^{app}$", "$options": "i"}}
                            for app in applications
                        ]
                    }
                ]

        if app_filters:
            if application_type and applications:
                query["$and"] = [
                    {
                        "$or": [
                            {"Application": {"$regex": f"^{app}$", "$options": "i"}}
                            for app in applications
                        ]
                    },
                    {"Message Type": {"$regex": "^message$", "$options": "i"}},
                ]
            elif application_type:
                query["$or"] = app_filters
            else:
                query.update(app_filters[0])

    if interaction_type:
        query["analysis_summary.interaction_type"] = {"$in": interaction_type}
    if entities:
        query["analysis_summary.entities"] = {"$in": entities}
    if entities_classes:
        and_conditions = []
        for class_name in entities_classes:
            and_conditions.append(
                {
                    f"analysis_summary.entities_classification.{class_name}": {
                        "$exists": True,
                        "$ne": [],
                    }
                }
            )
        query["$and"] = and_conditions
    if alert is not None:
        query["alert"] = alert

    if to_filter:
        query["To"] = {"$in": to_filter}

    if from_filter:
        query["From"] = {"$in": from_filter}

    return query


@router.get("/message-filter-analysis/{case_id}")
async def get_filtered_messages_analysis(
    case_id: str,
    sematic_search: Optional[str] = Query(None),
    top_topic: Optional[List[str]] = Query(None),
    toxicity_score: Optional[int] = Query(None),
    sentiment_aspect: Optional[List[str]] = Query(None),
    emotion: Optional[List[str]] = Query(None),
    language: Optional[List[str]] = Query(None),
    risk_level: Optional[List[str]] = Query(None),
    application_type: Optional[List[str]] = Query(None),
    applications: Optional[List[str]] = Query(None),
    interaction_type: Optional[List[str]] = Query(None),
    entities: Optional[List[str]] = Query(None),
    entities_classes: Optional[List[str]] = Query(None),
    alert: Optional[bool] = Query(None),
    to: Optional[List[str]] = Query(None),
    from_: Optional[List[str]] = Query(None, alias="from"),
    _: str = Depends(get_current_user),
):
    try:
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    base_query = {}
    if sematic_search and sematic_search.strip() != "":
        model_profile = case.get("model_profile", {})
        models_profile = (
            model_profile if isinstance(model_profile, dict) else model_profile
        )
        collection = db[f"{case['name']}_{case_id}"]
        rag = ArabicRagAnalyzer(collection, collection_case, case_id, models_profile)
        items = rag.semantic_search(sematic_search)
        if not items or len(items) == 0:
            return JSONResponse(
                content={"error": "No messages found with the given filters"},
                status_code=404,
            )
        mongo_ids = [item.payload.get("mongo_id") for item in items]
        base_query["_id"] = {"$in": [ObjectId(id) for id in mongo_ids]}

    params = {
        "base_query": base_query,
        "top_topic": (
            top_topic[0].split(",") if top_topic and top_topic[0] != "" else None
        ),
        "toxicity_score": toxicity_score if toxicity_score else None,
        "sentiment_aspect": (
            sentiment_aspect[0].split(",")
            if sentiment_aspect and sentiment_aspect[0] != ""
            else None
        ),
        "emotion": emotion[0].split(",") if emotion and emotion[0] != "" else None,
        "language": language[0].split(",") if language and language[0] != "" else None,
        "risk_level": (
            risk_level[0].split(",") if risk_level and risk_level[0] != "" else None
        ),
        "application_type": (
            application_type[0].split(",")
            if application_type and application_type[0] != ""
            else None
        ),
        "applications": (
            applications[0].split(",")
            if applications and applications[0] != ""
            else None
        ),
        "interaction_type": (
            interaction_type[0].split(",")
            if interaction_type and interaction_type[0] != ""
            else None
        ),
        "entities": entities if entities and len(entities) > 0 else None,
        "entities_classes": (
            entities_classes if entities_classes and len(entities_classes) > 0 else None
        ),
        "alert": alert if alert else None,
        "to": to[0].split(",") if to and to[0] != "" else None,
        "from": from_[0].split(",") if from_ and from_[0] != "" else None,
    }

    query = _build_message_filter_query(params)
    collection = db[f"{case['name']}_{case_id}"]

    from utils.pipelines_v1 import (
        bar_and_pipe_chart_pipeline,
        heatmap_of_appilication_against_emotion_piepline,
        stacked_bar_risk_per_lanaguage,
        top_card_data_pipeline,
    )

    match_stage = {"$match": query} if query else {"$match": {}}

    bar_pipeline = [match_stage] + bar_and_pipe_chart_pipeline
    bar_result = await collection.aggregate(bar_pipeline).to_list(length=1)

    heatmap_pipeline = [match_stage] + heatmap_of_appilication_against_emotion_piepline
    heatmap_result = await collection.aggregate(heatmap_pipeline).to_list(length=100)

    stacked_pipeline = [match_stage] + stacked_bar_risk_per_lanaguage
    stacked_result = await collection.aggregate(stacked_pipeline).to_list(length=100)

    top_cards_pipeline = [match_stage] + top_card_data_pipeline
    top_cards_result = await collection.aggregate(top_cards_pipeline).to_list(length=1)

    # Use message-level timestamp when available (e.g., 'Date'), then social_media_analysis.text_metadata.timestamp,
    # then fallback to ingestion_timestamp. This prevents all docs from sharing the same ingestion time.
    timeline_pipeline = [
        match_stage,
        {
            "$addFields": {
                "ts": {
                    "$ifNull": [
                        {
                            "$dateFromString": {
                                "dateString": {"$toString": "$Date"},
                                "onError": None,
                            }
                        },
                        {
                            "$dateFromString": {
                                "dateString": {
                                    "$substrBytes": [{"$toString": "$Name"}, 0, 19]
                                },
                                "format": "%Y-%m-%d %H:%M:%S",
                                "timezone": "UTC",
                                "onError": None,
                            }
                        },
                        {
                            "$dateFromString": {
                                "dateString": {
                                    "$toString": "$social_media_analysis.text_metadata.timestamp"
                                },
                                "onError": None,
                            }
                        },
                        {"$toDate": "$ingestion_timestamp"},
                    ]
                }
            }
        },
        {
            "$project": {
                "day": {"$dateToString": {"format": "%Y-%m-%d", "date": "$ts"}},
                "risk_score": {
                    "$switch": {
                        "branches": [
                            {
                                "case": {
                                    "$eq": ["$analysis_summary.risk_level", "high"]
                                },
                                "then": 3,
                            },
                            {
                                "case": {
                                    "$eq": ["$analysis_summary.risk_level", "medium"]
                                },
                                "then": 2,
                            },
                            {
                                "case": {
                                    "$eq": ["$analysis_summary.risk_level", "low"]
                                },
                                "then": 1,
                            },
                        ],
                        "default": 0,
                    }
                },
                "toxicity": {
                    "$ifNull": [{"$toDouble": "$analysis_summary.toxicity_score"}, 0.0]
                },
            }
        },
        {
            "$group": {
                "_id": "$day",
                "count": {"$sum": 1},
                "avg_risk_level": {"$avg": "$risk_score"},
                "avg_toxicity": {"$avg": "$toxicity"},
            }
        },
        {"$sort": {"_id": 1}},
    ]
    timeline_result = await collection.aggregate(timeline_pipeline).to_list(length=1000)

    response = {
        "bar_and_donut": bar_result[0] if bar_result else {},
        "heatmap": {"data": heatmap_result},
        "stacked_bar": {"data": stacked_result},
        "top_cards": top_cards_result[0] if top_cards_result else {},
        "timeline": timeline_result,
    }

    # Sanitize any NaN/Infinity values to ensure JSON compliance
    response = sanitize_nan_values_recursive(response)

    return JSONResponse(content=jsonable_encoder(response), status_code=200)


@router.post("/test-analyzer")
async def test_analysis(text: str):
    analyzer = ArabicSocialAnalyzer("dawdaw", collection_case, "wdawdawd", "awdawdawd")
    result = analyzer.analyze_content("جاك الذي يعيش في كراتشي طبيب جيد")
    return JSONResponse(
        content=result,
        status_code=200,
    )


@router.post("/analyze-user-rag-query")
async def analyze_user_rag_query(
    case_id: str = Body(...), query: str = Body(...), _: str = Depends(get_current_user)
):
    """
    Analyze user RAG query using only the llama_basic_params, llama_advanced_params, and prompt
    stored in the model_profile object inside the case document.
    """
    try:
        print(f"Analyzing RAG query for case_id: {case_id} with query: {query}")
        # Validate and fetch the case
        try:
            obj_case_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid case_id format. Must be a valid ObjectId string.",
            )
        case = await collection_case.find_one({"_id": obj_case_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        collection = db[f"{case['name']}_{case_id}"]
        print("case", case)
        # Get model_profile directly from the case (already an object)
        model_profile = case.get("model_profile", {})
        if not isinstance(model_profile, dict) or not model_profile:
            raise HTTPException(
                status_code=404, detail="Model profile not found in case."
            )

        # Extract Llama params and prompt from model_profile
        llama_profile = model_profile.get("llama", {})
        llama_basic_params = llama_profile.get("basic_params", {})
        llama_advanced_params = llama_profile.get("advanced_params", {})
        llama_prompt = llama_profile.get("prompt", "")

        # Fallback to defaults if needed
        default_basic = getattr(
            LlamaClient,
            "DEFAULT_BASIC_PARAMS",
            {"temperature": 0.6, "num_ctx": 2048, "num_tokens": 512},
        )
        default_advanced = getattr(LlamaClient, "DEFAULT_ADVANCED_PARAMS", {})
        basic_params = {**default_basic, **llama_basic_params}
        advanced_params = {**default_advanced, **llama_advanced_params}

        # Get chat history from database
        chat_collection = db["chat_histories"]
        chat_history = (
            await chat_collection.find({"case_id": obj_case_id, "user_id": ObjectId(_)})
            .sort("timestamp", -1)
            .limit(10)
            .to_list(None)
        )

        analyzer = ArabicRagAnalyzer(
            collection, collection_case, case_id, model_profile
        )
        analyzer.llama_basic_params = basic_params
        analyzer.llama_advanced_params = advanced_params
        analyzer.chat_history = [
            {
                "query": chat["query"],
                "response": chat["response"],
                "mongo_ids": chat.get("mongo_ids", []),
            }
            for chat in reversed(chat_history)
        ]

        # Use async summarization API
        summary = await analyzer.summarize_messages_async(query)

        if summary is None or summary.get("success") == False:
            raise HTTPException(status_code=404, detail="Failed to generate summary.")

        logger.info(f"Summary Response: {summary}")

        # Save chat history in a single collection
        chat_entry = {
            "case_id": obj_case_id,
            "case_name": case["name"],
            "user_id": ObjectId(_),
            "query": query,
            "response": summary.get("summary"),
            "mongo_ids": summary.get("mongo_ids"),
            "timestamp": datetime.utcnow(),
            "type": "rag_query",
            "llama_settings": {
                "basic_params": basic_params,
                "advanced_params": advanced_params,
                "prompt": llama_prompt,
            },
        }
        await chat_collection.insert_one(chat_entry)

        return {
            "summary": summary.get("summary"),
            "mongo_ids": summary.get("mongo_ids"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing user query: {e}")


@router.get("/chat-history/{case_id}")
async def get_chat_history(
    case_id: str,
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(10, alias="limit", le=100),
    _: str = Depends(get_current_user),
):
    try:
        # Validate case exists
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Get the chat history from the single collection
        chat_collection = db["chat_histories"]

        # Get total count for pagination
        total_count = await chat_collection.count_documents(
            {"case_id": ObjectId(case_id), "user_id": ObjectId(_)}
        )

        # Calculate skip based on page and limit
        skip = (page - 1) * limit

        # Query the chat history with pagination
        chat_history = (
            await chat_collection.find(
                {"case_id": ObjectId(case_id), "user_id": ObjectId(_)}
            )
            .sort("timestamp", -1)
            .skip(skip)
            .limit(limit)
            .to_list(None)
        )

        # Process the results
        result = []
        for chat in chat_history:
            chat["_id"] = str(chat["_id"])
            chat["case_id"] = str(chat["case_id"])
            chat["user_id"] = str(chat["user_id"])
            chat["timestamp"] = chat["timestamp"].isoformat()
            result.append(chat)

        return JSONResponse(
            content={
                "data": result,
                "pagination": {
                    "total": total_count,
                    "page": page,
                    "limit": limit,
                    "total_pages": math.ceil(total_count / limit),
                },
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving chat history: {str(e)}"
        )


# New endpoint to get all chat histories for a user
@router.get("/user-chat-histories")
async def get_user_chat_histories(
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(10, alias="limit", le=100),
    _: str = Depends(get_current_user),
):
    try:
        chat_collection = db["chat_histories"]

        # Get total count for pagination
        total_count = await chat_collection.count_documents({"user_id": ObjectId(_)})

        # Calculate skip based on page and limit
        skip = (page - 1) * limit

        # Query all chat histories for the user with pagination
        chat_histories = (
            await chat_collection.find({"user_id": ObjectId(_)})
            .sort("timestamp", -1)
            .skip(skip)
            .limit(limit)
            .to_list(None)
        )

        # Process the results
        result = []
        for chat in chat_histories:
            chat["_id"] = str(chat["_id"])
            chat["case_id"] = str(chat["case_id"])
            chat["user_id"] = str(chat["user_id"])
            chat["timestamp"] = chat["timestamp"].isoformat()
            result.append(chat)

        return JSONResponse(
            content={
                "data": result,
                "pagination": {
                    "total": total_count,
                    "page": page,
                    "limit": limit,
                    "total_pages": math.ceil(total_count / limit),
                },
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error retrieving user chat histories: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving user chat histories: {str(e)}"
        )


@router.get("/case/{case_id}/contacts")
async def get_case_contacts(case_id: str, _: str = Depends(get_current_user)):
    """Get all unique contacts with their message counts from communication-type records."""
    try:
        # Validate and get case
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        collection = db[f"{case['name']}_{case_id}"]

        # Pipeline to get unique contacts with message counts
        pipeline = [
            {
                "$match": {
                    "Message Type": {
                        "$in": ["message", "call", "contact", "email", "sms", "chat"]
                    }
                }
            },
            {
                "$project": {
                    "entries": {
                        "$concatArrays": [
                            [
                                {
                                    "contact": {"$ifNull": ["$From", ""]},
                                    "name": {"$ifNull": ["$Name", ""]},
                                    "phone_numbers": {
                                        "$ifNull": ["$phone_numbers", []]
                                    },
                                    "email": {"$ifNull": ["$email", ""]},
                                    "direction": "from",
                                    "message_type": "$Message Type",
                                    "application": "$Application",
                                }
                            ],
                            {
                                "$cond": {
                                    "if": {"$eq": [{"$type": "$To"}, "string"]},
                                    "then": {
                                        "$map": {
                                            "input": {"$split": ["$To", ","]},
                                            "as": "to",
                                            "in": {
                                                "contact": {"$trim": {"input": "$$to"}},
                                                "name": {"$ifNull": ["$Name", ""]},
                                                "phone_numbers": {
                                                    "$ifNull": ["$phone_numbers", []]
                                                },
                                                "email": {"$ifNull": ["$email", ""]},
                                                "direction": "to",
                                                "message_type": "$Message Type",
                                                "application": "$Application",
                                            },
                                        }
                                    },
                                    "else": [],
                                }
                            },
                        ]
                    }
                }
            },
            {"$unwind": "$entries"},
            {"$match": {"entries.contact": {"$ne": ""}}},
            {
                "$group": {
                    "_id": "$entries.contact",
                    "names": {"$addToSet": "$entries.name"},
                    "phone_numbers": {"$addToSet": "$entries.phone_numbers"},
                    "emails": {"$addToSet": "$entries.email"},
                    "sent": {
                        "$sum": {
                            "$cond": [{"$eq": ["$entries.direction", "from"]}, 1, 0]
                        }
                    },
                    "received": {
                        "$sum": {"$cond": [{"$eq": ["$entries.direction", "to"]}, 1, 0]}
                    },
                    "types": {"$addToSet": "$entries.message_type"},
                    "applications": {"$addToSet": "$entries.application"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "identifier": "$_id",
                    "name": {
                        "$cond": [
                            {
                                "$gt": [
                                    {
                                        "$size": {
                                            "$filter": {
                                                "input": "$names",
                                                "cond": {"$ne": ["$$this", ""]},
                                            }
                                        }
                                    },
                                    0,
                                ]
                            },
                            {
                                "$arrayElemAt": [
                                    {
                                        "$filter": {
                                            "input": "$names",
                                            "cond": {"$ne": ["$$this", ""]},
                                        }
                                    },
                                    0,
                                ]
                            },
                            "$_id",
                        ]
                    },
                    "number": {
                        "$cond": [
                            {
                                "$gt": [
                                    {
                                        "$size": {
                                            "$reduce": {
                                                "input": "$phone_numbers",
                                                "initialValue": [],
                                                "in": {
                                                    "$concatArrays": [
                                                        "$$value",
                                                        "$$this",
                                                    ]
                                                },
                                            }
                                        }
                                    },
                                    0,
                                ]
                            },
                            {
                                "$arrayElemAt": [
                                    {
                                        "$reduce": {
                                            "input": "$phone_numbers",
                                            "initialValue": [],
                                            "in": {
                                                "$concatArrays": ["$$value", "$$this"]
                                            },
                                        }
                                    },
                                    0,
                                ]
                            },
                            {
                                "$cond": [
                                    {
                                        "$gt": [
                                            {
                                                "$size": {
                                                    "$filter": {
                                                        "input": "$emails",
                                                        "cond": {"$ne": ["$$this", ""]},
                                                    }
                                                }
                                            },
                                            0,
                                        ]
                                    },
                                    {
                                        "$arrayElemAt": [
                                            {
                                                "$filter": {
                                                    "input": "$emails",
                                                    "cond": {"$ne": ["$$this", ""]},
                                                }
                                            },
                                            0,
                                        ]
                                    },
                                    "$_id",
                                ]
                            },
                        ]
                    },
                    "total": {"$add": ["$sent", "$received"]},
                    "sent": "$sent",
                    "received": "$received",
                    "types": 1,
                    "applications": 1,
                }
            },
            {"$sort": {"total": -1, "name": 1}},
        ]

        contacts = await collection.aggregate(pipeline).to_list(None)
        # Sanitize NaN/Inf before returning
        safe_contacts = [sanitize_nan_values(c) for c in contacts]
        return JSONResponse(content={"data": safe_contacts}, status_code=200)

    except Exception as e:
        logger.error(f"Error getting contacts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/case/{case_id}/contact-messages/{contact}")
async def get_contact_messages(
    case_id: str,
    contact: str,
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(10, alias="limit", le=100),
    _: str = Depends(get_current_user),
):
    """Get all messages where the specified contact appears in 'from' or 'to' fields."""
    try:
        # Validate and get case
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        collection = db[f"{case['name']}_{case_id}"]

        # Query to match messages with the contact and case_id
        query = {
            "$and": [
                {"case_id": ObjectId(case_id)},
                {
                    "$or": [
                        {"From": contact},
                        {"To": {"$regex": re.escape(contact), "$options": "i"}},
                        {"Name": contact},
                        {"email": contact},
                        {"phone_numbers": contact},
                    ]
                },
                {
                    "Message Type": {
                        "$in": ["message", "call", "contact", "email", "sms", "chat"]
                    }
                },
            ]
        }

        # Get all messages matching the query
        messages = await collection.find(query).sort("timestamp", -1).to_list(None)

        # Process messages
        processed_messages = []
        for msg in messages:
            msg["_id"] = str(msg["_id"])
            if "case_id" in msg:
                msg["case_id"] = str(msg["case_id"])
            msg = sanitize_nan_values(msg)
            processed_messages.append(msg)

        return JSONResponse(content=processed_messages, status_code=200)

    except Exception as e:
        logger.error(f"Error getting contact messages: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{case_id}/applications")
async def get_case_applications(case_id: str, _: str = Depends(get_current_user)):
    """Return distinct Application values (and their counts) for messages in a case."""
    try:
        # Validate and fetch case
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    try:
        collection = db[f"{case['name']}_{case_id}"]

        pipeline = [
            {"$match": {"Application": {"$exists": True, "$ne": None, "$ne": ""}}},
            {"$group": {"_id": "$Application", "count": {"$sum": 1}}},
            {"$sort": {"count": -1, "_id": 1}},
        ]

        results = await collection.aggregate(pipeline).to_list(None)

        # Normalize results to a friendly structure
        apps = [
            {"application": r.get("_id"), "count": r.get("count", 0)} for r in results
        ]

        return JSONResponse(
            content={
                "case_id": case_id,
                "case_name": case.get("name"),
                "applications": apps,
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Error getting applications for case {case_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/{case_id}")
async def get_analytics(case_id: str):
    try:
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    # Extracting entityClasses from case to provide context
    entities_classes = case.get("entitiesClasses", [])
    logger.info(f"Entities classes from case: {entities_classes}")

    try:
        collection = db[f"{case['name']}_{case_id}"]

        # Only include records with Message Type in the allowed set (all types to be analyzed)
        allowed_types = [
            "call",
            "contact",
            "email",
            "browsing_history",
            "chat",
            "message",
            "sms",
            "location",
            "note",
        ]
        match_stage = {"$match": {"Message Type": {"$in": allowed_types}}}

        # Execute pipelines_v1 with Message Type filter
        bar_and_donut = await collection.aggregate(
            [match_stage] + pipelines_v1.bar_and_pipe_chart_pipeline
        ).to_list(None)
        heatmap = await collection.aggregate(
            [match_stage]
            + pipelines_v1.heatmap_of_appilication_against_emotion_piepline
        ).to_list(None)
        stacked_bar = await collection.aggregate(
            [match_stage] + pipelines_v1.stacked_bar_risk_per_lanaguage
        ).to_list(None)
        # Top cards: count all allowed message types
        top_cards_pipeline = [match_stage] + pipelines_v1.top_card_data_pipeline
        top_cards = await collection.aggregate(top_cards_pipeline).to_list(None)
        side_cards = await collection.aggregate(
            [match_stage] + pipelines_v1.side_card_data_pipeline
        ).to_list(None)
        area_chart = await collection.aggregate(
            [match_stage] + pipelines_v1.area_chart_entities_pipeline
        ).to_list(None)

        entities_classes_vs_emotion = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_classes_vs_emotion_pipeline
        ).to_list(None)
        applications_vs_topics = await collection.aggregate(
            [match_stage] + pipelines_v1.applications_vs_topics_pipeline
        ).to_list(None)
        topics_vs_emotions = await collection.aggregate(
            [match_stage] + pipelines_v1.topics_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_emotions = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_applications = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_vs_applications_pipeline
        ).to_list(None)
        entities_vs_topics = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_vs_topics_pipeline
        ).to_list(None)

        # Get distinct entity classes from the collection
        distinct_entity_classes = []

        # First try to get entity classes from the case configuration
        if entities_classes and len(entities_classes) > 0:
            distinct_entity_classes = entities_classes
        else:
            # If not available in case config, try to extract from the data
            entity_classes_result = await collection.aggregate(
                [
                    {
                        "$match": {
                            "analysis_summary.entities_classification": {
                                "$exists": True
                            }
                        }
                    },
                    {
                        "$project": {
                            "entity_classes": {
                                "$objectToArray": "$analysis_summary.entities_classification"
                            }
                        }
                    },
                    {"$unwind": "$entity_classes"},
                    {"$group": {"_id": "$entity_classes.k"}},
                ]
            ).to_list(None)

            distinct_entity_classes = [ec["_id"] for ec in entity_classes_result]

        logger.info(f"Distinct entity classes found: {distinct_entity_classes}")

        # Generate entity class specific heatmaps
        entity_classes_heatmaps = []

        for entity_class in distinct_entity_classes:
            entity_class_data = {
                "entity_class": entity_class,
                "vs_emotions": await collection.aggregate(
                    pipeline=pipelines_v1.entity_class_vs_emotions_pipeline(
                        entity_class
                    )
                ).to_list(None),
                "vs_applications": await collection.aggregate(
                    pipeline=pipelines_v1.entity_class_vs_applications_pipeline(
                        entity_class
                    )
                ).to_list(None),
                "vs_topics": await collection.aggregate(
                    pipeline=pipelines_v1.entity_class_vs_topics_pipeline(entity_class)
                ).to_list(None),
            }
            entity_classes_heatmaps.append(entity_class_data)

    except Exception as e:
        logger.error(f"Error executing pipelines_v1: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

    # Sanitize the data
    bar_and_donut = sanitize_nan_values_recursive(bar_and_donut)
    heatmap = sanitize_nan_values_recursive(heatmap)
    stacked_bar = sanitize_nan_values_recursive(stacked_bar)

    # Filter out note and location types with 0 count from top_cards
    for card in top_cards:
        if "applications_message_count" in card:
            # Remove note and location if count is 0
            if (
                "note" in card["applications_message_count"]
                and card["applications_message_count"]["note"] == 0
            ):
                del card["applications_message_count"]["note"]
            if (
                "location" in card["applications_message_count"]
                and card["applications_message_count"]["location"] == 0
            ):
                del card["applications_message_count"]["location"]

        if "top_3_applications_message_count" in card:
            # Remove note and location if count is 0
            if (
                "note" in card["top_3_applications_message_count"]
                and card["top_3_applications_message_count"]["note"] == 0
            ):
                del card["top_3_applications_message_count"]["note"]
            if (
                "location" in card["top_3_applications_message_count"]
                and card["top_3_applications_message_count"]["location"] == 0
            ):
                del card["top_3_applications_message_count"]["location"]

        if "message_types_count" in card:
            # Remove note and location if count is 0
            if (
                "note" in card["message_types_count"]
                and card["message_types_count"]["note"] == 0
            ):
                del card["message_types_count"]["note"]
            if (
                "location" in card["message_types_count"]
                and card["message_types_count"]["location"] == 0
            ):
                del card["message_types_count"]["location"]

    top_cards = sanitize_nan_values_recursive(top_cards)
    side_cards = sanitize_nan_values_recursive(side_cards)
    area_chart = sanitize_nan_values_recursive(area_chart)

    # Sanitize existing heatmaps
    entities_classes_vs_emotion = sanitize_nan_values_recursive(
        entities_classes_vs_emotion
    )
    applications_vs_topics = sanitize_nan_values_recursive(applications_vs_topics)
    topics_vs_emotions = sanitize_nan_values_recursive(topics_vs_emotions)
    entities_vs_emotions = sanitize_nan_values_recursive(entities_vs_emotions)
    entities_vs_applications = sanitize_nan_values_recursive(entities_vs_applications)
    entities_vs_topics = sanitize_nan_values_recursive(entities_vs_topics)

    # Sanitize entity class specific heatmaps
    for entity_class_data in entity_classes_heatmaps:
        entity_class_data["vs_emotions"] = sanitize_nan_values_recursive(
            entity_class_data["vs_emotions"]
        )
        entity_class_data["vs_applications"] = sanitize_nan_values_recursive(
            entity_class_data["vs_applications"]
        )
        entity_class_data["vs_topics"] = sanitize_nan_values_recursive(
            entity_class_data["vs_topics"]
        )

    # Ensure all top_cards have applications_message_count
    for card in top_cards:
        if (
            "applications_message_count" not in card
            and "top_3_applications_message_count" in card
        ):
            card["applications_message_count"] = card.get(
                "applications_message_count",
                card.get("top_3_applications_message_count", {}),
            )

    analytics_data = {
        "bar_and_donut": {
            "data": bar_and_donut,
            "chart_type": "bar_and_donut",
            "description": "Bar and Donut chart showing the distribution of topics, sentiments etc.",
        },
        "heatmap": {
            "data": heatmap,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against emotions.",
        },
        "stacked_bar": {
            "data": stacked_bar,
            "chart_type": "stacked_bar",
            "description": "Stacked bar chart showing the distribution of risk levels per language.",
        },
        "area_chart": {
            "data": area_chart,
            "chart_type": "area_chart",
            "description": "Area chart showing the top 10 most repeated entities",
        },
        "top_cards": {
            "data": top_cards,
            "chart_type": "top_cards",
            "description": "Top cards showing the unique users, total messages, alerted messages, high risk messages",
        },
        "side_cards": {
            "data": side_cards,
            "chart_type": "side_cards",
            "description": "Side cards showing the last 5 messages, top users, messages by language, messages by risk level, most_occurring_entities",
        },
        # Existing heatmaps
        "entities_classes_vs_emotion": {
            "data": entities_classes_vs_emotion,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of entity classes against emotions.",
        },
        "applications_vs_topics": {
            "data": applications_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against top topics.",
        },
        "topics_vs_emotions": {
            "data": topics_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top topics against emotions.",
        },
        "entities_vs_emotions": {
            "data": entities_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against emotions.",
        },
        "entities_vs_applications": {
            "data": entities_vs_applications,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against applications.",
        },
        "entities_vs_topics": {
            "data": entities_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against topics.",
        },
        # New entity class specific heatmaps
        "entity_classes_heatmaps": entity_classes_heatmaps,
    }

    return JSONResponse(
        content=jsonable_encoder(analytics_data),
        status_code=200,
    )


@router.get("/analytics-filtered/{case_id}")
async def get_filtered_analytics(
    case_id: str,
    sematic_search: Optional[str] = Query(None),
    top_topic: Optional[List[str]] = Query(None),
    toxicity_score: Optional[int] = Query(None),
    sentiment_aspect: Optional[List[str]] = Query(None),
    emotion: Optional[List[str]] = Query(None),
    language: Optional[List[str]] = Query(None),
    risk_level: Optional[List[str]] = Query(None),
    application_type: Optional[List[str]] = Query(None),
    interaction_type: Optional[List[str]] = Query(None),
    entities: Optional[List[str]] = Query(None),
    entities_classes: Optional[List[str]] = Query(None),
    alert: Optional[bool] = Query(None),
    applications: Optional[List[str]] = Query(None),
    timestamp_from: Optional[str] = Query(None),
    timestamp_to: Optional[str] = Query(None),
    _: str = Depends(get_current_user),
):
    try:
        # Validate ObjectId
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    # Build query based on filters
    query = {}
    if sematic_search and sematic_search.strip() != "":
        model_profile = case.get("model_profile", {})

        if not isinstance(model_profile, dict):
            # If model_profile is an ID string, fetch the profile
            try:
                if isinstance(model_profile, str):
                    models_profile = await models_master_collection.find_one(
                        {"_id": ObjectId(model_profile)}
                    )
                    if not models_profile:
                        raise HTTPException(
                            status_code=404, detail="Model Profile not found."
                        )
                else:
                    # If it's already a dictionary, use it directly
                    models_profile = model_profile
            except Exception as e:
                # Fallback to default profile
                models_profile = await models_master_collection.find().to_list(None)
                if len(models_profile) > 1:
                    models_profile = models_profile[1]
                else:
                    raise HTTPException(
                        status_code=404, detail="No model profiles available."
                    )
        else:
            # Use the model profile directly from the case
            models_profile = model_profile

        collection = db[f"{case['name']}_{case_id}"]
        rag = ArabicRagAnalyzer(collection, collection_case, case_id, models_profile)

        try:
            results = rag.semantic_search(sematic_search)
            if not results:
                raise HTTPException(
                    status_code=404, detail="No messages found with the given filters"
                )

            # Extract mongo_ids from the search results
            mongo_ids = []
            for result in results:
                if result.payload and "mongo_id" in result.payload:
                    mongo_ids.append(result.payload["mongo_id"])

            if not mongo_ids:
                raise HTTPException(
                    status_code=404, detail="No valid messages found in search results"
                )

            query["_id"] = {"$in": [ObjectId(id_) for id_ in mongo_ids]}
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error performing semantic search: {str(e)}"
            )

    # Process filter parameters
    top_topic = (
        top_topic[0].split(",") if top_topic and top_topic[0].split(",") != "" else None
    )
    toxicity_score = toxicity_score if toxicity_score else None
    sentiment_aspect = (
        sentiment_aspect[0].split(",")
        if sentiment_aspect and sentiment_aspect[0] != ""
        else None
    )
    emotion = emotion[0].split(",") if emotion and emotion[0] != "" else None
    language = language[0].split(",") if language and language[0] != "" else None
    risk_level = (
        risk_level[0].split(",") if risk_level and risk_level[0] != "" else None
    )
    application_type = (
        application_type[0].split(",")
        if application_type and application_type[0] != ""
        else None
    )
    interaction_type = (
        interaction_type[0].split(",")
        if interaction_type and interaction_type[0] != ""
        else None
    )
    entities = entities if entities and len(entities) > 0 else None
    entities_classes = (
        entities_classes if entities_classes and len(entities_classes) > 0 else None
    )
    applications = (
        applications[0].split(",")
        if applications and applications[0].split(",") != ""
        else None
    )

    # Process timestamp filters
    if timestamp_from or timestamp_to:
        timestamp_query = {}
        if timestamp_from:
            try:
                from_date = datetime.fromisoformat(
                    timestamp_from.replace("Z", "+00:00")
                )
                timestamp_query["$gte"] = from_date
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid timestamp_from format: {str(e)}"
                )

        if timestamp_to:
            try:
                to_date = datetime.fromisoformat(timestamp_to.replace("Z", "+00:00"))
                timestamp_query["$lte"] = to_date
            except ValueError as e:
                raise HTTPException(
                    status_code=400, detail=f"Invalid timestamp_to format: {str(e)}"
                )

        if timestamp_query:
            query["timestamp"] = timestamp_query

    # Add filters to query
    if top_topic:
        query["analysis_summary.top_topic"] = {"$in": top_topic}
    if toxicity_score:
        query["analysis_summary.toxicity_score"] = {"$gte": int(toxicity_score)}
    if sentiment_aspect:
        query["analysis_summary.sentiment_aspect"] = {"$in": sentiment_aspect}
    if emotion:
        query["analysis_summary.emotion"] = {"$in": emotion}
    if language:
        query["analysis_summary.language"] = {"$in": language}
    if risk_level:
        query["analysis_summary.risk_level"] = {"$in": risk_level}
    if application_type:
        logger.info(f"application_type: {application_type}")
        # Map normalized application names to actual database values
        application_mapping = {
            "browsing_history": [
                "WebBookmark",
                "VisitedPage",
                "BrowserHistory",
                "Search",
            ],
            "location": ["Location Services"],
            "email": ["Email"],
            "sms/mms": ["SMS/MMS"],
            "message": [
                "SMS/MMS"
            ],  # For analytics, 'message' means Application='SMS/MMS' or Message Type='message'
            "chat": ["Chat"],
            "contact": ["Contact"],
            "call": [
                "call"
            ],  # For calls, we need to check Message Type instead of Application
            "note": ["Note"],
        }
        # Handle special case for calls
        if "call" in application_type:
            if "Message Type" not in query:
                query["Message Type"] = {"$in": ["call"]}
            else:
                existing_types = query["Message Type"].get("$in", [])
                if "call" not in existing_types:
                    existing_types.append("call")
                    query["Message Type"]["$in"] = existing_types
            application_type = [app for app in application_type if app != "call"]
        expanded_applications = []
        for app_type in application_type:
            if app_type.lower() in application_mapping:
                expanded_applications.extend(application_mapping[app_type.lower()])
            else:
                if app_type in application_mapping:
                    expanded_applications.extend(application_mapping[app_type])
                else:
                    expanded_applications.append(app_type)
        # Special handling for 'message' type: match both Application and Message Type
        if "message" in [a.lower() for a in application_type]:
            query["$or"] = [
                {"Application": {"$in": expanded_applications}},
                {"Message Type": {"$in": ["message"]}},
            ]
        elif expanded_applications:
            query["$or"] = [
                {"Application": {"$in": expanded_applications}},
                {
                    "$and": [
                        {"Application": {"$exists": False}},
                        {
                            "Message Type": {
                                "$in": [
                                    app.lower()
                                    for app in application_type
                                    if app.lower()
                                    in ["email", "message", "chat", "contact", "call"]
                                ]
                            }
                        },
                    ]
                },
            ]
    if interaction_type:
        query["analysis_summary.interaction_type"] = {"$in": interaction_type}
    if entities:
        query["analysis_summary.entities"] = {"$in": entities}
    if entities_classes:
        and_conditions = []
        for class_name in entities_classes:
            and_conditions.append(
                {
                    f"analysis_summary.entities_classification.{class_name}": {
                        "$exists": True,
                        "$ne": [],
                    }
                }
            )
        query["$and"] = and_conditions
    if alert is not None:
        query["alert"] = alert

    # Add applications filter
    if applications:
        if "$or" not in query:
            query["$or"] = []
        app_conditions = [
            {"Application": {"$regex": f"^{re.escape(app)}$", "$options": "i"}}
            for app in applications
        ]
        message_type_conditions = [
            {"Message Type": {"$regex": f"^{re.escape(app.lower())}$", "$options": "i"}}
            for app in applications
        ]
        query["$or"].extend(app_conditions + message_type_conditions)

    try:
        collection = db[f"{case['name']}_{case_id}"]

        # Add match stage with query to all pipelines_v1
        match_stage = {"$match": query}

        # Get total number of messages for the query
        total_messages = await collection.count_documents(query)

        # Execute pipelines_v1 with filter
        bar_and_donut = await collection.aggregate(
            [match_stage] + pipelines_v1.bar_and_pipe_chart_pipeline
        ).to_list(None)
        heatmap = await collection.aggregate(
            [match_stage]
            + pipelines_v1.heatmap_of_appilication_against_emotion_piepline
        ).to_list(None)
        stacked_bar = await collection.aggregate(
            [match_stage] + pipelines_v1.stacked_bar_risk_per_lanaguage
        ).to_list(None)
        top_cards = await collection.aggregate(
            [match_stage] + pipelines_v1.top_card_data_pipeline
        ).to_list(None)
        side_cards = await collection.aggregate(
            [match_stage] + pipelines_v1.side_card_data_pipeline
        ).to_list(None)
        area_chart = await collection.aggregate(
            [match_stage] + pipelines_v1.area_chart_entities_pipeline
        ).to_list(None)

        # Execute heatmap pipelines_v1 with filter
        entities_classes_vs_emotion = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_classes_vs_emotion_pipeline
        ).to_list(None)
        applications_vs_topics = await collection.aggregate(
            [match_stage] + pipelines_v1.applications_vs_topics_pipeline
        ).to_list(None)
        topics_vs_emotions = await collection.aggregate(
            [match_stage] + pipelines_v1.topics_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_emotions = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_applications = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_vs_applications_pipeline
        ).to_list(None)
        entities_vs_topics = await collection.aggregate(
            [match_stage] + pipelines_v1.entities_vs_topics_pipeline
        ).to_list(None)

        # Get distinct entity classes
        distinct_entity_classes = case.get("entitiesClasses", [])
        if not distinct_entity_classes:
            entity_classes_result = await collection.aggregate(
                [
                    match_stage,
                    {
                        "$match": {
                            "analysis_summary.entities_classification": {
                                "$exists": True
                            }
                        }
                    },
                    {
                        "$project": {
                            "entity_classes": {
                                "$objectToArray": "$analysis_summary.entities_classification"
                            }
                        }
                    },
                    {"$unwind": "$entity_classes"},
                    {"$group": {"_id": "$entity_classes.k"}},
                ]
            ).to_list(None)
            distinct_entity_classes = [ec["_id"] for ec in entity_classes_result]

        logger.info(
            f"Distinct entity classes found for RAG query analytics: {distinct_entity_classes}"
        )

        # Generate entity class specific heatmaps
        entity_classes_heatmaps = []
        for entity_class in distinct_entity_classes:
            entity_class_data = {
                "entity_class": entity_class,
                "vs_emotions": await collection.aggregate(
                    [match_stage]
                    + pipelines_v1.entity_class_vs_emotions_pipeline(entity_class)
                ).to_list(None),
                "vs_applications": await collection.aggregate(
                    [match_stage]
                    + pipelines_v1.entity_class_vs_applications_pipeline(entity_class)
                ).to_list(None),
                "vs_topics": await collection.aggregate(
                    [match_stage]
                    + pipelines_v1.entity_class_vs_topics_pipeline(entity_class)
                ).to_list(None),
            }
            entity_classes_heatmaps.append(entity_class_data)

    except Exception as e:
        logger.error(f"Error executing pipelines_v1: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

    # Filter out note and location types with 0 count from top_cards
    for card in top_cards:
        if "applications_message_count" in card:
            # Remove note and location if count is 0
            if (
                "note" in card["applications_message_count"]
                and card["applications_message_count"]["note"] == 0
            ):
                del card["applications_message_count"]["note"]
            if (
                "location" in card["applications_message_count"]
                and card["applications_message_count"]["location"] == 0
            ):
                del card["applications_message_count"]["location"]

        if "top_3_applications_message_count" in card:
            # Remove note and location if count is 0
            if (
                "note" in card["top_3_applications_message_count"]
                and card["top_3_applications_message_count"]["note"] == 0
            ):
                del card["top_3_applications_message_count"]["note"]
            if (
                "location" in card["top_3_applications_message_count"]
                and card["top_3_applications_message_count"]["location"] == 0
            ):
                del card["top_3_applications_message_count"]["location"]

        if "message_types_count" in card:
            # Remove note and location if count is 0
            if (
                "note" in card["message_types_count"]
                and card["message_types_count"]["note"] == 0
            ):
                del card["message_types_count"]["note"]
            if (
                "location" in card["message_types_count"]
                and card["message_types_count"]["location"] == 0
            ):
                del card["message_types_count"]["location"]

    # Sanitize all data
    bar_and_donut = sanitize_nan_values_recursive(bar_and_donut)
    heatmap = sanitize_nan_values_recursive(heatmap)
    stacked_bar = sanitize_nan_values_recursive(stacked_bar)
    top_cards = sanitize_nan_values_recursive(top_cards)
    side_cards = sanitize_nan_values_recursive(side_cards)
    area_chart = sanitize_nan_values_recursive(area_chart)
    entities_classes_vs_emotion = sanitize_nan_values_recursive(
        entities_classes_vs_emotion
    )
    applications_vs_topics = sanitize_nan_values_recursive(applications_vs_topics)
    topics_vs_emotions = sanitize_nan_values_recursive(topics_vs_emotions)
    entities_vs_emotions = sanitize_nan_values_recursive(entities_vs_emotions)
    entities_vs_applications = sanitize_nan_values_recursive(entities_vs_applications)
    entities_vs_topics = sanitize_nan_values_recursive(entities_vs_topics)

    # Sanitize entity class heatmaps
    for entity_class_data in entity_classes_heatmaps:
        entity_class_data["vs_emotions"] = sanitize_nan_values_recursive(
            entity_class_data["vs_emotions"]
        )
        entity_class_data["vs_applications"] = sanitize_nan_values_recursive(
            entity_class_data["vs_applications"]
        )
        entity_class_data["vs_topics"] = sanitize_nan_values_recursive(
            entity_class_data["vs_topics"]
        )

    # Set default top_cards if empty
    if not top_cards or len(top_cards) == 0:
        top_cards = [
            {
                "totalMessages": 0,
                "highRiskMessages": 0,
                "uniqueUsers": 0,
                "alertMessages": 0,
                "top_3_applications_message_count": {},
                "applications_message_count": {},
                "message_types_count": {},
                "top_3_entities_classes": {},
            }
        ]

    analytics_data = {
        "bar_and_donut": {
            "data": bar_and_donut,
            "chart_type": "bar_and_donut",
            "description": "Bar and Donut chart showing the distribution of topics, sentiments etc. for RAG query source messages.",
        },
        "heatmap": {
            "data": heatmap,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against emotions for RAG query source messages.",
        },
        "stacked_bar": {
            "data": stacked_bar,
            "chart_type": "stacked_bar",
            "description": "Stacked bar chart showing the distribution of risk levels per language for RAG query source messages.",
        },
        "area_chart": {
            "data": area_chart,
            "chart_type": "area_chart",
            "description": "Area chart showing the top 10 most repeated entities in RAG query source messages.",
        },
        "top_cards": {
            "data": top_cards,
            "chart_type": "top_cards",
            "description": "Top cards showing the unique users, total messages, alerted messages, high risk messages for RAG query source messages.",
        },
        "side_cards": {
            "data": side_cards,
            "chart_type": "side_cards",
            "description": "Side cards showing the last 5 messages, top users, messages by language, messages by risk level, most_occurring_entities for RAG query source messages.",
        },
        "entities_classes_vs_emotion": {
            "data": entities_classes_vs_emotion,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of entity classes against emotions for RAG query source messages.",
        },
        "applications_vs_topics": {
            "data": applications_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against top topics for RAG query source messages.",
        },
        "topics_vs_emotions": {
            "data": topics_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top topics against emotions for RAG query source messages.",
        },
        "entities_vs_emotions": {
            "data": entities_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against emotions for RAG query source messages.",
        },
        "entities_vs_applications": {
            "data": entities_vs_applications,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against applications for RAG query source messages.",
        },
        "entities_vs_topics": {
            "data": entities_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against topics for RAG query source messages.",
        },
        "entity_classes_heatmaps": entity_classes_heatmaps,
        "query_info": {
            "total_source_messages": total_messages,
            "case_id": case_id,
            "case_name": case["name"],
        },
    }

    return JSONResponse(
        content=jsonable_encoder(analytics_data),
        status_code=200,
    )


@router.post("/get-messages-by-ids")
async def get_messages_by_ids(
    request: GetMessagesByIdsRequest, _: str = Depends(get_current_user)
):
    try:
        # Validate case exists
        case = await collection_case.find_one({"_id": ObjectId(request.case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Get the collection for this case
        collection = db[f"{case['name']}_{request.case_id}"]

        # Convert string IDs to ObjectId
        object_ids = [ObjectId(id) for id in request.message_ids]

        # Get total count for pagination
        total_count = len(object_ids)

        # Calculate skip based on page and limit
        skip = (request.page - 1) * request.limit

        # Apply pagination to the object_ids list
        paginated_ids = object_ids[skip : skip + request.limit]

        # Query the messages with pagination
        messages = await collection.find({"_id": {"$in": paginated_ids}}).to_list(None)

        # Process the results
        result = []
        for message in messages:
            # Convert ObjectIds to strings
            message["_id"] = str(message["_id"])
            message["case_id"] = str(message["case_id"])
            # Sanitize any NaN values
            message = sanitize_nan_values(message)
            result.append(message)

        return JSONResponse(
            content={
                "data": result,
                "pagination": {
                    "total": total_count,
                    "page": request.page,
                    "limit": request.limit,
                    "total_pages": math.ceil(total_count / request.limit),
                },
            },
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error retrieving messages by IDs: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving messages: {str(e)}"
        )


@router.post("/rag-query-analytics")
async def get_rag_query_analytics(
    request: RAGQueryAnalyticsRequest, _: str = Depends(get_current_user)
):
    """
    Generate analytics for the specific messages that were used to answer a RAG query.
    This analyzes only the source messages that the analyzer used to generate the summary.
    """
    try:
        # Validate case exists
        case = await collection_case.find_one({"_id": ObjectId(request.case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Get the collection for this case
        collection = db[f"{case['name']}_{request.case_id}"]

        # Convert string IDs to ObjectId
        object_ids = [ObjectId(id) for id in request.mongo_ids]

        # Create match stage to filter only the specific messages
        match_stage = {"$match": {"_id": {"$in": object_ids}}}

        # Only include records with Message Type in the allowed set (all types to be analyzed)
        allowed_types = [
            "call",
            "contact",
            "email",
            "browsing_history",
            "chat",
            "message",
            "sms",
        ]
        message_type_match = {"$match": {"Message Type": {"$in": allowed_types}}}

        # Combine both match stages
        combined_match = {
            "$match": {
                "_id": {"$in": object_ids},
                "Message Type": {"$in": allowed_types},
            }
        }

        # Execute pipelines_v1 with the specific message filter
        bar_and_donut = await collection.aggregate(
            [combined_match] + pipelines_v1.bar_and_pipe_chart_pipeline
        ).to_list(None)
        heatmap = await collection.aggregate(
            [combined_match]
            + pipelines_v1.heatmap_of_appilication_against_emotion_piepline
        ).to_list(None)
        stacked_bar = await collection.aggregate(
            [combined_match] + pipelines_v1.stacked_bar_risk_per_lanaguage
        ).to_list(None)
        top_cards = await collection.aggregate(
            [combined_match] + pipelines_v1.top_card_data_pipeline
        ).to_list(None)
        side_cards = await collection.aggregate(
            [combined_match] + pipelines_v1.side_card_data_pipeline
        ).to_list(None)
        area_chart = await collection.aggregate(
            [combined_match] + pipelines_v1.area_chart_entities_pipeline
        ).to_list(None)

        # Execute heatmap pipelines_v1 with the specific message filter
        entities_classes_vs_emotion = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_classes_vs_emotion_pipeline
        ).to_list(None)
        applications_vs_topics = await collection.aggregate(
            [combined_match] + pipelines_v1.applications_vs_topics_pipeline
        ).to_list(None)
        topics_vs_emotions = await collection.aggregate(
            [combined_match] + pipelines_v1.topics_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_emotions = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_applications = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_vs_applications_pipeline
        ).to_list(None)
        entities_vs_topics = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_vs_topics_pipeline
        ).to_list(None)

        # Get distinct entity classes from the case configuration
        entities_classes = case.get("entitiesClasses", [])
        distinct_entity_classes = []

        if entities_classes and len(entities_classes) > 0:
            distinct_entity_classes = entities_classes
        else:
            # If not available in case config, try to extract from the filtered data
            entity_classes_result = await collection.aggregate(
                [
                    combined_match,
                    {
                        "$match": {
                            "analysis_summary.entities_classification": {
                                "$exists": True
                            }
                        }
                    },
                    {
                        "$project": {
                            "entity_classes": {
                                "$objectToArray": "$analysis_summary.entities_classification"
                            }
                        }
                    },
                    {"$unwind": "$entity_classes"},
                    {"$group": {"_id": "$entity_classes.k"}},
                ]
            ).to_list(None)

            distinct_entity_classes = [ec["_id"] for ec in entity_classes_result]

        logger.info(
            f"Distinct entity classes found for RAG query analytics: {distinct_entity_classes}"
        )

        # Generate entity class specific heatmaps
        entity_classes_heatmaps = []

        for entity_class in distinct_entity_classes:
            entity_class_data = {
                "entity_class": entity_class,
                "vs_emotions": await collection.aggregate(
                    [combined_match]
                    + pipelines_v1.entity_class_vs_emotions_pipeline(entity_class)
                ).to_list(None),
                "vs_applications": await collection.aggregate(
                    [combined_match]
                    + pipelines_v1.entity_class_vs_applications_pipeline(entity_class)
                ).to_list(None),
                "vs_topics": await collection.aggregate(
                    [combined_match]
                    + pipelines_v1.entity_class_vs_topics_pipeline(entity_class)
                ).to_list(None),
            }
            entity_classes_heatmaps.append(entity_class_data)

    except Exception as e:
        logger.error(f"Error executing RAG query analytics pipelines_v1: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

    # Sanitize all data
    bar_and_donut = sanitize_nan_values_recursive(bar_and_donut)
    heatmap = sanitize_nan_values_recursive(heatmap)
    stacked_bar = sanitize_nan_values_recursive(stacked_bar)
    top_cards = sanitize_nan_values_recursive(top_cards)
    side_cards = sanitize_nan_values_recursive(side_cards)
    area_chart = sanitize_nan_values_recursive(area_chart)
    entities_classes_vs_emotion = sanitize_nan_values_recursive(
        entities_classes_vs_emotion
    )
    applications_vs_topics = sanitize_nan_values_recursive(applications_vs_topics)
    topics_vs_emotions = sanitize_nan_values_recursive(topics_vs_emotions)
    entities_vs_emotions = sanitize_nan_values_recursive(entities_vs_emotions)
    entities_vs_applications = sanitize_nan_values_recursive(entities_vs_applications)
    entities_vs_topics = sanitize_nan_values_recursive(entities_vs_topics)

    # Sanitize entity class heatmaps
    for entity_class_data in entity_classes_heatmaps:
        entity_class_data["vs_emotions"] = sanitize_nan_values_recursive(
            entity_class_data["vs_emotions"]
        )
        entity_class_data["vs_applications"] = sanitize_nan_values_recursive(
            entity_class_data["vs_applications"]
        )
        entity_class_data["vs_topics"] = sanitize_nan_values_recursive(
            entity_class_data["vs_topics"]
        )

    # Set default top_cards if empty
    if not top_cards or len(top_cards) == 0:
        top_cards = [
            {
                "totalMessages": 0,
                "highRiskMessages": 0,
                "uniqueUsers": 0,
                "alertMessages": 0,
                "top_3_applications_message_count": {},
                "applications_message_count": {},
                "message_types_count": {},
                "top_3_entities_classes": {},
            }
        ]

    analytics_data = {
        "bar_and_donut": {
            "data": bar_and_donut,
            "chart_type": "bar_and_donut",
            "description": "Bar and Donut chart showing the distribution of topics, sentiments etc. for RAG query source messages.",
        },
        "heatmap": {
            "data": heatmap,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against emotions for RAG query source messages.",
        },
        "stacked_bar": {
            "data": stacked_bar,
            "chart_type": "stacked_bar",
            "description": "Stacked bar chart showing the distribution of risk levels per language for RAG query source messages.",
        },
        "area_chart": {
            "data": area_chart,
            "chart_type": "area_chart",
            "description": "Area chart showing the top 10 most repeated entities in RAG query source messages.",
        },
        "top_cards": {
            "data": top_cards,
            "chart_type": "top_cards",
            "description": "Top cards showing the unique users, total messages, alerted messages, high risk messages for RAG query source messages.",
        },
        "side_cards": {
            "data": side_cards,
            "chart_type": "side_cards",
            "description": "Side cards showing the last 5 messages, top users, messages by language, messages by risk level, most_occurring_entities for RAG query source messages.",
        },
        "entities_classes_vs_emotion": {
            "data": entities_classes_vs_emotion,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of entity classes against emotions for RAG query source messages.",
        },
        "applications_vs_topics": {
            "data": applications_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against top topics for RAG query source messages.",
        },
        "topics_vs_emotions": {
            "data": topics_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top topics against emotions for RAG query source messages.",
        },
        "entities_vs_emotions": {
            "data": entities_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against emotions for RAG query source messages.",
        },
        "entities_vs_applications": {
            "data": entities_vs_applications,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against applications for RAG query source messages.",
        },
        "entities_vs_topics": {
            "data": entities_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against topics for RAG query source messages.",
        },
        "entity_classes_heatmaps": entity_classes_heatmaps,
        "query_info": {
            "total_source_messages": len(request.mongo_ids),
            "case_id": request.case_id,
            "case_name": case["name"],
        },
    }

    return JSONResponse(
        content=jsonable_encoder(analytics_data),
        status_code=200,
    )


@router.get("/rag-query-analytics-by-chat-id/{chat_id}")
async def get_rag_query_analytics_by_chat_id(
    chat_id: str, _: str = Depends(get_current_user)
):
    """
    Generate analytics for the specific messages that were used to answer a RAG query
    by providing the chat history entry ID. This retrieves the mongo_ids from the chat history
    and generates analytics for those specific messages.
    """
    try:
        # Get the chat history entry
        chat_collection = db["chat_histories"]
        chat_entry = await chat_collection.find_one({"_id": ObjectId(chat_id)})

        if not chat_entry:
            raise HTTPException(status_code=404, detail="Chat history entry not found")

        # Check if the user has access to this chat entry
        if str(chat_entry["user_id"]) != _:
            raise HTTPException(
                status_code=403, detail="Access denied to this chat entry"
            )

        # Get the mongo_ids from the chat entry
        mongo_ids = chat_entry.get("mongo_ids", [])
        if not mongo_ids:
            raise HTTPException(
                status_code=404, detail="No source messages found for this chat entry"
            )

        # Get case information
        case_id = str(chat_entry["case_id"])
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Get the collection for this case
        collection = db[f"{case['name']}_{case_id}"]

        # Convert string IDs to ObjectId
        object_ids = [ObjectId(id) for id in mongo_ids]

        # Create match stage to filter only the specific messages
        match_stage = {"$match": {"_id": {"$in": object_ids}}}

        # Only include records with Message Type in the allowed set (all types to be analyzed)
        allowed_types = [
            "call",
            "contact",
            "email",
            "browsing_history",
            "chat",
            "message",
            "sms",
        ]
        message_type_match = {"$match": {"Message Type": {"$in": allowed_types}}}

        # Combine both match stages
        combined_match = {
            "$match": {
                "_id": {"$in": object_ids},
                "Message Type": {"$in": allowed_types},
            }
        }

        # Execute pipelines_v1 with the specific message filter
        bar_and_donut = await collection.aggregate(
            [combined_match] + pipelines_v1.bar_and_pipe_chart_pipeline
        ).to_list(None)
        heatmap = await collection.aggregate(
            [combined_match]
            + pipelines_v1.heatmap_of_appilication_against_emotion_piepline
        ).to_list(None)
        stacked_bar = await collection.aggregate(
            [combined_match] + pipelines_v1.stacked_bar_risk_per_lanaguage
        ).to_list(None)
        top_cards = await collection.aggregate(
            [combined_match] + pipelines_v1.top_card_data_pipeline
        ).to_list(None)
        side_cards = await collection.aggregate(
            [combined_match] + pipelines_v1.side_card_data_pipeline
        ).to_list(None)
        area_chart = await collection.aggregate(
            [combined_match] + pipelines_v1.area_chart_entities_pipeline
        ).to_list(None)

        # Execute heatmap pipelines_v1 with the specific message filter
        entities_classes_vs_emotion = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_classes_vs_emotion_pipeline
        ).to_list(None)
        applications_vs_topics = await collection.aggregate(
            [combined_match] + pipelines_v1.applications_vs_topics_pipeline
        ).to_list(None)
        topics_vs_emotions = await collection.aggregate(
            [combined_match] + pipelines_v1.topics_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_emotions = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_applications = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_vs_applications_pipeline
        ).to_list(None)
        entities_vs_topics = await collection.aggregate(
            [combined_match] + pipelines_v1.entities_vs_topics_pipeline
        ).to_list(None)

        # Get distinct entity classes from the case configuration
        entities_classes = case.get("entitiesClasses", [])
        distinct_entity_classes = []

        if entities_classes and len(entities_classes) > 0:
            distinct_entity_classes = entities_classes
        else:
            # If not available in case config, try to extract from the filtered data
            entity_classes_result = await collection.aggregate(
                [
                    combined_match,
                    {
                        "$match": {
                            "analysis_summary.entities_classification": {
                                "$exists": True
                            }
                        }
                    },
                    {
                        "$project": {
                            "entity_classes": {
                                "$objectToArray": "$analysis_summary.entities_classification"
                            }
                        }
                    },
                    {"$unwind": "$entity_classes"},
                    {"$group": {"_id": "$entity_classes.k"}},
                ]
            ).to_list(None)

            distinct_entity_classes = [ec["_id"] for ec in entity_classes_result]

        logger.info(
            f"Distinct entity classes found for RAG query analytics: {distinct_entity_classes}"
        )

        # Generate entity class specific heatmaps
        entity_classes_heatmaps = []

        for entity_class in distinct_entity_classes:
            entity_class_data = {
                "entity_class": entity_class,
                "vs_emotions": await collection.aggregate(
                    [combined_match]
                    + pipelines_v1.entity_class_vs_emotions_pipeline(entity_class)
                ).to_list(None),
                "vs_applications": await collection.aggregate(
                    [combined_match]
                    + pipelines_v1.entity_class_vs_applications_pipeline(entity_class)
                ).to_list(None),
                "vs_topics": await collection.aggregate(
                    [combined_match]
                    + pipelines_v1.entity_class_vs_topics_pipeline(entity_class)
                ).to_list(None),
            }
            entity_classes_heatmaps.append(entity_class_data)

    except Exception as e:
        logger.error(f"Error executing RAG query analytics pipelines_v1: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

    # Sanitize all data
    bar_and_donut = sanitize_nan_values_recursive(bar_and_donut)
    heatmap = sanitize_nan_values_recursive(heatmap)
    stacked_bar = sanitize_nan_values_recursive(stacked_bar)
    top_cards = sanitize_nan_values_recursive(top_cards)
    side_cards = sanitize_nan_values_recursive(side_cards)
    area_chart = sanitize_nan_values_recursive(area_chart)
    entities_classes_vs_emotion = sanitize_nan_values_recursive(
        entities_classes_vs_emotion
    )
    applications_vs_topics = sanitize_nan_values_recursive(applications_vs_topics)
    topics_vs_emotions = sanitize_nan_values_recursive(topics_vs_emotions)
    entities_vs_emotions = sanitize_nan_values_recursive(entities_vs_emotions)
    entities_vs_applications = sanitize_nan_values_recursive(entities_vs_applications)
    entities_vs_topics = sanitize_nan_values_recursive(entities_vs_topics)

    # Sanitize entity class heatmaps
    for entity_class_data in entity_classes_heatmaps:
        entity_class_data["vs_emotions"] = sanitize_nan_values_recursive(
            entity_class_data["vs_emotions"]
        )
        entity_class_data["vs_applications"] = sanitize_nan_values_recursive(
            entity_class_data["vs_applications"]
        )
        entity_class_data["vs_topics"] = sanitize_nan_values_recursive(
            entity_class_data["vs_topics"]
        )

    # Set default top_cards if empty
    if not top_cards or len(top_cards) == 0:
        top_cards = [
            {
                "totalMessages": 0,
                "highRiskMessages": 0,
                "uniqueUsers": 0,
                "alertMessages": 0,
                "top_3_applications_message_count": {},
                "applications_message_count": {},
                "message_types_count": {},
                "top_3_entities_classes": {},
            }
        ]

    analytics_data = {
        "bar_and_donut": {
            "data": bar_and_donut,
            "chart_type": "bar_and_donut",
            "description": "Bar and Donut chart showing the distribution of topics, sentiments etc. for RAG query source messages.",
        },
        "heatmap": {
            "data": heatmap,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against emotions for RAG query source messages.",
        },
        "stacked_bar": {
            "data": stacked_bar,
            "chart_type": "stacked_bar",
            "description": "Stacked bar chart showing the distribution of risk levels per language for RAG query source messages.",
        },
        "area_chart": {
            "data": area_chart,
            "chart_type": "area_chart",
            "description": "Area chart showing the top 10 most repeated entities in RAG query source messages.",
        },
        "top_cards": {
            "data": top_cards,
            "chart_type": "top_cards",
            "description": "Top cards showing the unique users, total messages, alerted messages, high risk messages for RAG query source messages.",
        },
        "side_cards": {
            "data": side_cards,
            "chart_type": "side_cards",
            "description": "Side cards showing the last 5 messages, top users, messages by language, messages by risk level, most_occurring_entities for RAG query source messages.",
        },
        "entities_classes_vs_emotion": {
            "data": entities_classes_vs_emotion,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of entity classes against emotions for RAG query source messages.",
        },
        "applications_vs_topics": {
            "data": applications_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of applications against top topics for RAG query source messages.",
        },
        "topics_vs_emotions": {
            "data": topics_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top topics against emotions for RAG query source messages.",
        },
        "entities_vs_emotions": {
            "data": entities_vs_emotions,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against emotions for RAG query source messages.",
        },
        "entities_vs_applications": {
            "data": entities_vs_applications,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against applications for RAG query source messages.",
        },
        "entities_vs_topics": {
            "data": entities_vs_topics,
            "chart_type": "heatmap",
            "description": "Heatmap showing the distribution of top 10 entities against topics for RAG query source messages.",
        },
        "entity_classes_heatmaps": entity_classes_heatmaps,
        "query_info": {
            "total_source_messages": len(mongo_ids),
            "case_id": case_id,
            "case_name": case["name"],
            "chat_id": chat_id,
            "original_query": chat_entry.get("query", ""),
            "timestamp": (
                chat_entry.get("timestamp", "").isoformat()
                if chat_entry.get("timestamp")
                else None
            ),
        },
    }

    return JSONResponse(
        content=jsonable_encoder(analytics_data),
        status_code=200,
    )


@router.get("/debug/cases-list")
async def debug_list_cases(_: str = Depends(get_current_user)):
    """Debug endpoint to list all case IDs and names."""
    try:
        cases = await collection_case.find({}).limit(20).to_list(20)

        case_list = []
        for case in cases:
            case_list.append(
                {
                    "id": str(case["_id"]),
                    "name": case.get("name", "Unknown"),
                    "status": case.get("status", "Unknown"),
                    "noteClassifications": case.get("noteClassifications", []),
                    "browsingHistoryClassifications": case.get(
                        "browsingHistoryClassifications", []
                    ),
                }
            )

        return JSONResponse(
            content={"total_cases": len(case_list), "cases": case_list},
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error listing cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{case_id}/geolocations")
async def get_case_geolocations(case_id: str, _: str = Depends(get_current_user)):
    """Return precomputed geolocation data only (no on-demand recomputation)."""
    logger.info(f"Getting precomputed geolocation data for case_id: {case_id}")
    try:
        # Validate ObjectId format first
        try:
            object_id = ObjectId(case_id)
            logger.info(f"Valid ObjectId format: {object_id}")
        except Exception as e:
            logger.error(f"Invalid ObjectId format for case_id '{case_id}': {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid ObjectId format: {case_id}"
            )

        # Check if case exists
        case = await collection_case.find_one({"_id": object_id})
        if not case:
            logger.error(f"Case not found with _id: {object_id}")
            raise HTTPException(
                status_code=404, detail=f"Case not found with ID: {case_id}"
            )

        case_name = case.get("name")
        logger.info(f"Case found: {case_name} with ID: {str(case['_id'])}")

        # Read strictly from the precomputed geolocations collection
        geo_collection = db[f"{case_name}_{case_id}_geolocations"]

        geolocations_ready = bool(case.get("geolocations_ready", False))
        # Use count_documents to check if any docs exist
        docs_count = await geo_collection.count_documents({})

        # If not marked ready and nothing is stored yet, return 202 Accepted
        if not geolocations_ready and docs_count == 0:
            return JSONResponse(
                content={
                    "geolocations": [],
                    "total_locations": 0,
                    "case_info": {
                        "case_id": str(case["_id"]),
                        "case_name": case_name,
                    },
                },
                status_code=202,
            )

        # Fetch all precomputed geolocation documents
        docs = await geo_collection.find({}).to_list(None)

        # Normalize and sanitize for JSON response
        result = []
        for d in docs:
            # Remove Mongo internal _id
            d.pop("_id", None)
            # Ensure timestamp is a string
            ts = d.get("timestamp")
            if isinstance(ts, datetime):
                d["timestamp"] = ts.isoformat()
            result.append(d)

        logger.info(
            f"Returning {len(result)} precomputed geolocations for case {case_name}"
        )

        return JSONResponse(
            content={
                "geolocations": result,
                "total_locations": len(result),
                "case_info": {
                    "case_id": str(case["_id"]),
                    "case_name": case_name,
                },
            },
            status_code=200,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_case_geolocations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{case_id}/classification-summary")
async def get_case_classification_summary(
    case_id: str, _: str = Depends(get_current_user)
):
    """Get a summary of all classifications with counts and examples for a case."""
    logger.info(f"Getting classification summary for case_id: {case_id}")

    try:
        # Validate ObjectId format first
        try:
            object_id = ObjectId(case_id)
            logger.info(f"Valid ObjectId format: {object_id}")
        except Exception as e:
            logger.error(f"Invalid ObjectId format for case_id '{case_id}': {e}")
            raise HTTPException(
                status_code=400, detail=f"Invalid ObjectId format: {case_id}"
            )

        # Check if case exists
        case = await collection_case.find_one({"_id": object_id})
        if not case:
            logger.error(f"Case not found with _id: {object_id}")
            raise HTTPException(
                status_code=404, detail=f"Case not found with ID: {case_id}"
            )

        logger.info(f"Case found: {case.get('name')} with ID: {str(case['_id'])}")

        # Get the collection for this case
        collection = db[f"{case['name']}_{case_id}"]

        # Get configured classifications
        note_classes = case.get("noteClassifications", [])
        browsing_classes = case.get("browsingHistoryClassifications", [])

        # Initialize summary structure
        summary = {
            "case_info": {
                "case_id": str(case["_id"]),
                "case_name": case["name"],
                "noteClassifications": note_classes,
                "browsingHistoryClassifications": browsing_classes,
            },
            "note_classifications": {},
            "browsing_history_classifications": {},
            "total_classified_messages": 0,
            "classification_stats": {
                "notes_processed": 0,
                "browsing_history_processed": 0,
                "notes_classified": 0,
                "browsing_history_classified": 0,
            },
        }

        # Process note classifications
        if note_classes:
            for class_name in note_classes:
                # Count messages with this classification
                count = await collection.count_documents(
                    {
                        "Message Type": "note",
                        "analysis_summary.note_classification": class_name,
                    }
                )

                # Get a sample message
                sample = await collection.find_one(
                    {
                        "Message Type": "note",
                        "analysis_summary.note_classification": class_name,
                    }
                )

                summary["note_classifications"][class_name] = {
                    "count": count,
                    "sample": (
                        {
                            "text": (
                                sample.get("Preview Text", "")[:100] + "..."
                                if sample and sample.get("Preview Text")
                                else ""
                            ),
                            "title": sample.get("Title", "") if sample else "",
                            "timestamp": sample.get("timestamp") if sample else None,
                        }
                        if sample
                        else None
                    ),
                }

        # Process browsing history classifications
        if browsing_classes:
            for class_name in browsing_classes:
                # Count messages with this classification
                count = await collection.count_documents(
                    {
                        "Message Type": "browsing_history",
                        "analysis_summary.browsing_history_classification": class_name,
                    }
                )

                # Get a sample message
                sample = await collection.find_one(
                    {
                        "Message Type": "browsing_history",
                        "analysis_summary.browsing_history_classification": class_name,
                    }
                )

                summary["browsing_history_classifications"][class_name] = {
                    "count": count,
                    "sample": (
                        {
                            "text": (
                                sample.get("Preview Text", "")[:100] + "..."
                                if sample and sample.get("Preview Text")
                                else ""
                            ),
                            "url": sample.get("url", "") if sample else "",
                            "title": sample.get("title", "") if sample else "",
                            "timestamp": sample.get("timestamp") if sample else None,
                        }
                        if sample
                        else None
                    ),
                }

        # Get overall statistics
        total_notes = await collection.count_documents({"Message Type": "note"})
        total_browsing = await collection.count_documents(
            {"Message Type": "browsing_history"}
        )

        total_classified_notes = await collection.count_documents(
            {
                "Message Type": "note",
                "analysis_summary.note_classification": {"$exists": True, "$ne": None},
            }
        )

        total_classified_browsing = await collection.count_documents(
            {
                "Message Type": "browsing_history",
                "analysis_summary.browsing_history_classification": {
                    "$exists": True,
                    "$ne": None,
                },
            }
        )

        summary["classification_stats"] = {
            "notes_processed": total_notes,
            "browsing_history_processed": total_browsing,
            "notes_classified": total_classified_notes,
            "browsing_history_classified": total_classified_browsing,
        }

        summary["total_classified_messages"] = (
            total_classified_notes + total_classified_browsing
        )

        # Add top classifications by count
        summary["top_classifications"] = {
            "notes": sorted(
                [
                    (class_name, data["count"])
                    for class_name, data in summary["note_classifications"].items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
            "browsing_history": sorted(
                [
                    (class_name, data["count"])
                    for class_name, data in summary[
                        "browsing_history_classifications"
                    ].items()
                ],
                key=lambda x: x[1],
                reverse=True,
            )[:5],
        }

        return JSONResponse(
            content=summary,
            status_code=200,
        )

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_case_classification_summary: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
