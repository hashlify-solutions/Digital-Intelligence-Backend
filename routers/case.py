import math
import os
from fastapi import APIRouter, Depends, HTTPException, Form
from fastapi import UploadFile, BackgroundTasks, HTTPException, Query, File
from typing import Optional
from bson import ObjectId
from utils.auth import get_current_user
from config.db import db, collection_case, users_collection
from config.db import qdrant_client
from pathlib import Path
from analyzer import ArabicSocialAnalyzer
from fastapi.responses import JSONResponse
from typing import List, Optional
from setup import setup_logging
from rag import ArabicRagAnalyzer
from utils import pipelines
from fastapi.encoders import jsonable_encoder
from config.db import (
    processing_profiles_collection, 
    models_master_collection, 
    ufdr_files_collection,
    ufdr_calls_collection,
    ufdr_chats_collection,
    ufdr_emails_collection,
    ufdr_locations_collection,
    ufdr_notes_collection,
    ufdr_searched_items_collection,
    ufdr_user_accounts_collection,
    ufdr_audio_collection,
    ufdr_photos_collection,
    ufdr_videos_collection,
    ufdr_photo_detected_faces_collection,
    ufdr_photo_detected_objects_collection,
    ufdr_video_detected_faces_collection,
    ufdr_video_detected_objects_collection,
    ufdr_video_screenshots_collection,
    detector_matches_collection
)
from schemas.case import GetMessagesByIdsRequest
from tasks.celery_tasks import process_csv_upload, process_ufdr_upload
from celery_app import celery_app
from utils.helpers import sanitize_nan_values, sanitize_nan_values_recursive, serialize_mongodb_document
import shutil
from datetime import datetime, timezone
from config.settings import settings

router = APIRouter()
logger = setup_logging()
UPLOAD_DIR = settings.upload_dir

@router.post("/upload-data", response_model=dict)
async def upload_case_data(
    file: UploadFile = File(...),
    topics: str = Form(...),
    sentiments: str = Form(...),
    interactions: str = Form(...),
    entitiesClasses: str = Form(...),
    caseName: str = Form(...),
    category: str = Form(...),
    is_rag: bool = Form(...),
    alert_id: Optional[str] = Form(None),
    models_profile_id: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=400, detail="Only CSV files are allowed in CSV mode."
            )

        if is_rag:
            topics_list = topics.split(",") if topics else []
            sentiments_list = sentiments.split(",") if sentiments else []
            interactions_list = interactions.split(",") if interactions else []

            if not all([topics_list, sentiments_list, interactions_list]):
                raise HTTPException(
                    status_code=400,
                    detail="Topics, Sentiments, and Interactions are required for RAG data.",
                )

        logger.info(f"Uploading file: {file.filename}")

        # Convert form data to lists
        topics_list = topics.split(",") if topics else []
        sentiments_list = sentiments.split(",") if sentiments else []
        interactions_list = interactions.split(",") if interactions else []
        entities_classes_list = entitiesClasses.split(",") if entitiesClasses else []

        if models_profile_id:
            models_profile = await processing_profiles_collection.find_one(
                {"_id": ObjectId(models_profile_id)}
            )
            if models_profile:
                logger.info("Model profile has been found with data")
                logger.info(models_profile)
            if not models_profile:
                raise HTTPException(status_code=404, detail="Model Profile not found.")
        else:
            models_profile = await models_master_collection.find_one(
                {"name": "DEFAULT PROFILE"}
            )
            logger.info(
                "Using the default model profile from models_master collection with data"
            )
            logger.info(models_profile)

        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        case_data = {
            "name": caseName,
            "status": "pending",
            "user_id": ObjectId(user_id),
            "topics": topics_list,
            "sentiments": sentiments_list,
            "interactions": interactions_list,
            "is_rag": is_rag,
            "model_profile": models_profile.get("_id", ""),
            "entitiesClasses": entities_classes_list,
            "category": category,
            "processing_started_at": datetime.now(timezone.utc).isoformat(),
        }

        if alert_id:
            case_data["alert_id"] = ObjectId(alert_id)

        case = await collection_case.insert_one(case_data)

        folder_path = os.path.join(f"{UPLOAD_DIR}/{caseName}_{case.inserted_id}/csv")
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(folder_path, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Converting models_profile to a serializable format
        models_profile_serializable = {
            "_id": str(models_profile.get("_id", "")),
            "classifier": models_profile.get("classifier", {}),
            "toxic": models_profile.get("toxic", {}),
            "emotion": models_profile.get("emotion", {}),
            "embeddings": models_profile.get("embeddings", {}),
        }

        task_result = process_csv_upload.delay(
            file_path=file_path,
            case_name=caseName,
            case_id=str(case.inserted_id),
            alert_id=str(alert_id) if alert_id else None,
            topics=topics_list,
            sentiments=sentiments_list,
            interactions=interactions_list,
            entitiesClasses=entities_classes_list,
            is_rag=is_rag,
            models_profile=models_profile_serializable,
        )

        return JSONResponse(
            content={
                "message": "File uploaded successfully. Processing started.",
                "case_id": str(case.inserted_id),
                "task_id": task_result.id,
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Error uploading case: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading case: {str(e)}")


@router.post("/upload-ufdr-data", response_model=dict)
async def upload_ufdr_data(
    file: UploadFile = File(...),
    topics: str = Form(...),
    sentiments: str = Form(...),
    interactions: str = Form(...),
    entitiesClasses: str = Form(...),
    caseName: str = Form(...),
    category: str = Form(...),
    is_rag: bool = Form(...),
    alert_id: Optional[str] = Form(None),
    models_profile_id: Optional[str] = Form(None),
    case_id: Optional[str] = Form(None),
    user_id: str = Depends(get_current_user),
):
    """
    Upload UFDR data file to an existing case or create a new case.
    
    If case_id is provided, the UFDR file will be uploaded to the existing case.
    If case_id is not provided, a new case will be created with the provided case details.
    """
    try:
        # Validate UFDR file extension
        if not file.filename.lower().endswith((".ufdr", ".UFDR")):
            raise HTTPException(
                status_code=400, detail="Only UFDR files are allowed."
            )

        if is_rag:
            topics_list = topics.split(",") if topics else []
            sentiments_list = sentiments.split(",") if sentiments else []
            interactions_list = interactions.split(",") if interactions else []

            if not all([topics_list, sentiments_list, interactions_list]):
                raise HTTPException(
                    status_code=400,
                    detail="Topics, Sentiments, and Interactions are required for RAG data.",
                )

        logger.info(f"Uploading UFDR file: {file.filename}")

        # Convert form data to lists
        topics_list = topics.split(",") if topics else []
        sentiments_list = sentiments.split(",") if sentiments else []
        interactions_list = interactions.split(",") if interactions else []
        entities_classes_list = entitiesClasses.split(",") if entitiesClasses else []

        # Get model profile
        if models_profile_id:
            models_profile = await processing_profiles_collection.find_one(
                {"_id": ObjectId(models_profile_id)}
            )
            if models_profile:
                logger.info("Model profile has been found with data")
                logger.info(models_profile)
            if not models_profile:
                raise HTTPException(status_code=404, detail="Model Profile not found.")
        else:
            models_profile = await models_master_collection.find_one(
                {"name": "DEFAULT PROFILE"}
            )
            logger.info(
                "Using the default model profile from models_master collection with data"
            )
            logger.info(models_profile)

        # Validate user
        user = await users_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        # Handle case creation or use existing case
        if case_id:
            # Validate case_id format
            if not ObjectId.is_valid(case_id):
                raise HTTPException(status_code=400, detail="Invalid case ID format")
            
            # Check if case exists and user has access
            existing_case = await collection_case.find_one({
                "_id": ObjectId(case_id),
                "user_id": ObjectId(user_id)
            })
            if not existing_case:
                raise HTTPException(status_code=404, detail="Case not found or access denied")
            
            case_object_id = ObjectId(case_id)
            case_name = existing_case.get("name")
            topics_list = existing_case.get("topics", [])
            sentiments_list = existing_case.get("sentiments", [])
            interactions_list = existing_case.get("interactions", [])
            entities_classes_list = existing_case.get("entitiesClasses", [])
            logger.info(f"Using existing case: {case_id}")

            await collection_case.update_one(
                {"_id": case_object_id},
                {"$set": {"processing_started_at": datetime.now(timezone.utc).isoformat()}},
            )
        else:
            # Create new case
            case_data = {
                "name": caseName,
                "status": "pending",
                "category": category,
                "user_id": ObjectId(user_id),
                "topics": topics_list,
                "sentiments": sentiments_list,
                "interactions": interactions_list,
                "is_rag": is_rag,
                "model_profile": models_profile.get("_id", ""),
                "entitiesClasses": entities_classes_list,
                "processing_started_at": datetime.now(timezone.utc).isoformat(),
            }

            if alert_id:
                case_data["alert_id"] = ObjectId(alert_id)

            case = await collection_case.insert_one(case_data)
            case_object_id = case.inserted_id
            case_name = case_data.get("name")
            logger.info(f"Created new case: {case_object_id}")

        # Create UFDR file record
        ufdr_file_data = {
            "name": file.filename,
            "caseId": case_object_id,
            "file_size": 0,  
            "created_at": datetime.now(),  
            "updated_at": datetime.now()   
        }
        ufdr_file = await ufdr_files_collection.insert_one(ufdr_file_data)

        # Create directory path for UFDR file with ufdr_file ID
        directory_path = os.path.join(f"{UPLOAD_DIR}/{case_name}_{case_object_id}/ufdr/{ufdr_file.inserted_id}")
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        
        # Create full file path
        file_path = os.path.join(directory_path, file.filename)
        
        # Stream large file to disk to handle large UFDR files efficiently
        logger.info(f"Starting to write large UFDR file: {file.filename}")
        
        # Check available disk space before starting upload
        free_space = shutil.disk_usage(UPLOAD_DIR).free
        logger.info(f"Available disk space: {free_space / (1024**3):.2f} GB")
        
        chunk_size = 1024 * 1024  # 1MB chunks for faster file writing with large files
        total_bytes_written = 0
        
        with open(file_path, "wb") as f:
            while chunk := await file.read(chunk_size):
                f.write(chunk)
                total_bytes_written += len(chunk)
                # Log progress for very large files every 1GB
                if total_bytes_written % (1024 * 1024 * 1024) == 0:
                    logger.info(f"Written {total_bytes_written / (1024**3):.2f} GB of {file.filename}")
        
        logger.info(f"Successfully saved UFDR file to: {file_path} (Total size: {total_bytes_written / (1024**3):.2f} GB)")

        # Update UFDR file record with actual file size
        await ufdr_files_collection.update_one(
            {"_id": ufdr_file.inserted_id},
            {"$set": {"file_size": total_bytes_written, "updated_at": datetime.now()}} 
        )

        # Now that file is saved, pass the file path to Celery task for processing
        task_result = process_ufdr_upload.delay(
            input_ufdr_path=file_path,
            output_ufdr_path=directory_path,
            case_name=case_name,
            case_id=str(case_object_id),
            ufdr_file_id=str(ufdr_file.inserted_id),
            alert_id=str(alert_id) if alert_id else None,
            topics=topics_list,
            sentiments=sentiments_list,
            interactions=interactions_list,
            entitiesClasses=entities_classes_list,
            model_profile=serialize_mongodb_document(models_profile),
        )

        return JSONResponse(
            content={
                "message": "UFDR file uploaded successfully. Processing started.",
                "case_id": str(case_object_id),
                "ufdr_file_id": str(ufdr_file.inserted_id),
                "task_id": str(task_result.id),
                "file_type": "ufdr",
                "file_size_bytes": total_bytes_written,
                "file_size_gb": round(total_bytes_written / (1024**3), 2),
                "file_path": file_path
            },
            status_code=200,
        )

    except Exception as e:
        logger.error(f"Error uploading UFDR case: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading UFDR case: {str(e)}")


@router.delete("/{case_id}")
async def delete_case(case_id: str, user_id: str = Depends(get_current_user)):
    try:
        case = await collection_case.find_one(
            {"_id": ObjectId(case_id), "user_id": ObjectId(user_id)}
        )
        if not case:
            raise HTTPException(
                status_code=404,
                detail="Case not found or you don't have permission to access it",
            )
        await collection_case.delete_one(
            {"_id": ObjectId(case_id), "user_id": ObjectId(user_id)}
        )
        case_messages_collection = db[f"{case['name']}_{case_id}"]
        await case_messages_collection.drop()
        qdrant_client.delete_collection(f"case_{case_id}")
        qdrant_client.delete_collection(f"case_{case_id}_media")
        case_dir = os.path.join(UPLOAD_DIR, f"{case['name']}_{case_id}")
        if os.path.exists(case_dir):
            shutil.rmtree(case_dir)
            logger.info(f"Deleted case directory: {case_dir}")
        # Delete all UFDR data for this case if any exists
        try:
            # Check if there are any UFDR files for this case
            ufdr_files_count = await ufdr_files_collection.count_documents({"caseId": ObjectId(case_id)})
            
            if ufdr_files_count > 0:
                logger.info(f"Found {ufdr_files_count} UFDR files for case {case_id}. Deleting UFDR data...")
                
                # Delete all UFDR data by calling the internal logic
                # (We'll replicate the logic from the delete_all_ufdr_data_for_case endpoint)
                
                # Get all UFDR files for this case
                ufdr_files = await ufdr_files_collection.find({"caseId": ObjectId(case_id)}).to_list(None)
                
                # Delete data from all shared collections for this case
                collections_to_clean = [
                    ("ufdr_calls", ufdr_calls_collection),
                    ("ufdr_chats", ufdr_chats_collection),
                    ("ufdr_emails", ufdr_emails_collection),
                    ("ufdr_locations", ufdr_locations_collection),
                    ("ufdr_notes", ufdr_notes_collection),
                    ("ufdr_searched_items", ufdr_searched_items_collection),
                    ("ufdr_user_accounts", ufdr_user_accounts_collection),
                    ("ufdr_audio", ufdr_audio_collection),
                    ("ufdr_photos", ufdr_photos_collection),
                    ("ufdr_videos", ufdr_videos_collection),
                    ("ufdr_photo_detected_faces", ufdr_photo_detected_faces_collection),
                    ("ufdr_photo_detected_objects", ufdr_photo_detected_objects_collection),
                    ("ufdr_video_detected_faces", ufdr_video_detected_faces_collection),
                    ("ufdr_video_detected_objects", ufdr_video_detected_objects_collection),
                    ("ufdr_video_screenshots", ufdr_video_screenshots_collection),
                    ("detector_matches", detector_matches_collection)
                ]
                
                # Delete from each collection using case_id
                total_deleted = 0
                for collection_name, collection in collections_to_clean:
                    try:
                        delete_result = await collection.delete_many({"case_id": ObjectId(case_id)})
                        if delete_result.deleted_count > 0:
                            total_deleted += delete_result.deleted_count
                            logger.info(f"Deleted {delete_result.deleted_count} documents from {collection_name}")
                    except Exception as e:
                        logger.error(f"Error deleting from {collection_name}: {str(e)}")
                
                # Delete UFDR files and directories from disk
                for ufdr_file in ufdr_files:
                    ufdr_file_id = str(ufdr_file["_id"])
                    try:
                        # Delete UFDR directory from disk
                        ufdr_directory = os.path.join(f"{UPLOAD_DIR}/{case['name']}_{case_id}/ufdr/{ufdr_file_id}")
                        if os.path.exists(ufdr_directory):
                            shutil.rmtree(ufdr_directory)
                            logger.info(f"Deleted UFDR directory: {ufdr_directory}")
                    except Exception as e:
                        logger.error(f"Error deleting UFDR directory {ufdr_file_id}: {str(e)}")
                
                # Delete all UFDR file records
                await ufdr_files_collection.delete_many({"caseId": ObjectId(case_id)})
                
                logger.info(f"Successfully deleted all UFDR data for case {case_id}. Total documents deleted: {total_deleted}")
            else:
                logger.info(f"No UFDR files found for case {case_id}")
                
        except Exception as e:
            logger.error(f"Error deleting UFDR data for case {case_id}: {str(e)}")
            # Don't raise the exception here as we want the case deletion to continue
        return JSONResponse(
            content={"message": "Case deleted successfully"}, status_code=200
        )
    except Exception as e:
        logger.error(f"Error deleting case: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting case: {str(e)}")


@router.get("/task-status/{task_id}")
async def check_task_status(task_id: str, _: str = Depends(get_current_user)):
    """Check the status of a Celery task"""
    try:
        result = celery_app.AsyncResult(task_id)
        task_info = {
            "task_id": task_id,
            "status": result.status,
            "result": result.result,
        }
        return JSONResponse(
            content=task_info,
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error checking task status: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error checking task status: {str(e)}"
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
        cases.append(case)

    return JSONResponse(
        content=cases,
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
async def get_case(case_id: str, _: str = Depends(get_current_user)):
    try:
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid ObjectId format")

    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    try:
        collection = db[f"{case['name']}_{case_id}"]
        case_data_cursor = await collection.find(
            {"case_id": ObjectId(case_id)}
        ).to_list(None)
        case_data = []
        for data in case_data_cursor:
            data["_id"] = str(data["_id"])
            data["case_id"] = str(data["case_id"])
            data = sanitize_nan_values(data)
            case_data.append(data)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Collection not found")

    return JSONResponse(
        content=case_data,
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
    interaction_type: Optional[List[str]] = Query(None),
    entities: Optional[List[str]] = Query(None),
    entities_classes: Optional[List[str]] = Query(None),
    alert: Optional[bool] = Query(None),
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
        models_profile_id = case.get("model_profile", "")

        if models_profile_id:
            models_profile = await processing_profiles_collection.find_one(
                {"_id": ObjectId(models_profile_id)}
            )
            if not models_profile:
                raise HTTPException(status_code=404, detail="Model Profile not found.")
        else:
            models_profile = await models_master_collection.find().to_list(None)
            models_profile = models_profile[1]

        rag = ArabicRagAnalyzer(
            collection_case, collection_case, case_id, models_profile
        )
        items = rag.semantic_search(query=sematic_search, collection_name=f"case_{case_id}", limit=3, score_threshold=0.15)
        if not items or len(items) == 0:
            raise HTTPException(
                status_code=404, detail="No messages found with the given filters"
            )
        # log each item in items
        logger.info(f"Items found: {items}")
        mongo_ids = [item.payload.get("mongo_id") for item in items]
        query["_id"] = {"$in": [ObjectId(id) for id in mongo_ids]}
        logger.info(f"Query after semantic search: {query}")

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

    alert = alert if alert else None
    if top_topic:
        logger.info(f"top topic: {top_topic}")
        query["analysis_summary.top_topic"] = {"$in": top_topic}
    if toxicity_score:
        logger.info(f"toxicity: f{top_topic}")
        query["analysis_summary.toxicity_score"] = {"$gte": int(toxicity_score)}
    if sentiment_aspect:
        logger.info(f"sentiment: f{top_topic}")
        query["analysis_summary.sentiment_aspect"] = {"$in": sentiment_aspect}
    if emotion:
        logger.info(f"emotion: f{emotion}")
        query["analysis_summary.emotion"] = {"$in": emotion}
    if language:
        logger.info(f"language: f{language}")
        query["analysis_summary.language"] = {"$in": language}
    if risk_level:
        logger.info(f"risk_level: f{risk_level}")
        query["analysis_summary.risk_level"] = {"$in": risk_level}
    if application_type:
        logger.info(f"application_type: f{application_type}")
        query["Application"] = {"$in": application_type}
    if interaction_type:
        logger.info(f"interaction_type: f{interaction_type}")
        query["analysis_summary.interaction_type"] = {"$in": interaction_type}
    if entities:
        logger.info(f"entities: f{entities}")
        query["analysis_summary.entities"] = {"$in": entities}
    if entities_classes:
        logger.info(f"entities_classes: {entities_classes}")
        # Create an $or array of conditions to check if any of the specified classes exist as keys
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
        logger.info(f"alert filter: {alert}")
        query["alert"] = alert

    casesItems = []
    try:
        # Access the collection based on case
        collection = db[f"{case['name']}_{case_id}"]
        logger.info(f"Querying collection: {case['name']}_{case_id} with query {query}")

        # Count total documents matching the query for pagination
        total_count = await collection.count_documents(query)
        if total_count == 0:
            raise HTTPException(
                status_code=404, detail="No messages found with the given filters"
            )

        # Calculate skip based on page and limit
        skip = (page - 1) * limit

        # Fetch paginated data
        case_data_cursor = (
            await collection.find(query).skip(skip).limit(limit).to_list(None)
        )
        logger.info(f"Found {len(case_data_cursor)} messages for page {page}")

        # Process the result
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
                "total_pages": math.ceil(total_count / limit),  # Ceiling division
            },
        },
        status_code=200,
    )


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
    case_id: str, query: str, _: str = Depends(get_current_user)
):
    try:
        id = case_id
        case = await collection_case.find_one({"_id": ObjectId(id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        collection = db[f"{case['name']}_{id}"]

        models_profile_id = case.get("model_profile", "")

        if models_profile_id:
            models_profile = await processing_profiles_collection.find_one(
                {"_id": ObjectId(models_profile_id)}
            )
            if not models_profile:
                raise HTTPException(status_code=404, detail="Model Profile not found.")
        else:
            models_profile = await models_master_collection.find().to_list(None)
            models_profile = models_profile[1]

        analyzer = ArabicRagAnalyzer(collection, collection_case, id, models_profile)
        summary = analyzer.summarize_messages(query)

        if summary is None or summary.get("success") == False:
            raise HTTPException(status_code=404, detail="Failed to generate summary.")

        logger.info(f"Summary Response: {summary}")

        return {
            "summary": summary.get("summary"),
            "mongo_ids": summary.get("mongo_ids"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing user query: {e}")


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

        # Execute existing pipelines
        bar_and_donut = await collection.aggregate(
            pipeline=pipelines.bar_and_pipe_chart_pipeline
        ).to_list(None)
        heatmap = await collection.aggregate(
            pipeline=pipelines.heatmap_of_appilication_against_emotion_piepline
        ).to_list(None)
        stacked_bar = await collection.aggregate(
            pipeline=pipelines.stacked_bar_risk_per_lanaguage
        ).to_list(None)
        top_cards = await collection.aggregate(
            pipeline=pipelines.top_card_data_pipeline
        ).to_list(None)
        side_cards = await collection.aggregate(
            pipeline=pipelines.side_card_data_pipeline
        ).to_list(None)
        area_chart = await collection.aggregate(
            pipeline=pipelines.area_chart_entities_pipeline
        ).to_list(None)

        entities_classes_vs_emotion = await collection.aggregate(
            pipeline=pipelines.entities_classes_vs_emotion_pipeline
        ).to_list(None)
        applications_vs_topics = await collection.aggregate(
            pipeline=pipelines.applications_vs_topics_pipeline
        ).to_list(None)
        topics_vs_emotions = await collection.aggregate(
            pipeline=pipelines.topics_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_emotions = await collection.aggregate(
            pipeline=pipelines.entities_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_applications = await collection.aggregate(
            pipeline=pipelines.entities_vs_applications_pipeline
        ).to_list(None)
        entities_vs_topics = await collection.aggregate(
            pipeline=pipelines.entities_vs_topics_pipeline
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
                    pipeline=pipelines.entity_class_vs_emotions_pipeline(entity_class)
                ).to_list(None),
                "vs_applications": await collection.aggregate(
                    pipeline=pipelines.entity_class_vs_applications_pipeline(
                        entity_class
                    )
                ).to_list(None),
                "vs_topics": await collection.aggregate(
                    pipeline=pipelines.entity_class_vs_topics_pipeline(entity_class)
                ).to_list(None),
            }
            entity_classes_heatmaps.append(entity_class_data)

    except Exception as e:
        logger.error(f"Error executing pipelines: {e}")
        raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

    # Sanitize the data
    bar_and_donut = sanitize_nan_values_recursive(bar_and_donut)
    heatmap = sanitize_nan_values_recursive(heatmap)
    stacked_bar = sanitize_nan_values_recursive(stacked_bar)
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

    # If top_cards is empty or doesn't contain the expected fields, provide defaults
    if not top_cards or len(top_cards) == 0:
        top_cards = [
            {
                "totalMessages": 0,
                "highRiskMessages": 0,
                "uniqueUsers": 0,
                "alertMessages": 0,
                "top_3_applications_message_count": {},
                "top_3_entities_classes": {},
            }
        ]

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
        models_profile_id = case.get("model_profile", "")

        if models_profile_id:
            models_profile = await processing_profiles_collection.find_one(
                {"_id": ObjectId(models_profile_id)}
            )
            if not models_profile:
                raise HTTPException(status_code=404, detail="Model Profile not found.")
        else:
            models_profile = await models_master_collection.find().to_list(None)
            models_profile = models_profile[1]

        rag = ArabicRagAnalyzer(
            collection_case, collection_case, case_id, models_profile
        )
        items = rag.semantic_search(query=sematic_search, collection_name=f"case_{case_id}", limit=3, score_threshold=0.15)
        if not items or len(items) == 0:
            raise HTTPException(
                status_code=404, detail="No messages found with the given filters"
            )
        mongo_ids = [item.payload.get("mongo_id") for item in items]
        query["_id"] = {"$in": [ObjectId(id) for id in mongo_ids]}

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
        query["Application"] = {"$in": application_type}
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

    try:
        collection = db[f"{case['name']}_{case_id}"]

        # Add match stage with query to all pipelines
        match_stage = {"$match": query}

        # Execute pipelines with filter
        bar_and_donut = await collection.aggregate(
            [match_stage] + pipelines.bar_and_pipe_chart_pipeline
        ).to_list(None)
        heatmap = await collection.aggregate(
            [match_stage] + pipelines.heatmap_of_appilication_against_emotion_piepline
        ).to_list(None)
        stacked_bar = await collection.aggregate(
            [match_stage] + pipelines.stacked_bar_risk_per_lanaguage
        ).to_list(None)
        top_cards = await collection.aggregate(
            [match_stage] + pipelines.top_card_data_pipeline
        ).to_list(None)
        side_cards = await collection.aggregate(
            [match_stage] + pipelines.side_card_data_pipeline
        ).to_list(None)
        area_chart = await collection.aggregate(
            [match_stage] + pipelines.area_chart_entities_pipeline
        ).to_list(None)

        # Execute heatmap pipelines with filter
        entities_classes_vs_emotion = await collection.aggregate(
            [match_stage] + pipelines.entities_classes_vs_emotion_pipeline
        ).to_list(None)
        applications_vs_topics = await collection.aggregate(
            [match_stage] + pipelines.applications_vs_topics_pipeline
        ).to_list(None)
        topics_vs_emotions = await collection.aggregate(
            [match_stage] + pipelines.topics_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_emotions = await collection.aggregate(
            [match_stage] + pipelines.entities_vs_emotions_pipeline
        ).to_list(None)
        entities_vs_applications = await collection.aggregate(
            [match_stage] + pipelines.entities_vs_applications_pipeline
        ).to_list(None)
        entities_vs_topics = await collection.aggregate(
            [match_stage] + pipelines.entities_vs_topics_pipeline
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

        # Generate entity class specific heatmaps
        entity_classes_heatmaps = []
        for entity_class in distinct_entity_classes:
            entity_class_data = {
                "entity_class": entity_class,
                "vs_emotions": await collection.aggregate(
                    [match_stage]
                    + pipelines.entity_class_vs_emotions_pipeline(entity_class)
                ).to_list(None),
                "vs_applications": await collection.aggregate(
                    [match_stage]
                    + pipelines.entity_class_vs_applications_pipeline(entity_class)
                ).to_list(None),
                "vs_topics": await collection.aggregate(
                    [match_stage]
                    + pipelines.entity_class_vs_topics_pipeline(entity_class)
                ).to_list(None),
            }
            entity_classes_heatmaps.append(entity_class_data)

    except Exception as e:
        logger.error(f"Error executing pipelines: {e}")
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
                "top_3_entities_classes": {},
            }
        ]

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
        "entity_classes_heatmaps": entity_classes_heatmaps,
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


@router.get("/ufdr-files/{case_id}")
async def get_ufdr_files_by_case(case_id: str, _: str = Depends(get_current_user)):
    """Get all UFDR files for a specific case"""
    try:
        # Validate case exists
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Get all UFDR files for this case
        ufdr_files = await ufdr_files_collection.find(
            {"caseId": ObjectId(case_id)}
        ).to_list(None)
        
        # Convert ObjectIds to strings
        for ufdr_file in ufdr_files:
            ufdr_file["_id"] = str(ufdr_file["_id"])
            ufdr_file["caseId"] = str(ufdr_file["caseId"])
        
        return JSONResponse(
            content={
                "case_id": case_id,
                "ufdr_files": ufdr_files,
                "total_files": len(ufdr_files)
            },
            status_code=200,
        )
        
    except Exception as e:
        logger.error(f"Error retrieving UFDR files: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving UFDR files: {str(e)}")


@router.get("/ufdr-data/{ufdr_file_id}/{data_type}")
async def get_ufdr_data(
    ufdr_file_id: str,
    data_type: str,
    page: int = Query(1, alias="page", ge=1),
    limit: int = Query(10, alias="limit", le=100),
    _: str = Depends(get_current_user)
):
    """Get UFDR data by type (calls, chats, emails, etc.) for a specific UFDR file"""
    try:
        # Validate UFDR file exists
        ufdr_file = await ufdr_files_collection.find_one({"_id": ObjectId(ufdr_file_id)})
        if not ufdr_file:
            raise HTTPException(status_code=404, detail="UFDR file not found")
        
        # Map data types to collections
        collection_map = {
            "calls": ufdr_calls_collection,
            "chats": ufdr_chats_collection,
            "emails": ufdr_emails_collection,
            "locations": ufdr_locations_collection,
            "notes": ufdr_notes_collection,
            "searched_items": ufdr_searched_items_collection,
            "user_accounts": ufdr_user_accounts_collection,
            "audio": ufdr_audio_collection,
            "photos": ufdr_photos_collection,
            "videos": ufdr_videos_collection
        }
        
        if data_type not in collection_map:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data type. Available types: {list(collection_map.keys())}"
            )
        
        collection = collection_map[data_type]
        
        # Build query
        query = {"ufdr_id": ObjectId(ufdr_file_id)}
        
        # Get total count
        total_count = await collection.count_documents(query)
        
        # Calculate pagination
        skip = (page - 1) * limit
        
        # Get paginated data
        data = await collection.find(query).skip(skip).limit(limit).sort("created_at", -1).to_list(None)
        
        # Convert ObjectIds to strings
        for item in data:
            item["_id"] = str(item["_id"])
            item["ufdr_id"] = str(item["ufdr_id"])
            item["case_id"] = str(item["case_id"])
            item["created_at"] = str(item["created_at"])
            item["updated_at"] = str(item["updated_at"])
            item = sanitize_nan_values(item)
        
        return JSONResponse(
            content={
                "ufdr_file_id": ufdr_file_id,
                "data_type": data_type,
                "data": data,
                "pagination": {
                    "total": total_count,
                    "page": page,
                    "limit": limit,
                    "total_pages": math.ceil(total_count / limit) if total_count > 0 else 0,
                },
            },
            status_code=200,
        )
        
    except Exception as e:
        logger.error(f"Error retrieving UFDR data: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving UFDR data: {str(e)}")


@router.get("/ufdr-analytics/{case_id}")
async def get_ufdr_analytics_by_case(case_id: str, _: str = Depends(get_current_user)):
    """Get analytics across all UFDR files for a case"""
    try:
        # Validate case exists
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        case_object_id = ObjectId(case_id)
        
        # Get counts for each data type
        analytics = {
            "case_id": case_id,
            "total_calls": await ufdr_calls_collection.count_documents({"case_id": case_object_id}),
            "total_chats": await ufdr_chats_collection.count_documents({"case_id": case_object_id}),
            "total_emails": await ufdr_emails_collection.count_documents({"case_id": case_object_id}),
            "total_locations": await ufdr_locations_collection.count_documents({"case_id": case_object_id}),
            "total_notes": await ufdr_notes_collection.count_documents({"case_id": case_object_id}),
            "total_searched_items": await ufdr_searched_items_collection.count_documents({"case_id": case_object_id}),
            "total_user_accounts": await ufdr_user_accounts_collection.count_documents({"case_id": case_object_id}),
            "total_audio_files": await ufdr_audio_collection.count_documents({"case_id": case_object_id}),
            "total_photos": await ufdr_photos_collection.count_documents({"case_id": case_object_id}),
            "total_videos": await ufdr_videos_collection.count_documents({"case_id": case_object_id}),
        }
        
        # Get UFDR files summary
        ufdr_files = await ufdr_files_collection.find({"caseId": case_object_id}).to_list(None)
        analytics["total_ufdr_files"] = len(ufdr_files)
        analytics["ufdr_files_summary"] = []
        
        for ufdr_file in ufdr_files:
            summary = {
                "ufdr_file_id": str(ufdr_file["_id"]),
                "name": ufdr_file["name"],
                "file_size_gb": round(ufdr_file["file_size"] / (1024**3), 2),
                "associated_schemas": ufdr_file.get("associated_schema_names", []),
                "created_at": ufdr_file.get("created_at")
            }
            analytics["ufdr_files_summary"].append(summary)
        
        return JSONResponse(content=analytics, status_code=200)
        
    except Exception as e:
        logger.error(f"Error retrieving UFDR analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving UFDR analytics: {str(e)}")


@router.delete("/ufdr-data/{ufdr_file_id}")
async def delete_ufdr_data(
    ufdr_file_id: str,
    case_id: str = Query(..., description="Case ID for validation"),
    user_id: str = Depends(get_current_user)
):
    """
    Delete all UFDR data (both from disk and database) for a specific UFDR file.
    
    This endpoint will:
    1. Validate permissions (user owns the case)
    2. Remove all data from shared database collections
    3. Delete all files and directories from disk storage
    4. Remove the UFDR file record
    """
    try:
        logger.info(f"Starting UFDR deletion for ufdr_file_id: {ufdr_file_id}, case_id: {case_id}")
        
        # Validate ObjectId formats
        try:
            ufdr_object_id = ObjectId(ufdr_file_id)
            case_object_id = ObjectId(case_id)
            user_object_id = ObjectId(user_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid ObjectId format")
        
        # Validate case exists and user has permission
        case = await collection_case.find_one({
            "_id": case_object_id,
            "user_id": user_object_id
        })
        if not case:
            raise HTTPException(
                status_code=404, 
                detail="Case not found or you don't have permission to access it"
            )
        
        # Validate UFDR file exists and belongs to the case
        ufdr_file = await ufdr_files_collection.find_one({
            "_id": ufdr_object_id,
            "caseId": case_object_id
        })
        if not ufdr_file:
            raise HTTPException(
                status_code=404, 
                detail="UFDR file not found or doesn't belong to this case"
            )
        
        case_name = case["name"]
        deletion_summary = {
            "ufdr_file_id": ufdr_file_id,
            "case_id": case_id,
            "ufdr_filename": ufdr_file["name"],
            "deleted_from_collections": [],
            "deleted_document_counts": {},
            "disk_cleanup": {"success": False, "deleted_paths": []},
            "errors": []
        }
        
        # 1. Delete data from all shared collections
        collections_to_clean = [
            ("ufdr_calls", ufdr_calls_collection),
            ("ufdr_chats", ufdr_chats_collection),
            ("ufdr_emails", ufdr_emails_collection),
            ("ufdr_locations", ufdr_locations_collection),
            ("ufdr_notes", ufdr_notes_collection),
            ("ufdr_searched_items", ufdr_searched_items_collection),
            ("ufdr_user_accounts", ufdr_user_accounts_collection),
            ("ufdr_audio", ufdr_audio_collection),
            ("ufdr_photos", ufdr_photos_collection),
            ("ufdr_videos", ufdr_videos_collection),
            ("ufdr_photo_detected_faces", ufdr_photo_detected_faces_collection),
            ("ufdr_photo_detected_objects", ufdr_photo_detected_objects_collection),
            ("ufdr_video_detected_faces", ufdr_video_detected_faces_collection),
            ("ufdr_video_detected_objects", ufdr_video_detected_objects_collection),
            ("ufdr_video_screenshots", ufdr_video_screenshots_collection),
            ("detector_matches", detector_matches_collection)
        ]
        
        # Delete from each collection
        for collection_name, collection in collections_to_clean:
            try:
                # Count documents before deletion
                count_before = await collection.count_documents({"ufdr_id": ufdr_object_id})
                
                if count_before > 0:
                    # Delete documents
                    delete_result = await collection.delete_many({"ufdr_id": ufdr_object_id})
                    
                    deletion_summary["deleted_from_collections"].append(collection_name)
                    deletion_summary["deleted_document_counts"][collection_name] = delete_result.deleted_count
                    
                    logger.info(f"Deleted {delete_result.deleted_count} documents from {collection_name}")
                else:
                    logger.info(f"No documents found in {collection_name} for ufdr_id: {ufdr_file_id}")
                    
            except Exception as e:
                error_msg = f"Error deleting from {collection_name}: {str(e)}"
                logger.error(error_msg)
                deletion_summary["errors"].append(error_msg)
        
        # 2. Delete files and directories from disk
        try:
            # Construct the UFDR directory path
            ufdr_directory = os.path.join(f"{UPLOAD_DIR}/{case_name}_{case_id}/ufdr/{ufdr_file_id}")
            
            if os.path.exists(ufdr_directory):
                # Use shutil.rmtree to recursively delete the entire directory
                shutil.rmtree(ufdr_directory)
                deletion_summary["disk_cleanup"]["success"] = True
                deletion_summary["disk_cleanup"]["deleted_paths"].append(ufdr_directory)
                logger.info(f"Successfully deleted directory: {ufdr_directory}")
            else:
                logger.warning(f"UFDR directory not found: {ufdr_directory}")
                deletion_summary["disk_cleanup"]["success"] = True  # Not an error if already missing
                
        except Exception as e:
            error_msg = f"Error deleting disk files: {str(e)}"
            logger.error(error_msg)
            deletion_summary["errors"].append(error_msg)
        
        # 3. Delete the UFDR file record from ufdr_files collection
        try:
            await ufdr_files_collection.delete_one({"_id": ufdr_object_id})
            logger.info(f"Deleted UFDR file record: {ufdr_file_id}")
        except Exception as e:
            error_msg = f"Error deleting UFDR file record: {str(e)}"
            logger.error(error_msg)
            deletion_summary["errors"].append(error_msg)
        
        # Calculate total deleted documents
        total_documents_deleted = sum(deletion_summary["deleted_document_counts"].values())
        
        # Determine response status
        has_errors = len(deletion_summary["errors"]) > 0
        status_code = 200 if not has_errors else 207  # 207 = Multi-Status (partial success)
        
        response_content = {
            "message": "UFDR data deletion completed" + (" with some errors" if has_errors else " successfully"),
            "deletion_summary": deletion_summary,
            "total_documents_deleted": total_documents_deleted,
            "collections_affected": len(deletion_summary["deleted_from_collections"]),
            "disk_cleanup_success": deletion_summary["disk_cleanup"]["success"],
            "has_errors": has_errors
        }
        
        logger.info(f"UFDR deletion completed for {ufdr_file_id}. Documents deleted: {total_documents_deleted}")
        
        return JSONResponse(content=response_content, status_code=status_code)
        
    except HTTPException:
        # Re-raise HTTP exceptions (they're already properly formatted)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during UFDR deletion: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during UFDR deletion: {str(e)}"
        )


@router.delete("/ufdr-data/case/{case_id}/all")
async def delete_all_ufdr_data_for_case(
    case_id: str,
    user_id: str = Depends(get_current_user)
):
    """
    Delete ALL UFDR data for a specific case (all UFDR files and their data).
    
    This endpoint will:
    1. Validate permissions (user owns the case)
    2. Find all UFDR files for the case
    3. Delete all data from shared database collections for the case
    4. Delete all UFDR directories from disk storage
    5. Remove all UFDR file records
    """
    try:
        logger.info(f"Starting bulk UFDR deletion for case_id: {case_id}")
        
        # Validate ObjectId formats
        try:
            case_object_id = ObjectId(case_id)
            user_object_id = ObjectId(user_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid ObjectId format")
        
        # Validate case exists and user has permission
        case = await collection_case.find_one({
            "_id": case_object_id,
            "user_id": user_object_id
        })
        if not case:
            raise HTTPException(
                status_code=404, 
                detail="Case not found or you don't have permission to access it"
            )
        
        case_name = case["name"]
        
        # Get all UFDR files for this case
        ufdr_files = await ufdr_files_collection.find({"caseId": case_object_id}).to_list(None)
        
        if not ufdr_files:
            return JSONResponse(
                content={
                    "message": "No UFDR files found for this case",
                    "case_id": case_id,
                    "total_ufdr_files": 0
                },
                status_code=200
            )
        
        deletion_summary = {
            "case_id": case_id,
            "case_name": case_name,
            "total_ufdr_files": len(ufdr_files),
            "ufdr_files_processed": [],
            "deleted_from_collections": [],
            "deleted_document_counts": {},
            "disk_cleanup": {"success": True, "deleted_paths": []},
            "errors": []
        }
        
        # 1. Delete data from all shared collections for this case
        collections_to_clean = [
            ("ufdr_calls", ufdr_calls_collection),
            ("ufdr_chats", ufdr_chats_collection),
            ("ufdr_emails", ufdr_emails_collection),
            ("ufdr_locations", ufdr_locations_collection),
            ("ufdr_notes", ufdr_notes_collection),
            ("ufdr_searched_items", ufdr_searched_items_collection),
            ("ufdr_user_accounts", ufdr_user_accounts_collection),
            ("ufdr_audio", ufdr_audio_collection),
            ("ufdr_photos", ufdr_photos_collection),
            ("ufdr_videos", ufdr_videos_collection),
            ("ufdr_photo_detected_faces", ufdr_photo_detected_faces_collection),
            ("ufdr_photo_detected_objects", ufdr_photo_detected_objects_collection),
            ("ufdr_video_detected_faces", ufdr_video_detected_faces_collection),
            ("ufdr_video_detected_objects", ufdr_video_detected_objects_collection),
            ("ufdr_video_screenshots", ufdr_video_screenshots_collection),
            ("detector_matches", detector_matches_collection)
        ]
        
        # Delete from each collection using case_id
        for collection_name, collection in collections_to_clean:
            try:
                # Count documents before deletion
                count_before = await collection.count_documents({"case_id": case_object_id})
                
                if count_before > 0:
                    # Delete documents
                    delete_result = await collection.delete_many({"case_id": case_object_id})
                    
                    deletion_summary["deleted_from_collections"].append(collection_name)
                    deletion_summary["deleted_document_counts"][collection_name] = delete_result.deleted_count
                    
                    logger.info(f"Deleted {delete_result.deleted_count} documents from {collection_name}")
                else:
                    logger.info(f"No documents found in {collection_name} for case_id: {case_id}")
                    
            except Exception as e:
                error_msg = f"Error deleting from {collection_name}: {str(e)}"
                logger.error(error_msg)
                deletion_summary["errors"].append(error_msg)
        
        # 2. Delete files and directories from disk for each UFDR file
        for ufdr_file in ufdr_files:
            ufdr_file_id = str(ufdr_file["_id"])
            ufdr_filename = ufdr_file["name"]
            
            try:
                # Construct the UFDR directory path
                ufdr_directory = os.path.join(f"{UPLOAD_DIR}/{case_name}_{case_id}/ufdr/{ufdr_file_id}")
                
                if os.path.exists(ufdr_directory):
                    # Use shutil.rmtree to recursively delete the entire directory
                    shutil.rmtree(ufdr_directory)
                    deletion_summary["disk_cleanup"]["deleted_paths"].append(ufdr_directory)
                    logger.info(f"Successfully deleted directory: {ufdr_directory}")
                else:
                    logger.warning(f"UFDR directory not found: {ufdr_directory}")
                
                deletion_summary["ufdr_files_processed"].append({
                    "ufdr_file_id": ufdr_file_id,
                    "filename": ufdr_filename,
                    "disk_deleted": True
                })
                
            except Exception as e:
                error_msg = f"Error deleting disk files for {ufdr_filename}: {str(e)}"
                logger.error(error_msg)
                deletion_summary["errors"].append(error_msg)
                deletion_summary["disk_cleanup"]["success"] = False
                
                deletion_summary["ufdr_files_processed"].append({
                    "ufdr_file_id": ufdr_file_id,
                    "filename": ufdr_filename,
                    "disk_deleted": False,
                    "error": str(e)
                })
        
        # 3. Delete all UFDR file records
        try:
            delete_result = await ufdr_files_collection.delete_many({"caseId": case_object_id})
            logger.info(f"Deleted {delete_result.deleted_count} UFDR file records")
        except Exception as e:
            error_msg = f"Error deleting UFDR file records: {str(e)}"
            logger.error(error_msg)
            deletion_summary["errors"].append(error_msg)
        
        # 4. Optionally delete the main UFDR directory if empty
        try:
            main_ufdr_dir = os.path.join(f"{UPLOAD_DIR}/{case_name}_{case_id}/ufdr")
            if os.path.exists(main_ufdr_dir) and not os.listdir(main_ufdr_dir):
                os.rmdir(main_ufdr_dir)
                deletion_summary["disk_cleanup"]["deleted_paths"].append(main_ufdr_dir)
                logger.info(f"Deleted empty UFDR directory: {main_ufdr_dir}")
        except Exception as e:
            logger.warning(f"Could not delete main UFDR directory: {str(e)}")
        
        # Calculate total deleted documents
        total_documents_deleted = sum(deletion_summary["deleted_document_counts"].values())
        
        # Determine response status
        has_errors = len(deletion_summary["errors"]) > 0
        status_code = 200 if not has_errors else 207  # 207 = Multi-Status (partial success)
        
        response_content = {
            "message": f"Bulk UFDR deletion completed" + (" with some errors" if has_errors else " successfully"),
            "deletion_summary": deletion_summary,
            "total_documents_deleted": total_documents_deleted,
            "collections_affected": len(deletion_summary["deleted_from_collections"]),
            "disk_cleanup_success": deletion_summary["disk_cleanup"]["success"],
            "has_errors": has_errors
        }
        
        logger.info(f"Bulk UFDR deletion completed for case {case_id}. Files processed: {len(ufdr_files)}, Documents deleted: {total_documents_deleted}")
        
        return JSONResponse(content=response_content, status_code=status_code)
        
    except HTTPException:
        # Re-raise HTTP exceptions (they're already properly formatted)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during bulk UFDR deletion: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error during bulk UFDR deletion: {str(e)}"
        )
