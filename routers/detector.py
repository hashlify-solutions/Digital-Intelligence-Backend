import os
import shutil
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Form, File, UploadFile, Query
from fastapi.responses import JSONResponse
from bson import ObjectId
from config.settings import settings
from utils.auth import get_current_user
from config.db import (
    detectors_collection,
    detector_matches_collection,
    detector_settings_collection,
    ufdr_photo_detected_faces_collection,
    ufdr_video_detected_faces_collection,
    ufdr_photo_detected_objects_collection,
    ufdr_video_detected_objects_collection,
    collection_case
)
from schemas.detector import (
    DetectorResponse,
    DetectorUpdate,
    DetectorMatch,
    DetectorMatchSummary,
    DetectorSettings,
    DetectorSettingsUpdate,
    AnalyzeDetectorsRequest,
    DetectorAnalysisResponse,
)
from tasks.celery_tasks import (
    process_detector_embedding_task,
    analyze_detector_matches_task
)
from model_registry import ModelRegistry


UPLOAD_DIR = settings.upload_dir
logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/case/{case_id}/detectors", response_model=DetectorResponse)
async def upload_detector(
    case_id: str,
    name: str = Form(...),
    type: str = Form(...),
    description: Optional[str] = Form(None),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload a detector image for a case"""
    try:
        # Validate case_id format
        if not ObjectId.is_valid(case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format")

        # Validate detector type
        if type not in ["person", "object"]:
            raise HTTPException(
                status_code=400, detail="Type must be 'person' or 'object'"
            )

        # Validate file type
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        
        # Find the case with case_id
        case = await collection_case.find_one({"_id": ObjectId(case_id)})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Create detector directory
        detector_dir = os.path.join(UPLOAD_DIR, f"{case['name']}_{case_id}/detectors")
        os.makedirs(detector_dir, exist_ok=True)

        # Create detector document
        current_time = datetime.now(timezone.utc)
        detector_doc = {
            "case_id": ObjectId(case_id),
            "name": name,
            "type": type,
            "description": description,
            "has_embedding": False,
            "created_at": current_time,
            "updated_at": current_time,
            "user_id": ObjectId(user_id),
        }
        result = await detectors_collection.insert_one(detector_doc)
        detector_id = str(result.inserted_id)
        logger.info(f"Created detector {detector_id} for case {case_id}")

        # Save uploaded file
        file_extension = os.path.splitext(file.filename)[1]
        safe_name = "".join(
            c for c in name if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        filename = f"{safe_name}_{detector_id}{file_extension}"
        file_path = os.path.join(detector_dir, filename)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved detector image to: {file_path}")

        detector_doc["image_path"] = file_path
        await detectors_collection.update_one({"_id": ObjectId(detector_id)}, {"$set": {"image_path": file_path}})

        # Validate image based on detector type
        if type == "person":
            face_client = ModelRegistry.get_model("face_embeddings")
            is_valid, message = face_client.validate_face_image(file_path)
        else:
            object_client = ModelRegistry.get_model("object_embeddings")
            is_valid, message = object_client.validate_object_image(file_path)

        if not is_valid:
            # Clean up file if validation failed
            os.remove(file_path)
            await detectors_collection.delete_one({"_id": ObjectId(detector_id)})
            logger.error(f"Invalid {type} image: {message}")
            logger.error(f"Deleted detector {detector_id} for case {case_id}")
            raise HTTPException(
                status_code=400, detail=f"Invalid {type} image: {message}"
            )

        # Trigger embedding generation
        embedding_task = process_detector_embedding_task.delay(detector_id, case_id)
        logger.info(
            f"Started embedding generation task {embedding_task.id} for detector {detector_id}"
        )

        # Return detector response
        detector_doc["_id"] = result.inserted_id
        
        # Remove _id from detector_doc to avoid ObjectId conflict and set it as string id
        detector_response_data = {k: v for k, v in detector_doc.items() if k != "_id"}
        detector_response_data.update({
            "_id": str(detector_doc["_id"]),  # Convert ObjectId to string for _id field
            "case_id": str(detector_doc["case_id"]),
            "user_id": str(detector_doc["user_id"]),
        })
        
        return DetectorResponse(**detector_response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading detector: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error uploading detector: {str(e)}"
        )


@router.get("/case/{case_id}/detectors", response_model=List[DetectorResponse])
async def list_detectors(
    case_id: str,
    type: Optional[str] = Query(None, description="Filter by detector type"),
    user_id: str = Depends(get_current_user),
):
    """List all detectors for a case"""
    try:
        # Validate case_id format
        if not ObjectId.is_valid(case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format")

        # Build query
        query = {"case_id": ObjectId(case_id)}
        if type and type in ["person", "object"]:
            query["type"] = type

        # Get detectors
        detectors = (
            await detectors_collection.find(query)
            .sort("created_at", -1)
            .to_list(length=None)
        )

        # Convert to response format
        response_detectors = []
        for detector in detectors:
            # Remove _id from detector to avoid ObjectId conflict and set it as string _id
            detector_data = {k: v for k, v in detector.items() if k != "_id"}
            detector_data.update({
                "_id": str(detector["_id"]),
                "case_id": str(detector["case_id"]),
                "user_id": str(detector["user_id"]),
            })
            response_detectors.append(DetectorResponse(**detector_data))

        return response_detectors

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing detectors: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error listing detectors: {str(e)}"
        )


@router.get("/case/{case_id}/detectors/{detector_id}", response_model=DetectorResponse)
async def get_detector(
    case_id: str,
    detector_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get a specific detector"""
    try:
        # Validate IDs
        if not ObjectId.is_valid(case_id) or not ObjectId.is_valid(detector_id):
            raise HTTPException(status_code=400, detail="Invalid ID format")

        # Get detector
        detector = await detectors_collection.find_one(
            {"_id": ObjectId(detector_id), "case_id": ObjectId(case_id)}
        )

        if not detector:
            raise HTTPException(status_code=404, detail="Detector not found")

        # Remove _id from detector to avoid ObjectId conflict and set it as string _id
        detector_data = {k: v for k, v in detector.items() if k != "_id"}
        detector_data.update({
            "_id": str(detector["_id"]),
            "case_id": str(detector["case_id"]),
            "user_id": str(detector["user_id"]),
        })
        
        return DetectorResponse(**detector_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detector: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting detector: {str(e)}")


@router.put("/case/{case_id}/detectors/{detector_id}", response_model=DetectorResponse)
async def update_detector(
    case_id: str,
    detector_id: str,
    detector_update: DetectorUpdate,
    user_id: str = Depends(get_current_user),
):
    """Update a detector"""
    try:
        # Validate IDs
        if not ObjectId.is_valid(case_id) or not ObjectId.is_valid(detector_id):
            raise HTTPException(status_code=400, detail="Invalid ID format")

        # Build update document
        update_doc = {"updated_at": datetime.utcnow()}
        if detector_update.name is not None:
            update_doc["name"] = detector_update.name
        if detector_update.description is not None:
            update_doc["description"] = detector_update.description

        # Update detector
        result = await detectors_collection.update_one(
            {"_id": ObjectId(detector_id), "case_id": ObjectId(case_id)},
            {"$set": update_doc},
        )

        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Detector not found")

        # Get updated detector
        detector = await detectors_collection.find_one({"_id": ObjectId(detector_id)})

        # Remove _id from detector to avoid ObjectId conflict and set it as string _id
        detector_data = {k: v for k, v in detector.items() if k != "_id"}
        detector_data.update({
            "_id": str(detector["_id"]),
            "case_id": str(detector["case_id"]),
            "user_id": str(detector["user_id"]),
        })
        
        return DetectorResponse(**detector_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating detector: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating detector: {str(e)}"
        )


@router.delete("/case/{case_id}/detectors/{detector_id}")
async def delete_detector(
    case_id: str,
    detector_id: str,
    user_id: str = Depends(get_current_user),
):
    """Delete a detector"""
    try:
        # Validate IDs
        if not ObjectId.is_valid(case_id) or not ObjectId.is_valid(detector_id):
            raise HTTPException(status_code=400, detail="Invalid ID format")

        # Get detector to get image path
        detector = await detectors_collection.find_one(
            {"_id": ObjectId(detector_id), "case_id": ObjectId(case_id)}
        )

        if not detector:
            raise HTTPException(status_code=404, detail="Detector not found")

        # Delete detector from database
        await detectors_collection.delete_one({"_id": ObjectId(detector_id)})

        # Delete associated matches
        await detector_matches_collection.delete_many(
            {"detector_id": ObjectId(detector_id)}
        )

        # Delete image file
        try:
            if os.path.exists(detector["image_path"]):
                os.remove(detector["image_path"])
                logger.info(f"Deleted detector image: {detector['image_path']}")
        except Exception as e:
            logger.warning(f"Could not delete detector image: {str(e)}")

        return JSONResponse(
            content={"message": "Detector deleted successfully"}, status_code=200
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting detector: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting detector: {str(e)}"
        )


@router.post(
    "/case/{case_id}/detectors/analyze", response_model=DetectorAnalysisResponse
)
async def analyze_detectors(
    case_id: str,
    request: AnalyzeDetectorsRequest,
    user_id: str = Depends(get_current_user),
):
    """Trigger analysis of all detectors against detected items"""
    try:
        # Validate case_id format
        if not ObjectId.is_valid(case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format")

        # Count detectors to be processed
        query = {"case_id": ObjectId(case_id), "has_embedding": True}
        if request.detector_type:
            query["type"] = request.detector_type

        detectors_count = await detectors_collection.count_documents(query)

        if detectors_count == 0:
            raise HTTPException(
                status_code=400,
                detail="No detectors with embeddings found for analysis",
            )

        # Estimate detected items count (rough estimate)
        detected_items_count = 0
        if not request.detector_type or request.detector_type == "person":
            detected_items_count += (
                await ufdr_photo_detected_faces_collection.count_documents(
                    {"case_id": ObjectId(case_id)}
                )
            )
            detected_items_count += (
                await ufdr_video_detected_faces_collection.count_documents(
                    {"case_id": ObjectId(case_id)}
                )
            )

        if not request.detector_type or request.detector_type == "object":
            detected_items_count += (
                await ufdr_photo_detected_objects_collection.count_documents(
                    {"case_id": ObjectId(case_id)}
                )
            )
            detected_items_count += (
                await ufdr_video_detected_objects_collection.count_documents(
                    {"case_id": ObjectId(case_id)}
                )
            )

        # Start analysis task
        analysis_task = analyze_detector_matches_task.delay(
            case_id, request.detector_type
        )

        logger.info(
            f"Started detector analysis task {analysis_task.id} for case {case_id}"
        )

        return DetectorAnalysisResponse(
            case_id=case_id,
            analysis_started=True,
            task_id=analysis_task.id,
            message="Detector analysis started successfully",
            detectors_processed=detectors_count,
            detected_items_to_analyze=detected_items_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting detector analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error starting detector analysis: {str(e)}"
        )


@router.get("/case/{case_id}/detector-matches", response_model=DetectorMatchSummary)
async def get_detector_matches(
    case_id: str,
    detector_id: Optional[str] = Query(None, description="Filter by detector ID"),
    confidence_level: Optional[str] = Query(
        None, description="Filter by confidence level"
    ),
    limit: int = Query(
        100, ge=1, le=1000, description="Maximum number of matches to return"
    ),
    user_id: str = Depends(get_current_user),
):
    """Get detector matches for a case"""
    try:
        # Validate case_id format
        if not ObjectId.is_valid(case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format")

        # Build query
        query = {"case_id": ObjectId(case_id)}
        if detector_id:
            if not ObjectId.is_valid(detector_id):
                raise HTTPException(
                    status_code=400, detail="Invalid detector ID format"
                )
            query["detector_id"] = ObjectId(detector_id)

        if confidence_level and confidence_level in ["high", "medium", "low"]:
            query["confidence_level"] = confidence_level

        # Get matches
        matches = (
            await detector_matches_collection.find(query)
            .sort("similarity_score", -1)
            .limit(limit)
            .to_list(length=None)
        )

        # Get match statistics
        total_matches = await detector_matches_collection.count_documents(
            {"case_id": ObjectId(case_id)}
        )
        high_confidence = await detector_matches_collection.count_documents(
            {"case_id": ObjectId(case_id), "confidence_level": "high"}
        )
        medium_confidence = await detector_matches_collection.count_documents(
            {"case_id": ObjectId(case_id), "confidence_level": "medium"}
        )
        low_confidence = await detector_matches_collection.count_documents(
            {"case_id": ObjectId(case_id), "confidence_level": "low"}
        )

        # Get detector statistics
        detector_stats_pipeline = [
            {"$match": {"case_id": ObjectId(case_id)}},
            {"$group": {"_id": "$detector_type", "count": {"$sum": 1}}},
        ]
        detector_stats_raw = await detector_matches_collection.aggregate(
            detector_stats_pipeline
        ).to_list(length=None)
        detector_stats = {stat["_id"]: stat["count"] for stat in detector_stats_raw}

        # Convert matches to response format
        response_matches = []
        for match in matches:
            # Remove _id from match to avoid ObjectId conflict and set it as string _id
            match_data = {k: v for k, v in match.items() if k != "_id"}
            match_data.update({
                "_id": str(match["_id"]),
                "case_id": str(match["case_id"]),
                "detector_id": str(match["detector_id"]),
                "detected_item_id": str(match["detected_item_id"]),
            })
            response_matches.append(DetectorMatch(**match_data))

        return DetectorMatchSummary(
            case_id=case_id,
            total_matches=total_matches,
            high_confidence_matches=high_confidence,
            medium_confidence_matches=medium_confidence,
            low_confidence_matches=low_confidence,
            detector_stats=detector_stats,
            matches=response_matches,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detector matches: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting detector matches: {str(e)}"
        )


@router.get("/case/{case_id}/detector-settings", response_model=DetectorSettings)
async def get_detector_settings(
    case_id: str,
    user_id: str = Depends(get_current_user),
):
    """Get detector settings for a case"""
    try:
        # Validate case_id format
        if not ObjectId.is_valid(case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format")

        # Get settings
        settings = await detector_settings_collection.find_one(
            {"case_id": ObjectId(case_id)}
        )

        if not settings:
            # Return default settings
            return DetectorSettings(case_id=case_id, user_id=user_id)

        return DetectorSettings(
            **{
                **settings,
                "case_id": str(settings["case_id"]),
                "user_id": str(settings.get("user_id", user_id)),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting detector settings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error getting detector settings: {str(e)}"
        )


@router.put("/case/{case_id}/detector-settings", response_model=DetectorSettings)
async def update_detector_settings(
    case_id: str,
    settings_update: DetectorSettingsUpdate,
    user_id: str = Depends(get_current_user),
):
    """Update detector settings for a case"""
    try:
        # Validate case_id format
        if not ObjectId.is_valid(case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format")

        # Build update document
        update_doc = {"updated_at": datetime.utcnow()}
        if settings_update.face_thresholds is not None:
            update_doc["face_thresholds"] = settings_update.face_thresholds
        if settings_update.object_thresholds is not None:
            update_doc["object_thresholds"] = settings_update.object_thresholds

        # Upsert settings
        result = await detector_settings_collection.update_one(
            {"case_id": ObjectId(case_id)},
            {
                "$set": update_doc,
                "$setOnInsert": {
                    "case_id": ObjectId(case_id),
                    "user_id": ObjectId(user_id),
                    "created_at": datetime.utcnow(),
                },
            },
            upsert=True,
        )

        # Get updated settings
        settings = await detector_settings_collection.find_one(
            {"case_id": ObjectId(case_id)}
        )

        return DetectorSettings(
            **{
                **settings,
                "case_id": str(settings["case_id"]),
                "user_id": str(settings["user_id"]),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating detector settings: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating detector settings: {str(e)}"
        )


@router.post("/case/{case_id}/detector-matches/reanalyze")
async def reanalyze_detector_matches(
    case_id: str,
    request: AnalyzeDetectorsRequest,
    user_id: str = Depends(get_current_user),
):
    """Re-run detector analysis with updated settings"""
    try:
        # Validate case_id format
        if not ObjectId.is_valid(case_id):
            raise HTTPException(status_code=400, detail="Invalid case ID format")

        # Clear existing matches if requested
        if request.recompute_embeddings:
            await detector_matches_collection.delete_many(
                {"case_id": ObjectId(case_id)}
            )
            logger.info(f"Cleared existing matches for case {case_id}")

        # Start analysis task
        analysis_task = analyze_detector_matches_task.delay(
            case_id, request.detector_type
        )

        logger.info(
            f"Started detector re-analysis task {analysis_task.id} for case {case_id}"
        )

        return JSONResponse(
            content={
                "message": "Detector re-analysis started successfully",
                "case_id": case_id,
                "task_id": analysis_task.id,
                "cleared_existing_matches": request.recompute_embeddings,
            },
            status_code=200,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting detector re-analysis: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error starting detector re-analysis: {str(e)}"
        )
