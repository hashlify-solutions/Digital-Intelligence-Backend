# Standard 3rd party dependencies imports 
import os
from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
import logging
from pathlib import Path
import json
# Internal local modules imports
from config.db import db
from utils.auth import get_current_user
from schemas.user import UserOut
from utils.case_mapping import get_actual_case_directory


logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/ufdr-metadata/{case_id}")
async def get_ufdr_metadata(
    case_id: str,
    media_type: Optional[str] = Query(None, description="Filter by media type: image, audio, video, document, archive, other"),
    file_extension: Optional[str] = Query(None, description="Filter by file extension"),
    deleted_state: Optional[str] = Query(None, description="Filter by deleted state: Intact, Deleted, etc."),
    source: Optional[str] = Query(None, description="Filter by source: UFDR-Media (XML), UFDR-Directory-Scan (directory scan)"),
    limit: int = Query(100, description="Maximum number of results to return"),
    offset: int = Query(0, description="Number of results to skip"),
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get comprehensive UFDR metadata for a case including all media files (images, audios, videos, documents, etc.)
    """
    try:
        # Validate case_id
        try:
            case_object_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        # Get database collections
        cases_collection = db["cases"]
        
        # Verify case exists and user has access
        case = await cases_collection.find_one({"_id": case_object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use case-specific collection for UFDR metadata
        case_collection = db[f"{case['name']}_{case_id}"]
        
        # Build filter query
        filter_query = {
            "case_id": case_object_id
        }
        
        # Filter by source (XML metadata, directory scan, or embedded content)
        if source and source.lower() != "null":
            if source == "UFDR-Media":
                filter_query["source"] = "UFDR-Media"
            elif source == "UFDR-Directory-Scan":
                filter_query["source"] = "UFDR-Directory-Scan"
            elif source == "UFDR-XML-Embedded":
                filter_query["source"] = "UFDR-XML-Embedded"
            else:
                # If source is specified but not valid, include all sources
                filter_query["source"] = {"$in": ["UFDR-Media", "UFDR-Directory-Scan", "UFDR-XML-Embedded"]}
        else:
            # Default: include all sources
            filter_query["source"] = {"$in": ["UFDR-Media", "UFDR-Directory-Scan", "UFDR-XML-Embedded"]}
        
        if media_type and media_type.lower() != "null":
            filter_query["media_type"] = media_type
        if file_extension and file_extension.lower() != "null":
            filter_query["file_extension"] = file_extension.lower()
        if deleted_state and deleted_state.lower() != "null":
            filter_query["deleted_state"] = deleted_state
        
        # Get total count
        total_count = await case_collection.count_documents(filter_query)
        
        # Adjust offset if it exceeds total count
        effective_offset = min(offset, total_count)
        
        # Get media files with pagination
        cursor = case_collection.find(filter_query).skip(effective_offset).limit(limit)
        media_files = await cursor.to_list(length=limit)
        
        # Convert ObjectId to string for JSON serialization and add media URLs
        for file in media_files:
            file["_id"] = str(file["_id"])
            file["case_id"] = str(file["case_id"])
            
            # Generate media URL only if file exists
            local_path = file.get("local_path", "")
            case_name = case.get("name", "")
            media_url = None
            
            # Get actual directory name using utility function
            actual_case_dir = get_actual_case_directory(case_name)
            
            if local_path and actual_case_dir:
                # Convert local path to URL path
                relative_path = local_path.replace("\\", "/")
                if relative_path.startswith("./"):
                    relative_path = relative_path[2:]
                # Remove leading 'files/' if present to avoid duplication
                if relative_path.startswith("files/"):
                    relative_path = relative_path[6:]  # Remove 'files/'
                
                # Construct full file path to check if file exists
                full_file_path = Path(".") / actual_case_dir / relative_path
                
                # Only generate URL if file actually exists
                if full_file_path.exists() and full_file_path.is_file():
                    media_url = f"/api/media/files/{actual_case_dir}/files/{relative_path}"
            
            file["media_url"] = media_url
        
        return {
            "case_id": case_id,
            "total_count": total_count,
            "returned_count": len(media_files),
            "offset": effective_offset,
            "limit": limit,
            "filters_applied": {
                "media_type": media_type,
                "file_extension": file_extension,
                "deleted_state": deleted_state,
                "source": source
            },
            "media_files": media_files,
            "pagination_info": {
                "requested_offset": offset,
                "effective_offset": effective_offset,
                "has_more": effective_offset + len(media_files) < total_count,
                "total_pages": max(1, (total_count + limit - 1) // limit) if limit > 0 else 1
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting UFDR metadata: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ufdr-metadata/{case_id}/summary")
async def get_ufdr_metadata_summary(
    case_id: str,
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get summary statistics of UFDR metadata for a case
    """
    try:
        # Validate case_id
        try:
            case_object_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        # Get database collections
        cases_collection = db["cases"]
        
        # Verify case exists and user has access
        case = await cases_collection.find_one({"_id": case_object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use case-specific collection for UFDR metadata
        case_collection = db[f"{case['name']}_{case_id}"]
        
        # Build aggregation pipeline for summary statistics
        pipeline = [
            {
                "$match": {
                    "case_id": case_object_id,
                    "source": {"$in": ["UFDR-Media", "UFDR-Directory-Scan", "UFDR-XML-Embedded"]}
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_files": {"$sum": 1},
                    "total_size_bytes": {"$sum": {"$toInt": {"$ifNull": ["$file_size", "0"]}}},
                    "media_types": {"$push": "$media_type"},
                    "file_extensions": {"$push": "$file_extension"},
                    "deleted_states": {"$push": "$deleted_state"},
                    "file_systems": {"$push": "$file_system"},
                    "has_exif_data": {"$sum": {"$cond": [{"$gt": [{"$size": {"$ifNull": ["$exif_data", []]}}, 0]}, 1, 0]}},
                    "embedded_files": {"$sum": {"$cond": [{"$eq": ["$embedded", "true"]}, 1, 0]}},
                    "related_files": {"$sum": {"$cond": [{"$eq": ["$is_related", "True"]}, 1, 0]}}
                }
            }
        ]
        
        result = await case_collection.aggregate(pipeline).to_list(1)
        
        if not result:
            return {
                "case_id": case_id,
                "summary": {
                    "total_files": 0,
                    "total_size_bytes": 0,
                    "media_type_counts": {},
                    "file_extension_counts": {},
                    "deleted_state_counts": {},
                    "file_system_counts": {},
                    "files_with_exif": 0,
                    "embedded_files": 0,
                    "related_files": 0
                }
            }
        
        summary_data = result[0]
        
        # Count occurrences of each category
        from collections import Counter
        
        media_type_counts = dict(Counter(summary_data.get("media_types", [])))
        file_extension_counts = dict(Counter(summary_data.get("file_extensions", [])))
        deleted_state_counts = dict(Counter(summary_data.get("deleted_states", [])))
        file_system_counts = dict(Counter(summary_data.get("file_systems", [])))
        
        return {
            "case_id": case_id,
            "summary": {
                "total_files": summary_data.get("total_files", 0),
                "total_size_bytes": summary_data.get("total_size_bytes", 0),
                "total_size_mb": round(summary_data.get("total_size_bytes", 0) / (1024 * 1024), 2),
                "media_type_counts": media_type_counts,
                "file_extension_counts": file_extension_counts,
                "deleted_state_counts": deleted_state_counts,
                "file_system_counts": file_system_counts,
                "files_with_exif": summary_data.get("has_exif_data", 0),
                "embedded_files": summary_data.get("embedded_files", 0),
                "related_files": summary_data.get("related_files", 0)
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting UFDR metadata summary: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ufdr-metadata/{case_id}/sources")
async def get_ufdr_metadata_by_source(
    case_id: str,
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get UFDR metadata statistics broken down by source (XML vs Directory Scan)
    """
    try:
        # Validate case_id
        try:
            case_object_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        # Get database collections
        cases_collection = db["cases"]
        
        # Verify case exists and user has access
        case = await cases_collection.find_one({"_id": case_object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use case-specific collection for UFDR metadata
        case_collection = db[f"{case['name']}_{case_id}"]
        
        # Build aggregation pipeline for source breakdown
        pipeline = [
            {
                "$match": {
                    "case_id": case_object_id,
                    "source": {"$in": ["UFDR-Media", "UFDR-Directory-Scan", "UFDR-XML-Embedded"]}
                }
            },
            {
                "$group": {
                    "_id": "$source",
                    "total_files": {"$sum": 1},
                    "total_size_bytes": {"$sum": {"$toInt": {"$ifNull": ["$file_size", "0"]}}},
                    "media_types": {"$push": "$media_type"},
                    "file_extensions": {"$push": "$file_extension"},
                    "has_exif_data": {"$sum": {"$cond": [{"$gt": [{"$size": {"$ifNull": ["$exif_data", []]}}, 0]}, 1, 0]}},
                    "embedded_files": {"$sum": {"$cond": [{"$eq": ["$embedded", "true"]}, 1, 0]}},
                    "related_files": {"$sum": {"$cond": [{"$eq": ["$is_related", "True"]}, 1, 0]}}
                }
            }
        ]
        
        results = await case_collection.aggregate(pipeline).to_list(None)
        
        # Process results
        source_breakdown = {}
        total_files = 0
        total_size_bytes = 0
        
        for result in results:
            source_name = result["_id"]
            source_data = {
                "total_files": result["total_files"],
                "total_size_bytes": result["total_size_bytes"],
                "total_size_mb": round(result["total_size_bytes"] / (1024 * 1024), 2),
                "files_with_exif": result["has_exif_data"],
                "embedded_files": result["embedded_files"],
                "related_files": result["related_files"]
            }
            
            # Count media types and extensions
            from collections import Counter
            source_data["media_type_counts"] = dict(Counter(result["media_types"]))
            source_data["file_extension_counts"] = dict(Counter(result["file_extensions"]))
            
            source_breakdown[source_name] = source_data
            total_files += result["total_files"]
            total_size_bytes += result["total_size_bytes"]
        
        return {
            "case_id": case_id,
            "total_files": total_files,
            "total_size_bytes": total_size_bytes,
            "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
            "source_breakdown": source_breakdown,
            "sources": {
                "UFDR-Media": "Files extracted from UFDR XML metadata",
                "UFDR-Directory-Scan": "Files found by scanning extracted directory structure",
                "UFDR-XML-Embedded": "Files extracted from embedded base64 content in XML"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting UFDR metadata by source: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ufdr-metadata/{case_id}/images")
async def get_ufdr_images(
    case_id: str,
    include_exif: bool = Query(True, description="Include EXIF data in response"),
    limit: int = Query(50, description="Maximum number of images to return"),
    offset: int = Query(0, description="Number of images to skip"),
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get all image files from UFDR with optional EXIF data
    """
    try:
        # Validate case_id
        try:
            case_object_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        # Get database collections
        cases_collection = db["cases"]
        
        # Verify case exists and user has access
        case = await cases_collection.find_one({"_id": case_object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use case-specific collection for UFDR metadata
        case_collection = db[f"{case['name']}_{case_id}"]
        
        # Build filter for images
        filter_query = {
            "case_id": case_object_id,
            "source": "UFDR-Media",
            "media_type": "image"
        }
        
        # Get total count
        total_count = await case_collection.count_documents(filter_query)
        
        # Adjust offset if it exceeds total count
        effective_offset = min(offset, total_count)
        
        # Get images with pagination
        cursor = case_collection.find(filter_query).skip(effective_offset).limit(limit)
        images = await cursor.to_list(length=limit)
        
        # Process images
        processed_images = []
        for image in images:
            # Generate media URL
            local_path = image.get("local_path", "")
            case_name = case.get("name", "")
            media_url = None
            
            if local_path and case_name:
                # Convert local path to URL path
                relative_path = local_path.replace("\\", "/")
                if relative_path.startswith("./"):
                    relative_path = relative_path[2:]
                
                # Get actual directory name using utility function
                actual_case_dir = get_actual_case_directory(case_name)
                
                # Remove leading 'files/' if present to avoid duplication
                if relative_path.startswith("files/"):
                    relative_path = relative_path[6:]  # Remove 'files/'
                
                # Generate media URL only if file exists
                full_file_path = Path(".") / actual_case_dir / relative_path
                
                # Only generate URL if file actually exists
                if full_file_path.exists() and full_file_path.is_file():
                    media_url = f"/api/media/files/{actual_case_dir}/files/{relative_path}"
                else:
                    media_url = None
            
            image_data = {
                "_id": str(image["_id"]),
                "file_id": image.get("file_id"),
                "file_name": image.get("file_name"),
                "file_path": image.get("file_path"),
                "local_path": image.get("local_path"),
                "media_url": media_url,
                "file_size": image.get("file_size"),
                "file_size_bytes": image.get("file_size_bytes"),
                "file_extension": image.get("file_extension"),
                "sha256": image.get("sha256"),
                "md5": image.get("md5"),
                "creation_time": image.get("creation_time"),
                "modify_time": image.get("modify_time"),
                "access_time": image.get("access_time"),
                "deleted_state": image.get("deleted_state"),
                "embedded": image.get("embedded"),
                "is_related": image.get("is_related"),
                "file_system": image.get("file_system"),
                "source_file": image.get("source_file"),
                "ingestion_timestamp": image.get("ingestion_timestamp")
            }
            
            if include_exif and image.get("exif_data"):
                image_data["exif_data"] = image.get("exif_data")
            
            processed_images.append(image_data)
        
        return {
            "case_id": case_id,
            "total_images": total_count,
            "returned_images": len(processed_images),
            "offset": effective_offset,
            "limit": limit,
            "include_exif": include_exif,
            "images": processed_images,
            "pagination_info": {
                "requested_offset": offset,
                "effective_offset": effective_offset,
                "has_more": effective_offset + len(processed_images) < total_count,
                "total_pages": max(1, (total_count + limit - 1) // limit) if limit > 0 else 1
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting UFDR images: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ufdr-metadata/{case_id}/audio")
async def get_ufdr_audio_files(
    case_id: str,
    limit: int = Query(50, description="Maximum number of audio files to return"),
    offset: int = Query(0, description="Number of audio files to skip"),
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get all audio files from UFDR
    """
    try:
        # Validate case_id
        try:
            case_object_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        # Get database collections
        cases_collection = db["cases"]
        
        # Verify case exists and user has access
        case = await cases_collection.find_one({"_id": case_object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use case-specific collection for UFDR metadata
        case_collection = db[f"{case['name']}_{case_id}"]
        
        # Build filter for audio files
        filter_query = {
            "case_id": case_object_id,
            "source": "UFDR-Media",
            "media_type": "audio"
        }
        
        # Get total count
        total_count = await case_collection.count_documents(filter_query)
        
        # Adjust offset if it exceeds total count
        effective_offset = min(offset, total_count)
        
        # Get audio files with pagination
        cursor = case_collection.find(filter_query).skip(effective_offset).limit(limit)
        audio_files = await cursor.to_list(length=limit)
        
        # Process audio files
        processed_audio = []
        for audio in audio_files:
            # Generate media URL
            local_path = audio.get("local_path", "")
            case_name = case.get("name", "")
            media_url = None
            
            if local_path and case_name:
                # Convert local path to URL path
                relative_path = local_path.replace("\\", "/")
                if relative_path.startswith("./"):
                    relative_path = relative_path[2:]
                
                # Get actual directory name using utility function
                actual_case_dir = get_actual_case_directory(case_name)
                
                # Remove leading 'files/' if present to avoid duplication
                if relative_path.startswith("files/"):
                    relative_path = relative_path[6:]  # Remove 'files/'
                
                # Generate media URL only if file exists
                full_file_path = Path(".") / actual_case_dir / relative_path
                
                # Only generate URL if file actually exists
                if full_file_path.exists() and full_file_path.is_file():
                    media_url = f"/api/media/files/{actual_case_dir}/files/{relative_path}"
                else:
                    media_url = None
            
            audio_data = {
                "_id": str(audio["_id"]),
                "file_id": audio.get("file_id"),
                "file_name": audio.get("file_name"),
                "file_path": audio.get("file_path"),
                "local_path": audio.get("local_path"),
                "media_url": media_url,
                "file_size": audio.get("file_size"),
                "file_size_bytes": audio.get("file_size_bytes"),
                "file_extension": audio.get("file_extension"),
                "sha256": audio.get("sha256"),
                "md5": audio.get("md5"),
                "creation_time": audio.get("creation_time"),
                "modify_time": audio.get("modify_time"),
                "access_time": audio.get("access_time"),
                "deleted_state": audio.get("deleted_state"),
                "embedded": audio.get("embedded"),
                "is_related": audio.get("is_related"),
                "file_system": audio.get("file_system"),
                "source_file": audio.get("source_file"),
                "ingestion_timestamp": audio.get("ingestion_timestamp"),
                "additional_metadata": audio.get("additional_metadata", {})
            }
            processed_audio.append(audio_data)
        
        return {
            "case_id": case_id,
            "total_audio_files": total_count,
            "returned_audio_files": len(processed_audio),
            "offset": effective_offset,
            "limit": limit,
            "audio_files": processed_audio,
            "pagination_info": {
                "requested_offset": offset,
                "effective_offset": effective_offset,
                "has_more": effective_offset + len(processed_audio) < total_count,
                "total_pages": max(1, (total_count + limit - 1) // limit) if limit > 0 else 1
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting UFDR audio files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ufdr-metadata/{case_id}/video")
async def get_ufdr_video_files(
    case_id: str,
    limit: int = Query(50, description="Maximum number of video files to return"),
    offset: int = Query(0, description="Number of video files to skip"),
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get all video files from UFDR
    """
    try:
        # Validate case_id
        try:
            case_object_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        # Get database collections
        messages_collection = db["messages"]
        cases_collection = db["cases"]
        
        # Verify case exists and user has access
        case = await cases_collection.find_one({"_id": case_object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")
        
        # Use case-specific collection for UFDR metadata
        case_collection = db[f"{case['name']}_{case_id}"]

        # Build filter for video files
        filter_query = {
            "case_id": case_object_id,
            "source": "UFDR-Media",
            "media_type": "video"
        }

        # Get total count
        total_count = await case_collection.count_documents(filter_query)
        
        # Adjust offset if it exceeds total count
        effective_offset = min(offset, total_count)
        
        # Get video files with pagination
        cursor = case_collection.find(filter_query).skip(effective_offset).limit(limit)
        video_files = await cursor.to_list(length=limit)
        
        # Process video files
        processed_videos = []
        for video in video_files:
            # Generate media URL
            local_path = video.get("local_path", "")
            case_name = case.get("name", "")
            media_url = None
            
            if local_path and case_name:
                # Convert local path to URL path
                relative_path = local_path.replace("\\", "/")
                if relative_path.startswith("./"):
                    relative_path = relative_path[2:]
                
                # Get actual directory name using utility function
                actual_case_dir = get_actual_case_directory(case_name)
                
                # Remove leading 'files/' if present to avoid duplication
                if relative_path.startswith("files/"):
                    relative_path = relative_path[6:]  # Remove 'files/'
                
                # Generate media URL only if file exists
                full_file_path = Path(".") / actual_case_dir / relative_path
                
                # Only generate URL if file actually exists
                if full_file_path.exists() and full_file_path.is_file():
                    media_url = f"/api/media/files/{actual_case_dir}/files/{relative_path}"
                else:
                    media_url = None
            
            video_data = {
                "_id": str(video["_id"]),
                "file_id": video.get("file_id"),
                "file_name": video.get("file_name"),
                "file_path": video.get("file_path"),
                "local_path": video.get("local_path"),
                "media_url": media_url,
                "file_size": video.get("file_size"),
                "file_size_bytes": video.get("file_size_bytes"),
                "file_extension": video.get("file_extension"),
                "sha256": video.get("sha256"),
                "md5": video.get("md5"),
                "creation_time": video.get("creation_time"),
                "modify_time": video.get("modify_time"),
                "access_time": video.get("access_time"),
                "deleted_state": video.get("deleted_state"),
                "embedded": video.get("embedded"),
                "is_related": video.get("is_related"),
                "file_system": video.get("file_system"),
                "source_file": video.get("source_file"),
                "ingestion_timestamp": video.get("ingestion_timestamp"),
                "additional_metadata": video.get("additional_metadata", {})
            }
            processed_videos.append(video_data)
        
        return {
            "case_id": case_id,
            "total_video_files": total_count,
            "returned_video_files": len(processed_videos),
            "offset": effective_offset,
            "limit": limit,
            "video_files": processed_videos,
            "pagination_info": {
                "requested_offset": offset,
                "effective_offset": effective_offset,
                "has_more": effective_offset + len(processed_videos) < total_count,
                "total_pages": max(1, (total_count + limit - 1) // limit) if limit > 0 else 1
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting UFDR video files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ufdr-metadata/{case_id}/documents")
async def get_ufdr_documents(
    case_id: str,
    limit: int = Query(50, description="Maximum number of documents to return"),
    offset: int = Query(0, description="Number of documents to skip"),
    current_user: UserOut = Depends(get_current_user)
):
    """
    Get all document files from UFDR
    """
    try:
        # Validate case_id
        try:
            case_object_id = ObjectId(case_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid case ID format")
        
        # Get database collections
        cases_collection = db["cases"]

        # Verify case exists and user has access
        case = await cases_collection.find_one({"_id": case_object_id})
        if not case:
            raise HTTPException(status_code=404, detail="Case not found")

        # Use case-specific collection for UFDR metadata
        case_collection = db[f"{case['name']}_{case_id}"]

        # Build filter for documents
        filter_query = {
            "case_id": case_object_id,
            "source": "UFDR-Media",
            "media_type": "document"
        }

        # Get total count
        total_count = await case_collection.count_documents(filter_query)
        
        # Get documents with pagination
        cursor = case_collection.find(filter_query).skip(offset).limit(limit)
        documents = await cursor.to_list(length=limit)
        
        # Process documents
        processed_documents = []
        for doc in documents:
            doc_data = {
                "_id": str(doc["_id"]),
                "file_id": doc.get("file_id"),
                "file_name": doc.get("file_name"),
                "file_path": doc.get("file_path"),
                "local_path": doc.get("local_path"),
                "file_size": doc.get("file_size"),
                "file_size_bytes": doc.get("file_size_bytes"),
                "file_extension": doc.get("file_extension"),
                "sha256": doc.get("sha256"),
                "md5": doc.get("md5"),
                "creation_time": doc.get("creation_time"),
                "modify_time": doc.get("modify_time"),
                "access_time": doc.get("access_time"),
                "deleted_state": doc.get("deleted_state"),
                "embedded": doc.get("embedded"),
                "is_related": doc.get("is_related"),
                "file_system": doc.get("file_system"),
                "source_file": doc.get("source_file"),
                "ingestion_timestamp": doc.get("ingestion_timestamp"),
                "additional_metadata": doc.get("additional_metadata", {})
            }
            processed_documents.append(doc_data)
        
        return {
            "case_id": case_id,
            "total_documents": total_count,
            "returned_documents": len(processed_documents),
            "offset": offset,
            "limit": limit,
            "documents": processed_documents
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting UFDR documents: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
