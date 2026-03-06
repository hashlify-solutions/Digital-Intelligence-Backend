import json
import uuid
import asyncio
import threading
from datetime import datetime, timezone
from bson import ObjectId
import logging
import os
import torch
import cv2
# Prevent OpenCV from spawning its own threads (causes segfaults in multi-threaded Celery workers)
cv2.setNumThreads(0)
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from model_registry import ModelRegistry
import numpy as np
from config.db import (
    ufdr_photo_detected_faces_collection,
    ufdr_video_detected_faces_collection,
    ufdr_photo_detected_objects_collection,
    ufdr_video_detected_objects_collection,
)
from config.db import qdrant_client, db, collection_case
from config.settings import settings
from utils.helpers import (
    create_quadrant_collection_if_not_exists,
    robust_qdrant_upsert,
    cleanup_gpu_memory,
    _safe_update_case_processing_status,
    _safe_update_case_total_messages,
    _safe_mark_case_failed,
    _precompute_and_save_geolocations,
    _finalize_case_processing,
)
from qdrant_client.models import Distance, PointStruct
from ingester_v1 import CSVIngester
from ufdr_ingester import UFDRIngester
from analyzer_v1 import ArabicSocialAnalyzer

logger = logging.getLogger(__name__)

# Get compute configuration for parallel processing
_compute_config = settings.compute_config
_batch_size = _compute_config["batch_size"]
_max_workers = _compute_config["max_workers"]

# Global lock to serialize OpenCV/dlib operations that are not thread-safe
# Prevents segfaults when face detection, object detection, etc. run concurrently
_cv2_inference_lock = threading.Lock()


def is_valid_media_file(file_path: str) -> bool:
    """Check if a file is a valid media file (not a macOS resource fork or hidden file)

    Args:
        file_path: Path to the file to validate

    Returns:
        True if the file is a valid media file, False otherwise
    """
    filename = os.path.basename(file_path)
    # Skip macOS resource fork files (start with ._)
    if filename.startswith("._"):
        return False
    # Skip hidden files
    if filename.startswith("."):
        return False
    return True


async def save_json_data(json_path, collection, ufdr_file_id, case_id):
    """Helper method to save JSON data to shared collection"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list) and len(data) > 0:
            # Prepare documents for bulk insert
            documents = []
            current_time = datetime.now(timezone.utc)

            for item in data:
                document = {
                    **item,
                    "ufdr_id": ObjectId(ufdr_file_id),
                    "case_id": ObjectId(case_id),
                    "created_at": current_time,
                    "updated_at": current_time,
                }
                documents.append(document)

            if documents:
                await collection.insert_many(documents)
                logger.info(
                    f"Inserted {len(documents)} documents into {collection.name}"
                )
                return len(documents)

        return 0

    except Exception as e:
        logger.error(f"Error saving JSON data from {json_path}: {str(e)}")
        raise


async def save_media_files(media_dir, collection, ufdr_file_id, media_type, case_id):
    """Helper method to save media file metadata to shared collection"""
    try:
        documents = []
        current_time = datetime.now(timezone.utc)

        for filename in os.listdir(media_dir):
            # Skip macOS resource fork files and hidden files
            if filename.startswith("._") or filename.startswith("."):
                logger.debug(f"Skipping macOS resource fork/hidden file: {filename}")
                continue

            file_path = os.path.join(media_dir, filename)
            if os.path.isfile(file_path):
                # Determine file type based on extension
                file_ext = os.path.splitext(filename)[1].lower()

                document = {
                    "name": filename,
                    "type": file_ext,
                    "media_type": media_type,
                    "path": file_path,
                    "ufdr_id": ObjectId(ufdr_file_id),
                    "case_id": ObjectId(case_id),
                    "created_at": current_time,
                    "updated_at": current_time,
                }
                documents.append(document)

        if documents:
            await collection.insert_many(documents)
            logger.info(
                f"Inserted {len(documents)} {media_type} files into {collection.name}"
            )
            return len(documents)

        return 0

    except Exception as e:
        logger.error(f"Error saving media files from {media_dir}: {str(e)}")
        raise


async def process_face_detection_async(
    ufdr_photos_collection,
    ufdr_photo_detected_faces_collection,
    face_detector_client,
    output_ufdr_path,
    ufdr_file_id,
    case_id,
):
    """Async helper function to process face detection and save results with parallel processing"""
    try:
        logger.info(f"Starting async face detection for case {case_id} with batch processing")
        face_client = ModelRegistry.get_model("face_embeddings")
        
        # Get the ufdr file photos from the database (async)
        ufdr_file_photos = await ufdr_photos_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        # Filter valid media files
        valid_photos = [
            photo for photo in ufdr_file_photos 
            if is_valid_media_file(photo["path"]) and os.path.exists(photo["path"])
        ]
        
        logger.info(f"Processing {len(valid_photos)} valid photos for face detection")
        
        total_faces_detected = 0
        all_face_documents = []
        
        # Process images in batches using thread pool for CPU-bound detection
        loop = asyncio.get_event_loop()
        
        def _detect_faces_sync(input_image_path, output_path):
            """Thread-safe wrapper for face detection (OpenCV DNN is not thread-safe)."""
            with _cv2_inference_lock:
                return face_detector_client.process_image_with_face_detection(
                    input_image_path, output_path
                )

        async def process_single_photo(ufdr_photo):
            """Process a single photo for face detection."""
            input_image_path = ufdr_photo["path"]
            
            # Run CPU-bound detection in thread pool (serialized via lock)
            detection_result = await loop.run_in_executor(
                None,
                _detect_faces_sync,
                input_image_path,
                output_ufdr_path
            )
            
            return ufdr_photo, detection_result
        
        # Process photos sequentially to avoid concurrent OpenCV access
        # (the lock serializes anyway, so batching just adds overhead)
        batch_size = min(_batch_size, 8)  # Smaller batches for stability
        
        for i in range(0, len(valid_photos), batch_size):
            batch = valid_photos[i:i + batch_size]
            
            # Process batch concurrently (lock inside ensures safety)
            tasks = [process_single_photo(photo) for photo in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            current_time = datetime.now(timezone.utc)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in face detection: {result}")
                    continue
                    
                ufdr_photo, detection_result = result
                number_of_faces_saved = detection_result["number_of_faces_saved"]
                saved_file_paths = detection_result["saved_file_paths"]
                
                total_faces_detected += number_of_faces_saved
                
                # Prepare face documents for bulk insert
                for saved_file_path in saved_file_paths:
                    face_embedding = await loop.run_in_executor(
                        None,
                        face_client.extract_face_embedding,
                        saved_file_path
                    )
                    
                    face_document = {
                        "name": os.path.basename(saved_file_path),
                        "type": os.path.splitext(saved_file_path)[1],
                        "media_type": ufdr_photo["media_type"],
                        "path": saved_file_path,
                        "ufdr_id": ObjectId(ufdr_file_id),
                        "ufdr_photo_id": ObjectId(ufdr_photo["_id"]),
                        "case_id": ObjectId(case_id),
                        "created_at": current_time,
                        "updated_at": current_time,
                        "has_embedding": face_embedding is not None,
                        "embedding": face_embedding.tolist() if face_embedding is not None else None,
                    }
                    all_face_documents.append(face_document)
            
            # Bulk insert face documents for this batch
            if all_face_documents:
                await ufdr_photo_detected_faces_collection.insert_many(all_face_documents)
                logger.info(f"Batch {i//batch_size + 1}: Inserted {len(all_face_documents)} face documents")
                all_face_documents = []

        logger.info(
            f"Face detection completed for case {case_id}. Total faces detected: {total_faces_detected}"
        )
        return {
            "status": "completed",
            "case_id": case_id,
            "faces_detected": total_faces_detected,
        }

    except Exception as e:
        logger.error(f"Error in async face detection: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()


async def process_object_detection_async(
    ufdr_photos_collection,
    ufdr_photo_detected_objects_collection,
    object_detector_client,
    output_ufdr_path,
    ufdr_file_id,
    case_id,
):
    """Async helper function to process object detection and save results with parallel processing"""
    try:
        logger.info(f"Starting async object detection for case {case_id} with batch processing")
        object_client = ModelRegistry.get_model("object_embeddings")
        
        # Get the ufdr file photos from the database (async)
        ufdr_file_photos = await ufdr_photos_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        # Filter valid media files
        valid_photos = [
            photo for photo in ufdr_file_photos 
            if is_valid_media_file(photo["path"]) and os.path.exists(photo["path"])
        ]
        
        logger.info(f"Processing {len(valid_photos)} valid photos for object detection")

        total_objects_detected = 0
        all_object_documents = []
        
        # Process images in batches using thread pool for CPU-bound detection
        loop = asyncio.get_event_loop()
        
        def _detect_objects_sync(input_image_path, output_path):
            """Thread-safe wrapper for object detection (OpenCV/YOLO not thread-safe)."""
            with _cv2_inference_lock:
                return object_detector_client.detect_object_from_image(
                    input_image_path, output_path
                )

        async def process_single_photo(ufdr_photo):
            """Process a single photo for object detection."""
            input_image_path = ufdr_photo["path"]
            
            try:
                # Run CPU-bound detection in thread pool (serialized via lock)
                detected_objects_array = await loop.run_in_executor(
                    None,
                    _detect_objects_sync,
                    input_image_path,
                    output_ufdr_path
                )
                return ufdr_photo, detected_objects_array
            except Exception as e:
                logger.error(f"Error processing image {input_image_path}: {str(e)}")
                return ufdr_photo, []
        
        # Process photos in batches for memory efficiency
        batch_size = min(_batch_size, 8)  # Smaller batches for stability
        
        for i in range(0, len(valid_photos), batch_size):
            batch = valid_photos[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [process_single_photo(photo) for photo in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            current_time = datetime.now(timezone.utc)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in object detection: {result}")
                    continue
                    
                ufdr_photo, detected_objects_array = result
                total_objects_detected += len(detected_objects_array)
                
                # Prepare object documents for bulk insert
                for detected_object in detected_objects_array:
                    cropped_object_image_path = detected_object["cropped_object_image_path"]
                    
                    object_embedding = await loop.run_in_executor(
                        None,
                        object_client.extract_object_embedding,
                        cropped_object_image_path
                    )
                    
                    object_document = {
                        "name": os.path.basename(cropped_object_image_path),
                        "type": os.path.splitext(cropped_object_image_path)[1],
                        "class_name": detected_object["class_name"],
                        "media_type": ufdr_photo["media_type"],
                        "path": cropped_object_image_path,
                        "ufdr_id": ObjectId(ufdr_file_id),
                        "ufdr_photo_id": ObjectId(ufdr_photo["_id"]),
                        "case_id": ObjectId(case_id),
                        "created_at": current_time,
                        "updated_at": current_time,
                        "has_embedding": object_embedding is not None,
                        "embedding": object_embedding.tolist() if object_embedding is not None else None,
                    }
                    all_object_documents.append(object_document)
            
            # Bulk insert object documents for this batch
            if all_object_documents:
                await ufdr_photo_detected_objects_collection.insert_many(all_object_documents)
                logger.info(f"Batch {i//batch_size + 1}: Inserted {len(all_object_documents)} object documents")
                all_object_documents = []

        logger.info(
            f"Object detection completed for case {case_id}. Total objects detected: {total_objects_detected}"
        )
        return {
            "status": "completed",
            "case_id": case_id,
            "objects_detected": total_objects_detected,
        }

    except Exception as e:
        logger.error(f"Error in async object detection: {str(e)}")
        return {"status": "error", "case_id": case_id, "error": str(e)}

    finally:
        cleanup_gpu_memory()


async def process_video_face_detection_async(
    ufdr_video_screenshots_collection,
    ufdr_video_detected_faces_collection,
    face_detector_client,
    output_ufdr_path,
    ufdr_file_id,
    case_id,
):
    """Async helper function to process video face detection from screenshots with parallel processing"""
    try:
        logger.info(
            f"Starting async video face detection from screenshots for case {case_id} with batch processing"
        )
        face_client = ModelRegistry.get_model("face_embeddings")
        
        # Get the ufdr file video screenshots from the database (async)
        ufdr_video_screenshots = await ufdr_video_screenshots_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        # Filter valid media files
        valid_screenshots = [
            screenshot for screenshot in ufdr_video_screenshots 
            if is_valid_media_file(screenshot["path"]) and os.path.exists(screenshot["path"])
        ]
        
        logger.info(f"Processing {len(valid_screenshots)} valid screenshots for video face detection")

        total_faces_detected = 0
        all_face_documents = []
        
        # Process screenshots in batches using thread pool
        loop = asyncio.get_event_loop()
        
        def _detect_faces_sync(input_image_path, output_path):
            """Thread-safe wrapper for face detection."""
            with _cv2_inference_lock:
                return face_detector_client.process_image_with_face_detection(
                    input_image_path, output_path
                )

        async def process_single_screenshot(screenshot):
            """Process a single screenshot for face detection."""
            input_image_path = screenshot["path"]
            
            # Run CPU-bound detection in thread pool (serialized via lock)
            detection_result = await loop.run_in_executor(
                None,
                _detect_faces_sync,
                input_image_path,
                output_ufdr_path
            )
            
            return screenshot, detection_result
        
        # Process screenshots in batches for memory efficiency
        batch_size = min(_batch_size, 8)  # Smaller batches for stability
        
        for i in range(0, len(valid_screenshots), batch_size):
            batch = valid_screenshots[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [process_single_screenshot(screenshot) for screenshot in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            current_time = datetime.now(timezone.utc)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in video face detection: {result}")
                    continue
                    
                screenshot, detection_result = result
                number_of_faces_saved = detection_result["number_of_faces_saved"]
                saved_file_paths = detection_result["saved_file_paths"]
                
                total_faces_detected += number_of_faces_saved
                
                # Prepare face documents for bulk insert
                for saved_file_path in saved_file_paths:
                    face_embedding = await loop.run_in_executor(
                        None,
                        face_client.extract_face_embedding,
                        saved_file_path
                    )
                    
                    face_document = {
                        "name": os.path.basename(saved_file_path),
                        "type": os.path.splitext(saved_file_path)[1],
                        "media_type": screenshot["media_type"],
                        "path": saved_file_path,
                        "frame_number": screenshot["frame_number"],
                        "source_video_path": screenshot["source_video_path"],
                        "ufdr_id": ObjectId(ufdr_file_id),
                        "ufdr_video_id": ObjectId(screenshot["ufdr_video_id"]),
                        "ufdr_video_screenshot_id": ObjectId(screenshot["_id"]),
                        "case_id": ObjectId(case_id),
                        "created_at": current_time,
                        "updated_at": current_time,
                        "has_embedding": face_embedding is not None,
                        "embedding": face_embedding.tolist() if face_embedding is not None else None,
                    }
                    all_face_documents.append(face_document)
            
            # Bulk insert face documents for this batch
            if all_face_documents:
                await ufdr_video_detected_faces_collection.insert_many(all_face_documents)
                logger.info(f"Batch {i//batch_size + 1}: Inserted {len(all_face_documents)} video face documents")
                all_face_documents = []

        logger.info(
            f"Video face detection from screenshots completed for case {case_id}. Total faces detected: {total_faces_detected}"
        )
        return {
            "status": "completed",
            "case_id": case_id,
            "faces_detected": total_faces_detected,
        }

    except Exception as e:
        logger.error(f"Error in async video face detection: {str(e)}")
        return {"status": "error", "case_id": case_id, "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def process_video_object_detection_async(
    ufdr_video_screenshots_collection,
    ufdr_video_detected_objects_collection,
    object_detector_client,
    output_ufdr_path,
    ufdr_file_id,
    case_id,
):
    """Async helper function to process video object detection from screenshots with parallel processing"""
    try:
        logger.info(
            f"Starting async video object detection from screenshots for case {case_id} with batch processing"
        )
        object_embedding_client = ModelRegistry.get_model("object_embeddings")
        
        # Get the ufdr file video screenshots from the database (async)
        ufdr_video_screenshots = await ufdr_video_screenshots_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        # Filter valid media files
        valid_screenshots = [
            screenshot for screenshot in ufdr_video_screenshots 
            if is_valid_media_file(screenshot["path"]) and os.path.exists(screenshot["path"])
        ]
        
        logger.info(f"Processing {len(valid_screenshots)} valid screenshots for video object detection")

        total_objects_detected = 0
        all_object_documents = []
        
        # Process screenshots in batches using thread pool
        loop = asyncio.get_event_loop()
        
        def _detect_objects_sync(input_image_path, output_path):
            """Thread-safe wrapper for object detection."""
            with _cv2_inference_lock:
                return object_detector_client.detect_object_from_image(
                    input_image_path, output_path
                )

        async def process_single_screenshot(screenshot):
            """Process a single screenshot for object detection."""
            input_image_path = screenshot["path"]
            
            try:
                # Run CPU-bound detection in thread pool (serialized via lock)
                detected_objects_array = await loop.run_in_executor(
                    None,
                    _detect_objects_sync,
                    input_image_path,
                    output_ufdr_path
                )
                return screenshot, detected_objects_array
            except Exception as e:
                logger.error(f"Error processing screenshot {input_image_path}: {str(e)}")
                return screenshot, []
        
        # Process screenshots in batches for memory efficiency
        batch_size = min(_batch_size, 8)  # Smaller batches for stability
        
        for i in range(0, len(valid_screenshots), batch_size):
            batch = valid_screenshots[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [process_single_screenshot(screenshot) for screenshot in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            current_time = datetime.now(timezone.utc)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in video object detection: {result}")
                    continue
                    
                screenshot, detected_objects_array = result
                total_objects_detected += len(detected_objects_array)
                
                # Prepare object documents for bulk insert
                for detected_object in detected_objects_array:
                    cropped_object_image_path = detected_object["cropped_object_image_path"]
                    
                    object_embedding = await loop.run_in_executor(
                        None,
                        object_embedding_client.extract_object_embedding,
                        cropped_object_image_path
                    )
                    
                    object_document = {
                        "name": os.path.basename(cropped_object_image_path),
                        "type": os.path.splitext(cropped_object_image_path)[1],
                        "class_name": detected_object["class_name"],
                        "media_type": screenshot["media_type"],
                        "path": cropped_object_image_path,
                        "frame_number": screenshot["frame_number"],
                        "source_video_path": screenshot["source_video_path"],
                        "ufdr_id": ObjectId(ufdr_file_id),
                        "ufdr_video_id": ObjectId(screenshot["ufdr_video_id"]),
                        "ufdr_video_screenshot_id": ObjectId(screenshot["_id"]),
                        "case_id": ObjectId(case_id),
                        "created_at": current_time,
                        "updated_at": current_time,
                        "has_embedding": object_embedding is not None,
                        "embedding": object_embedding.tolist() if object_embedding is not None else None,
                    }
                    all_object_documents.append(object_document)
            
            # Bulk insert object documents for this batch
            if all_object_documents:
                await ufdr_video_detected_objects_collection.insert_many(all_object_documents)
                logger.info(f"Batch {i//batch_size + 1}: Inserted {len(all_object_documents)} video object documents")
                all_object_documents = []

        logger.info(
            f"Video object detection from screenshots completed for case {case_id}. Total objects detected: {total_objects_detected}"
        )
        return {
            "status": "completed",
            "case_id": case_id,
            "objects_detected": total_objects_detected,
        }

    except Exception as e:
        logger.error(f"Error in async video object detection: {str(e)}")
        return {"status": "error", "case_id": case_id, "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def segment_video_frames_async(
    ufdr_videos_collection,
    ufdr_video_screenshots_collection,
    output_ufdr_path,
    ufdr_file_id,
    case_id,
    frame_interval=30,
):
    """Async helper function to segment video frames and save them as screenshots"""
    try:
        logger.info(f"Starting async video frame segmentation for case {case_id}")

        # Get the ufdr file videos from the database (async)
        ufdr_file_videos = await ufdr_videos_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        total_frames_saved = 0

        for ufdr_video in ufdr_file_videos:
            input_video_path = ufdr_video["path"]

            # Skip macOS resource fork files and hidden files
            if not is_valid_media_file(input_video_path):
                logger.debug(f"Skipping invalid media file: {input_video_path}")
                continue

            if not os.path.exists(input_video_path):
                logger.warning(f"Video file does not exist: {input_video_path}")
                continue

            try:
                # Process video frame segmentation (sync operation)
                cap = cv2.VideoCapture(input_video_path)
                if not cap.isOpened():
                    logger.error(f"Cannot open video file: {input_video_path}")
                    continue

                # Create output directory for screenshots
                screenshots_dir = os.path.join(output_ufdr_path, "video_screenshots")
                os.makedirs(screenshots_dir, exist_ok=True)

                base_filename = os.path.splitext(os.path.basename(input_video_path))[0]
                frame_count = 0
                saved_frames = []

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Save every frame_interval frames
                    if frame_count % frame_interval == 0:
                        # Generate frame filename
                        frame_filename = f"{base_filename}_frame_{frame_count:06d}.jpg"
                        frame_path = os.path.join(screenshots_dir, frame_filename)

                        # Save frame as image
                        if cv2.imwrite(frame_path, frame):
                            saved_frames.append(
                                {
                                    "name": frame_filename,
                                    "type": ".jpg",
                                    "media_type": "photo",  # Screenshots are treated as photos
                                    "path": frame_path,
                                    "frame_number": frame_count,
                                    "source_video_path": input_video_path,
                                }
                            )
                            total_frames_saved += 1
                            logger.debug(
                                f"Saved frame {frame_count} from {input_video_path}"
                            )
                        else:
                            logger.warning(
                                f"Failed to save frame {frame_count} from {input_video_path}"
                            )

                    frame_count += 1

                cap.release()
                logger.info(
                    f"Segmented {len(saved_frames)} frames from {input_video_path}"
                )

                # Save the segmented frames to the database (async)
                if saved_frames:
                    screenshot_documents = []
                    current_time = datetime.now(timezone.utc)

                    for frame_data in saved_frames:
                        screenshot_document = {
                            "name": frame_data["name"],
                            "type": frame_data["type"],
                            "media_type": frame_data["media_type"],
                            "path": frame_data["path"],
                            "frame_number": frame_data["frame_number"],
                            "source_video_path": frame_data["source_video_path"],
                            "ufdr_id": ObjectId(ufdr_file_id),
                            "ufdr_video_id": ObjectId(ufdr_video["_id"]),
                            "case_id": ObjectId(case_id),
                            "created_at": current_time,
                            "updated_at": current_time,
                        }
                        screenshot_documents.append(screenshot_document)

                    await ufdr_video_screenshots_collection.insert_many(
                        screenshot_documents
                    )
                    logger.info(
                        f"Saved {len(screenshot_documents)} video screenshots to database"
                    )

            except Exception as e:
                logger.error(f"Error processing video {input_video_path}: {str(e)}")
                continue

        logger.info(
            f"Video frame segmentation completed for case {case_id}. Total frames saved: {total_frames_saved}"
        )
        return {
            "status": "completed",
            "case_id": case_id,
            "frames_saved": total_frames_saved,
        }

    except Exception as e:
        logger.error(f"Error in async video frame segmentation: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def process_detector_embedding_async(detector_id, case_id, detectors_collection):
    """Async helper function to generate embedding for a detector"""
    try:
        logger.info(f"Processing detector embedding for detector {detector_id}")

        # Get detector from database
        detector = await detectors_collection.find_one({"_id": ObjectId(detector_id)})
        if not detector:
            logger.error(f"Detector not found: {detector_id}")
            return {"status": "error", "message": "Detector not found"}

        detector_type = detector["type"]
        image_path = detector["image_path"]

        if not os.path.exists(image_path):
            logger.error(f"Detector image not found: {image_path}")
            return {"status": "error", "message": "Detector image not found"}

        # Generate embedding based on detector type
        embedding = None
        if detector_type == "person":
            face_client = ModelRegistry.get_model("face_embeddings")
            embedding = face_client.extract_face_embedding(image_path)
        elif detector_type == "object":
            object_client = ModelRegistry.get_model("object_embeddings")
            embedding = object_client.extract_object_embedding(image_path)

        if embedding is None:
            logger.error(f"Failed to generate embedding for detector {detector_id}")
            return {"status": "error", "message": "Failed to generate embedding"}

        # Convert numpy array to list for MongoDB storage
        embedding_list = embedding.tolist()

        # Update detector with embedding
        await detectors_collection.update_one(
            {"_id": ObjectId(detector_id)},
            {
                "$set": {
                    "embedding": embedding_list,
                    "has_embedding": True,
                    "updated_at": datetime.now(timezone.utc),
                }
            },
        )

        # Import locally to avoid circular import
        from tasks.celery_tasks import new_detector_match_task

        new_detector_match_task.delay(case_id, detector_id)
        logger.info(f"Successfully generated embedding for detector {detector_id}")

        return {
            "status": "completed",
            "detector_id": detector_id,
            "embedding_size": len(embedding_list),
        }

    except Exception as e:
        logger.error(f"Error processing detector embedding: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def analyze_detector_matches_async(
    case_id,
    detector_type,
    detectors_collection,
    detector_matches_collection,
    detector_settings_collection,
):
    """Async helper function to analyze all detected items against detectors"""
    try:
        logger.info(
            f"Starting detector match analysis for case {case_id}, type: {detector_type}"
        )
        current_time = datetime.now(timezone.utc)
        # Get detector settings for thresholds
        settings = await detector_settings_collection.find_one(
            {"case_id": ObjectId(case_id)}
        )
        if not settings:
            # Create default settings
            default_settings = {
                "case_id": ObjectId(case_id),
                "face_thresholds": {
                    "high_confidence": 0.9,
                    "medium_confidence": 0.75,
                    "low_confidence": 0.6,
                    "minimum_match": 0.96,
                },
                "object_thresholds": {
                    "high_confidence": 0.85,
                    "medium_confidence": 0.7,
                    "low_confidence": 0.55,
                    "minimum_match": 0.96,
                },
                "created_at": current_time,
                "updated_at": current_time,
            }
            await detector_settings_collection.insert_one(default_settings)
            settings = default_settings

        # Get detectors for this case
        detector_query = {"case_id": ObjectId(case_id), "has_embedding": True}
        if detector_type:
            detector_query["type"] = detector_type

        detectors = await detectors_collection.find(detector_query).to_list(length=None)

        if not detectors:
            logger.info(f"No detectors with embeddings found for case {case_id}")
            return {
                "status": "completed",
                "matches_found": 0,
                "message": "No detectors found",
            }

        total_matches = 0

        # Initialize embedding clients
        face_client = ModelRegistry.get_model("face_embeddings")
        object_client = ModelRegistry.get_model("object_embeddings")

        # Process each detector
        for detector in detectors:
            detector_id = detector["_id"]
            detector_embedding = np.array(detector["embedding"])
            detector_name = detector["name"]
            detector_type_current = detector["type"]

            # Get appropriate thresholds
            if detector_type_current == "person":
                thresholds = settings["face_thresholds"]
                # Get detected faces
                detected_items = []

                # Photo detected faces
                photo_faces = await ufdr_photo_detected_faces_collection.find(
                    {"case_id": ObjectId(case_id)}
                ).to_list(length=None)
                for face in photo_faces:
                    detected_items.append(
                        {
                            "item": face,
                            "type": "face",
                            "collection": "ufdr_photo_detected_faces",
                        }
                    )

                # Video detected faces
                video_faces = await ufdr_video_detected_faces_collection.find(
                    {"case_id": ObjectId(case_id)}
                ).to_list(length=None)
                for face in video_faces:
                    detected_items.append(
                        {
                            "item": face,
                            "type": "face",
                            "collection": "ufdr_video_detected_faces",
                        }
                    )

            elif detector_type_current == "object":
                thresholds = settings["object_thresholds"]
                # Get detected objects
                detected_items = []

                # Photo detected objects
                photo_objects = await ufdr_photo_detected_objects_collection.find(
                    {"case_id": ObjectId(case_id)}
                ).to_list(length=None)
                for obj in photo_objects:
                    detected_items.append(
                        {
                            "item": obj,
                            "type": "object",
                            "collection": "ufdr_photo_detected_objects",
                        }
                    )

                # Video detected objects
                video_objects = await ufdr_video_detected_objects_collection.find(
                    {"case_id": ObjectId(case_id)}
                ).to_list(length=None)
                for obj in video_objects:
                    detected_items.append(
                        {
                            "item": obj,
                            "type": "object",
                            "collection": "ufdr_video_detected_objects",
                        }
                    )

            # Process each detected item
            for detected_item_data in detected_items:
                detected_item = detected_item_data["item"]
                detected_item_type = detected_item_data["type"]
                detected_item_collection = detected_item_data["collection"]
                detected_item_path = detected_item["path"]

                if not os.path.exists(detected_item_path):
                    logger.warning(
                        f"Detected item file not found: {detected_item_path}"
                    )
                    continue

                # Generate embedding for detected item
                if detected_item_type == "face":
                    detected_embedding = face_client.extract_face_embedding(
                        detected_item_path
                    )
                else:
                    detected_embedding = object_client.extract_object_embedding(
                        detected_item_path
                    )

                if detected_embedding is None:
                    logger.warning(
                        f"Failed to generate embedding for detected item: {detected_item_path}"
                    )
                    continue

                # Compute similarity
                if detected_item_type == "face":
                    similarity = face_client.compute_similarity(
                        detector_embedding, detected_embedding
                    )
                else:
                    similarity = object_client.compute_similarity(
                        detector_embedding, detected_embedding
                    )

                # Check if similarity meets minimum threshold
                if similarity < thresholds["minimum_match"]:
                    continue

                # Determine confidence level
                if similarity >= thresholds["high_confidence"]:
                    confidence_level = "high"
                elif similarity >= thresholds["medium_confidence"]:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"

                # Create match record
                match_document = {
                    "case_id": ObjectId(case_id),
                    "ufdr_id": ObjectId(detected_item["ufdr_id"]),
                    "detector_id": ObjectId(detector_id),
                    "detector_name": detector_name,
                    "detector_type": detector_type_current,
                    "detected_item_type": detected_item_type,
                    "detected_item_id": ObjectId(detected_item["_id"]),
                    "detected_item_collection": detected_item_collection,
                    "detected_item_path": detected_item_path,
                    "similarity_score": similarity,
                    "confidence_level": confidence_level,
                    "match_threshold": thresholds["minimum_match"],
                    "created_at": current_time,
                }

                # Add metadata if available
                if "frame_number" in detected_item:
                    match_document["frame_number"] = detected_item["frame_number"]
                if "source_video_path" in detected_item:
                    match_document["source_video_path"] = detected_item[
                        "source_video_path"
                    ]

                # Check if match already exists (to avoid duplicates)
                existing_match = await detector_matches_collection.find_one(
                    {
                        "detector_id": ObjectId(detector_id),
                        "detected_item_id": ObjectId(detected_item["_id"]),
                    }
                )

                if existing_match:
                    # Update existing match if similarity is higher
                    if similarity > existing_match["similarity_score"]:
                        await detector_matches_collection.update_one(
                            {"_id": existing_match["_id"]},
                            {
                                "$set": {
                                    "similarity_score": similarity,
                                    "confidence_level": confidence_level,
                                    "updated_at": current_time,
                                }
                            },
                        )
                        logger.info(
                            f"Updated match for detector {detector_name} with higher similarity: {similarity:.3f}"
                        )
                else:
                    # Insert new match
                    await detector_matches_collection.insert_one(match_document)
                    total_matches += 1
                    logger.info(
                        f"Found match for detector {detector_name}: {similarity:.3f} ({confidence_level})"
                    )

        logger.info(
            f"Detector match analysis completed for case {case_id}. Total matches: {total_matches}"
        )
        return {
            "status": "completed",
            "case_id": case_id,
            "matches_found": total_matches,
        }

    except Exception as e:
        logger.error(f"Error in detector match analysis: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def new_detector_match_async(
    case_id,
    detector_id,
    detectors_collection,
    detector_matches_collection,
    detector_settings_collection,
    ufdr_photo_detected_faces_collection,
    ufdr_video_detected_faces_collection,
    ufdr_photo_detected_objects_collection,
    ufdr_video_detected_objects_collection,
):
    try:
        logger.info(
            f"Matching new detection for detector {detector_id} in case {case_id}"
        )
        current_time = datetime.now(timezone.utc)
        settings = await detector_settings_collection.find_one(
            {"case_id": ObjectId(case_id)}
        )
        if not settings:
            default_settings = {
                "case_id": ObjectId(case_id),
                "face_thresholds": {
                    "high_confidence": 0.9,
                    "medium_confidence": 0.75,
                    "low_confidence": 0.6,
                    "minimum_match": 0.96,
                },
                "object_thresholds": {
                    "high_confidence": 0.85,
                    "medium_confidence": 0.7,
                    "low_confidence": 0.55,
                    "minimum_match": 0.96,
                },
                "created_at": current_time,
                "updated_at": current_time,
            }
            await detector_settings_collection.insert_one(default_settings)
            settings = default_settings

        # Get detector with detector_id
        detector_doc = await detectors_collection.find_one(
            {"_id": ObjectId(detector_id), "has_embedding": True}
        )
        if not detector_doc:
            logger.info(f"No detector with id {detector_id} found for case {case_id}")
            return {
                "status": "completed",
                "matches_found": 0,
                "message": "No detectors found",
            }

        total_matches = 0
        # Initialize embedding clients
        face_embedding_client = ModelRegistry.get_model("face_embeddings")
        object_embedding_client = ModelRegistry.get_model("object_embeddings")
        detector_type_current = detector_doc["type"]
        detector_name = detector_doc["name"]
        detector_embedding = np.array(detector_doc["embedding"])

        detected_items = []
        # Get appropriate thresholds
        if detector_type_current == "person":
            thresholds = settings["face_thresholds"]
            # Get detected faces
            # Photo detected faces
            photo_faces = await ufdr_photo_detected_faces_collection.find(
                {"case_id": ObjectId(case_id), "has_embedding": True}
            ).to_list(length=None)
            for face in photo_faces:
                detected_items.append(
                    {
                        "item": face,
                        "type": "face",
                        "collection": "ufdr_photo_detected_faces",
                    }
                )
            # Video detected faces
            video_faces = await ufdr_video_detected_faces_collection.find(
                {"case_id": ObjectId(case_id), "has_embedding": True}
            ).to_list(length=None)
            for face in video_faces:
                detected_items.append(
                    {
                        "item": face,
                        "type": "face",
                        "collection": "ufdr_video_detected_faces",
                    }
                )

        elif detector_type_current == "object":
            thresholds = settings["object_thresholds"]
            # Get detected objects
            # Photo detected objects
            photo_objects = await ufdr_photo_detected_objects_collection.find(
                {"case_id": ObjectId(case_id), "has_embedding": True}
            ).to_list(length=None)
            for obj in photo_objects:
                detected_items.append(
                    {
                        "item": obj,
                        "type": "object",
                        "collection": "ufdr_photo_detected_objects",
                    }
                )
            # Video detected objects
            video_objects = await ufdr_video_detected_objects_collection.find(
                {"case_id": ObjectId(case_id), "has_embedding": True}
            ).to_list(length=None)
            for obj in video_objects:
                detected_items.append(
                    {
                        "item": obj,
                        "type": "object",
                        "collection": "ufdr_video_detected_objects",
                    }
                )

        # Process each detected item
        for detected_item_data in detected_items:
            detected_item = detected_item_data["item"]
            detected_item_type = detected_item_data["type"]
            detected_item_collection = detected_item_data["collection"]
            detected_item_path = detected_item["path"]
            detected_embedding = np.array(detected_item["embedding"])

            if not os.path.exists(detected_item_path):
                logger.warning(f"Detected item file not found: {detected_item_path}")
                continue

            # Compute similarity
            if (
                detected_item_type == "face"
                and detector_embedding is not None
                and detected_embedding is not None
            ):
                similarity = face_embedding_client.compute_similarity(
                    detector_embedding, detected_embedding
                )
            elif (
                detected_item_type == "object"
                and detector_embedding is not None
                and detected_embedding is not None
            ):
                similarity = object_embedding_client.compute_similarity(
                    detector_embedding, detected_embedding
                )

            # Check if similarity meets minimum threshold
            if similarity < thresholds["minimum_match"]:
                continue

            # Determine confidence level
            if similarity >= thresholds["high_confidence"]:
                confidence_level = "high"
            elif similarity >= thresholds["medium_confidence"]:
                confidence_level = "medium"
            else:
                confidence_level = "low"

            # Create match record
            match_document = {
                "case_id": ObjectId(case_id),
                "ufdr_id": ObjectId(detected_item["ufdr_id"]),
                "detector_id": ObjectId(detector_id),
                "detector_name": detector_name,
                "detector_type": detector_type_current,
                "detected_item_type": detected_item_type,
                "detected_item_id": ObjectId(detected_item["_id"]),
                "detected_item_collection": detected_item_collection,
                "detected_item_path": detected_item_path,
                "similarity_score": similarity,
                "confidence_level": confidence_level,
                "match_threshold": thresholds["minimum_match"],
                "created_at": current_time,
            }

            # Add metadata if available
            if "frame_number" in detected_item:
                match_document["frame_number"] = detected_item["frame_number"]
            if "source_video_path" in detected_item:
                match_document["source_video_path"] = detected_item["source_video_path"]

            # Insert new match
            await detector_matches_collection.insert_one(match_document)
            total_matches += 1
            logger.info(
                f"Found match for detector {detector_name}: {similarity:.3f} ({confidence_level})"
            )

        return {
            "status": "completed",
            "detector_id": detector_id,
            "case_id": case_id,
            "matches_found": total_matches,
        }

    except Exception as e:
        logger.error(f"Error in new_detector_match_async: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def process_detector_matches_async(
    case_id,
    ufdr_file_id,
    detectors_collection,
    detector_matches_collection,
    detector_settings_collection,
    ufdr_photo_detected_faces_collection,
    ufdr_video_detected_faces_collection,
    ufdr_photo_detected_objects_collection,
    ufdr_video_detected_objects_collection,
):
    try:
        current_time = datetime.now(
            timezone.utc
        )  # Define current_time at the beginning

        # Get all the detectors of a case
        detectors = await detectors_collection.find(
            {"case_id": ObjectId(case_id)}
        ).to_list(length=None)

        total_matches = 0
        # Initialize embedding clients
        face_embedding_client = ModelRegistry.get_model("face_embeddings")
        object_embedding_client = ModelRegistry.get_model("object_embeddings")

        settings = await detector_settings_collection.find_one(
            {"case_id": ObjectId(case_id)}
        )
        if not settings:
            default_settings = {
                "case_id": ObjectId(case_id),
                "face_thresholds": {
                    "high_confidence": 0.9,
                    "medium_confidence": 0.75,
                    "low_confidence": 0.6,
                    "minimum_match": 0.96,
                },
                "object_thresholds": {
                    "high_confidence": 0.85,
                    "medium_confidence": 0.7,
                    "low_confidence": 0.55,
                    "minimum_match": 0.96,
                },
                "created_at": current_time,
                "updated_at": current_time,
            }
            await detector_settings_collection.insert_one(default_settings)
            settings = default_settings

        for detector in detectors:
            detector_id = detector["_id"]

            logger.info(
                f"Processing detector matches for the case {case_id} for ufdr file {ufdr_file_id}"
            )
            current_time = datetime.now(timezone.utc)
            # Get detector with detector_id
            detector_doc = await detectors_collection.find_one(
                {"_id": ObjectId(detector_id), "has_embedding": True}
            )
            if not detector_doc:
                logger.info(
                    f"No detector with id {detector_id} found for case {case_id}"
                )
                continue

            detector_type_current = detector_doc["type"]
            detector_name = detector_doc["name"]
            detector_embedding = np.array(detector_doc["embedding"])

            detected_items = []
            # Get appropriate thresholds
            if detector_type_current == "person":
                thresholds = settings["face_thresholds"]
                # Get detected faces
                # Photo detected faces
                photo_faces = await ufdr_photo_detected_faces_collection.find(
                    {"ufdr_id": ObjectId(ufdr_file_id), "has_embedding": True}
                ).to_list(length=None)
                for face in photo_faces:
                    detected_items.append(
                        {
                            "item": face,
                            "type": "face",
                            "collection": "ufdr_photo_detected_faces",
                        }
                    )
                # Video detected faces
                video_faces = await ufdr_video_detected_faces_collection.find(
                    {"ufdr_id": ObjectId(ufdr_file_id), "has_embedding": True}
                ).to_list(length=None)
                for face in video_faces:
                    detected_items.append(
                        {
                            "item": face,
                            "type": "face",
                            "collection": "ufdr_video_detected_faces",
                        }
                    )

            elif detector_type_current == "object":
                thresholds = settings["object_thresholds"]
                # Get detected objects
                # Photo detected objects
                photo_objects = await ufdr_photo_detected_objects_collection.find(
                    {"ufdr_id": ObjectId(ufdr_file_id), "has_embedding": True}
                ).to_list(length=None)
                for obj in photo_objects:
                    detected_items.append(
                        {
                            "item": obj,
                            "type": "object",
                            "collection": "ufdr_photo_detected_objects",
                        }
                    )
                # Video detected objects
                video_objects = await ufdr_video_detected_objects_collection.find(
                    {"ufdr_id": ObjectId(ufdr_file_id), "has_embedding": True}
                ).to_list(length=None)
                for obj in video_objects:
                    detected_items.append(
                        {
                            "item": obj,
                            "type": "object",
                            "collection": "ufdr_video_detected_objects",
                        }
                    )

            # Process each detected item
            for detected_item_data in detected_items:
                detected_item = detected_item_data["item"]
                detected_item_type = detected_item_data["type"]
                detected_item_collection = detected_item_data["collection"]
                detected_item_path = detected_item["path"]
                detected_embedding = np.array(detected_item["embedding"])

                if not os.path.exists(detected_item_path):
                    logger.warning(
                        f"Detected item file not found: {detected_item_path}"
                    )
                    continue

                # Compute similarity
                if (
                    detected_item_type == "face"
                    and detector_embedding is not None
                    and detected_embedding is not None
                ):
                    similarity = face_embedding_client.compute_similarity(
                        detector_embedding, detected_embedding
                    )
                elif (
                    detected_item_type == "object"
                    and detector_embedding is not None
                    and detected_embedding is not None
                ):
                    similarity = object_embedding_client.compute_similarity(
                        detector_embedding, detected_embedding
                    )

                # Check if similarity meets minimum threshold
                if similarity < thresholds["minimum_match"]:
                    continue

                # Determine confidence level
                if similarity >= thresholds["high_confidence"]:
                    confidence_level = "high"
                elif similarity >= thresholds["medium_confidence"]:
                    confidence_level = "medium"
                else:
                    confidence_level = "low"

                # Create match record
                match_document = {
                    "case_id": ObjectId(case_id),
                    "ufdr_id": ObjectId(detected_item["ufdr_id"]),
                    "detector_id": ObjectId(detector_id),
                    "detector_name": detector_name,
                    "detector_type": detector_type_current,
                    "detected_item_type": detected_item_type,
                    "detected_item_id": ObjectId(detected_item["_id"]),
                    "detected_item_collection": detected_item_collection,
                    "detected_item_path": detected_item_path,
                    "similarity_score": similarity,
                    "confidence_level": confidence_level,
                    "match_threshold": thresholds["minimum_match"],
                    "created_at": current_time,
                }

                # Add metadata if available
                if "frame_number" in detected_item:
                    match_document["frame_number"] = detected_item["frame_number"]
                if "source_video_path" in detected_item:
                    match_document["source_video_path"] = detected_item[
                        "source_video_path"
                    ]

                # Insert new match
                await detector_matches_collection.insert_one(match_document)
                total_matches += 1
                logger.info(
                    f"Found match for detector {detector_name}: {similarity:.3f} ({confidence_level})"
                )

        return {
            "status": "completed",
            "ufdr_file_id": str(ufdr_file_id),
            "case_id": str(case_id),
            "matches_found": total_matches,
        }

    except Exception as e:
        logger.error(f"Error in process_detector_matches_async: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def analyze_audio_async(
    case_id,
    ufdr_file_id,
    ufdr_audio_collection,
    transcriber_client,
    analyzer_client,
    rag_analyzer_client,
    vector_size,
    is_llama_validation_enabled,
):
    """Analyze audio files with batched AI analysis and Qdrant upserts."""
    try:
        ufdr_audios = await ufdr_audio_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        if not ufdr_audios:
            logger.error(f"No audios found in UFDR file: {ufdr_file_id}")
            return {"status": "error", "message": "No audios found in UFDR file"}

        # Step 1: Transcribe all audios first (I/O bound, sequential)
        transcription_results = []
        for ufdr_audio in ufdr_audios:
            ufdr_audio_id = ufdr_audio["_id"]
            ufdr_audio_path = ufdr_audio["path"]

            if not is_valid_media_file(ufdr_audio_path):
                logger.debug(f"Skipping invalid media file: {ufdr_audio_path}")
                continue

            try:
                transribed_text_doc = transcriber_client.transcribe(ufdr_audio_path)
                if transribed_text_doc is None:
                    logger.warning(f"Transcription returned None for {ufdr_audio_path}, skipping")
                    continue
                if not transribed_text_doc.get("text"):
                    error_detail = transribed_text_doc.get("error", "empty transcription")
                    logger.warning(f"No transcription text for {ufdr_audio_path}: {error_detail}")
                    continue
                transcription_results.append({
                    "audio_id": ufdr_audio_id,
                    "audio_path": ufdr_audio_path,
                    "transcription": transribed_text_doc["text"],
                    "language": transribed_text_doc["language"],
                    "segments": transribed_text_doc["segments"],
                })
            except Exception as e:
                logger.error(f"Error transcribing audio {ufdr_audio_path}: {e}")
                continue

        if not transcription_results:
            logger.info("No valid audio transcriptions to analyze")
            return {"status": "completed", "total_audios_analyzed": 0}

        logger.info(f"Transcribed {len(transcription_results)} audios, starting batch AI analysis...")

        # Step 2: Batch AI analysis on all transcriptions
        all_texts = [r["transcription"] for r in transcription_results]
        
        # Use batch methods from analyzer's clients for GPU-optimal inference
        try:
            topic_results = analyzer_client.classifier_client.classify_batch(
                all_texts, analyzer_client.content_categories["topic"]
            )
            interaction_results = analyzer_client.classifier_client.classify_batch(
                all_texts, analyzer_client.content_categories["interaction_type"]
            )
            sentiment_results = analyzer_client.classifier_client.classify_batch(
                all_texts, analyzer_client.content_categories["sentiment_aspects"]
            )
            toxicity_results = analyzer_client.toxic_client.analyze_toxicity_batch(all_texts)
            emotion_results = analyzer_client.emotion_client.analyze_sentiment_batch(all_texts)
        except Exception as e:
            logger.error(f"Batch AI analysis failed, falling back to sequential: {e}")
            topic_results = interaction_results = sentiment_results = None
            toxicity_results = emotion_results = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

        # Step 3: Assemble results, bulk-update MongoDB, batch-upsert Qdrant
        from pymongo import UpdateOne as _UpdateOne
        bulk_ops = []
        qdrant_points = []
        new_quadrant_media_case_collection = f"case_{case_id}_media"
        create_quadrant_collection_if_not_exists(
            new_quadrant_media_case_collection, vector_size, Distance.COSINE
        )

        # Batch generate embeddings for all transcriptions
        try:
            embedding_client = rag_analyzer_client.embedding_client
            all_embeddings = embedding_client.create_embeddings_batch(all_texts)
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            all_embeddings = [None] * len(all_texts)

        for i, result in enumerate(transcription_results):
            try:
                if topic_results and i < len(topic_results):
                    # Use batch results to build analysis
                    from utils.translationmap import topic_translation_map, interaction_translation_map, sentiment_translation_map
                    topic_cls = topic_results[i] if topic_results else {"labels": [], "scores": []}
                    interact_cls = interaction_results[i] if interaction_results else {"labels": [], "scores": []}
                    sent_cls = sentiment_results[i] if sentiment_results else {"labels": [], "scores": []}
                    tox_cls = toxicity_results[i] if toxicity_results else {"score": 0.0, "label": "non-toxic"}
                    emo_cls = emotion_results[i] if emotion_results else {"score": 0.0, "label": "neutral"}
                    if tox_cls is None:
                        tox_cls = {"score": 0.0, "label": "non-toxic"}

                    analysis = {
                        "text_metadata": {
                            "content": result["transcription"][:500],
                            "length": len(result["transcription"]),
                            "language_info": {
                                "primary_language": result.get("language", "unknown"),
                                "is_multilingual": False,
                                "has_emojis": False,
                            },
                            "is_multilingual": False,
                        },
                        "content_analysis": {
                            "topics": [{"category": topic_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                       for l, s in zip((topic_cls.get("labels") or [])[:3], (topic_cls.get("scores") or [])[:3])],
                            "interaction": [{"type": interaction_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                           for l, s in zip((interact_cls.get("labels") or [])[:3], (interact_cls.get("scores") or [])[:3])],
                            "sentiment_aspects": [{"aspect": sentiment_translation_map.get(l, l), "strength": round(s * 100, 2)}
                                                 for l, s in zip((sent_cls.get("labels") or [])[:3], (sent_cls.get("scores") or [])[:3])],
                        },
                        "sentiment_metrics": {"emotion": emo_cls.get("label", "neutral"), "emotion_confidence": round(emo_cls.get("score", 0.0) * 100, 2)},
                        "toxicity": {"toxicity_score": round(tox_cls.get("score", 0.0) * 100, 2), "toxicity_label": tox_cls.get("label", "non-toxic")},
                        "entity": [],
                        "entities_classification": {},
                    }
                    analysis_summary_response = analyzer_client.process_analyze_content_response(analysis)
                else:
                    # Fallback: sequential analysis
                    analysis = await analyzer_client.analyze_content(result["transcription"])
                    analysis_summary_response = analyzer_client.process_analyze_content_response(analysis)

                collection_update_doc = {
                    "analysis_summary": analysis_summary_response,
                    "language": result["language"],
                    "transcription": result["transcription"],
                    "segments": result["segments"],
                    "updated_at": datetime.now(timezone.utc),
                }

                bulk_ops.append(
                    _UpdateOne(
                        {"_id": ObjectId(result["audio_id"])},
                        {"$set": collection_update_doc}
                    )
                )

                # Prepare Qdrant point
                embedding = all_embeddings[i] if i < len(all_embeddings) and all_embeddings[i] is not None else None
                if embedding is not None:
                    import numpy as np
                    normalized = rag_analyzer_client.normalize_embedding(np.array(embedding))
                    qdrant_points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        payload={
                            "type": "audio",
                            "ufdr_audio_id": str(result["audio_id"]),
                            "ufdr_audio_path": result["audio_path"],
                            "transcription": result["transcription"],
                            "case_id": str(case_id),
                            "ufdr_id": str(ufdr_file_id),
                        },
                        vector=normalized.tolist() if hasattr(normalized, 'tolist') else list(normalized),
                    ))

            except Exception as e:
                logger.error(f"Error processing audio {result['audio_id']}: {e}")
                continue

        # Step 4: Bulk write to MongoDB
        if bulk_ops:
            try:
                await ufdr_audio_collection.bulk_write(bulk_ops, ordered=False)
                logger.info(f"Bulk updated {len(bulk_ops)} audio documents")
            except Exception as e:
                logger.error(f"Bulk write failed for audio: {e}")

        # Step 5: Batch upsert to Qdrant
        if qdrant_points:
            success = robust_qdrant_upsert(
                collection_name=new_quadrant_media_case_collection,
                points=qdrant_points,
                max_retries=3,
            )
            if success:
                logger.info(f"Batch upserted {len(qdrant_points)} audio embeddings to Qdrant")
            else:
                logger.error(f"Batch Qdrant upsert failed for audio embeddings")

        total_audios_analyzed = len(bulk_ops)
        logger.info(f"Analyzed {total_audios_analyzed} audios (batched)")
        return {"status": "completed", "total_audios_analyzed": total_audios_analyzed}

    except Exception as e:
        logger.error(f"Error in analyze_audio_async: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()


async def analyze_video_async(
    case_id,
    ufdr_file_id,
    ufdr_video_collection,
    output_audio_dir,
    transcriber_client,
    analyzer_client,
    rag_analyzer_client,
    video_to_audio_converter_client,
    vector_size,
    is_llama_validation_enabled=False,
):
    """Analyze video files with batched AI analysis and Qdrant upserts."""
    try:
        ufdr_videos = await ufdr_video_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        if not ufdr_videos:
            logger.error(f"No videos found in UFDR file: {ufdr_file_id}")
            return {"status": "error", "message": "No videos found in UFDR file"}

        # Step 1: Convert + transcribe all videos (I/O bound, sequential)
        transcription_results = []
        for ufdr_video in ufdr_videos:
            ufdr_video_id = ufdr_video["_id"]
            ufdr_video_path = ufdr_video["path"]

            if not is_valid_media_file(ufdr_video_path):
                logger.debug(f"Skipping invalid media file: {ufdr_video_path}")
                continue

            try:
                video_filename = os.path.splitext(os.path.basename(ufdr_video_path))[0]
                safe_filename = "".join(
                    c for c in video_filename if c.isalnum() or c in (" ", "-", "_")
                ).rstrip()
                if not safe_filename:
                    safe_filename = f"video_{ufdr_video_id}"
                output_audio_path = os.path.join(output_audio_dir, f"{safe_filename}.mp3")

                converted_audio_path = video_to_audio_converter_client.convert_video_to_audio(
                    ufdr_video_path, output_audio_path
                )
                if not converted_audio_path:
                    logger.error(f"Error converting video to audio: {ufdr_video_path}")
                    continue

                transribed_text_doc = transcriber_client.transcribe(converted_audio_path)
                if transribed_text_doc is None:
                    logger.warning(f"Transcription returned None for video audio {converted_audio_path}, skipping")
                    continue
                if not transribed_text_doc.get("text"):
                    error_detail = transribed_text_doc.get("error", "empty transcription")
                    logger.warning(f"No transcription text for video audio {converted_audio_path}: {error_detail}")
                    continue

                transcription_results.append({
                    "video_id": ufdr_video_id,
                    "video_path": ufdr_video_path,
                    "converted_audio_path": converted_audio_path,
                    "transcription": transribed_text_doc["text"],
                    "language": transribed_text_doc["language"],
                    "segments": transribed_text_doc["segments"],
                })
            except Exception as e:
                logger.error(f"Error processing video {ufdr_video_path}: {e}")
                continue

        if not transcription_results:
            logger.info("No valid video transcriptions to analyze")
            return {"status": "completed", "total_videos_analyzed": 0}

        logger.info(f"Transcribed {len(transcription_results)} videos, starting batch AI analysis...")

        # Step 2: Batch AI analysis
        all_texts = [r["transcription"] for r in transcription_results]
        
        try:
            topic_results = analyzer_client.classifier_client.classify_batch(
                all_texts, analyzer_client.content_categories["topic"]
            )
            interaction_results = analyzer_client.classifier_client.classify_batch(
                all_texts, analyzer_client.content_categories["interaction_type"]
            )
            sentiment_results = analyzer_client.classifier_client.classify_batch(
                all_texts, analyzer_client.content_categories["sentiment_aspects"]
            )
            toxicity_results = analyzer_client.toxic_client.analyze_toxicity_batch(all_texts)
            emotion_results = analyzer_client.emotion_client.analyze_sentiment_batch(all_texts)
        except Exception as e:
            logger.error(f"Batch AI analysis failed for video, falling back to sequential: {e}")
            topic_results = interaction_results = sentiment_results = None
            toxicity_results = emotion_results = None
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except Exception:
                    pass

        # Step 3: Batch embeddings
        try:
            embedding_client = rag_analyzer_client.embedding_client
            all_embeddings = embedding_client.create_embeddings_batch(all_texts)
        except Exception as e:
            logger.error(f"Batch embedding generation failed for video: {e}")
            all_embeddings = [None] * len(all_texts)

        # Step 4: Assemble results, bulk update MongoDB, batch upsert Qdrant
        from pymongo import UpdateOne as _UpdateOne
        bulk_ops = []
        qdrant_points = []
        new_quadrant_media_case_collection = f"case_{case_id}_media"
        create_quadrant_collection_if_not_exists(
            new_quadrant_media_case_collection, vector_size, Distance.COSINE
        )

        for i, result in enumerate(transcription_results):
            try:
                if topic_results and i < len(topic_results):
                    from utils.translationmap import topic_translation_map, interaction_translation_map, sentiment_translation_map
                    topic_cls = topic_results[i] if topic_results else {"labels": [], "scores": []}
                    interact_cls = interaction_results[i] if interaction_results else {"labels": [], "scores": []}
                    sent_cls = sentiment_results[i] if sentiment_results else {"labels": [], "scores": []}
                    tox_cls = toxicity_results[i] if toxicity_results else {"score": 0.0, "label": "non-toxic"}
                    emo_cls = emotion_results[i] if emotion_results else {"score": 0.0, "label": "neutral"}
                    if tox_cls is None:
                        tox_cls = {"score": 0.0, "label": "non-toxic"}

                    analysis = {
                        "text_metadata": {
                            "content": result["transcription"][:500],
                            "length": len(result["transcription"]),
                            "language_info": {
                                "primary_language": result.get("language", "unknown"),
                                "is_multilingual": False,
                                "has_emojis": False,
                            },
                            "is_multilingual": False,
                        },
                        "content_analysis": {
                            "topics": [{"category": topic_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                       for l, s in zip((topic_cls.get("labels") or [])[:3], (topic_cls.get("scores") or [])[:3])],
                            "interaction": [{"type": interaction_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                           for l, s in zip((interact_cls.get("labels") or [])[:3], (interact_cls.get("scores") or [])[:3])],
                            "sentiment_aspects": [{"aspect": sentiment_translation_map.get(l, l), "strength": round(s * 100, 2)}
                                                 for l, s in zip((sent_cls.get("labels") or [])[:3], (sent_cls.get("scores") or [])[:3])],
                        },
                        "sentiment_metrics": {"emotion": emo_cls.get("label", "neutral"), "emotion_confidence": round(emo_cls.get("score", 0.0) * 100, 2)},
                        "toxicity": {"toxicity_score": round(tox_cls.get("score", 0.0) * 100, 2), "toxicity_label": tox_cls.get("label", "non-toxic")},
                        "entity": [],
                        "entities_classification": {},
                    }
                    analysis_summary_response = analyzer_client.process_analyze_content_response(analysis)
                else:
                    analysis = await analyzer_client.analyze_content(result["transcription"])
                    analysis_summary_response = analyzer_client.process_analyze_content_response(analysis)

                collection_update_doc = {
                    "converted_audio_path": result["converted_audio_path"],
                    "analysis_summary": analysis,
                    "language": result["language"],
                    "transcription": result["transcription"],
                    "segments": result["segments"],
                    "updated_at": datetime.now(timezone.utc),
                }

                bulk_ops.append(
                    _UpdateOne(
                        {"_id": ObjectId(result["video_id"])},
                        {"$set": collection_update_doc}
                    )
                )

                embedding = all_embeddings[i] if i < len(all_embeddings) and all_embeddings[i] is not None else None
                if embedding is not None:
                    import numpy as np
                    normalized = rag_analyzer_client.normalize_embedding(np.array(embedding))
                    qdrant_points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        payload={
                            "type": "video_transcription",
                            "ufdr_video_id": str(result["video_id"]),
                            "ufdr_video_path": result["video_path"],
                            "transcription": result["transcription"],
                            "case_id": str(case_id),
                            "ufdr_id": str(ufdr_file_id),
                        },
                        vector=normalized.tolist() if hasattr(normalized, 'tolist') else list(normalized),
                    ))

            except Exception as e:
                logger.error(f"Error processing video {result['video_id']}: {e}")
                continue

        # Bulk write to MongoDB
        if bulk_ops:
            try:
                await ufdr_video_collection.bulk_write(bulk_ops, ordered=False)
                logger.info(f"Bulk updated {len(bulk_ops)} video documents")
            except Exception as e:
                logger.error(f"Bulk write failed for video: {e}")

        # Batch upsert to Qdrant
        if qdrant_points:
            success = robust_qdrant_upsert(
                collection_name=new_quadrant_media_case_collection,
                points=qdrant_points,
                max_retries=3,
            )
            if success:
                logger.info(f"Batch upserted {len(qdrant_points)} video embeddings to Qdrant")

        total_videos_analyzed = len(bulk_ops)
        logger.info(f"Analyzed {total_videos_analyzed} videos (batched)")
        return {"status": "completed", "total_videos_analyzed": total_videos_analyzed}

    except Exception as e:
        logger.error(f"Error in analyze_video_async: {str(e)}")
        return {"status": "error", "error": str(e)}

    finally:
        cleanup_gpu_memory()


async def detect_nsfw_images_async(
    ufdr_file_id,
    ufdr_photos_collection,
    ufdr_video_collection,
    ufdr_video_screenshots_collection,
    nsfw_detector_client,
):
    """Async helper function to detect NSFW images with parallel batch processing"""
    try:
        logger.info(f"Starting NSFW image detection for the ufdr_file: {ufdr_file_id} with batch processing")

        ufdr_file_photos = await ufdr_photos_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        ufdr_file_videos = await ufdr_video_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        loop = asyncio.get_event_loop()
        
        # Filter valid photos
        valid_photos = [
            photo for photo in ufdr_file_photos 
            if is_valid_media_file(photo["path"]) and os.path.exists(photo["path"])
        ]
        
        logger.info(f"Processing {len(valid_photos)} valid photos for NSFW detection")
        
        # Process photos in batches
        batch_size = min(_batch_size, 64)  # NSFW detection is faster, can use larger batches
        nsfw_updates = []
        non_nsfw_updates = []
        
        for i in range(0, len(valid_photos), batch_size):
            batch = valid_photos[i:i + batch_size]
            
            async def detect_single_photo(photo):
                """Detect NSFW in a single photo."""
                is_nsfw = await loop.run_in_executor(
                    None,
                    nsfw_detector_client.is_nsfw_image,
                    photo["path"]
                )
                return photo, is_nsfw
            
            # Process batch concurrently
            tasks = [detect_single_photo(photo) for photo in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Error in NSFW detection: {result}")
                    continue
                    
                photo, is_nsfw = result
                if is_nsfw:
                    logger.info(f"NSFW image detected: {photo['path']}")
                    nsfw_updates.append(photo["_id"])
                else:
                    non_nsfw_updates.append(photo["_id"])
        
        # Bulk update NSFW status
        if nsfw_updates:
            await ufdr_photos_collection.update_many(
                {"_id": {"$in": [ObjectId(id) for id in nsfw_updates]}},
                {"$set": {"is_nsfw": True}},
            )
            logger.info(f"Marked {len(nsfw_updates)} photos as NSFW")
        
        if non_nsfw_updates:
            await ufdr_photos_collection.update_many(
                {"_id": {"$in": [ObjectId(id) for id in non_nsfw_updates]}},
                {"$set": {"is_nsfw": False}},
            )
            logger.info(f"Marked {len(non_nsfw_updates)} photos as not NSFW")

        # Detect NSFW images from the ufdr file videos (video screenshots)
        for ufdr_video in ufdr_file_videos:
            ufdr_video_id = ufdr_video["_id"]
            ufdr_file_video_screenshots = await ufdr_video_screenshots_collection.find(
                {"ufdr_video_id": ObjectId(ufdr_video_id)}
            ).to_list(length=None)

            # Default to not NSFW
            await ufdr_video_collection.update_one(
                {"_id": ObjectId(ufdr_video_id)},
                {"$set": {"is_nsfw": False}},
            )
            
            # Filter valid screenshots
            valid_screenshots = [
                screenshot for screenshot in ufdr_file_video_screenshots 
                if is_valid_media_file(screenshot["path"]) and os.path.exists(screenshot["path"])
            ]
            
            # Process screenshots in batches to find any NSFW content
            found_nsfw = False
            for i in range(0, len(valid_screenshots), batch_size):
                if found_nsfw:
                    break
                    
                batch = valid_screenshots[i:i + batch_size]
                
                async def detect_single_screenshot(screenshot):
                    """Detect NSFW in a single screenshot."""
                    is_nsfw = await loop.run_in_executor(
                        None,
                        nsfw_detector_client.is_nsfw_image,
                        screenshot["path"]
                    )
                    return screenshot, is_nsfw
                
                tasks = [detect_single_screenshot(screenshot) for screenshot in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in batch_results:
                    if isinstance(result, Exception):
                        continue
                        
                    screenshot, is_nsfw = result
                    if is_nsfw:
                        logger.info(
                            f"NSFW image detected from video {ufdr_video_id} on path: {screenshot['path']}"
                        )
                        await ufdr_video_collection.update_one(
                            {"_id": ObjectId(ufdr_video_id)},
                            {"$set": {"is_nsfw": True}},
                        )
                        found_nsfw = True
                        break

        logger.info(f"Completed NSFW image detection for the ufdr_file: {ufdr_file_id}")
        return {"status": "completed", "ufdr_file_id": ufdr_file_id}

    except Exception as e:
        logger.error(f"Error in detect_nsfw_images_async: {str(e)}")
        return {"status": "error", "ufdr_file_id": ufdr_file_id, "error": str(e)}

    finally:
        cleanup_gpu_memory()
        cleanup_gpu_memory()


async def generate_image_description_llava_async(
    ufdr_file_id,
    case_id,
    ufdr_photos_collection,
    case_collection,
    llava_client,
    llama_client,
    rag_analyzer_client,
    analyzer_client,
    vector_size,
    is_llama_validation_enabled=False,
):
    """Generate image descriptions with batched AI analysis + embeddings."""
    try:
        logger.info(
            f"Starting BATCHED image description generation for ufdr_file: {ufdr_file_id}"
        )
        ufdr_photos = await ufdr_photos_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        # Step 1: Generate descriptions for all photos (LLaVA is sequential per image)
        description_results = []
        for ufdr_photo in ufdr_photos:
            ufdr_photo_id = ufdr_photo["_id"]
            ufdr_photo_path = ufdr_photo["path"]

            if not is_valid_media_file(ufdr_photo_path):
                logger.debug(f"Skipping invalid media file: {ufdr_photo_path}")
                continue

            try:
                description = llava_client.describe_image(ufdr_photo_path)
                if description:
                    description_results.append({
                        "photo_id": ufdr_photo_id,
                        "photo_path": ufdr_photo_path,
                        "description": description,
                    })
                else:
                    logger.error(f"Error generating image description for photo {ufdr_photo_path}")
            except Exception as e:
                logger.error(f"Error describing image {ufdr_photo_path}: {e}")

        if not description_results:
            logger.info("No valid image descriptions generated")
            return {"status": "completed", "ufdr_file_id": ufdr_file_id}

        logger.info(f"Generated {len(description_results)} descriptions, starting batch AI analysis...")

        # Step 2: Batch AI analysis on all descriptions
        all_descriptions = [r["description"] for r in description_results]
        
        try:
            topic_results = analyzer_client.classifier_client.classify_batch(
                all_descriptions, analyzer_client.content_categories["topic"]
            )
            interaction_results = analyzer_client.classifier_client.classify_batch(
                all_descriptions, analyzer_client.content_categories["interaction_type"]
            )
            sentiment_results = analyzer_client.classifier_client.classify_batch(
                all_descriptions, analyzer_client.content_categories["sentiment_aspects"]
            )
            toxicity_results = analyzer_client.toxic_client.analyze_toxicity_batch(all_descriptions)
            emotion_results = analyzer_client.emotion_client.analyze_sentiment_batch(all_descriptions)
        except Exception as e:
            logger.error(f"Batch AI analysis failed for images: {e}")
            topic_results = interaction_results = sentiment_results = None
            toxicity_results = emotion_results = None

        # Step 3: Batch entity extraction via llama_client (still sequential per description for LLM)
        all_entities = []
        all_entity_classifications = []
        case_doc = await case_collection.find_one({"_id": ObjectId(case_id)})
        entities_classes = case_doc.get("entitiesClasses", []) if case_doc else []
        total_desc = len(description_results)

        for idx, desc_result in enumerate(description_results):
            logger.info(f"Entity extraction for image {idx + 1}/{total_desc}: {desc_result['photo_path']}")
            try:
                entities = llama_client.extract_entities(desc_result["description"])
                all_entities.append(entities or [])
            except Exception as e:
                logger.error(f"Entity extraction failed for photo {desc_result['photo_path']}: {e}")
                all_entities.append([])

            try:
                if all_entities[-1]:
                    ec = llama_client.classify_entities(all_entities[-1], entities_classes)
                    all_entity_classifications.append(ec or {})
                else:
                    all_entity_classifications.append({})
            except Exception as e:
                logger.error(f"Entity classification failed: {e}")
                all_entity_classifications.append({})

        logger.info(f"Entity extraction completed for all {total_desc} images")

        # Step 4: Batch embeddings
        try:
            embedding_client = rag_analyzer_client.embedding_client
            all_embeddings = embedding_client.create_embeddings_batch(all_descriptions)
        except Exception as e:
            logger.error(f"Batch embedding generation failed for images: {e}")
            all_embeddings = [None] * len(all_descriptions)

        # Step 5: Assemble results, bulk update MongoDB, batch upsert Qdrant
        from pymongo import UpdateOne as _UpdateOne
        bulk_ops = []
        qdrant_points = []
        new_quadrant_media_case_collection = f"case_{case_id}_media"
        create_quadrant_collection_if_not_exists(
            new_quadrant_media_case_collection, vector_size, Distance.COSINE
        )

        for i, desc_result in enumerate(description_results):
            try:
                if topic_results and i < len(topic_results):
                    from utils.translationmap import topic_translation_map, interaction_translation_map, sentiment_translation_map
                    topic_cls = topic_results[i] if topic_results else {"labels": [], "scores": []}
                    interact_cls = interaction_results[i] if interaction_results else {"labels": [], "scores": []}
                    sent_cls = sentiment_results[i] if sentiment_results else {"labels": [], "scores": []}
                    tox_cls = toxicity_results[i] if toxicity_results else {"score": 0.0, "label": "non-toxic"}
                    emo_cls = emotion_results[i] if emotion_results else {"score": 0.0, "label": "neutral"}
                    if tox_cls is None:
                        tox_cls = {"score": 0.0, "label": "non-toxic"}

                    analysis_summary = {
                        "content_analysis": {
                            "topics": [{"category": topic_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                       for l, s in zip((topic_cls.get("labels") or [])[:3], (topic_cls.get("scores") or [])[:3])],
                            "interaction": [{"type": interaction_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                           for l, s in zip((interact_cls.get("labels") or [])[:3], (interact_cls.get("scores") or [])[:3])],
                            "sentiment_aspects": [{"aspect": sentiment_translation_map.get(l, l), "strength": round(s * 100, 2)}
                                                 for l, s in zip((sent_cls.get("labels") or [])[:3], (sent_cls.get("scores") or [])[:3])],
                        },
                        "sentiment_metrics": {"emotion": emo_cls.get("label", "neutral"), "emotion_confidence": round(emo_cls.get("score", 0.0) * 100, 2)},
                        "toxicity": {"toxicity_score": round(tox_cls.get("score", 0.0) * 100, 2), "toxicity_label": tox_cls.get("label", "non-toxic")},
                        "entity": all_entities[i] if i < len(all_entities) else [],
                        "entities_classification": all_entity_classifications[i] if i < len(all_entity_classifications) else {},
                    }
                else:
                    analysis_summary = await analyzer_client.analyze_content(desc_result["description"])

                entities = all_entities[i] if i < len(all_entities) else []
                entities_classification = all_entity_classifications[i] if i < len(all_entity_classifications) else {}

                collection_update_doc = {
                    "analysis_summary": analysis_summary,
                    "description": desc_result["description"],
                    "entities": entities,
                    "entities_classification": entities_classification,
                }

                bulk_ops.append(
                    _UpdateOne(
                        {"_id": ObjectId(desc_result["photo_id"])},
                        {"$set": collection_update_doc}
                    )
                )

                embedding = all_embeddings[i] if i < len(all_embeddings) and all_embeddings[i] is not None else None
                if embedding is not None:
                    import numpy as np
                    normalized = rag_analyzer_client.normalize_embedding(np.array(embedding))
                    qdrant_points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        payload={
                            "type": "photo",
                            "ufdr_photo_id": str(desc_result["photo_id"]),
                            "ufdr_photo_path": desc_result["photo_path"],
                            "description": desc_result["description"],
                            "case_id": str(case_id),
                            "ufdr_id": str(ufdr_file_id),
                        },
                        vector=normalized.tolist() if hasattr(normalized, 'tolist') else list(normalized),
                    ))

            except Exception as e:
                logger.error(f"Error processing photo {desc_result['photo_id']}: {e}")
                continue

        # Bulk write to MongoDB
        if bulk_ops:
            try:
                await ufdr_photos_collection.bulk_write(bulk_ops, ordered=False)
                logger.info(f"Bulk updated {len(bulk_ops)} photo documents")
            except Exception as e:
                logger.error(f"Bulk write failed for photos: {e}")

        # Batch upsert to Qdrant
        if qdrant_points:
            success = robust_qdrant_upsert(
                collection_name=new_quadrant_media_case_collection,
                points=qdrant_points,
                max_retries=3,
            )
            if success:
                logger.info(f"Batch upserted {len(qdrant_points)} photo embeddings to Qdrant")

        logger.info(
            f"Completed BATCHED image description generation for ufdr_file: {ufdr_file_id} "
            f"({len(bulk_ops)} photos processed)"
        )
        return {"status": "completed", "ufdr_file_id": ufdr_file_id}

    except Exception as e:
        logger.error(f"Error in generate_image_description_llava_async: {str(e)}")
        return {"status": "error", "ufdr_file_id": ufdr_file_id, "error": str(e)}


async def generate_video_frame_description_llava_async(
    ufdr_file_id,
    case_id,
    ufdr_video_screenshots_collection,
    case_collection,
    llava_client,
    llama_client,
    analyzer_client,
    rag_analyzer_client,
    vector_size,
    is_llama_validation_enabled=False,
):
    """Generate video frame descriptions with batched AI analysis + embeddings."""
    try:
        logger.info(
            f"Starting BATCHED video frame description generation for ufdr_file: {ufdr_file_id}"
        )
        ufdr_video_screenshots = await ufdr_video_screenshots_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)

        # Step 1: Generate descriptions for all screenshots (LLaVA is sequential)
        description_results = []
        for ss in ufdr_video_screenshots:
            ss_id = ss["_id"]
            ss_path = ss["path"]

            if not is_valid_media_file(ss_path):
                logger.debug(f"Skipping invalid media file: {ss_path}")
                continue

            try:
                description = llava_client.describe_image(ss_path)
                if description:
                    description_results.append({
                        "ss_id": ss_id,
                        "ss_path": ss_path,
                        "source_video_path": ss.get("source_video_path"),
                        "frame_number": ss.get("frame_number"),
                        "description": description,
                    })
                else:
                    logger.error(f"Error generating description for screenshot {ss_path}")
            except Exception as e:
                logger.error(f"Error describing screenshot {ss_path}: {e}")

        if not description_results:
            logger.info("No valid video frame descriptions generated")
            return {"status": "completed", "ufdr_file_id": ufdr_file_id}

        logger.info(f"Generated {len(description_results)} frame descriptions, starting batch AI analysis...")

        # Step 2: Batch AI analysis
        all_descriptions = [r["description"] for r in description_results]
        
        try:
            topic_results = analyzer_client.classifier_client.classify_batch(
                all_descriptions, analyzer_client.content_categories["topic"]
            )
            interaction_results = analyzer_client.classifier_client.classify_batch(
                all_descriptions, analyzer_client.content_categories["interaction_type"]
            )
            sentiment_results = analyzer_client.classifier_client.classify_batch(
                all_descriptions, analyzer_client.content_categories["sentiment_aspects"]
            )
            toxicity_results = analyzer_client.toxic_client.analyze_toxicity_batch(all_descriptions)
            emotion_results = analyzer_client.emotion_client.analyze_sentiment_batch(all_descriptions)
        except Exception as e:
            logger.error(f"Batch AI analysis failed for video frames: {e}")
            topic_results = interaction_results = sentiment_results = None
            toxicity_results = emotion_results = None

        # Step 3: Entity extraction (LLM, sequential)
        all_entities = []
        all_entity_classifications = []
        case_doc = await case_collection.find_one({"_id": ObjectId(case_id)})
        entities_classes = case_doc.get("entitiesClasses", []) if case_doc else []
        total_desc = len(description_results)

        for idx, desc_result in enumerate(description_results):
            logger.info(f"Entity extraction for video frame {idx + 1}/{total_desc}: {desc_result['ss_path']}")
            try:
                entities = llama_client.extract_entities(desc_result["description"])
                all_entities.append(entities or [])
            except Exception as e:
                logger.error(f"Entity extraction failed for frame {desc_result['ss_path']}: {e}")
                all_entities.append([])

            try:
                if all_entities[-1]:
                    ec = llama_client.classify_entities(all_entities[-1], entities_classes)
                    all_entity_classifications.append(ec or {})
                else:
                    all_entity_classifications.append({})
            except Exception as e:
                logger.error(f"Entity classification failed: {e}")
                all_entity_classifications.append({})

        logger.info(f"Entity extraction completed for all {total_desc} video frames")

        # Step 4: Batch embeddings
        try:
            embedding_client = rag_analyzer_client.embedding_client
            all_embeddings = embedding_client.create_embeddings_batch(all_descriptions)
        except Exception as e:
            logger.error(f"Batch embedding generation failed for video frames: {e}")
            all_embeddings = [None] * len(all_descriptions)

        # Step 5: Assemble, bulk update MongoDB, batch upsert Qdrant
        from pymongo import UpdateOne as _UpdateOne
        bulk_ops = []
        qdrant_points = []
        new_quadrant_media_case_collection = f"case_{case_id}_media"
        create_quadrant_collection_if_not_exists(
            new_quadrant_media_case_collection, vector_size, Distance.COSINE
        )

        for i, desc_result in enumerate(description_results):
            try:
                if topic_results and i < len(topic_results):
                    from utils.translationmap import topic_translation_map, interaction_translation_map, sentiment_translation_map
                    topic_cls = topic_results[i] if topic_results else {"labels": [], "scores": []}
                    interact_cls = interaction_results[i] if interaction_results else {"labels": [], "scores": []}
                    sent_cls = sentiment_results[i] if sentiment_results else {"labels": [], "scores": []}
                    tox_cls = toxicity_results[i] if toxicity_results else {"score": 0.0, "label": "non-toxic"}
                    emo_cls = emotion_results[i] if emotion_results else {"score": 0.0, "label": "neutral"}
                    if tox_cls is None:
                        tox_cls = {"score": 0.0, "label": "non-toxic"}

                    analysis_summary = {
                        "content_analysis": {
                            "topics": [{"category": topic_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                       for l, s in zip((topic_cls.get("labels") or [])[:3], (topic_cls.get("scores") or [])[:3])],
                            "interaction": [{"type": interaction_translation_map.get(l, l), "confidence": round(s * 100, 2)}
                                           for l, s in zip((interact_cls.get("labels") or [])[:3], (interact_cls.get("scores") or [])[:3])],
                            "sentiment_aspects": [{"aspect": sentiment_translation_map.get(l, l), "strength": round(s * 100, 2)}
                                                 for l, s in zip((sent_cls.get("labels") or [])[:3], (sent_cls.get("scores") or [])[:3])],
                        },
                        "sentiment_metrics": {"emotion": emo_cls.get("label", "neutral"), "emotion_confidence": round(emo_cls.get("score", 0.0) * 100, 2)},
                        "toxicity": {"toxicity_score": round(tox_cls.get("score", 0.0) * 100, 2), "toxicity_label": tox_cls.get("label", "non-toxic")},
                        "entity": all_entities[i] if i < len(all_entities) else [],
                        "entities_classification": all_entity_classifications[i] if i < len(all_entity_classifications) else {},
                    }
                else:
                    analysis_summary = await analyzer_client.analyze_content(desc_result["description"])

                entities = all_entities[i] if i < len(all_entities) else []
                entities_classification = all_entity_classifications[i] if i < len(all_entity_classifications) else {}

                collection_update_doc = {
                    "description": desc_result["description"],
                    "entities": entities,
                    "entities_classification": entities_classification,
                    "analysis_summary": analysis_summary,
                }

                bulk_ops.append(
                    _UpdateOne(
                        {"_id": ObjectId(desc_result["ss_id"])},
                        {"$set": collection_update_doc}
                    )
                )

                embedding = all_embeddings[i] if i < len(all_embeddings) and all_embeddings[i] is not None else None
                if embedding is not None:
                    import numpy as np
                    normalized = rag_analyzer_client.normalize_embedding(np.array(embedding))
                    qdrant_points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        payload={
                            "type": "video_screenshot",
                            "ufdr_video_screenshot_id": str(desc_result["ss_id"]),
                            "ufdr_video_screenshot_path": desc_result["ss_path"],
                            "source_video_path": desc_result.get("source_video_path"),
                            "frame_number": desc_result.get("frame_number"),
                            "description": desc_result["description"],
                            "case_id": str(case_id),
                            "ufdr_id": str(ufdr_file_id),
                        },
                        vector=normalized.tolist() if hasattr(normalized, 'tolist') else list(normalized),
                    ))

            except Exception as e:
                logger.error(f"Error processing screenshot {desc_result['ss_id']}: {e}")
                continue

        # Bulk write to MongoDB
        if bulk_ops:
            try:
                await ufdr_video_screenshots_collection.bulk_write(bulk_ops, ordered=False)
                logger.info(f"Bulk updated {len(bulk_ops)} video screenshot documents")
            except Exception as e:
                logger.error(f"Bulk write failed for video screenshots: {e}")

        # Batch upsert to Qdrant
        if qdrant_points:
            success = robust_qdrant_upsert(
                collection_name=new_quadrant_media_case_collection,
                points=qdrant_points,
                max_retries=3,
            )
            if success:
                logger.info(f"Batch upserted {len(qdrant_points)} video screenshot embeddings to Qdrant")

        logger.info(
            f"Completed BATCHED video frame description generation for ufdr_file: {ufdr_file_id} "
            f"({len(bulk_ops)} frames processed)"
        )
        return {"status": "completed", "ufdr_file_id": ufdr_file_id}

    except Exception as e:
        logger.error(f"Error in generate_video_frame_description_llava_async: {str(e)}")
        return {"status": "error", "ufdr_file_id": ufdr_file_id, "error": str(e)}


async def generate_video_description_async(
    ufdr_file_id,
    case_id,
    ufdr_video_collection,
    ufdr_video_screenshots_collection,
    case_collection,
    llama_client,
    analyzer_client,
    rag_analyzer_client,
    vector_size,
    is_llama_validation_enabled=False,
):
    try:
        logger.info(
            f"Starting video description generation with Llava for the ufdr_file: {ufdr_file_id}"
        )
        ufdr_videos = await ufdr_video_collection.find(
            {"ufdr_id": ObjectId(ufdr_file_id)}
        ).to_list(length=None)
        case_doc = await case_collection.find_one({"_id": ObjectId(case_id)})
        entities_classes = case_doc["entitiesClasses"]

        for ufdr_video in ufdr_videos:
            try:
                video_description = ""
                video_entities = []
                ufdr_video_id = ufdr_video["_id"]
                ufdr_video_screenshots = (
                    await ufdr_video_screenshots_collection.find(
                        {"ufdr_video_id": ObjectId(ufdr_video_id)}
                    )
                    .sort("frame_number", 1)
                    .to_list(length=None)
                )

                for ufdr_video_screenshot in ufdr_video_screenshots:
                    ufdr_video_screenshot_path = ufdr_video_screenshot["path"]
                    ufdr_video_screenshot_frame_number = ufdr_video_screenshot[
                        "frame_number"
                    ]
                    ufdr_video_screenshot_description = ufdr_video_screenshot["description"]
                    ufdr_video_screenshot_entities = ufdr_video_screenshot["entities"]

                    logger.info(
                        f"Image description for video screenshot {ufdr_video_screenshot_path} frame number {ufdr_video_screenshot_frame_number} is\n: {ufdr_video_screenshot_description}"
                    )
                    if ufdr_video_screenshot_description:
                        video_description += f"Frame number {ufdr_video_screenshot_frame_number} Description: {ufdr_video_screenshot_description}\n"
                        video_entities.extend(ufdr_video_screenshot_entities)
                    else:
                        logger.error(
                            f"Description for video screenshot {ufdr_video_screenshot_path} frame number {ufdr_video_screenshot_frame_number} is not found"
                        )
                        continue

                video_entities = list(set(video_entities))
                logger.info(f"Video entities: {video_entities}")

                if video_description:
                    with open("prompts/video_description_summarization.txt", "r") as file:
                        video_frames_description_summarization_prompt = file.read()
                    final_video_description = llama_client.chat(
                        video_frames_description_summarization_prompt,
                        {"video_frames_description": video_description.strip()},
                    )
                    try:
                        video_entities_classification = llama_client.classify_entities(
                            video_entities, entities_classes
                        )
                    except Exception as e:
                        logger.error(
                            f"Error classifying video entities for video {ufdr_video_id}: {e}"
                        )
                        video_entities_classification = {}

                    logger.info(
                        f"Starting to analyze the video description for {ufdr_video_id} video"
                    )
                    analysis_summary = await analyzer_client.analyze_content(
                        video_description
                    )
                    analysis_summary_response = (
                        analyzer_client.process_analyze_content_response(analysis_summary)
                    )
                    logger.info(f"Analysis summary response: {analysis_summary_response}")
                    collection_update_doc = {
                        "description": (
                            final_video_description
                            if final_video_description
                            else video_description
                        ),
                        "entities": video_entities,
                        "entities_classification": video_entities_classification,
                        "analysis_summary": analysis_summary,
                    }
                    if is_llama_validation_enabled:
                        llama_validation_summary = (
                            await analyzer_client.validate_analysis_summary_via_llama(
                                video_description, analysis_summary_response
                            )
                        )
                        collection_update_doc["llama_validation_summary"] = (
                            llama_validation_summary
                        )
                    await ufdr_video_collection.update_one(
                        {"_id": ObjectId(ufdr_video_id)},
                        {"$set": collection_update_doc},
                    )
                    logger.info(
                        f"Description, entities and entities classification for {ufdr_video_id} video are saved in the database"
                    )

                    logger.info(
                        f"Starting to save the embeddings for {ufdr_video_id} video in the quadrant"
                    )
                    new_quadrant_media_case_collection = f"case_{case_id}_media"
                    create_quadrant_collection_if_not_exists(
                        new_quadrant_media_case_collection, vector_size, Distance.COSINE
                    )
                    video_description_embeddings = (
                        rag_analyzer_client.query_embedding_endpoint(
                            final_video_description
                        )
                    )
                    if video_description_embeddings is not None and len(video_description_embeddings) > 0:
                        logger.info(f"Embeddings exist for {ufdr_video_id} video")
                        video_point = PointStruct(
                            id=str(uuid.uuid4()),
                            payload={
                                "type": "video_description",
                                "ufdr_video_id": str(ufdr_video_id),
                                "ufdr_video_path": ufdr_video["path"],
                                "description": final_video_description,
                                "case_id": str(case_id),
                                "ufdr_id": str(ufdr_file_id),
                            },
                            vector=video_description_embeddings,
                        )
                        success = robust_qdrant_upsert(
                            collection_name=new_quadrant_media_case_collection,
                            points=[video_point],
                            max_retries=3,
                        )
                        if success:
                            logger.info(
                                f"Embeddings for {ufdr_video_id} video saved successfully in quadrant"
                            )
                        else:
                            logger.error(
                                f"Failed to save embeddings for {ufdr_video_id} video to quadrant after multiple attempts"
                            )
                    else:
                        logger.warning(
                            f"No embeddings generated for {ufdr_video_id} video, skipping quadrant upsert"
                        )
            except Exception as e:
                logger.error(f"Error processing video {ufdr_video_id} description: {e}")
                continue

        logger.info(
            f"Completed video description generation for the ufdr_file: {ufdr_file_id}"
        )
        return {"status": "completed", "ufdr_file_id": ufdr_file_id}

    except Exception as e:
        logger.error(f"Error in generate_video_description_async: {str(e)}")
        return {"status": "error", "ufdr_file_id": ufdr_file_id, "error": str(e)}


async def process_csv_upload_v1_helper(
    folder_path,
    file_path,
    file_extension,
    case_id,
    ufdr_file_id,
    case_name,
    alert_id,
    topics,
    sentiments,
    interactions,
    entities_classes,
    is_rag,
    models_profile,
    note_classifications,
    browsing_history_classifications,
    is_llama_validation_enabled,
):
    """Background task to process the case data"""
    try:
        collection = db[f"{case_name}_{case_id}"]
        # Choosing appropriate ingester based on file extension
        if file_extension == "csv":
            ingester = CSVIngester(collection, collection_case, case_id, models_profile)
        elif file_extension == "ufdr" or file_extension == "ufd":
            ingester = UFDRIngester(
                collection, collection_case, case_id, models_profile
            )

        # Get Llama model settings from models_profile
        llama_settings = models_profile.get("llama", {})
        basic_params = llama_settings.get("basic_params", None)
        advanced_params = llama_settings.get("advanced_params", None)

        analyzer = ArabicSocialAnalyzer(
            collection,
            collection_case,
            case_id,
            alert_id,
            topics,
            sentiments,
            interactions,
            entities_classes,
            models_profile,
            use_parallel_processing=True,
            note_classifications=note_classifications,
            browsing_history_classifications=browsing_history_classifications,
            is_llama_validation_enabled=is_llama_validation_enabled,
            # llama_basic_params=basic_params,
            # llama_advanced_params=advanced_params
        )

        # Update case status to processing
        await _safe_update_case_processing_status(case_id)

        # Process the data using the correct pipeline
        if file_extension == "csv":
            await ingester.ingest_csv_manually(file_path)
            if is_rag:
                await ingester.ingest_rag_data()
        else:
            await ingester.ingest_ufdr_file(file_path)

        # Count total messages after ingestion
        total_messages = await collection.count_documents({})

        # Update case with total messages count
        await _safe_update_case_total_messages(case_id, total_messages)

        await analyzer.process_documents()

        # Precompute and persist geolocations for this case
        # try:
        #     await _precompute_and_save_geolocations(case_id, case_name, models_profile)
        # except Exception as geo_err:
        #     logger.error(
        #         f"Failed to precompute geolocations for case {case_id}: {geo_err}"
        #     )

        # Triggering the ufdr media processing pipeline
        if file_extension == "ufdr" or file_extension == "ufd":
            # Lazy import to avoid circular dependency issues
            from tasks.celery_tasks import process_ufdr_upload

            logger.info(
                f"Triggering the ufdr media processing pipeline for the ufdr_file: {ufdr_file_id} after the ingestion and processing of the ufdr file's text data"
            )
            process_ufdr_upload.delay(
                input_ufdr_path=file_path,
                output_ufdr_path=folder_path,
                case_name=case_name,
                case_id=case_id,
                ufdr_file_id=ufdr_file_id,
                alert_id=alert_id,
                topics=topics,
                sentiments=sentiments,
                interactions=interactions,
                entitiesClasses=entities_classes,
                model_profile=models_profile,
                note_classifications=note_classifications,
                browsing_history_classifications=browsing_history_classifications,
                is_llama_validation_enabled=is_llama_validation_enabled,
            )
        else:
            await _finalize_case_processing(case_id)

    except Exception as e:
        logger.error(f"Error in background processing: {str(e)}")
        # Update case status to failed with retries
        await _safe_mark_case_failed(case_id, str(e))