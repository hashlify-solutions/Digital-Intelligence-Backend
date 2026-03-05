from celery import Task, chain, group
from celery_app import celery_app
from config.settings import settings
from ingester import CSVIngester
from analyzer import ArabicSocialAnalyzer
from analyzer_v1 import ArabicSocialAnalyzer as ArabicSocialAnalyzerV1
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from urllib.parse import quote_plus
from clients.ufdr_extracter.UfdrExtracter import UfdrExtracter
import os
from utils.celery_helpers import (
    save_json_data,
    save_media_files,
    process_face_detection_async,
    process_object_detection_async,
    process_video_face_detection_async,
    process_video_object_detection_async,
    segment_video_frames_async,
    process_detector_embedding_async,
    analyze_detector_matches_async,
    new_detector_match_async,
    analyze_audio_async,
    process_detector_matches_async,
    analyze_video_async,
    detect_nsfw_images_async,
    generate_image_description_llava_async,
    generate_video_frame_description_llava_async,
    generate_video_description_async,
    process_csv_upload_v1_helper,
)
from utils.async_helpers import run_async_in_thread, run_async_task, cleanup_thread_loop
from utils.helpers import _finalize_case_processing
from clients.video_to_audio.VideoToAudioConverter import VideoToAudioConverter
from rag import ArabicRagAnalyzer
from model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Platform-aware async runner
run_async = run_async_task


class DatabaseTask(Task):
    """Base task with database connection management"""

    _db = None
    _client = None
    _db_name = settings.mongo_database
    _mongo_uri = settings.mongo_connection_string

    @property
    def db(self):
        if self._db is None:
            # Create new motor client for Celery workers
            self._client = AsyncIOMotorClient(self._mongo_uri)
            self._db = self._client[self._db_name]
        return self._db

@celery_app.task(name="tasks.celery_tasks.finalize_case_processing_task")
def finalize_case_processing_task(case_id):
    """Record processing_completed_at and total_processing_time on the case document."""
    try:
        logger.info(f"Starting finalize_case_processing_task for case {case_id}")
        run_async(_finalize_case_processing(case_id))
        logger.info(f"Case processing finalized for case {case_id}")
        return {"status": "completed", "case_id": case_id}
    except Exception as e:
        logger.error(f"Error finalizing case processing for {case_id}: {e}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.ingest_csv_task"
)
def ingest_csv_task(self, file_path, case_name, case_id, models_profile):
    """Task to ingest CSV data into MongoDB"""
    try:
        logger.info(f"Starting CSV ingestion for case {case_id}")

        # Get the database from task instance
        db = self.db
        collection = db[f"{case_name}_{case_id}"]
        collection_all_cases = db["cases"]

        # Create ingester instance
        ingester = CSVIngester(
            collection, collection_all_cases, case_id, models_profile
        )

        # Run async function in sync context (platform-aware)
        run_async(ingester.ingest_csv_manually(file_path))

        logger.info(f"CSV ingestion completed for case {case_id}")
        return {"status": "completed", "case_id": case_id}

    except Exception as e:
        logger.error(f"Error in CSV ingestion task: {str(e)}")
        raise


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.ingest_rag_task"
)
def ingest_rag_task(self, case_name, case_id, models_profile):
    """Task to ingest RAG data into Qdrant"""
    try:
        logger.info(f"Starting RAG ingestion for case {case_id}")

        # Get the database from task instance
        db = self.db
        collection = db[f"{case_name}_{case_id}"]
        collection_all_cases = db["cases"]

        # Create ingester instance
        ingester = CSVIngester(
            collection, collection_all_cases, case_id, models_profile
        )

        # Run async function in sync context (platform-aware)
        run_async(ingester.ingest_rag_data())

        logger.info(f"RAG ingestion completed for case {case_id}")
        return {"status": "completed", "case_id": case_id}

    except Exception as e:
        logger.error(f"Error in RAG ingestion task: {str(e)}")
        raise


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.analyze_documents_task"
)
def analyze_documents_task(
    self,
    case_name,
    case_id,
    alert_id,
    topics,
    sentiments,
    interactions,
    entitiesClasses,
    models_profile,
):
    """Task to analyze documents with AI models"""
    try:
        logger.info(f"Starting document analysis for case {case_id}")

        # Get the database from task instance
        db = self.db
        collection = db[f"{case_name}_{case_id}"]
        collection_all_cases = db["cases"]

        # Create analyzer instance
        analyzer = ArabicSocialAnalyzer(
            collection,
            collection_all_cases,
            case_id,
            alert_id,
            topics,
            sentiments,
            interactions,
            entitiesClasses,
            models_profile,
        )

        # Run async function in sync context (platform-aware)
        run_async(analyzer.process_documents())

        logger.info(f"Document analysis completed for case {case_id}")
        return {"status": "completed", "case_id": case_id}

    except Exception as e:
        logger.error(f"Error in document analysis task: {str(e)}")
        raise


@celery_app.task(name="tasks.celery_tasks.process_csv_upload")
def process_csv_upload(
    file_path,
    case_name,
    case_id,
    alert_id,
    topics,
    sentiments,
    interactions,
    entitiesClasses,
    is_rag,
    models_profile,
):
    """Main task that orchestrates the entire CSV upload process"""
    try:
        # Create a chain of tasks
        # First: Always ingest CSV
        workflow = ingest_csv_task.si(file_path, case_name, case_id, models_profile)

        # Second: If RAG is enabled, ingest to Qdrant after CSV ingestion
        if is_rag:
            workflow = workflow | ingest_rag_task.si(case_name, case_id, models_profile)

        # Third: Always analyze documents after ingestion
        workflow = workflow | analyze_documents_task.si(
            case_name,
            case_id,
            alert_id,
            topics,
            sentiments,
            interactions,
            entitiesClasses,
            models_profile,
        )

        # Fourth: Record total processing time
        workflow = workflow | finalize_case_processing_task.si(case_id)

        # Execute the workflow
        result = workflow.apply_async()

        return {"status": "started", "case_id": case_id, "workflow_id": result.id}

    except Exception as e:
        logger.error(f"Error in process_csv_upload task: {str(e)}")
        raise


@celery_app.task(name="tasks.celery_tasks.process_csv_upload_v1")
def process_csv_upload_v1(
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
    is_llama_validation_enabled
):
    """Main task that orchestrates the entire v1 CSV upload process"""
    try:
        run_async(
            process_csv_upload_v1_helper(
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
                is_llama_validation_enabled
            )
        )
        return {"status": "completed", "case_id": case_id}
    except Exception as e:
        logger.error(f"Error in process_csv_upload_v1 task: {str(e)}")
        raise


@celery_app.task(name="tasks.celery_tasks.extract_ufdr_file_task")
def extract_ufdr_file_task(input_ufdr_path, output_ufdr_path):
    """Task to extract UFDR file to disk.
    
    Skips extraction if the output path already exists (i.e. Phase A already
    extracted the UFDR during ingest_ufdr_file), avoiding duplicate work.
    """
    try:
        # Check if already extracted by Phase A (ingest_ufdr_file)
        # if os.path.exists(output_ufdr_path) and os.listdir(output_ufdr_path):
        #     logger.info(
        #         f"UFDR already extracted at {output_ufdr_path}, skipping re-extraction"
        #     )
        #     # Clean up the original input file if it still exists
        #     if os.path.exists(input_ufdr_path):
        #         os.remove(input_ufdr_path)
        #     return {"status": "completed", "output_ufdr_path": output_ufdr_path, "skipped": True}
        
        ufdr_extracter = UfdrExtracter()
        ufdr_extracter.extract_ufdr(input_ufdr_path, output_ufdr_path)
        # removing the input UFDR file after extraction
        if os.path.exists(input_ufdr_path):
            os.remove(input_ufdr_path)
        return {"status": "completed", "output_ufdr_path": output_ufdr_path}
    except Exception as e:
        logger.error(f"Error in extract_ufdr_file_task task: {str(e)}")
        raise


@celery_app.task(
    base=DatabaseTask,
    bind=True,
    name="tasks.celery_tasks.save_extracted_ufdr_data_task",
)
def save_extracted_ufdr_data_task(
    self, output_ufdr_path, ufdr_file_id, case_name, case_id
):
    """Task to save extracted UFDR data to shared MongoDB collections"""
    try:
        logger.info(f"Starting UFDR data persistence for ufdr_file_id: {ufdr_file_id}")

        # Get the database from task instance
        db = self.db
        ufdr_files_collection = db["ufdr_files"]

        # Define JSON file mappings to shared collections
        json_files = {
            "Call.json": db["ufdr_calls"],
            "Chat.json": db["ufdr_chats"],
            "Email.json": db["ufdr_emails"],
            "Location.json": db["ufdr_locations"],
            "Note.json": db["ufdr_notes"],
            "SearchedItem.json": db["ufdr_searched_items"],
            "UserAccount.json": db["ufdr_user_accounts"],
        }

        # Process each JSON file
        for json_filename, collection in json_files.items():
            json_path = os.path.join(output_ufdr_path, json_filename)
            if os.path.exists(json_path):
                logger.info(
                    f"Processing {json_filename} into shared collection {collection.name}"
                )
                # Read and process JSON file
                document_count = run_async(
                    save_json_data(json_path, collection, ufdr_file_id, case_id)
                )
                logger.info(
                    f"Completed processing {json_filename} - {document_count} documents"
                )

        # Process media files using shared collections
        media_dir = os.path.join(output_ufdr_path, "media")
        if os.path.exists(media_dir):
            media_collections = {
                "audio": db["ufdr_audio"],
                "photo": db["ufdr_photos"],
                "video": db["ufdr_videos"],
            }

            for media_type, collection in media_collections.items():
                media_type_dir = os.path.join(media_dir, media_type)
                if os.path.exists(media_type_dir):
                    logger.info(
                        f"Processing {media_type} media files into shared collection {collection.name}"
                    )

                    # Process media files
                    file_count = run_async(
                        save_media_files(
                            media_type_dir,
                            collection,
                            ufdr_file_id,
                            media_type,
                            case_id,
                        )
                    )

                    logger.info(
                        f"Completed processing {media_type} media files ({file_count} files)"
                    )

        logger.info(f"UFDR data persistence completed for ufdr_file_id: {ufdr_file_id}")

        return {
            "status": "completed",
            "ufdr_file_id": ufdr_file_id,
            "case_id": case_id,
        }

    except Exception as e:
        logger.error(f"Error in save_extracted_ufdr_data_task: {str(e)}")
        raise


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.detect_faces_task"
)
def detect_faces_task(self, output_ufdr_path, ufdr_file_id, case_id):
    """Task to detect faces from extracted images"""
    try:
        logger.info(f"Starting face detection for case {case_id}")

        # Get the face detector client (cached via ModelRegistry)
        face_detector_client = ModelRegistry.get_model("face_detector", model_name="dnn")

        # Get the database from task instance
        db = self.db
        ufdr_photos_collection = db["ufdr_photos"]
        ufdr_photo_detected_faces_collection = db["ufdr_photo_detected_faces"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            process_face_detection_async(
                ufdr_photos_collection,
                ufdr_photo_detected_faces_collection,
                face_detector_client,
                output_ufdr_path,
                ufdr_file_id,
                case_id,
            )
        )

        logger.info(f"Face detection completed for case {case_id}")
        return result

    except Exception as e:
        logger.error(f"Error in detect_faces_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.detect_objects_task"
)
def detect_objects_task(self, output_ufdr_path, ufdr_file_id, case_id):
    """Task to detect objects from extracted images"""
    try:
        logger.info(f"Starting object detection for case {case_id}")

        # Get the object detector client (cached via ModelRegistry)
        object_detector_client = ModelRegistry.get_model("object_detector", model_name="yolo")

        # Get the database from task instance
        db = self.db
        ufdr_photos_collection = db["ufdr_photos"]
        ufdr_photo_detected_objects_collection = db["ufdr_photo_detected_objects"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            process_object_detection_async(
                ufdr_photos_collection,
                ufdr_photo_detected_objects_collection,
                object_detector_client,
                output_ufdr_path,
                ufdr_file_id,
                case_id,
            )
        )

        logger.info(f"Object detection completed for case {case_id}")
        return result

    except Exception as e:
        logger.error(f"Error in detect_objects_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.segment_video_in_frames_task"
)
def segment_video_in_frames_task(
    self, output_ufdr_path, ufdr_file_id, case_id, frame_interval=30
):
    """Task to segment video frames and save them as screenshots"""
    try:
        logger.info(f"Starting video frame segmentation for case {case_id}")

        # Get the database from task instance
        db = self.db
        ufdr_videos_collection = db["ufdr_videos"]
        ufdr_video_screenshots_collection = db["ufdr_video_screenshots"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            segment_video_frames_async(
                ufdr_videos_collection,
                ufdr_video_screenshots_collection,
                output_ufdr_path,
                ufdr_file_id,
                case_id,
                frame_interval,
            )
        )

        logger.info(f"Video frame segmentation completed for case {case_id}")
        return result

    except Exception as e:
        logger.error(f"Error in segment_video_in_frames_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.detect_video_faces_task"
)
def detect_video_faces_task(self, output_ufdr_path, ufdr_file_id, case_id):
    """Task to detect faces from video screenshots"""
    try:
        logger.info(
            f"Starting video face detection from screenshots for case {case_id}"
        )

        # Get the face detector client (cached via ModelRegistry)
        face_detector_client = ModelRegistry.get_model("face_detector", model_name="dnn")

        # Get the database from task instance
        db = self.db
        ufdr_video_screenshots_collection = db["ufdr_video_screenshots"]
        ufdr_video_detected_faces_collection = db["ufdr_video_detected_faces"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            process_video_face_detection_async(
                ufdr_video_screenshots_collection,
                ufdr_video_detected_faces_collection,
                face_detector_client,
                output_ufdr_path,
                ufdr_file_id,
                case_id,
            )
        )

        logger.info(
            f"Video face detection from screenshots completed for case {case_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in detect_video_faces_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.detect_video_objects_task"
)
def detect_video_objects_task(self, output_ufdr_path, ufdr_file_id, case_id):
    """Task to detect objects from video screenshots"""
    try:
        logger.info(
            f"Starting video object detection from screenshots for case {case_id}"
        )

        # Get the object detector client (cached via ModelRegistry)
        object_detector_client = ModelRegistry.get_model("object_detector", model_name="yolo")

        # Get the database from task instance
        db = self.db
        ufdr_video_screenshots_collection = db["ufdr_video_screenshots"]
        ufdr_video_detected_objects_collection = db["ufdr_video_detected_objects"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            process_video_object_detection_async(
                ufdr_video_screenshots_collection,
                ufdr_video_detected_objects_collection,
                object_detector_client,
                output_ufdr_path,
                ufdr_file_id,
                case_id,
            )
        )
        logger.info(
            f"Video object detection from screenshots completed for case {case_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in detect_video_objects_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask,
    bind=True,
    name="tasks.celery_tasks.process_detector_matches_task",
)
def process_detector_matches_task(self, case_id, ufdr_file_id):
    """Task to match a single new detection against existing detectors"""
    try:
        logger.info(
            f"Starting detection matching for the case {case_id} for ufdr file {ufdr_file_id}"
        )

        # Get the database from task instance
        db = self.db
        detectors_collection = db["detectors"]
        detector_matches_collection = db["detector_matches"]
        detector_settings_collection = db["detector_settings"]
        ufdr_photo_detected_faces_collection = db["ufdr_photo_detected_faces"]
        ufdr_video_detected_faces_collection = db["ufdr_video_detected_faces"]
        ufdr_photo_detected_objects_collection = db["ufdr_photo_detected_objects"]
        ufdr_video_detected_objects_collection = db["ufdr_video_detected_objects"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            process_detector_matches_async(
                case_id,
                ufdr_file_id,
                detectors_collection,
                detector_matches_collection,
                detector_settings_collection,
                ufdr_photo_detected_faces_collection,
                ufdr_video_detected_faces_collection,
                ufdr_photo_detected_objects_collection,
                ufdr_video_detected_objects_collection,
            )
        )

        logger.info(
            f"Detection matching completed for the case {case_id} for ufdr file {ufdr_file_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in process_detector_matches_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(name="tasks.celery_tasks.process_ufdr_upload")
def process_ufdr_upload(
    input_ufdr_path,
    output_ufdr_path,
    case_name,
    case_id,
    ufdr_file_id,
    alert_id=None,
    topics=None,
    sentiments=None,
    interactions=None,
    entitiesClasses=None,
    model_profile=None,
    note_classifications=None,
    browsing_history_classifications=None,
    is_llama_validation_enabled=False,
):
    """
    Main task that orchestrates the entire UFDR upload process.
    
    Aggressively parallelized workflow:
    After M2 (save to MongoDB), ALL independent tasks start immediately:
    - Image face+object detection (need photos in DB)
    - Video segmentation (need video metadata in DB)
    - Audio analysis (need audio metadata in DB)
    - NSFW detection (need photos in DB)
    - Image description via LLaVA (need photos in DB)
    
    After video segmentation:
    - Video face+object detection
    - Video frame description via LLaVA
    
    After all detections: detector matching
    After video frame descriptions: aggregate video description
    """
    try:
        logger.info(f"Starting AGGRESSIVELY PARALLELIZED UFDR workflow for case {case_id}")
        
        # Common task arguments for analyzer tasks
        analyzer_args = (
            ufdr_file_id, case_name, case_id, alert_id,
            topics, sentiments, interactions, entitiesClasses,
            model_profile, note_classifications,
            browsing_history_classifications, is_llama_validation_enabled,
        )
        
        llava_args = (
            ufdr_file_id, case_name, case_id, model_profile, alert_id,
            topics, sentiments, interactions, entitiesClasses,
            note_classifications, browsing_history_classifications,
            is_llama_validation_enabled,
        )
        
        video_desc_args = (
            ufdr_file_id, case_id, case_name, model_profile, alert_id,
            topics, sentiments, interactions, entitiesClasses,
            note_classifications, browsing_history_classifications,
            is_llama_validation_enabled,
        )
        
        # ---- Sequential prerequisites ----
        # Phase 1: Extract UFDR (skips if already extracted by Phase A)
        # Phase 2: Save metadata to MongoDB
        
        # ---- Parallel Group 1: All tasks that only need M2 done ----
        # These all start immediately after save_extracted_ufdr_data_task
        
        # Detection branch: image detection -> video segmentation -> video detection -> matching
        detection_workflow = chain(
            # Image face + object detection (parallel)
            group(
                detect_faces_task.si(output_ufdr_path, ufdr_file_id, case_id),
                detect_objects_task.si(output_ufdr_path, ufdr_file_id, case_id),
            ),
            # Video frame segmentation
            segment_video_in_frames_task.si(output_ufdr_path, ufdr_file_id, case_id),
            # Video face + object detection (parallel)
            group(
                detect_video_faces_task.si(output_ufdr_path, ufdr_file_id, case_id),
                detect_video_objects_task.si(output_ufdr_path, ufdr_file_id, case_id),
            ),
            # Detector matching
            process_detector_matches_task.si(case_id, ufdr_file_id),
        )
        
        # LLaVA description branch: image desc -> video frame desc -> video desc aggregation
        llava_workflow = chain(
            generate_image_description_llava_task.si(*llava_args),
            # Video frame desc needs segmented frames from detection branch,
            # but since we can't cross-reference branches in a simple chord,
            # we place it after image desc for model cache reuse
        )
        
        # Video description sub-workflow (needs video frames segmented)
        # This runs after the detection branch via a separate chain after the main parallel group
        
        # Build the optimized workflow
        workflow = chain(
            # Phase 1: Extract UFDR (skips if already extracted)
            extract_ufdr_file_task.si(input_ufdr_path, output_ufdr_path),
            
            # Phase 2: Save to MongoDB
            save_extracted_ufdr_data_task.si(
                output_ufdr_path, ufdr_file_id, case_name, case_id
            ),
            
            # Phase 3: AGGRESSIVELY PARALLEL - all independent tasks at once
            group(
                # Branch A: Full detection pipeline
                detection_workflow,
                
                # Branch B: Audio analysis (independent of detections)
                analyze_audio_task.si(*analyzer_args),
                
                # Branch C: Video analysis (independent of detections)
                analyze_video_task.si(*analyzer_args),
                
                # Branch D: NSFW detection (independent)
                detect_nsfw_images_task.si(ufdr_file_id),
                
                # Branch E: LLaVA image descriptions (independent)
                llava_workflow,
            ),
            
            # Phase 4: Video frame + video description (needs segmented frames from Branch A)
            generate_video_frame_description_llava_task.si(*llava_args),
            
            # Phase 5: Aggregate video descriptions
            generate_video_description_task.si(*video_desc_args),

            # Phase 6: Record total processing time
            finalize_case_processing_task.si(case_id),
        )

        # Execute the workflow
        result = workflow.apply_async()
        
        logger.info(
            f"AGGRESSIVELY PARALLELIZED workflow started for case {case_id}, "
            f"workflow_id={result.id}"
        )

        return {"status": "started", "case_id": case_id, "workflow_id": result.id}

    except Exception as e:
        logger.error(f"Error in process_ufdr_upload task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask,
    bind=True,
    name="tasks.celery_tasks.process_detector_embedding_task",
)
def process_detector_embedding_task(self, detector_id, case_id):
    """Task to generate embedding for a detector image"""
    try:
        logger.info(
            f"Starting detector embedding processing for detector {detector_id}"
        )

        # Get the database from task instance
        db = self.db
        detectors_collection = db["detectors"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            process_detector_embedding_async(detector_id, case_id, detectors_collection)
        )

        logger.info(
            f"Detector embedding processing completed for detector {detector_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in process_detector_embedding_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask,
    bind=True,
    name="tasks.celery_tasks.analyze_detector_matches_task",
)
def analyze_detector_matches_task(self, case_id, detector_type=None):
    """Task to analyze all detected items against case detectors"""
    try:
        logger.info(f"Starting detector match analysis for case {case_id}")

        # Get the database from task instance
        db = self.db
        detectors_collection = db["detectors"]
        detector_matches_collection = db["detector_matches"]
        detector_settings_collection = db["detector_settings"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            analyze_detector_matches_async(
                case_id,
                detector_type,
                detectors_collection,
                detector_matches_collection,
                detector_settings_collection,
            )
        )
        logger.info(f"Detector match analysis completed for case {case_id}")
        return result

    except Exception as e:
        logger.error(f"Error in analyze_detector_matches_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.new_detector_match_task"
)
def new_detector_match_task(self, case_id, detector_id):
    """Task to match a single new detection against existing detectors"""
    try:
        logger.info(
            f"Starting detection matching for the detector {detector_id} in case {case_id}"
        )

        # Get the database from task instance
        db = self.db
        detectors_collection = db["detectors"]
        detector_matches_collection = db["detector_matches"]
        detector_settings_collection = db["detector_settings"]
        ufdr_photo_detected_faces_collection = db["ufdr_photo_detected_faces"]
        ufdr_video_detected_faces_collection = db["ufdr_video_detected_faces"]
        ufdr_photo_detected_objects_collection = db["ufdr_photo_detected_objects"]
        ufdr_video_detected_objects_collection = db["ufdr_video_detected_objects"]

        # Run async function in sync context (platform-aware)
        result = run_async(
            new_detector_match_async(
                case_id,
                detector_id,
                detectors_collection,
                detector_matches_collection,
                detector_settings_collection,
                ufdr_photo_detected_faces_collection,
                ufdr_video_detected_faces_collection,
                ufdr_photo_detected_objects_collection,
                ufdr_video_detected_objects_collection,
            )
        )
        logger.info(
            f"Single detection matching completed for the detector {detector_id} in case {case_id}"
        )
        return result

    except Exception as e:
        logger.error(f"Error in match_new_detection_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.analyze_audio_task"
)
def analyze_audio_task(
    self,
    ufdr_file_id,
    case_name,
    case_id,
    alert_id,
    topics,
    sentiments,
    interactions,
    entitiesClasses,
    model_profile,
    note_classifications,
    browsing_history_classifications,
    is_llama_validation_enabled,
):
    """Task to match a single new detection against existing detectors"""
    try:
        logger.info(f"Starting audio transcription for the ufdr_file: {ufdr_file_id}")
        # Get the database from task instance
        db = self.db
        case_collection = db["cases"]
        single_case_collection = db[f"{case_name}_{case_id}"]
        ufdr_audio_collection = db["ufdr_audio"]
        transcriber_client = ModelRegistry.get_model("transcriber")
        # analyzer_client = ArabicSocialAnalyzer(
        #     case_id=case_id,
        #     alert_id=alert_id,
        #     topics=topics,
        #     sentiments=sentiments,
        #     interactions=interactions,
        #     entitiesClasses=entitiesClasses,
        #     model_profile=model_profile,
        # )
        analyzer_client_v1 = ArabicSocialAnalyzerV1(
            mongo_collection_case=None,
            mongo_collection__all_cases=None,
            case_id=case_id,
            alert_id=alert_id,
            topics=topics,
            sentiments=sentiments,
            interactions=interactions,
            entitiesClasses=entitiesClasses,
            model_profile=model_profile,
            use_parallel_processing=True,
            note_classifications=note_classifications,
            browsing_history_classifications=browsing_history_classifications,
            is_llama_validation_enabled=is_llama_validation_enabled,
        )
        rag_analyzer_client = ArabicRagAnalyzer(
            case_collection, single_case_collection, case_id, model_profile
        )
        embedding_obj = model_profile.get("embeddings", {})
        vector_size = embedding_obj.get("embedding_size", 512)
        # Run async function in sync context (platform-aware)
        result = run_async(
            analyze_audio_async(
                case_id,
                ufdr_file_id,
                ufdr_audio_collection,
                transcriber_client,
                analyzer_client_v1,
                rag_analyzer_client,
                vector_size,
                is_llama_validation_enabled,
            )
        )
        logger.info(f"Completed audio transcription for the ufdr_file: {ufdr_file_id}")
        return result

    except Exception as e:
        logger.error(f"Error in analyze_audio_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.analyze_video_task"
)
def analyze_video_task(
    self,
    ufdr_file_id,
    case_name,
    case_id,
    alert_id,
    topics,
    sentiments,
    interactions,
    entitiesClasses,
    model_profile,
    note_classifications,
    browsing_history_classifications,
    is_llama_validation_enabled,
):
    """Task to match a single new detection against existing detectors"""
    try:
        logger.info(f"Starting video analysis for the ufdr_file: {ufdr_file_id}")
        # Get the database from task instance
        db = self.db
        ufdr_video_collection = db["ufdr_videos"]
        case_collection = db["cases"]
        single_case_collection = db[f"{case_name}_{case_id}"]
        embedding_obj = model_profile.get("embeddings", {})
        vector_size = embedding_obj.get("embedding_size", 512)
        video_to_audio_converter_client = VideoToAudioConverter()
        transcriber_client = ModelRegistry.get_model("transcriber")
        # analyzer_client = ArabicSocialAnalyzer(
        #     case_id=case_id,
        #     alert_id=alert_id,
        #     topics=topics,
        #     sentiments=sentiments,
        #     interactions=interactions,
        #     entitiesClasses=entitiesClasses,
        #     model_profile=model_profile,
        # )
        analyzer_client_v1 = ArabicSocialAnalyzerV1(
            mongo_collection_case=None,
            mongo_collection__all_cases=None,
            case_id=case_id,
            alert_id=alert_id,
            topics=topics,
            sentiments=sentiments,
            interactions=interactions,
            entitiesClasses=entitiesClasses,
            model_profile=model_profile,
            use_parallel_processing=True,
            note_classifications=note_classifications,
            browsing_history_classifications=browsing_history_classifications,
            is_llama_validation_enabled=is_llama_validation_enabled,
        )
        rag_analyzer_client = ArabicRagAnalyzer(
            case_collection, single_case_collection, case_id, model_profile
        )

        # Normalize the upload directory path to remove any duplicate slashes
        base_dir = os.path.normpath(settings.upload_dir)
        output_audio_dir = os.path.normpath(
            os.path.join(
                base_dir,
                f"{case_name}_{case_id}",
                "ufdr",
                f"{ufdr_file_id}",
                "video_audio",
            )
        )

        # Ensure the directory exists
        os.makedirs(output_audio_dir, exist_ok=True)

        # Run async function in sync context (platform-aware)
        result = run_async(
            analyze_video_async(
                case_id,
                ufdr_file_id,
                ufdr_video_collection,
                output_audio_dir,
                transcriber_client,
                analyzer_client_v1,
                rag_analyzer_client,
                video_to_audio_converter_client,
                vector_size,
                is_llama_validation_enabled,
            )
        )
        logger.info(f"Completed video analysis for the ufdr_file: {ufdr_file_id}")
        return result

    except Exception as e:
        logger.error(f"Error in analyze_video_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask, bind=True, name="tasks.celery_tasks.detect_nsfw_images_task"
)
def detect_nsfw_images_task(self, ufdr_file_id):
    """Task to detect NSFW images from ufdr_photos and ufdr_video_screenshots"""
    try:
        logger.info(f"Starting NSFW image detection for the ufdr_file: {ufdr_file_id}")
        # Get the database from task instance
        db = self.db
        ufdr_photos_collection = db["ufdr_photos"]
        ufdr_video_collection = db["ufdr_videos"]
        ufdr_video_screenshots_collection = db["ufdr_video_screenshots"]
        nsfw_detector_client = ModelRegistry.get_model("nsfw_detector")
        # Run async function in sync context (platform-aware)
        result = run_async(
            detect_nsfw_images_async(
                ufdr_file_id,
                ufdr_photos_collection,
                ufdr_video_collection,
                ufdr_video_screenshots_collection,
                nsfw_detector_client,
            )
        )
        logger.info(f"Completed NSFW image detection for the ufdr_file: {ufdr_file_id}")
        return result
    except Exception as e:
        logger.error(f"Error in detect_nsfw_images_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask,
    bind=True,
    name="tasks.celery_tasks.generate_image_description_llava_task",
)
def generate_image_description_llava_task(
    self,
    ufdr_file_id,
    case_name,
    case_id,
    model_profile,
    alert_id,
    topics,
    sentiments,
    interactions,
    entitiesClasses,
    note_classifications,
    browsing_history_classifications,
    is_llama_validation_enabled,
):
    """Task to generate image description with Llava"""
    try:
        logger.info(
            f"Starting image description generation with Llava for the ufdr_file: {ufdr_file_id}"
        )
        db = self.db
        ufdr_photos_collection = db["ufdr_photos"]
        case_collection = db["cases"]
        single_case_collection = db[f"{case_name}_{case_id}"]
        llava_client = ModelRegistry.get_model("llava")
        llama_client = ModelRegistry.get_model("llama")
        rag_analyzer_client = ArabicRagAnalyzer(
            case_collection, single_case_collection, case_id, model_profile
        )
        embedding_obj = model_profile.get("embeddings", {})
        vector_size = embedding_obj.get("embedding_size", 512)

        # analyzer_client = ArabicSocialAnalyzer(
        #     case_id=case_id,
        #     alert_id=alert_id,
        #     topics=topics,
        #     sentiments=sentiments,
        #     interactions=interactions,
        #     entitiesClasses=entitiesClasses,
        #     model_profile=model_profile,
        # )
        analyzer_client_v1 = ArabicSocialAnalyzerV1(
            mongo_collection_case=None,
            mongo_collection__all_cases=None,
            case_id=case_id,
            alert_id=alert_id,
            topics=topics,
            sentiments=sentiments,
            interactions=interactions,
            entitiesClasses=entitiesClasses,
            model_profile=model_profile,
            use_parallel_processing=True,
            note_classifications=note_classifications,
            browsing_history_classifications=browsing_history_classifications, 
            is_llama_validation_enabled=is_llama_validation_enabled,
        )

        result = run_async(
            generate_image_description_llava_async(
                ufdr_file_id,
                case_id,
                ufdr_photos_collection,
                case_collection,
                llava_client,
                llama_client,
                rag_analyzer_client,
                analyzer_client_v1,
                vector_size,
                is_llama_validation_enabled,
            )
        )
        logger.info(
            f"Completed image description generation with Llava for the ufdr_file: {ufdr_file_id}"
        )
        return result
    except Exception as e:
        logger.error(f"Error in generate_image_description_llava_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask,
    bind=True,
    name="tasks.celery_tasks.generate_video_frame_description_llava_task",
)
def generate_video_frame_description_llava_task(
    self,
    ufdr_file_id,
    case_name,
    case_id,
    model_profile,
    alert_id,
    topics,
    sentiments,
    interactions,
    entitiesClasses,
    note_classifications,
    browsing_history_classifications,
    is_llama_validation_enabled,
):
    """Task to generate video frame description with Llava"""
    try:
        logger.info(
            f"Starting video frame description generation with Llava for the ufdr_file: {ufdr_file_id}"
        )
        db = self.db
        ufdr_video_screenshots_collection = db["ufdr_video_screenshots"]
        case_collection = db["cases"]
        single_case_collection = db[f"{case_name}_{case_id}"]
        llava_client = ModelRegistry.get_model("llava")
        llama_client = ModelRegistry.get_model("llama")
        embedding_obj = model_profile.get("embeddings", {})
        vector_size = embedding_obj.get("embedding_size", 512)
        # analyzer_client = ArabicSocialAnalyzer(
        #     case_id=case_id,
        #     alert_id=alert_id,
        #     topics=topics,
        #     sentiments=sentiments,
        #     interactions=interactions,
        #     entitiesClasses=entitiesClasses,
        #     model_profile=model_profile,
        # )
        analyzer_client_v1 = ArabicSocialAnalyzerV1(
            mongo_collection_case=None,
            mongo_collection__all_cases=None,
            case_id=case_id,
            alert_id=alert_id,
            topics=topics,
            sentiments=sentiments,
            interactions=interactions,
            entitiesClasses=entitiesClasses,
            model_profile=model_profile,
            use_parallel_processing=True,
            note_classifications=note_classifications,
            browsing_history_classifications=browsing_history_classifications,
            is_llama_validation_enabled=is_llama_validation_enabled,
        )
        rag_analyzer_client = ArabicRagAnalyzer(
            case_collection, single_case_collection, case_id, model_profile
        )

        result = run_async(
            generate_video_frame_description_llava_async(
                ufdr_file_id,
                case_id,
                ufdr_video_screenshots_collection,
                case_collection,
                llava_client,
                llama_client,
                analyzer_client_v1,
                rag_analyzer_client,
                vector_size,
                is_llama_validation_enabled,
            )
        )
        logger.info(
            f"Completed video frame description generation with Llava for the ufdr_file: {ufdr_file_id}"
        )
        return result
    except Exception as e:
        logger.error(f"Error in generate_video_frame_description_llava_task: {str(e)}")
        return {"status": "error", "error": str(e)}


@celery_app.task(
    base=DatabaseTask,
    bind=True,
    name="tasks.celery_tasks.generate_video_description_task",
)
def generate_video_description_task(
    self,
    ufdr_file_id,
    case_id,
    case_name,
    model_profile,
    alert_id,
    topics,
    sentiments,
    interactions,
    entitiesClasses,
    note_classifications,
    browsing_history_classifications,
    is_llama_validation_enabled,
):
    """Task to generate video description with Llava"""
    try:
        logger.info(
            f"Starting video description generation with Llava for the ufdr_file: {ufdr_file_id}"
        )

        db = self.db
        case_collection = db["cases"]
        single_case_collection = db[f"{case_name}_{case_id}"]
        ufdr_video_collection = db["ufdr_videos"]
        ufdr_video_screenshots_collection = db["ufdr_video_screenshots"]

        embedding_obj = model_profile.get("embeddings", {})
        vector_size = embedding_obj.get("embedding_size", 512)
        llama_client = ModelRegistry.get_model("llama")
        # analyzer_client = ArabicSocialAnalyzer(
        #     case_id=case_id,
        #     alert_id=alert_id,
        #     topics=topics,
        #     sentiments=sentiments,
        #     interactions=interactions,
        #     entitiesClasses=entitiesClasses,
        #     model_profile=model_profile,
        # )
        analyzer_client_v1 = ArabicSocialAnalyzerV1(
            mongo_collection_case=None,
            mongo_collection__all_cases=None,
            case_id=case_id,
            alert_id=alert_id,
            topics=topics,
            sentiments=sentiments,
            interactions=interactions,
            entitiesClasses=entitiesClasses,
            model_profile=model_profile,
            use_parallel_processing=True,
            note_classifications=note_classifications,
            browsing_history_classifications=browsing_history_classifications, 
            is_llama_validation_enabled=is_llama_validation_enabled,
        )
        rag_analyzer_client = ArabicRagAnalyzer(
            case_collection, single_case_collection, case_id, model_profile
        )

        result = run_async(
            generate_video_description_async(
                ufdr_file_id,
                case_id,
                ufdr_video_collection,
                ufdr_video_screenshots_collection,
                case_collection,
                llama_client,
                analyzer_client_v1,
                rag_analyzer_client,
                vector_size,
                is_llama_validation_enabled,
            )
        )
        logger.info(
            f"Completed video description generation with Llava for the ufdr_file: {ufdr_file_id}"
        )
        return result
    except Exception as e:
        logger.error(f"Error in generate_video_description_task: {str(e)}")
        return {"status": "error", "error": str(e)}
