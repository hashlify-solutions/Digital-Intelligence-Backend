import os
import multiprocessing
import warnings

# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# Safety environment variables — must be set BEFORE any library imports that
# touch CUDA/OpenMP/MKL.  These apply on ALL platforms (not just macOS) to
# prevent thread-oversubscription and CUDA-in-fork issues.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENCV_THREAD_COUNT", "0")
os.environ.setdefault("JOBLIB_MULTIPROCESSING", "0")

from config.settings import settings

if settings.platform.lower() == "mac":
    os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

from celery import Celery

# Create Celery instance
celery_app = Celery(
    settings.celery_app_name,
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks.celery_tasks"],
)

# Base configuration
base_config = {
    "task_serializer": "json",  # How tasks are serialized when sent to queue
    "accept_content": ["json"],  # Acceptable content types for task data
    "result_serializer": "json",  # How results are serialized when returned
    "timezone": "UTC",  # Timezone for task execution
    "enable_utc": True,  # Enable UTC timezone for task execution
    # Task execution settings
    "task_acks_late": True,  # Acknowledge tasks after they're executed
    "task_reject_on_worker_lost": True,  # Re-queue task if worker dies
    "worker_prefetch_multiplier": 1,  # Number of tasks to prefetch per worker
    "worker_max_tasks_per_child": 1000,  # Maximum number of tasks a worker can process before restarting
    # Task routing - you can define different queues for different tasks
    "task_routes": {
        # csv processing queue routes
        "tasks.celery_tasks.process_csv_upload": {
            "queue": "data_processing"
        },  # Route process_csv_upload to data_processing queue
        "tasks.celery_tasks.process_csv_upload_v1": {
            "queue": "data_processing"
        },  # Route process_csv_upload to data_processing queue
        "tasks.celery_tasks.ingest_csv_task": {
            "queue": "data_ingestion"
        },  # Route ingest_csv_task to data_ingestion queue
        "tasks.celery_tasks.ingest_rag_task": {
            "queue": "rag_processing"
        },  # Route ingest_rag_task to rag_processing queue
        "tasks.celery_tasks.analyze_documents_task": {
            "queue": "analysis"
        },  # Route analyze_documents_task to analysis queue
        # ufdr processing queue routes
        "tasks.celery_tasks.process_ufdr_upload": {
            "queue": "ufdr_processing"
        },  # Route process_ufdr_upload to ufdr_processing queue
        "tasks.celery_tasks.extract_ufdr_file_task": {
            "queue": "ufdr_extraction"
        },  # Route extract_ufdr_file_task to ufdr_extraction queue
        "tasks.celery_tasks.save_extracted_ufdr_data_task": {
            "queue": "data_ingestion"
        },  # Route save_extracted_ufdr_data_task to data_ingestion queue
        "tasks.celery_tasks.detect_faces_task": {
            "queue": "face_detection"
        },  # Route detect_faces_task to face_detection queue
        "tasks.celery_tasks.detect_objects_task": {
            "queue": "object_detection"
        },  # Route detect_objects_task to object_detection queue
        "tasks.celery_tasks.segment_video_in_frames_task": {
            "queue": "video_frames_segmentation"
        },  # Route segment_video_in_frames_task to video_frames_segmentation queue
        "tasks.celery_tasks.detect_video_faces_task": {
            "queue": "face_detection"
        },  # Route detect_video_faces_task to face_detection queue
        "tasks.celery_tasks.detect_video_objects_task": {
            "queue": "object_detection"
        },  # Route detect_video_objects_task to object_detection queue
        # detector processing queue routes
        "tasks.celery_tasks.process_detector_embedding_task": {
            "queue": "detector_processing"
        },  # Route process_detector_embedding_task to detector_processing queue
        "tasks.celery_tasks.process_detector_matches_task": {
            "queue": "detector_analysis"
        },  # Route process_detector_matches_task to detector_analysis queue
        "tasks.celery_tasks.analyze_detector_matches_task": {
            "queue": "detector_analysis"
        },  # Route analyze_detector_matches_task to detector_analysis queue
        "tasks.celery_tasks.new_detector_match_task": {
            "queue": "detector_analysis"
        },  # Route new_detector_match_task to detector_analysis queue
        "tasks.celery_tasks.analyze_audio_task": {
            "queue": "audio_analysis"
        },  # Route analyze_audio_task to analysis queue
        "tasks.celery_tasks.analyze_video_task": {
            "queue": "video_analysis"
        },  # Route analyze_video_task to video_analysis queue
        "tasks.celery_tasks.detect_nsfw_images_task": {
            "queue": "nsfw_detection"
        },  # Route detect_nsfw_images_task to nsfw_detection queue
        "tasks.celery_tasks.generate_image_description_llava_task": {
            "queue": "image_description_generation"
        },  # Route generate_image_description_llava_task to image_description_generation queue
        "tasks.celery_tasks.generate_video_frame_description_llava_task": {
            "queue": "image_description_generation"
        },  # Route generate_video_frame_description_llava_task to image_description_generation queue
        "tasks.celery_tasks.generate_video_description_task": {
            "queue": "video_description_generation"
        } # Route generate_video_description_task to video_description_generation queue
    },
    # Task time limits
    "task_soft_time_limit": 3600,  # 1 hour soft limit for tasks
    "task_time_limit": 7200,  # 2 hour hard limit for tasks
}

# Calculate worker concurrency WITHOUT touching CUDA/torch.
# settings.compute_config initializes CUDA which poisons forked subprocesses.
# We defer full hardware detection to worker startup (worker_ready signal).
_env_concurrency = settings.parallel_max_workers
if _env_concurrency:
    worker_concurrency = _env_concurrency
else:
    worker_concurrency = min(multiprocessing.cpu_count() * 2, 32)

# Use threads pool on ALL platforms.
# Reasons:
#  - GPU models are shared via ModelRegistry singleton (1 copy, not N copies)
#  - CUDA works in threads but NOT in forked processes
#  - Workload is I/O-bound (MongoDB, files) + GPU-bound (inference) — GIL is not
#    the bottleneck; PyTorch/CUDA releases GIL during GPU kernels
#  - Prevents "Cannot re-initialize CUDA in forked subprocess" errors on Linux
platform_config = {
    "worker_pool": "threads",
    "worker_concurrency": worker_concurrency,
}

import logging
_celery_logger = logging.getLogger(__name__)
_celery_logger.info(
    f"Celery worker config: pool=threads, concurrency={worker_concurrency}, "
    f"cpu={multiprocessing.cpu_count()}"
)

# Merge base config with platform-specific config
celery_app.conf.update({**base_config, **platform_config})


# Preload common AI models on worker startup to avoid first-request latency.
# worker_ready fires once the worker is fully booted (works for ALL pool types).
from celery.signals import worker_ready

@worker_ready.connect
def preload_models_on_worker_start(**kwargs):
    """Preload commonly used AI models when the Celery worker is ready."""
    try:
        from model_registry import preload_models
        _celery_logger.info("Preloading AI models on worker startup...")
        preload_models(["classifier", "toxic", "emotion", "embeddings"])
        _celery_logger.info("AI model preloading complete")
    except Exception as e:
        _celery_logger.warning(f"Model preloading failed (models will load on first use): {e}")
