#!/bin/bash

# Load environment variables
source .env 2>/dev/null || true

# Clear Redis cache before starting worker
echo "Clearing Redis cache..."
if docker compose exec redis redis-cli FLUSHALL 2>/dev/null; then
    echo "Redis cache cleared (via Docker)."
else
    redis-cli FLUSHALL 2>/dev/null && echo "Redis cache cleared (native)." || echo "Warning: Could not flush Redis cache."
fi

# Start Celery worker with threads pool on all platforms.
# Threads pool ensures:
#  - GPU models are loaded ONCE and shared across all threads (via ModelRegistry)
#  - No "Cannot re-initialize CUDA in forked subprocess" errors
#  - Works identically on macOS, Linux, and Windows
echo "Starting Celery worker..."
echo "Platform: ${PLATFORM}"

# Define the queues
QUEUES="data_processing,data_ingestion,rag_processing,analysis,ufdr_processing,ufdr_extraction,face_detection,object_detection,video_frames_segmentation,detector_processing,detector_analysis,audio_analysis,video_analysis,nsfw_detection,image_description_generation,video_description_generation"

# Concurrency is set dynamically in celery_app.py based on CPU count
# (or PARALLEL_MAX_WORKERS env var if set). No --concurrency flag here
# so the Python config is the single source of truth.
celery -A celery_app worker \
    --loglevel=info \
    --queues=$QUEUES \
    --pool=threads \
    --time-limit=7200 \
    --soft-time-limit=3600