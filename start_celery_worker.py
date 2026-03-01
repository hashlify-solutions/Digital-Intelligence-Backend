#!/usr/bin/env python
"""
Platform-aware Celery worker starter script.
This script automatically configures the Celery worker based on the platform.
"""

import os
import sys
import subprocess
import warnings

# Suppress pkg_resources deprecation warning
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

from config.settings import settings

def start_celery_worker():
    """Start Celery worker with threads pool on all platforms."""
    
    queues = [
        'data_processing',
        'data_ingestion', 
        'rag_processing',
        'analysis',
        'ufdr_processing',
        'ufdr_extraction',
        'face_detection',
        'object_detection',
        'video_frames_segmentation',
        'detector_processing',
        'detector_analysis',
        'audio_analysis',
        'video_analysis',
        'nsfw_detection',
        'image_description_generation',
        'video_description_generation',
    ]
    
    # Concurrency is set dynamically in celery_app.py based on CPU count
    # (or PARALLEL_MAX_WORKERS env var if set). No --concurrency flag here
    # so the Python config is the single source of truth.
    cmd = [
        'celery',
        '-A', 'celery_app',
        'worker',
        '--loglevel=info',
        f'--queues={",".join(queues)}',
        '--pool=threads',
        '--time-limit=7200',
        '--soft-time-limit=3600',
    ]
    
    platform = settings.platform.lower()
    if platform == 'mac':
        os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
    
    print(f"Starting Celery worker (threads pool, platform={platform})")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nShutting down Celery worker...")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Celery worker: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_celery_worker()
