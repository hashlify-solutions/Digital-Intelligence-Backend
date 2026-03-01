#!/bin/bash

# Start Celery beat for scheduled tasks (if needed)
echo "Starting Celery beat..."

celery -A celery_app beat --loglevel=info