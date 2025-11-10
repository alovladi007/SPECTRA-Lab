"""Celery application for background job processing.

Handles async execution of Ion Implantation and RTP runs with progress tracking,
cancellation support, and retry/backoff logic.
"""

import os
from celery import Celery
from kombu import Exchange, Queue

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "process_control",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "app.tasks.ion_tasks",
        "app.tasks.rtp_tasks",
    ]
)

# Task routing configuration
celery_app.conf.update(
    # Routing
    task_routes={
        "app.tasks.ion_tasks.*": {"queue": "ion"},
        "app.tasks.rtp_tasks.*": {"queue": "rtp"},
    },

    # Queues
    task_queues=(
        Queue("ion", Exchange("ion"), routing_key="ion"),
        Queue("rtp", Exchange("rtp"), routing_key="rtp"),
        Queue("default", Exchange("default"), routing_key="default"),
    ),

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Task execution
    task_track_started=True,
    task_send_sent_event=True,
    task_acks_late=True,  # Acknowledge after task completion
    worker_prefetch_multiplier=1,  # One task at a time per worker

    # Retries
    task_annotations={
        "*": {
            "max_retries": 3,
            "default_retry_delay": 60,  # 1 minute
            "retry_backoff": True,
            "retry_backoff_max": 600,  # 10 minutes max
            "retry_jitter": True,
        }
    },

    # Results
    result_expires=3600 * 24 * 7,  # 7 days
    result_backend_transport_options={
        "master_name": "mymaster"
    },

    # Time limits
    task_soft_time_limit=3600 * 2,  # 2 hours soft limit
    task_time_limit=3600 * 3,  # 3 hours hard limit

    # Timezone
    timezone="UTC",
    enable_utc=True,
)


if __name__ == "__main__":
    celery_app.start()
