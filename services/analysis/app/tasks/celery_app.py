"""
Celery Application Configuration

Configures Celery for asynchronous CVD run orchestration and task processing.
"""

from celery import Celery
from kombu import Queue
import os

# Celery configuration
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

# Create Celery app
celery_app = Celery(
    "spectra_analysis",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

# Celery configuration
celery_app.conf.update(
    # Task routing
    task_routes={
        "app.tasks.cvd_orchestration.*": {"queue": "cvd_runs"},
        "app.tasks.monitoring.*": {"queue": "monitoring"},
        "app.tasks.analytics.*": {"queue": "analytics"},
    },

    # Task queues
    task_queues=(
        Queue("cvd_runs", routing_key="cvd_runs"),
        Queue("monitoring", routing_key="monitoring"),
        Queue("analytics", routing_key="analytics"),
        Queue("default", routing_key="default"),
    ),

    # Task execution
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,

    # Task results
    result_expires=3600,  # 1 hour
    result_extended=True,

    # Task tracking
    task_track_started=True,
    task_send_sent_event=True,

    # Worker configuration
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,

    # Task retry
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Beat schedule (for periodic tasks)
    beat_schedule={
        "monitor-running-cvd-runs": {
            "task": "app.tasks.monitoring.monitor_all_running_runs",
            "schedule": 5.0,  # Every 5 seconds
        },
        "cleanup-old-tasks": {
            "task": "app.tasks.maintenance.cleanup_old_task_results",
            "schedule": 3600.0,  # Every hour
        },
    },
)

# Auto-discover tasks
celery_app.autodiscover_tasks([
    "app.tasks.cvd_orchestration",
    "app.tasks.monitoring",
    "app.tasks.analytics",
])


def get_celery_app():
    """Get configured Celery app instance"""
    return celery_app
