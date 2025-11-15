"""
Monitoring Tasks

Periodic tasks for monitoring running CVD processes and system health.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta

from .celery_app import celery_app
from ..realtime.events import emit_run_event, RunEventType, get_run_events

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.monitoring.monitor_all_running_runs")
def monitor_all_running_runs():
    """
    Monitor all currently running CVD runs

    This is a periodic task (runs every 5 seconds) that checks
    the status of all active runs and emits health metrics.
    """
    # In production, would query database for active runs
    # For now, just log
    logger.debug("Monitoring all running CVD runs")

    # Example: Check for stale runs (no updates in 5 minutes)
    # stale_runs = find_stale_runs(timeout_minutes=5)

    # for run_id in stale_runs:
    #     emit_run_event(
    #         run_id=run_id,
    #         event_type=RunEventType.WARNING,
    #         data={
    #             "warning_type": "stale_run",
    #             "message": "No updates received in 5 minutes",
    #         }
    #     )

    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@celery_app.task(name="app.tasks.monitoring.check_run_health")
def check_run_health(run_id: str) -> Dict[str, Any]:
    """
    Check health of a specific CVD run

    Args:
        run_id: Run identifier

    Returns:
        Health status dictionary
    """
    logger.info(f"Checking health for run_id={run_id}")

    # Get recent events
    events = get_run_events(run_id, limit=50)

    if not events:
        return {
            "run_id": run_id,
            "status": "no_events",
            "healthy": False,
        }

    # Get last event
    last_event = events[-1]
    last_event_time = datetime.fromisoformat(last_event.timestamp)
    time_since_last = (datetime.now() - last_event_time).total_seconds()

    # Check if stale (no updates in 5 minutes)
    is_stale = time_since_last > 300

    # Count warnings and errors
    warnings = [e for e in events if e.event_type == RunEventType.WARNING]
    errors = [e for e in events if e.event_type == RunEventType.ERROR]

    # Determine health status
    if errors:
        status = "unhealthy"
        healthy = False
    elif warnings:
        status = "degraded"
        healthy = True
    elif is_stale:
        status = "stale"
        healthy = False
    else:
        status = "healthy"
        healthy = True

    return {
        "run_id": run_id,
        "status": status,
        "healthy": healthy,
        "last_event_time": last_event.timestamp,
        "time_since_last_sec": time_since_last,
        "total_events": len(events),
        "warnings": len(warnings),
        "errors": len(errors),
    }


@celery_app.task(name="app.tasks.monitoring.emit_system_metrics")
def emit_system_metrics():
    """
    Emit system-level metrics (Celery worker stats, queue sizes, etc.)

    This helps monitor the health of the job queue system.
    """
    from celery.task.control import inspect

    i = inspect()

    # Get active tasks
    active = i.active()
    active_count = sum(len(tasks) for tasks in (active or {}).values())

    # Get scheduled tasks
    scheduled = i.scheduled()
    scheduled_count = sum(len(tasks) for tasks in (scheduled or {}).values())

    # Get worker stats
    stats = i.stats()
    worker_count = len(stats or {})

    logger.info(
        f"System metrics: workers={worker_count}, active={active_count}, scheduled={scheduled_count}"
    )

    return {
        "timestamp": datetime.now().isoformat(),
        "workers": worker_count,
        "active_tasks": active_count,
        "scheduled_tasks": scheduled_count,
        "queue_health": "ok" if worker_count > 0 else "no_workers",
    }


@celery_app.task(name="app.tasks.maintenance.cleanup_old_task_results")
def cleanup_old_task_results():
    """
    Clean up old Celery task results

    Runs hourly to prevent result backend from growing indefinitely.
    """
    logger.info("Cleaning up old task results")

    # Celery automatically expires results based on result_expires config
    # This task is mainly for custom cleanup if needed

    return {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
    }
