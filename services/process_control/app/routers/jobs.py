"""Job management API endpoints.

Cross-cutting job management endpoints for:
- Getting job status (any job type)
- Cancelling jobs
- Listing all jobs
- Viewing job logs and artifacts

These endpoints work with jobs from all process types (Ion, RTP, etc.)
"""

from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from app.auth import (
    User,
    Permission,
    require_permission,
    require_any_permission,
    require_org_access,
    AuditLogger,
)
from app.models.job import (
    JobStore,
    JobType,
    JobStatus as JobStatusModel,
    job_to_response,
)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(
    prefix="/api/jobs",
    tags=["jobs"],
)


# ============================================================================
# Response Models
# ============================================================================

class JobStatusResponse(BaseModel):
    """Job status response."""
    job_id: str
    run_id: str
    job_type: str
    status: str
    progress: float
    current_step: str

    # Timing
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]

    # Results
    error_message: Optional[str]
    retry_count: int
    logs_uri: Optional[str]
    artifacts: List[dict] = Field(default_factory=list)

    # Metadata
    metadata: dict = Field(default_factory=dict)


class JobListResponse(BaseModel):
    """List of jobs with pagination."""
    jobs: List[JobStatusResponse]
    total: int
    page: int
    page_size: int
    has_more: bool


class CancelJobResponse(BaseModel):
    """Response for job cancellation."""
    job_id: str
    run_id: str
    status: str
    message: str


# ============================================================================
# Endpoints
# ============================================================================

@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get job status",
    description="""
    Get status, progress, and results for any job (Ion, RTP, etc.).

    **Required permission**: One of `job:view_status`, `ion:view_run`, `rtp:view_run`

    **Job lifecycle**:
    1. `queued`: Job created, waiting to start
    2. `running`: Job is executing
    3. `completed`: Job finished successfully
    4. `failed`: Job encountered an error
    5. `cancelled`: Job was cancelled by user
    6. `retrying`: Job failed and is being retried

    **Response includes**:
    - Current status and progress (0-100%)
    - Timing information (created, started, completed)
    - Artifacts (logs, telemetry, charts)
    - Metadata (results, metrics)
    - Error message (if failed)
    """,
)
async def get_job_status(
    job_id: str,
    user: User = Depends(
        require_any_permission(
            Permission.JOB_VIEW_STATUS,
            Permission.ION_VIEW_RUN,
            Permission.RTP_VIEW_RUN,
        )
    ),
) -> JobStatusResponse:
    """Get job status by ID."""

    job = JobStore.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Check org access
    require_org_access(job.org_id, user)

    # Calculate duration
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="job",
        resource_id=job_id,
        action="view_status",
        success=True,
    )

    return JobStatusResponse(
        job_id=job.id,
        run_id=job.run_id,
        job_type=job.job_type.value,
        status=job.status.value,
        progress=job.progress,
        current_step=job.current_step,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        duration_seconds=duration,
        error_message=job.error_message,
        retry_count=job.retry_count,
        logs_uri=job.logs_uri,
        artifacts=job.artifacts,
        metadata=job.metadata,
    )


@router.get(
    "",
    response_model=JobListResponse,
    summary="List jobs",
    description="""
    List all jobs for the authenticated user's organization.

    Supports filtering by job type, status, and pagination.

    **Required permission**: `job:view_status`

    **Query parameters**:
    - `job_type`: Filter by job type (ion_implant, rtp, simulation, calibration)
    - `status`: Filter by status (queued, running, completed, failed, cancelled)
    - `page`: Page number (default: 1)
    - `page_size`: Results per page (default: 50, max: 100)

    **Response includes**:
    - List of jobs with full status
    - Total count of matching jobs
    - Pagination information
    """,
)
async def list_jobs(
    job_type: Optional[str] = Query(None, description="Filter by job type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=100, description="Results per page"),
    user: User = Depends(require_permission(Permission.JOB_VIEW_STATUS)),
) -> JobListResponse:
    """List jobs for user's organization."""

    # Parse filters
    job_type_filter = None
    if job_type:
        try:
            job_type_filter = JobType(job_type)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid job type: {job_type}",
            )

    status_filter = None
    if status:
        try:
            from app.models.job import JobStatus as JobStatusEnum
            status_filter = JobStatusEnum(status)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}",
            )

    # Calculate offset
    offset = (page - 1) * page_size

    # Query jobs
    jobs = JobStore.list_jobs(
        org_id=user.org_id,
        job_type=job_type_filter,
        status=status_filter,
        limit=page_size,
        offset=offset,
    )

    # Count total
    total = JobStore.count_jobs(
        org_id=user.org_id,
        job_type=job_type_filter,
        status=status_filter,
    )

    # Convert to response format
    job_responses = []
    for job in jobs:
        duration = None
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()

        job_responses.append(
            JobStatusResponse(
                job_id=job.id,
                run_id=job.run_id,
                job_type=job.job_type.value,
                status=job.status.value,
                progress=job.progress,
                current_step=job.current_step,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                duration_seconds=duration,
                error_message=job.error_message,
                retry_count=job.retry_count,
                logs_uri=job.logs_uri,
                artifacts=job.artifacts,
                metadata=job.metadata,
            )
        )

    has_more = (offset + page_size) < total

    return JobListResponse(
        jobs=job_responses,
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
    )


@router.post(
    "/{job_id}/cancel",
    response_model=CancelJobResponse,
    summary="Cancel job",
    description="""
    Cancel a running job (graceful cancellation).

    **Required permission**: `job:cancel`

    **Cancellation behavior**:
    - Jobs in `queued` or `running` status can be cancelled
    - Cancellation is graceful: current step completes before stopping
    - Status changes to `cancelled`
    - Celery task is revoked

    **Note**: Jobs that are `completed`, `failed`, or already `cancelled`
    cannot be cancelled again.
    """,
)
async def cancel_job(
    job_id: str,
    user: User = Depends(require_permission(Permission.JOB_CANCEL)),
) -> CancelJobResponse:
    """Cancel a running job."""

    job = JobStore.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Check org access
    require_org_access(job.org_id, user)

    # Check if cancellable
    from app.models.job import JobStatus as JobStatusEnum
    if job.status not in [JobStatusEnum.QUEUED, JobStatusEnum.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job in status: {job.status.value}",
        )

    # Mark as cancelled
    JobStore.update_job(
        job_id,
        status=JobStatusEnum.CANCELLED,
        completed_at=datetime.utcnow(),
    )

    # Revoke Celery task
    from app.celery_app import celery_app
    celery_app.control.revoke(job_id, terminate=True)

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="job",
        resource_id=job_id,
        action="cancel",
        success=True,
        metadata={"job_type": job.job_type.value},
    )

    return CancelJobResponse(
        job_id=job_id,
        run_id=job.run_id,
        status="cancelled",
        message=f"Job {job_id} (run {job.run_id}) has been cancelled",
    )


@router.post(
    "/{job_id}/retry",
    response_model=JobStatusResponse,
    summary="Retry failed job",
    description="""
    Retry a failed job.

    **Required permission**: `job:retry`

    **Retry behavior**:
    - Only jobs with `failed` status can be retried
    - Creates a new job with same parameters
    - Original job remains in `failed` status
    - New job starts with `queued` status

    **Use cases**:
    - Transient failures (network, timeouts)
    - After fixing underlying issues
    - Re-running with same parameters
    """,
)
async def retry_job(
    job_id: str,
    user: User = Depends(require_permission(Permission.JOB_RETRY)),
) -> JobStatusResponse:
    """Retry a failed job."""

    job = JobStore.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Check org access
    require_org_access(job.org_id, user)

    # Check if retryable
    from app.models.job import JobStatus as JobStatusEnum
    if job.status != JobStatusEnum.FAILED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot retry job in status: {job.status.value}. Only failed jobs can be retried.",
        )

    # Create new job with same parameters
    import uuid
    new_job_id = f"job-{uuid.uuid4().hex}"
    new_run_id = f"{job.job_type.value.upper()}-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    from app.models.job import Job
    new_job = Job(
        id=new_job_id,
        run_id=new_run_id,
        org_id=job.org_id,
        job_type=job.job_type,
        user_id=user.user_id,
        recipe_data=job.recipe_data,
    )

    JobStore.create_job(new_job)

    # Enqueue appropriate task
    if job.job_type == JobType.ION_IMPLANT:
        from app.tasks.ion_tasks import execute_ion_run
        execute_ion_run.apply_async(
            kwargs={
                "job_id": new_job_id,
                "run_id": new_run_id,
                "org_id": job.org_id,
                "recipe": job.recipe_data,
                "user_id": user.user_id,
            },
            task_id=new_job_id,
        )
    elif job.job_type == JobType.RTP:
        from app.tasks.rtp_tasks import execute_rtp_run
        execute_rtp_run.apply_async(
            kwargs={
                "job_id": new_job_id,
                "run_id": new_run_id,
                "org_id": job.org_id,
                "recipe": job.recipe_data,
                "user_id": user.user_id,
            },
            task_id=new_job_id,
        )

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="job",
        resource_id=job_id,
        action="retry",
        success=True,
        metadata={
            "original_job_id": job_id,
            "new_job_id": new_job_id,
            "job_type": job.job_type.value,
        },
    )

    return JobStatusResponse(
        job_id=new_job_id,
        run_id=new_run_id,
        job_type=new_job.job_type.value,
        status=new_job.status.value,
        progress=new_job.progress,
        current_step=new_job.current_step,
        created_at=new_job.created_at,
        started_at=new_job.started_at,
        completed_at=new_job.completed_at,
        duration_seconds=None,
        error_message=new_job.error_message,
        retry_count=new_job.retry_count,
        logs_uri=new_job.logs_uri,
        artifacts=new_job.artifacts,
        metadata=new_job.metadata,
    )


@router.get(
    "/{job_id}/logs",
    summary="Get job logs",
    description="""
    Get logs for a job.

    **Required permission**: `job:view_logs`

    Returns the logs URI. In production, this would fetch actual log content
    from storage (MinIO, S3, etc.).

    **Note**: For completed jobs, logs are stored in artifact storage.
    For running jobs, logs are not yet available.
    """,
)
async def get_job_logs(
    job_id: str,
    user: User = Depends(require_permission(Permission.JOB_VIEW_LOGS)),
) -> dict:
    """Get job logs."""

    job = JobStore.get_job(job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )

    # Check org access
    require_org_access(job.org_id, user)

    if not job.logs_uri:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Logs not available for job {job_id}. Job may still be running or logs were not generated.",
        )

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="job",
        resource_id=job_id,
        action="view_logs",
        success=True,
    )

    return {
        "job_id": job_id,
        "run_id": job.run_id,
        "logs_uri": job.logs_uri,
        "message": "In production, logs would be fetched from storage and returned here",
    }


# Export
__all__ = ["router"]
