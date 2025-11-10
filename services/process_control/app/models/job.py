"""Job models for background task tracking.

Tracks Celery tasks with progress, status, logs, and artifacts.
"""

from sqlalchemy import Column, String, Integer, Float, JSON, DateTime, Enum as SQLEnum, Text, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


# ============================================================================
# Enums
# ============================================================================

class JobStatus(str, PyEnum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobType(str, PyEnum):
    """Type of job."""
    ION_IMPLANT = "ion_implant"
    RTP = "rtp"
    SIMULATION = "simulation"
    CALIBRATION = "calibration"


# ============================================================================
# Database Models
# ============================================================================

class Job:
    """Job record for tracking background tasks.

    Note: This is a plain Python class. In production, inherit from SQLAlchemy Base.
    """

    def __init__(
        self,
        id: str,
        run_id: str,
        org_id: str,
        job_type: JobType,
        user_id: str,
        recipe_data: Dict,
        status: JobStatus = JobStatus.QUEUED,
    ):
        self.id = id
        self.run_id = run_id
        self.org_id = org_id
        self.job_type = job_type
        self.user_id = user_id
        self.status = status
        self.progress = 0.0
        self.current_step = "Initializing"
        self.recipe_data = recipe_data
        self.started_at = None
        self.completed_at = None
        self.error_message = None
        self.retry_count = 0
        self.logs_uri = None
        self.artifacts = []
        self.metadata = {}
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


# ============================================================================
# Pydantic Models (API Schemas)
# ============================================================================

class JobResponse(BaseModel):
    """Response for job creation."""
    run_id: str
    job_id: str
    status: JobStatus
    created_at: datetime


class JobProgress(BaseModel):
    """Job progress update."""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    current_step: str
    started_at: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None


class JobStatus(BaseModel):
    """Full job status."""
    job_id: str
    run_id: str
    job_type: JobType
    status: JobStatus
    progress: float
    current_step: str

    # Timing
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Results
    error_message: Optional[str] = None
    retry_count: int = 0
    logs_uri: Optional[str] = None
    artifacts: List[Dict[str, str]] = Field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class JobListResponse(BaseModel):
    """List of jobs with pagination."""
    jobs: List[JobStatus]
    total: int
    page: int
    page_size: int
    has_more: bool


class CancelJobResponse(BaseModel):
    """Response for job cancellation."""
    job_id: str
    status: JobStatus
    message: str


# ============================================================================
# In-Memory Job Store (for development)
# ============================================================================

class JobStore:
    """In-memory job storage for development.

    In production, replace with database queries.
    """

    _jobs: Dict[str, Job] = {}

    @classmethod
    def create_job(cls, job: Job) -> Job:
        """Create a new job."""
        cls._jobs[job.id] = job
        return job

    @classmethod
    def get_job(cls, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return cls._jobs.get(job_id)

    @classmethod
    def get_job_by_run_id(cls, run_id: str) -> Optional[Job]:
        """Get job by run ID."""
        for job in cls._jobs.values():
            if job.run_id == run_id:
                return job
        return None

    @classmethod
    def update_job(cls, job_id: str, **kwargs) -> Optional[Job]:
        """Update job fields."""
        job = cls._jobs.get(job_id)
        if job:
            for key, value in kwargs.items():
                setattr(job, key, value)
            job.updated_at = datetime.utcnow()
        return job

    @classmethod
    def list_jobs(
        cls,
        org_id: Optional[str] = None,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Job]:
        """List jobs with filters."""
        jobs = list(cls._jobs.values())

        # Filter
        if org_id:
            jobs = [j for j in jobs if j.org_id == org_id]
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        if status:
            jobs = [j for j in jobs if j.status == status]

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Paginate
        return jobs[offset:offset + limit]

    @classmethod
    def count_jobs(
        cls,
        org_id: Optional[str] = None,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
    ) -> int:
        """Count jobs with filters."""
        jobs = list(cls._jobs.values())

        if org_id:
            jobs = [j for j in jobs if j.org_id == org_id]
        if job_type:
            jobs = [j for j in jobs if j.job_type == job_type]
        if status:
            jobs = [j for j in jobs if j.status == status]

        return len(jobs)


# ============================================================================
# Helper Functions
# ============================================================================

def job_to_response(job: Job) -> JobStatus:
    """Convert Job model to JobStatusResponse."""
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    return JobStatus(
        job_id=job.id,
        run_id=job.run_id,
        job_type=job.job_type,
        status=job.status,
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


# Export
__all__ = [
    "JobStatus",
    "JobType",
    "Job",
    "JobResponse",
    "JobProgress",
    "JobStatusResponse",
    "JobListResponse",
    "CancelJobResponse",
    "JobStore",
    "job_to_response",
]
