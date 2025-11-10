"""Ion Implantation API endpoints.

REST API for:
- Creating and managing ion implantation runs
- Querying run status and results
- Simulating dose profiles
- Accessing SPC and VM data

All endpoints require authentication and RBAC permissions.
"""

from typing import Optional, List
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field

from app.auth import (
    User,
    Permission,
    require_permission,
    require_org_access,
    AuditLogger,
)
from app.models.job import (
    JobStore,
    JobType,
    Job,
    JobResponse,
    job_to_response,
)
from app.tasks.ion_tasks import (
    execute_ion_run,
    simulate_dose_profile,
    validate_ion_recipe,
)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(
    prefix="/api/ion",
    tags=["ion-implantation"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class IonRunRequest(BaseModel):
    """Request to create an ion implantation run."""

    # Recipe parameters
    species: str = Field(..., description="Dopant species (B, P, As, BF2, In, Sb)")
    energy_kev: float = Field(..., ge=1, le=200, description="Implant energy (keV)")
    dose_atoms_cm2: float = Field(..., ge=1e11, le=1e16, description="Target dose (atoms/cm²)")
    tilt_deg: float = Field(7.0, ge=0, le=90, description="Tilt angle (degrees)")
    twist_deg: float = Field(0.0, ge=0, le=90, description="Twist angle (degrees)")

    # Beam parameters
    beam_current_ma: float = Field(..., ge=0.1, le=50, description="Beam current (mA)")
    scan_speed_mm_s: float = Field(50.0, ge=1, le=100, description="Scan speed (mm/s)")

    # Wafer parameters
    wafer_diameter_mm: float = Field(300, description="Wafer diameter (mm)")
    wafer_id: Optional[str] = Field(None, description="Wafer identifier")

    # Optional metadata
    lot_id: Optional[str] = Field(None, description="Lot identifier")
    comments: Optional[str] = Field(None, description="Run comments")

    class Config:
        schema_extra = {
            "example": {
                "species": "P",
                "energy_kev": 40.0,
                "dose_atoms_cm2": 1e15,
                "tilt_deg": 7.0,
                "twist_deg": 0.0,
                "beam_current_ma": 5.0,
                "scan_speed_mm_s": 50.0,
                "wafer_diameter_mm": 300,
                "wafer_id": "W12345",
                "lot_id": "LOT-001",
                "comments": "Test run for process development",
            }
        }


class IonRunResponse(BaseModel):
    """Response for ion run creation."""
    run_id: str
    job_id: str
    status: str
    created_at: datetime
    message: str


class IonRunStatus(BaseModel):
    """Full ion run status with results."""
    run_id: str
    job_id: str
    status: str
    progress: float
    current_step: str

    # Recipe
    recipe: dict

    # Timing
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_seconds: Optional[float]

    # Results (if completed)
    final_dose_atoms_cm2: Optional[float] = None
    dose_error_pct: Optional[float] = None
    vm_prediction: Optional[dict] = None
    spc_alerts_count: Optional[int] = None

    # Artifacts
    logs_uri: Optional[str] = None
    artifacts: List[dict] = Field(default_factory=list)

    # Error (if failed)
    error_message: Optional[str] = None


class DoseProfileRequest(BaseModel):
    """Request for dose profile simulation."""
    species: str = Field(..., description="Dopant species")
    energy_kev: float = Field(..., ge=1, le=200, description="Implant energy (keV)")
    dose_atoms_cm2: float = Field(..., ge=1e11, le=1e16, description="Dose (atoms/cm²)")
    tilt_deg: float = Field(7.0, ge=0, le=90, description="Tilt angle")
    twist_deg: float = Field(0.0, ge=0, le=90, description="Twist angle")

    class Config:
        schema_extra = {
            "example": {
                "species": "B",
                "energy_kev": 20.0,
                "dose_atoms_cm2": 5e14,
                "tilt_deg": 7.0,
                "twist_deg": 0.0,
            }
        }


class DoseProfileResponse(BaseModel):
    """Response with 1D and 2D dose profiles."""
    profile_1d: dict
    profile_2d: dict
    metadata: dict


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/runs",
    response_model=IonRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create ion implantation run",
    description="""
    Create and enqueue an ion implantation run.

    The run will be validated against SOP and calibration, then enqueued
    as a Celery background job. Returns immediately with job ID for tracking.

    **Required permission**: `ion:create_run`

    **Validation checks**:
    - Species must be supported (B, P, As, BF2, In, Sb)
    - Energy range: 1-200 keV
    - Dose range: 1e11 - 1e16 atoms/cm²
    - Tilt/twist angles: 0-90°
    - Beam current: 0.1-50 mA
    - Scan speed: 1-100 mm/s

    **Process flow**:
    1. Validate recipe
    2. Create job record
    3. Enqueue Celery task
    4. Return job ID for progress tracking
    """,
)
async def create_ion_run(
    request: IonRunRequest,
    user: User = Depends(require_permission(Permission.ION_CREATE_RUN)),
) -> IonRunResponse:
    """Create and enqueue an ion implantation run."""

    # Generate IDs
    run_id = f"ION-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    job_id = f"job-{uuid.uuid4().hex}"

    # Convert request to recipe dict
    recipe = request.dict()

    # Validate recipe
    validation = validate_ion_recipe(recipe)
    if not validation["valid"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "message": "Recipe validation failed",
                "errors": validation["errors"],
                "warnings": validation["warnings"],
            },
        )

    # Create job record
    job = Job(
        id=job_id,
        run_id=run_id,
        org_id=user.org_id,
        job_type=JobType.ION_IMPLANT,
        user_id=user.user_id,
        recipe_data=recipe,
    )

    JobStore.create_job(job)

    # Enqueue Celery task
    execute_ion_run.apply_async(
        kwargs={
            "job_id": job_id,
            "run_id": run_id,
            "org_id": user.org_id,
            "recipe": recipe,
            "user_id": user.user_id,
        },
        task_id=job_id,
    )

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="ion_run",
        resource_id=run_id,
        action="create",
        success=True,
        metadata={
            "species": request.species,
            "energy_kev": request.energy_kev,
            "dose_atoms_cm2": request.dose_atoms_cm2,
        },
    )

    return IonRunResponse(
        run_id=run_id,
        job_id=job_id,
        status="queued",
        created_at=job.created_at,
        message=f"Ion implantation run {run_id} created and queued for execution",
    )


@router.get(
    "/runs/{run_id}",
    response_model=IonRunStatus,
    summary="Get ion run status",
    description="""
    Get status and results for an ion implantation run.

    Returns job status, progress, telemetry, SPC alerts, VM predictions,
    and artifacts (if completed).

    **Required permission**: `ion:view_run`

    **Status values**:
    - `queued`: Waiting to start
    - `running`: Currently executing
    - `completed`: Finished successfully
    - `failed`: Encountered an error
    - `cancelled`: Cancelled by user

    **Results** (if completed):
    - Final dose and error %
    - VM predictions (sheet resistance, junction depth, activation)
    - SPC alerts
    - Artifacts (2D profiles, charts, logs)
    """,
)
async def get_ion_run(
    run_id: str,
    user: User = Depends(require_permission(Permission.ION_VIEW_RUN)),
) -> IonRunStatus:
    """Get ion run status and results."""

    # Find job by run_id
    job = JobStore.get_job_by_run_id(run_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ion run {run_id} not found",
        )

    # Check org access
    require_org_access(job.org_id, user)

    # Calculate duration
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    # Extract results from metadata
    metadata = job.metadata or {}
    final_dose = metadata.get("final_dose_atoms_cm2")
    dose_error_pct = metadata.get("dose_error_pct")
    vm_prediction = metadata.get("vm_prediction")
    spc_alerts_count = metadata.get("spc_alerts_count")

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="ion_run",
        resource_id=run_id,
        action="view",
        success=True,
    )

    return IonRunStatus(
        run_id=run_id,
        job_id=job.id,
        status=job.status.value,
        progress=job.progress,
        current_step=job.current_step,
        recipe=job.recipe_data,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        duration_seconds=duration,
        final_dose_atoms_cm2=final_dose,
        dose_error_pct=dose_error_pct,
        vm_prediction=vm_prediction,
        spc_alerts_count=spc_alerts_count,
        logs_uri=job.logs_uri,
        artifacts=job.artifacts,
        error_message=job.error_message,
    )


@router.get(
    "/runs",
    response_model=List[IonRunStatus],
    summary="List ion runs",
    description="""
    List ion implantation runs for the authenticated user's organization.

    Supports filtering and pagination.

    **Required permission**: `ion:view_run`

    **Query parameters**:
    - `status`: Filter by job status (queued, running, completed, failed, cancelled)
    - `limit`: Maximum number of results (default: 50, max: 100)
    - `offset`: Number of results to skip (default: 0)
    """,
)
async def list_ion_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    user: User = Depends(require_permission(Permission.ION_VIEW_RUN)),
) -> List[IonRunStatus]:
    """List ion runs for user's organization."""

    # Parse status filter
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

    # Query jobs
    jobs = JobStore.list_jobs(
        org_id=user.org_id,
        job_type=JobType.ION_IMPLANT,
        status=status_filter,
        limit=limit,
        offset=offset,
    )

    # Convert to response format
    results = []
    for job in jobs:
        duration = None
        if job.started_at and job.completed_at:
            duration = (job.completed_at - job.started_at).total_seconds()

        metadata = job.metadata or {}

        results.append(
            IonRunStatus(
                run_id=job.run_id,
                job_id=job.id,
                status=job.status.value,
                progress=job.progress,
                current_step=job.current_step,
                recipe=job.recipe_data,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                duration_seconds=duration,
                final_dose_atoms_cm2=metadata.get("final_dose_atoms_cm2"),
                dose_error_pct=metadata.get("dose_error_pct"),
                vm_prediction=metadata.get("vm_prediction"),
                spc_alerts_count=metadata.get("spc_alerts_count"),
                logs_uri=job.logs_uri,
                artifacts=job.artifacts,
                error_message=job.error_message,
            )
        )

    return results


@router.post(
    "/simulate/dose_profile",
    response_model=DoseProfileResponse,
    summary="Simulate dose profile",
    description="""
    Calculate 1D depth profile and 2D lateral profile (SRIM-like).

    This is a synchronous calculation that returns immediately.
    Use this for dose profile visualization and planning.

    **Required permission**: `ion:simulate`

    **Returns**:
    - **1D profile**: Concentration vs depth with projected range and straggle
    - **2D profile**: Lateral dose distribution across wafer

    **Note**: This uses physics-based models (LSS theory) for fast estimation.
    For production, consider SRIM or detailed Monte Carlo simulations.
    """,
)
async def simulate_dose_profile_endpoint(
    request: DoseProfileRequest,
    user: User = Depends(require_permission(Permission.ION_SIMULATE)),
) -> DoseProfileResponse:
    """Simulate ion implantation dose profile."""

    # Run synchronous simulation
    result = simulate_dose_profile(
        species=request.species,
        energy_kev=request.energy_kev,
        dose_atoms_cm2=request.dose_atoms_cm2,
        tilt_deg=request.tilt_deg,
        twist_deg=request.twist_deg,
    )

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="ion_simulation",
        resource_id=f"{request.species}_{request.energy_kev}keV",
        action="simulate",
        success=True,
        metadata={
            "species": request.species,
            "energy_kev": request.energy_kev,
        },
    )

    return DoseProfileResponse(
        profile_1d=result["profile_1d"],
        profile_2d=result["profile_2d"],
        metadata=result["metadata"],
    )


@router.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel ion run",
    description="""
    Cancel a running ion implantation job.

    **Required permission**: `ion:cancel_run`

    **Note**: Cancellation is graceful. The job will complete its current
    step before stopping. Status will change to `cancelled`.
    """,
)
async def cancel_ion_run(
    run_id: str,
    user: User = Depends(require_permission(Permission.ION_CANCEL_RUN)),
) -> dict:
    """Cancel a running ion run."""

    # Find job
    job = JobStore.get_job_by_run_id(run_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ion run {run_id} not found",
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
        job.id,
        status=JobStatusEnum.CANCELLED,
        completed_at=datetime.utcnow(),
    )

    # Revoke Celery task
    from app.celery_app import celery_app
    celery_app.control.revoke(job.id, terminate=True)

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="ion_run",
        resource_id=run_id,
        action="cancel",
        success=True,
    )

    return {
        "run_id": run_id,
        "status": "cancelled",
        "message": f"Ion run {run_id} has been cancelled",
    }


# Export
__all__ = ["router"]
