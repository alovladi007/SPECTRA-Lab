"""RTP (Rapid Thermal Processing) API endpoints.

REST API for:
- Creating and managing RTP runs
- Querying run status and results
- Auto-tuning PID/MPC controllers
- Accessing temperature profiles, SPC, and VM data

All endpoints require authentication and RBAC permissions.
"""

from typing import Optional, List, Dict, Any
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
)
from app.tasks.rtp_tasks import (
    execute_rtp_run,
    tune_controller,
    validate_rtp_recipe,
)


# ============================================================================
# Router Setup
# ============================================================================

router = APIRouter(
    prefix="/api/rtp",
    tags=["rtp"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class RTPSegment(BaseModel):
    """Single temperature segment in RTP recipe."""
    target_temp_c: float = Field(..., ge=400, le=1200, description="Target temperature (°C)")
    duration_s: float = Field(..., ge=0, le=300, description="Segment duration (seconds)")
    ramp_rate_c_s: Optional[float] = Field(None, ge=0, le=100, description="Ramp rate (°C/s)")


class RTPRunRequest(BaseModel):
    """Request to create an RTP run."""

    # Temperature profile
    segments: List[RTPSegment] = Field(..., min_items=1, max_items=10)
    initial_temp_c: float = Field(25.0, description="Initial chamber temperature (°C)")

    # Gas flows
    n2_flow_slm: float = Field(10.0, ge=0, le=20, description="N2 flow rate (SLM)")
    o2_flow_slm: float = Field(0.0, ge=0, le=5, description="O2 flow rate (SLM)")

    # Controller selection
    controller_type: str = Field("pid", description="Controller type (pid or mpc)")

    # PID parameters (if controller_type == "pid")
    pid_kp: Optional[float] = Field(None, description="PID proportional gain")
    pid_ki: Optional[float] = Field(None, description="PID integral gain")
    pid_kd: Optional[float] = Field(None, description="PID derivative gain")

    # Wafer parameters
    wafer_id: Optional[str] = Field(None, description="Wafer identifier")

    # Ion implant context (for VM)
    ion_implant_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Ion implantation parameters (species, energy, dose) for VM prediction",
    )

    # Optional metadata
    lot_id: Optional[str] = Field(None, description="Lot identifier")
    comments: Optional[str] = Field(None, description="Run comments")

    class Config:
        schema_extra = {
            "example": {
                "segments": [
                    {"target_temp_c": 800, "duration_s": 5, "ramp_rate_c_s": 50},
                    {"target_temp_c": 1000, "duration_s": 10, "ramp_rate_c_s": 40},
                    {"target_temp_c": 1000, "duration_s": 30},  # Hold
                    {"target_temp_c": 400, "duration_s": 15, "ramp_rate_c_s": 40},  # Cool down
                ],
                "initial_temp_c": 25.0,
                "n2_flow_slm": 10.0,
                "o2_flow_slm": 0.0,
                "controller_type": "pid",
                "pid_kp": 5.0,
                "pid_ki": 0.5,
                "pid_kd": 1.0,
                "wafer_id": "W12345",
                "ion_implant_context": {
                    "species": "P",
                    "energy_kev": 40.0,
                    "dose_atoms_cm2": 1e15,
                },
                "lot_id": "LOT-001",
                "comments": "Activation anneal after P implant",
            }
        }


class RTPRunResponse(BaseModel):
    """Response for RTP run creation."""
    run_id: str
    job_id: str
    status: str
    created_at: datetime
    message: str


class RTPRunStatus(BaseModel):
    """Full RTP run status with results."""
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
    final_temp_c: Optional[float] = None
    temp_error_c: Optional[float] = None
    max_overshoot_c: Optional[float] = None
    avg_ramp_error_c: Optional[float] = None
    thermal_budget: Optional[float] = None
    vm_prediction: Optional[dict] = None
    spc_alerts_count: Optional[int] = None

    # Artifacts
    logs_uri: Optional[str] = None
    artifacts: List[dict] = Field(default_factory=list)

    # Error (if failed)
    error_message: Optional[str] = None


class ControllerTuningRequest(BaseModel):
    """Request for controller auto-tuning."""
    controller_type: str = Field(..., description="Controller type (pid or mpc)")
    recipe: Dict[str, Any] = Field(..., description="RTP recipe to optimize for")
    optimization_target: str = Field(
        "minimize_overshoot",
        description="Optimization objective (minimize_overshoot, minimize_ramp_error, minimize_settling_time)",
    )

    class Config:
        schema_extra = {
            "example": {
                "controller_type": "pid",
                "recipe": {
                    "segments": [
                        {"target_temp_c": 1000, "duration_s": 30},
                    ],
                },
                "optimization_target": "minimize_overshoot",
            }
        }


class ControllerTuningResponse(BaseModel):
    """Response with tuned controller parameters."""
    controller_type: str
    parameters: dict
    expected_performance: dict
    tuning_method: str


# ============================================================================
# Endpoints
# ============================================================================

@router.post(
    "/runs",
    response_model=RTPRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create RTP run",
    description="""
    Create and enqueue an RTP run.

    The run will be validated against thermal budgets and system limits,
    then enqueued as a Celery background job. Returns immediately with job ID.

    **Required permission**: `rtp:create_run`

    **Validation checks**:
    - Temperature range: 400-1200°C
    - Ramp rate: ≤100°C/s (up), ≤50°C/s (down)
    - Segment duration: 0-300 seconds
    - Gas flows: N2 (0-20 SLM), O2 (0-5 SLM)
    - Thermal budget warnings

    **Controller types**:
    - **PID**: Simple PID controller with Kp, Ki, Kd gains
    - **MPC**: Model Predictive Control with constraint handling

    **Process flow**:
    1. Validate recipe
    2. Create job record
    3. Enqueue Celery task
    4. Return job ID for progress tracking
    """,
)
async def create_rtp_run(
    request: RTPRunRequest,
    user: User = Depends(require_permission(Permission.RTP_CREATE_RUN)),
) -> RTPRunResponse:
    """Create and enqueue an RTP run."""

    # Generate IDs
    run_id = f"RTP-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    job_id = f"job-{uuid.uuid4().hex}"

    # Convert request to recipe dict
    recipe = request.dict()

    # Validate recipe
    validation = validate_rtp_recipe(recipe)
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
        job_type=JobType.RTP,
        user_id=user.user_id,
        recipe_data=recipe,
    )

    JobStore.create_job(job)

    # Enqueue Celery task
    execute_rtp_run.apply_async(
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
        resource_type="rtp_run",
        resource_id=run_id,
        action="create",
        success=True,
        metadata={
            "segments_count": len(request.segments),
            "peak_temp_c": max(seg.target_temp_c for seg in request.segments),
            "controller_type": request.controller_type,
        },
    )

    return RTPRunResponse(
        run_id=run_id,
        job_id=job_id,
        status="queued",
        created_at=job.created_at,
        message=f"RTP run {run_id} created and queued for execution",
    )


@router.get(
    "/runs/{run_id}",
    response_model=RTPRunStatus,
    summary="Get RTP run status",
    description="""
    Get status and results for an RTP run.

    Returns job status, progress, temperature profile, SPC alerts,
    VM predictions, and artifacts (if completed).

    **Required permission**: `rtp:view_run`

    **Status values**:
    - `queued`: Waiting to start
    - `running`: Currently executing
    - `completed`: Finished successfully
    - `failed`: Encountered an error
    - `cancelled`: Cancelled by user

    **Results** (if completed):
    - Final temperature and error
    - Max overshoot
    - Average ramp error
    - Thermal budget
    - VM predictions (activation, diffusion, oxide thickness)
    - SPC alerts
    - Artifacts (temperature profiles, control charts, logs)
    """,
)
async def get_rtp_run(
    run_id: str,
    user: User = Depends(require_permission(Permission.RTP_VIEW_RUN)),
) -> RTPRunStatus:
    """Get RTP run status and results."""

    # Find job by run_id
    job = JobStore.get_job_by_run_id(run_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RTP run {run_id} not found",
        )

    # Check org access
    require_org_access(job.org_id, user)

    # Calculate duration
    duration = None
    if job.started_at and job.completed_at:
        duration = (job.completed_at - job.started_at).total_seconds()

    # Extract results from metadata
    metadata = job.metadata or {}

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="rtp_run",
        resource_id=run_id,
        action="view",
        success=True,
    )

    return RTPRunStatus(
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
        final_temp_c=metadata.get("final_temp_c"),
        temp_error_c=metadata.get("temp_error_c"),
        max_overshoot_c=metadata.get("max_overshoot_c"),
        avg_ramp_error_c=metadata.get("avg_ramp_error_c"),
        thermal_budget=metadata.get("thermal_budget"),
        vm_prediction=metadata.get("vm_prediction"),
        spc_alerts_count=metadata.get("spc_alerts_count"),
        logs_uri=job.logs_uri,
        artifacts=job.artifacts,
        error_message=job.error_message,
    )


@router.get(
    "/runs",
    response_model=List[RTPRunStatus],
    summary="List RTP runs",
    description="""
    List RTP runs for the authenticated user's organization.

    Supports filtering and pagination.

    **Required permission**: `rtp:view_run`
    """,
)
async def list_rtp_runs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Results per page"),
    offset: int = Query(0, ge=0, description="Results to skip"),
    user: User = Depends(require_permission(Permission.RTP_VIEW_RUN)),
) -> List[RTPRunStatus]:
    """List RTP runs for user's organization."""

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
        job_type=JobType.RTP,
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
            RTPRunStatus(
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
                final_temp_c=metadata.get("final_temp_c"),
                temp_error_c=metadata.get("temp_error_c"),
                max_overshoot_c=metadata.get("max_overshoot_c"),
                avg_ramp_error_c=metadata.get("avg_ramp_error_c"),
                thermal_budget=metadata.get("thermal_budget"),
                vm_prediction=metadata.get("vm_prediction"),
                spc_alerts_count=metadata.get("spc_alerts_count"),
                logs_uri=job.logs_uri,
                artifacts=job.artifacts,
                error_message=job.error_message,
            )
        )

    return results


@router.post(
    "/tune/controller",
    response_model=ControllerTuningResponse,
    summary="Auto-tune controller",
    description="""
    Auto-tune PID or MPC controller parameters for a given recipe.

    Uses optimization algorithms to find optimal controller gains that
    minimize overshoot, ramp error, or settling time.

    **Required permission**: `rtp:tune_controller`

    **Tuning methods**:
    - **PID**: Relay feedback (Ziegler-Nichols variant)
    - **MPC**: Model-based optimization

    **Optimization targets**:
    - `minimize_overshoot`: Minimize temperature overshoot
    - `minimize_ramp_error`: Minimize tracking error during ramps
    - `minimize_settling_time`: Minimize time to reach steady-state

    **Returns**:
    - Recommended controller parameters
    - Expected performance metrics
    - Tuning method used

    **Note**: This runs a quick simulation to evaluate controller performance.
    For production, validate tuned parameters in hardware tests.
    """,
)
async def tune_controller_endpoint(
    request: ControllerTuningRequest,
    user: User = Depends(require_permission(Permission.RTP_TUNE_CONTROLLER)),
) -> ControllerTuningResponse:
    """Auto-tune controller parameters."""

    # Validate controller type
    if request.controller_type not in ["pid", "mpc"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid controller type: {request.controller_type}",
        )

    # Run tuning task (synchronous for now, can be async with Celery)
    result = tune_controller(
        controller_type=request.controller_type,
        recipe=request.recipe,
        optimization_target=request.optimization_target,
    )

    # Audit log
    AuditLogger.log_access(
        user=user,
        resource_type="controller_tuning",
        resource_id=request.controller_type,
        action="tune",
        success=True,
        metadata={
            "controller_type": request.controller_type,
            "optimization_target": request.optimization_target,
        },
    )

    return ControllerTuningResponse(
        controller_type=result["controller_type"],
        parameters=result["parameters"],
        expected_performance=result["expected_performance"],
        tuning_method=result["tuning_method"],
    )


@router.delete(
    "/runs/{run_id}",
    status_code=status.HTTP_200_OK,
    summary="Cancel RTP run",
    description="""
    Cancel a running RTP job.

    **Required permission**: `rtp:cancel_run`

    **Note**: Cancellation is graceful. The job will complete its current
    step before stopping. Status will change to `cancelled`.
    """,
)
async def cancel_rtp_run(
    run_id: str,
    user: User = Depends(require_permission(Permission.RTP_CANCEL_RUN)),
) -> dict:
    """Cancel a running RTP run."""

    # Find job
    job = JobStore.get_job_by_run_id(run_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"RTP run {run_id} not found",
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
        resource_type="rtp_run",
        resource_id=run_id,
        action="cancel",
        success=True,
    )

    return {
        "run_id": run_id,
        "status": "cancelled",
        "message": f"RTP run {run_id} has been cancelled",
    }


# Export
__all__ = ["router"]
