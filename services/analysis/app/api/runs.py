"""
Analysis API - Runs endpoints with calibration validation.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, UUID4
from datetime import datetime

from services.shared.db.deps import (
    get_db, get_current_user, require_engineer_or_above,
    check_instrument_calibration
)
from services.shared.db.models import Run, RunStatus, User, Instrument, Recipe

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class RunBase(BaseModel):
    instrument_id: UUID4
    sample_id: Optional[UUID4] = None
    recipe_id: Optional[UUID4] = None
    method: str
    parameters: dict


class RunCreate(RunBase):
    pass


class RunResponse(RunBase):
    id: UUID4
    organization_id: UUID4
    status: RunStatus
    operator_id: UUID4
    started_at: Optional[datetime]
    finished_at: Optional[datetime]
    error_message: Optional[str]
    blocked_reason: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class RunStatusUpdate(BaseModel):
    status: RunStatus
    error_message: Optional[str] = None


# ============================================================================
# Runs CRUD with Calibration Validation
# ============================================================================

@router.post("/", response_model=RunResponse, status_code=status.HTTP_201_CREATED)
def create_run(
    run_in: RunCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Create a new analytical run.

    **IMPORTANT**: Automatically validates instrument calibration.
    If calibration is expired, run is created in BLOCKED status.

    **Permissions**: engineer, pi, admin
    """
    # Verify instrument exists and belongs to org
    instrument = db.query(Instrument).filter(
        Instrument.id == run_in.instrument_id,
        Instrument.organization_id == current_user.organization_id
    ).first()

    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")

    # Check calibration status
    cal_status = check_instrument_calibration(str(run_in.instrument_id), db)

    # Determine initial run status based on calibration
    initial_status = RunStatus.QUEUED
    blocked_reason = None

    if not cal_status["valid"]:
        initial_status = RunStatus.BLOCKED

        if cal_status["status"] == "no_calibration":
            blocked_reason = f"Instrument '{instrument.name}' has no calibration certificate on record"
        elif cal_status["status"] == "expired":
            cal_info = cal_status["latest_calibration"]
            blocked_reason = (
                f"Instrument '{instrument.name}' calibration expired on "
                f"{cal_info['expires_at']}. Certificate ID: {cal_info['certificate_id']}"
            )

    # Create run
    run = Run(
        **run_in.dict(),
        organization_id=current_user.organization_id,
        created_by=current_user.id,
        status=initial_status,
        blocked_reason=blocked_reason
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    return run


@router.get("/", response_model=List[RunResponse])
def list_runs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[RunStatus] = None,
    instrument_id: Optional[UUID4] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List runs with filtering.

    **Auto-scoped to user's organization**
    """
    query = db.query(Run).filter(Run.organization_id == current_user.organization_id)

    if status_filter:
        query = query.filter(Run.status == status_filter)

    if instrument_id:
        query = query.filter(Run.instrument_id == instrument_id)

    runs = query.order_by(Run.created_at.desc()).offset(skip).limit(limit).all()
    return runs


@router.get("/{run_id}", response_model=RunResponse)
def get_run(
    run_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get run by ID."""
    run = db.query(Run).filter(
        Run.id == run_id,
        Run.organization_id == current_user.organization_id
    ).first()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return run


@router.patch("/{run_id}/status", response_model=RunResponse)
def update_run_status(
    run_id: UUID4,
    status_update: RunStatusUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Update run status.

    **Permissions**: engineer, pi, admin
    """
    run = db.query(Run).filter(
        Run.id == run_id,
        Run.organization_id == current_user.organization_id
    ).first()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    # Can't start blocked runs
    if run.status == RunStatus.BLOCKED and status_update.status == RunStatus.RUNNING:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "run_blocked",
                "message": "Cannot start run: blocked due to calibration issue",
                "blocked_reason": run.blocked_reason
            }
        )

    run.status = status_update.status
    if status_update.error_message:
        run.error_message = status_update.error_message

    # Set timestamps
    if status_update.status == RunStatus.RUNNING and not run.started_at:
        run.started_at = datetime.utcnow()
    elif status_update.status in [RunStatus.SUCCEEDED, RunStatus.FAILED]:
        if not run.finished_at:
            run.finished_at = datetime.utcnow()

    db.commit()
    db.refresh(run)
    return run


@router.post("/{run_id}/unblock", response_model=RunResponse)
def unblock_run(
    run_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Attempt to unblock a run by re-checking calibration.

    **Permissions**: engineer, pi, admin
    """
    run = db.query(Run).filter(
        Run.id == run_id,
        Run.organization_id == current_user.organization_id
    ).first()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != RunStatus.BLOCKED:
        raise HTTPException(status_code=400, detail="Run is not blocked")

    # Re-check calibration
    cal_status = check_instrument_calibration(str(run.instrument_id), db)

    if cal_status["valid"]:
        run.status = RunStatus.QUEUED
        run.blocked_reason = None
    else:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "calibration_still_invalid",
                "message": "Calibration is still invalid",
                "calibration_status": cal_status
            }
        )

    db.commit()
    db.refresh(run)
    return run
