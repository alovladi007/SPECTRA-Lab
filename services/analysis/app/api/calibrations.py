"""
Analysis API - Calibrations endpoints.
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
from services.shared.db.models import Calibration, CalibrationStatus, User, Instrument

router = APIRouter(prefix="/api/v1/calibrations", tags=["calibrations"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class CalibrationBase(BaseModel):
    instrument_id: UUID4
    certificate_id: str
    provider: Optional[str] = None
    issued_at: datetime
    expires_at: datetime
    standards_used: Optional[dict] = None
    uncertainty: Optional[dict] = None


class CalibrationCreate(CalibrationBase):
    pass


class CalibrationResponse(CalibrationBase):
    id: UUID4
    status: CalibrationStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CalibrationStatusResponse(BaseModel):
    instrument_id: UUID4
    valid: bool
    status: str
    latest_calibration: Optional[dict] = None


# ============================================================================
# Calibrations CRUD
# ============================================================================

@router.post("/", response_model=CalibrationResponse, status_code=status.HTTP_201_CREATED)
def create_calibration(
    cal_in: CalibrationCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Upload a new calibration certificate.

    **Permissions**: engineer, pi, admin
    """
    # Verify instrument exists and belongs to org
    instrument = db.query(Instrument).filter(
        Instrument.id == cal_in.instrument_id,
        Instrument.organization_id == current_user.organization_id
    ).first()

    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")

    # Validate dates
    if cal_in.expires_at <= cal_in.issued_at:
        raise HTTPException(
            status_code=400,
            detail="Expiration date must be after issue date"
        )

    # Determine status based on expiration
    now = datetime.utcnow()
    initial_status = CalibrationStatus.VALID if cal_in.expires_at > now else CalibrationStatus.EXPIRED

    calibration = Calibration(
        **cal_in.dict(),
        status=initial_status
    )
    db.add(calibration)
    db.commit()
    db.refresh(calibration)

    return calibration


@router.get("/", response_model=List[CalibrationResponse])
def list_calibrations(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    instrument_id: Optional[UUID4] = None,
    status_filter: Optional[CalibrationStatus] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List calibrations.

    **Auto-scoped to user's organization instruments**
    """
    # Get all instrument IDs for user's org
    instrument_ids = db.query(Instrument.id).filter(
        Instrument.organization_id == current_user.organization_id
    ).all()
    instrument_ids = [i[0] for i in instrument_ids]

    query = db.query(Calibration).filter(Calibration.instrument_id.in_(instrument_ids))

    if instrument_id:
        query = query.filter(Calibration.instrument_id == instrument_id)

    if status_filter:
        query = query.filter(Calibration.status == status_filter)

    calibrations = query.order_by(Calibration.expires_at.desc()).offset(skip).limit(limit).all()
    return calibrations


@router.get("/{calibration_id}", response_model=CalibrationResponse)
def get_calibration(
    calibration_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get calibration by ID."""
    # Verify access through instrument ownership
    calibration = db.query(Calibration).join(Instrument).filter(
        Calibration.id == calibration_id,
        Instrument.organization_id == current_user.organization_id
    ).first()

    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")

    return calibration


@router.get("/status/check", response_model=CalibrationStatusResponse)
def check_calibration_status(
    instrument_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Check current calibration status for an instrument.

    Returns validation status and details about the latest calibration.
    """
    # Verify instrument exists and belongs to org
    instrument = db.query(Instrument).filter(
        Instrument.id == instrument_id,
        Instrument.organization_id == current_user.organization_id
    ).first()

    if not instrument:
        raise HTTPException(status_code=404, detail="Instrument not found")

    # Get calibration status
    cal_status = check_instrument_calibration(str(instrument_id), db)

    return CalibrationStatusResponse(
        instrument_id=instrument_id,
        **cal_status
    )


@router.patch("/{calibration_id}/expire", response_model=CalibrationResponse)
def expire_calibration(
    calibration_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Manually expire a calibration (e.g., if instrument damaged).

    **Permissions**: engineer, pi, admin
    """
    # Verify access
    calibration = db.query(Calibration).join(Instrument).filter(
        Calibration.id == calibration_id,
        Instrument.organization_id == current_user.organization_id
    ).first()

    if not calibration:
        raise HTTPException(status_code=404, detail="Calibration not found")

    if calibration.status == CalibrationStatus.EXPIRED:
        raise HTTPException(status_code=400, detail="Calibration already expired")

    calibration.status = CalibrationStatus.EXPIRED
    db.commit()
    db.refresh(calibration)

    return calibration
