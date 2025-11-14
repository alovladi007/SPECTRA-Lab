"""
Calibration - FastAPI Routers
REST API endpoints for equipment calibration tracking and compliance
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, select, case
from sqlalchemy.orm import Session

import logging
import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.deps import get_db
from ..models.calibration import EquipmentCalibration, CalibrationStatus
from ..schemas.calibration import (
    CalibrationCreate,
    CalibrationUpdate,
    CalibrationSchema,
    CalibrationStatusCheck,
    CalibrationDashboard,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/calibration", tags=["Calibration"])


# ============================================================================
# Helper Functions
# ============================================================================

def update_calibration_status(calibration: EquipmentCalibration):
    """Update calibration status based on next_calibration_date"""
    calibration.update_status()


# ============================================================================
# Calibration Records
# ============================================================================

@router.post("/calibrations", response_model=CalibrationSchema, status_code=status.HTTP_201_CREATED)
def create_calibration(
    calibration: CalibrationCreate,
    db: Session = Depends(get_db),
):
    """Create a new calibration record"""
    try:
        # Create calibration record
        db_calibration = EquipmentCalibration(**calibration.model_dump())

        # Calculate and set initial status
        update_calibration_status(db_calibration)

        db.add(db_calibration)
        db.commit()
        db.refresh(db_calibration)

        logger.info(
            f"Created calibration: {db_calibration.id} for equipment {db_calibration.equipment_name}"
        )
        return db_calibration

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibrations", response_model=List[CalibrationSchema])
def list_calibrations(
    org_id: Optional[UUID] = Query(None, description="Filter by organization ID"),
    status_filter: Optional[CalibrationStatus] = Query(None, alias="status", description="Filter by status"),
    equipment_id: Optional[UUID] = Query(None, description="Filter by equipment ID"),
    equipment_type: Optional[str] = Query(None, description="Filter by equipment type"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    sort_by: str = Query("next_calibration_date", description="Field to sort by"),
    sort_desc: bool = Query(False, description="Sort in descending order"),
    db: Session = Depends(get_db),
):
    """
    List calibration records with filters and pagination

    Supports filtering by:
    - Organization ID
    - Calibration status (valid, due_soon, expired)
    - Equipment ID
    - Equipment type
    """
    try:
        query = select(EquipmentCalibration)

        # Apply filters
        filters = []
        if org_id:
            filters.append(EquipmentCalibration.org_id == org_id)
        if status_filter:
            filters.append(EquipmentCalibration.status == status_filter)
        if equipment_id:
            filters.append(EquipmentCalibration.equipment_id == equipment_id)
        if equipment_type:
            filters.append(EquipmentCalibration.equipment_type == equipment_type)

        if filters:
            query = query.where(and_(*filters))

        # Sorting
        sort_column = getattr(EquipmentCalibration, sort_by, EquipmentCalibration.next_calibration_date)
        query = query.order_by(sort_column.desc() if sort_desc else sort_column.asc())

        # Pagination
        query = query.offset(skip).limit(limit)

        result = db.execute(query)
        calibrations = result.scalars().all()

        # Update status for all records before returning
        for cal in calibrations:
            update_calibration_status(cal)

        db.commit()

        return calibrations

    except Exception as e:
        logger.exception(f"Error listing calibrations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/calibrations/{calibration_id}", response_model=CalibrationSchema)
def get_calibration(
    calibration_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific calibration record by ID"""
    try:
        query = select(EquipmentCalibration).where(EquipmentCalibration.id == calibration_id)
        result = db.execute(query)
        calibration = result.scalar_one_or_none()

        if not calibration:
            raise HTTPException(status_code=404, detail="Calibration record not found")

        # Update status before returning
        update_calibration_status(calibration)
        db.commit()

        return calibration

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/calibrations/{calibration_id}", response_model=CalibrationSchema)
def update_calibration(
    calibration_id: UUID,
    update_data: CalibrationUpdate,
    db: Session = Depends(get_db),
):
    """Update a calibration record"""
    try:
        query = select(EquipmentCalibration).where(EquipmentCalibration.id == calibration_id)
        result = db.execute(query)
        calibration = result.scalar_one_or_none()

        if not calibration:
            raise HTTPException(status_code=404, detail="Calibration record not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(calibration, key, value)

        # Recalculate status if dates changed
        if 'next_calibration_date' in update_dict or 'calibration_date' in update_dict:
            update_calibration_status(calibration)

        db.commit()
        db.refresh(calibration)

        logger.info(f"Updated calibration: {calibration_id}")
        return calibration

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating calibration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Equipment Status Check
# ============================================================================

@router.get("/equipment/{equipment_id}/status", response_model=CalibrationStatusCheck)
def get_equipment_calibration_status(
    equipment_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Check calibration status for a specific equipment

    Returns the most recent calibration record for the equipment
    """
    try:
        # Get most recent calibration for this equipment
        query = (
            select(EquipmentCalibration)
            .where(EquipmentCalibration.equipment_id == equipment_id)
            .order_by(EquipmentCalibration.calibration_date.desc())
            .limit(1)
        )
        result = db.execute(query)
        calibration = result.scalar_one_or_none()

        if not calibration:
            raise HTTPException(
                status_code=404,
                detail=f"No calibration records found for equipment {equipment_id}"
            )

        # Update status
        update_calibration_status(calibration)
        db.commit()

        # Calculate days until due
        now = datetime.now(calibration.next_calibration_date.tzinfo)
        days_until_due = (calibration.next_calibration_date - now).days

        return CalibrationStatusCheck(
            equipment_id=calibration.equipment_id,
            equipment_name=calibration.equipment_name,
            status=calibration.status,
            calibration_date=calibration.calibration_date,
            next_calibration_date=calibration.next_calibration_date,
            days_until_due=days_until_due,
            is_valid=calibration.is_valid,
            is_expired=calibration.is_expired,
            certificate_number=calibration.certificate_number,
            performed_by=calibration.performed_by,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting equipment calibration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dashboard & Analytics
# ============================================================================

@router.get("/dashboard", response_model=CalibrationDashboard)
def get_calibration_dashboard(
    org_id: Optional[UUID] = Query(None, description="Filter by organization ID"),
    db: Session = Depends(get_db),
):
    """
    Get calibration dashboard with summary statistics

    Provides:
    - Total calibration records
    - Count by status (valid, due_soon, expired)
    - Upcoming calibrations (this month, this quarter)
    - Compliance rate
    - Breakdown by equipment type
    """
    try:
        # Base query
        base_query = select(EquipmentCalibration)
        if org_id:
            base_query = base_query.where(EquipmentCalibration.org_id == org_id)

        # Get all calibrations
        result = db.execute(base_query)
        all_calibrations = result.scalars().all()

        # Update all statuses
        for cal in all_calibrations:
            update_calibration_status(cal)
        db.commit()

        # Calculate statistics
        total = len(all_calibrations)
        valid = sum(1 for cal in all_calibrations if cal.status == CalibrationStatus.VALID)
        due_soon = sum(1 for cal in all_calibrations if cal.status == CalibrationStatus.DUE_SOON)
        expired = sum(1 for cal in all_calibrations if cal.status == CalibrationStatus.EXPIRED)

        # Calculate compliance rate
        compliance_rate = (valid / total * 100) if total > 0 else 0.0

        # Calculate upcoming calibrations
        now = datetime.now()
        month_end = now + timedelta(days=30)
        quarter_end = now + timedelta(days=90)

        upcoming_this_month = sum(
            1 for cal in all_calibrations
            if now <= cal.next_calibration_date <= month_end
        )
        upcoming_this_quarter = sum(
            1 for cal in all_calibrations
            if now <= cal.next_calibration_date <= quarter_end
        )

        # Breakdown by status
        by_status = {
            "valid": valid,
            "due_soon": due_soon,
            "expired": expired,
        }

        # Breakdown by equipment type
        by_equipment_type = {}
        for cal in all_calibrations:
            if cal.equipment_type:
                by_equipment_type[cal.equipment_type] = by_equipment_type.get(cal.equipment_type, 0) + 1

        return CalibrationDashboard(
            total=total,
            valid=valid,
            due_soon=due_soon,
            expired=expired,
            scheduled=upcoming_this_quarter,
            compliance_rate=round(compliance_rate, 2),
            upcoming_this_month=upcoming_this_month,
            upcoming_this_quarter=upcoming_this_quarter,
            by_status=by_status,
            by_equipment_type=by_equipment_type,
        )

    except Exception as e:
        logger.exception(f"Error getting calibration dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
def health_check():
    """Health check endpoint for calibration service"""
    return {
        "status": "healthy",
        "service": "calibration-tracking",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
