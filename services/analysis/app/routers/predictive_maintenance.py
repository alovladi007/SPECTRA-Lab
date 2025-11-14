"""
Predictive Maintenance - FastAPI Routers
REST API endpoints for equipment health monitoring and failure prediction
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
from ..models.predictive_maintenance import (
    EquipmentHealth,
    MaintenanceEvent,
    EquipmentStatus,
    MaintenanceEventType
)
from ..schemas.predictive_maintenance import (
    EquipmentHealthCreate,
    EquipmentHealthUpdate,
    EquipmentHealthSchema,
    MaintenanceEventCreate,
    MaintenanceEventUpdate,
    MaintenanceEventSchema,
    PredictiveMaintenanceDashboard,
    FailurePrediction,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/predictive-maintenance", tags=["Predictive Maintenance"])


# ============================================================================
# Helper Functions
# ============================================================================

def update_equipment_status(equipment_health: EquipmentHealth):
    """Update equipment status based on health_score"""
    equipment_health.update_status()


# ============================================================================
# Equipment Health Records
# ============================================================================

@router.post("/health", response_model=EquipmentHealthSchema, status_code=status.HTTP_201_CREATED)
def create_equipment_health(
    health_record: EquipmentHealthCreate,
    db: Session = Depends(get_db),
):
    """Create a new equipment health record"""
    try:
        # Create health record
        db_health = EquipmentHealth(**health_record.model_dump())

        # Calculate and set initial status
        update_equipment_status(db_health)

        db.add(db_health)
        db.commit()
        db.refresh(db_health)

        logger.info(
            f"Created equipment health record: {db_health.id} for equipment {db_health.equipment_name}"
        )
        return db_health

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating equipment health record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=List[EquipmentHealthSchema])
def list_equipment_health(
    org_id: Optional[UUID] = Query(None, description="Filter by organization ID"),
    status_filter: Optional[EquipmentStatus] = Query(
        None, alias="status", description="Filter by status"
    ),
    equipment_type: Optional[str] = Query(None, description="Filter by equipment type"),
    equipment_id: Optional[UUID] = Query(None, description="Filter by specific equipment"),
    min_health_score: Optional[float] = Query(None, ge=0.0, le=100.0, description="Minimum health score"),
    max_health_score: Optional[float] = Query(None, ge=0.0, le=100.0, description="Maximum health score"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    sort_by: str = Query("health_score", description="Field to sort by"),
    sort_desc: bool = Query(True, description="Sort in descending order"),
    db: Session = Depends(get_db),
):
    """
    List equipment health records with filters and pagination

    Supports filtering by:
    - Organization ID
    - Equipment status (healthy, warning, critical)
    - Equipment type
    - Specific equipment ID
    - Health score range
    """
    try:
        query = select(EquipmentHealth)

        # Apply filters
        filters = []
        if org_id:
            filters.append(EquipmentHealth.org_id == org_id)
        if status_filter:
            filters.append(EquipmentHealth.status == status_filter)
        if equipment_type:
            filters.append(EquipmentHealth.equipment_type == equipment_type)
        if equipment_id:
            filters.append(EquipmentHealth.equipment_id == equipment_id)
        if min_health_score is not None:
            filters.append(EquipmentHealth.health_score >= min_health_score)
        if max_health_score is not None:
            filters.append(EquipmentHealth.health_score <= max_health_score)

        if filters:
            query = query.where(and_(*filters))

        # Sorting
        sort_column = getattr(EquipmentHealth, sort_by, EquipmentHealth.health_score)
        query = query.order_by(sort_column.desc() if sort_desc else sort_column.asc())

        # Pagination
        query = query.offset(skip).limit(limit)

        result = db.execute(query)
        health_records = result.scalars().all()

        # Update status for all records before returning
        for health in health_records:
            update_equipment_status(health)

        db.commit()

        return health_records

    except Exception as e:
        logger.exception(f"Error listing equipment health records: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health/{health_id}", response_model=EquipmentHealthSchema)
def get_equipment_health(
    health_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific equipment health record by ID"""
    try:
        query = select(EquipmentHealth).where(EquipmentHealth.id == health_id)
        result = db.execute(query)
        health_record = result.scalar_one_or_none()

        if not health_record:
            raise HTTPException(status_code=404, detail="Equipment health record not found")

        # Update status before returning
        update_equipment_status(health_record)
        db.commit()

        return health_record

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting equipment health record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/health/{health_id}", response_model=EquipmentHealthSchema)
def update_equipment_health(
    health_id: UUID,
    update_data: EquipmentHealthUpdate,
    db: Session = Depends(get_db),
):
    """Update an equipment health record"""
    try:
        query = select(EquipmentHealth).where(EquipmentHealth.id == health_id)
        result = db.execute(query)
        health_record = result.scalar_one_or_none()

        if not health_record:
            raise HTTPException(status_code=404, detail="Equipment health record not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(health_record, key, value)

        # Recalculate status if health_score changed
        if 'health_score' in update_dict:
            update_equipment_status(health_record)

        db.commit()
        db.refresh(health_record)

        logger.info(f"Updated equipment health record: {health_id}")
        return health_record

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating equipment health record: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/equipment/{equipment_id}/health", response_model=EquipmentHealthSchema)
def get_latest_equipment_health(
    equipment_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get the latest health record for a specific equipment

    Returns the most recent health record for the equipment
    """
    try:
        # Get most recent health record for this equipment
        query = (
            select(EquipmentHealth)
            .where(EquipmentHealth.equipment_id == equipment_id)
            .order_by(EquipmentHealth.created_at.desc())
            .limit(1)
        )
        result = db.execute(query)
        health_record = result.scalar_one_or_none()

        if not health_record:
            raise HTTPException(
                status_code=404,
                detail=f"No health records found for equipment {equipment_id}"
            )

        # Update status
        update_equipment_status(health_record)
        db.commit()

        return health_record

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting latest equipment health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Maintenance Events
# ============================================================================

@router.post("/events", response_model=MaintenanceEventSchema, status_code=status.HTTP_201_CREATED)
def create_maintenance_event(
    event: MaintenanceEventCreate,
    db: Session = Depends(get_db),
):
    """Create a new maintenance event"""
    try:
        # Create maintenance event
        db_event = MaintenanceEvent(**event.model_dump())

        db.add(db_event)
        db.commit()
        db.refresh(db_event)

        logger.info(
            f"Created maintenance event: {db_event.id} for equipment {db_event.equipment_id}"
        )
        return db_event

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating maintenance event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events", response_model=List[MaintenanceEventSchema])
def list_maintenance_events(
    org_id: Optional[UUID] = Query(None, description="Filter by organization ID"),
    equipment_id: Optional[UUID] = Query(None, description="Filter by equipment ID"),
    event_type: Optional[MaintenanceEventType] = Query(None, description="Filter by event type"),
    performed_date_from: Optional[datetime] = Query(None, description="Filter events from this date"),
    performed_date_to: Optional[datetime] = Query(None, description="Filter events until this date"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    sort_by: str = Query("performed_date", description="Field to sort by"),
    sort_desc: bool = Query(True, description="Sort in descending order"),
    db: Session = Depends(get_db),
):
    """
    List maintenance events with filters and pagination

    Supports filtering by:
    - Organization ID
    - Equipment ID
    - Event type (preventive, corrective, predictive, inspection)
    - Date range
    """
    try:
        query = select(MaintenanceEvent)

        # Apply filters
        filters = []
        if org_id:
            filters.append(MaintenanceEvent.org_id == org_id)
        if equipment_id:
            filters.append(MaintenanceEvent.equipment_id == equipment_id)
        if event_type:
            filters.append(MaintenanceEvent.event_type == event_type)
        if performed_date_from:
            filters.append(MaintenanceEvent.performed_date >= performed_date_from)
        if performed_date_to:
            filters.append(MaintenanceEvent.performed_date <= performed_date_to)

        if filters:
            query = query.where(and_(*filters))

        # Sorting
        sort_column = getattr(MaintenanceEvent, sort_by, MaintenanceEvent.performed_date)
        query = query.order_by(sort_column.desc() if sort_desc else sort_column.asc())

        # Pagination
        query = query.offset(skip).limit(limit)

        result = db.execute(query)
        events = result.scalars().all()

        return events

    except Exception as e:
        logger.exception(f"Error listing maintenance events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Dashboard & Analytics
# ============================================================================

@router.get("/dashboard", response_model=PredictiveMaintenanceDashboard)
def get_predictive_maintenance_dashboard(
    org_id: Optional[UUID] = Query(None, description="Filter by organization ID"),
    db: Session = Depends(get_db),
):
    """
    Get predictive maintenance dashboard with summary statistics

    Provides:
    - Total equipment being monitored
    - Count by status (healthy, warning, critical)
    - Average health score
    - Upcoming failure predictions
    - Total downtime and maintenance costs (last 30 days)
    - Breakdown by equipment type
    """
    try:
        # Base query for equipment health
        health_query = select(EquipmentHealth)
        if org_id:
            health_query = health_query.where(EquipmentHealth.org_id == org_id)

        # Get all equipment health records
        result = db.execute(health_query)
        all_health = result.scalars().all()

        # Update all statuses
        for health in all_health:
            update_equipment_status(health)
        db.commit()

        # Calculate health statistics
        total_equipment = len(all_health)
        healthy_count = sum(1 for h in all_health if h.status == EquipmentStatus.HEALTHY)
        warning_count = sum(1 for h in all_health if h.status == EquipmentStatus.WARNING)
        critical_count = sum(1 for h in all_health if h.status == EquipmentStatus.CRITICAL)

        # Calculate average health score
        avg_health_score = (
            sum(h.health_score for h in all_health) / total_equipment
            if total_equipment > 0
            else 0.0
        )

        # Calculate predictions this month
        now = datetime.now()
        month_end = now + timedelta(days=30)
        predictions_this_month = sum(
            1 for h in all_health
            if h.predicted_failure_date
            and now <= h.predicted_failure_date <= month_end
        )

        # Get maintenance events from last 30 days
        thirty_days_ago = now - timedelta(days=30)
        events_query = select(MaintenanceEvent).where(
            MaintenanceEvent.performed_date >= thirty_days_ago
        )
        if org_id:
            events_query = events_query.where(MaintenanceEvent.org_id == org_id)

        result = db.execute(events_query)
        recent_events = result.scalars().all()

        # Calculate downtime and costs
        total_downtime_hours = sum(e.downtime_hours for e in recent_events)
        total_maintenance_cost = sum(e.cost for e in recent_events)

        # Breakdown by status
        by_status = {
            "healthy": healthy_count,
            "warning": warning_count,
            "critical": critical_count,
        }

        # Breakdown by equipment type
        by_equipment_type = {}
        for health in all_health:
            if health.equipment_type:
                by_equipment_type[health.equipment_type] = (
                    by_equipment_type.get(health.equipment_type, 0) + 1
                )

        return PredictiveMaintenanceDashboard(
            total_equipment=total_equipment,
            healthy_count=healthy_count,
            warning_count=warning_count,
            critical_count=critical_count,
            avg_health_score=round(avg_health_score, 2),
            predictions_this_month=predictions_this_month,
            total_downtime_hours=round(total_downtime_hours, 2),
            total_maintenance_cost=round(total_maintenance_cost, 2),
            by_status=by_status,
            by_equipment_type=by_equipment_type,
        )

    except Exception as e:
        logger.exception(f"Error getting predictive maintenance dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions", response_model=List[FailurePrediction])
def get_failure_predictions(
    org_id: Optional[UUID] = Query(None, description="Filter by organization ID"),
    days_ahead: int = Query(90, ge=1, le=365, description="Number of days to look ahead"),
    min_probability: float = Query(0.0, ge=0.0, le=1.0, description="Minimum failure probability"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: Session = Depends(get_db),
):
    """
    Get upcoming failure predictions

    Returns equipment with predicted failures within the specified timeframe,
    sorted by predicted failure date (soonest first)
    """
    try:
        now = datetime.now()
        future_date = now + timedelta(days=days_ahead)

        # Query equipment with predicted failures in the timeframe
        query = select(EquipmentHealth).where(
            and_(
                EquipmentHealth.predicted_failure_date.isnot(None),
                EquipmentHealth.predicted_failure_date >= now,
                EquipmentHealth.predicted_failure_date <= future_date,
                EquipmentHealth.failure_probability >= min_probability,
            )
        )

        if org_id:
            query = query.where(EquipmentHealth.org_id == org_id)

        # Sort by predicted failure date (soonest first)
        query = query.order_by(EquipmentHealth.predicted_failure_date.asc())

        # Pagination
        query = query.offset(skip).limit(limit)

        result = db.execute(query)
        predictions = result.scalars().all()

        # Update statuses and calculate days until failure
        for pred in predictions:
            update_equipment_status(pred)
        db.commit()

        # Convert to FailurePrediction schema
        failure_predictions = []
        for pred in predictions:
            days_until = (
                (pred.predicted_failure_date - now).days
                if pred.predicted_failure_date
                else 0
            )
            failure_predictions.append(
                FailurePrediction(
                    equipment_id=pred.equipment_id,
                    equipment_name=pred.equipment_name,
                    equipment_type=pred.equipment_type,
                    health_score=pred.health_score,
                    predicted_failure_date=pred.predicted_failure_date,
                    days_until_failure=days_until,
                    failure_probability=pred.failure_probability,
                    confidence=pred.confidence,
                    status=pred.status,
                    recommendations=pred.recommendations,
                )
            )

        return failure_predictions

    except Exception as e:
        logger.exception(f"Error getting failure predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health-api/health")
def health_check():
    """Health check endpoint for predictive maintenance service"""
    return {
        "status": "healthy",
        "service": "predictive-maintenance",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }
