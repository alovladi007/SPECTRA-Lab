"""
SQLAlchemy models for Predictive Maintenance
Track equipment health, failure predictions, and maintenance events
"""

from sqlalchemy import Column, String, Float, Text, DateTime, Index, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
from enum import Enum
import uuid
from datetime import datetime
import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.base import Base


# Enums
class EquipmentStatus(str, Enum):
    """Equipment health status"""
    HEALTHY = "healthy"  # Good condition, health score >= 70
    WARNING = "warning"  # Needs attention, health score 40-69
    CRITICAL = "critical"  # Critical condition, health score < 40


class MaintenanceEventType(str, Enum):
    """Type of maintenance event"""
    PREVENTIVE = "preventive"  # Scheduled preventive maintenance
    CORRECTIVE = "corrective"  # Repair after failure
    PREDICTIVE = "predictive"  # Maintenance based on prediction
    INSPECTION = "inspection"  # Routine inspection


class EquipmentHealth(Base):
    """
    Equipment Health Record
    Tracks equipment health scores, failure predictions, and maintenance schedules
    """
    __tablename__ = "equipment_health"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Equipment information
    equipment_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    equipment_name = Column(String(255), nullable=False)
    equipment_type = Column(String(100), nullable=True)

    # Health metrics
    health_score = Column(Float, nullable=False)  # 0-100 scale
    predicted_failure_date = Column(DateTime(timezone=True), nullable=True, index=True)
    confidence = Column(Float, nullable=False)  # 0-1 scale
    failure_probability = Column(Float, nullable=False)  # 0-1 scale

    # Reliability metrics
    mtbf_hours = Column(Float, nullable=True)  # Mean Time Between Failures
    mttr_hours = Column(Float, nullable=True)  # Mean Time To Repair

    # Maintenance schedule
    last_maintenance_date = Column(DateTime(timezone=True), nullable=True)
    next_maintenance_date = Column(DateTime(timezone=True), nullable=True, index=True)

    # Data and analysis
    sensor_data = Column(JSONB, nullable=True)  # Latest sensor readings
    anomalies = Column(JSONB, nullable=True)  # Detected anomalies
    recommendations = Column(JSONB, nullable=True)  # Maintenance recommendations

    # Status (computed from health_score)
    status = Column(
        SQLEnum(EquipmentStatus),
        nullable=False,
        default=EquipmentStatus.HEALTHY,
        index=True
    )

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index('ix_equipment_health_org_equipment', 'org_id', 'equipment_id'),
        Index('ix_equipment_health_org_status', 'org_id', 'status'),
        Index('ix_equipment_health_predicted_failure', 'predicted_failure_date'),
    )

    @hybrid_property
    def days_until_failure(self) -> int:
        """Calculate days until predicted failure"""
        if self.predicted_failure_date:
            delta = self.predicted_failure_date - datetime.now(self.predicted_failure_date.tzinfo)
            return delta.days
        return 0

    @hybrid_property
    def days_until_maintenance(self) -> int:
        """Calculate days until next scheduled maintenance"""
        if self.next_maintenance_date:
            delta = self.next_maintenance_date - datetime.now(self.next_maintenance_date.tzinfo)
            return delta.days
        return 0

    @hybrid_property
    def is_healthy(self) -> bool:
        """Check if equipment is in healthy state"""
        return self.status == EquipmentStatus.HEALTHY

    @hybrid_property
    def is_critical(self) -> bool:
        """Check if equipment is in critical state"""
        return self.status == EquipmentStatus.CRITICAL

    def calculate_status(self) -> EquipmentStatus:
        """
        Calculate equipment status based on health_score
        Returns: EquipmentStatus enum value
        """
        if self.health_score >= 70:
            return EquipmentStatus.HEALTHY
        elif self.health_score >= 40:
            return EquipmentStatus.WARNING
        else:
            return EquipmentStatus.CRITICAL

    def update_status(self):
        """Update the status field based on current health score"""
        self.status = self.calculate_status()


class MaintenanceEvent(Base):
    """
    Maintenance Event Record
    Tracks all maintenance activities performed on equipment
    """
    __tablename__ = "maintenance_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Equipment information
    equipment_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Event details
    event_type = Column(
        SQLEnum(MaintenanceEventType),
        nullable=False,
        default=MaintenanceEventType.PREVENTIVE,
        index=True
    )
    performed_date = Column(DateTime(timezone=True), nullable=False, index=True)
    downtime_hours = Column(Float, nullable=False, default=0.0)  # Equipment downtime
    cost = Column(Float, nullable=False, default=0.0)  # Maintenance cost

    # Personnel and details
    technician = Column(String(255), nullable=True)  # Technician or team name
    description = Column(Text, nullable=True)  # Detailed description
    parts_replaced = Column(JSONB, nullable=True)  # Parts replaced during maintenance
    notes = Column(Text, nullable=True)  # Additional notes

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    # Indexes for common queries
    __table_args__ = (
        Index('ix_maintenance_events_org_equipment', 'org_id', 'equipment_id'),
        Index('ix_maintenance_events_org_type', 'org_id', 'event_type'),
        Index('ix_maintenance_events_performed_date', 'performed_date'),
    )

    @hybrid_property
    def total_cost(self) -> float:
        """Return the total cost of the maintenance event"""
        return self.cost

    @hybrid_property
    def is_planned(self) -> bool:
        """Check if this was a planned maintenance event"""
        return self.event_type in [MaintenanceEventType.PREVENTIVE, MaintenanceEventType.PREDICTIVE]

    @hybrid_property
    def is_emergency(self) -> bool:
        """Check if this was an emergency repair"""
        return self.event_type == MaintenanceEventType.CORRECTIVE
