"""
SQLAlchemy models for Equipment Calibration Tracking
Track calibration records, schedules, and compliance status
"""

from sqlalchemy import Column, String, Float, Integer, Text, DateTime, Index, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
from sqlalchemy.ext.hybrid import hybrid_property
from enum import Enum
import uuid
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.base import Base


# Enums for Calibration status
class CalibrationStatus(str, Enum):
    """Calibration status based on next calibration date"""
    VALID = "valid"  # Current and within calibration period
    DUE_SOON = "due_soon"  # Within 30 days of next calibration
    EXPIRED = "expired"  # Past next calibration date


class EquipmentCalibration(Base):
    """
    Equipment Calibration Record
    Tracks calibration history and compliance for laboratory equipment
    """
    __tablename__ = "equipment_calibrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    # Equipment information
    equipment_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    equipment_name = Column(String(255), nullable=False)
    equipment_type = Column(String(100), nullable=True)

    # Calibration dates
    calibration_date = Column(DateTime(timezone=True), nullable=False)
    next_calibration_date = Column(DateTime(timezone=True), nullable=False, index=True)
    interval_days = Column(Integer, nullable=False)  # Calibration interval in days

    # Calibration details
    calibration_standard = Column(String(255), nullable=True)  # e.g., "NIST", "ISO 17025"
    performed_by = Column(String(255), nullable=False)  # Technician or company name
    certificate_number = Column(String(100), nullable=True, unique=True)

    # Status (computed from next_calibration_date)
    status = Column(
        SQLEnum(CalibrationStatus),
        nullable=False,
        default=CalibrationStatus.VALID,
        index=True
    )

    # Additional information
    notes = Column(Text, nullable=True)
    metadata_json = Column(JSONB, nullable=True)

    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)

    # Indexes for common queries
    __table_args__ = (
        Index('ix_equipment_calibrations_org_equipment', 'org_id', 'equipment_id'),
        Index('ix_equipment_calibrations_org_status', 'org_id', 'status'),
        Index('ix_equipment_calibrations_next_date', 'next_calibration_date'),
    )

    @hybrid_property
    def days_until_due(self) -> int:
        """Calculate days until calibration is due"""
        if self.next_calibration_date:
            delta = self.next_calibration_date - datetime.now(self.next_calibration_date.tzinfo)
            return delta.days
        return 0

    @hybrid_property
    def is_valid(self) -> bool:
        """Check if calibration is currently valid"""
        return self.status == CalibrationStatus.VALID

    @hybrid_property
    def is_expired(self) -> bool:
        """Check if calibration has expired"""
        return self.status == CalibrationStatus.EXPIRED

    def calculate_status(self) -> CalibrationStatus:
        """
        Calculate calibration status based on next_calibration_date
        Returns: CalibrationStatus enum value
        """
        if not self.next_calibration_date:
            return CalibrationStatus.EXPIRED

        now = datetime.now(self.next_calibration_date.tzinfo)
        days_until = (self.next_calibration_date - now).days

        if days_until < 0:
            return CalibrationStatus.EXPIRED
        elif days_until <= 30:
            return CalibrationStatus.DUE_SOON
        else:
            return CalibrationStatus.VALID

    def update_status(self):
        """Update the status field based on current date"""
        self.status = self.calculate_status()
