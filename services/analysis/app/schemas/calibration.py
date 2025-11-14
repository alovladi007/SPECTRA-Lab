"""
Calibration - Pydantic v2 Schemas
API validation schemas for equipment calibration tracking
"""

from datetime import datetime
from typing import Any, Dict, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class CalibrationStatus(str, Enum):
    """Calibration status based on next calibration date"""
    VALID = "valid"
    DUE_SOON = "due_soon"
    EXPIRED = "expired"


# ============================================================================
# Calibration Schemas
# ============================================================================

class CalibrationBase(BaseModel):
    """Base schema for Calibration"""
    equipment_id: UUID = Field(..., description="Equipment UUID")
    equipment_name: str = Field(..., min_length=1, max_length=255, description="Equipment name")
    equipment_type: Optional[str] = Field(None, max_length=100, description="Equipment type/category")

    # Calibration dates
    calibration_date: datetime = Field(..., description="Date calibration was performed")
    next_calibration_date: datetime = Field(..., description="Next scheduled calibration date")
    interval_days: int = Field(..., gt=0, description="Calibration interval in days")

    # Calibration details
    calibration_standard: Optional[str] = Field(
        None,
        max_length=255,
        description="Calibration standard (e.g., NIST, ISO 17025)"
    )
    performed_by: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Technician or company that performed calibration"
    )
    certificate_number: Optional[str] = Field(
        None,
        max_length=100,
        description="Calibration certificate number"
    )

    # Additional information
    notes: Optional[str] = Field(None, description="Additional notes or remarks")
    metadata_json: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata as JSON",
        examples=[{
            "temperature_range": {"min": -20, "max": 100},
            "accuracy": "±0.1°C",
            "lab_name": "National Metrology Lab"
        }]
    )

    @field_validator('interval_days')
    @classmethod
    def validate_interval(cls, v: int) -> int:
        """Validate calibration interval is reasonable"""
        if v < 1:
            raise ValueError("Calibration interval must be at least 1 day")
        if v > 3650:  # 10 years
            raise ValueError("Calibration interval cannot exceed 10 years (3650 days)")
        return v

    @field_validator('next_calibration_date')
    @classmethod
    def validate_next_date(cls, v: datetime, info) -> datetime:
        """Validate next calibration date is after calibration date"""
        calibration_date = info.data.get('calibration_date')
        if calibration_date and v <= calibration_date:
            raise ValueError("Next calibration date must be after calibration date")
        return v


class CalibrationCreate(CalibrationBase):
    """Schema for creating a Calibration record"""
    org_id: UUID = Field(..., description="Organization UUID")


class CalibrationUpdate(BaseModel):
    """Schema for updating a Calibration record"""
    equipment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    equipment_type: Optional[str] = Field(None, max_length=100)
    calibration_date: Optional[datetime] = None
    next_calibration_date: Optional[datetime] = None
    interval_days: Optional[int] = Field(None, gt=0)
    calibration_standard: Optional[str] = Field(None, max_length=255)
    performed_by: Optional[str] = Field(None, min_length=1, max_length=255)
    certificate_number: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None
    metadata_json: Optional[Dict[str, Any]] = None


class CalibrationSchema(CalibrationBase):
    """Schema for Calibration response"""
    id: UUID
    org_id: UUID
    status: CalibrationStatus = Field(..., description="Calculated calibration status")
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Status Check Schema
# ============================================================================

class CalibrationStatusCheck(BaseModel):
    """Schema for equipment calibration status check"""
    equipment_id: UUID
    equipment_name: str
    status: CalibrationStatus
    calibration_date: datetime
    next_calibration_date: datetime
    days_until_due: int = Field(..., description="Days until calibration is due")
    is_valid: bool = Field(..., description="Whether calibration is currently valid")
    is_expired: bool = Field(..., description="Whether calibration has expired")
    certificate_number: Optional[str] = None
    performed_by: str

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Dashboard Summary Schema
# ============================================================================

class CalibrationDashboard(BaseModel):
    """Schema for calibration dashboard statistics"""
    total: int = Field(..., description="Total calibration records")
    valid: int = Field(..., description="Number of valid calibrations")
    due_soon: int = Field(..., description="Number of calibrations due soon (within 30 days)")
    expired: int = Field(..., description="Number of expired calibrations")
    scheduled: int = Field(..., description="Number of calibrations scheduled in next 90 days")

    # Additional statistics
    compliance_rate: float = Field(..., description="Percentage of equipment in valid status")
    upcoming_this_month: int = Field(..., description="Calibrations due this month")
    upcoming_this_quarter: int = Field(..., description="Calibrations due this quarter")

    # Optional breakdown by equipment type
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown by status"
    )
    by_equipment_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown by equipment type"
    )


# ============================================================================
# Query Schemas
# ============================================================================

class CalibrationQuery(BaseModel):
    """Schema for querying calibrations"""
    org_id: Optional[UUID] = None
    equipment_id: Optional[UUID] = None
    status: Optional[CalibrationStatus] = None
    equipment_type: Optional[str] = None

    # Date range filters
    calibration_date_from: Optional[datetime] = None
    calibration_date_to: Optional[datetime] = None
    next_calibration_date_from: Optional[datetime] = None
    next_calibration_date_to: Optional[datetime] = None

    # Pagination
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)

    # Sorting
    sort_by: str = Field(default="next_calibration_date", description="Field to sort by")
    sort_desc: bool = Field(default=False, description="Sort descending")
