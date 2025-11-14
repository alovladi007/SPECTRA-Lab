"""
Predictive Maintenance - Pydantic v2 Schemas
API validation schemas for equipment health monitoring and maintenance tracking
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class EquipmentStatus(str, Enum):
    """Equipment health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"


class MaintenanceEventType(str, Enum):
    """Type of maintenance event"""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"
    INSPECTION = "inspection"


# ============================================================================
# Equipment Health Schemas
# ============================================================================

class EquipmentHealthBase(BaseModel):
    """Base schema for Equipment Health"""
    equipment_id: UUID = Field(..., description="Equipment UUID")
    equipment_name: str = Field(..., min_length=1, max_length=255, description="Equipment name")
    equipment_type: Optional[str] = Field(None, max_length=100, description="Equipment type/category")

    # Health metrics
    health_score: float = Field(..., ge=0.0, le=100.0, description="Health score (0-100)")
    predicted_failure_date: Optional[datetime] = Field(
        None,
        description="Predicted failure date based on analysis"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    failure_probability: float = Field(..., ge=0.0, le=1.0, description="Failure probability (0-1)")

    # Reliability metrics
    mtbf_hours: Optional[float] = Field(None, ge=0.0, description="Mean Time Between Failures (hours)")
    mttr_hours: Optional[float] = Field(None, ge=0.0, description="Mean Time To Repair (hours)")

    # Maintenance schedule
    last_maintenance_date: Optional[datetime] = Field(None, description="Last maintenance date")
    next_maintenance_date: Optional[datetime] = Field(None, description="Next scheduled maintenance date")

    # Data and analysis
    sensor_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Latest sensor readings",
        examples=[{
            "temperature": 75.2,
            "vibration": 0.05,
            "pressure": 101.3,
            "runtime_hours": 1250
        }]
    )
    anomalies: Optional[Dict[str, Any]] = Field(
        None,
        description="Detected anomalies",
        examples=[{
            "detected": ["vibration_spike", "temperature_variance"],
            "severity": "medium",
            "timestamp": "2025-11-13T10:30:00Z"
        }]
    )
    recommendations: Optional[Dict[str, Any]] = Field(
        None,
        description="Maintenance recommendations",
        examples=[{
            "actions": ["inspect_bearings", "check_lubrication"],
            "priority": "high",
            "estimated_cost": 500.0
        }]
    )

    @field_validator('health_score')
    @classmethod
    def validate_health_score(cls, v: float) -> float:
        """Validate health score is within range"""
        if not 0.0 <= v <= 100.0:
            raise ValueError("Health score must be between 0 and 100")
        return round(v, 2)

    @field_validator('confidence', 'failure_probability')
    @classmethod
    def validate_probability(cls, v: float) -> float:
        """Validate probability values are within range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Probability values must be between 0 and 1")
        return round(v, 4)


class EquipmentHealthCreate(EquipmentHealthBase):
    """Schema for creating an Equipment Health record"""
    org_id: UUID = Field(..., description="Organization UUID")


class EquipmentHealthUpdate(BaseModel):
    """Schema for updating an Equipment Health record"""
    equipment_name: Optional[str] = Field(None, min_length=1, max_length=255)
    equipment_type: Optional[str] = Field(None, max_length=100)
    health_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    predicted_failure_date: Optional[datetime] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    failure_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    mtbf_hours: Optional[float] = Field(None, ge=0.0)
    mttr_hours: Optional[float] = Field(None, ge=0.0)
    last_maintenance_date: Optional[datetime] = None
    next_maintenance_date: Optional[datetime] = None
    sensor_data: Optional[Dict[str, Any]] = None
    anomalies: Optional[Dict[str, Any]] = None
    recommendations: Optional[Dict[str, Any]] = None


class EquipmentHealthSchema(EquipmentHealthBase):
    """Schema for Equipment Health response"""
    id: UUID
    org_id: UUID
    status: EquipmentStatus = Field(..., description="Calculated equipment status")
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Maintenance Event Schemas
# ============================================================================

class MaintenanceEventBase(BaseModel):
    """Base schema for Maintenance Event"""
    equipment_id: UUID = Field(..., description="Equipment UUID")
    event_type: MaintenanceEventType = Field(..., description="Type of maintenance event")
    performed_date: datetime = Field(..., description="Date maintenance was performed")
    downtime_hours: float = Field(default=0.0, ge=0.0, description="Equipment downtime in hours")
    cost: float = Field(default=0.0, ge=0.0, description="Maintenance cost")

    # Personnel and details
    technician: Optional[str] = Field(None, max_length=255, description="Technician or team name")
    description: Optional[str] = Field(None, description="Detailed description of work performed")
    parts_replaced: Optional[Dict[str, Any]] = Field(
        None,
        description="Parts replaced during maintenance",
        examples=[{
            "parts": [
                {"name": "bearing", "quantity": 2, "part_number": "BRG-123"},
                {"name": "seal", "quantity": 1, "part_number": "SEAL-456"}
            ]
        }]
    )
    notes: Optional[str] = Field(None, description="Additional notes or observations")

    @field_validator('downtime_hours', 'cost')
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Validate values are non-negative"""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return round(v, 2)


class MaintenanceEventCreate(MaintenanceEventBase):
    """Schema for creating a Maintenance Event"""
    org_id: UUID = Field(..., description="Organization UUID")


class MaintenanceEventUpdate(BaseModel):
    """Schema for updating a Maintenance Event"""
    event_type: Optional[MaintenanceEventType] = None
    performed_date: Optional[datetime] = None
    downtime_hours: Optional[float] = Field(None, ge=0.0)
    cost: Optional[float] = Field(None, ge=0.0)
    technician: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None
    parts_replaced: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


class MaintenanceEventSchema(MaintenanceEventBase):
    """Schema for Maintenance Event response"""
    id: UUID
    org_id: UUID
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Dashboard Summary Schema
# ============================================================================

class PredictiveMaintenanceDashboard(BaseModel):
    """Schema for predictive maintenance dashboard statistics"""
    total_equipment: int = Field(..., description="Total number of equipment being monitored")
    healthy_count: int = Field(..., description="Number of equipment in healthy status")
    warning_count: int = Field(..., description="Number of equipment in warning status")
    critical_count: int = Field(..., description="Number of equipment in critical status")
    avg_health_score: float = Field(..., description="Average health score across all equipment")
    predictions_this_month: int = Field(
        ...,
        description="Number of failure predictions for this month"
    )
    total_downtime_hours: float = Field(
        ...,
        description="Total equipment downtime hours (last 30 days)"
    )
    total_maintenance_cost: float = Field(
        ...,
        description="Total maintenance cost (last 30 days)"
    )

    # Breakdowns
    by_status: Dict[str, int] = Field(
        default_factory=dict,
        description="Equipment count by status"
    )
    by_equipment_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Equipment count by type"
    )


# ============================================================================
# Prediction Schema
# ============================================================================

class FailurePrediction(BaseModel):
    """Schema for failure prediction details"""
    equipment_id: UUID
    equipment_name: str
    equipment_type: Optional[str] = None
    health_score: float
    predicted_failure_date: datetime
    days_until_failure: int
    failure_probability: float
    confidence: float
    status: EquipmentStatus
    recommendations: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Query Schemas
# ============================================================================

class EquipmentHealthQuery(BaseModel):
    """Schema for querying equipment health records"""
    org_id: Optional[UUID] = None
    equipment_id: Optional[UUID] = None
    status: Optional[EquipmentStatus] = None
    equipment_type: Optional[str] = None

    # Health score range
    min_health_score: Optional[float] = Field(None, ge=0.0, le=100.0)
    max_health_score: Optional[float] = Field(None, ge=0.0, le=100.0)

    # Pagination
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)

    # Sorting
    sort_by: str = Field(default="health_score", description="Field to sort by")
    sort_desc: bool = Field(default=True, description="Sort descending")


class MaintenanceEventQuery(BaseModel):
    """Schema for querying maintenance events"""
    org_id: Optional[UUID] = None
    equipment_id: Optional[UUID] = None
    event_type: Optional[MaintenanceEventType] = None

    # Date range filters
    performed_date_from: Optional[datetime] = None
    performed_date_to: Optional[datetime] = None

    # Pagination
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)

    # Sorting
    sort_by: str = Field(default="performed_date", description="Field to sort by")
    sort_desc: bool = Field(default=True, description="Sort descending")
