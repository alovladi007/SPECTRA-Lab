"""
Oxidation Manufacturing - Pydantic v2 Schemas
API validation schemas for thermal oxidation manufacturing platform
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class OxidationFurnaceType(str, Enum):
    """Oxidation furnace types"""
    HORIZONTAL_TUBE = "horizontal_tube"
    VERTICAL_TUBE = "vertical_tube"
    BATCH = "batch"
    RAPID_THERMAL = "rapid_thermal"


class OxidationType(str, Enum):
    """Oxidation process types"""
    DRY = "dry"
    WET = "wet"
    STEAM = "steam"
    PYROGENIC = "pyrogenic"
    ANODIC = "anodic"


class OxideApplication(str, Enum):
    """Oxide layer applications"""
    GATE_OXIDE = "gate_oxide"
    FIELD_OXIDE = "field_oxide"
    MASK_OXIDE = "mask_oxide"
    PASSIVATION = "passivation"
    TUNNEL_OXIDE = "tunnel_oxide"


class RunStatus(str, Enum):
    """Oxidation run status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class MeasurementType(str, Enum):
    """Measurement method for oxide characterization"""
    ELLIPSOMETRY = "ellipsometry"
    REFLECTOMETRY = "reflectometry"
    CV = "cv"
    EDAX = "edax"


# ============================================================================
# Furnace Schemas
# ============================================================================

class OxidationFurnaceBase(BaseModel):
    """Base schema for Oxidation Furnace"""
    name: str = Field(..., min_length=1, max_length=255, description="Furnace name")
    furnace_type: OxidationFurnaceType
    manufacturer: Optional[str] = Field(None, max_length=100)
    model: Optional[str] = Field(None, max_length=100)
    serial_number: Optional[str] = Field(None, max_length=100)

    # Physical configuration
    tube_diameter_mm: Optional[float] = Field(None, gt=0)
    tube_length_mm: Optional[float] = Field(None, gt=0)
    num_temperature_zones: int = Field(default=1, ge=1, le=10)
    max_wafer_capacity: int = Field(default=25, ge=1)
    max_temperature_c: float = Field(default=1200.0, gt=0)

    # Oxidation capabilities
    supports_dry_oxidation: bool = Field(default=True)
    supports_wet_oxidation: bool = Field(default=False)
    supports_steam_oxidation: bool = Field(default=False)
    supports_pyrogenic: bool = Field(default=False)

    # Calibration and maintenance
    is_active: bool = True
    is_calibrated: bool = True
    temperature_uniformity_c: Optional[float] = Field(None, description="+/- temperature variation across zones")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Additional capabilities")

    # Status and metadata
    location: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


class OxidationFurnaceCreate(OxidationFurnaceBase):
    """Schema for creating an Oxidation Furnace"""
    org_id: UUID


class OxidationFurnaceUpdate(BaseModel):
    """Schema for updating an Oxidation Furnace"""
    name: Optional[str] = None
    is_active: Optional[bool] = None
    is_calibrated: Optional[bool] = None
    location: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    last_pm_date: Optional[datetime] = None
    next_pm_date: Optional[datetime] = None


class OxidationFurnaceSchema(OxidationFurnaceBase):
    """Schema for Oxidation Furnace response"""
    id: UUID
    org_id: UUID
    last_pm_date: Optional[datetime] = None
    next_pm_date: Optional[datetime] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Recipe Schemas
# ============================================================================

class OxidationRecipeBase(BaseModel):
    """Base schema for Oxidation Recipe"""
    name: str = Field(..., min_length=1, max_length=255)
    version: str = Field(default="1.0", max_length=50)
    description: Optional[str] = None

    # Process type
    oxidation_type: OxidationType
    application: Optional[OxideApplication] = None

    # Temperature profile
    temperature_c: float = Field(..., gt=0, description="Process temperature in Celsius")
    ramp_rate_c_per_min: float = Field(default=10.0, gt=0, description="Temperature ramp rate")
    time_minutes: float = Field(..., gt=0, description="Process duration in minutes")

    # Gas flows (sccm - standard cubic centimeters per minute)
    o2_flow_sccm: float = Field(default=0.0, ge=0)
    n2_flow_sccm: float = Field(default=0.0, ge=0)
    h2_flow_sccm: float = Field(default=0.0, ge=0)

    # Target specifications
    target_thickness_nm: Optional[float] = Field(None, gt=0)
    thickness_tolerance_nm: Optional[float] = Field(None, ge=0)

    # Advanced parameters (JSONB)
    temperature_profile: Optional[Dict[str, Any]] = Field(
        None,
        description="Multi-zone temperature profile",
        examples=[{
            "zones": {"zone1": 1000, "zone2": 1000, "zone3": 1000},
            "ramp_rate": 10,
            "soak_time": 30
        }]
    )

    gas_sequence: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Multi-step gas flow changes",
        examples=[[
            {"step": 1, "o2_sccm": 100, "n2_sccm": 500, "duration_min": 15},
            {"step": 2, "o2_sccm": 50, "n2_sccm": 500, "duration_min": 45}
        ]]
    )

    recipe_steps: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Sequential recipe execution steps",
        examples=[[
            {"step": 1, "name": "Load", "duration_min": 5},
            {"step": 2, "name": "Ramp", "duration_min": 20},
            {"step": 3, "name": "Soak", "duration_min": 60},
            {"step": 4, "name": "Cool", "duration_min": 30},
            {"step": 5, "name": "Unload", "duration_min": 5}
        ]]
    )

    # Safety and metadata
    safety_hazard_level: int = Field(default=1, ge=1, le=5)
    required_interlocks: Optional[List[str]] = None
    max_time_limit_min: Optional[float] = Field(None, gt=0)
    status: str = Field(default="DRAFT", description="Recipe status (DRAFT, APPROVED, DEPRECATED)")

    @field_validator('o2_flow_sccm', 'n2_flow_sccm', 'h2_flow_sccm')
    @classmethod
    def validate_flow_rates(cls, v: float) -> float:
        """Validate flow rates are non-negative"""
        if v < 0:
            raise ValueError("Flow rates must be non-negative")
        return v

    @field_validator('temperature_c', 'time_minutes')
    @classmethod
    def validate_positive_values(cls, v: float) -> float:
        """Validate temperature and time are positive"""
        if v <= 0:
            raise ValueError("Temperature and time must be positive")
        return v


class OxidationRecipeCreate(OxidationRecipeBase):
    """Schema for creating an Oxidation Recipe"""
    furnace_id: UUID
    org_id: UUID
    created_by_id: UUID


class OxidationRecipeUpdate(BaseModel):
    """Schema for updating an Oxidation Recipe"""
    name: Optional[str] = None
    description: Optional[str] = None
    temperature_c: Optional[float] = None
    ramp_rate_c_per_min: Optional[float] = None
    time_minutes: Optional[float] = None
    o2_flow_sccm: Optional[float] = None
    n2_flow_sccm: Optional[float] = None
    h2_flow_sccm: Optional[float] = None
    target_thickness_nm: Optional[float] = None
    thickness_tolerance_nm: Optional[float] = None
    temperature_profile: Optional[Dict[str, Any]] = None
    gas_sequence: Optional[List[Dict[str, Any]]] = None
    recipe_steps: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None


class OxidationRecipeSchema(OxidationRecipeBase):
    """Schema for Oxidation Recipe response"""
    id: UUID
    furnace_id: UUID
    org_id: UUID
    created_by_id: UUID
    run_count: int = Field(default=0, description="Number of times this recipe has been run")
    approval_date: Optional[datetime] = None
    approved_by_id: Optional[UUID] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Optional nested furnace
    furnace: Optional[OxidationFurnaceSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Run Schemas
# ============================================================================

class OxidationRunBase(BaseModel):
    """Base schema for Oxidation Run"""
    lot_id: Optional[str] = Field(None, max_length=100)
    wafer_ids: List[str] = Field(..., min_length=1, description="List of wafer IDs")
    wafer_count: int = Field(..., ge=1, description="Number of wafers")

    # Wafer positioning
    boat_position_map: Optional[Dict[str, Any]] = Field(
        None,
        description="Position mapping of wafers in boat",
        examples=[{"W001": 1, "W002": 2, "W003": 3}]
    )

    # Actual process conditions
    actual_temperature_c: Optional[float] = Field(None, gt=0)
    actual_time_minutes: Optional[float] = Field(None, gt=0)
    actual_thickness_nm: Optional[float] = Field(None, gt=0)

    # Measured results
    measured_thickness_nm: Optional[float] = Field(None, gt=0)
    thickness_uniformity_percent: Optional[float] = Field(None, ge=0, le=100)
    refractive_index: Optional[float] = None

    # Metadata
    run_number: Optional[str] = Field(None, max_length=100)
    operator: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None

    @field_validator('wafer_ids')
    @classmethod
    def validate_unique_wafers(cls, v: List[str]) -> List[str]:
        """Ensure wafer IDs are unique"""
        if len(v) != len(set(v)):
            raise ValueError("Wafer IDs must be unique")
        return v


class OxidationRunCreate(OxidationRunBase):
    """Schema for creating an Oxidation Run"""
    recipe_id: UUID
    furnace_id: UUID
    org_id: UUID
    operator_id: UUID


class OxidationRunUpdate(BaseModel):
    """Schema for updating an Oxidation Run"""
    status: Optional[RunStatus] = None
    actual_temperature_c: Optional[float] = None
    actual_time_minutes: Optional[float] = None
    actual_thickness_nm: Optional[float] = None
    measured_thickness_nm: Optional[float] = None
    thickness_uniformity_percent: Optional[float] = None
    refractive_index: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None


class OxidationRunSchema(OxidationRunBase):
    """Schema for Oxidation Run response"""
    id: UUID
    recipe_id: UUID
    furnace_id: UUID
    org_id: UUID
    operator_id: UUID
    status: RunStatus
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = Field(None, description="Actual duration in seconds")
    job_id: Optional[str] = Field(None, description="Celery task ID")
    job_progress: Optional[float] = Field(None, ge=0, le=100)
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Optional nested relationships
    recipe: Optional[OxidationRecipeSchema] = None
    furnace: Optional[OxidationFurnaceSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Result Schemas
# ============================================================================

class OxidationResultBase(BaseModel):
    """Base schema for Oxidation Result"""
    wafer_id: str = Field(..., max_length=100)
    measurement_type: MeasurementType

    # Thickness measurements
    thickness_nm: float = Field(..., gt=0, description="Measured oxide thickness in nm")
    thickness_std_dev: Optional[float] = Field(None, ge=0)
    uniformity_percent: Optional[float] = Field(None, ge=0, le=100)

    # Optical properties
    refractive_index: Optional[float] = Field(None, gt=0)
    extinction_coefficient: Optional[float] = Field(None, ge=0)

    # Electrical properties (for gate oxides)
    breakdown_voltage_v: Optional[float] = Field(None, gt=0)
    dielectric_constant: Optional[float] = Field(None, gt=0)
    interface_state_density: Optional[float] = Field(None, ge=0)

    # Measurement details
    measurement_points: Optional[Dict[str, Any]] = Field(
        None,
        description="Multi-point measurement data",
        examples=[{
            "points": [
                {"x_mm": 0, "y_mm": 0, "thickness_nm": 101.2},
                {"x_mm": 10, "y_mm": 0, "thickness_nm": 101.5}
            ]
        }]
    )

    measured_by: Optional[str] = Field(None, max_length=255)
    equipment_id: Optional[str] = Field(None, max_length=100)

    # Quality assessment
    uniformity_score: Optional[float] = Field(None, ge=0, le=100)
    pass_fail: Optional[bool] = None
    defect_count: Optional[int] = Field(None, ge=0)

    # References
    report_uri: Optional[str] = Field(None, max_length=500)
    raw_data_uri: Optional[str] = Field(None, max_length=500)


class OxidationResultCreate(OxidationResultBase):
    """Schema for creating an Oxidation Result"""
    run_id: UUID
    measurement_timestamp: Optional[datetime] = None


class OxidationResultUpdate(BaseModel):
    """Schema for updating an Oxidation Result"""
    thickness_nm: Optional[float] = None
    thickness_std_dev: Optional[float] = None
    uniformity_percent: Optional[float] = None
    refractive_index: Optional[float] = None
    extinction_coefficient: Optional[float] = None
    breakdown_voltage_v: Optional[float] = None
    dielectric_constant: Optional[float] = None
    interface_state_density: Optional[float] = None
    uniformity_score: Optional[float] = None
    pass_fail: Optional[bool] = None
    defect_count: Optional[int] = None
    measurement_points: Optional[Dict[str, Any]] = None


class OxidationResultSchema(OxidationResultBase):
    """Schema for Oxidation Result response"""
    id: UUID
    run_id: UUID
    org_id: UUID
    measured_at: datetime
    created_at: datetime

    # Optional nested run
    run: Optional[OxidationRunSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Telemetry Schemas
# ============================================================================

class OxidationTelemetryCreate(BaseModel):
    """Schema for creating Oxidation Telemetry point"""
    run_id: UUID
    ts: datetime

    # Temperature measurements
    temperature_zones_c: Dict[str, float] = Field(
        ...,
        description="Temperature readings from all zones",
        examples=[{"zone1": 1000.5, "zone2": 1000.2, "zone3": 999.8}]
    )
    temperature_setpoint_c: Optional[float] = None
    temperature_deviation_c: Optional[float] = None

    # Gas flow control
    o2_flow_sccm: Optional[float] = None
    n2_flow_sccm: Optional[float] = None
    h2_flow_sccm: Optional[float] = None
    total_flow_sccm: Optional[float] = None

    # Chamber conditions
    chamber_pressure_torr: Optional[float] = None

    # Derived values
    heating_rate_c_per_min: Optional[float] = None
    time_at_temperature_min: Optional[float] = None

    # Controller data
    pid_output_pct: Optional[Dict[str, float]] = Field(
        None,
        description="PID controller output per zone",
        examples=[{"zone1": 45.2, "zone2": 46.1, "zone3": 44.8}]
    )

    # Other signals
    other_signals: Optional[Dict[str, Any]] = None


class OxidationTelemetrySchema(OxidationTelemetryCreate):
    """Schema for Oxidation Telemetry response"""
    id: UUID
    org_id: UUID

    model_config = ConfigDict(from_attributes=True)


class OxidationTelemetryBulkCreate(BaseModel):
    """Schema for bulk telemetry insert"""
    run_id: UUID
    data_points: List[OxidationTelemetryCreate] = Field(..., min_length=1)

    @field_validator('data_points')
    @classmethod
    def validate_all_same_run(cls, v: List[OxidationTelemetryCreate], info) -> List[OxidationTelemetryCreate]:
        """Ensure all points have the same run_id"""
        run_id = info.data.get('run_id')
        for point in v:
            if point.run_id != run_id:
                raise ValueError("All telemetry points must have the same run_id")
        return v


# ============================================================================
# SPC Schemas
# ============================================================================

class OxidationSPCSeriesBase(BaseModel):
    """Base schema for SPC Series"""
    name: str = Field(..., min_length=1, max_length=255)
    parameter: str = Field(
        ...,
        max_length=100,
        description="Parameter being tracked (e.g., 'thickness', 'uniformity', 'refractive_index')"
    )

    # Control limits
    target: Optional[float] = None
    ucl: Optional[float] = Field(None, description="Upper Control Limit")
    lcl: Optional[float] = Field(None, description="Lower Control Limit")
    usl: Optional[float] = Field(None, description="Upper Specification Limit")
    lsl: Optional[float] = Field(None, description="Lower Specification Limit")

    # Statistics
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    cp: Optional[float] = Field(None, description="Process capability")
    cpk: Optional[float] = Field(None, description="Process capability index")

    # Status
    is_active: bool = True


class OxidationSPCSeriesCreate(OxidationSPCSeriesBase):
    """Schema for creating an SPC Series"""
    recipe_id: Optional[UUID] = None
    furnace_id: Optional[UUID] = None
    org_id: UUID


class OxidationSPCSeriesUpdate(BaseModel):
    """Schema for updating an SPC Series"""
    name: Optional[str] = None
    target: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None
    usl: Optional[float] = None
    lsl: Optional[float] = None
    mean: Optional[float] = None
    std_dev: Optional[float] = None
    cp: Optional[float] = None
    cpk: Optional[float] = None
    is_active: Optional[bool] = None


class OxidationSPCSeriesSchema(OxidationSPCSeriesBase):
    """Schema for SPC Series response"""
    id: UUID
    recipe_id: Optional[UUID]
    furnace_id: Optional[UUID]
    org_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class OxidationSPCPointBase(BaseModel):
    """Base schema for SPC Point"""
    value: float
    violation: bool = False
    violation_rules: Optional[List[str]] = Field(
        None,
        description="List of violated rules (e.g., ['western_electric_1', 'western_electric_2'])"
    )


class OxidationSPCPointCreate(OxidationSPCPointBase):
    """Schema for creating an SPC Point"""
    series_id: UUID
    run_id: Optional[UUID] = None
    ts: datetime


class OxidationSPCPointSchema(OxidationSPCPointBase):
    """Schema for SPC Point response"""
    id: UUID
    series_id: UUID
    run_id: Optional[UUID]
    org_id: UUID
    ts: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Query Schemas
# ============================================================================

class OxidationRunQuery(BaseModel):
    """Schema for querying Oxidation runs"""
    org_id: Optional[UUID] = None
    furnace_id: Optional[UUID] = None
    recipe_id: Optional[UUID] = None
    status: Optional[RunStatus] = None
    lot_id: Optional[str] = None

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Pagination
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)

    # Sorting
    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_desc: bool = Field(default=True, description="Sort descending")


class OxidationAnalyticsRequest(BaseModel):
    """Schema for analytics request"""
    metric: str = Field(
        ...,
        description="Metric to analyze (e.g., 'thickness', 'uniformity', 'refractive_index')"
    )
    aggregation: str = Field(
        default="mean",
        description="Aggregation function (mean, median, std, min, max)"
    )

    # Filters
    org_id: UUID
    furnace_id: Optional[UUID] = None
    recipe_id: Optional[UUID] = None

    start_date: datetime
    end_date: datetime

    # Grouping
    group_by: Optional[List[str]] = Field(
        None,
        description="Group by fields (e.g., ['recipe_id', 'furnace_id'])"
    )

    # Time binning
    time_bin: Optional[str] = Field(
        None,
        description="Time bin size (e.g., '1h', '1d', '1w')"
    )


class OxidationAnalyticsResponse(BaseModel):
    """Schema for analytics response"""
    metric: str
    aggregation: str
    data: List[Dict[str, Any]]
    summary: Dict[str, Any]


# ============================================================================
# Batch Operations
# ============================================================================

class OxidationBatchRunCreate(BaseModel):
    """Schema for creating batch Oxidation runs"""
    recipe_id: UUID
    furnace_id: UUID
    org_id: UUID
    lot_id: Optional[str] = None
    wafer_ids: List[str] = Field(..., min_length=1, max_length=100)
    operator_id: UUID

    @field_validator('wafer_ids')
    @classmethod
    def validate_unique_wafers(cls, v: List[str]) -> List[str]:
        """Ensure wafer IDs are unique"""
        if len(v) != len(set(v)):
            raise ValueError("Wafer IDs must be unique")
        return v


class OxidationBatchRunResponse(BaseModel):
    """Schema for batch run creation response"""
    run_ids: List[UUID]
    lot_id: Optional[str]
    total_runs: int
    status: str
