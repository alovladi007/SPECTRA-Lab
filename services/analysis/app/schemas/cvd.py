"""
CVD Platform - Pydantic v2 Schemas
Comprehensive API validation schemas for all CVD variants
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class PressureMode(str, Enum):
    """CVD Pressure Mode"""
    APCVD = "APCVD"  # Atmospheric Pressure
    LPCVD = "LPCVD"  # Low Pressure
    UHVCVD = "UHVCVD"  # Ultra-High Vacuum
    PECVD = "PECVD"  # Plasma-Enhanced
    HDP_CVD = "HDP_CVD"  # High-Density Plasma
    SACVD = "SACVD"  # Sub-Atmospheric


class EnergyMode(str, Enum):
    """CVD Energy Activation Mode"""
    THERMAL = "thermal"  # Pure thermal activation
    PLASMA = "plasma"  # Plasma-enhanced
    HOT_WIRE = "hot_wire"  # Hot-wire/filament
    LASER = "laser"  # Laser-assisted
    PHOTO = "photo"  # Photo-assisted
    MICROWAVE = "microwave"  # Microwave plasma
    REMOTE_PLASMA = "remote_plasma"  # Remote plasma source
    COMBUSTION = "combustion"  # Combustion CVD


class ReactorType(str, Enum):
    """CVD Reactor Configuration"""
    COLD_WALL = "cold_wall"
    HOT_WALL = "hot_wall"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    PANCAKE = "pancake"
    SHOWERHEAD = "showerhead"
    ROTATING_DISK = "rotating_disk"
    COLD_FINGER = "cold_finger"


class ChemistryType(str, Enum):
    """CVD Chemistry/Precursor Type"""
    MOCVD = "MOCVD"  # Metal-organic
    OMCVD = "OMCVD"  # Organo-metallic
    HCVD = "HCVD"  # Halide
    HYDRIDE = "hydride"  # SiH4, GeH4
    AACVD = "AACVD"  # Aerosol-assisted
    STANDARD = "standard"  # Standard chemistry
    ORGANOMETALLIC = "organometallic"  # Metal alkyls
    HALIDE = "halide"  # Chlorides, fluorides


class RunStatus(str, Enum):
    """CVD Run Status"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
    BLOCKED = "blocked"
    PAUSED = "paused"


class AlarmSeverity(str, Enum):
    """Alarm Severity Level"""
    INFO = "INFO"
    WARNING = "WARNING"
    ALARM = "ALARM"
    CRITICAL = "CRITICAL"


# ============================================================================
# Process Mode Schemas
# ============================================================================

class CVDProcessModeBase(BaseModel):
    """Base schema for CVD Process Mode"""
    name: str = Field(..., min_length=1, max_length=255, description="Process mode name")
    pressure_mode: PressureMode
    energy_mode: EnergyMode
    reactor_type: ReactorType
    chemistry_type: ChemistryType
    variant: Optional[str] = Field(None, max_length=100, description="Specific variant name (e.g., 'MOCVD', 'PECVD-ICP')")
    description: Optional[str] = Field(None, description="Process mode description")

    # Capabilities (includes ranges, materials, etc.)
    capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Process capabilities including pressure_range_pa, temperature_range_c, materials, etc."
    )


class CVDProcessModeCreate(CVDProcessModeBase):
    """Schema for creating a CVD Process Mode"""
    org_id: UUID


class CVDProcessModeUpdate(BaseModel):
    """Schema for updating a CVD Process Mode"""
    name: Optional[str] = None
    variant: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None


class CVDProcessModeSchema(CVDProcessModeBase):
    """Schema for CVD Process Mode response"""
    id: UUID
    org_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Recipe Schemas
# ============================================================================

class CVDRecipeBase(BaseModel):
    """Base schema for CVD Recipe"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    film_target: str = Field(..., max_length=100, description="Target film material (e.g., 'Si3N4', 'SiO2')")

    # Temperature profile (JSONB)
    temperature_profile: Dict[str, Any] = Field(
        ...,
        description="Temperature profile configuration",
        examples=[{
            "zones": [
                {"zone": 1, "setpoint_c": 650, "ramp_rate_c_per_min": 10},
                {"zone": 2, "setpoint_c": 650, "ramp_rate_c_per_min": 10}
            ],
            "soak_time_s": 300
        }]
    )

    # Gas flows (JSONB)
    gas_flows: Dict[str, Any] = Field(
        ...,
        description="Gas flow configuration",
        examples=[{
            "gases": [
                {"name": "SiH4", "flow_sccm": 100, "mfc_id": "MFC1"},
                {"name": "NH3", "flow_sccm": 50, "mfc_id": "MFC2"}
            ],
            "carrier_gas": "N2",
            "carrier_flow_sccm": 1000
        }]
    )

    # Pressure setpoints (JSONB)
    pressure_setpoints: Dict[str, Any] = Field(
        ...,
        description="Pressure setpoint configuration",
        examples=[{
            "base_pressure_pa": 50,
            "process_pressure_pa": 200,
            "throttle_valve_position_pct": 45
        }]
    )

    # Plasma settings (optional, for PECVD/HDPCVD)
    plasma_settings: Optional[Dict[str, Any]] = Field(
        None,
        description="Plasma configuration (RF/ICP power, frequency, etc.)",
        examples=[{
            "rf_power_w": 300,
            "frequency_mhz": 13.56,
            "bias_voltage_v": -150,
            "matching_mode": "auto"
        }]
    )

    # Recipe steps (JSONB)
    recipe_steps: List[Dict[str, Any]] = Field(
        ...,
        description="Sequential recipe steps",
        examples=[[
            {"step": 1, "name": "Pumpdown", "duration_s": 60, "action": "evacuate"},
            {"step": 2, "name": "Heatup", "duration_s": 300, "action": "ramp_temperature"},
            {"step": 3, "name": "Stabilize", "duration_s": 120, "action": "stabilize"},
            {"step": 4, "name": "Deposition", "duration_s": 600, "action": "deposit"},
            {"step": 5, "name": "Cooldown", "duration_s": 300, "action": "cool"}
        ]]
    )

    # Process parameters
    thickness_target_nm: float = Field(..., gt=0, description="Target film thickness in nm")
    uniformity_target_pct: Optional[float] = Field(None, gt=0, le=100, description="Target uniformity percentage")

    # Metadata
    tags: List[str] = Field(default_factory=list, description="Recipe tags for searching")
    version: int = Field(default=1, description="Recipe version number")

    @field_validator('recipe_steps')
    @classmethod
    def validate_steps(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate recipe steps are sequential"""
        if not v:
            raise ValueError("Recipe must have at least one step")

        step_numbers = [step.get('step', 0) for step in v]
        if step_numbers != sorted(step_numbers):
            raise ValueError("Recipe steps must be in sequential order")

        return v


class CVDRecipeCreate(CVDRecipeBase):
    """Schema for creating a CVD Recipe"""
    process_mode_id: UUID
    org_id: UUID


class CVDRecipeUpdate(BaseModel):
    """Schema for updating a CVD Recipe"""
    name: Optional[str] = None
    description: Optional[str] = None
    film_target: Optional[str] = None
    temperature_profile: Optional[Dict[str, Any]] = None
    gas_flows: Optional[Dict[str, Any]] = None
    pressure_setpoints: Optional[Dict[str, Any]] = None
    plasma_settings: Optional[Dict[str, Any]] = None
    recipe_steps: Optional[List[Dict[str, Any]]] = None
    thickness_target_nm: Optional[float] = None
    uniformity_target_pct: Optional[float] = None
    tags: Optional[List[str]] = None
    version: Optional[int] = None


class CVDRecipeSchema(CVDRecipeBase):
    """Schema for CVD Recipe response"""
    id: UUID
    process_mode_id: UUID
    org_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    created_by_id: UUID

    # Optional nested process mode
    process_mode: Optional[CVDProcessModeSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Run Schemas
# ============================================================================

class CVDRunBase(BaseModel):
    """Base schema for CVD Run"""
    lot_id: Optional[str] = Field(None, max_length=100)
    wafer_ids: List[str] = Field(default_factory=list, description="List of wafer IDs in this run")
    operator_id: Optional[UUID] = None

    # Actual conditions (may differ from recipe)
    actual_temperature_c: Optional[float] = None
    actual_pressure_pa: Optional[float] = None
    actual_time_s: Optional[float] = None

    # Run metadata
    run_number: Optional[str] = Field(None, max_length=50)
    notes: Optional[str] = None


class CVDRunCreate(CVDRunBase):
    """Schema for creating a CVD Run"""
    cvd_recipe_id: UUID
    process_mode_id: UUID
    instrument_id: UUID
    org_id: UUID


class CVDRunUpdate(BaseModel):
    """Schema for updating a CVD Run"""
    status: Optional[RunStatus] = None
    actual_temperature_c: Optional[float] = None
    actual_pressure_pa: Optional[float] = None
    actual_time_s: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    notes: Optional[str] = None


class CVDRunSchema(CVDRunBase):
    """Schema for CVD Run response"""
    id: UUID
    cvd_recipe_id: UUID
    process_mode_id: UUID
    instrument_id: UUID
    org_id: UUID
    status: RunStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    created_at: datetime

    # Optional nested relationships
    recipe: Optional[CVDRecipeSchema] = None
    process_mode: Optional[CVDProcessModeSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Telemetry Schemas
# ============================================================================

class CVDTelemetryCreate(BaseModel):
    """Schema for creating CVD Telemetry point"""
    run_id: UUID
    timestamp: datetime

    # Sensor readings (JSONB)
    temperatures: Dict[str, float] = Field(
        ...,
        description="Temperature readings from all zones",
        examples=[{"zone_1": 650.2, "zone_2": 649.8, "wafer": 645.5}]
    )
    pressures: Dict[str, float] = Field(
        ...,
        description="Pressure readings",
        examples=[{"chamber": 198.5, "foreline": 1200}]
    )
    gas_flows: Dict[str, float] = Field(
        ...,
        description="Gas flow readings",
        examples=[{"sih4": 99.8, "nh3": 50.1, "n2": 1005.2}]
    )

    # Optional sensor data
    plasma_parameters: Optional[Dict[str, float]] = Field(
        None,
        description="Plasma parameters (power, voltage, current)",
        examples=[{"rf_power_w": 298.5, "dc_bias_v": -152.3, "reflected_power_w": 2.1}]
    )
    rotation_speed_rpm: Optional[float] = None

    # Derived/calculated values
    deposition_rate_nm_per_min: Optional[float] = None
    uniformity_pct: Optional[float] = None


class CVDTelemetrySchema(CVDTelemetryCreate):
    """Schema for CVD Telemetry response"""
    id: UUID

    model_config = ConfigDict(from_attributes=True)


class CVDTelemetryBulkCreate(BaseModel):
    """Schema for bulk telemetry insert"""
    run_id: UUID
    data_points: List[CVDTelemetryCreate] = Field(..., min_length=1)

    @field_validator('data_points')
    @classmethod
    def validate_all_same_run(cls, v: List[CVDTelemetryCreate], info) -> List[CVDTelemetryCreate]:
        """Ensure all points have the same run_id"""
        run_id = info.data.get('run_id')
        for point in v:
            if point.run_id != run_id:
                raise ValueError("All telemetry points must have the same run_id")
        return v


# ============================================================================
# Result Schemas
# ============================================================================

class CVDResultBase(BaseModel):
    """Base schema for CVD Result"""
    wafer_id: str = Field(..., max_length=100)

    # Metrology results
    thickness_nm: Optional[float] = Field(None, gt=0)
    thickness_std_nm: Optional[float] = Field(None, ge=0)
    uniformity_pct: Optional[float] = Field(None, ge=0, le=100)

    # Thickness map (optional, for full wafer mapping)
    thickness_map: Optional[Dict[str, Any]] = Field(
        None,
        description="2D thickness map data",
        examples=[{
            "map_type": "radial",
            "points": [
                {"r_mm": 0, "theta_deg": 0, "thickness_nm": 100.5},
                {"r_mm": 25, "theta_deg": 0, "thickness_nm": 99.8}
            ]
        }]
    )

    # Additional metrology
    refractive_index: Optional[float] = None
    stress_mpa: Optional[float] = None
    composition: Optional[Dict[str, float]] = Field(
        None,
        description="Film composition (e.g., {'Si': 0.57, 'N': 0.43})"
    )

    # Quality flags
    pass_fail: Optional[bool] = None
    defect_count: Optional[int] = Field(None, ge=0)

    # VM predictions
    vm_predicted_thickness_nm: Optional[float] = None
    vm_confidence: Optional[float] = Field(None, ge=0, le=1)


class CVDResultCreate(CVDResultBase):
    """Schema for creating a CVD Result"""
    run_id: UUID
    measurement_timestamp: datetime


class CVDResultUpdate(BaseModel):
    """Schema for updating a CVD Result"""
    thickness_nm: Optional[float] = None
    thickness_std_nm: Optional[float] = None
    uniformity_pct: Optional[float] = None
    thickness_map: Optional[Dict[str, Any]] = None
    refractive_index: Optional[float] = None
    stress_mpa: Optional[float] = None
    composition: Optional[Dict[str, float]] = None
    pass_fail: Optional[bool] = None
    defect_count: Optional[int] = None


class CVDResultSchema(CVDResultBase):
    """Schema for CVD Result response"""
    id: UUID
    run_id: UUID
    measurement_timestamp: datetime
    created_at: datetime
    updated_at: datetime

    # Optional nested run
    run: Optional[CVDRunSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# SPC Schemas
# ============================================================================

class CVDSPCSeriesBase(BaseModel):
    """Base schema for SPC Series"""
    metric_name: str = Field(..., max_length=100, description="Metric being tracked (e.g., 'thickness', 'uniformity')")
    chart_type: str = Field(..., max_length=50, description="Control chart type (e.g., 'xbar', 'ewma', 'cusum')")

    # Control limits
    ucl: float = Field(..., description="Upper Control Limit")
    lcl: float = Field(..., description="Lower Control Limit")
    center_line: float = Field(..., description="Center line (target or mean)")

    usl: Optional[float] = Field(None, description="Upper Specification Limit")
    lsl: Optional[float] = Field(None, description="Lower Specification Limit")

    # Chart parameters
    subgroup_size: int = Field(default=1, ge=1)
    lambda_ewma: Optional[float] = Field(None, ge=0, le=1, description="EWMA smoothing parameter")
    k_cusum: Optional[float] = Field(None, gt=0, description="CUSUM reference value")

    # Status
    is_active: bool = True


class CVDSPCSeriesCreate(CVDSPCSeriesBase):
    """Schema for creating an SPC Series"""
    cvd_recipe_id: Optional[UUID] = None
    process_mode_id: Optional[UUID] = None
    org_id: UUID


class CVDSPCSeriesUpdate(BaseModel):
    """Schema for updating an SPC Series"""
    ucl: Optional[float] = None
    lcl: Optional[float] = None
    center_line: Optional[float] = None
    usl: Optional[float] = None
    lsl: Optional[float] = None
    is_active: Optional[bool] = None


class CVDSPCSeriesSchema(CVDSPCSeriesBase):
    """Schema for SPC Series response"""
    id: UUID
    cvd_recipe_id: Optional[UUID]
    process_mode_id: Optional[UUID]
    org_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class CVDSPCPointBase(BaseModel):
    """Base schema for SPC Point"""
    value: float
    subgroup_id: Optional[str] = Field(None, max_length=100)

    # Violations
    out_of_control: bool = False
    violation_rules: List[str] = Field(
        default_factory=list,
        description="List of violated rules (e.g., ['rule_1', 'rule_2'])"
    )


class CVDSPCPointCreate(CVDSPCPointBase):
    """Schema for creating an SPC Point"""
    series_id: UUID
    run_id: Optional[UUID] = None
    timestamp: datetime


class CVDSPCPointSchema(CVDSPCPointBase):
    """Schema for SPC Point response"""
    id: UUID
    series_id: UUID
    run_id: Optional[UUID]
    timestamp: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Advanced Query Schemas
# ============================================================================

class CVDRunQuery(BaseModel):
    """Schema for querying CVD runs"""
    org_id: Optional[UUID] = None
    process_mode_id: Optional[UUID] = None
    cvd_recipe_id: Optional[UUID] = None
    instrument_id: Optional[UUID] = None
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


class CVDAnalyticsRequest(BaseModel):
    """Schema for analytics request"""
    metric: str = Field(..., description="Metric to analyze (e.g., 'thickness', 'uniformity')")
    aggregation: str = Field(default="mean", description="Aggregation function (mean, median, std, min, max)")

    # Filters
    org_id: UUID
    process_mode_id: Optional[UUID] = None
    cvd_recipe_id: Optional[UUID] = None
    instrument_id: Optional[UUID] = None

    start_date: datetime
    end_date: datetime

    # Grouping
    group_by: Optional[List[str]] = Field(
        None,
        description="Group by fields (e.g., ['cvd_recipe_id', 'instrument_id'])"
    )

    # Time binning
    time_bin: Optional[str] = Field(
        None,
        description="Time bin size (e.g., '1h', '1d', '1w')"
    )


class CVDAnalyticsResponse(BaseModel):
    """Schema for analytics response"""
    metric: str
    aggregation: str
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Analytics results",
        examples=[[
            {"timestamp": "2024-01-01T00:00:00Z", "value": 100.5, "count": 25},
            {"timestamp": "2024-01-02T00:00:00Z", "value": 101.2, "count": 30}
        ]]
    )
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary statistics",
        examples=[{
            "total_count": 55,
            "mean": 100.85,
            "std": 2.3,
            "min": 95.2,
            "max": 105.8
        }]
    )


# ============================================================================
# Alarm Schemas
# ============================================================================

class CVDAlarmBase(BaseModel):
    """Base schema for CVD Alarm"""
    severity: AlarmSeverity
    alarm_type: str = Field(..., max_length=100, description="Alarm type (e.g., 'temperature_high', 'pressure_low')")
    message: str = Field(..., description="Alarm message")

    # Context
    parameter_name: Optional[str] = Field(None, max_length=100)
    parameter_value: Optional[float] = None
    threshold_value: Optional[float] = None


class CVDAlarmCreate(CVDAlarmBase):
    """Schema for creating a CVD Alarm"""
    run_id: Optional[UUID] = None
    instrument_id: Optional[UUID] = None
    timestamp: datetime


class CVDAlarmSchema(CVDAlarmBase):
    """Schema for CVD Alarm response"""
    id: UUID
    run_id: Optional[UUID]
    instrument_id: Optional[UUID]
    timestamp: datetime
    acknowledged: bool
    acknowledged_by: Optional[UUID]
    acknowledged_at: Optional[datetime]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class CVDAlarmAcknowledge(BaseModel):
    """Schema for acknowledging an alarm"""
    alarm_id: UUID
    acknowledged_by: UUID
    notes: Optional[str] = None


# ============================================================================
# Control Action Schemas
# ============================================================================

class CVDControlActionBase(BaseModel):
    """Base schema for Control Action (R2R, FDC)"""
    controller_type: str = Field(..., max_length=50, description="Controller type (e.g., 'R2R_EWMA', 'MPC', 'PID')")
    action_type: str = Field(..., max_length=50, description="Action type (e.g., 'adjust_temperature', 'adjust_time')")

    # Target and adjustment
    target_value: float
    measured_value: float
    error: float

    # Control action
    adjustment: Dict[str, Any] = Field(
        ...,
        description="Adjustment parameters",
        examples=[{
            "parameter": "process_time_s",
            "old_value": 600,
            "new_value": 620,
            "delta": 20
        }]
    )

    # Applied?
    applied: bool = False
    applied_at: Optional[datetime] = None


class CVDControlActionCreate(CVDControlActionBase):
    """Schema for creating a Control Action"""
    run_id: UUID
    cvd_recipe_id: UUID
    timestamp: datetime


class CVDControlActionSchema(CVDControlActionBase):
    """Schema for Control Action response"""
    id: UUID
    run_id: UUID
    cvd_recipe_id: UUID
    timestamp: datetime
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Batch Operations
# ============================================================================

class CVDBatchRunCreate(BaseModel):
    """Schema for creating batch CVD runs"""
    cvd_recipe_id: UUID
    process_mode_id: UUID
    instrument_id: UUID
    org_id: UUID
    lot_id: str
    wafer_ids: List[str] = Field(..., min_length=1, max_length=25, description="List of wafer IDs (max 25)")
    operator_id: Optional[UUID] = None

    @field_validator('wafer_ids')
    @classmethod
    def validate_unique_wafers(cls, v: List[str]) -> List[str]:
        """Ensure wafer IDs are unique"""
        if len(v) != len(set(v)):
            raise ValueError("Wafer IDs must be unique")
        return v


class CVDBatchRunResponse(BaseModel):
    """Schema for batch run creation response"""
    run_ids: List[UUID]
    lot_id: str
    total_runs: int
    status: str


# ============================================================================
# Export Schemas
# ============================================================================

class CVDExportRequest(BaseModel):
    """Schema for data export request"""
    export_type: str = Field(..., description="Export type (e.g., 'runs', 'telemetry', 'results', 'spc')")
    format: str = Field(default="csv", description="Export format (csv, json, parquet)")

    # Filters
    org_id: UUID
    start_date: datetime
    end_date: datetime

    cvd_recipe_ids: Optional[List[UUID]] = None
    instrument_ids: Optional[List[UUID]] = None

    # Options
    include_telemetry: bool = False
    include_results: bool = True


class CVDExportResponse(BaseModel):
    """Schema for export response"""
    export_id: UUID
    status: str
    file_url: Optional[str] = None
    file_size_bytes: Optional[int] = None
    record_count: int
    created_at: datetime
    expires_at: datetime
