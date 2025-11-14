"""
Diffusion Manufacturing - Pydantic v2 Schemas
API validation schemas for manufacturing-grade diffusion platform
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class FurnaceType(str, Enum):
    """Diffusion furnace types"""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BATCH = "batch"
    RTP_HYBRID = "rtp_hybrid"
    LAMP_HEATED = "lamp_heated"


class DopantType(str, Enum):
    """Dopant species for diffusion"""
    BORON = "boron"
    PHOSPHORUS = "phosphorus"
    ARSENIC = "arsenic"
    ANTIMONY = "antimony"
    GALLIUM = "gallium"
    INDIUM = "indium"


class DiffusionType(str, Enum):
    """Diffusion process types"""
    PREDEPOSITION = "predeposition"
    DRIVE_IN = "drive_in"
    TWO_STEP = "two_step"
    CO_DIFFUSION = "co_diffusion"
    GETTERING = "gettering"


class DopantSource(str, Enum):
    """Dopant delivery methods"""
    SOLID_SOURCE = "solid_source"
    LIQUID_SOURCE = "liquid_source"
    GAS_SOURCE = "gas_source"
    SPIN_ON = "spin_on"
    ION_IMPLANT_ANNEAL = "ion_implant_anneal"


class AmbientGas(str, Enum):
    """Furnace ambient atmospheres"""
    N2 = "N2"
    O2 = "O2"
    AR = "Ar"
    H2 = "H2"
    N2_O2 = "N2_O2"
    FORMING_GAS = "forming_gas"


class RunStatus(str, Enum):
    """Diffusion run status"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
    BLOCKED = "blocked"
    PAUSED = "paused"


# ============================================================================
# Furnace Schemas
# ============================================================================

class DiffusionFurnaceBase(BaseModel):
    """Base schema for Diffusion Furnace"""
    name: str = Field(..., min_length=1, max_length=255, description="Furnace name")
    furnace_type: FurnaceType
    manufacturer: Optional[str] = Field(None, max_length=100)
    model: Optional[str] = Field(None, max_length=100)
    serial_number: Optional[str] = Field(None, max_length=100)

    # Physical configuration
    tube_diameter_mm: Optional[float] = Field(None, gt=0)
    tube_length_mm: Optional[float] = Field(None, gt=0)
    num_temperature_zones: int = Field(default=3, ge=1, le=10)
    max_wafer_capacity: int = Field(default=25, ge=1)
    max_temperature_c: float = Field(default=1200.0, gt=0)

    # Capabilities
    supported_dopants: List[str] = Field(
        ...,
        description="List of supported dopants",
        examples=[["boron", "phosphorus", "arsenic"]]
    )
    supported_sources: List[str] = Field(
        ...,
        description="List of supported dopant sources",
        examples=[["solid_source", "liquid_source", "gas_source"]]
    )
    supported_ambients: List[str] = Field(
        ...,
        description="List of supported ambient gases",
        examples=[["N2", "O2", "forming_gas"]]
    )
    temperature_uniformity_c: Optional[float] = Field(None, description="+/- temperature variation across zones")
    capabilities: Dict[str, Any] = Field(default_factory=dict, description="Additional capabilities")

    # Status
    is_active: bool = True
    location: Optional[str] = Field(None, max_length=255)
    description: Optional[str] = None


class DiffusionFurnaceCreate(DiffusionFurnaceBase):
    """Schema for creating a Diffusion Furnace"""
    org_id: UUID


class DiffusionFurnaceUpdate(BaseModel):
    """Schema for updating a Diffusion Furnace"""
    name: Optional[str] = None
    is_active: Optional[bool] = None
    location: Optional[str] = None
    description: Optional[str] = None
    capabilities: Optional[Dict[str, Any]] = None
    last_pm_date: Optional[datetime] = None
    next_pm_date: Optional[datetime] = None


class DiffusionFurnaceSchema(DiffusionFurnaceBase):
    """Schema for Diffusion Furnace response"""
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

class DiffusionRecipeBase(BaseModel):
    """Base schema for Diffusion Recipe"""
    name: str = Field(..., min_length=1, max_length=255)
    version: int = Field(default=1, ge=1)
    description: Optional[str] = None

    # Process type
    diffusion_type: DiffusionType
    dopant: DopantType
    dopant_source: DopantSource
    target_conductivity_type: str = Field(..., pattern="^(n-type|p-type)$")

    # Temperature profile (JSONB)
    temperature_profile: Dict[str, Any] = Field(
        ...,
        description="Temperature profile configuration",
        examples=[{
            "ramp_rate_c_per_min": 5,
            "hold_temp_c": 1000,
            "hold_time_min": 30,
            "zones": {"zone1": 1000, "zone2": 1000, "zone3": 1000}
        }]
    )

    # Ambient control
    ambient_gas: AmbientGas
    flow_rate_slm: Optional[float] = Field(None, gt=0, description="Flow rate in standard liters per minute")
    ambient_sequence: Optional[List[Dict[str, Any]]] = Field(
        None,
        description="Multi-step ambient changes",
        examples=[[
            {"step": 1, "gas": "N2", "flow_slm": 5, "duration_min": 10},
            {"step": 2, "gas": "O2", "flow_slm": 2, "duration_min": 30}
        ]]
    )

    # Process targets
    target_junction_depth_um: Optional[float] = Field(None, gt=0)
    target_sheet_resistance_ohm_per_sq: Optional[float] = Field(None, gt=0)
    target_dose_cm2: Optional[float] = Field(None, gt=0)

    # Recipe steps (JSONB)
    recipe_steps: List[Dict[str, Any]] = Field(
        ...,
        description="Sequential recipe steps",
        examples=[[
            {"step": 1, "name": "Load", "duration_min": 5, "action": "load_wafers"},
            {"step": 2, "name": "Ramp", "duration_min": 20, "action": "ramp_temperature"},
            {"step": 3, "name": "Soak", "duration_min": 30, "action": "diffuse"},
            {"step": 4, "name": "Cool", "duration_min": 40, "action": "cooldown"},
            {"step": 5, "name": "Unload", "duration_min": 5, "action": "unload_wafers"}
        ]]
    )

    # Source control (for liquid/gas sources)
    source_temperature_c: Optional[float] = None
    carrier_flow_sccm: Optional[float] = None
    bubbler_settings: Optional[Dict[str, Any]] = None

    # Safety
    safety_hazard_level: int = Field(default=1, ge=1, le=5)
    required_interlocks: Optional[List[str]] = None
    max_time_limit_min: Optional[float] = Field(None, gt=0)

    # Metadata
    status: str = Field(default="draft", description="Recipe status (draft, approved, deprecated)")

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


class DiffusionRecipeCreate(DiffusionRecipeBase):
    """Schema for creating a Diffusion Recipe"""
    furnace_id: UUID
    org_id: UUID
    created_by_id: UUID


class DiffusionRecipeUpdate(BaseModel):
    """Schema for updating a Diffusion Recipe"""
    name: Optional[str] = None
    description: Optional[str] = None
    temperature_profile: Optional[Dict[str, Any]] = None
    ambient_gas: Optional[AmbientGas] = None
    flow_rate_slm: Optional[float] = None
    ambient_sequence: Optional[List[Dict[str, Any]]] = None
    target_junction_depth_um: Optional[float] = None
    target_sheet_resistance_ohm_per_sq: Optional[float] = None
    recipe_steps: Optional[List[Dict[str, Any]]] = None
    status: Optional[str] = None


class DiffusionRecipeSchema(DiffusionRecipeBase):
    """Schema for Diffusion Recipe response"""
    id: UUID
    furnace_id: UUID
    org_id: UUID
    created_by_id: UUID
    approval_date: Optional[datetime] = None
    approved_by_id: Optional[UUID] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    # Optional nested furnace
    furnace: Optional[DiffusionFurnaceSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Run Schemas
# ============================================================================

class DiffusionRunBase(BaseModel):
    """Base schema for Diffusion Run"""
    lot_id: Optional[UUID] = None
    wafer_ids: List[str] = Field(..., min_length=1, description="List of wafer IDs")
    wafer_count: int = Field(..., ge=1, description="Number of wafers")
    boat_position_map: Optional[Dict[str, Any]] = Field(
        None,
        description="Position mapping of wafers in boat",
        examples=[{"W001": 1, "W002": 2, "W003": 3}]
    )

    # Actual process conditions
    actual_peak_temp_c: Optional[float] = None
    actual_time_at_temp_min: Optional[float] = None
    actual_ambient: Optional[str] = None

    # Metadata
    run_number: Optional[str] = Field(None, max_length=100)
    notes: Optional[str] = None

    @field_validator('wafer_ids')
    @classmethod
    def validate_unique_wafers(cls, v: List[str]) -> List[str]:
        """Ensure wafer IDs are unique"""
        if len(v) != len(set(v)):
            raise ValueError("Wafer IDs must be unique")
        return v


class DiffusionRunCreate(DiffusionRunBase):
    """Schema for creating a Diffusion Run"""
    recipe_id: UUID
    furnace_id: UUID
    org_id: UUID
    operator_id: UUID


class DiffusionRunUpdate(BaseModel):
    """Schema for updating a Diffusion Run"""
    status: Optional[RunStatus] = None
    actual_peak_temp_c: Optional[float] = None
    actual_time_at_temp_min: Optional[float] = None
    actual_ambient: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    notes: Optional[str] = None


class DiffusionRunSchema(DiffusionRunBase):
    """Schema for Diffusion Run response"""
    id: UUID
    recipe_id: UUID
    furnace_id: UUID
    org_id: UUID
    operator_id: UUID
    status: RunStatus
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    duration_seconds: Optional[float]
    celery_task_id: Optional[str]
    job_progress: Optional[float]
    error_code: Optional[str]
    error_message: Optional[str]
    created_at: datetime

    # Optional nested relationships
    recipe: Optional[DiffusionRecipeSchema] = None
    furnace: Optional[DiffusionFurnaceSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# Telemetry Schemas
# ============================================================================

class DiffusionTelemetryCreate(BaseModel):
    """Schema for creating Diffusion Telemetry point"""
    run_id: UUID
    ts: datetime

    # Temperature measurements
    temperature_zones_c: Dict[str, float] = Field(
        ...,
        description="Temperature readings from all zones",
        examples=[{"zone1": 1000.5, "zone2": 1000.2, "zone3": 999.8, "wafer_avg": 1000.2}]
    )
    temperature_setpoint_c: Optional[float] = None
    temperature_deviation_c: Optional[float] = None

    # Ambient control
    ambient_gas: Optional[str] = None
    flow_rate_slm: Optional[float] = None
    flow_rate_setpoint_slm: Optional[float] = None

    # Pressure
    chamber_pressure_torr: Optional[float] = None

    # Source control (for liquid/gas sources)
    source_temperature_c: Optional[float] = None
    carrier_flow_sccm: Optional[float] = None
    bubbler_pressure_torr: Optional[float] = None

    # Furnace position
    boat_position_mm: Optional[float] = None

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


class DiffusionTelemetrySchema(DiffusionTelemetryCreate):
    """Schema for Diffusion Telemetry response"""
    id: UUID
    org_id: UUID

    model_config = ConfigDict(from_attributes=True)


class DiffusionTelemetryBulkCreate(BaseModel):
    """Schema for bulk telemetry insert"""
    run_id: UUID
    data_points: List[DiffusionTelemetryCreate] = Field(..., min_length=1)

    @field_validator('data_points')
    @classmethod
    def validate_all_same_run(cls, v: List[DiffusionTelemetryCreate], info) -> List[DiffusionTelemetryCreate]:
        """Ensure all points have the same run_id"""
        run_id = info.data.get('run_id')
        for point in v:
            if point.run_id != run_id:
                raise ValueError("All telemetry points must have the same run_id")
        return v


# ============================================================================
# Result Schemas
# ============================================================================

class DiffusionResultBase(BaseModel):
    """Base schema for Diffusion Result"""
    wafer_id: UUID

    # Electrical measurements
    sheet_resistance_ohm_per_sq: Optional[float] = Field(None, gt=0)
    sheet_resistance_std_pct: Optional[float] = Field(None, ge=0)
    sheet_resistance_uniformity_pct: Optional[float] = Field(None, ge=0, le=100)
    sheet_resistance_map: Optional[Dict[str, Any]] = Field(
        None,
        description="2D sheet resistance map",
        examples=[{
            "map_type": "radial",
            "points": [
                {"r_mm": 0, "theta_deg": 0, "rsh_ohm_per_sq": 50.5},
                {"r_mm": 25, "theta_deg": 0, "rsh_ohm_per_sq": 49.8}
            ]
        }]
    )

    # Junction characteristics
    junction_depth_um: Optional[float] = Field(None, gt=0)
    junction_depth_std_um: Optional[float] = Field(None, ge=0)

    # Dopant profile (from SIMS/spreading resistance)
    surface_concentration_cm3: Optional[float] = None
    peak_concentration_cm3: Optional[float] = None
    dose_cm2: Optional[float] = None
    dopant_profile_uri: Optional[str] = Field(None, max_length=500)

    # Quality metrics
    uniformity_score: Optional[float] = Field(None, ge=0, le=100)
    defect_count: Optional[int] = Field(None, ge=0)
    pass_fail: Optional[bool] = None

    # Four-point probe data
    four_point_probe_data: Optional[Dict[str, Any]] = None

    # C-V measurements
    cv_measurements: Optional[Dict[str, Any]] = None

    # VM predictions
    vm_predictions: Optional[Dict[str, Any]] = None

    # SPC snapshot
    spc_snapshot: Optional[Dict[str, Any]] = None

    # References
    report_uri: Optional[str] = Field(None, max_length=500)
    raw_data_uri: Optional[str] = Field(None, max_length=500)


class DiffusionResultCreate(DiffusionResultBase):
    """Schema for creating a Diffusion Result"""
    run_id: UUID
    measurement_timestamp: Optional[datetime] = None


class DiffusionResultUpdate(BaseModel):
    """Schema for updating a Diffusion Result"""
    sheet_resistance_ohm_per_sq: Optional[float] = None
    sheet_resistance_std_pct: Optional[float] = None
    sheet_resistance_uniformity_pct: Optional[float] = None
    sheet_resistance_map: Optional[Dict[str, Any]] = None
    junction_depth_um: Optional[float] = None
    junction_depth_std_um: Optional[float] = None
    surface_concentration_cm3: Optional[float] = None
    pass_fail: Optional[bool] = None
    defect_count: Optional[int] = None


class DiffusionResultSchema(DiffusionResultBase):
    """Schema for Diffusion Result response"""
    id: UUID
    run_id: UUID
    org_id: UUID
    measurement_timestamp: Optional[datetime]
    created_at: datetime

    # Optional nested run
    run: Optional[DiffusionRunSchema] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================================================
# SPC Schemas
# ============================================================================

class DiffusionSPCSeriesBase(BaseModel):
    """Base schema for SPC Series"""
    name: str = Field(..., min_length=1, max_length=255)
    parameter: str = Field(
        ...,
        max_length=100,
        description="Parameter being tracked (e.g., 'sheet_resistance', 'junction_depth', 'uniformity')"
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


class DiffusionSPCSeriesCreate(DiffusionSPCSeriesBase):
    """Schema for creating an SPC Series"""
    recipe_id: Optional[UUID] = None
    furnace_id: Optional[UUID] = None
    org_id: UUID


class DiffusionSPCSeriesUpdate(BaseModel):
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


class DiffusionSPCSeriesSchema(DiffusionSPCSeriesBase):
    """Schema for SPC Series response"""
    id: UUID
    recipe_id: Optional[UUID]
    furnace_id: Optional[UUID]
    org_id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class DiffusionSPCPointBase(BaseModel):
    """Base schema for SPC Point"""
    value: float
    violation: bool = False
    violation_rules: Optional[List[str]] = Field(
        None,
        description="List of violated rules (e.g., ['western_electric_1', 'western_electric_2'])"
    )


class DiffusionSPCPointCreate(DiffusionSPCPointBase):
    """Schema for creating an SPC Point"""
    series_id: UUID
    run_id: Optional[UUID] = None
    ts: datetime


class DiffusionSPCPointSchema(DiffusionSPCPointBase):
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

class DiffusionRunQuery(BaseModel):
    """Schema for querying Diffusion runs"""
    org_id: Optional[UUID] = None
    furnace_id: Optional[UUID] = None
    recipe_id: Optional[UUID] = None
    status: Optional[RunStatus] = None
    lot_id: Optional[UUID] = None

    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Pagination
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=100, ge=1, le=1000)

    # Sorting
    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_desc: bool = Field(default=True, description="Sort descending")


class DiffusionAnalyticsRequest(BaseModel):
    """Schema for analytics request"""
    metric: str = Field(
        ...,
        description="Metric to analyze (e.g., 'sheet_resistance', 'junction_depth', 'uniformity')"
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


class DiffusionAnalyticsResponse(BaseModel):
    """Schema for analytics response"""
    metric: str
    aggregation: str
    data: List[Dict[str, Any]]
    summary: Dict[str, Any]


# ============================================================================
# Batch Operations
# ============================================================================

class DiffusionBatchRunCreate(BaseModel):
    """Schema for creating batch Diffusion runs"""
    recipe_id: UUID
    furnace_id: UUID
    org_id: UUID
    lot_id: UUID
    wafer_ids: List[str] = Field(..., min_length=1, max_length=100)
    operator_id: UUID

    @field_validator('wafer_ids')
    @classmethod
    def validate_unique_wafers(cls, v: List[str]) -> List[str]:
        """Ensure wafer IDs are unique"""
        if len(v) != len(set(v)):
            raise ValueError("Wafer IDs must be unique")
        return v


class DiffusionBatchRunResponse(BaseModel):
    """Schema for batch run creation response"""
    run_ids: List[UUID]
    lot_id: UUID
    total_runs: int
    status: str
