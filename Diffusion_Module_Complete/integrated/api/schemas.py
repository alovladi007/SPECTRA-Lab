"""
Pydantic schemas for Diffusion & Oxidation module.

Defines all data models with validation, type safety, and unit handling.
"""

from typing import Optional, Literal, Union, List, Dict, Any
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum
import numpy as np


# ============================================================================
# Enums
# ============================================================================

class DopantType(str, Enum):
    """Supported dopant types."""
    BORON = "boron"
    PHOSPHORUS = "phosphorus"
    ARSENIC = "arsenic"
    ANTIMONY = "antimony"
    CUSTOM = "custom"


class SourceType(str, Enum):
    """Diffusion source types."""
    CONSTANT = "constant"  # Surface concentration held constant
    LIMITED = "limited"    # Fixed dose, Gaussian profile
    GAUSSIAN = "gaussian"  # Alias for LIMITED
    DELTA = "delta"        # Delta function source
    CUSTOM = "custom"      # User-defined profile


class OxidationAmbient(str, Enum):
    """Oxidation ambient types."""
    DRY = "dry"    # Dry O2
    WET = "wet"    # H2O or steam
    PYROGENIC = "pyrogenic"  # H2 + O2 combustion


class BoundaryCondition(str, Enum):
    """Boundary conditions for numerical solver."""
    DIRICHLET = "dirichlet"  # Fixed value
    NEUMANN = "neumann"      # Fixed flux (usually zero)
    ROBIN = "robin"          # Mixed


class RunStatus(str, Enum):
    """Status of a simulation or experiment run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ============================================================================
# Base Schemas
# ============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )


class TimestampedSchema(BaseSchema):
    """Schema with timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


class UUIDSchema(BaseSchema):
    """Schema with UUID primary key."""
    id: UUID = Field(default_factory=uuid4)


# ============================================================================
# Dopant & Material Schemas
# ============================================================================

class DopantProperties(BaseSchema):
    """Physical properties of a dopant."""
    name: DopantType
    d0: float = Field(..., description="Pre-exponential factor (cm²/s)", gt=0)
    ea: float = Field(..., description="Activation energy (eV)", gt=0)
    k_segregation: Optional[float] = Field(None, description="Segregation coefficient", gt=0)
    
    @field_validator("d0", "ea")
    @classmethod
    def validate_positive(cls, v):
        if v <= 0:
            raise ValueError("Value must be positive")
        return v


class MaterialProperties(BaseSchema):
    """Properties of the substrate material."""
    name: str = "silicon"
    lattice_constant: float = Field(5.431, description="Lattice constant (Å)")
    density: float = Field(2.329, description="Density (g/cm³)")
    atomic_density: float = Field(5.0e22, description="Atomic density (atoms/cm³)")


# ============================================================================
# Recipe Schemas
# ============================================================================

class DiffusionRecipe(UUIDSchema, TimestampedSchema):
    """Recipe for a diffusion process step."""
    name: str = Field(..., description="Recipe name")
    description: Optional[str] = None
    
    # Process parameters
    dopant: DopantType
    temperature: float = Field(..., description="Temperature (°C)", ge=600, le=1400)
    time: float = Field(..., description="Process time (minutes)", gt=0)
    source_type: SourceType
    
    # Source parameters
    surface_concentration: Optional[float] = Field(
        None,
        description="Surface concentration for constant source (atoms/cm³)",
        gt=0
    )
    dose: Optional[float] = Field(
        None,
        description="Total dose for limited source (atoms/cm²)",
        gt=0
    )
    
    # Background doping
    background_concentration: float = Field(
        1e15,
        description="Background doping (atoms/cm³)",
        gt=0
    )
    background_type: Literal["n", "p"] = "n"
    
    # Advanced parameters
    use_concentration_dependent_d: bool = Field(
        False,
        description="Use concentration-dependent diffusivity"
    )
    concentration_exponent: float = Field(1.0, ge=0, le=4)
    concentration_factor: float = Field(1.0, gt=0)
    
    # Metadata
    tool_id: Optional[str] = None
    operator: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v):
        if not (600 <= v <= 1400):
            raise ValueError("Temperature must be between 600-1400°C")
        return v
    
    @field_validator("dose")
    @classmethod
    def validate_dose_for_limited_source(cls, v, info):
        if info.data.get("source_type") in ["limited", "gaussian"] and v is None:
            raise ValueError("Dose required for limited/gaussian source")
        return v
    
    @field_validator("surface_concentration")
    @classmethod
    def validate_surface_conc_for_constant_source(cls, v, info):
        if info.data.get("source_type") == "constant" and v is None:
            raise ValueError("Surface concentration required for constant source")
        return v


class OxidationRecipe(UUIDSchema, TimestampedSchema):
    """Recipe for a thermal oxidation process step."""
    name: str = Field(..., description="Recipe name")
    description: Optional[str] = None
    
    # Process parameters
    temperature: float = Field(..., description="Temperature (°C)", ge=700, le=1300)
    time: float = Field(..., description="Process time (minutes)", gt=0)
    ambient: OxidationAmbient
    pressure: float = Field(1.0, description="Pressure (atm)", gt=0, le=10)
    
    # Initial conditions
    initial_oxide_thickness: float = Field(
        0.0,
        description="Initial oxide thickness (nm)",
        ge=0
    )
    
    # Flags
    use_massoud_model: bool = Field(
        True,
        description="Apply Massoud thin-oxide correction"
    )
    couple_with_diffusion: bool = Field(
        False,
        description="Couple with dopant redistribution"
    )
    
    # Metadata
    tool_id: Optional[str] = None
    operator: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class CoupledRecipe(UUIDSchema, TimestampedSchema):
    """Recipe for coupled diffusion + oxidation process."""
    name: str = Field(..., description="Recipe name")
    description: Optional[str] = None
    
    diffusion_recipe: DiffusionRecipe
    oxidation_recipe: OxidationRecipe
    
    # Coupling parameters
    track_interface: bool = Field(True, description="Track moving Si/SiO₂ interface")
    apply_segregation: bool = Field(True, description="Apply segregation at interface")


# ============================================================================
# Run Schemas
# ============================================================================

class DiffusionRun(UUIDSchema, TimestampedSchema):
    """Record of a diffusion simulation or measurement."""
    recipe_id: UUID
    run_id: Optional[UUID] = None  # Link to parent experiment run
    status: RunStatus = RunStatus.PENDING
    
    # Input parameters (copied from recipe for immutability)
    dopant: DopantType
    temperature: float
    time: float
    source_type: SourceType
    
    # Computed parameters
    diffusion_length: Optional[float] = Field(None, description="Diffusion length (nm)")
    junction_depth: Optional[float] = Field(None, description="Junction depth (nm)")
    peak_concentration: Optional[float] = Field(None, description="Peak concentration (cm⁻³)")
    sheet_resistance: Optional[float] = Field(None, description="Sheet resistance (Ω/□)")
    
    # Results storage
    profile_data_uri: Optional[str] = Field(None, description="S3 URI to profile data")
    results_json: Optional[Dict[str, Any]] = None
    
    # Quality metrics
    convergence_achieved: Optional[bool] = None
    numerical_error: Optional[float] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None
    traceback: Optional[str] = None


class OxidationRun(UUIDSchema, TimestampedSchema):
    """Record of an oxidation simulation or measurement."""
    recipe_id: UUID
    run_id: Optional[UUID] = None
    status: RunStatus = RunStatus.PENDING
    
    # Input parameters
    temperature: float
    time: float
    ambient: OxidationAmbient
    pressure: float
    
    # Computed parameters
    final_thickness: Optional[float] = Field(None, description="Final oxide thickness (nm)")
    growth_rate: Optional[float] = Field(None, description="Growth rate (nm/min)")
    deal_grove_b: Optional[float] = Field(None, description="B parameter (μm²/hr)")
    deal_grove_b_a: Optional[float] = Field(None, description="B/A parameter (μm/hr)")
    
    # Results storage
    thickness_vs_time_uri: Optional[str] = None
    results_json: Optional[Dict[str, Any]] = None
    
    # Timing
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Error handling
    error_message: Optional[str] = None


# ============================================================================
# FDC & SPC Schemas
# ============================================================================

class FurnaceFDCRecord(UUIDSchema):
    """Fault Detection and Classification record from furnace."""
    run_id: UUID
    timestamp: datetime
    
    # Zone temperatures (°C)
    zone_temps: Dict[str, float] = Field(
        ...,
        description="Temperature by zone (e.g., {'zone1': 950.2, ...})"
    )
    
    # Process parameters
    pressure: float = Field(..., description="Chamber pressure (Torr)")
    gas_flows: Dict[str, float] = Field(
        ...,
        description="Gas flows (e.g., {'O2': 2.5, 'N2': 1.0}) in slpm"
    )
    
    # Boat/wafer position
    boat_position: Optional[int] = None
    wafer_slots: Optional[List[int]] = None
    
    # Alarms/warnings
    alarm_flags: Dict[str, bool] = Field(default_factory=dict)
    warning_flags: Dict[str, bool] = Field(default_factory=dict)
    
    # Computed health metrics
    zone_uniformity: Optional[float] = Field(None, description="Temperature std dev across zones")
    pressure_stability: Optional[float] = Field(None, description="Pressure variation coefficient")
    flow_balance: Optional[float] = Field(None, description="Gas flow balance metric")


class SPCPoint(UUIDSchema):
    """Statistical Process Control data point."""
    run_id: UUID
    timestamp: datetime
    
    # KPI being monitored
    kpi_name: str = Field(..., description="e.g., 'junction_depth', 'sheet_resistance', 'oxide_thickness'")
    kpi_value: float
    kpi_unit: str
    
    # Control chart parameters
    target_value: Optional[float] = None
    ucl: Optional[float] = Field(None, description="Upper control limit")
    lcl: Optional[float] = Field(None, description="Lower control limit")
    usl: Optional[float] = Field(None, description="Upper spec limit")
    lsl: Optional[float] = Field(None, description="Lower spec limit")
    
    # Statistical metrics
    cpk: Optional[float] = None
    sigma_level: Optional[float] = None
    
    # Rule violations
    rule_violations: List[str] = Field(default_factory=list)
    severity: Literal["normal", "warning", "critical"] = "normal"
    
    # Context
    tool_id: Optional[str] = None
    recipe_id: Optional[UUID] = None
    lot_id: Optional[str] = None


class ToolEvent(UUIDSchema):
    """Equipment event log."""
    tool_id: str
    timestamp: datetime
    event_type: Literal["alarm", "maintenance", "calibration", "error", "warning", "info"]
    severity: Literal["low", "medium", "high", "critical"]
    message: str
    details: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


# ============================================================================
# Analysis Result Schemas
# ============================================================================

class DiffusionProfile(BaseSchema):
    """Concentration profile from diffusion."""
    depth: List[float] = Field(..., description="Depth array (nm)")
    concentration: List[float] = Field(..., description="Concentration array (cm⁻³)")
    
    # Metadata
    dopant: DopantType
    temperature: float
    time: float
    
    # Derived metrics
    junction_depth: Optional[float] = None
    peak_concentration: Optional[float] = None
    surface_concentration: Optional[float] = None
    
    @field_validator("depth", "concentration")
    @classmethod
    def validate_equal_length(cls, v, info):
        if "depth" in info.data and "concentration" in info.data:
            if len(info.data["depth"]) != len(info.data["concentration"]):
                raise ValueError("Depth and concentration arrays must have equal length")
        return v
    
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return np.array(self.depth), np.array(self.concentration)


class OxideGrowthCurve(BaseSchema):
    """Oxide thickness vs time curve."""
    time: List[float] = Field(..., description="Time array (minutes)")
    thickness: List[float] = Field(..., description="Thickness array (nm)")
    
    # Metadata
    temperature: float
    ambient: OxidationAmbient
    pressure: float
    
    # Deal-Grove parameters
    b_parameter: Optional[float] = None
    b_a_parameter: Optional[float] = None
    
    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays."""
        return np.array(self.time), np.array(self.thickness)


class CalibrationResult(UUIDSchema, TimestampedSchema):
    """Results from parameter calibration."""
    dopant: DopantType
    tool_id: Optional[str] = None
    
    # Calibrated parameters
    d0_mean: float
    d0_std: float
    ea_mean: float
    ea_std: float
    
    # Uncertainty quantification
    d0_credible_interval: tuple[float, float] = Field(..., description="95% CI for D0")
    ea_credible_interval: tuple[float, float] = Field(..., description="95% CI for Ea")
    
    # Calibration quality
    n_samples: int
    r_squared: float
    rmse: float
    
    # Data source
    calibration_runs: List[UUID] = Field(default_factory=list)
    method: Literal["least_squares", "mcmc", "svi"] = "least_squares"


# ============================================================================
# Request/Response Schemas for API
# ============================================================================

class DiffusionSimulationRequest(BaseSchema):
    """Request to simulate diffusion."""
    recipe: Union[UUID, DiffusionRecipe] = Field(..., description="Recipe ID or full recipe")
    
    # Optional overrides
    temperature: Optional[float] = None
    time: Optional[float] = None
    
    # Output options
    return_full_profile: bool = True
    save_to_storage: bool = True


class DiffusionSimulationResponse(BaseSchema):
    """Response from diffusion simulation."""
    run_id: UUID
    status: RunStatus
    
    # Key results
    junction_depth: Optional[float] = None
    sheet_resistance: Optional[float] = None
    peak_concentration: Optional[float] = None
    
    # Optional full profile
    profile: Optional[DiffusionProfile] = None
    
    # Storage URI if saved
    profile_uri: Optional[str] = None
    
    # Timing
    duration_seconds: float


class OxidationSimulationRequest(BaseSchema):
    """Request to simulate oxidation."""
    recipe: Union[UUID, OxidationRecipe]
    
    # Optional overrides
    temperature: Optional[float] = None
    time: Optional[float] = None
    target_thickness: Optional[float] = Field(
        None,
        description="If provided, find time to reach thickness (nm)"
    )
    
    # Output options
    return_full_curve: bool = True
    save_to_storage: bool = True


class OxidationSimulationResponse(BaseSchema):
    """Response from oxidation simulation."""
    run_id: UUID
    status: RunStatus
    
    # Key results
    final_thickness: Optional[float] = None
    growth_rate: Optional[float] = None
    time_to_target: Optional[float] = Field(
        None,
        description="Time to reach target thickness (minutes)"
    )
    
    # Optional full curve
    growth_curve: Optional[OxideGrowthCurve] = None
    
    # Storage URI if saved
    curve_uri: Optional[str] = None
    
    # Timing
    duration_seconds: float


class SPCMonitorRequest(BaseSchema):
    """Request to monitor KPI with SPC rules."""
    kpi_data: List[SPCPoint]
    
    # Rule configuration
    enable_western_electric: bool = True
    enable_nelson: bool = True
    enable_cusum: bool = False
    enable_bocpd: bool = False
    
    # Chart parameters
    target: Optional[float] = None
    sigma: Optional[float] = None


class SPCMonitorResponse(BaseSchema):
    """Response from SPC monitoring."""
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    change_points: List[int] = Field(default_factory=list)
    process_capability: Optional[Dict[str, float]] = None
    recommendation: Optional[str] = None


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Enums
    "DopantType",
    "SourceType",
    "OxidationAmbient",
    "BoundaryCondition",
    "RunStatus",
    
    # Base schemas
    "BaseSchema",
    "TimestampedSchema",
    "UUIDSchema",
    
    # Properties
    "DopantProperties",
    "MaterialProperties",
    
    # Recipes
    "DiffusionRecipe",
    "OxidationRecipe",
    "CoupledRecipe",
    
    # Runs
    "DiffusionRun",
    "OxidationRun",
    
    # FDC/SPC
    "FurnaceFDCRecord",
    "SPCPoint",
    "ToolEvent",
    
    # Results
    "DiffusionProfile",
    "OxideGrowthCurve",
    "CalibrationResult",
    
    # API
    "DiffusionSimulationRequest",
    "DiffusionSimulationResponse",
    "OxidationSimulationRequest",
    "OxidationSimulationResponse",
    "SPCMonitorRequest",
    "SPCMonitorResponse",
]
