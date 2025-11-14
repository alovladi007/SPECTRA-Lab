"""
SQLAlchemy models for Diffusion Manufacturing Module
Manufacturing-grade diffusion with equipment tracking, recipes, runs, telemetry, and SPC
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.base import Base


# Enums for Diffusion classifications
class FurnaceType(str, Enum):
    """Diffusion furnace types"""
    HORIZONTAL = "horizontal"  # Horizontal tube furnace
    VERTICAL = "vertical"  # Vertical furnace
    BATCH = "batch"  # Batch processing furnace
    RTP_HYBRID = "rtp_hybrid"  # Rapid thermal + diffusion hybrid
    LAMP_HEATED = "lamp_heated"  # Lamp-heated furnace


class DopantType(str, Enum):
    """Dopant species for diffusion"""
    BORON = "boron"  # B (p-type)
    PHOSPHORUS = "phosphorus"  # P (n-type)
    ARSENIC = "arsenic"  # As (n-type)
    ANTIMONY = "antimony"  # Sb (n-type)
    GALLIUM = "gallium"  # Ga (p-type)
    INDIUM = "indium"  # In (p-type)


class DiffusionType(str, Enum):
    """Diffusion process types"""
    PREDEPOSITION = "predeposition"  # High concentration, short time
    DRIVE_IN = "drive_in"  # Lower temp, long time redistribution
    TWO_STEP = "two_step"  # Predeposition + drive-in
    CO_DIFFUSION = "co_diffusion"  # Multiple dopants
    GETTERING = "gettering"  # Impurity gettering


class DopantSource(str, Enum):
    """Dopant delivery methods"""
    SOLID_SOURCE = "solid_source"  # BN, P2O5 wafers
    LIQUID_SOURCE = "liquid_source"  # BBr3, POCl3 bubblers
    GAS_SOURCE = "gas_source"  # B2H6, PH3
    SPIN_ON = "spin_on"  # Spin-on dopant films
    ION_IMPLANT_ANNEAL = "ion_implant_anneal"  # Post-implant activation


class AmbientGas(str, Enum):
    """Furnace ambient atmospheres"""
    N2 = "N2"  # Nitrogen
    O2 = "O2"  # Oxygen (oxidizing)
    AR = "Ar"  # Argon (inert)
    H2 = "H2"  # Hydrogen (reducing)
    N2_O2 = "N2_O2"  # Nitrogen + oxygen mix
    FORMING_GAS = "forming_gas"  # N2 + H2 (5-10% H2)


class RunStatus(str, Enum):
    """Diffusion run status"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
    BLOCKED = "blocked"
    PAUSED = "paused"


class DiffusionFurnace(Base):
    """
    Diffusion furnace equipment
    Represents a physical diffusion furnace with zones and capabilities
    """
    __tablename__ = "diffusion_furnaces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    furnace_type = Column(SQLEnum(FurnaceType), nullable=False)
    manufacturer = Column(String(100), nullable=True)
    model = Column(String(100), nullable=True)
    serial_number = Column(String(100), nullable=True)

    # Physical configuration
    tube_diameter_mm = Column(Float, nullable=True)
    tube_length_mm = Column(Float, nullable=True)
    num_temperature_zones = Column(Integer, nullable=False, default=3)
    max_wafer_capacity = Column(Integer, nullable=False, default=25)
    max_temperature_c = Column(Float, nullable=False, default=1200.0)

    # Capabilities
    supported_dopants = Column(JSONB, nullable=False)  # List of DopantType
    supported_sources = Column(JSONB, nullable=False)  # List of DopantSource
    supported_ambients = Column(JSONB, nullable=False)  # List of AmbientGas
    temperature_uniformity_c = Column(Float, nullable=True)  # +/- temp variation
    capabilities = Column(JSONB, nullable=True)  # Additional capabilities

    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    last_pm_date = Column(DateTime(timezone=True), nullable=True)
    next_pm_date = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    location = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    custom_metadata = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    recipes = relationship("DiffusionRecipe", back_populates="furnace")
    runs = relationship("DiffusionRun", back_populates="furnace")

    def __repr__(self):
        return f"<DiffusionFurnace {self.name} ({self.furnace_type})>"


class DiffusionRecipe(Base):
    """
    Diffusion recipe definition
    Process recipes with temperature profiles, ambient control, and safety
    """
    __tablename__ = "diffusion_recipes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    furnace_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_furnaces.id'), nullable=False)

    # Process type
    diffusion_type = Column(SQLEnum(DiffusionType), nullable=False)
    dopant = Column(SQLEnum(DopantType), nullable=False)
    dopant_source = Column(SQLEnum(DopantSource), nullable=False)
    target_conductivity_type = Column(String(10), nullable=False)  # n-type or p-type

    # Temperature profile (JSONB for flexibility)
    temperature_profile = Column(JSONB, nullable=False)
    # Example: {"ramp_rate_c_per_min": 5, "hold_temp_c": 1000, "hold_time_min": 30, "zones": {...}}

    # Ambient control
    ambient_gas = Column(SQLEnum(AmbientGas), nullable=False)
    flow_rate_slm = Column(Float, nullable=True)  # Standard liters per minute
    ambient_sequence = Column(JSONB, nullable=True)  # Multi-step ambient changes

    # Process targets
    target_junction_depth_um = Column(Float, nullable=True)
    target_sheet_resistance_ohm_per_sq = Column(Float, nullable=True)
    target_dose_cm2 = Column(Float, nullable=True)

    # Recipe steps
    recipe_steps = Column(JSONB, nullable=False)
    # Example: [{"step": "ramp", "duration_min": 20, ...}, {"step": "soak", ...}]

    # Source control (for liquid/gas sources)
    source_temperature_c = Column(Float, nullable=True)
    carrier_flow_sccm = Column(Float, nullable=True)
    bubbler_settings = Column(JSONB, nullable=True)

    # Safety
    safety_hazard_level = Column(Integer, nullable=False, default=1)
    required_interlocks = Column(JSONB, nullable=True)
    max_time_limit_min = Column(Float, nullable=True)

    # Metadata
    status = Column(String(50), nullable=False, default='draft')
    approval_date = Column(DateTime(timezone=True), nullable=True)
    approved_by_id = Column(UUID(as_uuid=True), nullable=True)
    description = Column(Text, nullable=True)
    custom_metadata = Column(JSONB, nullable=True)

    created_by_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    furnace = relationship("DiffusionFurnace", back_populates="recipes")
    runs = relationship("DiffusionRun", back_populates="recipe")

    def __repr__(self):
        return f"<DiffusionRecipe {self.name} v{self.version} ({self.dopant}-{self.diffusion_type})>"


class DiffusionRun(Base):
    """
    Diffusion run execution
    Tracks a single diffusion process run
    """
    __tablename__ = "diffusion_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    run_number = Column(String(100), nullable=False, unique=True)

    # Relationships
    recipe_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_recipes.id'), nullable=False)
    furnace_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_furnaces.id'), nullable=False)
    lot_id = Column(UUID(as_uuid=True), nullable=True)

    # Wafer tracking
    wafer_ids = Column(JSONB, nullable=False)  # List of wafer UUIDs
    wafer_count = Column(Integer, nullable=False)
    boat_position_map = Column(JSONB, nullable=True)  # Position mapping

    # Status
    status = Column(SQLEnum(RunStatus), nullable=False, default=RunStatus.QUEUED)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)

    # Actual process conditions (may differ from recipe)
    actual_peak_temp_c = Column(Float, nullable=True)
    actual_time_at_temp_min = Column(Float, nullable=True)
    actual_ambient = Column(String(50), nullable=True)

    # Pre/post flight
    preflight_summary = Column(JSONB, nullable=True)
    postflight_summary = Column(JSONB, nullable=True)

    # Error tracking
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    fault_data = Column(JSONB, nullable=True)

    # Job queue
    celery_task_id = Column(String(255), nullable=True, index=True)
    job_progress = Column(Float, nullable=True, default=0.0)

    # Metadata
    operator_id = Column(UUID(as_uuid=True), nullable=False)
    notes = Column(Text, nullable=True)
    custom_metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    recipe = relationship("DiffusionRecipe", back_populates="runs")
    furnace = relationship("DiffusionFurnace", back_populates="runs")
    telemetry = relationship("DiffusionTelemetry", back_populates="run", cascade="all, delete-orphan")
    results = relationship("DiffusionResult", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DiffusionRun {self.run_number} ({self.status.value})>"

    @property
    def is_complete(self) -> bool:
        return self.status in [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.ABORTED]

    @property
    def is_active(self) -> bool:
        return self.status in [RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.PAUSED]


class DiffusionTelemetry(Base):
    """
    Diffusion telemetry time-series data
    High-frequency sensor data during run (temperature zones, ambient, flow)
    """
    __tablename__ = "diffusion_telemetry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_runs.id', ondelete='CASCADE'), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)

    # Temperature measurements
    temperature_zones_c = Column(JSONB, nullable=False)
    # Example: {"zone1": 1000.5, "zone2": 1000.2, "zone3": 999.8, "wafer_avg": 1000.2}
    temperature_setpoint_c = Column(Float, nullable=True)
    temperature_deviation_c = Column(Float, nullable=True)

    # Ambient control
    ambient_gas = Column(String(50), nullable=True)
    flow_rate_slm = Column(Float, nullable=True)
    flow_rate_setpoint_slm = Column(Float, nullable=True)

    # Pressure (if monitored)
    chamber_pressure_torr = Column(Float, nullable=True)

    # Source control (for liquid/gas sources)
    source_temperature_c = Column(Float, nullable=True)
    carrier_flow_sccm = Column(Float, nullable=True)
    bubbler_pressure_torr = Column(Float, nullable=True)

    # Furnace position (for vertical furnaces)
    boat_position_mm = Column(Float, nullable=True)

    # Derived values
    heating_rate_c_per_min = Column(Float, nullable=True)
    time_at_temperature_min = Column(Float, nullable=True)

    # Controller data
    pid_output_pct = Column(JSONB, nullable=True)  # PID output per zone

    # Other signals
    other_signals = Column(JSONB, nullable=True)

    # Relationships
    run = relationship("DiffusionRun", back_populates="telemetry")

    def __repr__(self):
        return f"<DiffusionTelemetry run={self.run_id} ts={self.ts}>"


class DiffusionResult(Base):
    """
    Diffusion run results
    Post-process metrology: sheet resistance, junction depth, dopant profiles
    """
    __tablename__ = "diffusion_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_runs.id'), nullable=False, index=True)
    wafer_id = Column(UUID(as_uuid=True), nullable=False)  # Individual wafer result

    # Electrical measurements
    sheet_resistance_ohm_per_sq = Column(Float, nullable=True)
    sheet_resistance_std_pct = Column(Float, nullable=True)
    sheet_resistance_uniformity_pct = Column(Float, nullable=True)
    sheet_resistance_map = Column(JSONB, nullable=True)  # 2D map

    # Junction characteristics
    junction_depth_um = Column(Float, nullable=True)
    junction_depth_std_um = Column(Float, nullable=True)

    # Dopant profile (from SIMS/spreading resistance)
    surface_concentration_cm3 = Column(Float, nullable=True)
    peak_concentration_cm3 = Column(Float, nullable=True)
    dose_cm2 = Column(Float, nullable=True)
    dopant_profile_uri = Column(String(500), nullable=True)  # Link to full SIMS data

    # Quality metrics
    uniformity_score = Column(Float, nullable=True)
    defect_count = Column(Integer, nullable=True)
    pass_fail = Column(Boolean, nullable=True)

    # Four-point probe data (if available)
    four_point_probe_data = Column(JSONB, nullable=True)

    # C-V measurements (if available)
    cv_measurements = Column(JSONB, nullable=True)

    # VM predictions
    vm_predictions = Column(JSONB, nullable=True)

    # SPC snapshot
    spc_snapshot = Column(JSONB, nullable=True)

    # References
    report_uri = Column(String(500), nullable=True)
    raw_data_uri = Column(String(500), nullable=True)

    measurement_timestamp = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    run = relationship("DiffusionRun", back_populates="results")

    def __repr__(self):
        return f"<DiffusionResult wafer={self.wafer_id} Rsh={self.sheet_resistance_ohm_per_sq}Ω/sq>"


class DiffusionSPCSeries(Base):
    """SPC chart series for diffusion parameters"""
    __tablename__ = "diffusion_spc_series"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    parameter = Column(String(100), nullable=False)
    # Common parameters: sheet_resistance, junction_depth, uniformity, temperature_stability

    recipe_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_recipes.id'), nullable=True)
    furnace_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_furnaces.id'), nullable=True)

    # Control limits
    target = Column(Float, nullable=True)
    ucl = Column(Float, nullable=True)  # Upper control limit
    lcl = Column(Float, nullable=True)  # Lower control limit
    usl = Column(Float, nullable=True)  # Upper spec limit
    lsl = Column(Float, nullable=True)  # Lower spec limit

    # Statistics
    mean = Column(Float, nullable=True)
    std_dev = Column(Float, nullable=True)
    cp = Column(Float, nullable=True)  # Process capability
    cpk = Column(Float, nullable=True)  # Process capability index

    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    points = relationship("DiffusionSPCPoint", back_populates="series", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<DiffusionSPCSeries {self.name} ({self.parameter})>"


class DiffusionSPCPoint(Base):
    """Individual SPC data point"""
    __tablename__ = "diffusion_spc_points"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    series_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_spc_series.id'), nullable=False)
    run_id = Column(UUID(as_uuid=True), ForeignKey('diffusion_runs.id'), nullable=True)

    value = Column(Float, nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    violation = Column(Boolean, nullable=False, default=False)
    violation_rules = Column(JSONB, nullable=True)  # Which rules violated

    # Relationships
    series = relationship("DiffusionSPCSeries", back_populates="points")

    def __repr__(self):
        return f"<DiffusionSPCPoint series={self.series_id} value={self.value} violation={self.violation}>"
