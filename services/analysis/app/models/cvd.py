"""
SQLAlchemy models for CVD module
Supports all CVD variants with unified data model
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any

from ..database import Base


# Enums for CVD classifications
class PressureMode(str, Enum):
    """CVD pressure regimes"""
    APCVD = "APCVD"  # Atmospheric pressure
    LPCVD = "LPCVD"  # Low pressure
    UHVCVD = "UHVCVD"  # Ultra-high vacuum
    PECVD = "PECVD"  # Plasma-enhanced
    HDP_CVD = "HDP_CVD"  # High-density plasma
    SACVD = "SACVD"  # Sub-atmospheric


class EnergyMode(str, Enum):
    """CVD energy/activation methods"""
    THERMAL = "thermal"
    PLASMA = "plasma"
    HOT_WIRE = "hot_wire"
    LASER = "laser"
    PHOTO = "photo"
    MICROWAVE = "microwave"
    REMOTE_PLASMA = "remote_plasma"
    COMBUSTION = "combustion"


class ReactorType(str, Enum):
    """CVD reactor configurations"""
    COLD_WALL = "cold_wall"
    HOT_WALL = "hot_wall"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    PANCAKE = "pancake"
    SHOWERHEAD = "showerhead"
    ROTATING_DISK = "rotating_disk"
    COLD_FINGER = "cold_finger"


class ChemistryType(str, Enum):
    """CVD chemistry classifications"""
    MOCVD = "MOCVD"  # Metal-organic
    OMCVD = "OMCVD"  # Organo-metallic
    HCVD = "HCVD"  # Halide
    HYDRIDE = "hydride"
    AACVD = "AACVD"  # Aerosol-assisted
    STANDARD = "standard"
    ORGANOMETALLIC = "organometallic"
    HALIDE = "halide"


class RunStatus(str, Enum):
    """CVD run status"""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"
    BLOCKED = "blocked"
    PAUSED = "paused"


class CVDProcessMode(Base):
    """
    CVD process mode definition
    Represents a specific combination of pressure/energy/reactor/chemistry
    """
    __tablename__ = "cvd_process_modes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    pressure_mode = Column(SQLEnum(PressureMode), nullable=False)
    energy_mode = Column(SQLEnum(EnergyMode), nullable=False)
    reactor_type = Column(SQLEnum(ReactorType), nullable=False)
    chemistry_type = Column(SQLEnum(ChemistryType), nullable=False)
    variant = Column(String(100), nullable=True)  # ALCVD, pulsed, hybrid, SACVD, etc.

    description = Column(Text, nullable=True)
    default_recipes = Column(JSONB, nullable=True)
    capabilities = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    recipes = relationship("CVDRecipe", back_populates="process_mode")
    runs = relationship("CVDRun", back_populates="process_mode")

    def __repr__(self):
        return f"<CVDProcessMode {self.name} ({self.pressure_mode}-{self.energy_mode}-{self.reactor_type})>"

    @property
    def full_classification(self) -> str:
        """Get full classification string"""
        parts = [self.pressure_mode.value, self.energy_mode.value,
                self.reactor_type.value, self.chemistry_type.value]
        if self.variant:
            parts.append(self.variant)
        return " - ".join(parts)


class CVDRecipe(Base):
    """
    CVD recipe definition
    Supports all CVD variants with flexible JSONB fields
    """
    __tablename__ = "cvd_recipes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    process_mode_id = Column(UUID(as_uuid=True), ForeignKey('cvd_process_modes.id'), nullable=False)

    # Film targets
    film_target = Column(String(100), nullable=False)
    thickness_target_nm = Column(Float, nullable=False)
    uniformity_target_pct = Column(Float, nullable=False)

    # Process parameters (JSONB for flexibility)
    temperature_profile = Column(JSONB, nullable=False)
    pressure_setpoints = Column(JSONB, nullable=False)
    gas_flows = Column(JSONB, nullable=False)

    # Variant-specific settings
    plasma_settings = Column(JSONB, nullable=True)
    filament_settings = Column(JSONB, nullable=True)
    laser_settings = Column(JSONB, nullable=True)
    pulsing_scheme = Column(JSONB, nullable=True)

    # Recipe steps
    recipe_steps = Column(JSONB, nullable=False)

    # Safety
    safety_hazard_level = Column(Integer, nullable=False, default=1)
    required_interlocks = Column(JSONB, nullable=True)

    # Metadata
    status = Column(String(50), nullable=False, default='draft')
    approval_date = Column(DateTime(timezone=True), nullable=True)
    approved_by_id = Column(UUID(as_uuid=True), nullable=True)
    metadata = Column(JSONB, nullable=True)

    created_by_id = Column(UUID(as_uuid=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    process_mode = relationship("CVDProcessMode", back_populates="recipes")
    runs = relationship("CVDRun", back_populates="recipe")

    def __repr__(self):
        return f"<CVDRecipe {self.name} v{self.version} ({self.film_target})>"


class CVDRun(Base):
    """
    CVD run execution
    Tracks a single CVD process run
    """
    __tablename__ = "cvd_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    run_number = Column(String(100), nullable=False, unique=True)

    # Relationships
    cvd_recipe_id = Column(UUID(as_uuid=True), ForeignKey('cvd_recipes.id'), nullable=False)
    process_mode_id = Column(UUID(as_uuid=True), ForeignKey('cvd_process_modes.id'), nullable=False)
    instrument_id = Column(UUID(as_uuid=True), nullable=False)
    sample_id = Column(UUID(as_uuid=True), nullable=True)
    wafer_id = Column(UUID(as_uuid=True), nullable=True)

    # Status
    status = Column(SQLEnum(RunStatus), nullable=False, default=RunStatus.QUEUED)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    duration_seconds = Column(Float, nullable=True)

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
    metadata = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    recipe = relationship("CVDRecipe", back_populates="runs")
    process_mode = relationship("CVDProcessMode", back_populates="runs")
    telemetry = relationship("CVDTelemetry", back_populates="run", cascade="all, delete-orphan")
    result = relationship("CVDResult", back_populates="run", uselist=False, cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CVDRun {self.run_number} ({self.status.value})>"

    @property
    def is_complete(self) -> bool:
        return self.status in [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.ABORTED]

    @property
    def is_active(self) -> bool:
        return self.status in [RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.PAUSED]


class CVDTelemetry(Base):
    """
    CVD telemetry time-series data
    High-frequency sensor data during run
    """
    __tablename__ = "cvd_telemetry"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    cvd_run_id = Column(UUID(as_uuid=True), ForeignKey('cvd_runs.id', ondelete='CASCADE'), nullable=False, index=True)
    ts = Column(DateTime(timezone=True), nullable=False, index=True)

    # Core measurements
    chamber_pressure_torr = Column(Float, nullable=True)
    temperature_zones_c = Column(JSONB, nullable=True)
    gas_flows_sccm = Column(JSONB, nullable=True)

    # Plasma measurements
    rf_power_w = Column(Float, nullable=True)
    rf_reflected_w = Column(Float, nullable=True)
    dc_bias_v = Column(Float, nullable=True)
    plasma_impedance_ohm = Column(Float, nullable=True)
    microwave_power_w = Column(Float, nullable=True)

    # Hot-wire measurements
    filament_temp_c = Column(Float, nullable=True)
    filament_current_a = Column(Float, nullable=True)

    # Laser measurements
    laser_power_w = Column(Float, nullable=True)
    laser_energy_mj = Column(Float, nullable=True)

    # In-situ metrology
    qcm_rate_nm_per_s = Column(Float, nullable=True)
    qcm_thickness_nm = Column(Float, nullable=True)
    ellipsometer_thickness_nm = Column(Float, nullable=True)
    reflectometer_thickness_nm = Column(Float, nullable=True)

    # Derived values
    deposition_rate_nm_per_min = Column(Float, nullable=True)
    plasma_density_proxy = Column(Float, nullable=True)

    # References
    rga_spectrum_uri = Column(String(500), nullable=True)
    oes_spectrum_uri = Column(String(500), nullable=True)

    # Other signals
    other_signals = Column(JSONB, nullable=True)

    # Relationships
    run = relationship("CVDRun", back_populates="telemetry")

    def __repr__(self):
        return f"<CVDTelemetry run={self.cvd_run_id} ts={self.ts}>"


class CVDResult(Base):
    """
    CVD run results
    Post-process measurements and analysis
    """
    __tablename__ = "cvd_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    cvd_run_id = Column(UUID(as_uuid=True), ForeignKey('cvd_runs.id'), nullable=False, unique=True)

    # Film properties
    film_material = Column(String(100), nullable=False)
    thickness_mean_nm = Column(Float, nullable=True)
    thickness_std_nm = Column(Float, nullable=True)
    thickness_uniformity_pct = Column(Float, nullable=True)
    thickness_min_nm = Column(Float, nullable=True)
    thickness_max_nm = Column(Float, nullable=True)
    thickness_map = Column(JSONB, nullable=True)

    # Optical
    refractive_index = Column(Float, nullable=True)
    extinction_coefficient = Column(Float, nullable=True)

    # Mechanical
    stress_mpa = Column(Float, nullable=True)
    stress_type = Column(String(50), nullable=True)

    # Quality
    conformality_score = Column(Float, nullable=True)
    selectivity_score = Column(Float, nullable=True)
    step_coverage_pct = Column(Float, nullable=True)
    defect_density_per_cm2 = Column(Float, nullable=True)
    roughness_rms_nm = Column(Float, nullable=True)

    # Composition (for MOCVD/alloys)
    composition = Column(JSONB, nullable=True)

    # VM predictions
    vm_predictions = Column(JSONB, nullable=True)

    # SPC snapshot
    spc_snapshot = Column(JSONB, nullable=True)

    # References
    report_uri = Column(String(500), nullable=True)
    raw_data_uri = Column(String(500), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    run = relationship("CVDRun", back_populates="result")

    def __repr__(self):
        return f"<CVDResult run={self.cvd_run_id} thickness={self.thickness_mean_nm}nm>"


class CVDSPCSeries(Base):
    """SPC chart series for CVD parameters"""
    __tablename__ = "cvd_spc_series"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    parameter = Column(String(100), nullable=False)
    process_mode_id = Column(UUID(as_uuid=True), ForeignKey('cvd_process_modes.id'), nullable=True)
    instrument_id = Column(UUID(as_uuid=True), nullable=True)

    # Control limits
    target = Column(Float, nullable=True)
    ucl = Column(Float, nullable=True)
    lcl = Column(Float, nullable=True)
    usl = Column(Float, nullable=True)
    lsl = Column(Float, nullable=True)

    # Statistics
    mean = Column(Float, nullable=True)
    std_dev = Column(Float, nullable=True)
    cp = Column(Float, nullable=True)
    cpk = Column(Float, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    points = relationship("CVDSPCPoint", back_populates="series", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<CVDSPCSeries {self.name} ({self.parameter})>"


class CVDSPCPoint(Base):
    """Individual SPC data point"""
    __tablename__ = "cvd_spc_points"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    series_id = Column(UUID(as_uuid=True), ForeignKey('cvd_spc_series.id'), nullable=False)
    cvd_run_id = Column(UUID(as_uuid=True), ForeignKey('cvd_runs.id'), nullable=True)

    value = Column(Float, nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    violation = Column(Boolean, nullable=False, default=False)
    violation_rules = Column(JSONB, nullable=True)

    # Relationships
    series = relationship("CVDSPCSeries", back_populates="points")

    def __repr__(self):
        return f"<CVDSPCPoint series={self.series_id} value={self.value} violation={self.violation}>"
