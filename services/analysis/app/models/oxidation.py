"""
SQLAlchemy models for Oxidation Manufacturing Module
Thermal oxidation processes with furnace management, recipes, runs, and SPC
"""

from sqlalchemy import Column, String, Float, Integer, Boolean, Text, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime
import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.base import Base


# Enums for Oxidation classifications
class OxidationFurnaceType(str, Enum):
    """Oxidation furnace types"""
    HORIZONTAL_TUBE = "horizontal_tube"
    VERTICAL_TUBE = "vertical_tube"
    BATCH = "batch"
    RAPID_THERMAL = "rapid_thermal"


class OxidationType(str, Enum):
    """Oxidation process types"""
    DRY = "dry"  # Dry O2
    WET = "wet"  # H2O + O2
    STEAM = "steam"  # High-pressure steam
    PYROGENIC = "pyrogenic"  # H2 + O2 combustion
    ANODIC = "anodic"  # Electrochemical


class OxideApplication(str, Enum):
    """Oxide layer applications"""
    GATE_OXIDE = "gate_oxide"  # Gate dielectric
    FIELD_OXIDE = "field_oxide"  # Isolation oxide
    MASK_OXIDE = "mask_oxide"  # Process mask
    PASSIVATION = "passivation"  # Surface protection
    TUNNEL_OXIDE = "tunnel_oxide"  # Thin tunnel barrier


class RunStatus(str, Enum):
    """Oxidation run status"""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class OxidationFurnace(Base):
    """
    Oxidation furnace equipment
    Represents physical oxidation furnace with temperature zones
    """
    __tablename__ = "oxidation_furnaces"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    name = Column(String(255), nullable=False)
    furnace_type = Column(SQLEnum(OxidationFurnaceType), nullable=False)
    manufacturer = Column(String(100), nullable=True)
    model = Column(String(100), nullable=True)
    serial_number = Column(String(100), nullable=True)

    # Physical configuration
    tube_diameter_mm = Column(Float, nullable=True)
    tube_length_mm = Column(Float, nullable=True)
    num_temperature_zones = Column(Integer, default=1)
    max_temperature_c = Column(Float, nullable=False)
    max_wafer_capacity = Column(Integer, nullable=False)

    # Capabilities
    supports_dry_oxidation = Column(Boolean, default=True)
    supports_wet_oxidation = Column(Boolean, default=False)
    supports_steam_oxidation = Column(Boolean, default=False)
    supports_pyrogenic = Column(Boolean, default=False)

    # Status
    is_active = Column(Boolean, default=True)
    is_calibrated = Column(Boolean, default=True)
    last_pm_date = Column(DateTime(timezone=True), nullable=True)
    next_pm_date = Column(DateTime(timezone=True), nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    metadata_json = Column(JSONB, nullable=True)

    # Relationships
    recipes = relationship("OxidationRecipe", back_populates="furnace", cascade="all, delete-orphan")
    runs = relationship("OxidationRun", back_populates="furnace")


class OxidationRecipe(Base):
    """
    Oxidation process recipe
    Defines temperature profiles and ambient conditions
    """
    __tablename__ = "oxidation_recipes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    furnace_id = Column(UUID(as_uuid=True), ForeignKey("oxidation_furnaces.id"), nullable=False)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    oxidation_type = Column(SQLEnum(OxidationType), nullable=False)
    application = Column(SQLEnum(OxideApplication), nullable=True)

    # Process parameters
    temperature_c = Column(Float, nullable=False)
    time_minutes = Column(Float, nullable=False)
    ramp_rate_c_per_min = Column(Float, default=10.0)

    # Gas flows (sccm)
    o2_flow_sccm = Column(Float, default=0.0)
    n2_flow_sccm = Column(Float, default=0.0)
    h2_flow_sccm = Column(Float, default=0.0)

    # Target specifications
    target_thickness_nm = Column(Float, nullable=True)
    thickness_tolerance_nm = Column(Float, nullable=True)

    # Recipe status
    status = Column(String(50), default="DRAFT")  # DRAFT, APPROVED, DEPRECATED
    version = Column(String(50), default="1.0")
    run_count = Column(Integer, default=0)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    created_by = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    furnace = relationship("OxidationFurnace", back_populates="recipes")
    runs = relationship("OxidationRun", back_populates="recipe")


class OxidationRun(Base):
    """
    Oxidation process run
    Tracks execution of oxidation recipe on wafers
    """
    __tablename__ = "oxidation_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    furnace_id = Column(UUID(as_uuid=True), ForeignKey("oxidation_furnaces.id"), nullable=False)
    recipe_id = Column(UUID(as_uuid=True), ForeignKey("oxidation_recipes.id"), nullable=False)

    run_number = Column(String(100), nullable=True, index=True)
    lot_id = Column(String(100), nullable=True)
    wafer_ids = Column(JSONB, default=[])  # List of wafer IDs

    # Run status
    status = Column(SQLEnum(RunStatus), default=RunStatus.QUEUED, nullable=False, index=True)

    # Timestamps
    queued_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    # Actual process parameters (as-run)
    actual_temperature_c = Column(Float, nullable=True)
    actual_time_minutes = Column(Float, nullable=True)
    actual_thickness_nm = Column(Float, nullable=True)

    # Results
    measured_thickness_nm = Column(Float, nullable=True)
    thickness_uniformity_percent = Column(Float, nullable=True)
    refractive_index = Column(Float, nullable=True)

    # Job tracking
    job_id = Column(String(255), nullable=True)  # Celery task ID
    job_progress = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)

    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now(), nullable=True)
    operator = Column(String(255), nullable=True)
    notes = Column(Text, nullable=True)

    # Relationships
    furnace = relationship("OxidationFurnace", back_populates="runs")
    recipe = relationship("OxidationRecipe", back_populates="runs")
    results = relationship("OxidationResult", back_populates="run", cascade="all, delete-orphan")


class OxidationResult(Base):
    """
    Oxidation measurement results
    Stores ellipsometry, reflectometry, C-V measurements
    """
    __tablename__ = "oxidation_results"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("oxidation_runs.id"), nullable=False)
    org_id = Column(UUID(as_uuid=True), nullable=False, index=True)

    wafer_id = Column(String(100), nullable=False)
    measurement_type = Column(String(50), nullable=False)  # ELLIPSOMETRY, REFLECTOMETRY, CV

    # Thickness measurements
    thickness_nm = Column(Float, nullable=False)
    thickness_std_dev = Column(Float, nullable=True)
    uniformity_percent = Column(Float, nullable=True)

    # Optical properties
    refractive_index = Column(Float, nullable=True)
    extinction_coefficient = Column(Float, nullable=True)

    # Electrical properties (for gate oxides)
    breakdown_voltage_v = Column(Float, nullable=True)
    dielectric_constant = Column(Float, nullable=True)
    interface_state_density = Column(Float, nullable=True)

    # Measurement metadata
    measurement_points = Column(JSONB, nullable=True)  # Multi-point measurements
    measured_at = Column(DateTime(timezone=True), server_default=func.now())
    measured_by = Column(String(255), nullable=True)
    equipment_id = Column(String(100), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    notes = Column(Text, nullable=True)

    # Relationships
    run = relationship("OxidationRun", back_populates="results")
