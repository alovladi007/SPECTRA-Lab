"""
services/shared/db/models.py

Complete SQLAlchemy ORM models for SPECTRA-Lab Platform Session 17.

Includes:
- Organizations and multi-tenancy
- Users with RBAC
- Instruments and calibrations
- Samples, wafers, devices
- Recipes and approvals
- Runs and results
- ELN entries and signatures
- SOPs and custody
- SPC series and alerts
- ML model metadata
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, Enum, ARRAY, CheckConstraint, Index, BigInteger,
    Numeric, TIMESTAMP, Table
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid
import enum

from .base import Base

# ============================================================================
# Enums
# ============================================================================

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    PI = "pi"
    ENGINEER = "engineer"
    TECHNICIAN = "technician"
    VIEWER = "viewer"


class InstrumentStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class CalibrationStatus(str, enum.Enum):
    VALID = "valid"
    EXPIRED = "expired"


class RecipeStatus(str, enum.Enum):
    DRAFT = "draft"
    APPROVED = "approved"
    RETIRED = "retired"


class ApprovalState(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class RunStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"


# ============================================================================
# Mixins
# ============================================================================

class UUIDMixin:
    """Primary key mixin with UUID."""
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    """Timestamp mixin for created_at and updated_at."""
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class SoftDeleteMixin:
    """Soft delete mixin."""
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at = Column(TIMESTAMP(timezone=True), nullable=True)


# ============================================================================
# Organizations
# ============================================================================

class Organization(Base, UUIDMixin, TimestampMixin):
    """Multi-tenant organization."""
    __tablename__ = "organizations"
    
    name = Column(String(255), nullable=False, unique=True, index=True)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    settings = Column(JSONB, default={})
    
    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    instruments = relationship("Instrument", back_populates="organization", cascade="all, delete-orphan")
    samples = relationship("Sample", back_populates="organization", cascade="all, delete-orphan")
    recipes = relationship("Recipe", back_populates="organization", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="organization", cascade="all, delete-orphan")
    eln_entries = relationship("ELNEntry", back_populates="organization", cascade="all, delete-orphan")
    sops = relationship("SOP", back_populates="organization", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Organization(name='{self.name}')>"


# ============================================================================
# Users & Auth
# ============================================================================

class User(Base, UUIDMixin, TimestampMixin):
    """Platform user with RBAC."""
    __tablename__ = "users"
    __table_args__ = (
        Index('idx_users_org_email', 'organization_id', 'email', unique=True),
    )
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False, default=UserRole.VIEWER, index=True)
    password_hash = Column(String(255), nullable=True)  # NULL for OIDC users
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    last_login = Column(TIMESTAMP(timezone=True), nullable=True)
    extra_metadata = Column(JSONB, default={})
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    owned_recipes = relationship("Recipe", back_populates="owner", foreign_keys="Recipe.owner_id")
    approvals = relationship("RecipeApproval", back_populates="approver")
    runs_created = relationship("Run", back_populates="created_by_user", foreign_keys="Run.created_by")
    eln_entries = relationship("ELNEntry", back_populates="author")
    signatures = relationship("Signature", back_populates="signer")
    
    def __repr__(self):
        return f"<User(email='{self.email}', role='{self.role.value}')>"


class APIKey(Base, UUIDMixin, TimestampMixin):
    """API keys for service accounts."""
    __tablename__ = "api_keys"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    key_hash = Column(String(255), nullable=False, unique=True, index=True)
    scopes = Column(ARRAY(Text), default=[])
    last_used_at = Column(TIMESTAMP(timezone=True), nullable=True)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<APIKey(user_id='{self.user_id}')>"


# ============================================================================
# Instruments & Calibrations
# ============================================================================

class Instrument(Base, UUIDMixin, TimestampMixin):
    """Registered instrument."""
    __tablename__ = "instruments"
    __table_args__ = (
        Index('idx_instruments_org_name', 'organization_id', 'name', unique=True),
    )
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    vendor = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    serial = Column(String(255), nullable=True)
    interface = Column(String(100), nullable=False)  # "visa_usb", "visa_gpib", etc.
    location = Column(String(255), nullable=True)
    status = Column(Enum(InstrumentStatus), default=InstrumentStatus.OFFLINE, index=True)
    extra_metadata = Column(JSONB, default={})
    
    # Relationships
    organization = relationship("Organization", back_populates="instruments")
    calibrations = relationship("Calibration", back_populates="instrument", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="instrument")
    
    def __repr__(self):
        return f"<Instrument(name='{self.name}', model='{self.model}')>"


class Calibration(Base, UUIDMixin, TimestampMixin):
    """Instrument calibration certificate."""
    __tablename__ = "calibrations"
    __table_args__ = {'extend_existing': True}

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True)
    certificate_id = Column(String(255), nullable=True)
    issued_at = Column(TIMESTAMP(timezone=True), nullable=False)
    expires_at = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    provider = Column(String(255), nullable=True)
    results = Column(JSONB, default={})
    file_uri = Column(Text, nullable=True)
    status = Column(Enum(CalibrationStatus), nullable=False, default=CalibrationStatus.VALID, index=True)
    
    # Relationships
    instrument = relationship("Instrument", back_populates="calibrations")
    
    def __repr__(self):
        return f"<Calibration(instrument_id='{self.instrument_id}', expires='{self.expires_at}')>"


# ============================================================================
# Materials & Samples
# ============================================================================

class Material(Base, UUIDMixin, TimestampMixin):
    """Material library."""
    __tablename__ = "materials"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False, index=True)
    type = Column(String(100), nullable=False)
    extra_metadata = Column(JSONB, default={})
    
    # Relationships
    samples = relationship("Sample", back_populates="material")
    
    def __repr__(self):
        return f"<Material(name='{self.name}')>"


class Sample(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """Sample hierarchy: wafer/die/device."""
    __tablename__ = "samples"
    __table_args__ = (
        Index('idx_samples_org_barcode', 'organization_id', 'barcode', unique=True),
    )
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    material_type = Column(String(255), nullable=True)
    material_id = Column(UUID(as_uuid=True), ForeignKey("materials.id"), nullable=True, index=True)
    lot_code = Column(String(255), nullable=True)
    barcode = Column(String(255), nullable=True, index=True)
    location = Column(String(255), nullable=True)
    extra_metadata = Column(JSONB, default={})
    
    # Relationships
    organization = relationship("Organization", back_populates="samples")
    material = relationship("Material", back_populates="samples")
    wafers = relationship("Wafer", back_populates="sample", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="sample")
    custody_events = relationship("CustodyEvent", back_populates="sample", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Sample(name='{self.name}', barcode='{self.barcode}')>"


class Wafer(Base, UUIDMixin, TimestampMixin):
    """Wafer within a sample."""
    __tablename__ = "wafers"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    sample_id = Column(UUID(as_uuid=True), ForeignKey("samples.id", ondelete="CASCADE"), nullable=False, index=True)
    wafer_id_code = Column(String(255), nullable=False)
    diameter_mm = Column(Float, nullable=True)
    notch = Column(String(50), nullable=True)
    map_json = Column(JSONB, default={})
    
    # Relationships
    sample = relationship("Sample", back_populates="wafers")
    devices = relationship("Device", back_populates="wafer", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="wafer")
    
    def __repr__(self):
        return f"<Wafer(id='{self.wafer_id_code}')>"


class Device(Base, UUIDMixin, TimestampMixin):
    """Device/die on a wafer."""
    __tablename__ = "devices"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    wafer_id = Column(UUID(as_uuid=True), ForeignKey("wafers.id", ondelete="CASCADE"), nullable=False, index=True)
    device_label = Column(String(255), nullable=False)
    coordinates = Column(JSONB, default={})  # {"row": 5, "col": 10}
    extra_metadata = Column(JSONB, default={})
    
    # Relationships
    wafer = relationship("Wafer", back_populates="devices")
    runs = relationship("Run", back_populates="device")
    
    def __repr__(self):
        return f"<Device(label='{self.device_label}')>"


# ============================================================================
# Recipes & Approvals
# ============================================================================

class Recipe(Base, UUIDMixin, TimestampMixin):
    """Method template with parameters."""
    __tablename__ = "recipes"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    status = Column(Enum(RecipeStatus), default=RecipeStatus.DRAFT, nullable=False, index=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    sop_id = Column(UUID(as_uuid=True), ForeignKey("sops.id"), nullable=True, index=True)
    params = Column(JSONB, default={})
    
    # Relationships
    organization = relationship("Organization", back_populates="recipes")
    owner = relationship("User", back_populates="owned_recipes", foreign_keys=[owner_id])
    sop = relationship("SOP", back_populates="recipes")
    approvals = relationship("RecipeApproval", back_populates="recipe", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="recipe")
    
    def __repr__(self):
        return f"<Recipe(name='{self.name}', status='{self.status.value}')>"


class RecipeApproval(Base, UUIDMixin, TimestampMixin):
    """Recipe approval workflow."""
    __tablename__ = "recipe_approvals"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    recipe_id = Column(UUID(as_uuid=True), ForeignKey("recipes.id", ondelete="CASCADE"), nullable=False, index=True)
    approver_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    state = Column(Enum(ApprovalState), default=ApprovalState.PENDING, nullable=False, index=True)
    comment = Column(Text, nullable=True)
    signed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    signature_blob_uri = Column(Text, nullable=True)
    
    # Relationships
    recipe = relationship("Recipe", back_populates="approvals")
    approver = relationship("User", back_populates="approvals")
    
    def __repr__(self):
        return f"<RecipeApproval(recipe_id='{self.recipe_id}', state='{self.state.value}')>"


# ============================================================================
# Runs & Results
# ============================================================================

class Run(Base, UUIDMixin, TimestampMixin):
    """Experiment execution."""
    __tablename__ = "runs"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    recipe_id = Column(UUID(as_uuid=True), ForeignKey("recipes.id"), nullable=True, index=True)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=True, index=True)
    sample_id = Column(UUID(as_uuid=True), ForeignKey("samples.id"), nullable=True, index=True)
    wafer_id = Column(UUID(as_uuid=True), ForeignKey("wafers.id"), nullable=True, index=True)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.id"), nullable=True, index=True)
    method = Column(String(255), nullable=False)
    status = Column(Enum(RunStatus), default=RunStatus.QUEUED, nullable=False, index=True)
    started_at = Column(TIMESTAMP(timezone=True), nullable=True)
    finished_at = Column(TIMESTAMP(timezone=True), nullable=True)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    log_uri = Column(Text, nullable=True)
    blocked_reason = Column(Text, nullable=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="runs")
    recipe = relationship("Recipe", back_populates="runs")
    instrument = relationship("Instrument", back_populates="runs")
    sample = relationship("Sample", back_populates="runs")
    wafer = relationship("Wafer", back_populates="runs")
    device = relationship("Device", back_populates="runs")
    created_by_user = relationship("User", back_populates="runs_created", foreign_keys=[created_by])
    results = relationship("Result", back_populates="run", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Run(method='{self.method}', status='{self.status.value}')>"


class Result(Base, UUIDMixin, TimestampMixin):
    """Analysis result."""
    __tablename__ = "results"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    result_type = Column(String(255), nullable=False)
    metrics = Column(JSONB, default={})
    arrays_uri = Column(Text, nullable=True)
    images_uri = Column(Text, nullable=True)
    artifacts_uri = Column(Text, nullable=True)
    report_uri = Column(Text, nullable=True)
    
    # Relationships
    run = relationship("Run", back_populates="results")
    
    def __repr__(self):
        return f"<Result(type='{self.result_type}')>"


# ============================================================================
# Attachments
# ============================================================================

class Attachment(Base, UUIDMixin, TimestampMixin):
    """Generic file attachment."""
    __tablename__ = "attachments"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    entity_type = Column(String(100), nullable=False, index=True)
    entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    uri = Column(Text, nullable=False)
    content_type = Column(String(255), nullable=False)
    size_bytes = Column(BigInteger, nullable=False)
    checksum = Column(String(255), nullable=False)
    
    def __repr__(self):
        return f"<Attachment(entity_type='{self.entity_type}', size={self.size_bytes})>"


# ============================================================================
# ELN & LIMS
# ============================================================================

class ELNEntry(Base, UUIDMixin, TimestampMixin):
    """Electronic lab notebook entry."""
    __tablename__ = "eln_entries"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    body_markdown = Column(Text, nullable=False)
    linked_entities = Column(JSONB, default={})
    signed = Column(Boolean, default=False, nullable=False, index=True)
    signed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="eln_entries")
    author = relationship("User", back_populates="eln_entries")
    signatures = relationship("Signature", back_populates="entry", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ELNEntry(title='{self.title}')>"


class Signature(Base, UUIDMixin, TimestampMixin):
    """E-signature record."""
    __tablename__ = "signatures"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    entry_id = Column(UUID(as_uuid=True), ForeignKey("eln_entries.id", ondelete="CASCADE"), nullable=False, index=True)
    signer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    method = Column(String(100), nullable=False)  # "password", "pkcs11", etc.
    signature_blob_uri = Column(Text, nullable=True)
    signed_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    
    # Relationships
    entry = relationship("ELNEntry", back_populates="signatures")
    signer = relationship("User", back_populates="signatures")
    
    def __repr__(self):
        return f"<Signature(entry_id='{self.entry_id}')>"


class SOP(Base, UUIDMixin, TimestampMixin):
    """Standard operating procedure."""
    __tablename__ = "sops"
    __table_args__ = (
        Index('idx_sops_org_number', 'organization_id', 'number', unique=True),
    )
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    number = Column(String(100), nullable=False)
    title = Column(String(500), nullable=False)
    version = Column(String(50), nullable=False)
    body_md_uri = Column(Text, nullable=False)
    hazard_level = Column(String(50), nullable=True)
    retired_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="sops")
    recipes = relationship("Recipe", back_populates="sop")
    
    def __repr__(self):
        return f"<SOP(number='{self.number}', title='{self.title}')>"


class CustodyEvent(Base, UUIDMixin, TimestampMixin):
    """Sample chain-of-custody."""
    __tablename__ = "custody_events"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    sample_id = Column(UUID(as_uuid=True), ForeignKey("samples.id", ondelete="CASCADE"), nullable=False, index=True)
    from_user = Column(String(255), nullable=True)
    to_user = Column(String(255), nullable=False)
    location = Column(String(255), nullable=True)
    note = Column(Text, nullable=True)
    
    # Relationships
    sample = relationship("Sample", back_populates="custody_events")
    
    def __repr__(self):
        return f"<CustodyEvent(sample_id='{self.sample_id}', to='{self.to_user}')>"


# ============================================================================
# SPC
# ============================================================================

class SPCSeries(Base, UUIDMixin, TimestampMixin):
    """SPC control chart series."""
    __tablename__ = "spc_series"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    path = Column(String(500), nullable=False)  # "project/method/metric"
    entity_type = Column(String(100), nullable=False)
    entity_id = Column(UUID(as_uuid=True), nullable=True, index=True)
    metric = Column(String(255), nullable=False)
    subgroup_size = Column(Integer, default=1)
    spec_lcl = Column(Float, nullable=True)
    spec_ucl = Column(Float, nullable=True)
    ctrl_lcl = Column(Float, nullable=True)
    ctrl_ucl = Column(Float, nullable=True)
    ruleset = Column(String(255), default="western_electric")
    chart_type = Column(String(20), default="I-MR")  # I-MR, Xbar-R, EWMA, CUSUM
    control_limits = Column(JSONB, default={})
    spec_limits = Column(JSONB, default={})
    rules = Column(JSONB, default={})
    window_size = Column(Integer, nullable=True)
    ewma_lambda = Column(Float, nullable=True)

    # Relationships
    points = relationship("SPCPoint", back_populates="series", cascade="all, delete-orphan")
    alerts = relationship("SPCAlert", back_populates="series", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SPCSeries(name='{self.name}', metric='{self.metric}')>"


class SPCPoint(Base, UUIDMixin):
    """SPC data point."""
    __tablename__ = "spc_points"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    series_id = Column(UUID(as_uuid=True), ForeignKey("spc_series.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=True, index=True)
    ts = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    value = Column(Float, nullable=False)
    subgroup = Column(String(255), nullable=True)
    subgroup_values = Column(ARRAY(Float), nullable=True)
    moving_range = Column(Float, nullable=True)
    ewma_value = Column(Float, nullable=True)
    cusum_pos = Column(Float, nullable=True)
    cusum_neg = Column(Float, nullable=True)
    violations = Column(JSONB, default={})
    extra_metadata = Column(JSONB, default={})

    # Relationships
    series = relationship("SPCSeries", back_populates="points")
    run = relationship("Run")
    alerts = relationship("SPCAlert", back_populates="point", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<SPCPoint(series_id='{self.series_id}', value={self.value})>"


class SPCAlert(Base, UUIDMixin, TimestampMixin):
    """SPC alert/violation."""
    __tablename__ = "spc_alerts"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    series_id = Column(UUID(as_uuid=True), ForeignKey("spc_series.id", ondelete="CASCADE"), nullable=False, index=True)
    point_id = Column(UUID(as_uuid=True), ForeignKey("spc_points.id", ondelete="CASCADE"), nullable=False, index=True)
    rule = Column(String(255), nullable=False)
    window = Column(String(255), nullable=True)
    severity = Column(String(50), nullable=False)
    alert_type = Column(String(50), nullable=False)
    rule_violated = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    opened_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    closed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=False, index=True)
    acknowledged_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    acknowledged_at = Column(TIMESTAMP(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    summary = Column(Text, nullable=False)
    details = Column(JSONB, default={})

    # Relationships
    series = relationship("SPCSeries", back_populates="alerts")
    point = relationship("SPCPoint", back_populates="alerts")
    acknowledged_user = relationship("User")

    def __repr__(self):
        return f"<SPCAlert(rule='{self.rule}', severity='{self.severity}')>"


# ============================================================================
# Process Control: Ion Implantation
# ============================================================================

class ImplantDoseProfile(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """Ion implantation dose profile and characterization data."""
    __tablename__ = "implant_dose_profiles"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False, index=True)

    # Ion beam parameters
    ion_species = Column(String(10), nullable=False, index=True)  # B, P, As, Sb, etc.
    isotope = Column(Integer, nullable=True)  # e.g., 11 for B-11
    energy_keV = Column(Float, nullable=False)
    tilt_deg = Column(Float, nullable=False)  # Beam tilt angle
    twist_deg = Column(Float, nullable=False)  # Wafer rotation
    dose_cm2 = Column(Float, nullable=False)  # ions/cm²

    # SRIM/TRIM simulation results
    projected_range_nm = Column(Float, nullable=True)  # Rp
    straggle_nm = Column(Float, nullable=True)  # ΔRp
    channeling_metric = Column(Float, nullable=True)

    # Extended metrics
    damage_metrics = Column(JSONB, default={})  # DPA, vacancies, etc.
    beam_uniformity = Column(JSONB, default={})  # Spatial uniformity map
    wafer_map_uri = Column(String(500), nullable=True)
    sims_profile_uri = Column(String(500), nullable=True)  # SIMS depth profile

    # Relationships
    run = relationship("Run")
    telemetry = relationship("ImplantTelemetry", back_populates="profile", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_implant_dose_org_run', 'organization_id', 'run_id'),
    )

    def __repr__(self):
        return f"<ImplantDoseProfile(species='{self.ion_species}', energy={self.energy_keV}keV)>"


class ImplantTelemetry(Base, UUIDMixin):
    """Real-time telemetry data from ion implanter."""
    __tablename__ = "implant_telemetry"

    profile_id = Column(UUID(as_uuid=True), ForeignKey("implant_dose_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False, index=True)
    ts = Column(TIMESTAMP(timezone=True), nullable=False, index=True)

    # Beam parameters
    beam_current_mA = Column(Float, nullable=False)
    pressure_mTorr = Column(Float, nullable=False)
    accel_voltage_kV = Column(Float, nullable=False)
    analyzer_magnet_T = Column(Float, nullable=True)  # Mass analyzer

    # Beam steering
    steering_X = Column(Float, nullable=True)
    steering_Y = Column(Float, nullable=True)

    # Dose integration
    dose_count_C_cm2 = Column(Float, nullable=False)  # Cumulative dose

    # Diagnostics
    beam_profile_uri = Column(String(500), nullable=True)
    faraday_currents = Column(ARRAY(Float), nullable=True)  # Multi-cup array
    gas_flows = Column(JSONB, default={})  # Source gas flows
    extra_metadata = Column(JSONB, default={})

    # Relationships
    profile = relationship("ImplantDoseProfile", back_populates="telemetry")
    run = relationship("Run")

    __table_args__ = (
        Index('idx_implant_telem_run_ts', 'run_id', 'ts'),
    )

    def __repr__(self):
        return f"<ImplantTelemetry(current={self.beam_current_mA}mA)>"


# ============================================================================
# Process Control: Rapid Thermal Processing (RTP)
# ============================================================================

class RTPProfile(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """Rapid Thermal Processing temperature profiles and parameters."""
    __tablename__ = "rtp_profiles"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False, index=True)

    # Temperature profile
    recipe_curve = Column(JSONB, nullable=False)  # Ramp/soak segments
    peak_T_C = Column(Float, nullable=False)

    # Process parameters
    ambient_gas = Column(String(50), nullable=False)  # N2, O2, NH3, etc.
    pressure_Torr = Column(Float, nullable=False)
    emissivity = Column(Float, nullable=False)  # Wafer emissivity
    pyrometer_cal_id = Column(UUID(as_uuid=True), ForeignKey("calibrations.id"), nullable=True)

    # Multi-zone control
    zone_setpoints = Column(JSONB, default={})  # Per-zone lamp settings
    uniformity_metrics = Column(JSONB, default={})
    wafer_rotation_rpm = Column(Float, nullable=True)

    # Relationships
    run = relationship("Run")
    pyrometer_calibration = relationship("Calibration")
    telemetry = relationship("RTPTelemetry", back_populates="profile", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_rtp_profile_org_run', 'organization_id', 'run_id'),
        Index('idx_rtp_recipe_gin', 'recipe_curve', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<RTPProfile(peak={self.peak_T_C}°C, gas={self.ambient_gas})>"


class RTPTelemetry(Base, UUIDMixin):
    """Real-time telemetry from RTP system."""
    __tablename__ = "rtp_telemetry"

    profile_id = Column(UUID(as_uuid=True), ForeignKey("rtp_profiles.id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False, index=True)
    ts = Column(TIMESTAMP(timezone=True), nullable=False, index=True)

    # Temperature measurements
    setpoint_T_C = Column(Float, nullable=False)
    pyrometer_T_C = Column(Float, nullable=False)  # Primary measurement
    tc_T_C = Column(ARRAY(Float), nullable=True)  # Thermocouple array

    # Lamp control
    lamp_power_pct = Column(ARRAY(Float), nullable=False)  # Per-zone

    # Process parameters
    emissivity_used = Column(Float, nullable=False)
    chamber_pressure_Torr = Column(Float, nullable=False)
    flow_sccm = Column(JSONB, nullable=False)  # Gas flows

    # Controller states
    pid_state = Column(JSONB, default={})  # P, I, D, error, output
    mpc_state = Column(JSONB, default={})  # MPC state vector
    extra_metadata = Column(JSONB, default={})

    # Relationships
    profile = relationship("RTPProfile", back_populates="telemetry")
    run = relationship("Run")

    __table_args__ = (
        Index('idx_rtp_telem_run_ts', 'run_id', 'ts'),
        Index('idx_rtp_flow_gin', 'flow_sccm', postgresql_using='gin'),
    )

    def __repr__(self):
        return f"<RTPTelemetry(temp={self.pyrometer_T_C}°C)>"


# ============================================================================
# Virtual Metrology
# ============================================================================

class VMFeatureSet(Base, UUIDMixin, TimestampMixin):
    """Virtual Metrology feature engineering configuration."""
    __tablename__ = "vm_feature_sets"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(200), nullable=False)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=True, index=True)

    # Feature engineering
    features = Column(JSONB, nullable=False)  # Feature definitions
    target_metrics = Column(JSONB, nullable=False)  # What to predict
    preprocessing = Column(JSONB, default={})  # Scaling, transforms

    active = Column(Boolean, default=True, nullable=False, index=True)

    # Relationships
    instrument = relationship("Instrument")
    models = relationship("VMModel", back_populates="feature_set", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<VMFeatureSet(name='{self.name}')>"


class VMModel(Base, UUIDMixin, TimestampMixin):
    """Virtual Metrology predictive models."""
    __tablename__ = "vm_models"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_set_id = Column(UUID(as_uuid=True), ForeignKey("vm_feature_sets.id", ondelete="CASCADE"), nullable=False, index=True)

    # Model identification
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)  # neural, xgboost, physics

    # Model storage
    model_uri = Column(String(500), nullable=True)  # S3/Minio path
    hyperparameters = Column(JSONB, default={})
    performance_metrics = Column(JSONB, default={})  # R2, RMSE, MAE

    # Training data
    training_runs = Column(ARRAY(UUID(as_uuid=True)), nullable=True)
    validation_runs = Column(ARRAY(UUID(as_uuid=True)), nullable=True)

    # Deployment status
    deployed = Column(Boolean, default=False, nullable=False, index=True)
    approved_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    approved_at = Column(TIMESTAMP(timezone=True), nullable=True)

    # Relationships
    feature_set = relationship("VMFeatureSet", back_populates="models")
    approved_user = relationship("User")

    def __repr__(self):
        return f"<VMModel(name='{self.name}', version='{self.version}')>"


# ============================================================================
# ML Metadata
# ============================================================================

class FeatureSet(Base, UUIDMixin, TimestampMixin):
    """ML feature set definition."""
    __tablename__ = "feature_sets"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    spec = Column(JSONB, nullable=False)
    
    def __repr__(self):
        return f"<FeatureSet(name='{self.name}')>"


class MLModel(Base, UUIDMixin, TimestampMixin):
    """ML model registry."""
    __tablename__ = "ml_models"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    framework = Column(String(100), nullable=False)
    metrics = Column(JSONB, default={})
    uri = Column(Text, nullable=False)
    retired_at = Column(TIMESTAMP(timezone=True), nullable=True)
    
    def __repr__(self):
        return f"<MLModel(name='{self.name}', version='{self.version}')>"
