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
    ts = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    value = Column(Float, nullable=False)
    subgroup = Column(String(255), nullable=True)
    extra_metadata = Column(JSONB, default={})
    
    # Relationships
    series = relationship("SPCSeries", back_populates="points")
    
    def __repr__(self):
        return f"<SPCPoint(series_id='{self.series_id}', value={self.value})>"


class SPCAlert(Base, UUIDMixin, TimestampMixin):
    """SPC alert/violation."""
    __tablename__ = "spc_alerts"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    series_id = Column(UUID(as_uuid=True), ForeignKey("spc_series.id", ondelete="CASCADE"), nullable=False, index=True)
    rule = Column(String(255), nullable=False)
    window = Column(String(255), nullable=True)
    severity = Column(String(50), nullable=False)
    opened_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    closed_at = Column(TIMESTAMP(timezone=True), nullable=True)
    summary = Column(Text, nullable=False)
    details = Column(JSONB, default={})
    
    # Relationships
    series = relationship("SPCSeries", back_populates="alerts")
    
    def __repr__(self):
        return f"<SPCAlert(rule='{self.rule}', severity='{self.severity}')>"


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
