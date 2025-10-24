# services/instruments/app/models/__init__.py

"""
SQLAlchemy ORM Models for SemiconductorLab Platform

Complete data model with 28+ entities covering:
- Organizations, Users, Projects
- Instruments, Calibrations
- Materials, Samples
- Methods, Recipes, Runs
- Measurements, Results, Attachments
- Notebook Entries, Approvals
- Audit Logs, SPC Limits, ML Models
"""

from datetime import datetime
from typing import List, Optional
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text,
    ForeignKey, Enum, JSON, ARRAY, CheckConstraint, Index, BigInteger,
    Numeric, TIMESTAMP
)
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import uuid
import enum

# Create base class
Base = declarative_base()

# ============================================================================
# Enums
# ============================================================================

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    PI = "pi"
    ENGINEER = "engineer"
    TECHNICIAN = "technician"
    VIEWER = "viewer"

class ProjectStatus(str, enum.Enum):
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    ARCHIVED = "archived"

class InstrumentStatus(str, enum.Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"

class ConnectionType(str, enum.Enum):
    VISA_USB = "visa_usb"
    VISA_GPIB = "visa_gpib"
    VISA_TCPIP = "visa_tcpip"
    SERIAL = "serial"
    USB_RAW = "usb_raw"

class CalibrationStatus(str, enum.Enum):
    VALID = "valid"
    DUE = "due"
    OVERDUE = "overdue"
    INVALID = "invalid"

class SampleType(str, enum.Enum):
    WAFER = "wafer"
    DIE = "die"
    DEVICE = "device"
    COUPON = "coupon"
    TEST_STRUCTURE = "test_structure"

class MethodCategory(str, enum.Enum):
    ELECTRICAL = "electrical"
    OPTICAL = "optical"
    STRUCTURAL = "structural"
    CHEMICAL = "chemical"

class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"

class AttachmentType(str, enum.Enum):
    RAW_DATA = "raw_data"
    IMAGE = "image"
    REPORT = "report"
    NOTEBOOK = "notebook"
    OTHER = "other"

class ApprovalStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

class ModelStatus(str, enum.Enum):
    TRAINING = "training"
    DEPLOYED = "deployed"
    ARCHIVED = "archived"

# ============================================================================
# Mixin Classes
# ============================================================================

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

class UUIDMixin:
    """Mixin for UUID primary key"""
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

# ============================================================================
# Core Entities
# ============================================================================

class Organization(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "organizations"

    name = Column(String(255), nullable=False)
    slug = Column(String(100), nullable=False, unique=True, index=True)
    settings = Column(JSONB, default={})

    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    projects = relationship("Project", back_populates="organization", cascade="all, delete-orphan")
    instruments = relationship("Instrument", back_populates="organization", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Organization(name='{self.name}', slug='{self.slug}')>"

class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False, index=True)
    is_active = Column(Boolean, default=True)
    last_login = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})

    # Relationships
    organization = relationship("Organization", back_populates="users")
    owned_projects = relationship("Project", back_populates="owner", foreign_keys="Project.owner_id")
    operated_runs = relationship("Run", back_populates="operator", foreign_keys="Run.operator_id")

    def __repr__(self):
        return f"<User(email='{self.email}', role='{self.role.value}')>"

class Project(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "projects"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE, index=True)
    started_at = Column(TIMESTAMP(timezone=True))
    completed_at = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})

    # Relationships
    organization = relationship("Organization", back_populates="projects")
    owner = relationship("User", back_populates="owned_projects", foreign_keys=[owner_id])
    samples = relationship("Sample", back_populates="project", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project(name='{self.name}', status='{self.status.value}')>"

# ============================================================================
# Instruments & Calibrations
# ============================================================================

class Instrument(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "instruments"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    model = Column(String(255), nullable=False)
    vendor = Column(String(255), nullable=False)
    serial_number = Column(String(255))
    connection_type = Column(Enum(ConnectionType), nullable=False)
    connection_string = Column(String(500), nullable=False)
    driver = Column(String(255), nullable=False)
    capabilities = Column(ARRAY(Text), default=[])
    status = Column(Enum(InstrumentStatus), default=InstrumentStatus.OFFLINE, index=True)
    firmware_version = Column(String(100))
    last_seen = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})

    # Relationships
    organization = relationship("Organization", back_populates="instruments")
    calibrations = relationship("Calibration", back_populates="instrument", cascade="all, delete-orphan")
    runs = relationship("Run", back_populates="instrument")

    def __repr__(self):
        return f"<Instrument(name='{self.name}', model='{self.model}')>"

class Calibration(Base, UUIDMixin):
    __tablename__ = "calibrations"

    instrument_id = Column(UUID(as_uuid=True), ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True)
    performed_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    calibration_date = Column(TIMESTAMP(timezone=True), nullable=False)
    next_calibration_date = Column(TIMESTAMP(timezone=True), nullable=False, index=True)
    certificate_number = Column(String(255))
    certificate_url = Column(Text)
    standards_used = Column(ARRAY(Text))
    uncertainty_budget = Column(JSONB)
    notes = Column(Text)
    status = Column(Enum(CalibrationStatus), default=CalibrationStatus.VALID, index=True)
    metadata = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    instrument = relationship("Instrument", back_populates="calibrations")

    def __repr__(self):
        return f"<Calibration(instrument_id='{self.instrument_id}', status='{self.status.value}')>"

# ============================================================================
# Materials & Samples
# ============================================================================

class Material(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "materials"

    name = Column(String(255), nullable=False, unique=True, index=True)
    chemical_formula = Column(String(100))
    crystal_structure = Column(String(100))
    lattice_constants = Column(JSONB)
    band_gap = Column(JSONB)
    refractive_index = Column(JSONB)
    properties = Column(JSONB, default={})
    references = Column(ARRAY(Text))

    # Relationships
    samples = relationship("Sample", back_populates="material")

    def __repr__(self):
        return f"<Material(name='{self.name}')>"

class Sample(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "samples"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    parent_id = Column(UUID(as_uuid=True), ForeignKey("samples.id"), index=True)
    name = Column(String(255), nullable=False)
    type = Column(Enum(SampleType), nullable=False, index=True)
    material_id = Column(UUID(as_uuid=True), ForeignKey("materials.id"))
    barcode = Column(String(255), index=True)
    qr_code = Column(String(255))
    location = Column(String(255))
    custodian_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    received_date = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})

    # Relationships
    project = relationship("Project", back_populates="samples")
    material = relationship("Material", back_populates="samples")
    parent = relationship("Sample", remote_side="Sample.id", back_populates="children")
    children = relationship("Sample", back_populates="parent")
    runs = relationship("Run", back_populates="sample", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Sample(name='{self.name}', type='{self.type.value}')>"

# ============================================================================
# Methods & Recipes
# ============================================================================

class Method(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "methods"

    name = Column(String(255), nullable=False, unique=True)
    display_name = Column(String(255), nullable=False)
    category = Column(Enum(MethodCategory), nullable=False, index=True)
    description = Column(Text)
    parameter_schema = Column(JSONB, nullable=False)
    default_parameters = Column(JSONB, default={})
    units = Column(JSONB, default={})
    safety_warnings = Column(ARRAY(Text))

    # Relationships
    recipes = relationship("Recipe", back_populates="method")
    runs = relationship("Run", back_populates="method")

    def __repr__(self):
        return f"<Method(name='{self.name}', category='{self.category.value}')>"

class Recipe(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "recipes"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    method_id = Column(UUID(as_uuid=True), ForeignKey("methods.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    parameters = Column(JSONB, nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    is_template = Column(Boolean, default=False, index=True)
    metadata = Column(JSONB, default={})

    # Relationships
    method = relationship("Method", back_populates="recipes")
    runs = relationship("Run", back_populates="recipe")

    def __repr__(self):
        return f"<Recipe(name='{self.name}')>"

# ============================================================================
# Runs & Measurements
# ============================================================================

class Run(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "runs"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    method_id = Column(UUID(as_uuid=True), ForeignKey("methods.id"), nullable=False, index=True)
    sample_id = Column(UUID(as_uuid=True), ForeignKey("samples.id", ondelete="CASCADE"), nullable=False, index=True)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey("instruments.id"), nullable=False, index=True)
    operator_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    recipe_id = Column(UUID(as_uuid=True), ForeignKey("recipes.id"))
    status = Column(Enum(RunStatus), default=RunStatus.PENDING, index=True)
    parameters = Column(JSONB, nullable=False)
    progress = Column(Numeric(5, 2), default=0.0)
    started_at = Column(TIMESTAMP(timezone=True), index=True)
    completed_at = Column(TIMESTAMP(timezone=True))
    duration_seconds = Column(Integer)
    error_message = Column(Text)
    raw_data_uri = Column(Text)
    raw_data_hash = Column(String(64))
    environmental_conditions = Column(JSONB)
    metadata = Column(JSONB, default={})

    # Relationships
    organization = relationship("Organization")
    project = relationship("Project", back_populates="runs")
    method = relationship("Method", back_populates="runs")
    sample = relationship("Sample", back_populates="runs")
    instrument = relationship("Instrument", back_populates="runs")
    operator = relationship("User", back_populates="operated_runs", foreign_keys=[operator_id])
    recipe = relationship("Recipe", back_populates="runs")
    measurements = relationship("Measurement", back_populates="run", cascade="all, delete-orphan")
    results = relationship("Result", back_populates="run", cascade="all, delete-orphan")
    attachments = relationship("Attachment", back_populates="run", cascade="all, delete-orphan")
    approvals = relationship("Approval", back_populates="run", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Run(id='{self.id}', method='{self.method.name if self.method else 'N/A'}', status='{self.status.value}')>"

class Measurement(Base):
    __tablename__ = "measurements"

    time = Column(TIMESTAMP(timezone=True), nullable=False, primary_key=True)
    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, primary_key=True, index=True)
    sequence_number = Column(Integer, nullable=False)
    values = Column(JSONB, nullable=False)
    metadata = Column(JSONB, default={})

    # Relationships
    run = relationship("Run", back_populates="measurements")

    def __repr__(self):
        return f"<Measurement(run_id='{self.run_id}', seq={self.sequence_number})>"

class Result(Base, UUIDMixin):
    __tablename__ = "results"

    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    metric = Column(String(255), nullable=False, index=True)
    value = Column(Numeric, nullable=False)
    unit = Column(String(50), nullable=False)
    uncertainty = Column(Numeric)
    uncertainty_type = Column(String(50))
    fit_quality = Column(JSONB)
    analysis_method = Column(String(255))
    analysis_version = Column(String(50))
    metadata = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, index=True)

    # Relationships
    run = relationship("Run", back_populates="results")

    def __repr__(self):
        return f"<Result(metric='{self.metric}', value={self.value} {self.unit})>"

class Attachment(Base, UUIDMixin):
    __tablename__ = "attachments"

    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), index=True)
    filename = Column(String(500), nullable=False)
    mime_type = Column(String(100), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    storage_uri = Column(Text, nullable=False)
    file_hash = Column(String(64), nullable=False)
    attachment_type = Column(Enum(AttachmentType), nullable=False, index=True)
    description = Column(Text)
    metadata = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)

    # Relationships
    run = relationship("Run", back_populates="attachments")

    def __repr__(self):
        return f"<Attachment(filename='{self.filename}', type='{self.attachment_type.value}')>"

# ============================================================================
# ELN & Approvals
# ============================================================================

class NotebookEntry(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "notebook_entries"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, index=True)
    author_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    is_signed = Column(Boolean, default=False)
    signed_at = Column(TIMESTAMP(timezone=True))
    signature = Column(Text)
    related_runs = Column(ARRAY(UUID))
    tags = Column(ARRAY(Text))
    metadata = Column(JSONB, default={})

    def __repr__(self):
        return f"<NotebookEntry(title='{self.title}')>"

class Approval(Base, UUIDMixin):
    __tablename__ = "approvals"

    run_id = Column(UUID(as_uuid=True), ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True)
    approver_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    status = Column(Enum(ApprovalStatus), default=ApprovalStatus.PENDING, index=True)
    comments = Column(Text)
    approved_at = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    # Relationships
    run = relationship("Run", back_populates="approvals")

    def __repr__(self):
        return f"<Approval(run_id='{self.run_id}', status='{self.status.value}')>"

# ============================================================================
# SPC & ML
# ============================================================================

class SPCControlLimit(Base, UUIDMixin):
    __tablename__ = "spc_control_limits"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    method_id = Column(UUID(as_uuid=True), ForeignKey("methods.id"), nullable=False, index=True)
    instrument_id = Column(UUID(as_uuid=True), ForeignKey("instruments.id"))
    metric = Column(String(255), nullable=False, index=True)
    center_line = Column(Numeric, nullable=False)
    ucl = Column(Numeric, nullable=False)
    lcl = Column(Numeric, nullable=False)
    usl = Column(Numeric)
    lsl = Column(Numeric)
    sample_size = Column(Integer, nullable=False)
    confidence_level = Column(Numeric, default=0.95)
    calculation_method = Column(String(100))
    valid_from = Column(TIMESTAMP(timezone=True), nullable=False)
    valid_until = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    def __repr__(self):
        return f"<SPCControlLimit(metric='{self.metric}', CL={self.center_line})>"

class MLModel(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "ml_models"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)
    target_metric = Column(String(255), nullable=False)
    features = Column(ARRAY(Text), nullable=False)
    hyperparameters = Column(JSONB)
    training_metrics = Column(JSONB)
    validation_metrics = Column(JSONB)
    model_uri = Column(Text, nullable=False)
    status = Column(Enum(ModelStatus), default=ModelStatus.TRAINING, index=True)
    trained_by = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    deployed_at = Column(TIMESTAMP(timezone=True))
    metadata = Column(JSONB, default={})

    def __repr__(self):
        return f"<MLModel(name='{self.name}', version='{self.version}')>"

# ============================================================================
# Audit Log
# ============================================================================

class AuditLog(Base, UUIDMixin):
    __tablename__ = "audit_log"

    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    action = Column(String(100), nullable=False)
    entity_type = Column(String(100), nullable=False, index=True)
    entity_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    changes = Column(JSONB)
    ip_address = Column(INET)
    user_agent = Column(Text)
    timestamp = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False, index=True)

    def __repr__(self):
        return f"<AuditLog(action='{self.action}', entity_type='{self.entity_type}')>"
