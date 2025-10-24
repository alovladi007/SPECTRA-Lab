# services/instruments/app/models/__init__.py

"""
SQLAlchemy ORM Models for SemiconductorLab Platform

This module defines the complete data model with:
- Organizations, Users, Projects
- Instruments, Calibrations
- Materials, Samples
- Methods, Recipes, Runs
- Measurements, Results, Attachments
- Notebook Entries, Approvals
- Audit Logs, SPC Control Limits, ML Models
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

# Create base class for all models
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

# [Rest of model definitions...]

# Export all models
__all__ = [
    "Base",
    "Organization", "User", "Project",
    "Instrument", "Calibration",
    "Material", "Sample",
    "Method", "Recipe",
    "Run", "Measurement", "Result", "Attachment",
    "NotebookEntry", "Approval",
    "AuditLog",
    "SPCControlLimit", "MLModel",
    # Enums
    "UserRole", "ProjectStatus", "InstrumentStatus", "ConnectionType",
    "CalibrationStatus", "SampleType", "MethodCategory", "RunStatus",
    "AttachmentType", "ApprovalStatus", "ModelStatus",
]
