# services/instruments/app/schemas/__init__.py

"""
Pydantic Schemas for Request/Response Validation

These schemas provide:
- Type validation for API requests
- Serialization of database models to JSON
- Documentation for OpenAPI spec
- Separation of concerns (API vs DB layer)
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, EmailStr, UUID4, field_validator, ConfigDict
from decimal import Decimal
from enum import Enum

# Import enums from models
from ..models import (
    UserRole, ProjectStatus, InstrumentStatus, ConnectionType,
    CalibrationStatus, SampleType, MethodCategory, RunStatus,
    AttachmentType, ApprovalStatus, ModelStatus
)

# ============================================================================
# Base Schemas
# ============================================================================

class BaseSchema(BaseModel):
    """Base schema with common config"""
    model_config = ConfigDict(from_attributes=True, use_enum_values=True)

class TimestampSchema(BaseSchema):
    """Mixin for timestamps"""
    created_at: datetime
    updated_at: datetime

# ============================================================================
# Organization Schemas
# ============================================================================

class OrganizationBase(BaseSchema):
    name: str = Field(..., max_length=255)
    slug: str = Field(..., max_length=100, pattern=r'^[a-z0-9-]+$')
    settings: Dict[str, Any] = Field(default_factory=dict)

class OrganizationCreate(OrganizationBase):
    pass

class OrganizationUpdate(BaseSchema):
    name: Optional[str] = Field(None, max_length=255)
    settings: Optional[Dict[str, Any]] = None

class OrganizationResponse(OrganizationBase, TimestampSchema):
    id: UUID4

# [Rest of schema definitions...]

__all__ = [
    "OrganizationCreate", "OrganizationUpdate", "OrganizationResponse",
    "UserCreate", "UserUpdate", "UserResponse", "UserLogin", "TokenResponse",
    "ProjectCreate", "ProjectUpdate", "ProjectResponse",
    "InstrumentCreate", "InstrumentUpdate", "InstrumentResponse",
    "SampleCreate", "SampleUpdate", "SampleResponse",
    "RunCreate", "RunUpdate", "RunResponse",
    "ResultCreate", "ResultResponse",
    # ... and more
]
