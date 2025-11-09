"""
LIMS API - SOPs (Standard Operating Procedures) endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, UUID4
from datetime import datetime

from services.shared.db.deps import get_db, get_current_user, require_engineer_or_above
from services.shared.db.models import SOP, User

router = APIRouter(prefix="/api/sops", tags=["sops"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class SOPBase(BaseModel):
    number: str
    title: str
    version: str
    body_md_uri: str
    hazard_level: str = "low"


class SOPCreate(SOPBase):
    pass


class SOPResponse(SOPBase):
    id: UUID4
    organization_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# SOPs CRUD
# ============================================================================

@router.post("/", response_model=SOPResponse, status_code=status.HTTP_201_CREATED)
def create_sop(
    sop_in: SOPCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Create a new SOP.

    **Permissions**: engineer, pi, admin
    """
    sop = SOP(
        **sop_in.dict(),
        organization_id=current_user.organization_id
    )
    db.add(sop)
    db.commit()
    db.refresh(sop)
    return sop


@router.get("/", response_model=List[SOPResponse])
def list_sops(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List SOPs.

    **Auto-scoped to user's organization**
    """
    sops = db.query(SOP).filter(
        SOP.organization_id == current_user.organization_id
    ).order_by(SOP.number).offset(skip).limit(limit).all()

    return sops


@router.get("/{sop_id}", response_model=SOPResponse)
def get_sop(
    sop_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get SOP by ID."""
    sop = db.query(SOP).filter(
        SOP.id == sop_id,
        SOP.organization_id == current_user.organization_id
    ).first()

    if not sop:
        raise HTTPException(status_code=404, detail="SOP not found")

    return sop
