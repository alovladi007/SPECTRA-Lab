"""
LIMS API - ELN (Electronic Lab Notebook) endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, UUID4
from datetime import datetime

from services.shared.db.deps import get_db, get_current_user, require_engineer_or_above
from services.shared.db.models import ELNEntry, User, Signature

router = APIRouter(prefix="/api/eln", tags=["eln"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class ELNEntryBase(BaseModel):
    title: str
    body_markdown: str
    tags: Optional[List[str]] = []
    linked_runs: Optional[List[UUID4]] = []
    linked_samples: Optional[List[UUID4]] = []


class ELNEntryCreate(ELNEntryBase):
    pass


class ELNEntryResponse(ELNEntryBase):
    id: UUID4
    organization_id: UUID4
    author_id: UUID4
    is_signed: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class SignatureRequest(BaseModel):
    signature_hash: str  # Client-generated signature


class SignatureResponse(BaseModel):
    id: UUID4
    eln_entry_id: UUID4
    signer_id: UUID4
    signed_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# ELN CRUD
# ============================================================================

@router.post("/", response_model=ELNEntryResponse, status_code=status.HTTP_201_CREATED)
def create_eln_entry(
    entry_in: ELNEntryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Create a new ELN entry.

    **Permissions**: engineer, pi, admin
    """
    entry = ELNEntry(
        **entry_in.dict(),
        organization_id=current_user.organization_id,
        author_id=current_user.id
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


@router.get("/", response_model=List[ELNEntryResponse])
def list_eln_entries(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List ELN entries.

    **Auto-scoped to user's organization**
    """
    entries = db.query(ELNEntry).filter(
        ELNEntry.organization_id == current_user.organization_id
    ).order_by(ELNEntry.created_at.desc()).offset(skip).limit(limit).all()

    return entries


@router.get("/{entry_id}", response_model=ELNEntryResponse)
def get_eln_entry(
    entry_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get ELN entry by ID."""
    entry = db.query(ELNEntry).filter(
        ELNEntry.id == entry_id,
        ELNEntry.organization_id == current_user.organization_id
    ).first()

    if not entry:
        raise HTTPException(status_code=404, detail="ELN entry not found")

    return entry


@router.post("/{entry_id}/sign", response_model=SignatureResponse, status_code=status.HTTP_201_CREATED)
def sign_eln_entry(
    entry_id: UUID4,
    sig_in: SignatureRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Digitally sign an ELN entry.

    **Note**: This is a simplified signature. Production systems would use PKI.
    """
    entry = db.query(ELNEntry).filter(
        ELNEntry.id == entry_id,
        ELNEntry.organization_id == current_user.organization_id
    ).first()

    if not entry:
        raise HTTPException(status_code=404, detail="ELN entry not found")

    if entry.is_signed:
        raise HTTPException(status_code=400, detail="Entry already signed")

    # Create signature
    import hashlib
    content_hash = hashlib.sha256(entry.body_markdown.encode()).hexdigest()

    signature = Signature(
        eln_entry_id=entry_id,
        signer_id=current_user.id,
        signature_hash=sig_in.signature_hash,
        content_hash=content_hash
    )
    db.add(signature)

    # Mark entry as signed
    entry.is_signed = True
    db.commit()
    db.refresh(signature)

    return signature
