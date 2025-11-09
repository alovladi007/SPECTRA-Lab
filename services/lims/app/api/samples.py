"""
LIMS API - Samples endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, UUID4
from datetime import datetime

from services.shared.db.deps import get_db, get_current_user, require_engineer_or_above, PaginationParams
from services.shared.db.models import Sample, User, Wafer, Device

router = APIRouter(prefix="/api/samples", tags=["samples"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class SampleBase(BaseModel):
    name: str
    material_type: Optional[str] = None
    material_id: Optional[UUID4] = None
    lot_code: Optional[str] = None
    barcode: Optional[str] = None
    location: Optional[str] = None
    extra_metadata: dict = {}


class SampleCreate(SampleBase):
    pass


class SampleUpdate(BaseModel):
    name: Optional[str] = None
    location: Optional[str] = None
    extra_metadata: Optional[dict] = None


class SampleResponse(SampleBase):
    id: UUID4
    organization_id: UUID4
    is_deleted: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class WaferCreate(BaseModel):
    wafer_id_code: str
    diameter_mm: Optional[float] = None
    notch: Optional[str] = None
    map_json: dict = {}


class WaferResponse(BaseModel):
    id: UUID4
    sample_id: UUID4
    wafer_id_code: str
    diameter_mm: Optional[float]
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Samples CRUD
# ============================================================================

@router.post("/", response_model=SampleResponse, status_code=status.HTTP_201_CREATED)
def create_sample(
    sample_in: SampleCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Create a new sample.

    **Permissions**: engineer, pi, admin
    """
    sample = Sample(
        **sample_in.dict(),
        organization_id=current_user.organization_id
    )
    db.add(sample)
    db.commit()
    db.refresh(sample)
    return sample


@router.get("/", response_model=List[SampleResponse])
def list_samples(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    barcode: Optional[str] = None,
    location: Optional[str] = None,
    include_deleted: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List samples with pagination and filtering.

    **Auto-scoped to user's organization**
    """
    query = db.query(Sample).filter(Sample.organization_id == current_user.organization_id)

    if not include_deleted:
        query = query.filter(Sample.is_deleted == False)

    if barcode:
        query = query.filter(Sample.barcode.ilike(f"%{barcode}%"))

    if location:
        query = query.filter(Sample.location.ilike(f"%{location}%"))

    samples = query.order_by(Sample.created_at.desc()).offset(skip).limit(limit).all()
    return samples


@router.get("/{sample_id}", response_model=SampleResponse)
def get_sample(
    sample_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get sample by ID."""
    sample = db.query(Sample).filter(
        Sample.id == sample_id,
        Sample.organization_id == current_user.organization_id
    ).first()

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    return sample


@router.put("/{sample_id}", response_model=SampleResponse)
def update_sample(
    sample_id: UUID4,
    sample_in: SampleUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Update sample.

    **Permissions**: engineer, pi, admin
    """
    sample = db.query(Sample).filter(
        Sample.id == sample_id,
        Sample.organization_id == current_user.organization_id
    ).first()

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    update_data = sample_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(sample, field, value)

    db.commit()
    db.refresh(sample)
    return sample


@router.delete("/{sample_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_sample(
    sample_id: UUID4,
    hard_delete: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Soft delete (or hard delete) a sample.

    **Permissions**: engineer, pi, admin
    """
    sample = db.query(Sample).filter(
        Sample.id == sample_id,
        Sample.organization_id == current_user.organization_id
    ).first()

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    if hard_delete:
        db.delete(sample)
    else:
        sample.is_deleted = True
        sample.deleted_at = datetime.utcnow()

    db.commit()


# ============================================================================
# Wafers (nested resource)
# ============================================================================

@router.post("/{sample_id}/wafers", response_model=WaferResponse, status_code=status.HTTP_201_CREATED)
def create_wafer(
    sample_id: UUID4,
    wafer_in: WaferCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Add a wafer to a sample.

    **Permissions**: engineer, pi, admin
    """
    # Verify sample exists and belongs to user's org
    sample = db.query(Sample).filter(
        Sample.id == sample_id,
        Sample.organization_id == current_user.organization_id
    ).first()

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    wafer = Wafer(
        **wafer_in.dict(),
        sample_id=sample_id,
        organization_id=current_user.organization_id
    )
    db.add(wafer)
    db.commit()
    db.refresh(wafer)
    return wafer


@router.get("/{sample_id}/wafers", response_model=List[WaferResponse])
def list_sample_wafers(
    sample_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """List all wafers for a sample."""
    # Verify sample access
    sample = db.query(Sample).filter(
        Sample.id == sample_id,
        Sample.organization_id == current_user.organization_id
    ).first()

    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")

    wafers = db.query(Wafer).filter(Wafer.sample_id == sample_id).all()
    return wafers
