"""
LIMS API - Recipes endpoints with approval workflow.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, UUID4
from datetime import datetime

from services.shared.db.deps import (
    get_db, get_current_user, require_engineer_or_above,
    require_pi_or_admin, PaginationParams
)
from services.shared.db.models import (
    Recipe, RecipeStatus, RecipeApproval, ApprovalState, User
)

router = APIRouter(prefix="/api/recipes", tags=["recipes"])


# ============================================================================
# Pydantic Schemas
# ============================================================================

class RecipeBase(BaseModel):
    name: str
    method: str
    version: str
    description: Optional[str] = None
    parameters: dict


class RecipeCreate(RecipeBase):
    sop_id: Optional[UUID4] = None


class RecipeUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[dict] = None


class RecipeResponse(RecipeBase):
    id: UUID4
    organization_id: UUID4
    status: RecipeStatus
    owner_id: UUID4
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ApprovalRequest(BaseModel):
    comments: Optional[str] = None


class ApprovalResponse(BaseModel):
    id: UUID4
    recipe_id: UUID4
    approver_id: UUID4
    state: ApprovalState
    comments: Optional[str]
    approved_at: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Recipes CRUD
# ============================================================================

@router.post("/", response_model=RecipeResponse, status_code=status.HTTP_201_CREATED)
def create_recipe(
    recipe_in: RecipeCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Create a new recipe (draft status).

    **Permissions**: engineer, pi, admin
    """
    recipe = Recipe(
        **recipe_in.dict(),
        organization_id=current_user.organization_id,
        owner_id=current_user.id,
        status=RecipeStatus.DRAFT
    )
    db.add(recipe)
    db.commit()
    db.refresh(recipe)
    return recipe


@router.get("/", response_model=List[RecipeResponse])
def list_recipes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status_filter: Optional[RecipeStatus] = None,
    method: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List recipes with filtering.

    **Auto-scoped to user's organization**
    """
    query = db.query(Recipe).filter(Recipe.organization_id == current_user.organization_id)

    if status_filter:
        query = query.filter(Recipe.status == status_filter)

    if method:
        query = query.filter(Recipe.method.ilike(f"%{method}%"))

    recipes = query.order_by(Recipe.created_at.desc()).offset(skip).limit(limit).all()
    return recipes


@router.get("/{recipe_id}", response_model=RecipeResponse)
def get_recipe(
    recipe_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get recipe by ID."""
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.organization_id == current_user.organization_id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    return recipe


@router.put("/{recipe_id}", response_model=RecipeResponse)
def update_recipe(
    recipe_id: UUID4,
    recipe_in: RecipeUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Update recipe (only if draft or owned by user).

    **Permissions**: engineer, pi, admin (or owner)
    """
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.organization_id == current_user.organization_id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Can't edit approved recipes
    if recipe.status == RecipeStatus.APPROVED:
        raise HTTPException(
            status_code=400,
            detail="Cannot edit approved recipe. Create a new version instead."
        )

    # Check ownership or admin/PI role
    if recipe.owner_id != current_user.id and current_user.role.value not in ["admin", "pi"]:
        raise HTTPException(status_code=403, detail="Not authorized to edit this recipe")

    update_data = recipe_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(recipe, field, value)

    db.commit()
    db.refresh(recipe)
    return recipe


@router.delete("/{recipe_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_recipe(
    recipe_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_pi_or_admin)
):
    """
    Delete recipe.

    **Permissions**: pi, admin
    """
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.organization_id == current_user.organization_id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Can't delete approved recipes
    if recipe.status == RecipeStatus.APPROVED:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete approved recipe. Retire it instead."
        )

    db.delete(recipe)
    db.commit()


# ============================================================================
# Approval Workflow
# ============================================================================

@router.post("/{recipe_id}/request-approval", status_code=status.HTTP_201_CREATED)
def request_approval(
    recipe_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer_or_above)
):
    """
    Request approval for a recipe.

    **Permissions**: engineer, pi, admin (or owner)
    """
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.organization_id == current_user.organization_id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    if recipe.status != RecipeStatus.DRAFT:
        raise HTTPException(status_code=400, detail="Recipe is not in draft status")

    # Change status to pending
    recipe.status = RecipeStatus.PENDING_APPROVAL
    db.commit()

    return {"message": "Approval requested", "recipe_id": recipe_id, "status": "pending_approval"}


@router.post("/{recipe_id}/approve", response_model=ApprovalResponse, status_code=status.HTTP_201_CREATED)
def approve_recipe(
    recipe_id: UUID4,
    approval_in: ApprovalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_pi_or_admin)
):
    """
    Approve a recipe.

    **Permissions**: pi, admin only
    """
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.organization_id == current_user.organization_id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    if recipe.status != RecipeStatus.PENDING_APPROVAL:
        raise HTTPException(status_code=400, detail="Recipe is not pending approval")

    # Create approval record
    approval = RecipeApproval(
        recipe_id=recipe_id,
        approver_id=current_user.id,
        approver_role=current_user.role,
        state=ApprovalState.APPROVED,
        comments=approval_in.comments,
        approved_at=datetime.utcnow()
    )
    db.add(approval)

    # Update recipe status
    recipe.status = RecipeStatus.APPROVED
    db.commit()
    db.refresh(approval)

    return approval


@router.post("/{recipe_id}/reject", response_model=ApprovalResponse, status_code=status.HTTP_201_CREATED)
def reject_recipe(
    recipe_id: UUID4,
    approval_in: ApprovalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_pi_or_admin)
):
    """
    Reject a recipe.

    **Permissions**: pi, admin only
    """
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.organization_id == current_user.organization_id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    if recipe.status != RecipeStatus.PENDING_APPROVAL:
        raise HTTPException(status_code=400, detail="Recipe is not pending approval")

    # Create rejection record
    approval = RecipeApproval(
        recipe_id=recipe_id,
        approver_id=current_user.id,
        approver_role=current_user.role,
        state=ApprovalState.REJECTED,
        comments=approval_in.comments or "Rejected"
    )
    db.add(approval)

    # Revert recipe to draft
    recipe.status = RecipeStatus.DRAFT
    db.commit()
    db.refresh(approval)

    return approval


@router.get("/{recipe_id}/approvals", response_model=List[ApprovalResponse])
def get_recipe_approvals(
    recipe_id: UUID4,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get approval history for a recipe."""
    # Verify recipe exists and belongs to user's org
    recipe = db.query(Recipe).filter(
        Recipe.id == recipe_id,
        Recipe.organization_id == current_user.organization_id
    ).first()

    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    approvals = db.query(RecipeApproval).filter(
        RecipeApproval.recipe_id == recipe_id
    ).order_by(RecipeApproval.created_at.desc()).all()

    return approvals
