"""
Oxidation Manufacturing - FastAPI Routers
REST API endpoints for manufacturing-grade oxidation platform
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session, joinedload

import logging
import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.deps import get_db
from ..models.oxidation import (
    OxidationFurnace,
    OxidationRecipe,
    OxidationRun,
    OxidationResult,
    RunStatus,
)
from ..schemas.oxidation import (
    OxidationFurnaceCreate,
    OxidationFurnaceUpdate,
    OxidationFurnaceSchema,
    OxidationRecipeCreate,
    OxidationRecipeUpdate,
    OxidationRecipeSchema,
    OxidationRunCreate,
    OxidationRunUpdate,
    OxidationRunSchema,
    OxidationRunQuery,
    OxidationResultCreate,
    OxidationResultUpdate,
    OxidationResultSchema,
    OxidationAnalyticsRequest,
    OxidationAnalyticsResponse,
    OxidationBatchRunCreate,
    OxidationBatchRunResponse,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/oxidation", tags=["Oxidation Manufacturing"])


# ============================================================================
# Furnaces
# ============================================================================

@router.post("/furnaces", response_model=OxidationFurnaceSchema, status_code=status.HTTP_201_CREATED)
def create_furnace(
    furnace: OxidationFurnaceCreate,
    db: Session = Depends(get_db),
):
    """Create a new oxidation furnace"""
    try:
        db_furnace = OxidationFurnace(**furnace.model_dump())
        db.add(db_furnace)
        db.commit()
        db.refresh(db_furnace)

        logger.info(f"Created furnace: {db_furnace.id} - {db_furnace.name}")
        return db_furnace

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating furnace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/furnaces", response_model=List[OxidationFurnaceSchema])
def list_furnaces(
    org_id: Optional[UUID] = None,
    furnace_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List oxidation furnaces with filters"""
    try:
        query = select(OxidationFurnace)

        # Apply filters
        filters = []
        if org_id:
            filters.append(OxidationFurnace.org_id == org_id)
        if furnace_type:
            filters.append(OxidationFurnace.furnace_type == furnace_type)
        if is_active is not None:
            filters.append(OxidationFurnace.is_active == is_active)

        if filters:
            query = query.where(and_(*filters))

        query = query.offset(skip).limit(limit).order_by(OxidationFurnace.created_at.desc())

        result = db.execute(query)
        furnaces = result.scalars().all()

        return furnaces

    except Exception as e:
        logger.exception(f"Error listing furnaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/furnaces/{furnace_id}", response_model=OxidationFurnaceSchema)
def get_furnace(
    furnace_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific furnace by ID"""
    try:
        query = select(OxidationFurnace).where(OxidationFurnace.id == furnace_id)
        result = db.execute(query)
        furnace = result.scalar_one_or_none()

        if not furnace:
            raise HTTPException(status_code=404, detail="Furnace not found")

        return furnace

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting furnace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/furnaces/{furnace_id}", response_model=OxidationFurnaceSchema)
def update_furnace(
    furnace_id: UUID,
    update_data: OxidationFurnaceUpdate,
    db: Session = Depends(get_db),
):
    """Update a furnace"""
    try:
        query = select(OxidationFurnace).where(OxidationFurnace.id == furnace_id)
        result = db.execute(query)
        furnace = result.scalar_one_or_none()

        if not furnace:
            raise HTTPException(status_code=404, detail="Furnace not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(furnace, key, value)

        db.commit()
        db.refresh(furnace)

        logger.info(f"Updated furnace: {furnace_id}")
        return furnace

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating furnace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/furnaces/{furnace_id}/status")
def get_furnace_status(
    furnace_id: UUID,
    db: Session = Depends(get_db),
):
    """Get current status of a furnace"""
    try:
        # Check if furnace exists
        furnace_query = select(OxidationFurnace).where(OxidationFurnace.id == furnace_id)
        furnace = db.execute(furnace_query).scalar_one_or_none()
        if not furnace:
            raise HTTPException(status_code=404, detail="Furnace not found")

        # Check for active runs
        active_runs_query = (
            select(OxidationRun)
            .where(
                and_(
                    OxidationRun.furnace_id == furnace_id,
                    OxidationRun.status.in_([RunStatus.RUNNING, RunStatus.QUEUED])
                )
            )
        )
        active_runs = db.execute(active_runs_query).scalars().all()

        # Determine state
        if any(r.status == RunStatus.RUNNING for r in active_runs):
            state = "processing"
            current_run = next(r for r in active_runs if r.status == RunStatus.RUNNING)
            current_run_id = str(current_run.id)
        elif any(r.status == RunStatus.QUEUED for r in active_runs):
            state = "queued"
            current_run_id = None
        else:
            state = "idle"
            current_run_id = None

        return {
            "furnace_id": str(furnace_id),
            "state": state,
            "current_run_id": current_run_id,
            "is_active": furnace.is_active,
            "queued_runs": len([r for r in active_runs if r.status == RunStatus.QUEUED]),
            "message": f"Furnace is {state}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting furnace status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Recipes
# ============================================================================

@router.post("/recipes", response_model=OxidationRecipeSchema, status_code=status.HTTP_201_CREATED)
def create_recipe(
    recipe: OxidationRecipeCreate,
    db: Session = Depends(get_db),
):
    """Create a new oxidation recipe"""
    try:
        # Verify furnace exists
        furnace_query = select(OxidationFurnace).where(OxidationFurnace.id == recipe.furnace_id)
        furnace_result = db.execute(furnace_query)
        if not furnace_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Furnace not found")

        db_recipe = OxidationRecipe(**recipe.model_dump())
        db.add(db_recipe)
        db.commit()
        db.refresh(db_recipe)

        logger.info(f"Created recipe: {db_recipe.id} - {db_recipe.name}")
        return db_recipe

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating recipe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipes", response_model=List[OxidationRecipeSchema])
def list_recipes(
    org_id: Optional[UUID] = None,
    furnace_id: Optional[UUID] = None,
    oxidation_type: Optional[str] = None,
    application: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List oxidation recipes with filters"""
    try:
        query = select(OxidationRecipe).options(joinedload(OxidationRecipe.furnace))

        # Apply filters
        filters = []
        if org_id:
            filters.append(OxidationRecipe.org_id == org_id)
        if furnace_id:
            filters.append(OxidationRecipe.furnace_id == furnace_id)
        if oxidation_type:
            filters.append(OxidationRecipe.oxidation_type == oxidation_type)
        if application:
            filters.append(OxidationRecipe.application == application)
        if status_filter:
            filters.append(OxidationRecipe.status == status_filter)

        if filters:
            query = query.where(and_(*filters))

        query = query.offset(skip).limit(limit).order_by(OxidationRecipe.created_at.desc())

        result = db.execute(query)
        recipes = result.scalars().unique().all()

        return recipes

    except Exception as e:
        logger.exception(f"Error listing recipes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipes/{recipe_id}", response_model=OxidationRecipeSchema)
def get_recipe(
    recipe_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific recipe by ID"""
    try:
        query = select(OxidationRecipe).where(OxidationRecipe.id == recipe_id).options(
            joinedload(OxidationRecipe.furnace)
        )
        result = db.execute(query)
        recipe = result.scalar_one_or_none()

        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        return recipe

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting recipe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/recipes/{recipe_id}", response_model=OxidationRecipeSchema)
def update_recipe(
    recipe_id: UUID,
    update_data: OxidationRecipeUpdate,
    db: Session = Depends(get_db),
):
    """Update a recipe"""
    try:
        query = select(OxidationRecipe).where(OxidationRecipe.id == recipe_id)
        result = db.execute(query)
        recipe = result.scalar_one_or_none()

        if not recipe:
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(recipe, key, value)

        db.commit()
        db.refresh(recipe)

        logger.info(f"Updated recipe: {recipe_id}")
        return recipe

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating recipe: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Runs
# ============================================================================

@router.post("/runs", response_model=OxidationRunSchema, status_code=status.HTTP_201_CREATED)
def create_run(
    run: OxidationRunCreate,
    db: Session = Depends(get_db),
):
    """Create a new oxidation run"""
    try:
        # Verify recipe exists
        recipe_query = select(OxidationRecipe).where(OxidationRecipe.id == run.recipe_id)
        recipe_result = db.execute(recipe_query)
        if not recipe_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Verify furnace exists
        furnace_query = select(OxidationFurnace).where(OxidationFurnace.id == run.furnace_id)
        furnace_result = db.execute(furnace_query)
        if not furnace_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Furnace not found")

        # Generate run number if not provided
        run_data = run.model_dump()
        if not run_data.get('run_number'):
            run_data['run_number'] = f"OXI-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        db_run = OxidationRun(**run_data, status=RunStatus.QUEUED)
        db.add(db_run)
        db.commit()
        db.refresh(db_run)

        # TODO: Enqueue run for execution via Celery

        logger.info(f"Created run: {db_run.id} - {db_run.run_number}")
        return db_run

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/batch", response_model=OxidationBatchRunResponse, status_code=status.HTTP_201_CREATED)
def create_batch_runs(
    batch: OxidationBatchRunCreate,
    db: Session = Depends(get_db),
):
    """Create multiple runs for a batch of wafers"""
    try:
        # Verify recipe and furnace exist
        recipe_query = select(OxidationRecipe).where(OxidationRecipe.id == batch.recipe_id)
        if not db.execute(recipe_query).scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Recipe not found")

        furnace_query = select(OxidationFurnace).where(OxidationFurnace.id == batch.furnace_id)
        if not db.execute(furnace_query).scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Furnace not found")

        run_ids = []
        base_run_number = f"OXI-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        for idx, wafer_id in enumerate(batch.wafer_ids):
            run_number = f"{base_run_number}-{idx+1:03d}"

            db_run = OxidationRun(
                run_number=run_number,
                recipe_id=batch.recipe_id,
                furnace_id=batch.furnace_id,
                org_id=batch.org_id,
                lot_id=batch.lot_id,
                wafer_ids=[wafer_id],
                wafer_count=1,
                operator=batch.operator_id,
                status=RunStatus.QUEUED,
            )
            db.add(db_run)
            run_ids.append(db_run.id)

        db.commit()

        logger.info(f"Created batch of {len(run_ids)} runs for lot: {batch.lot_id}")

        return OxidationBatchRunResponse(
            run_ids=run_ids,
            lot_id=batch.lot_id,
            total_runs=len(run_ids),
            status="created"
        )

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating batch runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs", response_model=List[OxidationRunSchema])
def list_runs(
    org_id: Optional[UUID] = None,
    furnace_id: Optional[UUID] = None,
    recipe_id: Optional[UUID] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    lot_id: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "created_at",
    sort_desc: bool = True,
    db: Session = Depends(get_db),
):
    """List oxidation runs with filters and pagination"""
    try:
        query = select(OxidationRun).options(
            joinedload(OxidationRun.recipe),
            joinedload(OxidationRun.furnace)
        )

        # Apply filters
        filters = []
        if org_id:
            filters.append(OxidationRun.org_id == org_id)
        if furnace_id:
            filters.append(OxidationRun.furnace_id == furnace_id)
        if recipe_id:
            filters.append(OxidationRun.recipe_id == recipe_id)
        if status_filter:
            filters.append(OxidationRun.status == status_filter)
        if lot_id:
            filters.append(OxidationRun.lot_id == lot_id)
        if start_date:
            filters.append(OxidationRun.created_at >= start_date)
        if end_date:
            filters.append(OxidationRun.created_at <= end_date)

        if filters:
            query = query.where(and_(*filters))

        # Sorting
        sort_column = getattr(OxidationRun, sort_by, OxidationRun.created_at)
        query = query.order_by(sort_column.desc() if sort_desc else sort_column.asc())

        query = query.offset(skip).limit(limit)

        result = db.execute(query)
        runs = result.scalars().unique().all()

        return runs

    except Exception as e:
        logger.exception(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}", response_model=OxidationRunSchema)
def get_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific run by ID with relationships"""
    try:
        query = select(OxidationRun).where(OxidationRun.id == run_id).options(
            joinedload(OxidationRun.recipe),
            joinedload(OxidationRun.furnace)
        )
        result = db.execute(query)
        run = result.scalar_one_or_none()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        return run

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/runs/{run_id}", response_model=OxidationRunSchema)
def update_run(
    run_id: UUID,
    update_data: OxidationRunUpdate,
    db: Session = Depends(get_db),
):
    """Update a run"""
    try:
        query = select(OxidationRun).where(OxidationRun.id == run_id)
        result = db.execute(query)
        run = result.scalar_one_or_none()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(run, key, value)

        # Calculate duration if end_time is set
        if run.started_at and run.completed_at:
            run.duration_seconds = (run.completed_at - run.started_at).total_seconds()

        db.commit()
        db.refresh(run)

        logger.info(f"Updated run: {run_id}")
        return run

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Results
# ============================================================================

@router.post("/results", response_model=OxidationResultSchema, status_code=status.HTTP_201_CREATED)
def create_result(
    result_data: OxidationResultCreate,
    db: Session = Depends(get_db),
):
    """Create an oxidation result"""
    try:
        # Verify run exists
        run_query = select(OxidationRun).where(OxidationRun.id == result_data.run_id)
        run_result = db.execute(run_query)
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        db_result = OxidationResult(**result_data.model_dump(), org_id=run.org_id)
        db.add(db_result)
        db.commit()
        db.refresh(db_result)

        logger.info(f"Created result for run: {result_data.run_id}")
        return db_result

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/run/{run_id}", response_model=List[OxidationResultSchema])
def get_results_for_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get all results for a specific run"""
    try:
        query = select(OxidationResult).where(OxidationResult.run_id == run_id)
        result = db.execute(query)
        results = result.scalars().all()

        return results

    except Exception as e:
        logger.exception(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}", response_model=OxidationResultSchema)
def get_result(
    result_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific result by ID"""
    try:
        query = select(OxidationResult).where(OxidationResult.id == result_id).options(
            joinedload(OxidationResult.run)
        )
        result = db.execute(query)
        result_data = result.scalar_one_or_none()

        if not result_data:
            raise HTTPException(status_code=404, detail="Result not found")

        return result_data

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/results/{result_id}", response_model=OxidationResultSchema)
def update_result(
    result_id: UUID,
    update_data: OxidationResultUpdate,
    db: Session = Depends(get_db),
):
    """Update a result"""
    try:
        query = select(OxidationResult).where(OxidationResult.id == result_id)
        result = db.execute(query)
        result_obj = result.scalar_one_or_none()

        if not result_obj:
            raise HTTPException(status_code=404, detail="Result not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(result_obj, key, value)

        db.commit()
        db.refresh(result_obj)

        logger.info(f"Updated result: {result_id}")
        return result_obj

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics
# ============================================================================

@router.post("/analytics", response_model=OxidationAnalyticsResponse)
def get_analytics(
    request: OxidationAnalyticsRequest,
    db: Session = Depends(get_db),
):
    """Get analytics for oxidation metrics"""
    try:
        # Build query based on metric
        # This is a simplified version - real implementation would need more complex aggregations
        query = select(OxidationResult).join(OxidationRun)

        # Apply filters
        filters = [OxidationRun.org_id == request.org_id]
        if request.furnace_id:
            filters.append(OxidationRun.furnace_id == request.furnace_id)
        if request.recipe_id:
            filters.append(OxidationRun.recipe_id == request.recipe_id)

        filters.append(OxidationRun.created_at >= request.start_date)
        filters.append(OxidationRun.created_at <= request.end_date)

        query = query.where(and_(*filters))

        result = db.execute(query)
        results = result.scalars().all()

        # Calculate analytics based on metric
        metric_values = []
        if request.metric == "thickness":
            metric_values = [r.thickness_nm for r in results if r.thickness_nm]
        elif request.metric == "uniformity":
            metric_values = [r.uniformity_percent for r in results if r.uniformity_percent]
        elif request.metric == "refractive_index":
            metric_values = [r.refractive_index for r in results if r.refractive_index]
        elif request.metric == "breakdown_voltage":
            metric_values = [r.breakdown_voltage_v for r in results if r.breakdown_voltage_v]
        elif request.metric == "dielectric_constant":
            metric_values = [r.dielectric_constant for r in results if r.dielectric_constant]

        # Calculate summary statistics
        if metric_values:
            import statistics
            summary = {
                "count": len(metric_values),
                "mean": statistics.mean(metric_values),
                "median": statistics.median(metric_values),
                "std": statistics.stdev(metric_values) if len(metric_values) > 1 else 0,
                "min": min(metric_values),
                "max": max(metric_values),
            }
        else:
            summary = {"count": 0}

        return OxidationAnalyticsResponse(
            metric=request.metric,
            aggregation=request.aggregation,
            data=[{"value": v} for v in metric_values],
            summary=summary
        )

    except Exception as e:
        logger.exception(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health & Status
# ============================================================================

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "oxidation-manufacturing",
        "timestamp": datetime.utcnow().isoformat()
    }
