"""
Diffusion Manufacturing - FastAPI Routers
REST API endpoints for manufacturing-grade diffusion platform
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status, WebSocket, WebSocketDisconnect
from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, joinedload

import logging
import json
import asyncio
import sys
from pathlib import Path

# Add services/shared to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from db.deps import get_db
from ..models.diffusion import (
    DiffusionFurnace,
    DiffusionRecipe,
    DiffusionRun,
    DiffusionTelemetry,
    DiffusionResult,
    DiffusionSPCSeries,
    DiffusionSPCPoint,
    RunStatus,
)
from ..schemas.diffusion import (
    DiffusionFurnaceCreate,
    DiffusionFurnaceUpdate,
    DiffusionFurnaceSchema,
    DiffusionRecipeCreate,
    DiffusionRecipeUpdate,
    DiffusionRecipeSchema,
    DiffusionRunCreate,
    DiffusionRunUpdate,
    DiffusionRunSchema,
    DiffusionRunQuery,
    DiffusionTelemetryCreate,
    DiffusionTelemetrySchema,
    DiffusionTelemetryBulkCreate,
    DiffusionResultCreate,
    DiffusionResultUpdate,
    DiffusionResultSchema,
    DiffusionSPCSeriesCreate,
    DiffusionSPCSeriesUpdate,
    DiffusionSPCSeriesSchema,
    DiffusionSPCPointCreate,
    DiffusionSPCPointSchema,
    DiffusionAnalyticsRequest,
    DiffusionAnalyticsResponse,
    DiffusionBatchRunCreate,
    DiffusionBatchRunResponse,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/diffusion", tags=["Diffusion Manufacturing"])


# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    """Manage WebSocket connections for real-time telemetry"""

    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, run_id: str, websocket: WebSocket):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)
        logger.info(f"WebSocket connected for run: {run_id}")

    def disconnect(self, run_id: str, websocket: WebSocket):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]
        logger.info(f"WebSocket disconnected for run: {run_id}")

    async def broadcast(self, run_id: str, message: dict):
        """Broadcast telemetry data to all connected clients for a run"""
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to WebSocket: {e}")


manager = ConnectionManager()


# ============================================================================
# Furnaces
# ============================================================================

@router.post("/furnaces", response_model=DiffusionFurnaceSchema, status_code=status.HTTP_201_CREATED)
def create_furnace(
    furnace: DiffusionFurnaceCreate,
    db: Session = Depends(get_db),
):
    """Create a new diffusion furnace"""
    try:
        db_furnace = DiffusionFurnace(**furnace.model_dump())
        db.add(db_furnace)
        db.commit()
        db.refresh(db_furnace)

        logger.info(f"Created furnace: {db_furnace.id} - {db_furnace.name}")
        return db_furnace

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating furnace: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/furnaces", response_model=List[DiffusionFurnaceSchema])
def list_furnaces(
    org_id: Optional[UUID] = None,
    furnace_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List diffusion furnaces with filters"""
    try:
        query = select(DiffusionFurnace)

        # Apply filters
        filters = []
        if org_id:
            filters.append(DiffusionFurnace.org_id == org_id)
        if furnace_type:
            filters.append(DiffusionFurnace.furnace_type == furnace_type)
        if is_active is not None:
            filters.append(DiffusionFurnace.is_active == is_active)

        if filters:
            query = query.where(and_(*filters))

        query = query.offset(skip).limit(limit).order_by(DiffusionFurnace.created_at.desc())

        result = db.execute(query)
        furnaces = result.scalars().all()

        return furnaces

    except Exception as e:
        logger.exception(f"Error listing furnaces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/furnaces/{furnace_id}", response_model=DiffusionFurnaceSchema)
def get_furnace(
    furnace_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific furnace by ID"""
    try:
        query = select(DiffusionFurnace).where(DiffusionFurnace.id == furnace_id)
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


@router.patch("/furnaces/{furnace_id}", response_model=DiffusionFurnaceSchema)
def update_furnace(
    furnace_id: UUID,
    update_data: DiffusionFurnaceUpdate,
    db: Session = Depends(get_db),
):
    """Update a furnace"""
    try:
        query = select(DiffusionFurnace).where(DiffusionFurnace.id == furnace_id)
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


# ============================================================================
# Recipes
# ============================================================================

@router.post("/recipes", response_model=DiffusionRecipeSchema, status_code=status.HTTP_201_CREATED)
def create_recipe(
    recipe: DiffusionRecipeCreate,
    db: Session = Depends(get_db),
):
    """Create a new diffusion recipe"""
    try:
        # Verify furnace exists
        furnace_query = select(DiffusionFurnace).where(DiffusionFurnace.id == recipe.furnace_id)
        furnace_result = db.execute(furnace_query)
        if not furnace_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Furnace not found")

        db_recipe = DiffusionRecipe(**recipe.model_dump())
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


@router.get("/recipes", response_model=List[DiffusionRecipeSchema])
def list_recipes(
    org_id: Optional[UUID] = None,
    furnace_id: Optional[UUID] = None,
    dopant: Optional[str] = None,
    diffusion_type: Optional[str] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List diffusion recipes with filters"""
    try:
        query = select(DiffusionRecipe).options(joinedload(DiffusionRecipe.furnace))

        # Apply filters
        filters = []
        if org_id:
            filters.append(DiffusionRecipe.org_id == org_id)
        if furnace_id:
            filters.append(DiffusionRecipe.furnace_id == furnace_id)
        if dopant:
            filters.append(DiffusionRecipe.dopant == dopant)
        if diffusion_type:
            filters.append(DiffusionRecipe.diffusion_type == diffusion_type)
        if status_filter:
            filters.append(DiffusionRecipe.status == status_filter)

        if filters:
            query = query.where(and_(*filters))

        query = query.offset(skip).limit(limit).order_by(DiffusionRecipe.created_at.desc())

        result = db.execute(query)
        recipes = result.scalars().unique().all()

        return recipes

    except Exception as e:
        logger.exception(f"Error listing recipes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipes/{recipe_id}", response_model=DiffusionRecipeSchema)
def get_recipe(
    recipe_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific recipe by ID"""
    try:
        query = select(DiffusionRecipe).where(DiffusionRecipe.id == recipe_id).options(
            joinedload(DiffusionRecipe.furnace)
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


@router.patch("/recipes/{recipe_id}", response_model=DiffusionRecipeSchema)
def update_recipe(
    recipe_id: UUID,
    update_data: DiffusionRecipeUpdate,
    db: Session = Depends(get_db),
):
    """Update a recipe"""
    try:
        query = select(DiffusionRecipe).where(DiffusionRecipe.id == recipe_id)
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

@router.post("/runs", response_model=DiffusionRunSchema, status_code=status.HTTP_201_CREATED)
def create_run(
    run: DiffusionRunCreate,
    db: Session = Depends(get_db),
):
    """Create a new diffusion run"""
    try:
        # Verify recipe exists
        recipe_query = select(DiffusionRecipe).where(DiffusionRecipe.id == run.recipe_id)
        recipe_result = db.execute(recipe_query)
        if not recipe_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Recipe not found")

        # Verify furnace exists
        furnace_query = select(DiffusionFurnace).where(DiffusionFurnace.id == run.furnace_id)
        furnace_result = db.execute(furnace_query)
        if not furnace_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Furnace not found")

        # Generate run number if not provided
        run_data = run.model_dump()
        if not run_data.get('run_number'):
            run_data['run_number'] = f"DIFF-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

        db_run = DiffusionRun(**run_data, status=RunStatus.QUEUED)
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


@router.post("/runs/batch", response_model=DiffusionBatchRunResponse, status_code=status.HTTP_201_CREATED)
def create_batch_runs(
    batch: DiffusionBatchRunCreate,
    db: Session = Depends(get_db),
):
    """Create multiple runs for a batch of wafers"""
    try:
        # Verify recipe and furnace exist
        recipe_query = select(DiffusionRecipe).where(DiffusionRecipe.id == batch.recipe_id)
        if not db.execute(recipe_query).scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Recipe not found")

        furnace_query = select(DiffusionFurnace).where(DiffusionFurnace.id == batch.furnace_id)
        if not db.execute(furnace_query).scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Furnace not found")

        run_ids = []
        base_run_number = f"DIFF-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"

        for idx, wafer_id in enumerate(batch.wafer_ids):
            run_number = f"{base_run_number}-{idx+1:03d}"

            db_run = DiffusionRun(
                run_number=run_number,
                recipe_id=batch.recipe_id,
                furnace_id=batch.furnace_id,
                org_id=batch.org_id,
                lot_id=batch.lot_id,
                wafer_ids=[wafer_id],
                wafer_count=1,
                operator_id=batch.operator_id,
                status=RunStatus.QUEUED,
            )
            db.add(db_run)
            run_ids.append(db_run.id)

        db.commit()

        logger.info(f"Created batch of {len(run_ids)} runs for lot: {batch.lot_id}")

        return DiffusionBatchRunResponse(
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


@router.get("/runs", response_model=List[DiffusionRunSchema])
def list_runs(
    org_id: Optional[UUID] = None,
    furnace_id: Optional[UUID] = None,
    recipe_id: Optional[UUID] = None,
    status_filter: Optional[str] = Query(None, alias="status"),
    lot_id: Optional[UUID] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "created_at",
    sort_desc: bool = True,
    db: Session = Depends(get_db),
):
    """List diffusion runs with filters"""
    try:
        query = select(DiffusionRun).options(
            joinedload(DiffusionRun.recipe),
            joinedload(DiffusionRun.furnace)
        )

        # Apply filters
        filters = []
        if org_id:
            filters.append(DiffusionRun.org_id == org_id)
        if furnace_id:
            filters.append(DiffusionRun.furnace_id == furnace_id)
        if recipe_id:
            filters.append(DiffusionRun.recipe_id == recipe_id)
        if status_filter:
            filters.append(DiffusionRun.status == status_filter)
        if lot_id:
            filters.append(DiffusionRun.lot_id == lot_id)
        if start_date:
            filters.append(DiffusionRun.created_at >= start_date)
        if end_date:
            filters.append(DiffusionRun.created_at <= end_date)

        if filters:
            query = query.where(and_(*filters))

        # Sorting
        sort_column = getattr(DiffusionRun, sort_by, DiffusionRun.created_at)
        query = query.order_by(sort_column.desc() if sort_desc else sort_column.asc())

        query = query.offset(skip).limit(limit)

        result = db.execute(query)
        runs = result.scalars().unique().all()

        return runs

    except Exception as e:
        logger.exception(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}", response_model=DiffusionRunSchema)
def get_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific run by ID"""
    try:
        query = select(DiffusionRun).where(DiffusionRun.id == run_id).options(
            joinedload(DiffusionRun.recipe),
            joinedload(DiffusionRun.furnace)
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


@router.patch("/runs/{run_id}", response_model=DiffusionRunSchema)
def update_run(
    run_id: UUID,
    update_data: DiffusionRunUpdate,
    db: Session = Depends(get_db),
):
    """Update a run"""
    try:
        query = select(DiffusionRun).where(DiffusionRun.id == run_id)
        result = db.execute(query)
        run = result.scalar_one_or_none()

        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(run, key, value)

        # Calculate duration if end_time is set
        if run.start_time and run.end_time:
            run.duration_seconds = (run.end_time - run.start_time).total_seconds()

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
# Telemetry
# ============================================================================

@router.post("/telemetry", response_model=DiffusionTelemetrySchema, status_code=status.HTTP_201_CREATED)
def create_telemetry(
    telemetry: DiffusionTelemetryCreate,
    db: Session = Depends(get_db),
):
    """Create a single telemetry data point"""
    try:
        # Verify run exists
        run_query = select(DiffusionRun).where(DiffusionRun.id == telemetry.run_id)
        run_result = db.execute(run_query)
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        db_telemetry = DiffusionTelemetry(**telemetry.model_dump(), org_id=run.org_id)
        db.add(db_telemetry)
        db.commit()
        db.refresh(db_telemetry)

        # Broadcast to WebSocket clients
        asyncio.create_task(
            manager.broadcast(
                str(telemetry.run_id),
                {
                    "type": "telemetry",
                    "data": {
                        "ts": telemetry.ts.isoformat(),
                        "temperature_zones_c": telemetry.temperature_zones_c,
                        "ambient_gas": telemetry.ambient_gas,
                        "flow_rate_slm": telemetry.flow_rate_slm,
                    }
                }
            )
        )

        return db_telemetry

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/bulk", status_code=status.HTTP_201_CREATED)
def create_telemetry_bulk(
    bulk_data: DiffusionTelemetryBulkCreate,
    db: Session = Depends(get_db),
):
    """Bulk insert telemetry data points"""
    try:
        # Verify run exists
        run_query = select(DiffusionRun).where(DiffusionRun.id == bulk_data.run_id)
        run_result = db.execute(run_query)
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        # Insert all points
        for point in bulk_data.data_points:
            db_telemetry = DiffusionTelemetry(**point.model_dump(), org_id=run.org_id)
            db.add(db_telemetry)

        db.commit()

        logger.info(f"Bulk inserted {len(bulk_data.data_points)} telemetry points for run {bulk_data.run_id}")

        return {"status": "success", "count": len(bulk_data.data_points)}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error bulk creating telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry/run/{run_id}", response_model=List[DiffusionTelemetrySchema])
def get_telemetry_for_run(
    run_id: UUID,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 1000,
    db: Session = Depends(get_db),
):
    """Get telemetry data for a specific run"""
    try:
        query = select(DiffusionTelemetry).where(DiffusionTelemetry.run_id == run_id)

        # Apply time filters
        if start_time:
            query = query.where(DiffusionTelemetry.ts >= start_time)
        if end_time:
            query = query.where(DiffusionTelemetry.ts <= end_time)

        query = query.order_by(DiffusionTelemetry.ts.asc()).offset(skip).limit(limit)

        result = db.execute(query)
        telemetry = result.scalars().all()

        return telemetry

    except Exception as e:
        logger.exception(f"Error getting telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/telemetry/{run_id}")
async def telemetry_websocket(
    websocket: WebSocket,
    run_id: str,
):
    """WebSocket endpoint for real-time telemetry streaming"""
    await manager.connect(run_id, websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_json({"type": "heartbeat", "status": "connected"})
    except WebSocketDisconnect:
        manager.disconnect(run_id, websocket)
        logger.info(f"WebSocket disconnected for run: {run_id}")


# ============================================================================
# Results
# ============================================================================

@router.post("/results", response_model=DiffusionResultSchema, status_code=status.HTTP_201_CREATED)
def create_result(
    result_data: DiffusionResultCreate,
    db: Session = Depends(get_db),
):
    """Create a diffusion result"""
    try:
        # Verify run exists
        run_query = select(DiffusionRun).where(DiffusionRun.id == result_data.run_id)
        run_result = db.execute(run_query)
        run = run_result.scalar_one_or_none()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        db_result = DiffusionResult(**result_data.model_dump(), org_id=run.org_id)
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


@router.get("/results/run/{run_id}", response_model=List[DiffusionResultSchema])
def get_results_for_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get results for a specific run"""
    try:
        query = select(DiffusionResult).where(DiffusionResult.run_id == run_id)
        result = db.execute(query)
        results = result.scalars().all()

        return results

    except Exception as e:
        logger.exception(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{result_id}", response_model=DiffusionResultSchema)
def get_result(
    result_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific result by ID"""
    try:
        query = select(DiffusionResult).where(DiffusionResult.id == result_id).options(
            joinedload(DiffusionResult.run)
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


@router.patch("/results/{result_id}", response_model=DiffusionResultSchema)
def update_result(
    result_id: UUID,
    update_data: DiffusionResultUpdate,
    db: Session = Depends(get_db),
):
    """Update a result"""
    try:
        query = select(DiffusionResult).where(DiffusionResult.id == result_id)
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
# SPC
# ============================================================================

@router.post("/spc/series", response_model=DiffusionSPCSeriesSchema, status_code=status.HTTP_201_CREATED)
def create_spc_series(
    series: DiffusionSPCSeriesCreate,
    db: Session = Depends(get_db),
):
    """Create a new SPC series"""
    try:
        db_series = DiffusionSPCSeries(**series.model_dump())
        db.add(db_series)
        db.commit()
        db.refresh(db_series)

        logger.info(f"Created SPC series: {db_series.id} - {db_series.name}")
        return db_series

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating SPC series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spc/series", response_model=List[DiffusionSPCSeriesSchema])
def list_spc_series(
    org_id: Optional[UUID] = None,
    recipe_id: Optional[UUID] = None,
    furnace_id: Optional[UUID] = None,
    parameter: Optional[str] = None,
    is_active: Optional[bool] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List SPC series with filters"""
    try:
        query = select(DiffusionSPCSeries)

        # Apply filters
        filters = []
        if org_id:
            filters.append(DiffusionSPCSeries.org_id == org_id)
        if recipe_id:
            filters.append(DiffusionSPCSeries.recipe_id == recipe_id)
        if furnace_id:
            filters.append(DiffusionSPCSeries.furnace_id == furnace_id)
        if parameter:
            filters.append(DiffusionSPCSeries.parameter == parameter)
        if is_active is not None:
            filters.append(DiffusionSPCSeries.is_active == is_active)

        if filters:
            query = query.where(and_(*filters))

        query = query.offset(skip).limit(limit).order_by(DiffusionSPCSeries.created_at.desc())

        result = db.execute(query)
        series = result.scalars().all()

        return series

    except Exception as e:
        logger.exception(f"Error listing SPC series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spc/series/{series_id}", response_model=DiffusionSPCSeriesSchema)
def get_spc_series(
    series_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific SPC series by ID"""
    try:
        query = select(DiffusionSPCSeries).where(DiffusionSPCSeries.id == series_id)
        result = db.execute(query)
        series = result.scalar_one_or_none()

        if not series:
            raise HTTPException(status_code=404, detail="SPC series not found")

        return series

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting SPC series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/spc/series/{series_id}", response_model=DiffusionSPCSeriesSchema)
def update_spc_series(
    series_id: UUID,
    update_data: DiffusionSPCSeriesUpdate,
    db: Session = Depends(get_db),
):
    """Update an SPC series"""
    try:
        query = select(DiffusionSPCSeries).where(DiffusionSPCSeries.id == series_id)
        result = db.execute(query)
        series = result.scalar_one_or_none()

        if not series:
            raise HTTPException(status_code=404, detail="SPC series not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(series, key, value)

        db.commit()
        db.refresh(series)

        logger.info(f"Updated SPC series: {series_id}")
        return series

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating SPC series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spc/points", response_model=DiffusionSPCPointSchema, status_code=status.HTTP_201_CREATED)
def create_spc_point(
    point: DiffusionSPCPointCreate,
    db: Session = Depends(get_db),
):
    """Create a new SPC point"""
    try:
        # Verify series exists
        series_query = select(DiffusionSPCSeries).where(DiffusionSPCSeries.id == point.series_id)
        series_result = db.execute(series_query)
        series = series_result.scalar_one_or_none()
        if not series:
            raise HTTPException(status_code=404, detail="SPC series not found")

        db_point = DiffusionSPCPoint(**point.model_dump(), org_id=series.org_id)
        db.add(db_point)
        db.commit()
        db.refresh(db_point)

        logger.info(f"Created SPC point for series: {point.series_id}")
        return db_point

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating SPC point: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spc/points/{series_id}", response_model=List[DiffusionSPCPointSchema])
def get_spc_points(
    series_id: UUID,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """Get SPC points for a series"""
    try:
        query = (
            select(DiffusionSPCPoint)
            .where(DiffusionSPCPoint.series_id == series_id)
            .order_by(DiffusionSPCPoint.ts.desc())
            .limit(limit)
        )

        result = db.execute(query)
        points = result.scalars().all()

        return points

    except Exception as e:
        logger.exception(f"Error getting SPC points: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Analytics
# ============================================================================

@router.post("/analytics", response_model=DiffusionAnalyticsResponse)
def get_analytics(
    request: DiffusionAnalyticsRequest,
    db: Session = Depends(get_db),
):
    """Get analytics for diffusion metrics"""
    try:
        # Build query based on metric
        # This is a simplified version - real implementation would need more complex aggregations
        query = select(DiffusionResult).join(DiffusionRun)

        # Apply filters
        filters = [DiffusionRun.org_id == request.org_id]
        if request.furnace_id:
            filters.append(DiffusionRun.furnace_id == request.furnace_id)
        if request.recipe_id:
            filters.append(DiffusionRun.recipe_id == request.recipe_id)

        filters.append(DiffusionRun.created_at >= request.start_date)
        filters.append(DiffusionRun.created_at <= request.end_date)

        query = query.where(and_(*filters))

        result = db.execute(query)
        results = result.scalars().all()

        # Calculate analytics based on metric
        metric_values = []
        if request.metric == "sheet_resistance":
            metric_values = [r.sheet_resistance_ohm_per_sq for r in results if r.sheet_resistance_ohm_per_sq]
        elif request.metric == "junction_depth":
            metric_values = [r.junction_depth_um for r in results if r.junction_depth_um]
        elif request.metric == "uniformity":
            metric_values = [r.uniformity_score for r in results if r.uniformity_score]

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

        return DiffusionAnalyticsResponse(
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
        "service": "diffusion-manufacturing",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/furnaces/{furnace_id}/status")
def get_furnace_status(
    furnace_id: UUID,
    db: Session = Depends(get_db),
):
    """Get current status of a furnace"""
    try:
        # Check if furnace exists
        furnace_query = select(DiffusionFurnace).where(DiffusionFurnace.id == furnace_id)
        furnace = db.execute(furnace_query).scalar_one_or_none()
        if not furnace:
            raise HTTPException(status_code=404, detail="Furnace not found")

        # Check for active runs
        active_runs_query = (
            select(DiffusionRun)
            .where(
                and_(
                    DiffusionRun.furnace_id == furnace_id,
                    DiffusionRun.status.in_([RunStatus.RUNNING, RunStatus.QUEUED])
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
