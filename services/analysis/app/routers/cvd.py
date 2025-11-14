"""
CVD Platform - FastAPI Routers
REST API endpoints for all CVD operations
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
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
from ..models.cvd import (
    CVDProcessMode,
    CVDRecipe,
    CVDRun,
    CVDTelemetry,
    CVDResult,
    CVDSPCSeries,
    CVDSPCPoint,
)
from ..schemas.cvd import (
    CVDProcessModeCreate,
    CVDProcessModeUpdate,
    CVDProcessModeSchema,
    CVDRecipeCreate,
    CVDRecipeUpdate,
    CVDRecipeSchema,
    CVDRunCreate,
    CVDRunUpdate,
    CVDRunSchema,
    CVDRunQuery,
    CVDTelemetryCreate,
    CVDTelemetrySchema,
    CVDTelemetryBulkCreate,
    CVDResultCreate,
    CVDResultUpdate,
    CVDResultSchema,
    CVDSPCSeriesCreate,
    CVDSPCSeriesUpdate,
    CVDSPCSeriesSchema,
    CVDSPCPointCreate,
    CVDSPCPointSchema,
    CVDAnalyticsRequest,
    CVDAnalyticsResponse,
    CVDAlarmCreate,
    CVDAlarmSchema,
    CVDAlarmAcknowledge,
    CVDControlActionCreate,
    CVDControlActionSchema,
    CVDBatchRunCreate,
    CVDBatchRunResponse,
    CVDExportRequest,
    CVDExportResponse,
    RunStatus,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/cvd", tags=["CVD Platform"])


# ============================================================================
# Dependency Injection
# ============================================================================

# TODO: Add authentication/authorization dependencies
# async def get_current_user() -> User:
#     ...


# ============================================================================
# Process Modes
# ============================================================================

@router.post("/process-modes", response_model=CVDProcessModeSchema, status_code=status.HTTP_201_CREATED)
def create_process_mode(
    process_mode: CVDProcessModeCreate,
    db: Session = Depends(get_db),
):
    """Create a new CVD process mode"""
    try:
        db_process_mode = CVDProcessMode(**process_mode.model_dump())
        db.add(db_process_mode)
        db.commit()
        db.refresh(db_process_mode)

        logger.info(f"Created process mode: {db_process_mode.id}")
        return db_process_mode

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating process mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/process-modes", response_model=List[CVDProcessModeSchema])
def list_process_modes(
    org_id: Optional[UUID] = None,
    pressure_mode: Optional[str] = None,
    energy_mode: Optional[str] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List CVD process modes with filters"""
    try:
        query = select(CVDProcessMode)

        # Apply filters
        filters = []
        if org_id:
            filters.append(CVDProcessMode.org_id == org_id)
        if pressure_mode:
            filters.append(CVDProcessMode.pressure_mode == pressure_mode)
        if energy_mode:
            filters.append(CVDProcessMode.energy_mode == energy_mode)

        if filters:
            query = query.where(and_(*filters))

        query = query.offset(skip).limit(limit).order_by(CVDProcessMode.created_at.desc())

        result = db.execute(query)
        process_modes = result.scalars().all()

        return process_modes

    except Exception as e:
        logger.exception(f"Error listing process modes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/process-modes/{process_mode_id}", response_model=CVDProcessModeSchema)
def get_process_mode(
    process_mode_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific process mode by ID"""
    try:
        query = select(CVDProcessMode).where(CVDProcessMode.id == process_mode_id)
        result = db.execute(query)
        process_mode = result.scalar_one_or_none()

        if not process_mode:
            raise HTTPException(status_code=404, detail="Process mode not found")

        return process_mode

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting process mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/process-modes/{process_mode_id}", response_model=CVDProcessModeSchema)
def update_process_mode(
    process_mode_id: UUID,
    update_data: CVDProcessModeUpdate,
    db: Session = Depends(get_db),
):
    """Update a process mode"""
    try:
        query = select(CVDProcessMode).where(CVDProcessMode.id == process_mode_id)
        result = db.execute(query)
        process_mode = result.scalar_one_or_none()

        if not process_mode:
            raise HTTPException(status_code=404, detail="Process mode not found")

        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(process_mode, key, value)

        db.commit()
        db.refresh(process_mode)

        logger.info(f"Updated process mode: {process_mode_id}")
        return process_mode

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating process mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Recipes
# ============================================================================

@router.post("/recipes", response_model=CVDRecipeSchema, status_code=status.HTTP_201_CREATED)
def create_recipe(
    recipe: CVDRecipeCreate,
    db: Session = Depends(get_db),
):
    """Create a new CVD recipe"""
    try:
        # Verify process mode exists
        process_mode_query = select(CVDProcessMode).where(CVDProcessMode.id == recipe.process_mode_id)
        process_mode_result = db.execute(process_mode_query)
        if not process_mode_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Process mode not found")

        db_recipe = CVDRecipe(**recipe.model_dump())
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


@router.get("/recipes", response_model=List[CVDRecipeSchema])
def list_recipes(
    org_id: Optional[UUID] = None,
    process_mode_id: Optional[UUID] = None,
    search: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
):
    """List CVD recipes with filters"""
    try:
        query = select(CVDRecipe).options(joinedload(CVDRecipe.process_mode))

        # Apply filters
        filters = []
        if org_id:
            filters.append(CVDRecipe.org_id == org_id)
        if process_mode_id:
            filters.append(CVDRecipe.process_mode_id == process_mode_id)

        if search:
            filters.append(
                or_(
                    CVDRecipe.name.ilike(f"%{search}%"),
                    CVDRecipe.description.ilike(f"%{search}%"),
                )
            )

        if tags:
            # JSONB array contains
            for tag in tags:
                filters.append(CVDRecipe.tags.contains([tag]))

        if filters:
            query = query.where(and_(*filters))

        query = query.offset(skip).limit(limit).order_by(CVDRecipe.created_at.desc())

        result = db.execute(query)
        recipes = result.scalars().unique().all()

        return recipes

    except Exception as e:
        logger.exception(f"Error listing recipes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recipes/{recipe_id}", response_model=CVDRecipeSchema)
def get_recipe(
    recipe_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific recipe by ID"""
    try:
        query = select(CVDRecipe).where(CVDRecipe.id == recipe_id).options(joinedload(CVDRecipe.process_mode))
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


@router.patch("/recipes/{recipe_id}", response_model=CVDRecipeSchema)
def update_recipe(
    recipe_id: UUID,
    update_data: CVDRecipeUpdate,
    db: Session = Depends(get_db),
):
    """Update a recipe"""
    try:
        query = select(CVDRecipe).where(CVDRecipe.id == recipe_id)
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

@router.post("/runs", response_model=CVDRunSchema, status_code=status.HTTP_201_CREATED)
def create_run(
    run: CVDRunCreate,
    db: Session = Depends(get_db),
):
    """Create a new CVD run"""
    try:
        # Verify recipe exists
        recipe_query = select(CVDRecipe).where(CVDRecipe.id == run.cvd_recipe_id)
        recipe_result = db.execute(recipe_query)
        if not recipe_result.scalar_one_or_none():
            raise HTTPException(status_code=404, detail="Recipe not found")

        db_run = CVDRun(**run.model_dump(), status=RunStatus.QUEUED)
        db.add(db_run)
        db.commit()
        db.refresh(db_run)

        # TODO: Enqueue run for execution via Celery

        logger.info(f"Created run: {db_run.id}")
        return db_run

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating run: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/batch", response_model=CVDBatchRunResponse, status_code=status.HTTP_201_CREATED)
def create_batch_runs(
    batch: CVDBatchRunCreate,
    db: Session = Depends(get_db),
):
    """Create multiple runs for a batch of wafers"""
    try:
        run_ids = []

        for wafer_id in batch.wafer_ids:
            run_data = CVDRunCreate(
                recipe_id=batch.cvd_recipe_id,
                process_mode_id=batch.process_mode_id,
                tool_id=batch.instrument_id,
                organization_id=batch.org_id,
                lot_id=batch.lot_id,
                wafer_ids=[wafer_id],
                operator_id=batch.operator_id,
            )

            db_run = CVDRun(**run_data.model_dump(), status=RunStatus.QUEUED)
            db.add(db_run)
            run_ids.append(db_run.id)

        db.commit()

        logger.info(f"Created batch of {len(run_ids)} runs for lot {batch.lot_id}")

        return CVDBatchRunResponse(
            run_ids=run_ids,
            lot_id=batch.lot_id,
            total_runs=len(run_ids),
            status="queued",
        )

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating batch runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs", response_model=List[CVDRunSchema])
def list_runs(
    query_params: CVDRunQuery = Depends(),
    db: Session = Depends(get_db),
):
    """List CVD runs with filters"""
    try:
        query = select(CVDRun).options(
            joinedload(CVDRun.recipe),
            joinedload(CVDRun.process_mode),
        )

        # Apply filters
        filters = []
        if query_params.org_id:
            filters.append(CVDRun.org_id == query_params.org_id)
        if query_params.process_mode_id:
            filters.append(CVDRun.process_mode_id == query_params.process_mode_id)
        if query_params.cvd_recipe_id:
            filters.append(CVDRun.cvd_recipe_id == query_params.cvd_recipe_id)
        if query_params.instrument_id:
            filters.append(CVDRun.instrument_id == query_params.instrument_id)
        if query_params.status:
            filters.append(CVDRun.status == query_params.status)
        if query_params.lot_id:
            filters.append(CVDRun.lot_id == query_params.lot_id)

        if query_params.start_date:
            filters.append(CVDRun.created_at >= query_params.start_date)
        if query_params.end_date:
            filters.append(CVDRun.created_at <= query_params.end_date)

        if filters:
            query = query.where(and_(*filters))

        # Sorting
        if query_params.sort_desc:
            query = query.order_by(getattr(CVDRun, query_params.sort_by).desc())
        else:
            query = query.order_by(getattr(CVDRun, query_params.sort_by).asc())

        query = query.offset(query_params.skip).limit(query_params.limit)

        result = db.execute(query)
        runs = result.scalars().unique().all()

        return runs

    except Exception as e:
        logger.exception(f"Error listing runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}", response_model=CVDRunSchema)
def get_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get a specific run by ID"""
    try:
        query = select(CVDRun).where(CVDRun.id == run_id).options(
            joinedload(CVDRun.recipe),
            joinedload(CVDRun.process_mode),
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


@router.patch("/runs/{run_id}", response_model=CVDRunSchema)
def update_run(
    run_id: UUID,
    update_data: CVDRunUpdate,
    db: Session = Depends(get_db),
):
    """Update a run"""
    try:
        query = select(CVDRun).where(CVDRun.id == run_id)
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

        logger.info(f"Updated run: {run_id}, status: {run.status}")
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

@router.post("/telemetry", response_model=CVDTelemetrySchema, status_code=status.HTTP_201_CREATED)
def create_telemetry(
    telemetry: CVDTelemetryCreate,
    db: Session = Depends(get_db),
):
    """Create a single telemetry point"""
    try:
        db_telemetry = CVDTelemetry(**telemetry.model_dump())
        db.add(db_telemetry)
        db.commit()
        db.refresh(db_telemetry)

        return db_telemetry

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/telemetry/bulk", status_code=status.HTTP_201_CREATED)
def create_telemetry_bulk(
    bulk_data: CVDTelemetryBulkCreate,
    db: Session = Depends(get_db),
):
    """Bulk insert telemetry points"""
    try:
        db_telemetry_list = [CVDTelemetry(**point.model_dump()) for point in bulk_data.data_points]

        db.add_all(db_telemetry_list)
        db.commit()

        logger.info(f"Bulk inserted {len(db_telemetry_list)} telemetry points for run {bulk_data.run_id}")

        return {"status": "success", "count": len(db_telemetry_list)}

    except Exception as e:
        db.rollback()
        logger.exception(f"Error bulk creating telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/telemetry/run/{run_id}", response_model=List[CVDTelemetrySchema])
def get_telemetry_for_run(
    run_id: UUID,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 10000,
    db: Session = Depends(get_db),
):
    """Get telemetry data for a run"""
    try:
        query = select(CVDTelemetry).where(CVDTelemetry.run_id == run_id)

        if start_time:
            query = query.where(CVDTelemetry.timestamp >= start_time)
        if end_time:
            query = query.where(CVDTelemetry.timestamp <= end_time)

        query = query.order_by(CVDTelemetry.timestamp.asc()).offset(skip).limit(limit)

        result = db.execute(query)
        telemetry = result.scalars().all()

        return telemetry

    except Exception as e:
        logger.exception(f"Error getting telemetry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/telemetry/{run_id}")
async def websocket_telemetry_stream(
    websocket: WebSocket,
    run_id: UUID,
):
    """WebSocket endpoint for real-time telemetry streaming"""
    await websocket.accept()

    logger.info(f"WebSocket connection established for run {run_id}")

    try:
        # TODO: Connect to actual telemetry stream from tool/simulator
        # For now, simulate with periodic updates

        while True:
            # Simulate telemetry data
            data = {
                "run_id": str(run_id),
                "timestamp": datetime.utcnow().isoformat(),
                "temperatures": {"zone_1": 650.0, "zone_2": 649.5},
                "pressures": {"chamber": 100.0},
                "gas_flows": {"SiH4": 50.0, "N2": 1000.0},
            }

            await websocket.send_json(data)
            await asyncio.sleep(1.0)  # 1 Hz

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
        await websocket.close()


# ============================================================================
# Results
# ============================================================================

@router.post("/results", response_model=CVDResultSchema, status_code=status.HTTP_201_CREATED)
def create_result(
    result: CVDResultCreate,
    db: Session = Depends(get_db),
):
    """Create a CVD result (metrology data)"""
    try:
        db_result = CVDResult(**result.model_dump())
        db.add(db_result)
        db.commit()
        db.refresh(db_result)

        logger.info(f"Created result for run {result.run_id}, wafer {result.wafer_id}")
        return db_result

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating result: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/run/{run_id}", response_model=List[CVDResultSchema])
def get_results_for_run(
    run_id: UUID,
    db: Session = Depends(get_db),
):
    """Get all results for a run"""
    try:
        query = select(CVDResult).where(CVDResult.run_id == run_id).options(joinedload(CVDResult.run))
        result = db.execute(query)
        results = result.scalars().unique().all()

        return results

    except Exception as e:
        logger.exception(f"Error getting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SPC
# ============================================================================

@router.post("/spc/series", response_model=CVDSPCSeriesSchema, status_code=status.HTTP_201_CREATED)
def create_spc_series(
    series: CVDSPCSeriesCreate,
    db: Session = Depends(get_db),
):
    """Create an SPC control chart series"""
    try:
        db_series = CVDSPCSeries(**series.model_dump())
        db.add(db_series)
        db.commit()
        db.refresh(db_series)

        logger.info(f"Created SPC series: {db_series.id} - {db_series.metric_name}")
        return db_series

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating SPC series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spc/series", response_model=List[CVDSPCSeriesSchema])
def list_spc_series(
    org_id: Optional[UUID] = None,
    cvd_recipe_id: Optional[UUID] = None,
    metric_name: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List SPC series"""
    try:
        query = select(CVDSPCSeries)

        filters = []
        if org_id:
            filters.append(CVDSPCSeries.org_id == org_id)
        if cvd_recipe_id:
            filters.append(CVDSPCSeries.cvd_recipe_id == cvd_recipe_id)
        if metric_name:
            filters.append(CVDSPCSeries.metric_name == metric_name)

        if filters:
            query = query.where(and_(*filters))

        result = db.execute(query)
        series = result.scalars().all()

        return series

    except Exception as e:
        logger.exception(f"Error listing SPC series: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spc/points", response_model=CVDSPCPointSchema, status_code=status.HTTP_201_CREATED)
def create_spc_point(
    point: CVDSPCPointCreate,
    db: Session = Depends(get_db),
):
    """Add a point to an SPC series"""
    try:
        db_point = CVDSPCPoint(**point.model_dump())
        db.add(db_point)
        db.commit()
        db.refresh(db_point)

        return db_point

    except Exception as e:
        db.rollback()
        logger.exception(f"Error creating SPC point: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/spc/points/{series_id}", response_model=List[CVDSPCPointSchema])
def get_spc_points(
    series_id: UUID,
    limit: int = 1000,
    db: Session = Depends(get_db),
):
    """Get points for an SPC series"""
    try:
        query = (
            select(CVDSPCPoint)
            .where(CVDSPCPoint.series_id == series_id)
            .order_by(CVDSPCPoint.timestamp.desc())
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

@router.post("/analytics", response_model=CVDAnalyticsResponse)
def get_analytics(
    request: CVDAnalyticsRequest,
    db: Session = Depends(get_db),
):
    """Get analytics data for specified metric"""
    try:
        # Build query based on metric
        if request.metric == "thickness":
            base_query = select(CVDResult.thickness_nm, CVDResult.measurement_timestamp, CVDRun.cvd_recipe_id, CVDRun.instrument_id)
            base_query = base_query.join(CVDRun, CVDResult.run_id == CVDRun.id)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported metric: {request.metric}")

        # Apply filters
        filters = [CVDRun.org_id == request.org_id]

        if request.cvd_recipe_id:
            filters.append(CVDRun.cvd_recipe_id == request.cvd_recipe_id)
        if request.instrument_id:
            filters.append(CVDRun.instrument_id == request.instrument_id)

        filters.append(CVDResult.measurement_timestamp >= request.start_date)
        filters.append(CVDResult.measurement_timestamp <= request.end_date)

        base_query = base_query.where(and_(*filters))

        # Execute query
        result = db.execute(base_query)
        rows = result.all()

        # Process data
        values = [row[0] for row in rows if row[0] is not None]

        if not values:
            return CVDAnalyticsResponse(
                metric=request.metric,
                aggregation=request.aggregation,
                data=[],
                summary={"total_count": 0},
            )

        # Calculate summary statistics
        import numpy as np

        summary = {
            "total_count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

        # Format data (simplified)
        data = [
            {
                "timestamp": row[1].isoformat() if row[1] else None,
                "value": float(row[0]) if row[0] else None,
            }
            for row in rows if row[0] is not None
        ]

        return CVDAnalyticsResponse(
            metric=request.metric,
            aggregation=request.aggregation,
            data=data[:1000],  # Limit response size
            summary=summary,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error getting analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "cvd-platform",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Tool Status (for simulator integration)
# ============================================================================

@router.get("/tools/{tool_id}/status")
def get_tool_status(instrument_id: UUID):
    """Get real-time tool status from simulator/hardware"""
    # TODO: Integrate with CVDToolManager to get actual status
    return {
        "tool_id": str(tool_id),
        "state": "IDLE",
        "current_run_id": None,
        "message": "Tool status endpoint - integration pending",
    }
