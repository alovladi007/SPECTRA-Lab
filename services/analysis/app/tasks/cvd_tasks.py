"""
CVD Platform - Celery Tasks
Async task queue for CVD operations
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID
import logging
import asyncio

from celery import Celery, Task, group, chain
from celery.schedules import crontab
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from ..models.cvd import CVDRun, CVDRecipe, CVDResult, CVDTelemetry, CVDSPCPoint, CVDSPCSeries
from ..schemas.cvd import RunStatus
from ..tools.base import CVDToolManager, ToolState
from ..simulators.lpcvd_thermal import LPCVDThermalSimulator
from ..simulators.pecvd_plasma import PECVDPlasmaSimulator


logger = logging.getLogger(__name__)

# Initialize Celery app
app = Celery(
    "cvd_platform",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=7200,  # 2 hours max
    worker_prefetch_multiplier=1,
)

# Database setup for async tasks
DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/cvd_platform"
async_engine = create_async_engine(DATABASE_URL, echo=False, future=True)
AsyncSessionLocal = sessionmaker(async_engine, class_=AsyncSession, expire_on_commit=False)

# Global tool manager
tool_manager = CVDToolManager()


# ============================================================================
# Base Task Class
# ============================================================================

class CVDTask(Task):
    """Base task with error handling and logging"""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handler for task failure"""
        logger.error(f"Task {self.name} [{task_id}] failed: {exc}")
        logger.error(f"Exception info: {einfo}")

    def on_success(self, retval, task_id, args, kwargs):
        """Handler for task success"""
        logger.info(f"Task {self.name} [{task_id}] completed successfully")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Handler for task retry"""
        logger.warning(f"Task {self.name} [{task_id}] retrying: {exc}")


# ============================================================================
# Run Execution Tasks
# ============================================================================

@app.task(base=CVDTask, bind=True, max_retries=3)
def execute_cvd_run(self, run_id: str):
    """
    Execute a CVD run on simulator or hardware.
    This is the main task for running a process.

    Args:
        run_id: UUID of the run to execute
    """
    logger.info(f"Starting CVD run execution: {run_id}")

    # Use asyncio to run async function
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_execute_cvd_run_async(UUID(run_id)))

    return result


async def _execute_cvd_run_async(run_id: UUID) -> Dict[str, Any]:
    """Async implementation of run execution"""
    async with AsyncSessionLocal() as db:
        try:
            # Get run details
            query = select(CVDRun).where(CVDRun.id == run_id)
            result = await db.execute(query)
            run = result.scalar_one_or_none()

            if not run:
                raise ValueError(f"Run {run_id} not found")

            # Get recipe
            recipe_query = select(CVDRecipe).where(CVDRecipe.id == run.recipe_id)
            recipe_result = await db.execute(recipe_query)
            recipe = recipe_result.scalar_one_or_none()

            if not recipe:
                raise ValueError(f"Recipe {run.recipe_id} not found")

            # Update run status
            run.status = RunStatus.INITIALIZING
            run.start_time = datetime.utcnow()
            await db.commit()

            # Get or create tool simulator
            tool = tool_manager.get_tool(run.tool_id)

            if not tool:
                # Create simulator based on process mode
                process_mode = recipe.process_mode

                if process_mode.energy_mode.value == "THERMAL":
                    tool = LPCVDThermalSimulator(
                        tool_id=run.tool_id,
                        tool_name=f"LPCVD-Sim-{run.tool_id}",
                        material="Si",  # TODO: Get from recipe
                    )
                elif process_mode.energy_mode.value == "PLASMA":
                    tool = PECVDPlasmaSimulator(
                        tool_id=run.tool_id,
                        tool_name=f"PECVD-Sim-{run.tool_id}",
                        material="SiO2",  # TODO: Get from recipe
                    )
                else:
                    raise ValueError(f"Unsupported energy mode: {process_mode.energy_mode}")

                tool_manager.register_tool(tool)
                await tool.initialize_hardware()

            # Reset film thickness for new run
            if hasattr(tool, 'reset_film_thickness'):
                tool.reset_film_thickness()

            # Prepare recipe dict
            recipe_dict = {
                "id": str(recipe.id),
                "name": recipe.name,
                "temperature_profile": recipe.temperature_profile,
                "gas_flows": recipe.gas_flows,
                "pressure_profile": recipe.pressure_profile,
                "plasma_settings": recipe.plasma_settings,
                "recipe_steps": recipe.recipe_steps,
                "process_time_s": recipe.process_time_s,
            }

            # Telemetry callback
            telemetry_buffer = []

            async def telemetry_callback(telemetry_point):
                """Callback to save telemetry to database"""
                telemetry_data = CVDTelemetry(
                    run_id=run_id,
                    timestamp=telemetry_point.timestamp,
                    temperatures=telemetry_point.temperatures,
                    pressures=telemetry_point.pressures,
                    gas_flows=telemetry_point.gas_flows,
                    plasma_parameters=telemetry_point.plasma_parameters,
                    rotation_speed_rpm=telemetry_point.rotation_speed_rpm,
                )

                telemetry_buffer.append(telemetry_data)

                # Bulk insert every 10 points
                if len(telemetry_buffer) >= 10:
                    db.add_all(telemetry_buffer)
                    await db.commit()
                    telemetry_buffer.clear()

            # Execute recipe
            logger.info(f"Executing recipe {recipe.name} for run {run_id}")
            success = await tool.execute_recipe(recipe_dict, run_id, telemetry_callback)

            # Flush remaining telemetry
            if telemetry_buffer:
                db.add_all(telemetry_buffer)
                await db.commit()

            # Update run status
            if success:
                run.status = RunStatus.COMPLETED
                run.end_time = datetime.utcnow()
                run.duration_s = (run.end_time - run.start_time).total_seconds()

                # Get final results from simulator
                if hasattr(tool, 'get_film_thickness'):
                    final_thickness = tool.get_film_thickness()
                    final_uniformity = tool.get_uniformity()

                    # Save results for each wafer
                    for wafer_id in run.wafer_ids:
                        result = CVDResult(
                            run_id=run_id,
                            wafer_id=wafer_id,
                            thickness_nm=final_thickness,
                            uniformity_pct=final_uniformity,
                            measurement_timestamp=datetime.utcnow(),
                        )
                        db.add(result)

                logger.info(f"Run {run_id} completed successfully. Thickness: {final_thickness:.2f} nm")

            else:
                run.status = RunStatus.ERROR
                run.end_time = datetime.utcnow()
                logger.error(f"Run {run_id} failed")

            await db.commit()

            # Trigger post-processing tasks
            if success:
                # Chain tasks: VM prediction -> SPC update -> R2R control
                chain(
                    predict_virtual_metrology.s(str(run_id)),
                    update_spc_charts.s(str(run_id)),
                    calculate_r2r_control.s(str(run_id)),
                ).apply_async()

            return {
                "run_id": str(run_id),
                "status": run.status.value,
                "duration_s": run.duration_s,
                "success": success,
            }

        except Exception as e:
            logger.exception(f"Error executing run {run_id}: {e}")

            # Update run to error state
            if run:
                run.status = RunStatus.ERROR
                run.end_time = datetime.utcnow()
                await db.commit()

            raise


# ============================================================================
# Virtual Metrology Tasks
# ============================================================================

@app.task(base=CVDTask, bind=True)
def predict_virtual_metrology(self, run_id: str):
    """
    Predict film properties using virtual metrology models.

    Args:
        run_id: UUID of the run
    """
    logger.info(f"Running virtual metrology prediction for run {run_id}")

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_predict_vm_async(UUID(run_id)))

    return result


async def _predict_vm_async(run_id: UUID) -> Dict[str, Any]:
    """Async VM prediction"""
    async with AsyncSessionLocal() as db:
        try:
            # Get telemetry data
            query = select(CVDTelemetry).where(CVDTelemetry.run_id == run_id).order_by(CVDTelemetry.timestamp)
            result = await db.execute(query)
            telemetry_points = result.scalars().all()

            if not telemetry_points:
                logger.warning(f"No telemetry data for run {run_id}")
                return {"status": "no_data"}

            # Extract features (simplified)
            # TODO: Implement full feature engineering
            avg_temp = sum(
                sum(t.temperatures.values()) / len(t.temperatures) for t in telemetry_points
            ) / len(telemetry_points)

            avg_pressure = sum(
                sum(t.pressures.values()) / len(t.pressures) for t in telemetry_points
            ) / len(telemetry_points)

            # Placeholder prediction (TODO: Load and use actual ML model)
            predicted_thickness = avg_temp * 0.15  # Simplified linear model

            # Update results with VM prediction
            results_query = select(CVDResult).where(CVDResult.run_id == run_id)
            results_result = await db.execute(results_query)
            results = results_result.scalars().all()

            for result_obj in results:
                result_obj.vm_predicted_thickness_nm = predicted_thickness
                result_obj.vm_confidence = 0.85  # Placeholder

            await db.commit()

            logger.info(f"VM prediction complete for run {run_id}: {predicted_thickness:.2f} nm")

            return {
                "run_id": str(run_id),
                "predicted_thickness_nm": predicted_thickness,
                "confidence": 0.85,
            }

        except Exception as e:
            logger.exception(f"Error in VM prediction for run {run_id}: {e}")
            raise


# ============================================================================
# SPC Tasks
# ============================================================================

@app.task(base=CVDTask, bind=True)
def update_spc_charts(self, run_id: str):
    """
    Update SPC control charts with new run data.

    Args:
        run_id: UUID of the run
    """
    logger.info(f"Updating SPC charts for run {run_id}")

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_update_spc_async(UUID(run_id)))

    return result


async def _update_spc_async(run_id: UUID) -> Dict[str, Any]:
    """Async SPC update"""
    async with AsyncSessionLocal() as db:
        try:
            # Get results
            results_query = select(CVDResult).where(CVDResult.run_id == run_id)
            results_result = await db.execute(results_query)
            results = results_result.scalars().all()

            if not results:
                logger.warning(f"No results for run {run_id}")
                return {"status": "no_results"}

            # Get run details
            run_query = select(CVDRun).where(CVDRun.id == run_id)
            run_result = await db.execute(run_query)
            run = run_result.scalar_one_or_none()

            if not run:
                return {"status": "run_not_found"}

            # Find active SPC series for this recipe
            spc_query = select(CVDSPCSeries).where(
                CVDSPCSeries.recipe_id == run.recipe_id,
                CVDSPCSeries.is_active == True,
            )
            spc_result = await db.execute(spc_query)
            spc_series_list = spc_result.scalars().all()

            points_added = 0

            for series in spc_series_list:
                metric = series.metric_name

                # Get metric value from results
                for result_obj in results:
                    value = None

                    if metric == "thickness":
                        value = result_obj.thickness_nm
                    elif metric == "uniformity":
                        value = result_obj.uniformity_pct
                    elif metric == "stress":
                        value = result_obj.stress_mpa

                    if value is None:
                        continue

                    # Check for violations (simplified)
                    out_of_control = False
                    violation_rules = []

                    if value > series.ucl:
                        out_of_control = True
                        violation_rules.append("above_ucl")
                    elif value < series.lcl:
                        out_of_control = True
                        violation_rules.append("below_lcl")

                    if series.usl and value > series.usl:
                        out_of_control = True
                        violation_rules.append("above_usl")
                    elif series.lsl and value < series.lsl:
                        out_of_control = True
                        violation_rules.append("below_lsl")

                    # Create SPC point
                    spc_point = CVDSPCPoint(
                        series_id=series.id,
                        run_id=run_id,
                        timestamp=result_obj.measurement_timestamp,
                        value=value,
                        out_of_control=out_of_control,
                        violation_rules=violation_rules,
                    )

                    db.add(spc_point)
                    points_added += 1

                    if out_of_control:
                        logger.warning(
                            f"SPC violation for run {run_id}, series {series.id}: "
                            f"{metric}={value:.2f}, violations={violation_rules}"
                        )

            await db.commit()

            logger.info(f"SPC update complete for run {run_id}: {points_added} points added")

            return {
                "run_id": str(run_id),
                "points_added": points_added,
            }

        except Exception as e:
            logger.exception(f"Error updating SPC for run {run_id}: {e}")
            raise


# ============================================================================
# R2R Control Tasks
# ============================================================================

@app.task(base=CVDTask, bind=True)
def calculate_r2r_control(self, run_id: str):
    """
    Calculate Run-to-Run control adjustments.

    Args:
        run_id: UUID of the run
    """
    logger.info(f"Calculating R2R control for run {run_id}")

    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(_calculate_r2r_async(UUID(run_id)))

    return result


async def _calculate_r2r_async(run_id: UUID) -> Dict[str, Any]:
    """Async R2R calculation"""
    async with AsyncSessionLocal() as db:
        try:
            # Get run and results
            run_query = select(CVDRun).where(CVDRun.id == run_id)
            run_result = await db.execute(run_query)
            run = run_result.scalar_one_or_none()

            if not run:
                return {"status": "run_not_found"}

            results_query = select(CVDResult).where(CVDResult.run_id == run_id)
            results_result = await db.execute(results_query)
            results = results_result.scalars().all()

            if not results:
                return {"status": "no_results"}

            # Get recipe and target
            recipe_query = select(CVDRecipe).where(CVDRecipe.id == run.recipe_id)
            recipe_result = await db.execute(recipe_query)
            recipe = recipe_result.scalar_one_or_none()

            if not recipe or not recipe.target_thickness_nm:
                return {"status": "no_target"}

            target_thickness = recipe.target_thickness_nm

            # Calculate average measured thickness
            measured_thickness = sum(r.thickness_nm for r in results if r.thickness_nm) / len(results)

            # Calculate error
            error = target_thickness - measured_thickness
            error_pct = (error / target_thickness) * 100

            # Simple EWMA control
            lambda_ewma = 0.3  # Smoothing factor
            gain = 0.8  # Control gain

            # Calculate adjustment
            time_adjustment_s = gain * (error / target_thickness) * recipe.process_time_s

            # Constrain adjustment (max ±10%)
            max_adjustment = recipe.process_time_s * 0.10
            time_adjustment_s = max(-max_adjustment, min(max_adjustment, time_adjustment_s))

            new_process_time = recipe.process_time_s + time_adjustment_s

            logger.info(
                f"R2R control for run {run_id}: "
                f"Error={error:.2f} nm ({error_pct:.1f}%), "
                f"Adjustment={time_adjustment_s:.1f}s, "
                f"New time={new_process_time:.1f}s"
            )

            # TODO: Apply adjustment to recipe or create control action record

            return {
                "run_id": str(run_id),
                "target_thickness_nm": target_thickness,
                "measured_thickness_nm": measured_thickness,
                "error_nm": error,
                "error_pct": error_pct,
                "time_adjustment_s": time_adjustment_s,
                "new_process_time_s": new_process_time,
            }

        except Exception as e:
            logger.exception(f"Error in R2R control for run {run_id}: {e}")
            raise


# ============================================================================
# Data Export Tasks
# ============================================================================

@app.task(base=CVDTask, bind=True)
def export_run_data(self, run_id: str, export_format: str = "csv"):
    """
    Export run data to file.

    Args:
        run_id: UUID of the run
        export_format: Export format (csv, json, parquet)
    """
    logger.info(f"Exporting run {run_id} to {export_format}")

    # TODO: Implement full export logic
    # - Query all data (run, recipe, telemetry, results)
    # - Format as requested
    # - Save to file storage (S3/MinIO)
    # - Return download URL

    return {
        "run_id": run_id,
        "format": export_format,
        "status": "pending_implementation",
    }


# ============================================================================
# Periodic Tasks
# ============================================================================

@app.task(base=CVDTask)
def cleanup_old_telemetry():
    """Periodic task to clean up old telemetry data"""
    logger.info("Running telemetry cleanup task")

    # TODO: Implement cleanup
    # - Delete telemetry older than retention period
    # - Archive to cold storage if needed

    return {"status": "pending_implementation"}


@app.task(base=CVDTask)
def recalculate_spc_limits():
    """Periodic task to recalculate SPC control limits"""
    logger.info("Recalculating SPC control limits")

    # TODO: Implement SPC limit recalculation
    # - For each active SPC series
    # - Get last N points
    # - Recalculate UCL, LCL based on data
    # - Update series

    return {"status": "pending_implementation"}


@app.task(base=CVDTask)
def retrain_vm_models():
    """Periodic task to retrain VM models"""
    logger.info("Retraining VM models")

    # TODO: Implement model retraining
    # - Collect recent training data
    # - Train/update ML models
    # - Validate performance
    # - Deploy if improved

    return {"status": "pending_implementation"}


# ============================================================================
# Celery Beat Schedule (Periodic Tasks)
# ============================================================================

app.conf.beat_schedule = {
    "cleanup-old-telemetry": {
        "task": "cvd_tasks.cleanup_old_telemetry",
        "schedule": crontab(hour=2, minute=0),  # Daily at 2 AM
    },
    "recalculate-spc-limits": {
        "task": "cvd_tasks.recalculate_spc_limits",
        "schedule": crontab(hour=1, minute=0, day_of_week=0),  # Weekly on Sunday at 1 AM
    },
    "retrain-vm-models": {
        "task": "cvd_tasks.retrain_vm_models",
        "schedule": crontab(hour=3, minute=0, day_of_week=0),  # Weekly on Sunday at 3 AM
    },
}


# ============================================================================
# Task Utilities
# ============================================================================

def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get status of a Celery task"""
    result = app.AsyncResult(task_id)

    return {
        "task_id": task_id,
        "state": result.state,
        "info": result.info,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
    }


def cancel_task(task_id: str) -> bool:
    """Cancel a running task"""
    result = app.AsyncResult(task_id)
    result.revoke(terminate=True)
    logger.info(f"Task {task_id} cancelled")
    return True
