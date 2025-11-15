"""
CVD Run Orchestration Tasks

Celery tasks for orchestrating CVD runs (real or simulated).
Provides real-time progress updates and result storage.
"""

import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from celery import Task, chain, group, chord
from celery.exceptions import SoftTimeLimitExceeded

from .celery_app import celery_app
from ..realtime.events import emit_run_event, RunEventType
from ..drivers.hil_simulator import HILCVDSimulator, SimulationParameters
from ..drivers.physics_models import (
    ThicknessModel,
    StressModel,
    AdhesionModel,
    DepositionParameters,
    ProcessConditions,
    AdhesionFactors,
)

logger = logging.getLogger(__name__)


class CVDRunTask(Task):
    """Base task class for CVD runs with progress tracking"""

    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds"""
        logger.info(f"CVD run task {task_id} completed successfully")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails"""
        logger.error(f"CVD run task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried"""
        logger.warning(f"CVD run task {task_id} retrying: {exc}")


@celery_app.task(
    base=CVDRunTask,
    bind=True,
    name="app.tasks.cvd_orchestration.run_cvd_simulation",
    soft_time_limit=3600,  # 1 hour
    time_limit=3900,  # 1 hour 5 minutes
)
def run_cvd_simulation(
    self,
    run_id: str,
    recipe_params: Dict[str, Any],
    simulation_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a CVD simulation (HIL) with real-time progress updates

    Args:
        run_id: Unique run identifier
        recipe_params: Recipe parameters
        simulation_config: Optional simulation configuration

    Returns:
        Dictionary with simulation results
    """
    logger.info(f"Starting CVD simulation for run_id={run_id}")

    try:
        # Emit start event
        emit_run_event(
            run_id=run_id,
            event_type=RunEventType.RUN_STARTED,
            data={"recipe": recipe_params},
        )

        # Create simulator
        sim_params = SimulationParameters(**simulation_config) if simulation_config else SimulationParameters()

        simulator = HILCVDSimulator(
            mode=recipe_params.get("mode", "thermal"),
            enable_noise=sim_params.enable_noise,
            enable_drift=sim_params.enable_drift,
            enable_faults=sim_params.enable_faults,
        )

        # Configure recipe
        dep_params = DepositionParameters(
            temperature_c=recipe_params.get("temperature_c", 800.0),
            pressure_torr=recipe_params.get("pressure_torr", 0.5),
            precursor_flow_sccm=recipe_params.get("precursor_flow_sccm", 80.0),
            carrier_gas_flow_sccm=recipe_params.get("carrier_gas_flow_sccm", 500.0),
            film_material=recipe_params.get("film_material", "Si3N4"),
            target_thickness_nm=recipe_params.get("target_thickness_nm", 100.0),
        )

        # Calculate total time
        total_time = recipe_params.get("duration_sec", 3600.0)
        update_interval = 10.0  # Update every 10 seconds

        # Run simulation with progress updates
        elapsed = 0.0
        results_history = []

        while elapsed < total_time:
            # Update progress
            progress = (elapsed / total_time) * 100.0

            # Run simulation step
            step_time = min(update_interval, total_time - elapsed)
            result = simulator.run_deposition(
                params=dep_params,
                time_sec=step_time,
            )

            # Calculate current metrics
            current_thickness = result["thickness_nm"]
            current_rate = result["deposition_rate_nm_min"]

            # Emit progress event with metrics
            emit_run_event(
                run_id=run_id,
                event_type=RunEventType.PROGRESS_UPDATE,
                data={
                    "progress": progress,
                    "elapsed_sec": elapsed,
                    "total_sec": total_time,
                    "current_thickness_nm": current_thickness,
                    "deposition_rate_nm_min": current_rate,
                    "timestamp": datetime.now().isoformat(),
                },
            )

            # Check for risk indicators
            check_risk_indicators(
                run_id=run_id,
                thickness_nm=current_thickness,
                rate_nm_min=current_rate,
                params=dep_params,
            )

            results_history.append({
                "time_sec": elapsed,
                "thickness_nm": current_thickness,
                "rate_nm_min": current_rate,
            })

            # Update task state
            self.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "current_thickness": current_thickness,
                    "elapsed": elapsed,
                }
            )

            elapsed += step_time
            time.sleep(0.1)  # Small delay for realism

        # Final results
        final_result = simulator.run_deposition(params=dep_params, time_sec=0.0)

        # Calculate final metrics
        final_metrics = calculate_final_metrics(
            results=final_result,
            params=dep_params,
            history=results_history,
        )

        # Emit completion event
        emit_run_event(
            run_id=run_id,
            event_type=RunEventType.RUN_COMPLETED,
            data={
                "final_metrics": final_metrics,
                "duration_sec": total_time,
            },
        )

        logger.info(f"CVD simulation completed for run_id={run_id}")

        return {
            "run_id": run_id,
            "status": "completed",
            "metrics": final_metrics,
            "history": results_history[-10:],  # Last 10 points
        }

    except SoftTimeLimitExceeded:
        logger.error(f"CVD simulation timed out for run_id={run_id}")
        emit_run_event(
            run_id=run_id,
            event_type=RunEventType.RUN_FAILED,
            data={"error": "Simulation timeout"},
        )
        raise

    except Exception as e:
        logger.error(f"CVD simulation failed for run_id={run_id}: {e}")
        emit_run_event(
            run_id=run_id,
            event_type=RunEventType.RUN_FAILED,
            data={"error": str(e)},
        )
        raise


def check_risk_indicators(
    run_id: str,
    thickness_nm: float,
    rate_nm_min: float,
    params: DepositionParameters,
):
    """
    Check for risk indicators during deposition

    Emits warning events if risks are detected:
    - Stress risk (high predicted stress)
    - Adhesion risk (poor conditions)
    - Rate anomaly (unexpected deposition rate)
    """
    warnings = []

    # Stress risk check
    stress_model = StressModel()
    process_cond = ProcessConditions(
        temperature_c=params.temperature_c,
        pressure_torr=params.pressure_torr,
        deposition_rate_nm_min=rate_nm_min,
        film_thickness_nm=thickness_nm,
    )

    stress_result = stress_model.calculate_total_stress(process_cond)
    stress_mpa = stress_result["stress_mean_mpa"]

    # High compressive stress risk
    if stress_mpa < -400.0:
        warnings.append({
            "type": "high_compressive_stress",
            "severity": "WARNING",
            "message": f"High compressive stress detected: {stress_mpa:.1f} MPa",
            "value": stress_mpa,
        })

    # High tensile stress risk
    elif stress_mpa > 300.0:
        warnings.append({
            "type": "high_tensile_stress",
            "severity": "WARNING",
            "message": f"High tensile stress detected: {stress_mpa:.1f} MPa",
            "value": stress_mpa,
        })

    # Adhesion risk check (simplified - would use actual pre-clean data)
    if abs(stress_mpa) > 500.0:
        warnings.append({
            "type": "adhesion_risk",
            "severity": "WARNING",
            "message": f"Adhesion risk due to high stress: {stress_mpa:.1f} MPa",
            "recommendation": "Consider stress relief anneal or reduce deposition rate",
        })

    # Rate anomaly check
    expected_rate_min = 40.0
    expected_rate_max = 60.0

    if rate_nm_min < expected_rate_min:
        warnings.append({
            "type": "low_deposition_rate",
            "severity": "INFO",
            "message": f"Low deposition rate: {rate_nm_min:.1f} nm/min",
            "expected_range": f"{expected_rate_min}-{expected_rate_max}",
        })

    elif rate_nm_min > expected_rate_max:
        warnings.append({
            "type": "high_deposition_rate",
            "severity": "INFO",
            "message": f"High deposition rate: {rate_nm_min:.1f} nm/min",
            "expected_range": f"{expected_rate_min}-{expected_rate_max}",
        })

    # Emit warnings
    if warnings:
        emit_run_event(
            run_id=run_id,
            event_type=RunEventType.WARNING,
            data={"warnings": warnings},
        )


def calculate_final_metrics(
    results: Dict[str, Any],
    params: DepositionParameters,
    history: list,
) -> Dict[str, Any]:
    """Calculate comprehensive final metrics"""

    # Get final thickness
    final_thickness = results["thickness_nm"]
    final_rate = results["deposition_rate_nm_min"]

    # Calculate stress
    stress_model = StressModel()
    process_cond = ProcessConditions(
        temperature_c=params.temperature_c,
        pressure_torr=params.pressure_torr,
        deposition_rate_nm_min=final_rate,
        film_thickness_nm=final_thickness,
    )

    stress_result = stress_model.calculate_total_stress(process_cond)

    # Calculate adhesion
    adhesion_model = AdhesionModel()
    adhesion_factors = AdhesionFactors(
        film_stress_mpa=stress_result["stress_mean_mpa"],
        pre_clean_quality=0.95,  # Assume good pre-clean
        deposition_temp_c=params.temperature_c,
        film_thickness_nm=final_thickness,
    )

    adhesion_score, adhesion_class = adhesion_model.calculate_adhesion_score(adhesion_factors)

    return {
        "thickness_mean_nm": final_thickness,
        "thickness_uniformity_pct": results.get("wiw_uniformity_pct", 2.5),
        "deposition_rate_nm_min": final_rate,
        "stress_mean_mpa": stress_result["stress_mean_mpa"],
        "stress_type": stress_result["stress_type"].value,
        "intrinsic_stress_mpa": stress_result["intrinsic_stress_mpa"],
        "thermal_stress_mpa": stress_result["thermal_stress_mpa"],
        "adhesion_score": adhesion_score,
        "adhesion_class": adhesion_class.value,
        "film_material": params.film_material,
        "process_timestamp": datetime.now().isoformat(),
    }


@celery_app.task(name="app.tasks.cvd_orchestration.run_multi_step_recipe")
def run_multi_step_recipe(
    run_id: str,
    recipe_steps: list,
) -> Dict[str, Any]:
    """
    Run a multi-step recipe (chain of depositions)

    Args:
        run_id: Unique run identifier
        recipe_steps: List of recipe step configurations

    Returns:
        Combined results from all steps
    """
    logger.info(f"Starting multi-step recipe for run_id={run_id}, {len(recipe_steps)} steps")

    # Create task chain
    task_chain = chain(
        [
            run_cvd_simulation.s(
                run_id=f"{run_id}_step{i}",
                recipe_params=step,
            )
            for i, step in enumerate(recipe_steps)
        ]
    )

    # Execute chain
    result = task_chain.apply_async()

    return {
        "run_id": run_id,
        "status": "submitted",
        "task_id": result.id,
        "num_steps": len(recipe_steps),
    }


@celery_app.task(name="app.tasks.cvd_orchestration.cancel_cvd_run")
def cancel_cvd_run(run_id: str) -> Dict[str, Any]:
    """
    Cancel a running CVD simulation

    Args:
        run_id: Run identifier to cancel

    Returns:
        Cancellation status
    """
    logger.info(f"Cancelling CVD run: {run_id}")

    # Emit cancellation event
    emit_run_event(
        run_id=run_id,
        event_type=RunEventType.RUN_CANCELLED,
        data={"timestamp": datetime.now().isoformat()},
    )

    return {
        "run_id": run_id,
        "status": "cancelled",
    }
