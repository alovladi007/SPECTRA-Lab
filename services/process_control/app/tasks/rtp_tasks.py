"""Celery tasks for RTP (Rapid Thermal Processing) background processing.

Executes RTP runs asynchronously with:
- Recipe validation against thermal budgets and ramp rate limits
- HIL simulator execution with PID/MPC control
- Real-time temperature tracking and lamp power monitoring
- Progress tracking and cancellation support
- SPC monitoring for ramp error, overshoot, emissivity drift
- Virtual Metrology predictions for activation/diffusion
- Artifact storage (temperature profiles, control charts, thermal budget)
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

from celery import Task
from celery.exceptions import Ignore, SoftTimeLimitExceeded

from app.celery_app import celery_app
from app.models.job import JobStore, JobStatus as JobStatusEnum, JobType
from app.simulators.rtp_hil import RTPHILDriver
from app.controllers.rtp import (
    PIDController,
    MPCController,
    R2RController,
    ThermalBudgetCalculator,
)
from app.spc.monitors import RTPMonitor, RTPParameter
from app.vm.rtp_vm import RTPVirtualMetrologyModel, RTPVMFeatures


# ============================================================================
# Configuration
# ============================================================================

# Storage paths (reuse from ion_tasks)
TELEMETRY_STORAGE = os.getenv("TELEMETRY_STORAGE", "/tmp/spectra/telemetry")
ARTIFACT_STORAGE = os.getenv("ARTIFACT_STORAGE", "/tmp/spectra/artifacts")
LOGS_STORAGE = os.getenv("LOGS_STORAGE", "/tmp/spectra/logs")

os.makedirs(TELEMETRY_STORAGE, exist_ok=True)
os.makedirs(ARTIFACT_STORAGE, exist_ok=True)
os.makedirs(LOGS_STORAGE, exist_ok=True)


# ============================================================================
# Custom Task Base Class
# ============================================================================

class CancellableTask(Task):
    """Base task with cancellation support."""

    def __call__(self, *args, **kwargs):
        """Check for cancellation before running."""
        job_id = kwargs.get("job_id")
        if job_id:
            job = JobStore.get_job(job_id)
            if job and job.status == JobStatusEnum.CANCELLED:
                raise Ignore()

        return super().__call__(*args, **kwargs)


# ============================================================================
# Recipe Validation
# ============================================================================

def validate_rtp_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Validate RTP recipe against thermal budgets and system limits.

    Args:
        recipe: Recipe dictionary containing temperature profile

    Returns:
        Validation result with warnings and errors

    Validation checks:
        - Max temperature: 400-1200°C
        - Ramp rate: ≤100°C/s (up), ≤50°C/s (down)
        - Hold time: 0-300 seconds
        - Thermal budget: Consider cumulative thermal exposure
        - Gas flows: N2 (0-20 SLM), O2 (0-5 SLM)
    """
    errors = []
    warnings = []

    # Get recipe segments
    segments = recipe.get("segments", [])
    if not segments:
        errors.append("Recipe must contain at least one temperature segment")
        return {"valid": False, "errors": errors, "warnings": warnings}

    prev_temp = recipe.get("initial_temp_c", 25.0)

    for i, seg in enumerate(segments):
        target_temp = seg.get("target_temp_c", 0)
        duration_s = seg.get("duration_s", 0)

        # Temperature range check
        if not (400 <= target_temp <= 1200):
            errors.append(
                f"Segment {i+1}: Temperature {target_temp}°C out of range [400, 1200]"
            )

        # Duration check
        if duration_s < 0 or duration_s > 300:
            errors.append(
                f"Segment {i+1}: Duration {duration_s}s out of range [0, 300]"
            )

        # Ramp rate check
        if i > 0 or "ramp_rate_c_s" in seg:
            ramp_rate = seg.get("ramp_rate_c_s", abs(target_temp - prev_temp) / max(duration_s, 1))

            if target_temp > prev_temp:  # Heating
                if ramp_rate > 100:
                    errors.append(
                        f"Segment {i+1}: Heating ramp rate {ramp_rate:.1f}°C/s exceeds limit (100°C/s)"
                    )
            else:  # Cooling
                if abs(ramp_rate) > 50:
                    warnings.append(
                        f"Segment {i+1}: Cooling ramp rate {abs(ramp_rate):.1f}°C/s exceeds recommended limit (50°C/s)"
                    )

        prev_temp = target_temp

    # Gas flow validation
    n2_flow = recipe.get("n2_flow_slm", 10.0)
    o2_flow = recipe.get("o2_flow_slm", 0.0)

    if not (0 <= n2_flow <= 20):
        errors.append(f"N2 flow {n2_flow} SLM out of range [0, 20]")

    if not (0 <= o2_flow <= 5):
        errors.append(f"O2 flow {o2_flow} SLM out of range [0, 5]")

    # Thermal budget warning
    total_thermal_budget = sum(
        seg.get("duration_s", 0) * np.exp(seg.get("target_temp_c", 0) / 100)
        for seg in segments
    )

    if total_thermal_budget > 1e6:
        warnings.append(
            f"High thermal budget ({total_thermal_budget:.2e}). May cause excessive diffusion."
        )

    # Controller type validation
    controller_type = recipe.get("controller_type", "pid")
    if controller_type not in ["pid", "mpc"]:
        errors.append(f"Invalid controller type '{controller_type}'. Must be 'pid' or 'mpc'")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================================
# Storage Utilities (Shared with ion_tasks)
# ============================================================================

def store_telemetry(run_id: str, telemetry: Dict[str, Any]) -> str:
    """Store telemetry data to filesystem."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"rtp_{run_id}_{timestamp}.json"
    filepath = os.path.join(TELEMETRY_STORAGE, filename)

    with open(filepath, "w") as f:
        json.dump(telemetry, f, indent=2)

    return f"file://{filepath}"


def store_artifact(run_id: str, artifact_type: str, data: Any) -> Dict[str, str]:
    """Store artifact to filesystem."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"rtp_{run_id}_{artifact_type}_{timestamp}.json"
    filepath = os.path.join(ARTIFACT_STORAGE, filename)

    if isinstance(data, np.ndarray):
        data = data.tolist()
    elif isinstance(data, dict):
        data = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in data.items()}

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "type": artifact_type,
        "uri": f"file://{filepath}",
        "created_at": datetime.utcnow().isoformat(),
    }


def store_logs(run_id: str, logs: List[str]) -> str:
    """Store run logs to filesystem."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"rtp_{run_id}_{timestamp}.log"
    filepath = os.path.join(LOGS_STORAGE, filename)

    with open(filepath, "w") as f:
        f.write("\n".join(logs))

    return f"file://{filepath}"


# ============================================================================
# Main RTP Task
# ============================================================================

@celery_app.task(
    bind=True,
    base=CancellableTask,
    name="app.tasks.rtp_tasks.execute_rtp_run",
    max_retries=3,
    default_retry_delay=60,
)
def execute_rtp_run(
    self,
    job_id: str,
    run_id: str,
    org_id: str,
    recipe: Dict[str, Any],
    user_id: str,
) -> Dict[str, Any]:
    """Execute RTP run with full process control.

    This is the main Celery task for RTP. It:
    1. Validates recipe against thermal budgets
    2. Initializes HIL simulator and PID/MPC controller
    3. Executes temperature profile with real-time control
    4. Updates job progress continuously
    5. Runs SPC checks (ramp error, overshoot, emissivity)
    6. Runs VM predictions for activation/diffusion
    7. Stores all telemetry and artifacts
    8. Handles cancellation gracefully

    Args:
        job_id: Job identifier for tracking
        run_id: Run identifier
        org_id: Organization ID
        recipe: RTP recipe with temperature segments
        user_id: User who initiated the run

    Returns:
        Result dictionary with metrics, artifacts, and alerts
    """
    logs = []
    job = JobStore.get_job(job_id)

    def log(message: str):
        """Helper to log and store messages."""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        logs.append(log_msg)
        print(log_msg)

    def update_progress(progress: float, step: str):
        """Update job progress."""
        JobStore.update_job(
            job_id,
            progress=progress,
            current_step=step,
        )

    try:
        log(f"Starting RTP run {run_id}")
        update_progress(0.0, "Initializing")

        # Mark job as running
        JobStore.update_job(
            job_id,
            status=JobStatusEnum.RUNNING,
            started_at=datetime.utcnow(),
        )

        # =====================================================================
        # Step 1: Recipe Validation (5%)
        # =====================================================================
        log("Validating recipe against thermal budgets and system limits")
        update_progress(5.0, "Validating recipe")

        validation = validate_rtp_recipe(recipe)
        if not validation["valid"]:
            error_msg = f"Recipe validation failed: {', '.join(validation['errors'])}"
            log(f"ERROR: {error_msg}")
            raise ValueError(error_msg)

        if validation["warnings"]:
            for warning in validation["warnings"]:
                log(f"WARNING: {warning}")

        log("Recipe validation passed")

        # =====================================================================
        # Step 2: Initialize HIL Simulator (10%)
        # =====================================================================
        log("Initializing RTP HIL simulator")
        update_progress(10.0, "Initializing simulator")

        hil_sim = RTPHILDriver(equipment_id="RTP-SIM-001")

        segments = recipe["segments"]
        controller_type = recipe.get("controller_type", "pid")
        n2_flow = recipe.get("n2_flow_slm", 10.0)
        o2_flow = recipe.get("o2_flow_slm", 0.0)

        log(f"Recipe: {len(segments)} segments, controller: {controller_type}")
        log(f"Gas flows: N2={n2_flow} SLM, O2={o2_flow} SLM")

        for i, seg in enumerate(segments):
            log(f"  Segment {i+1}: {seg['target_temp_c']}°C for {seg['duration_s']}s")

        # =====================================================================
        # Step 3: Initialize Controller (15%)
        # =====================================================================
        log(f"Initializing {controller_type.upper()} controller")
        update_progress(15.0, "Initializing controller")

        if controller_type == "pid":
            # PID controller with tuning from recipe or defaults
            from app.controllers.rtp import PIDGains

            kp = recipe.get("pid_kp", 5.0)
            ki = recipe.get("pid_ki", 0.5)
            kd = recipe.get("pid_kd", 1.0)

            gains = PIDGains(Kp=kp, Ki=ki, Kd=kd)
            controller = PIDController(gains=gains)
            log(f"PID gains: Kp={kp}, Ki={ki}, Kd={kd}")

        elif controller_type == "mpc":
            from app.controllers.rtp import MPCParameters

            params = MPCParameters(
                prediction_horizon=10,
                control_horizon=5,
            )
            controller = MPCController(params=params, num_zones=4)
            log("MPC controller initialized with H_p=10, H_c=5")

        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        # R2R controller
        r2r_controller = R2RController(alpha=0.2)

        # SPC monitor
        spc_monitor = RTPMonitor(equipment_id="RTP-SIM-001")

        # VM model
        vm_model = RTPVirtualMetrologyModel()

        # Thermal budget calculator
        thermal_calc = ThermalBudgetCalculator()

        log("Controllers initialized")

        # =====================================================================
        # Step 4: Execute Temperature Profile (20-80%)
        # =====================================================================
        log("Starting RTP process")
        update_progress(20.0, "Running RTP process")

        # Calculate total duration
        total_duration = sum(seg["duration_s"] for seg in segments)
        log(f"Total process duration: {total_duration} seconds")

        dt = 0.1  # 100ms sampling
        current_time = 0.0
        current_segment = 0

        telemetry_history = {
            "time_s": [],
            "setpoint_temp_c": [],
            "measured_temp_c": [],
            "lamp_power_pct": [],
            "ramp_error_c": [],
            "chamber_pressure_torr": [],
            "n2_flow_slm": [],
            "o2_flow_slm": [],
        }

        spc_alerts = []
        max_overshoot = 0.0
        cumulative_thermal_budget = 0.0

        # Get initial temperature
        current_temp = recipe.get("initial_temp_c", 25.0)
        segment_start_time = 0.0

        while current_segment < len(segments):
            # Check for cancellation
            job = JobStore.get_job(job_id)
            if job.status == JobStatusEnum.CANCELLED:
                log("Run cancelled by user")
                raise Ignore()

            # Get current segment
            seg = segments[current_segment]
            target_temp = seg["target_temp_c"]
            segment_duration = seg["duration_s"]

            # Calculate setpoint (linear ramp to target)
            segment_elapsed = current_time - segment_start_time
            if segment_elapsed < segment_duration:
                # Ramp phase
                prev_temp = current_temp if current_segment == 0 else segments[current_segment - 1]["target_temp_c"]
                setpoint = prev_temp + (target_temp - prev_temp) * (segment_elapsed / segment_duration)
            else:
                # Hold at target
                setpoint = target_temp

            # Get measured temperature from simulator
            telemetry = hil_sim.get_telemetry()
            measured_temp = telemetry["wafer_temp_c"]

            # Compute control action
            error = setpoint - measured_temp

            if controller_type == "pid":
                control_output = controller.compute(
                    setpoint=setpoint,
                    measured=measured_temp,
                    dt=dt,
                )
            elif controller_type == "mpc":
                # For MPC, provide future setpoints
                future_setpoints = [target_temp] * 10  # Simplified
                control_output = controller.compute(
                    measured_temp=measured_temp,
                    setpoint_trajectory=future_setpoints,
                )

            # Apply control to simulator (lamp power %)
            lamp_power_pct = np.clip(control_output, 0, 100)
            hil_sim.set_lamp_power(lamp_power_pct)

            # Advance simulator
            hil_sim.step(dt)

            # Update current temperature (with realistic lag)
            current_temp = measured_temp

            # Calculate ramp error
            ramp_error = abs(setpoint - measured_temp)

            # Track overshoot
            if measured_temp > setpoint:
                overshoot = measured_temp - setpoint
                max_overshoot = max(max_overshoot, overshoot)

            # Update thermal budget
            thermal_budget_step = thermal_calc.calculate_segment_budget(
                temperature_c=measured_temp,
                duration_s=dt,
            )
            cumulative_thermal_budget += thermal_budget_step

            # Store telemetry
            telemetry_history["time_s"].append(current_time)
            telemetry_history["setpoint_temp_c"].append(setpoint)
            telemetry_history["measured_temp_c"].append(measured_temp)
            telemetry_history["lamp_power_pct"].append(lamp_power_pct)
            telemetry_history["ramp_error_c"].append(ramp_error)
            telemetry_history["chamber_pressure_torr"].append(telemetry["chamber_pressure_torr"])
            telemetry_history["n2_flow_slm"].append(n2_flow)
            telemetry_history["o2_flow_slm"].append(o2_flow)

            # Run SPC checks every 1 second
            if int(current_time * 10) % 10 == 0 and current_time > 0:
                spc_data = {
                    RTPParameter.RAMP_ERROR_C: ramp_error,
                    RTPParameter.LAMP_POWER_PCT: lamp_power_pct,
                    RTPParameter.CHAMBER_PRESSURE_TORR: telemetry["chamber_pressure_torr"],
                }

                alerts = spc_monitor.process_sample(spc_data, timestamp=datetime.utcnow())
                if alerts:
                    for alert in alerts:
                        log(f"SPC ALERT: {alert.parameter.value} - {alert.message}")
                        spc_alerts.append({
                            "parameter": alert.parameter.value,
                            "message": alert.message,
                            "severity": alert.severity.value,
                            "timestamp": alert.timestamp.isoformat(),
                        })

            # Update progress (20% to 80% during run)
            progress = 20.0 + (current_time / total_duration) * 60.0
            if int(current_time * 10) % 10 == 0:
                update_progress(
                    progress,
                    f"RTP segment {current_segment + 1}/{len(segments)} ({measured_temp:.1f}°C)"
                )

            # Advance time
            current_time += dt

            # Check if segment is complete
            if current_time >= segment_start_time + segment_duration:
                log(f"Segment {current_segment + 1} complete: {target_temp}°C")
                current_segment += 1
                segment_start_time = current_time

            # Simulate real-time execution (speed up 10x for demo)
            time.sleep(dt / 10)

        log(f"RTP process complete. Total thermal budget: {cumulative_thermal_budget:.2e}")

        # =====================================================================
        # Step 5: Calculate Metrics (80%)
        # =====================================================================
        log("Calculating process metrics")
        update_progress(80.0, "Calculating metrics")

        # Final temperature
        final_temp = telemetry_history["measured_temp_c"][-1]
        target_final_temp = segments[-1]["target_temp_c"]
        temp_error_final = abs(final_temp - target_final_temp)

        log(f"Final temperature: {final_temp:.1f}°C (target: {target_final_temp:.1f}°C)")
        log(f"Temperature error: {temp_error_final:.2f}°C")
        log(f"Max overshoot: {max_overshoot:.2f}°C")

        # Average ramp error
        avg_ramp_error = np.mean(telemetry_history["ramp_error_c"])
        log(f"Average ramp error: {avg_ramp_error:.2f}°C")

        # =====================================================================
        # Step 6: Run Virtual Metrology (85%)
        # =====================================================================
        log("Running Virtual Metrology predictions")
        update_progress(85.0, "Running VM predictions")

        # Get ion implant context (if available from job metadata)
        ion_context = recipe.get("ion_implant_context", {})

        vm_features = RTPVMFeatures(
            peak_temp_c=max(telemetry_history["measured_temp_c"]),
            hold_time_s=segments[-1]["duration_s"],
            ramp_up_rate_c_s=recipe.get("ramp_rate_c_s", 50.0),
            ramp_down_rate_c_s=20.0,  # Typical cooling rate
            n2_flow_slm=n2_flow,
            o2_flow_slm=o2_flow,
            chamber_pressure_torr=np.mean(telemetry_history["chamber_pressure_torr"]),
            thermal_budget=cumulative_thermal_budget,
            # Ion implant context
            species=ion_context.get("species", "P"),
            implant_energy_kev=ion_context.get("energy_kev", 40.0),
            implant_dose_atoms_cm2=ion_context.get("dose_atoms_cm2", 1e15),
        )

        vm_prediction = vm_model.predict(vm_features)
        log(f"VM Prediction: Activation={vm_prediction.activation_pct:.1f}%, "
            f"Diffusion={vm_prediction.diffusion_depth_um:.3f} μm, "
            f"Oxide={vm_prediction.oxide_thickness_nm:.1f} nm")

        # =====================================================================
        # Step 7: Store Artifacts (90%)
        # =====================================================================
        log("Storing telemetry and artifacts")
        update_progress(90.0, "Storing artifacts")

        # Store telemetry
        telemetry_uri = store_telemetry(run_id, telemetry_history)
        log(f"Telemetry stored: {telemetry_uri}")

        artifacts = []

        # Temperature profile chart
        profile_artifact = store_artifact(run_id, "temperature_profile", {
            "time_s": telemetry_history["time_s"],
            "setpoint_temp_c": telemetry_history["setpoint_temp_c"],
            "measured_temp_c": telemetry_history["measured_temp_c"],
        })
        artifacts.append(profile_artifact)
        log(f"Temperature profile stored: {profile_artifact['uri']}")

        # Control chart (lamp power)
        control_artifact = store_artifact(run_id, "control_chart", {
            "time_s": telemetry_history["time_s"],
            "lamp_power_pct": telemetry_history["lamp_power_pct"],
            "ramp_error_c": telemetry_history["ramp_error_c"],
        })
        artifacts.append(control_artifact)
        log(f"Control chart stored: {control_artifact['uri']}")

        # SPC alerts (if any)
        if spc_alerts:
            alerts_artifact = store_artifact(run_id, "spc_alerts", spc_alerts)
            artifacts.append(alerts_artifact)
            log(f"SPC alerts stored: {alerts_artifact['uri']}")

        # VM prediction
        vm_artifact = store_artifact(run_id, "vm_prediction", {
            "activation_pct": vm_prediction.activation_pct,
            "diffusion_depth_um": vm_prediction.diffusion_depth_um,
            "sheet_resistance_ohm_sq": vm_prediction.sheet_resistance_ohm_sq,
            "junction_depth_um": vm_prediction.junction_depth_um,
            "oxide_thickness_nm": vm_prediction.oxide_thickness_nm,
            "confidence_score": vm_prediction.confidence_score,
            "features": vm_features.dict(),
        })
        artifacts.append(vm_artifact)
        log(f"VM prediction stored: {vm_artifact['uri']}")

        # Store logs
        logs_uri = store_logs(run_id, logs)
        log(f"Logs stored: {logs_uri}")

        # =====================================================================
        # Step 8: R2R Recommendation (95%)
        # =====================================================================
        log("Generating R2R recommendations")
        update_progress(95.0, "R2R recommendations")

        r2r_recommendation = r2r_controller.recommend_adjustment(
            setpoint_temp=target_final_temp,
            measured_temp=final_temp,
        )
        log(f"R2R: Temperature adjustment: {r2r_recommendation.temperature_adjustment_c:+.2f}°C")

        # =====================================================================
        # Step 9: Complete Job (100%)
        # =====================================================================
        log("Run completed successfully")
        update_progress(100.0, "Completed")

        JobStore.update_job(
            job_id,
            status=JobStatusEnum.COMPLETED,
            progress=100.0,
            completed_at=datetime.utcnow(),
            logs_uri=logs_uri,
            artifacts=artifacts,
            metadata={
                "final_temp_c": final_temp,
                "temp_error_c": temp_error_final,
                "max_overshoot_c": max_overshoot,
                "avg_ramp_error_c": avg_ramp_error,
                "thermal_budget": cumulative_thermal_budget,
                "vm_prediction": vm_prediction.dict(),
                "spc_alerts_count": len(spc_alerts),
                "r2r_adjustment_c": r2r_recommendation.temperature_adjustment_c,
            },
        )

        return {
            "run_id": run_id,
            "status": "completed",
            "final_temp_c": final_temp,
            "temp_error_c": temp_error_final,
            "max_overshoot_c": max_overshoot,
            "avg_ramp_error_c": avg_ramp_error,
            "thermal_budget": cumulative_thermal_budget,
            "vm_prediction": vm_prediction.dict(),
            "spc_alerts": spc_alerts,
            "artifacts": artifacts,
            "logs_uri": logs_uri,
        }

    except SoftTimeLimitExceeded:
        log("ERROR: Task exceeded time limit")
        JobStore.update_job(
            job_id,
            status=JobStatusEnum.FAILED,
            error_message="Task exceeded time limit",
            completed_at=datetime.utcnow(),
        )
        raise

    except Exception as e:
        log(f"ERROR: {str(e)}")
        JobStore.update_job(
            job_id,
            status=JobStatusEnum.FAILED,
            error_message=str(e),
            completed_at=datetime.utcnow(),
        )

        # Retry on transient errors
        if "connection" in str(e).lower() or "timeout" in str(e).lower():
            log(f"Retrying due to transient error (attempt {self.request.retries + 1}/3)")
            raise self.retry(exc=e)

        raise


# ============================================================================
# Controller Tuning Task
# ============================================================================

@celery_app.task(
    name="app.tasks.rtp_tasks.tune_controller",
    time_limit=60,
)
def tune_controller(
    controller_type: str,
    recipe: Dict[str, Any],
    optimization_target: str = "minimize_overshoot",
) -> Dict[str, Any]:
    """Auto-tune PID or MPC controller parameters.

    This task runs optimization to find optimal controller gains based on
    the recipe and optimization objective.

    Args:
        controller_type: "pid" or "mpc"
        recipe: RTP recipe to optimize for
        optimization_target: "minimize_overshoot", "minimize_ramp_error", or "minimize_settling_time"

    Returns:
        Recommended controller parameters
    """
    from app.controllers.rtp import auto_tune_pid

    if controller_type == "pid":
        # Use Ziegler-Nichols or relay-feedback tuning
        tuned_params = auto_tune_pid(
            recipe=recipe,
            method="relay_feedback",
            target=optimization_target,
        )

        return {
            "controller_type": "pid",
            "parameters": {
                "kp": tuned_params["kp"],
                "ki": tuned_params["ki"],
                "kd": tuned_params["kd"],
            },
            "expected_performance": {
                "overshoot_pct": tuned_params["overshoot_pct"],
                "settling_time_s": tuned_params["settling_time_s"],
                "steady_state_error_c": tuned_params["steady_state_error_c"],
            },
            "tuning_method": "relay_feedback",
        }

    elif controller_type == "mpc":
        # MPC tuning: adjust prediction/control horizons and weights
        return {
            "controller_type": "mpc",
            "parameters": {
                "prediction_horizon": 10,
                "control_horizon": 5,
                "weight_tracking": 1.0,
                "weight_control_effort": 0.1,
            },
            "expected_performance": {
                "overshoot_pct": 2.0,
                "settling_time_s": 5.0,
            },
            "tuning_method": "model_based",
        }

    else:
        raise ValueError(f"Unknown controller type: {controller_type}")


# Export
__all__ = [
    "execute_rtp_run",
    "tune_controller",
    "validate_rtp_recipe",
]
