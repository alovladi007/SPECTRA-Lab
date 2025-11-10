"""Celery tasks for Ion Implantation background processing.

Executes ion implantation runs asynchronously with:
- Recipe validation against SOP + calibration
- HIL simulator execution with real-time telemetry
- Progress tracking and cancellation support
- SPC monitoring and alert generation
- Virtual Metrology predictions
- Artifact storage (logs, telemetry, 2D profiles)
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
from app.models.job import JobStore, JobStatus as JobStatusEnum, JobType, job_to_response
from app.simulators.ion_implant_hil import IonImplantHILDriver
from app.controllers.ion import (
    DoseIntegrator,
    ScanUniformityController,
    R2RController,
    BeamDriftDetector,
)
from app.spc.monitors import IonImplantMonitor, IonParameter
from app.spc.charts import SPCAlert
from app.vm.ion_vm import IonVirtualMetrologyModel, IonVMFeatures


# ============================================================================
# Configuration
# ============================================================================

# Storage paths
TELEMETRY_STORAGE = os.getenv("TELEMETRY_STORAGE", "/tmp/spectra/telemetry")
ARTIFACT_STORAGE = os.getenv("ARTIFACT_STORAGE", "/tmp/spectra/artifacts")
LOGS_STORAGE = os.getenv("LOGS_STORAGE", "/tmp/spectra/logs")

# Ensure directories exist
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
                raise Ignore()  # Don't execute cancelled tasks

        return super().__call__(*args, **kwargs)


# ============================================================================
# Recipe Validation
# ============================================================================

def validate_ion_recipe(recipe: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ion implantation recipe against SOP and calibration.

    Args:
        recipe: Recipe dictionary containing species, energy, dose, etc.

    Returns:
        Validation result with warnings and errors

    Validation checks:
        - Species supported: B, P, As, BF2, In, Sb
        - Energy range: 1-200 keV
        - Dose range: 1e11 - 1e16 atoms/cm²
        - Tilt/twist angles: 0-90°
        - Beam current: 0.1-50 mA
        - Scan speed: 1-100 mm/s
    """
    errors = []
    warnings = []

    # Species validation
    valid_species = ["B", "P", "As", "BF2", "In", "Sb"]
    species = recipe.get("species", "")
    if species not in valid_species:
        errors.append(f"Invalid species '{species}'. Must be one of {valid_species}")

    # Energy validation
    energy_kev = recipe.get("energy_kev", 0)
    if not (1 <= energy_kev <= 200):
        errors.append(f"Energy {energy_kev} keV out of range [1, 200]")

    # High energy channeling warning
    if energy_kev > 100 and recipe.get("tilt_deg", 0) < 7:
        warnings.append(
            f"High energy ({energy_kev} keV) with low tilt ({recipe.get('tilt_deg')}°). "
            "Risk of channeling. Recommend tilt ≥ 7°"
        )

    # Dose validation
    dose = recipe.get("dose_atoms_cm2", 0)
    if not (1e11 <= dose <= 1e16):
        errors.append(f"Dose {dose:.2e} atoms/cm² out of range [1e11, 1e16]")

    # Tilt/twist validation
    tilt = recipe.get("tilt_deg", 0)
    twist = recipe.get("twist_deg", 0)
    if not (0 <= tilt <= 90):
        errors.append(f"Tilt angle {tilt}° out of range [0, 90]")
    if not (0 <= twist <= 90):
        errors.append(f"Twist angle {twist}° out of range [0, 90]")

    # Beam current validation
    beam_current = recipe.get("beam_current_ma", 0)
    if not (0.1 <= beam_current <= 50):
        errors.append(f"Beam current {beam_current} mA out of range [0.1, 50]")

    # Scan parameters
    scan_speed = recipe.get("scan_speed_mm_s", 0)
    if not (1 <= scan_speed <= 100):
        errors.append(f"Scan speed {scan_speed} mm/s out of range [1, 100]")

    # Dose rate check (uniformity)
    if beam_current > 10 and scan_speed > 50:
        warnings.append(
            f"High beam current ({beam_current} mA) with fast scan ({scan_speed} mm/s). "
            "May cause non-uniformity."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ============================================================================
# Telemetry Storage
# ============================================================================

def store_telemetry(run_id: str, telemetry: Dict[str, Any]) -> str:
    """Store telemetry data to filesystem.

    In production, use MinIO/S3.

    Args:
        run_id: Run identifier
        telemetry: Telemetry dictionary

    Returns:
        URI to stored telemetry
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"ion_{run_id}_{timestamp}.json"
    filepath = os.path.join(TELEMETRY_STORAGE, filename)

    with open(filepath, "w") as f:
        json.dump(telemetry, f, indent=2)

    return f"file://{filepath}"


def store_artifact(run_id: str, artifact_type: str, data: Any) -> Dict[str, str]:
    """Store artifact (2D profile, chart, etc.) to filesystem.

    Args:
        run_id: Run identifier
        artifact_type: Type of artifact (e.g., "dose_profile_2d")
        data: Artifact data (dict, array, etc.)

    Returns:
        Artifact metadata dict with uri and type
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"ion_{run_id}_{artifact_type}_{timestamp}.json"
    filepath = os.path.join(ARTIFACT_STORAGE, filename)

    # Convert numpy arrays to lists for JSON serialization
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
    """Store run logs to filesystem.

    Args:
        run_id: Run identifier
        logs: List of log messages

    Returns:
        URI to stored logs
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"ion_{run_id}_{timestamp}.log"
    filepath = os.path.join(LOGS_STORAGE, filename)

    with open(filepath, "w") as f:
        f.write("\n".join(logs))

    return f"file://{filepath}"


# ============================================================================
# Main Ion Implantation Task
# ============================================================================

@celery_app.task(
    bind=True,
    base=CancellableTask,
    name="app.tasks.ion_tasks.execute_ion_run",
    max_retries=3,
    default_retry_delay=60,
)
def execute_ion_run(
    self,
    job_id: str,
    run_id: str,
    org_id: str,
    recipe: Dict[str, Any],
    user_id: str,
) -> Dict[str, Any]:
    """Execute ion implantation run with full process control.

    This is the main Celery task for Ion Implantation. It:
    1. Validates recipe against SOP + calibration
    2. Initializes HIL simulator and controllers
    3. Executes run with real-time telemetry
    4. Updates job progress continuously
    5. Runs SPC checks and generates alerts
    6. Runs VM predictions for metrology
    7. Stores all telemetry and artifacts
    8. Handles cancellation gracefully

    Args:
        job_id: Job identifier for tracking
        run_id: Run identifier
        org_id: Organization ID
        recipe: Ion implantation recipe
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
        log(f"Starting ion implantation run {run_id}")
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
        log("Validating recipe against SOP and calibration")
        update_progress(5.0, "Validating recipe")

        validation = validate_ion_recipe(recipe)
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
        log("Initializing HIL simulator")
        update_progress(10.0, "Initializing simulator")

        hil_sim = IonImplantHILDriver()

        # Configure from recipe
        species = recipe["species"]
        energy_kev = recipe["energy_kev"]
        dose_atoms_cm2 = recipe["dose_atoms_cm2"]
        beam_current_ma = recipe["beam_current_ma"]
        tilt_deg = recipe.get("tilt_deg", 7.0)
        twist_deg = recipe.get("twist_deg", 0.0)

        wafer_area_cm2 = np.pi * (recipe.get("wafer_diameter_mm", 300) / 10 / 2) ** 2

        log(f"Species: {species}, Energy: {energy_kev} keV, Dose: {dose_atoms_cm2:.2e} atoms/cm²")
        log(f"Beam current: {beam_current_ma} mA, Tilt: {tilt_deg}°, Twist: {twist_deg}°")

        # =====================================================================
        # Step 3: Initialize Controllers (15%)
        # =====================================================================
        log("Initializing dose integrator and controllers")
        update_progress(15.0, "Initializing controllers")

        dose_integrator = DoseIntegrator(
            target_dose=dose_atoms_cm2,
            wafer_area_cm2=wafer_area_cm2,
        )

        scan_controller = ScanUniformityController(
            kp=0.8,
            ki=0.2,
            kd=0.1,
            target_dose=dose_atoms_cm2,
        )

        # R2R controller (if previous runs available)
        r2r_controller = R2RController(
            target_dose=dose_atoms_cm2,
            alpha=0.3,
        )

        # Beam drift FDC
        beam_fdc = BeamDriftDetector(
            target_current=beam_current_ma,
            k=0.5,
            h=5.0,
        )

        # SPC monitor
        spc_monitor = IonImplantMonitor()

        # VM model
        vm_model = IonVirtualMetrologyModel()

        log("Controllers initialized")

        # =====================================================================
        # Step 4: Execute Run with Telemetry (20-80%)
        # =====================================================================
        log("Starting implantation process")
        update_progress(20.0, "Running implantation")

        # Simulate run duration based on dose and beam current
        # Q = I·t → t = Q/I = (Dose × Area) / (Current / q)
        # For simulation, scale down to reasonable time
        total_charge_coulombs = dose_atoms_cm2 * wafer_area_cm2 * 1.602e-19
        estimated_time_sec = total_charge_coulombs / (beam_current_ma * 1e-3)
        sim_duration_sec = min(estimated_time_sec / 100, 30)  # Scale down for demo

        log(f"Estimated simulation duration: {sim_duration_sec:.1f} seconds")

        dt = 0.1  # 100ms sampling
        num_steps = int(sim_duration_sec / dt)

        telemetry_history = {
            "time_s": [],
            "beam_current_ma": [],
            "chamber_pressure_torr": [],
            "analyzer_field_v": [],
            "integrated_dose_atoms_cm2": [],
            "dose_uniformity_pct": [],
            "wafer_temp_c": [],
        }

        spc_alerts = []
        current_dose = 0.0

        for step in range(num_steps):
            # Check for cancellation
            job = JobStore.get_job(job_id)
            if job.status == JobStatusEnum.CANCELLED:
                log("Run cancelled by user")
                raise Ignore()

            # Simulate telemetry
            t = step * dt
            telemetry = hil_sim.get_telemetry()

            # Add some realistic variation
            beam_current = beam_current_ma * (1 + np.random.normal(0, 0.02))
            pressure = telemetry["chamber_pressure_torr"] * (1 + np.random.normal(0, 0.05))
            field = energy_kev * 1000 * (1 + np.random.normal(0, 0.001))

            # Dose integration
            dose_rate = beam_current * 1e-3 / (1.602e-19 * wafer_area_cm2)  # atoms/cm²/s
            current_dose += dose_rate * dt

            # Scan uniformity (simulate)
            uniformity = 95.0 + np.random.normal(0, 2.0)

            # Store telemetry
            telemetry_history["time_s"].append(t)
            telemetry_history["beam_current_ma"].append(beam_current)
            telemetry_history["chamber_pressure_torr"].append(pressure)
            telemetry_history["analyzer_field_v"].append(field)
            telemetry_history["integrated_dose_atoms_cm2"].append(current_dose)
            telemetry_history["dose_uniformity_pct"].append(uniformity)
            telemetry_history["wafer_temp_c"].append(telemetry["wafer_temp_c"])

            # Run SPC checks every 10 steps (1 second)
            if step % 10 == 0 and step > 0:
                spc_data = {
                    IonParameter.BEAM_CURRENT_MA: beam_current,
                    IonParameter.CHAMBER_PRESSURE_TORR: pressure,
                    IonParameter.DOSE_UNIFORMITY_PCT: uniformity,
                    IonParameter.ANALYZER_FIELD_V: field,
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

            # Beam drift FDC
            fdc_alert = beam_fdc.check_drift(beam_current, t)
            if fdc_alert:
                log(f"FDC ALERT: {fdc_alert['message']}")

            # Update progress (20% to 80% during run)
            progress = 20.0 + (step / num_steps) * 60.0
            if step % 10 == 0:
                update_progress(progress, f"Implanting ({current_dose / dose_atoms_cm2 * 100:.1f}% dose)")

            # Simulate real-time execution
            time.sleep(dt / 10)  # Speed up 10x for demo

        log(f"Implantation complete. Final dose: {current_dose:.2e} atoms/cm²")

        # =====================================================================
        # Step 5: Final Dose Check and R2R Update (80%)
        # =====================================================================
        log("Checking final dose accuracy")
        update_progress(80.0, "Verifying dose")

        dose_error_pct = abs(current_dose - dose_atoms_cm2) / dose_atoms_cm2 * 100
        log(f"Dose error: {dose_error_pct:.2f}%")

        if dose_error_pct > 3.0:
            log(f"WARNING: Dose error {dose_error_pct:.2f}% exceeds 3% tolerance")

        # R2R recommendation for next run
        r2r_recommendation = r2r_controller.recommend_adjustment(
            target_dose=dose_atoms_cm2,
            measured_dose=current_dose,
        )
        log(f"R2R: Recommended beam current adjustment: {r2r_recommendation.dose_adjustment_pct:+.2f}%")

        # =====================================================================
        # Step 6: Generate 2D Dose Profile (85%)
        # =====================================================================
        log("Generating 2D dose profile")
        update_progress(85.0, "Generating profiles")

        profile_2d = hil_sim.get_dose_profile_2d()

        # =====================================================================
        # Step 7: Run Virtual Metrology (90%)
        # =====================================================================
        log("Running Virtual Metrology predictions")
        update_progress(90.0, "Running VM predictions")

        vm_features = IonVMFeatures(
            species=species,
            energy_kev=energy_kev,
            dose_atoms_cm2=current_dose,
            tilt_deg=tilt_deg,
            twist_deg=twist_deg,
            beam_current_ma=beam_current_ma,
            chamber_pressure_torr=np.mean(telemetry_history["chamber_pressure_torr"]),
            wafer_temp_c=np.mean(telemetry_history["wafer_temp_c"]),
        )

        vm_prediction = vm_model.predict(vm_features)
        log(f"VM Prediction: Rs={vm_prediction.sheet_resistance_ohm_sq:.1f} Ω/sq, "
            f"Xj={vm_prediction.junction_depth_um:.3f} μm, "
            f"Activation={vm_prediction.activation_pct:.1f}%")

        # =====================================================================
        # Step 8: Store Artifacts (95%)
        # =====================================================================
        log("Storing telemetry and artifacts")
        update_progress(95.0, "Storing artifacts")

        # Store telemetry
        telemetry_uri = store_telemetry(run_id, telemetry_history)
        log(f"Telemetry stored: {telemetry_uri}")

        # Store artifacts
        artifacts = []

        # 2D dose profile
        profile_artifact = store_artifact(run_id, "dose_profile_2d", profile_2d)
        artifacts.append(profile_artifact)
        log(f"2D profile stored: {profile_artifact['uri']}")

        # SPC charts (if any)
        if spc_alerts:
            alerts_artifact = store_artifact(run_id, "spc_alerts", spc_alerts)
            artifacts.append(alerts_artifact)
            log(f"SPC alerts stored: {alerts_artifact['uri']}")

        # VM predictions
        vm_artifact = store_artifact(run_id, "vm_prediction", {
            "sheet_resistance_ohm_sq": vm_prediction.sheet_resistance_ohm_sq,
            "junction_depth_um": vm_prediction.junction_depth_um,
            "activation_pct": vm_prediction.activation_pct,
            "confidence_score": vm_prediction.confidence_score,
            "features": vm_features.dict(),
        })
        artifacts.append(vm_artifact)
        log(f"VM prediction stored: {vm_artifact['uri']}")

        # Store logs
        logs_uri = store_logs(run_id, logs)
        log(f"Logs stored: {logs_uri}")

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
                "final_dose_atoms_cm2": current_dose,
                "dose_error_pct": dose_error_pct,
                "vm_prediction": vm_prediction.dict(),
                "spc_alerts_count": len(spc_alerts),
                "r2r_adjustment_pct": r2r_recommendation.dose_adjustment_pct,
            },
        )

        return {
            "run_id": run_id,
            "status": "completed",
            "final_dose_atoms_cm2": current_dose,
            "dose_error_pct": dose_error_pct,
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
# Simulation Task (Synchronous Dose Profile Calculator)
# ============================================================================

@celery_app.task(
    name="app.tasks.ion_tasks.simulate_dose_profile",
    time_limit=30,
)
def simulate_dose_profile(
    species: str,
    energy_kev: float,
    dose_atoms_cm2: float,
    tilt_deg: float = 7.0,
    twist_deg: float = 0.0,
) -> Dict[str, Any]:
    """Simulate 1D and 2D dose profiles (SRIM-like calculation).

    This is a quick synchronous task for dose profile visualization.

    Args:
        species: Dopant species
        energy_kev: Implant energy
        dose_atoms_cm2: Total dose
        tilt_deg: Tilt angle
        twist_deg: Twist angle

    Returns:
        Dictionary with 1D depth profile and 2D lateral profile
    """
    from app.models.ion_range import SRIMEstimator

    estimator = SRIMEstimator()

    # 1D depth profile
    profile_1d = estimator.get_depth_profile(
        species=species,
        energy_kev=energy_kev,
        dose=dose_atoms_cm2,
    )

    # 2D lateral profile (simplified)
    x = np.linspace(-150, 150, 100)  # mm
    y = np.linspace(-150, 150, 100)
    X, Y = np.meshgrid(x, y)

    # Gaussian distribution with edge roll-off
    r = np.sqrt(X**2 + Y**2)
    profile_2d = dose_atoms_cm2 * np.exp(-(r / 50)**2)

    return {
        "profile_1d": {
            "depth_nm": profile_1d["depth_nm"],
            "concentration_cm3": profile_1d["concentration_cm3"],
            "projected_range_nm": profile_1d["projected_range_nm"],
            "straggle_nm": profile_1d["straggle_nm"],
        },
        "profile_2d": {
            "x_mm": x.tolist(),
            "y_mm": y.tolist(),
            "dose_atoms_cm2": profile_2d.tolist(),
        },
        "metadata": {
            "species": species,
            "energy_kev": energy_kev,
            "dose_atoms_cm2": dose_atoms_cm2,
            "tilt_deg": tilt_deg,
            "twist_deg": twist_deg,
        },
    }


# Export
__all__ = [
    "execute_ion_run",
    "simulate_dose_profile",
    "validate_ion_recipe",
]
