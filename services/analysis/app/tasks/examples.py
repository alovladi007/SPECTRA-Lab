"""
Examples for Job Queue & Real-Time UX

Demonstrates:
- Submitting CVD runs to Celery
- Monitoring progress with real-time updates
- Receiving stress/adhesion risk indicators
"""

import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Submit Simple CVD Run
# =============================================================================

def example_submit_cvd_run():
    """
    Submit a CVD run to Celery task queue
    """
    logger.info("=" * 70)
    logger.info("Example 1: Submit CVD Run to Task Queue")
    logger.info("=" * 70)

    from .cvd_orchestration import run_cvd_simulation

    # Define run parameters
    run_id = f"CVD_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    recipe_params = {
        "mode": "thermal",
        "temperature_c": 800.0,
        "pressure_torr": 0.5,
        "precursor_flow_sccm": 80.0,
        "carrier_gas_flow_sccm": 500.0,
        "film_material": "Si3N4",
        "target_thickness_nm": 100.0,
        "duration_sec": 120.0,  # 2 minutes for demo
    }

    simulation_config = {
        "enable_noise": True,
        "enable_drift": False,
        "enable_faults": False,
    }

    # Submit task
    logger.info(f"Submitting CVD run: {run_id}")

    task = run_cvd_simulation.apply_async(
        args=[run_id, recipe_params, simulation_config],
        queue="cvd_runs",
    )

    logger.info(f"Task submitted: task_id={task.id}")
    logger.info(f"Task state: {task.state}")

    # Monitor task progress
    logger.info("\nMonitoring task progress...")

    while not task.ready():
        # Get current state
        if task.state == "PROGRESS":
            info = task.info
            progress = info.get("progress", 0)
            thickness = info.get("current_thickness", 0)

            logger.info(
                f"Progress: {progress:.1f}% | Thickness: {thickness:.2f} nm"
            )

        time.sleep(2)

    # Get result
    if task.successful():
        result = task.result
        logger.info("\n✅ Task completed successfully!")
        logger.info(f"Final metrics:")
        logger.info(f"  Thickness: {result['metrics']['thickness_mean_nm']:.2f} nm")
        logger.info(f"  Stress: {result['metrics']['stress_mean_mpa']:.1f} MPa")
        logger.info(f"  Adhesion: {result['metrics']['adhesion_score']:.1f}/100")
    else:
        logger.error(f"\n❌ Task failed: {task.result}")

    return task.result


# =============================================================================
# Example 2: Monitor Real-Time Events
# =============================================================================

def example_monitor_realtime_events():
    """
    Monitor real-time events for a CVD run

    NOTE: Requires Redis to be running
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Monitor Real-Time Events")
    logger.info("=" * 70)

    from .cvd_orchestration import run_cvd_simulation
    from ..realtime.events import subscribe_to_run, get_run_events

    # Submit run
    run_id = f"CVD_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    recipe_params = {
        "temperature_c": 780.0,
        "pressure_torr": 0.3,
        "precursor_flow_sccm": 80.0,
        "carrier_gas_flow_sccm": 500.0,
        "film_material": "Si3N4",
        "duration_sec": 60.0,  # 1 minute
    }

    # Start task in background
    task = run_cvd_simulation.apply_async(
        args=[run_id, recipe_params],
        queue="cvd_runs",
    )

    logger.info(f"Task started: {task.id}")
    logger.info(f"Subscribing to real-time events for {run_id}...")

    # Subscribe to events
    try:
        for event in subscribe_to_run(run_id):
            logger.info(f"\n[{event.event_type.value}] @ {event.timestamp}")

            if event.event_type.value == "progress_update":
                logger.info(f"  Progress: {event.data.get('progress', 0):.1f}%")
                logger.info(f"  Thickness: {event.data.get('current_thickness_nm', 0):.2f} nm")

            elif event.event_type.value == "warning":
                warnings = event.data.get("warnings", [])
                for w in warnings:
                    logger.warning(f"  ⚠️  {w['type']}: {w['message']}")

            elif event.event_type.value == "run_completed":
                logger.info("  ✅ Run completed!")
                metrics = event.data.get("final_metrics", {})
                logger.info(f"  Final thickness: {metrics.get('thickness_mean_nm', 0):.2f} nm")
                break

            elif event.event_type.value == "run_failed":
                logger.error(f"  ❌ Run failed: {event.data.get('error')}")
                break

    except KeyboardInterrupt:
        logger.info("\nStopped monitoring")

    # Get event history
    logger.info("\nFetching event history...")
    events = get_run_events(run_id)
    logger.info(f"Total events: {len(events)}")


# =============================================================================
# Example 3: Stress Risk Monitoring
# =============================================================================

def example_stress_risk_monitoring():
    """
    Demonstrate stress risk indicator during deposition
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Stress Risk Monitoring")
    logger.info("=" * 70)

    from .cvd_orchestration import run_cvd_simulation

    run_id = f"CVD_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # High temperature + high rate = high stress risk
    recipe_params = {
        "temperature_c": 850.0,  # High temp → more tensile stress
        "pressure_torr": 0.5,
        "precursor_flow_sccm": 120.0,  # High flow → high rate
        "carrier_gas_flow_sccm": 500.0,
        "film_material": "Si3N4",
        "duration_sec": 90.0,
    }

    logger.info("Running high-stress recipe...")
    logger.info(f"Temperature: {recipe_params['temperature_c']}°C (HIGH)")
    logger.info(f"Precursor flow: {recipe_params['precursor_flow_sccm']} sccm (HIGH)")

    task = run_cvd_simulation.apply_async(
        args=[run_id, recipe_params],
        queue="cvd_runs",
    )

    # Monitor for warnings
    logger.info("\nMonitoring for stress warnings...")

    from ..realtime.events import subscribe_to_run

    try:
        for event in subscribe_to_run(run_id):
            if event.event_type.value == "warning":
                warnings = event.data.get("warnings", [])
                for w in warnings:
                    if "stress" in w["type"]:
                        logger.warning(f"\n⚠️  STRESS RISK DETECTED!")
                        logger.warning(f"   Type: {w['type']}")
                        logger.warning(f"   Message: {w['message']}")
                        if "recommendation" in w:
                            logger.warning(f"   Recommendation: {w['recommendation']}")

            elif event.event_type.value in ["run_completed", "run_failed"]:
                break

    except KeyboardInterrupt:
        pass


# =============================================================================
# Example 4: Multi-Step Recipe
# =============================================================================

def example_multi_step_recipe():
    """
    Submit a multi-step CVD recipe (e.g., barrier + fill)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Multi-Step Recipe")
    logger.info("=" * 70)

    from .cvd_orchestration import run_multi_step_recipe

    run_id = f"MULTI_STEP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Step 1: TiN barrier layer (thin, high adhesion)
    step1 = {
        "mode": "plasma",
        "temperature_c": 350.0,
        "pressure_torr": 5.0,
        "precursor_flow_sccm": 50.0,
        "carrier_gas_flow_sccm": 500.0,
        "rf_power_w": 100.0,
        "film_material": "TiN",
        "target_thickness_nm": 20.0,
        "duration_sec": 30.0,
    }

    # Step 2: W fill layer (thick)
    step2 = {
        "mode": "thermal",
        "temperature_c": 400.0,
        "pressure_torr": 80.0,
        "precursor_flow_sccm": 100.0,
        "carrier_gas_flow_sccm": 1000.0,
        "film_material": "W",
        "target_thickness_nm": 200.0,
        "duration_sec": 60.0,
    }

    recipe_steps = [step1, step2]

    logger.info(f"Submitting multi-step recipe: {run_id}")
    logger.info(f"  Step 1: {step1['film_material']} barrier ({step1['target_thickness_nm']} nm)")
    logger.info(f"  Step 2: {step2['film_material']} fill ({step2['target_thickness_nm']} nm)")

    result = run_multi_step_recipe(run_id, recipe_steps)

    logger.info(f"\nRecipe submitted: task_id={result['task_id']}")
    logger.info(f"Total steps: {result['num_steps']}")


# =============================================================================
# Example 5: Task Cancellation
# =============================================================================

def example_task_cancellation():
    """
    Start a CVD run and cancel it mid-process
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Task Cancellation")
    logger.info("=" * 70)

    from .cvd_orchestration import run_cvd_simulation, cancel_cvd_run
    from celery.result import AsyncResult

    run_id = f"CVD_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    recipe_params = {
        "temperature_c": 800.0,
        "pressure_torr": 0.5,
        "precursor_flow_sccm": 80.0,
        "carrier_gas_flow_sccm": 500.0,
        "film_material": "Si3N4",
        "duration_sec": 300.0,  # 5 minutes
    }

    # Start task
    task = run_cvd_simulation.apply_async(
        args=[run_id, recipe_params],
        queue="cvd_runs",
    )

    logger.info(f"Task started: {task.id}")
    logger.info("Waiting 10 seconds before cancelling...")

    time.sleep(10)

    # Cancel task
    logger.info(f"\nCancelling run: {run_id}")
    task.revoke(terminate=True)

    # Emit cancellation event
    cancel_cvd_run(run_id)

    logger.info("Task cancelled")


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run job queue examples"""
    logger.info("\n" + "=" * 70)
    logger.info("Job Queue & Real-Time UX - Examples")
    logger.info("=" * 70)

    logger.info("\nNOTE: These examples require:")
    logger.info("  1. Redis server running (redis://localhost:6379)")
    logger.info("  2. Celery worker running:")
    logger.info("     celery -A app.tasks.celery_app worker --loglevel=info")
    logger.info("  3. Celery beat running (for periodic tasks):")
    logger.info("     celery -A app.tasks.celery_app beat --loglevel=info")

    choice = input("\nProceed with examples? (y/n): ")

    if choice.lower() != "y":
        logger.info("Exiting...")
        return

    # Run examples
    example_submit_cvd_run()

    logger.info("\n" + "=" * 70)
    logger.info("All examples completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
