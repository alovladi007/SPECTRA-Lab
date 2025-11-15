"""
Example Usage of CVD Tool Drivers

Demonstrates how to use the CVDTool abstraction layer with:
- HIL Simulator
- Real drivers (PECVD, LPCVD, MOCVD)
- Communication adapters
"""

import asyncio
import logging
from uuid import uuid4
from datetime import datetime

from .hil_simulator import HILCVDSimulator, PhysicsConfig, FaultInjectionConfig
from .thermal_cvd import LPCVDDriver
from .plasma_cvd import PECVDDriver
from .specialty_cvd import MOCVDDriver
from .cvd_tool import ToolState, TelemetryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: HIL Simulator for LPCVD Process
# =============================================================================

async def example_hil_lpcvd_simulation():
    """
    Simulate LPCVD Silicon Nitride deposition with realistic physics
    """
    logger.info("=" * 70)
    logger.info("Example 1: HIL LPCVD Simulation (Silicon Nitride)")
    logger.info("=" * 70)

    # Configure physics for Si₃N₄ LPCVD
    physics = PhysicsConfig(
        # Arrhenius parameters for SiH₄ + NH₃ → Si₃N₄
        pre_exponential_factor=1e12,
        activation_energy_kj_mol=120.0,
        pressure_exponent=0.5,

        # Film properties
        intrinsic_stress_mpa=-250.0,  # Compressive
        intrinsic_stress_std_mpa=30.0,
        base_adhesion_score=88.0,

        # Thermal stress (Si₃N₄ on Si)
        deposition_temp_c=780.0,
        film_cte_ppm_k=2.8,
        substrate_cte_ppm_k=2.6,

        # Material properties
        film_density_g_cm3=3.1,
        film_hardness_gpa=20.0,
        refractive_index_real=2.0,
    )

    # Create simulator
    simulator = HILCVDSimulator(
        tool_id="SIM-LPCVD-01",
        vendor="HIL Simulator",
        model="Virtual LPCVD",
        mode="LPCVD",
        physics_config=physics,
    )

    # Connect
    await simulator.connect()
    logger.info(f"Simulator state: {simulator.state}")

    # Get capabilities
    caps = await simulator.get_capabilities()
    logger.info(f"Tool: {caps.vendor} {caps.model}")
    logger.info(f"Modes: {caps.supported_modes}")
    logger.info(f"Temp range: {caps.min_temp_c}-{caps.max_temp_c}°C")
    logger.info(f"Pressure range: {caps.min_pressure_torr}-{caps.max_pressure_torr} Torr")

    # Create mock recipe
    class MockRecipe:
        recipe_name = "Si3N4_100nm"
        target_temp_c = 780.0
        target_pressure_torr = 0.3
        target_thickness_nm = 100.0
        film_material = "Si3N4"

    recipe = MockRecipe()

    # Configure recipe
    await simulator.configure(recipe)
    logger.info(f"Recipe '{recipe.recipe_name}' configured")

    # Start run
    run_id = uuid4()
    logger.info(f"Starting run {run_id}")
    await simulator.start_run(run_id)

    # Set process conditions
    simulator.chamber_temp_c = recipe.target_temp_c
    simulator.chamber_pressure_torr = recipe.target_pressure_torr
    simulator.gas_flows_sccm = {"SiH4": 80.0, "NH3": 200.0, "N2": 500.0}

    # Stream telemetry for 10 seconds
    logger.info("Streaming telemetry...")
    sample_count = 0
    max_samples = 10

    async for telemetry in simulator.stream_telemetry(run_id, interval_sec=1.0):
        sample_count += 1

        logger.info(f"\n--- Sample {sample_count} ---")
        logger.info(f"Time: {telemetry.elapsed_time_sec:.1f}s")
        logger.info(f"Temperature: {telemetry.measurements[TelemetryType.TEMPERATURE]:.1f}°C")
        logger.info(f"Pressure: {telemetry.measurements[TelemetryType.PRESSURE]:.4f} Torr")
        logger.info(f"Thickness: {telemetry.thickness_nm:.2f} nm")
        logger.info(f"Deposition Rate: {telemetry.deposition_rate_nm_min:.2f} nm/min")
        logger.info(f"Film Stress: {telemetry.stress_mpa:.1f} MPa")
        logger.info(f"Gas Flows: {telemetry.gas_flows_sccm}")

        if sample_count >= max_samples:
            break

    # Stop run
    await simulator.stop_run(run_id)
    logger.info("Run stopped")

    # Get final status
    status = await simulator.get_status()
    logger.info(f"Final state: {status.state}")
    logger.info(f"Final thickness: {simulator.deposited_thickness_nm:.2f} nm")
    logger.info(f"Final stress: {simulator.film_stress_mpa:.1f} MPa")
    logger.info(f"Adhesion score: {simulator.adhesion_score:.1f}/100")

    # Disconnect
    await simulator.disconnect()


# =============================================================================
# Example 2: HIL Simulator with Fault Injection
# =============================================================================

async def example_hil_with_faults():
    """
    Simulate CVD process with fault injection for FDC testing
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: HIL Simulation with Fault Injection")
    logger.info("=" * 70)

    # Configure fault injection
    faults = FaultInjectionConfig(
        enabled=True,
        temp_spike_probability=0.1,  # 10% chance per second
        temp_spike_magnitude_c=30.0,
        pressure_leak_probability=0.05,
        pressure_leak_rate_torr_min=0.01,
    )

    simulator = HILCVDSimulator(
        tool_id="SIM-PECVD-FAULT",
        mode="PECVD",
        fault_config=faults,
    )

    await simulator.connect()

    class MockRecipe:
        recipe_name = "SiO2_Fault_Test"
        target_temp_c = 300.0
        target_pressure_torr = 1.5

    await simulator.configure(MockRecipe())

    run_id = uuid4()
    await simulator.start_run(run_id)

    simulator.chamber_temp_c = 300.0
    simulator.chamber_pressure_torr = 1.5

    logger.info("Monitoring for faults...")
    sample_count = 0

    async for telemetry in simulator.stream_telemetry(run_id, interval_sec=0.5):
        sample_count += 1

        status = await simulator.get_status()

        if status.active_alarms:
            logger.warning(f"⚠️  ALARMS DETECTED: {status.active_alarms}")

        if sample_count >= 20:
            break

    await simulator.stop_run(run_id)

    # Check alarm history
    alarms = await simulator.get_alarms()
    if alarms:
        logger.info(f"\nTotal alarms during run: {len(alarms)}")
        for alarm in alarms:
            logger.info(f"  - {alarm}")

    await simulator.disconnect()


# =============================================================================
# Example 3: PECVD Driver (Stub)
# =============================================================================

async def example_pecvd_driver():
    """
    Use PECVD driver to control real tool (stub implementation)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: PECVD Driver (Stub)")
    logger.info("=" * 70)

    driver = PECVDDriver(
        tool_id="PECVD-FAB1-01",
        host="192.168.1.100",
        port=5025,
        vendor="Applied Materials",
        model="Centura",
    )

    # Connect
    await driver.connect()
    logger.info("Connected to PECVD tool")

    # Get capabilities
    caps = await driver.get_capabilities()
    logger.info(f"Tool: {caps.vendor} {caps.model}")
    logger.info(f"Max RF Power: {caps.max_rf_power_w}W")
    logger.info(f"RF Frequency: {caps.rf_frequency_mhz} MHz")
    logger.info(f"Available gases: {caps.available_gas_lines}")

    # Configure recipe
    class MockPECVDRecipe:
        recipe_name = "SiO2_PECVD_300nm"
        target_temp_c = 300.0
        target_pressure_torr = 1.5
        rf_power_w = 500.0

    await driver.configure(MockPECVDRecipe())

    # Start run
    run_id = uuid4()
    await driver.start_run(run_id)

    # Monitor (stub will return mock data)
    sample_count = 0
    async for telemetry in driver.stream_telemetry(run_id, interval_sec=1.0):
        logger.info(f"Temp: {telemetry.measurements[TelemetryType.TEMPERATURE]:.1f}°C, "
                   f"RF Power: {telemetry.measurements[TelemetryType.POWER]:.0f}W")

        sample_count += 1
        if sample_count >= 5:
            break

    # Stop
    await driver.stop_run(run_id)
    await driver.disconnect()


# =============================================================================
# Example 4: LPCVD Batch Processing
# =============================================================================

async def example_lpcvd_batch():
    """
    LPCVD batch processing (stub)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: LPCVD Batch Processing (Stub)")
    logger.info("=" * 70)

    driver = LPCVDDriver(
        tool_id="LPCVD-BATCH-01",
        host="192.168.1.101",
        port=4840,  # OPC-UA
        vendor="ASM",
        model="A412",
    )

    await driver.connect()

    caps = await driver.get_capabilities()
    logger.info(f"Max batch size: {caps.max_batch_size} wafers")

    class MockLPCVDRecipe:
        recipe_name = "PolySi_200nm_Batch"
        target_temp_c = 620.0
        target_pressure_torr = 0.2
        batch_size = 50

    await driver.configure(MockLPCVDRecipe())

    run_id = uuid4()
    await driver.start_run(run_id)

    # Monitor for a few samples
    sample_count = 0
    async for telemetry in driver.stream_telemetry(run_id, interval_sec=2.0):
        logger.info(f"Furnace temp: {telemetry.measurements[TelemetryType.TEMPERATURE]:.1f}°C")
        sample_count += 1
        if sample_count >= 3:
            break

    await driver.stop_run(run_id)
    await driver.disconnect()


# =============================================================================
# Example 5: MOCVD Epitaxial Growth
# =============================================================================

async def example_mocvd_epitaxy():
    """
    MOCVD GaN epitaxy (stub)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: MOCVD GaN Epitaxy (Stub)")
    logger.info("=" * 70)

    driver = MOCVDDriver(
        tool_id="MOCVD-GaN-01",
        host="192.168.1.102",
        port=4840,
        vendor="Aixtron",
        model="AIX 2800G4",
    )

    await driver.connect()

    caps = await driver.get_capabilities()
    logger.info(f"Available MO sources: {caps.available_gas_lines}")

    class MockMOCVDRecipe:
        recipe_name = "GaN_2um_Buffer"
        target_temp_c = 1050.0
        target_pressure_torr = 300.0

    await driver.configure(MockMOCVDRecipe())

    run_id = uuid4()
    await driver.start_run(run_id)

    # Monitor V/III ratio and growth rate
    sample_count = 0
    async for telemetry in driver.stream_telemetry(run_id, interval_sec=1.0):
        v_iii = telemetry.raw_data.get('v_iii_ratio') if telemetry.raw_data else None
        logger.info(f"Growth rate: {telemetry.deposition_rate_nm_min:.2f} nm/min, "
                   f"V/III: {v_iii}")

        sample_count += 1
        if sample_count >= 5:
            break

    await driver.stop_run(run_id)
    await driver.disconnect()


# =============================================================================
# Main: Run All Examples
# =============================================================================

async def main():
    """Run all examples"""
    logger.info("\n" + "=" * 70)
    logger.info("CVD Tool Driver Examples")
    logger.info("=" * 70)

    # Example 1: HIL LPCVD with realistic physics
    await example_hil_lpcvd_simulation()

    # Example 2: HIL with fault injection
    await example_hil_with_faults()

    # Example 3: PECVD driver stub
    await example_pecvd_driver()

    # Example 4: LPCVD batch
    await example_lpcvd_batch()

    # Example 5: MOCVD epitaxy
    await example_mocvd_epitaxy()

    logger.info("\n" + "=" * 70)
    logger.info("All examples completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
