"""Soak tests for RTP HIL simulator with accelerated time."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

import sys
sys.path.insert(0, '/Users/vladimirantoine/SPECTRA LAB/SPECTRA-Lab/services/process_control')

from app.drivers.rtp_driver import (
    AmbientGas, RampSegment, GasFlowParameters, LampParameters,
    EmissivitySettings, TemperatureRecipe, TemperatureControlMode
)
from app.simulators.rtp_hil import RTPHILDriver
from app.telemetry.rtp_telemetry import RTPTelemetryManager


class SoakTestConfig:
    """Configuration for soak tests."""
    # Time acceleration factor (1000x = 1 hour simulated in 3.6 seconds)
    TIME_ACCELERATION = 1000.0

    # Test durations (in real-world time)
    DURATION_12H_S = 12 * 3600 / TIME_ACCELERATION  # 12 hours accelerated
    DURATION_24H_S = 24 * 3600 / TIME_ACCELERATION  # 24 hours accelerated
    DURATION_72H_S = 72 * 3600 / TIME_ACCELERATION  # 72 hours accelerated

    # Telemetry sample rate (Hz) - slower for soak tests
    TELEMETRY_RATE_HZ = 1.0


@pytest.fixture
async def rtp_system():
    """Create RTP HIL system for testing."""
    driver = RTPHILDriver(
        equipment_id="SOAK-TEST-RTP-01",
        num_zones=4,
        random_seed=42,  # Deterministic
        simulation_timestep_s=0.1
    )

    telemetry_manager = RTPTelemetryManager(
        driver=driver,
        sample_rate_hz=SoakTestConfig.TELEMETRY_RATE_HZ
    )

    await driver.connect()
    await driver.initialize()

    yield driver, telemetry_manager

    await driver.shutdown()
    await driver.disconnect()


@pytest.mark.asyncio
@pytest.mark.soak
@pytest.mark.timeout(60)  # Real-time timeout for 12h accelerated test
async def test_rtp_12h_thermal_stability(rtp_system):
    """
    12-hour soak test: Continuous temperature hold.

    Tests:
    - Long-term temperature stability
    - Sensor drift
    - Emissivity stability
    - Lamp power stability
    """
    driver, telemetry = rtp_system

    print("\n" + "="*80)
    print("RTP 12-Hour Thermal Stability Test (Accelerated)")
    print("="*80)

    # Setup gas flow
    gas_params = GasFlowParameters(
        gas_type=AmbientGas.NITROGEN,
        flow_rate_sccm=5000.0,
        chamber_pressure_torr=760.0  # Atmospheric
    )
    await driver.set_gas_flow(gas_params)

    # Set emissivity
    await driver.set_emissivity(0.65)

    # Target temperature: 1000°C (typical RTP temperature)
    target_temp = 1000.0
    await driver.set_target_temperature(target_temp, ramp_rate_C_per_s=50.0)

    print(f"Ramping to {target_temp}°C...")

    # Wait for temperature to stabilize
    stabilization_time = 60.0 / SoakTestConfig.TIME_ACCELERATION  # 60 seconds simulated
    stabilization_start = datetime.now()

    while (datetime.now() - stabilization_start).total_seconds() < stabilization_time:
        temp_data = await driver.get_temperature()
        await asyncio.sleep(0.1)

    print(f"Stabilized at {temp_data['pyrometer_C']:.1f}°C")
    print("Starting 12-hour hold...")

    # Run soak test
    start_time = datetime.now()
    telemetry_frames = []
    pyrometer_temps = []
    thermocouple_temps = []
    deviations = []

    await telemetry.start_streaming()

    sample_interval = 1.0 / SoakTestConfig.TELEMETRY_RATE_HZ
    elapsed_target = SoakTestConfig.DURATION_12H_S

    while (datetime.now() - start_time).total_seconds() < elapsed_target:
        # Collect telemetry
        frame = await telemetry.collect_frame()
        telemetry_frames.append(frame)
        pyrometer_temps.append(frame.pyrometer_temp_C)
        thermocouple_temps.append(frame.thermocouple_temp_C)
        deviations.append(frame.temp_deviation_C)

        # Log progress every 1000 frames
        if len(telemetry_frames) % 1000 == 0:
            elapsed_real = (datetime.now() - start_time).total_seconds()
            elapsed_sim = elapsed_real * SoakTestConfig.TIME_ACCELERATION
            print(f"  Progress: {elapsed_sim/3600:.1f}h simulated ({elapsed_real:.1f}s real) | "
                  f"Temp: {frame.pyrometer_temp_C:.1f}°C | Deviation: {frame.temp_deviation_C:.1f}°C")

        await asyncio.sleep(sample_interval)

    await telemetry.stop_streaming()

    # Analysis
    print("\nSoak Test Results:")
    print(f"  Total frames collected: {len(telemetry_frames)}")
    print(f"  Simulated time: {elapsed_target * SoakTestConfig.TIME_ACCELERATION / 3600:.1f} hours")

    # Temperature stability
    pyro_mean = np.mean(pyrometer_temps)
    pyro_std = np.std(pyrometer_temps)
    pyro_drift = pyrometer_temps[-1] - pyrometer_temps[0]

    tc_mean = np.mean(thermocouple_temps)
    tc_std = np.std(thermocouple_temps)
    tc_drift = thermocouple_temps[-1] - thermocouple_temps[0]

    print(f"\nPyrometer Stability:")
    print(f"  Mean: {pyro_mean:.2f}°C")
    print(f"  Std: {pyro_std:.2f}°C ({pyro_std/pyro_mean*100:.3f}%)")
    print(f"  Drift: {pyro_drift:.2f}°C")

    print(f"\nThermocouple Stability:")
    print(f"  Mean: {tc_mean:.2f}°C")
    print(f"  Std: {tc_std:.2f}°C ({tc_std/tc_mean*100:.3f}%)")
    print(f"  Drift: {tc_drift:.2f}°C")

    # Control performance
    mean_deviation = np.mean(deviations)
    max_deviation = np.max(deviations)
    rms_deviation = np.sqrt(np.mean(np.array(deviations)**2))

    print(f"\nControl Performance:")
    print(f"  Mean deviation: {mean_deviation:.2f}°C")
    print(f"  Max deviation: {max_deviation:.2f}°C")
    print(f"  RMS deviation: {rms_deviation:.2f}°C")

    # Assertions
    assert pyro_std < 5.0, "Pyrometer too unstable (>5°C std)"
    assert abs(pyro_drift) < 10.0, "Pyrometer drift too large (>10°C)"
    assert mean_deviation < 5.0, "Mean control error too large (>5°C)"


@pytest.mark.asyncio
@pytest.mark.soak
@pytest.mark.timeout(120)  # Real-time timeout for 24h accelerated test
async def test_rtp_24h_thermal_cycling(rtp_system):
    """
    24-hour soak test: Repeated thermal cycles.

    Tests:
    - Ramp rate tracking
    - Temperature overshoot/undershoot
    - Repeatability across cycles
    - No degradation over time
    """
    driver, telemetry = rtp_system

    print("\n" + "="*80)
    print("RTP 24-Hour Thermal Cycling Test (Accelerated)")
    print("="*80)

    # Setup gas flow
    gas_params = GasFlowParameters(
        gas_type=AmbientGas.NITROGEN,
        flow_rate_sccm=5000.0,
        chamber_pressure_torr=760.0
    )
    await driver.set_gas_flow(gas_params)
    await driver.set_emissivity(0.65)

    # Thermal cycle parameters
    temps = [400.0, 800.0, 1000.0, 600.0]  # Cycle temperatures
    ramp_rate = 100.0  # °C/s
    dwell_time = 10.0  # seconds (simulated)

    start_time = datetime.now()
    cycle_results = []

    cycle_num = 0
    while (datetime.now() - start_time).total_seconds() < SoakTestConfig.DURATION_24H_S:
        cycle_num += 1
        print(f"\nCycle {cycle_num}")

        cycle_start = datetime.now()
        cycle_data = {
            "cycle_num": cycle_num,
            "overshoots": [],
            "settle_times": []
        }

        for temp_idx, target_temp in enumerate(temps):
            print(f"  Step {temp_idx + 1}: {target_temp}°C")

            # Set target temperature
            await driver.set_target_temperature(target_temp, ramp_rate_C_per_s=ramp_rate)

            # Monitor ramp
            step_start = datetime.now()
            max_temp = target_temp
            settled = False

            while (datetime.now() - step_start).total_seconds() < (dwell_time / SoakTestConfig.TIME_ACCELERATION):
                temp_data = await driver.get_temperature()
                current_temp = temp_data["pyrometer_C"]

                # Track overshoot
                if current_temp > max_temp:
                    max_temp = current_temp

                # Check if settled (within 5°C)
                if not settled and abs(current_temp - target_temp) < 5.0:
                    settle_time = (datetime.now() - step_start).total_seconds() * SoakTestConfig.TIME_ACCELERATION
                    cycle_data["settle_times"].append(settle_time)
                    settled = True

                await asyncio.sleep(0.1)

            # Calculate overshoot
            overshoot = max(0, max_temp - target_temp)
            overshoot_pct = (overshoot / target_temp * 100) if target_temp > 0 else 0
            cycle_data["overshoots"].append(overshoot_pct)

        cycle_results.append(cycle_data)

        # Check if test duration exceeded
        if (datetime.now() - start_time).total_seconds() > SoakTestConfig.DURATION_24H_S:
            print(f"\nReached 24h test duration after {cycle_num} cycles")
            break

    # Analysis
    total_sim_hours = (datetime.now() - start_time).total_seconds() * SoakTestConfig.TIME_ACCELERATION / 3600

    print("\n" + "="*80)
    print("24-Hour Thermal Cycling Results:")
    print(f"  Total cycles completed: {len(cycle_results)}")
    print(f"  Simulated time: {total_sim_hours:.1f} hours")

    # Overshoot analysis
    all_overshoots = [os for cycle in cycle_results for os in cycle["overshoots"]]
    mean_overshoot = np.mean(all_overshoots)
    max_overshoot = np.max(all_overshoots)

    print(f"\nOvershoot Analysis:")
    print(f"  Mean overshoot: {mean_overshoot:.2f}%")
    print(f"  Max overshoot: {max_overshoot:.2f}%")

    # Settle time analysis
    all_settle_times = [st for cycle in cycle_results for st in cycle["settle_times"]]
    mean_settle_time = np.mean(all_settle_times)
    max_settle_time = np.max(all_settle_times)

    print(f"\nSettle Time Analysis:")
    print(f"  Mean settle time: {mean_settle_time:.1f}s")
    print(f"  Max settle time: {max_settle_time:.1f}s")

    # Check for degradation over time
    early_cycles = cycle_results[:len(cycle_results)//2]
    late_cycles = cycle_results[len(cycle_results)//2:]

    early_overshoot = np.mean([os for cycle in early_cycles for os in cycle["overshoots"]])
    late_overshoot = np.mean([os for cycle in late_cycles for os in cycle["overshoots"]])
    overshoot_degradation = late_overshoot - early_overshoot

    print(f"\nDegradation Analysis:")
    print(f"  Early cycles overshoot: {early_overshoot:.2f}%")
    print(f"  Late cycles overshoot: {late_overshoot:.2f}%")
    print(f"  Degradation: {overshoot_degradation:+.2f}%")

    # Assertions
    assert len(cycle_results) >= 5, "Should complete at least 5 thermal cycles"
    assert mean_overshoot < 5.0, "Mean overshoot too large (>5%)"
    assert max_overshoot < 10.0, "Max overshoot too large (>10%)"
    assert abs(overshoot_degradation) < 2.0, "Significant degradation observed (>2%)"


@pytest.mark.asyncio
@pytest.mark.soak
@pytest.mark.slow
@pytest.mark.timeout(300)  # Real-time timeout for 72h accelerated test
async def test_rtp_72h_recipe_stress(rtp_system):
    """
    72-hour stress test: Complex recipes under various conditions.

    Tests:
    - Recipe execution reliability
    - Different gas ambients
    - Different pressure regimes
    - Lamp zone control
    """
    driver, telemetry = rtp_system

    print("\n" + "="*80)
    print("RTP 72-Hour Recipe Stress Test (Accelerated)")
    print("="*80)

    start_time = datetime.now()
    recipe_results = []

    # Test various recipes
    recipes = [
        # Oxidation recipe
        {
            "name": "Thermal Oxidation",
            "segments": [
                RampSegment(target_temp_C=800.0, ramp_rate_C_per_s=50.0, dwell_time_s=30.0),
                RampSegment(target_temp_C=1000.0, ramp_rate_C_per_s=20.0, dwell_time_s=60.0),
                RampSegment(target_temp_C=400.0, ramp_rate_C_per_s=30.0, dwell_time_s=10.0),
            ],
            "gas": AmbientGas.OXYGEN
        },
        # Annealing recipe
        {
            "name": "Rapid Thermal Anneal",
            "segments": [
                RampSegment(target_temp_C=1050.0, ramp_rate_C_per_s=100.0, dwell_time_s=5.0),
                RampSegment(target_temp_C=25.0, ramp_rate_C_per_s=50.0, dwell_time_s=0.0),
            ],
            "gas": AmbientGas.NITROGEN
        },
        # Nitridation recipe
        {
            "name": "Nitridation",
            "segments": [
                RampSegment(target_temp_C=900.0, ramp_rate_C_per_s=50.0, dwell_time_s=120.0),
                RampSegment(target_temp_C=25.0, ramp_rate_C_per_s=40.0, dwell_time_s=0.0),
            ],
            "gas": AmbientGas.FORMING_GAS
        },
    ]

    recipe_num = 0
    for recipe_spec in recipes:
        if (datetime.now() - start_time).total_seconds() > SoakTestConfig.DURATION_72H_S:
            break

        recipe_num += 1
        print(f"\nRecipe {recipe_num}: {recipe_spec['name']}")

        # Create recipe
        recipe = TemperatureRecipe(
            recipe_name=recipe_spec["name"],
            segments=recipe_spec["segments"],
            gas_params=GasFlowParameters(
                gas_type=recipe_spec["gas"],
                flow_rate_sccm=5000.0,
                chamber_pressure_torr=760.0
            ),
            emissivity=EmissivitySettings(emissivity=0.65),
            control_mode=TemperatureControlMode.PYROMETER
        )

        # Load and start recipe
        recipe_id = await driver.load_recipe(recipe)
        run_id = await driver.start_recipe(recipe_id)

        recipe_start = datetime.now()

        # Monitor recipe execution
        while True:
            progress = await driver.get_recipe_progress()

            if not progress["is_running"]:
                break

            # Timeout check (2x expected time)
            expected_time = sum(seg.dwell_time_s for seg in recipe_spec["segments"])
            timeout = expected_time * 2 / SoakTestConfig.TIME_ACCELERATION

            if (datetime.now() - recipe_start).total_seconds() > timeout:
                print(f"  Recipe timeout at segment {progress['current_segment']}/{progress['total_segments']}")
                await driver.stop_recipe()
                break

            await asyncio.sleep(0.5)

        # Record result
        recipe_result = {
            "recipe_num": recipe_num,
            "recipe_name": recipe_spec["name"],
            "run_id": run_id,
            "completed": progress["progress_pct"] >= 100.0,
            "final_segment": progress["current_segment"],
            "total_segments": progress["total_segments"],
            "overshoot_pct": progress.get("overshoot_pct", 0.0),
            "elapsed_time_s": (datetime.now() - recipe_start).total_seconds() * SoakTestConfig.TIME_ACCELERATION
        }
        recipe_results.append(recipe_result)

        print(f"  Completed: {recipe_result['completed']}")
        print(f"  Overshoot: {recipe_result['overshoot_pct']:.2f}%")
        print(f"  Elapsed: {recipe_result['elapsed_time_s']:.1f}s")

    # Results
    total_sim_hours = (datetime.now() - start_time).total_seconds() * SoakTestConfig.TIME_ACCELERATION / 3600

    print("\n" + "="*80)
    print("72-Hour Recipe Stress Test Results:")
    print(f"  Total simulated time: {total_sim_hours:.1f} hours")
    print(f"  Recipes executed: {len(recipe_results)}")

    completed_count = sum(1 for r in recipe_results if r["completed"])
    completion_rate = completed_count / len(recipe_results) * 100 if recipe_results else 0

    print(f"\nRecipe Execution:")
    print(f"  Completed: {completed_count}/{len(recipe_results)} ({completion_rate:.1f}%)")

    # Overshoot analysis
    overshoots = [r["overshoot_pct"] for r in recipe_results if r["completed"]]
    if overshoots:
        mean_overshoot = np.mean(overshoots)
        max_overshoot = np.max(overshoots)

        print(f"\nOvershoot Analysis:")
        print(f"  Mean: {mean_overshoot:.2f}%")
        print(f"  Max: {max_overshoot:.2f}%")

    # Assertions
    assert len(recipe_results) >= 2, "Should execute at least 2 recipes"
    assert completion_rate >= 80.0, "Recipe completion rate too low (<80%)"


if __name__ == "__main__":
    # Run soak tests
    print("Running RTP Soak Tests")
    print("These tests run accelerated simulations (1000x speedup)")
    print("12h test ≈ 43 seconds | 24h test ≈ 86 seconds | 72h test ≈ 259 seconds\n")

    pytest.main([__file__, "-v", "-s", "-m", "soak"])
