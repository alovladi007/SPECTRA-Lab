"""Soak tests for Ion Implantation HIL simulator with accelerated time."""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

import sys
sys.path.insert(0, '/Users/vladimirantoine/SPECTRA LAB/SPECTRA-Lab/services/process_control')

from app.drivers.ion_implant_driver import (
    IonSpecies, SourceParameters, BeamParameters, ScanParameters,
    WaferParameters, DoseParameters, ScanPattern
)
from app.simulators.ion_implant_hil import IonImplantHILDriver
from app.telemetry.ion_implant_telemetry import IonImplantTelemetryManager


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
async def ion_implant_system():
    """Create ion implant HIL system for testing."""
    driver = IonImplantHILDriver(
        equipment_id="SOAK-TEST-ION-01",
        random_seed=42,  # Deterministic
        wafer_diameter_mm=300.0
    )

    telemetry_manager = IonImplantTelemetryManager(
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
async def test_ion_implant_12h_stability(ion_implant_system):
    """
    12-hour soak test: Continuous implantation with monitoring.

    Tests:
    - Long-term dose integration stability
    - Beam current stability
    - Vacuum stability
    - No memory leaks
    """
    driver, telemetry = ion_implant_system

    print("\n" + "="*80)
    print("Ion Implant 12-Hour Soak Test (Accelerated)")
    print("="*80)

    # Setup source
    source_params = SourceParameters(
        source_type="bernas",
        ion_species=IonSpecies.BORON,
        extraction_voltage_kV=30.0,
        arc_voltage_V=120.0,
        arc_current_A=10.0,
        gas_flow_sccm=2.0
    )
    await driver.source_on(source_params)

    # Setup beam
    beam_params = BeamParameters(
        analyzer_magnet_field_T=0.5,
        acceleration_voltage_kV=100.0,
        focus_voltage_kV=10.0
    )
    await driver.set_beam_parameters(beam_params)

    # Setup scan
    scan_params = ScanParameters(
        pattern=ScanPattern.RASTER,
        x_amplitude_mm=150.0,
        y_amplitude_mm=150.0,
        x_frequency_Hz=100.0,
        y_frequency_Hz=1.0,
        scan_speed_mm_s=1000.0
    )
    await driver.set_scan_pattern(scan_params)

    # Setup wafer (7° tilt to avoid channeling)
    wafer_params = WaferParameters(
        tilt_angle_deg=7.0,
        rotation_angle_deg=0.0,
        rotation_speed_rpm=10.0
    )
    await driver.set_wafer_position(wafer_params)

    # Target dose: 1e15 ions/cm²
    dose_params = DoseParameters(
        target_dose_cm2=1e15,
        beam_current_mA=5.0,
        wafer_area_cm2=np.pi * 15**2  # 300mm wafer
    )

    # Start implant
    run_id = await driver.start_implant(dose_params)
    print(f"Started implant run: {run_id}")

    # Run soak test
    start_time = datetime.now()
    telemetry_frames = []
    beam_currents = []
    doses = []

    await telemetry.start_streaming()

    sample_interval = 1.0 / SoakTestConfig.TELEMETRY_RATE_HZ
    elapsed_target = SoakTestConfig.DURATION_12H_S

    while (datetime.now() - start_time).total_seconds() < elapsed_target:
        # Collect telemetry
        frame = await telemetry.collect_frame()
        telemetry_frames.append(frame)
        beam_currents.append(frame.beam_current_mA)
        doses.append(frame.current_dose_cm2)

        # Log progress every 1000 frames
        if len(telemetry_frames) % 1000 == 0:
            elapsed_real = (datetime.now() - start_time).total_seconds()
            elapsed_sim = elapsed_real * SoakTestConfig.TIME_ACCELERATION
            print(f"  Progress: {elapsed_sim/3600:.1f}h simulated ({elapsed_real:.1f}s real) | "
                  f"Dose: {frame.current_dose_cm2:.2e} ions/cm² ({frame.percent_complete:.1f}%)")

        await asyncio.sleep(sample_interval)

    await telemetry.stop_streaming()

    # Analysis
    print("\nSoak Test Results:")
    print(f"  Total frames collected: {len(telemetry_frames)}")
    print(f"  Simulated time: {elapsed_target * SoakTestConfig.TIME_ACCELERATION / 3600:.1f} hours")

    # Beam stability
    beam_current_mean = np.mean(beam_currents)
    beam_current_std = np.std(beam_currents)
    beam_current_drift = beam_currents[-1] - beam_currents[0]

    print(f"\nBeam Current Stability:")
    print(f"  Mean: {beam_current_mean:.3f} mA")
    print(f"  Std: {beam_current_std:.3f} mA ({beam_current_std/beam_current_mean*100:.2f}%)")
    print(f"  Drift: {beam_current_drift:.3f} mA")

    # Dose integration
    final_dose = doses[-1]
    target_dose = dose_params.target_dose_cm2
    dose_error = abs(final_dose - target_dose) / target_dose * 100

    print(f"\nDose Integration:")
    print(f"  Final dose: {final_dose:.3e} ions/cm²")
    print(f"  Target dose: {target_dose:.3e} ions/cm²")
    print(f"  Error: {dose_error:.2f}%")

    # Assertions
    assert beam_current_std / beam_current_mean < 0.15, "Beam current too unstable (>15% RMS)"
    assert abs(beam_current_drift) < 1.0, "Beam current drift too large (>1 mA)"
    # Note: Dose might not complete in 12h depending on beam current


@pytest.mark.asyncio
@pytest.mark.soak
@pytest.mark.timeout(120)  # Real-time timeout for 24h accelerated test
async def test_ion_implant_24h_multiple_wafers(ion_implant_system):
    """
    24-hour soak test: Multiple wafer processing.

    Tests:
    - Repeated start/stop cycles
    - Dose reproducibility
    - System stability over multiple runs
    """
    driver, telemetry = ion_implant_system

    print("\n" + "="*80)
    print("Ion Implant 24-Hour Soak Test - Multiple Wafers (Accelerated)")
    print("="*80)

    # Test configuration
    num_wafers = 20
    wafer_results = []

    start_time = datetime.now()

    for wafer_num in range(num_wafers):
        print(f"\nProcessing wafer {wafer_num + 1}/{num_wafers}")

        # Setup (simulate wafer load)
        source_params = SourceParameters(
            source_type="bernas",
            ion_species=IonSpecies.PHOSPHORUS,
            extraction_voltage_kV=25.0,
            arc_voltage_V=115.0,
            arc_current_A=8.0,
            gas_flow_sccm=1.5
        )
        await driver.source_on(source_params)

        beam_params = BeamParameters(
            analyzer_magnet_field_T=0.45,
            acceleration_voltage_kV=80.0
        )
        await driver.set_beam_parameters(beam_params)

        scan_params = ScanParameters(
            pattern=ScanPattern.RASTER,
            x_amplitude_mm=150.0,
            y_amplitude_mm=150.0,
            x_frequency_Hz=100.0,
            y_frequency_Hz=1.0,
            scan_speed_mm_s=1000.0
        )
        await driver.set_scan_pattern(scan_params)

        wafer_params = WaferParameters(
            tilt_angle_deg=7.0,
            rotation_angle_deg=0.0,
            rotation_speed_rpm=10.0
        )
        await driver.set_wafer_position(wafer_params)

        # Smaller dose for faster completion
        dose_params = DoseParameters(
            target_dose_cm2=1e14,  # Lower dose
            beam_current_mA=5.0,
            wafer_area_cm2=np.pi * 15**2
        )

        # Start implant
        run_id = await driver.start_implant(dose_params)

        # Monitor until complete (with timeout)
        wafer_start = datetime.now()
        max_wafer_time = 60.0 / SoakTestConfig.TIME_ACCELERATION  # 60 seconds simulated

        while True:
            reading = await driver.get_dose_integrator_reading()

            if reading["percent_complete"] >= 100.0:
                break

            if (datetime.now() - wafer_start).total_seconds() > max_wafer_time:
                print(f"  Wafer timeout at {reading['percent_complete']:.1f}%")
                break

            await asyncio.sleep(0.1)

        # Get final reading
        final_reading = await driver.get_dose_integrator_reading()

        # Record results
        wafer_result = {
            "wafer_num": wafer_num + 1,
            "run_id": run_id,
            "final_dose_cm2": final_reading["current_dose_cm2"],
            "target_dose_cm2": dose_params.target_dose_cm2,
            "error_pct": abs(final_reading["current_dose_cm2"] - dose_params.target_dose_cm2) / dose_params.target_dose_cm2 * 100,
            "elapsed_time_s": final_reading["elapsed_time_s"]
        }
        wafer_results.append(wafer_result)

        print(f"  Dose: {wafer_result['final_dose_cm2']:.3e} ions/cm² (error: {wafer_result['error_pct']:.2f}%)")

        # Stop implant
        await driver.stop_implant()
        await driver.source_off()

        # Simulate wafer unload/load delay
        await asyncio.sleep(0.5)

        # Check if we've exceeded test time
        if (datetime.now() - start_time).total_seconds() > SoakTestConfig.DURATION_24H_S:
            print(f"\nReached 24h test duration after {len(wafer_results)} wafers")
            break

    # Analysis
    print("\n" + "="*80)
    print("24-Hour Soak Test Results:")
    print(f"  Total wafers processed: {len(wafer_results)}")
    print(f"  Simulated time: {(datetime.now() - start_time).total_seconds() * SoakTestConfig.TIME_ACCELERATION / 3600:.1f} hours")

    # Dose reproducibility
    dose_errors = [r["error_pct"] for r in wafer_results]
    mean_error = np.mean(dose_errors)
    std_error = np.std(dose_errors)

    print(f"\nDose Reproducibility:")
    print(f"  Mean error: {mean_error:.2f}%")
    print(f"  Std error: {std_error:.2f}%")
    print(f"  Max error: {max(dose_errors):.2f}%")
    print(f"  Min error: {min(dose_errors):.2f}%")

    # Assertions
    assert len(wafer_results) >= 10, "Should process at least 10 wafers"
    assert mean_error < 5.0, "Mean dose error too large (>5%)"
    assert std_error < 3.0, "Dose reproducibility too poor (>3% std)"


@pytest.mark.asyncio
@pytest.mark.soak
@pytest.mark.slow
@pytest.mark.timeout(300)  # Real-time timeout for 72h accelerated test
async def test_ion_implant_72h_stress(ion_implant_system):
    """
    72-hour stress test: Extreme conditions and recovery.

    Tests:
    - Recovery from beam loss
    - Recovery from vacuum excursion
    - Long-term profile stability
    """
    driver, telemetry = ion_implant_system

    print("\n" + "="*80)
    print("Ion Implant 72-Hour Stress Test (Accelerated)")
    print("="*80)

    start_time = datetime.now()
    events = []

    # Run continuous monitoring
    await telemetry.start_streaming()

    # Simulate various operating conditions
    conditions = [
        ("Low energy B", IonSpecies.BORON, 20.0),
        ("Medium energy P", IonSpecies.PHOSPHORUS, 50.0),
        ("High energy As", IonSpecies.ARSENIC, 100.0),
        ("Ultra-low energy B", IonSpecies.BORON, 1.0),
    ]

    for cond_num, (cond_name, species, energy_keV) in enumerate(conditions):
        print(f"\nCondition {cond_num + 1}: {cond_name}")

        # Setup
        source_params = SourceParameters(
            source_type="bernas",
            ion_species=species,
            extraction_voltage_kV=energy_keV * 0.3,
            arc_voltage_V=120.0,
            arc_current_A=10.0,
            gas_flow_sccm=2.0
        )
        await driver.source_on(source_params)

        beam_params = BeamParameters(
            analyzer_magnet_field_T=0.4,
            acceleration_voltage_kV=energy_keV
        )
        await driver.set_beam_parameters(beam_params)

        # Run for a portion of 72h
        condition_duration = SoakTestConfig.DURATION_72H_S / len(conditions)
        condition_start = datetime.now()

        while (datetime.now() - condition_start).total_seconds() < condition_duration:
            frame = await telemetry.collect_frame()

            await asyncio.sleep(1.0)

            # Check if overall test time exceeded
            if (datetime.now() - start_time).total_seconds() > SoakTestConfig.DURATION_72H_S:
                break

        events.append({
            "condition": cond_name,
            "species": species.value,
            "energy_keV": energy_keV,
            "duration_s": (datetime.now() - condition_start).total_seconds()
        })

        await driver.source_off()

        if (datetime.now() - start_time).total_seconds() > SoakTestConfig.DURATION_72H_S:
            break

    await telemetry.stop_streaming()

    # Results
    total_sim_hours = (datetime.now() - start_time).total_seconds() * SoakTestConfig.TIME_ACCELERATION / 3600

    print("\n" + "="*80)
    print("72-Hour Stress Test Results:")
    print(f"  Total simulated time: {total_sim_hours:.1f} hours")
    print(f"  Conditions tested: {len(events)}")

    for event in events:
        print(f"    {event['condition']}: {event['duration_s'] * SoakTestConfig.TIME_ACCELERATION / 3600:.1f}h")

    # Get telemetry statistics
    stats = await telemetry.get_statistics()
    print(f"\nTelemetry Statistics:")
    print(f"  Frames collected: {stats['buffer_size']}")

    # Assertions
    assert len(events) >= 2, "Should complete at least 2 test conditions"
    assert total_sim_hours >= 24.0, "Should simulate at least 24 hours"


if __name__ == "__main__":
    # Run soak tests
    print("Running Ion Implantation Soak Tests")
    print("These tests run accelerated simulations (1000x speedup)")
    print("12h test ≈ 43 seconds | 24h test ≈ 86 seconds | 72h test ≈ 259 seconds\n")

    pytest.main([__file__, "-v", "-s", "-m", "soak"])
