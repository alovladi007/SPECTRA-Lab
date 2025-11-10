# Process Control Drivers, Simulators & Telemetry

## Overview

This document describes the hardware drivers, Hardware-in-Loop (HIL) simulators, and telemetry streaming infrastructure for Ion Implantation and RTP (Rapid Thermal Processing) systems.

## Architecture

```
services/process_control/
├── app/
│   ├── drivers/              # Hardware control interfaces
│   │   ├── ion_implant_driver.py
│   │   └── rtp_driver.py
│   ├── simulators/           # HIL physics simulations
│   │   ├── ion_implant_hil.py
│   │   └── rtp_hil.py
│   └── telemetry/            # Telemetry streaming
│       ├── ion_implant_telemetry.py
│       └── rtp_telemetry.py
└── tests/
    └── soak_tests/           # Long-duration stability tests
        ├── test_ion_implant_soak.py
        └── test_rtp_soak.py
```

## Ion Implantation Driver

### Features

**Control Interface:**
- Source on/off with ion species selection (B, P, As, Sb, N, O, Ar, Si)
- Extraction voltage (10-50 kV) and acceleration voltage control
- Analyzer magnet for mass selection
- Beam steering (X/Y offset control)
- Scan pattern configuration (raster, spiral, serpentine, static)
- Wafer tilt (0-7°) and rotation control
- Dose integrator with real-time monitoring

**Telemetry Streaming (configurable Hz):**
- Beam current (mA)
- Chamber pressure (source, analyzer, process, beamline in mTorr)
- Accelerator voltage (kV)
- Analyzer magnet field (T)
- Steering values (X, Y in mm)
- Dose count (C/cm²)
- Beam profile (2D map)
- Ion species and source parameters
- Interlock status
- Projected range and straggle

### HIL Simulator - SRIM-like Physics

**Depth Profile Calculation:**
- Lindhard-Scharff-Schiott (LSS) theory for ion stopping
- Projected range (Rp) calculation based on ion mass and energy
- Range straggle (ΔRp) with realistic 20-40% of Rp
- Lateral straggle perpendicular to beam
- Channeling effects for low tilt angles (<2°)
- Pearson-IV profile with asymmetric Gaussian + exponential tail

**Dose Integration Noise:**
- Shot noise (Poisson statistics: σ = sqrt(N))
- Integrator drift (0.1-0.5% per minute)
- Charge collection efficiency variation (±0.5%)
- Deterministic seed for reproducibility

**Scan Uniformity:**
- Radial non-uniformity with edge rolloff (2-5%)
- Corner bias from ion optics aberrations
- Scan pattern artifacts (raster banding, spiral variations)
- Beam divergence effects
- Random jitter (0.5% RMS)

**Beam Jitter & Drift:**
- High-frequency jitter from power supply ripple (50 μm amplitude)
- Low-frequency drift from thermal effects (0.1 mm/hour)

### Usage Example

```python
from app.simulators.ion_implant_hil import IonImplantHILDriver
from app.drivers.ion_implant_driver import *
from app.telemetry.ion_implant_telemetry import IonImplantTelemetryManager

# Create HIL driver with deterministic seed
driver = IonImplantHILDriver(
    equipment_id="ION-01",
    random_seed=42,
    wafer_diameter_mm=300.0
)

# Connect and initialize
await driver.connect()
await driver.initialize()

# Configure source
source_params = SourceParameters(
    source_type=IonSource.BERNAS,
    ion_species=IonSpecies.BORON,
    extraction_voltage_kV=30.0,
    arc_voltage_V=120.0,
    arc_current_A=10.0,
    gas_flow_sccm=2.0
)
await driver.source_on(source_params)

# Configure beam
beam_params = BeamParameters(
    analyzer_magnet_field_T=0.5,
    acceleration_voltage_kV=100.0,
    focus_voltage_kV=10.0
)
await driver.set_beam_parameters(beam_params)

# Configure scan
scan_params = ScanParameters(
    pattern=ScanPattern.RASTER,
    x_amplitude_mm=150.0,
    y_amplitude_mm=150.0,
    x_frequency_Hz=100.0,
    y_frequency_Hz=1.0,
    scan_speed_mm_s=1000.0
)
await driver.set_scan_pattern(scan_params)

# Set wafer position (7° tilt to avoid channeling)
wafer_params = WaferParameters(
    tilt_angle_deg=7.0,
    rotation_angle_deg=0.0,
    rotation_speed_rpm=10.0
)
await driver.set_wafer_position(wafer_params)

# Start implant
dose_params = DoseParameters(
    target_dose_cm2=1e15,
    beam_current_mA=5.0,
    wafer_area_cm2=707.0  # 300mm wafer
)
run_id = await driver.start_implant(dose_params)

# Setup telemetry streaming
telemetry = IonImplantTelemetryManager(driver, sample_rate_hz=10.0)
await telemetry.start_streaming()

# Stream telemetry
async for frame in telemetry.stream_telemetry():
    print(f"Dose: {frame.current_dose_cm2:.2e} ions/cm² ({frame.percent_complete:.1f}%)")
    print(f"Beam current: {frame.beam_current_mA:.2f} mA")
    print(f"Projected range: {frame.projected_range_nm:.1f} nm")

    if frame.percent_complete >= 100.0:
        break

await telemetry.stop_streaming()
await driver.stop_implant()
```

---

## RTP (Rapid Thermal Processing) Driver

### Features

**Control Interface:**
- Temperature ramp segments with configurable ramp rates
- Dwell time control at target temperatures
- Ambient gas selection (N₂, Ar, O₂, forming gas, vacuum)
- Chamber pressure control (vacuum to atmospheric)
- Multi-zone lamp power control (0-100% per zone)
- Emissivity settings for pyrometer correction
- Recipe execution with automatic segment progression

**Telemetry Streaming (configurable Hz):**
- Pyrometer temperature with emissivity error
- Thermocouple temperature
- Lamp power per zone (%)
- Chamber pressure (Torr)
- Gas flows (sccm)
- Computed setpoint deviations
- Recipe progress and segment info
- Temperature overshoot tracking

### HIL Simulator - Thermal Plant Model

**Physics-Based Thermal Model:**
- Multi-zone lamp control (typically 4-6 zones)
- Zone-specific time constants (2-3.5 seconds)
- Radiative cooling (Stefan-Boltzmann: ε·σ·T⁴)
- Convective cooling (gas-dependent)
- Thermal coupling between zones (10% cross-coupling)
- Wafer thermal mass (Si: 700 J/kg·K at 1000°C)

**Sensor Simulation:**
- Pyrometer: 100ms response time, emissivity-dependent
- Thermocouple: 500ms response time, direct contact
- Measurement noise (±2°C pyrometer, ±1°C thermocouple)
- Sensor lag modeled as first-order response

**Emissivity Drift:**
- Configurable drift rate per second
- Simulates oxide growth changing surface emissivity
- Affects pyrometer accuracy

**Ambient/Gas Effects:**
- Gas-dependent convective heat transfer coefficients
  - N₂: 1.0× baseline
  - Ar: 0.7× (lower thermal conductivity)
  - O₂: 1.1×
  - Forming gas (N₂/H₂): 1.3× (H₂ increases conductivity)
  - Vacuum: 0.01× (minimal convection)
- Pressure scaling (linear with pressure)
- Flow-rate-dependent forced convection

**Overshoot/Undershoot Behavior:**
- PID controller with tunable gains (Kp, Ki, Kd)
- Thermal inertia causes overshoot on fast ramps
- Automatic overshoot tracking and reporting

**Actuator Saturation:**
- Lamp power limited to 0-100%
- Saturation flag when power demand exceeds limits
- Realistic power-to-heat conversion

### Usage Example

```python
from app.simulators.rtp_hil import RTPHILDriver
from app.drivers.rtp_driver import *
from app.telemetry.rtp_telemetry import RTPTelemetryManager

# Create HIL driver
driver = RTPHILDriver(
    equipment_id="RTP-01",
    num_zones=4,
    random_seed=42,
    simulation_timestep_s=0.1
)

await driver.connect()
await driver.initialize()

# Create temperature recipe
recipe = TemperatureRecipe(
    recipe_name="Rapid Thermal Oxidation",
    segments=[
        RampSegment(target_temp_C=800.0, ramp_rate_C_per_s=50.0, dwell_time_s=30.0),
        RampSegment(target_temp_C=1000.0, ramp_rate_C_per_s=20.0, dwell_time_s=60.0),
        RampSegment(target_temp_C=400.0, ramp_rate_C_per_s=30.0, dwell_time_s=10.0),
    ],
    gas_params=GasFlowParameters(
        gas_type=AmbientGas.OXYGEN,
        flow_rate_sccm=5000.0,
        chamber_pressure_torr=760.0
    ),
    emissivity=EmissivitySettings(emissivity=0.65),
    control_mode=TemperatureControlMode.PYROMETER
)

# Load and start recipe
recipe_id = await driver.load_recipe(recipe)
run_id = await driver.start_recipe(recipe_id)

# Setup telemetry
telemetry = RTPTelemetryManager(driver, sample_rate_hz=10.0)
await telemetry.start_streaming()

# Monitor execution
async for frame in telemetry.stream_telemetry():
    print(f"Temp: {frame.pyrometer_temp_C:.1f}°C / {frame.setpoint_temp_C:.1f}°C")
    print(f"Segment: {frame.current_segment}/{frame.total_segments}")
    print(f"Overshoot: {frame.overshoot_pct:.2f}%")

    if not frame.recipe_progress_pct or frame.recipe_progress_pct >= 100.0:
        break

await telemetry.stop_streaming()
```

---

## Soak Tests with Accelerated Time

### Purpose

Long-duration stability testing (12-72 hours) at 1000× time acceleration to validate:
- System stability over extended periods
- Repeatability across multiple runs
- Recovery from fault conditions
- Memory leak detection
- Physics model accuracy

### Time Acceleration

Real-time execution | Simulated duration | Test type
--------------------|-------------------|----------
43 seconds          | 12 hours          | Continuous operation
86 seconds          | 24 hours          | Multiple cycles
259 seconds         | 72 hours          | Stress conditions

### Ion Implant Soak Tests

**12-Hour Stability Test:**
- Continuous implantation with monitoring
- Beam current stability (should be <15% RMS variation)
- Vacuum stability
- Dose integration accuracy
- No memory leaks

**24-Hour Multiple Wafers:**
- 20+ wafer processing cycles
- Dose reproducibility (mean error <5%, std <3%)
- Start/stop reliability
- System reset between wafers

**72-Hour Stress Test:**
- Multiple ion species (B, P, As)
- Energy range 1-100 keV
- Recovery from beam loss
- Long-term profile stability

### RTP Soak Tests

**12-Hour Thermal Stability:**
- Continuous 1000°C hold
- Temperature stability (<5°C std)
- Sensor drift (<10°C over 12h)
- Control deviation (<5°C mean)

**24-Hour Thermal Cycling:**
- 5+ complete thermal cycles (400-1000°C)
- Overshoot <5% mean, <10% max
- No performance degradation over time
- Repeatability across cycles

**72-Hour Recipe Stress:**
- Multiple complex recipes
- Different gas ambients (N₂, O₂, forming gas)
- Recipe completion rate >80%
- Overshoot consistency

### Running Soak Tests

```bash
# Run all Ion Implant soak tests
cd services/process_control
pytest tests/soak_tests/test_ion_implant_soak.py -v -s -m soak

# Run all RTP soak tests
pytest tests/soak_tests/test_rtp_soak.py -v -s -m soak

# Run only 12h tests (faster)
pytest tests/soak_tests/ -v -s -m soak -k "12h"

# Run 72h stress tests (slower)
pytest tests/soak_tests/ -v -s -m "soak and slow"
```

---

## Integration with Safety System

All drivers integrate with the safety & calibration system:

1. **Pre-run Calibration Checks:**
   - Ion Implant: Beam current integrator, analyzer magnet probe, vacuum gauge
   - RTP: Pyrometer, thermocouple, pressure gauge, mass flow controllers

2. **Safety Approval Requirements:**
   - Both Ion Implant and RTP are HIGH hazard level
   - Require dual approval before starting
   - Interlocks checked before each run

3. **Audit Trail:**
   - All runs logged with e-signatures
   - Telemetry data can be archived for compliance
   - Calibration certificates linked to run data

### Example with Safety Guards

```python
from app.safety import check_calibration_status, check_safety_approval

# Check calibrations before starting
instrument_ids = [
    uuid.UUID("..."),  # Beam current integrator
    uuid.UUID("..."),  # Analyzer magnet probe
    uuid.UUID("..."),  # Vacuum gauge
]

try:
    calibrations = await check_calibration_status(instrument_ids)
    print("All calibrations valid")
except HTTPException as e:
    print(f"Calibration error: {e.detail['message']}")
    # Block run
    return

# Check safety approval
try:
    approval = await check_safety_approval(
        run_id=uuid.UUID("..."),
        process_type="ion_implant",
        hazard_level=HazardLevel.HIGH
    )
    print("Safety approval granted")
except HTTPException as e:
    print(f"Approval error: {e.detail['message']}")
    # Block run
    return

# Proceed with run...
```

---

## Telemetry Export

Both telemetry systems support exporting data to JSON for analysis:

```python
from app.telemetry.ion_implant_telemetry import IonImplantTelemetryRecorder

# Record telemetry for a run
recorder = IonImplantTelemetryRecorder(telemetry_manager)

# Record for 60 seconds
frames = await recorder.record(duration_s=60.0)

# Export to JSON
await recorder.export_to_json_file("/path/to/telemetry_data.json")
```

JSON format:
```json
{
  "metadata": {
    "recording_start": "2025-01-09T10:00:00.000",
    "recording_end": "2025-01-09T10:01:00.000",
    "num_frames": 600,
    "sample_rate_hz": 10.0
  },
  "frames": [
    {
      "timestamp": "2025-01-09T10:00:00.000",
      "run_id": "RUN-20250109-100000",
      "status": "running",
      "beam_current_mA": 5.02,
      "current_dose_cm2": 1.23e14,
      ...
    },
    ...
  ]
}
```

---

## Performance Characteristics

### Ion Implant HIL
- Depth profile calculation: <10ms per calculation
- Uniformity map generation: <100ms for 50×50 grid
- Telemetry frame collection: <5ms
- Typical run time: 10-30 minutes for 1e15 ions/cm² at 5 mA

### RTP HIL
- Thermal simulation timestep: 0.1 seconds
- Update rate: 10 Hz (adjustable)
- Temperature stabilization: 30-60 seconds (depends on setpoint)
- Telemetry frame collection: <5ms
- Typical recipe time: 5-15 minutes

---

## Future Enhancements

1. **Real Hardware Integration:**
   - Replace mock drivers with actual SECS/GEM communication
   - Interface with equipment PLCs
   - Real-time hardware interlocks

2. **Advanced Physics:**
   - Monte Carlo ion trajectory simulation
   - Full SRIM integration
   - Non-uniform wafer temperature (RTP)
   - Multi-wafer batch processing

3. **Machine Learning:**
   - Predictive maintenance from telemetry
   - Automatic recipe optimization
   - Anomaly detection in real-time

4. **Database Integration:**
   - Store telemetry in time-series database
   - Historical trend analysis
   - Correlation with metrology results

5. **Visualization:**
   - Real-time 2D/3D telemetry dashboards
   - Depth profile visualization
   - Temperature map overlays

---

## References

- LSS Theory: J. Lindhard, M. Scharff, H. Schiott, "Range Concepts and Heavy Ion Ranges", Mat. Fys. Medd. 33 (1963)
- SRIM: J.F. Ziegler, J.P. Biersack, "The Stopping and Range of Ions in Matter" (2008)
- RTP: A. Kermani et al., "Rapid Thermal Processing: A Justifiable Technology", Microelectronics Journal (2000)
- Silicon Properties: CRC Handbook of Chemistry and Physics
