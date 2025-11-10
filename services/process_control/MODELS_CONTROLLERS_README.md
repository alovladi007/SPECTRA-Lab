# Process Control Models & Controllers

## Overview

This document describes the modeling, analysis, and control algorithms for **Ion Implantation** and **RTP (Rapid Thermal Processing)** systems. These modules provide physics-based simulation, real-time control, and post-run analysis capabilities.

## Architecture

```
services/process_control/
├── app/
│   ├── models/              # Physics models
│   │   ├── ion_range.py     # SRIM-like ion range estimator
│   │   └── rtp_thermal.py   # Thermal plant dynamics
│   ├── controllers/         # Control algorithms
│   │   ├── ion.py           # Dose integrator, R2R, FDC
│   │   └── rtp.py           # PID, MPC, R2R, thermal budget
└── tests/
    └── unit/               # Unit tests
        ├── test_ion_range.py
        ├── test_rtp_thermal.py
        ├── test_ion_controllers.py
        └── test_rtp_controllers.py
```

---

## Ion Implantation Models & Controllers

### Pre-Run Models

#### 1. SRIM-like Range Estimator

Physics-based ion range calculation using Lindhard-Scharff-Schiott (LSS) theory.

**Features:**
- Projected range (Rp) and straggle (ΔRp) calculation
- Energy-dependent stopping power
- Mass-dependent range scaling
- Tilt angle corrections
- Depth profile generation (Pearson-IV with channeling tail)

**Usage:**

```python
from app.models.ion_range import SRIMEstimator, IonSpecies

estimator = SRIMEstimator()

# Estimate range
range_params = estimator.estimate_range(
    ion_species=IonSpecies.BORON,
    energy_keV=10.0,
    tilt_angle_deg=7.0
)

print(f"Projected range: {range_params.projected_range_nm:.1f} nm")
print(f"Straggle: {range_params.range_straggle_nm:.1f} nm")

# Generate depth profile
profile = estimator.predict_depth_profile(
    ion_species=IonSpecies.BORON,
    energy_keV=10.0,
    dose_cm2=1e15,
    tilt_angle_deg=7.0
)

import matplotlib.pyplot as plt
plt.plot(profile.depth_nm, profile.concentration_cm3)
plt.xlabel("Depth (nm)")
plt.ylabel("Concentration (cm⁻³)")
plt.yscale("log")
plt.show()
```

#### 2. Channeling Risk Predictor

Assesses channeling probability based on crystal orientation and ion energy.

**Features:**
- Lindhard model: P_channeling ≈ (ψ_c / ψ)²
- Critical angle calculation (energy-dependent)
- Twist angle effect
- Risk classification (LOW, MEDIUM, HIGH)

**Usage:**

```python
from app.models.ion_range import ChannelingRiskPredictor

predictor = ChannelingRiskPredictor()

risk = predictor.assess_channeling_risk(
    ion_species=IonSpecies.BORON,
    energy_keV=10.0,
    tilt_angle_deg=0.5,  # Low tilt!
    twist_angle_deg=0.0
)

print(f"Risk level: {risk.risk_level}")
print(f"Channeling probability: {risk.channeling_pct:.1f}%")
print(f"Recommendations: {risk.recommendations}")
```

#### 3. Sheet Resistance Estimator

Estimates post-anneal sheet resistance from dose, energy, and RTP profile.

**Features:**
- Activation fraction vs anneal temperature/time
- Solid solubility limits
- Caughey-Thomas mobility model (concentration-dependent)
- Sheet resistance: Rs = 1 / (q · ∫ n(x) · μ(x) dx)

**Usage:**

```python
from app.models.ion_range import SheetResistanceEstimator

estimator = SheetResistanceEstimator()

result = estimator.estimate_sheet_resistance(
    ion_species=IonSpecies.BORON,
    energy_keV=10.0,
    dose_cm2=1e15,
    anneal_temp_C=1000.0,
    anneal_time_s=30.0
)

print(f"Sheet resistance: {result.sheet_resistance_ohm_per_sq:.1f} Ω/sq")
print(f"Activation fraction: {result.activation_fraction:.1%}")
print(f"Junction depth: {result.junction_depth_nm:.1f} nm")
```

### In-Run Controllers

#### 1. Dose Integrator

Real-time dose integration: **Q = ∫I(t)dt / A**

**Features:**
- Beam current integration with charge state correction
- Wafer area normalization
- Scan pattern correction factor
- Area correction for edge exclusion

**Usage:**

```python
from app.controllers.ion import DoseIntegrator

integrator = DoseIntegrator(
    wafer_area_cm2=707.0,  # 300mm wafer
    charge_state=1
)

# During implant (called at each telemetry update)
for t in range(1000):
    integrator.integrate(
        beam_current_mA=5.0,
        timestamp=t * 0.1,
        dt=0.1
    )

    current_dose = integrator.get_dose()
    percent_complete = (current_dose / 1e15) * 100

    if percent_complete >= 100.0:
        break

final_result = integrator.get_result()
print(f"Final dose: {final_result.final_dose_cm2:.2e} ions/cm²")
```

#### 2. Scan Uniformity Controller

Analyzes 2D dose map and calculates steering/amplitude corrections.

**Features:**
- Uniformity metric: (1 - 3σ/mean) × 100%
- Centroid calculation for steering correction
- Edge rolloff detection
- Auto-trim recommendations

**Usage:**

```python
from app.controllers.ion import ScanUniformityController
import numpy as np

controller = ScanUniformityController()

# Analyze dose map
uniformity = controller.analyze_uniformity(
    dose_map=measured_dose_2d,
    x_positions_mm=x_grid,
    y_positions_mm=y_grid
)

print(f"Uniformity: {uniformity.uniformity_pct:.1f}%")
print(f"Edge rolloff: {uniformity.edge_rolloff_pct:.1f}%")

# Calculate corrections
corrections = controller.calculate_corrections(uniformity)

print(f"Steering X: {corrections.steering_x_mm:.2f} mm")
print(f"Steering Y: {corrections.steering_y_mm:.2f} mm")
print(f"Amplitude correction: {corrections.amplitude_correction:.3f}")
```

#### 3. Run-to-Run (R2R) Controller

Wafer-to-wafer adjustments using EWMA (Exponentially Weighted Moving Average).

**Features:**
- Dose correction based on measurement error
- Tilt/twist adjustments for uniformity
- EWMA smoothing (α = 0.3)

**Usage:**

```python
from app.controllers.ion import R2RController

controller = R2RController(alpha=0.3)

# After each wafer
controller.update(
    measured_dose_cm2=0.95e15,
    target_dose_cm2=1.0e15,
    measured_uniformity_pct=92.0
)

recommendation = controller.get_recommendation()

print(f"Adjust dose by: {recommendation.recommended_dose_adjustment:.3f}×")
print(f"Recommended tilt: {recommendation.recommended_tilt_deg:.1f}°")
```

#### 4. Beam Drift Detector (FDC)

Fault Detection and Classification for beam position drift.

**Features:**
- Linear regression for drift rate
- Spike detection (sudden jumps)
- Compensation recommendations
- Pause/compensate actions

**Usage:**

```python
from app.controllers.ion import BeamDriftDetector

detector = BeamDriftDetector(
    drift_threshold_mm=1.0,
    spike_threshold_mm=2.0
)

# Real-time monitoring
for measurement in telemetry_stream:
    result = detector.update(
        x_position_mm=measurement.beam_x,
        y_position_mm=measurement.beam_y,
        timestamp=measurement.timestamp
    )

    if result.spike_detected:
        print("ALERT: Beam spike detected! Pausing...")
        driver.pause_implant()
        break

    if result.drift_detected:
        print(f"Drift detected: {result.drift_magnitude_mm:.2f} mm")
        print(f"Compensating steering...")
        driver.adjust_steering(
            result.recommended_steering_x_mm,
            result.recommended_steering_y_mm
        )
```

---

## RTP Models & Controllers

### Thermal Plant Model

Physics-based multi-zone thermal dynamics.

**Features:**
- Multi-zone lamp control (4-6 zones)
- Radiative cooling: ε·σ·A·(T⁴ - T_amb⁴)
- Convective cooling: h·A·(T - T_amb)
- Zone coupling (10% between adjacent zones)
- Temperature-dependent silicon properties
- Emissivity drift modeling
- Sensor lag (pyrometer: 100ms, thermocouple: 500ms)

**Usage:**

```python
from app.models.rtp_thermal import RTPThermalPlant
import numpy as np

plant = RTPThermalPlant(
    num_zones=4,
    max_lamp_power_W=10000.0
)

# Simulation loop
lamp_powers = np.array([80.0, 90.0, 90.0, 80.0])  # % per zone

for _ in range(100):
    state = plant.update(
        dt=0.1,
        lamp_powers_pct=lamp_powers,
        gas_flow_sccm=5000.0,
        chamber_pressure_Pa=101325.0
    )

    print(f"Wafer temp: {state.wafer_temperature_C:.1f}°C")
    print(f"Pyrometer: {state.pyrometer_reading_C:.1f}°C")
```

### Controllers

#### 1. PID Controller with Anti-Windup

Standard PID control with integral windup prevention and feed-forward.

**Features:**
- Proportional-Integral-Derivative control
- Anti-windup (stops integrating when saturated)
- Feed-forward (anticipates setpoint changes)
- Configurable gains (Kp, Ki, Kd)

**Usage:**

```python
from app.controllers.rtp import PIDController, PIDGains

gains = PIDGains(
    Kp=2.0,
    Ki=0.5,
    Kd=0.1,
    windup_limit=100.0,
    enable_feedforward=True,
    feedforward_gain=1.0
)

controller = PIDController(gains)

# Control loop
setpoint = 1000.0  # °C

for _ in range(1000):
    measured = plant.state.wafer_temperature_C

    lamp_power = controller.update(
        setpoint=setpoint,
        measured=measured,
        dt=0.1,
        output_limits=(0.0, 100.0)
    )

    plant.update(dt=0.1, lamp_powers_pct=np.ones(4) * lamp_power)
```

#### 2. Model Predictive Control (MPC)

Constraint-based optimal control for overshoot minimization.

**Features:**
- Prediction horizon (typically 20 steps)
- Control horizon (typically 10 steps)
- Overshoot constraints
- Lamp power rate limits
- Actuator saturation handling

**Usage:**

```python
from app.controllers.rtp import MPCController, MPCParameters

params = MPCParameters(
    prediction_horizon=20,
    control_horizon=10,
    max_overshoot_C=5.0,
    max_lamp_rate_pct_per_s=50.0
)

controller = MPCController(params, num_zones=4)

# Future setpoint trajectory
future_setpoints = [1000.0] * 20  # Hold at 1000°C

lamp_powers = controller.update(
    current_temp=plant.state.wafer_temperature_C,
    current_lamp_power=plant.state.lamp_powers_pct,
    setpoint=1000.0,
    future_setpoints=future_setpoints,
    dt=0.1
)

plant.update(dt=0.1, lamp_powers_pct=lamp_powers)
```

#### 3. Run-to-Run (R2R) Controller

Wafer-to-wafer adjustments for emissivity drift and lamp trims.

**Features:**
- Emissivity drift tracking
- Lamp power trim per zone
- Overshoot prediction (EWMA)

**Usage:**

```python
from app.controllers.rtp import R2RController

controller = R2RController(num_zones=4, alpha=0.3)

# After each wafer
controller.update(
    measured_emissivity=0.68,  # Drifted from 0.65
    lamp_powers_used=np.array([60.0, 55.0, 55.0, 58.0]),
    overshoot_observed=3.5
)

adjustments = controller.get_adjustments()

print(f"Emissivity correction: {adjustments['emissivity_correction']:.3f}")
print(f"Lamp power trim: {adjustments['lamp_power_trim']}")
print(f"Predicted overshoot: {adjustments['predicted_overshoot_pct']:.1f}%")
```

#### 4. Thermal Budget Calculator

Calculates activation dose: **∫ exp(-Ea/kT(t)) dt**

**Features:**
- Dopant-specific activation energies
- Real-time integration
- Equivalent time at 1000°C calculation

**Usage:**

```python
from app.controllers.rtp import ThermalBudgetCalculator

calculator = ThermalBudgetCalculator(dopant_species="boron")

# During RTP run
for temp_sample in rtp_profile:
    calculator.add_sample(
        temperature_C=temp_sample.temperature,
        dt=temp_sample.dt
    )

budget = calculator.get_budget()

print(f"Integrated budget: {budget.integrated_budget:.2e}")
print(f"Equivalent time at 1000°C: {budget.equivalent_time_at_1000C_s:.1f} s")
print(f"Peak activation rate: {budget.peak_activation_rate:.2e}")
```

#### 5. Performance Analyzer

Post-run analysis and tuning recommendations.

**Features:**
- Ramp fidelity (overshoot, undershoot, RMSE)
- Settling time calculation
- Dwell stability (std dev, drift)
- Automatic tuning recommendations

**Usage:**

```python
from app.controllers.rtp import PerformanceAnalyzer

# Analyze segment
fidelity = PerformanceAnalyzer.analyze_ramp_segment(
    segment_id=1,
    target_ramp_rate_C_per_s=50.0,
    setpoint_history=setpoint_array,
    measured_history=measured_array,
    time_history=time_array
)

print(f"RMSE: {fidelity.rmse_C:.2f}°C")
print(f"Peak overshoot: {fidelity.peak_overshoot_pct:.1f}%")
print(f"Settling time: {fidelity.settling_time_s:.1f} s")

# Generate full report
report = PerformanceAnalyzer.generate_performance_report(
    recipe_id="REC-001",
    run_id="RUN-001",
    ramp_fidelities=[fidelity],
    thermal_budget=budget,
    saturation_events=0,
    constraint_violations=0,
    start_time=0.0,
    end_time=300.0
)

print(f"Overall RMSE: {report.overall_rmse_C:.2f}°C")
print(f"Recommended tuning: {report.recommended_tuning}")
print(f"Drift flags: {report.drift_flags}")
```

---

## Integration Example: Complete Ion Implant Run

```python
from app.models.ion_range import SRIMEstimator, ChannelingRiskPredictor
from app.controllers.ion import DoseIntegrator, BeamDriftDetector
from app.simulators.ion_implant_hil import IonImplantHILDriver
from app.drivers.ion_implant_driver import *

# Pre-run analysis
estimator = SRIMEstimator()
range_params = estimator.estimate_range(IonSpecies.BORON, 10.0, 7.0)

channeling_risk = ChannelingRiskPredictor().assess_channeling_risk(
    IonSpecies.BORON, 10.0, 7.0, 0.0
)

print(f"Projected range: {range_params.projected_range_nm:.1f} nm")
print(f"Channeling risk: {channeling_risk.risk_level}")

# Setup driver
driver = IonImplantHILDriver(equipment_id="ION-01")
await driver.connect()
await driver.initialize()

# Configure source and beam
source_params = SourceParameters(
    source_type=IonSource.BERNAS,
    ion_species=IonSpecies.BORON,
    extraction_voltage_kV=30.0,
    arc_voltage_V=120.0,
    arc_current_A=10.0,
    gas_flow_sccm=2.0
)
await driver.source_on(source_params)

# Setup controllers
dose_integrator = DoseIntegrator(wafer_area_cm2=707.0)
drift_detector = BeamDriftDetector()

# Start implant
run_id = await driver.start_implant(DoseParameters(
    target_dose_cm2=1e15,
    beam_current_mA=5.0,
    wafer_area_cm2=707.0
))

# Monitor and control
async for frame in telemetry.stream_telemetry():
    # Dose integration
    dose_integrator.integrate(
        frame.beam_current_mA,
        frame.timestamp,
        frame.dt
    )

    # Drift detection
    drift_result = drift_detector.update(
        frame.steering_x_mm,
        frame.steering_y_mm,
        frame.timestamp
    )

    if drift_result.spike_detected:
        await driver.stop_implant()
        break

    if dose_integrator.get_dose() >= 1e15:
        await driver.stop_implant()
        break

print("Implant complete!")
```

---

## Integration Example: Complete RTP Run

```python
from app.models.rtp_thermal import RTPThermalPlant
from app.controllers.rtp import PIDController, PIDGains, ThermalBudgetCalculator
from app.simulators.rtp_hil import RTPHILDriver
from app.drivers.rtp_driver import *

# Setup driver
driver = RTPHILDriver(equipment_id="RTP-01")
await driver.connect()
await driver.initialize()

# Setup controllers
gains = PIDGains(Kp=2.0, Ki=0.5, Kd=0.1)
pid = PIDController(gains)

budget_calc = ThermalBudgetCalculator(dopant_species="boron")

# Load recipe
recipe = TemperatureRecipe(
    recipe_name="Boron Activation",
    segments=[
        RampSegment(target_temp_C=800.0, ramp_rate_C_per_s=50.0, dwell_time_s=30.0),
        RampSegment(target_temp_C=1000.0, ramp_rate_C_per_s=30.0, dwell_time_s=60.0),
        RampSegment(target_temp_C=400.0, ramp_rate_C_per_s=40.0, dwell_time_s=10.0),
    ],
    gas_params=GasFlowParameters(
        gas_type=AmbientGas.NITROGEN,
        flow_rate_sccm=5000.0,
        chamber_pressure_torr=760.0
    ),
    emissivity=EmissivitySettings(emissivity=0.65)
)

recipe_id = await driver.load_recipe(recipe)
run_id = await driver.start_recipe(recipe_id)

# Monitor and control
async for frame in telemetry.stream_telemetry():
    # Calculate thermal budget
    budget_calc.add_sample(frame.pyrometer_temp_C, frame.dt)

    # Log progress
    print(f"Temp: {frame.pyrometer_temp_C:.1f}°C / {frame.setpoint_temp_C:.1f}°C")
    print(f"Segment: {frame.current_segment}/{frame.total_segments}")

    if frame.recipe_progress_pct >= 100.0:
        break

# Post-run analysis
final_budget = budget_calc.get_budget()
print(f"Thermal budget: {final_budget.integrated_budget:.2e}")
print(f"Equivalent anneal: {final_budget.equivalent_time_at_1000C_s:.1f}s @ 1000°C")
```

---

## Running Tests

```bash
cd services/process_control

# Run all unit tests
pytest tests/unit/ -v

# Run specific test file
pytest tests/unit/test_ion_range.py -v

# Run with coverage
pytest tests/unit/ --cov=app/models --cov=app/controllers --cov-report=html
```

---

## Key Physics and Algorithms

### Ion Implantation

**LSS Theory:**
- Reduced energy: ε = 32.53 · M₂ · E / (Z₁ · Z₂ · (M₁ + M₂))
- Nuclear stopping: Sn(ε) ∝ ε / (1 + 0.5·ε^0.7)
- Electronic stopping: Se(ε) ∝ ε^0.5
- Range: Rp = ∫ [dE/dx]⁻¹ dx

**Channeling:**
- Lindhard critical angle: ψc = √(2·Z₁·Z₂·e² / E·d)
- Channeling probability: P ≈ (ψc / ψ)²

**Sheet Resistance:**
- Rs = 1 / (q · ∫ n(x) · μ(x) dx)
- Caughey-Thomas: μ(N) = μ_min + (μ_max - μ_min) / (1 + (N/N_ref)^α)

### RTP

**Radiative Heat Transfer:**
- Q_rad = ε · σ · A · (T⁴ - T_amb⁴)
- Stefan-Boltzmann: σ = 5.67×10⁻⁸ W/(m²·K⁴)

**Thermal Budget:**
- Activation rate: exp(-Ea / kT)
- Thermal budget: B = ∫ exp(-Ea / kT(t)) dt

**PID Control:**
- u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de/dt

---

## References

- **LSS Theory**: Lindhard, Scharff, Schiott, "Range Concepts and Heavy Ion Ranges", Mat. Fys. Medd. 33 (1963)
- **SRIM**: Ziegler, Biersack, "The Stopping and Range of Ions in Matter" (2008)
- **RTP**: Kermani et al., "Rapid Thermal Processing: A Justifiable Technology", Microelectronics Journal (2000)
- **Caughey-Thomas**: Caughey, Thomas, "Carrier Mobilities in Silicon Empirically Related to Doping and Field", Proc. IEEE (1967)

---

## Future Enhancements

1. **Advanced Physics:**
   - Monte Carlo trajectory simulation
   - Full SRIM integration via Python bindings
   - Non-uniform wafer temperature modeling
   - Crystallographic damage modeling

2. **Machine Learning:**
   - Neural network emissivity estimator
   - Predictive maintenance from telemetry
   - Automatic recipe optimization

3. **Integration:**
   - REST API endpoints for web frontend
   - Real-time WebSocket streaming
   - Database persistence for run history
   - Metrology correlation (SIMS, 4PP, SRP)

4. **Visualization:**
   - Real-time 2D/3D dose maps
   - Depth profile plotting
   - Temperature trajectory overlays
   - Controller performance dashboards
