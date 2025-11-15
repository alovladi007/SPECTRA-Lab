# Advanced CVD Physics Models

Comprehensive physics-based models for film thickness, stress, and adhesion prediction with reactor-specific corrections and Virtual Metrology (VM) integration.

## Overview

This package provides sophisticated models for:

1. **Film Thickness** - Arrhenius kinetics with reactor geometry effects
2. **Film Stress** - Intrinsic + thermal stress with multiple measurement methods
3. **Film Adhesion** - Multi-factor scoring with test simulations
4. **Reactor Geometry** - Showerhead, horizontal flow, batch furnace configurations
5. **VM Features** - Feature engineering for ML-based Virtual Metrology

## Package Structure

```
physics_models/
├── __init__.py                 - Package initialization & exports
├── thickness.py                - Film thickness modeling
├── stress.py                   - Film stress modeling
├── adhesion.py                 - Film adhesion modeling
├── reactor_geometry.py         - Reactor-specific models
├── vm_features.py              - VM feature engineering
├── examples.py                 - Usage examples
└── README.md                   - This documentation
```

## Quick Start

### Film Thickness Prediction

```python
from physics_models import ThicknessModel, DepositionParameters, CVDMode
from physics_models import ShowerheadReactor

# Create reactor model
reactor = ShowerheadReactor(
    wafer_diameter_mm=200.0,
    is_rotating=True,
    rotation_speed_rpm=20.0,
)

# Create thickness model
model = ThicknessModel(mode=CVDMode.THERMAL, reactor=reactor)

# Define process parameters
params = DepositionParameters(
    temperature_c=780.0,
    pressure_torr=0.3,
    precursor_flow_sccm=80.0,
    film_material="Si3N4",
)

# Predict thickness after 1 hour
result = model.predict_thickness(params, time_sec=3600.0)

print(f"Mean thickness: {result['mean_thickness_nm']:.1f} nm")
print(f"Deposition rate: {result['deposition_rate_nm_min']:.2f} nm/min")
print(f"WIW uniformity: {result['wiw_uniformity_pct']:.2f}%")
```

### Film Stress Calculation

```python
from physics_models import StressModel, ProcessConditions, get_material_properties

# Get material properties
material = get_material_properties("Si3N4")

# Create stress model
model = StressModel(material=material)

# Define process conditions
process = ProcessConditions(
    temperature_c=780.0,
    pressure_torr=0.3,
    deposition_rate_nm_min=50.0,
    film_thickness_nm=100.0,
)

# Calculate stress
stress_result = model.calculate_total_stress(process)

print(f"Total stress: {stress_result['stress_mean_mpa']:.1f} MPa")
print(f"Type: {stress_result['stress_type'].value}")
print(f"Intrinsic: {stress_result['intrinsic_stress_mpa']:.1f} MPa")
print(f"Thermal: {stress_result['thermal_stress_mpa']:.1f} MPa")
```

### Film Adhesion Scoring

```python
from physics_models import AdhesionModel, AdhesionFactors
from physics_models import simulate_tape_test, simulate_scratch_test

# Create model
model = AdhesionModel()

# Define factors
factors = AdhesionFactors(
    film_stress_mpa=-250.0,
    pre_clean_quality=0.95,
    surface_roughness_ra_nm=0.8,
    deposition_temp_c=780.0,
)

# Calculate score
score, classification = model.calculate_adhesion_score(factors)

print(f"Adhesion: {score:.1f}/100 ({classification.value})")

# Simulate tests
tape_result = simulate_tape_test(score)
scratch_result = simulate_scratch_test(score, film_hardness_gpa=20.0)

print(f"Tape test: {tape_result.notes}")
print(f"Scratch test: Critical load = {scratch_result.critical_load_n:.2f} N")
```

## Detailed Documentation

### 1. Film Thickness Modeling

#### Deposition Kinetics

**Arrhenius Rate Equation**:

```
rate = A * exp(-Ea/RT) * (P/P₀)^n * (F/F₀)^m * (Power/Power₀)^p
```

Where:
- `A` = pre-exponential factor
- `Ea` = activation energy
- `R` = gas constant
- `T` = temperature (K)
- `P` = pressure
- `F` = precursor flow rate
- `Power` = RF power (for PECVD)
- `n, m, p` = exponents

**Implementation**:

```python
from physics_models import DepositionRateCalculator, ArrheniusParameters

# Custom kinetic parameters
params = ArrheniusParameters(
    pre_exponential_A=1e12,
    activation_energy_kj_mol=120.0,
    pressure_exponent=0.5,
    flow_exponent=0.5,
)

calc = DepositionRateCalculator(mode=CVDMode.THERMAL, arrhenius_params=params)

rate = calc.calculate_rate(
    temperature_c=780.0,
    pressure_torr=0.3,
    precursor_flow_sccm=80.0,
)

print(f"Deposition rate: {rate:.2f} nm/min")
```

#### WIW Uniformity Calculation

Thickness uniformity depends on:
- **Reactor geometry** (showerhead, horizontal flow, batch)
- **Gas depletion** effects
- **Wafer rotation** (improves uniformity)
- **Flow pattern** (top-down, side injection, etc.)

**Example**:

```python
from physics_models import UniformityCalculator

calc = UniformityCalculator(reactor=showerhead_reactor)

# Get thickness map (49 points)
x_mm, y_mm, thickness_map = calc.calculate_wiw_map(
    base_thickness_nm=100.0,
    wafer_diameter_mm=200.0,
    num_points=49,
)

# Calculate uniformity percentage
uniformity_pct = calc.calculate_wiw_uniformity(thickness_map)

print(f"WIW uniformity: {uniformity_pct:.2f}%")
```

### 2. Film Stress Modeling

#### Total Stress Calculation

```
σ_total = σ_intrinsic + σ_thermal + σ_gradient
```

**Intrinsic Stress**:
- Process-dependent (rate, pressure, plasma power)
- Material microstructure (grain boundaries, defects)
- Ion bombardment effects (for PECVD)

**Thermal Stress** (CTE mismatch):

```
σ_thermal = [E/(1-ν)] * (α_film - α_substrate) * ΔT
```

Where:
- `E` = Young's modulus
- `ν` = Poisson's ratio
- `α` = coefficient of thermal expansion
- `ΔT` = temperature change (deposition → measurement)

**Gradient Stress**:
- Through-thickness variation
- Surface effects

#### Measurement Method Conversions

**Wafer Curvature** (Stoney's Equation):

```python
from physics_models import wafer_curvature_to_stress

stress_mpa = wafer_curvature_to_stress(
    curvature_1_m=2.5,  # Measured curvature
    film_thickness_nm=100.0,
    substrate_thickness_um=725.0,
)

print(f"Stress from curvature: {stress_mpa:.1f} MPa")
```

**XRD Peak Shift**:

```python
from physics_models import xrd_to_stress

stress_mpa = xrd_to_stress(
    d_measured_angstrom=3.150,
    d_unstressed_angstrom=3.145,
    film_youngs_modulus_gpa=250.0,
)

print(f"Stress from XRD: {stress_mpa:.1f} MPa")
```

**Nanoindentation**:

```python
from physics_models import nanoindentation_to_stress

stress_estimate = nanoindentation_to_stress(
    hardness_gpa=20.0,
    youngs_modulus_gpa=250.0,
)

print(f"Stress estimate: {stress_estimate:.1f} MPa")
```

#### Material Property Database

Built-in properties for common films:

- `SiO2` - Silicon dioxide
- `Si3N4` - Silicon nitride
- `TiN` - Titanium nitride
- `W` - Tungsten
- `Al` - Aluminum
- `Cu` - Copper
- `a-Si` - Amorphous silicon
- `DLC` - Diamond-like carbon
- `GaN` - Gallium nitride

```python
from physics_models import get_material_properties, MATERIAL_DATABASE

# Get properties
si3n4 = get_material_properties("Si3N4")

print(f"E = {si3n4.film_youngs_modulus_gpa} GPa")
print(f"ν = {si3n4.film_poisson_ratio}")
print(f"α = {si3n4.film_cte_ppm_k} ppm/K")
print(f"ρ = {si3n4.film_density_g_cm3} g/cm³")

# View all materials
for name in MATERIAL_DATABASE.keys():
    print(name)
```

### 3. Film Adhesion Modeling

#### Adhesion Score Calculation

```
score = base * stress_penalty * surface * interlayer * contamination * microstructure * process
```

**Factors**:

1. **Stress penalty**: High stress → lower adhesion
   - Magnitude effect
   - Gradient effect (edge delamination)

2. **Surface preparation**:
   - Clean quality (0-1 scale)
   - Roughness (optimal: 0.5-2 nm Ra)

3. **Interlayer compatibility**:
   - Chemical bonding
   - Lattice match
   - CTE compatibility

4. **Contamination penalty**:
   - Particles
   - Moisture
   - Organics

5. **Microstructure**:
   - Film density
   - Thickness (thinner → better)
   - Grain size

6. **Process conditions**:
   - Deposition temperature
   - Ion bombardment (moderate helps)

#### Adhesion Test Simulations

**Tape Test** (ASTM D3359):

```python
result = simulate_tape_test(adhesion_score=85.0, test_type="cross_cut")

# Classifications: 5B, 4B, 3B, 2B, 1B, 0B
print(result.notes)
```

**Scratch Test**:

```python
result = simulate_scratch_test(
    adhesion_score=85.0,
    film_hardness_gpa=20.0,
    film_thickness_nm=100.0,
)

print(f"Critical load: {result.critical_load_n:.2f} N")
print(f"Failure mode: {result.failure_mode.value}")
```

**Nanoindentation**:

```python
result = simulate_nanoindentation(
    adhesion_score=85.0,
    film_youngs_modulus_gpa=250.0,
)

print(f"Interfacial energy: {result.interfacial_energy_j_m2:.2f} J/m²")
```

**Stud Pull Test**:

```python
result = simulate_stud_pull(adhesion_score=85.0, stud_diameter_mm=2.0)

print(f"Pull-off force: {result.critical_load_n:.1f} N")
```

### 4. Reactor Geometry Models

#### Showerhead Reactor

Top-down gas injection through perforated plate.

**Key Parameters**:
- Gap between showerhead and wafer
- Hole pattern (hexagonal, concentric, random)
- Wafer rotation

**Uniformity**:
- Excellent with rotation
- Minimal gas depletion

```python
from physics_models import ShowerheadReactor

reactor = ShowerheadReactor(
    gap_mm=20.0,
    hole_pattern="hexagonal",
    is_rotating=True,
)

uniformity = reactor.calculate_uniformity_factor(
    radial_position_mm=95.0,
    pressure_torr=1.0,
    temperature_c=300.0,
)

print(f"Uniformity factor: {uniformity:.3f}")
```

#### Horizontal Flow Reactor

Side gas injection with horizontal flow across wafer.

**Key Parameters**:
- Flow direction
- Boundary layer thickness
- Wafer rotation

**Uniformity**:
- Gas depletion along flow direction
- Improved with rotation

```python
from physics_models import HorizontalFlowReactor

reactor = HorizontalFlowReactor(
    is_rotating=True,
    boundary_layer_mm=5.0,
)

uniformity = reactor.calculate_uniformity_factor(
    x_mm=100.0,  # Along flow
    y_mm=0.0,
    pressure_torr=0.3,
    flow_velocity_cm_s=10.0,
)
```

#### Batch Furnace Reactor

Hot-wall tube furnace with multiple wafers.

**Key Parameters**:
- Number of wafers (25-200)
- Wafer spacing
- Tube temperature gradient

**Uniformity**:
- Excellent WIW uniformity
- WTW variation from position in boat

```python
from physics_models import BatchFurnaceReactor

reactor = BatchFurnaceReactor(
    num_wafers=100,
    wafer_spacing_mm=10.0,
)

# Position factor for wafer #50
position_factor = reactor.calculate_boat_position_factor(
    wafer_index=50,
    temperature_c=780.0,
)

# WTW uniformity
wtw_uniformity = reactor.calculate_wtw_uniformity()

print(f"WTW uniformity: {wtw_uniformity:.2f}%")
```

### 5. Virtual Metrology Feature Engineering

#### Feature Extraction

```python
from physics_models import VMFeatureExtractor

extractor = VMFeatureExtractor()

features = extractor.extract_all_features(
    deposition_params=params,
    process_conditions=process,
    adhesion_factors=factors,
    telemetry=telemetry,  # Optional time-series data
)

print(f"Total features: {len(features)}")
```

**Feature Categories**:

1. **Process parameters** - Temperature, pressure, flows, power
2. **Physics predictions** - Thickness, stress, adhesion from models
3. **Telemetry statistics** - Mean, std, min, max, slope, CV
4. **Derived features** - Ratios, products, power density

#### Training Dataset Creation

```python
from physics_models import create_training_dataset

# Simulate multiple runs
params_list = [...]  # List of DepositionParameters
measured_thickness = [...]  # Ground truth measurements

dataset = create_training_dataset(
    deposition_params_list=params_list,
    measured_thickness_nm_list=measured_thickness,
    measured_stress_mpa_list=stress_measurements,  # Optional
    measured_adhesion_score_list=adhesion_measurements,  # Optional
)

X = dataset["X"]  # Feature matrix
y_thickness = dataset["y_thickness_nm"]  # Target
feature_names = dataset["feature_names"]

# Train ML model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y_thickness)
```

#### Feature Importance Analysis

```python
from physics_models import feature_importance_analysis

importance = feature_importance_analysis(X, y_thickness, feature_names)

# Top 10 features
for name, score in list(importance.items())[:10]:
    print(f"{name}: {score:.3f}")
```

## Advanced Usage

### Custom Material Properties

```python
from physics_models import MaterialProperties, StressModel

custom_material = MaterialProperties(
    film_name="MyFilm",
    film_youngs_modulus_gpa=150.0,
    film_poisson_ratio=0.25,
    film_cte_ppm_k=5.0,
    film_density_g_cm3=4.0,
)

stress_model = StressModel(material=custom_material)
```

### Custom Arrhenius Parameters

```python
from physics_models import ArrheniusParameters, DepositionRateCalculator

custom_kinetics = ArrheniusParameters(
    pre_exponential_A=5e11,
    activation_energy_kj_mol=100.0,
    pressure_exponent=0.3,
)

calc = DepositionRateCalculator(arrhenius_params=custom_kinetics)
```

### Telemetry Statistics

```python
from physics_models import TelemetryData
import numpy as np

# Create telemetry from real data
telemetry = TelemetryData(
    time_sec=time_array,
    temperature_c=temp_array,
    pressure_torr=pressure_array,
    precursor_flow_sccm=flow_array,
    rf_power_w=power_array,  # Optional
    bias_voltage_v=bias_array,  # Optional
)

# Extract statistics
features = extractor._extract_telemetry_features(telemetry)

print(f"Temp mean: {features['temp_mean']:.2f}°C")
print(f"Temp std: {features['temp_std']:.2f}°C")
print(f"Temp slope: {features['temp_slope']:.3f}")
```

## Integration with HIL Simulator

The physics models can be integrated with the HIL simulator for realistic process simulation:

```python
from app.drivers import HILCVDSimulator, PhysicsConfig
from physics_models import ThicknessModel, StressModel, AdhesionModel

# Create physics models
thickness_model = ThicknessModel()
stress_model = StressModel()
adhesion_model = AdhesionModel()

# Create simulator
sim = HILCVDSimulator(
    tool_id="SIM-01",
    mode="LPCVD",
)

# Override simulator's physics with advanced models
# (Implementation in simulator code)
```

## Performance Notes

- **Thickness calculation**: < 1 ms per point
- **Stress calculation**: < 1 ms
- **Adhesion scoring**: < 1 ms
- **WIW map generation**: < 10 ms for 49 points
- **VM feature extraction**: < 5 ms per run

## References

### Thickness & Deposition Kinetics
- Handbook of Thin Film Deposition (Seshan, 2012)
- CVD of Nonmetals (Sze & Grobe, 1988)

### Film Stress
- Freund & Suresh, "Thin Film Materials" (2004)
- Stoney's Equation: Proc. R. Soc. Lond. A (1909)
- Flinn, "Measurement and Interpretation of Stress in Copper Films" (1991)

### Adhesion
- Mittal, "Adhesion Measurement of Films and Coatings" (1995)
- ASTM D3359 - Tape Test Standard
- ASTM C1624 - Scratch Test Standard

### Reactor Modeling
- Jensen, "Chemical Vapor Deposition" (1993)
- Hitchman & Jensen, "Chemical Vapor Deposition" (1993)

## License

Internal use only - SPECTRA Lab
