# Diffusion & Oxidation Module - User Guide

**Version:** 11.0.0
**Date:** November 9, 2025
**Status:** Production Ready

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Dashboards](#dashboards)
6. [CLI Tools](#cli-tools)
7. [API Reference](#api-reference)
8. [Troubleshooting](#troubleshooting)

---

## Quickstart

### 5-Minute Start

```python
# Import the stable API
from session11.spectra import diffusion_oxidation as do

# 1. Calculate diffusion profile (ERFC)
depth_nm, concentration = do.diffusion.erfc_profile(
    dopant="boron",
    temp_c=1000,
    time_min=30
)

# 2. Calculate junction depth
xj = do.diffusion.junction_depth(concentration, depth_nm, background=1e15)
print(f"Junction depth: {xj:.1f} nm")

# 3. Plan oxidation
thickness_nm = do.oxidation.deal_grove_thickness(
    temp_c=1000,
    time_hr=2.0,
    ambient="dry"
)
print(f"Oxide thickness: {thickness_nm:.1f} nm")

# 4. Check SPC rules
import numpy as np
data = np.random.normal(100, 5, 200)
violations = do.spc.check_rules(data)
print(f"Found {len(violations)} violations")
```

---

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy scipy pandas matplotlib

# For dashboards
pip install streamlit plotly

# For ML features
pip install scikit-learn xgboost emcee

# For data I/O
pip install pyarrow fastparquet
```

### Setup

```bash
# Clone repository
git clone https://github.com/alovladi007/SPECTRA-Lab.git
cd SPECTRA-Lab/Diffusion_Module_Complete

# Install dependencies
pip install -r integrated/config/requirements.txt

# Verify installation
python -c "from session11.spectra import diffusion_oxidation as do; print('Success!')"
```

---

## Basic Usage

### 1. Diffusion Calculations

#### ERFC Analytical Solution

```python
from session11.spectra import diffusion_oxidation as do
import numpy as np
import matplotlib.pyplot as plt

# Constant source diffusion
depth, conc = do.diffusion.erfc_profile(
    dopant="boron",
    temp_c=1000,
    time_min=30,
    method="constant_source",
    surface_conc=1e20,
    background=1e15
)

# Plot
plt.semilogy(depth, conc)
plt.xlabel('Depth (nm)')
plt.ylabel('Concentration (cm⁻³)')
plt.show()
```

#### Limited Source (Gaussian)

```python
# Limited source diffusion
depth, conc = do.diffusion.erfc_profile(
    dopant="phosphorus",
    temp_c=950,
    time_min=60,
    method="limited_source",
    dose=1e14,  # cm⁻²
    background=1e15
)
```

#### Numerical Solver

```python
# For complex cases, use numerical solver
C0 = np.full(100, 1e15)  # Initial condition

depth, conc = do.diffusion.numerical_solve(
    initial_conc=C0,
    time_sec=1800,
    temp_c=1000,
    dopant="arsenic",
    bc_left=('dirichlet', 1e19),
    bc_right=('neumann', 0.0)
)
```

### 2. Oxidation Calculations

#### Forward Problem (thickness vs time)

```python
from session11.spectra import diffusion_oxidation as do

# Calculate oxide thickness
thickness_nm = do.oxidation.deal_grove_thickness(
    temp_c=1000,
    time_hr=2.0,
    ambient="dry",
    pressure=1.0,
    initial_thickness_nm=5.0
)

print(f"Final thickness: {thickness_nm:.1f} nm")
```

#### Inverse Problem (time to target)

```python
# How long to grow 100nm oxide?
time_hr = do.oxidation.time_to_target(
    target_thickness_nm=100.0,
    temp_c=1000,
    ambient="dry",
    initial_thickness_nm=5.0
)

print(f"Required time: {time_hr:.2f} hours")
```

#### Growth Rate

```python
# Instantaneous growth rate
rate = do.oxidation.growth_rate(
    thickness_nm=50.0,
    temp_c=1000,
    ambient="wet"
)

print(f"Growth rate: {rate:.2f} nm/hr")
```

### 3. Statistical Process Control

#### Western Electric Rules

```python
from session11.spectra import diffusion_oxidation as do
import numpy as np

# Generate process data
data = np.random.normal(100, 5, 200)

# Add a shift to simulate out-of-control
data[100:] += 10

# Check SPC rules
violations = do.spc.check_rules(data)

# Print violations
for v in violations:
    print(f"Rule {v['rule']} at index {v['index']}: {v['description']}")
```

#### EWMA Monitoring

```python
# Exponentially Weighted Moving Average
ewma_violations = do.spc.ewma_monitor(data, lambda_param=0.2)

print(f"Found {len(ewma_violations)} EWMA violations")
```

#### Change Point Detection

```python
# Bayesian Online Change Point Detection
changepoints = do.spc.detect_changepoints(data, threshold=0.5)

for cp in changepoints:
    print(f"Change point at index {cp['index']}, probability {cp['probability']:.2%}")
```

---

## Advanced Features

### 1. Parameter Calibration

```python
from session11.spectra import diffusion_oxidation as do
import numpy as np

# Experimental data
depth_data = np.array([0, 50, 100, 150, 200])
conc_data = np.array([1e19, 8e18, 5e18, 2e18, 1e18])

# Calibrate diffusion parameters
result = do.ml.calibrate_params(
    x_data=depth_data,
    y_data=conc_data,
    model_type="diffusion",
    method="least_squares",
    temp_celsius=1000,
    time_minutes=30,
    dopant="boron"
)

print(f"D0 = {result['parameters']['D0']:.3f} cm²/s")
print(f"Ea = {result['parameters']['Ea']:.3f} eV")
```

### 2. Feature Extraction for ML

```python
# Extract features from FDC data
fdc_data = {
    'temperature': [990, 995, 1000, 1005, 1000],
    'pressure': [100, 101, 100, 99, 100],
    'time': [0, 60, 120, 180, 240]
}

features = do.ml.extract_features(fdc_data)
print(f"Extracted {len(features)} features")
```

---

## Dashboards

### Launch Dashboards

```bash
# Diffusion Profile Viewer
streamlit run session11/dashboards/diffusion_viewer.py

# Oxide Thickness Planner
streamlit run session11/dashboards/oxide_planner.py

# SPC Monitor
streamlit run session11/dashboards/spc_monitor.py
```

### Dashboard Features

**Diffusion Viewer:**
- Interactive sliders for T, t, surface concentration
- Compare ERFC vs Numerical solvers
- Real-time junction depth calculation
- Download profiles as CSV

**Oxide Planner:**
- Forward and inverse calculations
- Temperature comparison
- Growth rate visualization
- Process planning tools

**SPC Monitor:**
- Upload CSV or generate synthetic data
- Western Electric rules
- EWMA and CUSUM monitoring
- Change point detection
- Violation summary tables

---

## CLI Tools

### Batch Diffusion Simulator

```bash
# Create input CSV
cat > runs.csv << EOF
run_id,dopant,time_minutes,temp_celsius,method,surface_conc,background
R001,B,30,1000,constant_source,1e19,1e15
R002,P,60,950,constant_source,5e18,1e15
EOF

# Run batch simulation
python session10/scripts/batch_diffusion_sim.py \
  --input runs.csv \
  --out results.parquet \
  --verbose
```

### Batch Oxidation Simulator

```bash
# Create recipe CSV
cat > recipes.csv << EOF
recipe_id,temp_celsius,time_hours,ambient
OX001,1000,2.0,dry
OX002,1100,1.0,wet
EOF

# Run simulation
python session10/scripts/batch_oxidation_sim.py \
  --input recipes.csv \
  --out ox_results.parquet \
  --verbose
```

### SPC Watch

```bash
# Monitor KPI time series
python session10/scripts/spc_watch.py \
  --series kpi.csv \
  --report spc_report.json \
  --methods all \
  --verbose
```

---

## API Reference

### Diffusion API

```python
do.diffusion.erfc_profile(dopant, temp_c, time_min, ...)
do.diffusion.junction_depth(concentration, depth_nm, background)
do.diffusion.numerical_solve(initial_conc, time_sec, temp_c, ...)
```

### Oxidation API

```python
do.oxidation.deal_grove_thickness(temp_c, time_hr, ambient, ...)
do.oxidation.time_to_target(target_thickness_nm, temp_c, ...)
do.oxidation.growth_rate(thickness_nm, temp_c, ambient, ...)
```

### SPC API

```python
do.spc.check_rules(data, centerline=None, sigma=None)
do.spc.ewma_monitor(data, lambda_param=0.2)
do.spc.detect_changepoints(data, threshold=0.5)
```

### ML API

```python
do.ml.extract_features(fdc_data)
do.ml.calibrate_params(x_data, y_data, model_type, ...)
```

---

## Troubleshooting

### Common Issues

**Import errors:**
```python
# Add parent directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Module not found:**
```bash
# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Install missing dependencies
pip install -r integrated/config/requirements.txt
```

**Dashboard won't start:**
```bash
# Check Streamlit installation
streamlit --version

# Install if missing
pip install streamlit

# Run with full path
streamlit run /full/path/to/dashboard.py
```

---

## Getting Help

- **Documentation:** See `docs/` folder
- **Examples:** See `integrated/examples/` folder
- **Tests:** Run `pytest` to verify installation
- **Issues:** Report at GitHub repository

---

**Next:** See [THEORY.md](THEORY.md) for mathematical background and [WORKFLOW.md](WORKFLOW.md) for Micron-aligned workflows.
