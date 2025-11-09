# Semiconductor Manufacturing Workflow

**Micron-Aligned Process Flow: Recipe → VM → SPC → Corrective Action**

---

## Table of Contents

1. [Overview](#overview)
2. [Recipe Development](#recipe-development)
3. [Virtual Metrology](#virtual-metrology)
4. [SPC Monitoring](#spc-monitoring)
5. [Corrective Actions](#corrective-actions)
6. [End-to-End Example](#end-to-end-example)
7. [Best Practices](#best-practices)

---

## Overview

### Manufacturing Flow

```
Recipe Definition
      ↓
Process Execution (Furnace)
      ↓
FDC Data Collection
      ↓
Virtual Metrology Prediction
      ↓
SPC Monitoring
      ↓
Violation Detection?
   /        \
  NO        YES
   ↓         ↓
Continue  Corrective Action
           ↓
      Feedback to Recipe
```

### Module Integration

**SPECTRA Diffusion Module** provides:
- Recipe planning and optimization
- Virtual metrology for real-time prediction
- SPC monitoring for process control
- Calibration for parameter refinement

---

## Recipe Development

### Step 1: Define Target Specs

```python
# Product requirements
target_junction_depth_nm = 300  # Target xj
target_tolerance_nm = 20        # ±20 nm tolerance
dopant = "boron"
background_doping = 1e15
```

### Step 2: Initial Recipe Calculation

```python
from session11.spectra import diffusion_oxidation as do

# Target: 300nm junction with boron
# Start with estimated conditions
temp_c = 1000
surface_conc = 1e19

# Iterate to find required time
for time_min in range(10, 120, 5):
    depth, conc = do.diffusion.erfc_profile(
        dopant=dopant,
        temp_c=temp_c,
        time_min=time_min,
        surface_conc=surface_conc,
        background=background_doping
    )

    xj = do.diffusion.junction_depth(conc, depth, background_doping)

    if abs(xj - target_junction_depth_nm) < 5:
        print(f"Recipe: {temp_c}°C, {time_min} min → xj = {xj:.1f} nm")
        break
```

**Output:**
```
Recipe: 1000°C, 45 min → xj = 298.3 nm
```

### Step 3: Recipe Optimization

Consider process constraints:
- Temperature uniformity (±5°C)
- Time repeatability (±2 min)
- Equipment capability

```python
# Sensitivity analysis
temps = [995, 1000, 1005]
times = [43, 45, 47]

results = []
for T in temps:
    for t in times:
        depth, conc = do.diffusion.erfc_profile(
            dopant=dopant,
            temp_c=T,
            time_min=t,
            surface_conc=surface_conc,
            background=background_doping
        )
        xj = do.diffusion.junction_depth(conc, depth, background_doping)
        results.append((T, t, xj))
        print(f"T={T}°C, t={t}min → xj={xj:.1f}nm")
```

**Robustness Check:**
- xj range: 285-312 nm
- All within tolerance? ✅ Yes
- Recipe is robust to ±5°C, ±2min variation

---

## Virtual Metrology

### Step 4: FDC Data Collection

During process execution, collect Furnace Data Collection (FDC) data:

```python
fdc_data_run_001 = {
    'temperature': [995, 998, 1000, 1002, 1000, 998],  # 6 zones
    'pressure': [100.5, 100.3, 100.2, 100.4, 100.3, 100.5],
    'time_minutes': 45.2,
    'boat_position': 3,
    'wafer_count': 150
}
```

### Step 5: Feature Extraction

```python
from session11.spectra import diffusion_oxidation as do

# Extract features for VM model
features = do.ml.extract_features(fdc_data_run_001)

# Example features:
# - temp_mean: 998.8°C
# - temp_std: 2.1°C
# - temp_ramp_rate: 5.2°C/min
# - pressure_mean: 100.37 mbar
# - thermal_budget: 44820 °C·min
```

### Step 6: VM Prediction

```python
# Predict junction depth from FDC data (requires trained model)
# For demonstration, use physics-based prediction

predicted_xj = do.diffusion.junction_depth(
    conc, depth, background_doping
)

print(f"VM Prediction: xj = {predicted_xj:.1f} nm")
```

**Typical VM Model:**
- Inputs: 29 FDC features
- Output: Junction depth, sheet resistance
- Model: XGBoost or Random Forest
- Accuracy: ±10 nm (vs ±30 nm for physical metrology)

### Step 7: Uncertainty Quantification

```python
# If using calibrated model with UQ
result = do.ml.calibrate_params(
    x_data=historical_x,
    y_data=historical_y,
    model_type="diffusion",
    method="mcmc",
    temp_celsius=1000,
    time_minutes=45,
    dopant="boron"
)

# Prediction with 95% confidence interval
print(f"Predicted: {predicted_xj:.1f} nm")
print(f"95% CI: [{result['uncertainties']['xj'][0]:.1f}, "
      f"{result['uncertainties']['xj'][1]:.1f}] nm")
```

---

## SPC Monitoring

### Step 8: Collect Historical Data

```python
import numpy as np

# Historical junction depths from past 200 runs
historical_xj = np.array([
    298, 302, 295, 301, 297,  # ... 200 values
    299, 303, 296, 305, 294
])
```

### Step 9: Run SPC Analysis

```python
from session11.spectra import diffusion_oxidation as do

# Check Western Electric rules
violations = do.spc.check_rules(historical_xj)

# Print violations
for v in violations:
    print(f"⚠️  Rule {v['rule']} at run {v['index']}: {v['description']}")
    print(f"   Value: {v['metric_value']:.1f} nm, Severity: {v['severity']}")
```

**Example Output:**
```
⚠️  Rule RULE_1 at run 173: Point beyond 3σ limit
   Value: 325.3 nm, Severity: CRITICAL

⚠️  Rule RULE_4 at run 195: 8 consecutive points same side
   Value: 285.1 nm, Severity: WARNING
```

### Step 10: Change Point Detection

```python
# Detect process shifts
changepoints = do.spc.detect_changepoints(historical_xj, threshold=0.7)

for cp in changepoints:
    print(f"🔍 Change point detected at run {cp['index']}")
    print(f"   Probability: {cp['probability']:.1%}")
    print(f"   Run length: {cp['run_length']}")
```

**Example Output:**
```
🔍 Change point detected at run 150
   Probability: 85.3%
   Run length: 0
```

**Interpretation:**
- Process shifted at run 150
- Likely root cause: Equipment maintenance, new boat, etc.

---

## Corrective Actions

### Step 11: Root Cause Analysis

When violations or change points detected:

1. **Review FDC Data**
   ```python
   # Compare FDC features before/after change point
   features_before = extract_features(fdc_data_runs_100_149)
   features_after = extract_features(fdc_data_runs_150_200)

   # Identify changed features
   for key in features_before:
       delta = abs(features_after[key] - features_before[key])
       if delta > threshold:
           print(f"Feature '{key}' changed by {delta:.2f}")
   ```

   **Example:**
   ```
   Feature 'temp_mean' changed by 3.5°C
   Feature 'pressure_std' changed by 2.1 mbar
   ```

2. **Correlate with Equipment Events**
   - PM (Preventive Maintenance) on 2025-01-15
   - Change point at run 150 (2025-01-16)
   - Likely cause: Temperature calibration drift

### Step 12: Recipe Adjustment

Based on root cause:

```python
# Temperature shifted by +3.5°C
# Adjust recipe to compensate

new_temp_c = 1000 - 3.5  # Reduce set point
new_time_min = 45        # Keep time same

# Verify new recipe
depth, conc = do.diffusion.erfc_profile(
    dopant=dopant,
    temp_c=new_temp_c,
    time_min=new_time_min,
    surface_conc=surface_conc,
    background=background_doping
)

xj_new = do.diffusion.junction_depth(conc, depth, background_doping)
print(f"Adjusted recipe → xj = {xj_new:.1f} nm")
```

**Output:**
```
Adjusted recipe → xj = 299.8 nm ✅ Back in spec
```

### Step 13: Verification

```python
# Run verification lots
verification_runs = [301, 298, 303, 297, 300]

# Check if back in control
violations_after = do.spc.check_rules(
    np.array(historical_xj.tolist() + verification_runs)
)

if len(violations_after) == 0:
    print("✅ Process back in control")
else:
    print("⚠️  Still seeing violations - further adjustment needed")
```

---

## End-to-End Example

### Complete Workflow Script

```python
from session11.spectra import diffusion_oxidation as do
import numpy as np

# 1. RECIPE DEVELOPMENT
print("="*60)
print("1. Recipe Development")
print("="*60)

target_xj = 300
dopant = "boron"
temp_c = 1000
time_min = 45
surface_conc = 1e19
background = 1e15

depth, conc = do.diffusion.erfc_profile(
    dopant=dopant, temp_c=temp_c, time_min=time_min,
    surface_conc=surface_conc, background=background
)
xj = do.diffusion.junction_depth(conc, depth, background)
print(f"Recipe: {temp_c}°C, {time_min}min → xj = {xj:.1f}nm")

# 2. VIRTUAL METROLOGY
print("\n" + "="*60)
print("2. Virtual Metrology")
print("="*60)

fdc_data = {
    'temperature': [998, 1000, 1002, 1000, 999, 1001],
    'pressure': [100.2, 100.5, 100.3, 100.4, 100.2, 100.6],
    'time_minutes': 45.1
}

features = do.ml.extract_features(fdc_data)
print(f"Extracted {len(features)} FDC features")
print(f"VM Prediction: xj ≈ {xj:.1f}nm")

# 3. SPC MONITORING
print("\n" + "="*60)
print("3. SPC Monitoring")
print("="*60)

# Simulated historical data
np.random.seed(42)
historical_data = np.random.normal(300, 10, 200)
historical_data[150:] += 15  # Simulate process shift

violations = do.spc.check_rules(historical_data)
print(f"Found {len(violations)} SPC violations")

changepoints = do.spc.detect_changepoints(historical_data, threshold=0.5)
print(f"Found {len(changepoints)} change points")

# 4. CORRECTIVE ACTION
print("\n" + "="*60)
print("4. Corrective Action")
print("="*60)

if changepoints:
    cp_index = changepoints[0]['index']
    print(f"Process shifted at run {cp_index}")

    # Calculate mean before/after
    mean_before = np.mean(historical_data[:cp_index])
    mean_after = np.mean(historical_data[cp_index:])
    shift = mean_after - mean_before

    print(f"Mean shift: {shift:.1f}nm")
    print(f"Recommended action: Adjust recipe or investigate equipment")

print("\n" + "="*60)
print("Workflow Complete")
print("="*60)
```

---

## Best Practices

### Recipe Development

1. **Start with physics-based calculation**
   - Use ERFC for initial estimates
   - Validate with numerical solver for complex cases

2. **Perform sensitivity analysis**
   - Temperature: ±5°C
   - Time: ±10%
   - Ensure robustness

3. **Document recipe parameters**
   - Target specs
   - Process window
   - Critical parameters

### Virtual Metrology

1. **Feature engineering is key**
   - Use domain knowledge
   - Extract thermal, stability, spatial features
   - 29+ features for best accuracy

2. **Model selection**
   - XGBoost for nonlinear relationships
   - Ridge/Lasso for interpretability
   - Ensemble for production

3. **Regular recalibration**
   - Weekly or after equipment PM
   - Use latest metrology data
   - Track model performance

### SPC Monitoring

1. **Choose appropriate methods**
   - Western Electric: General monitoring
   - EWMA: Small shifts
   - CUSUM: Persistent shifts
   - BOCPD: Change points

2. **Set reasonable thresholds**
   - False alarm rate < 1%
   - Sensitivity vs specificity tradeoff

3. **Root cause analysis**
   - Correlate with FDC data
   - Equipment events
   - Recipe changes

### Corrective Actions

1. **Quick response**
   - Investigate within 1 shift
   - Implement fix within 24 hours

2. **Verify effectiveness**
   - Run confirmation lots
   - Re-check SPC

3. **Close the loop**
   - Update recipe database
   - Document learnings
   - Train operators

---

**Summary:** This workflow integrates physics-based modeling, machine learning, and statistical process control for robust semiconductor manufacturing. The SPECTRA Diffusion Module provides all necessary tools in a unified framework.

**Next:** See dashboards for interactive visualization and CLI tools for batch processing.
