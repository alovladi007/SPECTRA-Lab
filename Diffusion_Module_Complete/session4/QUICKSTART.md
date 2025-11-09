# Quick Start Guide - Thermal Oxidation Simulation

## Installation

```bash
cd /home/claude/diffusion-sim
pip install -r requirements.txt --break-system-packages
```

## 1. Python API - Basic Usage

```python
# Import modules
from core import deal_grove, massoud
import numpy as np

# Example 1: Calculate oxide thickness at specific time
thickness = deal_grove.thickness_at_time(
    t=2.0,              # 2 hours
    T=1000,             # 1000°C
    ambient='dry'       # Dry O₂
)
print(f"Thickness: {thickness * 1000:.2f} nm")
# Output: Thickness: 82.38 nm

# Example 2: Calculate time to reach target thickness
time_required = massoud.time_to_thickness_with_correction(
    x_target=0.5,       # 500 nm = 0.5 μm
    T=1000,             # 1000°C
    ambient='dry',      # Dry O₂
    apply_correction=True
)
print(f"Time required: {time_required:.2f} hours")
# Output: Time required: 34.48 hours

# Example 3: Compare dry vs wet oxidation
times = np.array([0.5, 1.0, 2.0, 4.0])  # hours

dry_thickness = deal_grove.thickness_at_time(times, 1000, 'dry')
wet_thickness = deal_grove.thickness_at_time(times, 1000, 'wet')

print("\nDry vs Wet Comparison at 1000°C:")
print("Time (hr)  |  Dry (nm)  |  Wet (nm)  |  Ratio")
print("-" * 50)
for t, dry, wet in zip(times, dry_thickness, wet_thickness):
    print(f"{t:8.1f}   | {dry*1000:9.2f}  | {wet*1000:9.2f}  | {wet/dry:6.1f}×")

# Example 4: Get rate constants
B, B_over_A = deal_grove.get_rate_constants(T=1000, ambient='dry')
print(f"\nRate constants at 1000°C (dry):")
print(f"B = {B:.2e} μm²/hr")
print(f"B/A = {B_over_A:.2e} μm/hr")
```

## 2. REST API Usage

### Start the Server

```bash
cd /home/claude/diffusion-sim
python -m api.service
```

Server runs at: `http://localhost:8000`

### Interactive Documentation

Open in browser: `http://localhost:8000/docs`

### Make Requests (using curl)

```bash
# Basic simulation
curl -X POST "http://localhost:8000/oxidation/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 1000,
    "ambient": "dry",
    "time_points": [0.5, 1.0, 2.0, 4.0]
  }'

# With inverse problem
curl -X POST "http://localhost:8000/oxidation/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 1000,
    "ambient": "dry",
    "time_points": [0.5, 1.0, 2.0],
    "use_massoud": true,
    "target_thickness": 0.5
  }'

# Wet oxidation with pressure
curl -X POST "http://localhost:8000/oxidation/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 1100,
    "ambient": "wet",
    "time_points": [0.1, 0.5, 1.0],
    "pressure": 1.5
  }'
```

### Make Requests (using Python)

```python
import requests
import json

url = "http://localhost:8000/oxidation/simulate"
payload = {
    "temperature": 1000,
    "ambient": "dry",
    "time_points": [0.5, 1.0, 2.0, 4.0],
    "use_massoud": True,
    "target_thickness": 0.5
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Thickness: {result['thickness_nm']}")
print(f"Time to 500 nm: {result['inverse_solution']['time_required_hr']:.2f} hours")
```

## 3. Jupyter Notebook

```bash
cd /home/claude/diffusion-sim/notebooks
jupyter notebook 02_quickstart_oxidation.ipynb
```

The notebook includes:
- Temperature dependence plots
- Dry vs wet comparison
- Massoud correction examples
- Inverse problem demonstrations
- Contour plots for process planning

## 4. Testing & Validation

```bash
cd /home/claude/diffusion-sim

# Test Deal-Grove module
python -m core.deal_grove

# Test Massoud module
python -m core.massoud

# Test API endpoints
python tests/test_api.py

# Generate comprehensive validation plots
python validation_demo.py
```

## 5. Common Use Cases

### Calculate Oxide Growth Schedule

```python
from core import massoud

# Target: 100 nm oxide
# Temperature: 1000°C
# Ambient: Dry

target_nm = 100
target_um = target_nm / 1000.0

time_required = massoud.time_to_thickness_with_correction(
    x_target=target_um,
    T=1000,
    ambient='dry',
    apply_correction=True
)

print(f"To grow {target_nm} nm oxide at 1000°C (dry):")
print(f"Time required: {time_required:.2f} hours ({time_required*60:.1f} minutes)")
```

### Compare Temperature Effects

```python
from core import deal_grove
import numpy as np

temperatures = [900, 1000, 1100, 1200]
target_thickness = 0.1  # 100 nm

print("Time to 100 nm (dry oxidation):")
for T in temperatures:
    t = deal_grove.time_to_thickness(target_thickness, T, 'dry')
    print(f"{T}°C: {t:.2f} hours")
```

### Process Planning: Multiple Steps

```python
from core import massoud

# Step 1: Initial dry oxidation
step1_time = 1.0  # hours
step1_thickness = massoud.thickness_with_correction(
    t=step1_time,
    T=1000,
    ambient='dry',
    x_i=0.0,
    apply_correction=True
)

# Step 2: Wet oxidation on top of initial oxide
step2_time = 0.5  # hours
final_thickness = massoud.thickness_with_correction(
    t=step2_time,
    T=1000,
    ambient='wet',
    x_i=step1_thickness,
    apply_correction=True
)

print(f"Step 1 (dry): {step1_thickness*1000:.2f} nm after {step1_time} hr")
print(f"Step 2 (wet): {final_thickness*1000:.2f} nm after {step2_time} hr")
print(f"Total: {final_thickness*1000:.2f} nm")
```

## 6. Important Notes

### Units
- **Temperature**: Celsius (°C)
- **Time**: Hours
- **Thickness**: Micrometers (μm) in code, nanometers (nm) in output
- **Pressure**: Atmospheres (atm)

### Conversion
```python
# nm to μm
thickness_um = thickness_nm / 1000.0

# μm to nm
thickness_nm = thickness_um * 1000.0

# minutes to hours
time_hr = time_min / 60.0
```

### Temperature Range
- Typical: 900-1200°C
- Validated: 800-1300°C
- Rate constants valid for T > 700°C

### When to Use Massoud Correction
- **Always use** for oxides < 100 nm
- **Important** for oxides < 70 nm
- **Critical** for oxides < 30 nm
- Negligible for oxides > 200 nm

## 7. Troubleshooting

### Import Errors
```bash
# Make sure you're in the project directory
cd /home/claude/diffusion-sim

# Or add to Python path
import sys
sys.path.insert(0, '/home/claude/diffusion-sim')
```

### API Not Starting
```bash
# Check if port 8000 is available
lsof -i :8000

# Use different port
uvicorn api.service:app --port 8001
```

### Jupyter Kernel Not Found
```bash
pip install ipykernel --break-system-packages
python -m ipykernel install --user --name diffusion-sim
```

## 8. References

Full documentation: `/home/claude/diffusion-sim/README.md`  
Detailed summary: `/home/claude/diffusion-sim/SESSION4_SUMMARY.md`  
Validation plots: `/mnt/user-data/outputs/session4_validation.png`

## Quick Reference Card

```
DEAL-GROVE MODEL:  x² + A·x = B·(t + τ)

KEY EQUATIONS:
├─ Growth:     x(t) = (-A + √(A² + 4B(t+τ)))/2
├─ Rate:       dx/dt = B/(2x + A)
└─ Inverse:    t = (x² + Ax)/B - τ

RATE CONSTANTS @ 1000°C:
├─ Dry:   B = 9.35×10⁻³ μm²/hr, A = 144.5 nm
└─ Wet:   B = 3.15×10⁵ μm²/hr,  A = 65,979 nm

MASSOUD CORRECTION:
└─ x_corrected = x_DG + 20·exp(-x_DG/7)  [nm]

TYPICAL VALUES:
├─ 100 nm dry  @ 1000°C: ~1 hour
├─ 500 nm dry  @ 1000°C: ~34 hours
├─ 500 nm wet  @ 1000°C: ~6 minutes
└─ Wet is ~100× faster than dry
```

---

**Ready to use!** Start with the Jupyter notebook for an interactive introduction.
