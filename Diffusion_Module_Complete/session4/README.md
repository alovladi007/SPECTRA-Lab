# Diffusion-Sim: Semiconductor Process Simulation

A Python-based simulation framework for thermal oxidation and diffusion processes in semiconductor manufacturing.

## Features

### Session 4: Thermal Oxidation (Deal-Grove + Massoud)

- **Deal-Grove Model**: Industry-standard linear-parabolic oxidation kinetics
- **Massoud Correction**: Thin-oxide corrections for enhanced accuracy (<70 nm)
- **Temperature Dependence**: Arrhenius rate constants
- **Dry/Wet Oxidation**: O₂ and H₂O ambient support
- **Inverse Solver**: Calculate time to reach target thickness
- **REST API**: FastAPI service for web integration
- **Jupyter Notebooks**: Interactive demonstrations and visualizations

## Installation

```bash
# Clone repository
git clone <repository-url>
cd diffusion-sim

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Python API

```python
from core import deal_grove, massoud
import numpy as np

# Basic oxidation simulation
T = 1000  # Celsius
times = np.array([0.5, 1.0, 2.0, 4.0])  # hours

# Deal-Grove model
thickness = deal_grove.thickness_at_time(times, T, 'dry')
print(f"Thickness: {thickness * 1000} nm")

# With Massoud thin-oxide correction
thickness_corrected = massoud.thickness_with_correction(
    times, T, 'dry', apply_correction=True
)
print(f"Corrected: {thickness_corrected * 1000} nm")

# Inverse problem: time to reach target
target_thickness = 0.5  # μm
time_required = massoud.time_to_thickness_with_correction(
    target_thickness, T, 'dry'
)
print(f"Time to {target_thickness} μm: {time_required:.3f} hours")
```

### REST API

Start the server:
```bash
python -m api.service
```

Or with uvicorn:
```bash
uvicorn api.service:app --reload --host 0.0.0.0 --port 8000
```

Access documentation:
- Interactive docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Example API request:
```bash
curl -X POST "http://localhost:8000/oxidation/simulate" \
  -H "Content-Type: application/json" \
  -d '{
    "temperature": 1000,
    "ambient": "dry",
    "time_points": [0.5, 1.0, 2.0, 4.0],
    "pressure": 1.0,
    "use_massoud": true,
    "target_thickness": 0.5
  }'
```

### Jupyter Notebook

```bash
cd notebooks
jupyter notebook 02_quickstart_oxidation.ipynb
```

## Project Structure

```
diffusion-sim/
├── core/                      # Core physics models
│   ├── deal_grove.py         # Deal-Grove oxidation model
│   ├── massoud.py            # Massoud thin-oxide correction
│   └── __init__.py
├── api/                       # REST API service
│   ├── service.py            # FastAPI application
│   └── __init__.py
├── notebooks/                 # Jupyter notebooks
│   └── 02_quickstart_oxidation.ipynb
├── tests/                     # Unit tests
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Models

### Deal-Grove Model

The Deal-Grove model describes thermal oxidation of silicon:

```
x_ox² + A·x_ox = B·(t + τ)
```

Where:
- `x_ox`: oxide thickness (μm)
- `A`: linear rate constant (μm)
- `B`: parabolic rate constant (μm²/hr)
- `t`: oxidation time (hours)
- `τ`: time shift for initial oxide

Rate constants follow Arrhenius temperature dependence:
```
k = k₀ exp(-Ea / (kB·T))
```

**Regimes:**
- **Linear** (x << A): Growth rate ≈ B/A (constant)
- **Parabolic** (x >> A): Growth rate ≈ B/(2x) (decreasing)

### Massoud Correction

For thin oxides (<~70 nm), the Massoud model adds an exponential correction:

```
x_Massoud = x_DG + C·exp(-x_DG / L)
```

Where:
- `x_DG`: Deal-Grove predicted thickness
- `C`: Correction amplitude (~20 nm)
- `L`: Characteristic length (~7 nm)

This correction accounts for enhanced oxidation rates in the thin-oxide regime.

## API Endpoints

### POST /oxidation/simulate

Simulate thermal oxidation with Deal-Grove and optional Massoud correction.

**Request Body:**
```json
{
  "temperature": 1000,           // Temperature (°C)
  "ambient": "dry",              // "dry" or "wet"
  "time_points": [0.5, 1.0, 2.0], // Time array (hours)
  "pressure": 1.0,               // Partial pressure (atm)
  "initial_thickness": 0.0,      // Initial oxide (μm)
  "use_massoud": true,           // Apply correction
  "target_thickness": 0.5        // Optional: inverse problem
}
```

**Response:**
```json
{
  "time_points": [0.5, 1.0, 2.0],
  "thickness": [0.123, 0.174, 0.246],  // μm
  "thickness_nm": [123, 174, 246],     // nm
  "temperature": 1000,
  "ambient": "dry",
  "rate_constants": {
    "B": 0.00377,           // μm²/hr
    "B_over_A": 0.0125,     // μm/hr
    "A": 0.301              // μm
  },
  "massoud_applied": true,
  "inverse_solution": {       // If target_thickness provided
    "target_thickness_um": 0.5,
    "time_required_hr": 3.456,
    "growth_rate_at_target": 0.0025
  }
}
```

## Testing

Run the core modules directly:

```bash
# Test Deal-Grove model
python -m core.deal_grove

# Test Massoud correction
python -m core.massoud
```

Expected output includes validation calculations and comparisons with published data.

## Physical Constants and Parameters

### Deal-Grove Rate Constants (1 atm)

**Dry Oxidation (O₂):**
- B₀ = 7.72×10⁵ μm²/hr
- (B/A)₀ = 3.71×10⁶ μm/hr
- Ea(B) = 2.0 eV
- Ea(B/A) = 1.96 eV

**Wet Oxidation (H₂O):**
- B₀ = 3.86×10⁸ μm²/hr
- (B/A)₀ = 6.23×10⁸ μm/hr
- Ea(B) = 0.78 eV
- Ea(B/A) = 2.05 eV

### Typical Values at 1000°C

| Parameter | Dry O₂ | Wet H₂O |
|-----------|--------|---------|
| B (μm²/hr) | 0.0038 | 0.39 |
| B/A (μm/hr) | 0.0125 | 0.43 |
| A (μm) | 0.30 | 0.91 |

**Note:** Wet oxidation is ~100× faster than dry!

## Development Roadmap

- [x] Session 4: Deal-Grove + Massoud oxidation models
- [ ] Session 5: Dopant diffusion models
- [ ] Session 6: Coupled oxidation-diffusion
- [ ] Session 7: 2D spatial profiles
- [ ] Session 8: Process optimization

## References

1. Deal, B. E., & Grove, A. S. (1965). General relationship for the thermal oxidation of silicon. *Journal of Applied Physics*, 36(12), 3770-3778.

2. Massoud, H. Z., Plummer, J. D., & Irene, E. A. (1985). Thermal oxidation of silicon in dry oxygen: Accurate determination of the kinetic rate constants. *Journal of the Electrochemical Society*, 132(11), 2685-2693.

3. Plummer, J. D., Deal, M. D., & Griffin, P. B. (2000). *Silicon VLSI Technology: Fundamentals, Practice, and Modeling*. Prentice Hall.

## License

MIT License - See LICENSE file for details

## Version

Current version: **0.4.0** (Session 4: Thermal Oxidation)
Git tag: `diffusion-v4`
