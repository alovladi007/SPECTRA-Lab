# Session 8: Optical Methods II - Complete Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Theory & Principles](#theory--principles)
3. [Implementation Architecture](#implementation-architecture)
4. [Method Details](#method-details)
5. [API Reference](#api-reference)
6. [User Workflows](#user-workflows)
7. [Data Analysis](#data-analysis)
8. [Troubleshooting](#troubleshooting)
9. [Performance Specifications](#performance-specifications)
10. [Safety & Compliance](#safety--compliance)

---

## Executive Summary

Session 8 implements three advanced optical characterization techniques essential for semiconductor analysis:

- **Spectroscopic Ellipsometry**: Non-destructive measurement of thin film thickness and optical constants
- **Photoluminescence (PL) Spectroscopy**: Analysis of radiative recombination and material quality
- **Raman Spectroscopy**: Vibrational spectroscopy for composition, stress, and crystallinity

These methods provide complementary information about optical properties, electronic structure, and material quality with nanometer-scale sensitivity.

### Key Capabilities

| Method | Primary Information | Sensitivity | Spatial Resolution |
|--------|-------------------|-------------|-------------------|
| Ellipsometry | Film thickness, n & k | 0.1 nm thickness | ~50 μm spot |
| Photoluminescence | Bandgap, defects, QY | <1 meV energy | ~1 μm (confocal) |
| Raman | Stress, crystallinity | 0.1 cm⁻¹ shift | <1 μm (confocal) |

---

## Theory & Principles

### Spectroscopic Ellipsometry

#### Fundamental Equation
Ellipsometry measures the change in polarization state:

```
ρ = r_p / r_s = tan(Ψ) · exp(iΔ)
```

Where:
- ρ = Complex reflectance ratio
- r_p, r_s = Fresnel reflection coefficients for p and s polarization
- Ψ = Amplitude ratio (0° to 90°)
- Δ = Phase difference (0° to 360°)

#### Fresnel Equations
For interface between media with refractive indices n₁ and n₂:

```
r_p = (n₂cosθ₁ - n₁cosθ₂)/(n₂cosθ₁ + n₁cosθ₂)
r_s = (n₁cosθ₁ - n₂cosθ₂)/(n₁cosθ₁ + n₂cosθ₂)
```

#### Transfer Matrix Method
For multilayer stacks, we use 2×2 matrices:

```
M = ∏ᵢ Mᵢ = ∏ᵢ [cos(βᵢ)    i·sin(βᵢ)/nᵢ]
                [i·nᵢ·sin(βᵢ)   cos(βᵢ)]
```

Where βᵢ = 2π·nᵢ·dᵢ·cosθᵢ/λ

#### Dispersion Models

**Cauchy Model** (transparent materials):
```
n(λ) = A + B/λ² + C/λ⁴
```

**Sellmeier Model** (wide spectral range):
```
n²(λ) = 1 + Σᵢ (Bᵢλ²)/(λ² - Cᵢ)
```

**Tauc-Lorentz Model** (amorphous semiconductors):
Combines Tauc gap with Lorentz oscillator for ε₂:
```
ε₂(E) = A·E₀·C·(E-Eg)²/((E²-E₀²)² + C²E²) · 1/E  for E > Eg
ε₂(E) = 0                                        for E ≤ Eg
```

### Photoluminescence Spectroscopy

#### Radiative Recombination
PL intensity depends on:
```
I_PL ∝ B·n·p
```

Where B is the radiative recombination coefficient, n and p are electron and hole concentrations.

#### Temperature Dependence

**Varshni Equation** for bandgap:
```
Eg(T) = Eg(0) - αT²/(T + β)
```

**Intensity Quenching**:
```
I(T) = I₀/(1 + A·exp(-Ea/kT))
```

Where Ea is the activation energy for non-radiative processes.

#### Peak Types
1. **Band-edge emission**: Direct bandgap recombination
2. **Excitonic emission**: Bound electron-hole pairs
3. **Donor-Acceptor Pairs (DAP)**: Impurity-related
4. **Phonon replicas**: Coupling with lattice vibrations
5. **Deep level emission**: Defect states

### Raman Spectroscopy

#### Raman Scattering
Inelastic light scattering with energy change:
```
ΔE = ℏω_phonon
```

Stokes: ω_scattered = ω_laser - ω_phonon
Anti-Stokes: ω_scattered = ω_laser + ω_phonon

#### Selection Rules
For first-order Raman in crystals:
- Γ-point phonons only (q ≈ 0)
- Symmetry-allowed modes

#### Stress/Strain Analysis
Peak shift with stress:
```
Δω = Σᵢⱼ pᵢⱼ·εᵢⱼ
```

For uniaxial stress in Si:
```
Δω = -2.0ω₀·σ/E
```

Where σ is stress, E is Young's modulus.

#### Crystallinity Assessment
- **Peak position**: Crystal quality indicator
- **FWHM**: Grain size via phonon confinement
- **Intensity ratio**: Defect density (e.g., D/G in graphene)

---

## Implementation Architecture

### System Components

```
Session 8 Architecture
├── Backend Services
│   ├── EllipsometryAnalyzer
│   │   ├── Fresnel calculations
│   │   ├── Transfer matrix method
│   │   ├── Model fitting engine
│   │   └── Dispersion models
│   ├── PhotoluminescenceAnalyzer
│   │   ├── Peak detection
│   │   ├── Multi-peak fitting
│   │   ├── Temperature analysis
│   │   └── Quantum yield calculation
│   └── RamanAnalyzer
│       ├── Peak identification
│       ├── Stress calculation
│       ├── Crystallinity analysis
│       └── Mapping processor
├── Frontend Components
│   ├── EllipsometryInterface
│   │   ├── Layer stack builder
│   │   ├── Ψ-Δ visualizer
│   │   └── Fit results display
│   ├── PLInterface
│   │   ├── Spectrum display
│   │   ├── Temperature controller
│   │   └── Peak analyzer
│   └── RamanInterface
│       ├── Spectrum viewer
│       ├── Peak identifier
│       └── Stress calculator
└── Data Management
    ├── Measurement storage
    ├── Model database
    └── Reference libraries
```

### Data Flow

```mermaid
graph LR
    A[Instrument] --> B[Data Acquisition]
    B --> C[Raw Data Storage]
    C --> D[Processing Pipeline]
    D --> E[Analysis Engine]
    E --> F[Results Database]
    F --> G[Visualization]
    G --> H[Reports]
```

---

## Method Details

### Ellipsometry Workflow

#### 1. Measurement Setup
- **Angle Selection**: Typically 65-75° for best sensitivity
- **Spectral Range**: Match to film transparency
- **Spot Size**: Consider film uniformity
- **Calibration**: Use known standard (e.g., SiO₂/Si)

#### 2. Model Construction
```python
# Example: SiO₂ on Si
stack = LayerStack(
    layers=[
        {
            'thickness': 100,  # Initial guess
            'model': DispersionModel.CAUCHY,
            'params': {'A': 1.46, 'B': 0.00354, 'C': 0}
        }
    ],
    substrate={'n': 3.85, 'k': 0.02}  # Si
)
```

#### 3. Fitting Process
- **Parameters**: Select physically meaningful parameters
- **Constraints**: Apply reasonable bounds
- **Uniqueness**: Check for parameter correlation
- **Validation**: Compare MSE, check unphysical results

#### 4. Quality Metrics
- **MSE < 10**: Good fit for single layer
- **MSE < 20**: Acceptable for complex stacks
- **Confidence intervals**: Should be < 10% of value

### Photoluminescence Workflow

#### 1. Sample Preparation
- **Surface cleaning**: Remove contamination
- **Mounting**: Ensure good thermal contact
- **Alignment**: Optimize collection efficiency

#### 2. Measurement Conditions
```python
# Optimal conditions for semiconductors
params = {
    'temperature': 10,  # K - reduce thermal broadening
    'excitation_power': 1-10,  # mW - avoid saturation
    'excitation_wavelength': 532,  # nm - above bandgap
    'integration_time': 1-10  # s - improve S/N
}
```

#### 3. Data Analysis Pipeline
1. **Background subtraction**
2. **Spectral correction** (detector response)
3. **Peak identification**
4. **Multi-peak deconvolution**
5. **Parameter extraction**

#### 4. Temperature Series Analysis
```python
# Extract activation energy
temperatures = [10, 20, 50, 100, 150, 200, 250, 300]
for T in temperatures:
    spectrum = measure_pl(T)
    peaks = find_peaks(spectrum)
    intensities.append(peaks[0].intensity)

# Arrhenius plot
ln_I = np.log(intensities)
inv_T = 1000/temperatures
Ea = -slope * k_B  # Activation energy
```

### Raman Workflow

#### 1. Laser Selection
| Material | Recommended Laser | Reason |
|----------|------------------|---------|
| Si | 532 nm | Good signal, minimal heating |
| GaN | 325 nm | Resonance enhancement |
| Graphene | 532 nm | Standard for D/G ratio |
| Organic | 785 nm | Reduced fluorescence |

#### 2. Measurement Parameters
- **Power**: 0.1-10 mW (avoid damage)
- **Acquisition**: 1-100 s (balance S/N vs drift)
- **Grating**: 1800 gr/mm (0.5 cm⁻¹ resolution)
- **Confocal**: Yes for depth resolution

#### 3. Stress Analysis Example
```python
# Silicon stress measurement
reference_peak = 520.5  # cm⁻¹ (unstressed Si)
measured_peak = 522.3   # cm⁻¹

stress_result = calculate_stress(
    measured_position=measured_peak,
    reference_position=reference_peak,
    material='Si'
)

print(f"Stress: {stress_result['stress']:.2f} GPa")
print(f"Type: {stress_result['type']}")
```

#### 4. Crystallinity Evaluation
- **Sharp peaks**: High crystallinity
- **Broad peaks**: Amorphous or nanocrystalline
- **Peak shift**: Strain or composition change
- **New peaks**: Phase transformation

---

## API Reference

### Ellipsometry Endpoints

```http
POST /api/optical/advanced/ellipsometry/measure
{
  "sample_id": "uuid",
  "wavelength_start": 300,
  "wavelength_end": 800,
  "angle_of_incidence": 70
}
```

```http
POST /api/optical/advanced/ellipsometry/fit-model
{
  "measurement_id": "uuid",
  "layers": [
    {
      "thickness": 100,
      "model": "cauchy",
      "params": {"A": 1.46, "B": 0.003}
    }
  ],
  "fit_parameters": ["layer0_thickness"]
}
```

### PL Endpoints

```http
POST /api/optical/advanced/pl/measure
{
  "sample_id": "uuid",
  "excitation_wavelength": 532,
  "excitation_power": 10,
  "temperature": 10,
  "integration_time": 1.0
}
```

```http
POST /api/optical/advanced/pl/temperature-series
{
  "sample_id": "uuid",
  "temperatures": [10, 50, 100, 150, 200, 250, 300],
  "excitation_wavelength": 532
}
```

### Raman Endpoints

```http
POST /api/optical/advanced/raman/measure
{
  "sample_id": "uuid",
  "laser_wavelength": 532,
  "laser_power": 5,
  "acquisition_time": 10
}
```

```http
POST /api/optical/advanced/raman/stress-analysis
{
  "measurement_id": "uuid",
  "material": "Si",
  "reference_position": 520.5
}
```

---

## User Workflows

### Workflow 1: Determine SiO₂ Thickness on Si

1. **Mount sample** in ellipsometer
2. **Set angle** to 70°
3. **Measure** 400-800 nm spectrum
4. **Create model**:
   - Layer 1: SiO₂ (Cauchy)
   - Substrate: Si (tabulated)
5. **Fit thickness** as free parameter
6. **Validate**: MSE < 5, thickness uncertainty < 1%
7. **Report**: Thickness ± error

### Workflow 2: Analyze GaAs Material Quality

1. **Cool sample** to 10K
2. **Align** PL setup for maximum signal
3. **Measure** spectrum with low power
4. **Identify** peaks:
   - Band-edge emission
   - Donor-bound exciton
   - Acceptor-related
5. **Run** temperature series
6. **Extract**:
   - Eg(0) from Varshni fit
   - Activation energies
   - Defect concentrations
7. **Assess** material quality

### Workflow 3: Measure Stress in Si Device

1. **Focus** Raman on device region
2. **Acquire** spectrum with 532 nm laser
3. **Find** Si peak (~520.5 cm⁻¹)
4. **Measure** peak position precisely
5. **Calculate** stress from shift
6. **Map** if spatial distribution needed
7. **Correlate** with device performance

---

## Data Analysis

### Ellipsometry Data Processing

```python
# Complete analysis workflow
def analyze_ellipsometry(data: EllipsometryData):
    # 1. Initialize analyzer
    analyzer = EllipsometryAnalyzer()
    
    # 2. Create initial model
    stack = LayerStack(
        layers=[{'thickness': 100, 'model': DispersionModel.CAUCHY,
                'params': {'A': 1.46, 'B': 0.003, 'C': 0}}],
        substrate={'n': 3.85, 'k': 0.02}
    )
    
    # 3. Fit model
    result = analyzer.fit_model(
        data,
        stack,
        fit_parameters=['layer0_thickness', 'layer0_params_A']
    )
    
    # 4. Extract results
    thickness = result['parameters']['layer0_thickness']
    n_value = result['parameters']['layer0_params_A']
    mse = result['mse']
    
    # 5. Calculate confidence intervals
    # (Implementation depends on covariance matrix)
    
    return {
        'thickness': thickness,
        'refractive_index': n_value,
        'mse': mse,
        'quality': 'good' if mse < 5 else 'check'
    }
```

### PL Peak Analysis

```python
def analyze_pl_peaks(spectrum: PLSpectrum):
    # 1. Process spectrum
    processed = pl_analyzer.process_spectrum(spectrum)
    
    # 2. Find peaks
    peaks = pl_analyzer.find_peaks(processed)
    
    # 3. Fit peaks for accurate parameters
    fit_result = pl_analyzer.fit_peaks(processed, n_peaks=3)
    
    # 4. Identify emission types
    for peak in fit_result['peaks']:
        energy = peak['energy']
        
        if 1.41 < energy < 1.43:  # GaAs example
            peak['type'] = 'Band-edge'
        elif 1.48 < energy < 1.50:
            peak['type'] = 'Exciton'
        elif energy < 1.35:
            peak['type'] = 'Deep level'
    
    return fit_result
```

### Raman Stress Mapping

```python
def create_stress_map(raman_map_data):
    # 1. Analyze each spectrum
    nx, ny, _ = raman_map_data.shape
    stress_map = np.zeros((nx, ny))
    
    for i in range(nx):
        for j in range(ny):
            spectrum = RamanSpectrum(
                raman_shift=positions,
                intensity=raman_map_data[i, j, :]
            )
            
            # Find Si peak
            peaks = raman_analyzer.find_peaks(spectrum)
            si_peak = peaks['positions'][0]  # Assuming Si is main peak
            
            # Calculate stress
            stress = raman_analyzer.calculate_stress(
                si_peak, 520.5, 'Si'
            )
            stress_map[i, j] = stress['stress']
    
    # 2. Statistics
    return {
        'stress_map': stress_map,
        'mean_stress': np.mean(stress_map),
        'std_stress': np.std(stress_map),
        'max_stress': np.max(np.abs(stress_map))
    }
```

---

## Troubleshooting

### Ellipsometry Issues

| Problem | Possible Causes | Solutions |
|---------|----------------|-----------|
| Poor fit (high MSE) | Wrong model, incorrect starting values | Try different dispersion model, adjust initial parameters |
| Oscillating fit | Parameter correlation | Fix some parameters, use constraints |
| Thickness > expected | Back reflection | Use absorbing backing, model substrate backside |
| Negative thickness | Phase ambiguity | Add thickness constraint, check Δ unwrapping |

### PL Issues

| Problem | Possible Causes | Solutions |
|---------|----------------|-----------|
| No signal | Misalignment, wrong laser, sample issue | Check alignment, verify laser wavelength, test known sample |
| Broad peaks | High temperature, poor quality | Cool sample, reduce laser power |
| Unexpected peaks | Contamination, substrate | Clean sample, use band-pass filter |
| Intensity fluctuations | Temperature drift, laser instability | Stabilize temperature, check laser |

### Raman Issues

| Problem | Possible Causes | Solutions |
|---------|----------------|-----------|
| High fluorescence | Organic contamination, defects | Use longer wavelength laser (785, 1064 nm) |
| No peaks | Wrong focus, low signal | Optimize focus, increase power/time |
| Peak shift | Temperature, stress, calibration | Control temperature, check calibration |
| Broad peaks | Damage, poor crystallinity | Reduce laser power, check sample quality |

---

## Performance Specifications

### Ellipsometry Performance

| Parameter | Specification | Typical Value |
|-----------|--------------|---------------|
| Wavelength Range | 190-3300 nm | 245-1700 nm |
| Angle Range | 20°-90° | 45°-90° |
| Ψ Accuracy | ±0.02° | ±0.01° |
| Δ Accuracy | ±0.04° | ±0.02° |
| Thickness Resolution | 0.01 nm | 0.1 nm |
| Measurement Time | 5-60 s | 10 s |
| Spot Size | 25 μm - 5 mm | 100 μm |

### PL Performance

| Parameter | Specification | Typical Value |
|-----------|--------------|---------------|
| Spectral Range | 200-1700 nm | 350-1000 nm |
| Spectral Resolution | 0.01-1 nm | 0.1 nm |
| Temperature Range | 4-500 K | 10-300 K |
| Temperature Stability | ±10 mK | ±50 mK |
| Detection Limit | Single photon | 100 photons/s |
| Spatial Resolution | 0.5-50 μm | 2 μm |
| Time Resolution | ps-ns (TRPL) | - |

### Raman Performance

| Parameter | Specification | Typical Value |
|-----------|--------------|---------------|
| Spectral Range | 10-9000 cm⁻¹ | 50-4000 cm⁻¹ |
| Spectral Resolution | 0.1-10 cm⁻¹ | 1 cm⁻¹ |
| Spatial Resolution | 0.5-10 μm | 1 μm |
| Depth Resolution | 1-10 μm | 2 μm (confocal) |
| Detection Limit | Single molecule | 1% concentration |
| Mapping Speed | 0.1-10 s/point | 1 s/point |
| Stress Sensitivity | 10 MPa | 50 MPa |

---

## Safety & Compliance

### Laser Safety

#### Classification
- **Class 2**: Visible lasers < 1 mW (safe for accidental exposure)
- **Class 3B**: 5-500 mW (eye hazard, skin safe)
- **Class 4**: > 500 mW (eye and skin hazard, fire risk)

#### Safety Measures
1. **Engineering Controls**
   - Interlocked enclosures
   - Beam stops and attenuators
   - Warning lights

2. **Administrative Controls**
   - Trained operators only
   - Standard operating procedures
   - Regular safety audits

3. **Personal Protective Equipment**
   - Laser safety goggles (OD appropriate for wavelength)
   - No reflective jewelry
   - Appropriate clothing

### Cryogenic Safety (for PL)

1. **Liquid Nitrogen Handling**
   - Insulated gloves
   - Face shield
   - Closed-toe shoes
   - Well-ventilated area

2. **System Operation**
   - Slow cool-down/warm-up
   - Monitor pressure relief
   - Check for ice blockages

### Chemical Safety

1. **Sample Handling**
   - Some semiconductors toxic (GaAs, CdTe)
   - Use gloves
   - Proper disposal

2. **Cleaning Solvents**
   - Fume hood use
   - Appropriate PPE
   - Waste disposal protocols

---

## Appendix A: Common Materials Database

### Optical Constants (@ 633 nm)

| Material | n | k | Notes |
|----------|---|---|-------|
| Si | 3.85 | 0.02 | Crystalline |
| GaAs | 3.85 | 0.19 | @ 1.96 eV |
| SiO₂ | 1.46 | 0 | Thermal oxide |
| Si₃N₄ | 2.00 | 0 | LPCVD |
| Al₂O₃ | 1.77 | 0 | ALD |
| TiO₂ | 2.51 | 0 | Anatase |
| Au | 0.18 | 3.4 | Bulk |
| Al | 1.44 | 7.6 | Fresh |

### PL Peak Positions

| Material | Peak (nm) | Energy (eV) | Temperature | Type |
|----------|-----------|-------------|-------------|------|
| GaAs | 870 | 1.425 | 10K | Band-edge |
| GaN | 365 | 3.40 | 300K | Band-edge |
| InP | 920 | 1.35 | 300K | Band-edge |
| CdTe | 830 | 1.49 | 300K | Band-edge |
| Si | 1130 | 1.10 | 77K | Band-edge |
| ZnO | 380 | 3.26 | 300K | Band-edge |

### Raman Frequencies

| Material | Mode | Position (cm⁻¹) | FWHM (cm⁻¹) |
|----------|------|-----------------|--------------|
| Si | TO/LO | 520.5 | 3-4 |
| Ge | TO/LO | 300 | 4-5 |
| GaAs | TO | 268 | 3-4 |
| GaAs | LO | 292 | 3-4 |
| GaN | E2(high) | 568 | 4-6 |
| Diamond | sp³ C-C | 1332 | 1-2 |
| Graphene | G | 1580 | 10-20 |
| Graphene | 2D | 2700 | 20-40 |

---

## Appendix B: Fitting Guidelines

### Ellipsometry Fitting Strategy

1. **Start Simple**
   - Single layer, fix substrate
   - Fit thickness only
   - Add complexity gradually

2. **Parameter Selection**
   - Max 1 parameter per 10 data points
   - Avoid correlated parameters
   - Use physical constraints

3. **Multi-Sample Analysis**
   - Link common parameters
   - Global fit for consistency
   - Validate on different samples

### PL Fitting Functions

```python
# Gaussian peak
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma)**2)

# Lorentzian peak  
def lorentzian(x, amplitude, center, gamma):
    return amplitude * gamma**2 / ((x - center)**2 + gamma**2)

# Voigt profile (convolution of Gaussian and Lorentzian)
def voigt(x, amplitude, center, sigma, gamma):
    z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
```

---

## Appendix C: Quality Control Procedures

### Daily Checks

1. **Ellipsometer**
   - Straight-through (no sample): Ψ = 45°, Δ = 0°
   - Standard sample (25 nm SiO₂/Si)
   - Compare to reference: ±0.5 nm

2. **PL System**
   - Laser power stability: <2% variation
   - Wavelength calibration: Hg lamp lines
   - Detector dark counts: <100 cps

3. **Raman System**
   - Si reference: 520.5 ±0.5 cm⁻¹
   - Laser wavelength: ±0.1 nm
   - Focus reproducibility: ±1 μm

### Monthly Calibration

1. **Full wavelength calibration**
2. **Detector response correction**
3. **Polarizer alignment (ellipsometer)**
4. **Temperature sensor calibration**
5. **Stage position accuracy**

---

**Document Version:** 1.0.0  
**Last Updated:** October 2024  
**Session:** 8 - Optical Methods II  
**Status:** Complete
