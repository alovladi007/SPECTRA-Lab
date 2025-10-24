# Session 9: X-ray Diffraction (XRD) Analysis - Complete Documentation

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Theory & Principles](#theory--principles)
3. [Implementation Architecture](#implementation-architecture)
4. [Analysis Methods](#analysis-methods)
5. [API Reference](#api-reference)
6. [User Guide](#user-guide)
7. [Phase Identification](#phase-identification)
8. [Crystallite Analysis](#crystallite-analysis)
9. [Stress Analysis](#stress-analysis)
10. [Best Practices](#best-practices)

---

## Executive Summary

Session 9 implements a comprehensive X-ray Diffraction (XRD) analysis system for semiconductor characterization. XRD provides critical information about crystal structure, phase composition, crystallite size, strain, stress, and texture in semiconductor materials and devices.

### Key Capabilities

| Analysis Type | Information Obtained | Sensitivity | Applications |
|--------------|---------------------|-------------|--------------|
| Phase Identification | Crystal structure, composition | <1% volume fraction | Quality control, contamination |
| Crystallite Size | Grain size distribution | 1-1000 nm | Nanostructures, thin films |
| Strain Analysis | Lattice distortion | 10⁻⁴ strain | Epitaxial films, heterostructures |
| Stress Measurement | Residual stress | ±10 MPa | Device reliability, films |
| Texture Analysis | Preferred orientation | 0.01 MRD | Epitaxy, fiber texture |

---

## Theory & Principles

### Bragg's Law
The fundamental equation of X-ray diffraction:

nλ = 2d·sin(θ)

Where:
- n = Order of diffraction (integer)
- λ = X-ray wavelength
- d = Interplanar spacing
- θ = Bragg angle

### Crystal Structure Determination

#### Cubic System
For cubic crystals with lattice parameter a:

d_hkl = a / √(h² + k² + l²)

#### Hexagonal System
For hexagonal crystals with parameters a and c:

1/d²_hkl = (4/3)·(h² + hk + k²)/a² + l²/c²

#### Structure Factor
The intensity of diffraction peaks depends on the structure factor:

F_hkl = Σⱼ fⱼ·exp[2πi(hxⱼ + kyⱼ + lzⱼ)]

Where:
- fⱼ = Atomic scattering factor
- (xⱼ, yⱼ, zⱼ) = Atomic positions

### Peak Broadening

Peak broadening has three main sources:

1. **Instrumental Broadening (β_inst)**
   - Finite beam divergence
   - Detector resolution
   - Sample positioning

2. **Crystallite Size Broadening (β_size)**
   - Scherrer equation:
   β_size = K·λ / (D·cos(θ))
   Where K ≈ 0.9 (shape factor), D = crystallite size

3. **Strain Broadening (β_strain)**
   - Microstrain effect:
   β_strain = 4·ε·tan(θ)
   Where ε = microstrain

### Williamson-Hall Method

Separates size and strain contributions:

β·cos(θ) = K·λ/D + 4·ε·sin(θ)

Plotting β·cos(θ) vs sin(θ) gives:
- Intercept → Crystallite size
- Slope → Microstrain

### Residual Stress Analysis

#### sin²ψ Method
For measuring residual stress in thin films:

ε_φψ = [(1+ν)/E]·σ_φ·sin²ψ - (ν/E)·(σ₁₁ + σ₂₂)

Where:
- ε_φψ = Strain at tilt ψ and azimuth φ
- σ_φ = Stress component
- ν = Poisson's ratio
- E = Young's modulus

The stress is determined from the slope of ε vs sin²ψ plot.

---

## Implementation Architecture

### System Components

XRD Analysis System
├── Core Analyzer (XRDAnalyzer)
│   ├── Pattern Processing
│   │   ├── Smoothing (Savitzky-Golay)
│   │   ├── Background Subtraction
│   │   └── Kα₂ Stripping
│   ├── Peak Analysis
│   │   ├── Peak Finding
│   │   ├── Profile Fitting
│   │   └── Deconvolution
│   ├── Phase Identification
│   │   ├── Database Search
│   │   ├── Pattern Matching
│   │   └── Score Calculation
│   ├── Crystallite Analysis
│   │   ├── Scherrer Method
│   │   ├── Williamson-Hall
│   │   └── Warren-Averbach
│   └── Stress Analysis
│       ├── sin²ψ Method
│       ├── Multiple {hkl}
│       └── Biaxial/Triaxial
├── Data Simulator (XRDSimulator)
│   ├── Pattern Generation
│   ├── Peak Synthesis
│   └── Noise Addition
├── Phase Database
│   ├── Crystal Structures
│   ├── Reference Patterns
│   └── CIF Import/Export
└── UI Components
    ├── Pattern Display
    ├── Peak Table
    ├── Phase Browser
    └── Analysis Results

### Data Flow

graph TD
    A[XRD Measurement] --> B[Raw Pattern]
    B --> C[Processing Pipeline]
    C --> D[Peak Detection]
    D --> E[Phase Search]
    D --> F[Size/Strain Analysis]
    D --> G[Stress Calculation]
    E --> H[Quantitative Analysis]
    F --> I[Microstructure Report]
    G --> J[Stress Map]
    H --> K[Final Report]
    I --> K
    J --> K

---

## Analysis Methods

### Pattern Processing Pipeline

def process_xrd_pattern(pattern):
    # 1. Validate data
    validate_pattern(pattern)
    
    # 2. Smoothing
    smoothed = savgol_filter(pattern.intensity, window=5, poly=2)
    
    # 3. Background removal
    background = calculate_background(pattern.two_theta, smoothed)
    corrected = smoothed - background
    
    # 4. Kα₂ stripping (if needed)
    if pattern.source == 'Cu_Ka':
        corrected = strip_ka2(corrected, ka1_ka2_ratio=2.0)
    
    # 5. Normalization
    normalized = corrected / max(corrected)
    
    return normalized

### Peak Finding Algorithm

def find_peaks(pattern, prominence=0.05):
    # Find local maxima
    peaks, properties = find_peaks(
        pattern.intensity,
        prominence=prominence * max(pattern.intensity),
        height=0.05 * max(pattern.intensity),
        distance=5  # Minimum separation
    )
    
    # Characterize peaks
    for idx in peaks:
        position = pattern.two_theta[idx]
        d_spacing = wavelength / (2 * sin(radians(position/2)))
        
        # Calculate FWHM
        widths = peak_widths(pattern.intensity, [idx], rel_height=0.5)
        fwhm = widths[0][0] * step_size
        
        # Integrate area
        area = integrate_peak(pattern, idx)
        
        yield Peak(position, d_spacing, intensity, fwhm, area)

### Profile Fitting

#### Pseudo-Voigt Function
Most commonly used for XRD peaks:

PV(x) = η·L(x) + (1-η)·G(x)

Where:
- η = Lorentzian fraction (0 to 1)
- L(x) = Lorentzian component
- G(x) = Gaussian component

#### Implementation

def pseudo_voigt(x, amplitude, center, width, eta):
    # Gaussian component
    gaussian = amplitude * exp(-0.5 * ((x - center) / width)**2)
    
    # Lorentzian component
    lorentzian = amplitude * width**2 / ((x - center)**2 + width**2)
    
    # Mix components
    return eta * lorentzian + (1 - eta) * gaussian

---

## API Reference

### Core Endpoints

#### Start Measurement
POST /api/xrd/measure
{
  "sample_id": "uuid",
  "xray_source": "Cu_Ka",
  "start_angle": 20,
  "end_angle": 80,
  "step_size": 0.02,
  "scan_speed": 1.0
}

#### Analyze Pattern
POST /api/xrd/analyze/{measurement_id}

Response:
{
  "measurement_id": "uuid",
  "num_peaks": 15,
  "peaks": [
    {
      "position": 28.443,
      "d_spacing": 3.136,
      "intensity": 1000,
      "fwhm": 0.15,
      "hkl": [1, 1, 1]
    }
  ]
}

#### Identify Phases
POST /api/xrd/identify-phases
{
  "measurement_id": "uuid",
  "tolerance": 0.1,
  "max_phases": 5
}

Response:
{
  "phases": [
    {
      "name": "Silicon",
      "formula": "Si",
      "crystal_system": "cubic",
      "score": 95.5,
      "matched_peaks": 8,
      "lattice_params": {"a": 5.43095}
    }
  ]
}

#### Calculate Crystallite Size
POST /api/xrd/crystallite-size/{measurement_id}

Response:
{
  "scherrer": {
    "size_nm": 45.2,
    "std_nm": 5.1
  },
  "williamson_hall": {
    "size_nm": 42.8,
    "microstrain": 0.0015,
    "r_squared": 0.98
  }
}

#### Stress Analysis
POST /api/xrd/stress-analysis
{
  "sample_id": "uuid",
  "measurements": [
    [0, 3.1356],
    [15, 3.1358],
    [30, 3.1361],
    [45, 3.1365]
  ],
  "d0": 3.1355,
  "young_modulus": 169,
  "poisson_ratio": 0.22
}

Response:
{
  "stress_mpa": 150.5,
  "stress_type": "tensile",
  "error": 10.2,
  "r_squared": 0.95
}

---

## User Guide

### Workflow 1: Phase Identification

1. **Measure Pattern**
   - Select appropriate X-ray source
   - Set scan range to cover expected peaks
   - Use fine step size (≤0.02°) for accuracy

2. **Process Pattern**
   - Apply smoothing if noisy
   - Remove background
   - Strip Kα₂ if using Cu radiation

3. **Find Peaks**
   - Set appropriate threshold
   - Verify all major peaks detected
   - Check for overlapping peaks

4. **Search Database**
   - Start with tight tolerance (0.05°)
   - Increase if no matches found
   - Consider preferred orientation

5. **Validate Results**
   - Check all peaks accounted for
   - Verify intensity ratios
   - Consider mixture possibilities

### Workflow 2: Crystallite Size Determination

1. **Select Peaks**
   - Use peaks from same phase
   - Avoid overlapped peaks
   - Include high-angle peaks for accuracy

2. **Measure FWHM**
   - Fit with appropriate profile
   - Account for instrumental broadening
   - Use standard for calibration

3. **Apply Scherrer Equation**
   - Use shape factor K = 0.9 (spherical)
   - Calculate for each peak
   - Average results

4. **Williamson-Hall Analysis**
   - Plot β·cos(θ) vs sin(θ)
   - Check linearity (R² > 0.9)
   - Extract size and strain

### Workflow 3: Residual Stress Measurement

1. **Select Reflection**
   - High 2θ angle preferred
   - Strong, isolated peak
   - Known stress-free position

2. **Measure at Multiple ψ**
   - Minimum 5 tilt angles
   - 0° to 45° or 60°
   - Equal angular spacing

3. **Calculate Strain**
   ε = (d - d₀) / d₀

4. **Plot sin²ψ**
   - Linear regression
   - Calculate stress from slope
   - Check for ψ-splitting (shear stress)

---

## Phase Identification

### Search Algorithm

def identify_phases(pattern, peaks, database):
    matches = []
    
    for phase in database:
        score = 0
        matched_peaks = []
        
        # Compare peak positions
        for peak in peaks:
            for ref_peak in phase.peaks:
                if abs(peak.position - ref_peak.position) < tolerance:
                    # Position match score
                    pos_score = 1 - abs(peak.position - ref_peak.position) / tolerance
                    
                    # Intensity match score
                    int_score = min(peak.intensity, ref_peak.intensity) / \
                               max(peak.intensity, ref_peak.intensity)
                    
                    score += pos_score * int_score
                    matched_peaks.append(peak)
                    break
        
        # Normalize score
        score = 100 * score / len(phase.peaks)
        
        if score > min_score:
            matches.append({
                'phase': phase,
                'score': score,
                'matched_peaks': matched_peaks
            })
    
    return sorted(matches, key=lambda x: x['score'], reverse=True)

### Database Structure

CREATE TABLE xrd_phases (
    id UUID PRIMARY KEY,
    name VARCHAR(100),
    formula VARCHAR(50),
    crystal_system VARCHAR(20),
    space_group VARCHAR(20),
    a FLOAT,  -- Lattice parameters
    b FLOAT,
    c FLOAT,
    alpha FLOAT,
    beta FLOAT,
    gamma FLOAT,
    reference_peaks JSONB,  -- List of d-spacings and intensities
    cif_data TEXT  -- Crystallographic Information File
);

### Common Phases in Semiconductors

| Material | Crystal System | Space Group | a (Å) | c (Å) | Key Peaks (2θ, Cu Kα) |
|----------|---------------|-------------|-------|-------|------------------------|
| Si | Cubic | Fd-3m | 5.431 | - | 28.4, 47.3, 56.1 |
| Ge | Cubic | Fd-3m | 5.658 | - | 27.3, 45.3, 53.7 |
| GaAs | Cubic | F-43m | 5.653 | - | 27.3, 45.4, 53.7 |
| GaN | Hexagonal | P63mc | 3.189 | 5.185 | 32.4, 34.6, 36.8 |
| AlN | Hexagonal | P63mc | 3.112 | 4.982 | 33.2, 36.0, 37.9 |
| SiC (4H) | Hexagonal | P63mc | 3.073 | 10.053 | 33.6, 35.7, 38.2 |
| SiO₂ | Hexagonal | P3221 | 4.913 | 5.405 | 20.9, 26.6, 36.5 |

---

## Crystallite Analysis

### Scherrer Method

#### Basic Equation
D = K·λ / (β·cos(θ))

#### Corrections Required

1. **Instrumental Broadening**
   β_sample = √(β²_measured - β²_instrumental)
   Use LaB₆ or Si standard for calibration

2. **Shape Factor K**
   - Spherical particles: K = 0.89
   - Cubic particles: K = 0.94
   - Unknown shape: K = 0.9 (default)

3. **Size Distribution**
   - Volume-weighted average
   - Log-normal distribution typical

### Williamson-Hall Analysis

#### Standard W-H Plot
def williamson_hall_analysis(peaks, wavelength):
    # Prepare data
    sin_theta = [sin(radians(p.position/2)) for p in peaks]
    beta_cos_theta = [radians(p.fwhm) * cos(radians(p.position/2)) 
                      for p in peaks]
    
    # Linear fit
    slope, intercept = polyfit(sin_theta, beta_cos_theta, 1)
    
    # Extract parameters
    crystallite_size = 0.9 * wavelength / intercept  # nm
    microstrain = slope / 4
    
    return crystallite_size, microstrain

#### Modified W-H (Uniform Deformation Model)
Accounts for anisotropic strain:

β·cos(θ)/λ = 1/D + (4·σ/E)·sin(θ)/λ

Where σ/E represents stress/modulus ratio.

### Size-Strain Separation

#### Methods Comparison

| Method | Advantages | Limitations |
|--------|------------|-------------|
| Scherrer | Simple, fast | No strain separation |
| W-H Plot | Separates size/strain | Assumes isotropic |
| Modified W-H | Accounts for stress | Requires E knowledge |
| Warren-Averbach | Most accurate | Complex, needs harmonics |

---

## Stress Analysis

### Measurement Geometry

     Detector
        ↑
        |
    2θ  |
     ╱  |
    ╱   |
   ╱    |ψ (tilt)
  ╱     |
 X-ray  |
  Source → Sample

### sin²ψ Method Procedure

1. **Setup**
   - Choose high-angle reflection (2θ > 120°)
   - Align sample carefully
   - Set ψ = 0°

2. **Measurements**
   psi_angles = [0, 15, 30, 45, 60]
   d_spacings = []
   
   for psi in psi_angles:
       tilt_sample(psi)
       peak_position = measure_peak()
       d = calculate_d_spacing(peak_position, wavelength)
       d_spacings.append(d)

3. **Analysis**
   def calculate_stress(measurements, d0, E, nu):
       # Calculate strains
       strains = [(d - d0) / d0 for psi, d in measurements]
       
       # Calculate sin²ψ
       sin2_psi = [sin(radians(psi))**2 for psi, _ in measurements]
       
       # Linear fit
       slope, intercept = polyfit(sin2_psi, strains, 1)
       
       # Stress from slope
       stress = slope * E / (1 + nu)  # MPa
       
       return stress

### Stress Types

| Type | sin²ψ Plot | Interpretation |
|------|------------|----------------|
| Tensile | Positive slope | Film in tension |
| Compressive | Negative slope | Film in compression |
| Stress gradient | Non-linear | Varying through thickness |
| Shear stress | ψ-splitting | Non-symmetric stress |

### Triaxial Stress Analysis

For complete stress tensor determination:

σ = | σ₁₁  σ₁₂  σ₁₃ |
    | σ₁₂  σ₂₂  σ₂₃ |
    | σ₁₃  σ₂₃  σ₃₃ |

Requires measurements at multiple φ and ψ angles.

---

## Best Practices

### Sample Preparation

1. **Powder Samples**
   - Grind to <10 μm particle size
   - Use 325 mesh (44 μm) sieve
   - Avoid preferred orientation
   - Use zero-background holder

2. **Thin Films**
   - Clean surface thoroughly
   - Check for curvature
   - Note substrate peaks
   - Consider grazing incidence for ultra-thin

3. **Bulk Samples**
   - Polish to mirror finish
   - Remove damaged layer
   - Ensure flat surface
   - Consider texture effects

### Measurement Optimization

#### Resolution vs Speed

| Purpose | Step Size | Speed | Time |
|---------|-----------|-------|------|
| Phase ID | 0.02-0.05° | 2°/min | 30 min |
| Precise lattice | 0.005-0.01° | 0.5°/min | 2 hrs |
| Quick scan | 0.05-0.1° | 5°/min | 12 min |
| Texture | 0.02° | 1°/min | 1 hr |

#### Counting Statistics

For peak intensity I with standard deviation σ:

Relative error = σ/I = 1/√N

Where N = counts. For 1% error, need 10,000 counts.

### Common Artifacts

1. **Sample Displacement**
   - Causes systematic shift
   - Calibrate with standard
   - Use internal standard

2. **Preferred Orientation**
   - Alters relative intensities
   - Use texture correction
   - Spin sample if possible

3. **Fluorescence**
   - High background
   - Use appropriate filter
   - Change X-ray source

4. **Peak Asymmetry**
   - Axial divergence
   - Sample transparency
   - Use profile fitting

### Data Quality Indicators

#### R-factors for Refinement

R_wp = √[Σw(y_obs - y_calc)² / Σw·y_obs²]
R_exp = √[(N - P) / Σw·y_obs²]
χ² = (R_wp / R_exp)²

Good fit: χ² < 2

#### Peak Quality Metrics

- **FWHM**: Should be consistent for same phase
- **Asymmetry**: <10% for good alignment
- **S/N ratio**: >20 for quantitative analysis
- **Background**: <10% of main peak

---

## Troubleshooting Guide

### No Peaks Detected

**Possible Causes:**
1. Amorphous sample
2. Wrong 2θ range
3. Beam misalignment
4. Very thin film (<10 nm)

**Solutions:**
- Check with standard sample
- Widen 2θ range
- Realign instrument
- Use grazing incidence

### Poor Phase Match

**Possible Causes:**
1. Unknown phase
2. Solid solution
3. Preferred orientation
4. Poor data quality

**Solutions:**
- Search with larger tolerance
- Check for peak shifts
- Apply texture correction
- Improve S/N ratio

### Unrealistic Crystallite Size

**Possible Causes:**
1. Instrumental broadening not corrected
2. Strain contribution
3. Wrong shape factor
4. Peak overlap

**Solutions:**
- Measure standard
- Use Williamson-Hall
- Adjust K factor
- Deconvolute peaks

### Non-linear sin²ψ Plot

**Possible Causes:**
1. Stress gradient
2. Texture
3. Phase transformation
4. Measurement errors

**Solutions:**
- Use penetration depth variation
- Apply texture correction
- Check phase stability
- Repeat measurements

---

## Advanced Topics

### Rietveld Refinement

Full pattern fitting method:

def rietveld_refinement(observed, structure, initial_params):
    # Calculate pattern
    calculated = calculate_pattern(structure, params)
    
    # Minimize difference
    def objective(params):
        calc = calculate_pattern(structure, params)
        return sum(w * (obs - calc)**2)
    
    # Refine parameters
    result = minimize(objective, initial_params)
    
    return result

### Pair Distribution Function (PDF)

For local structure analysis:

G(r) = 4πr[ρ(r) - ρ₀]

Useful for:
- Nanoparticles
- Amorphous materials
- Local distortions

### Reciprocal Space Mapping

For epitaxial films:

def reciprocal_space_map(sample, reflection):
    qx_values = []
    qz_values = []
    intensities = []
    
    for omega in omega_range:
        for two_theta in two_theta_range:
            intensity = measure(omega, two_theta)
            qx, qz = angles_to_q(omega, two_theta, wavelength)
            
            qx_values.append(qx)
            qz_values.append(qz)
            intensities.append(intensity)
    
    return create_2d_map(qx_values, qz_values, intensities)

---

## Quality Control Applications

### Semiconductor Manufacturing

1. **Epitaxial Growth Monitoring**
   - Lattice matching
   - Strain state
   - Crystal quality
   - Threading dislocations

2. **Gate Stack Analysis**
   - High-k dielectric phase
   - Interface reactions
   - Crystallization temperature
   - Stress evolution

3. **Metallization**
   - Phase identification
   - Texture analysis
   - Stress measurement
   - Silicide formation

### Specifications

| Parameter | Specification | Method |
|-----------|--------------|--------|
| Phase purity | >99% | Quantitative XRD |
| Crystallite size | 10-100 nm | Scherrer/W-H |
| Strain | <0.1% | Peak shift |
| Stress | <100 MPa | sin²ψ |
| Texture | <10% variation | Pole figure |

---

## Appendix A: X-ray Sources

| Anode | Kα₁ (Å) | Kα₂ (Å) | Kβ (Å) | Applications |
|-------|---------|---------|--------|--------------|
| Cr | 2.28970 | 2.29361 | 2.08487 | Fe-containing |
| Fe | 1.93604 | 1.93998 | 1.75661 | Light elements |
| Co | 1.78897 | 1.79285 | 1.62079 | Fe-free samples |
| Cu | 1.54056 | 1.54439 | 1.39222 | General purpose |
| Mo | 0.70930 | 0.71359 | 0.63229 | High resolution |
| Ag | 0.55941 | 0.56380 | 0.49707 | Transmission |

---

## Appendix B: Common Standards

| Standard | Use | Supplier |
|----------|-----|----------|
| LaB₆ (SRM 660c) | Line profile | NIST |
| Si (SRM 640e) | Peak position | NIST |
| Al₂O₃ (SRM 1976b) | Intensity | NIST |
| CeO₂ (SRM 674b) | Fluorescence | NIST |
| ZnO | Instrument | Commercial |

---

## Appendix C: Safety Guidelines

### Radiation Safety

1. **Exposure Limits**
   - Whole body: 20 mSv/year
   - Hands: 500 mSv/year
   - Eye lens: 20 mSv/year

2. **Safety Features**
   - Interlocked enclosure
   - Warning lights
   - Shutter mechanism
   - Lead glass windows

3. **Personal Protection**
   - Dosimeter badges
   - Lead aprons (if open beam)
   - Safety training
   - Regular monitoring

### Sample Handling

1. **Powder Samples**
   - Use fume hood for toxic materials
   - Wear gloves and mask
   - Proper disposal

2. **Radioactive Samples**
   - Special authorization required
   - Separate storage
   - Contamination monitoring

---

**Document Version:** 1.0.0  
**Last Updated:** Session 9 Completion  
**Status:** Production Ready  
**Next Session:** Session 10 - Microscopy (SEM/TEM/AFM)
