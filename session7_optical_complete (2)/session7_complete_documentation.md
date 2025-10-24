# Session 7: Optical Methods I - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Theory & Physics](#theory--physics)
3. [Implementation Details](#implementation-details)
4. [API Reference](#api-reference)
5. [User Guide](#user-guide)
6. [Troubleshooting](#troubleshooting)
7. [Validation & QC](#validation--qc)
8. [Safety Protocols](#safety-protocols)

---

## Overview

Session 7 implements comprehensive optical spectroscopy methods for semiconductor characterization, including UV-Vis-NIR absorption/transmission spectroscopy and Fourier Transform Infrared (FTIR) spectroscopy. These techniques provide critical information about optical properties, bandgaps, film thickness, and material composition.

### Key Capabilities

#### UV-Vis-NIR Spectroscopy
- **Wavelength Range**: 190-3300 nm (typical)
- **Measurements**: Transmission, Absorption, Reflectance
- **Analysis**: Tauc plots, Bandgap determination, Film thickness
- **Materials**: Direct & indirect semiconductors
- **Accuracy**: <50 meV bandgap determination

#### FTIR Spectroscopy
- **Wavenumber Range**: 400-4000 cm⁻¹ (typical)
- **Resolution**: 0.5-16 cm⁻¹
- **Analysis**: Peak identification, Functional groups, Film thickness
- **Methods**: Transmission, ATR, Diffuse reflectance
- **Database**: 50+ semiconductor-relevant peaks

---

## Theory & Physics

### UV-Vis-NIR Spectroscopy

#### Beer-Lambert Law
The fundamental relationship governing optical absorption:

```
A = εcl = -log₁₀(T)
```

Where:
- A = Absorbance
- ε = Molar extinction coefficient
- c = Concentration
- l = Path length
- T = Transmittance

#### Absorption Coefficient
For semiconductors, we use the absorption coefficient α:

```
α = (2.303 × A) / d
```

Where d is the sample thickness in cm.

#### Tauc Analysis for Bandgap Determination

The Tauc relation connects the absorption coefficient to the optical bandgap:

```
(αhν)ⁿ = A(hν - Eₘ)
```

Where:
- α = Absorption coefficient
- h = Planck's constant
- ν = Frequency
- n = Transition exponent
- Eₘ = Optical bandgap
- A = Constant

Transition exponents:
- n = 2: Direct allowed transitions
- n = 2/3: Direct forbidden transitions
- n = 1/2: Indirect allowed transitions
- n = 1/3: Indirect forbidden transitions

### FTIR Spectroscopy

#### Vibrational Modes
FTIR detects molecular vibrations based on dipole moment changes:

```
ν = (1/2πc) × √(k/μ)
```

Where:
- ν = Wavenumber (cm⁻¹)
- c = Speed of light
- k = Force constant
- μ = Reduced mass

#### Common Semiconductor Peaks

| Bond | Position (cm⁻¹) | Type | Typical Material |
|------|-----------------|------|------------------|
| Si-O | 1050-1150 | Stretching | SiO₂ |
| Si-H | 2000-2200 | Stretching | a-Si:H |
| Si-N | 800-880 | Stretching | Si₃N₄ |
| Ga-As | 250-290 | Phonon | GaAs |
| In-P | 320-360 | Phonon | InP |
| C-H | 2800-3000 | Stretching | Organics |
| O-H | 3200-3600 | Stretching | H₂O, OH groups |

#### Film Thickness from Interference Fringes

For thin films showing interference fringes:

```
d = m / (2n × Δν)
```

Where:
- d = Film thickness
- m = Order difference between fringes
- n = Refractive index
- Δν = Wavenumber difference between fringes

---

## Implementation Details

### Architecture

```
session7_optical/
├── backend/
│   ├── core.py              # Main analysis modules
│   ├── api.py               # FastAPI endpoints
│   ├── models.py            # Database models
│   └── utils.py             # Helper functions
├── frontend/
│   ├── OpticalInterface.tsx # React UI components
│   ├── UVVisComponent.tsx   # UV-Vis-NIR interface
│   └── FTIRComponent.tsx    # FTIR interface
├── tests/
│   ├── test_uvvis.py        # UV-Vis tests
│   ├── test_ftir.py         # FTIR tests
│   └── test_integration.py  # Integration tests
└── data/
    ├── calibration/         # Calibration files
    └── references/          # Reference spectra
```

### Core Classes

#### UVVisNIRAnalyzer
```python
class UVVisNIRAnalyzer:
    def process_spectrum(data, smooth=True, baseline_correct=True)
    def calculate_absorption(transmission, reference=None)
    def calculate_absorption_coefficient(absorption, thickness)
    def tauc_analysis(data, thickness, bandgap_type, energy_range=None)
```

#### FTIRAnalyzer
```python
class FTIRAnalyzer:
    def process_ftir_spectrum(data, baseline_method, smooth=True)
    def find_peaks(data, prominence, distance, identify=True)
    def fit_peaks(data, peak_type='lorentzian', max_peaks=10)
    def calculate_film_thickness(data, n_substrate, angle=0)
```

### Data Models

#### SpectralData
```python
@dataclass
class SpectralData:
    wavelength: np.ndarray  # nm or cm⁻¹
    intensity: np.ndarray   # a.u. or %
    measurement_type: MeasurementType
    metadata: Dict[str, Any]
```

#### TaucResult
```python
@dataclass
class TaucResult:
    bandgap: float         # eV
    bandgap_error: float   # eV
    r_squared: float
    tauc_data: np.ndarray
    fit_data: np.ndarray
    bandgap_type: BandgapType
```

---

## API Reference

### Endpoints

#### UV-Vis-NIR

**Upload Spectrum**
```http
POST /api/optical/uvvis/upload
Content-Type: application/json

{
  "wavelength": [300, 301, 302, ...],
  "intensity": [90.5, 89.2, 88.1, ...],
  "measurement_type": "transmission",
  "metadata": {
    "sample_id": "GaAs-001",
    "thickness_mm": 0.5
  }
}
```

**Tauc Analysis**
```http
POST /api/optical/uvvis/tauc-analysis
Content-Type: application/json

{
  "spectrum_id": "uuid-here",
  "thickness_mm": 0.5,
  "bandgap_type": "direct_allowed",
  "energy_range_min": 1.0,
  "energy_range_max": 2.0
}
```

#### FTIR

**Upload Spectrum**
```http
POST /api/optical/ftir/upload
Content-Type: application/json

{
  "wavenumber": [400, 401, 402, ...],
  "transmittance": [95.2, 94.8, 94.5, ...],
  "measurement_type": "transmission",
  "metadata": {
    "sample_id": "SiO2-TF-001",
    "sample_type": "thin_film"
  }
}
```

**Peak Analysis**
```http
POST /api/optical/ftir/peaks
Content-Type: application/json

{
  "spectrum_id": "uuid-here",
  "prominence": 0.01,
  "distance": 10,
  "peak_type": "lorentzian",
  "max_peaks": 10
}
```

### Response Formats

**Tauc Analysis Response**
```json
{
  "bandgap": 1.42,
  "bandgap_error": 0.01,
  "r_squared": 0.995,
  "fit_range": [1.3, 1.6],
  "bandgap_type": "direct_allowed",
  "absorption_edge_nm": 873
}
```

**Peak Analysis Response**
```json
{
  "peaks": [
    {
      "position": 1080,
      "intensity": 30,
      "width": 50,
      "area": 1500,
      "assignment": "Si-O stretching"
    }
  ],
  "baseline": {...},
  "r_squared": 0.98,
  "film_thickness_um": 1.2
}
```

---

## User Guide

### UV-Vis-NIR Workflow

1. **Sample Preparation**
   - Clean sample surfaces
   - Measure thickness (critical for absorption coefficient)
   - Note sample orientation

2. **Measurement Setup**
   - Select wavelength range (typically 300-1100 nm for semiconductors)
   - Choose appropriate reference (air, substrate, or reference sample)
   - Set integration time for optimal S/N ratio
   - Average multiple scans if needed

3. **Data Processing**
   - Apply baseline correction (rubberband or ALS recommended)
   - Smooth data if noisy (Savitzky-Golay filter)
   - Convert transmission to absorption if needed

4. **Tauc Analysis**
   - Select correct transition type:
     - Direct: GaAs, GaN, CdTe, InP
     - Indirect: Si, Ge
   - Identify linear region in Tauc plot
   - Extract bandgap from x-intercept
   - Verify R² > 0.99 for reliable fit

5. **Results Interpretation**
   - Compare bandgap with literature values
   - Check for sub-gap absorption (defects)
   - Analyze interference fringes for film quality

### FTIR Workflow

1. **Sample Preparation**
   - Ensure sample is IR-transparent or use appropriate technique
   - For thin films, use polished substrates
   - Purge with N₂ to remove H₂O and CO₂

2. **Measurement Parameters**
   - Resolution: 4 cm⁻¹ (standard), 1-2 cm⁻¹ (high-res)
   - Scans: 32-128 for good S/N
   - Apodization: Happ-Genzel or Blackman-Harris
   - Background: Measure before sample

3. **Peak Analysis**
   - Identify major peaks using database
   - Fit peaks with appropriate function
   - Calculate integrated areas for quantification
   - Look for contamination peaks (C-H, O-H)

4. **Film Thickness Determination**
   - Count interference fringes
   - Measure fringe spacing in cm⁻¹
   - Apply thickness formula with correct n
   - Verify with other methods if possible

---

## Troubleshooting

### Common Issues - UV-Vis-NIR

**Problem: Noisy spectrum**
- Solution: Increase integration time, average more scans, check lamp alignment

**Problem: Negative absorption values**
- Solution: Remeasure reference, check for stray light, verify sample alignment

**Problem: Poor Tauc fit (low R²)**
- Solution: Adjust fit range, check for correct transition type, verify sample quality

**Problem: Unexpected bandgap value**
- Solution: Verify thickness measurement, check for substrate contribution, confirm material phase

### Common Issues - FTIR

**Problem: No peaks detected**
- Solution: Check sample thickness, verify IR transparency, increase resolution

**Problem: Baseline drift**
- Solution: Purge longer, check for temperature stability, apply baseline correction

**Problem: Peak identification errors**
- Solution: Verify wavenumber calibration, check for peak overlaps, consult literature

**Problem: Inconsistent film thickness**
- Solution: Verify refractive index, check fringe quality, ensure uniform film

---

## Validation & QC

### UV-Vis-NIR Validation

1. **Wavelength Accuracy**
   - Use holmium oxide filter
   - Check peaks at 361, 451, 536, 641 nm
   - Accuracy: ±0.5 nm

2. **Photometric Accuracy**
   - Use NIST SRM 930e filters
   - Verify at multiple wavelengths
   - Accuracy: ±0.5% T

3. **Bandgap Standards**
   - GaAs: 1.42 ± 0.01 eV
   - Si: 1.12 ± 0.01 eV (indirect)
   - GaN: 3.4 ± 0.05 eV

### FTIR Validation

1. **Wavenumber Accuracy**
   - Use polystyrene film
   - Check peaks at 3027, 2850, 1601, 1028 cm⁻¹
   - Accuracy: ±2 cm⁻¹

2. **Resolution Test**
   - Measure H₂O vapor lines
   - Verify specified resolution
   - Check for proper apodization

3. **Thickness Standards**
   - Use certified Si wafers with SiO₂
   - Compare with ellipsometry
   - Accuracy: ±5%

### Quality Control Metrics

| Parameter | UV-Vis-NIR | FTIR |
|-----------|------------|------|
| S/N Ratio | >100:1 | >1000:1 |
| Baseline Stability | <0.001 A/hr | <0.1% T/hr |
| Wavelength Repeatability | ±0.05 nm | ±0.5 cm⁻¹ |
| Photometric Repeatability | ±0.1% | ±0.2% |

---

## Safety Protocols

### UV-Vis-NIR Safety

1. **UV Radiation**
   - Never look directly at UV source
   - Use UV-blocking safety glasses
   - Keep sample compartment closed during measurement

2. **Deuterium Lamp**
   - Contains deuterium gas under pressure
   - Allow cooling before replacement
   - Handle with care to avoid breakage

### FTIR Safety

1. **Laser Radiation**
   - Class 2 HeNe laser for alignment
   - Do not stare into beam
   - Post laser warning signs

2. **Liquid Nitrogen (for MCT detector)**
   - Use appropriate PPE (gloves, face shield)
   - Ensure adequate ventilation
   - Follow cryogenic handling procedures

3. **Sample Handling**
   - Some semiconductors are toxic (CdTe, GaAs)
   - Use gloves when handling
   - Dispose according to regulations

### General Laboratory Safety

1. **Electrical Safety**
   - Ensure proper grounding
   - Check cables for damage
   - Follow lockout/tagout procedures

2. **Chemical Safety**
   - Use fume hood for solvent cleaning
   - Proper disposal of contaminated materials
   - Keep MSDS sheets accessible

3. **Emergency Procedures**
   - Know location of safety shower/eyewash
   - Have emergency contacts posted
   - Regular safety training required

---

## Performance Specifications

### UV-Vis-NIR Performance

| Specification | Value |
|---------------|-------|
| Wavelength Range | 190-3300 nm |
| Wavelength Accuracy | ±0.3 nm |
| Wavelength Repeatability | ±0.05 nm |
| Photometric Range | -4 to 4 A |
| Photometric Accuracy | ±0.003 A (at 1 A) |
| Stray Light | <0.00005% T |
| Baseline Stability | <0.0003 A/hr |
| Scan Speed | 0.5-8000 nm/min |

### FTIR Performance

| Specification | Value |
|---------------|-------|
| Wavenumber Range | 350-7800 cm⁻¹ |
| Resolution | 0.5-64 cm⁻¹ |
| Wavenumber Accuracy | ±0.01 cm⁻¹ |
| S/N Ratio | >45,000:1 (1 min, 4 cm⁻¹) |
| Scan Speed | 20 spectra/sec at 16 cm⁻¹ |
| Baseline Stability | <0.01% T/hr |

---

## Maintenance Schedule

### Daily
- Check N₂ purge flow (FTIR)
- Verify lamp hours (UV-Vis)
- Clean sample compartments

### Weekly
- Run validation standards
- Check and record performance metrics
- Clean optical surfaces with lens tissue

### Monthly
- Full system calibration
- Replace desiccant (FTIR)
- Check and clean detectors

### Annually
- Replace UV/Vis lamps as needed
- Service vacuum pump (if applicable)
- Professional alignment check
- Update firmware/software

---

## References

1. Tauc, J. (1968). "Optical properties and electronic structure of amorphous Ge and Si." Materials Research Bulletin, 3(1), 37-46.

2. Makuła, P., Pacia, M., & Macyk, W. (2018). "How to correctly determine the band gap energy of modified semiconductor photocatalysts based on UV–Vis spectra." Journal of Physical Chemistry Letters, 9(23), 6814-6817.

3. Griffiths, P. R., & De Haseth, J. A. (2007). "Fourier transform infrared spectrometry" (Vol. 171). John Wiley & Sons.

4. Smith, B. C. (2011). "Fundamentals of Fourier transform infrared spectroscopy." CRC press.

5. Swanepoel, R. (1983). "Determination of the thickness and optical constants of amorphous silicon." Journal of Physics E: Scientific Instruments, 16(12), 1214.

---

## Appendix A: Material Database

### Direct Bandgap Semiconductors

| Material | Bandgap (eV) | λ edge (nm) | Applications |
|----------|--------------|-------------|--------------|
| GaN | 3.4 | 365 | Blue LEDs, Power devices |
| GaAs | 1.42 | 873 | Solar cells, Lasers |
| InP | 1.35 | 918 | High-speed electronics |
| CdTe | 1.5 | 827 | Solar cells |
| ZnO | 3.3 | 376 | Transparent electronics |

### Indirect Bandgap Semiconductors

| Material | Bandgap (eV) | λ edge (nm) | Applications |
|----------|--------------|-------------|--------------|
| Si | 1.12 | 1107 | Microelectronics |
| Ge | 0.66 | 1879 | IR detectors |
| GaP | 2.26 | 549 | Green LEDs |
| AlAs | 2.16 | 574 | Heterostructures |

---

## Appendix B: FTIR Peak Assignments

### Silicon-based Materials

| Wavenumber (cm⁻¹) | Assignment | Material |
|-------------------|------------|----------|
| 460 | Si-O rocking | SiO₂ |
| 800 | Si-O bending | SiO₂ |
| 1080 | Si-O stretching | SiO₂ |
| 610 | Si-Si | c-Si |
| 2090 | Si-H stretching | a-Si:H |
| 840 | Si-N stretching | Si₃N₄ |
| 2160 | Si-H stretching | Si₃N₄:H |

### III-V Semiconductors

| Wavenumber (cm⁻¹) | Assignment | Material |
|-------------------|------------|----------|
| 268 | GaAs TO phonon | GaAs |
| 292 | GaAs LO phonon | GaAs |
| 345 | InP TO phonon | InP |
| 370 | InP LO phonon | InP |
| 532 | GaN E2 phonon | GaN |
| 734 | GaN A1(LO) | GaN |

---

## Appendix C: Troubleshooting Decision Tree

```
Spectrum Quality Issue
├── Noise Problems
│   ├── High Frequency → Increase averaging
│   ├── Low Frequency → Check vibrations
│   └── Random → Check detector cooling
├── Baseline Issues
│   ├── Drift → Temperature stabilization
│   ├── Curvature → Alignment check
│   └── Offset → Reference measurement
├── Peak Issues
│   ├── Missing → Check resolution
│   ├── Shifted → Calibration needed
│   └── Broadened → Sample quality
└── Intensity Issues
    ├── Too Low → Check source/alignment
    ├── Too High → Reduce slit/aperture
    └── Saturated → Use filters/reduce time
```

---

**Document Version:** 1.0.0  
**Last Updated:** October 2024  
**Session:** 7 - Optical Methods I  
**Status:** Complete
