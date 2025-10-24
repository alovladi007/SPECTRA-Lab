# Session 11: XPS/XRF Analysis - Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Theoretical Background](#theoretical-background)
3. [Implementation Details](#implementation-details)
4. [API Reference](#api-reference)
5. [User Guide](#user-guide)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Safety Guidelines](#safety-guidelines)

---

## Overview

### Purpose
The XPS/XRF Analysis module provides comprehensive surface and elemental analysis capabilities for semiconductor materials characterization. It enables chemical state identification, depth profiling, and quantitative elemental analysis.

### Key Features
- **XPS Analysis**
  - Chemical state identification
  - Depth profiling
  - Multiplet splitting analysis
  - Multiple background subtraction methods (Shirley, Tougaard)
  - Peak fitting with various profiles (Gaussian, Lorentzian, Voigt, Doniach-Sunjic)
  
- **XRF Analysis**
  - Elemental identification
  - Standardless quantification
  - Matrix corrections
  - Detection limit calculation
  - Escape and sum peak identification

### System Requirements
- Python 3.8+
- Node.js 14+
- 8GB RAM minimum
- 10GB free disk space

---

## Theoretical Background

### X-ray Photoelectron Spectroscopy (XPS)

#### Basic Principles
XPS is based on the photoelectric effect, where X-ray photons eject core electrons from atoms:

```
KE = hν - BE - φ
```

Where:
- KE = Kinetic energy of emitted electron
- hν = X-ray photon energy
- BE = Binding energy
- φ = Work function

#### Chemical Shifts
Chemical environment affects binding energies, enabling chemical state identification:
- Oxidation states cause systematic shifts
- Electronegativity differences influence peak positions
- Typical shifts: 1-5 eV

#### Quantification
Atomic concentration calculation:
```
Ci = (Ii/Si) / Σ(Ij/Sj) × 100%
```

Where:
- Ii = Peak area for element i
- Si = Sensitivity factor for element i

#### Background Subtraction

**Shirley Background:**
- Iterative calculation based on integrated intensity
- Suitable for metallic and semiconductor samples
- Accounts for inelastic scattering

**Tougaard Background:**
- Universal cross-section approach
- Better for insulators and polymers
- Parameters: B = 2866 eV²/eV, C = 1643 eV²

#### Peak Shapes

1. **Gaussian:** Instrumental broadening
   ```
   G(x) = A × exp(-(x-x₀)²/(2σ²))
   ```

2. **Lorentzian:** Natural line width
   ```
   L(x) = A × γ²/((x-x₀)² + γ²)
   ```

3. **Voigt:** Convolution of Gaussian and Lorentzian
   - Most accurate for XPS peaks
   - Accounts for both broadening mechanisms

4. **Doniach-Sunjic:** Asymmetric peaks in metals
   - Accounts for electron-hole pair excitation
   - Asymmetry parameter α (0-0.3)

### X-ray Fluorescence (XRF)

#### Basic Principles
XRF measures characteristic X-rays emitted when inner shell vacancies are filled:

1. Primary excitation creates inner shell vacancy
2. Outer electron fills vacancy
3. Characteristic X-ray emitted

#### Fundamental Parameters Method
Quantification without standards:
```
Ci = Ii / (σi × ωi × εi × Ai)
```

Where:
- σi = Photoionization cross-section
- ωi = Fluorescence yield
- εi = Detector efficiency
- Ai = Absorption correction

#### Matrix Effects
- **Absorption:** X-rays absorbed by matrix
- **Enhancement:** Secondary fluorescence
- **Corrections:** Sherman equation, fundamental parameters

#### Detection Limits
Minimum detectable concentration:
```
MDL = 3 × √(2B/t) / S
```

Where:
- B = Background count rate
- t = Measurement time
- S = Sensitivity

---

## Implementation Details

### Architecture

```
chemical/
├── analyzer.py           # Core analysis engines
│   ├── XPSAnalyzer      # XPS processing and fitting
│   ├── XRFAnalyzer      # XRF analysis
│   ├── ElementDatabase  # Element properties
│   └── ChemicalSimulator # Spectrum generation
├── api.py               # REST API endpoints
├── models.py            # Data models
└── utils.py             # Helper functions
```

### Core Classes

#### XPSAnalyzer
```python
class XPSAnalyzer:
    def __init__(self, source: XRaySource = XRaySource.AL_KA):
        self.source = source
        self.element_db = ElementDatabase()
        self.calibration = {'C1s': 284.5}
    
    def process_spectrum(self, be, intensity, smooth_window=5)
    def shirley_background(self, be, intensity, endpoints=None)
    def fit_peak(self, be, intensity, shape=PeakShape.VOIGT)
    def quantification(self, peaks)
    def depth_profile(self, spectra, etch_times, elements)
```

#### XRFAnalyzer
```python
class XRFAnalyzer:
    def __init__(self, excitation_energy: float = 50.0):
        self.excitation_energy = excitation_energy
        self.detector_resolution = 150  # eV at 5.9 keV
    
    def process_spectrum(self, energy, counts, smooth_window=5)
    def find_peaks(self, energy, counts, prominence=0.05)
    def quantification_fundamental_parameters(self, peaks, matrix='SiO2')
    def detection_limits(self, energy, counts, measurement_time=300)
```

### Data Processing Pipeline

#### XPS Workflow
1. **Data Import** → Load spectrum (BE, intensity)
2. **Calibration** → Align to reference peak (C 1s)
3. **Background** → Calculate and subtract
4. **Peak Finding** → Identify peaks automatically
5. **Peak Fitting** → Fit with appropriate profiles
6. **Quantification** → Calculate atomic %
7. **Reporting** → Generate analysis report

#### XRF Workflow
1. **Data Import** → Load spectrum (energy, counts)
2. **Dead Time** → Apply corrections
3. **Peak Search** → Find element lines
4. **Identification** → Match to database
5. **Quantification** → Fundamental parameters
6. **MDL** → Calculate detection limits
7. **Reporting** → Generate results

---

## API Reference

### Endpoints

#### POST /api/chemical/xps/analyze
Analyze XPS spectrum

**Request:**
```json
{
  "file": "spectrum.txt",
  "params": {
    "source": "Al Kα",
    "pass_energy": 20,
    "scans": 10
  }
}
```

**Response:**
```json
{
  "binding_energy": [...],
  "intensity": [...],
  "peaks": [
    {
      "position": 284.5,
      "element": "C 1s",
      "area": 15000
    }
  ],
  "composition": {
    "C": 35.2,
    "O": 42.8,
    "Si": 15.3,
    "N": 6.7
  }
}
```

#### POST /api/chemical/xps/fit_peak
Fit individual XPS peak

**Parameters:**
- `binding_energy`: Array of BE values
- `intensity`: Array of intensities
- `shape`: Peak shape ("Gaussian", "Lorentzian", "Voigt")
- `background_type`: Background method ("shirley", "tougaard")

#### POST /api/chemical/xrf/analyze
Analyze XRF spectrum

**Parameters:**
- `file`: Spectrum file
- `excitation_energy`: X-ray tube voltage (keV)
- `measurement_time`: Acquisition time (s)
- `atmosphere`: Measurement atmosphere

#### GET /api/chemical/elements
Get element database

**Response:**
```json
{
  "elements": [
    {
      "symbol": "Si",
      "name": "Silicon",
      "atomic_number": 14,
      "xps_peaks": ["2p3/2", "2p1/2", "2s"],
      "xrf_lines": ["Kα", "Kβ"]
    }
  ]
}
```

---

## User Guide

### Getting Started

1. **Start Services**
   ```bash
   ./start_session11_services.sh
   ```

2. **Access Interface**
   - Web UI: http://localhost:3011/chemical
   - API Docs: http://localhost:8011/docs

### XPS Analysis Workflow

1. **Load Spectrum**
   - Click "Load Spectrum" or drag-and-drop file
   - Supported formats: .txt, .csv, .vms

2. **Set Parameters**
   - Select X-ray source (Al Kα, Mg Kα)
   - Choose pass energy (resolution)
   - Set energy range

3. **Process Spectrum**
   - Apply smoothing if needed
   - Select background type
   - Click "Subtract Background"

4. **Identify Peaks**
   - Auto-find peaks or manual selection
   - Review element assignments
   - Adjust if necessary

5. **Fit Peaks**
   - Select peak shape
   - Adjust parameters
   - Check fit quality (R²)

6. **Quantification**
   - Review atomic percentages
   - Check sensitivity factors
   - Apply corrections if needed

7. **Export Results**
   - Generate report (PDF/Excel)
   - Save processed data

### XRF Analysis Workflow

1. **Measurement Setup**
   - Set excitation energy
   - Choose measurement time
   - Select atmosphere

2. **Acquire/Load Spectrum**
   - Start measurement or load file
   - Review spectrum quality

3. **Peak Identification**
   - Auto-identify elements
   - Check for overlaps
   - Mark escape/sum peaks

4. **Quantification**
   - Select quantification method
   - Apply matrix corrections
   - Review results

5. **Detection Limits**
   - Calculate MDLs
   - Review for trace elements

### Depth Profiling (XPS)

1. **Setup Parameters**
   - Define etch rate (nm/s)
   - Set time intervals
   - Select elements to track

2. **Acquire Series**
   - Take spectra at each depth
   - Monitor selected peaks

3. **Process Profile**
   - Plot concentration vs depth
   - Identify interfaces
   - Export profile data

---

## Best Practices

### Sample Preparation
- **XPS:**
  - Clean surface (avoid contamination)
  - Conductive samples preferred
  - Use charge neutralization for insulators
  - Store under inert atmosphere

- **XRF:**
  - Flat, homogeneous surface
  - Appropriate thickness (>critical depth)
  - Consider particle size effects
  - Use appropriate sample holders

### Data Acquisition
1. **Survey Scans First**
   - Wide energy range
   - Lower resolution
   - Identify all elements

2. **High-Resolution Scans**
   - Narrow energy windows
   - Higher pass energy (XPS)
   - Multiple scans for averaging

3. **Reference Samples**
   - Run standards regularly
   - Check calibration
   - Monitor instrument drift

### Peak Fitting Guidelines
1. **Constrain Parameters**
   - Physically reasonable FWHM
   - Known peak separations
   - Area ratios for multiplets

2. **Background Selection**
   - Shirley for metals/semiconductors
   - Tougaard for insulators
   - Linear for small regions

3. **Peak Shape Selection**
   - Voigt for most XPS peaks
   - Doniach-Sunjic for metals
   - Gaussian for low resolution

### Quantification Accuracy
1. **Sensitivity Factors**
   - Use appropriate RSF values
   - Consider X-ray source
   - Apply transmission corrections

2. **Matrix Effects (XRF)**
   - Consider absorption/enhancement
   - Use fundamental parameters
   - Validate with standards

3. **Depth Considerations**
   - Information depth ~5-10 nm (XPS)
   - Consider overlayers
   - Account for preferential sputtering

---

## Troubleshooting

### Common Issues

#### Poor Peak Fitting
**Symptoms:** Low R², residuals show structure
**Solutions:**
- Try different peak shapes
- Adjust background endpoints
- Add additional components
- Check for satellites

#### Incorrect Quantification
**Symptoms:** Unrealistic compositions
**Solutions:**
- Verify sensitivity factors
- Check peak integration limits
- Consider overlapping peaks
- Review background subtraction

#### Noisy Spectra
**Symptoms:** High noise, poor S/N
**Solutions:**
- Increase acquisition time
- Optimize pass energy
- Check sample charging
- Verify instrument alignment

#### Peak Position Shifts
**Symptoms:** Peaks at wrong BE/energy
**Solutions:**
- Calibrate to reference peak
- Check for charging
- Verify X-ray source
- Review energy scale

### Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| "Peak not found" | No peaks above threshold | Lower prominence parameter |
| "Fitting failed" | Poor initial parameters | Manual parameter adjustment |
| "Invalid spectrum" | Wrong file format | Check format, use converter |
| "Calibration failed" | No reference peak | Manual calibration needed |

---

## Safety Guidelines

### X-ray Safety
⚠️ **WARNING:** X-ray radiation hazard

1. **Shielding**
   - Never bypass interlocks
   - Ensure proper enclosure
   - Check for leaks regularly

2. **Personal Protection**
   - Wear dosimeter if required
   - Follow ALARA principles
   - Maintain safe distance

3. **Training**
   - Complete radiation safety training
   - Understand emergency procedures
   - Know exposure limits

### Vacuum Safety
1. **Pump Operation**
   - Follow startup/shutdown procedures
   - Monitor pressure gauges
   - Check pump oil levels

2. **Sample Handling**
   - Vent slowly to avoid damage
   - Use proper tools
   - Avoid contamination

### Chemical Safety
1. **Sample Preparation**
   - Use fume hood for solvents
   - Wear appropriate PPE
   - Dispose of waste properly

2. **Ion Sputtering**
   - Argon gas handling
   - Pressure relief systems
   - Ventilation requirements

### Emergency Procedures
1. **X-ray Exposure**
   - Shut down source immediately
   - Report to radiation safety officer
   - Seek medical evaluation

2. **Vacuum Failure**
   - Close gate valves
   - Shut down pumps
   - Protect samples/detector

3. **Power Failure**
   - System should safe itself
   - Document sample status
   - Follow restart procedures

---

## Appendices

### A. Element Reference Table
| Element | XPS BE (eV) | XRF Energy (keV) | RSF (Al Kα) |
|---------|-------------|------------------|-------------|
| C | 284.5 (1s) | 0.277 (Kα) | 0.278 |
| O | 532.5 (1s) | 0.525 (Kα) | 0.780 |
| Si | 99.3 (2p) | 1.740 (Kα) | 0.339 |
| N | 399.5 (1s) | 0.392 (Kα) | 0.477 |

### B. Common Chemical Shifts
| Species | BE Shift (eV) |
|---------|---------------|
| C-C | 0 (reference) |
| C-O | +1.5 |
| C=O | +3.5 |
| O-C=O | +4.5 |

### C. Abbreviations
- **XPS:** X-ray Photoelectron Spectroscopy
- **XRF:** X-ray Fluorescence
- **BE:** Binding Energy
- **KE:** Kinetic Energy
- **RSF:** Relative Sensitivity Factor
- **FWHM:** Full Width at Half Maximum
- **MDL:** Method Detection Limit

---

*Document Version: 1.0*  
*Last Updated: October 2024*  
*Session 11 - Chemical Analysis Module*
