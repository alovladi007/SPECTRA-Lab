# Session 7: Optical I - UV-Vis-NIR & FTIR Implementation Guide

**Session:** S7 - Optical I (UV-Vis-NIR, FTIR)  
**Duration:** Week 7 (5 days)  
**Date:** October 23-27, 2025  
**Status:** 🚀 STARTING NOW  
**Prerequisites:** Sessions 1-6 Complete ✅

---

## 📋 Session Overview

### Scope
Optical characterization methods focusing on:
- **UV-Vis-NIR** - Absorption/Transmission/Reflectance spectroscopy (200-2500 nm)
- **FTIR** - Fourier Transform Infrared spectroscopy (400-4000 cm⁻¹)
- **Band Gap Analysis** - Direct/indirect transitions via Tauc plots
- **Chemical Bond Identification** - Vibrational mode analysis

### Business Value
- Enable material quality assessment (optical properties)
- Identify chemical composition and contamination
- Determine semiconductor band gaps for device design
- Support thin film optimization (thickness, uniformity)
- Quality control for coatings and surface treatments

### Technical Challenges
- Baseline correction for varying substrates
- Interference fringe handling in thin films
- Peak deconvolution in overlapping regions
- Temperature/environmental compensation
- Large spectral data management (10K+ points)

---

## 🎯 Deliverables

### 1. UV-Vis-NIR Analysis Module
```python
class UVVisNIRAnalyzer:
    """
    Features:
    - Spectrum acquisition (200-2500 nm)
    - Transmission → Absorbance conversion (Beer-Lambert)
    - Reflectance correction (specular/diffuse)
    - Baseline correction (polynomial, spline, rubberband)
    - Interference fringe removal
    - Tauc plot generation (direct/indirect transitions)
    - Urbach tail analysis
    - Optical constants extraction (n, k)
    """
```

### 2. FTIR Analysis Module
```python
class FTIRAnalyzer:
    """
    Features:
    - Spectrum acquisition (400-4000 cm⁻¹)
    - Baseline correction (ALS, polynomial, manual)
    - Peak finding and fitting (Gaussian/Lorentzian)
    - Functional group identification
    - Quantitative analysis (peak area integration)
    - ATR correction factors
    - Library matching against reference spectra
    - Thickness calculation from fringes
    """
```

### 3. Spectral Processing Utilities
```python
class SpectralProcessor:
    """
    Common utilities:
    - Smoothing (Savitzky-Golay, moving average)
    - Derivative spectroscopy (1st/2nd derivatives)
    - Spectral arithmetic (addition, subtraction, ratio)
    - Multi-spectrum comparison
    - Principal Component Analysis (PCA)
    - Spectral stitching (multiple detectors)
    """
```

### 4. Frontend UI Components
- **Spectrum Viewer** - Interactive plot with zoom/pan/cursor
- **Peak Annotation Tool** - Manual/automatic peak labeling
- **Tauc Plot Generator** - Band gap extraction interface
- **Library Manager** - Reference spectra database
- **Batch Processor** - Multiple file analysis

### 5. Test Data & Validation
- Synthetic UV-Vis spectra (Si, GaAs, GaN, organics)
- Synthetic FTIR spectra (SiO₂, Si₃N₄, polymers)
- Known band gap materials for validation
- Standard peak libraries for FTIR

---

## 🏗️ Technical Architecture

### Data Flow
```
Spectrometer → Raw Spectrum → Preprocessing → Analysis → Results
     ↓              ↓              ↓            ↓          ↓
   Driver       Calibration    Baseline     Fitting    Database
```

### Key Algorithms

#### 1. Tauc Plot Analysis
```python
def calculate_tauc_plot(wavelength, absorbance, transition_type='direct'):
    """
    Extract optical band gap from absorption spectrum
    
    Tauc equation:
    - Direct: (αhν)² = A(hν - Eg)
    - Indirect: (αhν)^(1/2) = A(hν - Eg)
    
    Returns:
        band_gap: float (eV)
        r_squared: float (fit quality)
    """
```

#### 2. Baseline Correction (ALS)
```python
def asymmetric_least_squares(spectrum, lam=1e6, p=0.01, n_iter=10):
    """
    Asymmetric Least Squares baseline correction
    Penalizes positive deviations less than negative
    Ideal for spectroscopy with positive peaks
    """
```

#### 3. Peak Deconvolution
```python
def deconvolve_peaks(spectrum, n_peaks=None):
    """
    Fit multiple overlapping peaks
    Uses Levenberg-Marquardt with constraints
    Returns peak positions, areas, widths
    """
```

---

## 📊 Implementation Schedule

### Day 1: UV-Vis-NIR Core (Monday)
- [ ] Morning: UVVisNIRAnalyzer class structure
- [ ] Afternoon: Transmission/Absorbance/Reflectance modes
- [ ] Evening: Baseline correction algorithms

### Day 2: Band Gap Analysis (Tuesday)
- [ ] Morning: Tauc plot implementation (direct/indirect)
- [ ] Afternoon: Urbach tail fitting
- [ ] Evening: Optical constants extraction (n, k)

### Day 3: FTIR Implementation (Wednesday)
- [ ] Morning: FTIRAnalyzer class
- [ ] Afternoon: Peak finding and fitting
- [ ] Evening: Functional group library

### Day 4: Frontend UI (Thursday)
- [ ] Morning: Interactive spectrum viewer component
- [ ] Afternoon: Tauc plot interface
- [ ] Evening: Peak annotation tools

### Day 5: Integration & Testing (Friday)
- [ ] Morning: Integration with existing platform
- [ ] Afternoon: Test suite completion
- [ ] Evening: Documentation and deployment

---

## 🔧 Technical Requirements

### Dependencies
```json
{
  "python": {
    "scipy": ">=1.11.0",
    "numpy": ">=1.24.0",
    "pandas": ">=2.0.0",
    "lmfit": ">=1.2.0",
    "scikit-learn": ">=1.3.0",
    "matplotlib": ">=3.7.0"
  },
  "javascript": {
    "recharts": "^2.8.0",
    "plotly.js": "^2.26.0",
    "d3": "^7.8.0"
  }
}
```

### Instrument Drivers
- Ocean Optics (USB4000, HR4000, NIRQuest)
- Thermo Fisher (Nicolet iS50 FTIR)
- PerkinElmer (Lambda series)
- Shimadzu (UV-2600, IRTracer-100)

### Data Formats
- **Input:** CSV, TXT, SPA, SPC, JCAMP-DX
- **Output:** HDF5, CSV, JCAMP-DX
- **Reports:** PDF with embedded plots

---

## ✅ Acceptance Criteria

### Accuracy Requirements
- [ ] Band gap extraction: ±0.05 eV vs reference
- [ ] FTIR peak position: ±5 cm⁻¹
- [ ] Peak area quantification: ±5%
- [ ] Baseline correction: <2% residual

### Performance Requirements
- [ ] Spectrum processing: <1s for 10K points
- [ ] Tauc plot generation: <500ms
- [ ] Peak fitting: <2s for 10 peaks
- [ ] Batch processing: 100 spectra/minute

### Quality Requirements
- [ ] Unit test coverage: >90%
- [ ] Integration tests: All workflows
- [ ] Documentation: Complete API + user guide
- [ ] Code review: Approved by senior engineer

---

## 🎯 Success Metrics

| Metric | Target | Priority |
|--------|--------|----------|
| Band gap accuracy | ±0.05 eV | Critical |
| Peak identification | 95% correct | High |
| Processing speed | <2s/spectrum | High |
| UI responsiveness | <200ms | Medium |
| Memory usage | <1GB for 1000 spectra | Medium |

---

## 📚 Reference Materials

### Band Gap Values for Validation
| Material | Band Gap (eV) | Type | Test Tolerance |
|----------|---------------|------|----------------|
| Si | 1.12 | Indirect | ±0.05 |
| GaAs | 1.42 | Direct | ±0.05 |
| GaN | 3.4 | Direct | ±0.05 |
| TiO₂ | 3.2 | Indirect | ±0.05 |
| ZnO | 3.37 | Direct | ±0.05 |

### FTIR Peak Library (cm⁻¹)
| Peak | Assignment | Tolerance |
|------|------------|-----------|
| 1050-1150 | Si-O stretch | ±10 |
| 2900-3000 | C-H stretch | ±10 |
| 3200-3600 | O-H/N-H stretch | ±15 |
| 1600-1700 | C=O stretch | ±10 |
| 2100-2200 | C≡N stretch | ±5 |

---

## 🔄 Integration Points

### With Existing Modules
1. **Database:** Store spectra with metadata
2. **File Storage:** S3/MinIO for raw data
3. **Analysis Pipeline:** Chain with other methods
4. **Reporting:** Include in comprehensive reports
5. **SPC:** Monitor peak positions/intensities

### API Endpoints
```python
POST /api/v1/optical/uvvisnir/analyze
POST /api/v1/optical/ftir/analyze
GET  /api/v1/optical/spectra/{id}
POST /api/v1/optical/batch/process
GET  /api/v1/optical/libraries/peaks
```

---

## 📝 Documentation Requirements

1. **Theory Guide**
   - UV-Vis-NIR principles
   - FTIR fundamentals
   - Tauc plot methodology
   - Peak fitting algorithms

2. **User Manual**
   - Instrument setup
   - Measurement procedures
   - Data analysis workflow
   - Troubleshooting

3. **API Reference**
   - Endpoint documentation
   - Parameter descriptions
   - Response formats
   - Error codes

---

## 🚀 Next Steps

1. **Review and approve this implementation guide**
2. **Set up development environment**
3. **Begin UV-Vis-NIR analyzer implementation**
4. **Schedule daily standups for the week**
5. **Prepare test datasets**

---

**Ready to begin Session 7 implementation!**

*Next Session Preview: S8 - Optical II (Ellipsometry, PL/EL, Raman)*