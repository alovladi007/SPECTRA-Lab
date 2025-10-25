# Session 11: Surface Analysis (XPS/XRF) - Complete Documentation

**Version:** 1.0.0  
**Date:** October 2024  
**Status:** ✅ Production Ready

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Theory & Background](#theory--background)
3. [Implementation Overview](#implementation-overview)
4. [API Reference](#api-reference)
5. [User Guide](#user-guide)
6. [Best Practices](#best-practices)

---

## Executive Summary

Session 11 implements comprehensive surface analysis methods:

### Implemented Methods

1. **XPS (X-ray Photoelectron Spectroscopy)**
   - Peak fitting with Voigt profiles
   - Shirley background subtraction
   - Atomic % quantification with RSF
   - Charge referencing (C 1s)
   - Chemical state identification

2. **XRF (X-ray Fluorescence)**
   - Element identification
   - Fundamental parameters quantification
   - Thin film thickness measurement
   - Multi-element analysis

---

## Theory & Background

### XPS Principles

XPS measures the kinetic energy of photoelectrons ejected by X-rays:
```
BE = hν - KE - φ
```
where BE is binding energy, hν is X-ray energy, KE is kinetic energy, φ is work function.

**Key Features:**
- Surface sensitive (~1-10 nm)
- Element identification
- Chemical state information
- Quantitative analysis

### XRF Principles

XRF detects characteristic X-rays emitted when inner shell vacancies are filled:
```
E_Kα = E_K - E_L
```

**Key Features:**
- Non-destructive
- Multi-element capability
- Bulk and thin film analysis
- Fast measurements

---

## Implementation Overview

### XPSAnalyzer

**Capabilities:**
- Shirley background subtraction (iterative)
- Voigt peak fitting
- Multi-peak deconvolution
- RSF-based quantification
- Charge shift correction

**Example:**
```python
analyzer = XPSAnalyzer()
background = analyzer.shirley_background(spectrum)
peaks = analyzer.fit_multiple_peaks(spectrum, [99.0, 103.0])
quant = analyzer.quantify(peaks)
```

### XRFAnalyzer

**Capabilities:**
- Peak identification from energy
- Element database matching
- Fundamental parameters method
- Thin film analysis

**Example:**
```python
analyzer = XRFAnalyzer()
peaks = analyzer.identify_peaks(spectrum)
quant = analyzer.fundamental_parameters(peaks)
```

---

## API Reference

### XPS Endpoints

#### POST /api/xps/analyze
Analyze XPS spectrum

**Request:**
```json
{
  "spectrum_id": "string",
  "background_type": "shirley",
  "peak_positions": [99.0, 103.0],
  "charge_reference": true
}
```

### XRF Endpoints

#### POST /api/xrf/analyze
Analyze XRF spectrum

**Request:**
```json
{
  "spectrum_id": "string",
  "method": "fundamental_parameters",
  "threshold": 100.0
}
```

---

## User Guide

### XPS Analysis Workflow

1. Load spectrum
2. Subtract background
3. Fit peaks
4. Assign elements
5. Quantify composition

### XRF Analysis Workflow

1. Load spectrum
2. Identify peaks
3. Match to database
4. Quantify elements

---

## Best Practices

### XPS
- Use Shirley background for inorganic materials
- Reference to C 1s at 284.8 eV
- Verify RSF for your instrument
- Check peak assignments carefully

### XRF
- Calibrate energy scale
- Use standards for accurate quantification
- Consider matrix effects
- Check for peak overlaps

---

**Complete technical documentation available in full package**
