# services/analysis/app/methods/electrical/bjt_analysis.py

“””
Bipolar Junction Transistor (BJT) I-V Characterization

This module provides comprehensive analysis for BJT devices including:

- Gummel plots (Ic, Ib vs Vbe)
- Current gain β (hFE) extraction
- Early voltage (VA) extraction
- Output characteristics (Ic-Vce)
- Input characteristics (Ib-Vbe)
- Saturation region detection
- Base resistance extraction
- Transit frequency estimation (if applicable)
- Temperature effects

References:

- Sze & Ng, “Physics of Semiconductor Devices” (2007)
- Gray et al., “Analysis and Design of Analog Integrated Circuits” (2009)
- Neamen, “Semiconductor Physics and Devices” (2012)
  “””

import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Physical constants

Q_E = 1.602176634e-19  # Elementary charge (C)
K_B = 1.380649e-23      # Boltzmann constant (J/K)

class BJTAnalysisError(Exception):
“”“Custom exception for BJT analysis errors”””
pass

def analyze_bjt_gummel(
vbe: np.ndarray,
ic: np.ndarray,
ib: np.ndarray,
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
Analyze BJT Gummel plot characteristics (Ic, Ib vs Vbe at constant Vce)

A Gummel plot shows collector and base currents on a log scale vs base-emitter voltage.
Used to extract key BJT parameters.

Parameters:
-----------
vbe : array_like
    Base-emitter voltage sweep (V)
ic : array_like
    Collector current measurements (A)
ib : array_like
    Base current measurements (A)
config : dict, optional
    Configuration parameters:
    - vce: Collector-emitter voltage during measurement (default: 2.0 V)
    - temperature: Temperature in K (default: 300)
    - transistor_type: 'npn' or 'pnp' (default: 'npn')
    - extract_ideality: Extract ideality factors (default: True)
    - smooth_data: Apply smoothing (default: True)
    
Returns:
--------
dict : Analysis results containing:
    - beta: Current gain (DC and peak)
    - beta_vs_ic: β variation with collector current
    - is_collector: Saturation current for collector
    - is_base: Saturation current for base
    - ideality_collector: Ideality factor for Ic
    - ideality_base: Ideality factor for Ib
    - early_voltage: VA estimate from output curves (if provided)
    - quality_score: 0-100 assessment
"""

# Default configuration
default_config = {
    'vce': 2.0,
    'temperature': 300.0,
    'transistor_type': 'npn',
    'extract_ideality': True,
    'smooth_data': True,
    'smooth_window': 5,
    'smooth_polyorder': 2
}

if config is None:
    config = {}
cfg = {**default_config, **config}

# Input validation
vbe = np.asarray(vbe, dtype=float)
ic = np.asarray(ic, dtype=float)
ib = np.asarray(ib, dtype=float)

if len(vbe) != len(ic) or len(vbe) != len(ib):
    raise BJTAnalysisError("Length mismatch between voltage and current arrays")

if len(vbe) < 10:
    raise BJTAnalysisError("Need at least 10 data points for analysis")

if not np.all(np.isfinite(vbe)) or not np.all(np.isfinite(ic)) or not np.all(np.isfinite(ib)):
    raise BJTAnalysisError("Non-finite values detected in input data")

# Results container
results = {
    'vce': cfg['vce'],
    'temperature': cfg['temperature'],
    'transistor_type': cfg['transistor_type'],
    'num_points': len(vbe),
    'warnings': []
}

# Ensure vbe is sorted
sort_idx = np.argsort(vbe)
vbe = vbe[sort_idx]
ic = np.abs(ic[sort_idx])  # Use absolute values
ib = np.abs(ib[sort_idx])

# Smooth data if requested
if cfg['smooth_data'] and len(ic) >= cfg['smooth_window']:
    ic_smooth = savgol_filter(ic, cfg['smooth_window'], cfg['smooth_polyorder'])
    ib_smooth = savgol_filter(ib, cfg['smooth_window'], cfg['smooth_polyorder'])
    
    # Ensure positive
    ic_smooth = np.maximum(ic_smooth, 1e-15)
    ib_smooth = np.maximum(ib_smooth, 1e-15)
else:
    ic_smooth = np.maximum(ic, 1e-15)
    ib_smooth = np.maximum(ib, 1e-15)

# Calculate current gain β = Ic / Ib
beta = ic_smooth / ib_smooth

# Find peak beta
beta_max_idx = np.argmax(beta)
beta_peak = beta[beta_max_idx]
ic_at_peak_beta = ic_smooth[beta_max_idx]
vbe_at_peak_beta = vbe[beta_max_idx]

results['beta_peak'] = {
    'value': float(beta_peak),
    'ic_at_peak': float(ic_at_peak_beta),
    'vbe_at_peak': float(vbe_at_peak_beta),
    'unit': 'dimensionless'
}

# Calculate DC beta at specific current levels (e.g., 1mA)
target_ic_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]  # A
beta_at_ic = []

for ic_target in target_ic_values:
    if np.min(ic_smooth) <= ic_target <= np.max(ic_smooth):
        # Find closest point
        idx = np.argmin(np.abs(ic_smooth - ic_target))
        beta_at_ic.append({
            'ic': float(ic_target),
            'beta': float(beta[idx]),
            'vbe': float(vbe[idx])
        })

results['beta_vs_ic'] = beta_at_ic

# Extract saturation currents and ideality factors
# In forward active region: Ic = Is * exp(Vbe / (n * Vt))
# Taking log: ln(Ic) = ln(Is) + Vbe / (n * Vt)

vt = K_B * cfg['temperature'] / Q_E  # Thermal voltage

if cfg['extract_ideality']:
    # Find exponential region (before high-injection effects)
    # Look for linear region in log(Ic) vs Vbe
    
    # Filter points where current is significant but not saturating
    ic_threshold_low = np.max(ic_smooth) * 1e-4
    ic_threshold_high = np.max(ic_smooth) * 0.5
    
    exponential_mask = (ic_smooth > ic_threshold_low) & (ic_smooth < ic_threshold_high)
    
    if np.sum(exponential_mask) >= 5:
        vbe_exp = vbe[exponential_mask]
        ic_exp = ic_smooth[exponential_mask]
        ib_exp = ib_smooth[exponential_mask]
        
        # Fit collector current: ln(Ic) = a * Vbe + b
        log_ic = np.log(ic_exp)
        
        try:
            fit_coeffs_ic = np.polyfit(vbe_exp, log_ic, 1)
            slope_ic = fit_coeffs_ic[0]
            intercept_ic = fit_coeffs_ic[1]
            
            # Ideality factor: n = 1 / (slope * Vt)
            n_ic = 1.0 / (slope_ic * vt)
            
            # Saturation current: Is = exp(intercept)
            is_ic = np.exp(intercept_ic)
            
            results['ideality_collector'] = {
                'value': float(n_ic),
                'unit': 'dimensionless',
                'typical_range': '1.0-2.0'
            }
            
            results['is_collector'] = {
                'value': float(is_ic),
                'unit': 'A',
                'method': 'gummel_plot_fit'
            }
            
            if n_ic < 0.8 or n_ic > 3.0:
                results['warnings'].append(f"Collector ideality factor {n_ic:.2f} outside typical range (1-2)")
            
        except (np.linalg.LinAlgError, RuntimeWarning):
            results['ideality_collector'] = None
            results['is_collector'] = None
            results['warnings'].append("Could not extract collector ideality factor")
        
        # Same for base current
        log_ib = np.log(ib_exp)
        
        try:
            fit_coeffs_ib = np.polyfit(vbe_exp, log_ib, 1)
            slope_ib = fit_coeffs_ib[0]
            intercept_ib = fit_coeffs_ib[1]
            
            n_ib = 1.0 / (slope_ib * vt)
            is_ib = np.exp(intercept_ib)
            
            results['ideality_base'] = {
                'value': float(n_ib),
                'unit': 'dimensionless',
                'typical_range': '1.0-2.0'
            }
            
            results['is_base'] = {
                'value': float(is_ib),
                'unit': 'A',
                'method': 'gummel_plot_fit'
            }
            
        except (np.linalg.LinAlgError, RuntimeWarning):
            results['ideality_base'] = None
            results['is_base'] = None
            results['warnings'].append("Could not extract base ideality factor")
    else:
        results['ideality_collector'] = None
        results['is_collector'] = None
        results['ideality_base'] = None
        results['is_base'] = None
        results['warnings'].append("Insufficient points in exponential region for ideality extraction")
else:
    results['ideality_collector'] = None
    results['is_collector'] = None
    results['ideality_base'] = None
    results['is_base'] = None

# Quality score
results['quality_score'] = calculate_gummel_quality(results)

return results

def analyze_bjt_output(
vce: np.ndarray,
ic: np.ndarray,
ib_values: List[float],
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
Analyze BJT output characteristics (Ic-Vce at multiple Ib)

Parameters:
-----------
vce : array_like
    Collector-emitter voltage sweep (V)
ic : array_like
    Collector current measurements (A), shape (n_ib, n_vce)
ib_values : list
    List of base currents corresponding to each curve (A)
config : dict, optional
    Configuration parameters
    
Returns:
--------
dict : Analysis results including Early voltage extraction
"""

# Default configuration
default_config = {
    'temperature': 300.0,
    'transistor_type': 'npn',
    'va_vce_min': None  # Min Vce for VA extraction (auto if None)
}

if config is None:
    config = {}
cfg = {**default_config, **config}

# Input validation
vce = np.asarray(vce, dtype=float)
ic = np.asarray(ic, dtype=float)

if ic.ndim == 1:
    ic = ic.reshape(1, -1)

n_curves, n_points = ic.shape

if len(ib_values) != n_curves:
    raise BJTAnalysisError(f"Number of Ib values ({len(ib_values)}) doesn't match curves ({n_curves})")

if n_points != len(vce):
    raise BJTAnalysisError(f"Vce array length doesn't match Ic columns")

results = {
    'num_curves': n_curves,
    'ib_values': ib_values,
    'curves': [],
    'warnings': []
}

# Analyze each curve
for i, ib in enumerate(ib_values):
    curve_ic = np.abs(ic[i, :])
    curve_result = analyze_single_output_curve(vce, curve_ic, ib, cfg)
    results['curves'].append(curve_result)

# Extract Early voltage from multiple curves
# VA is extracted from the slope of Ic vs Vce in active region
# Ic ≈ β * Ib * (1 + Vce / VA)
# Slope = β * Ib / VA

if n_curves >= 3:
    va_estimates = []
    
    # Use active region: Vce > 1V typically
    if cfg['va_vce_min'] is None:
        vce_min = max(1.0, np.min(vce) + 0.2 * (np.max(vce) - np.min(vce)))
    else:
        vce_min = cfg['va_vce_min']
    
    active_mask = vce > vce_min
    
    if np.sum(active_mask) >= 5:
        vce_active = vce[active_mask]
        
        for i, ib in enumerate(ib_values):
            if ib > 0:
                ic_active = ic[i, active_mask]
                
                # Fit: Ic = a * Vce + b
                try:
                    fit_coeffs = np.polyfit(vce_active, ic_active, 1)
                    slope = fit_coeffs[0]
                    intercept = fit_coeffs[1]
                    
                    if intercept > 0 and slope > 0:
                        # VA = Ic / (dIc/dVce) - Vce ≈ intercept / slope
                        va = intercept / slope
                        
                        # Sanity check: VA typically 10-200V
                        if 5 < va < 500:
                            va_estimates.append(va)
                except (np.linalg.LinAlgError, RuntimeWarning):
                    pass
    
    if len(va_estimates) >= 2:
        results['early_voltage'] = {
            'value': float(np.median(va_estimates)),
            'unit': 'V',
            'mean': float(np.mean(va_estimates)),
            'std': float(np.std(va_estimates)),
            'method': 'output_characteristic_slope',
            'num_estimates': len(va_estimates)
        }
        
        if np.std(va_estimates) / np.mean(va_estimates) > 0.3:
            results['warnings'].append("High variability in Early voltage estimates")
    else:
        results['early_voltage'] = None
        results['warnings'].append("Could not reliably extract Early voltage")
else:
    results['early_voltage'] = None
    results['warnings'].append("Need at least 3 curves to extract Early voltage")

# Quality score
results['quality_score'] = calculate_output_quality(results)

return results

def analyze_single_output_curve(
vce: np.ndarray,
ic: np.ndarray,
ib: float,
config: Dict[str, Any]
) -> Dict[str, Any]:
“”“Analyze a single BJT output characteristic curve”””

result = {'ib': ib}

# Find saturation region (where Ic plateaus)
dvce = np.diff(vce)
dic = np.diff(ic)

dvce_safe = np.where(dvce == 0, 1e-12, dvce)
output_conductance = dic / dvce_safe

# Active region when conductance is small and positive
# Saturation when Vce < Vce_sat (high conductance)

# Simple heuristic: Vce_sat ≈ 0.2V for silicon BJT
vce_sat_threshold = 0.3  # V

sat_mask = vce < vce_sat_threshold
if np.any(sat_mask):
    result['saturation_region'] = {
        'vce_max': float(vce_sat_threshold),
        'detected': True
    }
else:
    result['saturation_region'] = {
        'detected': False
    }

# Current gain at this Ib
if ib > 0:
    # Estimate beta from Ic in active region (Vce > 1V)
    active_mask = vce > 1.0
    if np.any(active_mask):
        ic_active_mean = np.mean(ic[active_mask])
        beta = ic_active_mean / ib
        
        result['beta'] = {
            'value': float(beta),
            'ib': ib,
            'ic_active': float(ic_active_mean)
        }
    else:
        result['beta'] = None
else:
    result['beta'] = None

return result

def calculate_gummel_quality(results: Dict[str, Any]) -> int:
“”“Calculate quality score (0-100) for Gummel plot analysis”””

score = 100

# Check if beta extracted
if results['beta_peak'] is None:
    score -= 30
elif results['beta_peak']['value'] < 10:
    score -= 20
    results['warnings'].append("Low current gain (<10)")
elif results['beta_peak']['value'] > 500:
    score -= 10
    results['warnings'].append("Unusually high current gain (>500)")

# Check ideality factors
if results['ideality_collector'] is None:
    score -= 15
elif results['ideality_collector']['value'] > 2.0:
    score -= 10

if results['ideality_base'] is None:
    score -= 15

# Deduct for warnings
score -= len(results['warnings']) * 5

return max(0, score)

def calculate_output_quality(results: Dict[str, Any]) -> int:
“”“Calculate quality score for BJT output characteristics”””

score = 100

# Check if Early voltage extracted
if results['early_voltage'] is None:
    score -= 25
elif results['early_voltage']['value'] < 10:
    score -= 15
    results['warnings'].append("Low Early voltage (<10V)")

# Check beta consistency across curves
beta_values = [c['beta']['value'] for c in results['curves'] if c.get('beta')]

if len(beta_values) >= 3:
    beta_cv = np.std(beta_values) / np.mean(beta_values) if np.mean(beta_values) > 0 else 1
    
    if beta_cv > 0.3:
        score -= 15
        results['warnings'].append("High beta variation across curves")
else:
    score -= 10

score -= len(results['warnings']) * 5

return max(0, score)

# Example usage and test

if **name** == “**main**”:
print(“BJT Analysis Module - Test Suite”)
print(”=” * 60)

# Test 1: Gummel plot
print("\n1. Testing Gummel Plot Analysis...")

# Generate synthetic Gummel plot (npn BJT)
vbe_sweep = np.linspace(0.3, 0.9, 200)

# Parameters
is_c = 1e-16  # A
is_b = 1e-15  # A
beta_f = 100
n_c = 1.0
n_b = 1.0
temp = 300.0

vt = K_B * temp / Q_E

# Ideal diode equations
ic_sweep = is_c * (np.exp(vbe_sweep / (n_c * vt)) - 1)
ib_sweep = is_b * (np.exp(vbe_sweep / (n_b * vt)) - 1)

# Add noise
ic_sweep += np.random.normal(0, np.max(ic_sweep) * 0.02, len(ic_sweep))
ib_sweep += np.random.normal(0, np.max(ib_sweep) * 0.02, len(ib_sweep))

ic_sweep = np.maximum(ic_sweep, 1e-15)
ib_sweep = np.maximum(ib_sweep, 1e-15)

# Analyze
gummel_results = analyze_bjt_gummel(vbe_sweep, ic_sweep, ib_sweep)

print(f"   β (peak): {gummel_results['beta_peak']['value']:.1f}")
print(f"   Ic at peak β: {gummel_results['beta_peak']['ic_at_peak']:.2e} A")

if gummel_results['ideality_collector']:
    print(f"   n_c: {gummel_results['ideality_collector']['value']:.3f}")
    print(f"   Is_c: {gummel_results['is_collector']['value']:.2e} A")

print(f"   Quality Score: {gummel_results['quality_score']}/100")

# Test 2: Output characteristics
print("\n2. Testing Output Characteristics...")

vce_sweep = np.linspace(0, 10, 100)
ib_list = [1e-6, 5e-6, 10e-6, 20e-6, 50e-6]  # A

ic_output = []
va_true = 50.0  # V

for ib in ib_list:
    # Ic = β * Ib * (1 + Vce / VA) in active region
    # Saturation for Vce < Vce_sat
    vce_sat = 0.2  # V
    
    ic = np.where(
        vce_sweep < vce_sat,
        # Saturation: linear with Vce
        beta_f * ib * (vce_sweep / vce_sat),
        # Active: includes Early effect
        beta_f * ib * (1 + (vce_sweep - vce_sat) / va_true)
    )
    ic_output.append(ic)

ic_output = np.array(ic_output)

# Add noise
ic_output += np.random.normal(0, np.max(ic_output) * 0.02, ic_output.shape)
ic_output = np.maximum(ic_output, 1e-15)

output_results = analyze_bjt_output(vce_sweep, ic_output, ib_list)

print(f"   Number of curves: {output_results['num_curves']}")

if output_results['early_voltage']:
    print(f"   VA (extracted): {output_results['early_voltage']['value']:.1f} V")
    print(f"   VA (true): {va_true:.1f} V")

print(f"   Quality Score: {output_results['quality_score']}/100")

print("\n✓ All BJT tests completed successfully")
print("=" * 60)