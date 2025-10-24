# services/analysis/app/methods/electrical/cv_profiling.py

“””
Capacitance-Voltage (C-V) Profiling and Analysis

This module provides comprehensive C-V analysis for:

- MOS capacitors (accumulation, depletion, inversion)
- Schottky barrier diodes
- p-n junctions
- Doping concentration profiles (N_D, N_A vs depth)
- Flat-band voltage (V_fb) extraction
- Interface trap density (D_it) estimation
- Oxide thickness and capacitance
- Built-in potential extraction
- High-frequency vs quasi-static analysis

References:

- Schroder, “Semiconductor Material and Device Characterization” (2006)
- Nicollian & Brews, “MOS Physics and Technology” (1982)
- ASTM F1701 (Standard Test Method for Dopant Concentration Profiles)
  “””

import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Physical constants

Q_E = 1.602176634e-19  # Elementary charge (C)
K_B = 1.380649e-23      # Boltzmann constant (J/K)
EPS_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
EPS_SI = 11.7            # Relative permittivity of silicon
EPS_SIOZ = 3.9           # Relative permittivity of SiO2

class CVAnalysisError(Exception):
“”“Custom exception for C-V analysis errors”””
pass

def analyze_mos_cv(
voltage: np.ndarray,
capacitance: np.ndarray,
area: float,
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
Analyze MOS capacitor C-V characteristics

Parameters:
-----------
voltage : array_like
    Gate voltage sweep (V)
capacitance : array_like
    Measured capacitance (F)
area : float
    Capacitor area (m²)
config : dict, optional
    Configuration parameters:
    - frequency: Measurement frequency in Hz (default: 1e6)
    - temperature: Temperature in K (default: 300)
    - substrate_type: 'n' or 'p' (default: 'p')
    - substrate_doping: Doping concentration in cm⁻³ (optional, will be extracted if not provided)
    - oxide_thickness: Oxide thickness in m (optional)
    - smooth_data: Apply smoothing filter (default: True)
    
Returns:
--------
dict : Analysis results containing:
    - cox: Oxide capacitance (F and F/m²)
    - cfb: Flat-band capacitance
    - vfb: Flat-band voltage
    - vth: Threshold voltage (for strong inversion)
    - oxide_thickness: Extracted oxide thickness
    - substrate_doping: Extracted doping concentration
    - flatband_capacitance: Capacitance at flat-band
    - interface_trap_density: D_it estimate
    - quality_score: 0-100 assessment
"""

# Default configuration
default_config = {
    'frequency': 1e6,
    'temperature': 300.0,
    'substrate_type': 'p',
    'substrate_doping': None,
    'oxide_thickness': None,
    'smooth_data': True,
    'smooth_window': 5,
    'smooth_polyorder': 2
}

if config is None:
    config = {}
cfg = {**default_config, **config}

# Input validation
voltage = np.asarray(voltage, dtype=float)
capacitance = np.asarray(capacitance, dtype=float)

if len(voltage) != len(capacitance):
    raise CVAnalysisError("Length mismatch between voltage and capacitance arrays")

if len(voltage) < 10:
    raise CVAnalysisError("Need at least 10 data points for analysis")

if not np.all(np.isfinite(voltage)) or not np.all(np.isfinite(capacitance)):
    raise CVAnalysisError("Non-finite values detected in input data")

if area <= 0:
    raise CVAnalysisError("Capacitor area must be positive")

# Results container
results = {
    'area': area,
    'frequency': cfg['frequency'],
    'temperature': cfg['temperature'],
    'substrate_type': cfg['substrate_type'],
    'num_points': len(voltage),
    'warnings': []
}

# Ensure voltage is sorted
sort_idx = np.argsort(voltage)
voltage = voltage[sort_idx]
capacitance = capacitance[sort_idx]

# Smooth data if requested
if cfg['smooth_data'] and len(capacitance) >= cfg['smooth_window']:
    capacitance_smooth = savgol_filter(
        capacitance, cfg['smooth_window'], cfg['smooth_polyorder']
    )
else:
    capacitance_smooth = capacitance

# Extract Cox (oxide capacitance) from accumulation region
# For p-substrate: accumulation at negative voltages
# For n-substrate: accumulation at positive voltages

if cfg['substrate_type'] == 'p':
    # Accumulation at most negative voltages
    accum_mask = voltage < np.min(voltage) + 0.1 * (np.max(voltage) - np.min(voltage))
else:
    # Accumulation at most positive voltages
    accum_mask = voltage > np.max(voltage) - 0.1 * (np.max(voltage) - np.min(voltage))

if np.sum(accum_mask) >= 3:
    cox = np.mean(capacitance_smooth[accum_mask])
    cox_density = cox / area  # F/m²
    
    results['cox'] = {
        'value': float(cox),
        'unit': 'F',
        'density': float(cox_density),
        'unit_density': 'F/m^2',
        'density_uf_cm2': float(cox_density * 1e2),  # µF/cm²
        'unit_uf_cm2': 'µF/cm^2'
    }
    
    # Extract oxide thickness if not provided
    if cfg['oxide_thickness'] is None:
        # Cox = ε₀ * ε_ox * A / t_ox
        # t_ox = ε₀ * ε_ox * A / Cox
        tox = EPS_0 * EPS_SIOZ / cox_density
        
        results['oxide_thickness'] = {
            'value': float(tox),
            'unit': 'm',
            'value_nm': float(tox * 1e9),
            'unit_nm': 'nm',
            'method': 'extracted_from_cox'
        }
    else:
        results['oxide_thickness'] = {
            'value': cfg['oxide_thickness'],
            'unit': 'm',
            'value_nm': float(cfg['oxide_thickness'] * 1e9),
            'unit_nm': 'nm',
            'method': 'user_provided'
        }
else:
    results['warnings'].append("Could not determine Cox from accumulation region")
    results['cox'] = None
    results['oxide_thickness'] = None

# Extract flat-band voltage (Vfb)
# Method: Maximum slope of C-V curve (onset of depletion)

dv = np.diff(voltage)
dc = np.diff(capacitance_smooth)

dv_safe = np.where(dv == 0, 1e-12, dv)
dcdv = dc / dv_safe

voltage_deriv = (voltage[:-1] + voltage[1:]) / 2

if cfg['substrate_type'] == 'p':
    # Maximum negative slope
    max_slope_idx = np.argmin(dcdv)
else:
    # Maximum positive slope  
    max_slope_idx = np.argmax(dcdv)

vfb_estimate = voltage_deriv[max_slope_idx]

# Refine using theoretical flat-band capacitance
# C_fb = Cox / (1 + λ_d/t_ox) where λ_d is Debye length

if results['cox'] is not None:
    # Estimate Debye length (requires doping)
    if cfg['substrate_doping'] is not None:
        n_sub = cfg['substrate_doping']
    else:
        # Estimate from depletion capacitance
        n_sub = estimate_doping_from_cv(
            voltage, capacitance_smooth, area, 
            cfg['substrate_type'], cfg['temperature']
        )
    
    results['substrate_doping'] = {
        'value': float(n_sub),
        'unit': 'cm^-3',
        'method': 'user_provided' if cfg['substrate_doping'] else 'estimated'
    }
    
    # Debye length: λ_d = sqrt(ε_s * k * T / (q² * N))
    eps_s = EPS_0 * EPS_SI
    lambda_d = np.sqrt(eps_s * K_B * cfg['temperature'] / 
                      (Q_E**2 * n_sub * 1e6))  # Convert cm⁻³ to m⁻³
    
    # Theoretical flat-band capacitance
    if results['oxide_thickness']['value'] > 0:
        tox = results['oxide_thickness']['value']
        cfb_theory = results['cox']['value'] / (1 + lambda_d / tox)
        
        # Find voltage closest to this capacitance
        cap_diff = np.abs(capacitance_smooth - cfb_theory)
        vfb_refined_idx = np.argmin(cap_diff)
        vfb_refined = voltage[vfb_refined_idx]
        
        # Use refined value if reasonable (within 1V of initial estimate)
        if abs(vfb_refined - vfb_estimate) < 1.0:
            vfb_final = vfb_refined
        else:
            vfb_final = vfb_estimate
            results['warnings'].append("Refined Vfb differs significantly from initial estimate")
    else:
        vfb_final = vfb_estimate
    
    results['vfb'] = {
        'value': float(vfb_final),
        'unit': 'V',
        'method': 'maximum_slope'
    }
    
    results['cfb'] = {
        'value': float(np.interp(vfb_final, voltage, capacitance_smooth)),
        'unit': 'F'
    }
else:
    results['vfb'] = None
    results['cfb'] = None

# Extract threshold voltage (strong inversion onset)
# For p-substrate: Vth where C starts to increase again after minimum
# For n-substrate: Vth where C starts to decrease again after minimum

c_min = np.min(capacitance_smooth)
c_min_idx = np.argmin(capacitance_smooth)
v_at_c_min = voltage[c_min_idx]

if cfg['substrate_type'] == 'p':
    # Look for increase after minimum
    post_min = capacitance_smooth[c_min_idx:]
    v_post_min = voltage[c_min_idx:]
    
    # Find where C increases by 5% of Cox
    if results['cox'] is not None:
        c_threshold = c_min + 0.05 * results['cox']['value']
        above_thresh = post_min > c_threshold
        
        if np.any(above_thresh):
            vth_idx = np.argmax(above_thresh)
            vth = v_post_min[vth_idx]
            
            results['vth'] = {
                'value': float(vth),
                'unit': 'V',
                'method': 'inversion_onset'
            }
        else:
            results['vth'] = None
            results['warnings'].append("Could not determine threshold voltage")
    else:
        results['vth'] = None
else:
    # Similar logic for n-substrate (opposite polarity)
    pre_min = capacitance_smooth[:c_min_idx]
    v_pre_min = voltage[:c_min_idx]
    
    if len(pre_min) > 0 and results['cox'] is not None:
        c_threshold = c_min + 0.05 * results['cox']['value']
        above_thresh = pre_min > c_threshold
        
        if np.any(above_thresh):
            vth_idx = np.argmax(above_thresh[::-1])
            vth = v_pre_min[-(vth_idx+1)]
            
            results['vth'] = {
                'value': float(vth),
                'unit': 'V',
                'method': 'inversion_onset'
            }
        else:
            results['vth'] = None
            results['warnings'].append("Could not determine threshold voltage")
    else:
        results['vth'] = None

# Interface trap density (D_it) estimation
# From stretch-out: ΔV = q * D_it / Cox

if results['cox'] is not None and results['vfb'] is not None:
    # Measure stretch-out as deviation from ideal C-V
    # This is simplified; proper D_it extraction requires multiple frequencies
    
    # Find depletion region
    if cfg['substrate_type'] == 'p':
        depl_mask = (voltage > results['vfb']['value']) & (voltage < v_at_c_min + 0.5)
    else:
        depl_mask = (voltage < results['vfb']['value']) & (voltage > v_at_c_min - 0.5)
    
    if np.sum(depl_mask) >= 5:
        # Fit ideal depletion curve: 1/C² = a * V + b
        c_depl = capacitance_smooth[depl_mask]
        v_depl = voltage[depl_mask]
        
        try:
            # Linear fit of 1/C²
            inv_c_sq = 1.0 / (c_depl ** 2)
            fit_coeffs = np.polyfit(v_depl, inv_c_sq, 1)
            
            # Calculate ideal curve
            inv_c_sq_ideal = np.polyval(fit_coeffs, v_depl)
            c_ideal = 1.0 / np.sqrt(inv_c_sq_ideal)
            
            # Stretch-out: voltage difference at same capacitance
            stretch = np.mean(np.abs(v_depl - np.interp(c_depl, c_ideal, v_depl)))
            
            # D_it ≈ Cox * ΔV / q
            dit = (results['cox']['value'] / area) * stretch / Q_E * 1e-4  # Convert to cm⁻²eV⁻¹
            
            results['interface_trap_density'] = {
                'value': float(dit),
                'unit': 'cm^-2 eV^-1',
                'method': 'stretchout',
                'stretch_voltage': float(stretch)
            }
            
            if dit > 1e12:
                results['warnings'].append(f"High interface trap density: {dit:.2e} cm⁻²eV⁻¹")
                
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning):
            results['interface_trap_density'] = None
            results['warnings'].append("Could not extract interface trap density")
    else:
        results['interface_trap_density'] = None
else:
    results['interface_trap_density'] = None

# Quality score
results['quality_score'] = calculate_mos_cv_quality(results)

return results

def analyze_schottky_cv(
voltage: np.ndarray,
capacitance: np.ndarray,
area: float,
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
Analyze Schottky diode C-V characteristics and extract doping profile

The Mott-Schottky relationship:
1/C² = (2/(q * ε_s * N_D * A²)) * (V - V_bi - kT/q)

From slope: N_D = 2 / (q * ε_s * slope * A²)
From intercept: V_bi = V_intercept - kT/q
"""

# Default configuration
default_config = {
    'frequency': 1e6,
    'temperature': 300.0,
    'substrate_type': 'n',
    'relative_permittivity': EPS_SI,
    'extract_profile': True
}

if config is None:
    config = {}
cfg = {**default_config, **config}

# Input validation
voltage = np.asarray(voltage, dtype=float)
capacitance = np.asarray(capacitance, dtype=float)

if len(voltage) != len(capacitance):
    raise CVAnalysisError("Length mismatch between voltage and capacitance arrays")

if len(voltage) < 10:
    raise CVAnalysisError("Need at least 10 data points for analysis")

# Results container
results = {
    'area': area,
    'frequency': cfg['frequency'],
    'temperature': cfg['temperature'],
    'substrate_type': cfg['substrate_type'],
    'warnings': []
}

# Sort by voltage
sort_idx = np.argsort(voltage)
voltage = voltage[sort_idx]
capacitance = capacitance[sort_idx]

# Mott-Schottky plot: 1/C² vs V
inv_c_sq = 1.0 / (capacitance ** 2)

# Select linear region (avoid breakdown and series resistance effects)
# Typically reverse bias region: V < 0 for n-type
if cfg['substrate_type'] == 'n':
    linear_mask = voltage < -0.1  # Reverse bias
else:
    linear_mask = voltage > 0.1   # Reverse bias for p-type

if np.sum(linear_mask) < 5:
    results['warnings'].append("Insufficient points in linear region")
    linear_mask = np.ones(len(voltage), dtype=bool)  # Use all points

v_linear = voltage[linear_mask]
inv_c_sq_linear = inv_c_sq[linear_mask]

# Linear fit: 1/C² = slope * V + intercept
try:
    fit_coeffs = np.polyfit(v_linear, inv_c_sq_linear, 1)
    slope = fit_coeffs[0]
    intercept = fit_coeffs[1]
    
    # Calculate R² to assess linearity
    inv_c_sq_fit = np.polyval(fit_coeffs, v_linear)
    ss_res = np.sum((inv_c_sq_linear - inv_c_sq_fit) ** 2)
    ss_tot = np.sum((inv_c_sq_linear - np.mean(inv_c_sq_linear)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Extract doping concentration: N_D = 2 / (q * ε_s * slope * A²)
    eps_s = EPS_0 * cfg['relative_permittivity']
    
    if slope > 0:
        n_d = 2.0 / (Q_E * eps_s * slope * area**2)
        n_d_cm3 = n_d * 1e-6  # Convert m⁻³ to cm⁻³
        
        results['doping_concentration'] = {
            'value': float(n_d_cm3),
            'unit': 'cm^-3',
            'method': 'mott_schottky',
            'r_squared': float(r_squared)
        }
    else:
        results['doping_concentration'] = None
        results['warnings'].append("Negative slope in Mott-Schottky plot")
    
    # Extract built-in potential: V_bi = -intercept/slope - kT/q
    vt = K_B * cfg['temperature'] / Q_E
    v_bi = -intercept / slope - vt
    
    results['built_in_potential'] = {
        'value': float(v_bi),
        'unit': 'V',
        'method': 'mott_schottky_intercept'
    }
    
except (np.linalg.LinAlgError, ValueError):
    results['doping_concentration'] = None
    results['built_in_potential'] = None
    results['warnings'].append("Could not fit Mott-Schottky plot")

# Extract doping profile vs depth if requested
if cfg['extract_profile'] and results['doping_concentration'] is not None:
    profile = extract_doping_profile(
        voltage, capacitance, area, 
        cfg['relative_permittivity'], cfg['temperature']
    )
    results['doping_profile'] = profile
else:
    results['doping_profile'] = None

# Quality score
results['quality_score'] = calculate_schottky_cv_quality(results)

return results

def estimate_doping_from_cv(
voltage: np.ndarray,
capacitance: np.ndarray,
area: float,
substrate_type: str,
temperature: float
) -> float:
“””
Estimate substrate doping concentration from depletion capacitance

In depletion: C = sqrt(q * ε_s * N / (2 * (V - V_fb)))

Returns doping in cm⁻³
"""

# Find minimum capacitance (deep depletion)
c_min = np.min(capacitance)
c_min_idx = np.argmin(capacitance)
v_at_c_min = voltage[c_min_idx]

# Use Mott-Schottky: 1/C² ∝ V
# Select region around minimum
depl_window = 10
start_idx = max(0, c_min_idx - depl_window)
end_idx = min(len(capacitance), c_min_idx + depl_window)

v_depl = voltage[start_idx:end_idx]
c_depl = capacitance[start_idx:end_idx]

if len(v_depl) >= 5:
    inv_c_sq = 1.0 / (c_depl ** 2)
    
    try:
        fit_coeffs = np.polyfit(v_depl, inv_c_sq, 1)
        slope = fit_coeffs[0]
        
        if slope > 0:
            eps_s = EPS_0 * EPS_SI
            n_d = 2.0 / (Q_E * eps_s * slope * area**2)
            n_d_cm3 = n_d * 1e-6  # Convert to cm⁻³
            
            # Sanity check: typical range 1e14 - 1e18 cm⁻³
            if 1e14 <= n_d_cm3 <= 1e19:
                return n_d_cm3
    except (np.linalg.LinAlgError, ValueError):
        pass

# Default fallback
return 1e16  # cm⁻³

def extract_doping_profile(
voltage: np.ndarray,
capacitance: np.ndarray,
area: float,
eps_r: float,
temperature: float
) -> Dict[str, np.ndarray]:
“””
Extract doping concentration profile vs depth from C-V data

N(W) = -C³ / (q * ε_s * A² * dC/dV)
W = ε_s * A / C

Returns dict with 'depth' and 'concentration' arrays
"""

eps_s = EPS_0 * eps_r

# Calculate dC/dV
dv = np.diff(voltage)
dc = np.diff(capacitance)

dv_safe = np.where(dv == 0, 1e-12, dv)
dcdv = dc / dv_safe

c_deriv = (capacitance[:-1] + capacitance[1:]) / 2
v_deriv = (voltage[:-1] + voltage[1:]) / 2

# Calculate doping concentration
# N(W) = -C³ / (q * ε_s * A² * dC/dV)

concentration = -c_deriv**3 / (Q_E * eps_s * area**2 * dcdv)
concentration = np.abs(concentration) * 1e-6  # Convert to cm⁻³

# Calculate depletion depth
# W = ε_s * A / C
depth = eps_s * area / c_deriv
depth = depth * 1e9  # Convert to nm

# Filter out non-physical values
valid_mask = (concentration > 1e13) & (concentration < 1e20) & (depth > 0) & (depth < 1e4)

return {
    'depth': depth[valid_mask],
    'depth_unit': 'nm',
    'concentration': concentration[valid_mask],
    'concentration_unit': 'cm^-3'
}

def calculate_mos_cv_quality(results: Dict[str, Any]) -> int:
“”“Calculate quality score for MOS C-V analysis”””

score = 100

if results['cox'] is None:
    score -= 30

if results['vfb'] is None:
    score -= 20

if results['vth'] is None:
    score -= 15

if results['interface_trap_density'] is not None:
    if results['interface_trap_density']['value'] > 1e12:
        score -= 10

score -= len(results['warnings']) * 5

return max(0, score)

def calculate_schottky_cv_quality(results: Dict[str, Any]) -> int:
“”“Calculate quality score for Schottky C-V analysis”””

score = 100

if results['doping_concentration'] is None:
    score -= 40
elif 'r_squared' in results['doping_concentration']:
    r_sq = results['doping_concentration']['r_squared']
    if r_sq < 0.95:
        score -= 20
    elif r_sq < 0.98:
        score -= 10

if results['built_in_potential'] is None:
    score -= 20

score -= len(results['warnings']) * 5

return max(0, score)

# Example usage and test

if **name** == “**main**”:
print(“C-V Profiling Module - Test Suite”)
print(”=” * 60)

print("\n1. Testing MOS C-V Analysis...")

# Generate synthetic MOS C-V curve (p-substrate)
area = 1e-8  # 100 µm²
tox = 10e-9  # 10 nm
cox = EPS_0 * EPS_SIOZ * area / tox
n_sub = 1e16  # 1e16 cm⁻³
vfb_true = -0.9  # V

voltage_sweep = np.linspace(-2, 2, 200)

# Simplified MOS C-V model
capacitance_sweep = np.zeros_like(voltage_sweep)

for i, v in enumerate(voltage_sweep):
    if v < vfb_true - 0.5:
        # Accumulation
        capacitance_sweep[i] = cox
    elif v < vfb_true + 1.5:
        # Depletion
        # C = Cox / (1 + Cox * W_d / ε_s / A)
        # W_d ∝ sqrt(V - Vfb)
        wd = np.sqrt(2 * EPS_0 * EPS_SI * abs(v - vfb_true) / (Q_E * n_sub * 1e6))
        capacitance_sweep[i] = cox / (1 + cox * wd / (EPS_0 * EPS_SI * area))
    else:
        # Weak inversion (slowly rising)
        c_min = np.min(capacitance_sweep[capacitance_sweep > 0])
        capacitance_sweep[i] = c_min + (cox - c_min) * 0.1 * (v - vfb_true - 1.5)

# Add noise
capacitance_sweep += np.random.normal(0, cox * 0.02, len(capacitance_sweep))

# Analyze
mos_results = analyze_mos_cv(
    voltage_sweep, capacitance_sweep, area,
    config={
        'substrate_type': 'p',
        'substrate_doping': n_sub
    }
)

print(f"   Cox: {mos_results['cox']['density_uf_cm2']:.2f} µF/cm²")
print(f"   t_ox: {mos_results['oxide_thickness']['value_nm']:.2f} nm")
print(f"   Vfb: {mos_results['vfb']['value']:.3f} V (true: {vfb_true:.3f} V)")
print(f"   Quality Score: {mos_results['quality_score']}/100")

print("\n2. Testing Schottky C-V Analysis...")

# Generate synthetic Schottky C-V (Mott-Schottky)
n_d_true = 5e16  # cm⁻³
v_bi_true = 0.8  # V

voltage_schottky = np.linspace(-5, 0, 100)

# Mott-Schottky: 1/C² = slope * (V - V_bi)
eps_s = EPS_0 * EPS_SI
slope = 2.0 / (Q_E * eps_s * n_d_true * 1e6 * area**2)

inv_c_sq = slope * (voltage_schottky - v_bi_true)
inv_c_sq = np.maximum(inv_c_sq, 1e18)  # Floor

capacitance_schottky = 1.0 / np.sqrt(inv_c_sq)

# Add noise
capacitance_schottky += np.random.normal(0, np.mean(capacitance_schottky) * 0.02, 
                                          len(capacitance_schottky))

schottky_results = analyze_schottky_cv(
    voltage_schottky, capacitance_schottky, area
)

print(f"   N_D: {schottky_results['doping_concentration']['value']:.2e} cm⁻³")
print(f"   (true: {n_d_true:.2e} cm⁻³)")
print(f"   V_bi: {schottky_results['built_in_potential']['value']:.3f} V")
print(f"   (true: {v_bi_true:.3f} V)")
print(f"   R²: {schottky_results['doping_concentration']['r_squared']:.4f}")
print(f"   Quality Score: {schottky_results['quality_score']}/100")

print("\n✓ All C-V profiling tests completed successfully")
print("=" * 60)