# services/analysis/app/methods/electrical/mosfet_analysis.py

“””
MOSFET I-V Characterization and Parameter Extraction

This module provides comprehensive analysis for MOSFET devices including:

- Transfer characteristics (Id-Vgs) analysis
- Output characteristics (Id-Vds) analysis
- Threshold voltage extraction (linear extrapolation, constant current, transconductance methods)
- Transconductance (gm) calculation
- Output resistance and on-resistance
- Subthreshold slope
- Channel length modulation parameter (λ)
- Mobility extraction
- Safe operating area (SOA) validation

References:

- Schroder, “Semiconductor Material and Device Characterization” (2006)
- Sze & Ng, “Physics of Semiconductor Devices” (2007)
- IEEE Standard 1620-2008
  “””

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Physical constants

Q_E = 1.602176634e-19  # Elementary charge (C)
K_B = 1.380649e-23      # Boltzmann constant (J/K)
EPS_0 = 8.854187817e-12  # Vacuum permittivity (F/m)

class MOSFETAnalysisError(Exception):
“”“Custom exception for MOSFET analysis errors”””
pass

def analyze_mosfet_transfer(
vgs: np.ndarray,
ids: np.ndarray,
vds: float,
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
Analyze MOSFET transfer characteristics (Id-Vgs at constant Vds)

Parameters:
-----------
vgs : array_like
    Gate-source voltage sweep (V)
ids : array_like  
    Drain-source current measurements (A)
vds : float
    Drain-source voltage (V) - held constant during sweep
config : dict, optional
    Configuration parameters:
    - vth_method: 'linear_extrapolation' | 'constant_current' | 'transconductance' (default: 'linear_extrapolation')
    - vth_current_threshold: Current threshold for constant current method (default: 1e-7 A)
    - subthreshold_fit_points: Number of points for subthreshold slope (default: 50)
    - smooth_window: Savitzky-Golay filter window (default: 5)
    - smooth_polyorder: Polynomial order for smoothing (default: 2)
    - temperature: Temperature in K (default: 300)
    - width: Channel width in m (optional)
    - length: Channel length in m (optional)
    - cox: Gate oxide capacitance in F/m² (optional)
    
Returns:
--------
dict : Analysis results containing:
    - vth: Threshold voltage and extraction method
    - gm_max: Maximum transconductance
    - gm_peak_vgs: Vgs at peak gm
    - subthreshold_slope: Subthreshold swing (mV/decade)
    - ioff: Off-state current
    - ion: On-state current  
    - ion_ioff_ratio: Current on/off ratio
    - mobility: Carrier mobility (if width, length, cox provided)
    - quality_score: 0-100 assessment
    - warnings: List of warnings
"""

# Default configuration
default_config = {
    'vth_method': 'linear_extrapolation',
    'vth_current_threshold': 1e-7,
    'subthreshold_fit_points': 50,
    'smooth_window': 5,
    'smooth_polyorder': 2,
    'temperature': 300.0,
    'width': None,
    'length': None,
    'cox': None
}

if config is None:
    config = {}
cfg = {**default_config, **config}

# Input validation
vgs = np.asarray(vgs, dtype=float)
ids = np.asarray(ids, dtype=float)

if len(vgs) != len(ids):
    raise MOSFETAnalysisError("Length mismatch between vgs and ids arrays")

if len(vgs) < 10:
    raise MOSFETAnalysisError("Need at least 10 data points for analysis")

if not np.all(np.isfinite(vgs)) or not np.all(np.isfinite(ids)):
    raise MOSFETAnalysisError("Non-finite values detected in input data")

# Ensure vgs is sorted
sort_idx = np.argsort(vgs)
vgs = vgs[sort_idx]
ids = np.abs(ids[sort_idx])  # Use absolute value for n-MOS and p-MOS

# Results container
results = {
    'vds': vds,
    'num_points': len(vgs),
    'warnings': []
}

# Smooth data for derivative calculations
if len(ids) >= cfg['smooth_window']:
    ids_smooth = savgol_filter(ids, cfg['smooth_window'], cfg['smooth_polyorder'])
else:
    ids_smooth = ids
    results['warnings'].append("Too few points for smoothing, using raw data")

# Extract key currents
results['ioff'] = {
    'value': float(np.min(ids)),
    'unit': 'A',
    'vgs_at_min': float(vgs[np.argmin(ids)])
}

results['ion'] = {
    'value': float(np.max(ids)),
    'unit': 'A',
    'vgs_at_max': float(vgs[np.argmax(ids)])
}

# Ion/Ioff ratio
if results['ioff']['value'] > 0:
    results['ion_ioff_ratio'] = {
        'value': results['ion']['value'] / results['ioff']['value'],
        'decades': float(np.log10(results['ion']['value'] / results['ioff']['value']))
    }
else:
    results['ion_ioff_ratio'] = {
        'value': float('inf'),
        'decades': float('inf')
    }
    results['warnings'].append("Ioff too small for accurate ratio calculation")

# Calculate transconductance gm = dIds/dVgs
dvgs = np.diff(vgs)
dids = np.diff(ids_smooth)

# Avoid division by zero
dvgs_safe = np.where(dvgs == 0, 1e-12, dvgs)
gm = dids / dvgs_safe
vgs_gm = (vgs[:-1] + vgs[1:]) / 2  # Midpoint voltages

# Find maximum transconductance
gm_max_idx = np.argmax(gm)
results['gm_max'] = {
    'value': float(gm[gm_max_idx]),
    'unit': 'S',
    'vgs': float(vgs_gm[gm_max_idx])
}

# Threshold voltage extraction
results['vth'] = extract_threshold_voltage(
    vgs, ids, ids_smooth, gm, vgs_gm, cfg
)

# Subthreshold slope (S = dVgs / d(log10(Ids)))
# Find subthreshold region (before Vth)
if results['vth']['value'] is not None:
    subth_mask = vgs < results['vth']['value']
    if np.sum(subth_mask) >= cfg['subthreshold_fit_points']:
        vgs_subth = vgs[subth_mask][-cfg['subthreshold_fit_points']:]
        ids_subth = ids[subth_mask][-cfg['subthreshold_fit_points']:]
        
        # Linear fit: log10(Ids) vs Vgs
        # Filter out zero/negative currents
        valid_mask = ids_subth > 0
        if np.sum(valid_mask) >= 10:
            log_ids = np.log10(ids_subth[valid_mask])
            vgs_valid = vgs_subth[valid_mask]
            
            # Fit: log10(Ids) = a * Vgs + b
            # S = 1/a converted to mV/decade
            try:
                fit_coeffs = np.polyfit(vgs_valid, log_ids, 1)
                slope_per_v = fit_coeffs[0]  # decade/V
                
                if slope_per_v > 0:
                    subth_slope = 1000.0 / slope_per_v  # mV/decade
                    
                    # Theoretical limit: S_ideal = (k*T/q) * ln(10) ≈ 60 mV/decade at 300K
                    s_ideal = (K_B * cfg['temperature'] / Q_E) * np.log(10) * 1000  # mV/decade
                    
                    results['subthreshold_slope'] = {
                        'value': float(subth_slope),
                        'unit': 'mV/decade',
                        'ideal_at_T': float(s_ideal),
                        'dibl_factor': float(subth_slope / s_ideal)
                    }
                    
                    if subth_slope > 3 * s_ideal:
                        results['warnings'].append(f"Poor subthreshold slope: {subth_slope:.1f} mV/dec (>3x ideal)")
                else:
                    results['warnings'].append("Invalid subthreshold slope (negative)")
                    results['subthreshold_slope'] = None
                    
            except np.linalg.LinAlgError:
                results['warnings'].append("Could not fit subthreshold region")
                results['subthreshold_slope'] = None
        else:
            results['warnings'].append("Insufficient valid points in subthreshold region")
            results['subthreshold_slope'] = None
    else:
        results['warnings'].append("Too few points in subthreshold region")
        results['subthreshold_slope'] = None
else:
    results['subthreshold_slope'] = None

# Mobility extraction (if geometry provided)
if all(k in cfg and cfg[k] is not None for k in ['width', 'length', 'cox']):
    results['mobility'] = extract_mobility_transfer(
        vgs, ids, vds, results['vth']['value'], 
        cfg['width'], cfg['length'], cfg['cox'], results['warnings']
    )
else:
    results['mobility'] = None

# Quality score
results['quality_score'] = calculate_transfer_quality(results)

return results

def extract_threshold_voltage(
vgs: np.ndarray,
ids: np.ndarray,
ids_smooth: np.ndarray,
gm: np.ndarray,
vgs_gm: np.ndarray,
config: Dict[str, Any]
) -> Dict[str, Any]:
“”“Extract threshold voltage using specified method”””

method = config['vth_method']
vth_result = {'method': method, 'value': None, 'unit': 'V'}

if method == 'linear_extrapolation':
    # Find linear region (maximum gm region)
    gm_max_idx = np.argmax(gm)
    
    # Take points around peak gm for linear fit
    fit_window = min(10, len(gm) // 4)
    start_idx = max(0, gm_max_idx - fit_window // 2)
    end_idx = min(len(gm), gm_max_idx + fit_window // 2)
    
    if end_idx - start_idx >= 5:
        # Linear fit: Ids = a * Vgs + b
        vgs_fit = vgs_gm[start_idx:end_idx]
        ids_fit = ids_smooth[start_idx:end_idx]
        
        try:
            fit_coeffs = np.polyfit(vgs_fit, ids_fit, 1)
            # Extrapolate to Ids = 0
            vth = -fit_coeffs[1] / fit_coeffs[0]
            
            # Sanity check: Vth should be within measurement range +/- 50%
            vgs_range = vgs[-1] - vgs[0]
            if vgs[0] - 0.5*vgs_range <= vth <= vgs[-1] + 0.5*vgs_range:
                vth_result['value'] = float(vth)
                vth_result['fit_coefficients'] = fit_coeffs.tolist()
            else:
                vth_result['value'] = None
                vth_result['error'] = f"Extrapolated Vth={vth:.2f}V outside reasonable range"
                
        except np.linalg.LinAlgError:
            vth_result['value'] = None
            vth_result['error'] = "Linear fit failed"
    else:
        vth_result['value'] = None
        vth_result['error'] = "Insufficient points for linear fit"

elif method == 'constant_current':
    # Vth = Vgs when Ids crosses threshold current
    i_thresh = config['vth_current_threshold']
    
    # Find crossing point
    above_thresh = ids >= i_thresh
    if np.any(above_thresh):
        cross_idx = np.argmax(above_thresh)
        if cross_idx > 0:
            # Linear interpolation
            i1, i2 = ids[cross_idx-1], ids[cross_idx]
            v1, v2 = vgs[cross_idx-1], vgs[cross_idx]
            
            if i2 != i1:
                vth = v1 + (i_thresh - i1) * (v2 - v1) / (i2 - i1)
                vth_result['value'] = float(vth)
                vth_result['threshold_current'] = i_thresh
            else:
                vth_result['value'] = float(v1)
                vth_result['threshold_current'] = i_thresh
        else:
            vth_result['value'] = float(vgs[0])
            vth_result['threshold_current'] = i_thresh
            vth_result['warning'] = "Current above threshold at first point"
    else:
        vth_result['value'] = None
        vth_result['error'] = "Current never exceeds threshold"

elif method == 'transconductance':
    # Vth at maximum gm
    gm_max_idx = np.argmax(gm)
    vth_result['value'] = float(vgs_gm[gm_max_idx])
    vth_result['gm_at_vth'] = float(gm[gm_max_idx])

else:
    raise MOSFETAnalysisError(f"Unknown Vth extraction method: {method}")

return vth_result

def extract_mobility_transfer(
vgs: np.ndarray,
ids: np.ndarray,
vds: float,
vth: Optional[float],
width: float,
length: float,
cox: float,
warnings: List[str]
) -> Optional[Dict[str, Any]]:
“””
Extract carrier mobility from transfer characteristics

In linear region (Vds << Vgs-Vth):
Ids = μ * Cox * (W/L) * (Vgs - Vth) * Vds

Therefore: μ = Ids / [Cox * (W/L) * (Vgs - Vth) * Vds]
"""

if vth is None:
    warnings.append("Cannot extract mobility without valid Vth")
    return None

# Find linear region: Vgs > Vth and well above threshold
linear_mask = vgs > (vth + 0.5)  # At least 0.5V above Vth

if np.sum(linear_mask) < 5:
    warnings.append("Insufficient points in linear region for mobility extraction")
    return None

vgs_lin = vgs[linear_mask]
ids_lin = ids[linear_mask]

# Calculate mobility for each point
vgs_vth = vgs_lin - vth
denominator = cox * (width / length) * vgs_vth * vds

# Avoid division by zero
valid_mask = denominator > 0
if np.sum(valid_mask) < 3:
    warnings.append("Too few valid points for mobility calculation")
    return None

mobility_values = ids_lin[valid_mask] / denominator[valid_mask]

# Take median to reduce outlier influence
mobility_median = np.median(mobility_values)
mobility_mean = np.mean(mobility_values)
mobility_std = np.std(mobility_values)

return {
    'value': float(mobility_median),
    'unit': 'm^2/(V·s)',
    'unit_cgs': 'cm^2/(V·s)',
    'value_cgs': float(mobility_median * 1e4),  # Convert to cm²/(V·s)
    'mean': float(mobility_mean * 1e4),
    'std': float(mobility_std * 1e4),
    'cv_percent': float(100 * mobility_std / mobility_mean) if mobility_mean > 0 else None
}

def analyze_mosfet_output(
vds: np.ndarray,
ids: np.ndarray,
vgs_values: List[float],
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
Analyze MOSFET output characteristics (Id-Vds at multiple Vgs)

Parameters:
-----------
vds : array_like
    Drain-source voltage sweep (V)
ids : array_like
    Drain-source current measurements (A), shape (n_vgs, n_vds)
vgs_values : list
    List of gate voltages corresponding to each curve
config : dict, optional
    Configuration parameters
    
Returns:
--------
dict : Analysis results
"""

# Default configuration
default_config = {
    'ron_vds_max': 0.1,  # Max Vds for Ron calculation
    'lambda_vds_min': None,  # Min Vds for lambda extraction (auto if None)
    'temperature': 300.0
}

if config is None:
    config = {}
cfg = {**default_config, **config}

# Input validation
vds = np.asarray(vds, dtype=float)
ids = np.asarray(ids, dtype=float)

if ids.ndim == 1:
    # Single curve
    ids = ids.reshape(1, -1)

n_curves, n_points = ids.shape

if len(vgs_values) != n_curves:
    raise MOSFETAnalysisError(f"Number of Vgs values ({len(vgs_values)}) doesn't match number of curves ({n_curves})")

if n_points != len(vds):
    raise MOSFETAnalysisError(f"Vds array length ({len(vds)}) doesn't match Ids columns ({n_points})")

results = {
    'num_curves': n_curves,
    'vgs_values': vgs_values,
    'curves': [],
    'warnings': []
}

# Analyze each curve
for i, vgs in enumerate(vgs_values):
    curve_ids = np.abs(ids[i, :])
    curve_result = analyze_single_output_curve(vds, curve_ids, vgs, cfg)
    results['curves'].append(curve_result)

# Extract on-resistance from highest Vgs curve
max_vgs_idx = np.argmax(vgs_values)
results['ron'] = results['curves'][max_vgs_idx].get('ron')

# Quality score
results['quality_score'] = calculate_output_quality(results)

return results

def analyze_single_output_curve(
vds: np.ndarray,
ids: np.ndarray,
vgs: float,
config: Dict[str, Any]
) -> Dict[str, Any]:
“”“Analyze a single output characteristic curve”””

result = {'vgs': vgs}

# Find saturation point (where dIds/dVds becomes small)
dvds = np.diff(vds)
dids = np.diff(ids)

dvds_safe = np.where(dvds == 0, 1e-12, dvds)
output_conductance = dids / dvds_safe

# Saturation when conductance drops below threshold
# Typically gds < 0.1 * gds_max
gds_threshold = 0.1 * np.max(output_conductance) if np.max(output_conductance) > 0 else 0

sat_mask = output_conductance < gds_threshold
if np.any(sat_mask):
    sat_idx = np.argmax(sat_mask)
    result['vds_sat'] = {
        'value': float(vds[sat_idx]),
        'unit': 'V',
        'ids_at_sat': float(ids[sat_idx])
    }
else:
    result['vds_sat'] = None
    result['warning'] = "Saturation not clearly reached"

# On-resistance (linear region, low Vds)
ron_mask = vds <= config['ron_vds_max']
if np.sum(ron_mask) >= 3:
    vds_lin = vds[ron_mask]
    ids_lin = ids[ron_mask]
    
    # Linear fit: Ids = Vds / Ron
    if ids_lin[-1] > 1e-9:  # Avoid near-zero currents
        try:
            fit_coeffs = np.polyfit(vds_lin, ids_lin, 1)
            conductance = fit_coeffs[0]
            
            if conductance > 0:
                ron = 1.0 / conductance
                result['ron'] = {
                    'value': float(ron),
                    'unit': 'Ω',
                    'unit_display': 'Ohm',
                    'vgs': vgs,
                    'conductance': float(conductance)
                }
            else:
                result['ron'] = None
        except (np.linalg.LinAlgError, RuntimeWarning):
            result['ron'] = None
    else:
        result['ron'] = None
else:
    result['ron'] = None

# Channel length modulation (lambda): Ids_sat = K * (Vgs - Vth)^2 * (1 + lambda * Vds)
# In saturation, dIds/dVds = K * (Vgs - Vth)^2 * lambda
# Approximate: lambda ≈ (1/Ids) * (dIds/dVds) in saturation

if result['vds_sat'] is not None:
    sat_start = int(sat_idx * 0.9)  # Start slightly before saturation
    if sat_start < len(output_conductance) - 5:
        gds_sat = np.mean(output_conductance[sat_start:])
        ids_sat_mean = np.mean(ids[sat_start:])
        
        if ids_sat_mean > 1e-9 and gds_sat > 0:
            lambda_param = gds_sat / ids_sat_mean
            result['lambda'] = {
                'value': float(lambda_param),
                'unit': '1/V',
                'early_voltage': float(1.0 / lambda_param) if lambda_param > 0 else None
            }
        else:
            result['lambda'] = None
    else:
        result['lambda'] = None
else:
    result['lambda'] = None

return result

def calculate_transfer_quality(results: Dict[str, Any]) -> int:
“”“Calculate quality score (0-100) for transfer characteristics”””

score = 100

# Deduct points for issues
if results['vth']['value'] is None:
    score -= 30

if results['subthreshold_slope'] is None:
    score -= 20
elif results['subthreshold_slope']['dibl_factor'] > 2:
    score -= 10

if results['ion_ioff_ratio']['decades'] < 3:
    score -= 15
elif results['ion_ioff_ratio']['decades'] < 5:
    score -= 5

if len(results['warnings']) > 0:
    score -= len(results['warnings']) * 5

return max(0, score)

def calculate_output_quality(results: Dict[str, Any]) -> int:
“”“Calculate quality score (0-100) for output characteristics”””

score = 100

# Check if saturation is reached in most curves
curves_with_sat = sum(1 for c in results['curves'] if c.get('vds_sat') is not None)
sat_fraction = curves_with_sat / results['num_curves']

if sat_fraction < 0.5:
    score -= 30
elif sat_fraction < 0.8:
    score -= 15

# Check if Ron is extracted
if results['ron'] is None:
    score -= 20

if len(results['warnings']) > 0:
    score -= len(results['warnings']) * 5

return max(0, score)

# Example usage and test

if **name** == “**main**”:
print(“MOSFET Analysis Module - Test Suite”)
print(”=” * 60)

# Generate synthetic transfer curve (n-MOSFET)
print("\n1. Testing Transfer Characteristics...")

vgs_sweep = np.linspace(-1, 3, 200)
vth_true = 0.5
gm_max_true = 0.01  # S

# Subthreshold
subth_mask = vgs_sweep < vth_true
ids_subth = 1e-12 * np.exp((vgs_sweep[subth_mask] - vth_true) / 0.065)

# Above threshold (square law in saturation)
ids_linear = gm_max_true * (vgs_sweep[~subth_mask] - vth_true) ** 2

ids_sweep = np.concatenate([ids_subth, ids_linear])

# Add noise
ids_sweep += np.random.normal(0, np.max(ids_sweep) * 0.01, len(ids_sweep))
ids_sweep = np.maximum(ids_sweep, 1e-15)  # Floor at 1 fA

# Analyze
transfer_results = analyze_mosfet_transfer(
    vgs_sweep, ids_sweep, vds=0.1,
    config={
        'vth_method': 'linear_extrapolation',
        'width': 10e-6,
        'length': 1e-6,
        'cox': 3.45e-3  # 3.45 mF/m² for 10nm oxide
    }
)

print(f"   Vth (extracted): {transfer_results['vth']['value']:.3f} V")
print(f"   Vth (true): {vth_true:.3f} V")
print(f"   gm_max: {transfer_results['gm_max']['value']*1e3:.2f} mS")
print(f"   Ion/Ioff: {transfer_results['ion_ioff_ratio']['decades']:.1f} decades")

if transfer_results['subthreshold_slope']:
    print(f"   S: {transfer_results['subthreshold_slope']['value']:.1f} mV/dec")

print(f"   Quality Score: {transfer_results['quality_score']}/100")

# Test output characteristics
print("\n2. Testing Output Characteristics...")

vds_sweep = np.linspace(0, 5, 100)
vgs_list = [0.5, 1.0, 1.5, 2.0, 2.5]

ids_output = []
for vgs in vgs_list:
    if vgs < vth_true:
        ids = np.ones_like(vds_sweep) * 1e-12
    else:
        # Linear region: Ids ∝ Vds
        # Saturation: Ids ∝ (Vgs-Vth)^2
        vds_sat = vgs - vth_true
        ids = np.where(
            vds_sweep < vds_sat,
            # Linear
            0.001 * (vgs - vth_true) * vds_sweep * (1 - 0.5 * vds_sweep / vds_sat),
            # Saturation
            0.001 * (vgs - vth_true) ** 2 * (1 + 0.1 * (vds_sweep - vds_sat))
        )
    ids_output.append(ids)

ids_output = np.array(ids_output)

# Add noise
ids_output += np.random.normal(0, np.max(ids_output) * 0.01, ids_output.shape)
ids_output = np.maximum(ids_output, 1e-15)

output_results = analyze_mosfet_output(vds_sweep, ids_output, vgs_list)

print(f"   Number of curves: {output_results['num_curves']}")
print(f"   Ron (at Vgs={vgs_list[-1]}V): {output_results['ron']['value']:.2f} Ω")
print(f"   Quality Score: {output_results['quality_score']}/100")

print("\n✓ All tests completed successfully")
print("=" * 60)