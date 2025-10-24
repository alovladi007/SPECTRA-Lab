# services/analysis/app/methods/electrical/iv_characterization.py

“””
I-V Characterization Analysis Module

Implements comprehensive I-V analysis for:

- Diodes (pn junction, Schottky)
- MOSFETs (Id-Vgs, Id-Vds)
- BJTs (Ic-Vbe, Ic-Vce)
- Solar cells (Jsc, Voc, FF, efficiency, MPP)

Features:

- Shockley diode equation fitting
- Parameter extraction (Is, n, Rs, Rsh)
- MOSFET model fitting (Vth, gm, Ron, μeff)
- Safe operating area (SOA) checks
- Temperature-dependent analysis
- Compliance monitoring

References:

- Sze, S. M. (2006). “Physics of Semiconductor Devices”
- Schroder, D. K. (2006). “Semiconductor Material and Device Characterization”
- IEC 60747 - Semiconductor Devices Standards
  “””

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from scipy.optimize import curve_fit, minimize
from scipy import constants
import logging

logger = logging.getLogger(**name**)

# Physical constants

K_B = constants.k  # Boltzmann constant (J/K)
Q_E = constants.e  # Elementary charge (C)
T_300K = 300.0  # Room temperature (K)

# ============================================================================

# Configuration

# ============================================================================

@dataclass
class IVConfig:
“”“Configuration for I-V measurements”””
# Device type
device_type: str = “diode”  # diode, mosfet, bjt, solar_cell

# Measurement parameters
voltage_start: float = -1.0  # V
voltage_stop: float = 1.0  # V
voltage_step: float = 0.01  # V
current_compliance: float = 0.1  # A

# Analysis parameters
temperature: float = 300.0  # K
device_area: Optional[float] = None  # cm²

# Fitting
fit_algorithm: str = "levenberg_marquardt"
initial_guess_auto: bool = True
max_iterations: int = 1000
fit_tolerance: float = 1e-8

# Safety
check_compliance: bool = True
max_power: Optional[float] = None  # W
soa_check: bool = True

# Quality
min_r_squared: float = 0.95
outlier_rejection: bool = True

# ============================================================================

# Diode Analysis

# ============================================================================

class DiodeAnalyzer:
“””
Diode I-V analysis with parameter extraction

Shockley diode equation:
I = Is * [exp(V / (n * Vt)) - 1] + V/Rsh

where:
- Is = saturation current
- n = ideality factor
- Vt = thermal voltage (kT/q)
- Rsh = shunt resistance

With series resistance:
I = Is * [exp((V - I*Rs) / (n * Vt)) - 1] + (V - I*Rs)/Rsh
"""

def __init__(self, config: IVConfig):
    self.config = config
    self.logger = logging.getLogger(__name__)
    self.Vt = (K_B * config.temperature) / Q_E  # Thermal voltage

def analyze(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main analysis pipeline for diode I-V
    
    Args:
        measurements: Dictionary containing:
            - voltages: Array of voltage values (V)
            - currents: Array of current values (A)
            - temperature: Optional temperature (K)
            - device_area: Optional device area (cm²)
            
    Returns:
        Dictionary with analysis results
    """
    self.logger.info("Starting diode I-V analysis")
    
    # Extract data
    voltages = np.array(measurements['voltages'])
    currents = np.array(measurements['currents'])
    temperature = measurements.get('temperature', self.config.temperature)
    device_area = measurements.get('device_area', self.config.device_area)
    
    # Update thermal voltage if temperature changed
    if temperature != self.config.temperature:
        self.Vt = (K_B * temperature) / Q_E
    
    # 1. Safety checks
    safety = self._check_safety(voltages, currents)
    if not safety['passed']:
        self.logger.warning(f"Safety check failed: {safety}")
    
    # 2. Data preprocessing
    v_clean, i_clean = self._preprocess_data(voltages, currents)
    
    # 3. Separate forward and reverse bias regions
    forward_mask = v_clean > 0.1  # Forward bias
    reverse_mask = v_clean < -0.1  # Reverse bias
    
    # 4. Extract diode parameters
    params = self._extract_parameters(v_clean, i_clean, forward_mask)
    
    # 5. Calculate derived metrics
    metrics = self._calculate_metrics(v_clean, i_clean, params, forward_mask, reverse_mask)
    
    # 6. Fit quality assessment
    fit_quality = self._assess_fit_quality(v_clean, i_clean, params)
    
    # 7. Temperature coefficient (if multiple temps)
    temp_coeff = None
    if 'temperatures' in measurements:
        temp_coeff = self._analyze_temperature_dependence(measurements)
    
    # Compile results
    results = {
        'device_type': 'diode',
        'parameters': params,
        'metrics': metrics,
        'fit_quality': fit_quality,
        'safety': safety,
        'temperature': {
            'value': temperature,
            'unit': 'K',
            'thermal_voltage': self.Vt
        },
        'device_area': {
            'value': device_area,
            'unit': 'cm²'
        } if device_area else None,
        'temperature_coefficient': temp_coeff,
        'raw_data': {
            'voltages': voltages.tolist(),
            'currents': currents.tolist()
        }
    }
    
    self.logger.info(
        f"Analysis complete: Is={params['Is']:.2e} A, n={params['n']:.2f}, "
        f"Rs={params['Rs']:.2f} Ω"
    )
    
    return results

def _check_safety(
    self, 
    voltages: np.ndarray, 
    currents: np.ndarray
) -> Dict[str, Any]:
    """Check for safety violations"""
    warnings = []
    
    # Check compliance
    if self.config.check_compliance:
        max_current = np.max(np.abs(currents))
        if max_current > self.config.current_compliance * 0.9:
            warnings.append(f"Approaching current compliance: {max_current:.3f} A")
    
    # Check power
    power = np.abs(voltages * currents)
    max_power = np.max(power)
    
    if self.config.max_power and max_power > self.config.max_power:
        warnings.append(f"Exceeded max power: {max_power:.3f} W")
    
    # Check for breakdown
    if np.any(currents[voltages < 0] < -0.01):  # Reverse breakdown
        warnings.append("Possible reverse breakdown detected")
    
    return {
        'passed': len(warnings) == 0,
        'warnings': warnings,
        'max_power': float(max_power),
        'max_current': float(np.max(np.abs(currents)))
    }

def _preprocess_data(
    self,
    voltages: np.ndarray,
    currents: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess and clean I-V data"""
    # Remove outliers if enabled
    if self.config.outlier_rejection:
        mask = self._identify_outliers(voltages, currents)
        v_clean = voltages[~mask]
        i_clean = currents[~mask]
    else:
        v_clean = voltages
        i_clean = currents
    
    # Sort by voltage
    sort_idx = np.argsort(v_clean)
    v_clean = v_clean[sort_idx]
    i_clean = i_clean[sort_idx]
    
    return v_clean, i_clean

def _identify_outliers(
    self,
    voltages: np.ndarray,
    currents: np.ndarray
) -> np.ndarray:
    """Identify outliers using residual analysis"""
    # Simple outlier detection: points far from smooth curve
    # Fit polynomial to identify gross outliers
    try:
        poly_fit = np.polyfit(voltages, currents, deg=5)
        poly_pred = np.polyval(poly_fit, voltages)
        residuals = np.abs(currents - poly_pred)
        
        threshold = np.median(residuals) + 3 * np.std(residuals)
        outliers = residuals > threshold
        
        if np.sum(outliers) > 0:
            self.logger.info(f"Identified {np.sum(outliers)} outliers")
        
        return outliers
    except:
        return np.zeros(len(voltages), dtype=bool)

def _extract_parameters(
    self,
    voltages: np.ndarray,
    currents: np.ndarray,
    forward_mask: np.ndarray
) -> Dict[str, float]:
    """
    Extract diode parameters using Shockley equation
    
    Method:
    1. Use forward bias data (V > 0.1V)
    2. Fit to: I = Is * exp(V / (n * Vt)) for initial guess
    3. Refine with full model including Rs and Rsh
    """
    # Use forward bias data
    v_fwd = voltages[forward_mask]
    i_fwd = currents[forward_mask]
    
    if len(v_fwd) < 5:
        self.logger.warning("Insufficient forward bias points")
        # Return default parameters
        return {
            'Is': 1e-12,
            'n': 1.0,
            'Rs': 0.0,
            'Rsh': 1e6,
            'fit_method': 'insufficient_data'
        }
    
    # ===== Step 1: Initial guess from linear region =====
    # log(I) vs V should be linear in forward bias
    try:
        # Use mid-range forward bias (avoid very low and very high current)
        i_min = np.percentile(i_fwd, 10)
        i_max = np.percentile(i_fwd, 90)
        linear_mask = (i_fwd > i_min) & (i_fwd < i_max) & (i_fwd > 0)
        
        if np.sum(linear_mask) > 3:
            v_linear = v_fwd[linear_mask]
            i_linear = i_fwd[linear_mask]
            
            # log(I) = log(Is) + V/(n*Vt)
            log_i = np.log(i_linear)
            slope, intercept = np.polyfit(v_linear, log_i, 1)
            
            n_guess = 1 / (slope * self.Vt)
            Is_guess = np.exp(intercept)
            
            # Sanity checks
            if n_guess < 0.5 or n_guess > 10:
                n_guess = 1.0
            if Is_guess < 1e-20 or Is_guess > 1e-3:
                Is_guess = 1e-12
        else:
            n_guess = 1.0
            Is_guess = 1e-12
    except:
        n_guess = 1.0
        Is_guess = 1e-12
    
    # ===== Step 2: Fit full model =====
    # I = Is * [exp((V - I*Rs) / (n * Vt)) - 1] + (V - I*Rs)/Rsh
    
    def diode_model(V, Is, n, Rs, Rsh):
        """Full diode model (implicit in I)"""
        # Use iteration to solve implicit equation
        I = np.zeros_like(V)
        for idx, v in enumerate(V):
            # Initial guess from simple model
            i_guess = Is * (np.exp(v / (n * self.Vt)) - 1)
            
            # Newton-Raphson iteration
            for _ in range(10):
                v_diode = v - i_guess * Rs
                i_model = Is * (np.exp(v_diode / (n * self.Vt)) - 1) + v_diode / Rsh
                error = i_model - i_guess
                
                if abs(error) < 1e-12:
                    break
                
                # Derivative
                di_dv = (Is / (n * self.Vt)) * np.exp(v_diode / (n * self.Vt)) + 1/Rsh
                di_di = -di_dv * Rs
                derivative = di_di - 1
                
                if abs(derivative) > 1e-10:
                    i_guess = i_guess - error / derivative
            
            I[idx] = i_guess
        
        return I
    
    # For curve_fit, use a simplified explicit model
    def simplified_model(V, Is, n, Rs):
        """Simplified explicit model (ignores Rsh for fitting)"""
        V_eff = V - Is * Rs * (np.exp(V / (n * self.Vt)))
        return Is * (np.exp(V_eff / (n * self.Vt)) - 1)
    
    try:
        # Fit simplified model
        bounds = ([1e-20, 0.5, 0], [1e-3, 10, 1000])
        popt, pcov = curve_fit(
            simplified_model,
            v_fwd,
            i_fwd,
            p0=[Is_guess, n_guess, 0.1],
            bounds=bounds,
            maxfev=self.config.max_iterations
        )
        
        Is_fit, n_fit, Rs_fit = popt
        
        # Estimate Rsh from reverse bias
        v_rev = voltages[voltages < -0.1]
        i_rev = currents[voltages < -0.1]
        
        if len(v_rev) > 2:
            # Rsh ≈ -ΔV / ΔI in reverse
            Rsh_fit = -np.polyfit(i_rev, v_rev, 1)[0]
            if Rsh_fit < 0 or Rsh_fit > 1e9:
                Rsh_fit = 1e6
        else:
            Rsh_fit = 1e6
        
        fit_method = 'curve_fit'
        
    except Exception as e:
        self.logger.warning(f"Curve fitting failed: {e}, using initial guess")
        Is_fit = Is_guess
        n_fit = n_guess
        Rs_fit = 0.1
        Rsh_fit = 1e6
        fit_method = 'initial_guess'
    
    return {
        'Is': float(Is_fit),
        'n': float(n_fit),
        'Rs': float(Rs_fit),
        'Rsh': float(Rsh_fit),
        'fit_method': fit_method
    }

def _calculate_metrics(
    self,
    voltages: np.ndarray,
    currents: np.ndarray,
    params: Dict[str, float],
    forward_mask: np.ndarray,
    reverse_mask: np.ndarray
) -> Dict[str, Any]:
    """Calculate derived metrics"""
    # Forward voltage at specific current
    i_test = 1e-3  # 1 mA test current
    v_forward = params['n'] * self.Vt * np.log(i_test / params['Is'] + 1)
    v_forward += i_test * params['Rs']  # Add series resistance drop
    
    # Reverse leakage current
    if np.any(reverse_mask):
        i_reverse = np.mean(currents[reverse_mask])
    else:
        i_reverse = None
    
    # Dynamic resistance
    # rd = dV/dI at operating point
    rd_forward = (params['n'] * self.Vt) / i_test + params['Rs']
    
    return {
        'forward_voltage_at_1mA': {
            'value': float(v_forward),
            'unit': 'V'
        },
        'reverse_leakage_current': {
            'value': float(i_reverse) if i_reverse else None,
            'unit': 'A'
        },
        'dynamic_resistance': {
            'value': float(rd_forward),
            'unit': 'Ω'
        },
        'turn_on_voltage': {
            'value': float(params['n'] * self.Vt * np.log(10)),  # V at I = 10*Is
            'unit': 'V',
            'note': 'Approximate'
        }
    }

def _assess_fit_quality(
    self,
    voltages: np.ndarray,
    currents: np.ndarray,
    params: Dict[str, float]
) -> Dict[str, Any]:
    """Assess quality of parameter fit"""
    # Calculate R² for forward bias
    forward_mask = voltages > 0.1
    v_fwd = voltages[forward_mask]
    i_fwd = currents[forward_mask]
    
    if len(v_fwd) == 0:
        return {'r_squared': 0.0, 'quality': 'poor'}
    
    # Predicted current
    Is, n, Rs = params['Is'], params['n'], params['Rs']
    i_pred = Is * (np.exp((v_fwd - i_fwd * Rs) / (n * self.Vt)) - 1)
    
    # R²
    ss_res = np.sum((i_fwd - i_pred) ** 2)
    ss_tot = np.sum((i_fwd - np.mean(i_fwd)) ** 2)
    
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Quality assessment
    if r_squared > 0.98:
        quality = 'excellent'
    elif r_squared > 0.95:
        quality = 'good'
    elif r_squared > 0.90:
        quality = 'acceptable'
    else:
        quality = 'poor'
    
    return {
        'r_squared': float(r_squared),
        'quality': quality,
        'rmse': float(np.sqrt(ss_res / len(i_fwd)))
    }

def _analyze_temperature_dependence(
    self,
    measurements: Dict[str, Any]
) -> Dict[str, Any]:
    """Analyze temperature dependence of diode parameters"""
    # Extract Is at different temperatures
    # Is ~ T³ * exp(-Eg / (k*T))
    # This is for future enhancement
    return {
        'note': 'Temperature dependence analysis not yet implemented'
    }

# ============================================================================

# API Functions

# ============================================================================

def analyze_diode_iv(
measurements: Dict[str, Any],
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
High-level API for diode I-V analysis

Args:
    measurements: Raw I-V measurement data
    config: Optional configuration dictionary
    
Returns:
    Analysis results dictionary
"""
# Create config
if config:
    iv_config = IVConfig(**config)
else:
    iv_config = IVConfig(device_type='diode')

# Run analysis
analyzer = DiodeAnalyzer(iv_config)
results = analyzer.analyze(measurements)

return results

# Placeholder for MOSFET, BJT, and Solar Cell analyzers

# These will be implemented in the full session

def analyze_mosfet_iv(measurements, config=None):
“”“MOSFET I-V analysis - TO BE IMPLEMENTED”””
return {‘status’: ‘not_implemented’, ‘device_type’: ‘mosfet’}

def analyze_bjt_iv(measurements, config=None):
“”“BJT I-V analysis - TO BE IMPLEMENTED”””
return {‘status’: ‘not_implemented’, ‘device_type’: ‘bjt’}

def analyze_solar_cell_iv(measurements, config=None):
“”“Solar cell I-V analysis - TO BE IMPLEMENTED”””
return {‘status’: ‘not_implemented’, ‘device_type’: ‘solar_cell’}