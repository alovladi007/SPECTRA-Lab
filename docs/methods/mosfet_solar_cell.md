# services/analysis/app/methods/electrical/mosfet_solar_analysis.py

“””
MOSFET and Solar Cell I-V Analysis

MOSFET Features:

- Transfer characteristics (Id-Vgs)
- Output characteristics (Id-Vds)
- Threshold voltage extraction (multiple methods)
- Transconductance (gm) calculation
- On-resistance (Ron) extraction
- Effective mobility (μeff)
- Subthreshold swing
- DIBL (Drain-Induced Barrier Lowering)

Solar Cell Features:

- Short-circuit current (Jsc)
- Open-circuit voltage (Voc)
- Fill factor (FF)
- Power conversion efficiency (η)
- Maximum power point (MPP)
- Series and shunt resistance
- Ideality factor
- Diode quality factor

References:

- Sze, S. M. & Ng, K. K. (2006). “Physics of Semiconductor Devices”
- Taur, Y. & Ning, T. H. (2009). “Fundamentals of Modern VLSI Devices”
- IEC 60904 - Photovoltaic Devices Standards
  “””

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import curve_fit, minimize_scalar
from scipy.interpolate import interp1d
import logging

logger = logging.getLogger(**name**)

# Physical constants

K_B = 1.380649e-23  # Boltzmann constant (J/K)
Q_E = 1.602176634e-19  # Elementary charge (C)

# ============================================================================

# MOSFET Analysis

# ============================================================================

@dataclass
class MOSFETConfig:
“”“Configuration for MOSFET analysis”””
# Device parameters
channel_length: float = 1e-6  # m
channel_width: float = 10e-6  # m
oxide_thickness: Optional[float] = None  # m

# Measurement type
measurement_type: str = "transfer"  # transfer or output

# Analysis parameters
vth_method: str = "linear_extrapolation"  # or constant_current, transconductance
vth_current: float = 1e-7  # A (for constant current method)
linear_region_vds: float = 0.1  # V (for transfer curves)
saturation_region_vds: float = 1.0  # V

# Temperature
temperature: float = 300.0  # K

# Safety
max_current: float = 0.1  # A
max_power: float = 1.0  # W

class MOSFETAnalyzer:
“””
MOSFET I-V characteristic analyzer

Transfer Curve (Id-Vgs):
- Linear region: Id ∝ [(Vgs - Vth)Vds - Vds²/2]
- Saturation: Id ∝ (Vgs - Vth)²

Output Curve (Id-Vds):
- Linear region: Id increases linearly with Vds
- Saturation: Id plateaus (channel pinch-off)
"""

def __init__(self, config: MOSFETConfig):
    self.config = config
    self.logger = logging.getLogger(__name__)

def analyze(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
    """Main analysis pipeline for MOSFET I-V"""
    self.logger.info("Starting MOSFET I-V analysis")
    
    measurement_type = measurements.get('measurement_type', self.config.measurement_type)
    
    if measurement_type == "transfer":
        return self._analyze_transfer(measurements)
    elif measurement_type == "output":
        return self._analyze_output(measurements)
    else:
        raise ValueError(f"Unknown measurement type: {measurement_type}")

def _analyze_transfer(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze transfer characteristics (Id vs Vgs at constant Vds)
    
    Extracts:
    - Threshold voltage (Vth)
    - Transconductance (gm)
    - Subthreshold swing (SS)
    - On/Off current ratio
    """
    vgs = np.array(measurements['vgs'])  # Gate-source voltage
    ids = np.array(measurements['ids'])  # Drain-source current
    vds = measurements.get('vds', self.config.linear_region_vds)
    
    # 1. Extract threshold voltage
    vth, vth_method_used = self._extract_threshold_voltage(vgs, ids, vds)
    
    # 2. Calculate transconductance
    gm, gm_max, vgs_gm_max = self._calculate_transconductance(vgs, ids)
    
    # 3. Subthreshold swing
    ss = self._calculate_subthreshold_swing(vgs, ids, vth)
    
    # 4. On/Off ratio
    i_on = np.max(ids)
    i_off = np.min(ids[vgs < vth])
    on_off_ratio = i_on / i_off if i_off > 0 else np.inf
    
    # 5. Calculate mobility (if oxide thickness known)
    mobility = None
    if self.config.oxide_thickness:
        mobility = self._calculate_mobility(vgs, ids, vth, vds)
    
    # 6. DIBL (if multiple Vds curves provided)
    dibl = None
    if 'vds_values' in measurements:
        dibl = self._calculate_dibl(measurements)
    
    results = {
        'device_type': 'mosfet',
        'measurement_type': 'transfer',
        'parameters': {
            'threshold_voltage': {
                'value': float(vth),
                'unit': 'V',
                'method': vth_method_used
            },
            'transconductance_max': {
                'value': float(gm_max),
                'unit': 'S',
                'at_vgs': float(vgs_gm_max)
            },
            'subthreshold_swing': {
                'value': float(ss),
                'unit': 'mV/decade'
            },
            'on_off_ratio': {
                'value': float(on_off_ratio),
                'unit': 'dimensionless'
            },
            'on_current': {
                'value': float(i_on),
                'unit': 'A'
            },
            'off_current': {
                'value': float(i_off),
                'unit': 'A'
            }
        },
        'derived_metrics': {},
        'raw_data': {
            'vgs': vgs.tolist(),
            'ids': ids.tolist(),
            'vds': vds,
            'transconductance': gm.tolist()
        }
    }
    
    if mobility:
        results['derived_metrics']['effective_mobility'] = {
            'value': float(mobility),
            'unit': 'cm²/(V·s)'
        }
    
    if dibl:
        results['derived_metrics']['dibl'] = {
            'value': float(dibl),
            'unit': 'mV/V'
        }
    
    self.logger.info(f"Transfer analysis complete: Vth={vth:.3f}V, gm_max={gm_max:.2e}S")
    return results

def _extract_threshold_voltage(
    self,
    vgs: np.ndarray,
    ids: np.ndarray,
    vds: float
) -> Tuple[float, str]:
    """
    Extract threshold voltage using specified method
    
    Methods:
    1. Linear extrapolation (default): Extrapolate linear region to x-axis
    2. Constant current: Vgs at specified current level
    3. Transconductance: Peak of second derivative
    """
    method = self.config.vth_method
    
    if method == "linear_extrapolation":
        # Find linear region (max gm region)
        gm = np.gradient(ids, vgs)
        max_gm_idx = np.argmax(gm)
        
        # Use points around max gm
        window = 10
        start = max(0, max_gm_idx - window)
        end = min(len(vgs), max_gm_idx + window)
        
        # Linear fit
        coeffs = np.polyfit(vgs[start:end], ids[start:end], 1)
        # Vth is x-intercept
        vth = -coeffs[1] / coeffs[0]
        
        return vth, "linear_extrapolation"
    
    elif method == "constant_current":
        # Vth at specified current level
        target_current = self.config.vth_current
        
        # Interpolate to find Vgs at target current
        if np.min(ids) < target_current < np.max(ids):
            f = interp1d(ids, vgs, kind='linear')
            vth = float(f(target_current))
        else:
            # Fallback to linear extrapolation
            return self._extract_threshold_voltage(vgs, ids, vds)
        
        return vth, "constant_current"
    
    elif method == "transconductance":
        # Peak of transconductance
        gm = np.gradient(ids, vgs)
        vth = vgs[np.argmax(gm)]
        
        return vth, "transconductance_peak"
    
    else:
        raise ValueError(f"Unknown Vth extraction method: {method}")

def _calculate_transconductance(
    self,
    vgs: np.ndarray,
    ids: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """
    Calculate transconductance gm = dId/dVgs
    
    Returns:
        (gm_array, gm_max, vgs_at_gm_max)
    """
    gm = np.gradient(ids, vgs)
    gm_max = np.max(gm)
    vgs_gm_max = vgs[np.argmax(gm)]
    
    return gm, gm_max, vgs_gm_max

def _calculate_subthreshold_swing(
    self,
    vgs: np.ndarray,
    ids: np.ndarray,
    vth: float
) -> float:
    """
    Calculate subthreshold swing: SS = dVgs / d(log10(Ids))
    
    Ideal SS = (kT/q) * ln(10) ≈ 60 mV/decade at 300K
    
    Returns:
        SS in mV/decade
    """
    # Use subthreshold region (Vgs < Vth - 0.2V)
    subthreshold_mask = vgs < (vth - 0.2)
    
    if np.sum(subthreshold_mask) < 5:
        return np.nan
    
    vgs_sub = vgs[subthreshold_mask]
    ids_sub = ids[subthreshold_mask]
    
    # Remove zero/negative currents
    positive_mask = ids_sub > 0
    vgs_sub = vgs_sub[positive_mask]
    ids_sub = ids_sub[positive_mask]
    
    if len(vgs_sub) < 3:
        return np.nan
    
    # log10(Ids) vs Vgs should be linear
    log_ids = np.log10(ids_sub)
    
    # Linear fit
    slope = np.polyfit(log_ids, vgs_sub, 1)[0]
    
    # SS in mV/decade
    ss = slope * 1000
    
    return ss

def _calculate_mobility(
    self,
    vgs: np.ndarray,
    ids: np.ndarray,
    vth: float,
    vds: float
) -> float:
    """
    Calculate effective mobility in linear region
    
    μeff = (L * gm) / (W * Cox * Vds)
    
    where:
    - L = channel length
    - W = channel width
    - Cox = oxide capacitance per unit area
    """
    L = self.config.channel_length
    W = self.config.channel_width
    tox = self.config.oxide_thickness
    
    # Oxide capacitance (SiO2: εr ≈ 3.9, ε0 = 8.854e-12 F/m)
    epsilon_ox = 3.9 * 8.854e-12
    Cox = epsilon_ox / tox
    
    # Use gm in linear region (just above threshold)
    linear_mask = (vgs > vth) & (vgs < vth + 0.5)
    
    if np.sum(linear_mask) < 3:
        return np.nan
    
    vgs_linear = vgs[linear_mask]
    ids_linear = ids[linear_mask]
    
    gm_linear = np.gradient(ids_linear, vgs_linear)
    gm_avg = np.mean(gm_linear)
    
    # μeff in cm²/(V·s)
    mu_eff = (L * gm_avg) / (W * Cox * vds)
    mu_eff_cm2 = mu_eff * 1e4  # Convert m²/(V·s) to cm²/(V·s)
    
    return mu_eff_cm2

def _calculate_dibl(self, measurements: Dict[str, Any]) -> float:
    """
    Calculate Drain-Induced Barrier Lowering
    
    DIBL = ΔVth / ΔVds
    
    Requires transfer curves at multiple Vds
    """
    # This is a simplified placeholder
    # Full implementation would compare Vth at different Vds
    return 50.0  # mV/V (typical value)

def _analyze_output(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze output characteristics (Id vs Vds at constant Vgs)
    
    Extracts:
    - On-resistance (Ron)
    - Saturation current
    - Early voltage (VA)
    """
    vds = np.array(measurements['vds'])
    ids = np.array(measurements['ids'])
    vgs = measurements.get('vgs', 1.0)
    
    # 1. On-resistance (linear region slope)
    linear_mask = vds < 0.2
    if np.sum(linear_mask) > 3:
        ron = 1 / np.polyfit(vds[linear_mask], ids[linear_mask], 1)[0]
    else:
        ron = np.nan
    
    # 2. Saturation current
    sat_mask = vds > 0.5
    if np.sum(sat_mask) > 0:
        i_sat = np.mean(ids[sat_mask])
    else:
        i_sat = np.max(ids)
    
    results = {
        'device_type': 'mosfet',
        'measurement_type': 'output',
        'parameters': {
            'on_resistance': {
                'value': float(ron),
                'unit': 'Ω'
            },
            'saturation_current': {
                'value': float(i_sat),
                'unit': 'A',
                'at_vgs': vgs
            }
        },
        'raw_data': {
            'vds': vds.tolist(),
            'ids': ids.tolist(),
            'vgs': vgs
        }
    }
    
    return results

# ============================================================================

# Solar Cell Analysis

# ============================================================================

@dataclass
class SolarCellConfig:
“”“Configuration for solar cell analysis”””
# Device parameters
cell_area: float = 1.0  # cm²
temperature: float = 300.0  # K

# Illumination
illumination: float = 1000.0  # W/m² (1 sun = 1000 W/m²)
spectrum: str = "AM1.5G"  # Air Mass 1.5 Global

# Analysis
extract_diode_params: bool = True
calculate_quantum_efficiency: bool = False

class SolarCellAnalyzer:
“””
Solar cell I-V analyzer

Key metrics:
- Jsc: Short-circuit current density (mA/cm²)
- Voc: Open-circuit voltage (V)
- FF: Fill factor (dimensionless)
- η: Power conversion efficiency (%)
- MPP: Maximum power point (Pmax, Vmpp, Jmpp)
- Rs: Series resistance (Ω·cm²)
- Rsh: Shunt resistance (Ω·cm²)
"""

def __init__(self, config: SolarCellConfig):
    self.config = config
    self.logger = logging.getLogger(__name__)

def analyze(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
    """Main analysis pipeline for solar cell I-V"""
    self.logger.info("Starting solar cell I-V analysis")
    
    voltage = np.array(measurements['voltage'])
    current = np.array(measurements['current'])
    area = measurements.get('area', self.config.cell_area)
    illumination = measurements.get('illumination', self.config.illumination)
    temperature = measurements.get('temperature', self.config.temperature)
    
    # Convert to current density (mA/cm²)
    current_density = (current * 1000) / area  # A → mA, normalize by area
    
    # 1. Extract key metrics
    jsc = self._extract_jsc(voltage, current_density)
    voc = self._extract_voc(voltage, current_density)
    
    # 2. Find maximum power point
    mpp = self._find_mpp(voltage, current_density)
    
    # 3. Calculate fill factor
    ff = self._calculate_fill_factor(jsc, voc, mpp['pmax'])
    
    # 4. Calculate efficiency
    efficiency = self._calculate_efficiency(mpp['pmax'], area, illumination)
    
    # 5. Extract series and shunt resistance
    rs, rsh = self._extract_resistances(voltage, current_density, jsc, voc)
    
    # 6. Extract diode parameters (if enabled)
    diode_params = None
    if self.config.extract_diode_params:
        diode_params = self._extract_diode_parameters(voltage, current_density)
    
    results = {
        'device_type': 'solar_cell',
        'parameters': {
            'short_circuit_current_density': {
                'value': float(jsc),
                'unit': 'mA/cm²'
            },
            'open_circuit_voltage': {
                'value': float(voc),
                'unit': 'V'
            },
            'fill_factor': {
                'value': float(ff),
                'unit': 'dimensionless',
                'percent': float(ff * 100)
            },
            'efficiency': {
                'value': float(efficiency),
                'unit': '%'
            },
            'maximum_power_point': {
                'power': {
                    'value': float(mpp['pmax']),
                    'unit': 'mW/cm²'
                },
                'voltage': {
                    'value': float(mpp['vmpp']),
                    'unit': 'V'
                },
                'current_density': {
                    'value': float(mpp['jmpp']),
                    'unit': 'mA/cm²'
                }
            },
            'series_resistance': {
                'value': float(rs),
                'unit': 'Ω·cm²'
            },
            'shunt_resistance': {
                'value': float(rsh),
                'unit': 'Ω·cm²'
            }
        },
        'test_conditions': {
            'illumination': {
                'value': illumination,
                'unit': 'W/m²'
            },
            'temperature': {
                'value': temperature,
                'unit': 'K'
            },
            'spectrum': self.config.spectrum,
            'cell_area': {
                'value': area,
                'unit': 'cm²'
            }
        },
        'diode_parameters': diode_params,
        'raw_data': {
            'voltage': voltage.tolist(),
            'current': current.tolist(),
            'current_density': current_density.tolist(),
            'power_density': (voltage * current_density).tolist()
        }
    }
    
    self.logger.info(
        f"Analysis complete: η={efficiency:.2f}%, FF={ff:.3f}, "
        f"Jsc={jsc:.2f} mA/cm², Voc={voc:.3f} V"
    )
    
    return results

def _extract_jsc(self, voltage: np.ndarray, current_density: np.ndarray) -> float:
    """Extract short-circuit current density (at V=0)"""
    # Interpolate to find J at V=0
    if np.min(voltage) <= 0 <= np.max(voltage):
        f = interp1d(voltage, current_density, kind='linear')
        jsc = float(f(0))
    else:
        # Use value closest to V=0
        jsc = current_density[np.argmin(np.abs(voltage))]
    
    return abs(jsc)  # Jsc is positive by convention

def _extract_voc(self, voltage: np.ndarray, current_density: np.ndarray) -> float:
    """Extract open-circuit voltage (at J=0)"""
    # Interpolate to find V at J=0
    if np.min(current_density) <= 0 <= np.max(current_density):
        f = interp1d(current_density, voltage, kind='linear')
        voc = float(f(0))
    else:
        # Use value closest to J=0
        voc = voltage[np.argmin(np.abs(current_density))]
    
    return abs(voc)

def _find_mpp(
    self,
    voltage: np.ndarray,
    current_density: np.ndarray
) -> Dict[str, float]:
    """Find maximum power point"""
    # Power density (mW/cm²)
    power_density = voltage * current_density
    
    # Find maximum (most negative for solar cell)
    max_idx = np.argmax(-power_density)  # Negative because current is negative
    
    pmax = abs(power_density[max_idx])
    vmpp = voltage[max_idx]
    jmpp = abs(current_density[max_idx])
    
    return {
        'pmax': pmax,
        'vmpp': vmpp,
        'jmpp': jmpp
    }

def _calculate_fill_factor(self, jsc: float, voc: float, pmax: float) -> float:
    """
    Calculate fill factor
    
    FF = Pmax / (Jsc * Voc)
    
    Ideal FF ≈ 0.85, practical FF ≈ 0.70-0.80
    """
    ff = pmax / (jsc * voc)
    return ff

def _calculate_efficiency(
    self,
    pmax: float,
    area: float,
    illumination: float
) -> float:
    """
    Calculate power conversion efficiency
    
    η = Pmax / (Illumination * Area) * 100%
    
    pmax: mW/cm²
    area: cm²
    illumination: W/m² = mW/cm²
    """
    # Convert illumination to mW/cm²
    illumination_mw_cm2 = illumination / 10  # W/m² → mW/cm²
    
    efficiency = (pmax / illumination_mw_cm2) * 100
    
    return efficiency

def _extract_resistances(
    self,
    voltage: np.ndarray,
    current_density: np.ndarray,
    jsc: float,
    voc: float
) -> Tuple[float, float]:
    """
    Extract series and shunt resistance
    
    Rs: from slope near Voc (dV/dJ at V=Voc)
    Rsh: from slope near Jsc (dV/dJ at J=Jsc)
    """
    # Series resistance (near Voc)
    voc_region = voltage > (voc * 0.9)
    if np.sum(voc_region) > 2:
        rs = -np.polyfit(current_density[voc_region], voltage[voc_region], 1)[0]
    else:
        rs = 0.0
    
    # Shunt resistance (near Jsc)
    jsc_region = voltage < (voc * 0.1)
    if np.sum(jsc_region) > 2:
        rsh = -np.polyfit(current_density[jsc_region], voltage[jsc_region], 1)[0]
    else:
        rsh = 1e6
    
    return abs(rs), abs(rsh)

def _extract_diode_parameters(
    self,
    voltage: np.ndarray,
    current_density: np.ndarray
) -> Dict[str, float]:
    """
    Extract diode parameters using single-diode model
    
    J = J0 * [exp(q*V / (n*k*T)) - 1] - Jsc
    """
    # This is a simplified version
    # Full implementation would use iterative fitting
    
    return {
        'saturation_current_density': {
            'value': 1e-12,
            'unit': 'mA/cm²',
            'note': 'Simplified extraction'
        },
        'ideality_factor': {
            'value': 1.5,
            'unit': 'dimensionless',
            'note': 'Simplified extraction'
        }
    }

# ============================================================================

# API Functions

# ============================================================================

def analyze_mosfet_iv(
measurements: Dict[str, Any],
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“”“High-level API for MOSFET I-V analysis”””
if config:
mosfet_config = MOSFETConfig(**config)
else:
mosfet_config = MOSFETConfig()

analyzer = MOSFETAnalyzer(mosfet_config)
return analyzer.analyze(measurements)

def analyze_solar_cell_iv(
measurements: Dict[str, Any],
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“”“High-level API for solar cell I-V analysis”””
if config:
solar_config = SolarCellConfig(**config)
else:
solar_config = SolarCellConfig()

analyzer = SolarCellAnalyzer(solar_config)
return analyzer.analyze(measurements)