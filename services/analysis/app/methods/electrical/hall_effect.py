# services/analysis/app/methods/electrical/hall_effect.py

“””
Hall Effect Analysis

Implements:

- Hall coefficient measurement
- Carrier concentration calculation
- Hall mobility determination
- Semiconductor type detection (n/p)
- Temperature-dependent Hall
- Multi-field measurements

References:

- Hall, E. H. (1879). “On a New Action of the Magnet on Electric Currents”
- ASTM F76 - Standard Test Methods for Resistivity and Hall Coefficient
- Schroder, D. K. (2006). “Semiconductor Material and Device Characterization”
  “””

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(**name**)

# Physical constants

Q_E = 1.602176634e-19  # Elementary charge (C)

# ============================================================================

# Configuration

# ============================================================================

@dataclass
class HallConfig:
“”“Configuration for Hall effect measurements”””
# Measurement parameters
current: float = 1e-3  # A
magnetic_field: float = 0.5  # T (Tesla)
multi_field: bool = False  # Multiple field measurements
field_values: Optional[List[float]] = None  # T

# Sample geometry
sample_thickness: float = 500e-4  # cm (500 μm default)
sample_width: Optional[float] = None  # cm
sample_length: Optional[float] = None  # cm
geometry_factor: float = 1.0  # Correction for non-ideal geometry

# Sign detection
auto_detect_type: bool = True  # Automatically determine n/p type

# Temperature
temperature: Optional[float] = 300.0  # K

# Statistical analysis
num_measurements: int = 5
outlier_rejection: bool = True
outlier_threshold: float = 3.0  # Z-score threshold

# Physical constraints
max_mobility: float = 10000.0  # cm²/(V·s) - sanity check
min_mobility: float = 0.1  # cm²/(V·s)

# ============================================================================

# Hall Effect Analyzer

# ============================================================================

class HallAnalyzer:
“””
Hall Effect measurement analyzer

The Hall effect is the production of a voltage difference (Hall voltage)
across an electrical conductor when a magnetic field is applied
perpendicular to the current flow.
"""

def __init__(self, config: HallConfig):
    self.config = config
    self.logger = logging.getLogger(__name__)

def analyze(self, measurements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main analysis pipeline for Hall measurements
    
    Args:
        measurements: Dictionary containing:
            - hall_voltages: List of Hall voltage measurements (V)
            - currents: List of current values (A)
            - magnetic_fields: List of magnetic field values (T)
            - sheet_resistance: Optional pre-measured sheet resistance (Ω/sq)
            - temperature: Optional temperature (K)
            
    Returns:
        Dictionary with Hall analysis results
    """
    self.logger.info("Starting Hall effect analysis")
    
    # Extract data
    hall_voltages = np.array(measurements['hall_voltages'])
    currents = np.array(measurements['currents'])
    magnetic_fields = np.array(measurements.get('magnetic_fields', 
                                                [self.config.magnetic_field] * len(hall_voltages)))
    sheet_resistance = measurements.get('sheet_resistance', None)
    temperature = measurements.get('temperature', self.config.temperature)
    
    # 1. Calculate Hall coefficient
    if self.config.multi_field and len(np.unique(magnetic_fields)) > 1:
        # Multi-field: linear regression V_H vs B
        hall_coefficient, r_squared = self._calculate_hall_coefficient_multifield(
            hall_voltages, currents, magnetic_fields
        )
        measurement_type = "multi_field"
    else:
        # Single field
        hall_coefficient = self._calculate_hall_coefficient_single(
            hall_voltages, currents, magnetic_fields
        )
        r_squared = None
        measurement_type = "single_field"
    
    # 2. Determine semiconductor type (n/p)
    carrier_type = self._determine_carrier_type(hall_coefficient)
    
    # 3. Calculate carrier concentration
    carrier_concentration = self._calculate_carrier_concentration(
        hall_coefficient
    )
    
    # 4. Calculate Hall mobility (requires sheet resistance)
    hall_mobility = None
    conductivity = None
    if sheet_resistance is not None:
        hall_mobility = self._calculate_hall_mobility(
            hall_coefficient, sheet_resistance
        )
        conductivity = 1 / (sheet_resistance * self.config.sample_thickness)
    
    # 5. Statistical analysis
    stats_result = self._calculate_statistics(hall_voltages)
    
    # 6. Quality checks
    quality = self._assess_quality(
        hall_coefficient, 
        hall_mobility, 
        carrier_concentration,
        stats_result
    )
    
    # Compile results
    results = {
        'hall_coefficient': {
            'value': hall_coefficient,
            'unit': 'cm³/C',
            'r_squared': r_squared
        },
        'carrier_type': carrier_type,
        'carrier_concentration': {
            'value': carrier_concentration,
            'unit': 'cm⁻³',
            'in_scientific': f"{carrier_concentration:.2e}"
        },
        'hall_mobility': {
            'value': hall_mobility,
            'unit': 'cm²/(V·s)'
        } if hall_mobility else None,
        'conductivity': {
            'value': conductivity,
            'unit': 'S/cm'
        } if conductivity else None,
        'sheet_resistance': {
            'value': sheet_resistance,
            'unit': 'Ω/sq'
        } if sheet_resistance else None,
        'temperature': {
            'value': temperature,
            'unit': 'K'
        },
        'measurement': {
            'type': measurement_type,
            'num_points': len(hall_voltages),
            'field_range': [float(np.min(magnetic_fields)), float(np.max(magnetic_fields))]
        },
        'statistics': stats_result,
        'quality': quality,
        'raw_data': {
            'hall_voltages': hall_voltages.tolist(),
            'currents': currents.tolist(),
            'magnetic_fields': magnetic_fields.tolist()
        }
    }
    
    self.logger.info(
        f"Analysis complete: {carrier_type}, "
        f"n = {carrier_concentration:.2e} cm⁻³, "
        f"μ_H = {hall_mobility:.1f} cm²/(V·s)" if hall_mobility else ""
    )
    
    return results

def _calculate_hall_coefficient_single(
    self,
    hall_voltages: np.ndarray,
    currents: np.ndarray,
    magnetic_fields: np.ndarray
) -> float:
    """
    Calculate Hall coefficient from single-field measurements
    
    R_H = (V_H · t) / (I · B)
    
    where:
    - V_H = Hall voltage
    - t = sample thickness
    - I = current
    - B = magnetic field
    """
    # Average over all measurements
    R_H_values = (hall_voltages * self.config.sample_thickness) / (
        currents * magnetic_fields
    )
    
    # Apply geometry factor
    R_H_values *= self.config.geometry_factor
    
    # Outlier rejection
    if self.config.outlier_rejection:
        R_H_cleaned = self._reject_outliers_zscore(R_H_values)
    else:
        R_H_cleaned = R_H_values
    
    R_H = np.mean(R_H_cleaned)
    return float(R_H)

def _calculate_hall_coefficient_multifield(
    self,
    hall_voltages: np.ndarray,
    currents: np.ndarray,
    magnetic_fields: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate Hall coefficient from multi-field measurements using linear regression
    
    V_H = (R_H · I / t) · B
    
    Slope of V_H vs B gives: R_H · I / t
    
    Returns:
        (hall_coefficient, r_squared)
    """
    # For each unique field, calculate average Hall voltage
    unique_fields = np.unique(magnetic_fields)
    avg_voltages = []
    
    for B in unique_fields:
        mask = magnetic_fields == B
        avg_V = np.mean(hall_voltages[mask])
        avg_voltages.append(avg_V)
    
    avg_voltages = np.array(avg_voltages)
    
    # Linear regression: V_H vs B
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        unique_fields, avg_voltages
    )
    
    # Extract Hall coefficient
    # slope = R_H · I / t
    avg_current = np.mean(currents)
    R_H = (slope * self.config.sample_thickness) / avg_current
    
    # Apply geometry factor
    R_H *= self.config.geometry_factor
    
    r_squared = r_value ** 2
    
    self.logger.info(f"Multi-field regression: R² = {r_squared:.4f}")
    
    return float(R_H), float(r_squared)

def _determine_carrier_type(self, hall_coefficient: float) -> str:
    """
    Determine semiconductor type from Hall coefficient sign
    
    - Positive R_H → p-type (holes)
    - Negative R_H → n-type (electrons)
    """
    if hall_coefficient > 0:
        return "p-type"
    elif hall_coefficient < 0:
        return "n-type"
    else:
        self.logger.warning("Hall coefficient is zero - ambiguous type")
        return "unknown"

def _calculate_carrier_concentration(self, hall_coefficient: float) -> float:
    """
    Calculate carrier concentration from Hall coefficient
    
    For single carrier type:
    n = 1 / (q · R_H)
    
    where:
    - n = carrier concentration (cm⁻³)
    - q = elementary charge (1.602e-19 C)
    - R_H = Hall coefficient (cm³/C)
    
    Note: This assumes single carrier type and Hall scattering factor r_H = 1
    """
    # Take absolute value (sign only indicates type)
    R_H_abs = abs(hall_coefficient)
    
    n = 1 / (Q_E * R_H_abs)
    
    return float(n)

def _calculate_hall_mobility(
    self,
    hall_coefficient: float,
    sheet_resistance: float
) -> float:
    """
    Calculate Hall mobility
    
    μ_H = |R_H| / (ρ)
    
    where:
    - μ_H = Hall mobility (cm²/(V·s))
    - R_H = Hall coefficient (cm³/C)
    - ρ = resistivity (Ω·cm)
    """
    # Calculate resistivity from sheet resistance
    resistivity = sheet_resistance * self.config.sample_thickness  # Ω·cm
    
    # Calculate mobility
    R_H_abs = abs(hall_coefficient)
    mobility = R_H_abs / resistivity
    
    # Sanity check
    if mobility > self.config.max_mobility:
        self.logger.warning(
            f"Unusually high mobility: {mobility:.1f} cm²/(V·s) "
            f"(max expected: {self.config.max_mobility})"
        )
    
    if mobility < self.config.min_mobility:
        self.logger.warning(
            f"Unusually low mobility: {mobility:.1f} cm²/(V·s) "
            f"(min expected: {self.config.min_mobility})"
        )
    
    return float(mobility)

def _reject_outliers_zscore(
    self,
    data: np.ndarray
) -> np.ndarray:
    """Reject outliers using Z-score method"""
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    z_scores = np.abs((data - mean) / std)
    
    mask = z_scores < self.config.outlier_threshold
    cleaned = data[mask]
    
    if len(cleaned) < len(data):
        self.logger.info(
            f"Rejected {len(data) - len(cleaned)} outliers "
            f"({100 * (len(data) - len(cleaned)) / len(data):.1f}%)"
        )
    
    return cleaned

def _calculate_statistics(self, data: np.ndarray) -> Dict[str, float]:
    """Calculate statistical summary"""
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'cv_percent': float(100 * np.std(data, ddof=1) / np.mean(data)),
        'count': int(len(data))
    }

def _assess_quality(
    self,
    hall_coefficient: float,
    hall_mobility: Optional[float],
    carrier_concentration: float,
    statistics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Assess measurement quality
    
    Returns quality metrics and warnings
    """
    warnings = []
    quality_score = 100.0
    
    # Check coefficient of variation
    cv = statistics['cv_percent']
    if cv > 10:
        warnings.append(f"High variability in measurements (CV = {cv:.1f}%)")
        quality_score -= 20
    elif cv > 5:
        warnings.append(f"Moderate variability (CV = {cv:.1f}%)")
        quality_score -= 10
    
    # Check Hall coefficient magnitude
    if abs(hall_coefficient) < 1e-10:
        warnings.append("Very low Hall coefficient - check setup")
        quality_score -= 30
    
    # Check mobility (if available)
    if hall_mobility:
        if hall_mobility > self.config.max_mobility:
            warnings.append(f"Mobility exceeds expected maximum ({hall_mobility:.1f} cm²/(V·s))")
            quality_score -= 15
        if hall_mobility < self.config.min_mobility:
            warnings.append(f"Mobility below expected minimum ({hall_mobility:.1f} cm²/(V·s))")
            quality_score -= 15
    
    # Check carrier concentration plausibility
    if carrier_concentration < 1e10:
        warnings.append("Very low carrier concentration - check sample or measurement")
        quality_score -= 20
    elif carrier_concentration > 1e22:
        warnings.append("Very high carrier concentration - check calculation")
        quality_score -= 20
    
    quality_score = max(0, quality_score)
    
    if quality_score >= 90:
        quality_level = "excellent"
    elif quality_score >= 70:
        quality_level = "good"
    elif quality_score >= 50:
        quality_level = "acceptable"
    else:
        quality_level = "poor"
    
    return {
        'score': quality_score,
        'level': quality_level,
        'warnings': warnings
    }

# ============================================================================

# Helper Functions

# ============================================================================

def calculate_drift_velocity(
hall_voltage: float,
sample_width: float,
magnetic_field: float
) -> float:
“””
Calculate carrier drift velocity from Hall voltage

v_d = V_H / (w · B)

Args:
    hall_voltage: Hall voltage (V)
    sample_width: Sample width (cm)
    magnetic_field: Magnetic field (T)
    
Returns:
    Drift velocity (cm/s)
"""
v_d = hall_voltage / (sample_width * magnetic_field)
return v_d

def estimate_scattering_time(
mobility: float,
effective_mass: float
) -> float:
“””
Estimate mean scattering time from mobility

τ = (μ · m*) / q

Args:
    mobility: Carrier mobility (cm²/(V·s))
    effective_mass: Effective mass (in units of electron rest mass)
    
Returns:
    Scattering time (s)
"""
m_e = 9.10938356e-31  # Electron mass (kg)
m_star = effective_mass * m_e

# Convert mobility to SI: cm²/(V·s) → m²/(V·s)
mobility_si = mobility * 1e-4

tau = (mobility_si * m_star) / Q_E
return tau

# ============================================================================

# API Function

# ============================================================================

def analyze_hall_effect(
measurements: Dict[str, Any],
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
High-level API for Hall effect analysis

Args:
    measurements: Raw measurement data including:
        - hall_voltages: Hall voltage measurements
        - currents: Current values
        - magnetic_fields: Magnetic field values
        - sheet_resistance: Optional sheet resistance
    config: Optional configuration dictionary
    
Returns:
    Analysis results dictionary
"""
# Create config
if config:
    hall_config = HallConfig(**config)
else:
    hall_config = HallConfig()

# Run analysis
analyzer = HallAnalyzer(hall_config)
results = analyzer.analyze(measurements)

return results