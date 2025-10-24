# packages/common/semiconductorlab_common/units.py

“””
Unit Handling System with Pint Integration

Provides:

- Physical quantity validation
- Unit conversion
- Uncertainty propagation
- UCUM serialization for interoperability
- Dimensional analysis
  “””

import pint
from typing import Union, Dict, Any, Optional, Tuple
from decimal import Decimal
import numpy as np
from dataclasses import dataclass
import json

# ============================================================================

# Unit Registry Setup

# ============================================================================

# Create unit registry

ureg = pint.UnitRegistry()

# Define semiconductor-specific units

ureg.define(‘electron = e = elementary_charge’)
ureg.define(‘hole = elementary_charge’)

# Common semiconductor quantities (for convenience)

Q_ = ureg.Quantity

# ============================================================================

# Semiconductor Constants

# ============================================================================

class PhysicalConstants:
“”“Physical constants with units”””

# Fundamental constants
k = Q_(1.380649e-23, 'joule / kelvin')  # Boltzmann constant
q = Q_(1.602176634e-19, 'coulomb')  # Elementary charge
h = Q_(6.62607015e-34, 'joule * second')  # Planck constant
c = Q_(299792458, 'meter / second')  # Speed of light
epsilon_0 = Q_(8.8541878128e-12, 'farad / meter')  # Vacuum permittivity

@classmethod
def thermal_voltage(cls, temperature: Q_) -> Q_:
    """
    Calculate thermal voltage Vt = kT/q
    
    Args:
        temperature: Temperature (K)
        
    Returns:
        Thermal voltage (V)
    """
    return (cls.k * temperature / cls.q).to('volt')

@classmethod
def energy_to_wavelength(cls, energy: Q_) -> Q_:
    """
    Convert photon energy to wavelength
    E = hc/λ → λ = hc/E
    
    Args:
        energy: Photon energy (eV)
        
    Returns:
        Wavelength (nm)
    """
    return (cls.h * cls.c / energy).to('nanometer')

@classmethod
def wavelength_to_energy(cls, wavelength: Q_) -> Q_:
    """
    Convert wavelength to photon energy
    
    Args:
        wavelength: Wavelength (nm)
        
    Returns:
        Photon energy (eV)
    """
    return (cls.h * cls.c / wavelength).to('eV')

# ============================================================================

# Quantity Validation

# ============================================================================

@dataclass
class QuantitySpec:
“”“Specification for a physical quantity”””
name: str
dimensionality: str  # e.g., ‘[length]’, ‘[current]’, ‘[voltage]’
allowed_units: list  # e.g., [‘V’, ‘mV’, ‘kV’]
min_value: Optional[float] = None
max_value: Optional[float] = None

def validate(self, value: Union[Q_, float], unit: Optional[str] = None) -> Q_:
    """
    Validate a quantity against this spec
    
    Args:
        value: Value with or without units
        unit: Unit string (if value is unitless)
        
    Returns:
        Validated Quantity
        
    Raises:
        ValueError: If validation fails
    """
    # Convert to Quantity if needed
    if not isinstance(value, Q_):
        if unit is None:
            raise ValueError(f"Unit must be specified for {self.name}")
        value = Q_(value, unit)
    
    # Check dimensionality
    if value.dimensionality != ureg.parse_expression(self.dimensionality).dimensionality:
        raise ValueError(
            f"{self.name} must have dimensionality {self.dimensionality}, "
            f"got {value.dimensionality}"
        )
    
    # Check allowed units
    if self.allowed_units and str(value.units) not in self.allowed_units:
        # Try converting to one of the allowed units
        converted = False
        for allowed_unit in self.allowed_units:
            try:
                value = value.to(allowed_unit)
                converted = True
                break
            except pint.DimensionalityError:
                continue
        
        if not converted:
            raise ValueError(
                f"{self.name} must use one of {self.allowed_units}, "
                f"got {value.units}"
            )
    
    # Check range
    if self.min_value is not None and value.magnitude < self.min_value:
        raise ValueError(
            f"{self.name} must be >= {self.min_value}, got {value.magnitude}"
        )
    
    if self.max_value is not None and value.magnitude > self.max_value:
        raise ValueError(
            f"{self.name} must be <= {self.max_value}, got {value.magnitude}"
        )
    
    return value

# ============================================================================

# Common Quantity Specs

# ============================================================================

class CommonQuantities:
“”“Pre-defined quantity specifications”””

VOLTAGE = QuantitySpec(
    name="voltage",
    dimensionality="[length] ** 2 * [mass] / [current] / [time] ** 3",
    allowed_units=['V', 'mV', 'μV', 'kV']
)

CURRENT = QuantitySpec(
    name="current",
    dimensionality="[current]",
    allowed_units=['A', 'mA', 'μA', 'nA', 'pA']
)

RESISTANCE = QuantitySpec(
    name="resistance",
    dimensionality="[length] ** 2 * [mass] / [current] ** 2 / [time] ** 3",
    allowed_units=['Ω', 'ohm', 'kΩ', 'MΩ']
)

CAPACITANCE = QuantitySpec(
    name="capacitance",
    dimensionality="[current] ** 2 * [time] ** 4 / [length] ** 2 / [mass]",
    allowed_units=['F', 'pF', 'nF', 'μF']
)

TEMPERATURE = QuantitySpec(
    name="temperature",
    dimensionality="[temperature]",
    allowed_units=['K', 'degC'],
    min_value=0  # Absolute zero
)

LENGTH = QuantitySpec(
    name="length",
    dimensionality="[length]",
    allowed_units=['m', 'mm', 'μm', 'nm', 'Å']
)

ENERGY = QuantitySpec(
    name="energy",
    dimensionality="[length] ** 2 * [mass] / [time] ** 2",
    allowed_units=['J', 'eV', 'meV']
)

WAVELENGTH = QuantitySpec(
    name="wavelength",
    dimensionality="[length]",
    allowed_units=['nm', 'μm', 'Å']
)

FREQUENCY = QuantitySpec(
    name="frequency",
    dimensionality="1 / [time]",
    allowed_units=['Hz', 'kHz', 'MHz', 'GHz']
)

MOBILITY = QuantitySpec(
    name="mobility",
    dimensionality="[length] ** 2 / [time] / [length] / [mass] * [current] * [time] ** 2",
    allowed_units=['cm^2 / V / s', 'm^2 / V / s']
)

CARRIER_DENSITY = QuantitySpec(
    name="carrier_density",
    dimensionality="1 / [length] ** 3",
    allowed_units=['cm^-3', 'm^-3']
)

# ============================================================================

# Uncertainty Handling

# ============================================================================

class UncertainQuantity:
“””
Quantity with uncertainty

Supports:
- Gaussian error propagation
- Relative and absolute uncertainties
- Arithmetic operations
"""

def __init__(
    self,
    value: Union[Q_, float],
    uncertainty: Union[Q_, float],
    unit: Optional[str] = None,
    uncertainty_type: str = "absolute"  # or "relative"
):
    # Convert to Quantity
    if not isinstance(value, Q_):
        if unit is None:
            raise ValueError("Unit must be specified")
        value = Q_(value, unit)
    
    self.value = value
    
    # Handle uncertainty
    if uncertainty_type == "relative":
        # Relative uncertainty (e.g., 5% = 0.05)
        self.uncertainty = value * uncertainty
    else:
        # Absolute uncertainty
        if not isinstance(uncertainty, Q_):
            self.uncertainty = Q_(uncertainty, str(value.units))
        else:
            self.uncertainty = uncertainty.to(value.units)

@property
def relative_uncertainty(self) -> float:
    """Get relative uncertainty (dimensionless)"""
    return (self.uncertainty / self.value).magnitude

def __repr__(self) -> str:
    return f"{self.value.magnitude:.3e} ± {self.uncertainty.magnitude:.3e} {self.value.units}"

def __add__(self, other: 'UncertainQuantity') -> 'UncertainQuantity':
    """Add two uncertain quantities"""
    new_value = self.value + other.value
    new_uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
    return UncertainQuantity(new_value, new_uncertainty)

def __sub__(self, other: 'UncertainQuantity') -> 'UncertainQuantity':
    """Subtract two uncertain quantities"""
    new_value = self.value - other.value
    new_uncertainty = np.sqrt(self.uncertainty**2 + other.uncertainty**2)
    return UncertainQuantity(new_value, new_uncertainty)

def __mul__(self, other: Union['UncertainQuantity', float]) -> 'UncertainQuantity':
    """Multiply uncertain quantities"""
    if isinstance(other, UncertainQuantity):
        new_value = self.value * other.value
        # Relative uncertainties add in quadrature for multiplication
        rel_unc = np.sqrt(self.relative_uncertainty**2 + other.relative_uncertainty**2)
        new_uncertainty = new_value * rel_unc
        return UncertainQuantity(new_value, new_uncertainty)
    else:
        # Scalar multiplication
        return UncertainQuantity(self.value * other, self.uncertainty * abs(other))

def __truediv__(self, other: Union['UncertainQuantity', float]) -> 'UncertainQuantity':
    """Divide uncertain quantities"""
    if isinstance(other, UncertainQuantity):
        new_value = self.value / other.value
        rel_unc = np.sqrt(self.relative_uncertainty**2 + other.relative_uncertainty**2)
        new_uncertainty = new_value * rel_unc
        return UncertainQuantity(new_value, new_uncertainty)
    else:
        return UncertainQuantity(self.value / other, self.uncertainty / abs(other))

# ============================================================================

# UCUM Serialization

# ============================================================================

class UCUMSerializer:
“””
Unified Code for Units of Measure (UCUM) serialization

UCUM is the international standard for machine-readable units.
"""

# Mapping from Pint to UCUM
PINT_TO_UCUM = {
    'volt': 'V',
    'millivolt': 'mV',
    'ampere': 'A',
    'milliampere': 'mA',
    'microampere': 'uA',
    'nanoampere': 'nA',
    'picoampere': 'pA',
    'ohm': 'Ohm',
    'kilohm': 'kOhm',
    'megohm': 'MOhm',
    'farad': 'F',
    'picofarad': 'pF',
    'nanofarad': 'nF',
    'meter': 'm',
    'nanometer': 'nm',
    'micrometer': 'um',
    'kelvin': 'K',
    'degree_Celsius': 'Cel',
    'second': 's',
    'hertz': 'Hz',
}

@classmethod
def to_ucum(cls, quantity: Q_) -> Dict[str, Any]:
    """
    Serialize Quantity to UCUM format
    
    Returns:
        Dictionary with 'value', 'unit' (UCUM), and 'ucum_code'
    """
    unit_str = str(quantity.units)
    ucum_code = cls.PINT_TO_UCUM.get(unit_str, unit_str)
    
    return {
        'value': float(quantity.magnitude),
        'unit': ucum_code,
        'ucum_code': ucum_code
    }

@classmethod
def from_ucum(cls, value: float, ucum_code: str) -> Q_:
    """
    Deserialize from UCUM format
    
    Args:
        value: Numeric value
        ucum_code: UCUM unit code
        
    Returns:
        Pint Quantity
    """
    # Reverse mapping
    pint_unit = None
    for pint_name, ucum_name in cls.PINT_TO_UCUM.items():
        if ucum_name == ucum_code:
            pint_unit = pint_name
            break
    
    if pint_unit is None:
        # Try direct conversion
        pint_unit = ucum_code
    
    return Q_(value, pint_unit)

# ============================================================================

# Validation Decorators

# ============================================================================

def validate_quantity(spec: QuantitySpec):
“””
Decorator to validate function arguments

Example:
    @validate_quantity(CommonQuantities.VOLTAGE)
    def measure_iv(voltage: Q_) -> Q_:
        ...
"""
def decorator(func):
    def wrapper(*args, **kwargs):
        # Validate first argument (assume it's the quantity to validate)
        if args:
            validated = spec.validate(args[0])
            args = (validated,) + args[1:]
        return func(*args, **kwargs)
    return wrapper
return decorator

# ============================================================================

# Example Usage

# ============================================================================

def example_usage():
“”“Demonstrate unit handling system”””
print(”=” * 80)
print(“Unit Handling System - Example Usage”)
print(”=” * 80)

# 1. Basic Quantities
print("\n1. Basic Quantities:")
voltage = Q_(0.6, 'V')
current = Q_(1.5, 'mA')
print(f"   Voltage: {voltage}")
print(f"   Current: {current}")
print(f"   Resistance: {(voltage / current).to('ohm')}")

# 2. Unit Conversion
print("\n2. Unit Conversion:")
wavelength = Q_(550, 'nm')
print(f"   Wavelength: {wavelength}")
print(f"   In μm: {wavelength.to('micrometer')}")
print(f"   In Å: {wavelength.to('angstrom')}")

# 3. Physical Constants
print("\n3. Physical Constants:")
temp = Q_(300, 'K')
Vt = PhysicalConstants.thermal_voltage(temp)
print(f"   Temperature: {temp}")
print(f"   Thermal voltage (Vt): {Vt.to('mV')}")

energy = Q_(2.5, 'eV')
wl = PhysicalConstants.energy_to_wavelength(energy)
print(f"   Band gap: {energy}")
print(f"   Corresponding wavelength: {wl}")

# 4. Quantity Validation
print("\n4. Quantity Validation:")
try:
    valid_voltage = CommonQuantities.VOLTAGE.validate(0.5, 'V')
    print(f"   ✓ Valid voltage: {valid_voltage}")
except ValueError as e:
    print(f"   ✗ Invalid: {e}")

try:
    invalid = CommonQuantities.VOLTAGE.validate(10, 'A')  # Wrong dimensionality
    print(f"   This shouldn't print")
except ValueError as e:
    print(f"   ✓ Caught error: {e}")

# 5. Uncertain Quantities
print("\n5. Uncertain Quantities:")
v1 = UncertainQuantity(Q_(0.650, 'V'), Q_(0.005, 'V'))
v2 = UncertainQuantity(Q_(0.100, 'V'), Q_(0.002, 'V'))
print(f"   V1 = {v1}")
print(f"   V2 = {v2}")
print(f"   V1 - V2 = {v1 - v2}")
print(f"   Relative uncertainty: {v1.relative_uncertainty * 100:.2f}%")

# 6. Semiconductor Calculations
print("\n6. Semiconductor Calculations:")
mobility = Q_(1500, 'cm^2 / V / s')
carrier_density = Q_(1e16, 'cm^-3')
conductivity = PhysicalConstants.q * carrier_density * mobility
print(f"   Mobility: {mobility}")
print(f"   Carrier density: {carrier_density}")
print(f"   Conductivity: {conductivity.to('S / m')}")

# 7. UCUM Serialization
print("\n7. UCUM Serialization:")
current = Q_(1.5, 'mA')
ucum_dict = UCUMSerializer.to_ucum(current)
print(f"   Quantity: {current}")
print(f"   UCUM: {json.dumps(ucum_dict, indent=2)}")

# Deserialize
reconstructed = UCUMSerializer.from_ucum(ucum_dict['value'], ucum_dict['ucum_code'])
print(f"   Reconstructed: {reconstructed}")

# 8. Array Operations
print("\n8. Array Operations with Units:")
voltages = Q_(np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), 'V')
currents = Q_(np.array([0.0, 1e-6, 10e-6, 100e-6, 1e-3, 5e-3]), 'A')
resistances = voltages / currents
print(f"   Voltages: {voltages}")
print(f"   Currents: {currents.to('mA')}")
print(f"   Resistances: {resistances.to('ohm')}")

print("\n" + "=" * 80)
print("Unit handling demonstration complete!")
print("=" * 80)

if **name** == “**main**”:
example_usage()