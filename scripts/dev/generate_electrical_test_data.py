# scripts/dev/generate_electrical_test_data.py

“””
Test Data Generators for Electrical Characterization Methods

Generates realistic synthetic data for:

- Four-Point Probe (Van der Pauw)
- Hall Effect measurements

Includes physics-based models with realistic noise and artifacts.
“””

import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import uuid
from datetime import datetime, timedelta

# ============================================================================

# Configuration

# ============================================================================

@dataclass
class MaterialProperties:
“”“Physical properties of test materials”””
name: str
resistivity: float  # Ω·cm
carrier_type: str  # “n-type” or “p-type”
carrier_concentration: float  # cm⁻³
mobility: float  # cm²/(V·s)
temperature_coefficient: float  # 1/K
thickness: float  # cm

# Known reference materials

REFERENCE_MATERIALS = {
‘silicon_n’: MaterialProperties(
name=“Silicon (n-type, phosphorus doped)”,
resistivity=0.01,  # 10 mΩ·cm
carrier_type=“n-type”,
carrier_concentration=5e18,
mobility=1200.0,
temperature_coefficient=0.0045,
thickness=500e-4  # 500 μm
),
‘silicon_p’: MaterialProperties(
name=“Silicon (p-type, boron doped)”,
resistivity=0.05,
carrier_type=“p-type”,
carrier_concentration=1e18,
mobility=400.0,
temperature_coefficient=0.0040,
thickness=500e-4
),
‘gaas_n’: MaterialProperties(
name=“GaAs (n-type)”,
resistivity=0.001,
carrier_type=“n-type”,
carrier_concentration=5e17,
mobility=8500.0,
temperature_coefficient=0.0035,
thickness=350e-4
),
‘gaas_p’: MaterialProperties(
name=“GaAs (p-type)”,
resistivity=0.02,
carrier_type=“p-type”,
carrier_concentration=1e17,
mobility=400.0,
temperature_coefficient=0.0030,
thickness=350e-4
),
‘graphene’: MaterialProperties(
name=“Graphene (monolayer)”,
resistivity=1e-4,
carrier_type=“n-type”,
carrier_concentration=1e13,  # per cm² (2D)
mobility=15000.0,
temperature_coefficient=0.001,
thickness=3.35e-8  # ~3.35 Å
),
‘copper_thin_film’: MaterialProperties(
name=“Copper thin film”,
resistivity=1.7e-6,  # Bulk Cu: 1.7 μΩ·cm
carrier_type=“n-type”,
carrier_concentration=8.5e22,
mobility=43.0,
temperature_coefficient=0.0039,
thickness=100e-7  # 100 nm
)
}

# ============================================================================

# Four-Point Probe Generator

# ============================================================================

class FourPointProbeGenerator:
“”“Generate synthetic 4PP measurement data”””

def __init__(
    self,
    material: MaterialProperties,
    noise_level: float = 0.02,  # 2% noise
    add_contact_issues: bool = False
):
    self.material = material
    self.noise_level = noise_level
    self.add_contact_issues = add_contact_issues

def generate_single_point(
    self,
    current: float = 1e-3,
    temperature: float = 300.0
) -> Dict[str, Any]:
    """Generate single measurement point"""
    # Calculate ideal sheet resistance
    sheet_resistance = self.material.resistivity / self.material.thickness
    
    # Temperature compensation
    T_ref = 300.0
    alpha = self.material.temperature_coefficient
    sheet_resistance *= (1 + alpha * (temperature - T_ref))
    
    # Van der Pauw geometry factor (for arbitrary shape)
    geometry_factor = np.random.uniform(0.95, 1.05)  # Slight variation
    sheet_resistance *= geometry_factor
    
    # Calculate voltage
    voltage = sheet_resistance * current
    
    # Add noise
    voltage += voltage * np.random.normal(0, self.noise_level)
    
    # Add contact resistance effects (if enabled)
    if self.add_contact_issues:
        contact_resistance = np.random.uniform(10, 50)  # Ohms
        voltage += current * contact_resistance * 0.1  # 10% contribution
    
    return {
        'voltage': voltage,
        'current': current,
        'temperature': temperature,
        'timestamp': datetime.now().isoformat()
    }

def generate_van_der_pauw_set(
    self,
    num_configurations: int = 4,
    current: float = 1e-3,
    temperature: float = 300.0
) -> Dict[str, Any]:
    """
    Generate complete Van der Pauw measurement set
    
    4 configurations:
    1. R_AB,CD: Current through AB, voltage across CD
    2. R_BC,DA: Current through BC, voltage across DA
    3. R_CD,AB: Current through CD, voltage across AB
    4. R_DA,BC: Current through DA, voltage across BC
    """
    measurements = []
    configurations = ['R_AB,CD', 'R_BC,DA', 'R_CD,AB', 'R_DA,BC']
    
    for i in range(num_configurations):
        config_name = configurations[i % 4]
        
        # Small variation between configurations due to sample asymmetry
        asymmetry_factor = np.random.uniform(0.98, 1.02)
        
        point = self.generate_single_point(current, temperature)
        point['voltage'] *= asymmetry_factor
        point['configuration'] = config_name
        
        measurements.append(point)
    
    # Extract arrays
    voltages = [m['voltage'] for m in measurements]
    currents = [m['current'] for m in measurements]
    configs = [m['configuration'] for m in measurements]
    
    # Calculate expected sheet resistance
    expected_Rs = self.material.resistivity / self.material.thickness
    
    return {
        'voltages': voltages,
        'currents': currents,
        'configurations': configs,
        'temperature': temperature,
        'material': self.material.name,
        'expected_sheet_resistance': expected_Rs,
        'sample_thickness': self.material.thickness
    }

def generate_wafer_map(
    self,
    wafer_diameter: float = 200.0,  # mm
    die_size: float = 10.0,  # mm
    current: float = 1e-3,
    temperature: float = 300.0,
    add_uniformity_gradient: bool = True
) -> Dict[str, Any]:
    """
    Generate wafer map with multiple measurement points
    
    Simulates center-to-edge gradients and radial variations
    """
    # Generate grid of measurement positions
    positions = []
    measurements = []
    
    radius = wafer_diameter / 2
    num_points = int((wafer_diameter / die_size) ** 2 * 0.7)  # ~70% coverage
    
    for _ in range(num_points):
        # Random position within wafer
        r = radius * np.sqrt(np.random.random())
        theta = 2 * np.pi * np.random.random()
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Generate measurement
        point = self.generate_single_point(current, temperature)
        
        # Add spatial gradient (center-to-edge variation)
        if add_uniformity_gradient:
            # Typical 5-10% variation from center to edge
            radial_factor = 1 + 0.08 * (r / radius)
            point['voltage'] *= radial_factor
        
        positions.append((x, y))
        measurements.append(point)
    
    # Extract data
    voltages = [m['voltage'] for m in measurements]
    currents = [m['current'] for m in measurements]
    
    return {
        'voltages': voltages,
        'currents': currents,
        'positions': positions,
        'configurations': ['R_AB,CD'] * len(measurements),  # Same config for all
        'temperature': temperature,
        'material': self.material.name,
        'wafer_diameter': wafer_diameter,
        'die_size': die_size,
        'sample_thickness': self.material.thickness,
        'is_wafer_map': True
    }

# ============================================================================

# Hall Effect Generator

# ============================================================================

class HallEffectGenerator:
“”“Generate synthetic Hall effect measurement data”””

def __init__(
    self,
    material: MaterialProperties,
    noise_level: float = 0.03  # 3% noise
):
    self.material = material
    self.noise_level = noise_level

def generate_single_field(
    self,
    current: float = 1e-3,
    magnetic_field: float = 0.5,  # Tesla
    temperature: float = 300.0
) -> Dict[str, Any]:
    """Generate Hall measurement at single magnetic field"""
    # Calculate Hall coefficient
    Q_E = 1.602176634e-19  # Elementary charge
    
    if self.material.carrier_type == "n-type":
        R_H = -1 / (Q_E * self.material.carrier_concentration)
    else:  # p-type
        R_H = 1 / (Q_E * self.material.carrier_concentration)
    
    # Calculate Hall voltage
    # V_H = (R_H · I · B) / t
    hall_voltage = (R_H * current * magnetic_field) / self.material.thickness
    
    # Add noise
    hall_voltage += hall_voltage * np.random.normal(0, self.noise_level)
    
    # Add offset (misalignment, thermal EMF)
    offset = np.random.uniform(-1e-6, 1e-6)  # μV range
    hall_voltage += offset
    
    return {
        'hall_voltage': hall_voltage,
        'current': current,
        'magnetic_field': magnetic_field,
        'temperature': temperature,
        'timestamp': datetime.now().isoformat()
    }

def generate_multi_field(
    self,
    current: float = 1e-3,
    field_range: Tuple[float, float] = (-1.0, 1.0),  # Tesla
    num_fields: int = 11,
    temperature: float = 300.0
) -> Dict[str, Any]:
    """
    Generate multi-field Hall measurements for linear regression
    
    This is the preferred method as it eliminates offsets
    """
    field_values = np.linspace(field_range[0], field_range[1], num_fields)
    measurements = []
    
    for B in field_values:
        point = self.generate_single_field(current, B, temperature)
        measurements.append(point)
    
    # Extract arrays
    hall_voltages = [m['hall_voltage'] for m in measurements]
    currents = [m['current'] for m in measurements]
    magnetic_fields = [m['magnetic_field'] for m in measurements]
    
    # Calculate sheet resistance (for mobility calculation)
    sheet_resistance = self.material.resistivity / self.material.thickness
    
    # Expected results
    Q_E = 1.602176634e-19
    sign = -1 if self.material.carrier_type == "n-type" else 1
    expected_R_H = sign / (Q_E * self.material.carrier_concentration)
    expected_n = self.material.carrier_concentration
    expected_mu = self.material.mobility
    
    return {
        'hall_voltages': hall_voltages,
        'currents': currents,
        'magnetic_fields': magnetic_fields,
        'sheet_resistance': sheet_resistance,
        'temperature': temperature,
        'material': self.material.name,
        'sample_thickness': self.material.thickness,
        'expected_results': {
            'hall_coefficient': expected_R_H,
            'carrier_concentration': expected_n,
            'carrier_type': self.material.carrier_type,
            'hall_mobility': expected_mu
        }
    }

def generate_single_field_repeated(
    self,
    num_measurements: int = 10,
    current: float = 1e-3,
    magnetic_field: float = 0.5,
    temperature: float = 300.0
) -> Dict[str, Any]:
    """Generate repeated measurements at single field"""
    measurements = []
    
    for _ in range(num_measurements):
        point = self.generate_single_field(current, magnetic_field, temperature)
        measurements.append(point)
    
    hall_voltages = [m['hall_voltage'] for m in measurements]
    currents = [m['current'] for m in measurements]
    magnetic_fields = [m['magnetic_field'] for m in measurements]
    
    # Calculate sheet resistance
    sheet_resistance = self.material.resistivity / self.material.thickness
    
    return {
        'hall_voltages': hall_voltages,
        'currents': currents,
        'magnetic_fields': magnetic_fields,
        'sheet_resistance': sheet_resistance,
        'temperature': temperature,
        'material': self.material.name,
        'sample_thickness': self.material.thickness
    }

# ============================================================================

# Data Export Functions

# ============================================================================

def save_test_dataset(
data: Dict[str, Any],
filename: str,
output_dir: Path = Path(“data/test_data/electrical”)
):
“”“Save test dataset to JSON file”””
output_dir.mkdir(parents=True, exist_ok=True)
filepath = output_dir / filename

# Add metadata
data['metadata'] = {
    'generated_at': datetime.now().isoformat(),
    'generator_version': '1.0.0',
    'dataset_id': str(uuid.uuid4())
}

with open(filepath, 'w') as f:
    json.dump(data, f, indent=2)

print(f"✓ Saved: {filepath}")

def generate_all_test_data():
“”“Generate complete test dataset for Session 4”””
print(“Generating test data for Session 4: Electrical I”)
print(”=” * 60)

# Four-Point Probe datasets
print("\n1. Four-Point Probe Data")
print("-" * 60)

# Silicon n-type
gen_si_n = FourPointProbeGenerator(REFERENCE_MATERIALS['silicon_n'], noise_level=0.01)
data_4pp_si = gen_si_n.generate_van_der_pauw_set()
save_test_dataset(data_4pp_si, "four_point_probe/silicon_n_type.json")

# Silicon p-type wafer map
data_4pp_wafer = gen_si_n.generate_wafer_map()
save_test_dataset(data_4pp_wafer, "four_point_probe/silicon_wafer_map.json")

# GaAs
gen_gaas = FourPointProbeGenerator(REFERENCE_MATERIALS['gaas_n'], noise_level=0.015)
data_4pp_gaas = gen_gaas.generate_van_der_pauw_set()
save_test_dataset(data_4pp_gaas, "four_point_probe/gaas_n_type.json")

# Copper thin film
gen_cu = FourPointProbeGenerator(REFERENCE_MATERIALS['copper_thin_film'], noise_level=0.02)
data_4pp_cu = gen_cu.generate_van_der_pauw_set()
save_test_dataset(data_4pp_cu, "four_point_probe/copper_thin_film.json")

# Hall Effect datasets
print("\n2. Hall Effect Data")
print("-" * 60)

# Silicon n-type (multi-field)
hall_gen_si_n = HallEffectGenerator(REFERENCE_MATERIALS['silicon_n'], noise_level=0.02)
data_hall_si_n = hall_gen_si_n.generate_multi_field()
save_test_dataset(data_hall_si_n, "hall_effect/silicon_n_type_multifield.json")

# Silicon p-type (single field repeated)
hall_gen_si_p = HallEffectGenerator(REFERENCE_MATERIALS['silicon_p'], noise_level=0.02)
data_hall_si_p = hall_gen_si_p.generate_single_field_repeated()
save_test_dataset(data_hall_si_p, "hall_effect/silicon_p_type_single.json")

# GaAs p-type
hall_gen_gaas_p = HallEffectGenerator(REFERENCE_MATERIALS['gaas_p'], noise_level=0.025)
data_hall_gaas = hall_gen_gaas_p.generate_multi_field()
save_test_dataset(data_hall_gaas, "hall_effect/gaas_p_type_multifield.json")

# Graphene (2D material)
hall_gen_graphene = HallEffectGenerator(REFERENCE_MATERIALS['graphene'], noise_level=0.03)
data_hall_graphene = hall_gen_graphene.generate_multi_field(field_range=(-0.5, 0.5))
save_test_dataset(data_hall_graphene, "hall_effect/graphene.json")

print("\n" + "=" * 60)
print("✓ Test data generation complete!")
print(f"  - 4 Four-Point Probe datasets")
print(f"  - 4 Hall Effect datasets")
print(f"  - Location: data/test_data/electrical/")

if **name** == “**main**”:
generate_all_test_data()