"""
Advanced Film Stress Modeling

Comprehensive stress calculation including:
- Intrinsic stress (process-dependent)
- Thermal stress (CTE mismatch)
- Gradient stress (through-thickness variation)
- Multiple measurement methods (Stoney, XRD, nanoindentation)
- VM feature engineering
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class StressType(str, Enum):
    """Film stress classification"""
    TENSILE = "TENSILE"  # Positive stress (film in tension)
    COMPRESSIVE = "COMPRESSIVE"  # Negative stress (film in compression)
    MIXED = "MIXED"  # Both tensile and compressive regions
    NEUTRAL = "NEUTRAL"  # Near-zero stress


class StressMeasurementMethod(str, Enum):
    """Film stress measurement techniques"""
    WAFER_CURVATURE = "WAFER_CURVATURE"  # Stoney's equation from curvature
    XRD = "XRD"  # X-ray diffraction peak shifts
    RAMAN = "RAMAN"  # Raman spectroscopy peak shifts
    NANOINDENTATION = "NANOINDENTATION"  # Indentation-based estimation
    BULGE_TEST = "BULGE_TEST"  # Membrane bulge test
    CANTILEVER = "CANTILEVER"  # Microcantilever deflection


@dataclass
class MaterialProperties:
    """Material properties for stress calculations"""
    # Film properties
    film_name: str = "SiO2"
    film_youngs_modulus_gpa: float = 70.0  # GPa
    film_poisson_ratio: float = 0.17
    film_cte_ppm_k: float = 0.5  # Coefficient of thermal expansion (ppm/K)
    film_density_g_cm3: float = 2.2  # g/cm³

    # Substrate properties
    substrate_name: str = "Silicon"
    substrate_youngs_modulus_gpa: float = 130.0  # GPa
    substrate_poisson_ratio: float = 0.28
    substrate_cte_ppm_k: float = 2.6  # ppm/K
    substrate_thickness_um: float = 725.0  # μm (typical Si wafer)

    # Reference state
    deposition_temp_c: float = 400.0
    reference_temp_c: float = 25.0  # Room temperature


@dataclass
class ProcessConditions:
    """Process conditions affecting intrinsic stress"""
    # Temperature
    temperature_c: float = 400.0

    # Pressure
    pressure_torr: float = 1.0

    # Deposition rate
    deposition_rate_nm_min: float = 50.0

    # Plasma parameters (for PECVD)
    rf_power_w: float = 0.0
    bias_voltage_v: float = 0.0
    ion_energy_ev: float = 0.0

    # Gas composition
    precursor_fraction: float = 0.1  # Fraction of total flow

    # Film thickness
    film_thickness_nm: float = 100.0


class IntrinsicStressCalculator:
    """
    Calculate intrinsic stress from process parameters

    Intrinsic stress arises from:
    - Atomic peening (ion bombardment)
    - Grain boundary effects
    - Incorporation of impurities
    - Film microstructure (columnar, amorphous, nanocrystalline)
    """

    def __init__(self, material: Optional[MaterialProperties] = None):
        self.material = material or MaterialProperties()

    def calculate_intrinsic_stress(
        self,
        process: ProcessConditions,
    ) -> Tuple[float, StressType]:
        """
        Calculate intrinsic stress from process conditions

        Args:
            process: Process conditions

        Returns:
            Tuple of (stress_mpa, stress_type)
        """
        # Base intrinsic stress (material and process dependent)
        base_stress_mpa = 0.0

        # ====================================================================
        # Factor 1: Deposition rate effect
        # ====================================================================
        # Higher rate → more compressive (less time for relaxation)
        # Lower rate → more tensile (more atomic mobility)
        rate_factor = -50.0 * (process.deposition_rate_nm_min / 100.0)

        # ====================================================================
        # Factor 2: Temperature effect
        # ====================================================================
        # Higher T → more tensile (thermal relaxation)
        # Lower T → more compressive (frozen-in stress)
        temp_factor = 0.2 * (process.temperature_c - 400.0)

        # ====================================================================
        # Factor 3: Pressure effect (for CVD)
        # ====================================================================
        # Lower pressure → more tensile (less densification)
        # Higher pressure → more compressive
        pressure_factor = -20.0 * math.log10(process.pressure_torr + 0.1)

        # ====================================================================
        # Factor 4: Ion bombardment (for PECVD)
        # ====================================================================
        # Ion peening creates compressive stress
        if process.rf_power_w > 0:
            # Estimate ion energy from bias voltage
            ion_energy_ev = abs(process.bias_voltage_v) if process.bias_voltage_v else 50.0

            # Compressive stress from ion bombardment
            ion_factor = -100.0 * math.sqrt(ion_energy_ev / 100.0)
        else:
            ion_factor = 0.0

        # ====================================================================
        # Combine factors
        # ====================================================================
        intrinsic_stress = base_stress_mpa + rate_factor + temp_factor + pressure_factor + ion_factor

        # Classify stress type
        if intrinsic_stress > 50.0:
            stress_type = StressType.TENSILE
        elif intrinsic_stress < -50.0:
            stress_type = StressType.COMPRESSIVE
        else:
            stress_type = StressType.NEUTRAL

        return intrinsic_stress, stress_type


class ThermalStressCalculator:
    """
    Calculate thermal stress from CTE mismatch

    σ_thermal = [E / (1-ν)] * (α_film - α_substrate) * ΔT

    Where:
    - E = film Young's modulus
    - ν = film Poisson's ratio
    - α = coefficient of thermal expansion
    - ΔT = temperature change from deposition to measurement
    """

    def __init__(self, material: Optional[MaterialProperties] = None):
        self.material = material or MaterialProperties()

    def calculate_thermal_stress(
        self,
        deposition_temp_c: float,
        measurement_temp_c: float = 25.0,
    ) -> float:
        """
        Calculate thermal stress from CTE mismatch

        Args:
            deposition_temp_c: Deposition temperature (°C)
            measurement_temp_c: Measurement temperature (°C)

        Returns:
            Thermal stress (MPa)
        """
        # Temperature change
        delta_T = deposition_temp_c - measurement_temp_c

        # CTE mismatch
        alpha_film = self.material.film_cte_ppm_k * 1e-6  # Convert ppm/K to 1/K
        alpha_sub = self.material.substrate_cte_ppm_k * 1e-6

        delta_alpha = alpha_film - alpha_sub

        # Biaxial modulus: E / (1 - ν)
        E_gpa = self.material.film_youngs_modulus_gpa
        nu = self.material.film_poisson_ratio
        biaxial_modulus_gpa = E_gpa / (1.0 - nu)

        # Thermal stress (GPa)
        stress_gpa = biaxial_modulus_gpa * delta_alpha * delta_T

        # Convert to MPa
        stress_mpa = stress_gpa * 1000.0

        return stress_mpa


class StressModel:
    """
    Comprehensive stress model combining all contributions

    Total stress = Intrinsic + Thermal + Gradient
    """

    def __init__(self, material: Optional[MaterialProperties] = None):
        self.material = material or MaterialProperties()

        self.intrinsic_calc = IntrinsicStressCalculator(material)
        self.thermal_calc = ThermalStressCalculator(material)

    def calculate_total_stress(
        self,
        process: ProcessConditions,
        measurement_temp_c: float = 25.0,
    ) -> Dict[str, any]:
        """
        Calculate total film stress

        Args:
            process: Process conditions
            measurement_temp_c: Measurement temperature

        Returns:
            Dictionary with stress components and statistics
        """
        # Intrinsic stress
        intrinsic_stress, stress_type = self.intrinsic_calc.calculate_intrinsic_stress(process)

        # Thermal stress
        thermal_stress = self.thermal_calc.calculate_thermal_stress(
            deposition_temp_c=process.temperature_c,
            measurement_temp_c=measurement_temp_c,
        )

        # Gradient stress (through-thickness variation)
        # Stress typically varies through film thickness
        gradient_factor = 0.1  # MPa/nm
        gradient_stress = gradient_factor * process.film_thickness_nm

        # Total mean stress
        total_stress_mean = intrinsic_stress + thermal_stress

        # Standard deviation (from gradients and non-uniformity)
        stress_std = abs(gradient_stress)

        # Min/max estimates
        stress_min = total_stress_mean - 2 * stress_std
        stress_max = total_stress_mean + 2 * stress_std

        # Classify final stress type
        if total_stress_mean > 0:
            final_type = StressType.TENSILE
        elif total_stress_mean < 0:
            final_type = StressType.COMPRESSIVE
        else:
            final_type = StressType.NEUTRAL

        return {
            "stress_mean_mpa": total_stress_mean,
            "stress_std_mpa": stress_std,
            "stress_min_mpa": stress_min,
            "stress_max_mpa": stress_max,
            "stress_type": final_type,
            "intrinsic_stress_mpa": intrinsic_stress,
            "thermal_stress_mpa": thermal_stress,
            "gradient_mpa_per_nm": gradient_factor,
        }

    def extract_vm_features(
        self,
        process: ProcessConditions,
    ) -> Dict[str, float]:
        """
        Extract features for Virtual Metrology stress prediction

        Args:
            process: Process conditions

        Returns:
            Feature dictionary for ML models
        """
        stress_result = self.calculate_total_stress(process)

        features = {
            # Process parameters
            "temperature_c": process.temperature_c,
            "pressure_torr": process.pressure_torr,
            "deposition_rate_nm_min": process.deposition_rate_nm_min,
            "rf_power_w": process.rf_power_w,
            "bias_voltage_v": process.bias_voltage_v,
            "precursor_fraction": process.precursor_fraction,
            "film_thickness_nm": process.film_thickness_nm,

            # Predicted stress
            "predicted_stress_mean_mpa": stress_result["stress_mean_mpa"],
            "predicted_stress_std_mpa": stress_result["stress_std_mpa"],
            "intrinsic_stress_mpa": stress_result["intrinsic_stress_mpa"],
            "thermal_stress_mpa": stress_result["thermal_stress_mpa"],

            # Material properties
            "film_youngs_modulus_gpa": self.material.film_youngs_modulus_gpa,
            "cte_mismatch_ppm_k": (
                self.material.film_cte_ppm_k - self.material.substrate_cte_ppm_k
            ),
        }

        return features


# =============================================================================
# Stress Measurement Method Converters
# =============================================================================

def wafer_curvature_to_stress(
    curvature_1_m: float,
    film_thickness_nm: float,
    substrate_thickness_um: float = 725.0,
    substrate_youngs_modulus_gpa: float = 130.0,
    substrate_poisson_ratio: float = 0.28,
) -> float:
    """
    Calculate film stress from wafer curvature using Stoney's equation

    σ = (E_s / (1-ν_s)) * (t_s² / (6*t_f)) * κ

    Where:
    - σ = film stress
    - E_s = substrate Young's modulus
    - ν_s = substrate Poisson's ratio
    - t_s = substrate thickness
    - t_f = film thickness
    - κ = curvature (1/radius)

    Args:
        curvature_1_m: Wafer curvature (1/m)
        film_thickness_nm: Film thickness (nm)
        substrate_thickness_um: Substrate thickness (μm)
        substrate_youngs_modulus_gpa: Substrate E (GPa)
        substrate_poisson_ratio: Substrate ν

    Returns:
        Film stress (MPa)
    """
    # Convert units
    t_s_m = substrate_thickness_um * 1e-6  # μm to m
    t_f_m = film_thickness_nm * 1e-9  # nm to m

    # Biaxial modulus of substrate
    E_biaxial_pa = (substrate_youngs_modulus_gpa * 1e9) / (1.0 - substrate_poisson_ratio)

    # Stoney's equation
    stress_pa = E_biaxial_pa * (t_s_m**2 / (6.0 * t_f_m)) * curvature_1_m

    # Convert Pa to MPa
    stress_mpa = stress_pa / 1e6

    return stress_mpa


def xrd_to_stress(
    d_measured_angstrom: float,
    d_unstressed_angstrom: float,
    film_youngs_modulus_gpa: float = 70.0,
    film_poisson_ratio: float = 0.17,
    miller_indices: Tuple[int, int, int] = (111,),
) -> float:
    """
    Calculate film stress from XRD peak shift

    σ = (E / (1-ν)) * (Δd / d₀)

    Where:
    - Δd = d_measured - d_unstressed
    - d₀ = unstressed lattice spacing

    Args:
        d_measured_angstrom: Measured d-spacing (Å)
        d_unstressed_angstrom: Unstressed d-spacing (Å)
        film_youngs_modulus_gpa: Film E (GPa)
        film_poisson_ratio: Film ν
        miller_indices: Miller indices of diffraction peak

    Returns:
        Film stress (MPa)
    """
    # Strain
    strain = (d_measured_angstrom - d_unstressed_angstrom) / d_unstressed_angstrom

    # Biaxial modulus
    E_biaxial_gpa = film_youngs_modulus_gpa / (1.0 - film_poisson_ratio)

    # Stress (GPa)
    stress_gpa = E_biaxial_gpa * strain

    # Convert to MPa
    stress_mpa = stress_gpa * 1000.0

    return stress_mpa


def nanoindentation_to_stress(
    hardness_gpa: float,
    youngs_modulus_gpa: float,
) -> float:
    """
    Estimate residual stress from nanoindentation data

    This is an empirical correlation, not a direct measurement.
    Residual stress affects the load-displacement curve.

    Args:
        hardness_gpa: Film hardness from nanoindentation (GPa)
        youngs_modulus_gpa: Film modulus from nanoindentation (GPa)

    Returns:
        Estimated stress (MPa) - approximate
    """
    # Empirical correlation: σ_residual ~ 0.1 * H
    # This is a rough estimate only
    stress_estimate_gpa = 0.1 * hardness_gpa

    stress_estimate_mpa = stress_estimate_gpa * 1000.0

    return stress_estimate_mpa


# =============================================================================
# Material Property Database
# =============================================================================

MATERIAL_DATABASE = {
    "SiO2": MaterialProperties(
        film_name="SiO2",
        film_youngs_modulus_gpa=70.0,
        film_poisson_ratio=0.17,
        film_cte_ppm_k=0.5,
        film_density_g_cm3=2.2,
    ),
    "Si3N4": MaterialProperties(
        film_name="Si3N4",
        film_youngs_modulus_gpa=250.0,
        film_poisson_ratio=0.27,
        film_cte_ppm_k=2.8,
        film_density_g_cm3=3.1,
    ),
    "TiN": MaterialProperties(
        film_name="TiN",
        film_youngs_modulus_gpa=600.0,
        film_poisson_ratio=0.25,
        film_cte_ppm_k=9.4,
        film_density_g_cm3=5.4,
    ),
    "W": MaterialProperties(
        film_name="W",
        film_youngs_modulus_gpa=400.0,
        film_poisson_ratio=0.28,
        film_cte_ppm_k=4.5,
        film_density_g_cm3=19.3,
    ),
    "Al": MaterialProperties(
        film_name="Al",
        film_youngs_modulus_gpa=70.0,
        film_poisson_ratio=0.35,
        film_cte_ppm_k=23.1,
        film_density_g_cm3=2.7,
    ),
    "Cu": MaterialProperties(
        film_name="Cu",
        film_youngs_modulus_gpa=130.0,
        film_poisson_ratio=0.34,
        film_cte_ppm_k=16.5,
        film_density_g_cm3=8.96,
    ),
    "a-Si": MaterialProperties(
        film_name="a-Si",
        film_youngs_modulus_gpa=160.0,
        film_poisson_ratio=0.22,
        film_cte_ppm_k=2.6,
        film_density_g_cm3=2.3,
    ),
    "DLC": MaterialProperties(
        film_name="DLC",
        film_youngs_modulus_gpa=200.0,
        film_poisson_ratio=0.25,
        film_cte_ppm_k=1.0,
        film_density_g_cm3=2.0,
    ),
    "GaN": MaterialProperties(
        film_name="GaN",
        film_youngs_modulus_gpa=300.0,
        film_poisson_ratio=0.37,
        film_cte_ppm_k=5.59,
        film_density_g_cm3=6.15,
    ),
}


def get_material_properties(film_material: str) -> MaterialProperties:
    """
    Get material properties from database

    Args:
        film_material: Material name

    Returns:
        MaterialProperties object
    """
    return MATERIAL_DATABASE.get(film_material, MaterialProperties())
