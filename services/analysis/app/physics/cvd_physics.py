"""
CVD Platform - Enhanced Physics Models Library
Comprehensive physics-based models for CVD reactor simulation
Based on peer-reviewed semiconductor processing literature
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve


# ============================================================================
# Physical Constants
# ============================================================================

@dataclass
class PhysicalConstants:
    """Universal physical constants"""
    R_GAS = 8.314  # J/(mol·K) - Universal gas constant
    BOLTZMANN = 1.380649e-23  # J/K - Boltzmann constant
    AVOGADRO = 6.02214076e23  # 1/mol - Avogadro's number
    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴) - Stefan-Boltzmann constant
    PLANCK = 6.62607015e-34  # J·s - Planck constant
    ELEMENTARY_CHARGE = 1.602176634e-19  # C - Elementary charge


CONST = PhysicalConstants()


# ============================================================================
# Gas Flow Dynamics
# ============================================================================

class GasFlowModel:
    """
    Gas flow dynamics for CVD reactors.
    Implements Reynolds number calculations and flow regime determination.
    """

    @staticmethod
    def calculate_reynolds_number(
        density_kg_m3: float,
        velocity_m_s: float,
        length_m: float,
        viscosity_pa_s: float,
    ) -> float:
        """
        Calculate Reynolds number for flow regime determination.

        Re = ρVL/μ

        Args:
            density_kg_m3: Gas density (kg/m³)
            velocity_m_s: Characteristic velocity (m/s)
            length_m: Characteristic length (m)
            viscosity_pa_s: Dynamic viscosity (Pa·s)

        Returns:
            Reynolds number (dimensionless)
        """
        if viscosity_pa_s == 0:
            raise ValueError("Viscosity cannot be zero")

        Re = (density_kg_m3 * velocity_m_s * length_m) / viscosity_pa_s
        return Re

    @staticmethod
    def get_flow_regime(reynolds_number: float) -> str:
        """
        Determine flow regime from Reynolds number.

        Args:
            reynolds_number: Reynolds number

        Returns:
            Flow regime: 'laminar', 'transitional', or 'turbulent'
        """
        if reynolds_number < 2300:
            return "laminar"
        elif reynolds_number < 4000:
            return "transitional"
        else:
            return "turbulent"

    @staticmethod
    def calculate_gas_density(
        pressure_pa: float,
        temperature_k: float,
        molecular_weight_g_mol: float,
    ) -> float:
        """
        Calculate gas density using ideal gas law.

        ρ = PM/(RT)

        Args:
            pressure_pa: Pressure (Pa)
            temperature_k: Temperature (K)
            molecular_weight_g_mol: Molecular weight (g/mol)

        Returns:
            Density (kg/m³)
        """
        # Convert molecular weight to kg/mol
        mw_kg_mol = molecular_weight_g_mol / 1000.0

        # ρ = PM/(RT)
        density = (pressure_pa * mw_kg_mol) / (CONST.R_GAS * temperature_k)

        return density

    @staticmethod
    def calculate_gas_viscosity(
        temperature_k: float,
        gas_name: str,
    ) -> float:
        """
        Calculate gas dynamic viscosity using Sutherland's formula.

        μ = μ₀(T/T₀)^(3/2) * (T₀ + S)/(T + S)

        Args:
            temperature_k: Temperature (K)
            gas_name: Gas name

        Returns:
            Dynamic viscosity (Pa·s)
        """
        # Sutherland parameters for common gases (μ₀ in Pa·s, T₀ = 273.15 K, S in K)
        sutherland_params = {
            "N2": {"mu0": 1.663e-5, "T0": 273.15, "S": 107},
            "H2": {"mu0": 8.411e-6, "T0": 273.15, "S": 97},
            "Ar": {"mu0": 2.125e-5, "T0": 273.15, "S": 114},
            "SiH4": {"mu0": 1.2e-5, "T0": 273.15, "S": 150},  # Approximate
            "NH3": {"mu0": 9.82e-6, "T0": 273.15, "S": 370},
        }

        params = sutherland_params.get(gas_name, {"mu0": 1.8e-5, "T0": 273.15, "S": 110})

        mu0 = params["mu0"]
        T0 = params["T0"]
        S = params["S"]

        # Sutherland's formula
        viscosity = mu0 * (temperature_k / T0) ** 1.5 * (T0 + S) / (temperature_k + S)

        return viscosity


# ============================================================================
# Mass Transport
# ============================================================================

class MassTransportModel:
    """
    Mass transport models for species diffusion and convection.
    """

    @staticmethod
    def calculate_binary_diffusion_coefficient(
        temperature_k: float,
        pressure_pa: float,
        ma_g_mol: float,
        mb_g_mol: float,
        sigma_ab_angstrom: float = 3.5,
        omega_d: float = 1.0,
    ) -> float:
        """
        Calculate binary diffusion coefficient using Chapman-Enskog theory.

        D_AB = 0.001858 T^(3/2) √(1/M_A + 1/M_B) / (P σ_AB² Ω_D)

        Args:
            temperature_k: Temperature (K)
            pressure_pa: Pressure (Pa)
            ma_g_mol: Molecular weight of species A (g/mol)
            mb_g_mol: Molecular weight of species B (g/mol)
            sigma_ab_angstrom: Collision diameter (Å)
            omega_d: Collision integral (dimensionless)

        Returns:
            Binary diffusion coefficient (m²/s)
        """
        # Convert pressure to atm
        pressure_atm = pressure_pa / 101325.0

        # Calculate reduced mass factor
        mass_factor = math.sqrt(1.0 / ma_g_mol + 1.0 / mb_g_mol)

        # Chapman-Enskog formula (result in cm²/s)
        D_cm2_s = (
            0.001858
            * (temperature_k ** 1.5)
            * mass_factor
            / (pressure_atm * (sigma_ab_angstrom ** 2) * omega_d)
        )

        # Convert to m²/s
        D_m2_s = D_cm2_s * 1e-4

        return D_m2_s

    @staticmethod
    def calculate_knudsen_diffusion_coefficient(
        pore_diameter_m: float,
        temperature_k: float,
        molecular_weight_g_mol: float,
    ) -> float:
        """
        Calculate Knudsen diffusion coefficient for low-pressure transport.

        D_K = (d_pore/3)√(8RT/πM)

        Args:
            pore_diameter_m: Pore diameter (m)
            temperature_k: Temperature (K)
            molecular_weight_g_mol: Molecular weight (g/mol)

        Returns:
            Knudsen diffusion coefficient (m²/s)
        """
        mw_kg_mol = molecular_weight_g_mol / 1000.0

        D_K = (pore_diameter_m / 3.0) * math.sqrt(
            (8.0 * CONST.R_GAS * temperature_k) / (math.pi * mw_kg_mol)
        )

        return D_K

    @staticmethod
    def calculate_effective_diffusion_coefficient(
        D_bulk_m2_s: float,
        D_knudsen_m2_s: float,
    ) -> float:
        """
        Calculate effective diffusion coefficient combining bulk and Knudsen.

        1/D_eff = 1/D_bulk + 1/D_K

        Args:
            D_bulk_m2_s: Bulk diffusion coefficient (m²/s)
            D_knudsen_m2_s: Knudsen diffusion coefficient (m²/s)

        Returns:
            Effective diffusion coefficient (m²/s)
        """
        D_eff = 1.0 / (1.0 / D_bulk_m2_s + 1.0 / D_knudsen_m2_s)
        return D_eff

    @staticmethod
    def calculate_concentration_from_flow(
        flow_sccm: float,
        total_flow_sccm: float,
        pressure_pa: float,
        temperature_k: float,
    ) -> float:
        """
        Calculate molar concentration from gas flow rate.

        Args:
            flow_sccm: Species flow rate (sccm)
            total_flow_sccm: Total flow rate (sccm)
            pressure_pa: Pressure (Pa)
            temperature_k: Temperature (K)

        Returns:
            Molar concentration (mol/m³)
        """
        if total_flow_sccm == 0:
            return 0.0

        # Mole fraction
        x = flow_sccm / total_flow_sccm

        # Total molar concentration from ideal gas law: C = P/(RT)
        C_total = pressure_pa / (CONST.R_GAS * temperature_k)

        # Partial concentration
        C = x * C_total

        return C


# ============================================================================
# Reaction Kinetics
# ============================================================================

class ReactionKineticsModel:
    """
    Chemical reaction kinetics for CVD processes.
    Implements Arrhenius equations and surface reaction models.
    """

    @staticmethod
    def arrhenius_rate_constant(
        k0: float,
        activation_energy_j_mol: float,
        temperature_k: float,
    ) -> float:
        """
        Calculate reaction rate constant using Arrhenius equation.

        k(T) = k₀ exp(-E_a/RT)

        Args:
            k0: Pre-exponential factor
            activation_energy_j_mol: Activation energy (J/mol)
            temperature_k: Temperature (K)

        Returns:
            Rate constant (units depend on k0)
        """
        exponent = -activation_energy_j_mol / (CONST.R_GAS * temperature_k)
        k = k0 * math.exp(exponent)
        return k

    @staticmethod
    def silicon_deposition_rate(
        temperature_k: float,
        silane_concentration_mol_m3: float,
        k0: float = 1.0e8,  # s⁻¹
        ea: float = 170000,  # J/mol
        reaction_order: float = 1.0,
    ) -> float:
        """
        Calculate silicon deposition rate from silane.

        SiH₄(g) → Si(s) + 2H₂(g)

        r = k(T) [SiH₄]^n

        Args:
            temperature_k: Temperature (K)
            silane_concentration_mol_m3: SiH4 concentration (mol/m³)
            k0: Pre-exponential factor (s⁻¹)
            ea: Activation energy (J/mol)
            reaction_order: Reaction order

        Returns:
            Reaction rate (mol/(m²·s))
        """
        k = ReactionKineticsModel.arrhenius_rate_constant(k0, ea, temperature_k)
        rate = k * (silane_concentration_mol_m3 ** reaction_order)
        return rate

    @staticmethod
    def silicon_nitride_deposition_rate(
        temperature_k: float,
        silane_concentration_mol_m3: float,
        ammonia_concentration_mol_m3: float,
        k0: float = 5.0e7,
        ea: float = 150000,  # J/mol
        order_sih4: float = 1.0,
        order_nh3: float = 0.5,
    ) -> float:
        """
        Calculate silicon nitride deposition rate.

        3SiH₄ + 4NH₃ → Si₃N₄ + 12H₂

        r = k(T) [SiH₄]^a [NH₃]^b

        Args:
            temperature_k: Temperature (K)
            silane_concentration_mol_m3: SiH4 concentration (mol/m³)
            ammonia_concentration_mol_m3: NH3 concentration (mol/m³)
            k0: Pre-exponential factor
            ea: Activation energy (J/mol)
            order_sih4: Reaction order for SiH4
            order_nh3: Reaction order for NH3

        Returns:
            Reaction rate (mol/(m²·s))
        """
        k = ReactionKineticsModel.arrhenius_rate_constant(k0, ea, temperature_k)
        rate = k * (silane_concentration_mol_m3 ** order_sih4) * (ammonia_concentration_mol_m3 ** order_nh3)
        return rate

    @staticmethod
    def langmuir_hinshelwood_rate(
        k_ads: float,
        K_ads: float,
        C_gas: float,
    ) -> float:
        """
        Calculate reaction rate using Langmuir-Hinshelwood mechanism.

        r = (k_ads K_ads C_gas) / (1 + K_ads C_gas)

        Args:
            k_ads: Adsorption rate constant
            K_ads: Adsorption equilibrium constant
            C_gas: Gas concentration

        Returns:
            Reaction rate
        """
        numerator = k_ads * K_ads * C_gas
        denominator = 1.0 + K_ads * C_gas
        rate = numerator / denominator
        return rate


# ============================================================================
# Heat Transfer
# ============================================================================

class HeatTransferModel:
    """
    Heat transfer models for CVD reactors.
    Includes radiation, convection, and conduction.
    """

    @staticmethod
    def radiative_heat_flux(
        T_hot_k: float,
        T_cold_k: float,
        emissivity: float = 0.7,
    ) -> float:
        """
        Calculate radiative heat flux using Stefan-Boltzmann law.

        q_rad = ε σ (T_hot⁴ - T_cold⁴)

        Args:
            T_hot_k: Hot surface temperature (K)
            T_cold_k: Cold surface temperature (K)
            emissivity: Surface emissivity (0-1)

        Returns:
            Heat flux (W/m²)
        """
        q_rad = emissivity * CONST.STEFAN_BOLTZMANN * (T_hot_k ** 4 - T_cold_k ** 4)
        return q_rad

    @staticmethod
    def convective_heat_flux(
        T_surface_k: float,
        T_infinity_k: float,
        heat_transfer_coefficient_w_m2_k: float,
    ) -> float:
        """
        Calculate convective heat flux using Newton's law of cooling.

        q_conv = h(T_surface - T_∞)

        Args:
            T_surface_k: Surface temperature (K)
            T_infinity_k: Bulk fluid temperature (K)
            heat_transfer_coefficient_w_m2_k: Convective heat transfer coefficient (W/(m²·K))

        Returns:
            Heat flux (W/m²)
        """
        q_conv = heat_transfer_coefficient_w_m2_k * (T_surface_k - T_infinity_k)
        return q_conv

    @staticmethod
    def calculate_nusselt_number(
        reynolds_number: float,
        prandtl_number: float,
        correlation: str = "laminar_flat_plate",
    ) -> float:
        """
        Calculate Nusselt number for convective heat transfer.

        Args:
            reynolds_number: Reynolds number
            prandtl_number: Prandtl number (Pr = μC_p/k)
            correlation: Correlation type

        Returns:
            Nusselt number
        """
        if correlation == "laminar_flat_plate":
            # Nu = 0.664 Re^0.5 Pr^0.33
            Nu = 0.664 * (reynolds_number ** 0.5) * (prandtl_number ** 0.33)

        elif correlation == "turbulent_flat_plate":
            # Nu = 0.037 Re^0.8 Pr^0.33
            Nu = 0.037 * (reynolds_number ** 0.8) * (prandtl_number ** 0.33)

        elif correlation == "cylinder_cross_flow":
            # Nu = C Re^m Pr^n (simplified)
            C = 0.3
            m = 0.6
            n = 0.37
            Nu = C * (reynolds_number ** m) * (prandtl_number ** n)

        else:
            raise ValueError(f"Unknown correlation: {correlation}")

        return Nu

    @staticmethod
    def calculate_prandtl_number(
        viscosity_pa_s: float,
        specific_heat_j_kg_k: float,
        thermal_conductivity_w_m_k: float,
    ) -> float:
        """
        Calculate Prandtl number.

        Pr = μC_p/k

        Args:
            viscosity_pa_s: Dynamic viscosity (Pa·s)
            specific_heat_j_kg_k: Specific heat capacity (J/(kg·K))
            thermal_conductivity_w_m_k: Thermal conductivity (W/(m·K))

        Returns:
            Prandtl number (dimensionless)
        """
        Pr = (viscosity_pa_s * specific_heat_j_kg_k) / thermal_conductivity_w_m_k
        return Pr


# ============================================================================
# Deposition Rate Models
# ============================================================================

class DepositionRateModel:
    """
    Film deposition rate calculations.
    Converts reaction rates to growth rates.
    """

    @staticmethod
    def growth_rate_from_flux(
        molar_flux_mol_m2_s: float,
        molecular_weight_kg_mol: float,
        film_density_kg_m3: float,
    ) -> float:
        """
        Calculate film growth rate from molar flux.

        Growth Rate = (MW/ρ_film) × Flux

        Args:
            molar_flux_mol_m2_s: Molar flux (mol/(m²·s))
            molecular_weight_kg_mol: Molecular weight (kg/mol)
            film_density_kg_m3: Film density (kg/m³)

        Returns:
            Growth rate (nm/min)
        """
        # Calculate growth rate in m/s
        growth_rate_m_s = (molecular_weight_kg_mol / film_density_kg_m3) * molar_flux_mol_m2_s

        # Convert to nm/min
        growth_rate_nm_min = growth_rate_m_s * 1e9 * 60

        return growth_rate_nm_min

    @staticmethod
    def calculate_total_flux(
        diffusion_flux: float,
        reaction_flux: float,
        regime: str = "mixed",
    ) -> float:
        """
        Calculate total flux considering diffusion and reaction.

        For mixed regime: 1/J_total = 1/J_diffusion + 1/J_reaction

        Args:
            diffusion_flux: Mass transport limited flux
            reaction_flux: Reaction limited flux
            regime: 'diffusion', 'reaction', or 'mixed'

        Returns:
            Total flux
        """
        if regime == "diffusion":
            return diffusion_flux
        elif regime == "reaction":
            return reaction_flux
        elif regime == "mixed":
            # Resistances in series
            if diffusion_flux == 0 or reaction_flux == 0:
                return 0.0
            J_total = 1.0 / (1.0 / diffusion_flux + 1.0 / reaction_flux)
            return J_total
        else:
            raise ValueError(f"Unknown regime: {regime}")


# ============================================================================
# Film Uniformity Model
# ============================================================================

class UniformityModel:
    """
    Film thickness uniformity calculations.
    """

    @staticmethod
    def radial_thickness_profile(
        center_thickness: float,
        radial_position_mm: float,
        characteristic_length_mm: float,
    ) -> float:
        """
        Calculate thickness at radial position due to depletion effects.

        t(r) = t_center × exp(-r²/L²)

        Args:
            center_thickness: Thickness at center (nm)
            radial_position_mm: Radial position from center (mm)
            characteristic_length_mm: Characteristic depletion length (mm)

        Returns:
            Thickness at position (nm)
        """
        exponent = -(radial_position_mm ** 2) / (characteristic_length_mm ** 2)
        thickness = center_thickness * math.exp(exponent)
        return thickness

    @staticmethod
    def calculate_uniformity_metric(
        thickness_values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Calculate uniformity metrics from thickness map.

        Args:
            thickness_values: Array of thickness measurements

        Returns:
            Dictionary with uniformity metrics
        """
        t_mean = np.mean(thickness_values)
        t_std = np.std(thickness_values)
        t_min = np.min(thickness_values)
        t_max = np.max(thickness_values)

        # Uniformity = (max - min) / mean × 100%
        uniformity_range = ((t_max - t_min) / t_mean) * 100

        # 1-sigma uniformity
        uniformity_sigma = (t_std / t_mean) * 100

        return {
            "mean_nm": float(t_mean),
            "std_nm": float(t_std),
            "min_nm": float(t_min),
            "max_nm": float(t_max),
            "uniformity_range_pct": float(uniformity_range),
            "uniformity_sigma_pct": float(uniformity_sigma),
        }

    @staticmethod
    def rotation_effect_on_uniformity(
        base_uniformity_pct: float,
        rotation_speed_rpm: float,
        mixing_time_constant_s: float = 10.0,
    ) -> float:
        """
        Calculate uniformity improvement due to wafer rotation.

        U = U₀/(1 + ω·τ_mix)

        Args:
            base_uniformity_pct: Baseline uniformity without rotation (%)
            rotation_speed_rpm: Rotation speed (RPM)
            mixing_time_constant_s: Mixing time constant (s)

        Returns:
            Improved uniformity (%)
        """
        # Convert RPM to rad/s
        omega_rad_s = rotation_speed_rpm * (2 * math.pi / 60)

        # Apply rotation improvement
        improved_uniformity = base_uniformity_pct / (1.0 + omega_rad_s * mixing_time_constant_s)

        return improved_uniformity


# ============================================================================
# Sensitivity Analysis
# ============================================================================

class SensitivityAnalysis:
    """
    Sensitivity analysis for process parameters.
    """

    @staticmethod
    def calculate_sensitivity(
        parameter_values: np.ndarray,
        output_values: np.ndarray,
    ) -> float:
        """
        Calculate sensitivity (dOutput/dParameter).

        Args:
            parameter_values: Array of parameter values
            output_values: Array of corresponding output values

        Returns:
            Sensitivity (dOutput/dParameter)
        """
        # Linear fit
        coeffs = np.polyfit(parameter_values, output_values, 1)
        sensitivity = coeffs[0]  # Slope

        return float(sensitivity)

    @staticmethod
    def monte_carlo_uncertainty(
        nominal_params: Dict[str, float],
        param_uncertainties: Dict[str, float],
        model_function,
        n_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Monte Carlo uncertainty propagation.

        Args:
            nominal_params: Nominal parameter values
            param_uncertainties: Parameter standard deviations
            model_function: Function that takes params dict and returns output
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary with statistics
        """
        outputs = []

        for _ in range(n_samples):
            # Sample parameters
            sampled_params = {}
            for key, nominal in nominal_params.items():
                sigma = param_uncertainties.get(key, 0)
                sampled_params[key] = np.random.normal(nominal, sigma)

            # Evaluate model
            output = model_function(sampled_params)
            outputs.append(output)

        outputs = np.array(outputs)

        return {
            "mean": float(np.mean(outputs)),
            "std": float(np.std(outputs)),
            "min": float(np.min(outputs)),
            "max": float(np.max(outputs)),
            "percentile_5": float(np.percentile(outputs, 5)),
            "percentile_95": float(np.percentile(outputs, 95)),
        }
