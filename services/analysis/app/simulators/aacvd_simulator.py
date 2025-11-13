"""
AACVD (Aerosol-Assisted Chemical Vapor Deposition) Simulator
Specialized for solution-based precursor delivery via aerosol
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime, timedelta

from ..physics.cvd_physics import (
    calculate_arrhenius_rate,
    calculate_diffusion_coefficient,
)


@dataclass
class AerosolConfig:
    """Aerosol generation configuration"""
    carrier_gas: str  # "N2", "Air", "O2"
    carrier_flow_slm: float  # Standard liters per minute
    atomizer_frequency_khz: float  # Ultrasonic frequency
    droplet_size_mean_um: float  # Mean droplet diameter
    droplet_size_std_um: float  # Droplet size distribution std
    precursor_concentration_mol_l: float  # Molar concentration in solution
    solvent: str  # "Water", "Ethanol", "Methanol", "DMF", etc.


@dataclass
class AACVDReactorConfig:
    """AACVD reactor configuration"""
    reactor_type: str  # "horizontal_tube", "vertical_tube", "spray_pyrolysis"
    substrate_temp_c: float
    vaporizer_temp_c: float  # Temperature where aerosol is vaporized
    reactor_length_cm: float
    reactor_diameter_cm: float
    substrate_position_cm: float  # Distance from inlet
    pressure_pa: float = 101325.0  # Typically atmospheric


@dataclass
class PrecursorSolution:
    """Precursor solution properties"""
    name: str
    solute_formula: str
    concentration_mol_l: float
    solvent: str
    solvent_boiling_point_c: float
    precursor_decomp_temp_c: float
    metal_content_pct: float
    molar_mass_g_mol: float


# Common AACVD precursor solutions
AACVD_SOLUTIONS = {
    "SnCl2_water": PrecursorSolution(
        name="Tin(II) chloride in water",
        solute_formula="SnCl2",
        concentration_mol_l=0.1,
        solvent="Water",
        solvent_boiling_point_c=100.0,
        precursor_decomp_temp_c=250.0,
        metal_content_pct=62.6,
        molar_mass_g_mol=189.6,
    ),
    "ZnAc_methanol": PrecursorSolution(
        name="Zinc acetate in methanol",
        solute_formula="Zn(CH3COO)2",
        concentration_mol_l=0.05,
        solvent="Methanol",
        solvent_boiling_point_c=64.7,
        precursor_decomp_temp_c=300.0,
        metal_content_pct=35.8,
        molar_mass_g_mol=183.5,
    ),
    "TiOiPr_ethanol": PrecursorSolution(
        name="Titanium isopropoxide in ethanol",
        solute_formula="Ti[OCH(CH3)2]4",
        concentration_mol_l=0.2,
        solvent="Ethanol",
        solvent_boiling_point_c=78.4,
        precursor_decomp_temp_c=350.0,
        metal_content_pct=16.8,
        molar_mass_g_mol=284.2,
    ),
    "FeCl3_water": PrecursorSolution(
        name="Iron(III) chloride in water",
        solute_formula="FeCl3",
        concentration_mol_l=0.15,
        solvent="Water",
        solvent_boiling_point_c=100.0,
        precursor_decomp_temp_c=280.0,
        metal_content_pct=34.4,
        molar_mass_g_mol=162.2,
    ),
}


class AACVDSimulator:
    """
    Aerosol-Assisted CVD Simulator

    Simulates deposition from solution-based precursors:
    - Metal oxides (ZnO, SnO2, TiO2, Fe2O3)
    - Transparent conductive oxides (ITO, FTO)
    - Multi-component oxides
    - Aerosol droplet transport and evaporation
    - Substrate temperature effects
    """

    def __init__(
        self,
        reactor_config: AACVDReactorConfig,
        aerosol_config: AerosolConfig,
    ):
        self.reactor = reactor_config
        self.aerosol = aerosol_config
        self.time_step_s = 1.0  # 1 second time steps

    def simulate_deposition(
        self,
        precursor_solution: PrecursorSolution,
        duration_min: float,
        substrate_area_cm2: float = 25.0,  # 5cm x 5cm substrate
    ) -> Dict[str, Any]:
        """
        Simulate AACVD deposition process

        Args:
            precursor_solution: Precursor solution properties
            duration_min: Deposition duration in minutes
            substrate_area_cm2: Substrate area in cm²

        Returns:
            Simulation results including thickness, morphology, composition
        """

        # Calculate aerosol transport efficiency
        transport_eff = self._calculate_transport_efficiency(precursor_solution)

        # Calculate deposition rate
        deposition_rate_nm_min = self._calculate_deposition_rate(
            precursor_solution=precursor_solution,
            transport_efficiency=transport_eff,
            substrate_area_cm2=substrate_area_cm2,
        )

        # Calculate total thickness
        thickness_nm = deposition_rate_nm_min * duration_min

        # Calculate uniformity
        uniformity_pct = self._calculate_uniformity(
            reactor_type=self.reactor.reactor_type,
            substrate_position_cm=self.reactor.substrate_position_cm,
            reactor_length_cm=self.reactor.reactor_length_cm,
        )

        # Calculate morphology characteristics
        morphology = self._calculate_morphology(
            substrate_temp_c=self.reactor.substrate_temp_c,
            droplet_size_um=self.aerosol.droplet_size_mean_um,
            deposition_rate_nm_min=deposition_rate_nm_min,
        )

        # Estimate crystallinity
        crystallinity_pct = self._estimate_crystallinity(
            substrate_temp_c=self.reactor.substrate_temp_c,
            precursor_decomp_temp_c=precursor_solution.precursor_decomp_temp_c,
        )

        # Generate telemetry
        telemetry = self._generate_telemetry(
            duration_min=duration_min,
            precursor_solution=precursor_solution,
            deposition_rate_nm_min=deposition_rate_nm_min,
        )

        return {
            "thickness_nm": thickness_nm,
            "thickness_std_nm": thickness_nm * (1.0 - uniformity_pct / 100.0) / 3.0,
            "uniformity_pct": uniformity_pct,
            "deposition_rate_nm_min": deposition_rate_nm_min,
            "transport_efficiency_pct": transport_eff * 100.0,
            "morphology": morphology,
            "crystallinity_pct": crystallinity_pct,
            "substrate_temp_c": self.reactor.substrate_temp_c,
            "vaporizer_temp_c": self.reactor.vaporizer_temp_c,
            "droplet_size_um": self.aerosol.droplet_size_mean_um,
            "precursor_concentration_mol_l": precursor_solution.concentration_mol_l,
            "telemetry": telemetry,
        }

    def _calculate_transport_efficiency(
        self,
        precursor_solution: PrecursorSolution,
    ) -> float:
        """Calculate fraction of aerosol reaching and depositing on substrate"""

        # Droplet transport depends on:
        # 1. Aerosol generation efficiency
        # 2. Transport losses (wall deposition, settling)
        # 3. Vaporization efficiency
        # 4. Precursor decomposition efficiency

        # 1. Aerosol generation (ultrasonic atomizer)
        # Higher frequency -> smaller droplets -> better transport
        generation_eff = min(
            self.aerosol.atomizer_frequency_khz / 2000.0, 0.95
        )

        # 2. Transport losses
        # Calculate settling velocity (Stokes' law)
        droplet_diameter_m = self.aerosol.droplet_size_mean_um * 1e-6

        # Assume aerosol density ~ solvent density (1000 kg/m³ for water)
        droplet_density = 1000.0  # kg/m³
        air_viscosity = 1.8e-5  # Pa·s at room temp

        # Stokes settling velocity (m/s)
        g = 9.81  # m/s²
        settling_velocity = (
            (droplet_density * g * droplet_diameter_m ** 2)
            / (18 * air_viscosity)
        )

        # Residence time in reactor
        flow_velocity_m_s = (
            self.aerosol.carrier_flow_slm / 60.0  # L/s
            / (np.pi * (self.reactor.reactor_diameter_cm / 200.0) ** 2)  # m²
        )
        residence_time_s = self.reactor.reactor_length_cm / 100.0 / flow_velocity_m_s

        # Fraction lost to settling
        settling_loss = min(
            settling_velocity * residence_time_s / (self.reactor.reactor_diameter_cm / 100.0),
            0.5,
        )

        # Wall deposition losses (diffusion + impaction)
        wall_loss = 0.15  # Typical 15% wall loss

        transport_loss = settling_loss + wall_loss
        transport_eff = max(0.1, 1.0 - transport_loss) * generation_eff

        # 3. Vaporization efficiency
        # Check if vaporizer temp > solvent boiling point
        if self.reactor.vaporizer_temp_c > precursor_solution.solvent_boiling_point_c + 20:
            vaporization_eff = 0.95
        elif self.reactor.vaporizer_temp_c > precursor_solution.solvent_boiling_point_c:
            vaporization_eff = 0.7
        else:
            vaporization_eff = 0.3  # Incomplete vaporization

        # 4. Decomposition efficiency
        if self.reactor.substrate_temp_c > precursor_solution.precursor_decomp_temp_c + 50:
            decomp_eff = 0.9
        elif self.reactor.substrate_temp_c > precursor_solution.precursor_decomp_temp_c:
            decomp_eff = 0.6
        else:
            # Low temperature - incomplete decomposition
            temp_ratio = self.reactor.substrate_temp_c / precursor_solution.precursor_decomp_temp_c
            decomp_eff = max(0.1, temp_ratio - 0.5)

        overall_eff = transport_eff * vaporization_eff * decomp_eff

        return max(0.01, min(0.9, overall_eff))

    def _calculate_deposition_rate(
        self,
        precursor_solution: PrecursorSolution,
        transport_efficiency: float,
        substrate_area_cm2: float,
    ) -> float:
        """Calculate deposition rate in nm/min"""

        # Mass flow of precursor
        # Carrier flow (L/min) * aerosol loading
        # Aerosol loading ~ 1-10 g/m³ for ultrasonic atomizers
        aerosol_loading_g_m3 = 5.0
        carrier_flow_m3_min = self.aerosol.carrier_flow_slm / 1000.0

        # Mass of solution transported per minute
        solution_mass_g_min = carrier_flow_m3_min * aerosol_loading_g_m3

        # Moles of precursor per minute
        # Assume solution density ~ 1 g/mL
        solution_volume_ml_min = solution_mass_g_min  # Approx for dilute solutions
        precursor_mol_min = (
            solution_volume_ml_min / 1000.0  # L/min
            * precursor_solution.concentration_mol_l
        )

        # Moles actually deposited
        deposited_mol_min = precursor_mol_min * transport_efficiency

        # Convert to mass of deposited film
        # For metal oxides, need to account for oxidation
        # E.g., Zn -> ZnO (molar mass ratio)
        # Simplified: assume stoichiometric oxide formation
        film_molar_mass = precursor_solution.molar_mass_g_mol * 1.5  # Rough approximation
        deposited_mass_g_min = deposited_mol_min * film_molar_mass

        # Film density (typical oxide: 5 g/cm³)
        film_density_g_cm3 = 5.0

        # Volume deposited per minute
        volume_cm3_min = deposited_mass_g_min / film_density_g_cm3

        # Thickness increase per minute
        thickness_cm_min = volume_cm3_min / substrate_area_cm2

        # Convert to nm/min
        deposition_rate_nm_min = thickness_cm_min * 1e7

        # Add substrate temperature effect
        # Higher temp -> better crystallinity but can reduce sticking
        if self.reactor.substrate_temp_c < 200:
            temp_factor = 0.5
        elif self.reactor.substrate_temp_c < 400:
            temp_factor = 0.8 + (self.reactor.substrate_temp_c - 200) / 200 * 0.4
        elif self.reactor.substrate_temp_c < 600:
            temp_factor = 1.2
        else:
            # Very high temp - re-evaporation
            temp_factor = 1.2 * math.exp(-(self.reactor.substrate_temp_c - 600) / 200)

        deposition_rate_nm_min *= temp_factor

        return max(0.1, deposition_rate_nm_min)

    def _calculate_uniformity(
        self,
        reactor_type: str,
        substrate_position_cm: float,
        reactor_length_cm: float,
    ) -> float:
        """Calculate thickness uniformity percentage"""

        # Base uniformity depends on reactor design
        base_uniformity = {
            "horizontal_tube": 75.0,
            "vertical_tube": 80.0,
            "spray_pyrolysis": 70.0,
        }.get(reactor_type, 75.0)

        # Position effect - better uniformity in middle of reactor
        position_ratio = substrate_position_cm / reactor_length_cm
        if 0.4 < position_ratio < 0.6:
            # Middle region - best uniformity
            position_factor = 1.1
        elif 0.2 < position_ratio < 0.8:
            # Good region
            position_factor = 1.0
        else:
            # Near inlet/outlet - poorer uniformity
            position_factor = 0.9

        # Droplet size uniformity effect
        size_cv = self.aerosol.droplet_size_std_um / self.aerosol.droplet_size_mean_um
        size_factor = 1.0 - size_cv * 0.2

        uniformity = base_uniformity * position_factor * size_factor

        return min(95.0, max(60.0, uniformity))

    def _calculate_morphology(
        self,
        substrate_temp_c: float,
        droplet_size_um: float,
        deposition_rate_nm_min: float,
    ) -> Dict[str, Any]:
        """Calculate film morphology characteristics"""

        # Grain size depends on substrate temperature and deposition rate
        if substrate_temp_c < 300:
            grain_size_nm = 20 + substrate_temp_c / 10
            morphology_type = "amorphous/nanocrystalline"
        elif substrate_temp_c < 450:
            grain_size_nm = 50 + (substrate_temp_c - 300) / 3
            morphology_type = "polycrystalline"
        else:
            grain_size_nm = 100 + (substrate_temp_c - 450) / 5
            morphology_type = "columnar"

        # Slower deposition -> larger grains
        rate_factor = 1.0 / math.sqrt(deposition_rate_nm_min / 10.0)
        grain_size_nm *= rate_factor

        # Surface roughness (RMS)
        # Depends on droplet size and deposition conditions
        roughness_nm = droplet_size_um * 10 + grain_size_nm * 0.1

        # Porosity (lower temp -> higher porosity)
        if substrate_temp_c < 300:
            porosity_pct = 15.0 - substrate_temp_c / 30
        else:
            porosity_pct = 5.0 - (substrate_temp_c - 300) / 100

        porosity_pct = max(1.0, min(20.0, porosity_pct))

        return {
            "grain_size_nm": grain_size_nm,
            "morphology_type": morphology_type,
            "surface_roughness_rms_nm": roughness_nm,
            "porosity_pct": porosity_pct,
            "structure": "columnar" if substrate_temp_c > 450 else "granular",
        }

    def _estimate_crystallinity(
        self,
        substrate_temp_c: float,
        precursor_decomp_temp_c: float,
    ) -> float:
        """Estimate film crystallinity percentage"""

        # Crystallinity depends on substrate temperature
        # relative to precursor decomposition temperature

        temp_excess = substrate_temp_c - precursor_decomp_temp_c

        if temp_excess < 0:
            # Below decomposition - mostly amorphous
            crystallinity = 10.0
        elif temp_excess < 50:
            # Near decomposition - low crystallinity
            crystallinity = 10 + temp_excess
        elif temp_excess < 150:
            # Good crystallization range
            crystallinity = 60 + (temp_excess - 50) * 0.3
        else:
            # High temp - excellent crystallinity
            crystallinity = min(95.0, 90 + (temp_excess - 150) * 0.05)

        return crystallinity

    def _generate_telemetry(
        self,
        duration_min: float,
        precursor_solution: PrecursorSolution,
        deposition_rate_nm_min: float,
    ) -> List[Dict[str, Any]]:
        """Generate time-series telemetry data"""

        telemetry = []
        num_steps = int(duration_min * 60 / self.time_step_s)

        start_time = datetime.utcnow()

        for i in range(num_steps):
            timestamp = start_time + timedelta(seconds=i * self.time_step_s)

            # Add realistic noise
            substrate_temp_noise = np.random.normal(0, 2.0)
            vaporizer_temp_noise = np.random.normal(0, 1.0)
            flow_noise = np.random.normal(0, self.aerosol.carrier_flow_slm * 0.02)

            telemetry.append({
                "timestamp": timestamp.isoformat(),
                "substrate_temp_c": self.reactor.substrate_temp_c + substrate_temp_noise,
                "vaporizer_temp_c": self.reactor.vaporizer_temp_c + vaporizer_temp_noise,
                "carrier_flow_slm": self.aerosol.carrier_flow_slm + flow_noise,
                "pressure_pa": self.reactor.pressure_pa + np.random.normal(0, 10),
                "atomizer_frequency_khz": self.aerosol.atomizer_frequency_khz,
                "deposition_rate_nm_min": deposition_rate_nm_min + np.random.normal(
                    0, deposition_rate_nm_min * 0.1
                ),
                "cumulative_thickness_nm": (i * self.time_step_s / 60.0) * deposition_rate_nm_min,
            })

        return telemetry


def simulate_zno_deposition(
    substrate_temp_c: float = 400.0,
    vaporizer_temp_c: float = 120.0,
    concentration_mol_l: float = 0.05,
    carrier_flow_slm: float = 2.0,
    duration_min: float = 30.0,
    droplet_size_um: float = 3.0,
) -> Dict[str, Any]:
    """
    Convenience function for ZnO AACVD deposition

    Typical ZnO AACVD conditions:
    - Substrate temp: 350-500°C
    - Precursor: Zinc acetate in methanol/ethanol
    - Carrier: Air or N2
    - Deposition rate: 10-100 nm/min
    """

    reactor = AACVDReactorConfig(
        reactor_type="horizontal_tube",
        substrate_temp_c=substrate_temp_c,
        vaporizer_temp_c=vaporizer_temp_c,
        reactor_length_cm=50.0,
        reactor_diameter_cm=5.0,
        substrate_position_cm=25.0,
        pressure_pa=101325.0,
    )

    aerosol = AerosolConfig(
        carrier_gas="Air",
        carrier_flow_slm=carrier_flow_slm,
        atomizer_frequency_khz=1700.0,
        droplet_size_mean_um=droplet_size_um,
        droplet_size_std_um=droplet_size_um * 0.3,
        precursor_concentration_mol_l=concentration_mol_l,
        solvent="Methanol",
    )

    simulator = AACVDSimulator(reactor, aerosol)

    # Use zinc acetate solution
    precursor = AACVD_SOLUTIONS["ZnAc_methanol"]
    precursor.concentration_mol_l = concentration_mol_l

    return simulator.simulate_deposition(
        precursor_solution=precursor,
        duration_min=duration_min,
        substrate_area_cm2=25.0,
    )
