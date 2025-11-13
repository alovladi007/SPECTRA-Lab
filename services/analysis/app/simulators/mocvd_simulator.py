"""
MOCVD (Metal-Organic Chemical Vapor Deposition) Simulator
Specialized for III-V and II-VI compound semiconductors
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math
from datetime import datetime, timedelta

from ..physics.cvd_physics import (
    calculate_arrhenius_rate,
    calculate_pressure_dependent_rate,
    calculate_film_stress,
)


@dataclass
class MOCVDPrecursor:
    """MOCVD precursor properties"""
    name: str
    formula: str
    vapor_pressure_torr: float  # At operating temperature
    decomposition_temp_c: float
    decomposition_energy_kj_mol: float
    molar_mass_g_mol: float
    metal_content_pct: float  # Percentage of metal in precursor
    sticking_coefficient: float = 0.3


@dataclass
class MOCVDReactorConfig:
    """MOCVD reactor configuration"""
    reactor_type: str  # "horizontal", "vertical", "rotating_disk", "close_coupled_showerhead"
    pressure_pa: float
    susceptor_rotation_rpm: float = 0.0
    carrier_gas: str = "H2"  # H2 or N2
    total_flow_slm: float = 10.0  # Standard liters per minute
    reactor_volume_l: float = 20.0
    susceptor_diameter_mm: float = 200.0
    wafer_diameter_mm: float = 150.0
    growth_zone_height_mm: float = 10.0


# Common MOCVD Precursors
MOCVD_PRECURSORS = {
    # Group III precursors
    "TMGa": MOCVDPrecursor(
        name="Trimethylgallium",
        formula="Ga(CH3)3",
        vapor_pressure_torr=240.0,
        decomposition_temp_c=350.0,
        decomposition_energy_kj_mol=120.0,
        molar_mass_g_mol=114.83,
        metal_content_pct=60.7,
        sticking_coefficient=0.4,
    ),
    "TEGa": MOCVDPrecursor(
        name="Triethylgallium",
        formula="Ga(C2H5)3",
        vapor_pressure_torr=95.0,
        decomposition_temp_c=380.0,
        decomposition_energy_kj_mol=125.0,
        molar_mass_g_mol=156.93,
        metal_content_pct=44.4,
        sticking_coefficient=0.35,
    ),
    "TMIn": MOCVDPrecursor(
        name="Trimethylindium",
        formula="In(CH3)3",
        vapor_pressure_torr=2.0,
        decomposition_temp_c=400.0,
        decomposition_energy_kj_mol=130.0,
        molar_mass_g_mol=159.93,
        metal_content_pct=71.7,
        sticking_coefficient=0.45,
    ),
    "TMAl": MOCVDPrecursor(
        name="Trimethylaluminum",
        formula="Al(CH3)3",
        vapor_pressure_torr=12.0,
        decomposition_temp_c=450.0,
        decomposition_energy_kj_mol=140.0,
        molar_mass_g_mol=72.09,
        metal_content_pct=37.4,
        sticking_coefficient=0.5,
    ),
    # Group V precursors
    "AsH3": MOCVDPrecursor(
        name="Arsine",
        formula="AsH3",
        vapor_pressure_torr=760.0 * 1e3,
        decomposition_temp_c=300.0,
        decomposition_energy_kj_mol=60.0,
        molar_mass_g_mol=77.95,
        metal_content_pct=96.2,
        sticking_coefficient=0.2,
    ),
    "PH3": MOCVDPrecursor(
        name="Phosphine",
        formula="PH3",
        vapor_pressure_torr=760.0 * 1e3,
        decomposition_temp_c=350.0,
        decomposition_energy_kj_mol=65.0,
        molar_mass_g_mol=34.0,
        metal_content_pct=91.2,
        sticking_coefficient=0.25,
    ),
    "NH3": MOCVDPrecursor(
        name="Ammonia",
        formula="NH3",
        vapor_pressure_torr=760.0 * 1e3,
        decomposition_temp_c=500.0,
        decomposition_energy_kj_mol=110.0,
        molar_mass_g_mol=17.03,
        metal_content_pct=82.2,
        sticking_coefficient=0.1,
    ),
}


class MOCVDSimulator:
    """
    Metal-Organic CVD Simulator

    Simulates epitaxial growth of compound semiconductors:
    - GaN, AlN, InN (nitrides)
    - GaAs, InP, InGaAs (arsenides/phosphides)
    - Temperature-dependent growth kinetics
    - Composition control for alloys
    - Parasitic deposition effects
    """

    def __init__(self, reactor_config: MOCVDReactorConfig):
        self.reactor = reactor_config
        self.time_step_s = 0.1  # 100ms time steps

    def simulate_growth(
        self,
        precursor_flows: Dict[str, float],  # sccm for each precursor
        temperature_c: float,
        duration_s: float,
        v_iii_ratio: Optional[float] = None,  # V/III ratio (group V / group III)
    ) -> Dict[str, Any]:
        """
        Simulate MOCVD growth process

        Args:
            precursor_flows: Precursor flow rates in sccm
            temperature_c: Susceptor temperature in Celsius
            duration_s: Growth duration in seconds
            v_iii_ratio: Override V/III ratio (calculated from flows if None)

        Returns:
            Simulation results including thickness, composition, uniformity
        """

        # Calculate actual V/III ratio
        actual_v_iii = self._calculate_v_iii_ratio(precursor_flows)
        if v_iii_ratio is not None:
            actual_v_iii = v_iii_ratio

        # Calculate growth rate
        growth_rate_nm_min = self._calculate_growth_rate(
            precursor_flows=precursor_flows,
            temperature_c=temperature_c,
            v_iii_ratio=actual_v_iii,
        )

        # Calculate composition (for ternary/quaternary alloys)
        composition = self._calculate_composition(precursor_flows, temperature_c)

        # Calculate uniformity
        uniformity_pct = self._calculate_uniformity(
            temperature_c=temperature_c,
            rotation_rpm=self.reactor.susceptor_rotation_rpm,
            reactor_type=self.reactor.reactor_type,
        )

        # Calculate total thickness
        thickness_nm = growth_rate_nm_min * (duration_s / 60.0)

        # Calculate parasitic deposition fraction
        parasitic_fraction = self._calculate_parasitic_deposition(
            temperature_c=temperature_c,
            pressure_pa=self.reactor.pressure_pa,
            precursor_flows=precursor_flows,
        )

        # Adjust for parasitic losses
        effective_thickness_nm = thickness_nm * (1.0 - parasitic_fraction)

        # Generate time-series telemetry
        telemetry = self._generate_telemetry(
            duration_s=duration_s,
            temperature_c=temperature_c,
            precursor_flows=precursor_flows,
            growth_rate_nm_min=growth_rate_nm_min,
        )

        return {
            "thickness_nm": effective_thickness_nm,
            "thickness_std_nm": effective_thickness_nm * (1.0 - uniformity_pct / 100.0) / 3.0,
            "uniformity_pct": uniformity_pct,
            "growth_rate_nm_min": growth_rate_nm_min,
            "composition": composition,
            "v_iii_ratio": actual_v_iii,
            "parasitic_fraction": parasitic_fraction,
            "temperature_c": temperature_c,
            "pressure_pa": self.reactor.pressure_pa,
            "telemetry": telemetry,
        }

    def _calculate_v_iii_ratio(self, precursor_flows: Dict[str, float]) -> float:
        """Calculate V/III ratio from precursor flows"""
        group_iii_flow = 0.0
        group_v_flow = 0.0

        for precursor_name, flow_sccm in precursor_flows.items():
            if precursor_name not in MOCVD_PRECURSORS:
                continue

            precursor = MOCVD_PRECURSORS[precursor_name]

            # Group III: Ga, In, Al
            if any(metal in precursor.formula for metal in ["Ga", "In", "Al"]):
                group_iii_flow += flow_sccm
            # Group V: As, P, N
            elif any(elem in precursor.formula for elem in ["As", "P", "N"]):
                group_v_flow += flow_sccm

        if group_iii_flow == 0:
            return 0.0

        return group_v_flow / group_iii_flow

    def _calculate_growth_rate(
        self,
        precursor_flows: Dict[str, float],
        temperature_c: float,
        v_iii_ratio: float,
    ) -> float:
        """Calculate growth rate in nm/min"""

        # Sum group III precursor contributions
        total_group_iii_incorporation = 0.0

        for precursor_name, flow_sccm in precursor_flows.items():
            if precursor_name not in MOCVD_PRECURSORS:
                continue

            precursor = MOCVD_PRECURSORS[precursor_name]

            # Only consider group III for growth rate
            if not any(metal in precursor.formula for metal in ["Ga", "In", "Al"]):
                continue

            # Calculate decomposition rate (Arrhenius)
            decomp_rate = calculate_arrhenius_rate(
                temperature_k=temperature_c + 273.15,
                activation_energy_kj_mol=precursor.decomposition_energy_kj_mol,
                pre_exponential=1e13,  # s^-1
            )

            # Sticking coefficient and incorporation efficiency
            incorporation_eff = precursor.sticking_coefficient

            # V/III ratio effect on incorporation (optimal around 100-1000)
            if v_iii_ratio < 10:
                incorporation_eff *= (v_iii_ratio / 10.0)  # V-poor reduces incorporation
            elif v_iii_ratio > 5000:
                incorporation_eff *= 0.9  # Excessive V can reduce quality

            # Convert flow to molar flow
            # sccm -> mol/min: flow_sccm * (1 L / 1000 cm³) * (1 mol / 22.4 L)
            molar_flow_mol_min = flow_sccm / 22400.0

            # Atoms incorporated per minute
            atoms_per_min = (
                molar_flow_mol_min
                * 6.022e23  # Avogadro's number
                * (precursor.metal_content_pct / 100.0)
                * incorporation_eff
                * min(decomp_rate / 1e12, 1.0)  # Normalized decomposition
            )

            total_group_iii_incorporation += atoms_per_min

        # Calculate thickness from atom incorporation
        # Assuming GaN-like density: ~8.9e22 atoms/cm³
        atoms_per_cm3 = 8.9e22
        wafer_area_cm2 = np.pi * (self.reactor.wafer_diameter_mm / 20.0) ** 2

        # nm/min = (atoms/min) / (atoms/cm³) / (area cm²) * 1e7 (cm to nm)
        growth_rate_nm_min = (
            total_group_iii_incorporation / atoms_per_cm3 / wafer_area_cm2 * 1e7
        )

        # Temperature regime corrections
        if temperature_c < 500:
            # Surface kinetics limited (low temp)
            growth_rate_nm_min *= (temperature_c / 500.0) ** 2
        elif temperature_c > 800:
            # Desorption losses (high temp)
            desorption_factor = math.exp(-(temperature_c - 800) / 100.0)
            growth_rate_nm_min *= desorption_factor

        return max(0.0, growth_rate_nm_min)

    def _calculate_composition(
        self,
        precursor_flows: Dict[str, float],
        temperature_c: float,
    ) -> Dict[str, float]:
        """Calculate alloy composition (e.g., InGaN: x% In, (1-x)% Ga)"""

        composition = {}

        # Calculate molar flows for each metal
        metal_flows = {}

        for precursor_name, flow_sccm in precursor_flows.items():
            if precursor_name not in MOCVD_PRECURSORS:
                continue

            precursor = MOCVD_PRECURSORS[precursor_name]

            # Extract metal from formula
            for metal in ["Ga", "In", "Al", "As", "P", "N"]:
                if metal in precursor.formula:
                    if metal not in metal_flows:
                        metal_flows[metal] = 0.0

                    # Molar flow weighted by incorporation
                    molar_flow = flow_sccm / 22400.0
                    incorporation = precursor.sticking_coefficient

                    metal_flows[metal] += molar_flow * incorporation

        # Normalize to get fractions
        total_group_iii = sum(
            flow for metal, flow in metal_flows.items() if metal in ["Ga", "In", "Al"]
        )

        if total_group_iii > 0:
            for metal in ["Ga", "In", "Al"]:
                if metal in metal_flows:
                    composition[metal] = metal_flows[metal] / total_group_iii

        return composition

    def _calculate_uniformity(
        self,
        temperature_c: float,
        rotation_rpm: float,
        reactor_type: str,
    ) -> float:
        """Calculate thickness uniformity percentage"""

        # Base uniformity depends on reactor type
        base_uniformity = {
            "horizontal": 85.0,
            "vertical": 88.0,
            "rotating_disk": 95.0,
            "close_coupled_showerhead": 97.0,
        }.get(reactor_type, 85.0)

        # Rotation improves uniformity
        if rotation_rpm > 0:
            rotation_improvement = min(rotation_rpm / 100.0 * 5.0, 8.0)
            base_uniformity += rotation_improvement

        # Temperature uniformity effect (assume ±2C variation)
        temp_variation = 2.0
        temp_effect = 1.0 - (temp_variation / temperature_c) * 2.0

        uniformity = base_uniformity * temp_effect

        return min(99.0, max(70.0, uniformity))

    def _calculate_parasitic_deposition(
        self,
        temperature_c: float,
        pressure_pa: float,
        precursor_flows: Dict[str, float],
    ) -> float:
        """Calculate fraction of precursor lost to parasitic deposition"""

        # Parasitic deposition increases with:
        # 1. Higher temperature (more gas-phase reactions)
        # 2. Higher pressure (more collisions)
        # 3. Higher precursor concentrations

        # Temperature factor (increases exponentially above 600C)
        if temperature_c > 600:
            temp_factor = (temperature_c - 600) / 200.0
        else:
            temp_factor = 0.0

        # Pressure factor (normalized to 100 Torr = 13332 Pa)
        pressure_torr = pressure_pa / 133.32
        pressure_factor = pressure_torr / 100.0

        # Precursor concentration factor
        total_flow = sum(precursor_flows.values())
        conc_factor = total_flow / 100.0  # Normalized to 100 sccm

        # Combined parasitic fraction
        parasitic = (
            0.02  # Base 2% loss
            + temp_factor * 0.15
            + pressure_factor * 0.05
            + conc_factor * 0.03
        )

        return min(0.5, max(0.0, parasitic))  # Cap at 50%

    def _generate_telemetry(
        self,
        duration_s: float,
        temperature_c: float,
        precursor_flows: Dict[str, float],
        growth_rate_nm_min: float,
    ) -> List[Dict[str, Any]]:
        """Generate time-series telemetry data"""

        telemetry = []
        num_steps = int(duration_s / self.time_step_s)

        start_time = datetime.utcnow()

        for i in range(num_steps):
            timestamp = start_time + timedelta(seconds=i * self.time_step_s)

            # Add realistic noise
            temp_noise = np.random.normal(0, 0.5)
            pressure_noise = np.random.normal(0, self.reactor.pressure_pa * 0.01)

            telemetry.append({
                "timestamp": timestamp.isoformat(),
                "temperature_c": temperature_c + temp_noise,
                "pressure_pa": self.reactor.pressure_pa + pressure_noise,
                "precursor_flows": {
                    name: flow + np.random.normal(0, flow * 0.01)
                    for name, flow in precursor_flows.items()
                },
                "rotation_rpm": self.reactor.susceptor_rotation_rpm,
                "growth_rate_nm_min": growth_rate_nm_min + np.random.normal(0, growth_rate_nm_min * 0.05),
                "cumulative_thickness_nm": (i * self.time_step_s / 60.0) * growth_rate_nm_min,
            })

        return telemetry


def simulate_gan_growth(
    temperature_c: float = 1050.0,
    tmga_flow_sccm: float = 50.0,
    nh3_flow_sccm: float = 2000.0,
    duration_min: float = 60.0,
    rotation_rpm: float = 1000.0,
    pressure_torr: float = 200.0,
) -> Dict[str, Any]:
    """
    Convenience function for GaN growth simulation

    Typical GaN MOCVD conditions:
    - Temperature: 1000-1100°C
    - V/III ratio: 1000-5000
    - Pressure: 100-300 Torr
    - Growth rate: 1-5 µm/hr
    """

    reactor = MOCVDReactorConfig(
        reactor_type="rotating_disk",
        pressure_pa=pressure_torr * 133.32,
        susceptor_rotation_rpm=rotation_rpm,
        carrier_gas="H2",
        total_flow_slm=10.0,
        susceptor_diameter_mm=200.0,
        wafer_diameter_mm=100.0,
    )

    simulator = MOCVDSimulator(reactor)

    precursor_flows = {
        "TMGa": tmga_flow_sccm,
        "NH3": nh3_flow_sccm,
    }

    return simulator.simulate_growth(
        precursor_flows=precursor_flows,
        temperature_c=temperature_c,
        duration_s=duration_min * 60.0,
    )
