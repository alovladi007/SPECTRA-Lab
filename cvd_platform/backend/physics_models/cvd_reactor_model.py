"""
CVD Reactor Physics Model and Digital Twin
Implements multi-physics simulation of CVD processes:
- Gas flow (Navier-Stokes equations)
- Mass transport and diffusion
- Reaction kinetics (Arrhenius laws)
- Heat transfer (conduction, convection, radiation)
- Deposition rate prediction
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging
from scipy.integrate import odeint
from scipy.sparse import diags, linalg as sp_linalg

logger = logging.getLogger(__name__)


class ReactorGeometry(Enum):
    """CVD Reactor Types"""
    HORIZONTAL_BOAT = "horizontal_boat"
    VERTICAL_BOAT = "vertical_boat"
    SHOWERHEAD = "showerhead"
    SINGLE_WAFER = "single_wafer"
    BATCH = "batch"


@dataclass
class ReactorDimensions:
    """Physical dimensions of reactor"""
    length: float  # m
    diameter: float  # m
    susceptor_diameter: float  # m
    wafer_diameter: float  # m (typically 0.2, 0.3 m for 200mm, 300mm)
    gap_height: float  # m (wafer to showerhead/top)
    heater_zones: int
    inlet_diameter: float  # m
    outlet_diameter: float  # m


@dataclass
class ProcessConditions:
    """CVD Process Conditions"""
    temperature: float  # K
    pressure: float  # Pa
    gas_flows: Dict[str, float]  # species: flow rate (sccm)
    rotation_speed: float  # rpm
    deposition_time: float  # seconds
    susceptor_temp: float  # K
    wall_temp: float  # K


@dataclass
class GasProperties:
    """Properties of gas species"""
    name: str
    molecular_weight: float  # kg/mol
    diffusion_coeff: float  # m²/s
    viscosity: float  # Pa·s
    thermal_conductivity: float  # W/(m·K)
    heat_capacity: float  # J/(kg·K)


@dataclass
class ReactionParameters:
    """Chemical reaction kinetics parameters"""
    reaction: str
    pre_exponential: float  # k₀
    activation_energy: float  # J/mol (E_a)
    reaction_order: float  # n
    products: List[str]
    stoichiometry: Dict[str, float]


class CVDReactorModel:
    """
    Physics-based CVD reactor model implementing:
    1. Gas flow dynamics (Navier-Stokes)
    2. Species mass transport
    3. Surface reaction kinetics
    4. Heat transfer
    5. Deposition rate calculation
    """

    def __init__(self, geometry: ReactorGeometry, dimensions: ReactorDimensions):
        self.geometry = geometry
        self.dimensions = dimensions

        # Physical constants
        self.R_GAS = 8.314  # J/(mol·K) - Universal gas constant
        self.BOLTZMANN = 1.380649e-23  # J/K
        self.AVOGADRO = 6.02214076e23  # mol⁻¹
        self.STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)

        # Simulation mesh
        self.nr = 50  # Radial points
        self.nz = 100  # Axial/vertical points
        self.r_grid = np.linspace(0, dimensions.susceptor_diameter/2, self.nr)
        self.z_grid = np.linspace(0, dimensions.gap_height, self.nz)

        # Initialize fields
        self.velocity_field = None
        self.temperature_field = None
        self.concentration_fields = {}
        self.deposition_rate_field = None

        # Gas properties database
        self.gas_database = self._initialize_gas_database()

        # Reaction database
        self.reactions = self._initialize_reactions()

        logger.info(f"Initialized CVD reactor model: {geometry.value}")

    def _initialize_gas_database(self) -> Dict[str, GasProperties]:
        """Initialize thermophysical properties of common CVD gases"""
        return {
            "SiH4": GasProperties(  # Silane
                name="SiH4",
                molecular_weight=0.03209,  # kg/mol
                diffusion_coeff=1.0e-5,  # m²/s at STP
                viscosity=1.0e-5,  # Pa·s
                thermal_conductivity=0.025,  # W/(m·K)
                heat_capacity=1400.0  # J/(kg·K)
            ),
            "N2": GasProperties(  # Nitrogen
                name="N2",
                molecular_weight=0.02801,
                diffusion_coeff=2.0e-5,
                viscosity=1.76e-5,
                thermal_conductivity=0.026,
                heat_capacity=1040.0
            ),
            "H2": GasProperties(  # Hydrogen
                name="H2",
                molecular_weight=0.00201,
                diffusion_coeff=7.5e-5,
                viscosity=8.9e-6,
                thermal_conductivity=0.18,
                heat_capacity=14300.0
            ),
            "NH3": GasProperties(  # Ammonia
                name="NH3",
                molecular_weight=0.01703,
                diffusion_coeff=2.8e-5,
                viscosity=1.0e-5,
                thermal_conductivity=0.025,
                heat_capacity=2060.0
            ),
            "Ar": GasProperties(  # Argon
                name="Ar",
                molecular_weight=0.03995,
                diffusion_coeff=1.8e-5,
                viscosity=2.2e-5,
                thermal_conductivity=0.018,
                heat_capacity=520.0
            )
        }

    def _initialize_reactions(self) -> List[ReactionParameters]:
        """Initialize chemical reaction kinetics"""
        return [
            # Silicon deposition from silane
            # SiH4(g) → Si(s) + 2H2(g)
            ReactionParameters(
                reaction="SiH4 -> Si + 2H2",
                pre_exponential=1.0e8,  # 1/s
                activation_energy=1.7e5,  # J/mol (~170 kJ/mol)
                reaction_order=1.0,
                products=["Si", "H2"],
                stoichiometry={"SiH4": -1, "Si": 1, "H2": 2}
            ),
            # Silicon nitride deposition
            # 3SiH4 + 4NH3 → Si3N4 + 12H2
            ReactionParameters(
                reaction="3SiH4 + 4NH3 -> Si3N4 + 12H2",
                pre_exponential=5.0e7,
                activation_energy=1.5e5,
                reaction_order=1.5,
                products=["Si3N4", "H2"],
                stoichiometry={"SiH4": -3, "NH3": -4, "Si3N4": 1, "H2": 12}
            )
        ]

    def compute_reynolds_number(self, conditions: ProcessConditions,
                                char_length: float) -> float:
        """
        Calculate Reynolds number for flow regime determination.
        Re = ρVL/μ
        """
        # Get mixture properties
        mixture_props = self._calculate_mixture_properties(conditions)

        # Estimate velocity from flow rate
        total_flow_sccm = sum(conditions.gas_flows.values())
        total_flow_m3s = total_flow_sccm * 1.66667e-8  # Convert sccm to m³/s

        # Volumetric flow rate at process conditions
        flow_actual = total_flow_m3s * (conditions.temperature / 273.15) * (101325 / conditions.pressure)

        # Average velocity
        area = np.pi * (self.dimensions.inlet_diameter / 2) ** 2
        velocity = flow_actual / area

        # Reynolds number
        Re = (mixture_props["density"] * velocity * char_length /
              mixture_props["viscosity"])

        return Re

    def _calculate_mixture_properties(self, conditions: ProcessConditions) -> Dict[str, float]:
        """Calculate mixture-averaged thermophysical properties"""
        total_flow = sum(conditions.gas_flows.values())

        # Mole fractions
        mole_fractions = {
            species: flow / total_flow
            for species, flow in conditions.gas_flows.items()
        }

        # Mixture molecular weight
        MW_mix = sum(mole_fractions.get(species, 0) * self.gas_database[species].molecular_weight
                     for species in self.gas_database.keys())

        # Mixture density (ideal gas law: ρ = PM/(RT))
        density = (conditions.pressure * MW_mix) / (self.R_GAS * conditions.temperature)

        # Mixture viscosity (Wilke's mixing rule - simplified)
        viscosity = sum(mole_fractions.get(species, 0) * self.gas_database[species].viscosity
                       for species in self.gas_database.keys())

        # Mixture thermal conductivity
        thermal_cond = sum(mole_fractions.get(species, 0) * self.gas_database[species].thermal_conductivity
                          for species in self.gas_database.keys())

        # Mixture heat capacity
        heat_capacity = sum(mole_fractions.get(species, 0) * self.gas_database[species].heat_capacity
                           for species in self.gas_database.keys())

        return {
            "density": density,
            "viscosity": viscosity,
            "thermal_conductivity": thermal_cond,
            "heat_capacity": heat_capacity,
            "molecular_weight": MW_mix
        }

    def solve_velocity_field(self, conditions: ProcessConditions) -> np.ndarray:
        """
        Solve simplified Navier-Stokes equations for velocity field.
        For CVD, typically low Re (laminar flow).

        Simplified 2D axisymmetric flow:
        ∂u/∂r + u/r + ∂w/∂z = 0  (continuity)
        ρ(u∂u/∂r + w∂u/∂z) = -∂P/∂r + μ(∂²u/∂r² + ∂²u/∂z²)  (momentum-r)
        ρ(u∂w/∂r + w∂w/∂z) = -∂P/∂z + μ(∂²w/∂r² + ∂²w/∂z²)  (momentum-z)
        """
        logger.info("Solving velocity field...")

        mixture_props = self._calculate_mixture_properties(conditions)
        mu = mixture_props["viscosity"]
        rho = mixture_props["density"]

        # For simplification, use plug flow approximation with boundary layer
        # This is a simplified model - full CFD would use FEM/FVM solvers

        # Inlet velocity
        total_flow_sccm = sum(conditions.gas_flows.values())
        total_flow_m3s = total_flow_sccm * 1.66667e-8
        flow_actual = total_flow_m3s * (conditions.temperature / 273.15) * (101325 / conditions.pressure)

        inlet_area = np.pi * (self.dimensions.inlet_diameter / 2) ** 2
        v_inlet = flow_actual / inlet_area

        # Create velocity field (r, z, [vr, vz])
        vr = np.zeros((self.nr, self.nz))
        vz = np.zeros((self.nr, self.nz))

        # Simplified parabolic profile for laminar flow
        for j in range(self.nz):
            z_pos = self.z_grid[j]
            v_center = v_inlet * (1 - z_pos / self.dimensions.gap_height * 0.5)

            for i in range(self.nr):
                r_pos = self.r_grid[i]
                r_norm = r_pos / (self.dimensions.susceptor_diameter / 2)

                # Parabolic profile: v(r) = v_max(1 - r²/R²)
                vz[i, j] = v_center * (1 - r_norm ** 2)

                # Radial velocity component (small, for mass conservation)
                if j > 0:
                    vr[i, j] = -r_pos * (vz[i, j] - vz[i, j-1]) / (self.z_grid[j] - self.z_grid[j-1])

        self.velocity_field = np.stack([vr, vz], axis=-1)

        logger.info(f"Velocity field computed. Max velocity: {np.max(vz):.4f} m/s")
        return self.velocity_field

    def solve_temperature_field(self, conditions: ProcessConditions) -> np.ndarray:
        """
        Solve heat transfer equation:
        ρCp(v·∇T) = k∇²T + Q

        Includes:
        - Convection from gas flow
        - Conduction in gas and solid
        - Radiation from heated susceptor
        """
        logger.info("Solving temperature field...")

        mixture_props = self._calculate_mixture_properties(conditions)
        k_thermal = mixture_props["thermal_conductivity"]
        Cp = mixture_props["heat_capacity"]
        rho = mixture_props["density"]

        # Initialize temperature field
        T = np.zeros((self.nr, self.nz))

        # Boundary conditions
        T[:, 0] = conditions.susceptor_temp  # Bottom (susceptor)
        T[:, -1] = conditions.wall_temp  # Top (chamber wall)
        T[0, :] = conditions.temperature  # Centerline (symmetry)
        T[-1, :] = conditions.wall_temp  # Outer radius

        # Solve using finite difference (simplified)
        # Full solution would use FEM with iterative solver

        dr = self.r_grid[1] - self.r_grid[0]
        dz = self.z_grid[1] - self.z_grid[0]

        alpha = k_thermal / (rho * Cp)  # Thermal diffusivity

        # Iterative solver (Gauss-Seidel)
        max_iterations = 1000
        tolerance = 1e-4

        for iteration in range(max_iterations):
            T_old = T.copy()

            for i in range(1, self.nr - 1):
                for j in range(1, self.nz - 1):
                    r = self.r_grid[i]

                    # Laplacian in cylindrical coordinates
                    d2T_dr2 = (T[i+1, j] - 2*T[i, j] + T[i-1, j]) / dr**2
                    dT_dr = (T[i+1, j] - T[i-1, j]) / (2*dr)
                    d2T_dz2 = (T[i, j+1] - 2*T[i, j] + T[i, j-1]) / dz**2

                    laplacian = d2T_dr2 + dT_dr/r + d2T_dz2

                    # Update temperature
                    T[i, j] = T_old[i, j] + 0.25 * alpha * laplacian

            # Check convergence
            error = np.max(np.abs(T - T_old))
            if error < tolerance:
                logger.info(f"Temperature field converged in {iteration} iterations")
                break

        self.temperature_field = T

        logger.info(f"Temperature field computed. Range: {np.min(T):.1f} - {np.max(T):.1f} K")
        return T

    def solve_species_transport(self, species: str, conditions: ProcessConditions) -> np.ndarray:
        """
        Solve species mass transport equation:
        ∂C/∂t + v·∇C = D∇²C - R

        where:
        C = concentration (mol/m³)
        v = velocity field
        D = diffusion coefficient
        R = reaction rate
        """
        logger.info(f"Solving {species} transport...")

        if species not in self.gas_database:
            raise ValueError(f"Unknown species: {species}")

        gas_props = self.gas_database[species]
        D = gas_props.diffusion_coeff

        # Initialize concentration field
        C = np.zeros((self.nr, self.nz))

        # Inlet concentration (from mole fraction and ideal gas law)
        if species in conditions.gas_flows:
            total_flow = sum(conditions.gas_flows.values())
            mole_fraction = conditions.gas_flows[species] / total_flow
            # C = n/V = P*y/(RT) where y is mole fraction
            C_inlet = (conditions.pressure * mole_fraction) / (self.R_GAS * conditions.temperature)
        else:
            C_inlet = 0.0

        # Boundary conditions
        C[:, -1] = C_inlet  # Inlet (top)
        C[-1, :] = 0.0  # Walls (consumption or inert)

        # Solve transport equation
        dr = self.r_grid[1] - self.r_grid[0]
        dz = self.z_grid[1] - self.z_grid[0]

        max_iterations = 1000
        tolerance = 1e-6

        for iteration in range(max_iterations):
            C_old = C.copy()

            for i in range(1, self.nr - 1):
                for j in range(1, self.nz - 1):
                    r = self.r_grid[i]

                    # Diffusion terms (Laplacian)
                    d2C_dr2 = (C[i+1, j] - 2*C[i, j] + C[i-1, j]) / dr**2
                    dC_dr = (C[i+1, j] - C[i-1, j]) / (2*dr)
                    d2C_dz2 = (C[i, j+1] - 2*C[i, j] + C[i, j-1]) / dz**2

                    laplacian = d2C_dr2 + dC_dr/r + d2C_dz2

                    # Convection terms (if velocity field available)
                    if self.velocity_field is not None:
                        vr = self.velocity_field[i, j, 0]
                        vz = self.velocity_field[i, j, 1]
                        dC_dr_upwind = (C[i, j] - C[i-1, j]) / dr if vr > 0 else (C[i+1, j] - C[i, j]) / dr
                        dC_dz_upwind = (C[i, j] - C[i, j-1]) / dz if vz > 0 else (C[i, j+1] - C[i, j]) / dz
                        convection = -(vr * dC_dr_upwind + vz * dC_dz_upwind)
                    else:
                        convection = 0.0

                    # Reaction term (at surface only)
                    if j == 0:  # Wafer surface
                        T_surface = self.temperature_field[i, j] if self.temperature_field is not None else conditions.temperature
                        reaction_rate = self.calculate_reaction_rate(species, C[i, j], T_surface, conditions)
                    else:
                        reaction_rate = 0.0

                    # Update concentration
                    C[i, j] = C_old[i, j] + 0.2 * (D * laplacian + convection - reaction_rate)

                    # Ensure non-negative concentration
                    C[i, j] = max(C[i, j], 0.0)

            # Check convergence
            error = np.max(np.abs(C - C_old))
            if error < tolerance:
                logger.info(f"{species} concentration field converged in {iteration} iterations")
                break

        self.concentration_fields[species] = C

        logger.info(f"{species} transport computed. Max concentration: {np.max(C):.6f} mol/m³")
        return C

    def calculate_reaction_rate(self, species: str, concentration: float,
                               temperature: float, conditions: ProcessConditions) -> float:
        """
        Calculate surface reaction rate using Arrhenius equation:
        R = k₀ * C^n * exp(-Ea/RT)

        where:
        k₀ = pre-exponential factor
        C = concentration
        n = reaction order
        Ea = activation energy
        R = gas constant
        T = temperature
        """
        # Find reaction involving this species
        for reaction in self.reactions:
            if species in reaction.stoichiometry:
                k0 = reaction.pre_exponential
                Ea = reaction.activation_energy
                n = reaction.reaction_order

                # Arrhenius rate constant
                k = k0 * np.exp(-Ea / (self.R_GAS * temperature))

                # Reaction rate
                rate = k * (concentration ** n)

                return rate * abs(reaction.stoichiometry[species])

        return 0.0

    def calculate_deposition_rate(self, conditions: ProcessConditions) -> np.ndarray:
        """
        Calculate film deposition rate across wafer surface.

        Growth Rate = (MW/ρ_film) * k_s * C_surface

        where:
        MW = molecular weight of film
        ρ_film = film density
        k_s = surface reaction rate constant
        C_surface = precursor concentration at surface
        """
        logger.info("Calculating deposition rate...")

        # Get surface concentration of reactant (e.g., SiH4 for Si deposition)
        if "SiH4" in self.concentration_fields:
            C_surface = self.concentration_fields["SiH4"][:, 0]  # Surface concentrations
        else:
            logger.warning("SiH4 concentration not available, using estimate")
            C_surface = np.ones(self.nr) * 1.0  # mol/m³

        # Get surface temperature
        if self.temperature_field is not None:
            T_surface = self.temperature_field[:, 0]
        else:
            T_surface = np.ones(self.nr) * conditions.temperature

        # Calculate deposition rate at each radial position
        deposition_rate = np.zeros(self.nr)

        # Film properties (for Si)
        MW_Si = 0.02809  # kg/mol
        rho_Si = 2330  # kg/m³

        for i in range(self.nr):
            # Reaction rate (mol/(m²·s))
            reaction_rate = self.calculate_reaction_rate("SiH4", C_surface[i], T_surface[i], conditions)

            # Convert to growth rate (m/s)
            growth_rate = (MW_Si / rho_Si) * reaction_rate

            # Convert to nm/s
            deposition_rate[i] = growth_rate * 1e9

        self.deposition_rate_field = deposition_rate

        logger.info(f"Deposition rate: {np.mean(deposition_rate):.4f} ± {np.std(deposition_rate):.4f} nm/s")
        logger.info(f"Uniformity: {(np.std(deposition_rate)/np.mean(deposition_rate)*100):.2f}%")

        return deposition_rate

    def calculate_film_thickness(self, conditions: ProcessConditions) -> Dict[str, Any]:
        """
        Calculate final film thickness distribution after deposition.
        """
        if self.deposition_rate_field is None:
            self.calculate_deposition_rate(conditions)

        # Thickness = rate * time
        thickness = self.deposition_rate_field * conditions.deposition_time  # nm

        mean_thickness = np.mean(thickness)
        std_thickness = np.std(thickness)
        uniformity = (std_thickness / mean_thickness) * 100 if mean_thickness > 0 else 0

        # Create radial profile
        thickness_profile = [
            {"radius_mm": self.r_grid[i] * 1000, "thickness_nm": thickness[i]}
            for i in range(self.nr)
        ]

        result = {
            "mean_thickness": mean_thickness,
            "std_dev": std_thickness,
            "uniformity_percent": uniformity,
            "min_thickness": np.min(thickness),
            "max_thickness": np.max(thickness),
            "center_thickness": thickness[0],
            "edge_thickness": thickness[-1],
            "radial_profile": thickness_profile
        }

        logger.info(f"Film thickness: {mean_thickness:.2f} ± {std_thickness:.2f} nm")
        logger.info(f"Uniformity: {uniformity:.2f}%")

        return result

    def run_full_simulation(self, conditions: ProcessConditions) -> Dict[str, Any]:
        """
        Run complete CVD simulation:
        1. Velocity field
        2. Temperature field
        3. Species transport
        4. Deposition rate
        5. Film thickness
        """
        logger.info("=== Starting CVD Reactor Simulation ===")
        logger.info(f"Process conditions: T={conditions.temperature}K, P={conditions.pressure}Pa")

        # Step 1: Solve fluid flow
        self.solve_velocity_field(conditions)

        # Step 2: Solve heat transfer
        self.solve_temperature_field(conditions)

        # Step 3: Solve species transport for all reactants
        for species in conditions.gas_flows.keys():
            if species in self.gas_database:
                self.solve_species_transport(species, conditions)

        # Step 4: Calculate deposition rate
        self.calculate_deposition_rate(conditions)

        # Step 5: Calculate final thickness
        thickness_result = self.calculate_film_thickness(conditions)

        logger.info("=== CVD Simulation Complete ===")

        return {
            "thickness": thickness_result,
            "velocity_field": self.velocity_field,
            "temperature_field": self.temperature_field,
            "concentration_fields": self.concentration_fields,
            "deposition_rate": self.deposition_rate_field,
            "process_conditions": conditions
        }
