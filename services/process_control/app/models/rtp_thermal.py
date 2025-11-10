"""RTP thermal plant models.

Provides thermal dynamics modeling, emissivity compensation, zone coupling,
and sensor lag models for RTP systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


# ============================================================================
# Constants and Material Properties
# ============================================================================

# Physical constants
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)
BOLTZMANN_K = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602e-19  # C

# Silicon thermal properties (temperature-dependent)
class SiliconThermalProperties:
    """Temperature-dependent thermal properties of silicon."""

    @staticmethod
    def density(temp_C: float) -> float:
        """Density (kg/m³)."""
        return 2329  # Approximately constant

    @staticmethod
    def specific_heat(temp_C: float) -> float:
        """Specific heat capacity (J/kg·K)."""
        # Temperature-dependent (increases with T)
        T_K = temp_C + 273.15
        if T_K < 300:
            return 700
        elif T_K < 800:
            # Linear interpolation
            return 700 + (T_K - 300) * (900 - 700) / (800 - 300)
        else:
            return 900

    @staticmethod
    def thermal_conductivity(temp_C: float) -> float:
        """Thermal conductivity (W/m·K)."""
        # Decreases with temperature
        T_K = temp_C + 273.15
        return 150 / (T_K / 300)**1.3

    @staticmethod
    def emissivity(temp_C: float, surface_condition: str = "polished") -> float:
        """Emissivity (0-1)."""
        # Depends on surface condition and temperature
        if surface_condition == "polished":
            # Polished silicon
            return 0.60 + 0.0001 * temp_C  # Increases slightly with T
        elif surface_condition == "oxidized":
            # Oxidized silicon
            return 0.85 + 0.0001 * temp_C
        elif surface_condition == "rough":
            # Rough/textured
            return 0.70 + 0.0001 * temp_C
        else:
            return 0.65  # Default


# Wafer geometry
WAFER_DIAMETER_M = 0.3  # 300mm wafer
WAFER_THICKNESS_M = 0.775e-3  # 775 μm
WAFER_AREA_M2 = np.pi * (WAFER_DIAMETER_M / 2) ** 2
WAFER_VOLUME_M3 = WAFER_AREA_M2 * WAFER_THICKNESS_M


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ThermalZoneModel:
    """Thermal model for a single lamp zone."""
    zone_id: int
    lamp_power_max_W: float  # Maximum lamp power
    time_constant_s: float  # Thermal time constant
    position_m: float  # Radial position from center
    coupling_coefficients: Dict[int, float]  # Coupling to other zones


@dataclass
class EmissivityModel:
    """Emissivity model and compensation."""
    base_emissivity: float  # Base emissivity (0-1)
    drift_rate_per_s: float  # Emissivity drift rate
    temperature_coefficient: float  # dε/dT
    oxidation_effect: float  # Oxide thickness effect
    current_emissivity: float  # Current estimated emissivity


@dataclass
class SensorModel:
    """Temperature sensor model (pyrometer or thermocouple)."""
    sensor_type: str  # "pyrometer" or "thermocouple"
    time_constant_s: float  # Response time
    measurement_noise_C: float  # Measurement noise (std dev)
    bias_C: float  # Systematic bias
    is_emissivity_dependent: bool  # Pyrometer = True, TC = False


@dataclass
class ThermalState:
    """Complete thermal state of the RTP system."""
    timestamp_s: float
    zone_temperatures_C: np.ndarray  # Temperature of each zone
    wafer_temperature_C: float  # Average wafer temperature
    pyrometer_reading_C: float  # Pyrometer reading
    thermocouple_reading_C: float  # Thermocouple reading
    lamp_powers_pct: np.ndarray  # Lamp power percentages
    heat_flux_W: np.ndarray  # Heat flux per zone
    emissivity: float  # Current emissivity


# ============================================================================
# Thermal Plant Model
# ============================================================================

class RTPThermalPlant:
    """Physics-based thermal plant model for RTP."""

    def __init__(
        self,
        num_zones: int = 4,
        max_lamp_power_W: float = 10000.0,
        wafer_diameter_m: float = 0.3
    ):
        """
        Initialize RTP thermal plant.

        Args:
            num_zones: Number of lamp zones
            max_lamp_power_W: Maximum total lamp power
            wafer_diameter_m: Wafer diameter
        """
        self.num_zones = num_zones
        self.max_lamp_power_W = max_lamp_power_W
        self.wafer_diameter_m = wafer_diameter_m
        self.wafer_area_m2 = np.pi * (wafer_diameter_m / 2) ** 2

        # Create zone models
        self.zones = self._create_zone_models()

        # Current state
        self.state = ThermalState(
            timestamp_s=0.0,
            zone_temperatures_C=np.ones(num_zones) * 25.0,
            wafer_temperature_C=25.0,
            pyrometer_reading_C=25.0,
            thermocouple_reading_C=25.0,
            lamp_powers_pct=np.zeros(num_zones),
            heat_flux_W=np.zeros(num_zones),
            emissivity=0.65
        )

        # Emissivity model
        self.emissivity_model = EmissivityModel(
            base_emissivity=0.65,
            drift_rate_per_s=0.0,
            temperature_coefficient=0.0001,
            oxidation_effect=0.0,
            current_emissivity=0.65
        )

        # Sensor models
        self.pyrometer = SensorModel(
            sensor_type="pyrometer",
            time_constant_s=0.1,
            measurement_noise_C=2.0,
            bias_C=0.0,
            is_emissivity_dependent=True
        )

        self.thermocouple = SensorModel(
            sensor_type="thermocouple",
            time_constant_s=0.5,
            measurement_noise_C=1.0,
            bias_C=0.0,
            is_emissivity_dependent=False
        )

        # Ambient conditions
        self.ambient_temp_C = 25.0
        self.chamber_pressure_Pa = 101325.0  # Atmospheric
        self.gas_thermal_conductivity = 0.026  # W/m·K for N2

    def _create_zone_models(self) -> List[ThermalZoneModel]:
        """Create thermal zone models."""
        zones = []
        power_per_zone = self.max_lamp_power_W / self.num_zones

        for i in range(self.num_zones):
            # Position zones radially
            position_m = (i / self.num_zones) * (self.wafer_diameter_m / 2)

            # Time constants vary by zone (edge zones respond faster)
            if i == 0 or i == self.num_zones - 1:
                time_constant = 2.0  # Edge zones
            else:
                time_constant = 3.0  # Center zones

            # Coupling coefficients (adjacent zones)
            coupling = {}
            coupling_strength = 0.1  # 10% coupling

            if i > 0:
                coupling[i - 1] = coupling_strength
            if i < self.num_zones - 1:
                coupling[i + 1] = coupling_strength

            zone = ThermalZoneModel(
                zone_id=i,
                lamp_power_max_W=power_per_zone,
                time_constant_s=time_constant,
                position_m=position_m,
                coupling_coefficients=coupling
            )
            zones.append(zone)

        return zones

    def update(
        self,
        dt: float,
        lamp_powers_pct: np.ndarray,
        ambient_temp_C: Optional[float] = None,
        gas_flow_sccm: float = 5000.0,
        chamber_pressure_Pa: Optional[float] = None
    ) -> ThermalState:
        """
        Update thermal plant state by one time step.

        Args:
            dt: Time step (seconds)
            lamp_powers_pct: Lamp power for each zone (0-100%)
            ambient_temp_C: Ambient temperature
            gas_flow_sccm: Gas flow rate
            chamber_pressure_Pa: Chamber pressure

        Returns:
            Updated ThermalState
        """
        if ambient_temp_C is not None:
            self.ambient_temp_C = ambient_temp_C
        if chamber_pressure_Pa is not None:
            self.chamber_pressure_Pa = chamber_pressure_Pa

        # Update emissivity (drift over time)
        self.emissivity_model.current_emissivity += (
            self.emissivity_model.drift_rate_per_s * dt
        )
        self.emissivity_model.current_emissivity = np.clip(
            self.emissivity_model.current_emissivity, 0.1, 1.0
        )

        # Calculate heat inputs
        heat_inputs = self._calculate_heat_inputs(lamp_powers_pct)

        # Calculate heat losses (radiation + convection)
        heat_losses = self._calculate_heat_losses(gas_flow_sccm)

        # Calculate zone coupling
        coupling_heat = self._calculate_zone_coupling()

        # Update zone temperatures
        new_zone_temps = np.zeros(self.num_zones)

        for i, zone in enumerate(self.zones):
            # Net heat flux
            net_heat = (
                heat_inputs[i]
                - heat_losses[i]
                + coupling_heat[i]
            )

            # Temperature change (Q = m·c·ΔT)
            zone_mass = (self.wafer_area_m2 / self.num_zones) * WAFER_THICKNESS_M * SiliconThermalProperties.density(25)
            specific_heat = SiliconThermalProperties.specific_heat(self.state.zone_temperatures_C[i])

            dT_dt = net_heat / (zone_mass * specific_heat)

            # First-order lag with time constant
            tau = zone.time_constant_s
            new_zone_temps[i] = self.state.zone_temperatures_C[i] + (dt / tau) * (
                dT_dt * tau
            )

        # Update zone temperatures
        self.state.zone_temperatures_C = new_zone_temps

        # Calculate weighted average wafer temperature
        weights = np.array([0.8, 1.0, 1.0, 0.8][:self.num_zones])
        weights /= weights.sum()
        self.state.wafer_temperature_C = np.sum(weights * new_zone_temps)

        # Update sensor readings with lag
        self._update_sensor_readings(dt)

        # Update state
        self.state.timestamp_s += dt
        self.state.lamp_powers_pct = lamp_powers_pct.copy()
        self.state.heat_flux_W = heat_inputs - heat_losses
        self.state.emissivity = self.emissivity_model.current_emissivity

        return self.state

    def _calculate_heat_inputs(self, lamp_powers_pct: np.ndarray) -> np.ndarray:
        """Calculate heat input from lamps for each zone."""
        heat_inputs = np.zeros(self.num_zones)

        for i, zone in enumerate(self.zones):
            power_fraction = lamp_powers_pct[i] / 100.0
            power_fraction = np.clip(power_fraction, 0.0, 1.0)

            # Lamp power with efficiency (70% heating, 30% losses)
            efficiency = 0.7
            heat_inputs[i] = power_fraction * zone.lamp_power_max_W * efficiency

        return heat_inputs

    def _calculate_heat_losses(self, gas_flow_sccm: float) -> np.ndarray:
        """Calculate radiative and convective heat losses."""
        heat_losses = np.zeros(self.num_zones)

        for i in range(self.num_zones):
            T_zone = self.state.zone_temperatures_C[i] + 273.15  # K
            T_amb = self.ambient_temp_C + 273.15  # K

            zone_area = self.wafer_area_m2 / self.num_zones

            # Radiative loss (Stefan-Boltzmann)
            emissivity = self.emissivity_model.current_emissivity
            radiative_loss = (
                emissivity * STEFAN_BOLTZMANN * zone_area *
                (T_zone**4 - T_amb**4)
            )

            # Convective loss (depends on gas flow and pressure)
            h_conv = self._calculate_convection_coefficient(gas_flow_sccm)
            convective_loss = h_conv * zone_area * (T_zone - T_amb)

            heat_losses[i] = radiative_loss + convective_loss

        return heat_losses

    def _calculate_convection_coefficient(self, gas_flow_sccm: float) -> float:
        """Calculate convective heat transfer coefficient."""
        # Base coefficient for natural convection
        h_base = 10.0  # W/(m²·K)

        # Pressure effect
        pressure_factor = self.chamber_pressure_Pa / 101325.0

        # Flow rate effect (forced convection)
        flow_factor = 1.0 + 0.001 * gas_flow_sccm  # 0.1% per 100 sccm

        h_conv = h_base * pressure_factor * flow_factor

        return h_conv

    def _calculate_zone_coupling(self) -> np.ndarray:
        """Calculate thermal coupling between zones."""
        coupling_heat = np.zeros(self.num_zones)

        for i, zone in enumerate(self.zones):
            for neighbor_id, coeff in zone.coupling_coefficients.items():
                # Heat transfer proportional to temperature difference
                delta_T = self.state.zone_temperatures_C[neighbor_id] - self.state.zone_temperatures_C[i]
                coupling_heat[i] += coeff * 1000.0 * delta_T  # W

        return coupling_heat

    def _update_sensor_readings(self, dt: float):
        """Update sensor readings with lag and noise."""
        # Pyrometer (emissivity-dependent)
        true_temp = self.state.wafer_temperature_C

        # Emissivity error affects pyrometer
        emissivity_error = self.emissivity_model.current_emissivity / 0.65  # Relative to assumed
        apparent_temp = true_temp * (emissivity_error ** 0.25)

        # First-order lag
        tau_pyro = self.pyrometer.time_constant_s
        self.state.pyrometer_reading_C += (dt / tau_pyro) * (
            apparent_temp - self.state.pyrometer_reading_C
        )

        # Add noise
        noise_pyro = np.random.normal(0, self.pyrometer.measurement_noise_C)
        self.state.pyrometer_reading_C += noise_pyro

        # Thermocouple (direct contact, no emissivity dependence)
        tau_tc = self.thermocouple.time_constant_s
        self.state.thermocouple_reading_C += (dt / tau_tc) * (
            true_temp - self.state.thermocouple_reading_C
        )

        # Add noise
        noise_tc = np.random.normal(0, self.thermocouple.measurement_noise_C)
        self.state.thermocouple_reading_C += noise_tc

    def reset(self, initial_temp_C: float = 25.0):
        """Reset thermal plant to initial conditions."""
        self.state = ThermalState(
            timestamp_s=0.0,
            zone_temperatures_C=np.ones(self.num_zones) * initial_temp_C,
            wafer_temperature_C=initial_temp_C,
            pyrometer_reading_C=initial_temp_C,
            thermocouple_reading_C=initial_temp_C,
            lamp_powers_pct=np.zeros(self.num_zones),
            heat_flux_W=np.zeros(self.num_zones),
            emissivity=self.emissivity_model.base_emissivity
        )


# Export
__all__ = [
    "RTPThermalPlant",
    "ThermalZoneModel",
    "EmissivityModel",
    "SensorModel",
    "ThermalState",
    "SiliconThermalProperties",
    "STEFAN_BOLTZMANN",
]
