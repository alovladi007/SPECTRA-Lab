"""RTP Hardware-in-Loop (HIL) Simulator with thermal plant model."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import math

from app.drivers.rtp_driver import (
    RTPDriver, RTPStatus, AmbientGas, TemperatureControlMode,
    RampSegment, GasFlowParameters, LampParameters, EmissivitySettings, TemperatureRecipe
)

logger = logging.getLogger(__name__)


# ============================================================================
# Physical Constants and Models
# ============================================================================

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)

# Silicon properties
SI_DENSITY = 2.33e3  # kg/m³
SI_SPECIFIC_HEAT = 700  # J/(kg·K) at 1000°C
SI_THERMAL_CONDUCTIVITY = 25  # W/(m·K) at 1000°C

# Typical wafer dimensions
WAFER_DIAMETER_M = 0.3  # 300mm
WAFER_THICKNESS_M = 0.775e-3  # 775 μm
WAFER_AREA_M2 = np.pi * (WAFER_DIAMETER_M / 2) ** 2
WAFER_MASS_KG = WAFER_AREA_M2 * WAFER_THICKNESS_M * SI_DENSITY


@dataclass
class ThermalZoneState:
    """State of a single thermal zone."""
    temperature_C: float
    lamp_power_pct: float
    time_constant_s: float


class ThermalPlantModel:
    """Physics-based thermal plant model for RTP."""

    def __init__(
        self,
        num_zones: int = 4,
        random_seed: Optional[int] = None
    ):
        """
        Initialize thermal plant model.

        Args:
            num_zones: Number of lamp zones (typically 3-6)
            random_seed: Random seed for deterministic behavior
        """
        self.num_zones = num_zones
        self.rng = np.random.default_rng(seed=random_seed)

        # Zone states
        self.zones = [
            ThermalZoneState(
                temperature_C=25.0,
                lamp_power_pct=0.0,
                time_constant_s=2.0 + i * 0.5  # Zones have different time constants
            )
            for i in range(num_zones)
        ]

        # Wafer temperature (weighted average of zones)
        self.wafer_temp_C = 25.0
        self.wafer_temp_history: List[float] = [25.0]

        # Pyrometer and thermocouple states
        self.pyrometer_temp_C = 25.0
        self.thermocouple_temp_C = 25.0
        self.pyrometer_lag_s = 0.1  # 100ms response time
        self.thermocouple_lag_s = 0.5  # 500ms response time

        # Emissivity
        self.emissivity = 0.65
        self.emissivity_drift_rate = 0.0  # per second

        # Gas effects
        self.gas_type = AmbientGas.NITROGEN
        self.gas_flow_sccm = 0.0
        self.chamber_pressure_torr = 760.0

        # Controller state
        self.setpoint_C = 25.0
        self.max_lamp_power_W = 10000.0
        self.lamp_saturation_reached = False

        # PID controller (for automatic control)
        self.kp = 2.0  # Proportional gain
        self.ki = 0.1  # Integral gain
        self.kd = 0.5  # Derivative gain
        self.integral_error = 0.0
        self.last_error = 0.0

        # Simulation time
        self.simulation_time_s = 0.0
        self.last_update_time = datetime.now()

    def update(self, dt: float):
        """
        Update thermal plant state by one time step.

        Args:
            dt: Time step (seconds)
        """
        self.simulation_time_s += dt

        # Calculate target temperature based on setpoint and PID
        error = self.setpoint_C - self.wafer_temp_C
        self.integral_error += error * dt
        derivative_error = (error - self.last_error) / dt if dt > 0 else 0
        self.last_error = error

        # PID output (lamp power demand)
        pid_output = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        # Convert PID output to lamp power percentage (with saturation)
        base_power_pct = max(0, min(100, pid_output))

        # Update each zone with thermal coupling
        zone_temps = [zone.temperature_C for zone in self.zones]

        for i, zone in enumerate(self.zones):
            # Apply lamp power with zone-specific response
            target_power = zone.lamp_power_pct if zone.lamp_power_pct > 0 else base_power_pct

            # Check for actuator saturation
            self.lamp_saturation_reached = target_power >= 99.0

            # Power to heat (simplified model)
            power_W = (target_power / 100.0) * (self.max_lamp_power_W / self.num_zones)

            # Heat input
            heat_input_J_per_s = power_W

            # Radiative cooling (Stefan-Boltzmann)
            T_kelvin = zone.temperature_C + 273.15
            T_ambient_kelvin = 298.15  # 25°C
            radiative_loss = self.emissivity * STEFAN_BOLTZMANN * WAFER_AREA_M2 * (T_kelvin**4 - T_ambient_kelvin**4)

            # Convective cooling (depends on gas flow and pressure)
            convective_coeff = self._calculate_convective_coefficient()
            convective_loss = convective_coeff * WAFER_AREA_M2 * (zone.temperature_C - 25.0)

            # Net heat flux
            net_heat_flux = heat_input_J_per_s - radiative_loss - convective_loss

            # Thermal inertia (first-order response)
            wafer_heat_capacity = WAFER_MASS_KG * SI_SPECIFIC_HEAT / self.num_zones
            temp_rate = net_heat_flux / wafer_heat_capacity

            # Update zone temperature with time constant
            zone_target_temp = zone.temperature_C + temp_rate * dt
            zone.temperature_C += (zone_target_temp - zone.temperature_C) * (dt / zone.time_constant_s)

            # Thermal coupling between zones (heat diffusion)
            if i > 0:
                coupling_strength = 0.1  # 10% coupling
                zone.temperature_C += coupling_strength * (zone_temps[i-1] - zone.temperature_C) * dt
            if i < self.num_zones - 1:
                coupling_strength = 0.1
                zone.temperature_C += coupling_strength * (zone_temps[i+1] - zone.temperature_C) * dt

            # Add process noise
            noise = self.rng.normal(0, 0.5)  # ±0.5°C noise
            zone.temperature_C += noise * dt

        # Calculate wafer temperature (weighted average)
        # Center zones typically hotter due to edge losses
        weights = np.array([0.8, 1.0, 1.0, 0.8][:self.num_zones])
        weights /= weights.sum()
        self.wafer_temp_C = sum(w * z.temperature_C for w, z in zip(weights, self.zones))

        # Update sensor readings with lag
        self._update_sensors(dt)

        # Apply emissivity drift
        self.emissivity += self.emissivity_drift_rate * dt
        self.emissivity = max(0.1, min(1.0, self.emissivity))  # Clamp

        # Store history
        self.wafer_temp_history.append(self.wafer_temp_C)
        if len(self.wafer_temp_history) > 1000:
            self.wafer_temp_history.pop(0)

    def _calculate_convective_coefficient(self) -> float:
        """
        Calculate convective heat transfer coefficient based on gas properties.

        Returns:
            Convective coefficient (W/(m²·K))
        """
        # Base convective coefficient for air/nitrogen at atmospheric pressure
        h_base = 10.0  # W/(m²·K)

        # Gas type effects
        gas_factors = {
            AmbientGas.NITROGEN: 1.0,
            AmbientGas.ARGON: 0.7,  # Lower thermal conductivity
            AmbientGas.OXYGEN: 1.1,
            AmbientGas.FORMING_GAS: 1.3,  # H2 increases conductivity
            AmbientGas.VACUUM: 0.01  # Minimal convection
        }

        gas_factor = gas_factors.get(self.gas_type, 1.0)

        # Pressure effect (linear approximation)
        pressure_factor = self.chamber_pressure_torr / 760.0

        # Flow rate effect (forced convection)
        flow_factor = 1.0 + 0.001 * self.gas_flow_sccm  # 0.1% per 100 sccm

        return h_base * gas_factor * pressure_factor * flow_factor

    def _update_sensors(self, dt: float):
        """Update sensor readings with realistic lag and noise."""
        # Pyrometer (fast, but emissivity-dependent)
        # Pyrometer measures apparent temperature based on emissivity
        # True relationship: I = ε·σ·T⁴
        # With wrong emissivity: T_measured = T_true * (ε_true/ε_set)^0.25

        emissivity_error = 1.0  # Assume correct for now (could add drift)
        apparent_temp = self.wafer_temp_C * (emissivity_error ** 0.25)

        # First-order lag
        self.pyrometer_temp_C += (apparent_temp - self.pyrometer_temp_C) * (dt / self.pyrometer_lag_s)

        # Add measurement noise
        pyrometer_noise = self.rng.normal(0, 2.0)  # ±2°C
        self.pyrometer_temp_C += pyrometer_noise

        # Thermocouple (slower, but direct contact measurement)
        # Thermocouple lags behind due to thermal mass
        self.thermocouple_temp_C += (self.wafer_temp_C - self.thermocouple_temp_C) * (dt / self.thermocouple_lag_s)

        # Thermocouple noise (less than pyrometer)
        tc_noise = self.rng.normal(0, 1.0)  # ±1°C
        self.thermocouple_temp_C += tc_noise

    def set_lamp_power(self, zone_powers: List[float]):
        """Set lamp power for each zone."""
        if len(zone_powers) != self.num_zones:
            raise ValueError(f"Expected {self.num_zones} zone powers")

        for zone, power in zip(self.zones, zone_powers):
            zone.lamp_power_pct = max(0, min(100, power))

    def set_setpoint(self, setpoint_C: float):
        """Set temperature setpoint."""
        self.setpoint_C = setpoint_C

    def get_overshoot(self) -> float:
        """Calculate temperature overshoot percentage."""
        if len(self.wafer_temp_history) < 2:
            return 0.0

        max_temp = max(self.wafer_temp_history[-100:])  # Check last 100 samples
        overshoot = max(0, max_temp - self.setpoint_C)
        return (overshoot / self.setpoint_C * 100) if self.setpoint_C > 0 else 0.0

    def reset(self):
        """Reset thermal plant to room temperature."""
        for zone in self.zones:
            zone.temperature_C = 25.0
            zone.lamp_power_pct = 0.0

        self.wafer_temp_C = 25.0
        self.pyrometer_temp_C = 25.0
        self.thermocouple_temp_C = 25.0
        self.integral_error = 0.0
        self.last_error = 0.0
        self.wafer_temp_history = [25.0]


# ============================================================================
# HIL Driver Implementation
# ============================================================================

class RTPHILDriver(RTPDriver):
    """Hardware-in-Loop driver with thermal plant simulation."""

    def __init__(
        self,
        equipment_id: str,
        num_zones: int = 4,
        random_seed: Optional[int] = None,
        simulation_timestep_s: float = 0.1
    ):
        super().__init__(equipment_id)
        self.num_zones = num_zones
        self.simulation_timestep_s = simulation_timestep_s

        # Thermal plant
        self.thermal_plant = ThermalPlantModel(num_zones=num_zones, random_seed=random_seed)

        # State variables
        self._emissivity = 0.65
        self._gas_params: Optional[GasFlowParameters] = None
        self._chamber_pressure_torr = 760.0

        # Recipe execution
        self._current_recipe: Optional[TemperatureRecipe] = None
        self._recipe_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._recipe_start_time: Optional[datetime] = None
        self._current_segment_index = 0
        self._segment_start_time: Optional[datetime] = None

        # Last update time
        self._last_sim_update = datetime.now()

    async def connect(self) -> bool:
        logger.info(f"Connecting to HIL RTP {self.equipment_id}")
        self._is_connected = True
        self._last_sim_update = datetime.now()
        return True

    async def disconnect(self) -> bool:
        logger.info(f"Disconnecting from HIL RTP {self.equipment_id}")
        self._is_connected = False
        self.status = RTPStatus.IDLE
        return True

    async def initialize(self) -> bool:
        if not self._is_connected:
            raise RuntimeError("Not connected to RTP")

        logger.info("Initializing HIL RTP system")
        self.thermal_plant.reset()
        self.status = RTPStatus.IDLE
        return True

    async def shutdown(self) -> bool:
        logger.info("Shutting down HIL RTP")
        await self.stop_recipe()
        self.thermal_plant.set_setpoint(25.0)
        self.status = RTPStatus.SHUTDOWN
        return True

    def _update_simulation(self):
        """Update thermal simulation based on elapsed time."""
        now = datetime.now()
        elapsed = (now - self._last_sim_update).total_seconds()

        # Run multiple simulation steps if needed
        num_steps = int(elapsed / self.simulation_timestep_s)
        for _ in range(min(num_steps, 100)):  # Limit to prevent runaway
            self.thermal_plant.update(self.simulation_timestep_s)

        self._last_sim_update = now

    # Temperature control
    async def set_target_temperature(self, temp_C: float, ramp_rate_C_per_s: Optional[float] = None) -> bool:
        logger.info(f"HIL: Setting target temperature: {temp_C}°C")
        self.thermal_plant.set_setpoint(temp_C)

        # Update status
        current_temp = self.thermal_plant.wafer_temp_C
        if temp_C > current_temp + 5:
            self.status = RTPStatus.HEATING
        elif temp_C < current_temp - 5:
            self.status = RTPStatus.COOLING
        else:
            self.status = RTPStatus.AT_TEMPERATURE

        return True

    async def get_temperature(self) -> Dict[str, float]:
        self._update_simulation()

        return {
            "pyrometer_C": self.thermal_plant.pyrometer_temp_C,
            "thermocouple_C": self.thermal_plant.thermocouple_temp_C,
            "setpoint_C": self.thermal_plant.setpoint_C,
            "deviation_C": abs(self.thermal_plant.wafer_temp_C - self.thermal_plant.setpoint_C),
            "zone_temps_C": [z.temperature_C for z in self.thermal_plant.zones]
        }

    async def set_emissivity(self, emissivity: float) -> bool:
        if not 0.1 <= emissivity <= 1.0:
            raise ValueError("Emissivity must be between 0.1 and 1.0")

        logger.info(f"HIL: Setting emissivity: {emissivity}")
        self._emissivity = emissivity
        self.thermal_plant.emissivity = emissivity
        return True

    async def get_emissivity(self) -> float:
        return self._emissivity

    # Lamp control
    async def set_lamp_power(self, params: LampParameters) -> bool:
        if len(params.zone_powers_pct) != self.num_zones:
            raise ValueError(f"Expected {self.num_zones} zone powers")

        logger.info(f"HIL: Setting lamp powers: {params.zone_powers_pct}")
        self.thermal_plant.set_lamp_power(params.zone_powers_pct)
        self.thermal_plant.max_lamp_power_W = params.max_power_W
        return True

    async def get_lamp_power(self) -> Dict[str, float]:
        self._update_simulation()

        result = {
            f"zone_{i+1}_pct": zone.lamp_power_pct
            for i, zone in enumerate(self.thermal_plant.zones)
        }
        result["saturation_reached"] = self.thermal_plant.lamp_saturation_reached
        return result

    # Gas control
    async def set_gas_flow(self, params: GasFlowParameters) -> bool:
        logger.info(f"HIL: Setting gas flow: {params.gas_type} @ {params.flow_rate_sccm} sccm")
        self._gas_params = params
        self._chamber_pressure_torr = params.chamber_pressure_torr

        # Update thermal plant
        self.thermal_plant.gas_type = params.gas_type
        self.thermal_plant.gas_flow_sccm = params.flow_rate_sccm
        self.thermal_plant.chamber_pressure_torr = params.chamber_pressure_torr

        return True

    async def get_gas_flow(self) -> GasFlowParameters:
        if self._gas_params is None:
            return GasFlowParameters(
                gas_type=AmbientGas.NITROGEN,
                flow_rate_sccm=0.0,
                chamber_pressure_torr=760.0
            )
        return self._gas_params

    async def set_chamber_pressure(self, pressure_torr: float) -> bool:
        logger.info(f"HIL: Setting chamber pressure: {pressure_torr} Torr")
        self._chamber_pressure_torr = pressure_torr
        self.thermal_plant.chamber_pressure_torr = pressure_torr
        return True

    async def get_chamber_pressure(self) -> float:
        return self._chamber_pressure_torr

    # Recipe execution
    async def load_recipe(self, recipe: TemperatureRecipe) -> str:
        logger.info(f"HIL: Loading recipe: {recipe.recipe_name}")
        self._current_recipe = recipe
        self._recipe_id = f"RECIPE-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._current_segment_index = 0

        # Set gas parameters
        await self.set_gas_flow(recipe.gas_params)

        # Set emissivity
        await self.set_emissivity(recipe.emissivity.emissivity)

        return self._recipe_id

    async def start_recipe(self, recipe_id: str) -> str:
        if self._recipe_id != recipe_id:
            raise ValueError(f"Recipe {recipe_id} not loaded")

        if self._current_recipe is None:
            raise RuntimeError("No recipe loaded")

        logger.info(f"HIL: Starting recipe execution: {self._current_recipe.recipe_name}")
        self._run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._recipe_start_time = datetime.now()
        self._segment_start_time = datetime.now()
        self._current_segment_index = 0
        self.status = RTPStatus.RUNNING_RECIPE

        # Start first segment
        if self._current_recipe.segments:
            first_segment = self._current_recipe.segments[0]
            await self.set_target_temperature(first_segment.target_temp_C, first_segment.ramp_rate_C_per_s)

        return self._run_id

    async def pause_recipe(self) -> bool:
        logger.info("HIL: Pausing recipe")
        # Hold current temperature
        return True

    async def resume_recipe(self) -> bool:
        logger.info("HIL: Resuming recipe")
        self.status = RTPStatus.RUNNING_RECIPE
        return True

    async def stop_recipe(self) -> bool:
        logger.info("HIL: Stopping recipe")
        self.status = RTPStatus.COOLING
        await self.set_target_temperature(25.0)
        self._run_id = None
        return True

    async def get_recipe_progress(self) -> Dict[str, Any]:
        self._update_simulation()

        if self._current_recipe is None or self._run_id is None:
            return {
                "is_running": False,
                "run_id": None,
                "recipe_name": None,
                "current_segment": 0,
                "total_segments": 0,
                "elapsed_time_s": 0.0,
                "progress_pct": 0.0
            }

        elapsed = (datetime.now() - self._recipe_start_time).total_seconds() if self._recipe_start_time else 0.0
        num_segments = len(self._current_recipe.segments)

        # Check if we should advance to next segment
        if self._segment_start_time and self._current_segment_index < num_segments:
            segment = self._current_recipe.segments[self._current_segment_index]
            segment_elapsed = (datetime.now() - self._segment_start_time).total_seconds()

            # Estimate time to reach target
            current_temp = self.thermal_plant.wafer_temp_C
            temp_diff = abs(segment.target_temp_C - current_temp)
            ramp_time = temp_diff / max(segment.ramp_rate_C_per_s, 0.1) if segment.ramp_rate_C_per_s > 0 else 0

            # If reached target and dwell complete, move to next segment
            if temp_diff < 5.0 and segment_elapsed >= ramp_time + segment.dwell_time_s:
                self._current_segment_index += 1
                self._segment_start_time = datetime.now()

                if self._current_segment_index < num_segments:
                    next_segment = self._current_recipe.segments[self._current_segment_index]
                    await self.set_target_temperature(next_segment.target_temp_C, next_segment.ramp_rate_C_per_s)
                else:
                    # Recipe complete
                    await self.stop_recipe()

        return {
            "is_running": self.status == RTPStatus.RUNNING_RECIPE,
            "run_id": self._run_id,
            "recipe_name": self._current_recipe.recipe_name,
            "current_segment": self._current_segment_index,
            "total_segments": num_segments,
            "elapsed_time_s": elapsed,
            "progress_pct": (self._current_segment_index / num_segments * 100) if num_segments > 0 else 0.0,
            "current_temp_C": self.thermal_plant.wafer_temp_C,
            "target_temp_C": self.thermal_plant.setpoint_C,
            "overshoot_pct": self.thermal_plant.get_overshoot()
        }

    # Status and diagnostics
    async def get_status(self) -> RTPStatus:
        self._update_simulation()
        return self.status

    async def check_interlocks(self) -> Dict[str, bool]:
        """All interlocks OK in simulation."""
        return {
            "chamber_door": True,
            "cooling_water": True,
            "lamp_overtemp": not self.thermal_plant.lamp_saturation_reached,
            "pyrometer_valid": True,
            "pressure_ok": True,
            "e_stop": True
        }

    # Additional HIL-specific methods
    def get_thermal_plant(self) -> ThermalPlantModel:
        """Get access to thermal plant model for analysis."""
        return self.thermal_plant


# Export
__all__ = [
    "RTPHILDriver",
    "ThermalPlantModel",
    "ThermalZoneState"
]
