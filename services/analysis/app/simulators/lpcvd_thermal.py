"""
CVD Platform - LPCVD Thermal HIL Simulator
Hardware-in-the-Loop simulator for Low Pressure CVD (thermal activation)
Implements physics-based reactor model for Silicon and Silicon Nitride deposition
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
import asyncio
import logging
import math

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d

from ..tools.base import (
    CVDToolBase,
    TelemetryPoint,
    RecipeValidationResult,
    ToolCapabilities,
    ToolState,
)


logger = logging.getLogger(__name__)


# ============================================================================
# LPCVD Physics Constants
# ============================================================================

class LPCVDConstants:
    """Physical constants for LPCVD simulation"""

    # Universal constants
    R_GAS = 8.314  # J/(mol·K) - Universal gas constant
    BOLTZMANN = 1.380649e-23  # J/K

    # Silicon deposition (SiH4 → Si + 2H2)
    SI_K0 = 1.0e8  # s^-1 - Pre-exponential factor
    SI_EA = 170000  # J/mol - Activation energy
    SI_ORDER = 1.0  # Reaction order
    SI_MW = 28.0855  # g/mol - Silicon molecular weight
    SI_DENSITY = 2329  # kg/m³ - Silicon film density

    # Silicon Nitride deposition (3SiH4 + 4NH3 → Si3N4 + 12H2)
    SIN_K0 = 5.0e7  # (sccm)^(-(a+b)) s^-1
    SIN_EA = 150000  # J/mol
    SIN_ORDER_SIH4 = 1.0  # a
    SIN_ORDER_NH3 = 0.5  # b
    SIN_MW = 140.28  # g/mol - Si3N4 molecular weight
    SIN_DENSITY = 3100  # kg/m³

    # Mass transport
    D_SIH4_N2 = 0.18  # cm²/s at STP - Binary diffusion coefficient
    D_NH3_N2 = 0.23  # cm²/s at STP

    # Reactor geometry (typical horizontal tube furnace)
    TUBE_RADIUS = 0.075  # m (150mm diameter)
    TUBE_LENGTH = 1.5  # m
    WAFER_RADIUS = 0.100  # m (200mm wafer)


# ============================================================================
# LPCVD Reactor Physics Model
# ============================================================================

class LPCVDReactorModel:
    """Physics-based LPCVD reactor model"""

    def __init__(self, constants: LPCVDConstants = LPCVDConstants()):
        self.const = constants
        self.current_state = {
            "temperature_c": 25.0,
            "pressure_pa": 101325.0,
            "gas_concentrations": {},
            "film_thickness_nm": 0.0,
        }

    def calculate_reaction_rate(
        self,
        temperature_c: float,
        pressure_pa: float,
        gas_flows: Dict[str, float],
        material: str = "Si",
    ) -> float:
        """
        Calculate surface reaction rate using Arrhenius kinetics.

        Args:
            temperature_c: Temperature in Celsius
            pressure_pa: Pressure in Pascals
            gas_flows: Gas flow rates in sccm
            material: Material being deposited ('Si' or 'Si3N4')

        Returns:
            Reaction rate in mol/(m²·s)
        """
        T_K = temperature_c + 273.15

        if material == "Si":
            # Silicon deposition from silane
            k = self.const.SI_K0 * math.exp(-self.const.SI_EA / (self.const.R_GAS * T_K))

            # Surface concentration (simplified ideal gas)
            C_sih4 = self._flow_to_concentration(
                gas_flows.get("SiH4", 0.0),
                sum(gas_flows.values()),
                pressure_pa,
                T_K,
            )

            # Rate = k * [SiH4]^n
            rate = k * (C_sih4 ** self.const.SI_ORDER)

        elif material == "Si3N4":
            # Silicon nitride deposition
            k = self.const.SIN_K0 * math.exp(-self.const.SIN_EA / (self.const.R_GAS * T_K))

            total_flow = sum(gas_flows.values())
            C_sih4 = self._flow_to_concentration(gas_flows.get("SiH4", 0.0), total_flow, pressure_pa, T_K)
            C_nh3 = self._flow_to_concentration(gas_flows.get("NH3", 0.0), total_flow, pressure_pa, T_K)

            # Rate = k * [SiH4]^a * [NH3]^b
            rate = k * (C_sih4 ** self.const.SIN_ORDER_SIH4) * (C_nh3 ** self.const.SIN_ORDER_NH3)

        else:
            raise ValueError(f"Unknown material: {material}")

        return rate

    def calculate_deposition_rate(
        self,
        temperature_c: float,
        pressure_pa: float,
        gas_flows: Dict[str, float],
        material: str = "Si",
    ) -> float:
        """
        Calculate film deposition rate in nm/min.

        Args:
            temperature_c: Temperature in Celsius
            pressure_pa: Pressure in Pascals
            gas_flows: Gas flow rates in sccm
            material: Material being deposited

        Returns:
            Deposition rate in nm/min
        """
        # Get reaction rate (mol/(m²·s))
        rate_mol = self.calculate_reaction_rate(temperature_c, pressure_pa, gas_flows, material)

        # Convert to nm/min
        if material == "Si":
            mw = self.const.SI_MW * 1e-3  # kg/mol
            rho = self.const.SI_DENSITY  # kg/m³
        elif material == "Si3N4":
            mw = self.const.SIN_MW * 1e-3  # kg/mol
            rho = self.const.SIN_DENSITY  # kg/m³
        else:
            raise ValueError(f"Unknown material: {material}")

        # Rate (nm/min) = (rate_mol * MW / rho) * 1e9 * 60
        deposition_rate_nm_min = (rate_mol * mw / rho) * 1e9 * 60

        return deposition_rate_nm_min

    def calculate_uniformity(
        self,
        temperature_c: float,
        pressure_pa: float,
        gas_flows: Dict[str, float],
        rotation_speed_rpm: float = 0.0,
    ) -> float:
        """
        Calculate film thickness uniformity across wafer.

        Args:
            temperature_c: Temperature
            pressure_pa: Pressure
            gas_flows: Gas flows
            rotation_speed_rpm: Wafer rotation speed

        Returns:
            Uniformity percentage (1-sigma)
        """
        # Simplified model: uniformity improves with rotation and lower deposition rate
        base_uniformity = 5.0  # % at ideal conditions

        # Rotation effect: uniformity improves with rotation
        if rotation_speed_rpm > 0:
            rotation_factor = 1.0 / (1.0 + rotation_speed_rpm / 10.0)
        else:
            rotation_factor = 2.0  # No rotation = worse uniformity

        # Temperature effect: higher temp = more edge effects
        temp_factor = 1.0 + (temperature_c - 650) / 1000.0

        # Pressure effect: lower pressure = better uniformity
        pressure_factor = 1.0 + (pressure_pa - 50) / 500.0

        uniformity = base_uniformity * rotation_factor * temp_factor * pressure_factor

        # Add random variation
        uniformity += np.random.normal(0, 0.5)

        return max(1.0, min(15.0, uniformity))  # Clamp to 1-15%

    def simulate_temperature_ramp(
        self,
        T_initial: float,
        T_target: float,
        ramp_rate: float,
        time_step: float = 1.0,
    ) -> List[float]:
        """
        Simulate temperature ramp with thermal mass effects.

        Args:
            T_initial: Initial temperature (°C)
            T_target: Target temperature (°C)
            ramp_rate: Ramp rate (°C/min)
            time_step: Time step (seconds)

        Returns:
            List of temperatures over time
        """
        ramp_rate_per_sec = ramp_rate / 60.0
        delta_T = T_target - T_initial
        ramp_time_s = abs(delta_T) / ramp_rate_per_sec

        # Generate time points
        t = np.arange(0, ramp_time_s, time_step)

        # First-order thermal response (realistic overshoot)
        tau = 30.0  # Time constant (seconds)
        temperatures = T_initial + delta_T * (1 - np.exp(-t / tau))

        # Add small noise
        temperatures += np.random.normal(0, 0.5, len(temperatures))

        return temperatures.tolist()

    def _flow_to_concentration(
        self,
        flow_sccm: float,
        total_flow_sccm: float,
        pressure_pa: float,
        temperature_k: float,
    ) -> float:
        """
        Convert gas flow to molar concentration.

        Args:
            flow_sccm: Gas flow rate (sccm)
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

        # Ideal gas: C = P/(RT)
        C_total = pressure_pa / (self.const.R_GAS * temperature_k)

        # Partial concentration
        C = x * C_total

        return C


# ============================================================================
# LPCVD Tool Simulator
# ============================================================================

class LPCVDThermalSimulator(CVDToolBase):
    """
    LPCVD Thermal CVD Tool Simulator.
    Implements hardware-in-the-loop simulation with physics-based reactor model.
    """

    def __init__(
        self,
        tool_id: UUID,
        tool_name: str,
        material: str = "Si",
        tube_zones: int = 3,
    ):
        """
        Initialize LPCVD simulator.

        Args:
            tool_id: Tool UUID
            tool_name: Tool name
            material: Material to deposit ('Si' or 'Si3N4')
            tube_zones: Number of temperature zones
        """
        # Define LPCVD capabilities
        capabilities = ToolCapabilities(
            min_pressure_pa=10.0,
            max_pressure_pa=200.0,
            pressure_control_resolution_pa=0.1,
            min_temperature_c=300.0,
            max_temperature_c=900.0,
            temperature_zones=tube_zones,
            max_ramp_rate_c_per_min=20.0,
            max_gas_lines=6,
            max_total_flow_sccm=5000.0,
            available_gases=["N2", "SiH4", "NH3", "H2"],
            has_plasma=False,
            has_rotation=False,  # Typical LPCVD doesn't rotate
            wafer_sizes_mm=[200.0],
            max_wafers_per_batch=25,  # Batch furnace
            cassette_loading=True,
            has_load_lock=False,
        )

        super().__init__(
            tool_id=tool_id,
            tool_name=tool_name,
            capabilities=capabilities,
            hardware_interface=None,
            simulation_mode=True,
        )

        self.material = material
        self.tube_zones = tube_zones

        # Physics model
        self.reactor_model = LPCVDReactorModel()

        # Simulated hardware state
        self._tube_temperatures: List[float] = [25.0] * tube_zones
        self._setpoint_temperatures: List[float] = [25.0] * tube_zones
        self._chamber_pressure: float = 101325.0  # Pa (atmospheric)
        self._setpoint_pressure: float = 101325.0
        self._gas_flows_actual: Dict[str, float] = {}
        self._gas_flows_setpoint: Dict[str, float] = {}

        # Process state
        self._film_thickness_nm: float = 0.0
        self._deposition_rate_nm_min: float = 0.0
        self._uniformity_pct: float = 0.0

        # PID controllers (simplified)
        self._temp_pid_gains = {"Kp": 2.0, "Ki": 0.1, "Kd": 0.5}
        self._pressure_pid_gains = {"Kp": 0.5, "Ki": 0.05, "Kd": 0.1}

        logger.info(f"LPCVD Thermal Simulator initialized: {tool_name}, Material: {material}")

    # ========================================================================
    # Abstract Method Implementations
    # ========================================================================

    async def initialize_hardware(self) -> bool:
        """Initialize simulated hardware"""
        logger.info(f"Initializing LPCVD simulator {self.tool_name}...")

        # Simulate initialization sequence
        await asyncio.sleep(2.0)

        # Initialize temperature zones to ambient
        self._tube_temperatures = [25.0] * self.tube_zones
        self._setpoint_temperatures = [25.0] * self.tube_zones

        # Set atmospheric pressure
        self._chamber_pressure = 101325.0
        self._setpoint_pressure = 101325.0

        # Close all gas flows
        self._gas_flows_actual = {gas: 0.0 for gas in self.capabilities.available_gases}
        self._gas_flows_setpoint = {gas: 0.0 for gas in self.capabilities.available_gases}

        # Reset process variables
        self._film_thickness_nm = 0.0
        self._deposition_rate_nm_min = 0.0

        logger.info(f"LPCVD simulator {self.tool_name} initialized successfully")
        return True

    async def shutdown_hardware(self) -> bool:
        """Shutdown simulated hardware"""
        logger.info(f"Shutting down LPCVD simulator {self.tool_name}...")

        # Close all gas flows
        self._gas_flows_setpoint = {gas: 0.0 for gas in self.capabilities.available_gases}
        await asyncio.sleep(1.0)

        # Vent to atmosphere
        self._setpoint_pressure = 101325.0
        await asyncio.sleep(2.0)

        # Cool down (fast in simulation)
        self._setpoint_temperatures = [25.0] * self.tube_zones
        await asyncio.sleep(1.0)

        logger.info(f"LPCVD simulator {self.tool_name} shutdown complete")
        return True

    async def validate_recipe(self, recipe: Dict[str, Any]) -> RecipeValidationResult:
        """Validate recipe against LPCVD capabilities"""
        errors = []
        warnings = []

        # Check temperature profile
        temp_profile = recipe.get("temperature_profile", {})
        zones = temp_profile.get("zones", [])

        if len(zones) != self.tube_zones:
            errors.append(f"Recipe has {len(zones)} zones, tool has {self.tube_zones} zones")

        for zone in zones:
            temp = zone.get("setpoint_c", 0)
            if temp < self.capabilities.min_temperature_c or temp > self.capabilities.max_temperature_c:
                errors.append(
                    f"Zone {zone.get('zone')} temperature {temp}°C out of range "
                    f"({self.capabilities.min_temperature_c}-{self.capabilities.max_temperature_c}°C)"
                )

            ramp_rate = zone.get("ramp_rate_c_per_min", 0)
            if ramp_rate > self.capabilities.max_ramp_rate_c_per_min:
                warnings.append(
                    f"Zone {zone.get('zone')} ramp rate {ramp_rate}°C/min exceeds "
                    f"maximum {self.capabilities.max_ramp_rate_c_per_min}°C/min"
                )

        # Check pressure profile
        pressure_profile = recipe.get("pressure_profile", {})
        process_pressure = pressure_profile.get("process_pressure_pa", 0)

        if process_pressure < self.capabilities.min_pressure_pa or process_pressure > self.capabilities.max_pressure_pa:
            errors.append(
                f"Process pressure {process_pressure} Pa out of range "
                f"({self.capabilities.min_pressure_pa}-{self.capabilities.max_pressure_pa} Pa)"
            )

        # Check gas flows
        gas_flows = recipe.get("gas_flows", {})
        gases = gas_flows.get("gases", [])
        total_flow = sum(g.get("flow_sccm", 0) for g in gases)

        if total_flow > self.capabilities.max_total_flow_sccm:
            errors.append(
                f"Total gas flow {total_flow} sccm exceeds maximum {self.capabilities.max_total_flow_sccm} sccm"
            )

        for gas in gases:
            gas_name = gas.get("name")
            if gas_name not in self.capabilities.available_gases:
                errors.append(f"Gas {gas_name} not available on this tool")

        # Material-specific checks
        if self.material == "Si3N4":
            # Need both SiH4 and NH3
            gas_names = [g.get("name") for g in gases]
            if "SiH4" not in gas_names or "NH3" not in gas_names:
                errors.append("Si3N4 deposition requires both SiH4 and NH3")

        return RecipeValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def execute_recipe_step(self, step: Dict[str, Any]) -> bool:
        """Execute a single recipe step"""
        step_name = step.get("name", "Unknown")
        action = step.get("action", "").lower()
        duration_s = step.get("duration_s", 0)

        logger.info(f"Executing step: {step_name} (action: {action}, duration: {duration_s}s)")

        try:
            if action == "evacuate" or action == "pumpdown":
                await self._simulate_pumpdown(duration_s, step.get("target_pressure_pa", 50.0))

            elif action == "ramp_temperature" or action == "heat":
                await self._simulate_temperature_ramp(duration_s, step.get("target_temperatures", []))

            elif action == "stabilize":
                await self._simulate_stabilization(duration_s)

            elif action == "deposit" or action == "process":
                await self._simulate_deposition(duration_s, step)

            elif action == "cool":
                await self._simulate_cooling(duration_s, step.get("target_temperature_c", 25.0))

            elif action == "vent":
                await self._simulate_venting(duration_s)

            else:
                logger.warning(f"Unknown action: {action}")
                await asyncio.sleep(duration_s)

            return True

        except Exception as e:
            logger.exception(f"Error executing step {step_name}: {e}")
            return False

    async def read_telemetry(self) -> TelemetryPoint:
        """Read current telemetry from simulator"""
        # Update simulated hardware (simple first-order dynamics)
        await self._update_simulated_hardware()

        # Create telemetry point
        telemetry = TelemetryPoint(
            timestamp=datetime.utcnow(),
            temperatures={f"zone_{i+1}": self._tube_temperatures[i] for i in range(self.tube_zones)},
            pressures={
                "chamber": self._chamber_pressure,
                "foreline": 1200.0,  # Simulated foreline pressure
            },
            gas_flows=self._gas_flows_actual.copy(),
            plasma_parameters=None,  # No plasma in thermal LPCVD
            rotation_speed_rpm=None,
            valve_positions={"throttle": 50.0},  # Simulated
            heater_powers={f"zone_{i+1}": self._calculate_heater_power(i) for i in range(self.tube_zones)},
        )

        # Update stored values for status
        self._temperatures = telemetry.temperatures
        self._pressures = telemetry.pressures
        self._gas_flows = telemetry.gas_flows

        return telemetry

    async def emergency_stop(self) -> bool:
        """Execute emergency stop"""
        logger.critical(f"EMERGENCY STOP - {self.tool_name}")

        # Immediate actions
        self._gas_flows_setpoint = {gas: 0.0 for gas in self.capabilities.available_gases}
        self._setpoint_pressure = 101325.0  # Vent to atmosphere

        # Set state to error
        await self.set_state(ToolState.ERROR)

        return True

    # ========================================================================
    # Simulation Methods
    # ========================================================================

    async def _simulate_pumpdown(self, duration_s: float, target_pressure_pa: float):
        """Simulate chamber pumpdown"""
        logger.info(f"Pumping down to {target_pressure_pa} Pa")

        # Set target
        self._setpoint_pressure = target_pressure_pa

        # Simulate pumpdown curve (exponential)
        start_pressure = self._chamber_pressure
        time_steps = int(duration_s / self._telemetry_interval_s)

        for i in range(time_steps):
            # Exponential approach to target
            tau = duration_s / 3.0  # Time constant
            t = i * self._telemetry_interval_s
            self._chamber_pressure = target_pressure_pa + (start_pressure - target_pressure_pa) * math.exp(-t / tau)

            await asyncio.sleep(self._telemetry_interval_s)

        # Ensure we reach target
        self._chamber_pressure = target_pressure_pa

    async def _simulate_temperature_ramp(self, duration_s: float, target_temps: List[Dict[str, Any]]):
        """Simulate temperature ramp"""
        logger.info(f"Ramping temperature to {target_temps}")

        # Set setpoints
        for zone_config in target_temps:
            zone_idx = zone_config.get("zone", 1) - 1
            if 0 <= zone_idx < self.tube_zones:
                self._setpoint_temperatures[zone_idx] = zone_config.get("setpoint_c", 25.0)

        # Simulate ramp with first-order dynamics
        time_steps = int(duration_s / self._telemetry_interval_s)

        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)
            # Hardware update happens in read_telemetry -> _update_simulated_hardware

    async def _simulate_stabilization(self, duration_s: float):
        """Simulate stabilization period"""
        logger.info(f"Stabilizing for {duration_s}s")

        time_steps = int(duration_s / self._telemetry_interval_s)
        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)

    async def _simulate_deposition(self, duration_s: float, step: Dict[str, Any]):
        """Simulate film deposition"""
        logger.info(f"Depositing {self.material} for {duration_s}s")

        # Get process conditions from step
        recipe = self._current_recipe or {}
        temp_profile = recipe.get("temperature_profile", {})
        gas_flows_config = recipe.get("gas_flows", {})

        # Set gas flows
        gases = gas_flows_config.get("gases", [])
        for gas in gases:
            gas_name = gas.get("name")
            flow_sccm = gas.get("flow_sccm", 0.0)
            self._gas_flows_setpoint[gas_name] = flow_sccm

        # Simulate deposition
        time_steps = int(duration_s / self._telemetry_interval_s)

        for _ in range(time_steps):
            # Calculate instantaneous deposition rate
            avg_temp = np.mean(self._tube_temperatures)
            self._deposition_rate_nm_min = self.reactor_model.calculate_deposition_rate(
                temperature_c=avg_temp,
                pressure_pa=self._chamber_pressure,
                gas_flows=self._gas_flows_actual,
                material=self.material,
            )

            # Integrate thickness
            self._film_thickness_nm += self._deposition_rate_nm_min * (self._telemetry_interval_s / 60.0)

            # Calculate uniformity
            self._uniformity_pct = self.reactor_model.calculate_uniformity(
                temperature_c=avg_temp,
                pressure_pa=self._chamber_pressure,
                gas_flows=self._gas_flows_actual,
                rotation_speed_rpm=0.0,
            )

            await asyncio.sleep(self._telemetry_interval_s)

        logger.info(f"Deposition complete. Thickness: {self._film_thickness_nm:.2f} nm")

    async def _simulate_cooling(self, duration_s: float, target_temp_c: float):
        """Simulate cooling"""
        logger.info(f"Cooling to {target_temp_c}°C")

        # Set all zones to target
        self._setpoint_temperatures = [target_temp_c] * self.tube_zones

        time_steps = int(duration_s / self._telemetry_interval_s)
        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)

    async def _simulate_venting(self, duration_s: float):
        """Simulate venting to atmosphere"""
        logger.info("Venting chamber to atmosphere")

        self._setpoint_pressure = 101325.0  # Atmospheric pressure

        time_steps = int(duration_s / self._telemetry_interval_s)
        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)

    async def _update_simulated_hardware(self):
        """Update simulated hardware state with realistic dynamics"""
        dt = self._telemetry_interval_s

        # Temperature control (first-order + noise)
        for i in range(self.tube_zones):
            tau_temp = 30.0  # seconds
            error = self._setpoint_temperatures[i] - self._tube_temperatures[i]
            self._tube_temperatures[i] += (error / tau_temp) * dt
            # Add noise
            self._tube_temperatures[i] += np.random.normal(0, 0.3)

        # Pressure control
        tau_pressure = 5.0  # seconds
        pressure_error = self._setpoint_pressure - self._chamber_pressure
        self._chamber_pressure += (pressure_error / tau_pressure) * dt
        self._chamber_pressure += np.random.normal(0, 0.5)

        # Gas flow control (fast dynamics)
        for gas in self._gas_flows_setpoint:
            tau_flow = 2.0  # seconds
            flow_error = self._gas_flows_setpoint[gas] - self._gas_flows_actual.get(gas, 0.0)
            self._gas_flows_actual[gas] = self._gas_flows_actual.get(gas, 0.0) + (flow_error / tau_flow) * dt
            self._gas_flows_actual[gas] += np.random.normal(0, 0.5)
            self._gas_flows_actual[gas] = max(0, self._gas_flows_actual[gas])  # No negative flow

    def _calculate_heater_power(self, zone_index: int) -> float:
        """Calculate heater power for a zone (simulated)"""
        # Simple proportional to temperature error
        error = self._setpoint_temperatures[zone_index] - self._tube_temperatures[zone_index]
        power_pct = min(100, max(0, 50 + error * 2.0))  # 0-100%
        return power_pct

    # ========================================================================
    # Public Interface
    # ========================================================================

    def get_film_thickness(self) -> float:
        """Get current film thickness"""
        return self._film_thickness_nm

    def get_deposition_rate(self) -> float:
        """Get current deposition rate"""
        return self._deposition_rate_nm_min

    def get_uniformity(self) -> float:
        """Get current uniformity"""
        return self._uniformity_pct

    def reset_film_thickness(self):
        """Reset film thickness (for new run)"""
        self._film_thickness_nm = 0.0
        logger.info("Film thickness reset to 0")
