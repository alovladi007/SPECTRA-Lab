"""
CVD Platform - PECVD Plasma HIL Simulator
Hardware-in-the-Loop simulator for Plasma-Enhanced CVD
Supports RF/ICP plasma with substrate biasing
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
import asyncio
import logging
import math

import numpy as np

from ..tools.base import (
    CVDToolBase,
    TelemetryPoint,
    RecipeValidationResult,
    ToolCapabilities,
    ToolState,
)


logger = logging.getLogger(__name__)


# ============================================================================
# PECVD Physics Model
# ============================================================================

class PECVDPlasmaModel:
    """Physics-based PECVD plasma and deposition model"""

    def __init__(self):
        # Constants
        self.R_GAS = 8.314  # J/(mol·K)
        self.ELEM_CHARGE = 1.602e-19  # Coulombs

        # Plasma parameters
        self.ION_MASS_AMU = 28  # N+ or SiH3+ (approx)
        self.ELECTRON_TEMP_EV = 3.0  # Typical electron temperature

    def calculate_plasma_density(
        self,
        rf_power_w: float,
        pressure_pa: float,
        gas_flows: Dict[str, float],
    ) -> float:
        """
        Calculate plasma density using simplified power balance.

        Args:
            rf_power_w: RF power in Watts
            pressure_pa: Pressure in Pascals
            gas_flows: Gas flow rates

        Returns:
            Plasma density in m^-3
        """
        # Simplified model: n_e ~ sqrt(Power / Pressure)
        # Typical: 10^15 - 10^17 m^-3

        base_density = 1e15  # m^-3

        # Power scaling
        power_factor = math.sqrt(rf_power_w / 100.0)  # Normalized to 100W

        # Pressure scaling (inverse)
        pressure_factor = math.sqrt(100.0 / pressure_pa)  # Normalized to 100 Pa

        density = base_density * power_factor * pressure_factor

        # Add variation
        density *= (1.0 + np.random.normal(0, 0.05))

        return max(1e14, min(1e18, density))

    def calculate_ion_energy(
        self,
        dc_bias_v: float,
        pressure_pa: float,
    ) -> float:
        """
        Calculate ion bombardment energy.

        Args:
            dc_bias_v: DC bias voltage (negative)
            pressure_pa: Pressure

        Returns:
            Ion energy in eV
        """
        # Ion energy ~ |V_bias|
        # Reduced at higher pressures due to collisions

        sheath_potential = abs(dc_bias_v)

        # Collision factor (higher pressure = more collisions = less energy)
        collision_factor = 1.0 / (1.0 + pressure_pa / 50.0)

        ion_energy = sheath_potential * collision_factor

        return ion_energy

    def calculate_deposition_rate_pecvd(
        self,
        temperature_c: float,
        pressure_pa: float,
        rf_power_w: float,
        gas_flows: Dict[str, float],
        material: str = "SiO2",
    ) -> float:
        """
        Calculate PECVD deposition rate.

        Args:
            temperature_c: Temperature
            pressure_pa: Pressure
            rf_power_w: RF power
            gas_flows: Gas flows
            material: Material being deposited

        Returns:
            Deposition rate in nm/min
        """
        # PECVD rate depends on:
        # 1. Plasma density (power, pressure)
        # 2. Precursor concentration (flows)
        # 3. Surface temperature (secondary effect)

        # Base rate
        if material == "SiO2":
            # SiH4 + O2 → SiO2
            base_rate = 50.0  # nm/min at reference conditions

            # Precursor concentrations
            sih4_flow = gas_flows.get("SiH4", 0.0)
            o2_flow = gas_flows.get("O2", 0.0) + gas_flows.get("N2O", 0.0)

            if sih4_flow == 0 or o2_flow == 0:
                return 0.0

            # Flow factor (normalized to 50 sccm SiH4, 500 sccm O2)
            flow_factor = math.sqrt((sih4_flow / 50.0) * (o2_flow / 500.0))

        elif material == "Si3N4":
            # SiH4 + NH3 → Si3N4 (plasma)
            base_rate = 40.0  # nm/min

            sih4_flow = gas_flows.get("SiH4", 0.0)
            nh3_flow = gas_flows.get("NH3", 0.0)

            if sih4_flow == 0 or nh3_flow == 0:
                return 0.0

            flow_factor = math.sqrt((sih4_flow / 50.0) * (nh3_flow / 100.0))

        elif material == "SiN":
            # Alias for Si3N4
            return self.calculate_deposition_rate_pecvd(
                temperature_c, pressure_pa, rf_power_w, gas_flows, "Si3N4"
            )

        else:
            logger.warning(f"Unknown PECVD material: {material}, using default")
            base_rate = 30.0
            flow_factor = 1.0

        # Power factor (linear approximation)
        power_factor = rf_power_w / 300.0  # Normalized to 300W

        # Pressure factor (parabolic, peak around 100-200 Pa)
        optimal_pressure = 150.0
        pressure_factor = 1.0 - ((pressure_pa - optimal_pressure) / optimal_pressure) ** 2
        pressure_factor = max(0.1, pressure_factor)

        # Temperature factor (weak dependence for PECVD)
        # Typical 250-350°C
        temp_factor = 1.0 + (temperature_c - 300.0) / 1000.0

        # Calculate rate
        rate = base_rate * flow_factor * power_factor * pressure_factor * temp_factor

        # Add noise
        rate *= (1.0 + np.random.normal(0, 0.03))

        return max(0, rate)

    def calculate_film_stress(
        self,
        rf_power_w: float,
        dc_bias_v: float,
        temperature_c: float,
    ) -> float:
        """
        Calculate film stress (compressive/tensile).

        Args:
            rf_power_w: RF power
            dc_bias_v: DC bias
            temperature_c: Temperature

        Returns:
            Stress in MPa (positive = tensile, negative = compressive)
        """
        # Ion bombardment → compressive stress
        # Temperature → tensile stress

        # Base stress
        base_stress = -100.0  # MPa (compressive)

        # Ion energy effect
        ion_energy = abs(dc_bias_v)
        ion_stress = -ion_energy * 0.5  # More negative = more compressive

        # Temperature effect
        temp_stress = (temperature_c - 300) * 0.3  # Tensile

        total_stress = base_stress + ion_stress + temp_stress

        # Add variation
        total_stress += np.random.normal(0, 10)

        return total_stress


# ============================================================================
# PECVD Tool Simulator
# ============================================================================

class PECVDPlasmaSimulator(CVDToolBase):
    """
    PECVD Tool Simulator with RF plasma.
    Supports parallel-plate and ICP configurations.
    """

    def __init__(
        self,
        tool_id: UUID,
        tool_name: str,
        material: str = "SiO2",
        plasma_type: str = "RF",  # 'RF' or 'ICP'
    ):
        """
        Initialize PECVD simulator.

        Args:
            tool_id: Tool UUID
            tool_name: Tool name
            material: Material to deposit
            plasma_type: Plasma type ('RF' or 'ICP')
        """
        # Define PECVD capabilities
        capabilities = ToolCapabilities(
            min_pressure_pa=50.0,
            max_pressure_pa=500.0,
            pressure_control_resolution_pa=0.5,
            min_temperature_c=150.0,
            max_temperature_c=400.0,
            temperature_zones=1,  # Single substrate heater
            max_ramp_rate_c_per_min=30.0,
            max_gas_lines=8,
            max_total_flow_sccm=10000.0,
            available_gases=["N2", "SiH4", "NH3", "N2O", "O2", "Ar", "H2", "CF4"],
            has_plasma=True,
            plasma_types=[plasma_type],
            max_plasma_power_w=2000.0,
            plasma_frequency_mhz=[13.56, 27.12] if plasma_type == "RF" else [13.56],
            has_rotation=True,
            max_rotation_speed_rpm=100.0,
            wafer_sizes_mm=[200.0, 300.0],
            max_wafers_per_batch=1,  # Single wafer
            has_load_lock=True,
            has_endpoint_detection=True,
        )

        super().__init__(
            tool_id=tool_id,
            tool_name=tool_name,
            capabilities=capabilities,
            hardware_interface=None,
            simulation_mode=True,
        )

        self.material = material
        self.plasma_type = plasma_type

        # Physics model
        self.plasma_model = PECVDPlasmaModel()

        # Simulated hardware state
        self._substrate_temperature: float = 25.0
        self._setpoint_temperature: float = 25.0

        self._chamber_pressure: float = 101325.0
        self._setpoint_pressure: float = 101325.0

        self._gas_flows_actual: Dict[str, float] = {}
        self._gas_flows_setpoint: Dict[str, float] = {}

        # Plasma parameters
        self._rf_power_actual: float = 0.0
        self._rf_power_setpoint: float = 0.0
        self._reflected_power: float = 0.0

        self._dc_bias: float = 0.0  # Volts (negative)
        self._dc_bias_setpoint: float = 0.0

        self._plasma_on: bool = False
        self._plasma_density: float = 0.0  # m^-3

        # Rotation
        self._rotation_speed: float = 0.0
        self._rotation_setpoint: float = 0.0

        # Process state
        self._film_thickness_nm: float = 0.0
        self._deposition_rate_nm_min: float = 0.0
        self._uniformity_pct: float = 0.0
        self._film_stress_mpa: float = 0.0

        logger.info(f"PECVD {plasma_type} Simulator initialized: {tool_name}, Material: {material}")

    # ========================================================================
    # Abstract Method Implementations
    # ========================================================================

    async def initialize_hardware(self) -> bool:
        """Initialize PECVD simulator"""
        logger.info(f"Initializing PECVD simulator {self.tool_name}...")

        await asyncio.sleep(2.0)

        # Initialize to safe state
        self._substrate_temperature = 25.0
        self._setpoint_temperature = 25.0
        self._chamber_pressure = 101325.0
        self._setpoint_pressure = 101325.0

        self._gas_flows_actual = {gas: 0.0 for gas in self.capabilities.available_gases}
        self._gas_flows_setpoint = {gas: 0.0 for gas in self.capabilities.available_gases}

        self._rf_power_actual = 0.0
        self._rf_power_setpoint = 0.0
        self._dc_bias = 0.0
        self._plasma_on = False

        self._rotation_speed = 0.0
        self._film_thickness_nm = 0.0

        logger.info(f"PECVD simulator {self.tool_name} initialized")
        return True

    async def shutdown_hardware(self) -> bool:
        """Shutdown PECVD simulator"""
        logger.info(f"Shutting down PECVD simulator {self.tool_name}...")

        # Turn off plasma
        self._plasma_on = False
        self._rf_power_setpoint = 0.0
        await asyncio.sleep(1.0)

        # Close gas flows
        self._gas_flows_setpoint = {gas: 0.0 for gas in self.capabilities.available_gases}
        await asyncio.sleep(1.0)

        # Vent
        self._setpoint_pressure = 101325.0
        await asyncio.sleep(2.0)

        # Cool
        self._setpoint_temperature = 25.0
        await asyncio.sleep(1.0)

        logger.info(f"PECVD simulator {self.tool_name} shutdown complete")
        return True

    async def validate_recipe(self, recipe: Dict[str, Any]) -> RecipeValidationResult:
        """Validate PECVD recipe"""
        errors = []
        warnings = []

        # Temperature
        temp_profile = recipe.get("temperature_profile", {})
        zones = temp_profile.get("zones", [])

        if len(zones) > 0:
            temp = zones[0].get("setpoint_c", 0)
            if temp < self.capabilities.min_temperature_c or temp > self.capabilities.max_temperature_c:
                errors.append(
                    f"Temperature {temp}°C out of range "
                    f"({self.capabilities.min_temperature_c}-{self.capabilities.max_temperature_c}°C)"
                )

        # Pressure
        pressure_profile = recipe.get("pressure_profile", {})
        process_pressure = pressure_profile.get("process_pressure_pa", 0)

        if process_pressure < self.capabilities.min_pressure_pa or process_pressure > self.capabilities.max_pressure_pa:
            errors.append(
                f"Pressure {process_pressure} Pa out of range "
                f"({self.capabilities.min_pressure_pa}-{self.capabilities.max_pressure_pa} Pa)"
            )

        # Plasma settings
        plasma_settings = recipe.get("plasma_settings", {})
        if plasma_settings:
            rf_power = plasma_settings.get("rf_power_w", 0)
            if rf_power > self.capabilities.max_plasma_power_w:
                errors.append(
                    f"RF power {rf_power}W exceeds maximum {self.capabilities.max_plasma_power_w}W"
                )

            frequency = plasma_settings.get("frequency_mhz", 13.56)
            if frequency not in self.capabilities.plasma_frequency_mhz:
                warnings.append(
                    f"Frequency {frequency} MHz may not be supported (available: {self.capabilities.plasma_frequency_mhz})"
                )

        # Gas flows
        gas_flows = recipe.get("gas_flows", {})
        gases = gas_flows.get("gases", [])
        total_flow = sum(g.get("flow_sccm", 0) for g in gases)

        if total_flow > self.capabilities.max_total_flow_sccm:
            errors.append(f"Total flow {total_flow} sccm exceeds maximum")

        for gas in gases:
            gas_name = gas.get("name")
            if gas_name not in self.capabilities.available_gases:
                errors.append(f"Gas {gas_name} not available")

        # Material-specific validation
        gas_names = [g.get("name") for g in gases]

        if self.material == "SiO2":
            if "SiH4" not in gas_names:
                errors.append("SiO2 deposition requires SiH4")
            if "O2" not in gas_names and "N2O" not in gas_names:
                errors.append("SiO2 deposition requires O2 or N2O")

        elif self.material == "Si3N4":
            if "SiH4" not in gas_names or "NH3" not in gas_names:
                errors.append("Si3N4 deposition requires SiH4 and NH3")

        return RecipeValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    async def execute_recipe_step(self, step: Dict[str, Any]) -> bool:
        """Execute PECVD recipe step"""
        step_name = step.get("name", "Unknown")
        action = step.get("action", "").lower()
        duration_s = step.get("duration_s", 0)

        logger.info(f"Executing PECVD step: {step_name} ({action}, {duration_s}s)")

        try:
            if action == "evacuate" or action == "pumpdown":
                await self._simulate_pumpdown(duration_s, step.get("target_pressure_pa", 100.0))

            elif action == "heat" or action == "ramp_temperature":
                await self._simulate_heating(duration_s, step.get("target_temperature_c", 300.0))

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
            logger.exception(f"Error executing PECVD step {step_name}: {e}")
            return False

    async def read_telemetry(self) -> TelemetryPoint:
        """Read PECVD telemetry"""
        await self._update_simulated_hardware()

        telemetry = TelemetryPoint(
            timestamp=datetime.utcnow(),
            temperatures={
                "substrate": self._substrate_temperature,
                "electrode": self._substrate_temperature + 20.0,  # Simulated
            },
            pressures={
                "chamber": self._chamber_pressure,
                "foreline": 2000.0,
            },
            gas_flows=self._gas_flows_actual.copy(),
            plasma_parameters={
                "rf_power_w": self._rf_power_actual,
                "reflected_power_w": self._reflected_power,
                "dc_bias_v": self._dc_bias,
                "plasma_density_m3": self._plasma_density,
                "plasma_on": float(self._plasma_on),
            },
            rotation_speed_rpm=self._rotation_speed,
            valve_positions={"throttle": 45.0},
            heater_powers={"substrate": self._calculate_heater_power()},
        )

        # Store for status
        self._temperatures = telemetry.temperatures
        self._pressures = telemetry.pressures
        self._gas_flows = telemetry.gas_flows
        self._plasma_parameters = telemetry.plasma_parameters

        return telemetry

    async def emergency_stop(self) -> bool:
        """PECVD emergency stop"""
        logger.critical(f"EMERGENCY STOP - {self.tool_name}")

        # Turn off plasma immediately
        self._plasma_on = False
        self._rf_power_setpoint = 0.0

        # Close gas flows
        self._gas_flows_setpoint = {gas: 0.0 for gas in self.capabilities.available_gases}

        # Vent chamber
        self._setpoint_pressure = 101325.0

        await self.set_state(ToolState.ERROR)
        return True

    # ========================================================================
    # PECVD-Specific Simulation
    # ========================================================================

    async def _simulate_pumpdown(self, duration_s: float, target_pressure_pa: float):
        """Simulate pumpdown"""
        logger.info(f"Pumping down to {target_pressure_pa} Pa")
        self._setpoint_pressure = target_pressure_pa

        start_pressure = self._chamber_pressure
        time_steps = int(duration_s / self._telemetry_interval_s)

        for i in range(time_steps):
            tau = duration_s / 3.0
            t = i * self._telemetry_interval_s
            self._chamber_pressure = target_pressure_pa + (start_pressure - target_pressure_pa) * math.exp(-t / tau)
            await asyncio.sleep(self._telemetry_interval_s)

        self._chamber_pressure = target_pressure_pa

    async def _simulate_heating(self, duration_s: float, target_temp_c: float):
        """Simulate substrate heating"""
        logger.info(f"Heating to {target_temp_c}°C")
        self._setpoint_temperature = target_temp_c

        time_steps = int(duration_s / self._telemetry_interval_s)
        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)

    async def _simulate_stabilization(self, duration_s: float):
        """Simulate stabilization"""
        logger.info(f"Stabilizing for {duration_s}s")

        time_steps = int(duration_s / self._telemetry_interval_s)
        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)

    async def _simulate_deposition(self, duration_s: float, step: Dict[str, Any]):
        """Simulate PECVD deposition with plasma"""
        logger.info(f"PECVD deposition for {duration_s}s")

        recipe = self._current_recipe or {}

        # Set gas flows
        gas_flows_config = recipe.get("gas_flows", {})
        gases = gas_flows_config.get("gases", [])
        for gas in gases:
            gas_name = gas.get("name")
            flow_sccm = gas.get("flow_sccm", 0.0)
            self._gas_flows_setpoint[gas_name] = flow_sccm

        # Set plasma parameters
        plasma_settings = recipe.get("plasma_settings", {})
        self._rf_power_setpoint = plasma_settings.get("rf_power_w", 300.0)
        self._dc_bias_setpoint = plasma_settings.get("bias_voltage_v", -150.0)

        # Turn on plasma
        self._plasma_on = True

        # Set rotation
        self._rotation_setpoint = step.get("rotation_rpm", 20.0)

        # Deposition loop
        time_steps = int(duration_s / self._telemetry_interval_s)

        for _ in range(time_steps):
            # Calculate deposition rate
            self._deposition_rate_nm_min = self.plasma_model.calculate_deposition_rate_pecvd(
                temperature_c=self._substrate_temperature,
                pressure_pa=self._chamber_pressure,
                rf_power_w=self._rf_power_actual,
                gas_flows=self._gas_flows_actual,
                material=self.material,
            )

            # Integrate thickness
            self._film_thickness_nm += self._deposition_rate_nm_min * (self._telemetry_interval_s / 60.0)

            # Calculate stress
            self._film_stress_mpa = self.plasma_model.calculate_film_stress(
                rf_power_w=self._rf_power_actual,
                dc_bias_v=self._dc_bias,
                temperature_c=self._substrate_temperature,
            )

            # Uniformity (improves with rotation)
            base_uniformity = 3.0  # %
            rotation_factor = 1.0 if self._rotation_speed > 0 else 2.5
            self._uniformity_pct = base_uniformity * rotation_factor
            self._uniformity_pct += np.random.normal(0, 0.3)

            await asyncio.sleep(self._telemetry_interval_s)

        # Turn off plasma
        self._plasma_on = False
        self._rf_power_setpoint = 0.0

        logger.info(f"PECVD deposition complete. Thickness: {self._film_thickness_nm:.2f} nm, Stress: {self._film_stress_mpa:.1f} MPa")

    async def _simulate_cooling(self, duration_s: float, target_temp_c: float):
        """Simulate cooling"""
        logger.info(f"Cooling to {target_temp_c}°C")
        self._setpoint_temperature = target_temp_c

        time_steps = int(duration_s / self._telemetry_interval_s)
        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)

    async def _simulate_venting(self, duration_s: float):
        """Simulate venting"""
        logger.info("Venting to atmosphere")
        self._setpoint_pressure = 101325.0

        time_steps = int(duration_s / self._telemetry_interval_s)
        for _ in range(time_steps):
            await asyncio.sleep(self._telemetry_interval_s)

    async def _update_simulated_hardware(self):
        """Update PECVD hardware state"""
        dt = self._telemetry_interval_s

        # Temperature control
        tau_temp = 20.0
        error = self._setpoint_temperature - self._substrate_temperature
        self._substrate_temperature += (error / tau_temp) * dt
        self._substrate_temperature += np.random.normal(0, 0.5)

        # Pressure control
        tau_pressure = 3.0
        pressure_error = self._setpoint_pressure - self._chamber_pressure
        self._chamber_pressure += (pressure_error / tau_pressure) * dt
        self._chamber_pressure += np.random.normal(0, 1.0)

        # Gas flows
        for gas in self._gas_flows_setpoint:
            tau_flow = 1.5
            flow_error = self._gas_flows_setpoint[gas] - self._gas_flows_actual.get(gas, 0.0)
            self._gas_flows_actual[gas] = self._gas_flows_actual.get(gas, 0.0) + (flow_error / tau_flow) * dt
            self._gas_flows_actual[gas] += np.random.normal(0, 1.0)
            self._gas_flows_actual[gas] = max(0, self._gas_flows_actual[gas])

        # RF power control
        tau_power = 0.5  # Fast
        power_error = self._rf_power_setpoint - self._rf_power_actual
        self._rf_power_actual += (power_error / tau_power) * dt
        self._rf_power_actual = max(0, self._rf_power_actual)

        # Reflected power (matching)
        if self._plasma_on:
            self._reflected_power = self._rf_power_actual * 0.02  # 2% reflected (good match)
            self._reflected_power += np.random.normal(0, 0.5)
        else:
            self._reflected_power = 0.0

        # DC bias
        tau_bias = 0.2
        bias_error = self._dc_bias_setpoint - self._dc_bias
        self._dc_bias += (bias_error / tau_bias) * dt

        # Plasma density
        if self._plasma_on and self._rf_power_actual > 50:
            self._plasma_density = self.plasma_model.calculate_plasma_density(
                rf_power_w=self._rf_power_actual,
                pressure_pa=self._chamber_pressure,
                gas_flows=self._gas_flows_actual,
            )
        else:
            self._plasma_density = 0.0

        # Rotation
        tau_rotation = 2.0
        rotation_error = self._rotation_setpoint - self._rotation_speed
        self._rotation_speed += (rotation_error / tau_rotation) * dt
        self._rotation_speed = max(0, self._rotation_speed)

    def _calculate_heater_power(self) -> float:
        """Calculate heater power"""
        error = self._setpoint_temperature - self._substrate_temperature
        power_pct = min(100, max(0, 50 + error * 3.0))
        return power_pct

    # ========================================================================
    # Public Interface
    # ========================================================================

    def get_film_thickness(self) -> float:
        """Get film thickness"""
        return self._film_thickness_nm

    def get_deposition_rate(self) -> float:
        """Get deposition rate"""
        return self._deposition_rate_nm_min

    def get_uniformity(self) -> float:
        """Get uniformity"""
        return self._uniformity_pct

    def get_film_stress(self) -> float:
        """Get film stress"""
        return self._film_stress_mpa

    def get_plasma_density(self) -> float:
        """Get plasma density"""
        return self._plasma_density

    def reset_film_thickness(self):
        """Reset film thickness"""
        self._film_thickness_nm = 0.0
        logger.info("Film thickness reset")
