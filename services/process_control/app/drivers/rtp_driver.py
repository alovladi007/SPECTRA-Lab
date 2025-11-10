"""RTP (Rapid Thermal Processing) driver interface and implementation."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class AmbientGas(str, Enum):
    """Ambient gas types."""
    NITROGEN = "N2"
    ARGON = "Ar"
    OXYGEN = "O2"
    FORMING_GAS = "N2_H2"  # 95% N2 + 5% H2
    VACUUM = "vacuum"


class RTPStatus(str, Enum):
    """RTP system status."""
    IDLE = "idle"
    HEATING = "heating"
    AT_TEMPERATURE = "at_temperature"
    COOLING = "cooling"
    RUNNING_RECIPE = "running_recipe"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class TemperatureControlMode(str, Enum):
    """Temperature control modes."""
    PYROMETER = "pyrometer"
    THERMOCOUPLE = "thermocouple"
    DUAL = "dual"  # Use both sensors


@dataclass
class RampSegment:
    """Single ramp segment in temperature recipe."""
    target_temp_C: float
    ramp_rate_C_per_s: float
    dwell_time_s: float = 0.0  # Hold at target for this duration


@dataclass
class GasFlowParameters:
    """Gas flow control parameters."""
    gas_type: AmbientGas
    flow_rate_sccm: float
    chamber_pressure_torr: float


@dataclass
class LampParameters:
    """Lamp power control parameters."""
    zone_powers_pct: List[float]  # Power for each lamp zone (0-100%)
    max_power_W: float = 10000.0  # Maximum lamp power


@dataclass
class EmissivitySettings:
    """Emissivity correction settings."""
    emissivity: float = 0.65  # Silicon emissivity (0.65 at 950°C typical)
    wavelength_nm: float = 950.0  # Pyrometer wavelength
    use_correction: bool = True


@dataclass
class TemperatureRecipe:
    """Complete temperature recipe."""
    recipe_name: str
    segments: List[RampSegment]
    gas_params: GasFlowParameters
    emissivity: EmissivitySettings
    control_mode: TemperatureControlMode = TemperatureControlMode.PYROMETER


# ============================================================================
# Abstract Driver Interface
# ============================================================================

class RTPDriver(ABC):
    """Abstract interface for RTP system control."""

    def __init__(self, equipment_id: str):
        self.equipment_id = equipment_id
        self.status = RTPStatus.IDLE
        self._is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to RTP hardware/simulator."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from RTP system."""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize RTP system."""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Safe shutdown of RTP system."""
        pass

    # Temperature control
    @abstractmethod
    async def set_target_temperature(self, temp_C: float, ramp_rate_C_per_s: Optional[float] = None) -> bool:
        """Set target temperature with optional ramp rate."""
        pass

    @abstractmethod
    async def get_temperature(self) -> Dict[str, float]:
        """Get current temperature readings from all sensors."""
        pass

    @abstractmethod
    async def set_emissivity(self, emissivity: float) -> bool:
        """Set emissivity correction for pyrometer."""
        pass

    @abstractmethod
    async def get_emissivity(self) -> float:
        """Get current emissivity setting."""
        pass

    # Lamp control
    @abstractmethod
    async def set_lamp_power(self, params: LampParameters) -> bool:
        """Set lamp power for all zones."""
        pass

    @abstractmethod
    async def get_lamp_power(self) -> Dict[str, float]:
        """Get current lamp power for all zones."""
        pass

    # Gas control
    @abstractmethod
    async def set_gas_flow(self, params: GasFlowParameters) -> bool:
        """Set gas type and flow rate."""
        pass

    @abstractmethod
    async def get_gas_flow(self) -> GasFlowParameters:
        """Get current gas flow parameters."""
        pass

    @abstractmethod
    async def set_chamber_pressure(self, pressure_torr: float) -> bool:
        """Set chamber pressure."""
        pass

    @abstractmethod
    async def get_chamber_pressure(self) -> float:
        """Get current chamber pressure."""
        pass

    # Recipe execution
    @abstractmethod
    async def load_recipe(self, recipe: TemperatureRecipe) -> str:
        """Load a temperature recipe. Returns recipe_id."""
        pass

    @abstractmethod
    async def start_recipe(self, recipe_id: str) -> str:
        """Start recipe execution. Returns run_id."""
        pass

    @abstractmethod
    async def pause_recipe(self) -> bool:
        """Pause current recipe."""
        pass

    @abstractmethod
    async def resume_recipe(self) -> bool:
        """Resume paused recipe."""
        pass

    @abstractmethod
    async def stop_recipe(self) -> bool:
        """Stop current recipe."""
        pass

    @abstractmethod
    async def get_recipe_progress(self) -> Dict[str, Any]:
        """Get current recipe execution progress."""
        pass

    # Status and diagnostics
    @abstractmethod
    async def get_status(self) -> RTPStatus:
        """Get overall RTP status."""
        pass

    @abstractmethod
    async def check_interlocks(self) -> Dict[str, bool]:
        """Check all safety interlocks."""
        pass


# ============================================================================
# Mock Hardware Implementation
# ============================================================================

class RTPMockDriver(RTPDriver):
    """Mock driver for testing (returns realistic values)."""

    def __init__(self, equipment_id: str, num_lamp_zones: int = 4):
        super().__init__(equipment_id)
        self.num_lamp_zones = num_lamp_zones

        # State variables
        self._current_temp_C = 25.0  # Room temperature
        self._target_temp_C = 25.0
        self._emissivity = 0.65
        self._lamp_powers = [0.0] * num_lamp_zones
        self._gas_params: Optional[GasFlowParameters] = None
        self._chamber_pressure_torr = 760.0  # Atmospheric

        # Recipe execution
        self._current_recipe: Optional[TemperatureRecipe] = None
        self._recipe_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._recipe_start_time: Optional[datetime] = None
        self._current_segment_index = 0

    async def connect(self) -> bool:
        logger.info(f"Connecting to mock RTP {self.equipment_id}")
        await asyncio.sleep(0.1)
        self._is_connected = True
        return True

    async def disconnect(self) -> bool:
        logger.info(f"Disconnecting from mock RTP {self.equipment_id}")
        self._is_connected = False
        self.status = RTPStatus.IDLE
        return True

    async def initialize(self) -> bool:
        if not self._is_connected:
            raise RuntimeError("Not connected to RTP")

        logger.info("Initializing RTP system")
        await asyncio.sleep(0.5)
        self.status = RTPStatus.IDLE
        return True

    async def shutdown(self) -> bool:
        logger.info("Shutting down RTP")
        await self.stop_recipe()
        self.status = RTPStatus.SHUTDOWN
        return True

    # Temperature control
    async def set_target_temperature(self, temp_C: float, ramp_rate_C_per_s: Optional[float] = None) -> bool:
        logger.info(f"Setting target temperature: {temp_C}°C")
        self._target_temp_C = temp_C

        # Simple temperature tracking (instant for mock)
        if ramp_rate_C_per_s:
            logger.info(f"  Ramp rate: {ramp_rate_C_per_s}°C/s")

        # Update status
        if temp_C > self._current_temp_C:
            self.status = RTPStatus.HEATING
        elif temp_C < self._current_temp_C:
            self.status = RTPStatus.COOLING

        await asyncio.sleep(0.1)
        return True

    async def get_temperature(self) -> Dict[str, float]:
        # Simulate gradual temperature change
        delta_T = self._target_temp_C - self._current_temp_C
        if abs(delta_T) > 1.0:
            self._current_temp_C += delta_T * 0.1  # 10% step towards target
        else:
            self._current_temp_C = self._target_temp_C
            if self.status in [RTPStatus.HEATING, RTPStatus.COOLING]:
                self.status = RTPStatus.AT_TEMPERATURE

        # Add some noise
        import random
        noise = random.uniform(-2.0, 2.0)

        return {
            "pyrometer_C": self._current_temp_C + noise,
            "thermocouple_C": self._current_temp_C + noise * 0.5,
            "setpoint_C": self._target_temp_C,
            "deviation_C": abs(self._current_temp_C - self._target_temp_C)
        }

    async def set_emissivity(self, emissivity: float) -> bool:
        if not 0.1 <= emissivity <= 1.0:
            raise ValueError("Emissivity must be between 0.1 and 1.0")

        logger.info(f"Setting emissivity: {emissivity}")
        self._emissivity = emissivity
        await asyncio.sleep(0.05)
        return True

    async def get_emissivity(self) -> float:
        return self._emissivity

    # Lamp control
    async def set_lamp_power(self, params: LampParameters) -> bool:
        if len(params.zone_powers_pct) != self.num_lamp_zones:
            raise ValueError(f"Expected {self.num_lamp_zones} zone powers, got {len(params.zone_powers_pct)}")

        logger.info(f"Setting lamp powers: {params.zone_powers_pct}")
        self._lamp_powers = params.zone_powers_pct.copy()
        await asyncio.sleep(0.1)
        return True

    async def get_lamp_power(self) -> Dict[str, float]:
        return {
            f"zone_{i+1}_pct": power
            for i, power in enumerate(self._lamp_powers)
        }

    # Gas control
    async def set_gas_flow(self, params: GasFlowParameters) -> bool:
        logger.info(f"Setting gas flow: {params.gas_type} @ {params.flow_rate_sccm} sccm")
        self._gas_params = params
        self._chamber_pressure_torr = params.chamber_pressure_torr
        await asyncio.sleep(0.2)
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
        logger.info(f"Setting chamber pressure: {pressure_torr} Torr")
        self._chamber_pressure_torr = pressure_torr
        await asyncio.sleep(0.1)
        return True

    async def get_chamber_pressure(self) -> float:
        return self._chamber_pressure_torr

    # Recipe execution
    async def load_recipe(self, recipe: TemperatureRecipe) -> str:
        logger.info(f"Loading recipe: {recipe.recipe_name}")
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

        logger.info(f"Starting recipe execution: {self._current_recipe.recipe_name}")
        self._run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self._recipe_start_time = datetime.now()
        self._current_segment_index = 0
        self.status = RTPStatus.RUNNING_RECIPE

        return self._run_id

    async def pause_recipe(self) -> bool:
        logger.info("Pausing recipe")
        # In real system, would hold current temperature
        await asyncio.sleep(0.05)
        return True

    async def resume_recipe(self) -> bool:
        logger.info("Resuming recipe")
        self.status = RTPStatus.RUNNING_RECIPE
        await asyncio.sleep(0.05)
        return True

    async def stop_recipe(self) -> bool:
        logger.info("Stopping recipe")
        self.status = RTPStatus.COOLING
        self._target_temp_C = 25.0
        self._run_id = None
        await asyncio.sleep(0.05)
        return True

    async def get_recipe_progress(self) -> Dict[str, Any]:
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

        return {
            "is_running": True,
            "run_id": self._run_id,
            "recipe_name": self._current_recipe.recipe_name,
            "current_segment": self._current_segment_index,
            "total_segments": num_segments,
            "elapsed_time_s": elapsed,
            "progress_pct": (self._current_segment_index / num_segments * 100) if num_segments > 0 else 0.0,
            "current_temp_C": self._current_temp_C,
            "target_temp_C": self._target_temp_C
        }

    # Status and diagnostics
    async def get_status(self) -> RTPStatus:
        return self.status

    async def check_interlocks(self) -> Dict[str, bool]:
        """Check all safety interlocks."""
        return {
            "chamber_door": True,
            "cooling_water": True,
            "lamp_overtemp": True,
            "pyrometer_valid": True,
            "pressure_ok": True,
            "e_stop": True
        }


# Export
__all__ = [
    "RTPDriver",
    "RTPMockDriver",
    "AmbientGas",
    "RTPStatus",
    "TemperatureControlMode",
    "RampSegment",
    "GasFlowParameters",
    "LampParameters",
    "EmissivitySettings",
    "TemperatureRecipe"
]
