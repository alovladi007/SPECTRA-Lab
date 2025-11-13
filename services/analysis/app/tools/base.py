"""
CVD Platform - Unified Tool Abstraction Layer
Base classes and interfaces for all CVD tool variants
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Protocol, Union
from uuid import UUID, uuid4
import asyncio
import logging

import numpy as np


# ============================================================================
# Type Definitions
# ============================================================================

logger = logging.getLogger(__name__)


class ToolState(str, Enum):
    """Tool operational state"""
    OFFLINE = "OFFLINE"
    IDLE = "IDLE"
    INITIALIZING = "INITIALIZING"
    PUMPING_DOWN = "PUMPING_DOWN"
    HEATING = "HEATING"
    STABILIZING = "STABILIZING"
    PROCESSING = "PROCESSING"
    COOLING = "COOLING"
    VENTING = "VENTING"
    ERROR = "ERROR"
    MAINTENANCE = "MAINTENANCE"


class ControlMode(str, Enum):
    """Control mode for tool operation"""
    MANUAL = "MANUAL"
    AUTOMATIC = "AUTOMATIC"
    REMOTE = "REMOTE"


@dataclass
class ToolCapabilities:
    """Describes tool capabilities and constraints"""
    # Pressure capabilities
    min_pressure_pa: float
    max_pressure_pa: float
    pressure_control_resolution_pa: float = 0.1

    # Temperature capabilities
    min_temperature_c: float
    max_temperature_c: float
    temperature_zones: int = 1
    max_ramp_rate_c_per_min: float = 50.0

    # Gas handling
    max_gas_lines: int = 6
    max_total_flow_sccm: float = 10000.0
    available_gases: List[str] = field(default_factory=list)

    # Plasma capabilities (optional)
    has_plasma: bool = False
    plasma_types: List[str] = field(default_factory=list)  # ['RF', 'ICP', 'DC', 'MW']
    max_plasma_power_w: float = 0.0
    plasma_frequency_mhz: List[float] = field(default_factory=list)

    # Mechanical capabilities
    has_rotation: bool = False
    max_rotation_speed_rpm: float = 0.0

    has_tilt: bool = False
    max_tilt_angle_deg: float = 0.0

    # Wafer handling
    wafer_sizes_mm: List[float] = field(default_factory=lambda: [200.0, 300.0])
    max_wafers_per_batch: int = 1
    cassette_loading: bool = False

    # Advanced features
    has_load_lock: bool = False
    has_endpoint_detection: bool = False
    has_in_situ_metrology: bool = False

    # Safety interlocks
    has_gas_leak_detection: bool = True
    has_emergency_stop: bool = True
    has_seismic_sensor: bool = False


@dataclass
class TelemetryPoint:
    """Single telemetry data point"""
    timestamp: datetime
    temperatures: Dict[str, float]  # Zone temperatures
    pressures: Dict[str, float]  # Chamber, foreline pressures
    gas_flows: Dict[str, float]  # Gas flow rates
    plasma_parameters: Optional[Dict[str, float]] = None
    rotation_speed_rpm: Optional[float] = None
    valve_positions: Optional[Dict[str, float]] = None
    heater_powers: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "temperatures": self.temperatures,
            "pressures": self.pressures,
            "gas_flows": self.gas_flows,
            "plasma_parameters": self.plasma_parameters,
            "rotation_speed_rpm": self.rotation_speed_rpm,
            "valve_positions": self.valve_positions,
            "heater_powers": self.heater_powers,
        }


@dataclass
class RecipeValidationResult:
    """Result of recipe validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ToolStatus:
    """Current tool status"""
    tool_id: UUID
    state: ToolState
    control_mode: ControlMode
    current_recipe_id: Optional[UUID] = None
    current_run_id: Optional[UUID] = None
    elapsed_time_s: float = 0.0
    estimated_remaining_s: float = 0.0

    # Current readings
    current_telemetry: Optional[TelemetryPoint] = None

    # Alarms and warnings
    active_alarms: List[Dict[str, Any]] = field(default_factory=list)
    active_warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Interlocks
    interlocks_active: bool = False
    interlock_reasons: List[str] = field(default_factory=list)


# ============================================================================
# Hardware Interface Protocol
# ============================================================================

class HardwareInterface(Protocol):
    """Protocol for hardware communication layer"""

    async def connect(self) -> bool:
        """Establish connection to hardware"""
        ...

    async def disconnect(self) -> bool:
        """Disconnect from hardware"""
        ...

    async def read_register(self, address: int) -> int:
        """Read hardware register"""
        ...

    async def write_register(self, address: int, value: int) -> bool:
        """Write hardware register"""
        ...

    async def send_command(self, command: str) -> str:
        """Send command and get response"""
        ...


# ============================================================================
# Abstract CVD Tool Base Class
# ============================================================================

class CVDToolBase(ABC):
    """
    Abstract base class for all CVD tool implementations.
    Provides unified interface for tool control, recipe execution, and telemetry.
    """

    def __init__(
        self,
        tool_id: UUID,
        tool_name: str,
        capabilities: ToolCapabilities,
        hardware_interface: Optional[HardwareInterface] = None,
        simulation_mode: bool = False,
    ):
        """
        Initialize CVD tool.

        Args:
            tool_id: Unique tool identifier
            tool_name: Tool name/designation
            capabilities: Tool capabilities descriptor
            hardware_interface: Hardware communication interface (None for simulation)
            simulation_mode: If True, run in HIL simulation mode
        """
        self.tool_id = tool_id
        self.tool_name = tool_name
        self.capabilities = capabilities
        self.hardware = hardware_interface
        self.simulation_mode = simulation_mode

        # State
        self._state = ToolState.OFFLINE
        self._control_mode = ControlMode.AUTOMATIC
        self._current_recipe: Optional[Dict[str, Any]] = None
        self._current_run_id: Optional[UUID] = None
        self._start_time: Optional[datetime] = None

        # Telemetry streaming
        self._telemetry_interval_s: float = 1.0
        self._telemetry_subscribers: List[Callable] = []

        # Alarms and interlocks
        self._alarms: List[Dict[str, Any]] = []
        self._warnings: List[Dict[str, Any]] = []
        self._interlocks_active: bool = False
        self._interlock_reasons: List[str] = []

        # Process variables (updated by subclasses)
        self._temperatures: Dict[str, float] = {}
        self._pressures: Dict[str, float] = {}
        self._gas_flows: Dict[str, float] = {}
        self._plasma_parameters: Dict[str, float] = {}
        self._rotation_speed: Optional[float] = None

        logger.info(f"Initialized {tool_name} (ID: {tool_id}, Simulation: {simulation_mode})")

    # ========================================================================
    # Abstract Methods - Must be implemented by subclasses
    # ========================================================================

    @abstractmethod
    async def initialize_hardware(self) -> bool:
        """
        Initialize tool hardware.
        Called when transitioning from OFFLINE to IDLE.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def shutdown_hardware(self) -> bool:
        """
        Safely shutdown tool hardware.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def validate_recipe(self, recipe: Dict[str, Any]) -> RecipeValidationResult:
        """
        Validate recipe against tool capabilities.

        Args:
            recipe: Recipe dictionary

        Returns:
            Validation result with errors/warnings
        """
        pass

    @abstractmethod
    async def execute_recipe_step(self, step: Dict[str, Any]) -> bool:
        """
        Execute a single recipe step.

        Args:
            step: Recipe step dictionary

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def read_telemetry(self) -> TelemetryPoint:
        """
        Read current telemetry from tool.

        Returns:
            Current telemetry point
        """
        pass

    @abstractmethod
    async def emergency_stop(self) -> bool:
        """
        Execute emergency stop sequence.

        Returns:
            True if successful
        """
        pass

    # ========================================================================
    # State Management
    # ========================================================================

    @property
    def state(self) -> ToolState:
        """Get current tool state"""
        return self._state

    async def set_state(self, new_state: ToolState) -> bool:
        """
        Set tool state with validation.

        Args:
            new_state: Desired state

        Returns:
            True if state change successful
        """
        old_state = self._state

        # Validate state transition
        if not self._validate_state_transition(old_state, new_state):
            logger.error(f"Invalid state transition: {old_state} -> {new_state}")
            return False

        self._state = new_state
        logger.info(f"Tool {self.tool_name} state: {old_state} -> {new_state}")

        # Execute state-specific actions
        await self._on_state_change(old_state, new_state)

        return True

    def _validate_state_transition(self, from_state: ToolState, to_state: ToolState) -> bool:
        """Validate if state transition is allowed"""
        # Define valid transitions
        valid_transitions = {
            ToolState.OFFLINE: [ToolState.IDLE, ToolState.INITIALIZING],
            ToolState.IDLE: [ToolState.OFFLINE, ToolState.PUMPING_DOWN, ToolState.MAINTENANCE],
            ToolState.INITIALIZING: [ToolState.IDLE, ToolState.ERROR],
            ToolState.PUMPING_DOWN: [ToolState.HEATING, ToolState.ERROR, ToolState.IDLE],
            ToolState.HEATING: [ToolState.STABILIZING, ToolState.ERROR, ToolState.COOLING],
            ToolState.STABILIZING: [ToolState.PROCESSING, ToolState.ERROR, ToolState.COOLING],
            ToolState.PROCESSING: [ToolState.COOLING, ToolState.ERROR],
            ToolState.COOLING: [ToolState.VENTING, ToolState.ERROR, ToolState.IDLE],
            ToolState.VENTING: [ToolState.IDLE, ToolState.ERROR],
            ToolState.ERROR: [ToolState.IDLE, ToolState.OFFLINE],
            ToolState.MAINTENANCE: [ToolState.IDLE, ToolState.OFFLINE],
        }

        return to_state in valid_transitions.get(from_state, [])

    async def _on_state_change(self, old_state: ToolState, new_state: ToolState):
        """Hook for state change actions"""
        if new_state == ToolState.ERROR:
            await self._handle_error_state()
        elif new_state == ToolState.IDLE and old_state == ToolState.PROCESSING:
            await self._cleanup_after_run()

    # ========================================================================
    # Recipe Execution
    # ========================================================================

    async def execute_recipe(
        self,
        recipe: Dict[str, Any],
        run_id: UUID,
        telemetry_callback: Optional[Callable[[TelemetryPoint], None]] = None,
    ) -> bool:
        """
        Execute complete recipe.

        Args:
            recipe: Recipe dictionary with steps
            run_id: Run identifier
            telemetry_callback: Optional callback for telemetry streaming

        Returns:
            True if successful
        """
        # Validate recipe
        validation = await self.validate_recipe(recipe)
        if not validation.is_valid:
            logger.error(f"Recipe validation failed: {validation.errors}")
            return False

        if validation.warnings:
            logger.warning(f"Recipe warnings: {validation.warnings}")

        # Check tool is in valid state
        if self._state not in [ToolState.IDLE, ToolState.STABILIZING]:
            logger.error(f"Cannot execute recipe in state {self._state}")
            return False

        # Set up run
        self._current_recipe = recipe
        self._current_run_id = run_id
        self._start_time = datetime.utcnow()

        if telemetry_callback:
            self._telemetry_subscribers.append(telemetry_callback)

        # Start telemetry streaming
        telemetry_task = asyncio.create_task(self._telemetry_stream_loop())

        try:
            # Execute recipe steps
            steps = recipe.get("recipe_steps", [])
            for step in steps:
                logger.info(f"Executing step {step.get('step')}: {step.get('name')}")

                # Update state based on step action
                await self._update_state_for_step(step)

                # Execute step
                success = await self.execute_recipe_step(step)
                if not success:
                    logger.error(f"Step {step.get('step')} failed")
                    await self.set_state(ToolState.ERROR)
                    return False

                # Check for interlocks
                if self._interlocks_active:
                    logger.error(f"Interlocks active: {self._interlock_reasons}")
                    await self.emergency_stop()
                    return False

            # Recipe completed successfully
            await self.set_state(ToolState.COOLING)
            logger.info(f"Recipe execution completed successfully for run {run_id}")
            return True

        except Exception as e:
            logger.exception(f"Exception during recipe execution: {e}")
            await self.set_state(ToolState.ERROR)
            return False

        finally:
            # Stop telemetry streaming
            telemetry_task.cancel()
            if telemetry_callback in self._telemetry_subscribers:
                self._telemetry_subscribers.remove(telemetry_callback)

            await self._cleanup_after_run()

    async def _update_state_for_step(self, step: Dict[str, Any]):
        """Update tool state based on step action"""
        action = step.get("action", "").lower()

        state_map = {
            "evacuate": ToolState.PUMPING_DOWN,
            "pumpdown": ToolState.PUMPING_DOWN,
            "ramp_temperature": ToolState.HEATING,
            "heat": ToolState.HEATING,
            "stabilize": ToolState.STABILIZING,
            "deposit": ToolState.PROCESSING,
            "process": ToolState.PROCESSING,
            "cool": ToolState.COOLING,
            "vent": ToolState.VENTING,
        }

        new_state = state_map.get(action)
        if new_state:
            await self.set_state(new_state)

    async def _cleanup_after_run(self):
        """Cleanup after run completion"""
        self._current_recipe = None
        self._current_run_id = None
        self._start_time = None

    # ========================================================================
    # Telemetry Streaming
    # ========================================================================

    async def _telemetry_stream_loop(self):
        """Async loop for continuous telemetry streaming"""
        while True:
            try:
                # Read current telemetry
                telemetry = await self.read_telemetry()

                # Notify subscribers
                for subscriber in self._telemetry_subscribers:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            await subscriber(telemetry)
                        else:
                            subscriber(telemetry)
                    except Exception as e:
                        logger.error(f"Telemetry subscriber error: {e}")

                # Wait for next interval
                await asyncio.sleep(self._telemetry_interval_s)

            except asyncio.CancelledError:
                logger.info("Telemetry streaming stopped")
                break
            except Exception as e:
                logger.error(f"Telemetry streaming error: {e}")
                await asyncio.sleep(self._telemetry_interval_s)

    def set_telemetry_interval(self, interval_s: float):
        """Set telemetry sampling interval"""
        if interval_s < 0.1:
            raise ValueError("Telemetry interval must be >= 0.1 seconds")
        self._telemetry_interval_s = interval_s

    async def stream_telemetry(self) -> AsyncGenerator[TelemetryPoint, None]:
        """
        Async generator for telemetry streaming.

        Yields:
            Telemetry points
        """
        while self._state not in [ToolState.OFFLINE, ToolState.ERROR]:
            telemetry = await self.read_telemetry()
            yield telemetry
            await asyncio.sleep(self._telemetry_interval_s)

    # ========================================================================
    # Alarm and Interlock Management
    # ========================================================================

    def add_alarm(self, severity: str, alarm_type: str, message: str, **kwargs):
        """Add alarm to active alarms list"""
        alarm = {
            "id": str(uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "alarm_type": alarm_type,
            "message": message,
            **kwargs,
        }
        self._alarms.append(alarm)
        logger.warning(f"ALARM [{severity}]: {message}")

        # Activate interlock for critical alarms
        if severity == "CRITICAL":
            self.activate_interlock(message)

    def clear_alarm(self, alarm_id: str):
        """Clear alarm by ID"""
        self._alarms = [a for a in self._alarms if a["id"] != alarm_id]

    def activate_interlock(self, reason: str):
        """Activate safety interlock"""
        self._interlocks_active = True
        if reason not in self._interlock_reasons:
            self._interlock_reasons.append(reason)
        logger.critical(f"INTERLOCK ACTIVATED: {reason}")

    def clear_interlock(self, reason: str):
        """Clear specific interlock"""
        if reason in self._interlock_reasons:
            self._interlock_reasons.remove(reason)

        if not self._interlock_reasons:
            self._interlocks_active = False
            logger.info("All interlocks cleared")

    async def _handle_error_state(self):
        """Handle transition to error state"""
        # Safe shutdown sequence
        logger.error("Entering error state - executing safe shutdown")
        # Subclasses should override to implement specific shutdown

    # ========================================================================
    # Status and Monitoring
    # ========================================================================

    def get_status(self) -> ToolStatus:
        """Get current tool status"""
        elapsed = 0.0
        remaining = 0.0

        if self._start_time and self._current_recipe:
            elapsed = (datetime.utcnow() - self._start_time).total_seconds()
            total_time = self._current_recipe.get("process_time_s", 0)
            remaining = max(0, total_time - elapsed)

        return ToolStatus(
            tool_id=self.tool_id,
            state=self._state,
            control_mode=self._control_mode,
            current_recipe_id=self._current_recipe.get("id") if self._current_recipe else None,
            current_run_id=self._current_run_id,
            elapsed_time_s=elapsed,
            estimated_remaining_s=remaining,
            current_telemetry=None,  # Call read_telemetry() separately
            active_alarms=self._alarms.copy(),
            active_warnings=self._warnings.copy(),
            interlocks_active=self._interlocks_active,
            interlock_reasons=self._interlock_reasons.copy(),
        )

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def check_capability(self, capability: str) -> bool:
        """Check if tool has specific capability"""
        return getattr(self.capabilities, capability, False)

    def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return self.capabilities

    def __repr__(self) -> str:
        return f"CVDTool(id={self.tool_id}, name={self.tool_name}, state={self._state})"


# ============================================================================
# Tool Manager
# ============================================================================

class CVDToolManager:
    """
    Manages multiple CVD tool instances.
    Provides tool discovery, allocation, and monitoring.
    """

    def __init__(self):
        self._tools: Dict[UUID, CVDToolBase] = {}
        self._tool_name_index: Dict[str, UUID] = {}
        logger.info("CVD Tool Manager initialized")

    def register_tool(self, tool: CVDToolBase):
        """Register a tool with the manager"""
        self._tools[tool.tool_id] = tool
        self._tool_name_index[tool.tool_name] = tool.tool_id
        logger.info(f"Registered tool: {tool.tool_name} ({tool.tool_id})")

    def unregister_tool(self, tool_id: UUID):
        """Unregister a tool"""
        if tool_id in self._tools:
            tool = self._tools[tool_id]
            del self._tools[tool_id]
            del self._tool_name_index[tool.tool_name]
            logger.info(f"Unregistered tool: {tool.tool_name}")

    def get_tool(self, tool_id: UUID) -> Optional[CVDToolBase]:
        """Get tool by ID"""
        return self._tools.get(tool_id)

    def get_tool_by_name(self, tool_name: str) -> Optional[CVDToolBase]:
        """Get tool by name"""
        tool_id = self._tool_name_index.get(tool_name)
        return self._tools.get(tool_id) if tool_id else None

    def list_tools(self, filter_state: Optional[ToolState] = None) -> List[CVDToolBase]:
        """List all tools, optionally filtered by state"""
        tools = list(self._tools.values())

        if filter_state:
            tools = [t for t in tools if t.state == filter_state]

        return tools

    def get_available_tools(self) -> List[CVDToolBase]:
        """Get tools available for processing (IDLE state)"""
        return self.list_tools(filter_state=ToolState.IDLE)

    async def shutdown_all(self):
        """Shutdown all registered tools"""
        logger.info("Shutting down all tools...")
        for tool in self._tools.values():
            try:
                await tool.shutdown_hardware()
            except Exception as e:
                logger.error(f"Error shutting down {tool.tool_name}: {e}")
