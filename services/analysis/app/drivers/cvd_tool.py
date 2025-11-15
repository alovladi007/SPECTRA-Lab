"""
CVD Tool Protocol Interface

Defines the standard interface for all CVD tools (real hardware and simulators).
This abstraction enables vendor-agnostic process control and monitoring.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, AsyncIterator, Optional, Dict, Any, List
from uuid import UUID


class ToolState(str, Enum):
    """Tool operational state"""
    OFFLINE = "OFFLINE"  # Tool not connected/powered
    IDLE = "IDLE"  # Ready for operation
    CONFIGURING = "CONFIGURING"  # Loading recipe
    RUNNING = "RUNNING"  # Process executing
    PAUSED = "PAUSED"  # Process paused
    STOPPING = "STOPPING"  # Controlled shutdown
    ERROR = "ERROR"  # Fault condition
    MAINTENANCE = "MAINTENANCE"  # Under maintenance


class TelemetryType(str, Enum):
    """Types of telemetry data"""
    TEMPERATURE = "TEMPERATURE"  # °C
    PRESSURE = "PRESSURE"  # Torr or Pa
    FLOW_RATE = "FLOW_RATE"  # sccm
    POWER = "POWER"  # W (RF/DC/Filament)
    VOLTAGE = "VOLTAGE"  # V
    CURRENT = "CURRENT"  # A
    FREQUENCY = "FREQUENCY"  # MHz (RF)
    THICKNESS = "THICKNESS"  # nm (in-situ)
    DEPOSITION_RATE = "DEPOSITION_RATE"  # nm/min
    STRESS = "STRESS"  # MPa (in-situ)
    REFLECTANCE = "REFLECTANCE"  # % (optical monitoring)
    EMISSION = "EMISSION"  # a.u. (OES)
    VALVE_POSITION = "VALVE_POSITION"  # %
    ROTATION_SPEED = "ROTATION_SPEED"  # RPM


@dataclass
class ToolCapabilities:
    """
    Advertises what this tool can do

    Used by schedulers and process engineers to understand tool capabilities
    without vendor-specific knowledge.
    """
    tool_id: str
    vendor: str
    model: str

    # Supported CVD modes (from CVDProcessMode.mode_name)
    supported_modes: List[str]  # e.g., ["LPCVD", "UHVCVD"]

    # Temperature range
    min_temp_c: float
    max_temp_c: float

    # Pressure range
    min_pressure_torr: float
    max_pressure_torr: float

    # Chamber specs
    max_wafer_diameter_mm: int  # 100, 150, 200, 300
    max_batch_size: int  # Number of wafers

    # Gas delivery
    available_gas_lines: List[str]  # e.g., ["SiH4", "NH3", "N2", "H2"]
    max_flow_rate_sccm: Dict[str, float]  # Per gas line

    # Energy source capabilities
    has_rf_plasma: bool = False
    has_dc_plasma: bool = False
    has_microwave_plasma: bool = False
    has_filament_heater: bool = False
    rf_frequency_mhz: Optional[float] = None
    max_rf_power_w: Optional[float] = None

    # In-situ metrology
    has_thickness_monitor: bool = False
    has_optical_monitor: bool = False
    has_stress_monitor: bool = False
    has_oes_monitor: bool = False  # Optical Emission Spectroscopy

    # Communication protocol
    comm_protocol: str = "SCPI"  # SCPI, OPC-UA, SECS-II, etc.

    # Additional metadata
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None
    installation_date: Optional[datetime] = None
    last_pm_date: Optional[datetime] = None  # Preventive maintenance


@dataclass
class ToolStatus:
    """Current status of the tool"""
    state: ToolState
    cvd_run_id: Optional[UUID] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    elapsed_time_sec: Optional[float] = None
    estimated_remaining_sec: Optional[float] = None

    # Latest critical parameters
    chamber_temp_c: Optional[float] = None
    chamber_pressure_torr: Optional[float] = None

    # Alarms and warnings
    active_alarms: List[str] = field(default_factory=list)
    active_warnings: List[str] = field(default_factory=list)

    # Error info (if state=ERROR)
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CVDTelemetry:
    """
    Real-time telemetry data point

    Emitted during process runs for monitoring, SPC, and FDC.
    """
    cvd_run_id: UUID
    timestamp: datetime
    elapsed_time_sec: float

    # Process step info
    step_number: int
    step_name: Optional[str] = None

    # Telemetry measurements
    # Key = parameter name, Value = measurement
    measurements: Dict[TelemetryType, float] = field(default_factory=dict)

    # Optional: Zone-specific measurements (multi-zone furnaces)
    zone_temps_c: Optional[Dict[str, float]] = None  # {"zone1": 850.0, "zone2": 900.0}
    zone_powers_w: Optional[Dict[str, float]] = None

    # Optional: Gas flow rates by line
    gas_flows_sccm: Optional[Dict[str, float]] = None  # {"SiH4": 120.0, "NH3": 500.0}

    # Optional: In-situ metrology
    thickness_nm: Optional[float] = None
    deposition_rate_nm_min: Optional[float] = None
    stress_mpa: Optional[float] = None
    reflectance_pct: Optional[float] = None

    # Optional: Plasma parameters
    rf_forward_power_w: Optional[float] = None
    rf_reflected_power_w: Optional[float] = None
    bias_voltage_v: Optional[float] = None

    # Raw data for advanced analysis
    raw_data: Optional[Dict[str, Any]] = None


class ToolError(Exception):
    """Base exception for tool-related errors"""
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class CVDTool(Protocol):
    """
    Protocol interface for CVD tools

    All real drivers and HIL simulators must implement this interface.
    This enables vendor-agnostic process execution and monitoring.

    Example implementations:
    - ThermalCVDDriver (APCVD, LPCVD, UHVCVD)
    - PlasmaCVDDriver (PECVD, HDP, MPCVD, RPCVD)
    - MOCVDDriver
    - HILSimulator (physics-based simulation)
    """

    async def get_capabilities(self) -> ToolCapabilities:
        """
        Get tool capabilities

        Returns:
            ToolCapabilities describing what this tool can do
        """
        ...

    async def connect(self) -> None:
        """
        Establish connection to the tool

        Raises:
            ToolError: If connection fails
        """
        ...

    async def disconnect(self) -> None:
        """
        Disconnect from the tool
        """
        ...

    async def configure(self, recipe: Any) -> None:  # recipe: CVDRecipe
        """
        Load and validate a recipe

        Args:
            recipe: CVDRecipe object containing process parameters

        Raises:
            ToolError: If recipe is invalid or incompatible with tool
        """
        ...

    async def start_run(self, cvd_run_id: UUID) -> None:
        """
        Start executing the configured recipe

        Args:
            cvd_run_id: Unique identifier for this run

        Raises:
            ToolError: If no recipe configured or tool not ready
        """
        ...

    async def stop_run(self, cvd_run_id: UUID) -> None:
        """
        Stop the current run (controlled shutdown)

        Args:
            cvd_run_id: Run to stop
        """
        ...

    async def pause_run(self, cvd_run_id: UUID) -> None:
        """
        Pause the current run (if supported)

        Args:
            cvd_run_id: Run to pause

        Raises:
            ToolError: If pause not supported
        """
        ...

    async def resume_run(self, cvd_run_id: UUID) -> None:
        """
        Resume a paused run

        Args:
            cvd_run_id: Run to resume
        """
        ...

    async def abort_run(self, cvd_run_id: UUID) -> None:
        """
        Emergency stop (immediate shutdown)

        Args:
            cvd_run_id: Run to abort
        """
        ...

    async def get_status(self, cvd_run_id: Optional[UUID] = None) -> ToolStatus:
        """
        Get current tool status

        Args:
            cvd_run_id: Optional run ID to get status for

        Returns:
            ToolStatus with current state and parameters
        """
        ...

    async def stream_telemetry(
        self,
        cvd_run_id: UUID,
        interval_sec: float = 1.0
    ) -> AsyncIterator[CVDTelemetry]:
        """
        Stream real-time telemetry during a run

        Args:
            cvd_run_id: Run to monitor
            interval_sec: Sampling interval in seconds

        Yields:
            CVDTelemetry data points

        Example:
            async for telemetry in tool.stream_telemetry(run_id):
                await process_telemetry(telemetry)
        """
        ...

    async def get_alarms(self) -> List[Dict[str, Any]]:
        """
        Get active alarms and warnings

        Returns:
            List of alarm dictionaries with code, severity, message, timestamp
        """
        ...

    async def clear_alarms(self) -> None:
        """
        Clear acknowledged alarms
        """
        ...

    async def run_diagnostics(self) -> Dict[str, Any]:
        """
        Run tool self-diagnostics

        Returns:
            Diagnostic results with health check status
        """
        ...
