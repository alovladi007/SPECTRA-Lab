"""
Thermal CVD Driver Stubs

Vendor-agnostic base implementations for thermal CVD tools:
- APCVD (Atmospheric Pressure CVD)
- LPCVD (Low Pressure CVD)
- UHVCVD (Ultra-High Vacuum CVD)

These stubs provide the communication framework and can be extended
for specific vendors (Applied Materials, ASM, Centrotherm, etc.)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Optional, Dict, Any, List
from uuid import UUID

from .cvd_tool import (
    CVDTool,
    ToolState,
    ToolStatus,
    ToolCapabilities,
    CVDTelemetry,
    TelemetryType,
    ToolError,
)

logger = logging.getLogger(__name__)


class ThermalCVDDriverBase(ABC):
    """
    Base class for thermal CVD drivers

    Implements common functionality for APCVD, LPCVD, UHVCVD tools.
    Subclasses implement vendor-specific communication protocols.
    """

    def __init__(
        self,
        tool_id: str,
        host: str,
        port: int,
        vendor: str,
        model: str,
        mode: str,
    ):
        self.tool_id = tool_id
        self.host = host
        self.port = port
        self.vendor = vendor
        self.model = model
        self.mode = mode

        self.connected = False
        self.state = ToolState.OFFLINE
        self.current_recipe: Optional[Any] = None
        self.current_run_id: Optional[UUID] = None

        # Tool-specific parameters (to be set by subclass)
        self.max_temp_c: float = 1200.0
        self.min_temp_c: float = 400.0
        self.max_pressure_torr: float = 1000.0
        self.min_pressure_torr: float = 0.001

        logger.info(f"Thermal CVD driver initialized: {tool_id} @ {host}:{port}")

    # =========================================================================
    # CVDTool Protocol Implementation
    # =========================================================================

    async def get_capabilities(self) -> ToolCapabilities:
        """Get tool capabilities"""
        return ToolCapabilities(
            tool_id=self.tool_id,
            vendor=self.vendor,
            model=self.model,
            supported_modes=[self.mode],
            min_temp_c=self.min_temp_c,
            max_temp_c=self.max_temp_c,
            min_pressure_torr=self.min_pressure_torr,
            max_pressure_torr=self.max_pressure_torr,
            max_wafer_diameter_mm=200,
            max_batch_size=self._get_max_batch_size(),
            available_gas_lines=self._get_available_gas_lines(),
            max_flow_rate_sccm=self._get_max_flow_rates(),
            has_rf_plasma=False,  # Thermal CVD has no plasma
            has_thickness_monitor=True,
            has_optical_monitor=True,
            comm_protocol=self._get_comm_protocol(),
        )

    async def connect(self) -> None:
        """Establish connection to the tool"""
        logger.info(f"Connecting to {self.vendor} {self.model} at {self.host}:{self.port}")

        try:
            await self._establish_connection()
            self.connected = True
            self.state = ToolState.IDLE
            logger.info(f"Connected to {self.tool_id}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise ToolError(f"Connection failed: {e}", error_code="CONN_FAILED")

    async def disconnect(self) -> None:
        """Disconnect from the tool"""
        logger.info(f"Disconnecting from {self.tool_id}")
        await self._close_connection()
        self.connected = False
        self.state = ToolState.OFFLINE

    async def configure(self, recipe: Any) -> None:
        """Load recipe into tool"""
        if not self.connected:
            raise ToolError("Not connected to tool", error_code="NOT_CONNECTED")

        if self.state not in [ToolState.IDLE, ToolState.ERROR]:
            raise ToolError(f"Cannot configure in state {self.state}", error_code="INVALID_STATE")

        logger.info(f"Configuring recipe: {recipe.recipe_name if hasattr(recipe, 'recipe_name') else 'Unknown'}")
        self.state = ToolState.CONFIGURING

        try:
            # Validate recipe
            self._validate_recipe(recipe)

            # Send recipe to tool
            await self._send_recipe(recipe)

            self.current_recipe = recipe
            self.state = ToolState.IDLE
            logger.info("Recipe configured successfully")
        except Exception as e:
            self.state = ToolState.ERROR
            logger.error(f"Recipe configuration failed: {e}")
            raise ToolError(f"Recipe configuration failed: {e}", error_code="CONFIG_FAILED")

    async def start_run(self, cvd_run_id: UUID) -> None:
        """Start process execution"""
        if self.state != ToolState.IDLE:
            raise ToolError(f"Cannot start run in state {self.state}", error_code="INVALID_STATE")

        if not self.current_recipe:
            raise ToolError("No recipe configured", error_code="NO_RECIPE")

        logger.info(f"Starting run {cvd_run_id}")
        self.current_run_id = cvd_run_id

        try:
            await self._start_process()
            self.state = ToolState.RUNNING
            logger.info(f"Run {cvd_run_id} started")
        except Exception as e:
            self.state = ToolState.ERROR
            logger.error(f"Failed to start run: {e}")
            raise ToolError(f"Failed to start run: {e}", error_code="START_FAILED")

    async def stop_run(self, cvd_run_id: UUID) -> None:
        """Stop process (controlled shutdown)"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active", error_code="INVALID_RUN_ID")

        logger.info(f"Stopping run {cvd_run_id}")
        self.state = ToolState.STOPPING

        try:
            await self._stop_process()
            self.state = ToolState.IDLE
            self.current_run_id = None
            logger.info(f"Run {cvd_run_id} stopped")
        except Exception as e:
            logger.error(f"Failed to stop run: {e}")
            raise ToolError(f"Failed to stop run: {e}", error_code="STOP_FAILED")

    async def pause_run(self, cvd_run_id: UUID) -> None:
        """Pause process (if supported)"""
        raise ToolError("Pause not supported for thermal CVD", error_code="NOT_SUPPORTED")

    async def resume_run(self, cvd_run_id: UUID) -> None:
        """Resume paused process"""
        raise ToolError("Resume not supported for thermal CVD", error_code="NOT_SUPPORTED")

    async def abort_run(self, cvd_run_id: UUID) -> None:
        """Emergency stop"""
        logger.warning(f"ABORTING run {cvd_run_id}")

        try:
            await self._abort_process()
            self.state = ToolState.ERROR
            self.current_run_id = None
        except Exception as e:
            logger.error(f"Abort failed: {e}")
            raise ToolError(f"Abort failed: {e}", error_code="ABORT_FAILED")

    async def get_status(self, cvd_run_id: Optional[UUID] = None) -> ToolStatus:
        """Get current tool status"""
        if not self.connected:
            return ToolStatus(state=ToolState.OFFLINE)

        try:
            status_data = await self._read_status()
            return self._parse_status(status_data)
        except Exception as e:
            logger.error(f"Failed to read status: {e}")
            return ToolStatus(
                state=ToolState.ERROR,
                error_message=str(e),
            )

    async def stream_telemetry(
        self,
        cvd_run_id: UUID,
        interval_sec: float = 1.0
    ) -> AsyncIterator[CVDTelemetry]:
        """Stream real-time telemetry"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active", error_code="INVALID_RUN_ID")

        logger.info(f"Starting telemetry stream for run {cvd_run_id}")
        start_time = datetime.utcnow()

        while self.state == ToolState.RUNNING:
            try:
                # Read telemetry from tool
                telemetry_data = await self._read_telemetry()

                # Parse and yield telemetry
                elapsed = (datetime.utcnow() - start_time).total_seconds()
                telemetry = self._parse_telemetry(cvd_run_id, telemetry_data, elapsed)
                yield telemetry

                await asyncio.sleep(interval_sec)
            except Exception as e:
                logger.error(f"Telemetry read failed: {e}")
                break

        logger.info(f"Telemetry stream ended for run {cvd_run_id}")

    async def get_alarms(self) -> List[Dict[str, Any]]:
        """Get active alarms"""
        try:
            alarms_data = await self._read_alarms()
            return self._parse_alarms(alarms_data)
        except Exception as e:
            logger.error(f"Failed to read alarms: {e}")
            return []

    async def clear_alarms(self) -> None:
        """Clear acknowledged alarms"""
        try:
            await self._clear_alarms()
        except Exception as e:
            logger.error(f"Failed to clear alarms: {e}")
            raise ToolError(f"Failed to clear alarms: {e}", error_code="CLEAR_FAILED")

    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run tool self-diagnostics"""
        try:
            diag_data = await self._run_diagnostics()
            return self._parse_diagnostics(diag_data)
        except Exception as e:
            logger.error(f"Diagnostics failed: {e}")
            return {
                "status": "FAILED",
                "error": str(e),
            }

    # =========================================================================
    # Abstract Methods (vendor-specific implementation required)
    # =========================================================================

    @abstractmethod
    async def _establish_connection(self) -> None:
        """Establish connection to tool (vendor-specific)"""
        pass

    @abstractmethod
    async def _close_connection(self) -> None:
        """Close connection to tool"""
        pass

    @abstractmethod
    async def _send_recipe(self, recipe: Any) -> None:
        """Send recipe to tool"""
        pass

    @abstractmethod
    async def _start_process(self) -> None:
        """Start process execution"""
        pass

    @abstractmethod
    async def _stop_process(self) -> None:
        """Stop process"""
        pass

    @abstractmethod
    async def _abort_process(self) -> None:
        """Emergency stop"""
        pass

    @abstractmethod
    async def _read_status(self) -> Dict[str, Any]:
        """Read current status from tool"""
        pass

    @abstractmethod
    async def _read_telemetry(self) -> Dict[str, Any]:
        """Read telemetry data"""
        pass

    @abstractmethod
    async def _read_alarms(self) -> List[Dict[str, Any]]:
        """Read alarms"""
        pass

    @abstractmethod
    async def _clear_alarms(self) -> None:
        """Clear alarms"""
        pass

    @abstractmethod
    async def _run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics"""
        pass

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _validate_recipe(self, recipe: Any) -> None:
        """Validate recipe against tool capabilities"""
        if hasattr(recipe, 'target_temp_c'):
            if not (self.min_temp_c <= recipe.target_temp_c <= self.max_temp_c):
                raise ValueError(
                    f"Temperature {recipe.target_temp_c}°C outside range "
                    f"[{self.min_temp_c}, {self.max_temp_c}]"
                )

        if hasattr(recipe, 'target_pressure_torr'):
            if not (self.min_pressure_torr <= recipe.target_pressure_torr <= self.max_pressure_torr):
                raise ValueError(
                    f"Pressure {recipe.target_pressure_torr} Torr outside range "
                    f"[{self.min_pressure_torr}, {self.max_pressure_torr}]"
                )

    def _parse_status(self, data: Dict[str, Any]) -> ToolStatus:
        """Parse status data into ToolStatus object"""
        return ToolStatus(
            state=self.state,
            cvd_run_id=self.current_run_id,
            chamber_temp_c=data.get('temperature_c'),
            chamber_pressure_torr=data.get('pressure_torr'),
            active_alarms=data.get('alarms', []),
            timestamp=datetime.utcnow(),
        )

    def _parse_telemetry(
        self,
        cvd_run_id: UUID,
        data: Dict[str, Any],
        elapsed_sec: float,
    ) -> CVDTelemetry:
        """Parse telemetry data into CVDTelemetry object"""
        return CVDTelemetry(
            cvd_run_id=cvd_run_id,
            timestamp=datetime.utcnow(),
            elapsed_time_sec=elapsed_sec,
            step_number=data.get('step', 1),
            measurements={
                TelemetryType.TEMPERATURE: data.get('temperature_c', 0.0),
                TelemetryType.PRESSURE: data.get('pressure_torr', 0.0),
            },
            gas_flows_sccm=data.get('gas_flows', {}),
        )

    def _parse_alarms(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse alarm data"""
        return data

    def _parse_diagnostics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse diagnostics data"""
        return {
            "status": "HEALTHY",
            "checks": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @abstractmethod
    def _get_comm_protocol(self) -> str:
        """Get communication protocol name"""
        pass

    @abstractmethod
    def _get_max_batch_size(self) -> int:
        """Get maximum batch size"""
        pass

    @abstractmethod
    def _get_available_gas_lines(self) -> List[str]:
        """Get available gas lines"""
        pass

    @abstractmethod
    def _get_max_flow_rates(self) -> Dict[str, float]:
        """Get maximum flow rates per gas"""
        pass


# =============================================================================
# Concrete Driver Implementations (Stubs)
# =============================================================================


class APCVDDriver(ThermalCVDDriverBase):
    """
    Atmospheric Pressure CVD Driver

    For tools operating at ~760 Torr (1 atm).
    Common for oxide and nitride films.
    """

    def __init__(self, tool_id: str, host: str, port: int, vendor: str = "Generic", model: str = "APCVD"):
        super().__init__(tool_id, host, port, vendor, model, "APCVD")
        self.max_pressure_torr = 1000.0
        self.min_pressure_torr = 500.0

    def _get_comm_protocol(self) -> str:
        return "SCPI"

    def _get_max_batch_size(self) -> int:
        return 25

    def _get_available_gas_lines(self) -> List[str]:
        return ["SiH4", "O2", "N2", "NH3"]

    def _get_max_flow_rates(self) -> Dict[str, float]:
        return {"SiH4": 500.0, "O2": 2000.0, "N2": 5000.0, "NH3": 2000.0}

    # Stub implementations
    async def _establish_connection(self) -> None:
        logger.info(f"[STUB] Establishing SCPI connection to {self.host}:{self.port}")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info(f"[STUB] Sending recipe to APCVD tool")

    async def _start_process(self) -> None:
        logger.info("[STUB] Starting APCVD process")

    async def _stop_process(self) -> None:
        logger.info("[STUB] Stopping APCVD process")

    async def _abort_process(self) -> None:
        logger.warning("[STUB] Aborting APCVD process")

    async def _read_status(self) -> Dict[str, Any]:
        return {"temperature_c": 850.0, "pressure_torr": 760.0, "alarms": []}

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 850.0,
            "pressure_torr": 760.0,
            "step": 1,
            "gas_flows": {"SiH4": 120.0, "NH3": 500.0},
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {"chamber": "OK", "heater": "OK", "gas_delivery": "OK"}


class LPCVDDriver(ThermalCVDDriverBase):
    """
    Low Pressure CVD Driver

    For tools operating at 0.1-10 Torr.
    Excellent uniformity and conformality.
    """

    def __init__(self, tool_id: str, host: str, port: int, vendor: str = "Generic", model: str = "LPCVD"):
        super().__init__(tool_id, host, port, vendor, model, "LPCVD")
        self.max_pressure_torr = 10.0
        self.min_pressure_torr = 0.1

    def _get_comm_protocol(self) -> str:
        return "OPC-UA"

    def _get_max_batch_size(self) -> int:
        return 100  # Batch furnaces can handle many wafers

    def _get_available_gas_lines(self) -> List[str]:
        return ["SiH4", "NH3", "N2", "O2", "DCS"]  # DCS = Dichlorosilane

    def _get_max_flow_rates(self) -> Dict[str, float]:
        return {"SiH4": 200.0, "NH3": 500.0, "N2": 1000.0, "O2": 500.0, "DCS": 300.0}

    # Stub implementations
    async def _establish_connection(self) -> None:
        logger.info(f"[STUB] Establishing OPC-UA connection to {self.host}:{self.port}")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing OPC-UA connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info(f"[STUB] Sending recipe to LPCVD tool")

    async def _start_process(self) -> None:
        logger.info("[STUB] Starting LPCVD process")

    async def _stop_process(self) -> None:
        logger.info("[STUB] Stopping LPCVD process")

    async def _abort_process(self) -> None:
        logger.warning("[STUB] Aborting LPCVD process")

    async def _read_status(self) -> Dict[str, Any]:
        return {"temperature_c": 780.0, "pressure_torr": 0.3, "alarms": []}

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 780.0,
            "pressure_torr": 0.3,
            "step": 1,
            "gas_flows": {"SiH4": 80.0, "NH3": 200.0},
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {"furnace": "OK", "vacuum": "OK", "gas_delivery": "OK"}


class UHVCVDDriver(ThermalCVDDriverBase):
    """
    Ultra-High Vacuum CVD Driver

    For tools operating at <1e-6 Torr.
    Ultra-clean deposition for advanced devices.
    """

    def __init__(self, tool_id: str, host: str, port: int, vendor: str = "Generic", model: str = "UHVCVD"):
        super().__init__(tool_id, host, port, vendor, model, "UHVCVD")
        self.max_pressure_torr = 0.01
        self.min_pressure_torr = 1e-7

    def _get_comm_protocol(self) -> str:
        return "SECS-II"

    def _get_max_batch_size(self) -> int:
        return 1  # Single-wafer processing

    def _get_available_gas_lines(self) -> List[str]:
        return ["SiH4", "GeH4", "H2", "N2"]  # GeH4 for SiGe

    def _get_max_flow_rates(self) -> Dict[str, float]:
        return {"SiH4": 100.0, "GeH4": 50.0, "H2": 500.0, "N2": 200.0}

    # Stub implementations
    async def _establish_connection(self) -> None:
        logger.info(f"[STUB] Establishing SECS-II connection to {self.host}:{self.port}")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing SECS-II connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info(f"[STUB] Sending recipe to UHVCVD tool")

    async def _start_process(self) -> None:
        logger.info("[STUB] Starting UHVCVD process")

    async def _stop_process(self) -> None:
        logger.info("[STUB] Stopping UHVCVD process")

    async def _abort_process(self) -> None:
        logger.warning("[STUB] Aborting UHVCVD process")

    async def _read_status(self) -> Dict[str, Any]:
        return {"temperature_c": 550.0, "pressure_torr": 1e-6, "alarms": []}

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 550.0,
            "pressure_torr": 1e-6,
            "step": 1,
            "gas_flows": {"SiH4": 50.0, "H2": 200.0},
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {"uhv_chamber": "OK", "turbo_pump": "OK", "load_lock": "OK"}
