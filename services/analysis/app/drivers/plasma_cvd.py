"""
Plasma CVD Driver Stubs

Vendor-agnostic base implementations for plasma-enhanced CVD tools:
- PECVD (Plasma-Enhanced CVD) - RF capacitively coupled
- HDP-CVD (High-Density Plasma CVD) - Inductively coupled
- MPCVD (Microwave Plasma CVD) - ECR or 2.45 GHz
- RPCVD (Remote Plasma CVD) - Downstream plasma

These stubs provide the communication framework and can be extended
for specific vendors (Applied Materials, Lam Research, Oxford Instruments, etc.)
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


class PlasmaCVDDriverBase(ABC):
    """
    Base class for plasma CVD drivers

    Implements common functionality for PECVD, HDP, MPCVD, RPCVD tools.
    Adds plasma-specific parameters (RF power, bias, etc.)
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

        # Tool-specific parameters
        self.max_temp_c: float = 400.0  # Plasma CVD is lower temp
        self.min_temp_c: float = 100.0
        self.max_pressure_torr: float = 10.0
        self.min_pressure_torr: float = 0.1

        # Plasma-specific
        self.max_rf_power_w: float = 2000.0
        self.rf_frequency_mhz: float = 13.56  # Standard RF frequency

        logger.info(f"Plasma CVD driver initialized: {tool_id} @ {host}:{port}")

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
            max_wafer_diameter_mm=300,
            max_batch_size=self._get_max_batch_size(),
            available_gas_lines=self._get_available_gas_lines(),
            max_flow_rate_sccm=self._get_max_flow_rates(),
            has_rf_plasma=True,
            has_dc_plasma=self._has_dc_plasma(),
            has_microwave_plasma=self._has_microwave_plasma(),
            rf_frequency_mhz=self.rf_frequency_mhz,
            max_rf_power_w=self.max_rf_power_w,
            has_thickness_monitor=True,
            has_optical_monitor=True,
            has_stress_monitor=True,
            has_oes_monitor=True,  # OES common in plasma tools
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
            self._validate_recipe(recipe)
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
        """Pause process (some plasma tools support this)"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active", error_code="INVALID_RUN_ID")

        logger.info(f"Pausing run {cvd_run_id}")
        try:
            await self._pause_process()
            self.state = ToolState.PAUSED
        except Exception as e:
            raise ToolError(f"Pause failed: {e}", error_code="PAUSE_FAILED")

    async def resume_run(self, cvd_run_id: UUID) -> None:
        """Resume paused process"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active", error_code="INVALID_RUN_ID")

        logger.info(f"Resuming run {cvd_run_id}")
        try:
            await self._resume_process()
            self.state = ToolState.RUNNING
        except Exception as e:
            raise ToolError(f"Resume failed: {e}", error_code="RESUME_FAILED")

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
        """Stream real-time telemetry with plasma parameters"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active", error_code="INVALID_RUN_ID")

        logger.info(f"Starting telemetry stream for run {cvd_run_id}")
        start_time = datetime.utcnow()

        while self.state == ToolState.RUNNING:
            try:
                telemetry_data = await self._read_telemetry()

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
            return {"status": "FAILED", "error": str(e)}

    # =========================================================================
    # Abstract Methods (vendor-specific implementation required)
    # =========================================================================

    @abstractmethod
    async def _establish_connection(self) -> None:
        """Establish connection to tool"""
        pass

    @abstractmethod
    async def _close_connection(self) -> None:
        """Close connection"""
        pass

    @abstractmethod
    async def _send_recipe(self, recipe: Any) -> None:
        """Send recipe to tool"""
        pass

    @abstractmethod
    async def _start_process(self) -> None:
        """Start process"""
        pass

    @abstractmethod
    async def _stop_process(self) -> None:
        """Stop process"""
        pass

    @abstractmethod
    async def _pause_process(self) -> None:
        """Pause process"""
        pass

    @abstractmethod
    async def _resume_process(self) -> None:
        """Resume process"""
        pass

    @abstractmethod
    async def _abort_process(self) -> None:
        """Emergency stop"""
        pass

    @abstractmethod
    async def _read_status(self) -> Dict[str, Any]:
        """Read status"""
        pass

    @abstractmethod
    async def _read_telemetry(self) -> Dict[str, Any]:
        """Read telemetry"""
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

        # Validate plasma-specific parameters
        if hasattr(recipe, 'rf_power_w'):
            if recipe.rf_power_w > self.max_rf_power_w:
                raise ValueError(
                    f"RF power {recipe.rf_power_w}W exceeds max {self.max_rf_power_w}W"
                )

    def _parse_status(self, data: Dict[str, Any]) -> ToolStatus:
        """Parse status data"""
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
        """Parse telemetry data with plasma parameters"""
        return CVDTelemetry(
            cvd_run_id=cvd_run_id,
            timestamp=datetime.utcnow(),
            elapsed_time_sec=elapsed_sec,
            step_number=data.get('step', 1),
            measurements={
                TelemetryType.TEMPERATURE: data.get('temperature_c', 0.0),
                TelemetryType.PRESSURE: data.get('pressure_torr', 0.0),
                TelemetryType.POWER: data.get('rf_power_w', 0.0),
            },
            gas_flows_sccm=data.get('gas_flows', {}),
            rf_forward_power_w=data.get('rf_forward_power_w'),
            rf_reflected_power_w=data.get('rf_reflected_power_w'),
            bias_voltage_v=data.get('bias_voltage_v'),
        )

    def _parse_alarms(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse alarms"""
        return data

    def _parse_diagnostics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse diagnostics"""
        return {
            "status": "HEALTHY",
            "checks": data,
            "timestamp": datetime.utcnow().isoformat(),
        }

    @abstractmethod
    def _get_comm_protocol(self) -> str:
        """Get communication protocol"""
        pass

    @abstractmethod
    def _get_max_batch_size(self) -> int:
        """Get max batch size"""
        pass

    @abstractmethod
    def _get_available_gas_lines(self) -> List[str]:
        """Get available gas lines"""
        pass

    @abstractmethod
    def _get_max_flow_rates(self) -> Dict[str, float]:
        """Get max flow rates"""
        pass

    def _has_dc_plasma(self) -> bool:
        """Whether tool has DC plasma capability"""
        return False

    def _has_microwave_plasma(self) -> bool:
        """Whether tool has microwave plasma"""
        return False


# =============================================================================
# Concrete Driver Implementations (Stubs)
# =============================================================================


class PECVDDriver(PlasmaCVDDriverBase):
    """
    Plasma-Enhanced CVD Driver (Capacitively Coupled)

    13.56 MHz RF plasma, parallel plate reactor.
    Standard for SiO₂, SiN, a-Si, DLC films.
    """

    def __init__(self, tool_id: str, host: str, port: int, vendor: str = "Generic", model: str = "PECVD"):
        super().__init__(tool_id, host, port, vendor, model, "PECVD")
        self.max_temp_c = 400.0
        self.min_temp_c = 150.0
        self.rf_frequency_mhz = 13.56

    def _get_comm_protocol(self) -> str:
        return "SCPI"

    def _get_max_batch_size(self) -> int:
        return 1  # Single wafer

    def _get_available_gas_lines(self) -> List[str]:
        return ["SiH4", "NH3", "N2O", "N2", "Ar", "CF4", "O2"]

    def _get_max_flow_rates(self) -> Dict[str, float]:
        return {
            "SiH4": 500.0,
            "NH3": 2000.0,
            "N2O": 2000.0,
            "N2": 5000.0,
            "Ar": 5000.0,
            "CF4": 200.0,
            "O2": 1000.0,
        }

    # Stub implementations
    async def _establish_connection(self) -> None:
        logger.info(f"[STUB] Establishing SCPI connection to PECVD tool")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info(f"[STUB] Sending recipe to PECVD tool")

    async def _start_process(self) -> None:
        logger.info("[STUB] Starting PECVD process (striking plasma)")

    async def _stop_process(self) -> None:
        logger.info("[STUB] Stopping PECVD process")

    async def _pause_process(self) -> None:
        logger.info("[STUB] Pausing PECVD process")

    async def _resume_process(self) -> None:
        logger.info("[STUB] Resuming PECVD process")

    async def _abort_process(self) -> None:
        logger.warning("[STUB] Aborting PECVD process")

    async def _read_status(self) -> Dict[str, Any]:
        return {
            "temperature_c": 300.0,
            "pressure_torr": 1.5,
            "rf_power_w": 500.0,
            "alarms": [],
        }

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 300.0,
            "pressure_torr": 1.5,
            "rf_power_w": 500.0,
            "rf_forward_power_w": 500.0,
            "rf_reflected_power_w": 5.0,
            "bias_voltage_v": -150.0,
            "step": 1,
            "gas_flows": {"SiH4": 150.0, "N2O": 800.0},
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {
            "rf_generator": "OK",
            "matcher": "OK",
            "chamber": "OK",
            "gas_delivery": "OK",
        }


class HDPCVDDriver(PlasmaCVDDriverBase):
    """
    High-Density Plasma CVD Driver (Inductively Coupled)

    High ion density, excellent gap fill, lower damage.
    Used for advanced dielectrics and metallization.
    """

    def __init__(self, tool_id: str, host: str, port: int, vendor: str = "Generic", model: str = "HDP-CVD"):
        super().__init__(tool_id, host, port, vendor, model, "HDP")
        self.max_temp_c = 400.0
        self.min_temp_c = 200.0
        self.rf_frequency_mhz = 13.56  # Source and bias

    def _get_comm_protocol(self) -> str:
        return "SECS-II"

    def _get_max_batch_size(self) -> int:
        return 1

    def _get_available_gas_lines(self) -> List[str]:
        return ["SiH4", "O2", "Ar", "He", "N2"]

    def _get_max_flow_rates(self) -> Dict[str, float]:
        return {"SiH4": 300.0, "O2": 1000.0, "Ar": 3000.0, "He": 2000.0, "N2": 1000.0}

    # Stub implementations
    async def _establish_connection(self) -> None:
        logger.info(f"[STUB] Establishing SECS-II connection to HDP-CVD tool")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info(f"[STUB] Sending recipe to HDP-CVD tool")

    async def _start_process(self) -> None:
        logger.info("[STUB] Starting HDP-CVD process")

    async def _stop_process(self) -> None:
        logger.info("[STUB] Stopping HDP-CVD process")

    async def _pause_process(self) -> None:
        logger.info("[STUB] Pausing HDP-CVD process")

    async def _resume_process(self) -> None:
        logger.info("[STUB] Resuming HDP-CVD process")

    async def _abort_process(self) -> None:
        logger.warning("[STUB] Aborting HDP-CVD process")

    async def _read_status(self) -> Dict[str, Any]:
        return {
            "temperature_c": 350.0,
            "pressure_torr": 0.01,
            "rf_power_w": 1500.0,
            "alarms": [],
        }

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 350.0,
            "pressure_torr": 0.01,
            "rf_power_w": 1500.0,
            "rf_forward_power_w": 1500.0,
            "rf_reflected_power_w": 10.0,
            "bias_voltage_v": -200.0,
            "step": 1,
            "gas_flows": {"SiH4": 100.0, "O2": 400.0, "Ar": 1000.0},
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {
            "icp_source": "OK",
            "bias_generator": "OK",
            "vacuum": "OK",
        }


class MPCVDDriver(PlasmaCVDDriverBase):
    """
    Microwave Plasma CVD Driver (ECR or 2.45 GHz)

    High ion density, low damage, excellent for diamond/DLC.
    """

    def __init__(self, tool_id: str, host: str, port: int, vendor: str = "Generic", model: str = "MPCVD"):
        super().__init__(tool_id, host, port, vendor, model, "MPCVD")
        self.max_temp_c = 1200.0  # Can go high for diamond
        self.min_temp_c = 400.0
        self.rf_frequency_mhz = 2450.0  # Microwave

    def _has_microwave_plasma(self) -> bool:
        return True

    def _get_comm_protocol(self) -> str:
        return "OPC-UA"

    def _get_max_batch_size(self) -> int:
        return 1

    def _get_available_gas_lines(self) -> List[str]:
        return ["CH4", "H2", "Ar", "N2", "O2"]

    def _get_max_flow_rates(self) -> Dict[str, float]:
        return {"CH4": 500.0, "H2": 3000.0, "Ar": 2000.0, "N2": 1000.0, "O2": 500.0}

    # Stub implementations
    async def _establish_connection(self) -> None:
        logger.info(f"[STUB] Establishing OPC-UA connection to MPCVD tool")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info(f"[STUB] Sending recipe to MPCVD tool")

    async def _start_process(self) -> None:
        logger.info("[STUB] Starting MPCVD process")

    async def _stop_process(self) -> None:
        logger.info("[STUB] Stopping MPCVD process")

    async def _pause_process(self) -> None:
        raise ToolError("Pause not supported for MPCVD", error_code="NOT_SUPPORTED")

    async def _resume_process(self) -> None:
        raise ToolError("Resume not supported for MPCVD", error_code="NOT_SUPPORTED")

    async def _abort_process(self) -> None:
        logger.warning("[STUB] Aborting MPCVD process")

    async def _read_status(self) -> Dict[str, Any]:
        return {
            "temperature_c": 800.0,
            "pressure_torr": 20.0,
            "rf_power_w": 1200.0,
            "alarms": [],
        }

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 800.0,
            "pressure_torr": 20.0,
            "rf_power_w": 1200.0,
            "step": 1,
            "gas_flows": {"CH4": 200.0, "H2": 1500.0},
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {
            "magnetron": "OK",
            "waveguide": "OK",
            "substrate_heater": "OK",
        }


class RPCVDDriver(PlasmaCVDDriverBase):
    """
    Remote Plasma CVD Driver (Downstream Plasma)

    Plasma generated remotely, low damage to substrate.
    Used for sensitive materials and low-k dielectrics.
    """

    def __init__(self, tool_id: str, host: str, port: int, vendor: str = "Generic", model: str = "RPCVD"):
        super().__init__(tool_id, host, port, vendor, model, "RPCVD")
        self.max_temp_c = 300.0
        self.min_temp_c = 100.0

    def _get_comm_protocol(self) -> str:
        return "SCPI"

    def _get_max_batch_size(self) -> int:
        return 1

    def _get_available_gas_lines(self) -> List[str]:
        return ["SiH4", "O2", "N2", "H2", "He"]

    def _get_max_flow_rates(self) -> Dict[str, float]:
        return {"SiH4": 300.0, "O2": 1000.0, "N2": 2000.0, "H2": 1500.0, "He": 2000.0}

    # Stub implementations
    async def _establish_connection(self) -> None:
        logger.info(f"[STUB] Establishing SCPI connection to RPCVD tool")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info(f"[STUB] Sending recipe to RPCVD tool")

    async def _start_process(self) -> None:
        logger.info("[STUB] Starting RPCVD process")

    async def _stop_process(self) -> None:
        logger.info("[STUB] Stopping RPCVD process")

    async def _pause_process(self) -> None:
        logger.info("[STUB] Pausing RPCVD process")

    async def _resume_process(self) -> None:
        logger.info("[STUB] Resuming RPCVD process")

    async def _abort_process(self) -> None:
        logger.warning("[STUB] Aborting RPCVD process")

    async def _read_status(self) -> Dict[str, Any]:
        return {
            "temperature_c": 250.0,
            "pressure_torr": 0.5,
            "rf_power_w": 300.0,
            "alarms": [],
        }

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 250.0,
            "pressure_torr": 0.5,
            "rf_power_w": 300.0,
            "step": 1,
            "gas_flows": {"SiH4": 100.0, "O2": 300.0, "He": 500.0},
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {
            "remote_plasma_source": "OK",
            "chamber": "OK",
            "gas_delivery": "OK",
        }
