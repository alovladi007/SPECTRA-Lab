"""
Specialty CVD Driver Stubs

Vendor-agnostic base implementations for specialty CVD techniques:
- MOCVD (Metal-Organic CVD) - For III-V compounds (GaN, GaAs, InP)
- AACVD (Aerosol-Assisted CVD) - For nanomaterials and complex oxides

These techniques require specialized precursor delivery and temperature control.
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


class MOCVDDriver:
    """
    Metal-Organic CVD Driver

    For epitaxial growth of III-V compounds (GaN, GaAs, InP, AlGaN, etc.)
    and II-VI compounds (ZnO, CdS, CdTe).

    Uses metal-organic precursors (TMGa, TEGa, TMAl, TMIn, etc.)
    with hydrides (NH₃, AsH₃, PH₃).

    Critical parameters:
    - Temperature: 500-1200°C
    - Pressure: 50-760 Torr
    - V/III ratio (e.g., NH₃/TMGa ratio for GaN)
    - Carrier gas flow (H₂, N₂)
    """

    def __init__(
        self,
        tool_id: str,
        host: str,
        port: int,
        vendor: str = "Generic",
        model: str = "MOCVD",
    ):
        self.tool_id = tool_id
        self.host = host
        self.port = port
        self.vendor = vendor
        self.model = model

        self.connected = False
        self.state = ToolState.OFFLINE
        self.current_recipe: Optional[Any] = None
        self.current_run_id: Optional[UUID] = None

        # MOCVD-specific parameters
        self.max_temp_c = 1200.0
        self.min_temp_c = 500.0
        self.max_pressure_torr = 760.0
        self.min_pressure_torr = 50.0

        logger.info(f"MOCVD driver initialized: {tool_id} @ {host}:{port}")

    # =========================================================================
    # CVDTool Protocol Implementation
    # =========================================================================

    async def get_capabilities(self) -> ToolCapabilities:
        """Get MOCVD tool capabilities"""
        return ToolCapabilities(
            tool_id=self.tool_id,
            vendor=self.vendor,
            model=self.model,
            supported_modes=["MOCVD"],
            min_temp_c=self.min_temp_c,
            max_temp_c=self.max_temp_c,
            min_pressure_torr=self.min_pressure_torr,
            max_pressure_torr=self.max_pressure_torr,
            max_wafer_diameter_mm=150,
            max_batch_size=1,  # Single wafer epitaxy
            available_gas_lines=[
                # Metal-organic precursors
                "TMGa",  # Trimethylgallium
                "TEGa",  # Triethylgallium
                "TMAl",  # Trimethylaluminum
                "TMIn",  # Trimethylindium
                # Hydrides
                "NH3",  # Ammonia
                "AsH3",  # Arsine
                "PH3",  # Phosphine
                # Dopants
                "SiH4",  # n-type dopant
                "Cp2Mg",  # Bis(cyclopentadienyl)magnesium (p-type)
                # Carrier gases
                "H2",
                "N2",
            ],
            max_flow_rate_sccm={
                "TMGa": 100.0,
                "TEGa": 100.0,
                "TMAl": 100.0,
                "TMIn": 100.0,
                "NH3": 10000.0,
                "AsH3": 500.0,
                "PH3": 500.0,
                "SiH4": 50.0,
                "Cp2Mg": 50.0,
                "H2": 20000.0,
                "N2": 20000.0,
            },
            has_rf_plasma=False,  # Thermal MOCVD
            has_thickness_monitor=True,  # In-situ reflectometry
            has_optical_monitor=True,
            has_stress_monitor=True,  # Wafer curvature
            comm_protocol="OPC-UA",
        )

    async def connect(self) -> None:
        """Connect to MOCVD tool"""
        logger.info(f"Connecting to MOCVD tool at {self.host}:{self.port}")
        try:
            await self._establish_connection()
            self.connected = True
            self.state = ToolState.IDLE
            logger.info(f"Connected to {self.tool_id}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise ToolError(f"Connection failed: {e}", error_code="CONN_FAILED")

    async def disconnect(self) -> None:
        """Disconnect from MOCVD tool"""
        logger.info(f"Disconnecting from {self.tool_id}")
        await self._close_connection()
        self.connected = False
        self.state = ToolState.OFFLINE

    async def configure(self, recipe: Any) -> None:
        """Load recipe (growth recipe with multi-layer structure)"""
        if not self.connected:
            raise ToolError("Not connected", error_code="NOT_CONNECTED")

        logger.info(f"Configuring MOCVD recipe: {recipe.recipe_name if hasattr(recipe, 'recipe_name') else 'Unknown'}")
        self.state = ToolState.CONFIGURING

        try:
            await self._send_recipe(recipe)
            self.current_recipe = recipe
            self.state = ToolState.IDLE
            logger.info("Recipe configured")
        except Exception as e:
            self.state = ToolState.ERROR
            raise ToolError(f"Configuration failed: {e}", error_code="CONFIG_FAILED")

    async def start_run(self, cvd_run_id: UUID) -> None:
        """Start epitaxial growth"""
        if self.state != ToolState.IDLE:
            raise ToolError(f"Cannot start in state {self.state}", error_code="INVALID_STATE")

        logger.info(f"Starting MOCVD run {cvd_run_id}")
        self.current_run_id = cvd_run_id

        try:
            await self._start_growth()
            self.state = ToolState.RUNNING
            logger.info(f"Growth started")
        except Exception as e:
            self.state = ToolState.ERROR
            raise ToolError(f"Start failed: {e}", error_code="START_FAILED")

    async def stop_run(self, cvd_run_id: UUID) -> None:
        """Stop growth (controlled cooldown)"""
        logger.info(f"Stopping run {cvd_run_id}")
        self.state = ToolState.STOPPING

        try:
            await self._stop_growth()
            self.state = ToolState.IDLE
            self.current_run_id = None
        except Exception as e:
            raise ToolError(f"Stop failed: {e}", error_code="STOP_FAILED")

    async def pause_run(self, cvd_run_id: UUID) -> None:
        """Pause not typically supported for epitaxial growth"""
        raise ToolError("Pause not supported for MOCVD", error_code="NOT_SUPPORTED")

    async def resume_run(self, cvd_run_id: UUID) -> None:
        """Resume not supported"""
        raise ToolError("Resume not supported for MOCVD", error_code="NOT_SUPPORTED")

    async def abort_run(self, cvd_run_id: UUID) -> None:
        """Emergency stop"""
        logger.warning(f"ABORTING run {cvd_run_id}")
        await self._abort_growth()
        self.state = ToolState.ERROR
        self.current_run_id = None

    async def get_status(self, cvd_run_id: Optional[UUID] = None) -> ToolStatus:
        """Get current status"""
        if not self.connected:
            return ToolStatus(state=ToolState.OFFLINE)

        try:
            status_data = await self._read_status()
            return ToolStatus(
                state=self.state,
                cvd_run_id=self.current_run_id,
                chamber_temp_c=status_data.get('temperature_c'),
                chamber_pressure_torr=status_data.get('pressure_torr'),
                active_alarms=status_data.get('alarms', []),
            )
        except Exception as e:
            return ToolStatus(state=ToolState.ERROR, error_message=str(e))

    async def stream_telemetry(
        self,
        cvd_run_id: UUID,
        interval_sec: float = 1.0
    ) -> AsyncIterator[CVDTelemetry]:
        """Stream telemetry with V/III ratio and growth rate"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active", error_code="INVALID_RUN_ID")

        logger.info(f"Starting telemetry stream for MOCVD run {cvd_run_id}")
        start_time = datetime.utcnow()

        while self.state == ToolState.RUNNING:
            try:
                data = await self._read_telemetry()
                elapsed = (datetime.utcnow() - start_time).total_seconds()

                telemetry = CVDTelemetry(
                    cvd_run_id=cvd_run_id,
                    timestamp=datetime.utcnow(),
                    elapsed_time_sec=elapsed,
                    step_number=data.get('layer_number', 1),
                    step_name=data.get('layer_name'),
                    measurements={
                        TelemetryType.TEMPERATURE: data.get('temperature_c', 0.0),
                        TelemetryType.PRESSURE: data.get('pressure_torr', 0.0),
                        TelemetryType.DEPOSITION_RATE: data.get('growth_rate_um_hr', 0.0),
                    },
                    gas_flows_sccm=data.get('gas_flows', {}),
                    thickness_nm=data.get('thickness_nm'),
                    deposition_rate_nm_min=data.get('growth_rate_um_hr', 0.0) * 1000 / 60,  # Convert to nm/min
                    raw_data={
                        'v_iii_ratio': data.get('v_iii_ratio'),
                        'susceptor_rotation_rpm': data.get('rotation_rpm'),
                    },
                )

                yield telemetry
                await asyncio.sleep(interval_sec)
            except Exception as e:
                logger.error(f"Telemetry failed: {e}")
                break

    async def get_alarms(self) -> List[Dict[str, Any]]:
        """Get alarms"""
        try:
            return await self._read_alarms()
        except Exception as e:
            logger.error(f"Failed to read alarms: {e}")
            return []

    async def clear_alarms(self) -> None:
        """Clear alarms"""
        await self._clear_alarms()

    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics"""
        try:
            diag = await self._run_diagnostics()
            return {
                "status": "HEALTHY",
                "checks": diag,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    # =========================================================================
    # MOCVD-specific stub implementations
    # =========================================================================

    async def _establish_connection(self) -> None:
        logger.info("[STUB] Establishing OPC-UA connection to MOCVD tool")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info("[STUB] Sending multi-layer growth recipe to MOCVD tool")

    async def _start_growth(self) -> None:
        logger.info("[STUB] Starting epitaxial growth (heating, stabilizing)")

    async def _stop_growth(self) -> None:
        logger.info("[STUB] Stopping growth (cooling under protective atmosphere)")

    async def _abort_growth(self) -> None:
        logger.warning("[STUB] Emergency stop")

    async def _read_status(self) -> Dict[str, Any]:
        return {
            "temperature_c": 1050.0,
            "pressure_torr": 300.0,
            "alarms": [],
        }

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 1050.0,
            "pressure_torr": 300.0,
            "layer_number": 1,
            "layer_name": "GaN Buffer",
            "growth_rate_um_hr": 2.5,
            "thickness_nm": 500.0,
            "v_iii_ratio": 5000.0,  # NH3/TMGa ratio
            "rotation_rpm": 1000.0,
            "gas_flows": {
                "TMGa": 50.0,
                "NH3": 5000.0,
                "H2": 10000.0,
            },
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {
            "reactor_chamber": "OK",
            "susceptor_heater": "OK",
            "rotation_motor": "OK",
            "precursor_bubblers": "OK",
            "exhaust_scrubber": "OK",
        }


class AACVDDriver:
    """
    Aerosol-Assisted CVD Driver

    For nanomaterials, complex oxides, and functional films.
    Uses aerosolized liquid precursors instead of gases.

    Applications:
    - Transparent conducting oxides (ITO, FTO, ZnO:Al)
    - Superconductors (YBCO)
    - Photocatalysts (TiO₂, ZnO)
    - Sensors and energy materials

    Key parameters:
    - Aerosol generation (ultrasonic, pneumatic)
    - Carrier gas flow
    - Substrate temperature (300-600°C)
    - Precursor solution composition
    """

    def __init__(
        self,
        tool_id: str,
        host: str,
        port: int,
        vendor: str = "Generic",
        model: str = "AACVD",
    ):
        self.tool_id = tool_id
        self.host = host
        self.port = port
        self.vendor = vendor
        self.model = model

        self.connected = False
        self.state = ToolState.OFFLINE
        self.current_recipe: Optional[Any] = None
        self.current_run_id: Optional[UUID] = None

        # AACVD-specific parameters
        self.max_temp_c = 600.0
        self.min_temp_c = 300.0
        self.max_pressure_torr = 760.0  # Atmospheric or reduced
        self.min_pressure_torr = 50.0

        logger.info(f"AACVD driver initialized: {tool_id} @ {host}:{port}")

    # =========================================================================
    # CVDTool Protocol Implementation
    # =========================================================================

    async def get_capabilities(self) -> ToolCapabilities:
        """Get AACVD tool capabilities"""
        return ToolCapabilities(
            tool_id=self.tool_id,
            vendor=self.vendor,
            model=self.model,
            supported_modes=["AACVD"],
            min_temp_c=self.min_temp_c,
            max_temp_c=self.max_temp_c,
            min_pressure_torr=self.min_pressure_torr,
            max_pressure_torr=self.max_pressure_torr,
            max_wafer_diameter_mm=100,
            max_batch_size=1,
            available_gas_lines=[
                "N2",  # Carrier gas
                "Air",  # Oxidizing atmosphere
                "O2",  # Oxidizing
                "Ar",  # Inert
            ],
            max_flow_rate_sccm={
                "N2": 5000.0,
                "Air": 5000.0,
                "O2": 2000.0,
                "Ar": 3000.0,
            },
            has_rf_plasma=False,
            has_thickness_monitor=True,
            has_optical_monitor=True,
            comm_protocol="SCPI",
        )

    async def connect(self) -> None:
        """Connect to AACVD tool"""
        logger.info(f"Connecting to AACVD tool at {self.host}:{self.port}")
        try:
            await self._establish_connection()
            self.connected = True
            self.state = ToolState.IDLE
            logger.info(f"Connected to {self.tool_id}")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise ToolError(f"Connection failed: {e}", error_code="CONN_FAILED")

    async def disconnect(self) -> None:
        """Disconnect"""
        logger.info(f"Disconnecting from {self.tool_id}")
        await self._close_connection()
        self.connected = False
        self.state = ToolState.OFFLINE

    async def configure(self, recipe: Any) -> None:
        """Load recipe (precursor solution + deposition parameters)"""
        if not self.connected:
            raise ToolError("Not connected", error_code="NOT_CONNECTED")

        logger.info(f"Configuring AACVD recipe")
        self.state = ToolState.CONFIGURING

        try:
            await self._send_recipe(recipe)
            self.current_recipe = recipe
            self.state = ToolState.IDLE
            logger.info("Recipe configured")
        except Exception as e:
            self.state = ToolState.ERROR
            raise ToolError(f"Configuration failed: {e}", error_code="CONFIG_FAILED")

    async def start_run(self, cvd_run_id: UUID) -> None:
        """Start aerosol deposition"""
        if self.state != ToolState.IDLE:
            raise ToolError(f"Cannot start in state {self.state}", error_code="INVALID_STATE")

        logger.info(f"Starting AACVD run {cvd_run_id}")
        self.current_run_id = cvd_run_id

        try:
            await self._start_deposition()
            self.state = ToolState.RUNNING
        except Exception as e:
            self.state = ToolState.ERROR
            raise ToolError(f"Start failed: {e}", error_code="START_FAILED")

    async def stop_run(self, cvd_run_id: UUID) -> None:
        """Stop deposition"""
        logger.info(f"Stopping run {cvd_run_id}")
        self.state = ToolState.STOPPING

        try:
            await self._stop_deposition()
            self.state = ToolState.IDLE
            self.current_run_id = None
        except Exception as e:
            raise ToolError(f"Stop failed: {e}", error_code="STOP_FAILED")

    async def pause_run(self, cvd_run_id: UUID) -> None:
        """Pause deposition (can stop aerosol generation)"""
        logger.info(f"Pausing run {cvd_run_id}")
        await self._pause_deposition()
        self.state = ToolState.PAUSED

    async def resume_run(self, cvd_run_id: UUID) -> None:
        """Resume deposition"""
        logger.info(f"Resuming run {cvd_run_id}")
        await self._resume_deposition()
        self.state = ToolState.RUNNING

    async def abort_run(self, cvd_run_id: UUID) -> None:
        """Emergency stop"""
        logger.warning(f"ABORTING run {cvd_run_id}")
        await self._abort_deposition()
        self.state = ToolState.ERROR
        self.current_run_id = None

    async def get_status(self, cvd_run_id: Optional[UUID] = None) -> ToolStatus:
        """Get status"""
        if not self.connected:
            return ToolStatus(state=ToolState.OFFLINE)

        try:
            status_data = await self._read_status()
            return ToolStatus(
                state=self.state,
                cvd_run_id=self.current_run_id,
                chamber_temp_c=status_data.get('temperature_c'),
                chamber_pressure_torr=status_data.get('pressure_torr'),
                active_alarms=status_data.get('alarms', []),
            )
        except Exception as e:
            return ToolStatus(state=ToolState.ERROR, error_message=str(e))

    async def stream_telemetry(
        self,
        cvd_run_id: UUID,
        interval_sec: float = 1.0
    ) -> AsyncIterator[CVDTelemetry]:
        """Stream telemetry with aerosol parameters"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active", error_code="INVALID_RUN_ID")

        logger.info(f"Starting telemetry stream for AACVD run {cvd_run_id}")
        start_time = datetime.utcnow()

        while self.state == ToolState.RUNNING:
            try:
                data = await self._read_telemetry()
                elapsed = (datetime.utcnow() - start_time).total_seconds()

                telemetry = CVDTelemetry(
                    cvd_run_id=cvd_run_id,
                    timestamp=datetime.utcnow(),
                    elapsed_time_sec=elapsed,
                    step_number=1,
                    measurements={
                        TelemetryType.TEMPERATURE: data.get('temperature_c', 0.0),
                        TelemetryType.PRESSURE: data.get('pressure_torr', 0.0),
                    },
                    gas_flows_sccm=data.get('carrier_gas_flows', {}),
                    raw_data={
                        'aerosol_flow_rate_ml_min': data.get('aerosol_flow_ml_min'),
                        'ultrasonic_power_w': data.get('ultrasonic_power_w'),
                        'solution_level_ml': data.get('solution_level_ml'),
                    },
                )

                yield telemetry
                await asyncio.sleep(interval_sec)
            except Exception as e:
                logger.error(f"Telemetry failed: {e}")
                break

    async def get_alarms(self) -> List[Dict[str, Any]]:
        """Get alarms"""
        try:
            return await self._read_alarms()
        except Exception as e:
            return []

    async def clear_alarms(self) -> None:
        """Clear alarms"""
        await self._clear_alarms()

    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics"""
        try:
            diag = await self._run_diagnostics()
            return {
                "status": "HEALTHY",
                "checks": diag,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}

    # =========================================================================
    # AACVD-specific stub implementations
    # =========================================================================

    async def _establish_connection(self) -> None:
        logger.info("[STUB] Establishing SCPI connection to AACVD tool")
        await asyncio.sleep(0.1)

    async def _close_connection(self) -> None:
        logger.info("[STUB] Closing connection")

    async def _send_recipe(self, recipe: Any) -> None:
        logger.info("[STUB] Sending recipe (precursor solution + parameters)")

    async def _start_deposition(self) -> None:
        logger.info("[STUB] Starting aerosol generation and deposition")

    async def _stop_deposition(self) -> None:
        logger.info("[STUB] Stopping aerosol deposition")

    async def _pause_deposition(self) -> None:
        logger.info("[STUB] Pausing aerosol generation")

    async def _resume_deposition(self) -> None:
        logger.info("[STUB] Resuming aerosol generation")

    async def _abort_deposition(self) -> None:
        logger.warning("[STUB] Emergency stop")

    async def _read_status(self) -> Dict[str, Any]:
        return {
            "temperature_c": 450.0,
            "pressure_torr": 760.0,
            "alarms": [],
        }

    async def _read_telemetry(self) -> Dict[str, Any]:
        return {
            "temperature_c": 450.0,
            "pressure_torr": 760.0,
            "aerosol_flow_ml_min": 2.0,
            "ultrasonic_power_w": 50.0,
            "solution_level_ml": 150.0,
            "carrier_gas_flows": {
                "N2": 2000.0,
                "O2": 500.0,
            },
        }

    async def _read_alarms(self) -> List[Dict[str, Any]]:
        return []

    async def _clear_alarms(self) -> None:
        logger.info("[STUB] Clearing alarms")

    async def _run_diagnostics(self) -> Dict[str, Any]:
        return {
            "ultrasonic_generator": "OK",
            "substrate_heater": "OK",
            "carrier_gas_delivery": "OK",
            "exhaust_system": "OK",
        }
