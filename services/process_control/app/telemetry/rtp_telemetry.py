"""RTP telemetry streaming."""

import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from app.drivers.rtp_driver import RTPDriver, RTPStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Telemetry Data Classes
# ============================================================================

@dataclass
class RTPTelemetryFrame:
    """Single frame of RTP telemetry data."""
    timestamp: str
    run_id: Optional[str]
    status: str

    # Temperature readings
    pyrometer_temp_C: float
    thermocouple_temp_C: float
    setpoint_temp_C: float
    temp_deviation_C: float

    # Lamp power
    lamp_powers_pct: List[float]
    lamp_saturation: bool

    # Gas and pressure
    gas_type: Optional[str]
    gas_flow_sccm: float
    chamber_pressure_torr: float

    # Recipe progress (if running)
    recipe_name: Optional[str] = None
    current_segment: Optional[int] = None
    total_segments: Optional[int] = None
    recipe_progress_pct: Optional[float] = None
    elapsed_time_s: Optional[float] = None

    # Zone temperatures (if available)
    zone_temps_C: Optional[List[float]] = None

    # Interlocks
    interlocks_ok: bool = True
    interlock_status: Optional[Dict[str, bool]] = None

    # Computed metrics
    overshoot_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Filter out None values for cleaner JSON
        return {k: v for k, v in data.items() if v is not None}


# ============================================================================
# Telemetry Manager
# ============================================================================

class RTPTelemetryManager:
    """Manages telemetry streaming for RTP system."""

    def __init__(
        self,
        driver: RTPDriver,
        sample_rate_hz: float = 10.0,
        max_buffer_size: int = 1000
    ):
        """
        Initialize telemetry manager.

        Args:
            driver: RTP driver instance
            sample_rate_hz: Telemetry sampling rate (Hz)
            max_buffer_size: Maximum number of frames to buffer
        """
        self.driver = driver
        self.sample_rate_hz = sample_rate_hz
        self.sample_period_s = 1.0 / sample_rate_hz
        self.max_buffer_size = max_buffer_size

        self._is_streaming = False
        self._buffer: List[RTPTelemetryFrame] = []
        self._buffer_lock = asyncio.Lock()

    async def start_streaming(self):
        """Start telemetry streaming."""
        if self._is_streaming:
            logger.warning("Telemetry streaming already active")
            return

        logger.info(f"Starting RTP telemetry streaming at {self.sample_rate_hz} Hz")
        self._is_streaming = True

    async def stop_streaming(self):
        """Stop telemetry streaming."""
        logger.info("Stopping RTP telemetry streaming")
        self._is_streaming = False

    async def collect_frame(self) -> RTPTelemetryFrame:
        """Collect a single telemetry frame from the driver."""
        timestamp = datetime.now().isoformat()

        # Get status
        status = await self.driver.get_status()

        # Get temperature readings
        temp_data = await self.driver.get_temperature()

        # Get lamp power
        lamp_data = await self.driver.get_lamp_power()
        lamp_powers = [lamp_data.get(f"zone_{i+1}_pct", 0.0) for i in range(4)]  # Assume 4 zones
        lamp_saturation = lamp_data.get("saturation_reached", False)

        # Get gas flow
        gas_params = await self.driver.get_gas_flow()

        # Get chamber pressure
        chamber_pressure = await self.driver.get_chamber_pressure()

        # Get recipe progress
        recipe_progress = await self.driver.get_recipe_progress()

        # Get interlocks
        interlocks = await self.driver.check_interlocks()
        all_interlocks_ok = all(interlocks.values())

        # Extract zone temperatures if available
        zone_temps = temp_data.get("zone_temps_C")

        # Build telemetry frame
        frame = RTPTelemetryFrame(
            timestamp=timestamp,
            run_id=recipe_progress.get("run_id"),
            status=status.value,
            # Temperature
            pyrometer_temp_C=temp_data.get("pyrometer_C", 0.0),
            thermocouple_temp_C=temp_data.get("thermocouple_C", 0.0),
            setpoint_temp_C=temp_data.get("setpoint_C", 0.0),
            temp_deviation_C=temp_data.get("deviation_C", 0.0),
            # Lamps
            lamp_powers_pct=lamp_powers,
            lamp_saturation=lamp_saturation,
            # Gas
            gas_type=gas_params.gas_type.value,
            gas_flow_sccm=gas_params.flow_rate_sccm,
            chamber_pressure_torr=chamber_pressure,
            # Recipe
            recipe_name=recipe_progress.get("recipe_name"),
            current_segment=recipe_progress.get("current_segment"),
            total_segments=recipe_progress.get("total_segments"),
            recipe_progress_pct=recipe_progress.get("progress_pct"),
            elapsed_time_s=recipe_progress.get("elapsed_time_s"),
            # Zones
            zone_temps_C=zone_temps,
            # Interlocks
            interlocks_ok=all_interlocks_ok,
            interlock_status=interlocks,
            # Metrics
            overshoot_pct=recipe_progress.get("overshoot_pct")
        )

        # Add to buffer
        async with self._buffer_lock:
            self._buffer.append(frame)
            if len(self._buffer) > self.max_buffer_size:
                self._buffer.pop(0)  # Remove oldest

        return frame

    async def stream_telemetry(self) -> AsyncGenerator[RTPTelemetryFrame, None]:
        """
        Async generator that yields telemetry frames at specified rate.

        Usage:
            async for frame in telemetry_manager.stream_telemetry():
                process(frame)
        """
        await self.start_streaming()

        try:
            while self._is_streaming:
                frame = await self.collect_frame()
                yield frame
                await asyncio.sleep(self.sample_period_s)
        finally:
            await self.stop_streaming()

    async def get_latest_frame(self) -> Optional[RTPTelemetryFrame]:
        """Get the most recent telemetry frame from buffer."""
        async with self._buffer_lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    async def get_buffer(self, max_frames: Optional[int] = None) -> List[RTPTelemetryFrame]:
        """Get buffered telemetry frames."""
        async with self._buffer_lock:
            if max_frames:
                return self._buffer[-max_frames:]
            return self._buffer.copy()

    async def clear_buffer(self):
        """Clear telemetry buffer."""
        async with self._buffer_lock:
            self._buffer.clear()

    def set_sample_rate(self, rate_hz: float):
        """Change telemetry sample rate."""
        if rate_hz <= 0 or rate_hz > 1000:
            raise ValueError("Sample rate must be between 0 and 1000 Hz")

        logger.info(f"Changing telemetry rate: {self.sample_rate_hz} -> {rate_hz} Hz")
        self.sample_rate_hz = rate_hz
        self.sample_period_s = 1.0 / rate_hz

    async def get_statistics(self) -> Dict[str, Any]:
        """Get telemetry statistics."""
        async with self._buffer_lock:
            if not self._buffer:
                return {
                    "buffer_size": 0,
                    "sample_rate_hz": self.sample_rate_hz,
                    "is_streaming": self._is_streaming
                }

            # Calculate statistics from buffer
            pyro_temps = [f.pyrometer_temp_C for f in self._buffer]
            tc_temps = [f.thermocouple_temp_C for f in self._buffer]
            deviations = [f.temp_deviation_C for f in self._buffer]

            return {
                "buffer_size": len(self._buffer),
                "sample_rate_hz": self.sample_rate_hz,
                "is_streaming": self._is_streaming,
                "oldest_timestamp": self._buffer[0].timestamp,
                "latest_timestamp": self._buffer[-1].timestamp,
                "pyrometer_stats": {
                    "mean_C": sum(pyro_temps) / len(pyro_temps),
                    "min_C": min(pyro_temps),
                    "max_C": max(pyro_temps),
                    "std_C": (sum((x - sum(pyro_temps)/len(pyro_temps))**2 for x in pyro_temps) / len(pyro_temps))**0.5
                },
                "thermocouple_stats": {
                    "mean_C": sum(tc_temps) / len(tc_temps),
                    "min_C": min(tc_temps),
                    "max_C": max(tc_temps),
                    "std_C": (sum((x - sum(tc_temps)/len(tc_temps))**2 for x in tc_temps) / len(tc_temps))**0.5
                },
                "control_performance": {
                    "mean_deviation_C": sum(deviations) / len(deviations),
                    "max_deviation_C": max(deviations),
                    "rms_deviation_C": (sum(d**2 for d in deviations) / len(deviations))**0.5
                }
            }


# ============================================================================
# Batch Telemetry Recorder
# ============================================================================

class RTPTelemetryRecorder:
    """Records telemetry to file for batch processing."""

    def __init__(self, manager: RTPTelemetryManager):
        self.manager = manager
        self._recording = False
        self._recorded_frames: List[RTPTelemetryFrame] = []

    async def start_recording(self):
        """Start recording telemetry."""
        logger.info("Starting telemetry recording")
        self._recording = True
        self._recorded_frames.clear()

    async def stop_recording(self) -> List[RTPTelemetryFrame]:
        """Stop recording and return recorded frames."""
        logger.info(f"Stopping telemetry recording. Recorded {len(self._recorded_frames)} frames")
        self._recording = False
        return self._recorded_frames.copy()

    async def record(self, duration_s: float) -> List[RTPTelemetryFrame]:
        """
        Record telemetry for specified duration.

        Args:
            duration_s: Recording duration in seconds

        Returns:
            List of recorded telemetry frames
        """
        await self.start_recording()

        start_time = datetime.now()
        async for frame in self.manager.stream_telemetry():
            if self._recording:
                self._recorded_frames.append(frame)

            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration_s:
                break

        return await self.stop_recording()

    def get_recorded_frames(self) -> List[RTPTelemetryFrame]:
        """Get all recorded frames."""
        return self._recorded_frames.copy()

    def export_to_dict(self) -> List[Dict[str, Any]]:
        """Export recorded frames as list of dictionaries."""
        return [frame.to_dict() for frame in self._recorded_frames]

    async def export_to_json_file(self, filepath: str):
        """Export recorded frames to JSON file."""
        import json

        data = {
            "metadata": {
                "recording_start": self._recorded_frames[0].timestamp if self._recorded_frames else None,
                "recording_end": self._recorded_frames[-1].timestamp if self._recorded_frames else None,
                "num_frames": len(self._recorded_frames),
                "sample_rate_hz": self.manager.sample_rate_hz
            },
            "frames": self.export_to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported {len(self._recorded_frames)} frames to {filepath}")


# Export
__all__ = [
    "RTPTelemetryFrame",
    "RTPTelemetryManager",
    "RTPTelemetryRecorder"
]
