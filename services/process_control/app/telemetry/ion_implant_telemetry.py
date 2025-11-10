"""Ion Implantation telemetry streaming."""

import asyncio
from typing import Dict, Any, Optional, AsyncGenerator, List
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

from app.drivers.ion_implant_driver import IonImplantDriver, ImplantStatus

logger = logging.getLogger(__name__)


# ============================================================================
# Telemetry Data Classes
# ============================================================================

@dataclass
class IonImplantTelemetryFrame:
    """Single frame of ion implant telemetry data."""
    timestamp: str
    run_id: Optional[str]
    status: str

    # Beam parameters
    beam_current_mA: float
    beam_voltage_kV: float
    analyzer_field_T: float
    beam_position_x_mm: float
    beam_position_y_mm: float

    # Vacuum
    source_pressure_mTorr: float
    analyzer_pressure_mTorr: float
    process_pressure_mTorr: float
    beamline_pressure_mTorr: float

    # Dose integration
    current_dose_cm2: float
    target_dose_cm2: float
    percent_complete: float
    integrated_charge_C: float
    elapsed_time_s: float

    # Profile info (if available)
    projected_range_nm: Optional[float] = None
    range_straggle_nm: Optional[float] = None
    lateral_straggle_nm: Optional[float] = None

    # Source info
    ion_species: Optional[str] = None
    extraction_voltage_kV: Optional[float] = None
    arc_current_A: Optional[float] = None

    # Wafer info
    wafer_tilt_deg: Optional[float] = None
    wafer_rotation_deg: Optional[float] = None

    # Interlocks
    interlocks_ok: bool = True
    interlock_status: Optional[Dict[str, bool]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Filter out None values for cleaner JSON
        return {k: v for k, v in data.items() if v is not None}


# ============================================================================
# Telemetry Manager
# ============================================================================

class IonImplantTelemetryManager:
    """Manages telemetry streaming for ion implanter."""

    def __init__(
        self,
        driver: IonImplantDriver,
        sample_rate_hz: float = 10.0,
        max_buffer_size: int = 1000
    ):
        """
        Initialize telemetry manager.

        Args:
            driver: Ion implant driver instance
            sample_rate_hz: Telemetry sampling rate (Hz)
            max_buffer_size: Maximum number of frames to buffer
        """
        self.driver = driver
        self.sample_rate_hz = sample_rate_hz
        self.sample_period_s = 1.0 / sample_rate_hz
        self.max_buffer_size = max_buffer_size

        self._is_streaming = False
        self._buffer: List[IonImplantTelemetryFrame] = []
        self._buffer_lock = asyncio.Lock()

    async def start_streaming(self):
        """Start telemetry streaming."""
        if self._is_streaming:
            logger.warning("Telemetry streaming already active")
            return

        logger.info(f"Starting ion implant telemetry streaming at {self.sample_rate_hz} Hz")
        self._is_streaming = True

    async def stop_streaming(self):
        """Stop telemetry streaming."""
        logger.info("Stopping ion implant telemetry streaming")
        self._is_streaming = False

    async def collect_frame(self) -> IonImplantTelemetryFrame:
        """Collect a single telemetry frame from the driver."""
        timestamp = datetime.now().isoformat()

        # Get status
        status = await self.driver.get_status()

        # Get beam status
        beam_status = await self.driver.get_beam_status()
        beam_position = await self.driver.get_beam_position()

        # Get vacuum
        vacuum = await self.driver.get_vacuum_pressure()

        # Get dose integrator
        dose_reading = await self.driver.get_dose_integrator_reading()

        # Get source status
        source_status = await self.driver.get_source_status()

        # Get wafer position
        wafer_pos = await self.driver.get_wafer_position()

        # Get interlocks
        interlocks = await self.driver.check_interlocks()
        all_interlocks_ok = all(interlocks.values())

        # Build telemetry frame
        frame = IonImplantTelemetryFrame(
            timestamp=timestamp,
            run_id=dose_reading.get("run_id"),
            status=status.value,
            # Beam
            beam_current_mA=beam_status.get("beam_current_mA", 0.0),
            beam_voltage_kV=beam_status.get("acceleration_voltage_kV", 0.0),
            analyzer_field_T=beam_status.get("analyzer_field_T", 0.0),
            beam_position_x_mm=beam_position[0],
            beam_position_y_mm=beam_position[1],
            # Vacuum
            source_pressure_mTorr=vacuum.get("source_chamber_mTorr", 0.0),
            analyzer_pressure_mTorr=vacuum.get("analyzer_chamber_mTorr", 0.0),
            process_pressure_mTorr=vacuum.get("process_chamber_mTorr", 0.0),
            beamline_pressure_mTorr=vacuum.get("beamline_mTorr", 0.0),
            # Dose
            current_dose_cm2=dose_reading.get("current_dose_cm2", 0.0),
            target_dose_cm2=dose_reading.get("target_dose_cm2", 0.0),
            percent_complete=dose_reading.get("percent_complete", 0.0),
            integrated_charge_C=dose_reading.get("integrated_charge_C", 0.0),
            elapsed_time_s=dose_reading.get("elapsed_time_s", 0.0),
            # Profile
            projected_range_nm=dose_reading.get("projected_range_nm"),
            range_straggle_nm=dose_reading.get("range_straggle_nm"),
            lateral_straggle_nm=dose_reading.get("lateral_straggle_nm"),
            # Source
            ion_species=source_status.get("ion_species"),
            extraction_voltage_kV=source_status.get("extraction_voltage_kV"),
            arc_current_A=source_status.get("arc_current_A"),
            # Wafer
            wafer_tilt_deg=wafer_pos.tilt_angle_deg,
            wafer_rotation_deg=wafer_pos.rotation_angle_deg,
            # Interlocks
            interlocks_ok=all_interlocks_ok,
            interlock_status=interlocks
        )

        # Add to buffer
        async with self._buffer_lock:
            self._buffer.append(frame)
            if len(self._buffer) > self.max_buffer_size:
                self._buffer.pop(0)  # Remove oldest

        return frame

    async def stream_telemetry(self) -> AsyncGenerator[IonImplantTelemetryFrame, None]:
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

    async def get_latest_frame(self) -> Optional[IonImplantTelemetryFrame]:
        """Get the most recent telemetry frame from buffer."""
        async with self._buffer_lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    async def get_buffer(self, max_frames: Optional[int] = None) -> List[IonImplantTelemetryFrame]:
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
            beam_currents = [f.beam_current_mA for f in self._buffer]
            doses = [f.current_dose_cm2 for f in self._buffer]

            return {
                "buffer_size": len(self._buffer),
                "sample_rate_hz": self.sample_rate_hz,
                "is_streaming": self._is_streaming,
                "oldest_timestamp": self._buffer[0].timestamp,
                "latest_timestamp": self._buffer[-1].timestamp,
                "beam_current_stats": {
                    "mean_mA": sum(beam_currents) / len(beam_currents),
                    "min_mA": min(beam_currents),
                    "max_mA": max(beam_currents),
                    "std_mA": (sum((x - sum(beam_currents)/len(beam_currents))**2 for x in beam_currents) / len(beam_currents))**0.5
                },
                "dose_stats": {
                    "latest_cm2": doses[-1],
                    "rate_cm2_per_s": (doses[-1] - doses[0]) / max(1.0, len(self._buffer) * self.sample_period_s)
                }
            }


# ============================================================================
# Batch Telemetry Recorder
# ============================================================================

class IonImplantTelemetryRecorder:
    """Records telemetry to file for batch processing."""

    def __init__(self, manager: IonImplantTelemetryManager):
        self.manager = manager
        self._recording = False
        self._recorded_frames: List[IonImplantTelemetryFrame] = []

    async def start_recording(self):
        """Start recording telemetry."""
        logger.info("Starting telemetry recording")
        self._recording = True
        self._recorded_frames.clear()

    async def stop_recording(self) -> List[IonImplantTelemetryFrame]:
        """Stop recording and return recorded frames."""
        logger.info(f"Stopping telemetry recording. Recorded {len(self._recorded_frames)} frames")
        self._recording = False
        return self._recorded_frames.copy()

    async def record(self, duration_s: float) -> List[IonImplantTelemetryFrame]:
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

    def get_recorded_frames(self) -> List[IonImplantTelemetryFrame]:
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
    "IonImplantTelemetryFrame",
    "IonImplantTelemetryManager",
    "IonImplantTelemetryRecorder"
]
