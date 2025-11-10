"""Ion Implantation driver interface and implementation."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class IonSource(str, Enum):
    """Ion source types."""
    FREEMAN = "freeman"
    BERNAS = "bernas"
    INDIRECTLY_HEATED_CATHODE = "ihc"


class IonSpecies(str, Enum):
    """Common ion species."""
    BORON = "B"
    PHOSPHORUS = "P"
    ARSENIC = "As"
    ANTIMONY = "Sb"
    NITROGEN = "N"
    OXYGEN = "O"
    ARGON = "Ar"
    SILICON = "Si"


class ScanPattern(str, Enum):
    """Beam scan patterns."""
    RASTER = "raster"
    SPIRAL = "spiral"
    SERPENTINE = "serpentine"
    STATIC = "static"


class ImplantStatus(str, Enum):
    """Implanter status."""
    IDLE = "idle"
    WARMING_UP = "warming_up"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class SourceParameters:
    """Ion source parameters."""
    source_type: IonSource
    ion_species: IonSpecies
    extraction_voltage_kV: float  # 10-50 kV typical
    arc_voltage_V: float  # 100-150 V typical
    arc_current_A: float  # 5-20 A typical
    gas_flow_sccm: float  # Gas feed rate


@dataclass
class BeamParameters:
    """Beam line parameters."""
    analyzer_magnet_field_T: float  # Mass analysis
    acceleration_voltage_kV: float  # Total energy (extraction + accel)
    focus_voltage_kV: Optional[float] = None
    decel_voltage_kV: Optional[float] = None  # For low energy implants


@dataclass
class ScanParameters:
    """Beam scanning parameters."""
    pattern: ScanPattern
    x_amplitude_mm: float
    y_amplitude_mm: float
    x_frequency_Hz: float
    y_frequency_Hz: float
    scan_speed_mm_s: float


@dataclass
class WaferParameters:
    """Wafer positioning parameters."""
    tilt_angle_deg: float  # 0-7° typical (channeling control)
    rotation_angle_deg: float  # 0-360°
    rotation_speed_rpm: float  # For uniformity


@dataclass
class DoseParameters:
    """Dose integration parameters."""
    target_dose_cm2: float  # Target dose (ions/cm²)
    beam_current_mA: float  # Expected beam current
    wafer_area_cm2: float  # Active implant area
    charge_state: int = 1  # Ion charge state (usually +1)


# ============================================================================
# Abstract Driver Interface
# ============================================================================

class IonImplantDriver(ABC):
    """Abstract interface for ion implanter control."""

    def __init__(self, equipment_id: str):
        self.equipment_id = equipment_id
        self.status = ImplantStatus.IDLE
        self._is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to implanter hardware/simulator."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from implanter."""
        pass

    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize implanter systems."""
        pass

    @abstractmethod
    async def shutdown(self) -> bool:
        """Safe shutdown of implanter."""
        pass

    # Source control
    @abstractmethod
    async def source_on(self, params: SourceParameters) -> bool:
        """Turn on ion source with specified parameters."""
        pass

    @abstractmethod
    async def source_off(self) -> bool:
        """Turn off ion source."""
        pass

    @abstractmethod
    async def get_source_status(self) -> Dict[str, Any]:
        """Get current source status."""
        pass

    # Beam line control
    @abstractmethod
    async def set_beam_parameters(self, params: BeamParameters) -> bool:
        """Set beam line parameters."""
        pass

    @abstractmethod
    async def set_analyzer_magnet(self, field_tesla: float) -> bool:
        """Set analyzer magnet field for mass selection."""
        pass

    @abstractmethod
    async def get_beam_status(self) -> Dict[str, Any]:
        """Get current beam status."""
        pass

    # Beam steering
    @abstractmethod
    async def set_beam_steering(self, x_offset_mm: float, y_offset_mm: float) -> bool:
        """Set beam steering offset."""
        pass

    @abstractmethod
    async def get_beam_position(self) -> Tuple[float, float]:
        """Get current beam position (x, y) in mm."""
        pass

    # Scanning
    @abstractmethod
    async def set_scan_pattern(self, params: ScanParameters) -> bool:
        """Configure beam scanning pattern."""
        pass

    @abstractmethod
    async def start_scan(self) -> bool:
        """Start beam scanning."""
        pass

    @abstractmethod
    async def stop_scan(self) -> bool:
        """Stop beam scanning."""
        pass

    # Wafer handling
    @abstractmethod
    async def set_wafer_position(self, params: WaferParameters) -> bool:
        """Set wafer tilt and rotation."""
        pass

    @abstractmethod
    async def get_wafer_position(self) -> WaferParameters:
        """Get current wafer position."""
        pass

    # Dose control
    @abstractmethod
    async def start_implant(self, params: DoseParameters) -> str:
        """Start implantation run. Returns run_id."""
        pass

    @abstractmethod
    async def pause_implant(self) -> bool:
        """Pause current implant."""
        pass

    @abstractmethod
    async def resume_implant(self) -> bool:
        """Resume paused implant."""
        pass

    @abstractmethod
    async def stop_implant(self) -> bool:
        """Stop current implant."""
        pass

    @abstractmethod
    async def get_dose_integrator_reading(self) -> Dict[str, Any]:
        """Get current dose integrator reading."""
        pass

    # Status and diagnostics
    @abstractmethod
    async def get_status(self) -> ImplantStatus:
        """Get overall implanter status."""
        pass

    @abstractmethod
    async def get_vacuum_pressure(self) -> Dict[str, float]:
        """Get vacuum pressures (source, analyzer, process chamber)."""
        pass

    @abstractmethod
    async def check_interlocks(self) -> Dict[str, bool]:
        """Check all safety interlocks."""
        pass


# ============================================================================
# Mock Hardware Implementation
# ============================================================================

class IonImplantMockDriver(IonImplantDriver):
    """Mock driver for testing (returns realistic values)."""

    def __init__(self, equipment_id: str):
        super().__init__(equipment_id)
        self._source_params: Optional[SourceParameters] = None
        self._beam_params: Optional[BeamParameters] = None
        self._scan_params: Optional[ScanParameters] = None
        self._wafer_params: Optional[WaferParameters] = None
        self._dose_params: Optional[DoseParameters] = None
        self._beam_steering = (0.0, 0.0)
        self._current_dose = 0.0
        self._implant_start_time: Optional[datetime] = None
        self._is_scanning = False
        self._run_id: Optional[str] = None

    async def connect(self) -> bool:
        logger.info(f"Connecting to mock implanter {self.equipment_id}")
        await asyncio.sleep(0.1)  # Simulate connection delay
        self._is_connected = True
        return True

    async def disconnect(self) -> bool:
        logger.info(f"Disconnecting from mock implanter {self.equipment_id}")
        self._is_connected = False
        self.status = ImplantStatus.IDLE
        return True

    async def initialize(self) -> bool:
        if not self._is_connected:
            raise RuntimeError("Not connected to implanter")

        logger.info("Initializing implanter systems")
        await asyncio.sleep(0.5)  # Simulate init time
        self.status = ImplantStatus.READY
        return True

    async def shutdown(self) -> bool:
        logger.info("Shutting down implanter")
        await self.source_off()
        await self.stop_scan()
        self.status = ImplantStatus.SHUTDOWN
        return True

    # Source control
    async def source_on(self, params: SourceParameters) -> bool:
        if not self._is_connected:
            raise RuntimeError("Not connected to implanter")

        logger.info(f"Turning on ion source: {params.ion_species} @ {params.extraction_voltage_kV} kV")
        self._source_params = params
        await asyncio.sleep(0.2)  # Simulate warmup
        return True

    async def source_off(self) -> bool:
        logger.info("Turning off ion source")
        self._source_params = None
        await asyncio.sleep(0.1)
        return True

    async def get_source_status(self) -> Dict[str, Any]:
        if self._source_params is None:
            return {
                "is_on": False,
                "ion_species": None,
                "extraction_voltage_kV": 0.0,
                "arc_voltage_V": 0.0,
                "arc_current_A": 0.0
            }

        return {
            "is_on": True,
            "ion_species": self._source_params.ion_species,
            "extraction_voltage_kV": self._source_params.extraction_voltage_kV,
            "arc_voltage_V": self._source_params.arc_voltage_V,
            "arc_current_A": self._source_params.arc_current_A,
            "gas_flow_sccm": self._source_params.gas_flow_sccm
        }

    # Beam line control
    async def set_beam_parameters(self, params: BeamParameters) -> bool:
        logger.info(f"Setting beam parameters: {params.acceleration_voltage_kV} kV")
        self._beam_params = params
        await asyncio.sleep(0.1)
        return True

    async def set_analyzer_magnet(self, field_tesla: float) -> bool:
        logger.info(f"Setting analyzer magnet: {field_tesla} T")
        if self._beam_params:
            self._beam_params.analyzer_magnet_field_T = field_tesla
        await asyncio.sleep(0.1)
        return True

    async def get_beam_status(self) -> Dict[str, Any]:
        if self._beam_params is None:
            return {
                "beam_on": False,
                "acceleration_voltage_kV": 0.0,
                "analyzer_field_T": 0.0,
                "beam_current_mA": 0.0
            }

        # Simulate beam current based on source
        beam_current = 5.0 if self._source_params else 0.0

        return {
            "beam_on": self._source_params is not None,
            "acceleration_voltage_kV": self._beam_params.acceleration_voltage_kV,
            "analyzer_field_T": self._beam_params.analyzer_magnet_field_T,
            "beam_current_mA": beam_current,
            "focus_voltage_kV": self._beam_params.focus_voltage_kV
        }

    # Beam steering
    async def set_beam_steering(self, x_offset_mm: float, y_offset_mm: float) -> bool:
        logger.info(f"Setting beam steering: ({x_offset_mm}, {y_offset_mm}) mm")
        self._beam_steering = (x_offset_mm, y_offset_mm)
        await asyncio.sleep(0.05)
        return True

    async def get_beam_position(self) -> Tuple[float, float]:
        return self._beam_steering

    # Scanning
    async def set_scan_pattern(self, params: ScanParameters) -> bool:
        logger.info(f"Setting scan pattern: {params.pattern}")
        self._scan_params = params
        await asyncio.sleep(0.1)
        return True

    async def start_scan(self) -> bool:
        if self._scan_params is None:
            raise RuntimeError("Scan parameters not set")

        logger.info("Starting beam scan")
        self._is_scanning = True
        await asyncio.sleep(0.05)
        return True

    async def stop_scan(self) -> bool:
        logger.info("Stopping beam scan")
        self._is_scanning = False
        await asyncio.sleep(0.05)
        return True

    # Wafer handling
    async def set_wafer_position(self, params: WaferParameters) -> bool:
        logger.info(f"Setting wafer position: tilt={params.tilt_angle_deg}°, rotation={params.rotation_angle_deg}°")
        self._wafer_params = params
        await asyncio.sleep(0.2)
        return True

    async def get_wafer_position(self) -> WaferParameters:
        if self._wafer_params is None:
            return WaferParameters(tilt_angle_deg=0.0, rotation_angle_deg=0.0, rotation_speed_rpm=0.0)
        return self._wafer_params

    # Dose control
    async def start_implant(self, params: DoseParameters) -> str:
        if not self._source_params or not self._beam_params:
            raise RuntimeError("Source and beam parameters must be set before implanting")

        logger.info(f"Starting implant: {params.target_dose_cm2:.2e} ions/cm²")
        self._dose_params = params
        self._current_dose = 0.0
        self._implant_start_time = datetime.now()
        self._run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.status = ImplantStatus.RUNNING

        # Start scan if not already scanning
        if not self._is_scanning and self._scan_params:
            await self.start_scan()

        return self._run_id

    async def pause_implant(self) -> bool:
        logger.info("Pausing implant")
        self.status = ImplantStatus.PAUSED
        await asyncio.sleep(0.05)
        return True

    async def resume_implant(self) -> bool:
        logger.info("Resuming implant")
        self.status = ImplantStatus.RUNNING
        await asyncio.sleep(0.05)
        return True

    async def stop_implant(self) -> bool:
        logger.info("Stopping implant")
        await self.stop_scan()
        self.status = ImplantStatus.READY
        self._implant_start_time = None
        self._run_id = None
        await asyncio.sleep(0.05)
        return True

    async def get_dose_integrator_reading(self) -> Dict[str, Any]:
        if self._dose_params is None:
            return {
                "current_dose_cm2": 0.0,
                "target_dose_cm2": 0.0,
                "percent_complete": 0.0,
                "integrated_charge_C": 0.0,
                "elapsed_time_s": 0.0,
                "run_id": None
            }

        # Simulate dose accumulation (in real driver, this would read from integrator)
        if self.status == ImplantStatus.RUNNING and self._implant_start_time:
            elapsed = (datetime.now() - self._implant_start_time).total_seconds()
            # Simple linear accumulation model
            dose_rate = self._dose_params.beam_current_mA * 1e-3 / (1.6e-19 * self._dose_params.wafer_area_cm2)
            self._current_dose = min(dose_rate * elapsed, self._dose_params.target_dose_cm2)

        percent_complete = (self._current_dose / self._dose_params.target_dose_cm2) * 100 if self._dose_params.target_dose_cm2 > 0 else 0.0

        # Calculate integrated charge
        integrated_charge = self._current_dose * self._dose_params.wafer_area_cm2 * 1.6e-19

        elapsed = (datetime.now() - self._implant_start_time).total_seconds() if self._implant_start_time else 0.0

        return {
            "current_dose_cm2": self._current_dose,
            "target_dose_cm2": self._dose_params.target_dose_cm2,
            "percent_complete": percent_complete,
            "integrated_charge_C": integrated_charge,
            "elapsed_time_s": elapsed,
            "run_id": self._run_id
        }

    # Status and diagnostics
    async def get_status(self) -> ImplantStatus:
        return self.status

    async def get_vacuum_pressure(self) -> Dict[str, float]:
        # Simulate realistic vacuum pressures (in mTorr)
        return {
            "source_chamber_mTorr": 1e-4,
            "analyzer_chamber_mTorr": 5e-6,
            "process_chamber_mTorr": 1e-5,
            "beamline_mTorr": 2e-6
        }

    async def check_interlocks(self) -> Dict[str, bool]:
        """Check all safety interlocks."""
        return {
            "chamber_door": True,
            "beam_shutter": True,
            "vacuum_ok": True,
            "cooling_water": True,
            "e_stop": True,
            "x_ray_level": True,
            "ground_fault": True
        }


# Export
__all__ = [
    "IonImplantDriver",
    "IonImplantMockDriver",
    "IonSource",
    "IonSpecies",
    "ScanPattern",
    "ImplantStatus",
    "SourceParameters",
    "BeamParameters",
    "ScanParameters",
    "WaferParameters",
    "DoseParameters"
]
