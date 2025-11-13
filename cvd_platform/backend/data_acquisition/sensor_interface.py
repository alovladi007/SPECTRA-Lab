"""
Sensor Interface Module for CVD Platform
Provides unified interface for all sensor types in CVD equipment.
Supports: Temperature, Pressure, Mass Flow, QCM, RGA, Ellipsometer, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import asyncio
import logging
import numpy as np

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Enumeration of supported sensor types"""
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    MASS_FLOW = "mass_flow"
    QCM = "qcm"  # Quartz Crystal Microbalance
    RGA = "rga"  # Residual Gas Analyzer
    ELLIPSOMETER = "ellipsometer"
    REFLECTOMETER = "reflectometer"
    ROTATION = "rotation"
    HEATER_ZONE = "heater_zone"
    VALVE_STATE = "valve_state"
    POWER = "power"


class SensorStatus(Enum):
    """Sensor health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULTY = "faulty"
    OFFLINE = "offline"
    CALIBRATING = "calibrating"


@dataclass
class SensorReading:
    """Data structure for a sensor reading"""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    value: float
    unit: str
    status: SensorStatus
    confidence: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "timestamp": self.timestamp.isoformat(),
            "value": self.value,
            "unit": self.unit,
            "status": self.status.value,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class SensorInterface(ABC):
    """Abstract base class for all sensor interfaces"""

    def __init__(self, sensor_id: str, sensor_type: SensorType, config: Dict[str, Any]):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.config = config
        self.status = SensorStatus.OFFLINE
        self.last_reading: Optional[SensorReading] = None
        self.calibration_data: Dict[str, Any] = {}

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to sensor"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from sensor"""
        pass

    @abstractmethod
    async def read(self) -> SensorReading:
        """Read current value from sensor"""
        pass

    @abstractmethod
    async def calibrate(self, reference_value: Optional[float] = None) -> bool:
        """Calibrate sensor against reference"""
        pass

    async def health_check(self) -> SensorStatus:
        """Check sensor health"""
        try:
            reading = await self.read()
            if reading.confidence > 0.9:
                self.status = SensorStatus.HEALTHY
            elif reading.confidence > 0.7:
                self.status = SensorStatus.DEGRADED
            else:
                self.status = SensorStatus.FAULTY
        except Exception as e:
            logger.error(f"Health check failed for {self.sensor_id}: {e}")
            self.status = SensorStatus.FAULTY
        return self.status

    def apply_calibration(self, raw_value: float) -> float:
        """Apply calibration correction to raw value"""
        if not self.calibration_data:
            return raw_value

        # Linear calibration: y = mx + b
        slope = self.calibration_data.get("slope", 1.0)
        offset = self.calibration_data.get("offset", 0.0)
        return slope * raw_value + offset


class TemperatureSensor(SensorInterface):
    """
    Temperature sensor interface.
    Supports thermocouples (K, J, T types) and RTDs (Pt100, Pt1000).
    """

    def __init__(self, sensor_id: str, config: Dict[str, Any]):
        super().__init__(sensor_id, SensorType.TEMPERATURE, config)
        self.thermocouple_type = config.get("thermocouple_type", "K")
        self.zone_id = config.get("zone_id", 0)
        self.min_temp = config.get("min_temp", 0.0)  # Celsius
        self.max_temp = config.get("max_temp", 1200.0)

    async def connect(self) -> bool:
        """Connect to temperature sensor via PLC/Modbus"""
        try:
            # Simulate connection - replace with actual driver
            await asyncio.sleep(0.1)
            self.status = SensorStatus.HEALTHY
            logger.info(f"Connected to temperature sensor {self.sensor_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.sensor_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from sensor"""
        self.status = SensorStatus.OFFLINE
        return True

    async def read(self) -> SensorReading:
        """Read temperature value"""
        try:
            # Simulate reading - replace with actual driver call
            raw_value = np.random.normal(800.0, 5.0)  # Simulated 800°C ± 5°C
            calibrated_value = self.apply_calibration(raw_value)

            # Validate reading
            if calibrated_value < self.min_temp or calibrated_value > self.max_temp:
                confidence = 0.3
                status = SensorStatus.FAULTY
            else:
                confidence = 0.95
                status = SensorStatus.HEALTHY

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                value=calibrated_value,
                unit="°C",
                status=status,
                confidence=confidence,
                metadata={
                    "zone_id": self.zone_id,
                    "thermocouple_type": self.thermocouple_type,
                    "raw_value": raw_value
                }
            )
            self.last_reading = reading
            return reading

        except Exception as e:
            logger.error(f"Failed to read from {self.sensor_id}: {e}")
            raise

    async def calibrate(self, reference_value: Optional[float] = None) -> bool:
        """Calibrate temperature sensor"""
        try:
            self.status = SensorStatus.CALIBRATING
            logger.info(f"Calibrating {self.sensor_id}...")

            # Perform multi-point calibration
            # In production, this would involve physical reference temperatures
            calibration_points = [
                (100.0, 100.5),  # (reference, measured)
                (500.0, 502.0),
                (800.0, 798.0)
            ]

            # Calculate linear fit
            refs = np.array([p[0] for p in calibration_points])
            meas = np.array([p[1] for p in calibration_points])
            slope, offset = np.polyfit(meas, refs, 1)

            self.calibration_data = {
                "slope": float(slope),
                "offset": float(offset),
                "calibration_date": datetime.utcnow().isoformat(),
                "calibration_points": calibration_points
            }

            self.status = SensorStatus.HEALTHY
            logger.info(f"Calibration complete for {self.sensor_id}: "
                       f"slope={slope:.6f}, offset={offset:.6f}")
            return True

        except Exception as e:
            logger.error(f"Calibration failed for {self.sensor_id}: {e}")
            self.status = SensorStatus.FAULTY
            return False


class PressureSensor(SensorInterface):
    """
    Pressure sensor interface.
    Supports capacitance manometers and Pirani gauges.
    """

    def __init__(self, sensor_id: str, config: Dict[str, Any]):
        super().__init__(sensor_id, SensorType.PRESSURE, config)
        self.pressure_range = config.get("pressure_range", "0-1000")  # Torr
        self.location = config.get("location", "chamber")

    async def connect(self) -> bool:
        try:
            await asyncio.sleep(0.1)
            self.status = SensorStatus.HEALTHY
            logger.info(f"Connected to pressure sensor {self.sensor_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.sensor_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        self.status = SensorStatus.OFFLINE
        return True

    async def read(self) -> SensorReading:
        """Read pressure value"""
        try:
            # Simulate reading - replace with actual driver
            raw_value = np.random.normal(10.0, 0.2)  # Simulated 10 Torr ± 0.2
            calibrated_value = self.apply_calibration(raw_value)

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                value=calibrated_value,
                unit="Torr",
                status=SensorStatus.HEALTHY,
                confidence=0.95,
                metadata={
                    "location": self.location,
                    "pressure_range": self.pressure_range,
                    "raw_value": raw_value
                }
            )
            self.last_reading = reading
            return reading

        except Exception as e:
            logger.error(f"Failed to read from {self.sensor_id}: {e}")
            raise

    async def calibrate(self, reference_value: Optional[float] = None) -> bool:
        """Calibrate pressure sensor"""
        try:
            self.status = SensorStatus.CALIBRATING
            logger.info(f"Calibrating {self.sensor_id}...")

            # Zero calibration at vacuum
            await asyncio.sleep(2.0)  # Wait for vacuum stabilization
            zero_reading = 0.001  # Simulated

            # Full-scale calibration
            if reference_value:
                measured = await self.read()
                slope = reference_value / measured.value
                self.calibration_data = {
                    "slope": slope,
                    "offset": -zero_reading,
                    "calibration_date": datetime.utcnow().isoformat()
                }

            self.status = SensorStatus.HEALTHY
            return True

        except Exception as e:
            logger.error(f"Calibration failed for {self.sensor_id}: {e}")
            self.status = SensorStatus.FAULTY
            return False


class MassFlowController(SensorInterface):
    """
    Mass Flow Controller (MFC) interface.
    Controls and measures gas flow rates.
    """

    def __init__(self, sensor_id: str, config: Dict[str, Any]):
        super().__init__(sensor_id, SensorType.MASS_FLOW, config)
        self.gas_type = config.get("gas_type", "N2")
        self.max_flow = config.get("max_flow", 1000.0)  # sccm
        self.setpoint = 0.0

    async def connect(self) -> bool:
        try:
            await asyncio.sleep(0.1)
            self.status = SensorStatus.HEALTHY
            logger.info(f"Connected to MFC {self.sensor_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.sensor_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        await self.set_flow(0.0)
        self.status = SensorStatus.OFFLINE
        return True

    async def read(self) -> SensorReading:
        """Read actual flow rate"""
        try:
            # Simulate reading - actual flow follows setpoint with lag
            raw_value = np.random.normal(self.setpoint, self.setpoint * 0.01)
            calibrated_value = self.apply_calibration(raw_value)

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                value=calibrated_value,
                unit="sccm",
                status=SensorStatus.HEALTHY,
                confidence=0.98,
                metadata={
                    "gas_type": self.gas_type,
                    "setpoint": self.setpoint,
                    "max_flow": self.max_flow,
                    "raw_value": raw_value
                }
            )
            self.last_reading = reading
            return reading

        except Exception as e:
            logger.error(f"Failed to read from {self.sensor_id}: {e}")
            raise

    async def set_flow(self, setpoint: float) -> bool:
        """Set flow rate setpoint"""
        try:
            if setpoint < 0 or setpoint > self.max_flow:
                logger.error(f"Setpoint {setpoint} out of range for {self.sensor_id}")
                return False

            self.setpoint = setpoint
            # Send command to MFC hardware
            logger.info(f"Set {self.sensor_id} flow to {setpoint} sccm")
            return True

        except Exception as e:
            logger.error(f"Failed to set flow for {self.sensor_id}: {e}")
            return False

    async def calibrate(self, reference_value: Optional[float] = None) -> bool:
        """Calibrate MFC using reference flow meter"""
        try:
            self.status = SensorStatus.CALIBRATING
            logger.info(f"Calibrating {self.sensor_id}...")

            # Multi-point calibration at different flow rates
            calibration_points = []
            for setpoint in [100, 500, 1000]:  # sccm
                await self.set_flow(setpoint)
                await asyncio.sleep(5.0)  # Wait for stabilization
                reading = await self.read()
                # In production, measure with reference flow meter
                reference = setpoint * (1.0 + np.random.normal(0, 0.005))
                calibration_points.append((reference, reading.value))

            # Calculate calibration curve
            refs = np.array([p[0] for p in calibration_points])
            meas = np.array([p[1] for p in calibration_points])
            slope, offset = np.polyfit(meas, refs, 1)

            self.calibration_data = {
                "slope": float(slope),
                "offset": float(offset),
                "calibration_date": datetime.utcnow().isoformat(),
                "gas_type": self.gas_type
            }

            self.status = SensorStatus.HEALTHY
            logger.info(f"Calibration complete for {self.sensor_id}")
            return True

        except Exception as e:
            logger.error(f"Calibration failed for {self.sensor_id}: {e}")
            self.status = SensorStatus.FAULTY
            return False


class QuartzCrystalMicrobalance(SensorInterface):
    """
    Quartz Crystal Microbalance (QCM) interface.
    Measures deposition rate and film thickness in-situ.
    """

    def __init__(self, sensor_id: str, config: Dict[str, Any]):
        super().__init__(sensor_id, SensorType.QCM, config)
        self.crystal_frequency = config.get("crystal_frequency", 6e6)  # Hz
        self.z_ratio = config.get("z_ratio", 0.5)  # Material-dependent
        self.film_density = config.get("film_density", 2.33)  # g/cm³ for Si
        self.accumulated_thickness = 0.0  # nm

    async def connect(self) -> bool:
        try:
            await asyncio.sleep(0.1)
            self.status = SensorStatus.HEALTHY
            logger.info(f"Connected to QCM {self.sensor_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.sensor_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        self.status = SensorStatus.OFFLINE
        return True

    async def read(self) -> SensorReading:
        """Read deposition rate from QCM"""
        try:
            # Simulate QCM reading - frequency shift
            # Sauerbrey equation: Δf = -(2f₀²/A√(ρ_qμ_q)) * Δm
            frequency_shift = np.random.normal(-500, 50)  # Hz (negative = deposition)

            # Convert to deposition rate (nm/s)
            # Rate ∝ frequency shift
            deposition_rate = abs(frequency_shift) * 0.01  # Simplified conversion

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                value=deposition_rate,
                unit="nm/s",
                status=SensorStatus.HEALTHY,
                confidence=0.90,
                metadata={
                    "frequency_shift": frequency_shift,
                    "accumulated_thickness": self.accumulated_thickness,
                    "crystal_frequency": self.crystal_frequency
                }
            )
            self.last_reading = reading
            return reading

        except Exception as e:
            logger.error(f"Failed to read from {self.sensor_id}: {e}")
            raise

    async def get_thickness(self) -> float:
        """Get accumulated film thickness"""
        return self.accumulated_thickness

    async def reset_thickness(self) -> None:
        """Reset thickness accumulator (after wafer change)"""
        self.accumulated_thickness = 0.0
        logger.info(f"Reset thickness for {self.sensor_id}")

    async def calibrate(self, reference_value: Optional[float] = None) -> bool:
        """Calibrate QCM using reference thickness measurement"""
        try:
            self.status = SensorStatus.CALIBRATING
            logger.info(f"Calibrating {self.sensor_id}...")

            if reference_value:
                # Compare QCM thickness to ex-situ measurement
                measured_thickness = self.accumulated_thickness
                if measured_thickness > 0:
                    tooling_factor = reference_value / measured_thickness
                    self.calibration_data = {
                        "tooling_factor": tooling_factor,
                        "calibration_date": datetime.utcnow().isoformat(),
                        "reference_thickness": reference_value
                    }
                    logger.info(f"QCM tooling factor: {tooling_factor:.4f}")

            self.status = SensorStatus.HEALTHY
            return True

        except Exception as e:
            logger.error(f"Calibration failed for {self.sensor_id}: {e}")
            self.status = SensorStatus.FAULTY
            return False


class Ellipsometer(SensorInterface):
    """
    Ellipsometer interface for in-situ and ex-situ thickness measurement.
    Measures film thickness, refractive index, and uniformity.
    """

    def __init__(self, sensor_id: str, config: Dict[str, Any]):
        super().__init__(sensor_id, SensorType.ELLIPSOMETER, config)
        self.wavelength = config.get("wavelength", 632.8)  # nm (HeNe laser)
        self.angle_of_incidence = config.get("angle_of_incidence", 70.0)  # degrees
        self.measurement_points = config.get("measurement_points", 49)  # Wafer map

    async def connect(self) -> bool:
        try:
            await asyncio.sleep(0.1)
            self.status = SensorStatus.HEALTHY
            logger.info(f"Connected to ellipsometer {self.sensor_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.sensor_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        self.status = SensorStatus.OFFLINE
        return True

    async def read(self) -> SensorReading:
        """Read film thickness from ellipsometry"""
        try:
            # Simulate ellipsometry measurement
            # Returns Ψ (psi) and Δ (delta) which are converted to thickness
            psi = np.random.normal(30.0, 0.5)  # degrees
            delta = np.random.normal(120.0, 1.0)  # degrees

            # Model fitting to extract thickness (simplified)
            thickness = self._model_fit(psi, delta)

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                value=thickness,
                unit="nm",
                status=SensorStatus.HEALTHY,
                confidence=0.95,
                metadata={
                    "psi": psi,
                    "delta": delta,
                    "wavelength": self.wavelength,
                    "angle": self.angle_of_incidence
                }
            )
            self.last_reading = reading
            return reading

        except Exception as e:
            logger.error(f"Failed to read from {self.sensor_id}: {e}")
            raise

    def _model_fit(self, psi: float, delta: float) -> float:
        """
        Fit optical model to extract thickness.
        In production, use proper Fresnel equations and multi-layer models.
        """
        # Simplified model for demonstration
        # Real implementation would use proper optical models
        thickness = (delta / 360.0) * (self.wavelength / 2.0) * 100
        return thickness

    async def measure_wafer_map(self, points: int = 49) -> Dict[str, Any]:
        """Measure thickness across wafer at multiple points"""
        try:
            thicknesses = []
            coordinates = []

            # Typical 49-point wafer map in polar coordinates
            for r in [0, 30, 60, 90, 120]:  # mm from center
                for theta in np.linspace(0, 360, 8 if r > 0 else 1, endpoint=False):
                    reading = await self.read()
                    # Add position-dependent variation
                    thickness = reading.value + np.random.normal(0, 2.0)
                    thicknesses.append(thickness)
                    coordinates.append({"r": r, "theta": theta})

            mean_thickness = np.mean(thicknesses)
            std_dev = np.std(thicknesses)
            uniformity = (std_dev / mean_thickness) * 100 if mean_thickness > 0 else 0

            return {
                "mean_thickness": mean_thickness,
                "std_dev": std_dev,
                "uniformity_percent": uniformity,
                "min_thickness": np.min(thicknesses),
                "max_thickness": np.max(thicknesses),
                "range": np.max(thicknesses) - np.min(thicknesses),
                "measurements": [
                    {"coord": coord, "thickness": thick}
                    for coord, thick in zip(coordinates, thicknesses)
                ]
            }

        except Exception as e:
            logger.error(f"Wafer map measurement failed for {self.sensor_id}: {e}")
            raise

    async def calibrate(self, reference_value: Optional[float] = None) -> bool:
        """Calibrate ellipsometer using reference standard"""
        try:
            self.status = SensorStatus.CALIBRATING
            logger.info(f"Calibrating {self.sensor_id}...")

            # Measure reference standard (e.g., SiO2 on Si)
            if reference_value:
                reading = await self.read()
                correction_factor = reference_value / reading.value
                self.calibration_data = {
                    "correction_factor": correction_factor,
                    "calibration_date": datetime.utcnow().isoformat(),
                    "reference_standard": reference_value
                }

            self.status = SensorStatus.HEALTHY
            return True

        except Exception as e:
            logger.error(f"Calibration failed for {self.sensor_id}: {e}")
            self.status = SensorStatus.FAULTY
            return False


class ResidualGasAnalyzer(SensorInterface):
    """
    Residual Gas Analyzer (RGA) interface.
    Mass spectrometer for gas composition monitoring.
    """

    def __init__(self, sensor_id: str, config: Dict[str, Any]):
        super().__init__(sensor_id, SensorType.RGA, config)
        self.mass_range = config.get("mass_range", "1-300")  # amu
        self.monitored_species = config.get("monitored_species", [
            "H2", "N2", "O2", "H2O", "Ar", "SiH4"
        ])

    async def connect(self) -> bool:
        try:
            await asyncio.sleep(0.1)
            self.status = SensorStatus.HEALTHY
            logger.info(f"Connected to RGA {self.sensor_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to {self.sensor_id}: {e}")
            return False

    async def disconnect(self) -> bool:
        self.status = SensorStatus.OFFLINE
        return True

    async def read(self) -> SensorReading:
        """Read dominant species partial pressure"""
        try:
            # Simulate RGA reading - dominant peak
            # Return total pressure reading
            total_pressure = np.random.normal(1e-6, 1e-7)  # Torr

            reading = SensorReading(
                sensor_id=self.sensor_id,
                sensor_type=self.sensor_type,
                timestamp=datetime.utcnow(),
                value=total_pressure,
                unit="Torr",
                status=SensorStatus.HEALTHY,
                confidence=0.90,
                metadata={
                    "mass_range": self.mass_range
                }
            )
            self.last_reading = reading
            return reading

        except Exception as e:
            logger.error(f"Failed to read from {self.sensor_id}: {e}")
            raise

    async def scan_spectrum(self) -> Dict[str, float]:
        """Perform full mass spectrum scan"""
        try:
            spectrum = {}
            mass_to_species = {
                2: "H2", 28: "N2", 32: "O2", 18: "H2O",
                40: "Ar", 32: "SiH4"
            }

            for mass, species in mass_to_species.items():
                # Simulate partial pressure for each species
                partial_pressure = np.random.lognormal(-15, 2)  # Torr
                spectrum[species] = partial_pressure

            logger.info(f"RGA scan complete: {spectrum}")
            return spectrum

        except Exception as e:
            logger.error(f"Spectrum scan failed for {self.sensor_id}: {e}")
            raise

    async def calibrate(self, reference_value: Optional[float] = None) -> bool:
        """Calibrate RGA sensitivity"""
        try:
            self.status = SensorStatus.CALIBRATING
            logger.info(f"Calibrating {self.sensor_id}...")

            # Calibration typically uses known leak rates
            await asyncio.sleep(5.0)

            self.calibration_data = {
                "calibration_date": datetime.utcnow().isoformat(),
                "electron_multiplier_voltage": 1800  # V
            }

            self.status = SensorStatus.HEALTHY
            return True

        except Exception as e:
            logger.error(f"Calibration failed for {self.sensor_id}: {e}")
            self.status = SensorStatus.FAULTY
            return False


# Sensor Factory
class SensorFactory:
    """Factory for creating sensor instances"""

    @staticmethod
    def create_sensor(sensor_type: str, sensor_id: str, config: Dict[str, Any]) -> SensorInterface:
        """Create sensor instance based on type"""
        sensor_map = {
            "temperature": TemperatureSensor,
            "pressure": PressureSensor,
            "mass_flow": MassFlowController,
            "qcm": QuartzCrystalMicrobalance,
            "ellipsometer": Ellipsometer,
            "rga": ResidualGasAnalyzer
        }

        sensor_class = sensor_map.get(sensor_type.lower())
        if not sensor_class:
            raise ValueError(f"Unknown sensor type: {sensor_type}")

        return sensor_class(sensor_id, config)
