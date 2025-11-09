"""
Data schemas for MES/SPC/FDC ingestion with strict typing and unit validation.

Provides Pydantic models for:
- MES diffusion run data
- FDC furnace sensor data
- SPC control chart data

All schemas include:
- Strict type validation
- Unit field specifications
- Timezone-aware timestamps
- Data provenance tracking

Status: PRODUCTION - Session 6
"""

from typing import Optional, Literal, Dict, Any, List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, field_validator, ConfigDict
from decimal import Decimal
import pytz


# ============================================================================
# Enumerations
# ============================================================================

class TemperatureUnit(str, Enum):
    """Temperature units."""
    CELSIUS = "C"
    KELVIN = "K"
    FAHRENHEIT = "F"


class TimeUnit(str, Enum):
    """Time units."""
    SECONDS = "s"
    MINUTES = "min"
    HOURS = "hr"


class ConcentrationUnit(str, Enum):
    """Dopant concentration units."""
    PER_CM3 = "cm^-3"
    PER_M3 = "m^-3"
    ATOMS_PER_CM3 = "atoms/cm^3"


class LengthUnit(str, Enum):
    """Length/depth units."""
    NANOMETERS = "nm"
    MICROMETERS = "um"
    ANGSTROMS = "A"


class DopantType(str, Enum):
    """Dopant species."""
    BORON = "B"
    PHOSPHORUS = "P"
    ARSENIC = "As"
    ANTIMONY = "Sb"


class ProcessType(str, Enum):
    """Diffusion process types."""
    PREDEPOSITION = "predeposition"
    DRIVE_IN = "drive_in"
    TWO_STEP = "two_step"
    OXIDATION = "oxidation"


class AmbientType(str, Enum):
    """Furnace ambient types."""
    DRY_O2 = "dry_O2"
    WET_O2 = "wet_O2"
    N2 = "N2"
    STEAM = "steam"


class RunStatus(str, Enum):
    """MES run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


# ============================================================================
# Base Models with Provenance
# ============================================================================

class DataProvenance(BaseModel):
    """Data provenance tracking."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    source_system: str = Field(..., description="Source system (MES/FDC/SPC)")
    source_file: Optional[str] = Field(None, description="Original filename")
    ingestion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(pytz.UTC),
        description="UTC timestamp when data was ingested"
    )
    data_version: str = Field(default="1.0", description="Data schema version")
    user: Optional[str] = Field(None, description="User who uploaded data")

    @field_validator('ingestion_timestamp')
    @classmethod
    def ensure_utc(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return pytz.UTC.localize(v)
        return v.astimezone(pytz.UTC)


# ============================================================================
# MES Diffusion Run Schema
# ============================================================================

class MESProcessParameters(BaseModel):
    """Process parameters from MES."""
    model_config = ConfigDict(extra='forbid')

    temperature: Decimal = Field(..., description="Process temperature", ge=0)
    temperature_unit: TemperatureUnit = Field(default=TemperatureUnit.CELSIUS)

    time: Decimal = Field(..., description="Process time", gt=0)
    time_unit: TimeUnit = Field(default=TimeUnit.MINUTES)

    ambient: AmbientType = Field(..., description="Furnace ambient")

    pressure: Optional[Decimal] = Field(None, description="Pressure", ge=0)
    pressure_unit: Literal["atm", "torr", "Pa"] = "atm"

    flow_rate: Optional[Decimal] = Field(None, description="Gas flow rate", ge=0)
    flow_rate_unit: Literal["sccm", "slm"] = "sccm"

    @field_validator('temperature', 'time', 'pressure', 'flow_rate', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert numeric values to Decimal for precision."""
        if v is None:
            return v
        return Decimal(str(v))


class MESDopantSpec(BaseModel):
    """Dopant specification from MES."""
    model_config = ConfigDict(extra='forbid')

    dopant_type: DopantType = Field(..., description="Dopant species")

    target_concentration: Optional[Decimal] = Field(
        None, description="Target surface concentration", ge=0
    )
    concentration_unit: ConcentrationUnit = Field(
        default=ConcentrationUnit.PER_CM3
    )

    target_junction_depth: Optional[Decimal] = Field(
        None, description="Target junction depth", ge=0
    )
    depth_unit: LengthUnit = Field(default=LengthUnit.NANOMETERS)

    target_sheet_resistance: Optional[Decimal] = Field(
        None, description="Target sheet resistance (ohms/square)", ge=0
    )


class MESRun(BaseModel):
    """Complete MES diffusion run record."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    # Identifiers
    run_id: str = Field(..., description="Unique run identifier", min_length=1)
    lot_id: str = Field(..., description="Wafer lot ID", min_length=1)
    wafer_id: Optional[str] = Field(None, description="Individual wafer ID")

    # Process info
    process_type: ProcessType = Field(..., description="Process type")
    recipe_name: str = Field(..., description="Recipe name", min_length=1)
    equipment_id: str = Field(..., description="Equipment ID", min_length=1)

    # Timestamps (all UTC)
    start_time: datetime = Field(..., description="Run start time (UTC)")
    end_time: Optional[datetime] = Field(None, description="Run end time (UTC)")

    # Process parameters
    parameters: MESProcessParameters = Field(..., description="Process parameters")

    # Dopant spec
    dopant: MESDopantSpec = Field(..., description="Dopant specification")

    # Status
    status: RunStatus = Field(default=RunStatus.PENDING, description="Run status")

    # Metrology results (filled post-process)
    measured_junction_depth: Optional[Decimal] = Field(None, ge=0)
    measured_sheet_resistance: Optional[Decimal] = Field(None, ge=0)

    # Provenance
    provenance: DataProvenance = Field(..., description="Data provenance")

    @field_validator('start_time', 'end_time')
    @classmethod
    def ensure_utc_timestamps(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure all timestamps are UTC."""
        if v is None:
            return None
        if v.tzinfo is None:
            return pytz.UTC.localize(v)
        return v.astimezone(pytz.UTC)

    @field_validator('end_time')
    @classmethod
    def end_after_start(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Validate end time is after start time."""
        if v is None:
            return None
        start = info.data.get('start_time')
        if start and v < start:
            raise ValueError("end_time must be after start_time")
        return v


# ============================================================================
# FDC Furnace Sensor Data Schema
# ============================================================================

class FDCSensorReading(BaseModel):
    """Single FDC sensor reading."""
    model_config = ConfigDict(extra='forbid')

    timestamp: datetime = Field(..., description="Reading timestamp (UTC)")

    # Temperature readings
    temperature: Decimal = Field(..., description="Temperature reading")
    temperature_unit: TemperatureUnit = Field(default=TemperatureUnit.CELSIUS)
    temperature_setpoint: Optional[Decimal] = Field(None, description="Setpoint")

    # Pressure readings
    pressure: Optional[Decimal] = Field(None, description="Pressure reading", ge=0)
    pressure_unit: Literal["atm", "torr", "mbar"] = "torr"

    # Flow readings
    flow_rate: Optional[Decimal] = Field(None, description="Flow rate", ge=0)
    flow_unit: Literal["sccm", "slm"] = "sccm"

    # Alarm flags
    temp_alarm: bool = Field(default=False, description="Temperature alarm")
    pressure_alarm: bool = Field(default=False, description="Pressure alarm")

    @field_validator('timestamp')
    @classmethod
    def ensure_utc_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return pytz.UTC.localize(v)
        return v.astimezone(pytz.UTC)


class FDCFurnaceData(BaseModel):
    """Complete FDC furnace sensor dataset."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    # Identifiers
    run_id: str = Field(..., description="Associated run ID", min_length=1)
    equipment_id: str = Field(..., description="Equipment ID", min_length=1)
    zone: Optional[str] = Field(None, description="Furnace zone")

    # Sensor readings
    readings: List[FDCSensorReading] = Field(
        ..., description="Time-series sensor readings", min_length=1
    )

    # Sampling info
    sampling_rate_seconds: Decimal = Field(
        ..., description="Sampling interval", gt=0
    )

    # Provenance
    provenance: DataProvenance = Field(..., description="Data provenance")

    @field_validator('readings')
    @classmethod
    def readings_chronological(cls, v: List[FDCSensorReading]) -> List[FDCSensorReading]:
        """Ensure readings are in chronological order."""
        if len(v) < 2:
            return v
        for i in range(1, len(v)):
            if v[i].timestamp <= v[i-1].timestamp:
                raise ValueError("Readings must be in chronological order")
        return v


# ============================================================================
# SPC Control Chart Data Schema
# ============================================================================

class SPCLimits(BaseModel):
    """SPC control limits."""
    model_config = ConfigDict(extra='forbid')

    ucl: Decimal = Field(..., description="Upper control limit")
    lcl: Decimal = Field(..., description="Lower control limit")
    usl: Optional[Decimal] = Field(None, description="Upper spec limit")
    lsl: Optional[Decimal] = Field(None, description="Lower spec limit")
    target: Optional[Decimal] = Field(None, description="Target value")

    @field_validator('ucl', 'lcl', 'usl', 'lsl', 'target', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        """Convert to Decimal."""
        if v is None:
            return v
        return Decimal(str(v))


class SPCDataPoint(BaseModel):
    """Single SPC data point."""
    model_config = ConfigDict(extra='forbid')

    timestamp: datetime = Field(..., description="Measurement timestamp (UTC)")
    run_id: str = Field(..., description="Run ID", min_length=1)

    # Measurement
    value: Decimal = Field(..., description="Measured value")
    unit: str = Field(..., description="Measurement unit", min_length=1)

    # Subgroup info
    subgroup_id: Optional[str] = Field(None, description="Subgroup identifier")
    sample_size: int = Field(default=1, description="Sample size", ge=1)

    # Flags
    out_of_control: bool = Field(default=False, description="OOC flag")
    violation_rules: List[str] = Field(
        default_factory=list, description="Violated rules"
    )

    @field_validator('timestamp')
    @classmethod
    def ensure_utc_timestamp(cls, v: datetime) -> datetime:
        """Ensure timestamp is UTC."""
        if v.tzinfo is None:
            return pytz.UTC.localize(v)
        return v.astimezone(pytz.UTC)


class SPCChart(BaseModel):
    """Complete SPC control chart."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    # Chart identification
    chart_id: str = Field(..., description="Chart identifier", min_length=1)
    chart_type: Literal["xbar", "r", "s", "i", "mr", "p", "np", "c", "u"] = Field(
        ..., description="Chart type"
    )

    # Metric info
    metric_name: str = Field(..., description="Metric name", min_length=1)
    metric_unit: str = Field(..., description="Metric unit", min_length=1)

    # Control limits
    limits: SPCLimits = Field(..., description="Control limits")

    # Data points
    data_points: List[SPCDataPoint] = Field(
        ..., description="Chart data points", min_length=1
    )

    # Statistics
    mean: Optional[Decimal] = Field(None, description="Process mean")
    std_dev: Optional[Decimal] = Field(None, description="Standard deviation")
    cpk: Optional[Decimal] = Field(None, description="Process capability", ge=0)

    # Provenance
    provenance: DataProvenance = Field(..., description="Data provenance")

    @field_validator('data_points')
    @classmethod
    def data_points_chronological(cls, v: List[SPCDataPoint]) -> List[SPCDataPoint]:
        """Ensure data points are chronological."""
        if len(v) < 2:
            return v
        for i in range(1, len(v)):
            if v[i].timestamp < v[i-1].timestamp:
                raise ValueError("Data points must be in chronological order")
        return v


# ============================================================================
# Export all schemas
# ============================================================================

__all__ = [
    # Enums
    "TemperatureUnit",
    "TimeUnit",
    "ConcentrationUnit",
    "LengthUnit",
    "DopantType",
    "ProcessType",
    "AmbientType",
    "RunStatus",
    # Base models
    "DataProvenance",
    # MES schemas
    "MESProcessParameters",
    "MESDopantSpec",
    "MESRun",
    # FDC schemas
    "FDCSensorReading",
    "FDCFurnaceData",
    # SPC schemas
    "SPCLimits",
    "SPCDataPoint",
    "SPCChart",
]
