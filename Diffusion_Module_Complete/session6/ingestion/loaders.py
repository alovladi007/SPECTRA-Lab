"""
Data loaders for MES/SPC/FDC file ingestion.

Provides parsers for:
- MES diffusion run CSV exports
- FDC furnace sensor Parquet files
- SPC control chart CSV files

All loaders perform:
- Schema validation
- Unit normalization
- Timezone conversion to UTC
- Data quality checks

Status: PRODUCTION - Session 6
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime
import pytz
from decimal import Decimal
import warnings

from session6.data.schemas import (
    MESRun,
    MESProcessParameters,
    MESDopantSpec,
    FDCFurnaceData,
    FDCSensorReading,
    SPCChart,
    SPCDataPoint,
    SPCLimits,
    DataProvenance,
    TemperatureUnit,
    TimeUnit,
    ConcentrationUnit,
    LengthUnit,
    DopantType,
    ProcessType,
    AmbientType,
    RunStatus,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_timestamp(
    ts_str: str,
    source_tz: str = "UTC",
    format_hint: Optional[str] = None
) -> datetime:
    """
    Parse timestamp string to UTC datetime.

    Args:
        ts_str: Timestamp string
        source_tz: Source timezone (default UTC)
        format_hint: Optional datetime format string

    Returns:
        UTC datetime object
    """
    if format_hint:
        dt = datetime.strptime(ts_str, format_hint)
    else:
        # Try common formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%m/%d/%Y %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(ts_str, fmt)
                break
            except ValueError:
                continue
        else:
            # Fallback to pandas
            dt = pd.to_datetime(ts_str).to_pydatetime()

    # Add timezone if naive
    if dt.tzinfo is None:
        tz = pytz.timezone(source_tz)
        dt = tz.localize(dt)

    # Convert to UTC
    return dt.astimezone(pytz.UTC)


def _normalize_temperature(
    value: float,
    from_unit: str,
    to_unit: str = "C"
) -> Decimal:
    """Normalize temperature to target unit."""
    value = Decimal(str(value))

    if from_unit == to_unit:
        return value

    # Convert to Celsius first
    if from_unit == "K":
        celsius = value - Decimal("273.15")
    elif from_unit == "F":
        celsius = (value - Decimal("32")) * Decimal("5") / Decimal("9")
    else:
        celsius = value

    # Convert to target unit
    if to_unit == "C":
        return celsius
    elif to_unit == "K":
        return celsius + Decimal("273.15")
    elif to_unit == "F":
        return celsius * Decimal("9") / Decimal("5") + Decimal("32")
    else:
        raise ValueError(f"Unknown temperature unit: {to_unit}")


def _normalize_time(
    value: float,
    from_unit: str,
    to_unit: str = "min"
) -> Decimal:
    """Normalize time to target unit."""
    value = Decimal(str(value))

    if from_unit == to_unit:
        return value

    # Convert to seconds first
    conversion_to_seconds = {
        "s": Decimal("1"),
        "sec": Decimal("1"),
        "min": Decimal("60"),
        "hr": Decimal("3600"),
        "hour": Decimal("3600"),
    }

    if from_unit not in conversion_to_seconds:
        raise ValueError(f"Unknown time unit: {from_unit}")

    seconds = value * conversion_to_seconds[from_unit]

    # Convert to target unit
    if to_unit == "s":
        return seconds
    elif to_unit == "min":
        return seconds / Decimal("60")
    elif to_unit == "hr":
        return seconds / Decimal("3600")
    else:
        raise ValueError(f"Unknown time unit: {to_unit}")


# ============================================================================
# MES Diffusion Run Loader
# ============================================================================

def load_mes_diffusion_runs(
    filepath: Path,
    source_tz: str = "US/Pacific",
    user: Optional[str] = None
) -> List[MESRun]:
    """
    Load MES diffusion run data from CSV.

    Expected CSV columns:
    - run_id: Unique run identifier
    - lot_id: Wafer lot ID
    - wafer_id: Optional wafer ID
    - process_type: predeposition, drive_in, etc.
    - recipe_name: Recipe name
    - equipment_id: Equipment identifier
    - start_time: Run start timestamp
    - end_time: Run end timestamp (optional)
    - temperature: Process temperature
    - temperature_unit: C, K, or F
    - time: Process time
    - time_unit: s, min, or hr
    - ambient: Furnace ambient (dry_O2, wet_O2, etc.)
    - pressure: Optional pressure
    - pressure_unit: Optional pressure unit
    - dopant_type: B, P, As, or Sb
    - target_concentration: Optional target concentration
    - concentration_unit: Optional concentration unit
    - target_junction_depth: Optional target depth
    - depth_unit: Optional depth unit
    - status: pending, running, completed, failed, aborted

    Args:
        filepath: Path to CSV file
        source_tz: Source timezone for timestamps
        user: Optional user who uploaded data

    Returns:
        List of validated MESRun objects

    Examples:
        >>> runs = load_mes_diffusion_runs(Path("mes_runs.csv"))
        >>> print(f"Loaded {len(runs)} runs")
        >>> print(f"First run: {runs[0].run_id}")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = {
        'run_id', 'lot_id', 'process_type', 'recipe_name',
        'equipment_id', 'start_time', 'temperature', 'temperature_unit',
        'time', 'time_unit', 'ambient', 'dopant_type'
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Create provenance
    provenance = DataProvenance(
        source_system="MES",
        source_file=filepath.name,
        user=user
    )

    # Parse each row
    runs = []
    for idx, row in df.iterrows():
        try:
            # Parse timestamps
            start_time = _parse_timestamp(row['start_time'], source_tz)
            end_time = None
            if pd.notna(row.get('end_time')):
                end_time = _parse_timestamp(row['end_time'], source_tz)

            # Normalize temperature and time
            temperature = _normalize_temperature(
                row['temperature'],
                row['temperature_unit'],
                "C"
            )
            time = _normalize_time(
                row['time'],
                row['time_unit'],
                "min"
            )

            # Process parameters
            params = MESProcessParameters(
                temperature=temperature,
                temperature_unit=TemperatureUnit.CELSIUS,
                time=time,
                time_unit=TimeUnit.MINUTES,
                ambient=AmbientType(row['ambient'].lower()),
                pressure=Decimal(str(row['pressure'])) if pd.notna(row.get('pressure')) else None,
                pressure_unit=row.get('pressure_unit', 'atm'),
                flow_rate=Decimal(str(row['flow_rate'])) if pd.notna(row.get('flow_rate')) else None,
                flow_rate_unit=row.get('flow_rate_unit', 'sccm')
            )

            # Dopant specification
            dopant = MESDopantSpec(
                dopant_type=DopantType(row['dopant_type']),
                target_concentration=Decimal(str(row['target_concentration'])) if pd.notna(row.get('target_concentration')) else None,
                concentration_unit=ConcentrationUnit(row.get('concentration_unit', 'cm^-3')),
                target_junction_depth=Decimal(str(row['target_junction_depth'])) if pd.notna(row.get('target_junction_depth')) else None,
                depth_unit=LengthUnit(row.get('depth_unit', 'nm')),
                target_sheet_resistance=Decimal(str(row['target_sheet_resistance'])) if pd.notna(row.get('target_sheet_resistance')) else None
            )

            # Create MES run
            run = MESRun(
                run_id=str(row['run_id']),
                lot_id=str(row['lot_id']),
                wafer_id=str(row['wafer_id']) if pd.notna(row.get('wafer_id')) else None,
                process_type=ProcessType(row['process_type'].lower()),
                recipe_name=str(row['recipe_name']),
                equipment_id=str(row['equipment_id']),
                start_time=start_time,
                end_time=end_time,
                parameters=params,
                dopant=dopant,
                status=RunStatus(row.get('status', 'pending').lower()),
                measured_junction_depth=Decimal(str(row['measured_junction_depth'])) if pd.notna(row.get('measured_junction_depth')) else None,
                measured_sheet_resistance=Decimal(str(row['measured_sheet_resistance'])) if pd.notna(row.get('measured_sheet_resistance')) else None,
                provenance=provenance
            )

            runs.append(run)

        except Exception as e:
            warnings.warn(f"Row {idx} parsing error: {e}")
            continue

    return runs


# ============================================================================
# FDC Furnace Data Loader
# ============================================================================

def load_fdc_furnace_data(
    filepath: Path,
    run_id: str,
    equipment_id: str,
    zone: Optional[str] = None,
    source_tz: str = "UTC",
    user: Optional[str] = None
) -> FDCFurnaceData:
    """
    Load FDC furnace sensor data from Parquet.

    Expected Parquet columns:
    - timestamp: Sensor reading timestamp
    - temperature: Temperature reading
    - temperature_unit: Optional temperature unit
    - temperature_setpoint: Optional setpoint
    - pressure: Optional pressure reading
    - pressure_unit: Optional pressure unit
    - flow_rate: Optional flow rate
    - flow_unit: Optional flow unit
    - temp_alarm: Optional temperature alarm flag
    - pressure_alarm: Optional pressure alarm flag

    Args:
        filepath: Path to Parquet file
        run_id: Associated run ID
        equipment_id: Equipment identifier
        zone: Optional furnace zone
        source_tz: Source timezone
        user: Optional user

    Returns:
        Validated FDCFurnaceData object

    Examples:
        >>> fdc = load_fdc_furnace_data(
        ...     Path("fdc_data.parquet"),
        ...     run_id="RUN_001",
        ...     equipment_id="FURNACE_01"
        ... )
        >>> print(f"Loaded {len(fdc.readings)} sensor readings")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read Parquet
    table = pq.read_table(filepath)
    df = table.to_pandas()

    # Validate required columns
    if 'timestamp' not in df.columns or 'temperature' not in df.columns:
        raise ValueError("Missing required columns: timestamp, temperature")

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Calculate sampling rate (median interval)
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
    intervals = df['timestamp_dt'].diff().dropna()
    sampling_rate = intervals.median().total_seconds()

    # Create provenance
    provenance = DataProvenance(
        source_system="FDC",
        source_file=filepath.name,
        user=user
    )

    # Parse each reading
    readings = []
    for idx, row in df.iterrows():
        try:
            timestamp = _parse_timestamp(str(row['timestamp']), source_tz)

            # Normalize temperature
            temp_unit = row.get('temperature_unit', 'C')
            temperature = _normalize_temperature(
                row['temperature'],
                temp_unit,
                "C"
            )

            reading = FDCSensorReading(
                timestamp=timestamp,
                temperature=temperature,
                temperature_unit=TemperatureUnit.CELSIUS,
                temperature_setpoint=Decimal(str(row['temperature_setpoint'])) if pd.notna(row.get('temperature_setpoint')) else None,
                pressure=Decimal(str(row['pressure'])) if pd.notna(row.get('pressure')) else None,
                pressure_unit=row.get('pressure_unit', 'torr'),
                flow_rate=Decimal(str(row['flow_rate'])) if pd.notna(row.get('flow_rate')) else None,
                flow_unit=row.get('flow_unit', 'sccm'),
                temp_alarm=bool(row.get('temp_alarm', False)),
                pressure_alarm=bool(row.get('pressure_alarm', False))
            )

            readings.append(reading)

        except Exception as e:
            warnings.warn(f"Row {idx} parsing error: {e}")
            continue

    if not readings:
        raise ValueError("No valid readings parsed")

    return FDCFurnaceData(
        run_id=run_id,
        equipment_id=equipment_id,
        zone=zone,
        readings=readings,
        sampling_rate_seconds=Decimal(str(sampling_rate)),
        provenance=provenance
    )


# ============================================================================
# SPC Chart Data Loader
# ============================================================================

def load_spc_chart_data(
    filepath: Path,
    chart_id: str,
    chart_type: str,
    metric_name: str,
    metric_unit: str,
    source_tz: str = "UTC",
    user: Optional[str] = None
) -> SPCChart:
    """
    Load SPC control chart data from CSV.

    Expected CSV columns:
    - timestamp: Measurement timestamp
    - run_id: Associated run ID
    - value: Measured value
    - subgroup_id: Optional subgroup identifier
    - sample_size: Optional sample size (default 1)
    - out_of_control: Optional OOC flag
    - violation_rules: Optional comma-separated violated rules
    - ucl: Upper control limit (same for all rows)
    - lcl: Lower control limit (same for all rows)
    - usl: Optional upper spec limit
    - lsl: Optional lower spec limit
    - target: Optional target value

    Args:
        filepath: Path to CSV file
        chart_id: Chart identifier
        chart_type: Chart type (xbar, r, s, i, mr, p, np, c, u)
        metric_name: Metric name
        metric_unit: Metric unit
        source_tz: Source timezone
        user: Optional user

    Returns:
        Validated SPCChart object

    Examples:
        >>> chart = load_spc_chart_data(
        ...     Path("spc_junction_depth.csv"),
        ...     chart_id="JD_001",
        ...     chart_type="xbar",
        ...     metric_name="Junction Depth",
        ...     metric_unit="nm"
        ... )
        >>> print(f"Loaded {len(chart.data_points)} data points")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Read CSV
    df = pd.read_csv(filepath)

    # Validate required columns
    required_cols = {'timestamp', 'run_id', 'value', 'ucl', 'lcl'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort by timestamp
    df = df.sort_values('timestamp')

    # Create provenance
    provenance = DataProvenance(
        source_system="SPC",
        source_file=filepath.name,
        user=user
    )

    # Extract control limits (assume constant across dataset)
    first_row = df.iloc[0]
    limits = SPCLimits(
        ucl=Decimal(str(first_row['ucl'])),
        lcl=Decimal(str(first_row['lcl'])),
        usl=Decimal(str(first_row['usl'])) if pd.notna(first_row.get('usl')) else None,
        lsl=Decimal(str(first_row['lsl'])) if pd.notna(first_row.get('lsl')) else None,
        target=Decimal(str(first_row['target'])) if pd.notna(first_row.get('target')) else None
    )

    # Parse each data point
    data_points = []
    for idx, row in df.iterrows():
        try:
            timestamp = _parse_timestamp(str(row['timestamp']), source_tz)

            # Parse violation rules
            violation_rules = []
            if pd.notna(row.get('violation_rules')):
                violation_rules = [r.strip() for r in str(row['violation_rules']).split(',')]

            point = SPCDataPoint(
                timestamp=timestamp,
                run_id=str(row['run_id']),
                value=Decimal(str(row['value'])),
                unit=metric_unit,
                subgroup_id=str(row['subgroup_id']) if pd.notna(row.get('subgroup_id')) else None,
                sample_size=int(row.get('sample_size', 1)),
                out_of_control=bool(row.get('out_of_control', False)),
                violation_rules=violation_rules
            )

            data_points.append(point)

        except Exception as e:
            warnings.warn(f"Row {idx} parsing error: {e}")
            continue

    if not data_points:
        raise ValueError("No valid data points parsed")

    # Calculate statistics
    values = [float(dp.value) for dp in data_points]
    mean = Decimal(str(sum(values) / len(values)))
    std_dev = Decimal(str(pd.Series(values).std()))

    # Calculate Cpk if spec limits exist
    cpk = None
    if limits.usl is not None and limits.lsl is not None and std_dev > 0:
        cpu = (limits.usl - mean) / (3 * std_dev)
        cpl = (mean - limits.lsl) / (3 * std_dev)
        cpk = min(cpu, cpl)

    return SPCChart(
        chart_id=chart_id,
        chart_type=chart_type,
        metric_name=metric_name,
        metric_unit=metric_unit,
        limits=limits,
        data_points=data_points,
        mean=mean,
        std_dev=std_dev,
        cpk=cpk,
        provenance=provenance
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "load_mes_diffusion_runs",
    "load_fdc_furnace_data",
    "load_spc_chart_data",
]
