"""
Data writers for standardized Parquet and JSON output with provenance.

Provides writers for:
- MES diffusion run data
- FDC furnace sensor data
- SPC control chart data

All writers include:
- Schema validation
- Data provenance metadata
- Compression
- Partitioning support

Status: PRODUCTION - Session 6
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from datetime import datetime
import pytz

from session6.data.schemas import (
    MESRun,
    FDCFurnaceData,
    SPCChart,
    DataProvenance,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _add_metadata(
    table: pa.Table,
    provenance: DataProvenance,
    schema_version: str = "1.0"
) -> pa.Table:
    """
    Add provenance metadata to PyArrow table.

    Args:
        table: PyArrow table
        provenance: Data provenance
        schema_version: Schema version

    Returns:
        Table with metadata
    """
    metadata = {
        b'source_system': provenance.source_system.encode(),
        b'source_file': (provenance.source_file or '').encode(),
        b'ingestion_timestamp': provenance.ingestion_timestamp.isoformat().encode(),
        b'data_version': provenance.data_version.encode(),
        b'schema_version': schema_version.encode(),
        b'user': (provenance.user or '').encode(),
    }

    # Merge with existing metadata
    existing_metadata = table.schema.metadata or {}
    existing_metadata.update(metadata)

    return table.replace_schema_metadata(existing_metadata)


# ============================================================================
# MES Run Writers
# ============================================================================

def write_mes_runs_parquet(
    runs: List[MESRun],
    output_path: Path,
    partition_by: Optional[str] = None,
    compression: str = "snappy"
) -> Path:
    """
    Write MES runs to Parquet with provenance metadata.

    Args:
        runs: List of MES runs
        output_path: Output file or directory path
        partition_by: Optional partition column (e.g., 'lot_id', 'process_type')
        compression: Compression codec (snappy, gzip, brotli, zstd)

    Returns:
        Path to written file(s)

    Examples:
        >>> runs = load_mes_diffusion_runs(Path("mes_data.csv"))
        >>> write_mes_runs_parquet(
        ...     runs,
        ...     Path("output/mes_runs.parquet"),
        ...     partition_by="lot_id"
        ... )
    """
    if not runs:
        raise ValueError("No runs to write")

    output_path = Path(output_path)

    # Convert to records
    records = []
    for run in runs:
        record = {
            'run_id': run.run_id,
            'lot_id': run.lot_id,
            'wafer_id': run.wafer_id,
            'process_type': run.process_type.value,
            'recipe_name': run.recipe_name,
            'equipment_id': run.equipment_id,
            'start_time': run.start_time,
            'end_time': run.end_time,
            'temperature': float(run.parameters.temperature),
            'temperature_unit': run.parameters.temperature_unit.value,
            'time': float(run.parameters.time),
            'time_unit': run.parameters.time_unit.value,
            'ambient': run.parameters.ambient.value,
            'pressure': float(run.parameters.pressure) if run.parameters.pressure else None,
            'pressure_unit': run.parameters.pressure_unit,
            'flow_rate': float(run.parameters.flow_rate) if run.parameters.flow_rate else None,
            'flow_rate_unit': run.parameters.flow_rate_unit,
            'dopant_type': run.dopant.dopant_type.value,
            'target_concentration': float(run.dopant.target_concentration) if run.dopant.target_concentration else None,
            'concentration_unit': run.dopant.concentration_unit.value,
            'target_junction_depth': float(run.dopant.target_junction_depth) if run.dopant.target_junction_depth else None,
            'depth_unit': run.dopant.depth_unit.value,
            'target_sheet_resistance': float(run.dopant.target_sheet_resistance) if run.dopant.target_sheet_resistance else None,
            'status': run.status.value,
            'measured_junction_depth': float(run.measured_junction_depth) if run.measured_junction_depth else None,
            'measured_sheet_resistance': float(run.measured_sheet_resistance) if run.measured_sheet_resistance else None,
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Convert to PyArrow table
    table = pa.Table.from_pandas(df)

    # Add metadata
    table = _add_metadata(table, runs[0].provenance, "MES_v1.0")

    # Write with optional partitioning
    if partition_by and partition_by in df.columns:
        output_path.mkdir(parents=True, exist_ok=True)
        pq.write_to_dataset(
            table,
            root_path=str(output_path),
            partition_cols=[partition_by],
            compression=compression
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            table,
            str(output_path),
            compression=compression
        )

    return output_path


def write_mes_runs_json(
    runs: List[MESRun],
    output_path: Path,
    indent: int = 2
) -> Path:
    """
    Write MES runs to JSON with provenance.

    Args:
        runs: List of MES runs
        output_path: Output file path
        indent: JSON indentation

    Returns:
        Path to written file

    Examples:
        >>> runs = load_mes_diffusion_runs(Path("mes_data.csv"))
        >>> write_mes_runs_json(runs, Path("output/mes_runs.json"))
    """
    if not runs:
        raise ValueError("No runs to write")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    data = {
        'schema_version': 'MES_v1.0',
        'exported_at': datetime.now(pytz.UTC).isoformat(),
        'runs': [run.model_dump(mode='json') for run in runs]
    }

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

    return output_path


# ============================================================================
# FDC Data Writers
# ============================================================================

def write_fdc_data_parquet(
    fdc_data: FDCFurnaceData,
    output_path: Path,
    compression: str = "snappy"
) -> Path:
    """
    Write FDC furnace data to Parquet.

    Args:
        fdc_data: FDC furnace data
        output_path: Output file path
        compression: Compression codec

    Returns:
        Path to written file

    Examples:
        >>> fdc = load_fdc_furnace_data(
        ...     Path("fdc.parquet"),
        ...     run_id="RUN_001",
        ...     equipment_id="FURN_01"
        ... )
        >>> write_fdc_data_parquet(fdc, Path("output/fdc_clean.parquet"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert readings to records
    records = []
    for reading in fdc_data.readings:
        record = {
            'timestamp': reading.timestamp,
            'temperature': float(reading.temperature),
            'temperature_unit': reading.temperature_unit.value,
            'temperature_setpoint': float(reading.temperature_setpoint) if reading.temperature_setpoint else None,
            'pressure': float(reading.pressure) if reading.pressure else None,
            'pressure_unit': reading.pressure_unit,
            'flow_rate': float(reading.flow_rate) if reading.flow_rate else None,
            'flow_unit': reading.flow_unit,
            'temp_alarm': reading.temp_alarm,
            'pressure_alarm': reading.pressure_alarm,
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Add metadata columns
    df['run_id'] = fdc_data.run_id
    df['equipment_id'] = fdc_data.equipment_id
    df['zone'] = fdc_data.zone
    df['sampling_rate_seconds'] = float(fdc_data.sampling_rate_seconds)

    # Convert to PyArrow table
    table = pa.Table.from_pandas(df)

    # Add metadata
    table = _add_metadata(table, fdc_data.provenance, "FDC_v1.0")

    # Write
    pq.write_table(table, str(output_path), compression=compression)

    return output_path


def write_fdc_data_json(
    fdc_data: FDCFurnaceData,
    output_path: Path,
    indent: int = 2
) -> Path:
    """
    Write FDC data to JSON.

    Args:
        fdc_data: FDC furnace data
        output_path: Output file path
        indent: JSON indentation

    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    data = {
        'schema_version': 'FDC_v1.0',
        'exported_at': datetime.now(pytz.UTC).isoformat(),
        'data': fdc_data.model_dump(mode='json')
    }

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

    return output_path


# ============================================================================
# SPC Chart Writers
# ============================================================================

def write_spc_chart_parquet(
    chart: SPCChart,
    output_path: Path,
    compression: str = "snappy"
) -> Path:
    """
    Write SPC chart data to Parquet.

    Args:
        chart: SPC chart data
        output_path: Output file path
        compression: Compression codec

    Returns:
        Path to written file

    Examples:
        >>> chart = load_spc_chart_data(
        ...     Path("spc.csv"),
        ...     chart_id="JD_001",
        ...     chart_type="xbar",
        ...     metric_name="Junction Depth",
        ...     metric_unit="nm"
        ... )
        >>> write_spc_chart_parquet(chart, Path("output/spc_clean.parquet"))
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert data points to records
    records = []
    for point in chart.data_points:
        record = {
            'timestamp': point.timestamp,
            'run_id': point.run_id,
            'value': float(point.value),
            'unit': point.unit,
            'subgroup_id': point.subgroup_id,
            'sample_size': point.sample_size,
            'out_of_control': point.out_of_control,
            'violation_rules': ','.join(point.violation_rules) if point.violation_rules else None,
        }
        records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Add chart metadata columns
    df['chart_id'] = chart.chart_id
    df['chart_type'] = chart.chart_type
    df['metric_name'] = chart.metric_name
    df['metric_unit'] = chart.metric_unit
    df['ucl'] = float(chart.limits.ucl)
    df['lcl'] = float(chart.limits.lcl)
    df['usl'] = float(chart.limits.usl) if chart.limits.usl else None
    df['lsl'] = float(chart.limits.lsl) if chart.limits.lsl else None
    df['target'] = float(chart.limits.target) if chart.limits.target else None
    df['mean'] = float(chart.mean) if chart.mean else None
    df['std_dev'] = float(chart.std_dev) if chart.std_dev else None
    df['cpk'] = float(chart.cpk) if chart.cpk else None

    # Convert to PyArrow table
    table = pa.Table.from_pandas(df)

    # Add metadata
    table = _add_metadata(table, chart.provenance, "SPC_v1.0")

    # Write
    pq.write_table(table, str(output_path), compression=compression)

    return output_path


def write_spc_chart_json(
    chart: SPCChart,
    output_path: Path,
    indent: int = 2
) -> Path:
    """
    Write SPC chart to JSON.

    Args:
        chart: SPC chart data
        output_path: Output file path
        indent: JSON indentation

    Returns:
        Path to written file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serializable format
    data = {
        'schema_version': 'SPC_v1.0',
        'exported_at': datetime.now(pytz.UTC).isoformat(),
        'chart': chart.model_dump(mode='json')
    }

    # Write JSON
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=indent, default=str)

    return output_path


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "write_mes_runs_parquet",
    "write_mes_runs_json",
    "write_fdc_data_parquet",
    "write_fdc_data_json",
    "write_spc_chart_parquet",
    "write_spc_chart_json",
]
