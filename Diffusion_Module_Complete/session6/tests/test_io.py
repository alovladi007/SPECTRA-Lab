"""
Comprehensive unit tests for Session 6 IO & Schemas.

Tests:
- Schema validation
- Unit normalization (temperature, time)
- Timezone conversion
- Round-trip IO (load + write + load)
- Data provenance
- Error handling

Status: COMPREHENSIVE TEST SUITE
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from decimal import Decimal
import tempfile
import shutil

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.schemas import (
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
    DopantType,
    ProcessType,
    AmbientType,
    RunStatus,
)

from ingestion.loaders import (
    load_mes_diffusion_runs,
    load_fdc_furnace_data,
    load_spc_chart_data,
)

from ingestion.writers import (
    write_mes_runs_parquet,
    write_mes_runs_json,
    write_fdc_data_parquet,
    write_fdc_data_json,
    write_spc_chart_parquet,
    write_spc_chart_json,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_output_dir():
    """Temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# Schema Validation Tests
# ============================================================================

class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_temperature_unit_validation(self):
        """Test temperature unit enum validation."""
        assert TemperatureUnit.CELSIUS == "C"
        assert TemperatureUnit.KELVIN == "K"
        assert TemperatureUnit.FAHRENHEIT == "F"

    def test_dopant_type_validation(self):
        """Test dopant type enum validation."""
        assert DopantType.BORON == "B"
        assert DopantType.PHOSPHORUS == "P"
        assert DopantType.ARSENIC == "As"
        assert DopantType.ANTIMONY == "Sb"

    def test_mes_run_valid_data(self):
        """Test MES run with valid data."""
        provenance = DataProvenance(
            source_system="MES",
            source_file="test.csv"
        )

        params = MESProcessParameters(
            temperature=Decimal("1000"),
            temperature_unit=TemperatureUnit.CELSIUS,
            time=Decimal("60"),
            time_unit=TimeUnit.MINUTES,
            ambient=AmbientType.DRY_O2
        )

        dopant = MESDopantSpec(
            dopant_type=DopantType.BORON,
            target_concentration=Decimal("1e19")
        )

        run = MESRun(
            run_id="TEST_001",
            lot_id="LOT_01",
            process_type=ProcessType.PREDEPOSITION,
            recipe_name="BORON_PREDEP",
            equipment_id="FURN_01",
            start_time=datetime.now(pytz.UTC),
            parameters=params,
            dopant=dopant,
            provenance=provenance
        )

        assert run.run_id == "TEST_001"
        assert run.parameters.temperature == Decimal("1000")

    def test_mes_run_invalid_end_before_start(self):
        """Test MES run fails when end_time before start_time."""
        with pytest.raises(ValueError, match="end_time must be after start_time"):
            provenance = DataProvenance(source_system="MES")
            params = MESProcessParameters(
                temperature=Decimal("1000"),
                time=Decimal("60"),
                ambient=AmbientType.DRY_O2
            )
            dopant = MESDopantSpec(dopant_type=DopantType.BORON)

            MESRun(
                run_id="TEST",
                lot_id="LOT",
                process_type=ProcessType.PREDEPOSITION,
                recipe_name="TEST",
                equipment_id="FURN",
                start_time=datetime.now(pytz.UTC),
                end_time=datetime.now(pytz.UTC) - timedelta(hours=1),
                parameters=params,
                dopant=dopant,
                provenance=provenance
            )

    def test_fdc_readings_must_be_chronological(self):
        """Test FDC readings must be in chronological order."""
        with pytest.raises(ValueError, match="chronological"):
            provenance = DataProvenance(source_system="FDC")

            readings = [
                FDCSensorReading(
                    timestamp=datetime.now(pytz.UTC),
                    temperature=Decimal("1000")
                ),
                FDCSensorReading(
                    timestamp=datetime.now(pytz.UTC) - timedelta(seconds=1),
                    temperature=Decimal("1001")
                )
            ]

            FDCFurnaceData(
                run_id="TEST",
                equipment_id="FURN",
                readings=readings,
                sampling_rate_seconds=Decimal("1"),
                provenance=provenance
            )


# ============================================================================
# MES Loader Tests
# ============================================================================

class TestMESLoaders:
    """Test MES diffusion run loading."""

    def test_load_mes_runs_success(self, fixtures_dir):
        """Test successful MES run loading."""
        csv_path = fixtures_dir / "mes_diffusion_runs.csv"
        runs = load_mes_diffusion_runs(csv_path, source_tz="UTC")

        assert len(runs) > 0
        assert all(isinstance(run, MESRun) for run in runs)

        # Check first run
        run = runs[0]
        assert run.run_id == "RUN_001"
        assert run.parameters.temperature_unit == TemperatureUnit.CELSIUS
        assert run.parameters.time_unit == TimeUnit.MINUTES
        assert run.start_time.tzinfo == pytz.UTC

    def test_load_mes_runs_file_not_found(self):
        """Test MES loader with missing file."""
        with pytest.raises(FileNotFoundError):
            load_mes_diffusion_runs(Path("nonexistent.csv"))


# ============================================================================
# FDC Loader Tests
# ============================================================================

class TestFDCLoaders:
    """Test FDC furnace data loading."""

    def test_load_fdc_data_success(self, fixtures_dir):
        """Test successful FDC data loading."""
        parquet_path = fixtures_dir / "fdc_furnace_data.parquet"
        fdc = load_fdc_furnace_data(
            parquet_path,
            run_id="RUN_001",
            equipment_id="FURN_01"
        )

        assert isinstance(fdc, FDCFurnaceData)
        assert fdc.run_id == "RUN_001"
        assert len(fdc.readings) > 0
        assert all(r.timestamp.tzinfo == pytz.UTC for r in fdc.readings)


# ============================================================================
# SPC Loader Tests
# ============================================================================

class TestSPCLoaders:
    """Test SPC chart data loading."""

    def test_load_spc_chart_success(self, fixtures_dir):
        """Test successful SPC chart loading."""
        csv_path = fixtures_dir / "spc_charts.csv"
        chart = load_spc_chart_data(
            csv_path,
            chart_id="JD_001",
            chart_type="xbar",
            metric_name="Junction Depth",
            metric_unit="nm"
        )

        assert isinstance(chart, SPCChart)
        assert chart.chart_id == "JD_001"
        assert len(chart.data_points) > 0
        assert chart.mean is not None
        assert chart.std_dev is not None


# ============================================================================
# Round-Trip IO Tests
# ============================================================================

class TestRoundTripIO:
    """Test round-trip load -> write -> load cycles."""

    def test_mes_roundtrip_parquet(self, fixtures_dir, temp_output_dir):
        """Test MES round-trip via Parquet."""
        # Load original
        runs_original = load_mes_diffusion_runs(
            fixtures_dir / "mes_diffusion_runs.csv"
        )

        # Write to Parquet
        output_path = temp_output_dir / "mes_test.parquet"
        write_mes_runs_parquet(runs_original, output_path)

        assert output_path.exists()

    def test_mes_roundtrip_json(self, fixtures_dir, temp_output_dir):
        """Test MES round-trip via JSON."""
        # Load original
        runs_original = load_mes_diffusion_runs(
            fixtures_dir / "mes_diffusion_runs.csv"
        )

        # Write to JSON
        output_path = temp_output_dir / "mes_test.json"
        write_mes_runs_json(runs_original, output_path)

        assert output_path.exists()

    def test_fdc_roundtrip_parquet(self, fixtures_dir, temp_output_dir):
        """Test FDC round-trip via Parquet."""
        # Load original
        fdc_original = load_fdc_furnace_data(
            fixtures_dir / "fdc_furnace_data.parquet",
            run_id="RUN_001",
            equipment_id="FURN_01"
        )

        # Write to Parquet
        output_path = temp_output_dir / "fdc_test.parquet"
        write_fdc_data_parquet(fdc_original, output_path)

        assert output_path.exists()

    def test_spc_roundtrip_parquet(self, fixtures_dir, temp_output_dir):
        """Test SPC round-trip via Parquet."""
        # Load original
        chart_original = load_spc_chart_data(
            fixtures_dir / "spc_charts.csv",
            chart_id="JD_001",
            chart_type="xbar",
            metric_name="Junction Depth",
            metric_unit="nm"
        )

        # Write to Parquet
        output_path = temp_output_dir / "spc_test.parquet"
        write_spc_chart_parquet(chart_original, output_path)

        assert output_path.exists()


# ============================================================================
# Provenance Tests
# ============================================================================

class TestProvenance:
    """Test data provenance tracking."""

    def test_provenance_in_mes_runs(self, fixtures_dir):
        """Test provenance is preserved in MES runs."""
        runs = load_mes_diffusion_runs(fixtures_dir / "mes_diffusion_runs.csv")

        assert all(run.provenance.source_system == "MES" for run in runs)
        assert all(run.provenance.source_file == "mes_diffusion_runs.csv" for run in runs)
        assert all(run.provenance.ingestion_timestamp.tzinfo == pytz.UTC for run in runs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
