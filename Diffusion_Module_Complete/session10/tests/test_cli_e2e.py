"""
End-to-End tests for Session 10 CLI tools.

Tests all three CLI tools with fixtures and validates outputs.
"""

import pytest
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def diffusion_input_csv(temp_dir):
    """Create sample diffusion input CSV."""
    csv_path = temp_dir / "diffusion_runs.csv"

    data = {
        'run_id': ['R001', 'R002', 'R003', 'R004'],
        'dopant': ['B', 'P', 'As', 'boron'],
        'time_minutes': [30, 60, 45, 30],
        'temp_celsius': [1000, 950, 1050, 1000],
        'method': ['constant_source', 'limited_source', 'constant_source', 'limited_source'],
        'surface_conc': [1e19, None, 5e18, None],
        'dose': [None, 1e14, None, 5e13],
        'background': [1e15, 1e15, 1e15, 1e15]
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def oxidation_input_csv(temp_dir):
    """Create sample oxidation input CSV."""
    csv_path = temp_dir / "oxidation_recipes.csv"

    data = {
        'recipe_id': ['OX001', 'OX002', 'OX003', 'OX004'],
        'temp_celsius': [1000, 1100, 950, 1000],
        'time_hours': [2.0, 1.0, 3.0, 1.5],
        'ambient': ['dry', 'wet', 'dry', 'wet'],
        'pressure': [1.0, 1.0, 0.8, 1.0],
        'initial_thickness_nm': [0.0, 5.0, 0.0, 10.0]
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def spc_input_csv(temp_dir):
    """Create sample SPC time series CSV."""
    csv_path = temp_dir / "kpi_series.csv"

    # Generate time series with some violations
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    n_points = 100

    timestamps = []
    values = []

    for i in range(n_points):
        timestamps.append((start_time + timedelta(hours=i)).isoformat())

        # Base value with some variation
        if i < 50:
            value = 100.0 + np.random.normal(0, 2)
        else:
            # Introduce shift (should trigger SPC rules)
            value = 110.0 + np.random.normal(0, 2)

        # Add outlier
        if i == 25:
            value = 125.0  # Should trigger Rule 1

        values.append(value)

    data = {
        'timestamp': timestamps,
        'value': values
    }

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    return csv_path


class TestBatchDiffusionSim:
    """Test batch_diffusion_sim.py CLI tool."""

    def test_basic_simulation(self, diffusion_input_csv, temp_dir):
        """Test basic diffusion simulation."""
        output_path = temp_dir / "diffusion_results.parquet"

        cmd = [
            "python3",
            "session10/scripts/batch_diffusion_sim.py",
            "--input", str(diffusion_input_csv),
            "--out", str(output_path),
            "--method", "erfc",
            "--verbose"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check success
        assert result.returncode in [0, 2], f"Command failed: {result.stderr}"

        # Check output file exists
        assert output_path.exists(), "Output parquet file not created"

        # Load and validate results
        df = pd.read_parquet(output_path)

        # Check columns
        expected_cols = [
            'run_id', 'dopant', 'temp_celsius', 'time_minutes',
            'junction_depth_nm', 'sheet_resistance_ohm_sq',
            'peak_concentration', 'status', 'error_message'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check all runs processed
        assert len(df) == 4, f"Expected 4 runs, got {len(df)}"

        # Check at least some succeeded
        n_success = (df['status'] == 'SUCCESS').sum()
        assert n_success > 0, "No successful runs"

        # Validate successful runs have valid junction depths
        success_df = df[df['status'] == 'SUCCESS']
        assert (success_df['junction_depth_nm'] > 0).all(), "Invalid junction depths"

    def test_numerical_solver(self, diffusion_input_csv, temp_dir):
        """Test with numerical solver."""
        output_path = temp_dir / "diffusion_numerical.parquet"

        cmd = [
            "python3",
            "session10/scripts/batch_diffusion_sim.py",
            "--input", str(diffusion_input_csv),
            "--out", str(output_path),
            "--method", "numerical"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode in [0, 2]
        assert output_path.exists()


class TestBatchOxidationSim:
    """Test batch_oxidation_sim.py CLI tool."""

    def test_basic_oxidation(self, oxidation_input_csv, temp_dir):
        """Test basic oxidation simulation."""
        output_path = temp_dir / "oxidation_results.parquet"

        cmd = [
            "python3",
            "session10/scripts/batch_oxidation_sim.py",
            "--input", str(oxidation_input_csv),
            "--out", str(output_path),
            "--verbose"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check success
        assert result.returncode in [0, 2], f"Command failed: {result.stderr}"

        # Check output file exists
        assert output_path.exists(), "Output parquet file not created"

        # Load and validate results
        df = pd.read_parquet(output_path)

        # Check columns
        expected_cols = [
            'recipe_id', 'temp_celsius', 'time_hours', 'ambient',
            'final_thickness_nm', 'growth_thickness_nm', 'growth_rate_nm_hr',
            'B_parabolic_nm2_hr', 'A_linear_nm', 'status'
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

        # Check all runs processed
        assert len(df) == 4, f"Expected 4 runs, got {len(df)}"

        # Check all succeeded
        assert (df['status'] == 'SUCCESS').all(), "Some runs failed"

        # Validate thicknesses are positive
        assert (df['final_thickness_nm'] > 0).all(), "Invalid thicknesses"

        # Validate growth thickness
        assert (df['growth_thickness_nm'] >= 0).all(), "Invalid growth thickness"

    def test_dry_vs_wet(self, temp_dir):
        """Test dry vs wet oxidation produces different results."""
        csv_path = temp_dir / "dry_wet.csv"

        data = {
            'recipe_id': ['DRY', 'WET'],
            'temp_celsius': [1000, 1000],
            'time_hours': [2.0, 2.0],
            'ambient': ['dry', 'wet'],
        }
        pd.DataFrame(data).to_csv(csv_path, index=False)

        output_path = temp_dir / "dry_wet_results.parquet"

        cmd = [
            "python3",
            "session10/scripts/batch_oxidation_sim.py",
            "--input", str(csv_path),
            "--out", str(output_path)
        ]

        subprocess.run(cmd, check=True)

        df = pd.read_parquet(output_path)

        # Wet oxidation should produce thicker oxide
        dry_thickness = df[df['ambient'] == 'dry']['final_thickness_nm'].values[0]
        wet_thickness = df[df['ambient'] == 'wet']['final_thickness_nm'].values[0]

        assert wet_thickness > dry_thickness, "Wet oxidation should be thicker than dry"


class TestSPCWatch:
    """Test spc_watch.py CLI tool."""

    def test_basic_spc_monitoring(self, spc_input_csv, temp_dir):
        """Test basic SPC monitoring."""
        report_path = temp_dir / "spc_report.json"

        cmd = [
            "python3",
            "session10/scripts/spc_watch.py",
            "--series", str(spc_input_csv),
            "--report", str(report_path),
            "--methods", "rules",
            "--verbose"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Check success
        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check report file exists
        assert report_path.exists(), "Report JSON not created"

        # Load and validate report
        with open(report_path, 'r') as f:
            report = json.load(f)

        # Check structure
        assert 'metadata' in report
        assert 'summary' in report
        assert 'results' in report

        # Check metadata
        assert 'generated_at' in report['metadata']
        assert 'methods_applied' in report['metadata']

        # Check summary
        summary = report['summary']
        assert 'n_observations' in summary
        assert summary['n_observations'] == 100
        assert 'mean' in summary
        assert 'std' in summary

        # Check results
        assert 'spc_rules' in report['results']
        spc_results = report['results']['spc_rules']
        assert 'n_violations' in spc_results
        assert 'violations' in spc_results

    def test_multiple_methods(self, spc_input_csv, temp_dir):
        """Test with multiple SPC methods."""
        report_path = temp_dir / "spc_multi.json"

        cmd = [
            "python3",
            "session10/scripts/spc_watch.py",
            "--series", str(spc_input_csv),
            "--report", str(report_path),
            "--methods", "rules", "ewma", "cusum", "bocpd"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        assert result.returncode == 0

        with open(report_path, 'r') as f:
            report = json.load(f)

        # Check all methods present
        results = report['results']
        assert 'spc_rules' in results
        assert 'ewma' in results
        assert 'cusum' in results
        assert 'bocpd' in results

    def test_all_methods_shortcut(self, spc_input_csv, temp_dir):
        """Test 'all' methods shortcut."""
        report_path = temp_dir / "spc_all.json"

        cmd = [
            "python3",
            "session10/scripts/spc_watch.py",
            "--series", str(spc_input_csv),
            "--report", str(report_path),
            "--methods", "all"
        ]

        subprocess.run(cmd, check=True)

        with open(report_path, 'r') as f:
            report = json.load(f)

        # All four methods should be present
        assert len(report['results']) == 4


class TestIntegration:
    """Integration tests combining multiple tools."""

    def test_full_workflow(self, temp_dir):
        """Test complete workflow: diffusion -> oxidation -> SPC monitoring."""

        # 1. Run diffusion simulations
        diff_csv = temp_dir / "diff.csv"
        diff_data = {
            'run_id': ['D1', 'D2', 'D3'],
            'dopant': ['B', 'P', 'As'],
            'time_minutes': [30, 30, 30],
            'temp_celsius': [1000, 1000, 1000],
            'method': ['constant_source'] * 3,
            'surface_conc': [1e19] * 3,
            'background': [1e15] * 3
        }
        pd.DataFrame(diff_data).to_csv(diff_csv, index=False)

        diff_out = temp_dir / "diff_out.parquet"
        subprocess.run([
            "python3", "session10/scripts/batch_diffusion_sim.py",
            "--input", str(diff_csv),
            "--out", str(diff_out)
        ], check=True)

        # Verify diffusion results
        diff_df = pd.read_parquet(diff_out)
        assert len(diff_df) == 3
        assert (diff_df['status'] == 'SUCCESS').all()

        # 2. Run oxidation simulations
        ox_csv = temp_dir / "ox.csv"
        ox_data = {
            'recipe_id': ['O1', 'O2'],
            'temp_celsius': [1000, 1100],
            'time_hours': [1.0, 2.0],
            'ambient': ['dry', 'wet']
        }
        pd.DataFrame(ox_data).to_csv(ox_csv, index=False)

        ox_out = temp_dir / "ox_out.parquet"
        subprocess.run([
            "python3", "session10/scripts/batch_oxidation_sim.py",
            "--input", str(ox_csv),
            "--out", str(ox_out)
        ], check=True)

        # Verify oxidation results
        ox_df = pd.read_parquet(ox_out)
        assert len(ox_df) == 2

        # 3. Create KPI series from junction depths
        kpi_csv = temp_dir / "kpi.csv"
        timestamps = [
            (datetime(2025, 1, 1) + timedelta(days=i)).isoformat()
            for i in range(len(diff_df))
        ]
        kpi_data = {
            'timestamp': timestamps,
            'value': diff_df['junction_depth_nm'].tolist()
        }
        pd.DataFrame(kpi_data).to_csv(kpi_csv, index=False)

        spc_report = temp_dir / "spc.json"
        subprocess.run([
            "python3", "session10/scripts/spc_watch.py",
            "--series", str(kpi_csv),
            "--report", str(spc_report),
            "--methods", "rules"
        ], check=True)

        # Verify SPC report
        with open(spc_report, 'r') as f:
            report = json.load(f)
        assert report['summary']['n_observations'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
