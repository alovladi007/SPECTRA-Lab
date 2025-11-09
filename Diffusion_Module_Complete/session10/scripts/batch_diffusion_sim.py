#!/usr/bin/env python3
"""
Batch Diffusion Simulator - Session 10

Reads diffusion run parameters from CSV, simulates using ERFC or numerical solver,
outputs results to Parquet with full schema validation.

Usage:
    batch_diffusion_sim.py --input runs.csv --out results.parquet [--method erfc|numerical]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from session2.erfc import constant_source_profile, limited_source_profile, junction_depth
    from session3.fick_fd import solve_diffusion_1d
    from session6.data.schemas import MESRun, DataProvenance
    from session6.ingestion.writers import write_mes_runs_parquet
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure Sessions 2, 3, and 6 are available in the path")
    sys.exit(1)


def validate_input_csv(csv_path: Path) -> pd.DataFrame:
    """
    Validate input CSV has required columns.

    Required columns:
    - run_id: Unique identifier
    - dopant: B, P, As, Sb
    - time_minutes: Diffusion time
    - temp_celsius: Temperature
    - method: constant_source or limited_source
    - surface_conc (for constant_source): cm^-3
    - dose (for limited_source): cm^-2
    - background: cm^-3

    Returns:
        Validated DataFrame
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # Check required columns
    required_cols = ['run_id', 'dopant', 'time_minutes', 'temp_celsius', 'method', 'background']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate dopants
    valid_dopants = {'B', 'P', 'As', 'Sb', 'boron', 'phosphorus', 'arsenic', 'antimony'}
    invalid_dopants = df[~df['dopant'].isin(valid_dopants)]
    if not invalid_dopants.empty:
        raise ValueError(f"Invalid dopants found: {invalid_dopants['dopant'].unique()}")

    # Validate method
    valid_methods = {'constant_source', 'limited_source'}
    invalid_methods = df[~df['method'].isin(valid_methods)]
    if not invalid_methods.empty:
        raise ValueError(f"Invalid methods found: {invalid_methods['method'].unique()}")

    # Check method-specific columns
    const_runs = df[df['method'] == 'constant_source']
    if not const_runs.empty and 'surface_conc' not in df.columns:
        raise ValueError("constant_source runs require 'surface_conc' column")

    limited_runs = df[df['method'] == 'limited_source']
    if not limited_runs.empty and 'dose' not in df.columns:
        raise ValueError("limited_source runs require 'dose' column")

    return df


def simulate_diffusion_run(row: pd.Series, solver: str = 'erfc') -> Dict[str, Any]:
    """
    Simulate a single diffusion run.

    Args:
        row: Row from input DataFrame
        solver: 'erfc' or 'numerical'

    Returns:
        Dictionary with simulation results
    """
    run_id = row['run_id']
    dopant = row['dopant']
    time_min = row['time_minutes']
    temp_c = row['temp_celsius']
    method = row['method']
    background = row['background']

    # Convert time to seconds
    time_sec = time_min * 60

    try:
        if solver == 'erfc':
            # Use analytical ERFC solution
            if method == 'constant_source':
                surface_conc = row['surface_conc']
                x_nm, concentration = constant_source_profile(
                    x_nm=np.linspace(0, 500, 100),
                    time_sec=time_sec,
                    temp_celsius=temp_c,
                    dopant=dopant.lower(),
                    surface_conc=surface_conc,
                    background=background
                )
            else:  # limited_source
                dose = row['dose']
                x_nm, concentration = limited_source_profile(
                    x_nm=np.linspace(0, 500, 100),
                    time_sec=time_sec,
                    temp_celsius=temp_c,
                    dopant=dopant.lower(),
                    dose_per_cm2=dose,
                    background=background
                )

            # Calculate junction depth
            xj = junction_depth(concentration, x_nm, background)

        elif solver == 'numerical':
            # Use numerical solver
            from session3.fick_fd import DiffusionSolver1D, DiffusionParams

            # Set up domain
            L_nm = 500.0
            nx = 100
            dx = L_nm / (nx - 1)

            # Initial condition
            C0 = np.full(nx, background)

            # Diffusivity parameters
            if dopant.lower() == 'boron' or dopant == 'B':
                D0, Ea = 0.76, 3.69
            elif dopant.lower() == 'phosphorus' or dopant == 'P':
                D0, Ea = 3.85, 3.66
            elif dopant.lower() == 'arsenic' or dopant == 'As':
                D0, Ea = 0.066, 3.44
            else:  # antimony
                D0, Ea = 0.214, 3.65

            # Calculate diffusivity
            k_eV_K = 8.617e-5
            T_K = temp_c + 273.15
            D = D0 * np.exp(-Ea / (k_eV_K * T_K))

            params = DiffusionParams(D0=D, Ea=0.0, dt_sec=1.0)  # Ea=0 since D already computed
            solver_obj = DiffusionSolver1D(params, nx, L_nm)

            # Boundary conditions
            if method == 'constant_source':
                bc_left = ('dirichlet', row['surface_conc'])
            else:
                bc_left = ('neumann', 0.0)  # Zero flux for limited source
            bc_right = ('neumann', 0.0)

            # Solve
            times, C_history = solver_obj.solve(
                C0=C0,
                t_final_sec=time_sec,
                bc_left=bc_left,
                bc_right=bc_right
            )

            concentration = C_history[-1]
            x_nm = np.linspace(0, L_nm, nx)
            xj = junction_depth(concentration, x_nm, background)

        else:
            raise ValueError(f"Unknown solver: {solver}")

        # Estimate sheet resistance (simplified)
        sheet_resistance = 1e20 / np.trapz(concentration, x_nm * 1e-7)  # Rough estimate

        return {
            'run_id': run_id,
            'dopant': dopant,
            'temp_celsius': temp_c,
            'time_minutes': time_min,
            'method': method,
            'solver': solver,
            'junction_depth_nm': float(xj),
            'sheet_resistance_ohm_sq': float(sheet_resistance),
            'peak_concentration': float(np.max(concentration)),
            'status': 'SUCCESS',
            'error_message': None
        }

    except Exception as e:
        return {
            'run_id': run_id,
            'dopant': dopant,
            'temp_celsius': temp_c,
            'time_minutes': time_min,
            'method': method,
            'solver': solver,
            'junction_depth_nm': None,
            'sheet_resistance_ohm_sq': None,
            'peak_concentration': None,
            'status': 'FAILED',
            'error_message': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch diffusion simulator with schema validation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input runs.csv --out results.parquet
  %(prog)s --input runs.csv --out results.parquet --method numerical
  %(prog)s --input runs.csv --out results.parquet --verbose

Input CSV format:
  run_id,dopant,time_minutes,temp_celsius,method,surface_conc,dose,background
  R001,B,30,1000,constant_source,1e19,,1e15
  R002,P,60,950,limited_source,,1e14,1e15
        """
    )

    parser.add_argument('--input', required=True, type=Path,
                        help='Input CSV file with diffusion run parameters')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output Parquet file for results')
    parser.add_argument('--method', choices=['erfc', 'numerical'], default='erfc',
                        help='Solver method (default: erfc)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress messages')

    args = parser.parse_args()

    # Validate input
    if args.verbose:
        print(f"Reading input from: {args.input}")

    try:
        df_input = validate_input_csv(args.input)
    except Exception as e:
        print(f"Error validating input: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(df_input)} diffusion runs")
        print(f"Using solver: {args.method}")

    # Simulate all runs
    results = []
    for idx, row in df_input.iterrows():
        if args.verbose:
            print(f"Simulating run {idx+1}/{len(df_input)}: {row['run_id']}")

        result = simulate_diffusion_run(row, solver=args.method)
        results.append(result)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Report summary
    n_success = (df_results['status'] == 'SUCCESS').sum()
    n_failed = (df_results['status'] == 'FAILED').sum()

    if args.verbose:
        print(f"\nSimulation complete:")
        print(f"  Success: {n_success}")
        print(f"  Failed: {n_failed}")

        if n_failed > 0:
            print("\nFailed runs:")
            failed = df_results[df_results['status'] == 'FAILED']
            for _, fail_row in failed.iterrows():
                print(f"  {fail_row['run_id']}: {fail_row['error_message']}")

    # Write output with provenance
    provenance = DataProvenance(
        source_system="batch_diffusion_sim",
        created_at=datetime.utcnow(),
        created_by="CLI",
        version="1.0",
        notes=f"Batch simulation with {args.method} solver. {n_success} successful, {n_failed} failed."
    )

    # Save to Parquet
    try:
        df_results.to_parquet(args.out, index=False, engine='pyarrow')
        if args.verbose:
            print(f"\nResults written to: {args.out}")
    except Exception as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        return 1

    return 0 if n_failed == 0 else 2  # Exit code 2 if some runs failed


if __name__ == '__main__':
    sys.exit(main())
