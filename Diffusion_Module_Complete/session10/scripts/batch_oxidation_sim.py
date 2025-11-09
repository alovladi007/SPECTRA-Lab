#!/usr/bin/env python3
"""
Batch Oxidation Simulator - Session 10

Reads oxidation recipe parameters from CSV, simulates using Deal-Grove model,
outputs results to Parquet with full schema validation.

Usage:
    batch_oxidation_sim.py --input recipes.csv --out results.parquet [--verbose]
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
    from session4.deal_grove import (
        thickness_at_time,
        time_to_thickness,
        growth_rate,
        get_rate_constants
    )
except ImportError as e:
    print(f"Error importing Deal-Grove module: {e}")
    print("Ensure Session 4 (deal_grove.py) is available in the path")
    sys.exit(1)


def validate_input_csv(csv_path: Path) -> pd.DataFrame:
    """
    Validate input CSV has required columns.

    Required columns:
    - recipe_id: Unique identifier for the recipe
    - temp_celsius: Temperature (°C)
    - time_hours: Oxidation time (hours)
    - ambient: dry or wet oxidation

    Optional columns:
    - pressure: Partial pressure of oxidant (atm), default 1.0
    - initial_thickness_nm: Initial oxide thickness (nm), default 0.0

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
    required_cols = ['recipe_id', 'temp_celsius', 'time_hours', 'ambient']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Validate ambient
    valid_ambients = {'dry', 'wet', 'Dry', 'Wet', 'DRY', 'WET'}
    invalid_ambients = df[~df['ambient'].isin(valid_ambients)]
    if not invalid_ambients.empty:
        raise ValueError(f"Invalid ambient values found: {invalid_ambients['ambient'].unique()}")

    # Add optional columns with defaults
    if 'pressure' not in df.columns:
        df['pressure'] = 1.0

    if 'initial_thickness_nm' not in df.columns:
        df['initial_thickness_nm'] = 0.0

    # Validate numeric columns
    numeric_cols = ['temp_celsius', 'time_hours', 'pressure', 'initial_thickness_nm']
    for col in numeric_cols:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except Exception:
                    raise ValueError(f"Column '{col}' must contain numeric values")

            # Check for negative values
            if (df[col] < 0).any():
                raise ValueError(f"Column '{col}' contains negative values")

    # Validate temperature range (typical semiconductor processing)
    if (df['temp_celsius'] < 700).any() or (df['temp_celsius'] > 1300).any():
        print("Warning: Some temperatures are outside typical range (700-1300°C)", file=sys.stderr)

    # Validate time
    if (df['time_hours'] > 100).any():
        print("Warning: Some oxidation times exceed 100 hours", file=sys.stderr)

    return df


def simulate_oxidation_run(row: pd.Series) -> Dict[str, Any]:
    """
    Simulate a single oxidation run using Deal-Grove model.

    Args:
        row: Row from input DataFrame

    Returns:
        Dictionary with simulation results
    """
    recipe_id = row['recipe_id']
    temp_c = row['temp_celsius']
    time_hr = row['time_hours']
    ambient = row['ambient'].lower()
    pressure = row.get('pressure', 1.0)
    initial_thickness_nm = row.get('initial_thickness_nm', 0.0)

    # Convert initial thickness from nm to μm
    x_i_um = initial_thickness_nm / 1000.0

    try:
        # Calculate oxide thickness using Deal-Grove model
        thickness_um = thickness_at_time(
            t=time_hr,
            T=temp_c,
            ambient=ambient,
            pressure=pressure,
            x_i=x_i_um
        )

        # Convert thickness from μm to nm
        thickness_nm = thickness_um * 1000.0

        # Calculate instantaneous growth rate
        if thickness_um > 0:
            rate_um_hr = growth_rate(
                x_ox=thickness_um,
                T=temp_c,
                ambient=ambient,
                pressure=pressure
            )
            rate_nm_hr = rate_um_hr * 1000.0
        else:
            rate_nm_hr = 0.0

        # Get Deal-Grove parameters
        B, B_over_A = get_rate_constants(temp_c, ambient, pressure)
        A = B / B_over_A

        # Convert to nm units for output
        B_nm2_hr = B * 1e6  # μm²/hr to nm²/hr
        A_nm = A * 1000.0   # μm to nm

        # Calculate growth thickness (final - initial)
        growth_thickness_nm = thickness_nm - initial_thickness_nm

        return {
            'recipe_id': recipe_id,
            'temp_celsius': temp_c,
            'time_hours': time_hr,
            'ambient': ambient,
            'pressure': pressure,
            'initial_thickness_nm': float(initial_thickness_nm),
            'final_thickness_nm': float(thickness_nm),
            'growth_thickness_nm': float(growth_thickness_nm),
            'growth_rate_nm_hr': float(rate_nm_hr),
            'B_parabolic_nm2_hr': float(B_nm2_hr),
            'A_linear_nm': float(A_nm),
            'status': 'SUCCESS',
            'error_message': None
        }

    except Exception as e:
        return {
            'recipe_id': recipe_id,
            'temp_celsius': temp_c,
            'time_hours': time_hr,
            'ambient': ambient,
            'pressure': pressure,
            'initial_thickness_nm': float(initial_thickness_nm),
            'final_thickness_nm': None,
            'growth_thickness_nm': None,
            'growth_rate_nm_hr': None,
            'B_parabolic_nm2_hr': None,
            'A_linear_nm': None,
            'status': 'FAILED',
            'error_message': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Batch oxidation simulator using Deal-Grove model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input recipes.csv --out results.parquet
  %(prog)s --input recipes.csv --out results.parquet --verbose

Input CSV format:
  recipe_id,temp_celsius,time_hours,ambient,pressure,initial_thickness_nm
  OX001,1000,2.0,dry,1.0,0
  OX002,1100,1.0,wet,1.0,5.0

Required columns: recipe_id, temp_celsius, time_hours, ambient
Optional columns: pressure (default 1.0), initial_thickness_nm (default 0.0)
        """
    )

    parser.add_argument('--input', required=True, type=Path,
                        help='Input CSV file with oxidation recipe parameters')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output Parquet file for results')
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
        print(f"Loaded {len(df_input)} oxidation recipes")

    # Simulate all recipes
    results = []
    for idx, row in df_input.iterrows():
        if args.verbose:
            print(f"Simulating recipe {idx+1}/{len(df_input)}: {row['recipe_id']}")

        result = simulate_oxidation_run(row)
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
            print("\nFailed recipes:")
            failed = df_results[df_results['status'] == 'FAILED']
            for _, fail_row in failed.iterrows():
                print(f"  {fail_row['recipe_id']}: {fail_row['error_message']}")

        # Summary statistics for successful runs
        if n_success > 0:
            success_df = df_results[df_results['status'] == 'SUCCESS']
            print(f"\nSummary Statistics (successful runs):")
            print(f"  Oxide thickness: {success_df['final_thickness_nm'].mean():.2f} ± {success_df['final_thickness_nm'].std():.2f} nm")
            print(f"  Growth rate: {success_df['growth_rate_nm_hr'].mean():.2f} ± {success_df['growth_rate_nm_hr'].std():.2f} nm/hr")

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
