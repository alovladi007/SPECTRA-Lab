#!/usr/bin/env python3
"""
SPC Watch - Session 10

Monitors KPI time series data for out-of-control conditions using:
- Western Electric & Nelson Rules
- EWMA (Exponentially Weighted Moving Average)
- CUSUM (Cumulative Sum)
- BOCPD (Bayesian Online Change Point Detection)

Usage:
    spc_watch.py --series kpi.csv --report spc.json [--methods all] [--verbose]
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from session7.spc import (
        quick_rule_check,
        quick_ewma_check,
        quick_cusum_check,
        quick_bocpd_check,
        RuleViolationType,
    )
except ImportError as e:
    print(f"Error importing SPC modules: {e}")
    print("Ensure Session 7 (SPC module) is available in the path")
    sys.exit(1)


def validate_input_csv(csv_path: Path) -> pd.DataFrame:
    """
    Validate input CSV has required columns.

    Required columns:
    - timestamp: Timestamp (ISO format or parseable date/time)
    - value: Numeric KPI value

    Optional columns:
    - kpi_name: Name of the KPI being monitored

    Returns:
        Validated DataFrame with parsed timestamps
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Input file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # Check required columns
    required_cols = ['timestamp', 'value']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Parse timestamps
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except Exception as e:
        raise ValueError(f"Failed to parse timestamps: {e}")

    # Validate numeric values
    if not pd.api.types.is_numeric_dtype(df['value']):
        try:
            df['value'] = pd.to_numeric(df['value'])
        except Exception:
            raise ValueError("Column 'value' must contain numeric values")

    # Check for NaN values
    if df['value'].isna().any():
        n_nan = df['value'].isna().sum()
        raise ValueError(f"Found {n_nan} NaN values in 'value' column")

    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def run_spc_rules(data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
    """
    Run Western Electric & Nelson SPC rules.

    Args:
        data: KPI values
        timestamps: Timestamps for each value

    Returns:
        Dictionary with violations and summary
    """
    violations = quick_rule_check(data, timestamps=timestamps)

    violations_list = []
    for v in violations:
        violations_list.append({
            'rule': v.rule.value,
            'index': int(v.index),
            'timestamp': v.timestamp.isoformat() if v.timestamp else None,
            'severity': v.severity.value,
            'description': v.description,
            'affected_indices': [int(i) for i in v.affected_indices],
            'metric_value': float(v.metric_value)
        })

    # Count violations by rule
    rule_counts = {}
    for v in violations:
        rule = v.rule.value
        rule_counts[rule] = rule_counts.get(rule, 0) + 1

    # Count by severity
    severity_counts = {}
    for v in violations:
        sev = v.severity.value
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    return {
        'method': 'spc_rules',
        'n_violations': len(violations),
        'violations': violations_list,
        'rule_counts': rule_counts,
        'severity_counts': severity_counts
    }


def run_ewma(data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
    """
    Run EWMA (Exponentially Weighted Moving Average) monitoring.

    Args:
        data: KPI values
        timestamps: Timestamps for each value

    Returns:
        Dictionary with EWMA violations and statistics
    """
    violations = quick_ewma_check(data, timestamps=timestamps)

    violations_list = []
    for v in violations:
        violations_list.append({
            'index': int(v.index),
            'timestamp': v.timestamp.isoformat() if v.timestamp else None,
            'ewma_value': float(v.ewma_value),
            'control_limit': float(v.control_limit),
            'side': v.side,
            'description': v.description
        })

    return {
        'method': 'ewma',
        'n_violations': len(violations),
        'violations': violations_list
    }


def run_cusum(data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
    """
    Run CUSUM (Cumulative Sum) monitoring.

    Args:
        data: KPI values
        timestamps: Timestamps for each value

    Returns:
        Dictionary with CUSUM violations
    """
    violations = quick_cusum_check(data, timestamps=timestamps)

    violations_list = []
    for v in violations:
        violations_list.append({
            'index': int(v.index),
            'timestamp': v.timestamp.isoformat() if v.timestamp else None,
            'cusum_value': float(v.cusum_value),
            'threshold': float(v.threshold),
            'side': v.side,
            'description': v.description
        })

    return {
        'method': 'cusum',
        'n_violations': len(violations),
        'violations': violations_list
    }


def run_bocpd(data: np.ndarray, timestamps: List[datetime]) -> Dict[str, Any]:
    """
    Run BOCPD (Bayesian Online Change Point Detection).

    Args:
        data: KPI values
        timestamps: Timestamps for each value

    Returns:
        Dictionary with detected change points
    """
    changepoints = quick_bocpd_check(data, timestamps=timestamps)

    changepoints_list = []
    for cp in changepoints:
        changepoints_list.append({
            'index': int(cp.index),
            'timestamp': cp.timestamp.isoformat() if cp.timestamp else None,
            'probability': float(cp.probability),
            'run_length': int(cp.run_length) if cp.run_length is not None else None,
            'description': cp.description
        })

    return {
        'method': 'bocpd',
        'n_changepoints': len(changepoints),
        'changepoints': changepoints_list
    }


def generate_summary(results: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for the report.

    Args:
        results: Results from all SPC methods
        df: Input DataFrame

    Returns:
        Summary dictionary
    """
    data = df['value'].values

    summary = {
        'n_observations': int(len(data)),
        'start_time': df['timestamp'].min().isoformat(),
        'end_time': df['timestamp'].max().isoformat(),
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'median': float(np.median(data)),
        'total_violations': 0,
        'total_changepoints': 0
    }

    # Count total violations
    if 'spc_rules' in results:
        summary['total_violations'] += results['spc_rules']['n_violations']
    if 'ewma' in results:
        summary['total_violations'] += results['ewma']['n_violations']
    if 'cusum' in results:
        summary['total_violations'] += results['cusum']['n_violations']
    if 'bocpd' in results:
        summary['total_changepoints'] = results['bocpd']['n_changepoints']

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='SPC monitoring for KPI time series data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --series kpi.csv --report spc.json
  %(prog)s --series kpi.csv --report spc.json --methods rules ewma cusum
  %(prog)s --series kpi.csv --report spc.json --methods all --verbose

Input CSV format:
  timestamp,value
  2025-01-01T00:00:00,100.5
  2025-01-01T01:00:00,102.3
  2025-01-01T02:00:00,98.7

Methods:
  rules  - Western Electric & Nelson SPC rules
  ewma   - Exponentially Weighted Moving Average
  cusum  - Cumulative Sum control chart
  bocpd  - Bayesian Online Change Point Detection
  all    - All methods (default)
        """
    )

    parser.add_argument('--series', required=True, type=Path,
                        help='Input CSV file with KPI time series data')
    parser.add_argument('--report', required=True, type=Path,
                        help='Output JSON file for SPC report')
    parser.add_argument('--methods', nargs='+',
                        choices=['rules', 'ewma', 'cusum', 'bocpd', 'all'],
                        default=['all'],
                        help='SPC methods to apply (default: all)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress messages')

    args = parser.parse_args()

    # Expand 'all' methods
    if 'all' in args.methods:
        methods = ['rules', 'ewma', 'cusum', 'bocpd']
    else:
        methods = args.methods

    # Validate input
    if args.verbose:
        print(f"Reading KPI data from: {args.series}")

    try:
        df = validate_input_csv(args.series)
    except Exception as e:
        print(f"Error validating input: {e}", file=sys.stderr)
        return 1

    if args.verbose:
        print(f"Loaded {len(df)} observations")
        print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"Value range: {df['value'].min():.2f} to {df['value'].max():.2f}")
        print(f"Methods to apply: {', '.join(methods)}")

    # Extract data
    data = df['value'].values
    timestamps = df['timestamp'].tolist()

    # Run SPC methods
    results = {}

    try:
        if 'rules' in methods:
            if args.verbose:
                print("\nRunning SPC rules...")
            results['spc_rules'] = run_spc_rules(data, timestamps)
            if args.verbose:
                print(f"  Found {results['spc_rules']['n_violations']} violations")

        if 'ewma' in methods:
            if args.verbose:
                print("\nRunning EWMA...")
            results['ewma'] = run_ewma(data, timestamps)
            if args.verbose:
                print(f"  Found {results['ewma']['n_violations']} violations")

        if 'cusum' in methods:
            if args.verbose:
                print("\nRunning CUSUM...")
            results['cusum'] = run_cusum(data, timestamps)
            if args.verbose:
                print(f"  Found {results['cusum']['n_violations']} violations")

        if 'bocpd' in methods:
            if args.verbose:
                print("\nRunning BOCPD...")
            results['bocpd'] = run_bocpd(data, timestamps)
            if args.verbose:
                print(f"  Found {results['bocpd']['n_changepoints']} change points")

    except Exception as e:
        print(f"Error running SPC analysis: {e}", file=sys.stderr)
        return 1

    # Generate summary
    summary = generate_summary(results, df)

    # Create report
    report = {
        'metadata': {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'input_file': str(args.series),
            'methods_applied': methods,
            'tool_version': '10.0.0'
        },
        'summary': summary,
        'results': results
    }

    # Write report
    try:
        with open(args.report, 'w') as f:
            json.dump(report, f, indent=2)
        if args.verbose:
            print(f"\nReport written to: {args.report}")
    except Exception as e:
        print(f"Error writing report: {e}", file=sys.stderr)
        return 1

    # Print summary
    if args.verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total observations: {summary['n_observations']}")
        print(f"Total violations: {summary['total_violations']}")
        print(f"Total change points: {summary['total_changepoints']}")

        if summary['total_violations'] > 0 or summary['total_changepoints'] > 0:
            print(f"\n⚠️  Process may be out of control - review report for details")
        else:
            print(f"\n✅  No violations or change points detected - process in control")

    return 0


if __name__ == '__main__':
    sys.exit(main())
