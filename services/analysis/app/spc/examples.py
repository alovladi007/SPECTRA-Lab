"""
Examples for SPC Module

Demonstrates usage of X-bar/R, EWMA, CUSUM charts and Western Electric rules
for monitoring CVD film properties.
"""

import numpy as np
import logging
from datetime import datetime, timedelta

from .series import SPCSeries, SPCDataPoint, SPCMetric, create_spc_series
from .charts import XBarRChart, EWMAChart, CUSUMChart
from .rules import WesternElectricRules, check_all_rules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: X-bar/R Chart for Thickness Mean
# =============================================================================

def example_xbar_chart():
    """
    Demonstrate X-bar/R chart for thickness monitoring
    """
    logger.info("=" * 70)
    logger.info("Example 1: X-bar/R Chart for Thickness Mean")
    logger.info("=" * 70)

    # Generate synthetic thickness data (in-control process)
    np.random.seed(42)
    n_runs = 50
    process_mean = 100.0  # nm
    process_std = 2.0     # nm

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # In-control data
    thickness_values = np.random.normal(process_mean, process_std, n_runs)

    # Create SPC series
    series = create_spc_series(
        metric=SPCMetric.THICKNESS_MEAN,
        values=thickness_values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-01",
        recipe_id="SiN_LPCVD",
        film_material="Si3N4",
    )

    # Create X-bar/R chart (individuals chart with n=1)
    chart = XBarRChart(series, subgroup_size=1)

    # Run chart analysis
    result = chart.run_chart(baseline_points=20)

    # Print results
    logger.info(f"Chart type: {result.chart_type.value}")
    logger.info(f"Center line: {result.control_limits.center_line:.2f} nm")
    logger.info(f"UCL: {result.control_limits.upper_control_limit:.2f} nm")
    logger.info(f"LCL: {result.control_limits.lower_control_limit:.2f} nm")
    logger.info(f"Process mean: {result.control_limits.process_mean:.2f} nm")
    logger.info(f"Process std: {result.control_limits.process_std:.2f} nm")
    logger.info(f"Out-of-control points: {len(result.out_of_control)}")

    # Check Western Electric rules
    violations = check_all_rules(result.chart_values, result.control_limits)
    logger.info(f"Western Electric violations: {len(violations)}")

    for v in violations[:5]:  # Show first 5
        logger.info(f"  [{v.severity}] Point {v.point_index}: {v.description}")


# =============================================================================
# Example 2: EWMA Chart for Detecting Gradual Drift
# =============================================================================

def example_ewma_chart():
    """
    Demonstrate EWMA chart for detecting gradual drift in stress
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: EWMA Chart for Stress Drift Detection")
    logger.info("=" * 70)

    # Generate synthetic stress data with gradual drift
    np.random.seed(43)
    n_runs = 60
    process_mean = -250.0  # MPa (compressive)
    process_std = 10.0     # MPa

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # Create drift: gradual increase from run 30 onwards
    stress_values = np.random.normal(process_mean, process_std, n_runs)

    # Add drift (stress becomes less compressive)
    for i in range(30, n_runs):
        drift = (i - 30) * 0.5  # +0.5 MPa per run
        stress_values[i] += drift

    # Create SPC series
    series = create_spc_series(
        metric=SPCMetric.STRESS_MEAN,
        values=stress_values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-01",
        recipe_id="SiN_LPCVD",
        film_material="Si3N4",
    )

    # Create EWMA chart
    chart = EWMAChart(series, lambda_=0.2)

    # Run chart analysis
    result = chart.run_chart(baseline_points=25)

    # Print results
    logger.info(f"Chart type: {result.chart_type.value}")
    logger.info(f"Center line: {result.control_limits.center_line:.2f} MPa")
    logger.info(f"UCL: {result.control_limits.upper_control_limit:.2f} MPa")
    logger.info(f"LCL: {result.control_limits.lower_control_limit:.2f} MPa")
    logger.info(f"Out-of-control points: {len(result.out_of_control)}")

    if len(result.out_of_control) > 0:
        logger.info(f"First out-of-control at point: {result.out_of_control[0]}")
        logger.info(f"  Original value: {result.original_values[result.out_of_control[0]]:.2f} MPa")
        logger.info(f"  EWMA value: {result.chart_values[result.out_of_control[0]]:.2f} MPa")

    # Check Western Electric rules
    violations = check_all_rules(result.chart_values, result.control_limits)
    logger.info(f"Western Electric violations: {len(violations)}")


# =============================================================================
# Example 3: CUSUM Chart for Detecting Small Shifts
# =============================================================================

def example_cusum_chart():
    """
    Demonstrate CUSUM chart for detecting small sustained shifts
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: CUSUM Chart for Adhesion Score Shift")
    logger.info("=" * 70)

    # Generate synthetic adhesion score data with shift
    np.random.seed(44)
    n_runs = 50
    process_mean = 85.0  # Initial good adhesion
    process_std = 3.0

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # Create shift: sudden drop at run 25
    adhesion_values = np.random.normal(process_mean, process_std, n_runs)

    # Add shift (adhesion drops)
    for i in range(25, n_runs):
        adhesion_values[i] -= 5.0  # -5 point drop

    # Create SPC series
    series = create_spc_series(
        metric=SPCMetric.ADHESION_SCORE,
        values=adhesion_values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-01",
        recipe_id="TiN_PECVD",
        film_material="TiN",
    )

    # Create CUSUM chart
    chart = CUSUMChart(series, k_sigma=0.5, h_sigma=5.0)

    # Run chart analysis
    result = chart.run_chart(baseline_points=20)

    # Print results
    logger.info(f"Chart type: {result.chart_type.value}")
    logger.info(f"Target: {result.control_limits.center_line:.2f}")
    logger.info(f"Decision interval H: {result.control_limits.upper_control_limit:.2f}")
    logger.info(f"Out-of-control points: {len(result.out_of_control)}")

    if len(result.out_of_control) > 0:
        logger.info(f"First out-of-control at point: {result.out_of_control[0]}")
        logger.info(f"  Original value: {result.original_values[result.out_of_control[0]]:.2f}")
        logger.info(f"  CUSUM value: {result.chart_values[result.out_of_control[0]]:.2f}")


# =============================================================================
# Example 4: Western Electric Rules Demonstration
# =============================================================================

def example_western_electric_rules():
    """
    Demonstrate all Western Electric rules with synthetic patterns
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Western Electric Rules")
    logger.info("=" * 70)

    # Generate data with various rule violations
    np.random.seed(45)
    n_runs = 100
    process_mean = 100.0
    process_std = 2.0

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # Generate mostly in-control data
    values = np.random.normal(process_mean, process_std, n_runs)

    # Inject violations:

    # Rule 1: Point beyond 3σ
    values[10] = process_mean + 4 * process_std

    # Rule 4: 8 consecutive points above center
    for i in range(20, 28):
        values[i] = process_mean + 0.5 * process_std

    # Rule 5: 6 points trending up
    for i in range(40, 46):
        values[i] = process_mean + (i - 40) * 0.3 * process_std

    # Create series
    series = create_spc_series(
        metric=SPCMetric.THICKNESS_UNIFORMITY,
        values=values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-02",
        recipe_id="SiO2_PECVD",
        film_material="SiO2",
    )

    # Create chart
    chart = XBarRChart(series, subgroup_size=1)
    result = chart.run_chart()

    # Check all rules
    violations = check_all_rules(result.chart_values, result.control_limits)

    logger.info(f"Total violations found: {len(violations)}")
    logger.info("\nViolations by rule:")

    # Group by violation type
    violation_counts = {}
    for v in violations:
        rule_name = v.violation_type.value
        violation_counts[rule_name] = violation_counts.get(rule_name, 0) + 1

    for rule, count in sorted(violation_counts.items()):
        logger.info(f"  {rule}: {count} violations")

    logger.info("\nDetailed violations:")
    for v in violations[:10]:  # Show first 10
        logger.info(f"  Point {v.point_index} [{v.severity}]: {v.description}")


# =============================================================================
# Example 5: Multi-Tool Comparison
# =============================================================================

def example_multi_tool_comparison():
    """
    Demonstrate SPC monitoring across multiple tools
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Multi-Tool SPC Comparison")
    logger.info("=" * 70)

    # Generate data for 3 tools
    tools = ["CVD-01", "CVD-02", "CVD-03"]
    np.random.seed(46)

    n_runs_per_tool = 30

    all_values = []
    all_timestamps = []
    all_run_ids = []
    all_tools = []

    for tool_idx, tool_id in enumerate(tools):
        # Each tool has slightly different mean (tool-to-tool variation)
        tool_mean = 100.0 + tool_idx * 1.0  # CVD-01: 100, CVD-02: 101, CVD-03: 102
        tool_std = 2.0

        values = np.random.normal(tool_mean, tool_std, n_runs_per_tool)
        timestamps = [datetime.now() + timedelta(hours=tool_idx * n_runs_per_tool + i) for i in range(n_runs_per_tool)]
        run_ids = [f"{tool_id}_RUN_{i:03d}" for i in range(n_runs_per_tool)]

        all_values.extend(values)
        all_timestamps.extend(timestamps)
        all_run_ids.extend(run_ids)
        all_tools.extend([tool_id] * n_runs_per_tool)

    # Create overall series
    overall_series = SPCSeries(metric=SPCMetric.THICKNESS_MEAN, recipe_id="SiN_LPCVD")

    for value, timestamp, run_id, tool_id in zip(all_values, all_timestamps, all_run_ids, all_tools):
        point = SPCDataPoint(
            timestamp=timestamp,
            value=value,
            run_id=run_id,
            tool_id=tool_id,
        )
        overall_series.add_point(point)

    overall_series.calculate_statistics()

    logger.info(f"Overall process mean: {overall_series.process_mean:.2f} nm")
    logger.info(f"Overall process std: {overall_series.process_std:.2f} nm")

    # Analyze each tool separately
    logger.info("\nPer-tool statistics:")

    for tool_id in tools:
        tool_series = overall_series.filter_by_tool(tool_id)

        chart = XBarRChart(tool_series, subgroup_size=1)
        result = chart.run_chart()

        logger.info(f"\n{tool_id}:")
        logger.info(f"  Mean: {result.control_limits.process_mean:.2f} nm")
        logger.info(f"  Std: {result.control_limits.process_std:.2f} nm")
        logger.info(f"  UCL: {result.control_limits.upper_control_limit:.2f} nm")
        logger.info(f"  LCL: {result.control_limits.lower_control_limit:.2f} nm")
        logger.info(f"  Out-of-control: {len(result.out_of_control)}")


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run all SPC examples"""
    logger.info("\n" + "=" * 70)
    logger.info("SPC Module - Examples")
    logger.info("=" * 70)

    example_xbar_chart()
    example_ewma_chart()
    example_cusum_chart()
    example_western_electric_rules()
    example_multi_tool_comparison()

    logger.info("\n" + "=" * 70)
    logger.info("All SPC examples completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
