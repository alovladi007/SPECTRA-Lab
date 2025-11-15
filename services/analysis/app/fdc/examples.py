"""
Examples for FDC Module

Demonstrates fault detection and classification for CVD processes.
"""

import numpy as np
import logging
from datetime import datetime, timedelta

from .detector import FDCDetector, detect_faults
from .classifiers import classify_fault_root_cause
from .patterns import PatternDetector

from ..spc.series import SPCSeries, SPCMetric, create_spc_series
from ..spc.charts import XBarRChart, EWMAChart
from ..spc.rules import check_all_rules

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Thickness Drift Detection
# =============================================================================

def example_thickness_drift():
    """
    Demonstrate detection of thickness drift due to MFC calibration
    """
    logger.info("=" * 70)
    logger.info("Example 1: Thickness Drift Detection (MFC Calibration)")
    logger.info("=" * 70)

    # Generate thickness data with gradual upward drift
    np.random.seed(50)
    n_runs = 60
    process_mean = 100.0  # nm
    process_std = 2.0

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # Create drift starting at run 20
    thickness_values = np.random.normal(process_mean, process_std, n_runs)

    for i in range(20, n_runs):
        drift = (i - 20) * 0.1  # +0.1 nm per run
        thickness_values[i] += drift

    # Create series
    series = create_spc_series(
        metric=SPCMetric.THICKNESS_MEAN,
        values=thickness_values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-01",
        recipe_id="SiN_LPCVD",
        film_material="Si3N4",
    )

    # Run EWMA chart
    chart = EWMAChart(series, lambda_=0.2)
    result = chart.run_chart(baseline_points=20)

    # Check violations
    violations = check_all_rules(result.chart_values, result.control_limits)

    # Detect faults
    faults = detect_faults(series, result, violations)

    logger.info(f"Detected {len(faults)} fault(s)")

    for fault in faults:
        logger.info(f"\nFault: {fault.fault_type.value}")
        logger.info(f"  Severity: {fault.severity.value}")
        logger.info(f"  Description: {fault.description}")

        # Classify root cause
        fault_classified = classify_fault_root_cause(fault)

        if fault_classified.root_cause:
            logger.info(f"  Root cause: {fault_classified.root_cause.value} (confidence: {fault_classified.root_cause_confidence:.2f})")
            logger.info(f"  Action: {fault_classified.recommended_action}")


# =============================================================================
# Example 2: Stress Shift Detection
# =============================================================================

def example_stress_shift():
    """
    Demonstrate detection of sudden stress shift due to recipe change
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Stress Shift Detection (Recipe Change)")
    logger.info("=" * 70)

    # Generate stress data with sudden shift
    np.random.seed(51)
    n_runs = 50
    process_mean = -250.0  # MPa
    process_std = 10.0

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # Create sudden shift at run 25
    stress_values = np.random.normal(process_mean, process_std, n_runs)

    for i in range(25, n_runs):
        stress_values[i] += 30.0  # +30 MPa shift (less compressive)

    # Create series
    series = create_spc_series(
        metric=SPCMetric.STRESS_MEAN,
        values=stress_values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-01",
        recipe_id="SiN_LPCVD",
        film_material="Si3N4",
    )

    # Run X-bar chart
    chart = XBarRChart(series, subgroup_size=1)
    result = chart.run_chart(baseline_points=20)

    # Detect faults
    violations = check_all_rules(result.chart_values, result.control_limits)
    faults = detect_faults(series, result, violations)

    logger.info(f"Detected {len(faults)} fault(s)")

    for fault in faults:
        logger.info(f"\nFault: {fault.fault_type.value}")
        logger.info(f"  Severity: {fault.severity.value}")
        logger.info(f"  Description: {fault.description}")

        # Statistics
        if fault.statistics:
            logger.info(f"  Statistics:")
            for key, value in fault.statistics.items():
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.2f}")

        # Classify root cause
        fault_classified = classify_fault_root_cause(fault)

        if fault_classified.root_cause:
            logger.info(f"  Root cause: {fault_classified.root_cause.value} (confidence: {fault_classified.root_cause_confidence:.2f})")
            logger.info(f"  Action: {fault_classified.recommended_action}")


# =============================================================================
# Example 3: Adhesion Drop Detection
# =============================================================================

def example_adhesion_drop():
    """
    Demonstrate detection of sudden adhesion drop due to pre-clean failure
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Adhesion Drop Detection (Pre-clean Failure)")
    logger.info("=" * 70)

    # Generate adhesion data with sudden drop
    np.random.seed(52)
    n_runs = 40
    process_mean = 88.0  # Good adhesion
    process_std = 2.5

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # Create sudden drop at run 20
    adhesion_values = np.random.normal(process_mean, process_std, n_runs)

    for i in range(20, n_runs):
        adhesion_values[i] -= 15.0  # Drop to ~73 (MARGINAL)

    # Create series
    series = create_spc_series(
        metric=SPCMetric.ADHESION_SCORE,
        values=adhesion_values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-02",
        recipe_id="TiN_PECVD",
        film_material="TiN",
    )

    # Run chart
    chart = XBarRChart(series, subgroup_size=1)
    result = chart.run_chart(baseline_points=15)

    # Detect faults
    violations = check_all_rules(result.chart_values, result.control_limits)
    faults = detect_faults(series, result, violations)

    logger.info(f"Detected {len(faults)} fault(s)")

    for fault in faults:
        logger.info(f"\nFault: {fault.fault_type.value}")
        logger.info(f"  Severity: {fault.severity.value}")
        logger.info(f"  Description: {fault.description}")

        # Classify root cause
        fault_classified = classify_fault_root_cause(fault)

        if fault_classified.root_cause:
            logger.info(f"  Root cause: {fault_classified.root_cause.value} (confidence: {fault_classified.root_cause_confidence:.2f})")
            logger.info(f"  Action: {fault_classified.recommended_action}")

            # Show all hypotheses
            if "root_cause_hypotheses" in fault_classified.statistics:
                logger.info(f"\n  All root cause hypotheses:")
                for hyp in fault_classified.statistics["root_cause_hypotheses"][:3]:
                    logger.info(f"    - {hyp['root_cause']} (conf: {hyp['confidence']:.2f})")
                    logger.info(f"      {hyp['reasoning']}")


# =============================================================================
# Example 4: Pattern Detection
# =============================================================================

def example_pattern_detection():
    """
    Demonstrate advanced pattern detection
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Advanced Pattern Detection")
    logger.info("=" * 70)

    # Generate data with multiple patterns
    np.random.seed(53)
    n_runs = 100

    # Base signal
    t = np.arange(n_runs)

    # Add trend
    trend = 0.05 * t

    # Add cycle
    cycle = 2.0 * np.sin(2 * np.pi * t / 15)  # Period = 15

    # Add noise
    noise = np.random.normal(0, 0.5, n_runs)

    # Combine
    values = 100.0 + trend + cycle + noise

    # Detect patterns
    detector = PatternDetector(values)

    trend_pattern = detector.detect_trend()
    shift_pattern = detector.detect_shift()
    cyclic_pattern = detector.detect_cycles()

    logger.info("Pattern Detection Results:")

    if trend_pattern and trend_pattern.is_significant:
        logger.info(f"\nTrend detected:")
        logger.info(f"  Slope: {trend_pattern.slope:.4f}/run")
        logger.info(f"  R²: {trend_pattern.r_squared:.3f}")
        logger.info(f"  Significant: {trend_pattern.is_significant}")

    if shift_pattern and shift_pattern.is_significant:
        logger.info(f"\nShift detected:")
        logger.info(f"  Change point: Run {shift_pattern.change_point_index}")
        logger.info(f"  Magnitude: {shift_pattern.shift_magnitude:.2f}")
        logger.info(f"  Significant: {shift_pattern.is_significant}")

    if cyclic_pattern and cyclic_pattern.is_significant:
        logger.info(f"\nCycle detected:")
        logger.info(f"  Period: {cyclic_pattern.period:.1f} runs")
        logger.info(f"  Amplitude: {cyclic_pattern.amplitude:.2f}")
        logger.info(f"  Significant: {cyclic_pattern.is_significant}")


# =============================================================================
# Example 5: Increased Variation Detection
# =============================================================================

def example_increased_variation():
    """
    Demonstrate detection of increased process variation
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Increased Variation Detection (Temperature Instability)")
    logger.info("=" * 70)

    # Generate uniformity data with increased variation
    np.random.seed(54)
    n_runs = 60
    process_mean = 2.5  # % uniformity
    process_std_baseline = 0.3
    process_std_degraded = 0.8

    timestamps = [datetime.now() + timedelta(hours=i) for i in range(n_runs)]
    run_ids = [f"RUN_{i:03d}" for i in range(n_runs)]

    # First 35 runs: normal variation
    # Last 25 runs: increased variation
    uniformity_values = np.concatenate([
        np.random.normal(process_mean, process_std_baseline, 35),
        np.random.normal(process_mean, process_std_degraded, 25),
    ])

    # Create series
    series = create_spc_series(
        metric=SPCMetric.THICKNESS_UNIFORMITY,
        values=uniformity_values.tolist(),
        timestamps=timestamps,
        run_ids=run_ids,
        tool_id="CVD-03",
        recipe_id="SiO2_PECVD",
        film_material="SiO2",
    )

    # Run chart
    chart = XBarRChart(series, subgroup_size=1)
    result = chart.run_chart(baseline_points=30)

    # Detect faults
    violations = check_all_rules(result.chart_values, result.control_limits)
    faults = detect_faults(series, result, violations)

    logger.info(f"Detected {len(faults)} fault(s)")

    for fault in faults:
        logger.info(f"\nFault: {fault.fault_type.value}")
        logger.info(f"  Severity: {fault.severity.value}")
        logger.info(f"  Description: {fault.description}")

        if fault.statistics:
            logger.info(f"  Statistics:")
            for key, value in fault.statistics.items():
                if isinstance(value, float):
                    logger.info(f"    {key}: {value:.3f}")

        # Classify root cause
        fault_classified = classify_fault_root_cause(fault)

        if fault_classified.root_cause:
            logger.info(f"  Root cause: {fault_classified.root_cause.value} (confidence: {fault_classified.root_cause_confidence:.2f})")
            logger.info(f"  Action: {fault_classified.recommended_action}")


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run all FDC examples"""
    logger.info("\n" + "=" * 70)
    logger.info("FDC Module - Examples")
    logger.info("=" * 70)

    example_thickness_drift()
    example_stress_shift()
    example_adhesion_drop()
    example_pattern_detection()
    example_increased_variation()

    logger.info("\n" + "=" * 70)
    logger.info("All FDC examples completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
