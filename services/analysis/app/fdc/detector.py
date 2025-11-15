"""
Fault Detection

Detects faults from SPC violations and time-series patterns.
"""

import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..spc.series import SPCSeries, SPCMetric
from ..spc.charts import SPCChartResult
from ..spc.rules import SPCViolation, SPCViolationType


class FaultType(str, Enum):
    """Types of faults in CVD processes"""
    # Drift faults
    GRADUAL_DRIFT_UPWARD = "gradual_drift_upward"
    GRADUAL_DRIFT_DOWNWARD = "gradual_drift_downward"

    # Shift faults
    SUDDEN_SHIFT_UPWARD = "sudden_shift_upward"
    SUDDEN_SHIFT_DOWNWARD = "sudden_shift_downward"

    # Variation faults
    INCREASED_VARIATION = "increased_variation"
    DECREASED_VARIATION = "decreased_variation"

    # Pattern faults
    CYCLIC_PATTERN = "cyclic_pattern"
    ALTERNATING_PATTERN = "alternating_pattern"

    # Out-of-control
    OUT_OF_SPEC_HIGH = "out_of_spec_high"
    OUT_OF_SPEC_LOW = "out_of_spec_low"

    # Equipment-specific
    TOOL_DEGRADATION = "tool_degradation"
    CALIBRATION_DRIFT = "calibration_drift"


class FaultSeverity(str, Enum):
    """Severity levels for faults"""
    CRITICAL = "CRITICAL"    # Immediate action required
    WARNING = "WARNING"      # Action required soon
    INFO = "INFO"           # Informational


class RootCause(str, Enum):
    """Potential root causes"""
    # Thermal
    HEATER_DEGRADATION = "heater_degradation"
    TEMPERATURE_CONTROLLER_DRIFT = "temperature_controller_drift"
    COOLING_SYSTEM_ISSUE = "cooling_system_issue"

    # Gas delivery
    MFC_CALIBRATION_DRIFT = "mfc_calibration_drift"
    GAS_LINE_CONTAMINATION = "gas_line_contamination"
    PRECURSOR_DEPLETION = "precursor_depletion"

    # Vacuum
    PUMP_DEGRADATION = "pump_degradation"
    LEAK_DETECTED = "leak_detected"
    PRESSURE_CONTROLLER_DRIFT = "pressure_controller_drift"

    # Plasma
    RF_GENERATOR_DRIFT = "rf_generator_drift"
    MATCHING_NETWORK_ISSUE = "matching_network_issue"

    # Contamination
    CHAMBER_CONTAMINATION = "chamber_contamination"
    PRECLEAN_FAILURE = "preclean_failure"
    PARTICLE_GENERATION = "particle_generation"

    # Recipe/Software
    RECIPE_ERROR = "recipe_error"
    VM_MODEL_MISTUNE = "vm_model_mistune"
    PCM_CONTROLLER_ISSUE = "pcm_controller_issue"

    # Unknown
    UNKNOWN = "unknown"


@dataclass
class Fault:
    """Detected fault"""
    fault_type: FaultType
    severity: FaultSeverity
    metric: SPCMetric
    description: str

    # When the fault occurred
    detection_timestamp: datetime
    first_occurrence_index: Optional[int] = None
    last_occurrence_index: Optional[int] = None

    # Context
    tool_id: Optional[str] = None
    recipe_id: Optional[str] = None
    film_material: Optional[str] = None

    # Root cause (if classified)
    root_cause: Optional[RootCause] = None
    root_cause_confidence: float = 0.0  # 0-1

    # Supporting evidence
    spc_violations: List[SPCViolation] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)

    # Recommended action
    recommended_action: Optional[str] = None


class FDCDetector:
    """
    Fault Detection and Classification Detector

    Analyzes SPC chart results and time-series patterns to detect faults.
    """

    def __init__(
        self,
        series: SPCSeries,
        spc_result: Optional[SPCChartResult] = None,
    ):
        """
        Args:
            series: SPC series data
            spc_result: Optional SPC chart result
        """
        self.series = series
        self.spc_result = spc_result
        self.faults: List[Fault] = []

    def detect_all_faults(
        self,
        spc_violations: Optional[List[SPCViolation]] = None,
    ) -> List[Fault]:
        """
        Detect all faults from SPC violations and time-series analysis

        Args:
            spc_violations: Optional list of SPC violations

        Returns:
            List of detected faults
        """
        self.faults = []

        # 1. Detect faults from SPC violations
        if spc_violations:
            self.faults.extend(self._detect_from_spc_violations(spc_violations))

        # 2. Detect drift faults
        self.faults.extend(self._detect_drift())

        # 3. Detect shift faults
        self.faults.extend(self._detect_shifts())

        # 4. Detect variation changes
        self.faults.extend(self._detect_variation_changes())

        # 5. Detect out-of-spec conditions
        self.faults.extend(self._detect_out_of_spec())

        # Sort by severity and timestamp
        self.faults.sort(key=lambda f: (f.severity.value, f.detection_timestamp))

        return self.faults

    def _detect_from_spc_violations(
        self,
        violations: List[SPCViolation],
    ) -> List[Fault]:
        """Convert SPC violations to faults"""
        faults = []

        # Group violations by type
        violation_groups: Dict[SPCViolationType, List[SPCViolation]] = {}
        for v in violations:
            if v.violation_type not in violation_groups:
                violation_groups[v.violation_type] = []
            violation_groups[v.violation_type].append(v)

        # Rule 1: Beyond 3σ → Critical out-of-spec
        if SPCViolationType.RULE_1_BEYOND_3SIGMA in violation_groups:
            for v in violation_groups[SPCViolationType.RULE_1_BEYOND_3SIGMA]:
                value = self.series.get_values()[v.point_index]
                cl = self.spc_result.control_limits.center_line if self.spc_result else self.series.process_mean

                if value > cl:
                    fault_type = FaultType.OUT_OF_SPEC_HIGH
                else:
                    fault_type = FaultType.OUT_OF_SPEC_LOW

                faults.append(Fault(
                    fault_type=fault_type,
                    severity=FaultSeverity.CRITICAL,
                    metric=self.series.metric,
                    description=f"{self.series.metric.value} out of control limits",
                    detection_timestamp=datetime.now(),
                    first_occurrence_index=v.point_index,
                    last_occurrence_index=v.point_index,
                    tool_id=self.series.tool_id,
                    recipe_id=self.series.recipe_id,
                    film_material=self.series.film_material,
                    spc_violations=[v],
                ))

        # Rule 4 & 5: Trends → Gradual drift
        trend_rules = [
            SPCViolationType.RULE_4_EIGHT_ONE_SIDE,
            SPCViolationType.RULE_5_SIX_TREND,
        ]

        for rule in trend_rules:
            if rule in violation_groups:
                for v in violation_groups[rule]:
                    # Check if upward or downward
                    if v.related_indices and len(v.related_indices) >= 2:
                        start_val = self.series.get_values()[v.related_indices[0]]
                        end_val = self.series.get_values()[v.related_indices[-1]]

                        if end_val > start_val:
                            fault_type = FaultType.GRADUAL_DRIFT_UPWARD
                        else:
                            fault_type = FaultType.GRADUAL_DRIFT_DOWNWARD

                        faults.append(Fault(
                            fault_type=fault_type,
                            severity=FaultSeverity.WARNING,
                            metric=self.series.metric,
                            description=f"{self.series.metric.value} showing gradual drift",
                            detection_timestamp=datetime.now(),
                            first_occurrence_index=v.related_indices[0] if v.related_indices else v.point_index,
                            last_occurrence_index=v.point_index,
                            tool_id=self.series.tool_id,
                            recipe_id=self.series.recipe_id,
                            film_material=self.series.film_material,
                            spc_violations=[v],
                        ))

        # Rule 7: Alternating → Pattern fault
        if SPCViolationType.RULE_7_FOURTEEN_ALTERNATING in violation_groups:
            for v in violation_groups[SPCViolationType.RULE_7_FOURTEEN_ALTERNATING]:
                faults.append(Fault(
                    fault_type=FaultType.ALTERNATING_PATTERN,
                    severity=FaultSeverity.WARNING,
                    metric=self.series.metric,
                    description=f"{self.series.metric.value} showing alternating pattern",
                    detection_timestamp=datetime.now(),
                    first_occurrence_index=v.related_indices[0] if v.related_indices else v.point_index,
                    last_occurrence_index=v.point_index,
                    tool_id=self.series.tool_id,
                    recipe_id=self.series.recipe_id,
                    film_material=self.series.film_material,
                    spc_violations=[v],
                ))

        return faults

    def _detect_drift(self) -> List[Fault]:
        """
        Detect gradual drift using linear regression

        Drift is significant if slope > threshold
        """
        faults = []

        values = self.series.get_values()
        if len(values) < 10:
            return faults

        # Perform linear regression
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, deg=1)

        # Calculate R² to assess linearity
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Drift threshold (relative to process std)
        process_std = self.series.process_std if self.series.process_std else np.std(values)
        drift_threshold = 0.05 * process_std  # 0.05σ per point

        # Check for significant drift
        if abs(slope) > drift_threshold and r_squared > 0.5:
            if slope > 0:
                fault_type = FaultType.GRADUAL_DRIFT_UPWARD
                description = f"{self.series.metric.value} drifting upward (slope={slope:.3f}/run, R²={r_squared:.2f})"
            else:
                fault_type = FaultType.GRADUAL_DRIFT_DOWNWARD
                description = f"{self.series.metric.value} drifting downward (slope={slope:.3f}/run, R²={r_squared:.2f})"

            faults.append(Fault(
                fault_type=fault_type,
                severity=FaultSeverity.WARNING,
                metric=self.series.metric,
                description=description,
                detection_timestamp=datetime.now(),
                first_occurrence_index=0,
                last_occurrence_index=len(values) - 1,
                tool_id=self.series.tool_id,
                recipe_id=self.series.recipe_id,
                film_material=self.series.film_material,
                statistics={
                    "slope": slope,
                    "r_squared": r_squared,
                    "drift_rate_per_run": slope,
                },
            ))

        return faults

    def _detect_shifts(self) -> List[Fault]:
        """
        Detect sudden shifts using change-point detection

        Uses simple method: compare mean before and after each point
        """
        faults = []

        values = self.series.get_values()
        if len(values) < 20:
            return faults

        process_std = self.series.process_std if self.series.process_std else np.std(values)
        shift_threshold = 2.0 * process_std  # 2σ shift

        # Scan for change points
        min_segment_size = 10

        for i in range(min_segment_size, len(values) - min_segment_size):
            mean_before = np.mean(values[max(0, i-min_segment_size):i])
            mean_after = np.mean(values[i:min(len(values), i+min_segment_size)])

            shift_magnitude = mean_after - mean_before

            if abs(shift_magnitude) > shift_threshold:
                if shift_magnitude > 0:
                    fault_type = FaultType.SUDDEN_SHIFT_UPWARD
                    description = f"{self.series.metric.value} sudden increase at run {i} (Δ={shift_magnitude:.2f})"
                else:
                    fault_type = FaultType.SUDDEN_SHIFT_DOWNWARD
                    description = f"{self.series.metric.value} sudden decrease at run {i} (Δ={shift_magnitude:.2f})"

                faults.append(Fault(
                    fault_type=fault_type,
                    severity=FaultSeverity.WARNING,
                    metric=self.series.metric,
                    description=description,
                    detection_timestamp=datetime.now(),
                    first_occurrence_index=i,
                    last_occurrence_index=i,
                    tool_id=self.series.tool_id,
                    recipe_id=self.series.recipe_id,
                    film_material=self.series.film_material,
                    statistics={
                        "shift_magnitude": shift_magnitude,
                        "mean_before": mean_before,
                        "mean_after": mean_after,
                    },
                ))

                # Only report first major shift
                break

        return faults

    def _detect_variation_changes(self) -> List[Fault]:
        """
        Detect changes in process variation

        Compares std dev in recent window vs baseline
        """
        faults = []

        values = self.series.get_values()
        if len(values) < 20:
            return faults

        # Split into baseline (first 60%) and recent (last 40%)
        split_idx = int(len(values) * 0.6)
        baseline_values = values[:split_idx]
        recent_values = values[split_idx:]

        baseline_std = np.std(baseline_values, ddof=1)
        recent_std = np.std(recent_values, ddof=1)

        # Variance ratio (F-test concept)
        variance_ratio = (recent_std / baseline_std) ** 2 if baseline_std > 0 else 1.0

        # Threshold for significant change (F-distribution approximation)
        # For simplicity, use 2x or 0.5x as thresholds
        if variance_ratio > 2.0:
            faults.append(Fault(
                fault_type=FaultType.INCREASED_VARIATION,
                severity=FaultSeverity.WARNING,
                metric=self.series.metric,
                description=f"{self.series.metric.value} variation increased ({variance_ratio:.1f}x)",
                detection_timestamp=datetime.now(),
                first_occurrence_index=split_idx,
                last_occurrence_index=len(values) - 1,
                tool_id=self.series.tool_id,
                recipe_id=self.series.recipe_id,
                film_material=self.series.film_material,
                statistics={
                    "baseline_std": baseline_std,
                    "recent_std": recent_std,
                    "variance_ratio": variance_ratio,
                },
            ))

        elif variance_ratio < 0.5:
            faults.append(Fault(
                fault_type=FaultType.DECREASED_VARIATION,
                severity=FaultSeverity.INFO,
                metric=self.series.metric,
                description=f"{self.series.metric.value} variation decreased ({variance_ratio:.1f}x)",
                detection_timestamp=datetime.now(),
                first_occurrence_index=split_idx,
                last_occurrence_index=len(values) - 1,
                tool_id=self.series.tool_id,
                recipe_id=self.series.recipe_id,
                film_material=self.series.film_material,
                statistics={
                    "baseline_std": baseline_std,
                    "recent_std": recent_std,
                    "variance_ratio": variance_ratio,
                },
            ))

        return faults

    def _detect_out_of_spec(self) -> List[Fault]:
        """
        Detect out-of-spec conditions based on metric-specific limits

        These are different from SPC control limits - these are spec limits.
        """
        faults = []

        # Define spec limits for each metric
        spec_limits = self._get_spec_limits()

        if not spec_limits:
            return faults

        usl = spec_limits.get("upper_spec_limit")
        lsl = spec_limits.get("lower_spec_limit")

        values = self.series.get_values()

        for i, value in enumerate(values):
            if usl is not None and value > usl:
                faults.append(Fault(
                    fault_type=FaultType.OUT_OF_SPEC_HIGH,
                    severity=FaultSeverity.CRITICAL,
                    metric=self.series.metric,
                    description=f"{self.series.metric.value} exceeds upper spec limit (USL={usl})",
                    detection_timestamp=datetime.now(),
                    first_occurrence_index=i,
                    last_occurrence_index=i,
                    tool_id=self.series.tool_id,
                    recipe_id=self.series.recipe_id,
                    film_material=self.series.film_material,
                    statistics={"value": value, "spec_limit": usl},
                ))

            if lsl is not None and value < lsl:
                faults.append(Fault(
                    fault_type=FaultType.OUT_OF_SPEC_LOW,
                    severity=FaultSeverity.CRITICAL,
                    metric=self.series.metric,
                    description=f"{self.series.metric.value} below lower spec limit (LSL={lsl})",
                    detection_timestamp=datetime.now(),
                    first_occurrence_index=i,
                    last_occurrence_index=i,
                    tool_id=self.series.tool_id,
                    recipe_id=self.series.recipe_id,
                    film_material=self.series.film_material,
                    statistics={"value": value, "spec_limit": lsl},
                ))

        return faults

    def _get_spec_limits(self) -> Dict[str, float]:
        """
        Get specification limits for the metric

        These would normally come from a database or config file.
        For now, using typical values.
        """
        spec_limits = {}

        if self.series.metric == SPCMetric.THICKNESS_MEAN:
            # Example: 100nm ± 10nm
            spec_limits = {
                "upper_spec_limit": 110.0,
                "lower_spec_limit": 90.0,
            }

        elif self.series.metric == SPCMetric.THICKNESS_UNIFORMITY:
            # Uniformity should be < 5%
            spec_limits = {
                "upper_spec_limit": 5.0,
                "lower_spec_limit": None,
            }

        elif self.series.metric == SPCMetric.STRESS_MEAN:
            # Example: -250 MPa ± 100 MPa
            spec_limits = {
                "upper_spec_limit": -150.0,
                "lower_spec_limit": -350.0,
            }

        elif self.series.metric == SPCMetric.ADHESION_SCORE:
            # Adhesion should be > 70
            spec_limits = {
                "upper_spec_limit": None,
                "lower_spec_limit": 70.0,
            }

        return spec_limits


def detect_faults(
    series: SPCSeries,
    spc_result: Optional[SPCChartResult] = None,
    spc_violations: Optional[List[SPCViolation]] = None,
) -> List[Fault]:
    """
    Convenience function to detect all faults

    Args:
        series: SPC series data
        spc_result: Optional SPC chart result
        spc_violations: Optional SPC violations

    Returns:
        List of detected faults
    """
    detector = FDCDetector(series, spc_result)
    return detector.detect_all_faults(spc_violations)
