"""
Western Electric & Nelson Control Chart Rules - Session 7

Implements standard SPC rules for detecting out-of-control conditions:
- Western Electric Rules (Rules 1-4)
- Nelson Rules (Rules 1-8)

These rules are used to detect process shifts, trends, and instabilities
in control charts for semiconductor manufacturing.

Status: PRODUCTION READY ✅
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime


class RuleViolationType(str, Enum):
    """Types of control chart rule violations."""
    RULE_1 = "RULE_1"  # One point > 3σ
    RULE_2 = "RULE_2"  # 2/3 points > 2σ (same side)
    RULE_3 = "RULE_3"  # 4/5 points > 1σ (same side)
    RULE_4 = "RULE_4"  # 8 consecutive points (same side)
    RULE_5 = "RULE_5"  # 6 points trend (monotonic)
    RULE_6 = "RULE_6"  # 15 points within 1σ
    RULE_7 = "RULE_7"  # 14 points alternating
    RULE_8 = "RULE_8"  # 8 points beyond 1σ (both sides)


class Severity(str, Enum):
    """Severity level of rule violation."""
    CRITICAL = "CRITICAL"  # Immediate action required
    WARNING = "WARNING"    # Investigate soon
    INFO = "INFO"          # Monitor closely


@dataclass
class RuleViolation:
    """
    Represents a detected control chart rule violation.

    Attributes:
        rule: Type of rule violated
        index: Index in data array where violation detected
        timestamp: Timestamp of violation (if provided)
        severity: Severity level
        description: Human-readable description
        affected_indices: List of indices involved in violation
        metric_value: Actual metric value at violation point
    """
    rule: RuleViolationType
    index: int
    timestamp: Optional[datetime]
    severity: Severity
    description: str
    affected_indices: List[int]
    metric_value: float


@dataclass
class ControlLimits:
    """
    Control chart limits.

    Attributes:
        centerline: Process mean (CL)
        ucl: Upper control limit (+3σ)
        lcl: Lower control limit (-3σ)
        sigma: Process standard deviation
    """
    centerline: float
    ucl: float
    lcl: float
    sigma: float

    @classmethod
    def from_data(cls, data: np.ndarray, use_moving_range: bool = False) -> 'ControlLimits':
        """
        Calculate control limits from data.

        Args:
            data: Process measurements
            use_moving_range: If True, estimate sigma from moving range

        Returns:
            ControlLimits object
        """
        centerline = np.mean(data)

        if use_moving_range:
            # Estimate sigma from moving range (for individual charts)
            mr = np.abs(np.diff(data))
            mr_bar = np.mean(mr)
            sigma = mr_bar / 1.128  # d2 constant for n=2
        else:
            sigma = np.std(data, ddof=1)

        ucl = centerline + 3 * sigma
        lcl = centerline - 3 * sigma

        return cls(centerline=centerline, ucl=ucl, lcl=lcl, sigma=sigma)


class SPCRulesEngine:
    """
    Western Electric & Nelson Rules implementation.

    Detects out-of-control conditions using standard SPC rules.
    """

    def __init__(
        self,
        limits: ControlLimits,
        enabled_rules: Optional[List[RuleViolationType]] = None,
        timestamps: Optional[List[datetime]] = None
    ):
        """
        Initialize SPC rules engine.

        Args:
            limits: Control limits for the process
            enabled_rules: List of rules to check (None = all rules)
            timestamps: Optional timestamps for each data point
        """
        self.limits = limits
        self.enabled_rules = enabled_rules or list(RuleViolationType)
        self.timestamps = timestamps

    def check_all_rules(self, data: np.ndarray) -> List[RuleViolation]:
        """
        Check all enabled rules against data.

        Args:
            data: Process measurements

        Returns:
            List of detected violations
        """
        violations = []

        # Standardize data
        z_scores = (data - self.limits.centerline) / self.limits.sigma

        if RuleViolationType.RULE_1 in self.enabled_rules:
            violations.extend(self._check_rule_1(data, z_scores))

        if RuleViolationType.RULE_2 in self.enabled_rules:
            violations.extend(self._check_rule_2(data, z_scores))

        if RuleViolationType.RULE_3 in self.enabled_rules:
            violations.extend(self._check_rule_3(data, z_scores))

        if RuleViolationType.RULE_4 in self.enabled_rules:
            violations.extend(self._check_rule_4(data, z_scores))

        if RuleViolationType.RULE_5 in self.enabled_rules:
            violations.extend(self._check_rule_5(data, z_scores))

        if RuleViolationType.RULE_6 in self.enabled_rules:
            violations.extend(self._check_rule_6(data, z_scores))

        if RuleViolationType.RULE_7 in self.enabled_rules:
            violations.extend(self._check_rule_7(data, z_scores))

        if RuleViolationType.RULE_8 in self.enabled_rules:
            violations.extend(self._check_rule_8(data, z_scores))

        return sorted(violations, key=lambda v: v.index)

    def _get_timestamp(self, index: int) -> Optional[datetime]:
        """Get timestamp for given index."""
        if self.timestamps and index < len(self.timestamps):
            return self.timestamps[index]
        return None

    def _check_rule_1(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 1: One point beyond 3σ from centerline.
        (Western Electric Rule 1 / Nelson Rule 1)
        """
        violations = []

        for i, z in enumerate(z_scores):
            if np.abs(z) > 3:
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_1,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.CRITICAL,
                    description=f"Point beyond 3σ (z={z:.2f})",
                    affected_indices=[i],
                    metric_value=data[i]
                ))

        return violations

    def _check_rule_2(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 2: Two out of three consecutive points beyond 2σ on same side.
        (Western Electric Rule 2 / Nelson Rule 2)
        """
        violations = []

        for i in range(2, len(z_scores)):
            window = z_scores[i-2:i+1]

            # Check positive side
            beyond_2sigma_pos = np.sum(window > 2)
            if beyond_2sigma_pos >= 2:
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_2,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"2/3 points beyond +2σ",
                    affected_indices=list(range(i-2, i+1)),
                    metric_value=data[i]
                ))

            # Check negative side
            beyond_2sigma_neg = np.sum(window < -2)
            if beyond_2sigma_neg >= 2:
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_2,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"2/3 points beyond -2σ",
                    affected_indices=list(range(i-2, i+1)),
                    metric_value=data[i]
                ))

        return violations

    def _check_rule_3(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 3: Four out of five consecutive points beyond 1σ on same side.
        (Western Electric Rule 3 / Nelson Rule 3)
        """
        violations = []

        for i in range(4, len(z_scores)):
            window = z_scores[i-4:i+1]

            # Check positive side
            beyond_1sigma_pos = np.sum(window > 1)
            if beyond_1sigma_pos >= 4:
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_3,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"4/5 points beyond +1σ",
                    affected_indices=list(range(i-4, i+1)),
                    metric_value=data[i]
                ))

            # Check negative side
            beyond_1sigma_neg = np.sum(window < -1)
            if beyond_1sigma_neg >= 4:
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_3,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"4/5 points beyond -1σ",
                    affected_indices=list(range(i-4, i+1)),
                    metric_value=data[i]
                ))

        return violations

    def _check_rule_4(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 4: Eight consecutive points on same side of centerline.
        (Western Electric Rule 4 / Nelson Rule 4)
        """
        violations = []

        for i in range(7, len(z_scores)):
            window = z_scores[i-7:i+1]

            # Check if all positive
            if np.all(window > 0):
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_4,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"8 consecutive points above centerline",
                    affected_indices=list(range(i-7, i+1)),
                    metric_value=data[i]
                ))

            # Check if all negative
            if np.all(window < 0):
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_4,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"8 consecutive points below centerline",
                    affected_indices=list(range(i-7, i+1)),
                    metric_value=data[i]
                ))

        return violations

    def _check_rule_5(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 5: Six points in a row steadily increasing or decreasing.
        (Nelson Rule 5)
        """
        violations = []

        for i in range(5, len(data)):
            window = data[i-5:i+1]

            # Check increasing trend
            if np.all(np.diff(window) > 0):
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_5,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"6 points steadily increasing (trend)",
                    affected_indices=list(range(i-5, i+1)),
                    metric_value=data[i]
                ))

            # Check decreasing trend
            if np.all(np.diff(window) < 0):
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_5,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"6 points steadily decreasing (trend)",
                    affected_indices=list(range(i-5, i+1)),
                    metric_value=data[i]
                ))

        return violations

    def _check_rule_6(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 6: Fifteen points in a row within 1σ of centerline (both sides).
        (Nelson Rule 6 - indicates reduced variation)
        """
        violations = []

        for i in range(14, len(z_scores)):
            window = z_scores[i-14:i+1]

            if np.all(np.abs(window) < 1):
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_6,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.INFO,
                    description=f"15 points within 1σ (reduced variation)",
                    affected_indices=list(range(i-14, i+1)),
                    metric_value=data[i]
                ))

        return violations

    def _check_rule_7(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 7: Fourteen points in a row alternating up and down.
        (Nelson Rule 7 - indicates systematic variation)
        """
        violations = []

        for i in range(13, len(data)):
            window = data[i-13:i+1]
            diffs = np.diff(window)

            # Check if signs alternate
            sign_changes = np.diff(np.sign(diffs))
            if np.all(sign_changes != 0):
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_7,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.INFO,
                    description=f"14 points alternating up/down (systematic)",
                    affected_indices=list(range(i-13, i+1)),
                    metric_value=data[i]
                ))

        return violations

    def _check_rule_8(self, data: np.ndarray, z_scores: np.ndarray) -> List[RuleViolation]:
        """
        Rule 8: Eight points in a row with none within 1σ of centerline.
        (Nelson Rule 8 - indicates mixture or stratification)
        """
        violations = []

        for i in range(7, len(z_scores)):
            window = z_scores[i-7:i+1]

            if np.all(np.abs(window) > 1):
                violations.append(RuleViolation(
                    rule=RuleViolationType.RULE_8,
                    index=i,
                    timestamp=self._get_timestamp(i),
                    severity=Severity.WARNING,
                    description=f"8 points beyond 1σ (mixture/stratification)",
                    affected_indices=list(range(i-7, i+1)),
                    metric_value=data[i]
                ))

        return violations


def quick_rule_check(
    data: np.ndarray,
    centerline: Optional[float] = None,
    sigma: Optional[float] = None,
    timestamps: Optional[List[datetime]] = None,
    rules: Optional[List[RuleViolationType]] = None
) -> List[RuleViolation]:
    """
    Quick helper to check SPC rules on data.

    Args:
        data: Process measurements
        centerline: Process mean (if None, calculated from data)
        sigma: Process std dev (if None, calculated from data)
        timestamps: Optional timestamps for violations
        rules: Specific rules to check (if None, all rules checked)

    Returns:
        List of detected violations

    Example:
        >>> violations = quick_rule_check(junction_depths)
        >>> for v in violations:
        ...     print(f"{v.rule}: {v.description}")
    """
    if centerline is None or sigma is None:
        limits = ControlLimits.from_data(data)
    else:
        limits = ControlLimits(
            centerline=centerline,
            ucl=centerline + 3*sigma,
            lcl=centerline - 3*sigma,
            sigma=sigma
        )

    engine = SPCRulesEngine(limits, enabled_rules=rules, timestamps=timestamps)
    return engine.check_all_rules(data)
