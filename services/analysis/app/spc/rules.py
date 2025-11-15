"""
Western Electric Rules for SPC

Implements the Western Electric rules for detecting out-of-control conditions.

The 8 classic Western Electric rules:
1. One point beyond Zone A (>3σ from center)
2. Two out of three consecutive points in Zone A or beyond
3. Four out of five consecutive points in Zone B or beyond
4. Eight consecutive points in Zone C or beyond (one side of center)
5. Six points in a row steadily increasing or decreasing
6. Fifteen points in a row in Zone C (both sides of center)
7. Fourteen points in a row alternating up and down
8. Eight points in a row on both sides of center with none in Zone C
"""

import numpy as np
from typing import List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .charts import ControlLimits


class SPCViolationType(str, Enum):
    """Types of SPC rule violations"""
    # Western Electric Rules
    RULE_1_BEYOND_3SIGMA = "rule_1_beyond_3sigma"
    RULE_2_TWO_OF_THREE_BEYOND_2SIGMA = "rule_2_two_of_three_beyond_2sigma"
    RULE_3_FOUR_OF_FIVE_BEYOND_1SIGMA = "rule_3_four_of_five_beyond_1sigma"
    RULE_4_EIGHT_ONE_SIDE = "rule_4_eight_one_side"
    RULE_5_SIX_TREND = "rule_5_six_trend"
    RULE_6_FIFTEEN_WITHIN_1SIGMA = "rule_6_fifteen_within_1sigma"
    RULE_7_FOURTEEN_ALTERNATING = "rule_7_fourteen_alternating"
    RULE_8_EIGHT_AVOID_CENTER = "rule_8_eight_avoid_center"

    # Additional patterns
    GRADUAL_DRIFT = "gradual_drift"
    SUDDEN_SHIFT = "sudden_shift"
    INCREASED_VARIATION = "increased_variation"


@dataclass
class SPCViolation:
    """SPC rule violation"""
    violation_type: SPCViolationType
    point_index: int  # Index of the violating point
    description: str
    severity: str  # "CRITICAL", "WARNING", "INFO"
    related_indices: Optional[List[int]] = None  # Other points involved in pattern


class WesternElectricRules:
    """
    Western Electric Rules implementation

    Detects out-of-control patterns in SPC data.
    """

    def __init__(
        self,
        values: np.ndarray,
        control_limits: ControlLimits,
    ):
        """
        Args:
            values: Process values (chart values, e.g., from X-bar or EWMA)
            control_limits: Control limits for the chart
        """
        self.values = values
        self.control_limits = control_limits
        self.n = len(values)

        # Calculate zones
        self._calculate_zones()

    def _calculate_zones(self):
        """
        Calculate SPC zones

        Zone C: μ ± 1σ (center ± 1/3 of control limit range)
        Zone B: μ ± 2σ (center ± 2/3 of control limit range)
        Zone A: μ ± 3σ (beyond 2/3, up to control limits)
        Beyond: Outside control limits
        """
        cl = self.control_limits.center_line
        ucl = self.control_limits.upper_control_limit
        lcl = self.control_limits.lower_control_limit

        # Calculate zone boundaries
        sigma = (ucl - cl) / 3.0  # 1σ

        # Upper zones
        self.zone_c_upper = cl + sigma
        self.zone_b_upper = cl + 2 * sigma
        self.zone_a_upper = cl + 3 * sigma

        # Lower zones
        self.zone_c_lower = cl - sigma
        self.zone_b_lower = cl - 2 * sigma
        self.zone_a_lower = cl - 3 * sigma

    def _get_zone(self, value: float) -> int:
        """
        Get zone number for a value

        Returns:
            +3: Beyond UCL
            +2: Zone A upper
            +1: Zone B upper
            0: Zone C
            -1: Zone B lower
            -2: Zone A lower
            -3: Beyond LCL
        """
        cl = self.control_limits.center_line

        if value > self.zone_a_upper:
            return +3  # Beyond UCL
        elif value > self.zone_b_upper:
            return +2  # Zone A upper
        elif value > self.zone_c_upper:
            return +1  # Zone B upper
        elif value >= self.zone_c_lower:
            return 0   # Zone C
        elif value >= self.zone_b_lower:
            return -1  # Zone B lower
        elif value >= self.zone_a_lower:
            return -2  # Zone A lower
        else:
            return -3  # Beyond LCL

    def check_rule_1(self) -> List[SPCViolation]:
        """
        Rule 1: One point beyond Zone A (>3σ from center)

        This is a critical violation.
        """
        violations = []

        for i, value in enumerate(self.values):
            zone = self._get_zone(value)

            if abs(zone) >= 3:
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_1_BEYOND_3SIGMA,
                    point_index=i,
                    description=f"Point beyond 3σ control limit (Zone {'A+' if zone > 0 else 'A-'})",
                    severity="CRITICAL",
                ))

        return violations

    def check_rule_2(self) -> List[SPCViolation]:
        """
        Rule 2: Two out of three consecutive points in Zone A or beyond (same side)
        """
        violations = []

        if self.n < 3:
            return violations

        for i in range(2, self.n):
            window = [self._get_zone(self.values[j]) for j in range(i-2, i+1)]

            # Check upper side
            count_upper_a = sum(1 for z in window if z >= 2)
            if count_upper_a >= 2:
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_2_TWO_OF_THREE_BEYOND_2SIGMA,
                    point_index=i,
                    description="2 of 3 points beyond 2σ (upper)",
                    severity="WARNING",
                    related_indices=list(range(i-2, i+1)),
                ))

            # Check lower side
            count_lower_a = sum(1 for z in window if z <= -2)
            if count_lower_a >= 2:
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_2_TWO_OF_THREE_BEYOND_2SIGMA,
                    point_index=i,
                    description="2 of 3 points beyond 2σ (lower)",
                    severity="WARNING",
                    related_indices=list(range(i-2, i+1)),
                ))

        return violations

    def check_rule_3(self) -> List[SPCViolation]:
        """
        Rule 3: Four out of five consecutive points in Zone B or beyond (same side)
        """
        violations = []

        if self.n < 5:
            return violations

        for i in range(4, self.n):
            window = [self._get_zone(self.values[j]) for j in range(i-4, i+1)]

            # Check upper side (Zone B or beyond: zone >= 1)
            count_upper_b = sum(1 for z in window if z >= 1)
            if count_upper_b >= 4:
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_3_FOUR_OF_FIVE_BEYOND_1SIGMA,
                    point_index=i,
                    description="4 of 5 points beyond 1σ (upper)",
                    severity="WARNING",
                    related_indices=list(range(i-4, i+1)),
                ))

            # Check lower side
            count_lower_b = sum(1 for z in window if z <= -1)
            if count_lower_b >= 4:
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_3_FOUR_OF_FIVE_BEYOND_1SIGMA,
                    point_index=i,
                    description="4 of 5 points beyond 1σ (lower)",
                    severity="WARNING",
                    related_indices=list(range(i-4, i+1)),
                ))

        return violations

    def check_rule_4(self) -> List[SPCViolation]:
        """
        Rule 4: Eight consecutive points on one side of center line
        """
        violations = []

        if self.n < 8:
            return violations

        cl = self.control_limits.center_line

        for i in range(7, self.n):
            window = self.values[i-7:i+1]

            # All above center
            if all(v > cl for v in window):
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_4_EIGHT_ONE_SIDE,
                    point_index=i,
                    description="8 consecutive points above center",
                    severity="WARNING",
                    related_indices=list(range(i-7, i+1)),
                ))

            # All below center
            if all(v < cl for v in window):
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_4_EIGHT_ONE_SIDE,
                    point_index=i,
                    description="8 consecutive points below center",
                    severity="WARNING",
                    related_indices=list(range(i-7, i+1)),
                ))

        return violations

    def check_rule_5(self) -> List[SPCViolation]:
        """
        Rule 5: Six points in a row steadily increasing or decreasing
        """
        violations = []

        if self.n < 6:
            return violations

        for i in range(5, self.n):
            window = self.values[i-5:i+1]

            # Check if strictly increasing
            if all(window[j] < window[j+1] for j in range(5)):
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_5_SIX_TREND,
                    point_index=i,
                    description="6 points steadily increasing",
                    severity="WARNING",
                    related_indices=list(range(i-5, i+1)),
                ))

            # Check if strictly decreasing
            if all(window[j] > window[j+1] for j in range(5)):
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_5_SIX_TREND,
                    point_index=i,
                    description="6 points steadily decreasing",
                    severity="WARNING",
                    related_indices=list(range(i-5, i+1)),
                ))

        return violations

    def check_rule_6(self) -> List[SPCViolation]:
        """
        Rule 6: Fifteen points in a row in Zone C (within 1σ of center)
        """
        violations = []

        if self.n < 15:
            return violations

        for i in range(14, self.n):
            window = [self._get_zone(self.values[j]) for j in range(i-14, i+1)]

            # All in Zone C
            if all(z == 0 for z in window):
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_6_FIFTEEN_WITHIN_1SIGMA,
                    point_index=i,
                    description="15 points in Zone C (low variation)",
                    severity="INFO",
                    related_indices=list(range(i-14, i+1)),
                ))

        return violations

    def check_rule_7(self) -> List[SPCViolation]:
        """
        Rule 7: Fourteen points in a row alternating up and down
        """
        violations = []

        if self.n < 14:
            return violations

        for i in range(13, self.n):
            window = self.values[i-13:i+1]

            # Check alternating pattern
            is_alternating = True
            for j in range(13):
                if j % 2 == 0:
                    # Even: should go up
                    if window[j+1] <= window[j]:
                        is_alternating = False
                        break
                else:
                    # Odd: should go down
                    if window[j+1] >= window[j]:
                        is_alternating = False
                        break

            if is_alternating:
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_7_FOURTEEN_ALTERNATING,
                    point_index=i,
                    description="14 points alternating up/down",
                    severity="WARNING",
                    related_indices=list(range(i-13, i+1)),
                ))

        return violations

    def check_rule_8(self) -> List[SPCViolation]:
        """
        Rule 8: Eight points on both sides of center with none in Zone C
        """
        violations = []

        if self.n < 8:
            return violations

        cl = self.control_limits.center_line

        for i in range(7, self.n):
            window_values = self.values[i-7:i+1]
            window_zones = [self._get_zone(v) for v in window_values]

            # Check:
            # 1. None in Zone C
            # 2. Points on both sides of center
            none_in_c = all(z != 0 for z in window_zones)
            has_above = any(v > cl for v in window_values)
            has_below = any(v < cl for v in window_values)

            if none_in_c and has_above and has_below:
                violations.append(SPCViolation(
                    violation_type=SPCViolationType.RULE_8_EIGHT_AVOID_CENTER,
                    point_index=i,
                    description="8 points avoiding Zone C",
                    severity="WARNING",
                    related_indices=list(range(i-7, i+1)),
                ))

        return violations

    def check_all_rules(self) -> List[SPCViolation]:
        """
        Check all Western Electric rules

        Returns:
            List of all violations found
        """
        violations = []

        violations.extend(self.check_rule_1())
        violations.extend(self.check_rule_2())
        violations.extend(self.check_rule_3())
        violations.extend(self.check_rule_4())
        violations.extend(self.check_rule_5())
        violations.extend(self.check_rule_6())
        violations.extend(self.check_rule_7())
        violations.extend(self.check_rule_8())

        # Sort by point index
        violations.sort(key=lambda v: v.point_index)

        return violations


def check_all_rules(
    values: np.ndarray,
    control_limits: ControlLimits,
) -> List[SPCViolation]:
    """
    Convenience function to check all Western Electric rules

    Args:
        values: Process values
        control_limits: Control limits

    Returns:
        List of violations
    """
    rules = WesternElectricRules(values, control_limits)
    return rules.check_all_rules()


def filter_violations_by_severity(
    violations: List[SPCViolation],
    severity: str,
) -> List[SPCViolation]:
    """Filter violations by severity level"""
    return [v for v in violations if v.severity == severity]


def get_unique_violation_types(
    violations: List[SPCViolation],
) -> Set[SPCViolationType]:
    """Get set of unique violation types"""
    return {v.violation_type for v in violations}
