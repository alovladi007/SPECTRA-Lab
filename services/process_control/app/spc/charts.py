"""Statistical Process Control (SPC) charts.

Implements X-bar/R, EWMA, CUSUM charts with Western Electric rules,
alert detection, and process capability analysis.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import math


# ============================================================================
# Data Classes and Enums
# ============================================================================

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class RuleViolation(Enum):
    """Western Electric rule violations."""
    RULE1_BEYOND_3SIGMA = "rule1_beyond_3sigma"
    RULE2_NINE_SAME_SIDE = "rule2_nine_same_side"
    RULE3_SIX_TRENDING = "rule3_six_trending"
    RULE4_FOURTEEN_ALTERNATING = "rule4_fourteen_alternating"
    RULE5_TWO_OF_THREE_BEYOND_2SIGMA = "rule5_two_of_three_beyond_2sigma"
    RULE6_FOUR_OF_FIVE_BEYOND_1SIGMA = "rule6_four_of_five_beyond_1sigma"
    RULE7_FIFTEEN_WITHIN_1SIGMA = "rule7_fifteen_within_1sigma"
    RULE8_EIGHT_BEYOND_1SIGMA = "rule8_eight_beyond_1sigma"


@dataclass
class SPCAlert:
    """SPC alert with context."""
    timestamp: float
    parameter_name: str
    value: float
    rule_violated: RuleViolation
    severity: AlertSeverity
    message: str
    ucl: float  # Upper control limit
    lcl: float  # Lower control limit
    center_line: float

    # Deduplication
    alert_id: str = ""
    is_duplicate: bool = False


@dataclass
class ControlLimits:
    """Control limits for SPC charts."""
    center_line: float
    ucl: float  # Upper control limit
    lcl: float  # Lower control limit

    # For X-bar/R charts
    ucl_range: Optional[float] = None
    cl_range: Optional[float] = None


@dataclass
class SPCChartState:
    """State of an SPC chart."""
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    control_limits: Optional[ControlLimits] = None
    alerts: List[SPCAlert] = field(default_factory=list)

    # Alert deduplication
    recent_alerts: deque = field(default_factory=lambda: deque(maxlen=10))


# ============================================================================
# X-bar and R Chart
# ============================================================================

class XbarRChart:
    """X-bar (mean) and R (range) control chart.

    Used for monitoring process mean and variability when subgroups
    are available (e.g., multiple measurements per wafer/lot).
    """

    def __init__(
        self,
        parameter_name: str,
        subgroup_size: int = 5,
        window_size: int = 25
    ):
        """Initialize X-bar/R chart.

        Args:
            parameter_name: Name of parameter being monitored
            subgroup_size: Number of samples per subgroup
            window_size: Number of subgroups for calculating limits
        """
        self.parameter_name = parameter_name
        self.subgroup_size = subgroup_size
        self.window_size = window_size

        self.state = SPCChartState()

        # Constants for control limits (depends on subgroup size)
        self._set_constants()

    def _set_constants(self):
        """Set statistical constants based on subgroup size."""
        # A2 constant for X-bar chart UCL/LCL
        A2_values = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
        self.A2 = A2_values.get(self.subgroup_size, 0.577)

        # D3, D4 constants for R chart
        D3_values = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4_values = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

        self.D3 = D3_values.get(self.subgroup_size, 0)
        self.D4 = D4_values.get(self.subgroup_size, 2.114)

        # d2 constant for estimating sigma
        d2_values = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
        self.d2 = d2_values.get(self.subgroup_size, 2.326)

    def add_subgroup(self, subgroup: np.ndarray, timestamp: float):
        """Add a subgroup of measurements.

        Args:
            subgroup: Array of measurements (length = subgroup_size)
            timestamp: Timestamp of subgroup
        """
        if len(subgroup) != self.subgroup_size:
            raise ValueError(f"Subgroup size must be {self.subgroup_size}")

        # Calculate subgroup mean and range
        x_bar = np.mean(subgroup)
        r = np.max(subgroup) - np.min(subgroup)

        # Store
        self.state.data_points.append((x_bar, r))
        self.state.timestamps.append(timestamp)

        # Calculate control limits if enough data
        if len(self.state.data_points) >= self.window_size:
            self._calculate_control_limits()

            # Check for rule violations
            alerts = self._check_western_electric_rules()
            self.state.alerts.extend(alerts)

    def _calculate_control_limits(self):
        """Calculate control limits from historical data."""
        recent_data = list(self.state.data_points)[-self.window_size:]

        x_bars = [d[0] for d in recent_data]
        ranges = [d[1] for d in recent_data]

        # Grand mean (center line for X-bar chart)
        x_double_bar = np.mean(x_bars)

        # Average range (center line for R chart)
        r_bar = np.mean(ranges)

        # X-bar chart limits
        ucl_xbar = x_double_bar + self.A2 * r_bar
        lcl_xbar = x_double_bar - self.A2 * r_bar

        # R chart limits
        ucl_r = self.D4 * r_bar
        lcl_r = self.D3 * r_bar

        self.state.control_limits = ControlLimits(
            center_line=x_double_bar,
            ucl=ucl_xbar,
            lcl=lcl_xbar,
            ucl_range=ucl_r,
            cl_range=r_bar
        )

    def _check_western_electric_rules(self) -> List[SPCAlert]:
        """Check Western Electric rules for out-of-control conditions."""
        if self.state.control_limits is None:
            return []

        alerts = []

        # Get recent data points (X-bar values only)
        recent_xbars = [d[0] for d in list(self.state.data_points)[-30:]]
        recent_times = list(self.state.timestamps)[-30:]

        if len(recent_xbars) < 10:
            return []

        cl = self.state.control_limits.center_line
        ucl = self.state.control_limits.ucl
        lcl = self.state.control_limits.lcl
        sigma = (ucl - cl) / 3  # Estimate sigma

        # Latest value
        latest_val = recent_xbars[-1]
        latest_time = recent_times[-1]

        # Rule 1: One point beyond 3σ
        if latest_val > ucl or latest_val < lcl:
            alert = SPCAlert(
                timestamp=latest_time,
                parameter_name=self.parameter_name,
                value=latest_val,
                rule_violated=RuleViolation.RULE1_BEYOND_3SIGMA,
                severity=AlertSeverity.CRITICAL,
                message=f"{self.parameter_name} beyond 3σ limit: {latest_val:.2f}",
                ucl=ucl,
                lcl=lcl,
                center_line=cl
            )
            alert.alert_id = f"{self.parameter_name}_rule1_{int(latest_time)}"
            alerts.append(alert)

        # Rule 2: Nine consecutive points on same side of center line
        if len(recent_xbars) >= 9:
            last_nine = recent_xbars[-9:]
            if all(x > cl for x in last_nine) or all(x < cl for x in last_nine):
                alert = SPCAlert(
                    timestamp=latest_time,
                    parameter_name=self.parameter_name,
                    value=latest_val,
                    rule_violated=RuleViolation.RULE2_NINE_SAME_SIDE,
                    severity=AlertSeverity.WARNING,
                    message=f"{self.parameter_name}: 9 consecutive points on same side of CL",
                    ucl=ucl,
                    lcl=lcl,
                    center_line=cl
                )
                alert.alert_id = f"{self.parameter_name}_rule2_{int(latest_time)}"
                alerts.append(alert)

        # Rule 3: Six consecutive points trending up or down
        if len(recent_xbars) >= 6:
            last_six = recent_xbars[-6:]
            increasing = all(last_six[i] < last_six[i+1] for i in range(5))
            decreasing = all(last_six[i] > last_six[i+1] for i in range(5))

            if increasing or decreasing:
                alert = SPCAlert(
                    timestamp=latest_time,
                    parameter_name=self.parameter_name,
                    value=latest_val,
                    rule_violated=RuleViolation.RULE3_SIX_TRENDING,
                    severity=AlertSeverity.WARNING,
                    message=f"{self.parameter_name}: 6 consecutive points trending",
                    ucl=ucl,
                    lcl=lcl,
                    center_line=cl
                )
                alert.alert_id = f"{self.parameter_name}_rule3_{int(latest_time)}"
                alerts.append(alert)

        # Rule 5: Two of three consecutive points beyond 2σ
        if len(recent_xbars) >= 3:
            last_three = recent_xbars[-3:]
            beyond_2sigma = sum(1 for x in last_three if abs(x - cl) > 2 * sigma)

            if beyond_2sigma >= 2:
                alert = SPCAlert(
                    timestamp=latest_time,
                    parameter_name=self.parameter_name,
                    value=latest_val,
                    rule_violated=RuleViolation.RULE5_TWO_OF_THREE_BEYOND_2SIGMA,
                    severity=AlertSeverity.WARNING,
                    message=f"{self.parameter_name}: 2 of 3 points beyond 2σ",
                    ucl=ucl,
                    lcl=lcl,
                    center_line=cl
                )
                alert.alert_id = f"{self.parameter_name}_rule5_{int(latest_time)}"
                alerts.append(alert)

        # Deduplicate alerts
        alerts = self._deduplicate_alerts(alerts)

        return alerts

    def _deduplicate_alerts(self, new_alerts: List[SPCAlert]) -> List[SPCAlert]:
        """Remove duplicate alerts based on recent alert history."""
        deduplicated = []

        recent_alert_ids = {alert.alert_id for alert in self.state.recent_alerts}

        for alert in new_alerts:
            if alert.alert_id not in recent_alert_ids:
                deduplicated.append(alert)
                self.state.recent_alerts.append(alert)
            else:
                alert.is_duplicate = True

        return deduplicated


# ============================================================================
# EWMA (Exponentially Weighted Moving Average) Chart
# ============================================================================

class EWMAChart:
    """EWMA control chart for detecting small process shifts.

    More sensitive to small sustained shifts than X-bar charts.
    """

    def __init__(
        self,
        parameter_name: str,
        lambda_weight: float = 0.2,
        L: float = 3.0,
        window_size: int = 25
    ):
        """Initialize EWMA chart.

        Args:
            parameter_name: Name of parameter being monitored
            lambda_weight: Weighting factor (0 < λ ≤ 1), smaller = more smoothing
            L: Control limit multiplier (typically 2.7-3.0)
            window_size: Number of points for calculating baseline
        """
        self.parameter_name = parameter_name
        self.lambda_weight = lambda_weight
        self.L = L
        self.window_size = window_size

        self.state = SPCChartState()
        self.ewma_value = None
        self.baseline_mean = None
        self.baseline_std = None

    def add_point(self, value: float, timestamp: float):
        """Add a data point to the EWMA chart.

        Args:
            value: Measured value
            timestamp: Timestamp of measurement
        """
        self.state.data_points.append(value)
        self.state.timestamps.append(timestamp)

        # Calculate baseline statistics
        if len(self.state.data_points) >= self.window_size and self.baseline_mean is None:
            baseline_data = list(self.state.data_points)[:self.window_size]
            self.baseline_mean = np.mean(baseline_data)
            self.baseline_std = np.std(baseline_data, ddof=1)
            self.ewma_value = self.baseline_mean

        # Update EWMA
        if self.ewma_value is not None:
            self.ewma_value = self.lambda_weight * value + (1 - self.lambda_weight) * self.ewma_value

            # Calculate control limits
            self._calculate_control_limits()

            # Check for violations
            alerts = self._check_violations(value, timestamp)
            self.state.alerts.extend(alerts)

    def _calculate_control_limits(self):
        """Calculate EWMA control limits."""
        if self.baseline_mean is None or self.baseline_std is None:
            return

        n = len(self.state.data_points)

        # Control limit width (increases with sample size)
        lambda_factor = self.lambda_weight / (2 - self.lambda_weight)
        denominator = 1 - (1 - self.lambda_weight) ** (2 * n)

        if denominator > 0:
            sigma_ewma = self.baseline_std * math.sqrt(lambda_factor * denominator)
        else:
            sigma_ewma = 0

        ucl = self.baseline_mean + self.L * sigma_ewma
        lcl = self.baseline_mean - self.L * sigma_ewma

        self.state.control_limits = ControlLimits(
            center_line=self.baseline_mean,
            ucl=ucl,
            lcl=lcl
        )

    def _check_violations(self, value: float, timestamp: float) -> List[SPCAlert]:
        """Check if EWMA violates control limits."""
        if self.state.control_limits is None or self.ewma_value is None:
            return []

        alerts = []

        ucl = self.state.control_limits.ucl
        lcl = self.state.control_limits.lcl
        cl = self.state.control_limits.center_line

        if self.ewma_value > ucl or self.ewma_value < lcl:
            alert = SPCAlert(
                timestamp=timestamp,
                parameter_name=self.parameter_name,
                value=value,
                rule_violated=RuleViolation.RULE1_BEYOND_3SIGMA,
                severity=AlertSeverity.WARNING,
                message=f"{self.parameter_name} EWMA beyond control limits: {self.ewma_value:.2f}",
                ucl=ucl,
                lcl=lcl,
                center_line=cl
            )
            alert.alert_id = f"{self.parameter_name}_ewma_{int(timestamp)}"

            # Deduplicate
            recent_ids = {a.alert_id for a in self.state.recent_alerts}
            if alert.alert_id not in recent_ids:
                alerts.append(alert)
                self.state.recent_alerts.append(alert)

        return alerts


# ============================================================================
# CUSUM (Cumulative Sum) Chart
# ============================================================================

class CUSUMChart:
    """CUSUM control chart for detecting process shifts.

    Accumulates deviations from target, very sensitive to sustained shifts.
    """

    def __init__(
        self,
        parameter_name: str,
        target: Optional[float] = None,
        k: float = 0.5,  # Slack value (in sigma units)
        h: float = 5.0,  # Decision interval (in sigma units)
        window_size: int = 25
    ):
        """Initialize CUSUM chart.

        Args:
            parameter_name: Name of parameter
            target: Target value (if None, calculated from baseline)
            k: Reference value (typically 0.5σ)
            h: Decision interval (typically 4-5σ)
            window_size: Points for baseline calculation
        """
        self.parameter_name = parameter_name
        self.target = target
        self.k = k
        self.h = h
        self.window_size = window_size

        self.state = SPCChartState()

        self.cusum_high = 0.0  # Cumulative sum for high side
        self.cusum_low = 0.0   # Cumulative sum for low side

        self.baseline_std = None

    def add_point(self, value: float, timestamp: float):
        """Add a data point to CUSUM chart."""
        self.state.data_points.append(value)
        self.state.timestamps.append(timestamp)

        # Calculate baseline
        if len(self.state.data_points) >= self.window_size and self.baseline_std is None:
            baseline_data = list(self.state.data_points)[:self.window_size]
            if self.target is None:
                self.target = np.mean(baseline_data)
            self.baseline_std = np.std(baseline_data, ddof=1)

        # Update CUSUM
        if self.target is not None and self.baseline_std is not None:
            # Standardize
            z = (value - self.target) / self.baseline_std

            # Update CUSUM values
            self.cusum_high = max(0, self.cusum_high + z - self.k)
            self.cusum_low = max(0, self.cusum_low - z - self.k)

            # Check for violations
            alerts = self._check_violations(value, timestamp)
            self.state.alerts.extend(alerts)

    def _check_violations(self, value: float, timestamp: float) -> List[SPCAlert]:
        """Check if CUSUM exceeds decision interval."""
        if self.baseline_std is None:
            return []

        alerts = []

        # High-side violation
        if self.cusum_high > self.h:
            alert = SPCAlert(
                timestamp=timestamp,
                parameter_name=self.parameter_name,
                value=value,
                rule_violated=RuleViolation.RULE1_BEYOND_3SIGMA,
                severity=AlertSeverity.WARNING,
                message=f"{self.parameter_name} CUSUM high exceeded: {self.cusum_high:.2f}",
                ucl=self.h,
                lcl=-self.h,
                center_line=0.0
            )
            alert.alert_id = f"{self.parameter_name}_cusum_high_{int(timestamp)}"

            recent_ids = {a.alert_id for a in self.state.recent_alerts}
            if alert.alert_id not in recent_ids:
                alerts.append(alert)
                self.state.recent_alerts.append(alert)

                # Reset after detection
                self.cusum_high = 0.0

        # Low-side violation
        if self.cusum_low > self.h:
            alert = SPCAlert(
                timestamp=timestamp,
                parameter_name=self.parameter_name,
                value=value,
                rule_violated=RuleViolation.RULE1_BEYOND_3SIGMA,
                severity=AlertSeverity.WARNING,
                message=f"{self.parameter_name} CUSUM low exceeded: {self.cusum_low:.2f}",
                ucl=self.h,
                lcl=-self.h,
                center_line=0.0
            )
            alert.alert_id = f"{self.parameter_name}_cusum_low_{int(timestamp)}"

            recent_ids = {a.alert_id for a in self.state.recent_alerts}
            if alert.alert_id not in recent_ids:
                alerts.append(alert)
                self.state.recent_alerts.append(alert)

                # Reset after detection
                self.cusum_low = 0.0

        return alerts


# Export
__all__ = [
    "XbarRChart",
    "EWMAChart",
    "CUSUMChart",
    "SPCAlert",
    "ControlLimits",
    "AlertSeverity",
    "RuleViolation",
]
