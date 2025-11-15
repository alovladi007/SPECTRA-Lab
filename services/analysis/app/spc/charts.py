"""
SPC Charts Implementation

Implements standard SPC chart types:
- X-bar/R charts (mean and range)
- EWMA charts (exponentially weighted moving average)
- CUSUM charts (cumulative sum)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

from .series import SPCSeries, SPCDataPoint


class SPCChartType(str, Enum):
    """SPC chart types"""
    XBAR_R = "xbar_r"
    EWMA = "ewma"
    CUSUM = "cusum"
    INDIVIDUALS = "individuals"


@dataclass
class ControlLimits:
    """Control limits for SPC charts"""
    center_line: float
    upper_control_limit: float
    lower_control_limit: float
    upper_warning_limit: float
    lower_warning_limit: float

    # Additional statistics
    process_mean: float
    process_std: float


@dataclass
class SPCChartResult:
    """Result from SPC chart calculation"""
    chart_type: SPCChartType
    control_limits: ControlLimits
    chart_values: np.ndarray  # Values plotted on the chart
    original_values: np.ndarray  # Original data values
    out_of_control: List[int]  # Indices of out-of-control points


class SPCChart(ABC):
    """Abstract base class for SPC charts"""

    def __init__(self, series: SPCSeries):
        self.series = series
        self.control_limits: Optional[ControlLimits] = None

    @abstractmethod
    def calculate_control_limits(self, baseline_points: Optional[int] = None) -> ControlLimits:
        """Calculate control limits from data"""
        pass

    @abstractmethod
    def calculate_chart_values(self) -> np.ndarray:
        """Calculate values to plot on the chart"""
        pass

    def detect_out_of_control(
        self,
        chart_values: np.ndarray,
        control_limits: ControlLimits,
    ) -> List[int]:
        """
        Detect out-of-control points

        Points beyond control limits (±3σ)
        """
        out_of_control = []

        for i, value in enumerate(chart_values):
            if value > control_limits.upper_control_limit or value < control_limits.lower_control_limit:
                out_of_control.append(i)

        return out_of_control

    def run_chart(self, baseline_points: Optional[int] = None) -> SPCChartResult:
        """
        Run complete SPC chart analysis

        Args:
            baseline_points: Number of initial points to use for establishing control limits.
                           If None, uses all points.

        Returns:
            SPCChartResult with control limits and out-of-control points
        """
        # Calculate control limits
        control_limits = self.calculate_control_limits(baseline_points)
        self.control_limits = control_limits

        # Calculate chart values
        chart_values = self.calculate_chart_values()

        # Detect out-of-control points
        out_of_control = self.detect_out_of_control(chart_values, control_limits)

        return SPCChartResult(
            chart_type=self.get_chart_type(),
            control_limits=control_limits,
            chart_values=chart_values,
            original_values=self.series.get_values(),
            out_of_control=out_of_control,
        )

    @abstractmethod
    def get_chart_type(self) -> SPCChartType:
        """Get chart type identifier"""
        pass


# =============================================================================
# X-bar/R Chart
# =============================================================================

class XBarRChart(SPCChart):
    """
    X-bar and R Chart (Mean and Range)

    Traditional SPC chart for subgroup data.
    For CVD applications, each subgroup can be:
    - Multiple wafers in a lot
    - Multiple measurement points on a wafer
    - Multiple runs of the same recipe

    If individual measurements are provided (no subgroups),
    this becomes an Individuals chart (I-chart).
    """

    def __init__(
        self,
        series: SPCSeries,
        subgroup_size: int = 1,
    ):
        super().__init__(series)
        self.subgroup_size = subgroup_size

    def get_chart_type(self) -> SPCChartType:
        if self.subgroup_size == 1:
            return SPCChartType.INDIVIDUALS
        return SPCChartType.XBAR_R

    def calculate_subgroups(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate subgroup means and ranges

        Returns:
            (subgroup_means, subgroup_ranges)
        """
        n_points = len(values)
        n_subgroups = n_points // self.subgroup_size

        # Truncate to complete subgroups
        values_truncated = values[:n_subgroups * self.subgroup_size]

        # Reshape into subgroups
        subgroups = values_truncated.reshape(n_subgroups, self.subgroup_size)

        # Calculate means and ranges
        subgroup_means = np.mean(subgroups, axis=1)
        subgroup_ranges = np.ptp(subgroups, axis=1)  # peak-to-peak (max - min)

        return subgroup_means, subgroup_ranges

    def get_control_chart_constants(self, n: int) -> Dict[str, float]:
        """
        Get control chart constants for subgroup size n

        Constants from SPC literature (Montgomery, 2009)
        """
        # A2: Factor for X-bar chart control limits (3-sigma)
        A2_table = {
            1: 2.660,  # For individuals chart (moving range)
            2: 1.880,
            3: 1.023,
            4: 0.729,
            5: 0.577,
            6: 0.483,
            7: 0.419,
            8: 0.373,
            9: 0.337,
            10: 0.308,
        }

        # D3, D4: Factors for R chart control limits
        D3_table = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
        D4_table = {1: 3.267, 2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}

        # d2: Relationship between range and std dev
        d2_table = {1: 1.128, 2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}

        n_clamped = min(max(n, 1), 10)

        return {
            "A2": A2_table[n_clamped],
            "D3": D3_table[n_clamped],
            "D4": D4_table[n_clamped],
            "d2": d2_table[n_clamped],
        }

    def calculate_control_limits(self, baseline_points: Optional[int] = None) -> ControlLimits:
        """
        Calculate X-bar chart control limits

        For individuals chart (n=1), uses moving range method.
        """
        values = self.series.get_values()

        if baseline_points is not None:
            values_baseline = values[:baseline_points]
        else:
            values_baseline = values

        if self.subgroup_size == 1:
            # Individuals chart (I-chart) with moving range
            return self._calculate_individuals_limits(values_baseline)
        else:
            # X-bar/R chart with subgroups
            return self._calculate_xbar_limits(values_baseline)

    def _calculate_individuals_limits(self, values: np.ndarray) -> ControlLimits:
        """Calculate control limits for individuals chart using moving range"""
        # Calculate moving range (MR)
        moving_ranges = np.abs(np.diff(values))
        mean_range = np.mean(moving_ranges)

        # Process mean
        process_mean = np.mean(values)

        # Get constants
        constants = self.get_control_chart_constants(n=2)  # MR uses n=2
        d2 = constants["d2"]

        # Estimate process std from average moving range
        process_std = mean_range / d2

        # Control limits (±3σ)
        ucl = process_mean + 3.0 * process_std
        lcl = process_mean - 3.0 * process_std

        # Warning limits (±2σ)
        uwl = process_mean + 2.0 * process_std
        lwl = process_mean - 2.0 * process_std

        return ControlLimits(
            center_line=process_mean,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=uwl,
            lower_warning_limit=lwl,
            process_mean=process_mean,
            process_std=process_std,
        )

    def _calculate_xbar_limits(self, values: np.ndarray) -> ControlLimits:
        """Calculate control limits for X-bar chart with subgroups"""
        # Calculate subgroup means and ranges
        subgroup_means, subgroup_ranges = self.calculate_subgroups(values)

        # Overall mean and average range
        mean_of_means = np.mean(subgroup_means)
        mean_range = np.mean(subgroup_ranges)

        # Get constants
        constants = self.get_control_chart_constants(self.subgroup_size)
        A2 = constants["A2"]
        d2 = constants["d2"]

        # Estimate process std
        process_std = mean_range / d2

        # Control limits
        ucl = mean_of_means + A2 * mean_range
        lcl = mean_of_means - A2 * mean_range

        # Warning limits (±2σ)
        sigma_xbar = process_std / np.sqrt(self.subgroup_size)
        uwl = mean_of_means + 2.0 * sigma_xbar
        lwl = mean_of_means - 2.0 * sigma_xbar

        return ControlLimits(
            center_line=mean_of_means,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=uwl,
            lower_warning_limit=lwl,
            process_mean=mean_of_means,
            process_std=process_std,
        )

    def calculate_chart_values(self) -> np.ndarray:
        """Calculate values to plot on chart"""
        values = self.series.get_values()

        if self.subgroup_size == 1:
            # For individuals chart, plot individual values
            return values
        else:
            # For X-bar chart, plot subgroup means
            subgroup_means, _ = self.calculate_subgroups(values)
            return subgroup_means


# =============================================================================
# EWMA Chart
# =============================================================================

class EWMAChart(SPCChart):
    """
    EWMA Chart (Exponentially Weighted Moving Average)

    More sensitive to small shifts than X-bar chart.
    Good for detecting gradual drift in process mean.

    EWMA formula:
        Z_i = λ * X_i + (1 - λ) * Z_{i-1}

    where:
        λ = smoothing parameter (typically 0.1 to 0.3)
        X_i = current observation
        Z_i = EWMA value
    """

    def __init__(
        self,
        series: SPCSeries,
        lambda_: float = 0.2,
        L: float = 3.0,
    ):
        """
        Args:
            series: SPC series data
            lambda_: Smoothing parameter (0 < λ ≤ 1). Smaller = more smoothing.
            L: Width of control limits (multiples of std dev). Typically 3.0.
        """
        super().__init__(series)
        self.lambda_ = lambda_
        self.L = L

    def get_chart_type(self) -> SPCChartType:
        return SPCChartType.EWMA

    def calculate_control_limits(self, baseline_points: Optional[int] = None) -> ControlLimits:
        """Calculate EWMA control limits"""
        values = self.series.get_values()

        if baseline_points is not None:
            values_baseline = values[:baseline_points]
        else:
            values_baseline = values

        # Process mean and std
        process_mean = np.mean(values_baseline)
        process_std = np.std(values_baseline, ddof=1)

        # EWMA std dev (asymptotic)
        sigma_ewma = process_std * np.sqrt(self.lambda_ / (2.0 - self.lambda_))

        # Control limits (±L * σ_EWMA)
        ucl = process_mean + self.L * sigma_ewma
        lcl = process_mean - self.L * sigma_ewma

        # Warning limits (±2/3 * L * σ_EWMA)
        uwl = process_mean + (2.0/3.0) * self.L * sigma_ewma
        lwl = process_mean - (2.0/3.0) * self.L * sigma_ewma

        return ControlLimits(
            center_line=process_mean,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=uwl,
            lower_warning_limit=lwl,
            process_mean=process_mean,
            process_std=process_std,
        )

    def calculate_chart_values(self) -> np.ndarray:
        """Calculate EWMA values"""
        values = self.series.get_values()
        n = len(values)

        ewma = np.zeros(n)
        ewma[0] = values[0]  # Initialize with first value

        for i in range(1, n):
            ewma[i] = self.lambda_ * values[i] + (1.0 - self.lambda_) * ewma[i-1]

        return ewma


# =============================================================================
# CUSUM Chart
# =============================================================================

class CUSUMChart(SPCChart):
    """
    CUSUM Chart (Cumulative Sum)

    Very sensitive to small sustained shifts in process mean.
    Accumulates deviations from target.

    CUSUM formulas:
        C_i^+ = max(0, C_{i-1}^+ + (X_i - target) - K)
        C_i^- = max(0, C_{i-1}^- - (X_i - target) - K)

    where:
        K = slack parameter (typically 0.5 * σ)
        H = decision interval (typically 5 * σ)
    """

    def __init__(
        self,
        series: SPCSeries,
        k_sigma: float = 0.5,
        h_sigma: float = 5.0,
        target: Optional[float] = None,
    ):
        """
        Args:
            series: SPC series data
            k_sigma: Slack parameter as multiple of σ
            h_sigma: Decision interval as multiple of σ
            target: Target value (if None, uses process mean)
        """
        super().__init__(series)
        self.k_sigma = k_sigma
        self.h_sigma = h_sigma
        self.target = target

    def get_chart_type(self) -> SPCChartType:
        return SPCChartType.CUSUM

    def calculate_control_limits(self, baseline_points: Optional[int] = None) -> ControlLimits:
        """Calculate CUSUM control limits"""
        values = self.series.get_values()

        if baseline_points is not None:
            values_baseline = values[:baseline_points]
        else:
            values_baseline = values

        # Process mean and std
        process_mean = np.mean(values_baseline)
        process_std = np.std(values_baseline, ddof=1)

        # Target (center line)
        if self.target is None:
            target = process_mean
        else:
            target = self.target

        # Control limits (±H)
        H = self.h_sigma * process_std
        ucl = H
        lcl = -H

        # Warning limits (±2/3 * H)
        uwl = (2.0/3.0) * H
        lwl = -(2.0/3.0) * H

        return ControlLimits(
            center_line=target,
            upper_control_limit=ucl,
            lower_control_limit=lcl,
            upper_warning_limit=uwl,
            lower_warning_limit=lwl,
            process_mean=process_mean,
            process_std=process_std,
        )

    def calculate_chart_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate CUSUM values (both upper and lower)

        Returns:
            (cusum_upper, cusum_lower)
        """
        values = self.series.get_values()
        n = len(values)

        # Target and K
        if self.control_limits is not None:
            target = self.control_limits.center_line
            process_std = self.control_limits.process_std
        else:
            target = np.mean(values)
            process_std = np.std(values, ddof=1)

        K = self.k_sigma * process_std

        # Initialize CUSUM arrays
        cusum_upper = np.zeros(n)
        cusum_lower = np.zeros(n)

        for i in range(n):
            if i == 0:
                cusum_upper[i] = max(0.0, values[i] - target - K)
                cusum_lower[i] = max(0.0, target - values[i] - K)
            else:
                cusum_upper[i] = max(0.0, cusum_upper[i-1] + values[i] - target - K)
                cusum_lower[i] = max(0.0, cusum_lower[i-1] + target - values[i] - K)

        return cusum_upper, cusum_lower

    def run_chart(self, baseline_points: Optional[int] = None) -> SPCChartResult:
        """
        Run CUSUM chart analysis

        For CUSUM, chart_values contains both upper and lower CUSUM.
        Out-of-control points are where |CUSUM| > H.
        """
        # Calculate control limits
        control_limits = self.calculate_control_limits(baseline_points)
        self.control_limits = control_limits

        # Calculate CUSUM values
        cusum_upper, cusum_lower = self.calculate_chart_values()

        # Detect out-of-control points
        H = control_limits.upper_control_limit
        out_of_control = []

        for i in range(len(cusum_upper)):
            if cusum_upper[i] > H or cusum_lower[i] > H:
                out_of_control.append(i)

        # Combine upper and lower CUSUM for chart_values
        # Stack them for visualization
        chart_values = cusum_upper  # Primary chart shows upper CUSUM

        return SPCChartResult(
            chart_type=SPCChartType.CUSUM,
            control_limits=control_limits,
            chart_values=chart_values,
            original_values=self.series.get_values(),
            out_of_control=out_of_control,
        )
