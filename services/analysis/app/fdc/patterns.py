"""
Pattern Detection

Advanced pattern detection algorithms for time-series data.
"""

import numpy as np
from typing import List, Optional, Dict
from dataclasses import dataclass
from scipy import signal, stats


@dataclass
class TrendPattern:
    """Detected trend pattern"""
    start_index: int
    end_index: int
    slope: float
    r_squared: float
    is_significant: bool


@dataclass
class ShiftPattern:
    """Detected shift pattern"""
    change_point_index: int
    mean_before: float
    mean_after: float
    shift_magnitude: float
    is_significant: bool


@dataclass
class CyclicPattern:
    """Detected cyclic pattern"""
    period: float  # In number of samples
    amplitude: float
    phase: float
    frequency: float
    power: float  # Spectral power
    is_significant: bool


class PatternDetector:
    """
    Advanced pattern detection for time-series data

    Uses statistical methods to detect:
    - Trends (linear regression)
    - Shifts (change-point detection)
    - Cycles (FFT/periodogram analysis)
    """

    def __init__(
        self,
        values: np.ndarray,
        significance_level: float = 0.05,
    ):
        """
        Args:
            values: Time-series values
            significance_level: Statistical significance level (default 0.05)
        """
        self.values = values
        self.n = len(values)
        self.significance_level = significance_level

    def detect_trend(
        self,
        min_r_squared: float = 0.5,
    ) -> Optional[TrendPattern]:
        """
        Detect linear trend using regression

        Args:
            min_r_squared: Minimum R² for significance

        Returns:
            TrendPattern if significant trend detected
        """
        if self.n < 5:
            return None

        # Linear regression
        x = np.arange(self.n)
        slope, intercept = np.polyfit(x, self.values, deg=1)

        # Calculate R²
        y_pred = slope * x + intercept
        ss_res = np.sum((self.values - y_pred) ** 2)
        ss_tot = np.sum((self.values - np.mean(self.values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Statistical significance test (t-test for slope)
        # Null hypothesis: slope = 0
        se_slope = np.sqrt(ss_res / (self.n - 2)) / np.sqrt(np.sum((x - np.mean(x)) ** 2))
        t_stat = slope / se_slope if se_slope > 0 else 0
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), self.n - 2))

        is_significant = (r_squared >= min_r_squared) and (p_value < self.significance_level)

        return TrendPattern(
            start_index=0,
            end_index=self.n - 1,
            slope=slope,
            r_squared=r_squared,
            is_significant=is_significant,
        )

    def detect_shift(
        self,
        min_shift_magnitude: Optional[float] = None,
    ) -> Optional[ShiftPattern]:
        """
        Detect step change using simple change-point detection

        Args:
            min_shift_magnitude: Minimum magnitude for significance (in units of std dev)

        Returns:
            ShiftPattern if significant shift detected
        """
        if self.n < 10:
            return None

        if min_shift_magnitude is None:
            min_shift_magnitude = 1.5  # 1.5 std dev

        process_std = np.std(self.values, ddof=1)
        threshold = min_shift_magnitude * process_std

        # Find change point with maximum shift
        max_shift = 0
        max_shift_index = 0

        min_segment = max(5, int(self.n * 0.2))

        for i in range(min_segment, self.n - min_segment):
            mean_before = np.mean(self.values[:i])
            mean_after = np.mean(self.values[i:])

            shift = abs(mean_after - mean_before)

            if shift > max_shift:
                max_shift = shift
                max_shift_index = i

        if max_shift > threshold:
            mean_before = np.mean(self.values[:max_shift_index])
            mean_after = np.mean(self.values[max_shift_index:])

            # T-test for means
            t_stat, p_value = stats.ttest_ind(
                self.values[:max_shift_index],
                self.values[max_shift_index:],
                equal_var=False,
            )

            is_significant = p_value < self.significance_level

            return ShiftPattern(
                change_point_index=max_shift_index,
                mean_before=mean_before,
                mean_after=mean_after,
                shift_magnitude=mean_after - mean_before,
                is_significant=is_significant,
            )

        return None

    def detect_cycles(
        self,
        min_period: int = 3,
        max_period: Optional[int] = None,
    ) -> Optional[CyclicPattern]:
        """
        Detect cyclic patterns using FFT

        Args:
            min_period: Minimum period to consider
            max_period: Maximum period to consider (default: n/2)

        Returns:
            CyclicPattern if significant cycle detected
        """
        if self.n < 2 * min_period:
            return None

        if max_period is None:
            max_period = self.n // 2

        # Detrend data
        detrended = signal.detrend(self.values)

        # Compute periodogram
        frequencies, power = signal.periodogram(detrended)

        # Convert frequencies to periods
        # Skip DC component (index 0)
        periods = 1.0 / frequencies[1:]
        power_spectrum = power[1:]

        # Filter by period range
        valid_mask = (periods >= min_period) & (periods <= max_period)

        if not np.any(valid_mask):
            return None

        valid_periods = periods[valid_mask]
        valid_power = power_spectrum[valid_mask]

        # Find dominant frequency
        max_power_idx = np.argmax(valid_power)
        dominant_period = valid_periods[max_power_idx]
        dominant_power = valid_power[max_power_idx]
        dominant_freq = 1.0 / dominant_period

        # Estimate amplitude and phase by fitting sine wave
        t = np.arange(self.n)
        omega = 2 * np.pi * dominant_freq

        # Fit: y = A*sin(ωt + φ) + C
        # Use least squares: y = a*sin(ωt) + b*cos(ωt) + c
        A_matrix = np.column_stack([
            np.sin(omega * t),
            np.cos(omega * t),
            np.ones(self.n),
        ])

        coeffs, _, _, _ = np.linalg.lstsq(A_matrix, detrended, rcond=None)

        a, b, c = coeffs
        amplitude = np.sqrt(a**2 + b**2)
        phase = np.arctan2(b, a)

        # Test significance: compare power to noise level
        noise_power = np.median(valid_power)
        signal_to_noise = dominant_power / noise_power if noise_power > 0 else 0

        is_significant = signal_to_noise > 5.0  # Arbitrary threshold

        return CyclicPattern(
            period=dominant_period,
            amplitude=amplitude,
            phase=phase,
            frequency=dominant_freq,
            power=dominant_power,
            is_significant=is_significant,
        )


def detect_all_patterns(
    values: np.ndarray,
) -> Dict[str, Optional[object]]:
    """
    Convenience function to detect all pattern types

    Args:
        values: Time-series values

    Returns:
        Dictionary with detected patterns
    """
    detector = PatternDetector(values)

    return {
        "trend": detector.detect_trend(),
        "shift": detector.detect_shift(),
        "cycle": detector.detect_cycles(),
    }
