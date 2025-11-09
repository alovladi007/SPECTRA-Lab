"""
CUSUM (Cumulative Sum) Control Charts - Session 7

Implements CUSUM charts for detecting sustained shifts in process mean.
CUSUM is particularly effective for detecting small to moderate persistent shifts.

Status: PRODUCTION READY ✅
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class CUSUMViolation:
    """
    CUSUM control chart violation.

    Attributes:
        index: Index where violation occurred
        timestamp: Timestamp (if provided)
        cusum_high: Upper CUSUM value at violation
        cusum_low: Lower CUSUM value at violation
        data_value: Original data value
        direction: 'HIGH' or 'LOW'
        description: Human-readable description
    """
    index: int
    timestamp: Optional[datetime]
    cusum_high: float
    cusum_low: float
    data_value: float
    direction: str
    description: str


class CUSUMChart:
    """
    CUSUM (Cumulative Sum) control chart using tabular method.

    The CUSUM statistics are:
        C_i^+ = max(0, C_{i-1}^+ + (x_i - μ0 - K))  # Upper CUSUM
        C_i^- = max(0, C_{i-1}^- + (μ0 - x_i - K))  # Lower CUSUM

    Where:
        - K is the reference value (slack parameter), typically K = δ/2
        - δ is the expected shift size (in units of σ)
        - H is the decision interval (control limit), typically H = 4σ or 5σ

    A signal occurs when C^+ > H or C^- > H
    """

    def __init__(
        self,
        target: Optional[float] = None,
        sigma: Optional[float] = None,
        k: Optional[float] = None,
        h: Optional[float] = None,
        shift_size: float = 1.0
    ):
        """
        Initialize CUSUM chart.

        Args:
            target: Target mean μ0 (if None, estimated from data)
            sigma: Process std dev σ (if None, estimated from data)
            k: Reference value (if None, set to shift_size/2 * sigma)
            h: Decision interval (if None, set to 5.0 * sigma)
            shift_size: Expected shift size in units of σ (default 1.0)
        """
        self.target = target
        self.sigma = sigma
        self.k = k
        self.h = h
        self.shift_size = shift_size
        self.cusum_high: Optional[np.ndarray] = None
        self.cusum_low: Optional[np.ndarray] = None

    def compute_cusum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CUSUM statistics (tabular method).

        Args:
            data: Process measurements

        Returns:
            Tuple of (C^+, C^-) arrays
        """
        # Estimate parameters if not provided
        if self.target is None:
            self.target = np.mean(data)

        if self.sigma is None:
            self.sigma = np.std(data, ddof=1)

        if self.k is None:
            self.k = (self.shift_size / 2.0) * self.sigma

        if self.h is None:
            self.h = 5.0 * self.sigma

        n = len(data)
        cusum_high = np.zeros(n)
        cusum_low = np.zeros(n)

        for i in range(n):
            if i == 0:
                cusum_high[i] = max(0, data[i] - self.target - self.k)
                cusum_low[i] = max(0, self.target - data[i] - self.k)
            else:
                cusum_high[i] = max(0, cusum_high[i-1] + data[i] - self.target - self.k)
                cusum_low[i] = max(0, cusum_low[i-1] + self.target - data[i] - self.k)

        self.cusum_high = cusum_high
        self.cusum_low = cusum_low

        return cusum_high, cusum_low

    def detect_violations(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> List[CUSUMViolation]:
        """
        Detect CUSUM control limit violations.

        Args:
            data: Process measurements
            timestamps: Optional timestamps for violations

        Returns:
            List of detected violations
        """
        # Compute CUSUM if not already done
        if self.cusum_high is None or self.cusum_low is None:
            self.compute_cusum(data)

        violations = []

        for i in range(len(data)):
            c_high = self.cusum_high[i]
            c_low = self.cusum_low[i]

            timestamp = timestamps[i] if timestamps and i < len(timestamps) else None

            if c_high > self.h:
                violations.append(CUSUMViolation(
                    index=i,
                    timestamp=timestamp,
                    cusum_high=c_high,
                    cusum_low=c_low,
                    data_value=data[i],
                    direction='HIGH',
                    description=f"CUSUM High ({c_high:.3f}) > H ({self.h:.3f})"
                ))

            if c_low > self.h:
                violations.append(CUSUMViolation(
                    index=i,
                    timestamp=timestamp,
                    cusum_high=c_high,
                    cusum_low=c_low,
                    data_value=data[i],
                    direction='LOW',
                    description=f"CUSUM Low ({c_low:.3f}) > H ({self.h:.3f})"
                ))

        return violations

    def estimate_change_point(self, violation_index: int) -> int:
        """
        Estimate when the shift actually occurred.

        When CUSUM signals at time t, the shift likely occurred earlier.
        This backtracks to find the most likely change point.

        Args:
            violation_index: Index where CUSUM signaled

        Returns:
            Estimated change point index
        """
        if self.cusum_high is None or self.cusum_low is None:
            raise ValueError("CUSUM not computed. Call compute_cusum() first.")

        # Determine which CUSUM signaled
        if self.cusum_high[violation_index] > self.h:
            # Backtrack on upper CUSUM
            cusum = self.cusum_high
        else:
            # Backtrack on lower CUSUM
            cusum = self.cusum_low

        # Find where CUSUM last reset to zero
        for i in range(violation_index, -1, -1):
            if cusum[i] == 0:
                return i + 1

        return 0

    def get_arl(
        self,
        shift_size: float = 0.0,
        num_simulations: int = 1000,
        max_samples: int = 10000
    ) -> float:
        """
        Estimate Average Run Length (ARL) for a given shift size.

        Args:
            shift_size: Size of mean shift (in units of sigma)
            num_simulations: Number of Monte Carlo simulations
            max_samples: Maximum samples per simulation

        Returns:
            Estimated ARL
        """
        if self.sigma is None or self.target is None:
            raise ValueError("Target and sigma must be set to compute ARL")

        run_lengths = []

        for _ in range(num_simulations):
            # Generate data with shift
            data = np.random.normal(
                self.target + shift_size * self.sigma,
                self.sigma,
                size=max_samples
            )

            cusum_high, cusum_low = self.compute_cusum(data)

            # Find first violation
            violations_high = cusum_high > self.h
            violations_low = cusum_low > self.h

            violations = violations_high | violations_low

            if np.any(violations):
                run_length = np.argmax(violations) + 1
            else:
                run_length = max_samples

            run_lengths.append(run_length)

        return np.mean(run_lengths)


class FastInitialResponse_CUSUM(CUSUMChart):
    """
    CUSUM with Fast Initial Response (FIR).

    FIR-CUSUM starts with a head start value instead of zero,
    making it more sensitive to shifts that occur at startup.
    """

    def __init__(
        self,
        target: Optional[float] = None,
        sigma: Optional[float] = None,
        k: Optional[float] = None,
        h: Optional[float] = None,
        shift_size: float = 1.0,
        head_start: float = 0.5
    ):
        """
        Initialize FIR-CUSUM.

        Args:
            target: Target mean
            sigma: Process std dev
            k: Reference value
            h: Decision interval
            shift_size: Expected shift size in sigma units
            head_start: Initial CUSUM value as fraction of H (default 0.5)
        """
        super().__init__(target, sigma, k, h, shift_size)
        self.head_start = head_start

    def compute_cusum(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute FIR-CUSUM statistics with head start.

        Args:
            data: Process measurements

        Returns:
            Tuple of (C^+, C^-) arrays
        """
        # Estimate parameters if not provided
        if self.target is None:
            self.target = np.mean(data)

        if self.sigma is None:
            self.sigma = np.std(data, ddof=1)

        if self.k is None:
            self.k = (self.shift_size / 2.0) * self.sigma

        if self.h is None:
            self.h = 5.0 * self.sigma

        # Initial values with head start
        initial_value = self.head_start * self.h

        n = len(data)
        cusum_high = np.zeros(n)
        cusum_low = np.zeros(n)

        for i in range(n):
            if i == 0:
                cusum_high[i] = max(0, initial_value + data[i] - self.target - self.k)
                cusum_low[i] = max(0, initial_value + self.target - data[i] - self.k)
            else:
                cusum_high[i] = max(0, cusum_high[i-1] + data[i] - self.target - self.k)
                cusum_low[i] = max(0, cusum_low[i-1] + self.target - data[i] - self.k)

        self.cusum_high = cusum_high
        self.cusum_low = cusum_low

        return cusum_high, cusum_low


def quick_cusum_check(
    data: np.ndarray,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    shift_size: float = 1.0,
    timestamps: Optional[List[datetime]] = None,
    use_fir: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[CUSUMViolation]]:
    """
    Quick helper to run CUSUM check on data.

    Args:
        data: Process measurements
        target: Target mean (if None, estimated from data)
        sigma: Process std dev (if None, estimated from data)
        shift_size: Expected shift size in sigma units
        timestamps: Optional timestamps for violations
        use_fir: If True, use FIR-CUSUM with head start

    Returns:
        Tuple of (C^+, C^-, violations list)

    Example:
        >>> c_high, c_low, violations = quick_cusum_check(junction_depths)
        >>> if violations:
        ...     print(f"Detected {len(violations)} violations")
    """
    if use_fir:
        chart = FastInitialResponse_CUSUM(
            target=target,
            sigma=sigma,
            shift_size=shift_size
        )
    else:
        chart = CUSUMChart(
            target=target,
            sigma=sigma,
            shift_size=shift_size
        )

    c_high, c_low = chart.compute_cusum(data)
    violations = chart.detect_violations(data, timestamps)

    return c_high, c_low, violations


def optimal_cusum_params(
    target_arl0: float = 370.0,
    shift_size: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal CUSUM parameters for given ARL and shift size.

    Args:
        target_arl0: Target in-control ARL (default 370)
        shift_size: Expected shift size in sigma units

    Returns:
        Tuple of (optimal K, optimal H) in sigma units

    Note:
        These are approximate values based on standard tables.
        For precise ARL matching, use numerical search with get_arl().
    """
    # K is typically set to half the expected shift
    k_opt = shift_size / 2.0

    # H depends on desired ARL0 and shift size
    # These are approximate values from Montgomery (2009)
    if shift_size <= 0.5:
        if target_arl0 >= 500:
            h_opt = 8.01
        elif target_arl0 >= 370:
            h_opt = 5.0
        else:
            h_opt = 4.0
    elif shift_size <= 1.0:
        if target_arl0 >= 500:
            h_opt = 5.5
        elif target_arl0 >= 370:
            h_opt = 5.0
        else:
            h_opt = 4.0
    else:
        if target_arl0 >= 500:
            h_opt = 5.0
        elif target_arl0 >= 370:
            h_opt = 4.5
        else:
            h_opt = 4.0

    return k_opt, h_opt
