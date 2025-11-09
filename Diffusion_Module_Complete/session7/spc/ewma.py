"""
EWMA (Exponentially Weighted Moving Average) Control Charts - Session 7

Implements EWMA charts for detecting small shifts in process mean.
EWMA is more sensitive to small shifts than Shewhart charts.

Status: PRODUCTION READY ✅
"""

from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from datetime import datetime


@dataclass
class EWMALimits:
    """
    EWMA control limits.

    Attributes:
        centerline: Target mean
        ucl: Upper control limit array (time-varying)
        lcl: Lower control limit array (time-varying)
        lambda_: Smoothing parameter (0 < λ ≤ 1)
        L: Control limit width (typically 2.7-3.0)
    """
    centerline: float
    ucl: np.ndarray
    lcl: np.ndarray
    lambda_: float
    L: float


@dataclass
class EWMAViolation:
    """
    EWMA control chart violation.

    Attributes:
        index: Index where violation occurred
        timestamp: Timestamp (if provided)
        ewma_value: EWMA statistic value
        data_value: Original data value
        limit_exceeded: 'UCL' or 'LCL'
        description: Human-readable description
    """
    index: int
    timestamp: Optional[datetime]
    ewma_value: float
    data_value: float
    limit_exceeded: str
    description: str


class EWMAChart:
    """
    EWMA (Exponentially Weighted Moving Average) control chart.

    The EWMA statistic is:
        Z_t = λ * X_t + (1-λ) * Z_{t-1}

    Where:
        - λ (lambda) is the smoothing parameter (0 < λ ≤ 1)
        - X_t is the current observation
        - Z_{t-1} is the previous EWMA value

    Small λ (e.g., 0.05-0.2) is more sensitive to small shifts.
    Large λ (e.g., 0.8-1.0) behaves like a Shewhart chart.
    """

    def __init__(
        self,
        lambda_: float = 0.2,
        L: float = 3.0,
        target: Optional[float] = None,
        sigma: Optional[float] = None
    ):
        """
        Initialize EWMA chart.

        Args:
            lambda_: Smoothing parameter (0 < λ ≤ 1). Default 0.2
            L: Control limit multiplier. Default 3.0 (corresponds to ~99.73% coverage)
            target: Target mean (if None, estimated from data)
            sigma: Process std dev (if None, estimated from data)
        """
        if not (0 < lambda_ <= 1):
            raise ValueError("lambda must be in (0, 1]")

        self.lambda_ = lambda_
        self.L = L
        self.target = target
        self.sigma = sigma
        self.ewma_values: Optional[np.ndarray] = None
        self.limits: Optional[EWMALimits] = None

    def compute_ewma(
        self,
        data: np.ndarray,
        initial_ewma: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute EWMA statistics.

        Args:
            data: Process measurements
            initial_ewma: Starting value (if None, uses target or first observation)

        Returns:
            Array of EWMA values
        """
        ewma = np.zeros(len(data))

        if initial_ewma is None:
            initial_ewma = self.target if self.target is not None else data[0]

        ewma[0] = self.lambda_ * data[0] + (1 - self.lambda_) * initial_ewma

        for i in range(1, len(data)):
            ewma[i] = self.lambda_ * data[i] + (1 - self.lambda_) * ewma[i-1]

        self.ewma_values = ewma
        return ewma

    def compute_limits(self, data: np.ndarray, n: Optional[int] = None) -> EWMALimits:
        """
        Compute EWMA control limits.

        The control limits are time-varying:
            UCL(t) = μ + L * σ * sqrt(λ/(2-λ) * (1 - (1-λ)^(2t)))
            LCL(t) = μ - L * σ * sqrt(λ/(2-λ) * (1 - (1-λ)^(2t)))

        For large t, limits converge to:
            UCL(∞) = μ + L * σ * sqrt(λ/(2-λ))
            LCL(∞) = μ - L * σ * sqrt(λ/(2-λ))

        Args:
            data: Process measurements
            n: Length of data (if None, uses len(data))

        Returns:
            EWMALimits object
        """
        if self.target is None:
            self.target = np.mean(data)

        if self.sigma is None:
            self.sigma = np.std(data, ddof=1)

        if n is None:
            n = len(data)

        # Time-varying control limits
        t = np.arange(1, n + 1)
        variance_multiplier = np.sqrt(
            (self.lambda_ / (2 - self.lambda_)) * (1 - (1 - self.lambda_)**(2*t))
        )

        ucl = self.target + self.L * self.sigma * variance_multiplier
        lcl = self.target - self.L * self.sigma * variance_multiplier

        self.limits = EWMALimits(
            centerline=self.target,
            ucl=ucl,
            lcl=lcl,
            lambda_=self.lambda_,
            L=self.L
        )

        return self.limits

    def detect_violations(
        self,
        data: np.ndarray,
        timestamps: Optional[List[datetime]] = None
    ) -> List[EWMAViolation]:
        """
        Detect EWMA control limit violations.

        Args:
            data: Process measurements
            timestamps: Optional timestamps for violations

        Returns:
            List of detected violations
        """
        # Compute EWMA if not already done
        if self.ewma_values is None:
            self.compute_ewma(data)

        # Compute limits if not already done
        if self.limits is None:
            self.compute_limits(data)

        violations = []

        for i in range(len(self.ewma_values)):
            ewma_val = self.ewma_values[i]
            ucl = self.limits.ucl[i]
            lcl = self.limits.lcl[i]

            timestamp = timestamps[i] if timestamps and i < len(timestamps) else None

            if ewma_val > ucl:
                violations.append(EWMAViolation(
                    index=i,
                    timestamp=timestamp,
                    ewma_value=ewma_val,
                    data_value=data[i],
                    limit_exceeded='UCL',
                    description=f"EWMA ({ewma_val:.3f}) > UCL ({ucl:.3f})"
                ))

            elif ewma_val < lcl:
                violations.append(EWMAViolation(
                    index=i,
                    timestamp=timestamp,
                    ewma_value=ewma_val,
                    data_value=data[i],
                    limit_exceeded='LCL',
                    description=f"EWMA ({ewma_val:.3f}) < LCL ({lcl:.3f})"
                ))

        return violations

    def get_arl(self, shift_size: float, num_simulations: int = 10000) -> float:
        """
        Estimate Average Run Length (ARL) for a given shift size.

        ARL is the average number of samples before a shift is detected.

        Args:
            shift_size: Size of mean shift (in units of sigma)
            num_simulations: Number of Monte Carlo simulations

        Returns:
            Estimated ARL
        """
        if self.sigma is None:
            raise ValueError("Sigma must be set to compute ARL")

        run_lengths = []

        for _ in range(num_simulations):
            # Generate shifted data
            data = np.random.normal(
                self.target + shift_size * self.sigma,
                self.sigma,
                size=1000
            )

            ewma = self.compute_ewma(data)
            limits = self.compute_limits(data)

            # Find first violation
            violations = (ewma > limits.ucl) | (ewma < limits.lcl)
            if np.any(violations):
                run_length = np.argmax(violations) + 1
            else:
                run_length = len(data)

            run_lengths.append(run_length)

        return np.mean(run_lengths)


def quick_ewma_check(
    data: np.ndarray,
    lambda_: float = 0.2,
    L: float = 3.0,
    target: Optional[float] = None,
    sigma: Optional[float] = None,
    timestamps: Optional[List[datetime]] = None
) -> Tuple[np.ndarray, List[EWMAViolation]]:
    """
    Quick helper to run EWMA check on data.

    Args:
        data: Process measurements
        lambda_: Smoothing parameter (default 0.2)
        L: Control limit multiplier (default 3.0)
        target: Target mean (if None, estimated from data)
        sigma: Process std dev (if None, estimated from data)
        timestamps: Optional timestamps for violations

    Returns:
        Tuple of (EWMA values, violations list)

    Example:
        >>> ewma_vals, violations = quick_ewma_check(junction_depths, lambda_=0.1)
        >>> if violations:
        ...     print(f"Detected {len(violations)} violations")
    """
    chart = EWMAChart(lambda_=lambda_, L=L, target=target, sigma=sigma)
    ewma_vals = chart.compute_ewma(data)
    chart.compute_limits(data)
    violations = chart.detect_violations(data, timestamps)

    return ewma_vals, violations


def optimal_ewma_params(
    target_arl0: float = 370.0,
    shift_size: float = 1.0
) -> Tuple[float, float]:
    """
    Find optimal EWMA parameters for given ARL and shift size.

    Args:
        target_arl0: Target in-control ARL (default 370 ≈ Shewhart chart)
        shift_size: Expected shift size in sigma units

    Returns:
        Tuple of (optimal lambda, optimal L)

    Note:
        This uses approximate formulas. For precise ARL matching,
        use numerical search with get_arl().
    """
    # Approximate optimal lambda based on shift size
    # Small shifts → smaller lambda
    if shift_size <= 0.5:
        lambda_opt = 0.05
    elif shift_size <= 1.0:
        lambda_opt = 0.10
    elif shift_size <= 1.5:
        lambda_opt = 0.20
    else:
        lambda_opt = 0.40

    # Approximate L to achieve target ARL0
    # This is a simplified approximation
    if target_arl0 >= 500:
        L_opt = 3.09
    elif target_arl0 >= 370:
        L_opt = 3.00
    elif target_arl0 >= 200:
        L_opt = 2.86
    else:
        L_opt = 2.70

    return lambda_opt, L_opt
