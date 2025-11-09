"""
Cumulative Sum (CUSUM) control charts.

CUSUM charts accumulate deviations from a target value, making them
highly sensitive to sustained shifts in the process mean.

Two-sided CUSUM uses upper (C⁺) and lower (C⁻) statistics:
C⁺_t = max(0, x_t - (μ₀ + K) + C⁺_{t-1})
C⁻_t = max(0, (μ₀ - K) - x_t + C⁻_{t-1})

Where:
- K is the reference value (typically 0.5σ)
- H is the decision interval (typically 5σ)

Signal when C⁺ > H or C⁻ > H.

Reference:
- Page, E.S., Biometrika 41, 100-115 (1954)

Will be implemented in Session 7.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from ..data.schemas import SPCPoint


class CUSUM:
    """
    Cumulative Sum control chart.
    
    Very sensitive to sustained shifts (0.5σ to 2σ).
    
    Status: STUB - To be implemented in Session 7
    """
    
    def __init__(
        self,
        K: float = 0.5,
        H: float = 5.0,
        target: Optional[float] = None,
        sigma: Optional[float] = None
    ):
        """
        Initialize CUSUM chart.
        
        Args:
            K: Reference value (typically 0.5σ for 1σ shift detection)
            H: Decision interval (typically 5σ)
            target: Target value (computed from data if None)
            sigma: Process standard deviation (computed if None)
        """
        self.K = K
        self.H = H
        self.target = target
        self.sigma = sigma
        self.C_plus = 0.0
        self.C_minus = 0.0
        
        raise NotImplementedError("Session 7: CUSUM initialization")
    
    def update(
        self,
        x: float
    ) -> Tuple[float, float]:
        """
        Update CUSUM statistics with new observation.
        
        C⁺_t = max(0, x_t - (μ₀ + K) + C⁺_{t-1})
        C⁻_t = max(0, (μ₀ - K) - x_t + C⁻_{t-1})
        
        Args:
            x: New observation
        
        Returns:
            (C_plus, C_minus) updated statistics
        
        Status: STUB - To be implemented in Session 7
        """
        raise NotImplementedError("Session 7: CUSUM update")
    
    def check_violation(
        self
    ) -> Optional[Dict[str, Any]]:
        """
        Check if CUSUM violates decision interval.
        
        Returns:
            Violation details if C⁺ > H or C⁻ > H, None otherwise
        
        Status: STUB - To be implemented in Session 7
        """
        raise NotImplementedError("Session 7: CUSUM violation check")
    
    def reset(self):
        """
        Reset CUSUM statistics to zero.
        
        Typically done after investigating and correcting an out-of-control
        condition.
        
        Status: STUB - To be implemented in Session 7
        """
        raise NotImplementedError("Session 7: CUSUM reset")
    
    def process_series(
        self,
        data: List[SPCPoint]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], List[Dict[str, Any]]]:
        """
        Process a series of observations.
        
        Args:
            data: SPC data points
        
        Returns:
            (C_plus_series, C_minus_series, violations)
        
        Status: STUB - To be implemented in Session 7
        """
        raise NotImplementedError("Session 7: CUSUM process series")
    
    def detect_shift(
        self,
        C_plus: NDArray[np.float64],
        C_minus: NDArray[np.float64]
    ) -> Optional[Tuple[int, str]]:
        """
        Detect when and in which direction a shift occurred.
        
        Args:
            C_plus: Upper CUSUM series
            C_minus: Lower CUSUM series
        
        Returns:
            (shift_index, direction) where direction is "up" or "down"
            None if no shift detected
        
        Status: STUB - To be implemented in Session 7
        """
        raise NotImplementedError("Session 7: CUSUM shift detection")
    
    def estimate_shift_time(
        self,
        C: NDArray[np.float64],
        detection_index: int
    ) -> int:
        """
        Estimate when the shift actually started (before detection).
        
        Uses the CUSUM trajectory to backtrack to the shift origin.
        
        Args:
            C: CUSUM series (C⁺ or C⁻)
            detection_index: Index where H was exceeded
        
        Returns:
            Estimated shift start index
        
        Status: STUB - To be implemented in Session 7
        """
        raise NotImplementedError("Session 7: CUSUM shift time estimation")


class TabularCUSUM:
    """
    Tabular CUSUM for easier manual implementation.
    
    Maintains a simplified version of CUSUM suitable for manual charting.
    
    Status: STUB - To be implemented in Session 7
    """
    
    def __init__(
        self,
        K: float = 0.5,
        H: float = 5.0,
        target: Optional[float] = None,
        sigma: Optional[float] = None
    ):
        """
        Initialize tabular CUSUM.
        
        Args:
            K: Reference value
            H: Decision interval
            target: Target value
            sigma: Process standard deviation
        """
        self.K = K
        self.H = H
        self.target = target
        self.sigma = sigma
        
        raise NotImplementedError("Session 7: Tabular CUSUM initialization")
    
    def generate_table(
        self,
        data: List[SPCPoint]
    ) -> Dict[str, List[float]]:
        """
        Generate tabular CUSUM table.
        
        Args:
            data: SPC data points
        
        Returns:
            Dictionary with columns: x, C+, C-, N+ (count), N- (count)
        
        Status: STUB - To be implemented in Session 7
        """
        raise NotImplementedError("Session 7: Tabular CUSUM table generation")


def cusum_arl(
    K: float,
    H: float,
    shift: float
) -> float:
    """
    Calculate Average Run Length (ARL) for CUSUM chart.
    
    Args:
        K: Reference value (in σ units)
        H: Decision interval (in σ units)
        shift: Shift magnitude (in σ units)
    
    Returns:
        Average run length
    
    Status: STUB - To be implemented in Session 7
    """
    raise NotImplementedError("Session 7: CUSUM ARL calculation")


def optimize_cusum_params(
    target_arl0: float = 370.0,
    shift_to_detect: float = 1.0
) -> Tuple[float, float]:
    """
    Optimize CUSUM parameters (K, H) for given detection criteria.
    
    Args:
        target_arl0: Desired in-control ARL (typically 370)
        shift_to_detect: Shift magnitude to detect quickly (in σ)
    
    Returns:
        (optimal_K, optimal_H)
    
    Status: STUB - To be implemented in Session 7
    """
    raise NotImplementedError("Session 7: CUSUM parameter optimization")


def fast_initial_response(
    data: List[SPCPoint],
    K: float,
    H: float,
    FIR_factor: float = 0.5
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    CUSUM with Fast Initial Response (FIR) for startup periods.
    
    Starts CUSUM at H/2 instead of 0 to detect shifts quickly during startup.
    
    Args:
        data: SPC data points
        K: Reference value
        H: Decision interval
        FIR_factor: Initial value as fraction of H (typically 0.5)
    
    Returns:
        (C_plus_series, C_minus_series) with FIR
    
    Status: STUB - To be implemented in Session 7
    """
    raise NotImplementedError("Session 7: CUSUM FIR")


__all__ = [
    "CUSUM",
    "TabularCUSUM",
    "cusum_arl",
    "optimize_cusum_params",
    "fast_initial_response",
]
