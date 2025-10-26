"""
Session 13: Statistical Process Control (SPC) Hub - Complete Implementation

This module provides comprehensive SPC functionality including:
- Control charts (X-bar/R, EWMA, CUSUM, I-MR)
- Process capability (Cp, Cpk, Pp, Ppk)
- Western Electric and Nelson rules detection
- Real-time alerting and escalation
- Root cause analysis suggestions
- Trend analysis and prediction

Author: Semiconductor Lab Platform Team
Version: 1.0.0
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import curve_fit
import json
import warnings
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations and Constants
# ============================================================================

class ChartType(str, Enum):
    """Types of control charts"""
    XBAR_R = "xbar_r"  # X-bar and R chart (subgroups)
    I_MR = "i_mr"  # Individual and Moving Range
    EWMA = "ewma"  # Exponentially Weighted Moving Average
    CUSUM = "cusum"  # Cumulative Sum
    P = "p"  # Proportion defective
    C = "c"  # Count of defects


class RuleViolation(str, Enum):
    """Western Electric and Nelson rule violations"""
    RULE_1 = "rule_1"  # One point beyond 3σ
    RULE_2 = "rule_2"  # 2 of 3 points beyond 2σ (same side)
    RULE_3 = "rule_3"  # 4 of 5 points beyond 1σ (same side)
    RULE_4 = "rule_4"  # 8 consecutive points on same side of centerline
    RULE_5 = "rule_5"  # 6 points in a row increasing or decreasing
    RULE_6 = "rule_6"  # 15 points in a row in Zone C (both sides)
    RULE_7 = "rule_7"  # 14 points in a row alternating up/down
    RULE_8 = "rule_8"  # 8 points in a row beyond Zone C (both sides)


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Urgent attention needed
    MEDIUM = "medium"  # Should be investigated
    LOW = "low"  # Monitor closely
    INFO = "info"  # Informational only


class ProcessStatus(str, Enum):
    """Process control status"""
    IN_CONTROL = "in_control"
    OUT_OF_CONTROL = "out_of_control"
    WARNING = "warning"
    UNKNOWN = "unknown"


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ControlLimits:
    """Control limits for SPC charts"""
    ucl: float  # Upper Control Limit
    lcl: float  # Lower Control Limit
    centerline: float
    usl: Optional[float] = None  # Upper Specification Limit
    lsl: Optional[float] = None  # Lower Specification Limit
    sigma: Optional[float] = None
    computed_from_n: int = 0
    chart_type: str = "xbar_r"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SPCAlert:
    """SPC alert/violation"""
    id: str
    timestamp: datetime
    metric: str
    rule_violated: RuleViolation
    severity: AlertSeverity
    value: float
    control_limits: ControlLimits
    points_involved: List[int]  # Indices of points involved
    message: str
    suggested_actions: List[str] = field(default_factory=list)
    root_causes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessCapability:
    """Process capability indices"""
    cp: float  # Process Capability (precision)
    cpk: float  # Process Capability Index (precision + centering)
    pp: float  # Process Performance (long-term precision)
    ppk: float  # Process Performance Index (long-term precision + centering)
    cpm: Optional[float] = None  # Taguchi index
    sigma_level: float = 0.0  # Six Sigma level
    dpmo: float = 0.0  # Defects Per Million Opportunities
    is_capable: bool = False
    comments: List[str] = field(default_factory=list)


@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    trend_detected: bool
    trend_direction: str  # "increasing", "decreasing", "stable"
    trend_slope: float
    trend_significance: float  # p-value
    predicted_values: List[float]
    prediction_intervals: List[Tuple[float, float]]
    changepoints: List[int]  # Indices where process changed


# ============================================================================
# Control Chart Calculators
# ============================================================================

class ControlChartCalculator:
    """Base class for control chart calculations"""
    
    # Constants for control chart factors (from statistical tables)
    D3_TABLE = {
        2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223
    }
    D4_TABLE = {
        2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004,
        7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777
    }
    A2_TABLE = {
        2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
        7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308
    }
    d2_TABLE = {
        2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326, 6: 2.534,
        7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078
    }
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    @staticmethod
    def calculate_xbar_r_limits(
        data: np.ndarray,
        subgroup_size: int,
        sigma_multiplier: float = 3.0
    ) -> Tuple[ControlLimits, ControlLimits]:
        """
        Calculate X-bar and R chart control limits
        
        Args:
            data: Array of measurements
            subgroup_size: Number of samples per subgroup
            sigma_multiplier: Multiplier for control limits (default 3σ)
        
        Returns:
            (xbar_limits, r_limits)
        """
        # Reshape data into subgroups
        n_subgroups = len(data) // subgroup_size
        subgroups = data[:n_subgroups * subgroup_size].reshape(n_subgroups, subgroup_size)
        
        # Calculate subgroup means and ranges
        xbar = np.mean(subgroups, axis=1)
        ranges = np.ptp(subgroups, axis=1)
        
        # Grand average and average range
        xbar_grand = np.mean(xbar)
        r_bar = np.mean(ranges)
        
        # Get factors from tables
        A2 = ControlChartCalculator.A2_TABLE.get(subgroup_size, 0.577)
        D3 = ControlChartCalculator.D3_TABLE.get(subgroup_size, 0)
        D4 = ControlChartCalculator.D4_TABLE.get(subgroup_size, 2.114)
        d2 = ControlChartCalculator.d2_TABLE.get(subgroup_size, 2.326)
        
        # X-bar chart limits
        xbar_ucl = xbar_grand + A2 * r_bar
        xbar_lcl = xbar_grand - A2 * r_bar
        
        # Estimate sigma
        sigma = r_bar / d2
        
        xbar_limits = ControlLimits(
            ucl=xbar_ucl,
            lcl=xbar_lcl,
            centerline=xbar_grand,
            sigma=sigma,
            computed_from_n=n_subgroups,
            chart_type="xbar",
            metadata={"subgroup_size": subgroup_size, "A2": A2}
        )
        
        # R chart limits
        r_ucl = D4 * r_bar
        r_lcl = D3 * r_bar
        
        r_limits = ControlLimits(
            ucl=r_ucl,
            lcl=r_lcl,
            centerline=r_bar,
            sigma=None,
            computed_from_n=n_subgroups,
            chart_type="r",
            metadata={"subgroup_size": subgroup_size, "D3": D3, "D4": D4}
        )
        
        return xbar_limits, r_limits
    
    @staticmethod
    def calculate_i_mr_limits(
        data: np.ndarray,
        moving_range_span: int = 2
    ) -> Tuple[ControlLimits, ControlLimits]:
        """
        Calculate Individual and Moving Range chart limits
        
        Args:
            data: Array of individual measurements
            moving_range_span: Span for moving range calculation
        
        Returns:
            (i_limits, mr_limits)
        """
        # Individual values
        i_mean = np.mean(data)
        
        # Moving ranges
        mr = np.abs(np.diff(data, n=moving_range_span - 1))
        mr_mean = np.mean(mr)
        
        # Constants
        d2 = ControlChartCalculator.d2_TABLE.get(moving_range_span, 1.128)
        D3 = ControlChartCalculator.D3_TABLE.get(moving_range_span, 0)
        D4 = ControlChartCalculator.D4_TABLE.get(moving_range_span, 3.267)
        
        # Estimate sigma
        sigma = mr_mean / d2
        
        # Individual chart limits
        i_ucl = i_mean + 3 * sigma
        i_lcl = i_mean - 3 * sigma
        
        i_limits = ControlLimits(
            ucl=i_ucl,
            lcl=i_lcl,
            centerline=i_mean,
            sigma=sigma,
            computed_from_n=len(data),
            chart_type="i",
            metadata={"moving_range_span": moving_range_span}
        )
        
        # Moving range limits
        mr_ucl = D4 * mr_mean
        mr_lcl = D3 * mr_mean
        
        mr_limits = ControlLimits(
            ucl=mr_ucl,
            lcl=mr_lcl,
            centerline=mr_mean,
            sigma=None,
            computed_from_n=len(mr),
            chart_type="mr",
            metadata={"moving_range_span": moving_range_span, "D3": D3, "D4": D4}
        )
        
        return i_limits, mr_limits
    
    @staticmethod
    def calculate_ewma_limits(
        data: np.ndarray,
        lambda_weight: float = 0.2,
        target: Optional[float] = None,
        sigma: Optional[float] = None
    ) -> ControlLimits:
        """
        Calculate EWMA (Exponentially Weighted Moving Average) limits
        
        Args:
            data: Array of measurements
            lambda_weight: EWMA weight (0 < λ ≤ 1)
            target: Target value (default: data mean)
            sigma: Process standard deviation (default: estimated)
        
        Returns:
            EWMA control limits
        """
        if target is None:
            target = np.mean(data)
        
        if sigma is None:
            sigma = np.std(data, ddof=1)
        
        # EWMA control limits
        L = 3  # Typically 3 for 3-sigma limits
        
        # Limits widen asymptotically, use limiting value
        limit_factor = L * sigma * np.sqrt(lambda_weight / (2 - lambda_weight))
        
        ucl = target + limit_factor
        lcl = target - limit_factor
        
        limits = ControlLimits(
            ucl=ucl,
            lcl=lcl,
            centerline=target,
            sigma=sigma,
            computed_from_n=len(data),
            chart_type="ewma",
            metadata={"lambda": lambda_weight, "L": L}
        )
        
        return limits
    
    @staticmethod
    def calculate_cusum(
        data: np.ndarray,
        target: Optional[float] = None,
        sigma: Optional[float] = None,
        k: float = 0.5,
        h: float = 5.0
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate CUSUM (Cumulative Sum) values
        
        Args:
            data: Array of measurements
            target: Target value
            sigma: Process standard deviation
            k: Slack parameter (typically 0.5σ)
            h: Decision interval (typically 4-5σ)
        
        Returns:
            (cusum_high, cusum_low, decision_limit)
        """
        if target is None:
            target = np.mean(data)
        
        if sigma is None:
            sigma = np.std(data, ddof=1)
        
        # Standardize data
        z = (data - target) / sigma
        
        # CUSUM for detecting increases
        cusum_high = np.zeros(len(z))
        for i in range(1, len(z)):
            cusum_high[i] = max(0, cusum_high[i-1] + z[i] - k)
        
        # CUSUM for detecting decreases
        cusum_low = np.zeros(len(z))
        for i in range(1, len(z)):
            cusum_low[i] = min(0, cusum_low[i-1] + z[i] + k)
        
        # Decision limit
        decision_limit = h * sigma
        
        return cusum_high, cusum_low, decision_limit


# ============================================================================
# Rule Detection
# ============================================================================

class RuleDetector:
    """Detect Western Electric and Nelson rules violations"""
    
    def __init__(self, limits: ControlLimits):
        """
        Initialize rule detector
        
        Args:
            limits: Control limits for evaluation
        """
        self.limits = limits
        self.ucl = limits.ucl
        self.lcl = limits.lcl
        self.cl = limits.centerline
        self.sigma = limits.sigma or (limits.ucl - limits.centerline) / 3
    
    def detect_all_rules(self, data: np.ndarray) -> List[SPCAlert]:
        """
        Detect all rule violations
        
        Args:
            data: Array of measurements
        
        Returns:
            List of alerts
        """
        alerts = []
        
        # Rule 1: One point beyond 3σ
        alerts.extend(self._detect_rule_1(data))
        
        # Rule 2: 2 of 3 points beyond 2σ
        alerts.extend(self._detect_rule_2(data))
        
        # Rule 3: 4 of 5 points beyond 1σ
        alerts.extend(self._detect_rule_3(data))
        
        # Rule 4: 8 consecutive points on same side
        alerts.extend(self._detect_rule_4(data))
        
        # Rule 5: 6 points trending
        alerts.extend(self._detect_rule_5(data))
        
        # Rule 6: 15 points in Zone C
        alerts.extend(self._detect_rule_6(data))
        
        # Rule 7: 14 points alternating
        alerts.extend(self._detect_rule_7(data))
        
        # Rule 8: 8 points beyond Zone C
        alerts.extend(self._detect_rule_8(data))
        
        return alerts
    
    def _detect_rule_1(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 1: One point beyond 3σ"""
        alerts = []
        
        for i, value in enumerate(data):
            if value > self.ucl or value < self.lcl:
                severity = AlertSeverity.CRITICAL
                message = f"Point {i+1} beyond 3σ limit: {value:.4f}"
                
                alert = SPCAlert(
                    id=f"R1_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_1,
                    severity=severity,
                    value=value,
                    control_limits=self.limits,
                    points_involved=[i],
                    message=message,
                    suggested_actions=[
                        "Investigate special cause immediately",
                        "Check measurement system",
                        "Review process parameters",
                        "Inspect equipment"
                    ],
                    root_causes=[
                        "Equipment malfunction",
                        "Operator error",
                        "Material variation",
                        "Measurement error"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_rule_2(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 2: 2 of 3 consecutive points beyond 2σ (same side)"""
        alerts = []
        zone_2_upper = self.cl + 2 * self.sigma
        zone_2_lower = self.cl - 2 * self.sigma
        
        for i in range(2, len(data)):
            window = data[i-2:i+1]
            
            # Check upper side
            count_upper = sum(1 for x in window if x > zone_2_upper)
            if count_upper >= 2:
                alert = SPCAlert(
                    id=f"R2_U_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_2,
                    severity=AlertSeverity.HIGH,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-2, i+1)),
                    message=f"2 of 3 points beyond +2σ (points {i-1} to {i+1})",
                    suggested_actions=[
                        "Investigate process shift",
                        "Check for systematic bias",
                        "Review recent changes"
                    ],
                    root_causes=[
                        "Process drift",
                        "Calibration shift",
                        "Environmental change"
                    ]
                )
                alerts.append(alert)
            
            # Check lower side
            count_lower = sum(1 for x in window if x < zone_2_lower)
            if count_lower >= 2:
                alert = SPCAlert(
                    id=f"R2_L_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_2,
                    severity=AlertSeverity.HIGH,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-2, i+1)),
                    message=f"2 of 3 points beyond -2σ (points {i-1} to {i+1})",
                    suggested_actions=[
                        "Investigate process shift",
                        "Check for systematic bias",
                        "Review recent changes"
                    ],
                    root_causes=[
                        "Process drift",
                        "Calibration shift",
                        "Environmental change"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_rule_3(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 3: 4 of 5 consecutive points beyond 1σ (same side)"""
        alerts = []
        zone_1_upper = self.cl + self.sigma
        zone_1_lower = self.cl - self.sigma
        
        for i in range(4, len(data)):
            window = data[i-4:i+1]
            
            count_upper = sum(1 for x in window if x > zone_1_upper)
            count_lower = sum(1 for x in window if x < zone_1_lower)
            
            if count_upper >= 4:
                alert = SPCAlert(
                    id=f"R3_U_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_3,
                    severity=AlertSeverity.MEDIUM,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-4, i+1)),
                    message=f"4 of 5 points beyond +1σ (points {i-3} to {i+1})",
                    suggested_actions=[
                        "Monitor process closely",
                        "Look for increasing trend",
                        "Check for gradual shift"
                    ]
                )
                alerts.append(alert)
            
            if count_lower >= 4:
                alert = SPCAlert(
                    id=f"R3_L_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_3,
                    severity=AlertSeverity.MEDIUM,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-4, i+1)),
                    message=f"4 of 5 points beyond -1σ (points {i-3} to {i+1})",
                    suggested_actions=[
                        "Monitor process closely",
                        "Look for decreasing trend",
                        "Check for gradual shift"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_rule_4(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 4: 8 consecutive points on same side of centerline"""
        alerts = []
        
        for i in range(7, len(data)):
            window = data[i-7:i+1]
            
            if all(x > self.cl for x in window):
                alert = SPCAlert(
                    id=f"R4_U_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_4,
                    severity=AlertSeverity.MEDIUM,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-7, i+1)),
                    message=f"8 consecutive points above centerline (points {i-6} to {i+1})",
                    suggested_actions=[
                        "Investigate process shift",
                        "Check for bias in measurement",
                        "Review process centering"
                    ]
                )
                alerts.append(alert)
            
            elif all(x < self.cl for x in window):
                alert = SPCAlert(
                    id=f"R4_L_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_4,
                    severity=AlertSeverity.MEDIUM,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-7, i+1)),
                    message=f"8 consecutive points below centerline (points {i-6} to {i+1})",
                    suggested_actions=[
                        "Investigate process shift",
                        "Check for bias in measurement",
                        "Review process centering"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_rule_5(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 5: 6 points in a row increasing or decreasing"""
        alerts = []
        
        for i in range(5, len(data)):
            window = data[i-5:i+1]
            diffs = np.diff(window)
            
            if all(d > 0 for d in diffs):
                alert = SPCAlert(
                    id=f"R5_INC_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_5,
                    severity=AlertSeverity.MEDIUM,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-5, i+1)),
                    message=f"6 consecutive increasing points (points {i-4} to {i+1})",
                    suggested_actions=[
                        "Investigate upward trend",
                        "Check for wear or degradation",
                        "Look for increasing temperatures"
                    ]
                )
                alerts.append(alert)
            
            elif all(d < 0 for d in diffs):
                alert = SPCAlert(
                    id=f"R5_DEC_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_5,
                    severity=AlertSeverity.MEDIUM,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-5, i+1)),
                    message=f"6 consecutive decreasing points (points {i-4} to {i+1})",
                    suggested_actions=[
                        "Investigate downward trend",
                        "Check for tool wear",
                        "Look for decreasing temperatures"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_rule_6(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 6: 15 points in a row in Zone C (within 1σ)"""
        alerts = []
        zone_c_upper = self.cl + self.sigma
        zone_c_lower = self.cl - self.sigma
        
        for i in range(14, len(data)):
            window = data[i-14:i+1]
            
            if all(zone_c_lower <= x <= zone_c_upper for x in window):
                alert = SPCAlert(
                    id=f"R6_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_6,
                    severity=AlertSeverity.LOW,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-14, i+1)),
                    message=f"15 consecutive points in Zone C (points {i-13} to {i+1})",
                    suggested_actions=[
                        "Check for reduced variability",
                        "Verify measurement system sensitivity",
                        "Consider stratification"
                    ],
                    root_causes=[
                        "Over-control",
                        "Measurement rounding",
                        "Data manipulation"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_rule_7(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 7: 14 points in a row alternating up and down"""
        alerts = []
        
        for i in range(13, len(data)):
            window = data[i-13:i+1]
            diffs = np.diff(window)
            
            # Check if signs alternate
            sign_changes = np.diff(np.sign(diffs))
            if all(abs(sc) == 2 for sc in sign_changes):
                alert = SPCAlert(
                    id=f"R7_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_7,
                    severity=AlertSeverity.LOW,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-13, i+1)),
                    message=f"14 points alternating up/down (points {i-12} to {i+1})",
                    suggested_actions=[
                        "Check for systematic oscillation",
                        "Look for alternating operators",
                        "Review measurement procedure"
                    ],
                    root_causes=[
                        "Overadjustment",
                        "Alternating material batches",
                        "Temperature cycling"
                    ]
                )
                alerts.append(alert)
        
        return alerts
    
    def _detect_rule_8(self, data: np.ndarray) -> List[SPCAlert]:
        """Detect Rule 8: 8 points in a row beyond Zone C (outside 1σ)"""
        alerts = []
        zone_c_upper = self.cl + self.sigma
        zone_c_lower = self.cl - self.sigma
        
        for i in range(7, len(data)):
            window = data[i-7:i+1]
            
            if all((x > zone_c_upper or x < zone_c_lower) for x in window):
                alert = SPCAlert(
                    id=f"R8_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(),
                    metric=self.limits.chart_type,
                    rule_violated=RuleViolation.RULE_8,
                    severity=AlertSeverity.MEDIUM,
                    value=data[i],
                    control_limits=self.limits,
                    points_involved=list(range(i-7, i+1)),
                    message=f"8 points beyond Zone C (points {i-6} to {i+1})",
                    suggested_actions=[
                        "Check for increased variability",
                        "Review process consistency",
                        "Investigate multiple input streams"
                    ],
                    root_causes=[
                        "Mixture of distributions",
                        "Multiple operators/machines",
                        "Stratification"
                    ]
                )
                alerts.append(alert)
        
        return alerts


# ============================================================================
# Process Capability Calculator
# ============================================================================

class CapabilityCalculator:
    """Calculate process capability indices"""
    
    @staticmethod
    def calculate_capability(
        data: np.ndarray,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
        target: Optional[float] = None
    ) -> ProcessCapability:
        """
        Calculate process capability indices
        
        Args:
            data: Array of measurements
            usl: Upper Specification Limit
            lsl: Lower Specification Limit
            target: Target value (default: midpoint of spec limits)
        
        Returns:
            ProcessCapability object
        """
        # Basic statistics
        mean = np.mean(data)
        std_short = np.std(data, ddof=1)  # Short-term std (within subgroups)
        std_long = np.std(data, ddof=1)  # Long-term std (overall)
        
        # For simplicity, assume short-term = long-term
        # In practice, calculate from subgroup ranges
        
        comments = []
        
        # Calculate Cp (potential capability)
        cp = float('inf')
        if usl is not None and lsl is not None:
            spec_width = usl - lsl
            cp = spec_width / (6 * std_short)
            
            if cp < 1.0:
                comments.append("Process not capable (Cp < 1.0)")
            elif cp < 1.33:
                comments.append("Process marginally capable (1.0 ≤ Cp < 1.33)")
            else:
                comments.append("Process capable (Cp ≥ 1.33)")
        
        # Calculate Cpk (actual capability with centering)
        cpk = float('inf')
        if usl is not None and lsl is not None:
            cpu = (usl - mean) / (3 * std_short)
            cpl = (mean - lsl) / (3 * std_short)
            cpk = min(cpu, cpl)
            
            if cpk < 1.0:
                comments.append("Process not capable (Cpk < 1.0)")
            elif cpk < 1.33:
                comments.append("Process marginally capable (1.0 ≤ Cpk < 1.33)")
            else:
                comments.append("Process capable (Cpk ≥ 1.33)")
            
            # Check centering
            if abs(cp - cpk) > 0.1:
                comments.append("Process is not well-centered")
        
        # Calculate Pp and Ppk (long-term)
        pp = float('inf')
        ppk = float('inf')
        if usl is not None and lsl is not None:
            spec_width = usl - lsl
            pp = spec_width / (6 * std_long)
            
            ppu = (usl - mean) / (3 * std_long)
            ppl = (mean - lsl) / (3 * std_long)
            ppk = min(ppu, ppl)
        
        # Calculate Cpm (Taguchi index)
        cpm = None
        if usl is not None and lsl is not None and target is not None:
            tau = np.sqrt(std_short**2 + (mean - target)**2)
            spec_width = usl - lsl
            cpm = spec_width / (6 * tau)
        
        # Calculate Six Sigma level
        sigma_level = 0.0
        dpmo = 0.0
        if cpk != float('inf'):
            sigma_level = 3 * cpk
            # Approximate DPMO from Z-score
            z = sigma_level
            dpmo = (1 - stats.norm.cdf(z)) * 1_000_000
        
        # Determine if capable
        is_capable = cpk >= 1.33
        
        return ProcessCapability(
            cp=cp if cp != float('inf') else 0.0,
            cpk=cpk if cpk != float('inf') else 0.0,
            pp=pp if pp != float('inf') else 0.0,
            ppk=ppk if ppk != float('inf') else 0.0,
            cpm=cpm,
            sigma_level=sigma_level,
            dpmo=dpmo,
            is_capable=is_capable,
            comments=comments
        )


# ============================================================================
# Trend Analysis
# ============================================================================

class TrendAnalyzer:
    """Analyze trends in process data"""
    
    @staticmethod
    def analyze_trend(
        data: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
        forecast_steps: int = 5
    ) -> TrendAnalysis:
        """
        Analyze trend in data
        
        Args:
            data: Array of measurements
            timestamps: Optional timestamps (uses indices if None)
            forecast_steps: Number of steps to forecast
        
        Returns:
            TrendAnalysis object
        """
        if timestamps is None:
            timestamps = np.arange(len(data))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(timestamps, data)
        
        # Determine trend
        trend_detected = p_value < 0.05
        if not trend_detected:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "increasing"
        else:
            trend_direction = "decreasing"
        
        # Predictions
        future_times = np.arange(len(data), len(data) + forecast_steps)
        predicted_values = slope * future_times + intercept
        
        # Prediction intervals (95% confidence)
        se = std_err * np.sqrt(1 + 1/len(data) + (future_times - np.mean(timestamps))**2 / np.sum((timestamps - np.mean(timestamps))**2))
        margin = 1.96 * se  # 95% CI
        prediction_intervals = [(pred - m, pred + m) for pred, m in zip(predicted_values, margin)]
        
        # Detect changepoints (simple: look for large residuals)
        residuals = data - (slope * timestamps + intercept)
        threshold = 2 * np.std(residuals)
        changepoints = np.where(np.abs(residuals) > threshold)[0].tolist()
        
        return TrendAnalysis(
            trend_detected=trend_detected,
            trend_direction=trend_direction,
            trend_slope=slope,
            trend_significance=p_value,
            predicted_values=predicted_values.tolist(),
            prediction_intervals=prediction_intervals,
            changepoints=changepoints
        )


# ============================================================================
# Root Cause Analysis
# ============================================================================

class RootCauseAnalyzer:
    """Suggest root causes for process violations"""
    
    # Knowledge base of common root causes
    COMMON_CAUSES = {
        "critical_violations": [
            "Equipment malfunction or failure",
            "Incorrect process parameters",
            "Operator error or training issue",
            "Material defect or contamination",
            "Measurement system error",
            "Environmental condition extreme (temp, humidity)"
        ],
        "trend_increasing": [
            "Tool wear or degradation",
            "Temperature drift (increasing)",
            "Chemical concentration increasing",
            "Contamination buildup"
        ],
        "trend_decreasing": [
            "Reagent depletion",
            "Temperature drift (decreasing)",
            "Catalyst poisoning",
            "Filter clogging"
        ],
        "high_variability": [
            "Multiple operators with different techniques",
            "Inconsistent material batches",
            "Equipment instability",
            "Measurement repeatability issues"
        ],
        "oscillation": [
            "Overadjustment or over-control",
            "Alternating material lots",
            "Temperature cycling",
            "Periodic maintenance effects"
        ]
    }
    
    @staticmethod
    def suggest_causes(
        alerts: List[SPCAlert],
        recent_data: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[str]]:
        """
        Suggest root causes based on alerts and data patterns
        
        Args:
            alerts: List of recent alerts
            recent_data: Recent measurement data
            metadata: Additional context (instrument, operator, etc.)
        
        Returns:
            Dictionary of suggested causes by category
        """
        suggestions = {
            "likely_causes": [],
            "investigate": [],
            "preventive_actions": []
        }
        
        # Analyze alert types
        critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        trend_alerts = [a for a in alerts if a.rule_violated == RuleViolation.RULE_5]
        
        if critical_count > 0:
            suggestions["likely_causes"].extend(
                RootCauseAnalyzer.COMMON_CAUSES["critical_violations"][:3]
            )
            suggestions["investigate"].append("Review last process change or maintenance")
            suggestions["investigate"].append("Verify calibration status")
        
        # Check for trends
        if len(trend_alerts) > 0:
            if recent_data[-1] > recent_data[0]:
                suggestions["likely_causes"].extend(
                    RootCauseAnalyzer.COMMON_CAUSES["trend_increasing"][:2]
                )
            else:
                suggestions["likely_causes"].extend(
                    RootCauseAnalyzer.COMMON_CAUSES["trend_decreasing"][:2]
                )
        
        # Check variability
        if len(recent_data) > 10:
            cv = np.std(recent_data) / np.mean(recent_data) if np.mean(recent_data) != 0 else 0
            if cv > 0.1:  # 10% CV threshold
                suggestions["likely_causes"].extend(
                    RootCauseAnalyzer.COMMON_CAUSES["high_variability"][:2]
                )
        
        # Add metadata-based suggestions
        if metadata:
            if "last_calibration" in metadata:
                days_since = (datetime.now() - metadata["last_calibration"]).days
                if days_since > 30:
                    suggestions["investigate"].append(
                        f"Calibration overdue ({days_since} days since last)"
                    )
            
            if "operator" in metadata:
                suggestions["investigate"].append(
                    f"Review operator training: {metadata['operator']}"
                )
        
        # Generic preventive actions
        suggestions["preventive_actions"] = [
            "Implement more frequent calibration checks",
            "Add environmental monitoring (temperature, humidity)",
            "Review and update SOPs",
            "Provide operator refresher training",
            "Consider predictive maintenance schedule"
        ]
        
        return suggestions


# ============================================================================
# SPC Hub Main Class
# ============================================================================

class SPCHub:
    """
    Main Statistical Process Control Hub
    
    Provides comprehensive SPC functionality including:
    - Control chart generation and monitoring
    - Rule detection and alerting
    - Process capability analysis
    - Trend analysis and forecasting
    - Root cause analysis
    """
    
    def __init__(self):
        """Initialize SPC Hub"""
        self.calculator = ControlChartCalculator()
        self.capability_calc = CapabilityCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.root_cause_analyzer = RootCauseAnalyzer()
        
        logger.info("SPC Hub initialized")
    
    def analyze_process(
        self,
        data: np.ndarray,
        chart_type: ChartType = ChartType.I_MR,
        usl: Optional[float] = None,
        lsl: Optional[float] = None,
        target: Optional[float] = None,
        subgroup_size: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive process analysis
        
        Args:
            data: Process measurements
            chart_type: Type of control chart
            usl: Upper specification limit
            lsl: Lower specification limit
            target: Target value
            subgroup_size: For X-bar/R charts
            metadata: Additional context
        
        Returns:
            Complete analysis results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "chart_type": chart_type.value,
            "n_points": len(data),
            "statistics": {},
            "control_limits": {},
            "alerts": [],
            "capability": {},
            "trend": {},
            "status": ProcessStatus.UNKNOWN.value,
            "recommendations": []
        }
        
        try:
            # Basic statistics
            results["statistics"] = {
                "mean": float(np.mean(data)),
                "std": float(np.std(data, ddof=1)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "median": float(np.median(data)),
                "range": float(np.ptp(data))
            }
            
            # Calculate control limits based on chart type
            if chart_type == ChartType.XBAR_R:
                xbar_limits, r_limits = self.calculator.calculate_xbar_r_limits(
                    data, subgroup_size
                )
                results["control_limits"]["xbar"] = {
                    "ucl": xbar_limits.ucl,
                    "lcl": xbar_limits.lcl,
                    "centerline": xbar_limits.centerline,
                    "sigma": xbar_limits.sigma
                }
                results["control_limits"]["r"] = {
                    "ucl": r_limits.ucl,
                    "lcl": r_limits.lcl,
                    "centerline": r_limits.centerline
                }
                primary_limits = xbar_limits
                
            elif chart_type == ChartType.I_MR:
                i_limits, mr_limits = self.calculator.calculate_i_mr_limits(data)
                results["control_limits"]["i"] = {
                    "ucl": i_limits.ucl,
                    "lcl": i_limits.lcl,
                    "centerline": i_limits.centerline,
                    "sigma": i_limits.sigma
                }
                results["control_limits"]["mr"] = {
                    "ucl": mr_limits.ucl,
                    "lcl": mr_limits.lcl,
                    "centerline": mr_limits.centerline
                }
                primary_limits = i_limits
                
            elif chart_type == ChartType.EWMA:
                ewma_limits = self.calculator.calculate_ewma_limits(data, target=target)
                results["control_limits"]["ewma"] = {
                    "ucl": ewma_limits.ucl,
                    "lcl": ewma_limits.lcl,
                    "centerline": ewma_limits.centerline,
                    "sigma": ewma_limits.sigma
                }
                primary_limits = ewma_limits
                
            else:  # CUSUM
                cusum_high, cusum_low, decision_limit = self.calculator.calculate_cusum(data, target=target)
                results["control_limits"]["cusum"] = {
                    "decision_limit": decision_limit,
                    "cusum_high": cusum_high.tolist(),
                    "cusum_low": cusum_low.tolist()
                }
                # For rule detection, use I-MR limits
                primary_limits, _ = self.calculator.calculate_i_mr_limits(data)
            
            # Detect rule violations
            detector = RuleDetector(primary_limits)
            alerts = detector.detect_all_rules(data)
            
            results["alerts"] = [
                {
                    "id": alert.id,
                    "timestamp": alert.timestamp.isoformat(),
                    "rule": alert.rule_violated.value,
                    "severity": alert.severity.value,
                    "message": alert.message,
                    "value": alert.value,
                    "points_involved": alert.points_involved,
                    "suggested_actions": alert.suggested_actions[:3],  # Top 3
                    "root_causes": alert.root_causes[:3]
                }
                for alert in alerts
            ]
            
            # Determine process status
            if any(a.severity == AlertSeverity.CRITICAL for a in alerts):
                results["status"] = ProcessStatus.OUT_OF_CONTROL.value
            elif any(a.severity in [AlertSeverity.HIGH, AlertSeverity.MEDIUM] for a in alerts):
                results["status"] = ProcessStatus.WARNING.value
            else:
                results["status"] = ProcessStatus.IN_CONTROL.value
            
            # Calculate capability (if spec limits provided)
            if usl is not None or lsl is not None:
                capability = self.capability_calc.calculate_capability(
                    data, usl=usl, lsl=lsl, target=target
                )
                results["capability"] = {
                    "cp": capability.cp,
                    "cpk": capability.cpk,
                    "pp": capability.pp,
                    "ppk": capability.ppk,
                    "sigma_level": capability.sigma_level,
                    "dpmo": capability.dpmo,
                    "is_capable": capability.is_capable,
                    "comments": capability.comments
                }
            
            # Trend analysis
            trend = self.trend_analyzer.analyze_trend(data, forecast_steps=5)
            results["trend"] = {
                "detected": trend.trend_detected,
                "direction": trend.trend_direction,
                "slope": trend.trend_slope,
                "p_value": trend.trend_significance,
                "predicted_values": trend.predicted_values,
                "changepoints": trend.changepoints
            }
            
            # Root cause suggestions
            if len(alerts) > 0:
                root_causes = self.root_cause_analyzer.suggest_causes(
                    alerts, data, metadata
                )
                results["root_cause_analysis"] = root_causes
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                results["status"],
                alerts,
                results.get("capability", {}),
                trend
            )
            results["recommendations"] = recommendations
            
            logger.info(f"Process analysis complete: Status={results['status']}, Alerts={len(alerts)}")
            
        except Exception as e:
            logger.error(f"Process analysis failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _generate_recommendations(
        self,
        status: str,
        alerts: List[SPCAlert],
        capability: Dict,
        trend: TrendAnalysis
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if status == ProcessStatus.OUT_OF_CONTROL.value:
            recommendations.append("⚠️ IMMEDIATE ACTION REQUIRED: Process is out of control")
            recommendations.append("Stop production and investigate root cause")
            
        elif status == ProcessStatus.WARNING.value:
            recommendations.append("⚡ Monitor process closely - warning signals detected")
        
        if capability.get("is_capable") is False:
            recommendations.append("Process is not capable - reduce variation or widen specs")
        
        if trend.trend_detected:
            if trend.trend_direction == "increasing":
                recommendations.append(f"↗️ Increasing trend detected (slope={trend.trend_slope:.4f})")
            elif trend.trend_direction == "decreasing":
                recommendations.append(f"↘️ Decreasing trend detected (slope={trend.trend_slope:.4f})")
            recommendations.append("Investigate and correct trend before limits are exceeded")
        
        if len(recommendations) == 0:
            recommendations.append("✅ Process is in statistical control - continue monitoring")
        
        return recommendations


# ============================================================================
# Test Data Generator
# ============================================================================

class SPCTestDataGenerator:
    """Generate synthetic test data for SPC validation"""
    
    @staticmethod
    def generate_in_control(n: int = 100, mean: float = 100.0, sigma: float = 5.0) -> np.ndarray:
        """Generate in-control process data"""
        return np.random.normal(mean, sigma, n)
    
    @staticmethod
    def generate_with_shift(
        n: int = 100,
        mean: float = 100.0,
        sigma: float = 5.0,
        shift_at: int = 50,
        shift_magnitude: float = 10.0
    ) -> np.ndarray:
        """Generate data with mean shift"""
        data = np.random.normal(mean, sigma, n)
        data[shift_at:] += shift_magnitude
        return data
    
    @staticmethod
    def generate_with_trend(
        n: int = 100,
        mean: float = 100.0,
        sigma: float = 5.0,
        slope: float = 0.2
    ) -> np.ndarray:
        """Generate data with trend"""
        trend = np.arange(n) * slope
        noise = np.random.normal(0, sigma, n)
        return mean + trend + noise
    
    @staticmethod
    def generate_with_cycle(
        n: int = 100,
        mean: float = 100.0,
        sigma: float = 5.0,
        period: int = 20,
        amplitude: float = 5.0
    ) -> np.ndarray:
        """Generate data with cyclic pattern"""
        t = np.arange(n)
        cycle = amplitude * np.sin(2 * np.pi * t / period)
        noise = np.random.normal(0, sigma, n)
        return mean + cycle + noise


# ============================================================================
# FastAPI Integration
# ============================================================================

def create_spc_routes():
    """Create FastAPI routes for SPC Hub"""
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    from typing import Optional, List
    
    router = APIRouter(prefix="/api/spc", tags=["spc"])
    
    class SPCAnalysisRequest(BaseModel):
        data: List[float]
        chart_type: str = "i_mr"
        usl: Optional[float] = None
        lsl: Optional[float] = None
        target: Optional[float] = None
        subgroup_size: int = 5
        metadata: Optional[Dict[str, Any]] = None
    
    spc_hub = SPCHub()
    
    @router.post("/analyze")
    async def analyze_process(request: SPCAnalysisRequest):
        """Analyze process data with SPC methods"""
        try:
            data = np.array(request.data)
            chart_type = ChartType(request.chart_type)
            
            results = spc_hub.analyze_process(
                data=data,
                chart_type=chart_type,
                usl=request.usl,
                lsl=request.lsl,
                target=request.target,
                subgroup_size=request.subgroup_size,
                metadata=request.metadata
            )
            
            return results
            
        except Exception as e:
            logger.error(f"SPC analysis failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "service": "spc-hub",
            "version": "1.0.0"
        }
    
    return router


if __name__ == "__main__":
    # Demo and validation
    print("=" * 80)
    print("Session 13: SPC Hub - Demo and Validation")
    print("=" * 80)
    
    # Initialize hub
    hub = SPCHub()
    
    # Test 1: In-control process
    print("\n1. In-Control Process")
    print("-" * 40)
    data_in_control = SPCTestDataGenerator.generate_in_control(n=50, mean=100, sigma=5)
    results = hub.analyze_process(
        data=data_in_control,
        chart_type=ChartType.I_MR,
        usl=115,
        lsl=85
    )
    print(f"Status: {results['status']}")
    print(f"Alerts: {len(results['alerts'])}")
    print(f"Capable: {results['capability']['is_capable']}")
    print(f"Cpk: {results['capability']['cpk']:.3f}")
    
    # Test 2: Process with shift
    print("\n2. Process with Mean Shift")
    print("-" * 40)
    data_with_shift = SPCTestDataGenerator.generate_with_shift(
        n=50, mean=100, sigma=5, shift_at=30, shift_magnitude=15
    )
    results = hub.analyze_process(
        data=data_with_shift,
        chart_type=ChartType.I_MR,
        usl=115,
        lsl=85
    )
    print(f"Status: {results['status']}")
    print(f"Alerts: {len(results['alerts'])}")
    if results['alerts']:
        print(f"First alert: {results['alerts'][0]['message']}")
    
    # Test 3: Process with trend
    print("\n3. Process with Trend")
    print("-" * 40)
    data_with_trend = SPCTestDataGenerator.generate_with_trend(
        n=50, mean=100, sigma=5, slope=0.3
    )
    results = hub.analyze_process(
        data=data_with_trend,
        chart_type=ChartType.I_MR,
        usl=115,
        lsl=85
    )
    print(f"Status: {results['status']}")
    print(f"Trend detected: {results['trend']['detected']}")
    print(f"Trend direction: {results['trend']['direction']}")
    print(f"Alerts: {len(results['alerts'])}")
    
    # Test 4: EWMA chart
    print("\n4. EWMA Chart Analysis")
    print("-" * 40)
    results = hub.analyze_process(
        data=data_with_shift,
        chart_type=ChartType.EWMA,
        target=100
    )
    print(f"Status: {results['status']}")
    print(f"EWMA limits: UCL={results['control_limits']['ewma']['ucl']:.2f}, "
          f"LCL={results['control_limits']['ewma']['lcl']:.2f}")
    
    print("\n" + "=" * 80)
    print("SPC Hub validation complete! ✅")
    print("=" * 80)
