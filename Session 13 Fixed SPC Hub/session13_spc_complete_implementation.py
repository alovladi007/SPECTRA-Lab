"""
Session 13: SPC Hub - Complete Backend Implementation
Semiconductor Lab Platform

Statistical Process Control (SPC) analysis suite:
- X-bar/R, EWMA, CUSUM control charts
- Cp/Cpk process capability indices
- Western Electric and Nelson rule detection
- Alert generation and triage
- Root cause analysis suggestions

Author: Semiconductor Lab Platform Team
Version: 1.0.0
Date: October 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations
# ============================================================================

class ChartType(Enum):
    """Types of control charts"""
    XBAR_R = "xbar_r"  # X-bar and R charts (variables data)
    EWMA = "ewma"      # Exponentially Weighted Moving Average
    CUSUM = "cusum"    # Cumulative Sum
    P = "p"            # Proportion (attributes data)
    C = "c"            # Count (attributes data)


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RuleViolation(Enum):
    """Western Electric and Nelson rules"""
    # Western Electric Rules
    RULE_1 = "1_point_beyond_3sigma"
    RULE_2 = "2_of_3_beyond_2sigma"
    RULE_3 = "4_of_5_beyond_1sigma"
    RULE_4 = "8_consecutive_same_side"
    
    # Nelson Rules (additional)
    RULE_5 = "6_points_trending"
    RULE_6 = "14_points_alternating"
    RULE_7 = "15_points_within_1sigma"
    RULE_8 = "8_points_beyond_1sigma"


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class DataPoint:
    """Single data point in time series"""
    timestamp: datetime
    value: float
    subgroup: str
    run_id: str
    metadata: Dict[str, Any] = None


@dataclass
class ControlLimits:
    """Control limits for a metric"""
    ucl: float  # Upper Control Limit
    lcl: float  # Lower Control Limit
    cl: float   # Center Line
    sigma: float
    sample_size: int
    computed_from: List[str]  # Run IDs


@dataclass
class SPCAlert:
    """SPC alert"""
    id: str
    metric: str
    severity: AlertSeverity
    rule_violated: RuleViolation
    message: str
    timestamp: datetime
    data_points: List[DataPoint]
    suggested_actions: List[str]


@dataclass
class ProcessCapability:
    """Process capability indices"""
    cp: float   # Process capability
    cpk: float  # Process capability (accounting for centering)
    cpl: float  # Lower capability index
    cpu: float  # Upper capability index
    sigma: float
    mean: float
    lsl: Optional[float]  # Lower Specification Limit
    usl: Optional[float]  # Upper Specification Limit


# ============================================================================
# X-bar and R Chart Analysis
# ============================================================================

class XbarRChart:
    """
    X-bar and R control charts for variables data.
    
    Monitors process mean (X-bar) and variability (R).
    """
    
    # Constants for control limits (from statistical tables)
    D3_TABLE = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
    D4_TABLE = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}
    A2_TABLE = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
    
    def __init__(self, subgroup_size: int = 5):
        """
        Initialize X-bar/R chart.
        
        Args:
            subgroup_size: Number of measurements per subgroup (typically 2-10)
        """
        if subgroup_size < 2 or subgroup_size > 10:
            raise ValueError("Subgroup size must be between 2 and 10")
        
        self.subgroup_size = subgroup_size
        self.D3 = self.D3_TABLE[subgroup_size]
        self.D4 = self.D4_TABLE[subgroup_size]
        self.A2 = self.A2_TABLE[subgroup_size]
    
    def compute_control_limits(
        self,
        data: List[DataPoint],
        sigma_level: float = 3.0
    ) -> Tuple[ControlLimits, ControlLimits]:
        """
        Compute X-bar and R control limits.
        
        Args:
            data: Historical data points
            sigma_level: Number of sigma for control limits (typically 3)
        
        Returns:
            Tuple of (xbar_limits, r_limits)
        """
        # Group data by subgroup
        subgroups = {}
        for point in data:
            if point.subgroup not in subgroups:
                subgroups[point.subgroup] = []
            subgroups[point.subgroup].append(point.value)
        
        # Calculate X-bar and R for each subgroup
        xbar_values = []
        r_values = []
        run_ids = []
        
        for subgroup_id, values in subgroups.items():
            if len(values) == self.subgroup_size:
                xbar_values.append(np.mean(values))
                r_values.append(np.max(values) - np.min(values))
                run_ids.extend([p.run_id for p in data if p.subgroup == subgroup_id])
        
        # Grand average and average range
        xbar_bar = np.mean(xbar_values)
        r_bar = np.mean(r_values)
        
        # X-bar chart limits
        xbar_ucl = xbar_bar + self.A2 * r_bar
        xbar_lcl = xbar_bar - self.A2 * r_bar
        
        xbar_limits = ControlLimits(
            ucl=xbar_ucl,
            lcl=xbar_lcl,
            cl=xbar_bar,
            sigma=r_bar / 1.128,  # d2 constant for subgroup size
            sample_size=len(xbar_values),
            computed_from=list(set(run_ids))
        )
        
        # R chart limits
        r_ucl = self.D4 * r_bar
        r_lcl = self.D3 * r_bar
        
        r_limits = ControlLimits(
            ucl=r_ucl,
            lcl=r_lcl,
            cl=r_bar,
            sigma=r_bar,
            sample_size=len(r_values),
            computed_from=list(set(run_ids))
        )
        
        return xbar_limits, r_limits
    
    def check_rules(
        self,
        data: List[DataPoint],
        limits: ControlLimits
    ) -> List[SPCAlert]:
        """
        Check Western Electric and Nelson rules for violations.
        
        Args:
            data: Data points to check
            limits: Control limits
        
        Returns:
            List of alerts
        """
        alerts = []
        values = [p.value for p in data]
        
        # Rule 1: One point beyond 3σ
        alerts.extend(self._check_rule_1(data, limits))
        
        # Rule 2: 2 of 3 consecutive points beyond 2σ
        alerts.extend(self._check_rule_2(data, limits))
        
        # Rule 3: 4 of 5 consecutive points beyond 1σ
        alerts.extend(self._check_rule_3(data, limits))
        
        # Rule 4: 8 consecutive points on same side of centerline
        alerts.extend(self._check_rule_4(data, limits))
        
        # Rule 5: 6 points in a row trending up or down
        alerts.extend(self._check_rule_5(data, limits))
        
        # Rule 6: 14 points alternating up and down
        alerts.extend(self._check_rule_6(data, limits))
        
        # Rule 7: 15 points in a row within 1σ
        alerts.extend(self._check_rule_7(data, limits))
        
        # Rule 8: 8 points in a row beyond 1σ from centerline
        alerts.extend(self._check_rule_8(data, limits))
        
        return alerts
    
    def _check_rule_1(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for points beyond 3σ"""
        alerts = []
        for point in data:
            if point.value > limits.ucl or point.value < limits.lcl:
                alert = SPCAlert(
                    id=f"alert_{point.run_id}_{int(point.timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.CRITICAL,
                    rule_violated=RuleViolation.RULE_1,
                    message=f"Point beyond 3σ control limits (value: {point.value:.3f})",
                    timestamp=point.timestamp,
                    data_points=[point],
                    suggested_actions=[
                        "Check instrument calibration",
                        "Verify measurement procedure",
                        "Inspect for special causes",
                        "Review operator training"
                    ]
                )
                alerts.append(alert)
        return alerts
    
    def _check_rule_2(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for 2 of 3 consecutive points beyond 2σ"""
        alerts = []
        sigma_2_upper = limits.cl + 2 * limits.sigma
        sigma_2_lower = limits.cl - 2 * limits.sigma
        
        for i in range(len(data) - 2):
            window = data[i:i+3]
            beyond_count = sum(1 for p in window if p.value > sigma_2_upper or p.value < sigma_2_lower)
            
            if beyond_count >= 2:
                alert = SPCAlert(
                    id=f"alert_rule2_{int(window[0].timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.HIGH,
                    rule_violated=RuleViolation.RULE_2,
                    message="2 of 3 consecutive points beyond 2σ",
                    timestamp=window[-1].timestamp,
                    data_points=window,
                    suggested_actions=[
                        "Investigate process shift",
                        "Check for material batch change",
                        "Review recent process changes"
                    ]
                )
                alerts.append(alert)
                break  # Only report once per sequence
        
        return alerts
    
    def _check_rule_3(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for 4 of 5 consecutive points beyond 1σ"""
        alerts = []
        sigma_1_upper = limits.cl + limits.sigma
        sigma_1_lower = limits.cl - limits.sigma
        
        for i in range(len(data) - 4):
            window = data[i:i+5]
            beyond_count = sum(1 for p in window if p.value > sigma_1_upper or p.value < sigma_1_lower)
            
            if beyond_count >= 4:
                alert = SPCAlert(
                    id=f"alert_rule3_{int(window[0].timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.MEDIUM,
                    rule_violated=RuleViolation.RULE_3,
                    message="4 of 5 consecutive points beyond 1σ",
                    timestamp=window[-1].timestamp,
                    data_points=window,
                    suggested_actions=[
                        "Monitor for continuing trend",
                        "Check process stability",
                        "Review control plan"
                    ]
                )
                alerts.append(alert)
                break
        
        return alerts
    
    def _check_rule_4(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for 8 consecutive points on same side of centerline"""
        alerts = []
        
        for i in range(len(data) - 7):
            window = data[i:i+8]
            all_above = all(p.value > limits.cl for p in window)
            all_below = all(p.value < limits.cl for p in window)
            
            if all_above or all_below:
                side = "above" if all_above else "below"
                alert = SPCAlert(
                    id=f"alert_rule4_{int(window[0].timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.MEDIUM,
                    rule_violated=RuleViolation.RULE_4,
                    message=f"8 consecutive points {side} centerline",
                    timestamp=window[-1].timestamp,
                    data_points=window,
                    suggested_actions=[
                        "Process may have shifted",
                        "Recalculate control limits if shift is permanent",
                        "Investigate root cause of shift"
                    ]
                )
                alerts.append(alert)
                break
        
        return alerts
    
    def _check_rule_5(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for 6 points in a row trending"""
        alerts = []
        
        for i in range(len(data) - 5):
            window = data[i:i+6]
            values = [p.value for p in window]
            
            # Check for monotonic increase or decrease
            increasing = all(values[j] < values[j+1] for j in range(5))
            decreasing = all(values[j] > values[j+1] for j in range(5))
            
            if increasing or decreasing:
                direction = "increasing" if increasing else "decreasing"
                alert = SPCAlert(
                    id=f"alert_rule5_{int(window[0].timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.MEDIUM,
                    rule_violated=RuleViolation.RULE_5,
                    message=f"6 points trending {direction}",
                    timestamp=window[-1].timestamp,
                    data_points=window,
                    suggested_actions=[
                        "Check for tool wear or drift",
                        "Verify environmental conditions",
                        "Review PM schedule"
                    ]
                )
                alerts.append(alert)
                break
        
        return alerts
    
    def _check_rule_6(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for 14 points alternating up and down"""
        alerts = []
        
        if len(data) < 14:
            return alerts
        
        for i in range(len(data) - 13):
            window = data[i:i+14]
            values = [p.value for p in window]
            
            # Check alternating pattern
            alternating = all(
                (values[j] < values[j+1] and values[j+1] > values[j+2]) or
                (values[j] > values[j+1] and values[j+1] < values[j+2])
                for j in range(12)
            )
            
            if alternating:
                alert = SPCAlert(
                    id=f"alert_rule6_{int(window[0].timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.LOW,
                    rule_violated=RuleViolation.RULE_6,
                    message="14 points alternating up and down",
                    timestamp=window[-1].timestamp,
                    data_points=window,
                    suggested_actions=[
                        "Check for systematic variation",
                        "Review measurement system",
                        "Investigate operator technique"
                    ]
                )
                alerts.append(alert)
                break
        
        return alerts
    
    def _check_rule_7(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for 15 points within 1σ"""
        alerts = []
        sigma_1_upper = limits.cl + limits.sigma
        sigma_1_lower = limits.cl - limits.sigma
        
        if len(data) < 15:
            return alerts
        
        for i in range(len(data) - 14):
            window = data[i:i+15]
            all_within = all(sigma_1_lower < p.value < sigma_1_upper for p in window)
            
            if all_within:
                alert = SPCAlert(
                    id=f"alert_rule7_{int(window[0].timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.LOW,
                    rule_violated=RuleViolation.RULE_7,
                    message="15 points within 1σ (stratification)",
                    timestamp=window[-1].timestamp,
                    data_points=window,
                    suggested_actions=[
                        "Process may be over-controlled",
                        "Check for data manipulation",
                        "Verify measurement system variation"
                    ]
                )
                alerts.append(alert)
                break
        
        return alerts
    
    def _check_rule_8(self, data: List[DataPoint], limits: ControlLimits) -> List[SPCAlert]:
        """Check for 8 points beyond 1σ"""
        alerts = []
        sigma_1_upper = limits.cl + limits.sigma
        sigma_1_lower = limits.cl - limits.sigma
        
        for i in range(len(data) - 7):
            window = data[i:i+8]
            beyond_count = sum(1 for p in window if p.value > sigma_1_upper or p.value < sigma_1_lower)
            
            if beyond_count >= 8:
                alert = SPCAlert(
                    id=f"alert_rule8_{int(window[0].timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.MEDIUM,
                    rule_violated=RuleViolation.RULE_8,
                    message="8 points beyond 1σ (mixture)",
                    timestamp=window[-1].timestamp,
                    data_points=window,
                    suggested_actions=[
                        "Check for multiple populations",
                        "Review subgrouping strategy",
                        "Investigate process consistency"
                    ]
                )
                alerts.append(alert)
                break
        
        return alerts


# ============================================================================
# EWMA (Exponentially Weighted Moving Average)
# ============================================================================

class EWMAChart:
    """
    EWMA control chart for detecting small process shifts.
    
    More sensitive to small shifts than X-bar charts.
    """
    
    def __init__(self, lambda_: float = 0.2):
        """
        Initialize EWMA chart.
        
        Args:
            lambda_: Weighting factor (typically 0.05 to 0.3)
        """
        if not 0 < lambda_ <= 1:
            raise ValueError("Lambda must be between 0 and 1")
        
        self.lambda_ = lambda_
    
    def compute_control_limits(
        self,
        data: List[DataPoint],
        sigma_level: float = 3.0
    ) -> ControlLimits:
        """
        Compute EWMA control limits.
        
        Args:
            data: Historical data points
            sigma_level: Number of sigma for control limits
        
        Returns:
            Control limits
        """
        values = [p.value for p in data]
        mean = np.mean(values)
        sigma = np.std(values, ddof=1)
        
        # EWMA control limits
        # σ_EWMA = σ * sqrt(λ / (2 - λ))
        sigma_ewma = sigma * np.sqrt(self.lambda_ / (2 - self.lambda_))
        
        ucl = mean + sigma_level * sigma_ewma
        lcl = mean - sigma_level * sigma_ewma
        
        return ControlLimits(
            ucl=ucl,
            lcl=lcl,
            cl=mean,
            sigma=sigma_ewma,
            sample_size=len(data),
            computed_from=[p.run_id for p in data]
        )
    
    def calculate_ewma(self, data: List[DataPoint], initial_ewma: Optional[float] = None) -> List[float]:
        """
        Calculate EWMA values.
        
        Args:
            data: Data points
            initial_ewma: Initial EWMA value (defaults to mean of first few points)
        
        Returns:
            List of EWMA values
        """
        values = [p.value for p in data]
        
        if initial_ewma is None:
            initial_ewma = np.mean(values[:5]) if len(values) >= 5 else values[0]
        
        ewma_values = [initial_ewma]
        for value in values:
            ewma = self.lambda_ * value + (1 - self.lambda_) * ewma_values[-1]
            ewma_values.append(ewma)
        
        return ewma_values[1:]  # Remove initial value
    
    def check_violations(
        self,
        data: List[DataPoint],
        limits: ControlLimits
    ) -> List[SPCAlert]:
        """
        Check for EWMA violations.
        
        Args:
            data: Data points
            limits: Control limits
        
        Returns:
            List of alerts
        """
        alerts = []
        ewma_values = self.calculate_ewma(data)
        
        for i, (point, ewma_value) in enumerate(zip(data, ewma_values)):
            if ewma_value > limits.ucl or ewma_value < limits.lcl:
                alert = SPCAlert(
                    id=f"alert_ewma_{point.run_id}_{int(point.timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.HIGH,
                    rule_violated=RuleViolation.RULE_1,
                    message=f"EWMA value beyond control limits (EWMA: {ewma_value:.3f})",
                    timestamp=point.timestamp,
                    data_points=[point],
                    suggested_actions=[
                        "Small process shift detected",
                        "Investigate recent process changes",
                        "Review preceding measurements"
                    ]
                )
                alerts.append(alert)
        
        return alerts


# ============================================================================
# CUSUM (Cumulative Sum)
# ============================================================================

class CUSUMChart:
    """
    CUSUM control chart for detecting sustained shifts.
    
    Accumulates deviations from target to detect persistent changes.
    """
    
    def __init__(self, k: float = 0.5, h: float = 5.0):
        """
        Initialize CUSUM chart.
        
        Args:
            k: Reference value (typically 0.5σ)
            h: Decision interval (typically 4-5σ)
        """
        self.k = k
        self.h = h
    
    def compute_control_limits(
        self,
        data: List[DataPoint]
    ) -> Tuple[float, float, float]:
        """
        Compute CUSUM parameters.
        
        Args:
            data: Historical data points
        
        Returns:
            Tuple of (target, sigma, h_limit)
        """
        values = [p.value for p in data]
        target = np.mean(values)
        sigma = np.std(values, ddof=1)
        h_limit = self.h * sigma
        
        return target, sigma, h_limit
    
    def calculate_cusum(
        self,
        data: List[DataPoint],
        target: float,
        sigma: float
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate CUSUM values (upper and lower).
        
        Args:
            data: Data points
            target: Target value
            sigma: Process standard deviation
        
        Returns:
            Tuple of (cusum_high, cusum_low)
        """
        values = [p.value for p in data]
        k_value = self.k * sigma
        
        cusum_high = [0]
        cusum_low = [0]
        
        for value in values:
            # Upper CUSUM
            sh = max(0, cusum_high[-1] + (value - target) - k_value)
            cusum_high.append(sh)
            
            # Lower CUSUM
            sl = max(0, cusum_low[-1] + (target - value) - k_value)
            cusum_low.append(sl)
        
        return cusum_high[1:], cusum_low[1:]
    
    def check_violations(
        self,
        data: List[DataPoint],
        target: float,
        sigma: float,
        h_limit: float
    ) -> List[SPCAlert]:
        """
        Check for CUSUM violations.
        
        Args:
            data: Data points
            target: Target value
            sigma: Process standard deviation
            h_limit: Decision interval
        
        Returns:
            List of alerts
        """
        alerts = []
        cusum_high, cusum_low = self.calculate_cusum(data, target, sigma)
        
        for i, (point, ch, cl) in enumerate(zip(data, cusum_high, cusum_low)):
            if ch > h_limit:
                alert = SPCAlert(
                    id=f"alert_cusum_high_{point.run_id}_{int(point.timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.HIGH,
                    rule_violated=RuleViolation.RULE_1,
                    message=f"CUSUM high exceeded decision interval (CUSUM+: {ch:.3f})",
                    timestamp=point.timestamp,
                    data_points=[point],
                    suggested_actions=[
                        "Sustained upward shift detected",
                        "Process mean has increased",
                        "Investigate cause of persistent deviation"
                    ]
                )
                alerts.append(alert)
            
            if cl > h_limit:
                alert = SPCAlert(
                    id=f"alert_cusum_low_{point.run_id}_{int(point.timestamp.timestamp())}",
                    metric="process",
                    severity=AlertSeverity.HIGH,
                    rule_violated=RuleViolation.RULE_1,
                    message=f"CUSUM low exceeded decision interval (CUSUM-: {cl:.3f})",
                    timestamp=point.timestamp,
                    data_points=[point],
                    suggested_actions=[
                        "Sustained downward shift detected",
                        "Process mean has decreased",
                        "Investigate cause of persistent deviation"
                    ]
                )
                alerts.append(alert)
        
        return alerts


# ============================================================================
# Process Capability Analysis
# ============================================================================

class CapabilityAnalysis:
    """
    Process capability (Cp, Cpk) analysis.
    
    Measures how well process fits within specification limits.
    """
    
    @staticmethod
    def calculate_capability(
        data: List[float],
        lsl: Optional[float] = None,
        usl: Optional[float] = None
    ) -> ProcessCapability:
        """
        Calculate process capability indices.
        
        Args:
            data: Process data
            lsl: Lower Specification Limit
            usl: Upper Specification Limit
        
        Returns:
            Process capability indices
        """
        mean = np.mean(data)
        sigma = np.std(data, ddof=1)
        
        # Cp: Potential capability (ignores centering)
        if lsl is not None and usl is not None:
            cp = (usl - lsl) / (6 * sigma)
        else:
            cp = np.nan
        
        # Cpk: Actual capability (accounts for centering)
        cpu = (usl - mean) / (3 * sigma) if usl is not None else np.nan
        cpl = (mean - lsl) / (3 * sigma) if lsl is not None else np.nan
        
        if not np.isnan(cpu) and not np.isnan(cpl):
            cpk = min(cpu, cpl)
        elif not np.isnan(cpu):
            cpk = cpu
        elif not np.isnan(cpl):
            cpk = cpl
        else:
            cpk = np.nan
        
        return ProcessCapability(
            cp=cp,
            cpk=cpk,
            cpu=cpu,
            cpl=cpl,
            sigma=sigma,
            mean=mean,
            lsl=lsl,
            usl=usl
        )
    
    @staticmethod
    def interpret_capability(capability: ProcessCapability) -> str:
        """
        Interpret capability indices.
        
        Args:
            capability: Process capability
        
        Returns:
            Interpretation string
        """
        if np.isnan(capability.cpk):
            return "Cannot assess - no specification limits"
        
        if capability.cpk >= 2.0:
            return "Excellent (6σ capable)"
        elif capability.cpk >= 1.67:
            return "Very Good (5σ capable)"
        elif capability.cpk >= 1.33:
            return "Adequate (4σ capable)"
        elif capability.cpk >= 1.0:
            return "Marginal (3σ capable, some rejects expected)"
        else:
            return "Poor (High defect rate, process not capable)"


# ============================================================================
# SPC Manager - Main Orchestration
# ============================================================================

class SPCManager:
    """
    Main SPC analysis manager.
    
    Orchestrates control chart analysis, alert generation, and reporting.
    """
    
    def __init__(self):
        self.xbar_r = XbarRChart(subgroup_size=5)
        self.ewma = EWMAChart(lambda_=0.2)
        self.cusum = CUSUMChart(k=0.5, h=5.0)
    
    def analyze_metric(
        self,
        metric_name: str,
        data: List[DataPoint],
        chart_type: ChartType = ChartType.XBAR_R,
        lsl: Optional[float] = None,
        usl: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive SPC analysis for a metric.
        
        Args:
            metric_name: Name of metric being analyzed
            data: Historical data points
            chart_type: Type of control chart
            lsl: Lower specification limit (for Cp/Cpk)
            usl: Upper specification limit (for Cp/Cpk)
        
        Returns:
            Complete SPC analysis results
        """
        logger.info(f"Starting SPC analysis for metric: {metric_name}")
        
        if len(data) < 20:
            logger.warning(f"Insufficient data for {metric_name}: {len(data)} points (minimum 20)")
            return {
                "error": "Insufficient data",
                "message": "At least 20 data points required for SPC analysis"
            }
        
        results = {
            "metric": metric_name,
            "chart_type": chart_type.value,
            "data_count": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
        # Compute control limits and check rules
        if chart_type == ChartType.XBAR_R:
            xbar_limits, r_limits = self.xbar_r.compute_control_limits(data)
            alerts = self.xbar_r.check_rules(data, xbar_limits)
            
            results["xbar_limits"] = {
                "ucl": xbar_limits.ucl,
                "lcl": xbar_limits.lcl,
                "cl": xbar_limits.cl,
                "sigma": xbar_limits.sigma
            }
            results["r_limits"] = {
                "ucl": r_limits.ucl,
                "lcl": r_limits.lcl,
                "cl": r_limits.cl
            }
        
        elif chart_type == ChartType.EWMA:
            limits = self.ewma.compute_control_limits(data)
            ewma_values = self.ewma.calculate_ewma(data)
            alerts = self.ewma.check_violations(data, limits)
            
            results["control_limits"] = {
                "ucl": limits.ucl,
                "lcl": limits.lcl,
                "cl": limits.cl,
                "sigma": limits.sigma
            }
            results["ewma_values"] = ewma_values
        
        elif chart_type == ChartType.CUSUM:
            target, sigma, h_limit = self.cusum.compute_control_limits(data)
            cusum_high, cusum_low = self.cusum.calculate_cusum(data, target, sigma)
            alerts = self.cusum.check_violations(data, target, sigma, h_limit)
            
            results["control_limits"] = {
                "target": target,
                "sigma": sigma,
                "h_limit": h_limit
            }
            results["cusum_high"] = cusum_high
            results["cusum_low"] = cusum_low
        
        # Process capability analysis
        values = [p.value for p in data]
        capability = CapabilityAnalysis.calculate_capability(values, lsl, usl)
        
        results["capability"] = {
            "cp": capability.cp if not np.isnan(capability.cp) else None,
            "cpk": capability.cpk if not np.isnan(capability.cpk) else None,
            "cpu": capability.cpu if not np.isnan(capability.cpu) else None,
            "cpl": capability.cpl if not np.isnan(capability.cpl) else None,
            "mean": capability.mean,
            "sigma": capability.sigma,
            "interpretation": CapabilityAnalysis.interpret_capability(capability)
        }
        
        # Alerts
        results["alerts"] = [
            {
                "id": alert.id,
                "severity": alert.severity.value,
                "rule": alert.rule_violated.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "suggested_actions": alert.suggested_actions
            }
            for alert in alerts
        ]
        
        # Summary statistics
        results["statistics"] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)),
            "range": float(np.ptp(values))
        }
        
        logger.info(f"SPC analysis complete: {len(alerts)} alerts generated")
        
        return results


# ============================================================================
# Synthetic Data Generators (for testing)
# ============================================================================

def generate_in_control_data(
    n_points: int = 50,
    mean: float = 100.0,
    sigma: float = 2.0,
    subgroup_size: int = 5
) -> List[DataPoint]:
    """Generate in-control process data"""
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(n_points):
        subgroup = f"subgroup_{i // subgroup_size}"
        value = np.random.normal(mean, sigma)
        
        point = DataPoint(
            timestamp=base_time + timedelta(hours=i),
            value=value,
            subgroup=subgroup,
            run_id=f"run_{i}",
            metadata={"operator": "tech_1", "instrument": "tool_A"}
        )
        data.append(point)
    
    return data


def generate_shift_data(
    n_points: int = 50,
    mean: float = 100.0,
    sigma: float = 2.0,
    shift_point: int = 25,
    shift_amount: float = 3.0
) -> List[DataPoint]:
    """Generate data with mean shift"""
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(n_points):
        subgroup = f"subgroup_{i // 5}"
        current_mean = mean if i < shift_point else mean + shift_amount
        value = np.random.normal(current_mean, sigma)
        
        point = DataPoint(
            timestamp=base_time + timedelta(hours=i),
            value=value,
            subgroup=subgroup,
            run_id=f"run_{i}",
            metadata={"operator": "tech_1"}
        )
        data.append(point)
    
    return data


def generate_trend_data(
    n_points: int = 50,
    mean: float = 100.0,
    sigma: float = 2.0,
    drift_rate: float = 0.1
) -> List[DataPoint]:
    """Generate data with linear trend"""
    data = []
    base_time = datetime.now() - timedelta(days=30)
    
    for i in range(n_points):
        subgroup = f"subgroup_{i // 5}"
        current_mean = mean + drift_rate * i
        value = np.random.normal(current_mean, sigma)
        
        point = DataPoint(
            timestamp=base_time + timedelta(hours=i),
            value=value,
            subgroup=subgroup,
            run_id=f"run_{i}",
            metadata={"drift": "tool_wear"}
        )
        data.append(point)
    
    return data


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example: Analyze in-control process
    print("=" * 80)
    print("SPC Analysis Example - In Control Process")
    print("=" * 80)
    
    manager = SPCManager()
    
    # Generate synthetic data
    data = generate_in_control_data(n_points=50, mean=100.0, sigma=2.0)
    
    # Run analysis
    results = manager.analyze_metric(
        metric_name="sheet_resistance",
        data=data,
        chart_type=ChartType.XBAR_R,
        lsl=94.0,
        usl=106.0
    )
    
    print(f"\nMetric: {results['metric']}")
    print(f"Data points: {results['data_count']}")
    print(f"\nControl Limits (X-bar):")
    print(f"  UCL: {results['xbar_limits']['ucl']:.2f}")
    print(f"  CL:  {results['xbar_limits']['cl']:.2f}")
    print(f"  LCL: {results['xbar_limits']['lcl']:.2f}")
    print(f"\nProcess Capability:")
    print(f"  Cp:  {results['capability']['cp']:.3f}")
    print(f"  Cpk: {results['capability']['cpk']:.3f}")
    print(f"  {results['capability']['interpretation']}")
    print(f"\nAlerts: {len(results['alerts'])}")
    
    # Example: Analyze process with shift
    print("\n" + "=" * 80)
    print("SPC Analysis Example - Process with Shift")
    print("=" * 80)
    
    shift_data = generate_shift_data(
        n_points=50,
        mean=100.0,
        sigma=2.0,
        shift_point=25,
        shift_amount=4.0
    )
    
    shift_results = manager.analyze_metric(
        metric_name="thickness",
        data=shift_data,
        chart_type=ChartType.XBAR_R,
        lsl=94.0,
        usl=106.0
    )
    
    print(f"\nMetric: {shift_results['metric']}")
    print(f"Alerts: {len(shift_results['alerts'])}")
    for alert in shift_results['alerts']:
        print(f"\n  {alert['severity'].upper()}: {alert['message']}")
        print(f"  Rule: {alert['rule']}")
        print(f"  Actions: {', '.join(alert['suggested_actions'][:2])}")
    
    print("\n" + "=" * 80)
    print("SPC Module - Ready for Production")
    print("=" * 80)
