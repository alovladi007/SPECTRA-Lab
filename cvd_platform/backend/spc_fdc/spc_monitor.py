"""
Statistical Process Control (SPC) and Fault Detection & Classification (FDC)
Implements comprehensive quality monitoring for CVD processes:
- Control charts (X-bar, R, EWMA, CUSUM)
- Process capability indices (Cp, Cpk, Pp, Ppk)
- Fault detection and classification
- Alarm management
- AI/ML-enhanced anomaly detection
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ChartType(Enum):
    """Control chart types"""
    XBAR = "xbar"  # Mean chart
    R = "r"  # Range chart
    S = "s"  # Standard deviation chart
    EWMA = "ewma"  # Exponentially weighted moving average
    CUSUM = "cusum"  # Cumulative sum
    INDIVIDUALS = "individuals"  # Individual measurements (I chart)


class AlarmLevel(Enum):
    """Alarm severity levels"""
    INFO = "info"
    WARNING = "warning"
    ALARM = "alarm"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of control limit violations"""
    ABOVE_UCL = "above_ucl"
    BELOW_LCL = "below_lcl"
    RUN_ABOVE_CL = "run_above_cl"  # 7+ consecutive points above centerline
    RUN_BELOW_CL = "run_below_cl"
    TREND_UP = "trend_up"  # 6+ consecutive increasing points
    TREND_DOWN = "trend_down"
    TWO_OF_THREE_ZONE_A = "2_of_3_zone_a"  # 2 of 3 points in Zone A
    FOUR_OF_FIVE_ZONE_B = "4_of_5_zone_b"  # 4 of 5 points in Zone B


@dataclass
class ControlLimits:
    """Control chart limits"""
    ucl: float  # Upper control limit
    lcl: float  # Lower control limit
    cl: float  # Center line (target or mean)
    usl: Optional[float] = None  # Upper specification limit
    lsl: Optional[float] = None  # Lower specification limit
    ucl_warning: Optional[float] = None  # Warning limit (2-sigma)
    lcl_warning: Optional[float] = None


@dataclass
class SPCViolation:
    """SPC rule violation"""
    violation_type: ViolationType
    value: float
    timestamp: datetime
    chart_id: str
    parameter_name: str
    severity: AlarmLevel
    message: str
    recommended_action: str


@dataclass
class ProcessCapability:
    """Process capability metrics"""
    cp: float  # Capability (potential)
    cpk: float  # Capability (actual, accounting for centering)
    pp: float  # Performance (potential)
    ppk: float  # Performance (actual)
    sigma_level: float  # Process sigma level
    dpmo: float  # Defects per million opportunities


class ControlChart:
    """
    Base class for control charts.
    Implements common functionality for all chart types.
    """

    def __init__(self,
                 chart_id: str,
                 parameter_name: str,
                 chart_type: ChartType,
                 target: Optional[float] = None,
                 spec_limits: Optional[Tuple[float, float]] = None,
                 subgroup_size: int = 1,
                 max_points: int = 1000):
        """
        Args:
            chart_id: Unique chart identifier
            parameter_name: Parameter being monitored (e.g., "thickness")
            chart_type: Type of control chart
            target: Target value
            spec_limits: (LSL, USL) specification limits
            subgroup_size: Subgroup size for X-bar/R charts
            max_points: Maximum number of points to retain
        """
        self.chart_id = chart_id
        self.parameter_name = parameter_name
        self.chart_type = chart_type
        self.target = target
        self.subgroup_size = subgroup_size
        self.max_points = max_points

        # Specification limits
        self.lsl = spec_limits[0] if spec_limits else None
        self.usl = spec_limits[1] if spec_limits else None

        # Control limits (calculated from data)
        self.control_limits: Optional[ControlLimits] = None

        # Data storage
        self.data_points: deque = deque(maxlen=max_points)
        self.violations: List[SPCViolation] = []

        # Statistical values
        self.mean = 0.0
        self.std_dev = 0.0
        self.range_bar = 0.0

        logger.info(f"Created {chart_type.value} chart for {parameter_name}")

    def add_point(self, value: float, timestamp: Optional[datetime] = None) -> List[SPCViolation]:
        """
        Add data point to chart and check for violations.

        Args:
            value: Measurement value
            timestamp: Measurement timestamp

        Returns:
            List of violations detected
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        self.data_points.append({
            "value": value,
            "timestamp": timestamp
        })

        # Recalculate control limits if needed
        if len(self.data_points) >= 30 and self.control_limits is None:
            self.calculate_control_limits()

        # Check for violations
        violations = self.check_violations(value, timestamp)

        if violations:
            self.violations.extend(violations)
            for violation in violations:
                logger.warning(f"SPC Violation: {violation.message}")

        return violations

    def calculate_control_limits(self):
        """Calculate control limits from data"""
        raise NotImplementedError("Subclasses must implement calculate_control_limits")

    def check_violations(self, value: float, timestamp: datetime) -> List[SPCViolation]:
        """Check for control rule violations"""
        violations = []

        if self.control_limits is None:
            return violations

        # Rule 1: Point beyond control limits
        if value > self.control_limits.ucl:
            violations.append(SPCViolation(
                violation_type=ViolationType.ABOVE_UCL,
                value=value,
                timestamp=timestamp,
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.CRITICAL,
                message=f"{self.parameter_name} = {value:.2f} exceeds UCL = {self.control_limits.ucl:.2f}",
                recommended_action="Check process parameters immediately. Possible equipment malfunction."
            ))

        elif value < self.control_limits.lcl:
            violations.append(SPCViolation(
                violation_type=ViolationType.BELOW_LCL,
                value=value,
                timestamp=timestamp,
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.CRITICAL,
                message=f"{self.parameter_name} = {value:.2f} below LCL = {self.control_limits.lcl:.2f}",
                recommended_action="Check process parameters immediately. Possible equipment malfunction."
            ))

        # Rule 2: Run above/below centerline (7+ consecutive points)
        run_violations = self._check_run_rules()
        violations.extend(run_violations)

        # Rule 3: Trend (6+ consecutive increasing/decreasing)
        trend_violations = self._check_trend_rules()
        violations.extend(trend_violations)

        # Rule 4: Zone tests (Western Electric rules)
        zone_violations = self._check_zone_rules()
        violations.extend(zone_violations)

        return violations

    def _check_run_rules(self) -> List[SPCViolation]:
        """Check for runs (7+ consecutive points on same side of centerline)"""
        violations = []

        if len(self.data_points) < 7:
            return violations

        recent_points = list(self.data_points)[-7:]
        values = [p["value"] for p in recent_points]
        cl = self.control_limits.cl

        # Check if all points above centerline
        if all(v > cl for v in values):
            violations.append(SPCViolation(
                violation_type=ViolationType.RUN_ABOVE_CL,
                value=values[-1],
                timestamp=recent_points[-1]["timestamp"],
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.WARNING,
                message=f"Run detected: 7 consecutive points above centerline",
                recommended_action="Check for process shift. Consider R2R adjustment."
            ))

        # Check if all points below centerline
        elif all(v < cl for v in values):
            violations.append(SPCViolation(
                violation_type=ViolationType.RUN_BELOW_CL,
                value=values[-1],
                timestamp=recent_points[-1]["timestamp"],
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.WARNING,
                message=f"Run detected: 7 consecutive points below centerline",
                recommended_action="Check for process shift. Consider R2R adjustment."
            ))

        return violations

    def _check_trend_rules(self) -> List[SPCViolation]:
        """Check for trends (6+ consecutive increasing/decreasing points)"""
        violations = []

        if len(self.data_points) < 6:
            return violations

        recent_points = list(self.data_points)[-6:]
        values = [p["value"] for p in recent_points]

        # Check increasing trend
        is_increasing = all(values[i] < values[i+1] for i in range(len(values)-1))
        if is_increasing:
            violations.append(SPCViolation(
                violation_type=ViolationType.TREND_UP,
                value=values[-1],
                timestamp=recent_points[-1]["timestamp"],
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.ALARM,
                message=f"Upward trend detected: 6 consecutive increasing points",
                recommended_action="Process drift detected. Check chamber condition, PM schedule."
            ))

        # Check decreasing trend
        is_decreasing = all(values[i] > values[i+1] for i in range(len(values)-1))
        if is_decreasing:
            violations.append(SPCViolation(
                violation_type=ViolationType.TREND_DOWN,
                value=values[-1],
                timestamp=recent_points[-1]["timestamp"],
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.ALARM,
                message=f"Downward trend detected: 6 consecutive decreasing points",
                recommended_action="Process drift detected. Check chamber condition, PM schedule."
            ))

        return violations

    def _check_zone_rules(self) -> List[SPCViolation]:
        """Check Western Electric zone rules"""
        violations = []

        if self.control_limits is None or len(self.data_points) < 5:
            return violations

        cl = self.control_limits.cl
        ucl = self.control_limits.ucl
        lcl = self.control_limits.lcl

        # Define zones
        # Zone A: 2σ to 3σ
        # Zone B: 1σ to 2σ
        # Zone C: 0 to 1σ

        sigma = (ucl - cl) / 3

        # Get recent points
        recent_points = list(self.data_points)[-5:]
        values = [p["value"] for p in recent_points]

        # Zone A boundaries
        zone_a_upper = cl + 2 * sigma
        zone_a_lower = cl - 2 * sigma

        # Zone B boundaries
        zone_b_upper = cl + sigma
        zone_b_lower = cl - sigma

        # Rule: 2 out of 3 consecutive points in Zone A or beyond
        last_3 = values[-3:]
        zone_a_count = sum(1 for v in last_3 if v > zone_a_upper or v < zone_a_lower)

        if zone_a_count >= 2:
            violations.append(SPCViolation(
                violation_type=ViolationType.TWO_OF_THREE_ZONE_A,
                value=values[-1],
                timestamp=recent_points[-1]["timestamp"],
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.WARNING,
                message="2 of 3 points in Zone A or beyond",
                recommended_action="Monitor process closely. Increased variability detected."
            ))

        # Rule: 4 out of 5 consecutive points in Zone B or beyond
        zone_b_count = sum(1 for v in values if v > zone_b_upper or v < zone_b_lower)

        if zone_b_count >= 4:
            violations.append(SPCViolation(
                violation_type=ViolationType.FOUR_OF_FIVE_ZONE_B,
                value=values[-1],
                timestamp=recent_points[-1]["timestamp"],
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.WARNING,
                message="4 of 5 points in Zone B or beyond",
                recommended_action="Monitor process closely. Increased variability detected."
            ))

        return violations

    def get_chart_data(self) -> Dict[str, Any]:
        """Get chart data for visualization"""
        return {
            "chart_id": self.chart_id,
            "parameter_name": self.parameter_name,
            "chart_type": self.chart_type.value,
            "control_limits": {
                "ucl": self.control_limits.ucl if self.control_limits else None,
                "lcl": self.control_limits.lcl if self.control_limits else None,
                "cl": self.control_limits.cl if self.control_limits else None,
                "usl": self.usl,
                "lsl": self.lsl
            },
            "data_points": [
                {"value": p["value"], "timestamp": p["timestamp"].isoformat()}
                for p in self.data_points
            ],
            "num_violations": len(self.violations),
            "recent_violations": [v.__dict__ for v in self.violations[-10:]]
        }


class XBarChart(ControlChart):
    """X-bar chart for subgroup means"""

    def __init__(self, chart_id: str, parameter_name: str,
                 target: Optional[float] = None,
                 spec_limits: Optional[Tuple[float, float]] = None,
                 subgroup_size: int = 5):
        super().__init__(chart_id, parameter_name, ChartType.XBAR,
                        target, spec_limits, subgroup_size)

    def calculate_control_limits(self):
        """Calculate X-bar control limits"""
        if len(self.data_points) < 25:
            logger.warning(f"Insufficient data for {self.chart_id} limits (need 25+, have {len(self.data_points)})")
            return

        # Calculate grand mean
        values = [p["value"] for p in self.data_points]
        grand_mean = np.mean(values)

        # Calculate average range
        ranges = []
        for i in range(len(values) - self.subgroup_size + 1):
            subgroup = values[i:i+self.subgroup_size]
            ranges.append(max(subgroup) - min(subgroup))

        r_bar = np.mean(ranges)

        # Constants for X-bar chart (depends on subgroup size)
        # A2 factors for n=2 to 10
        A2_factors = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483,
                     7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}

        A2 = A2_factors.get(self.subgroup_size, 0.577)

        # Calculate control limits
        ucl = grand_mean + A2 * r_bar
        lcl = grand_mean - A2 * r_bar

        self.control_limits = ControlLimits(
            ucl=ucl,
            lcl=lcl,
            cl=grand_mean,
            usl=self.usl,
            lsl=self.lsl
        )

        logger.info(f"{self.chart_id}: CL={grand_mean:.2f}, UCL={ucl:.2f}, LCL={lcl:.2f}")


class EWMAChart(ControlChart):
    """Exponentially Weighted Moving Average chart"""

    def __init__(self, chart_id: str, parameter_name: str,
                 target: float,
                 spec_limits: Optional[Tuple[float, float]] = None,
                 lambda_ewma: float = 0.2):
        super().__init__(chart_id, parameter_name, ChartType.EWMA,
                        target, spec_limits)
        self.lambda_ewma = lambda_ewma
        self.ewma_value = target
        self.ewma_history: List[float] = []

    def add_point(self, value: float, timestamp: Optional[datetime] = None) -> List[SPCViolation]:
        """Add point and update EWMA"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Update EWMA
        self.ewma_value = self.lambda_ewma * value + (1 - self.lambda_ewma) * self.ewma_value
        self.ewma_history.append(self.ewma_value)

        self.data_points.append({
            "value": value,
            "ewma": self.ewma_value,
            "timestamp": timestamp
        })

        # Calculate control limits if needed
        if len(self.data_points) >= 30 and self.control_limits is None:
            self.calculate_control_limits()

        # Check violations on EWMA value (not raw value)
        violations = self.check_violations(self.ewma_value, timestamp)

        return violations

    def calculate_control_limits(self):
        """Calculate EWMA control limits"""
        values = [p["value"] for p in self.data_points]
        mean = np.mean(values)
        std_dev = np.std(values)

        # EWMA standard deviation
        n = len(self.data_points)
        L = 3  # Width of control limits (typically 2.7 to 3)

        # EWMA control limits
        # UCL/LCL = μ ± L*σ*sqrt(λ/(2-λ))
        sigma_ewma = std_dev * np.sqrt(self.lambda_ewma / (2 - self.lambda_ewma))

        ucl = mean + L * sigma_ewma
        lcl = mean - L * sigma_ewma

        self.control_limits = ControlLimits(
            ucl=ucl,
            lcl=lcl,
            cl=mean,
            usl=self.usl,
            lsl=self.lsl
        )

        logger.info(f"{self.chart_id} EWMA: CL={mean:.2f}, UCL={ucl:.2f}, LCL={lcl:.2f}")


class CUSUMChart(ControlChart):
    """Cumulative Sum (CUSUM) chart for detecting small shifts"""

    def __init__(self, chart_id: str, parameter_name: str,
                 target: float,
                 spec_limits: Optional[Tuple[float, float]] = None,
                 k: float = 0.5,
                 h: float = 5.0):
        """
        Args:
            k: Reference value (typically 0.5*sigma)
            h: Decision interval (typically 4-5*sigma)
        """
        super().__init__(chart_id, parameter_name, ChartType.CUSUM,
                        target, spec_limits)
        self.k = k
        self.h = h
        self.cusum_high = 0.0
        self.cusum_low = 0.0

    def add_point(self, value: float, timestamp: Optional[datetime] = None) -> List[SPCViolation]:
        """Add point and update CUSUM"""
        if timestamp is None:
            timestamp = datetime.utcnow()

        # Calculate CUSUM
        # C+ = max(0, C+ + (xi - μ - k))
        # C- = max(0, C- + (μ - xi - k))

        self.cusum_high = max(0, self.cusum_high + (value - self.target - self.k))
        self.cusum_low = max(0, self.cusum_low + (self.target - value - self.k))

        self.data_points.append({
            "value": value,
            "cusum_high": self.cusum_high,
            "cusum_low": self.cusum_low,
            "timestamp": timestamp
        })

        # Check violations
        violations = []

        if self.cusum_high > self.h:
            violations.append(SPCViolation(
                violation_type=ViolationType.ABOVE_UCL,
                value=value,
                timestamp=timestamp,
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.ALARM,
                message=f"CUSUM high exceeded: {self.cusum_high:.2f} > {self.h:.2f}",
                recommended_action="Upward process shift detected. Investigate root cause."
            ))
            # Reset CUSUM
            self.cusum_high = 0.0

        if self.cusum_low > self.h:
            violations.append(SPCViolation(
                violation_type=ViolationType.BELOW_LCL,
                value=value,
                timestamp=timestamp,
                chart_id=self.chart_id,
                parameter_name=self.parameter_name,
                severity=AlarmLevel.ALARM,
                message=f"CUSUM low exceeded: {self.cusum_low:.2f} > {self.h:.2f}",
                recommended_action="Downward process shift detected. Investigate root cause."
            ))
            # Reset CUSUM
            self.cusum_low = 0.0

        return violations

    def calculate_control_limits(self):
        """CUSUM uses decision interval h instead of traditional limits"""
        pass  # Not applicable for CUSUM


def calculate_process_capability(data: np.ndarray,
                                lsl: float,
                                usl: float,
                                target: Optional[float] = None) -> ProcessCapability:
    """
    Calculate process capability indices.

    Cp = (USL - LSL) / (6σ)  - Potential capability
    Cpk = min((USL - μ) / (3σ), (μ - LSL) / (3σ))  - Actual capability

    Args:
        data: Process data
        lsl: Lower specification limit
        usl: Upper specification limit
        target: Target value (optional)

    Returns:
        ProcessCapability metrics
    """
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation

    # Cp - Potential capability (centering not considered)
    cp = (usl - lsl) / (6 * std_dev) if std_dev > 0 else 0

    # Cpk - Actual capability (considers centering)
    cpu = (usl - mean) / (3 * std_dev) if std_dev > 0 else 0
    cpl = (mean - lsl) / (3 * std_dev) if std_dev > 0 else 0
    cpk = min(cpu, cpl)

    # Pp and Ppk (using overall standard deviation)
    pp = cp  # Same as Cp for now
    ppk = cpk  # Same as Cpk for now

    # Sigma level
    z_score = min(cpu, cpl)
    sigma_level = z_score

    # DPMO (Defects Per Million Opportunities)
    # Using standard normal distribution
    from scipy.stats import norm
    dpmo = (1 - norm.cdf(z_score)) * 1e6

    capability = ProcessCapability(
        cp=cp,
        cpk=cpk,
        pp=pp,
        ppk=ppk,
        sigma_level=sigma_level,
        dpmo=dpmo
    )

    logger.info(f"Process Capability: Cp={cp:.2f}, Cpk={cpk:.2f}, Sigma={sigma_level:.2f}")

    return capability


class SPCManager:
    """
    High-level SPC Manager.
    Manages multiple control charts and provides unified interface.
    """

    def __init__(self):
        self.charts: Dict[str, ControlChart] = {}
        self.all_violations: List[SPCViolation] = []

        logger.info("Initialized SPC Manager")

    def create_chart(self,
                    chart_id: str,
                    parameter_name: str,
                    chart_type: ChartType = ChartType.XBAR,
                    target: Optional[float] = None,
                    spec_limits: Optional[Tuple[float, float]] = None,
                    **kwargs) -> ControlChart:
        """Create and register a control chart"""

        if chart_type == ChartType.XBAR:
            chart = XBarChart(chart_id, parameter_name, target, spec_limits,
                            kwargs.get("subgroup_size", 5))
        elif chart_type == ChartType.EWMA:
            chart = EWMAChart(chart_id, parameter_name, target, spec_limits,
                            kwargs.get("lambda_ewma", 0.2))
        elif chart_type == ChartType.CUSUM:
            chart = CUSUMChart(chart_id, parameter_name, target, spec_limits,
                             kwargs.get("k", 0.5), kwargs.get("h", 5.0))
        else:
            chart = ControlChart(chart_id, parameter_name, chart_type,
                               target, spec_limits)

        self.charts[chart_id] = chart
        logger.info(f"Created chart: {chart_id} ({chart_type.value})")

        return chart

    def add_measurement(self,
                       chart_id: str,
                       value: float,
                       timestamp: Optional[datetime] = None) -> List[SPCViolation]:
        """Add measurement to chart"""
        if chart_id not in self.charts:
            logger.error(f"Chart {chart_id} not found")
            return []

        violations = self.charts[chart_id].add_point(value, timestamp)
        self.all_violations.extend(violations)

        return violations

    def get_active_alarms(self, lookback_hours: int = 24) -> List[SPCViolation]:
        """Get active alarms from recent period"""
        cutoff = datetime.utcnow() - timedelta(hours=lookback_hours)
        active_alarms = [
            v for v in self.all_violations
            if v.timestamp > cutoff and v.severity in [AlarmLevel.ALARM, AlarmLevel.CRITICAL]
        ]
        return active_alarms

    def get_chart_summary(self) -> Dict[str, Any]:
        """Get summary of all charts"""
        summary = {
            "total_charts": len(self.charts),
            "total_violations": len(self.all_violations),
            "active_alarms": len(self.get_active_alarms()),
            "charts": {
                chart_id: chart.get_chart_data()
                for chart_id, chart in self.charts.items()
            }
        }
        return summary
