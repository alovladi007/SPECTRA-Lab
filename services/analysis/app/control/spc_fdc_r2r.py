"""
CVD Platform - SPC/FDC/R2R Control System
Statistical Process Control, Fault Detection & Classification, and Run-to-Run Control
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
import math

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest


logger = logging.getLogger(__name__)


# ============================================================================
# SPC Control Charts
# ============================================================================

class ControlChartType(str, Enum):
    """SPC control chart types"""
    XBAR = "xbar"  # X-bar chart
    INDIVIDUALS = "individuals"  # I-chart
    EWMA = "ewma"  # Exponentially Weighted Moving Average
    CUSUM = "cusum"  # Cumulative Sum
    RANGE = "range"  # R-chart
    STDEV = "stdev"  # S-chart


@dataclass
class ControlLimits:
    """Control chart limits"""
    ucl: float  # Upper Control Limit
    lcl: float  # Lower Control Limit
    center_line: float  # Center line (target or mean)
    usl: Optional[float] = None  # Upper Specification Limit
    lsl: Optional[float] = None  # Lower Specification Limit


@dataclass
class SPCViolation:
    """SPC rule violation"""
    rule_name: str
    description: str
    severity: str  # 'WARNING', 'ALARM', 'CRITICAL'
    point_indices: List[int]
    timestamp: datetime


class SPCChart:
    """
    Base class for SPC control charts.
    """

    def __init__(
        self,
        chart_type: ControlChartType,
        subgroup_size: int = 1,
    ):
        """
        Initialize SPC chart.

        Args:
            chart_type: Type of control chart
            subgroup_size: Size of subgroups for grouped charts
        """
        self.chart_type = chart_type
        self.subgroup_size = subgroup_size
        self.limits: Optional[ControlLimits] = None

    def calculate_limits(
        self,
        data: np.ndarray,
        target: Optional[float] = None,
        sigma_multiplier: float = 3.0,
    ) -> ControlLimits:
        """
        Calculate control limits from historical data.

        Args:
            data: Historical data array
            target: Target value (if known)
            sigma_multiplier: Multiplier for control limits (default 3 sigma)

        Returns:
            Control limits
        """
        raise NotImplementedError

    def check_violations(
        self,
        data: np.ndarray,
        limits: ControlLimits,
    ) -> List[SPCViolation]:
        """
        Check for control chart rule violations.

        Args:
            data: Data points to check
            limits: Control limits

        Returns:
            List of violations
        """
        violations = []

        # Western Electric Rules
        violations.extend(self._check_western_electric_rules(data, limits))

        return violations

    def _check_western_electric_rules(
        self,
        data: np.ndarray,
        limits: ControlLimits,
    ) -> List[SPCViolation]:
        """
        Check Western Electric rules for control charts.

        Rules:
        1. One point beyond 3-sigma
        2. Two out of three consecutive points beyond 2-sigma (same side)
        3. Four out of five consecutive points beyond 1-sigma (same side)
        4. Eight consecutive points on same side of center line
        5. Six consecutive points increasing or decreasing (trend)
        6. Fifteen consecutive points within 1-sigma (stratification)
        7. Fourteen consecutive points alternating up and down
        8. Eight consecutive points beyond 1-sigma (either side)

        Args:
            data: Data array
            limits: Control limits

        Returns:
            List of violations
        """
        violations = []

        center = limits.center_line
        ucl = limits.ucl
        lcl = limits.lcl

        # Calculate sigma
        sigma = (ucl - center) / 3.0

        # Rule 1: One point beyond 3-sigma
        beyond_3sigma = np.where((data > ucl) | (data < lcl))[0]
        if len(beyond_3sigma) > 0:
            violations.append(
                SPCViolation(
                    rule_name="rule_1",
                    description="One or more points beyond 3-sigma limits",
                    severity="CRITICAL",
                    point_indices=beyond_3sigma.tolist(),
                    timestamp=datetime.utcnow(),
                )
            )

        # Rule 2: Two out of three beyond 2-sigma (same side)
        ucl_2sigma = center + 2 * sigma
        lcl_2sigma = center - 2 * sigma

        for i in range(len(data) - 2):
            window = data[i : i + 3]
            beyond_upper = np.sum(window > ucl_2sigma)
            beyond_lower = np.sum(window < lcl_2sigma)

            if beyond_upper >= 2 or beyond_lower >= 2:
                violations.append(
                    SPCViolation(
                        rule_name="rule_2",
                        description="Two out of three consecutive points beyond 2-sigma",
                        severity="ALARM",
                        point_indices=[i, i + 1, i + 2],
                        timestamp=datetime.utcnow(),
                    )
                )

        # Rule 3: Four out of five beyond 1-sigma (same side)
        ucl_1sigma = center + sigma
        lcl_1sigma = center - sigma

        for i in range(len(data) - 4):
            window = data[i : i + 5]
            beyond_upper = np.sum(window > ucl_1sigma)
            beyond_lower = np.sum(window < lcl_1sigma)

            if beyond_upper >= 4 or beyond_lower >= 4:
                violations.append(
                    SPCViolation(
                        rule_name="rule_3",
                        description="Four out of five consecutive points beyond 1-sigma",
                        severity="WARNING",
                        point_indices=list(range(i, i + 5)),
                        timestamp=datetime.utcnow(),
                    )
                )

        # Rule 4: Eight consecutive points on same side of center
        for i in range(len(data) - 7):
            window = data[i : i + 8]
            all_above = np.all(window > center)
            all_below = np.all(window < center)

            if all_above or all_below:
                violations.append(
                    SPCViolation(
                        rule_name="rule_4",
                        description="Eight consecutive points on same side of center line",
                        severity="WARNING",
                        point_indices=list(range(i, i + 8)),
                        timestamp=datetime.utcnow(),
                    )
                )

        # Rule 5: Six consecutive points increasing or decreasing
        for i in range(len(data) - 5):
            window = data[i : i + 6]
            diffs = np.diff(window)

            all_increasing = np.all(diffs > 0)
            all_decreasing = np.all(diffs < 0)

            if all_increasing or all_decreasing:
                violations.append(
                    SPCViolation(
                        rule_name="rule_5",
                        description="Six consecutive points showing trend",
                        severity="WARNING",
                        point_indices=list(range(i, i + 6)),
                        timestamp=datetime.utcnow(),
                    )
                )

        return violations


class XBarChart(SPCChart):
    """X-bar control chart for grouped data"""

    def __init__(self, subgroup_size: int = 5):
        super().__init__(ControlChartType.XBAR, subgroup_size)

    def calculate_limits(
        self,
        data: np.ndarray,
        target: Optional[float] = None,
        sigma_multiplier: float = 3.0,
    ) -> ControlLimits:
        """Calculate X-bar chart limits"""

        # Reshape data into subgroups
        n_subgroups = len(data) // self.subgroup_size
        data_reshaped = data[: n_subgroups * self.subgroup_size].reshape(n_subgroups, self.subgroup_size)

        # Calculate subgroup means and ranges
        means = np.mean(data_reshaped, axis=1)
        ranges = np.ptp(data_reshaped, axis=1)

        # Grand mean
        grand_mean = np.mean(means) if target is None else target

        # Average range
        r_bar = np.mean(ranges)

        # Control chart constants (for n=5, A2=0.577)
        # For other n, use lookup table or formula
        control_chart_constants = {
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

        A2 = control_chart_constants.get(self.subgroup_size, 0.577)

        # Calculate limits
        ucl = grand_mean + A2 * r_bar
        lcl = grand_mean - A2 * r_bar

        self.limits = ControlLimits(
            ucl=ucl,
            lcl=lcl,
            center_line=grand_mean,
        )

        return self.limits


class EWMAChart(SPCChart):
    """Exponentially Weighted Moving Average chart"""

    def __init__(self, lambda_param: float = 0.2):
        """
        Initialize EWMA chart.

        Args:
            lambda_param: Smoothing parameter (0 < λ ≤ 1)
        """
        super().__init__(ControlChartType.EWMA)
        self.lambda_param = lambda_param

    def calculate_ewma(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate EWMA values.

        z_i = λx_i + (1-λ)z_{i-1}

        Args:
            data: Input data

        Returns:
            EWMA values
        """
        ewma = np.zeros(len(data))
        ewma[0] = data[0]

        for i in range(1, len(data)):
            ewma[i] = self.lambda_param * data[i] + (1 - self.lambda_param) * ewma[i - 1]

        return ewma

    def calculate_limits(
        self,
        data: np.ndarray,
        target: Optional[float] = None,
        sigma_multiplier: float = 3.0,
    ) -> ControlLimits:
        """Calculate EWMA chart limits"""

        # Calculate process mean and std dev
        center = np.mean(data) if target is None else target
        sigma = np.std(data, ddof=1)

        # EWMA control limits
        # UCL = μ + L·σ·√(λ/(2-λ))
        # LCL = μ - L·σ·√(λ/(2-λ))

        L = sigma_multiplier
        factor = math.sqrt(self.lambda_param / (2 - self.lambda_param))

        ucl = center + L * sigma * factor
        lcl = center - L * sigma * factor

        self.limits = ControlLimits(
            ucl=ucl,
            lcl=lcl,
            center_line=center,
        )

        return self.limits


class CUSUMChart(SPCChart):
    """Cumulative Sum control chart"""

    def __init__(self, k: float = 0.5, h: float = 5.0):
        """
        Initialize CUSUM chart.

        Args:
            k: Reference value (typically 0.5 * process shift)
            h: Decision interval (typically 4-5)
        """
        super().__init__(ControlChartType.CUSUM)
        self.k = k
        self.h = h

    def calculate_cusum(
        self,
        data: np.ndarray,
        target: float,
        sigma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate CUSUM statistics.

        C+ = max(0, x_i - (μ + kσ) + C+_{i-1})
        C- = max(0, (μ - kσ) - x_i + C-_{i-1})

        Args:
            data: Input data
            target: Target mean
            sigma: Process standard deviation

        Returns:
            (C_plus, C_minus) arrays
        """
        n = len(data)
        C_plus = np.zeros(n)
        C_minus = np.zeros(n)

        for i in range(n):
            if i == 0:
                C_plus[i] = max(0, data[i] - (target + self.k * sigma))
                C_minus[i] = max(0, (target - self.k * sigma) - data[i])
            else:
                C_plus[i] = max(0, data[i] - (target + self.k * sigma) + C_plus[i - 1])
                C_minus[i] = max(0, (target - self.k * sigma) - data[i] + C_minus[i - 1])

        return C_plus, C_minus

    def calculate_limits(
        self,
        data: np.ndarray,
        target: Optional[float] = None,
        sigma_multiplier: float = 3.0,
    ) -> ControlLimits:
        """Calculate CUSUM chart limits"""

        center = np.mean(data) if target is None else target
        sigma = np.std(data, ddof=1)

        # CUSUM decision limit
        H = self.h * sigma

        self.limits = ControlLimits(
            ucl=H,
            lcl=0,
            center_line=0,
        )

        return self.limits


# ============================================================================
# Process Capability Analysis
# ============================================================================

class ProcessCapability:
    """Process capability indices (Cp, Cpk, Pp, Ppk)"""

    @staticmethod
    def calculate_cp(
        data: np.ndarray,
        usl: float,
        lsl: float,
    ) -> float:
        """
        Calculate Cp (Process Capability).

        Cp = (USL - LSL) / (6σ)

        Args:
            data: Process data
            usl: Upper Specification Limit
            lsl: Lower Specification Limit

        Returns:
            Cp value
        """
        sigma = np.std(data, ddof=1)
        cp = (usl - lsl) / (6 * sigma)
        return cp

    @staticmethod
    def calculate_cpk(
        data: np.ndarray,
        usl: float,
        lsl: float,
    ) -> float:
        """
        Calculate Cpk (Process Capability Index).

        Cpk = min((USL - μ)/(3σ), (μ - LSL)/(3σ))

        Args:
            data: Process data
            usl: Upper Specification Limit
            lsl: Lower Specification Limit

        Returns:
            Cpk value
        """
        mean = np.mean(data)
        sigma = np.std(data, ddof=1)

        cpu = (usl - mean) / (3 * sigma)
        cpl = (mean - lsl) / (3 * sigma)

        cpk = min(cpu, cpl)
        return cpk

    @staticmethod
    def calculate_pp(
        data: np.ndarray,
        usl: float,
        lsl: float,
    ) -> float:
        """
        Calculate Pp (Process Performance).

        Similar to Cp but uses population std dev.

        Args:
            data: Process data
            usl: Upper Specification Limit
            lsl: Lower Specification Limit

        Returns:
            Pp value
        """
        sigma = np.std(data, ddof=0)  # Population std dev
        pp = (usl - lsl) / (6 * sigma)
        return pp

    @staticmethod
    def calculate_ppk(
        data: np.ndarray,
        usl: float,
        lsl: float,
    ) -> float:
        """
        Calculate Ppk (Process Performance Index).

        Args:
            data: Process data
            usl: Upper Specification Limit
            lsl: Lower Specification Limit

        Returns:
            Ppk value
        """
        mean = np.mean(data)
        sigma = np.std(data, ddof=0)  # Population std dev

        ppu = (usl - mean) / (3 * sigma)
        ppl = (mean - lsl) / (3 * sigma)

        ppk = min(ppu, ppl)
        return ppk


# ============================================================================
# Fault Detection & Classification (FDC)
# ============================================================================

class FaultDetector:
    """
    Fault Detection and Classification using statistical methods and ML.
    """

    def __init__(self):
        self.anomaly_detector = None
        self.is_trained = False

    def train(
        self,
        normal_data: np.ndarray,
        contamination: float = 0.1,
    ):
        """
        Train fault detector on normal operation data.

        Args:
            normal_data: Data from normal operation (n_samples, n_features)
            contamination: Expected proportion of outliers
        """
        # Use Isolation Forest for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
        )

        self.anomaly_detector.fit(normal_data)
        self.is_trained = True

        logger.info(f"FaultDetector trained on {len(normal_data)} samples")

    def detect(
        self,
        data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect faults in new data.

        Args:
            data: New data to check (n_samples, n_features)

        Returns:
            (predictions, anomaly_scores)
            predictions: -1 for anomalies, 1 for normal
            anomaly_scores: Lower scores indicate anomalies
        """
        if not self.is_trained:
            raise RuntimeError("FaultDetector must be trained first")

        predictions = self.anomaly_detector.predict(data)
        scores = self.anomaly_detector.score_samples(data)

        return predictions, scores

    def classify_fault(
        self,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Classify fault type based on features.

        Args:
            features: Process features

        Returns:
            Fault classification result
        """
        # Simple rule-based classification
        # TODO: Implement ML-based classification

        fault_type = "unknown"
        confidence = 0.0

        # Temperature-related faults
        if "temperature_deviation" in features:
            if abs(features["temperature_deviation"]) > 10:
                fault_type = "temperature_excursion"
                confidence = min(abs(features["temperature_deviation"]) / 50, 1.0)

        # Pressure-related faults
        if "pressure_deviation" in features:
            if abs(features["pressure_deviation"]) > 20:
                fault_type = "pressure_deviation"
                confidence = min(abs(features["pressure_deviation"]) / 100, 1.0)

        # Flow-related faults
        if "flow_deviation" in features:
            if abs(features["flow_deviation"]) > 5:
                fault_type = "gas_flow_anomaly"
                confidence = min(abs(features["flow_deviation"]) / 20, 1.0)

        return {
            "fault_type": fault_type,
            "confidence": confidence,
            "features": features,
        }


# ============================================================================
# Run-to-Run (R2R) Control
# ============================================================================

class R2RController:
    """Base class for Run-to-Run controllers"""

    def __init__(self, target: float):
        """
        Initialize R2R controller.

        Args:
            target: Target value for controlled variable
        """
        self.target = target

    def calculate_adjustment(
        self,
        measured_value: float,
        current_recipe_param: float,
    ) -> float:
        """
        Calculate recipe parameter adjustment.

        Args:
            measured_value: Measured output from previous run
            current_recipe_param: Current recipe parameter value

        Returns:
            Adjusted recipe parameter
        """
        raise NotImplementedError


class EWMAController(R2RController):
    """
    EWMA R2R Controller.
    Uses Exponentially Weighted Moving Average for control.
    """

    def __init__(
        self,
        target: float,
        lambda_ewma: float = 0.3,
        gain: float = 0.8,
    ):
        """
        Initialize EWMA controller.

        Args:
            target: Target value
            lambda_ewma: EWMA smoothing parameter
            gain: Control gain
        """
        super().__init__(target)
        self.lambda_ewma = lambda_ewma
        self.gain = gain
        self.ewma_error = 0.0

    def calculate_adjustment(
        self,
        measured_value: float,
        current_recipe_param: float,
    ) -> float:
        """Calculate EWMA control adjustment"""

        # Calculate error
        error = self.target - measured_value

        # Update EWMA error
        self.ewma_error = self.lambda_ewma * error + (1 - self.lambda_ewma) * self.ewma_error

        # Calculate adjustment
        adjustment = self.gain * self.ewma_error

        # Apply adjustment
        new_param = current_recipe_param + adjustment

        return new_param


class PIDController(R2RController):
    """
    PID R2R Controller.
    Proportional-Integral-Derivative control.
    """

    def __init__(
        self,
        target: float,
        kp: float = 1.0,
        ki: float = 0.1,
        kd: float = 0.05,
    ):
        """
        Initialize PID controller.

        Args:
            target: Target value
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        super().__init__(target)
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.previous_error = 0.0

    def calculate_adjustment(
        self,
        measured_value: float,
        current_recipe_param: float,
    ) -> float:
        """Calculate PID control adjustment"""

        # Calculate error
        error = self.target - measured_value

        # Update integral
        self.integral += error

        # Calculate derivative
        derivative = error - self.previous_error

        # PID formula
        adjustment = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Update previous error
        self.previous_error = error

        # Apply adjustment
        new_param = current_recipe_param + adjustment

        return new_param


class MPCController(R2RController):
    """
    Model Predictive Control for R2R.
    Uses process model to predict optimal control actions.
    """

    def __init__(
        self,
        target: float,
        process_model,
        horizon: int = 5,
    ):
        """
        Initialize MPC controller.

        Args:
            target: Target value
            process_model: Process model function
            horizon: Prediction horizon
        """
        super().__init__(target)
        self.process_model = process_model
        self.horizon = horizon

    def calculate_adjustment(
        self,
        measured_value: float,
        current_recipe_param: float,
    ) -> float:
        """Calculate MPC optimal control action"""

        # Simplified MPC: minimize tracking error over horizon
        def objective(u):
            # u is the control action (adjustment)
            predicted_output = self.process_model(current_recipe_param + u)
            cost = (predicted_output - self.target) ** 2
            return cost

        # Optimize
        result = minimize(
            objective,
            x0=0.0,
            method="SLSQP",
            bounds=[(-50, 50)],  # Constrain adjustment range
        )

        adjustment = result.x[0]
        new_param = current_recipe_param + adjustment

        return new_param


# ============================================================================
# Drift Detection and Compensation
# ============================================================================

class DriftDetector:
    """Detect and compensate for process drift"""

    @staticmethod
    def detect_linear_drift(
        timestamps: np.ndarray,
        values: np.ndarray,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Detect linear drift using linear regression.

        Args:
            timestamps: Time points
            values: Measured values
            confidence: Confidence level for significance

        Returns:
            Drift detection result
        """
        # Normalize timestamps
        t = (timestamps - timestamps[0]) / 3600.0  # Convert to hours

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(t, values)

        # Check significance
        is_significant = p_value < (1 - confidence)

        # Calculate drift rate (per hour)
        drift_rate = slope

        return {
            "is_drifting": is_significant,
            "drift_rate_per_hour": drift_rate,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "slope": slope,
            "intercept": intercept,
        }

    @staticmethod
    def compensate_drift(
        current_param: float,
        drift_rate: float,
        time_elapsed_hours: float,
    ) -> float:
        """
        Compensate for detected drift.

        Args:
            current_param: Current parameter value
            drift_rate: Detected drift rate (per hour)
            time_elapsed_hours: Time since last calibration

        Returns:
            Compensated parameter value
        """
        compensation = -drift_rate * time_elapsed_hours
        compensated_param = current_param + compensation

        return compensated_param
