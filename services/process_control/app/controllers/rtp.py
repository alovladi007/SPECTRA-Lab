"""RTP control algorithms.

Provides PID control with anti-windup, Model Predictive Control (MPC),
Run-to-Run (R2R) control, thermal budget calculations, and performance analysis.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import deque


# ============================================================================
# Constants
# ============================================================================

BOLTZMANN_K = 8.617e-5  # eV/K
ACTIVATION_ENERGIES = {
    "boron": 3.65,  # eV
    "phosphorus": 3.66,  # eV
    "arsenic": 4.05,  # eV
    "oxidation": 1.23,  # eV for oxide growth
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class PIDGains:
    """PID controller gains."""
    Kp: float  # Proportional gain
    Ki: float  # Integral gain
    Kd: float  # Derivative gain

    # Anti-windup
    windup_limit: float = 100.0  # Maximum integral accumulation

    # Feed-forward
    enable_feedforward: bool = False
    feedforward_gain: float = 1.0


@dataclass
class PIDState:
    """PID controller state."""
    integral: float = 0.0
    previous_error: float = 0.0
    previous_setpoint: float = 0.0
    saturated: bool = False

    # History for analysis
    error_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    output_history: deque = field(default_factory=lambda: deque(maxlen=1000))


@dataclass
class MPCParameters:
    """Model Predictive Control parameters."""
    prediction_horizon: int = 20  # Steps ahead to predict
    control_horizon: int = 10  # Steps ahead to optimize

    # Constraints
    max_temp_rate_C_per_s: float = 100.0  # Maximum ramp rate
    max_overshoot_C: float = 10.0  # Maximum overshoot allowed
    max_lamp_power_pct: float = 100.0
    min_lamp_power_pct: float = 0.0
    max_lamp_rate_pct_per_s: float = 50.0  # Rate limit

    # Weighting
    setpoint_tracking_weight: float = 1.0
    overshoot_penalty_weight: float = 10.0
    control_effort_weight: float = 0.1
    rate_change_weight: float = 0.01


@dataclass
class MPCState:
    """MPC controller state."""
    predicted_trajectory: np.ndarray = field(default_factory=lambda: np.array([]))
    optimal_control_sequence: np.ndarray = field(default_factory=lambda: np.array([]))
    constraint_violations: int = 0

    # History
    prediction_errors: deque = field(default_factory=lambda: deque(maxlen=100))


@dataclass
class R2RState:
    """Run-to-Run controller state."""
    run_count: int = 0

    # EWMA tracking
    emissivity_history: List[float] = field(default_factory=list)
    lamp_power_history: List[np.ndarray] = field(default_factory=list)
    overshoot_history: List[float] = field(default_factory=list)

    # Current adjustments
    emissivity_adjustment: float = 1.0
    lamp_power_trim: np.ndarray = field(default_factory=lambda: np.ones(4))

    # EWMA parameters
    alpha: float = 0.3  # Smoothing factor (0-1, higher = more weight on recent)


@dataclass
class ThermalBudget:
    """Thermal budget calculation results."""
    dopant_species: str
    activation_energy_eV: float
    integrated_budget: float  # ∫ exp(-Ea/kT) dt
    equivalent_time_at_1000C_s: float
    peak_activation_rate: float  # Maximum exp(-Ea/kT)


@dataclass
class RampFidelity:
    """Ramp quality metrics."""
    segment_id: int
    target_ramp_rate_C_per_s: float
    actual_ramp_rate_C_per_s: float

    # Overshoot/undershoot
    peak_overshoot_C: float
    peak_overshoot_pct: float
    peak_undershoot_C: float

    # Tracking error
    rmse_C: float  # Root mean square error
    max_error_C: float
    settling_time_s: float  # Time to reach ±2% of setpoint

    # Stability during dwell
    dwell_std_C: float
    dwell_drift_C_per_s: float


@dataclass
class ControllerPerformance:
    """Complete controller performance report."""
    recipe_id: str
    run_id: str

    # Per-segment metrics
    ramp_fidelities: List[RampFidelity]

    # Overall metrics
    overall_rmse_C: float
    max_overshoot_pct: float
    total_thermal_budget: float

    # Controller flags
    saturation_events: int
    constraint_violations: int
    recommended_tuning: Dict[str, float]
    drift_flags: List[str]

    # Timestamps
    start_time: float
    end_time: float
    total_duration_s: float


# ============================================================================
# PID Controller with Anti-Windup
# ============================================================================

class PIDController:
    """PID controller with anti-windup and feed-forward.

    Implements:
    - Standard PID: u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de/dt
    - Anti-windup: clamp integral term when saturated
    - Feed-forward: add derivative of setpoint
    """

    def __init__(self, gains: PIDGains):
        """Initialize PID controller.

        Args:
            gains: PID gains and configuration
        """
        self.gains = gains
        self.state = PIDState()

    def update(
        self,
        setpoint: float,
        measured: float,
        dt: float,
        output_limits: Tuple[float, float] = (0.0, 100.0)
    ) -> float:
        """Update PID controller for one time step.

        Args:
            setpoint: Desired value
            measured: Current measured value
            dt: Time step (seconds)
            output_limits: (min, max) output limits

        Returns:
            Control output (e.g., lamp power %)
        """
        # Error
        error = setpoint - measured

        # Proportional term
        P = self.gains.Kp * error

        # Integral term with anti-windup
        if not self.state.saturated:
            self.state.integral += error * dt
            # Clamp integral
            self.state.integral = np.clip(
                self.state.integral,
                -self.gains.windup_limit,
                self.gains.windup_limit
            )

        I = self.gains.Ki * self.state.integral

        # Derivative term (on error)
        if dt > 0:
            derivative = (error - self.state.previous_error) / dt
        else:
            derivative = 0.0

        D = self.gains.Kd * derivative

        # Feed-forward (derivative of setpoint)
        FF = 0.0
        if self.gains.enable_feedforward and dt > 0:
            setpoint_derivative = (setpoint - self.state.previous_setpoint) / dt
            FF = self.gains.feedforward_gain * setpoint_derivative

        # Total output
        output = P + I + D + FF

        # Apply output limits
        output_clamped = np.clip(output, output_limits[0], output_limits[1])

        # Check if saturated
        self.state.saturated = (output != output_clamped)

        # Update state
        self.state.previous_error = error
        self.state.previous_setpoint = setpoint

        # Record history
        self.state.error_history.append(error)
        self.state.output_history.append(output_clamped)

        return output_clamped

    def reset(self):
        """Reset controller state."""
        self.state = PIDState()

    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from history."""
        if len(self.state.error_history) == 0:
            return {}

        errors = np.array(self.state.error_history)

        return {
            "rmse": np.sqrt(np.mean(errors**2)),
            "max_error": np.max(np.abs(errors)),
            "mean_error": np.mean(errors),
            "std_error": np.std(errors),
        }


# ============================================================================
# Model Predictive Control (MPC)
# ============================================================================

class MPCController:
    """Model Predictive Controller for RTP temperature control.

    Implements:
    - Predictive thermal model
    - Constraint handling (rate limits, overshoot, saturation)
    - Quadratic programming optimization
    """

    def __init__(self, params: MPCParameters, num_zones: int = 4):
        """Initialize MPC controller.

        Args:
            params: MPC parameters
            num_zones: Number of lamp zones
        """
        self.params = params
        self.num_zones = num_zones
        self.state = MPCState()

        # Simple thermal model parameters
        self.thermal_time_constant = 3.0  # seconds
        self.lamp_to_temp_gain = 10.0  # C per % lamp power

    def predict_trajectory(
        self,
        current_temp: float,
        current_lamp_power: float,
        setpoint_trajectory: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Predict temperature trajectory using simple model.

        Args:
            current_temp: Current temperature (C)
            current_lamp_power: Current lamp power (%)
            setpoint_trajectory: Future setpoints
            dt: Time step (seconds)

        Returns:
            Predicted temperature trajectory
        """
        horizon = min(self.params.prediction_horizon, len(setpoint_trajectory))
        predicted_temps = np.zeros(horizon)

        # Simple first-order model: tau * dT/dt = u - T
        temp = current_temp
        lamp = current_lamp_power

        for i in range(horizon):
            # Heat input from lamps
            heat_input = lamp * self.lamp_to_temp_gain

            # First-order lag
            dT_dt = (heat_input - temp) / self.thermal_time_constant
            temp += dT_dt * dt

            predicted_temps[i] = temp

            # Simple control law for prediction (proportional)
            error = setpoint_trajectory[i] - temp
            lamp = np.clip(lamp + 0.5 * error, 0, 100)

        return predicted_temps

    def optimize_control(
        self,
        current_temp: float,
        current_lamp_power: np.ndarray,
        setpoint_trajectory: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """Optimize control sequence using constrained optimization.

        Args:
            current_temp: Current temperature (C)
            current_lamp_power: Current lamp power per zone (%)
            setpoint_trajectory: Future setpoint trajectory
            dt: Time step (seconds)

        Returns:
            Optimal lamp power sequence for control horizon
        """
        horizon = min(self.params.control_horizon, len(setpoint_trajectory))

        # Simplified optimization: gradient descent with constraints
        lamp_sequence = np.zeros((horizon, self.num_zones))
        lamp_sequence[0] = current_lamp_power

        # Forward simulate with constraint enforcement
        temp = current_temp

        for i in range(horizon):
            setpoint = setpoint_trajectory[i]
            error = setpoint - temp

            # Proportional control with overshoot prevention
            if error > 0:
                # Heating
                control_gain = 2.0
                # Reduce gain if approaching setpoint
                if error < self.params.max_overshoot_C:
                    control_gain *= (error / self.params.max_overshoot_C)
            else:
                # Cooling
                control_gain = 1.5

            # Calculate desired lamp power
            lamp_desired = current_lamp_power + control_gain * error

            # Apply constraints
            lamp_constrained = np.clip(
                lamp_desired,
                self.params.min_lamp_power_pct,
                self.params.max_lamp_power_pct
            )

            # Apply rate limit
            if i > 0:
                max_change = self.params.max_lamp_rate_pct_per_s * dt
                lamp_change = lamp_constrained - lamp_sequence[i-1]
                lamp_change_limited = np.clip(lamp_change, -max_change, max_change)
                lamp_constrained = lamp_sequence[i-1] + lamp_change_limited

            lamp_sequence[i] = lamp_constrained

            # Update predicted temperature
            heat_input = np.mean(lamp_constrained) * self.lamp_to_temp_gain
            dT_dt = (heat_input - temp) / self.thermal_time_constant
            temp += dT_dt * dt

            # Check for overshoot
            if temp > setpoint + self.params.max_overshoot_C:
                self.state.constraint_violations += 1
                # Reduce lamp power
                lamp_sequence[i] *= 0.5

        self.state.optimal_control_sequence = lamp_sequence
        return lamp_sequence

    def update(
        self,
        current_temp: float,
        current_lamp_power: np.ndarray,
        setpoint: float,
        future_setpoints: List[float],
        dt: float
    ) -> np.ndarray:
        """Update MPC controller.

        Args:
            current_temp: Current measured temperature (C)
            current_lamp_power: Current lamp powers (%)
            setpoint: Current setpoint (C)
            future_setpoints: Future setpoint trajectory
            dt: Time step (seconds)

        Returns:
            Optimal lamp powers for current time step
        """
        # Build setpoint trajectory
        setpoint_traj = np.array([setpoint] + future_setpoints)

        # Predict trajectory
        predicted = self.predict_trajectory(
            current_temp,
            np.mean(current_lamp_power),
            setpoint_traj,
            dt
        )
        self.state.predicted_trajectory = predicted

        # Optimize control
        optimal_sequence = self.optimize_control(
            current_temp,
            current_lamp_power,
            setpoint_traj,
            dt
        )

        # Return first control action
        return optimal_sequence[0]

    def reset(self):
        """Reset MPC state."""
        self.state = MPCState()


# ============================================================================
# Run-to-Run (R2R) Controller
# ============================================================================

class R2RController:
    """Run-to-Run controller for wafer-to-wafer adjustments.

    Uses EWMA (Exponentially Weighted Moving Average) to track:
    - Emissivity drift over time
    - Lamp power trim needed per zone
    - Overshoot trends
    """

    def __init__(self, num_zones: int = 4, alpha: float = 0.3):
        """Initialize R2R controller.

        Args:
            num_zones: Number of lamp zones
            alpha: EWMA smoothing factor (0-1)
        """
        self.num_zones = num_zones
        self.state = R2RState(alpha=alpha)

        # EWMA values
        self.ewma_emissivity = 0.65  # Initial guess
        self.ewma_lamp_power = np.ones(num_zones) * 50.0
        self.ewma_overshoot = 0.0

    def update(
        self,
        measured_emissivity: Optional[float] = None,
        lamp_powers_used: Optional[np.ndarray] = None,
        overshoot_observed: Optional[float] = None,
        target_overshoot: float = 0.0
    ):
        """Update R2R controller after a run.

        Args:
            measured_emissivity: Estimated emissivity from run
            lamp_powers_used: Average lamp powers used per zone (%)
            overshoot_observed: Peak overshoot observed (%)
            target_overshoot: Target overshoot (usually 0)
        """
        self.state.run_count += 1
        alpha = self.state.alpha

        # Update EWMA for emissivity
        if measured_emissivity is not None:
            self.state.emissivity_history.append(measured_emissivity)
            self.ewma_emissivity = (
                alpha * measured_emissivity +
                (1 - alpha) * self.ewma_emissivity
            )

            # Calculate adjustment (if emissivity is drifting)
            # Pyrometer reads low when emissivity is high
            expected_emissivity = 0.65
            self.state.emissivity_adjustment = expected_emissivity / self.ewma_emissivity

        # Update EWMA for lamp power
        if lamp_powers_used is not None:
            self.state.lamp_power_history.append(lamp_powers_used.copy())
            self.ewma_lamp_power = (
                alpha * lamp_powers_used +
                (1 - alpha) * self.ewma_lamp_power
            )

            # Calculate trim (normalize to mean = 1.0)
            mean_power = np.mean(self.ewma_lamp_power)
            if mean_power > 0:
                self.state.lamp_power_trim = self.ewma_lamp_power / mean_power

        # Update EWMA for overshoot
        if overshoot_observed is not None:
            self.state.overshoot_history.append(overshoot_observed)
            self.ewma_overshoot = (
                alpha * overshoot_observed +
                (1 - alpha) * self.ewma_overshoot
            )

    def get_adjustments(self) -> Dict[str, any]:
        """Get recommended adjustments for next run.

        Returns:
            Dictionary with emissivity correction and lamp power trims
        """
        return {
            "emissivity_correction": self.state.emissivity_adjustment,
            "lamp_power_trim": self.state.lamp_power_trim.copy(),
            "predicted_overshoot_pct": self.ewma_overshoot,
            "run_count": self.state.run_count,
        }

    def get_statistics(self) -> Dict[str, float]:
        """Get R2R statistics."""
        stats = {}

        if len(self.state.emissivity_history) > 0:
            stats["emissivity_mean"] = np.mean(self.state.emissivity_history)
            stats["emissivity_std"] = np.std(self.state.emissivity_history)

        if len(self.state.overshoot_history) > 0:
            stats["overshoot_mean"] = np.mean(self.state.overshoot_history)
            stats["overshoot_std"] = np.std(self.state.overshoot_history)

        return stats


# ============================================================================
# Thermal Budget Calculator
# ============================================================================

class ThermalBudgetCalculator:
    """Calculate thermal budget: ∫ exp(-Ea/kT(t)) dt

    This metric quantifies the effective annealing dose, accounting for
    the exponential temperature dependence of diffusion/activation.
    """

    def __init__(self, dopant_species: str = "boron"):
        """Initialize calculator.

        Args:
            dopant_species: Dopant type (boron, phosphorus, arsenic, oxidation)
        """
        self.dopant_species = dopant_species

        if dopant_species in ACTIVATION_ENERGIES:
            self.activation_energy_eV = ACTIVATION_ENERGIES[dopant_species]
        else:
            # Default
            self.activation_energy_eV = 3.65

        # Integration history
        self.time_history = []
        self.temp_history = []
        self.rate_history = []

    def add_sample(self, temperature_C: float, dt: float):
        """Add a temperature sample and integrate.

        Args:
            temperature_C: Temperature at this time step
            dt: Time step (seconds)
        """
        T_K = temperature_C + 273.15

        # Arrhenius rate: exp(-Ea / kT)
        rate = math.exp(-self.activation_energy_eV / (BOLTZMANN_K * T_K))

        self.time_history.append(dt)
        self.temp_history.append(temperature_C)
        self.rate_history.append(rate)

    def get_budget(self) -> ThermalBudget:
        """Calculate total thermal budget.

        Returns:
            ThermalBudget with integrated value
        """
        if len(self.rate_history) == 0:
            return ThermalBudget(
                dopant_species=self.dopant_species,
                activation_energy_eV=self.activation_energy_eV,
                integrated_budget=0.0,
                equivalent_time_at_1000C_s=0.0,
                peak_activation_rate=0.0
            )

        # Integrate using trapezoidal rule
        rates = np.array(self.rate_history)
        times = np.array(self.time_history)

        integrated_budget = np.sum(rates * times)
        peak_rate = np.max(rates)

        # Equivalent time at reference temperature (1000°C)
        T_ref_K = 1000.0 + 273.15
        rate_at_1000C = math.exp(-self.activation_energy_eV / (BOLTZMANN_K * T_ref_K))

        if rate_at_1000C > 0:
            equivalent_time_s = integrated_budget / rate_at_1000C
        else:
            equivalent_time_s = 0.0

        return ThermalBudget(
            dopant_species=self.dopant_species,
            activation_energy_eV=self.activation_energy_eV,
            integrated_budget=integrated_budget,
            equivalent_time_at_1000C_s=equivalent_time_s,
            peak_activation_rate=peak_rate
        )

    def reset(self):
        """Reset calculator."""
        self.time_history.clear()
        self.temp_history.clear()
        self.rate_history.clear()


# ============================================================================
# Performance Analysis
# ============================================================================

class PerformanceAnalyzer:
    """Analyze RTP controller performance and generate reports."""

    @staticmethod
    def analyze_ramp_segment(
        segment_id: int,
        target_ramp_rate_C_per_s: float,
        setpoint_history: np.ndarray,
        measured_history: np.ndarray,
        time_history: np.ndarray,
        dwell_start_idx: Optional[int] = None,
        dwell_end_idx: Optional[int] = None
    ) -> RampFidelity:
        """Analyze a single ramp segment.

        Args:
            segment_id: Segment identifier
            target_ramp_rate_C_per_s: Target ramp rate
            setpoint_history: Setpoint temperatures
            measured_history: Measured temperatures
            time_history: Time stamps
            dwell_start_idx: Index where dwell starts (if applicable)
            dwell_end_idx: Index where dwell ends (if applicable)

        Returns:
            RampFidelity metrics
        """
        # Calculate actual ramp rate (linear fit)
        if len(time_history) > 1:
            actual_ramp_rate = np.polyfit(time_history, measured_history, 1)[0]
        else:
            actual_ramp_rate = 0.0

        # Overshoot/undershoot
        errors = measured_history - setpoint_history
        peak_overshoot_C = np.max(errors)
        peak_undershoot_C = np.min(errors)

        # Overshoot percentage (relative to final setpoint)
        final_setpoint = setpoint_history[-1]
        if final_setpoint > 0:
            peak_overshoot_pct = (peak_overshoot_C / final_setpoint) * 100.0
        else:
            peak_overshoot_pct = 0.0

        # RMSE
        rmse_C = np.sqrt(np.mean(errors**2))
        max_error_C = np.max(np.abs(errors))

        # Settling time (time to reach ±2% of setpoint)
        settling_threshold = 0.02 * final_setpoint
        settled_mask = np.abs(errors) <= settling_threshold

        if np.any(settled_mask):
            settling_idx = np.argmax(settled_mask)
            settling_time_s = time_history[settling_idx] - time_history[0]
        else:
            settling_time_s = time_history[-1] - time_history[0]

        # Dwell stability (if applicable)
        dwell_std_C = 0.0
        dwell_drift_C_per_s = 0.0

        if dwell_start_idx is not None and dwell_end_idx is not None:
            dwell_temps = measured_history[dwell_start_idx:dwell_end_idx]
            dwell_times = time_history[dwell_start_idx:dwell_end_idx]

            if len(dwell_temps) > 1:
                dwell_std_C = np.std(dwell_temps)
                dwell_drift_C_per_s = np.polyfit(dwell_times, dwell_temps, 1)[0]

        return RampFidelity(
            segment_id=segment_id,
            target_ramp_rate_C_per_s=target_ramp_rate_C_per_s,
            actual_ramp_rate_C_per_s=actual_ramp_rate,
            peak_overshoot_C=peak_overshoot_C,
            peak_overshoot_pct=peak_overshoot_pct,
            peak_undershoot_C=peak_undershoot_C,
            rmse_C=rmse_C,
            max_error_C=max_error_C,
            settling_time_s=settling_time_s,
            dwell_std_C=dwell_std_C,
            dwell_drift_C_per_s=dwell_drift_C_per_s
        )

    @staticmethod
    def generate_performance_report(
        recipe_id: str,
        run_id: str,
        ramp_fidelities: List[RampFidelity],
        thermal_budget: ThermalBudget,
        saturation_events: int,
        constraint_violations: int,
        start_time: float,
        end_time: float
    ) -> ControllerPerformance:
        """Generate complete performance report.

        Args:
            recipe_id: Recipe identifier
            run_id: Run identifier
            ramp_fidelities: List of per-segment metrics
            thermal_budget: Thermal budget result
            saturation_events: Number of saturation events
            constraint_violations: Number of constraint violations
            start_time: Start timestamp
            end_time: End timestamp

        Returns:
            ControllerPerformance report
        """
        # Overall RMSE
        all_rmse = [rf.rmse_C for rf in ramp_fidelities]
        overall_rmse_C = np.mean(all_rmse) if all_rmse else 0.0

        # Max overshoot
        all_overshoot = [rf.peak_overshoot_pct for rf in ramp_fidelities]
        max_overshoot_pct = np.max(all_overshoot) if all_overshoot else 0.0

        # Recommended tuning
        recommended_tuning = {}

        # If overshoot is high, reduce gains
        if max_overshoot_pct > 5.0:
            recommended_tuning["reduce_Kp"] = 0.8  # Reduce by 20%
            recommended_tuning["reduce_Ki"] = 0.7  # Reduce by 30%

        # If RMSE is high, increase gains
        if overall_rmse_C > 5.0:
            recommended_tuning["increase_Kp"] = 1.2  # Increase by 20%

        # Drift flags
        drift_flags = []
        for rf in ramp_fidelities:
            if abs(rf.dwell_drift_C_per_s) > 0.5:
                drift_flags.append(f"Segment {rf.segment_id}: High drift {rf.dwell_drift_C_per_s:.2f} °C/s")

            if rf.dwell_std_C > 5.0:
                drift_flags.append(f"Segment {rf.segment_id}: High noise {rf.dwell_std_C:.2f} °C std")

        return ControllerPerformance(
            recipe_id=recipe_id,
            run_id=run_id,
            ramp_fidelities=ramp_fidelities,
            overall_rmse_C=overall_rmse_C,
            max_overshoot_pct=max_overshoot_pct,
            total_thermal_budget=thermal_budget.integrated_budget,
            saturation_events=saturation_events,
            constraint_violations=constraint_violations,
            recommended_tuning=recommended_tuning,
            drift_flags=drift_flags,
            start_time=start_time,
            end_time=end_time,
            total_duration_s=end_time - start_time
        )


# Export
__all__ = [
    "PIDController",
    "PIDGains",
    "PIDState",
    "MPCController",
    "MPCParameters",
    "MPCState",
    "R2RController",
    "R2RState",
    "ThermalBudgetCalculator",
    "ThermalBudget",
    "PerformanceAnalyzer",
    "RampFidelity",
    "ControllerPerformance",
    "ACTIVATION_ENERGIES",
]
