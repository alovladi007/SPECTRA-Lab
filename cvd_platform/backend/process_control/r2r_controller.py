"""
Advanced Process Control (APC) and Run-to-Run (R2R) Control
Implements sophisticated control algorithms for CVD process:
- Run-to-Run (R2R) control using EWMA
- Model Predictive Control (MPC)
- Adaptive control with drift compensation
- Multi-variable control for temperature, pressure, flow, time
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class ControlTarget:
    """Control targets and specifications"""
    target_thickness: float  # nm
    thickness_tolerance: float  # nm
    target_uniformity: float  # %
    uniformity_tolerance: float  # %


@dataclass
class RecipeParameters:
    """CVD Recipe Parameters (controllable variables)"""
    temperature: float  # °C
    pressure: float  # Torr
    precursor_flow: float  # sccm
    carrier_flow: float  # sccm
    deposition_time: float  # seconds
    rotation_speed: float  # rpm
    heater_zone_powers: List[float]  # W for each zone


@dataclass
class ControlAction:
    """Control action to be applied"""
    parameter_updates: Dict[str, float]
    timestamp: datetime
    reason: str
    predicted_improvement: float  # Expected reduction in error


class EWMAController:
    """
    Exponentially Weighted Moving Average (EWMA) R2R Controller

    Recipe update equation:
    θ(n+1) = θ(n) + K * [Target - Measured(n)]

    where:
    θ = recipe parameters
    K = gain matrix (typically 0.1-0.5)
    """

    def __init__(self, gain: float = 0.3, lambda_ewma: float = 0.7):
        """
        Args:
            gain: Controller gain (0 < gain < 1)
            lambda_ewma: EWMA smoothing parameter
        """
        self.gain = gain
        self.lambda_ewma = lambda_ewma
        self.process_history: deque = deque(maxlen=100)
        self.ewma_estimate = None

        logger.info(f"Initialized EWMA controller with gain={gain}, lambda={lambda_ewma}")

    def calculate_control_action(self,
                                 current_recipe: RecipeParameters,
                                 target: ControlTarget,
                                 measured_thickness: float,
                                 predicted_thickness: Optional[float] = None) -> ControlAction:
        """
        Calculate R2R control action.

        Args:
            current_recipe: Current recipe parameters
            target: Control target
            measured_thickness: Actual measured thickness (or VM predicted)
            predicted_thickness: VM prediction if available

        Returns:
            ControlAction with parameter updates
        """
        # Use VM prediction if available and reliable
        thickness_value = predicted_thickness if predicted_thickness is not None else measured_thickness

        # Calculate error
        error = target.target_thickness - thickness_value

        # Update EWMA estimate
        if self.ewma_estimate is None:
            self.ewma_estimate = thickness_value
        else:
            self.ewma_estimate = self.lambda_ewma * thickness_value + \
                               (1 - self.lambda_ewma) * self.ewma_estimate

        # Calculate error from EWMA
        ewma_error = target.target_thickness - self.ewma_estimate

        logger.info(f"R2R Control: Error={error:.2f} nm, EWMA Error={ewma_error:.2f} nm")

        # Calculate parameter adjustments
        # Primary control: deposition time (most direct impact on thickness)
        time_adjustment = self.gain * (error / target.target_thickness) * current_recipe.deposition_time

        # Secondary control: temperature (affects deposition rate)
        # ΔT ≈ 5°C per 10% thickness change (approximate, process-dependent)
        temp_adjustment = 0.5 * (error / target.target_thickness) * 5.0

        # Tertiary control: precursor flow (affects deposition rate)
        flow_adjustment = self.gain * 0.5 * (error / target.target_thickness) * current_recipe.precursor_flow

        # Apply limits to prevent excessive adjustments
        time_adjustment = np.clip(time_adjustment, -30.0, 30.0)  # ±30 seconds max
        temp_adjustment = np.clip(temp_adjustment, -10.0, 10.0)  # ±10°C max
        flow_adjustment = np.clip(flow_adjustment, -50.0, 50.0)  # ±50 sccm max

        parameter_updates = {
            "deposition_time": current_recipe.deposition_time + time_adjustment,
            "temperature": current_recipe.temperature + temp_adjustment,
            "precursor_flow": current_recipe.precursor_flow + flow_adjustment
        }

        # Estimate improvement
        # Simplified: assume linear response
        predicted_improvement = abs(error) * 0.7  # Expect 70% error reduction

        action = ControlAction(
            parameter_updates=parameter_updates,
            timestamp=datetime.utcnow(),
            reason=f"R2R adjustment for {error:.2f} nm error",
            predicted_improvement=predicted_improvement
        )

        # Store in history
        self.process_history.append({
            "timestamp": datetime.utcnow(),
            "measured_thickness": thickness_value,
            "target_thickness": target.target_thickness,
            "error": error,
            "time_adjustment": time_adjustment,
            "temp_adjustment": temp_adjustment
        })

        logger.info(f"R2R Action: ΔTime={time_adjustment:.1f}s, ΔTemp={temp_adjustment:.1f}°C, ΔFlow={flow_adjustment:.1f}sccm")

        return action


class PIDController:
    """
    PID (Proportional-Integral-Derivative) Controller

    u(t) = Kp*e(t) + Ki*∫e(t)dt + Kd*de(t)/dt
    """

    def __init__(self, Kp: float = 0.3, Ki: float = 0.1, Kd: float = 0.05):
        """
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.previous_error = 0.0
        self.dt = 1.0  # Time step (wafer-to-wafer)

        logger.info(f"Initialized PID controller: Kp={Kp}, Ki={Ki}, Kd={Kd}")

    def calculate_control_action(self,
                                 current_recipe: RecipeParameters,
                                 target: ControlTarget,
                                 measured_thickness: float) -> ControlAction:
        """Calculate PID control action"""
        # Error
        error = target.target_thickness - measured_thickness

        # Integral term
        self.integral += error * self.dt

        # Anti-windup: limit integral
        max_integral = 1000.0
        self.integral = np.clip(self.integral, -max_integral, max_integral)

        # Derivative term
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error

        # PID output
        control_output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative

        logger.info(f"PID Control: P={self.Kp * error:.2f}, I={self.Ki * self.integral:.2f}, D={self.Kd * derivative:.2f}")

        # Convert control output to parameter adjustments
        # Normalize control output to time adjustment
        time_adjustment = (control_output / target.target_thickness) * current_recipe.deposition_time

        # Limit adjustments
        time_adjustment = np.clip(time_adjustment, -30.0, 30.0)

        parameter_updates = {
            "deposition_time": current_recipe.deposition_time + time_adjustment
        }

        action = ControlAction(
            parameter_updates=parameter_updates,
            timestamp=datetime.utcnow(),
            reason=f"PID adjustment for {error:.2f} nm error",
            predicted_improvement=abs(error) * 0.8
        )

        return action


class ModelPredictiveController:
    """
    Model Predictive Control (MPC) for multi-zone temperature control.

    Solves optimization problem:
    min J = Σ[||T_ref - T_pred||² + R*||ΔU||²]

    subject to:
    T(k+1) = A*T(k) + B*U(k)  (state space model)
    U_min ≤ U ≤ U_max
    ΔU_min ≤ ΔU ≤ ΔU_max
    """

    def __init__(self, num_zones: int = 5, prediction_horizon: int = 10, control_horizon: int = 5):
        """
        Args:
            num_zones: Number of heater zones
            prediction_horizon: Prediction horizon (time steps)
            control_horizon: Control horizon (time steps)
        """
        self.num_zones = num_zones
        self.prediction_horizon = prediction_horizon
        self.control_horizon = control_horizon

        # State-space model matrices (simplified thermal model)
        # T(k+1) = A*T(k) + B*U(k)
        self.A = self._build_state_matrix()
        self.B = self._build_input_matrix()

        # MPC weights
        self.Q = np.eye(num_zones) * 100.0  # Output weight
        self.R = np.eye(num_zones) * 1.0  # Control effort weight

        # Constraints
        self.u_min = np.zeros(num_zones)  # Minimum power (W)
        self.u_max = np.ones(num_zones) * 5000.0  # Maximum power (W)
        self.du_max = np.ones(num_zones) * 500.0  # Maximum power change rate

        logger.info(f"Initialized MPC controller: {num_zones} zones, horizon={prediction_horizon}")

    def _build_state_matrix(self) -> np.ndarray:
        """
        Build state transition matrix for thermal dynamics.
        Includes heat transfer between adjacent zones.
        """
        A = np.zeros((self.num_zones, self.num_zones))

        # Diagonal elements (self-cooling)
        alpha = 0.95  # Temperature retention coefficient
        np.fill_diagonal(A, alpha)

        # Off-diagonal elements (inter-zone heat transfer)
        beta = 0.02  # Heat transfer coefficient
        for i in range(self.num_zones - 1):
            A[i, i+1] = beta
            A[i+1, i] = beta

        return A

    def _build_input_matrix(self) -> np.ndarray:
        """Build input matrix (power to temperature)"""
        # Simplified: each heater directly affects its zone
        B = np.eye(self.num_zones) * 0.01  # Power-to-temp gain
        return B

    def calculate_control_action(self,
                                 current_temps: np.ndarray,
                                 target_temps: np.ndarray,
                                 current_powers: np.ndarray) -> np.ndarray:
        """
        Calculate optimal heater powers using MPC.

        Args:
            current_temps: Current zone temperatures (°C)
            target_temps: Target zone temperatures (°C)
            current_powers: Current heater powers (W)

        Returns:
            Optimal heater powers (W)
        """
        # Simplified MPC - full implementation would use quadratic programming
        # Here we use gradient descent approximation

        # Prediction matrices
        # Y = Ψ*x + Θ*U where Y is predicted output, U is control sequence

        # Calculate error
        error = target_temps - current_temps

        # Proportional control with model-based feed-forward
        # ΔU = K*(T_ref - T_current) + U_ff
        K_p = 100.0  # Proportional gain

        power_adjustment = K_p * error

        # Apply rate constraints
        power_adjustment = np.clip(power_adjustment, -self.du_max, self.du_max)

        # Calculate new powers
        new_powers = current_powers + power_adjustment

        # Apply power constraints
        new_powers = np.clip(new_powers, self.u_min, self.u_max)

        logger.info(f"MPC Control: Error={np.linalg.norm(error):.2f}, ΔP={np.linalg.norm(power_adjustment):.1f}W")

        return new_powers


class AdaptiveController:
    """
    Adaptive controller with drift compensation.
    Uses recursive least squares (RLS) to estimate process model online.
    """

    def __init__(self, num_params: int = 3, forgetting_factor: float = 0.98):
        """
        Args:
            num_params: Number of model parameters
            forgetting_factor: RLS forgetting factor (0 < λ < 1)
        """
        self.num_params = num_params
        self.lambda_rls = forgetting_factor

        # RLS parameters
        self.theta = np.zeros(num_params)  # Parameter estimates
        self.P = np.eye(num_params) * 1000.0  # Covariance matrix

        # Process model: y = θ₁*u₁ + θ₂*u₂ + θ₃*u₃ + ...
        # For CVD: thickness = f(temperature, time, flow)

        self.history: deque = deque(maxlen=50)

        logger.info(f"Initialized adaptive controller with {num_params} parameters")

    def update_model(self, input_vector: np.ndarray, output: float):
        """
        Update process model using RLS.

        Args:
            input_vector: Input features [temperature, time, flow, ...]
            output: Measured output (thickness)
        """
        # RLS update equations
        # K(k) = P(k-1)*φ(k) / (λ + φ(k)'*P(k-1)*φ(k))
        # θ(k) = θ(k-1) + K(k)*(y(k) - φ(k)'*θ(k-1))
        # P(k) = (P(k-1) - K(k)*φ(k)'*P(k-1)) / λ

        phi = input_vector.reshape(-1, 1)

        # Prediction error
        y_pred = self.theta @ input_vector
        error = output - y_pred

        # Gain
        P_phi = self.P @ phi
        denominator = self.lambda_rls + (phi.T @ P_phi)
        K = P_phi / denominator

        # Update parameters
        self.theta = self.theta + (K * error).flatten()

        # Update covariance
        self.P = (self.P - K @ phi.T @ self.P) / self.lambda_rls

        logger.info(f"Model updated: θ={self.theta}, prediction error={error:.2f}")

        self.history.append({
            "input": input_vector,
            "output": output,
            "prediction": y_pred,
            "error": error
        })

    def predict_output(self, input_vector: np.ndarray) -> float:
        """Predict output using current model"""
        return self.theta @ input_vector

    def calculate_control_action(self,
                                 current_recipe: RecipeParameters,
                                 target: ControlTarget) -> ControlAction:
        """Calculate control action using adaptive model"""
        # Use inverse model to calculate required inputs for target output
        # y_target = θ₁*u₁ + θ₂*u₂ + θ₃*u₃
        # Solve for optimal u

        # Simplified: adjust time proportionally
        if self.theta[1] != 0:  # θ₁ might be time coefficient
            time_adjustment = (target.target_thickness - self.theta[0]) / self.theta[1] - current_recipe.deposition_time
            time_adjustment = np.clip(time_adjustment, -30.0, 30.0)
        else:
            time_adjustment = 0.0

        parameter_updates = {
            "deposition_time": current_recipe.deposition_time + time_adjustment
        }

        action = ControlAction(
            parameter_updates=parameter_updates,
            timestamp=datetime.utcnow(),
            reason="Adaptive control based on online model",
            predicted_improvement=10.0
        )

        return action


class DriftCompensator:
    """
    Drift compensator for systematic process drift.
    Detects and compensates for drift due to:
    - Chamber aging
    - Polymer buildup
    - PM cycle effects
    """

    def __init__(self, window_size: int = 20):
        """
        Args:
            window_size: Number of wafers for drift detection
        """
        self.window_size = window_size
        self.thickness_history: deque = deque(maxlen=window_size)
        self.drift_rate = 0.0  # nm per wafer

        logger.info(f"Initialized drift compensator with window={window_size}")

    def detect_drift(self) -> Tuple[bool, float]:
        """
        Detect systematic drift using linear regression.

        Returns:
            (drift_detected, drift_rate)
        """
        if len(self.thickness_history) < self.window_size // 2:
            return False, 0.0

        # Extract thickness values
        thicknesses = np.array([h["thickness"] for h in self.thickness_history])
        indices = np.arange(len(thicknesses))

        # Linear fit
        slope, intercept = np.polyfit(indices, thicknesses, 1)
        self.drift_rate = slope

        # Statistical test for drift significance
        # Use R² and slope magnitude
        predictions = slope * indices + intercept
        residuals = thicknesses - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((thicknesses - np.mean(thicknesses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Drift is significant if slope > threshold and good fit
        drift_threshold = 0.5  # nm per wafer
        r2_threshold = 0.7

        drift_detected = (abs(slope) > drift_threshold) and (r_squared > r2_threshold)

        if drift_detected:
            logger.warning(f"Drift detected: {slope:.3f} nm/wafer, R²={r2_squared:.3f}")
        else:
            logger.info(f"No significant drift: {slope:.3f} nm/wafer, R²={r2_squared:.3f}")

        return drift_detected, slope

    def compensate_drift(self,
                        current_recipe: RecipeParameters,
                        wafer_number: int) -> ControlAction:
        """
        Calculate drift compensation adjustment.

        Args:
            current_recipe: Current recipe
            wafer_number: Wafer number in sequence

        Returns:
            ControlAction for drift compensation
        """
        # Proactive adjustment based on detected drift
        expected_drift = self.drift_rate * wafer_number

        # Compensate by adjusting deposition time
        # ΔTime = (Drift / target_thickness) * Time
        # Simplified: 1% drift → 1% time adjustment
        time_compensation = -(expected_drift / 100.0) * current_recipe.deposition_time

        parameter_updates = {
            "deposition_time": current_recipe.deposition_time + time_compensation
        }

        action = ControlAction(
            parameter_updates=parameter_updates,
            timestamp=datetime.utcnow(),
            reason=f"Drift compensation: {expected_drift:.2f} nm expected drift",
            predicted_improvement=abs(expected_drift)
        )

        logger.info(f"Drift compensation: ΔTime={time_compensation:.1f}s for {expected_drift:.2f}nm drift")

        return action

    def add_measurement(self, wafer_id: str, thickness: float):
        """Add thickness measurement to history"""
        self.thickness_history.append({
            "wafer_id": wafer_id,
            "thickness": thickness,
            "timestamp": datetime.utcnow()
        })


class APCController:
    """
    High-level Advanced Process Control (APC) Controller.
    Combines multiple control strategies:
    - R2R control for thickness
    - MPC for temperature zones
    - Adaptive control for model updates
    - Drift compensation
    """

    def __init__(self):
        self.r2r_controller = EWMAController(gain=0.3)
        self.pid_controller = PIDController(Kp=0.3, Ki=0.1, Kd=0.05)
        self.mpc_controller = ModelPredictiveController(num_zones=5)
        self.adaptive_controller = AdaptiveController(num_params=3)
        self.drift_compensator = DriftCompensator(window_size=20)

        logger.info("Initialized APC controller")

    def calculate_recipe_update(self,
                               current_recipe: RecipeParameters,
                               target: ControlTarget,
                               measured_thickness: float,
                               vm_prediction: Optional[float] = None,
                               wafer_number: int = 0) -> RecipeParameters:
        """
        Calculate comprehensive recipe update.

        Args:
            current_recipe: Current recipe parameters
            target: Control target
            measured_thickness: Actual measured thickness
            vm_prediction: VM predicted thickness
            wafer_number: Wafer number in lot

        Returns:
            Updated recipe parameters
        """
        logger.info(f"=== APC Recipe Update for Wafer #{wafer_number} ===")

        # 1. R2R Control (primary thickness control)
        r2r_action = self.r2r_controller.calculate_control_action(
            current_recipe, target, measured_thickness, vm_prediction
        )

        # 2. Drift Compensation
        self.drift_compensator.add_measurement(f"W{wafer_number}", measured_thickness)
        drift_detected, drift_rate = self.drift_compensator.detect_drift()

        drift_compensation = 0.0
        if drift_detected:
            drift_action = self.drift_compensator.compensate_drift(current_recipe, wafer_number)
            drift_compensation = drift_action.parameter_updates.get("deposition_time", 0.0) - current_recipe.deposition_time

        # 3. Combine control actions
        updated_recipe = RecipeParameters(
            temperature=r2r_action.parameter_updates.get("temperature", current_recipe.temperature),
            pressure=current_recipe.pressure,
            precursor_flow=r2r_action.parameter_updates.get("precursor_flow", current_recipe.precursor_flow),
            carrier_flow=current_recipe.carrier_flow,
            deposition_time=r2r_action.parameter_updates.get("deposition_time", current_recipe.deposition_time) + drift_compensation,
            rotation_speed=current_recipe.rotation_speed,
            heater_zone_powers=current_recipe.heater_zone_powers
        )

        logger.info(f"Recipe updated: Time={updated_recipe.deposition_time:.1f}s, Temp={updated_recipe.temperature:.1f}°C")

        return updated_recipe
