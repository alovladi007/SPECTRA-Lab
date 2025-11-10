"""Advanced control algorithms for RTP and Ion Implantation."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import signal, optimize
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PIDConfig:
    """PID controller configuration."""
    Kp: float = 1.0  # Proportional gain
    Ki: float = 0.1  # Integral gain
    Kd: float = 0.01  # Derivative gain
    setpoint: float = 0.0
    output_min: float = -100.0
    output_max: float = 100.0
    integral_min: float = -100.0
    integral_max: float = 100.0
    derivative_filter_tau: float = 0.1  # Low-pass filter time constant
    sample_time: float = 0.1  # seconds


class PIDController:
    """Enhanced PID controller with anti-windup and derivative filtering."""
    
    def __init__(self, config: PIDConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset controller state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.derivative_filtered = 0.0
        self.last_output = 0.0
        self.last_time = None
        
    def update(self, measurement: float, dt: Optional[float] = None) -> float:
        """Update controller and return control output."""
        if dt is None:
            dt = self.config.sample_time
            
        # Calculate error
        error = self.config.setpoint - measurement
        
        # Proportional term
        P = self.config.Kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        
        # Apply integral limits (anti-windup)
        self.integral = np.clip(
            self.integral,
            self.config.integral_min,
            self.config.integral_max
        )
        I = self.config.Ki * self.integral
        
        # Derivative term with filtering
        if self.last_error is not None:
            derivative_raw = (error - self.last_error) / dt
            
            # Low-pass filter on derivative
            alpha = dt / (self.config.derivative_filter_tau + dt)
            self.derivative_filtered = (
                alpha * derivative_raw + 
                (1 - alpha) * self.derivative_filtered
            )
        else:
            self.derivative_filtered = 0.0
            
        D = self.config.Kd * self.derivative_filtered
        
        # Calculate output
        output = P + I + D
        
        # Apply output limits
        output_limited = np.clip(
            output,
            self.config.output_min,
            self.config.output_max
        )
        
        # Back-calculation anti-windup
        if output != output_limited:
            # Reduce integral if output was saturated
            self.integral -= (output - output_limited) * dt / self.config.Ki
            
        self.last_error = error
        self.last_output = output_limited
        
        return output_limited
        
    def set_setpoint(self, setpoint: float):
        """Update setpoint."""
        self.config.setpoint = setpoint
        
    def set_gains(self, Kp: float, Ki: float, Kd: float):
        """Update controller gains."""
        self.config.Kp = Kp
        self.config.Ki = Ki
        self.config.Kd = Kd
        
    def get_state(self) -> Dict[str, float]:
        """Get controller state."""
        return {
            'setpoint': self.config.setpoint,
            'error': self.last_error if self.last_error is not None else 0.0,
            'integral': self.integral,
            'derivative': self.derivative_filtered,
            'output': self.last_output,
            'Kp': self.config.Kp,
            'Ki': self.config.Ki,
            'Kd': self.config.Kd,
        }


@dataclass
class MPCConfig:
    """Model Predictive Control configuration."""
    prediction_horizon: int = 10
    control_horizon: int = 3
    sample_time: float = 0.1
    state_weight: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    control_weight: float = 0.1
    output_min: float = -100.0
    output_max: float = 100.0
    state_min: Optional[np.ndarray] = None
    state_max: Optional[np.ndarray] = None


class MPCController:
    """Model Predictive Controller for optimal control."""
    
    def __init__(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, config: MPCConfig):
        """
        Initialize MPC controller.
        
        Args:
            A: State transition matrix (n x n)
            B: Control input matrix (n x m)
            C: Output matrix (p x n)
            config: MPC configuration
        """
        self.A = A
        self.B = B
        self.C = C
        self.config = config
        
        self.n_states = A.shape[0]
        self.n_inputs = B.shape[1] if len(B.shape) > 1 else 1
        self.n_outputs = C.shape[0] if len(C.shape) > 1 else 1
        
        # Build prediction matrices
        self._build_prediction_matrices()
        
        # State estimation (simple observer)
        self.x_est = np.zeros(self.n_states)
        
    def _build_prediction_matrices(self):
        """Build matrices for prediction."""
        Np = self.config.prediction_horizon
        Nc = self.config.control_horizon
        
        # Build Phi matrix (prediction of states)
        self.Phi = np.zeros((self.n_states * Np, self.n_states))
        for i in range(Np):
            self.Phi[i*self.n_states:(i+1)*self.n_states, :] = np.linalg.matrix_power(self.A, i+1)
            
        # Build Theta matrix (effect of control inputs)
        self.Theta = np.zeros((self.n_states * Np, self.n_inputs * Nc))
        for i in range(Np):
            for j in range(min(i+1, Nc)):
                if i-j >= 0:
                    self.Theta[i*self.n_states:(i+1)*self.n_states, 
                              j*self.n_inputs:(j+1)*self.n_inputs] = \
                        np.linalg.matrix_power(self.A, i-j) @ self.B
                        
    def update(self, measurement: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Update MPC and calculate control action.
        
        Args:
            measurement: Current measurement vector
            reference: Reference trajectory over prediction horizon
            
        Returns:
            Control action
        """
        # Simple state estimation (could use Kalman filter)
        y_est = self.C @ self.x_est
        innovation = measurement - y_est
        L = 0.5  # Observer gain (simplified)
        self.x_est = self.A @ self.x_est + L * innovation
        
        # Solve optimization problem
        u_opt = self._solve_qp(self.x_est, reference)
        
        # Apply first control action
        u = u_opt[:self.n_inputs]
        
        # Update state estimate with control
        self.x_est = self.A @ self.x_est + self.B.reshape(-1) * u[0]
        
        return u
        
    def _solve_qp(self, x0: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Solve quadratic programming problem."""
        Nc = self.config.control_horizon
        
        # Build weight matrices
        Q = np.kron(np.eye(self.config.prediction_horizon), 
                    np.diag(self.config.state_weight))
        R = self.config.control_weight * np.eye(self.n_inputs * Nc)
        
        # Prediction
        x_pred = self.Phi @ x0
        
        # QP matrices
        H = self.Theta.T @ Q @ self.Theta + R
        f = -2 * self.Theta.T @ Q @ (reference - x_pred)
        
        # Constraints
        bounds = [(self.config.output_min, self.config.output_max)] * (self.n_inputs * Nc)
        
        # Solve
        result = optimize.minimize(
            lambda u: 0.5 * u.T @ H @ u + f.T @ u,
            np.zeros(self.n_inputs * Nc),
            method='SLSQP',
            bounds=bounds
        )
        
        return result.x
        
    def get_prediction(self, u_sequence: np.ndarray) -> np.ndarray:
        """Get predicted trajectory for given control sequence."""
        x_pred = self.Phi @ self.x_est + self.Theta @ u_sequence
        y_pred = np.array([self.C @ x_pred[i*self.n_states:(i+1)*self.n_states] 
                          for i in range(self.config.prediction_horizon)])
        return y_pred


@dataclass
class R2RConfig:
    """Run-to-Run control configuration."""
    ewma_lambda: float = 0.3  # EWMA filter parameter
    gain: float = 0.5  # Controller gain
    deadband: float = 0.1  # Control deadband
    max_adjustment: float = 10.0  # Maximum adjustment per run
    target_window: int = 10  # Runs to consider for target calculation
    enable_drift_compensation: bool = True
    enable_context_switching: bool = True


class R2RController:
    """Run-to-Run controller with EWMA filtering and drift compensation."""
    
    def __init__(self, config: R2RConfig):
        self.config = config
        self.reset()
        
    def reset(self):
        """Reset controller state."""
        self.run_history = deque(maxlen=100)
        self.ewma_estimate = None
        self.drift_estimate = 0.0
        self.context_models = {}  # For different product types
        self.current_context = 'default'
        
    def update(self, 
               measurement: float, 
               target: float,
               recipe_adjustment: float = 0.0,
               context: Optional[str] = None) -> float:
        """
        Update R2R controller and calculate recipe adjustment.
        
        Args:
            measurement: Post-process measurement
            target: Target value
            recipe_adjustment: Current recipe adjustment
            context: Context identifier (e.g., product type)
            
        Returns:
            New recipe adjustment
        """
        if context and self.config.enable_context_switching:
            self.current_context = context
            
        # Store run data
        self.run_history.append({
            'measurement': measurement,
            'target': target,
            'adjustment': recipe_adjustment,
            'context': self.current_context,
            'error': target - measurement
        })
        
        # Update EWMA estimate
        if self.ewma_estimate is None:
            self.ewma_estimate = measurement
        else:
            self.ewma_estimate = (
                self.config.ewma_lambda * measurement + 
                (1 - self.config.ewma_lambda) * self.ewma_estimate
            )
            
        # Calculate error
        error = target - self.ewma_estimate
        
        # Check deadband
        if abs(error) < self.config.deadband:
            return recipe_adjustment
            
        # Update drift estimate if enabled
        if self.config.enable_drift_compensation and len(self.run_history) > 5:
            self._update_drift_estimate()
            
        # Calculate control action
        adjustment = recipe_adjustment + self.config.gain * error
        
        # Add drift compensation
        if self.config.enable_drift_compensation:
            adjustment += self.drift_estimate
            
        # Apply limits
        adjustment = np.clip(
            adjustment,
            recipe_adjustment - self.config.max_adjustment,
            recipe_adjustment + self.config.max_adjustment
        )
        
        return adjustment
        
    def _update_drift_estimate(self):
        """Estimate process drift using recent history."""
        recent_runs = list(self.run_history)[-20:]
        
        if len(recent_runs) < 5:
            return
            
        # Extract errors for current context
        if self.config.enable_context_switching:
            errors = [r['error'] for r in recent_runs 
                     if r['context'] == self.current_context]
        else:
            errors = [r['error'] for r in recent_runs]
            
        if len(errors) < 5:
            return
            
        # Simple linear regression for drift
        x = np.arange(len(errors))
        y = np.array(errors)
        
        # Calculate slope (drift rate)
        slope = np.polyfit(x, y, 1)[0]
        
        # Update drift estimate (filtered)
        alpha = 0.1  # Drift filter constant
        self.drift_estimate = alpha * slope + (1 - alpha) * self.drift_estimate
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get controller statistics."""
        if not self.run_history:
            return {}
            
        recent_runs = list(self.run_history)[-self.config.target_window:]
        errors = [r['error'] for r in recent_runs]
        
        return {
            'ewma_estimate': self.ewma_estimate,
            'drift_estimate': self.drift_estimate,
            'mean_error': np.mean(errors) if errors else 0.0,
            'std_error': np.std(errors) if errors else 0.0,
            'runs_processed': len(self.run_history),
            'current_context': self.current_context
        }
        
    def predict_next(self, recipe_adjustment: float) -> float:
        """Predict next measurement given recipe adjustment."""
        if self.ewma_estimate is None:
            return 0.0
            
        # Simple linear prediction
        prediction = self.ewma_estimate + recipe_adjustment / self.config.gain
        
        # Add drift prediction
        if self.config.enable_drift_compensation:
            prediction -= self.drift_estimate
            
        return prediction


class AdaptiveController:
    """Adaptive controller that switches between PID, MPC based on conditions."""
    
    def __init__(self):
        # Initialize sub-controllers
        self.pid = PIDController(PIDConfig())
        self.mpc = None  # Initialized when model is available
        self.r2r = R2RController(R2RConfig())
        
        # Controller selection
        self.active_controller = 'PID'
        self.performance_metrics = {}
        
    def select_controller(self, 
                         process_state: Dict[str, Any],
                         performance_history: List[Dict]) -> str:
        """Select best controller based on current conditions."""
        
        # Simple rule-based selection
        if process_state.get('transient', False):
            # Use PID for fast transient response
            return 'PID'
        elif process_state.get('model_available', False) and \
             process_state.get('constraints_active', False):
            # Use MPC when model is available and constraints matter
            return 'MPC'
        elif process_state.get('steady_state', False) and \
             len(performance_history) > 10:
            # Use R2R for steady-state drift compensation
            return 'R2R'
        else:
            # Default to PID
            return 'PID'
            
    def update(self, 
               measurement: float,
               setpoint: float,
               process_state: Dict[str, Any]) -> Tuple[float, str]:
        """
        Update adaptive controller.
        
        Returns:
            Tuple of (control_output, active_controller)
        """
        # Select controller
        self.active_controller = self.select_controller(
            process_state,
            list(self.performance_metrics.values())
        )
        
        # Execute selected controller
        if self.active_controller == 'PID':
            self.pid.set_setpoint(setpoint)
            output = self.pid.update(measurement)
        elif self.active_controller == 'MPC' and self.mpc:
            reference = np.array([setpoint] * self.mpc.config.prediction_horizon)
            output = self.mpc.update(np.array([measurement]), reference)[0]
        elif self.active_controller == 'R2R':
            output = self.r2r.update(measurement, setpoint)
        else:
            # Fallback to PID
            self.pid.set_setpoint(setpoint)
            output = self.pid.update(measurement)
            
        # Store performance metrics
        self.performance_metrics[len(self.performance_metrics)] = {
            'controller': self.active_controller,
            'measurement': measurement,
            'setpoint': setpoint,
            'output': output,
            'error': abs(setpoint - measurement)
        }
        
        return output, self.active_controller
        
    def auto_tune_pid(self, 
                      step_response: List[float],
                      sample_time: float) -> Tuple[float, float, float]:
        """Auto-tune PID gains using relay feedback or step response."""
        
        # Simplified Ziegler-Nichols from step response
        y = np.array(step_response)
        
        # Find inflection point
        dy = np.diff(y)
        inflection_idx = np.argmax(dy)
        
        # Estimate delay and time constant
        L = inflection_idx * sample_time  # Delay
        T = len(y) * sample_time / 3  # Time constant (rough estimate)
        
        # Ziegler-Nichols tuning rules
        Kp = 1.2 * T / L
        Ki = Kp / (2 * L)
        Kd = Kp * L * 0.5
        
        # Update PID gains
        self.pid.set_gains(Kp, Ki, Kd)
        
        return Kp, Ki, Kd
