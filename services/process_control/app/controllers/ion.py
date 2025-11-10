"""Ion Implantation control algorithms.

Provides dose integration, scan uniformity correction, R2R control,
and fault detection & classification (FDC) for ion implantation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Deque
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DoseIntegrationResult:
    """Result of dose integration calculation."""
    current_dose_cm2: float
    target_dose_cm2: float
    percent_complete: float
    integrated_charge_C: float
    elapsed_time_s: float
    dose_rate_cm2_per_s: float
    uniformity_correction_factor: float
    beam_current_avg_mA: float


@dataclass
class UniformityMap:
    """2D dose uniformity map."""
    x_positions_mm: np.ndarray
    y_positions_mm: np.ndarray
    relative_dose: np.ndarray  # Normalized to mean = 1.0
    mean_dose: float
    std_dose: float
    uniformity_pct: float  # (1 - 3σ/mean) × 100%
    edge_rolloff_pct: float
    measurement_timestamp: float


@dataclass
class ScanCorrectionParameters:
    """Scan pattern correction parameters."""
    x_amplitude_correction: float  # Multiplicative correction
    y_amplitude_correction: float
    x_steering_offset_mm: float  # Additive offset
    y_steering_offset_mm: float
    dwell_time_map: Optional[np.ndarray] = None  # 2D dwell time corrections
    correction_timestamp: float = 0.0


@dataclass
class R2RState:
    """Run-to-run control state."""
    wafer_count: int
    dose_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    uniformity_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    tilt_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    twist_history: Deque[float] = field(default_factory=lambda: deque(maxlen=10))
    recommended_dose_adjustment: float = 1.0
    recommended_tilt_deg: float = 7.0
    recommended_twist_deg: float = 0.0


@dataclass
class BeamDriftDetection:
    """Beam drift detection result."""
    is_drift_detected: bool
    drift_magnitude_mm: float
    drift_rate_mm_per_s: float
    drift_direction_deg: float  # Angle from +X axis
    recommended_action: str  # "continue", "compensate", "pause"
    steering_correction_x_mm: float
    steering_correction_y_mm: float


@dataclass
class FDCAlert:
    """Fault detection and classification alert."""
    alert_type: str  # "beam_loss", "vacuum_excursion", "dose_deviation", "drift"
    severity: str  # "warning", "error", "critical"
    message: str
    timestamp: float
    recommended_action: str
    parameters: Dict


# ============================================================================
# Dose Integrator
# ============================================================================

class DoseIntegrator:
    """
    Dose integration with area and scan corrections.

    Implements: Q = ∫I(t)dt / A with scan pattern corrections.
    """

    def __init__(
        self,
        wafer_area_cm2: float,
        charge_state: int = 1,
        integration_window_s: float = 0.1
    ):
        """
        Initialize dose integrator.

        Args:
            wafer_area_cm2: Active implant area
            charge_state: Ion charge state (usually +1)
            integration_window_s: Integration time window
        """
        self.wafer_area_cm2 = wafer_area_cm2
        self.charge_state = charge_state
        self.integration_window_s = integration_window_s

        # Integration state
        self.integrated_charge_C = 0.0
        self.start_time: Optional[float] = None
        self.beam_current_history: List[Tuple[float, float]] = []  # (time, current)

        # Corrections
        self.area_correction = 1.0
        self.scan_correction = 1.0

    def start(self, timestamp: float = None):
        """Start dose integration."""
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        self.start_time = timestamp
        self.integrated_charge_C = 0.0
        self.beam_current_history.clear()
        logger.info(f"Dose integration started at t={timestamp:.3f}")

    def integrate(
        self,
        beam_current_mA: float,
        timestamp: float = None,
        dt: float = None
    ) -> float:
        """
        Integrate dose for one time step.

        Args:
            beam_current_mA: Beam current in mA
            timestamp: Current timestamp
            dt: Time step (if not using timestamp)

        Returns:
            Current integrated charge (Coulombs)
        """
        if self.start_time is None:
            raise RuntimeError("Integration not started. Call start() first.")

        if timestamp is None:
            timestamp = datetime.now().timestamp()

        # Store history
        self.beam_current_history.append((timestamp, beam_current_mA))

        # Calculate time step
        if dt is None:
            if len(self.beam_current_history) > 1:
                dt = timestamp - self.beam_current_history[-2][0]
            else:
                dt = self.integration_window_s

        # Integrate: Q = I × t (convert mA to A)
        charge_increment = (beam_current_mA * 1e-3) * dt
        self.integrated_charge_C += charge_increment

        return self.integrated_charge_C

    def get_dose(self, apply_corrections: bool = True) -> float:
        """
        Get current dose in ions/cm².

        Args:
            apply_corrections: Apply area and scan corrections

        Returns:
            Dose in ions/cm²
        """
        # Q (Coulombs) → number of ions → ions/cm²
        elementary_charge = 1.602e-19  # Coulombs per ion

        num_ions = self.integrated_charge_C / (elementary_charge * self.charge_state)

        # Dose = ions / area
        dose_cm2 = num_ions / self.wafer_area_cm2

        # Apply corrections
        if apply_corrections:
            dose_cm2 *= self.area_correction * self.scan_correction

        return dose_cm2

    def calculate_dose_result(self, target_dose_cm2: float) -> DoseIntegrationResult:
        """Calculate complete dose integration result."""
        current_dose = self.get_dose()
        elapsed = datetime.now().timestamp() - self.start_time if self.start_time else 0.0

        # Calculate average beam current
        if self.beam_current_history:
            currents = [c for _, c in self.beam_current_history]
            avg_current = np.mean(currents)
        else:
            avg_current = 0.0

        # Dose rate
        dose_rate = current_dose / elapsed if elapsed > 0 else 0.0

        return DoseIntegrationResult(
            current_dose_cm2=current_dose,
            target_dose_cm2=target_dose_cm2,
            percent_complete=(current_dose / target_dose_cm2 * 100) if target_dose_cm2 > 0 else 0.0,
            integrated_charge_C=self.integrated_charge_C,
            elapsed_time_s=elapsed,
            dose_rate_cm2_per_s=dose_rate,
            uniformity_correction_factor=self.scan_correction,
            beam_current_avg_mA=avg_current
        )

    def set_area_correction(self, correction: float):
        """Set area correction factor."""
        self.area_correction = correction

    def set_scan_correction(self, correction: float):
        """Set scan uniformity correction factor."""
        self.scan_correction = correction


# ============================================================================
# Scan Uniformity Correction
# ============================================================================

class ScanUniformityController:
    """Auto-trim scan pattern and steering to flatten 2D dose map."""

    def __init__(
        self,
        target_uniformity_pct: float = 95.0,
        correction_gain: float = 0.5
    ):
        """
        Initialize scan uniformity controller.

        Args:
            target_uniformity_pct: Target uniformity (%)
            correction_gain: Correction gain (0-1)
        """
        self.target_uniformity_pct = target_uniformity_pct
        self.correction_gain = correction_gain

        self.current_uniformity_map: Optional[UniformityMap] = None
        self.correction_params = ScanCorrectionParameters(
            x_amplitude_correction=1.0,
            y_amplitude_correction=1.0,
            x_steering_offset_mm=0.0,
            y_steering_offset_mm=0.0
        )

    def analyze_uniformity(
        self,
        dose_map: np.ndarray,
        x_positions_mm: np.ndarray,
        y_positions_mm: np.ndarray
    ) -> UniformityMap:
        """
        Analyze 2D dose uniformity map.

        Args:
            dose_map: 2D dose map (normalized or absolute)
            x_positions_mm: X positions
            y_positions_mm: Y positions

        Returns:
            UniformityMap analysis
        """
        # Normalize dose map (mean = 1.0)
        mean_dose = np.mean(dose_map[dose_map > 0])
        relative_dose = dose_map / mean_dose

        # Statistics
        std_dose = np.std(relative_dose[dose_map > 0])

        # Uniformity: (1 - 3σ/mean) × 100%
        uniformity_pct = (1.0 - 3 * std_dose) * 100

        # Edge rolloff (compare edge to center)
        center_mask = (np.abs(x_positions_mm) < 50) & (np.abs(y_positions_mm) < 50)
        edge_mask = (np.abs(x_positions_mm) > 100) | (np.abs(y_positions_mm) > 100)

        center_mean = np.mean(relative_dose[center_mask]) if np.any(center_mask) else 1.0
        edge_mean = np.mean(relative_dose[edge_mask]) if np.any(edge_mask) else 1.0
        edge_rolloff_pct = (1.0 - edge_mean / center_mean) * 100

        self.current_uniformity_map = UniformityMap(
            x_positions_mm=x_positions_mm,
            y_positions_mm=y_positions_mm,
            relative_dose=relative_dose,
            mean_dose=mean_dose,
            std_dose=std_dose,
            uniformity_pct=uniformity_pct,
            edge_rolloff_pct=edge_rolloff_pct,
            measurement_timestamp=datetime.now().timestamp()
        )

        return self.current_uniformity_map

    def calculate_corrections(
        self,
        uniformity_map: Optional[UniformityMap] = None
    ) -> ScanCorrectionParameters:
        """
        Calculate scan corrections to improve uniformity.

        Args:
            uniformity_map: Uniformity map (uses current if None)

        Returns:
            ScanCorrectionParameters
        """
        if uniformity_map is None:
            uniformity_map = self.current_uniformity_map

        if uniformity_map is None:
            logger.warning("No uniformity map available for correction")
            return self.correction_params

        # Calculate centroid offset (beam steering correction)
        x_centroid = np.sum(uniformity_map.x_positions_mm * uniformity_map.relative_dose) / np.sum(uniformity_map.relative_dose)
        y_centroid = np.sum(uniformity_map.y_positions_mm * uniformity_map.relative_dose) / np.sum(uniformity_map.relative_dose)

        # Steering correction (move beam to center centroid)
        steering_x = -x_centroid * self.correction_gain
        steering_y = -y_centroid * self.correction_gain

        # Amplitude correction (expand/contract scan for edge rolloff)
        if uniformity_map.edge_rolloff_pct > 5.0:
            # Significant edge rolloff - increase scan amplitude
            amplitude_correction = 1.0 + (uniformity_map.edge_rolloff_pct / 100) * self.correction_gain
        else:
            amplitude_correction = 1.0

        # Update correction parameters
        self.correction_params.x_steering_offset_mm += steering_x
        self.correction_params.y_steering_offset_mm += steering_y
        self.correction_params.x_amplitude_correction = amplitude_correction
        self.correction_params.y_amplitude_correction = amplitude_correction
        self.correction_params.correction_timestamp = datetime.now().timestamp()

        logger.info(
            f"Scan corrections: steering=({steering_x:.3f}, {steering_y:.3f}) mm, "
            f"amplitude={amplitude_correction:.4f}"
        )

        return self.correction_params


# ============================================================================
# Run-to-Run (R2R) Control
# ============================================================================

class R2RController:
    """Wafer-to-wafer adjustments based on measured non-uniformity."""

    def __init__(
        self,
        ewma_weight: float = 0.3,
        control_limits_sigma: float = 3.0
    ):
        """
        Initialize R2R controller.

        Args:
            ewma_weight: EWMA weight (0-1, higher = more responsive)
            control_limits_sigma: Control limit in sigma units
        """
        self.ewma_weight = ewma_weight
        self.control_limits_sigma = control_limits_sigma

        self.state = R2RState(wafer_count=0)

    def update(
        self,
        measured_dose_cm2: float,
        target_dose_cm2: float,
        measured_uniformity_pct: float,
        measured_tilt_deg: Optional[float] = None,
        measured_twist_deg: Optional[float] = None
    ):
        """
        Update R2R state with new wafer measurements.

        Args:
            measured_dose_cm2: Measured dose
            target_dose_cm2: Target dose
            measured_uniformity_pct: Measured uniformity
            measured_tilt_deg: Measured tilt angle
            measured_twist_deg: Measured twist angle
        """
        self.state.wafer_count += 1

        # Add to history
        self.state.dose_history.append(measured_dose_cm2 / target_dose_cm2)  # Normalized
        self.state.uniformity_history.append(measured_uniformity_pct)

        if measured_tilt_deg is not None:
            self.state.tilt_history.append(measured_tilt_deg)
        if measured_twist_deg is not None:
            self.state.twist_history.append(measured_twist_deg)

        # Calculate EWMA for dose
        if len(self.state.dose_history) > 1:
            ewma_dose = self._calculate_ewma(self.state.dose_history)

            # Dose adjustment: compensate for drift
            dose_error = 1.0 - ewma_dose
            self.state.recommended_dose_adjustment = 1.0 + dose_error * 0.5  # 50% correction
        else:
            self.state.recommended_dose_adjustment = 1.0

        # Calculate EWMA for uniformity
        if len(self.state.uniformity_history) > 1:
            ewma_uniformity = self._calculate_ewma(self.state.uniformity_history)

            # If uniformity is degrading, recommend tilt adjustment
            if ewma_uniformity < 90.0:
                # Increase tilt slightly to reduce channeling
                self.state.recommended_tilt_deg = min(self.state.recommended_tilt_deg + 0.5, 10.0)
            elif ewma_uniformity > 95.0 and self.state.recommended_tilt_deg > 7.0:
                # Can reduce tilt if uniformity is good
                self.state.recommended_tilt_deg = max(self.state.recommended_tilt_deg - 0.2, 7.0)

        logger.info(
            f"R2R update (wafer {self.state.wafer_count}): "
            f"dose_adj={self.state.recommended_dose_adjustment:.4f}, "
            f"tilt={self.state.recommended_tilt_deg:.1f}°"
        )

    def _calculate_ewma(self, history: Deque[float]) -> float:
        """Calculate exponentially weighted moving average."""
        if not history:
            return 0.0

        ewma = history[0]
        for value in list(history)[1:]:
            ewma = self.ewma_weight * value + (1 - self.ewma_weight) * ewma

        return ewma

    def get_recommendations(self) -> Dict:
        """Get R2R control recommendations."""
        return {
            "dose_adjustment_factor": self.state.recommended_dose_adjustment,
            "recommended_tilt_deg": self.state.recommended_tilt_deg,
            "recommended_twist_deg": self.state.recommended_twist_deg,
            "wafer_count": self.state.wafer_count,
            "dose_trend": "stable" if abs(self.state.recommended_dose_adjustment - 1.0) < 0.05 else "drifting"
        }


# ============================================================================
# Beam Drift FDC
# ============================================================================

class BeamDriftDetector:
    """Detect beam drift/spikes and recommend compensation."""

    def __init__(
        self,
        drift_threshold_mm: float = 0.5,
        spike_threshold_mm: float = 2.0,
        window_size: int = 10
    ):
        """
        Initialize beam drift detector.

        Args:
            drift_threshold_mm: Drift detection threshold
            spike_threshold_mm: Spike detection threshold
            window_size: Moving window size
        """
        self.drift_threshold_mm = drift_threshold_mm
        self.spike_threshold_mm = spike_threshold_mm
        self.window_size = window_size

        self.position_history: Deque[Tuple[float, float, float]] = deque(maxlen=window_size)  # (time, x, y)

    def update(
        self,
        x_position_mm: float,
        y_position_mm: float,
        timestamp: float = None
    ) -> BeamDriftDetection:
        """
        Update drift detector with new beam position.

        Args:
            x_position_mm: Beam X position
            y_position_mm: Beam Y position
            timestamp: Timestamp

        Returns:
            BeamDriftDetection result
        """
        if timestamp is None:
            timestamp = datetime.now().timestamp()

        self.position_history.append((timestamp, x_position_mm, y_position_mm))

        if len(self.position_history) < 3:
            # Not enough data
            return BeamDriftDetection(
                is_drift_detected=False,
                drift_magnitude_mm=0.0,
                drift_rate_mm_per_s=0.0,
                drift_direction_deg=0.0,
                recommended_action="continue",
                steering_correction_x_mm=0.0,
                steering_correction_y_mm=0.0
            )

        # Calculate drift vector
        positions = list(self.position_history)

        # Linear regression for drift rate
        times = np.array([p[0] for p in positions])
        x_positions = np.array([p[1] for p in positions])
        y_positions = np.array([p[2] for p in positions])

        # Normalize time
        t = times - times[0]

        # Fit linear trend
        x_drift_rate = np.polyfit(t, x_positions, 1)[0] if len(t) > 1 else 0.0
        y_drift_rate = np.polyfit(t, y_positions, 1)[0] if len(t) > 1 else 0.0

        # Total drift magnitude
        drift_magnitude = np.sqrt(x_drift_rate**2 + y_drift_rate**2)

        # Drift direction
        drift_direction_deg = np.rad2deg(np.arctan2(y_drift_rate, x_drift_rate))

        # Check for spike (sudden large change)
        if len(positions) >= 2:
            last_delta_x = positions[-1][1] - positions[-2][1]
            last_delta_y = positions[-1][2] - positions[-2][2]
            spike_magnitude = np.sqrt(last_delta_x**2 + last_delta_y**2)

            if spike_magnitude > self.spike_threshold_mm:
                # Spike detected
                return BeamDriftDetection(
                    is_drift_detected=True,
                    drift_magnitude_mm=spike_magnitude,
                    drift_rate_mm_per_s=spike_magnitude / (t[-1] - t[-2]) if len(t) > 1 else 0.0,
                    drift_direction_deg=drift_direction_deg,
                    recommended_action="pause",  # Pause for spike
                    steering_correction_x_mm=-last_delta_x,
                    steering_correction_y_mm=-last_delta_y
                )

        # Check for slow drift
        if drift_magnitude > self.drift_threshold_mm:
            # Drift detected - recommend compensation
            recommended_action = "compensate"

            # Steering correction (opposite of drift)
            steering_x = -x_positions[-1] * 0.5  # 50% correction
            steering_y = -y_positions[-1] * 0.5
        else:
            recommended_action = "continue"
            steering_x = 0.0
            steering_y = 0.0

        return BeamDriftDetection(
            is_drift_detected=drift_magnitude > self.drift_threshold_mm,
            drift_magnitude_mm=drift_magnitude,
            drift_rate_mm_per_s=drift_magnitude,
            drift_direction_deg=drift_direction_deg,
            recommended_action=recommended_action,
            steering_correction_x_mm=steering_x,
            steering_correction_y_mm=steering_y
        )


# Export
__all__ = [
    "DoseIntegrator",
    "ScanUniformityController",
    "R2RController",
    "BeamDriftDetector",
    "DoseIntegrationResult",
    "UniformityMap",
    "ScanCorrectionParameters",
    "R2RState",
    "BeamDriftDetection",
    "FDCAlert",
]
