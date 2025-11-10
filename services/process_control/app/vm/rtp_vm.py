"""Virtual Metrology (VM) model for RTP (Rapid Thermal Processing).

Predicts activation fraction, diffusion depth, and sheet resistance change
based on RTP recipe and real-time plant response using machine learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pickle


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RTPVMFeatures:
    """Feature vector for RTP VM model."""
    # Recipe parameters
    peak_temp_C: float
    total_dwell_time_s: float
    ramp_rate_up_C_per_s: float
    ramp_rate_down_C_per_s: float
    num_segments: int

    # Gas ambient
    gas_type: str  # N2, O2, forming_gas, Ar, vacuum
    gas_flow_sccm: float
    chamber_pressure_torr: float

    # Initial wafer state (from ion implant)
    initial_dose_cm2: Optional[float] = None
    initial_energy_keV: Optional[float] = None
    ion_species: Optional[str] = None

    # Real-time plant response
    actual_peak_temp_C: float = 0.0
    actual_dwell_time_s: float = 0.0
    avg_tracking_error_C: float = 0.0
    max_overshoot_C: float = 0.0
    thermal_budget: float = 0.0  # ∫exp(-Ea/kT)dt
    lamp_power_avg_pct: float = 0.0

    # Emissivity
    emissivity_setting: float = 0.65
    emissivity_drift: float = 0.0

    # Wafer properties
    wafer_backside_coating: str = "none"  # none, oxide, nitride
    substrate_type: str = "silicon"


@dataclass
class RTPVMPrediction:
    """VM prediction result for RTP."""
    # Predicted outcomes
    predicted_activation_fraction: float
    predicted_diffusion_depth_nm: float
    predicted_sheet_resistance_change_pct: float  # % change from initial
    predicted_oxide_thickness_nm: Optional[float] = None  # If oxidizing ambient

    # Uncertainty estimates
    activation_fraction_std: float = 0.0
    diffusion_depth_std: float = 0.0

    # Model metadata
    model_version: str = "v1.0"
    prediction_confidence: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)

    # Process quality indicators
    process_quality_score: float = 0.0  # 0-100
    uniformity_prediction_pct: float = 0.0


# ============================================================================
# Feature Engineering
# ============================================================================

class RTPVMFeatureEngineer:
    """Feature engineering for RTP VM model."""

    GAS_TYPE_ENCODING = {
        "nitrogen": 1,
        "n2": 1,
        "oxygen": 2,
        "o2": 2,
        "forming_gas": 3,
        "argon": 4,
        "ar": 4,
        "vacuum": 5,
    }

    ION_SPECIES_ENCODING = {
        "boron": 1,
        "phosphorus": 2,
        "arsenic": 3,
    }

    @staticmethod
    def encode_features(features: RTPVMFeatures) -> np.ndarray:
        """Encode features into numerical array.

        Args:
            features: RTPVMFeatures object

        Returns:
            Numpy array of encoded features
        """
        feature_vector = []

        # Temperature features
        feature_vector.append(features.peak_temp_C / 1000.0)  # Normalize to ~1
        feature_vector.append(features.actual_peak_temp_C / 1000.0)
        feature_vector.append(features.avg_tracking_error_C / 10.0)
        feature_vector.append(features.max_overshoot_C / 10.0)

        # Time features
        feature_vector.append(features.total_dwell_time_s / 60.0)  # Normalize to minutes
        feature_vector.append(features.actual_dwell_time_s / 60.0)
        feature_vector.append(np.log10(features.total_dwell_time_s + 1))

        # Ramp rate features
        feature_vector.append(features.ramp_rate_up_C_per_s / 100.0)
        feature_vector.append(features.ramp_rate_down_C_per_s / 100.0)

        # Thermal budget (critical for activation/diffusion)
        feature_vector.append(np.log10(features.thermal_budget + 1e-10))
        feature_vector.append(features.thermal_budget)

        # Gas features
        gas_code = RTPVMFeatureEngineer.GAS_TYPE_ENCODING.get(
            features.gas_type.lower(), 0
        )
        feature_vector.append(gas_code)
        feature_vector.append(features.gas_flow_sccm / 5000.0)  # Normalize
        feature_vector.append(features.chamber_pressure_torr / 760.0)

        # Emissivity
        feature_vector.append(features.emissivity_setting)
        feature_vector.append(features.emissivity_drift)

        # Initial state (if available)
        if features.initial_dose_cm2:
            feature_vector.append(np.log10(features.initial_dose_cm2))
            feature_vector.append(features.initial_energy_keV or 0.0)
            ion_code = RTPVMFeatureEngineer.ION_SPECIES_ENCODING.get(
                (features.ion_species or "").lower(), 0
            )
            feature_vector.append(ion_code)
        else:
            feature_vector.extend([0.0, 0.0, 0.0])

        # Process control features
        feature_vector.append(features.lamp_power_avg_pct / 100.0)
        feature_vector.append(features.num_segments)

        # Derived features
        # Temperature-time interaction (key for diffusion)
        temp_time_product = (features.actual_peak_temp_C / 1000.0) * (features.actual_dwell_time_s / 60.0)
        feature_vector.append(temp_time_product)

        # Ramp asymmetry
        ramp_asymmetry = abs(features.ramp_rate_up_C_per_s - features.ramp_rate_down_C_per_s)
        feature_vector.append(ramp_asymmetry / 100.0)

        return np.array(feature_vector, dtype=np.float32)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get feature names."""
        return [
            "peak_temp_norm",
            "actual_peak_temp_norm",
            "avg_tracking_error_norm",
            "max_overshoot_norm",
            "dwell_time_min",
            "actual_dwell_time_min",
            "log_dwell_time",
            "ramp_up_norm",
            "ramp_down_norm",
            "log_thermal_budget",
            "thermal_budget",
            "gas_type_code",
            "gas_flow_norm",
            "pressure_norm",
            "emissivity",
            "emissivity_drift",
            "log_dose",
            "energy_keV",
            "ion_species_code",
            "lamp_power_norm",
            "num_segments",
            "temp_time_product",
            "ramp_asymmetry",
        ]


# ============================================================================
# RTP VM Model
# ============================================================================

class RTPVirtualMetrologyModel:
    """Virtual metrology model for predicting RTP outcomes.

    Predicts:
    - Dopant activation fraction
    - Diffusion depth increase
    - Sheet resistance change
    - Oxide thickness (if applicable)
    """

    def __init__(self, model_version: str = "v1.0"):
        """Initialize RTP VM model.

        Args:
            model_version: Model version
        """
        self.model_version = model_version
        self.feature_engineer = RTPVMFeatureEngineer()

        # Model placeholders
        self.activation_model = None
        self.diffusion_model = None
        self.oxide_model = None

        # Training statistics
        self.training_stats = {
            "n_samples": 0,
            "r2_score_activation": 0.0,
            "r2_score_diffusion": 0.0,
        }

    def predict(self, features: RTPVMFeatures) -> RTPVMPrediction:
        """Predict RTP outcomes from features.

        Args:
            features: RTP process features

        Returns:
            RTPVMPrediction with predicted outcomes
        """
        # Encode features
        X = self.feature_engineer.encode_features(features)

        # Predictions (using heuristics as placeholders)
        activation, activation_std = self._predict_activation_heuristic(features)
        diffusion_depth, diffusion_std = self._predict_diffusion_heuristic(features)
        rs_change = self._predict_rs_change_heuristic(features, activation)

        # Oxide thickness (if oxidizing ambient)
        oxide_thickness = None
        if features.gas_type.lower() in ["oxygen", "o2"]:
            oxide_thickness = self._predict_oxide_thickness_heuristic(features)

        # Process quality score
        quality_score = self._calculate_quality_score(features)

        # Uniformity prediction
        uniformity_pred = self._predict_uniformity(features)

        # Confidence
        confidence = self._calculate_confidence(X)

        # Feature importance
        importance = self._calculate_feature_importance()

        return RTPVMPrediction(
            predicted_activation_fraction=activation,
            predicted_diffusion_depth_nm=diffusion_depth,
            predicted_sheet_resistance_change_pct=rs_change,
            predicted_oxide_thickness_nm=oxide_thickness,
            activation_fraction_std=activation_std,
            diffusion_depth_std=diffusion_std,
            model_version=self.model_version,
            prediction_confidence=confidence,
            feature_importance=importance,
            process_quality_score=quality_score,
            uniformity_prediction_pct=uniformity_pred
        )

    def _predict_activation_heuristic(
        self,
        features: RTPVMFeatures
    ) -> Tuple[float, float]:
        """Heuristic activation prediction (Arrhenius-based)."""
        # Activation depends on thermal budget
        # activation ≈ 1 - exp(-k * thermal_budget)

        # Base activation from thermal budget
        if features.thermal_budget > 0:
            base_activation = 1.0 - np.exp(-0.5 * features.thermal_budget)
        else:
            base_activation = 0.0

        # Temperature threshold effects
        if features.actual_peak_temp_C < 800:
            # Low temperature - limited activation
            base_activation *= 0.3
        elif features.actual_peak_temp_C < 900:
            base_activation *= 0.6
        elif features.actual_peak_temp_C < 1000:
            base_activation *= 0.85

        # Dwell time factor
        if features.actual_dwell_time_s < 10:
            base_activation *= 0.7
        elif features.actual_dwell_time_s > 120:
            base_activation *= 1.05  # Slight boost for long anneals

        # Gas ambient effect
        if features.gas_type.lower() in ["oxygen", "o2"]:
            # Oxidizing ambient can reduce activation slightly
            base_activation *= 0.95

        # Clamp to realistic range
        activation = np.clip(base_activation, 0.0, 0.95)

        # Uncertainty based on process variability
        if features.avg_tracking_error_C > 5.0:
            activation_std = 0.10  # High uncertainty
        else:
            activation_std = 0.05  # Low uncertainty

        return activation, activation_std

    def _predict_diffusion_heuristic(
        self,
        features: RTPVMFeatures
    ) -> Tuple[float, float]:
        """Heuristic diffusion depth prediction.

        Uses simplified Fick's 2nd law: x_diff ≈ sqrt(D * t)
        where D = D0 * exp(-Ea/kT)
        """
        # Diffusion coefficient approximation
        if features.ion_species and features.ion_species.lower() == "boron":
            D0 = 0.76  # cm²/s
            Ea = 3.65  # eV
        elif features.ion_species and features.ion_species.lower() == "phosphorus":
            D0 = 3.85
            Ea = 3.66
        else:
            D0 = 1.0
            Ea = 3.5

        # Average D during anneal (rough approximation using peak temp)
        if features.actual_peak_temp_C > 0:
            T_K = features.actual_peak_temp_C + 273.15
            k_B = 8.617e-5  # eV/K
            D_avg = D0 * np.exp(-Ea / (k_B * T_K))
        else:
            D_avg = 0

        # Diffusion length
        t_seconds = features.actual_dwell_time_s
        diffusion_length_cm = np.sqrt(D_avg * t_seconds)
        diffusion_depth_nm = diffusion_length_cm * 1e7  # Convert to nm

        # Add contribution from ramps (approximation)
        ramp_contribution = 0.1 * diffusion_depth_nm  # 10% from ramps
        total_diffusion = diffusion_depth_nm + ramp_contribution

        # Uncertainty
        diffusion_std = total_diffusion * 0.15  # 15% uncertainty

        return total_diffusion, diffusion_std

    def _predict_rs_change_heuristic(
        self,
        features: RTPVMFeatures,
        activation: float
    ) -> float:
        """Predict sheet resistance change.

        Rs_new = Rs_old * (activation_old / activation_new)
        So % change = (activation_old / activation_new - 1) * 100
        """
        # Assume initial activation was low (e.g., 5%)
        initial_activation = 0.05

        if activation > initial_activation:
            # Rs decreases as activation increases
            rs_ratio = initial_activation / activation
            rs_change_pct = (rs_ratio - 1.0) * 100.0
        else:
            rs_change_pct = 0.0

        return rs_change_pct

    def _predict_oxide_thickness_heuristic(self, features: RTPVMFeatures) -> float:
        """Predict oxide growth (Deal-Grove model approximation)."""
        if features.gas_type.lower() not in ["oxygen", "o2"]:
            return 0.0

        # Simplified Deal-Grove
        # For thin oxides: X = B/A * t
        # For thick oxides: X² = B * t

        T_K = features.actual_peak_temp_C + 273.15
        t_seconds = features.actual_dwell_time_s

        # Rough approximation (real model would use proper Deal-Grove)
        if features.actual_peak_temp_C > 1000:
            # Fast growth
            growth_rate = 50.0  # nm/min
        elif features.actual_peak_temp_C > 900:
            growth_rate = 10.0
        else:
            growth_rate = 2.0

        oxide_nm = growth_rate * (t_seconds / 60.0)

        return oxide_nm

    def _calculate_quality_score(self, features: RTPVMFeatures) -> float:
        """Calculate process quality score (0-100)."""
        score = 100.0

        # Penalize large tracking errors
        score -= min(features.avg_tracking_error_C * 2, 20)

        # Penalize overshoot
        score -= min(features.max_overshoot_C, 15)

        # Penalize emissivity drift
        score -= abs(features.emissivity_drift) * 20

        # Reward good dwell time
        if 20 < features.actual_dwell_time_s < 120:
            score += 5

        return max(score, 0.0)

    def _predict_uniformity(self, features: RTPVMFeatures) -> float:
        """Predict wafer uniformity."""
        base_uniformity = 95.0

        # Degrade with tracking error
        base_uniformity -= features.avg_tracking_error_C * 0.5

        # Degrade with overshoot
        base_uniformity -= features.max_overshoot_C * 0.3

        return max(base_uniformity, 80.0)

    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence."""
        # Mock confidence
        return 0.80

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Feature importance (mock)."""
        return {
            "thermal_budget": 0.30,
            "actual_peak_temp_norm": 0.25,
            "actual_dwell_time_min": 0.20,
            "avg_tracking_error_norm": 0.10,
            "gas_type_code": 0.08,
            "emissivity": 0.05,
            "max_overshoot_norm": 0.02,
        }

    def save_model(self, filepath: str):
        """Save model."""
        model_data = {
            "version": self.model_version,
            "training_stats": self.training_stats,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.model_version = model_data["version"]
        self.training_stats = model_data["training_stats"]


# Export
__all__ = [
    "RTPVMFeatures",
    "RTPVMPrediction",
    "RTPVMFeatureEngineer",
    "RTPVirtualMetrologyModel",
]
