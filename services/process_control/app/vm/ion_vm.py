"""Virtual Metrology (VM) model for Ion Implantation.

Predicts sheet resistance, junction depth, and activation from ion implant
parameters and RTP thermal budget using machine learning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pickle
import json


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class IonVMFeatures:
    """Feature vector for Ion VM model."""
    # Ion implant parameters
    ion_species: str  # Will be encoded
    energy_keV: float
    dose_cm2: float
    tilt_angle_deg: float
    twist_angle_deg: float

    # Beam parameters
    beam_current_mA: float
    dose_uniformity_pct: float

    # Vacuum/pressure
    avg_source_pressure_mtorr: float
    avg_process_pressure_mtorr: float

    # RTP parameters (if available)
    rtp_peak_temp_C: Optional[float] = None
    rtp_dwell_time_s: Optional[float] = None
    rtp_thermal_budget: Optional[float] = None  # ∫exp(-Ea/kT)dt
    rtp_ramp_rate_C_per_s: Optional[float] = None

    # Contextual
    wafer_lot_id: Optional[str] = None
    substrate_type: str = "silicon"  # silicon, sige, etc.


@dataclass
class IonVMPrediction:
    """VM prediction result for ion implantation."""
    # Predicted metrology
    predicted_sheet_resistance_ohm_per_sq: float
    predicted_junction_depth_nm: float
    predicted_activation_fraction: float

    # Uncertainty estimates
    sheet_resistance_std: float
    junction_depth_std: float
    activation_fraction_std: float

    # Model metadata
    model_version: str
    prediction_confidence: float  # 0-1
    feature_importance: Dict[str, float] = field(default_factory=dict)


# ============================================================================
# Feature Engineering
# ============================================================================

class IonVMFeatureEngineer:
    """Feature engineering for Ion VM model."""

    ION_SPECIES_ENCODING = {
        "boron": 1,
        "phosphorus": 2,
        "arsenic": 3,
        "antimony": 4,
        "nitrogen": 5,
        "oxygen": 6,
        "argon": 7,
        "silicon": 8,
    }

    @staticmethod
    def encode_features(features: IonVMFeatures) -> np.ndarray:
        """Encode features into numerical array for model input.

        Args:
            features: IonVMFeatures object

        Returns:
            Numpy array of encoded features
        """
        feature_vector = []

        # Ion species (one-hot or ordinal)
        ion_code = IonVMFeatureEngineer.ION_SPECIES_ENCODING.get(
            features.ion_species.lower(), 0
        )
        feature_vector.append(ion_code)

        # Energy features
        feature_vector.append(features.energy_keV)
        feature_vector.append(np.log10(features.energy_keV + 1))  # Log transform

        # Dose features
        feature_vector.append(np.log10(features.dose_cm2))  # Always log scale
        feature_vector.append(features.dose_cm2 / 1e15)  # Normalized

        # Angle features
        feature_vector.append(features.tilt_angle_deg)
        feature_vector.append(features.twist_angle_deg)
        feature_vector.append(np.cos(np.radians(features.tilt_angle_deg)))  # Channeling risk

        # Beam features
        feature_vector.append(features.beam_current_mA)
        feature_vector.append(features.dose_uniformity_pct)

        # Vacuum features
        feature_vector.append(np.log10(features.avg_source_pressure_mtorr + 1))
        feature_vector.append(np.log10(features.avg_process_pressure_mtorr + 1))

        # RTP features (if available)
        if features.rtp_peak_temp_C is not None:
            feature_vector.append(features.rtp_peak_temp_C / 1000.0)  # Normalize
            feature_vector.append(features.rtp_dwell_time_s or 0.0)
            feature_vector.append(np.log10((features.rtp_thermal_budget or 0.0) + 1))
            feature_vector.append((features.rtp_ramp_rate_C_per_s or 0.0) / 100.0)
        else:
            # Fill with zeros if RTP data not available
            feature_vector.extend([0.0, 0.0, 0.0, 0.0])

        # Derived features
        # Projected range approximation (simple power law)
        if ion_code == 1:  # Boron
            approx_range = 1.2 * (features.energy_keV ** 1.7)
        elif ion_code == 2:  # Phosphorus
            approx_range = 1.0 * (features.energy_keV ** 1.6)
        elif ion_code == 3:  # Arsenic
            approx_range = 0.8 * (features.energy_keV ** 1.3)
        else:
            approx_range = 1.0 * (features.energy_keV ** 1.5)

        feature_vector.append(approx_range / 100.0)  # Normalize

        # Dose x Energy interaction
        feature_vector.append(np.log10(features.dose_cm2) * np.log10(features.energy_keV))

        return np.array(feature_vector, dtype=np.float32)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Get feature names for interpretability."""
        return [
            "ion_species_code",
            "energy_keV",
            "log_energy_keV",
            "log_dose_cm2",
            "dose_normalized",
            "tilt_angle_deg",
            "twist_angle_deg",
            "cos_tilt_angle",
            "beam_current_mA",
            "dose_uniformity_pct",
            "log_source_pressure",
            "log_process_pressure",
            "rtp_peak_temp_norm",
            "rtp_dwell_time_s",
            "log_thermal_budget",
            "rtp_ramp_rate_norm",
            "approx_range_norm",
            "dose_energy_interaction",
        ]


# ============================================================================
# Ion VM Model
# ============================================================================

class IonVirtualMetrologyModel:
    """Virtual metrology model for predicting ion implant results.

    Uses gradient boosting or random forest to predict:
    - Sheet resistance (Ω/sq)
    - Junction depth (nm)
    - Activation fraction (0-1)
    """

    def __init__(self, model_version: str = "v1.0"):
        """Initialize Ion VM model.

        Args:
            model_version: Model version identifier
        """
        self.model_version = model_version
        self.feature_engineer = IonVMFeatureEngineer()

        # Model placeholders (would be trained ML models)
        self.sheet_resistance_model = None
        self.junction_depth_model = None
        self.activation_model = None

        # Model statistics (from training)
        self.training_stats = {
            "n_samples": 0,
            "feature_means": None,
            "feature_stds": None,
            "r2_score_rs": 0.0,
            "r2_score_jd": 0.0,
            "r2_score_activation": 0.0,
        }

    def predict(self, features: IonVMFeatures) -> IonVMPrediction:
        """Predict metrology results from ion implant features.

        Args:
            features: Ion implant and RTP features

        Returns:
            IonVMPrediction with predicted metrology
        """
        # Encode features
        X = self.feature_engineer.encode_features(features)

        # For now, use physics-based heuristics (would be replaced with trained ML models)
        predicted_rs, rs_std = self._predict_sheet_resistance_heuristic(features)
        predicted_jd, jd_std = self._predict_junction_depth_heuristic(features)
        predicted_act, act_std = self._predict_activation_heuristic(features)

        # Calculate prediction confidence based on feature coverage
        confidence = self._calculate_confidence(X)

        # Calculate feature importance (mock - would come from model)
        feature_importance = self._calculate_feature_importance()

        return IonVMPrediction(
            predicted_sheet_resistance_ohm_per_sq=predicted_rs,
            predicted_junction_depth_nm=predicted_jd,
            predicted_activation_fraction=predicted_act,
            sheet_resistance_std=rs_std,
            junction_depth_std=jd_std,
            activation_fraction_std=act_std,
            model_version=self.model_version,
            prediction_confidence=confidence,
            feature_importance=feature_importance
        )

    def _predict_sheet_resistance_heuristic(
        self,
        features: IonVMFeatures
    ) -> Tuple[float, float]:
        """Heuristic prediction for sheet resistance (placeholder for ML model)."""
        # Use Caughey-Thomas approximation
        # Rs = 1 / (q * ∫ n(x) * μ(x) dx)

        # Approximate junction depth
        if features.ion_species.lower() == "boron":
            approx_range = 1.2 * (features.energy_keV ** 1.7)
        elif features.ion_species.lower() == "phosphorus":
            approx_range = 1.0 * (features.energy_keV ** 1.6)
        else:
            approx_range = 0.8 * (features.energy_keV ** 1.3)

        junction_depth_nm = approx_range * 1.2  # Approximate

        # Activation fraction from RTP
        if features.rtp_peak_temp_C and features.rtp_peak_temp_C > 900:
            base_activation = 0.6 + 0.002 * (features.rtp_peak_temp_C - 900)
        else:
            base_activation = 0.3

        base_activation = min(base_activation, 0.95)

        # Active dose
        active_dose = features.dose_cm2 * base_activation

        # Mobility (simplified Caughey-Thomas)
        if features.ion_species.lower() == "boron":
            mu_max = 450  # cm²/Vs
            N_ref = 1.7e17
        else:  # n-type
            mu_max = 1400
            N_ref = 1e17

        avg_concentration = active_dose / (junction_depth_nm * 1e-7)  # cm⁻³
        mobility = mu_max / (1 + (avg_concentration / N_ref) ** 0.7)

        # Sheet resistance
        q = 1.602e-19
        sheet_integral = active_dose * mobility
        sheet_resistance = 1.0 / (q * sheet_integral)

        # Uncertainty estimate (would come from model)
        rs_std = sheet_resistance * 0.15  # 15% uncertainty

        return sheet_resistance, rs_std

    def _predict_junction_depth_heuristic(
        self,
        features: IonVMFeatures
    ) -> Tuple[float, float]:
        """Heuristic prediction for junction depth."""
        # Power law approximation
        if features.ion_species.lower() == "boron":
            approx_range = 1.2 * (features.energy_keV ** 1.7)
        elif features.ion_species.lower() == "phosphorus":
            approx_range = 1.0 * (features.energy_keV ** 1.6)
        elif features.ion_species.lower() == "arsenic":
            approx_range = 0.8 * (features.energy_keV ** 1.3)
        else:
            approx_range = 1.0 * (features.energy_keV ** 1.5)

        # Junction depth is approximately 1.2x projected range
        junction_depth = approx_range * 1.2

        # RTP can cause some diffusion
        if features.rtp_thermal_budget and features.rtp_thermal_budget > 1e-3:
            # Very simplified diffusion
            additional_depth = 5.0 * np.sqrt(features.rtp_thermal_budget)
            junction_depth += additional_depth

        # Uncertainty
        jd_std = junction_depth * 0.10  # 10% uncertainty

        return junction_depth, jd_std

    def _predict_activation_heuristic(
        self,
        features: IonVMFeatures
    ) -> Tuple[float, float]:
        """Heuristic prediction for activation fraction."""
        if features.rtp_peak_temp_C is None:
            # No anneal
            activation = 0.05  # Only minimal activation
        elif features.rtp_peak_temp_C < 800:
            activation = 0.1 + 0.0005 * features.rtp_peak_temp_C
        elif features.rtp_peak_temp_C < 1000:
            activation = 0.3 + 0.003 * (features.rtp_peak_temp_C - 800)
        else:
            activation = 0.7 + 0.001 * (features.rtp_peak_temp_C - 1000)

        # Dwell time effect
        if features.rtp_dwell_time_s:
            time_factor = min(features.rtp_dwell_time_s / 60.0, 1.0)
            activation *= (0.7 + 0.3 * time_factor)

        activation = min(activation, 0.95)  # Cap at 95%

        # Uncertainty
        act_std = 0.08  # 8% absolute uncertainty

        return activation, act_std

    def _calculate_confidence(self, X: np.ndarray) -> float:
        """Calculate prediction confidence based on feature coverage."""
        # Simple heuristic: confidence decreases for extreme values
        # (Would be based on training data distribution in real implementation)

        # Assume features are reasonably normalized
        # Calculate how many features are > 3 std from mean
        if self.training_stats["feature_means"] is None:
            # No training data, assume moderate confidence
            return 0.7

        return 0.85  # Mock value

    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance (mock - would come from trained model)."""
        feature_names = self.feature_engineer.get_feature_names()

        # Mock importance (in real model, from SHAP or model.feature_importances_)
        importance = {
            "energy_keV": 0.25,
            "log_dose_cm2": 0.20,
            "rtp_peak_temp_norm": 0.18,
            "log_thermal_budget": 0.15,
            "ion_species_code": 0.10,
            "tilt_angle_deg": 0.05,
            "dose_uniformity_pct": 0.04,
            "beam_current_mA": 0.03,
        }

        return importance

    def save_model(self, filepath: str):
        """Save model to file.

        Args:
            filepath: Path to save model
        """
        model_data = {
            "version": self.model_version,
            "training_stats": self.training_stats,
            # In real implementation, would save sklearn/xgboost models
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath: str):
        """Load model from file.

        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model_version = model_data["version"]
        self.training_stats = model_data["training_stats"]


# Export
__all__ = [
    "IonVMFeatures",
    "IonVMPrediction",
    "IonVMFeatureEngineer",
    "IonVirtualMetrologyModel",
]
