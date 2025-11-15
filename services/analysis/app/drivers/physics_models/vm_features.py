"""
Virtual Metrology Feature Engineering

Utilities for extracting features from process data and physics models
for training ML-based Virtual Metrology models.

VM models predict film properties (thickness, stress, adhesion) from
process telemetry without physical measurement.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from .thickness import ThicknessModel, DepositionParameters
from .stress import StressModel, ProcessConditions, MaterialProperties
from .adhesion import AdhesionModel, AdhesionFactors


@dataclass
class TelemetryData:
    """Process telemetry time series"""
    time_sec: np.ndarray
    temperature_c: np.ndarray
    pressure_torr: np.ndarray
    precursor_flow_sccm: np.ndarray
    rf_power_w: Optional[np.ndarray] = None
    bias_voltage_v: Optional[np.ndarray] = None


class VMFeatureExtractor:
    """
    Extract comprehensive features for VM model training

    Features include:
    1. Process parameters (static and time-averaged)
    2. Physics-based predictions
    3. Statistical features from telemetry
    4. Derived/engineered features
    """

    def __init__(self):
        self.thickness_model = ThicknessModel()
        self.stress_model = StressModel()
        self.adhesion_model = AdhesionModel()

    def extract_all_features(
        self,
        deposition_params: DepositionParameters,
        process_conditions: ProcessConditions,
        adhesion_factors: AdhesionFactors,
        telemetry: Optional[TelemetryData] = None,
    ) -> Dict[str, float]:
        """
        Extract comprehensive feature set

        Args:
            deposition_params: Deposition parameters for thickness model
            process_conditions: Process conditions for stress model
            adhesion_factors: Factors for adhesion model
            telemetry: Optional time-series telemetry data

        Returns:
            Dictionary of all features
        """
        features = {}

        # ====================================================================
        # 1. Process parameter features
        # ====================================================================
        features.update(self._extract_process_features(deposition_params))

        # ====================================================================
        # 2. Physics-based prediction features
        # ====================================================================
        features.update(self._extract_thickness_features(deposition_params))
        features.update(self._extract_stress_features(process_conditions))
        features.update(self._extract_adhesion_features(adhesion_factors))

        # ====================================================================
        # 3. Telemetry statistical features
        # ====================================================================
        if telemetry is not None:
            features.update(self._extract_telemetry_features(telemetry))

        # ====================================================================
        # 4. Derived/engineered features
        # ====================================================================
        features.update(self._extract_derived_features(features))

        return features

    def _extract_process_features(
        self,
        params: DepositionParameters,
    ) -> Dict[str, float]:
        """Extract process parameter features"""
        return {
            "temp_c": params.temperature_c,
            "pressure_torr": params.pressure_torr,
            "precursor_flow_sccm": params.precursor_flow_sccm,
            "carrier_flow_sccm": params.carrier_gas_flow_sccm,
            "dilution_flow_sccm": params.dilution_gas_flow_sccm,
            "total_flow_sccm": (
                params.precursor_flow_sccm +
                params.carrier_gas_flow_sccm +
                params.dilution_gas_flow_sccm
            ),
            "rf_power_w": params.rf_power_w,
            "rf_frequency_mhz": params.rf_frequency_mhz,
            "bias_voltage_v": params.bias_voltage_v,
            "rotation_speed_rpm": params.rotation_speed_rpm,
            "wafer_diameter_mm": params.wafer_diameter_mm,
        }

    def _extract_thickness_features(
        self,
        params: DepositionParameters,
    ) -> Dict[str, float]:
        """Extract thickness-related features from physics model"""
        return self.thickness_model.extract_vm_features(
            params=params,
            time_sec=3600.0,  # Nominal 1 hour
        )

    def _extract_stress_features(
        self,
        process: ProcessConditions,
    ) -> Dict[str, float]:
        """Extract stress-related features from physics model"""
        return self.stress_model.extract_vm_features(process)

    def _extract_adhesion_features(
        self,
        factors: AdhesionFactors,
    ) -> Dict[str, float]:
        """Extract adhesion-related features from physics model"""
        return self.adhesion_model.extract_vm_features(factors)

    def _extract_telemetry_features(
        self,
        telemetry: TelemetryData,
    ) -> Dict[str, float]:
        """
        Extract statistical features from telemetry time series

        Features:
        - Mean, std, min, max, range
        - Slope (linear fit)
        - Stability metrics (coefficient of variation)
        """
        features = {}

        # Temperature features
        features.update(self._compute_stats("temp", telemetry.temperature_c))

        # Pressure features
        features.update(self._compute_stats("pressure", telemetry.pressure_torr))

        # Flow features
        features.update(self._compute_stats("flow", telemetry.precursor_flow_sccm))

        # RF power features (if available)
        if telemetry.rf_power_w is not None:
            features.update(self._compute_stats("rf_power", telemetry.rf_power_w))

        # Bias voltage features (if available)
        if telemetry.bias_voltage_v is not None:
            features.update(self._compute_stats("bias", telemetry.bias_voltage_v))

        return features

    def _compute_stats(
        self,
        name: str,
        data: np.ndarray,
    ) -> Dict[str, float]:
        """Compute statistical features for a signal"""
        stats = {
            f"{name}_mean": np.mean(data),
            f"{name}_std": np.std(data),
            f"{name}_min": np.min(data),
            f"{name}_max": np.max(data),
            f"{name}_range": np.max(data) - np.min(data),
            f"{name}_cv": np.std(data) / (np.mean(data) + 1e-6),  # Coefficient of variation
        }

        # Linear trend (slope)
        if len(data) > 1:
            time_norm = np.linspace(0, 1, len(data))
            slope, _ = np.polyfit(time_norm, data, deg=1)
            stats[f"{name}_slope"] = slope

        return stats

    def _extract_derived_features(
        self,
        features: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Extract derived/engineered features

        These are combinations or transformations of base features
        that may improve ML model performance
        """
        derived = {}

        # Temperature-pressure interaction
        if "temp_c" in features and "pressure_torr" in features:
            derived["temp_pressure_product"] = features["temp_c"] * features["pressure_torr"]
            derived["temp_pressure_ratio"] = features["temp_c"] / (features["pressure_torr"] + 1e-6)

        # Flow-related ratios
        if "precursor_flow_sccm" in features and "total_flow_sccm" in features:
            derived["precursor_fraction"] = (
                features["precursor_flow_sccm"] / (features["total_flow_sccm"] + 1e-6)
            )

        # Power density (for PECVD)
        if "rf_power_w" in features and "wafer_diameter_mm" in features:
            wafer_area_cm2 = np.pi * (features["wafer_diameter_mm"] / 20.0)**2
            derived["power_density_w_cm2"] = features["rf_power_w"] / wafer_area_cm2

        # Stress-thickness interaction
        if "film_stress_mpa" in features and "film_thickness_nm" in features:
            derived["stress_thickness_product"] = (
                abs(features["film_stress_mpa"]) * features["film_thickness_nm"]
            )

        return derived


def create_training_dataset(
    deposition_params_list: List[DepositionParameters],
    measured_thickness_nm_list: List[float],
    measured_stress_mpa_list: Optional[List[float]] = None,
    measured_adhesion_score_list: Optional[List[float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Create training dataset for VM models

    Args:
        deposition_params_list: List of deposition parameters for each run
        measured_thickness_nm_list: Measured thickness values (ground truth)
        measured_stress_mpa_list: Measured stress values (optional)
        measured_adhesion_score_list: Measured adhesion scores (optional)

    Returns:
        Dictionary with X (features) and y (targets) arrays
    """
    extractor = VMFeatureExtractor()

    feature_list = []
    for params in deposition_params_list:
        # Create dummy process conditions and adhesion factors
        process_cond = ProcessConditions(
            temperature_c=params.temperature_c,
            pressure_torr=params.pressure_torr,
            deposition_rate_nm_min=50.0,  # Nominal
            rf_power_w=params.rf_power_w,
        )

        adhesion_fac = AdhesionFactors(
            film_stress_mpa=0.0,  # Will be updated with prediction
            pre_clean_quality=1.0,  # Assume good
            deposition_temp_c=params.temperature_c,
        )

        features = extractor.extract_all_features(
            deposition_params=params,
            process_conditions=process_cond,
            adhesion_factors=adhesion_fac,
        )

        feature_list.append(features)

    # Convert to arrays
    feature_names = list(feature_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in feature_list])

    dataset = {
        "X": X,
        "feature_names": feature_names,
        "y_thickness_nm": np.array(measured_thickness_nm_list),
    }

    if measured_stress_mpa_list is not None:
        dataset["y_stress_mpa"] = np.array(measured_stress_mpa_list)

    if measured_adhesion_score_list is not None:
        dataset["y_adhesion_score"] = np.array(measured_adhesion_score_list)

    return dataset


def feature_importance_analysis(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Analyze feature importance using correlation

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: List of feature names

    Returns:
        Dictionary of feature_name: importance_score
    """
    importance = {}

    for i, name in enumerate(feature_names):
        # Pearson correlation coefficient
        correlation = np.corrcoef(X[:, i], y)[0, 1]
        importance[name] = abs(correlation)  # Absolute value

    # Sort by importance
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    return importance_sorted
