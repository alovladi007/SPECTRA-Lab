"""
CVD Virtual Metrology - Feature Store
Feature engineering and storage for VM models
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
import joblib


logger = logging.getLogger(__name__)


# ============================================================================
# Feature Definitions
# ============================================================================

@dataclass
class FeatureDefinition:
    """Definition of a feature for VM models"""
    name: str
    description: str
    feature_type: str  # 'statistical', 'temporal', 'derived', 'raw'
    source_parameters: List[str]  # Source telemetry parameters
    calculation_method: str  # Method to calculate feature
    is_active: bool = True


class CVDFeatureEngineer:
    """
    Feature engineering for CVD virtual metrology.
    Extracts and transforms telemetry data into ML features.
    """

    def __init__(self):
        """Initialize feature engineer"""
        self.feature_definitions = self._define_features()
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None

    def _define_features(self) -> List[FeatureDefinition]:
        """Define standard CVD features"""
        features = [
            # Statistical features - Temperature
            FeatureDefinition(
                name="temp_mean",
                description="Mean temperature across all zones",
                feature_type="statistical",
                source_parameters=["temperatures"],
                calculation_method="mean",
            ),
            FeatureDefinition(
                name="temp_std",
                description="Temperature standard deviation",
                feature_type="statistical",
                source_parameters=["temperatures"],
                calculation_method="std",
            ),
            FeatureDefinition(
                name="temp_max",
                description="Maximum temperature",
                feature_type="statistical",
                source_parameters=["temperatures"],
                calculation_method="max",
            ),
            FeatureDefinition(
                name="temp_min",
                description="Minimum temperature",
                feature_type="statistical",
                source_parameters=["temperatures"],
                calculation_method="min",
            ),
            FeatureDefinition(
                name="temp_range",
                description="Temperature range (max - min)",
                feature_type="derived",
                source_parameters=["temperatures"],
                calculation_method="range",
            ),

            # Statistical features - Pressure
            FeatureDefinition(
                name="pressure_mean",
                description="Mean pressure",
                feature_type="statistical",
                source_parameters=["pressures"],
                calculation_method="mean",
            ),
            FeatureDefinition(
                name="pressure_std",
                description="Pressure standard deviation",
                feature_type="statistical",
                source_parameters=["pressures"],
                calculation_method="std",
            ),

            # Statistical features - Gas Flows
            FeatureDefinition(
                name="total_gas_flow",
                description="Total gas flow rate",
                feature_type="derived",
                source_parameters=["gas_flows"],
                calculation_method="sum",
            ),
            FeatureDefinition(
                name="gas_flow_ratio",
                description="Ratio of process gas to carrier gas",
                feature_type="derived",
                source_parameters=["gas_flows"],
                calculation_method="ratio",
            ),

            # Temporal features
            FeatureDefinition(
                name="temp_rate_of_change",
                description="Rate of temperature change",
                feature_type="temporal",
                source_parameters=["temperatures"],
                calculation_method="derivative",
            ),
            FeatureDefinition(
                name="pressure_stability",
                description="Pressure stability (inverse of std/mean)",
                feature_type="temporal",
                source_parameters=["pressures"],
                calculation_method="coefficient_of_variation",
            ),

            # Process time features
            FeatureDefinition(
                name="deposition_time",
                description="Total deposition time",
                feature_type="raw",
                source_parameters=["process_time"],
                calculation_method="direct",
            ),

            # Plasma features (if applicable)
            FeatureDefinition(
                name="rf_power_mean",
                description="Mean RF power",
                feature_type="statistical",
                source_parameters=["plasma_parameters"],
                calculation_method="mean",
            ),
            FeatureDefinition(
                name="dc_bias_mean",
                description="Mean DC bias voltage",
                feature_type="statistical",
                source_parameters=["plasma_parameters"],
                calculation_method="mean",
            ),
        ]

        return features

    def extract_features(
        self,
        telemetry_data: List[Dict[str, Any]],
        recipe_data: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Extract features from telemetry data.

        Args:
            telemetry_data: List of telemetry points
            recipe_data: Optional recipe parameters

        Returns:
            DataFrame with extracted features
        """
        if not telemetry_data:
            raise ValueError("No telemetry data provided")

        features = {}

        # Convert to DataFrame for easier processing
        df = pd.DataFrame(telemetry_data)

        # Extract statistical features
        features.update(self._extract_statistical_features(df))

        # Extract temporal features
        features.update(self._extract_temporal_features(df))

        # Extract derived features
        features.update(self._extract_derived_features(df))

        # Extract recipe features if available
        if recipe_data:
            features.update(self._extract_recipe_features(recipe_data))

        # Convert to DataFrame
        feature_df = pd.DataFrame([features])

        return feature_df

    def _extract_statistical_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract statistical features from telemetry"""
        features = {}

        # Temperature features
        if "temperatures" in df.columns:
            temp_values = []
            for temps in df["temperatures"]:
                if isinstance(temps, dict):
                    temp_values.extend(temps.values())

            if temp_values:
                features["temp_mean"] = np.mean(temp_values)
                features["temp_std"] = np.std(temp_values)
                features["temp_max"] = np.max(temp_values)
                features["temp_min"] = np.min(temp_values)
                features["temp_range"] = np.max(temp_values) - np.min(temp_values)
                features["temp_median"] = np.median(temp_values)

        # Pressure features
        if "pressures" in df.columns:
            pressure_values = []
            for pressures in df["pressures"]:
                if isinstance(pressures, dict):
                    pressure_values.extend(pressures.values())

            if pressure_values:
                features["pressure_mean"] = np.mean(pressure_values)
                features["pressure_std"] = np.std(pressure_values)
                features["pressure_median"] = np.median(pressure_values)

        # Gas flow features
        if "gas_flows" in df.columns:
            flow_values = []
            for flows in df["gas_flows"]:
                if isinstance(flows, dict):
                    flow_values.extend(flows.values())

            if flow_values:
                features["gas_flow_mean"] = np.mean(flow_values)
                features["gas_flow_std"] = np.std(flow_values)
                features["total_gas_flow"] = np.sum(flow_values)

        # Plasma features (if present)
        if "plasma_parameters" in df.columns:
            rf_power_values = []
            dc_bias_values = []

            for plasma in df["plasma_parameters"]:
                if isinstance(plasma, dict):
                    if "rf_power_w" in plasma:
                        rf_power_values.append(plasma["rf_power_w"])
                    if "dc_bias_v" in plasma:
                        dc_bias_values.append(plasma["dc_bias_v"])

            if rf_power_values:
                features["rf_power_mean"] = np.mean(rf_power_values)
                features["rf_power_std"] = np.std(rf_power_values)

            if dc_bias_values:
                features["dc_bias_mean"] = np.mean(dc_bias_values)
                features["dc_bias_std"] = np.std(dc_bias_values)

        return features

    def _extract_temporal_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract temporal/time-series features"""
        features = {}

        # Temperature rate of change
        if "temperatures" in df.columns and len(df) > 1:
            temp_series = []
            for temps in df["temperatures"]:
                if isinstance(temps, dict):
                    temp_series.append(np.mean(list(temps.values())))

            if len(temp_series) > 1:
                temp_diff = np.diff(temp_series)
                features["temp_rate_of_change"] = np.mean(temp_diff)
                features["temp_rate_std"] = np.std(temp_diff)

        # Pressure stability
        if "pressures" in df.columns:
            pressure_series = []
            for pressures in df["pressures"]:
                if isinstance(pressures, dict):
                    chamber_pressure = pressures.get("chamber", 0)
                    pressure_series.append(chamber_pressure)

            if len(pressure_series) > 0:
                pressure_mean = np.mean(pressure_series)
                pressure_std = np.std(pressure_series)
                if pressure_mean > 0:
                    features["pressure_cv"] = pressure_std / pressure_mean
                features["pressure_stability"] = 1.0 / (1.0 + features.get("pressure_cv", 0))

        return features

    def _extract_derived_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract derived/calculated features"""
        features = {}

        # Process duration
        if "timestamp" in df.columns and len(df) > 1:
            timestamps = pd.to_datetime(df["timestamp"])
            duration_s = (timestamps.max() - timestamps.min()).total_seconds()
            features["process_duration_s"] = duration_s
            features["process_duration_min"] = duration_s / 60.0

        # Temperature uniformity
        if "temp_mean" in features and "temp_std" in features and features["temp_mean"] > 0:
            features["temp_uniformity"] = features["temp_std"] / features["temp_mean"]

        return features

    def _extract_recipe_features(self, recipe_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from recipe parameters"""
        features = {}

        # Target thickness
        if "target_thickness_nm" in recipe_data:
            features["recipe_target_thickness"] = recipe_data["target_thickness_nm"]

        # Process time
        if "process_time_s" in recipe_data:
            features["recipe_process_time"] = recipe_data["process_time_s"]

        # Temperature setpoints
        if "temperature_profile" in recipe_data:
            temp_profile = recipe_data["temperature_profile"]
            if "zones" in temp_profile:
                zones = temp_profile["zones"]
                if zones:
                    setpoints = [z.get("setpoint_c", 0) for z in zones]
                    features["recipe_temp_setpoint_mean"] = np.mean(setpoints)
                    features["recipe_temp_setpoint_std"] = np.std(setpoints)

        # Pressure setpoint
        if "pressure_profile" in recipe_data:
            pressure_profile = recipe_data["pressure_profile"]
            if "process_pressure_pa" in pressure_profile:
                features["recipe_pressure_setpoint"] = pressure_profile["process_pressure_pa"]

        # Plasma settings
        if "plasma_settings" in recipe_data:
            plasma = recipe_data["plasma_settings"]
            if isinstance(plasma, dict):
                if "rf_power_w" in plasma:
                    features["recipe_rf_power"] = plasma["rf_power_w"]
                if "bias_voltage_v" in plasma:
                    features["recipe_dc_bias"] = plasma["bias_voltage_v"]

        return features

    def fit_scaler(self, X: pd.DataFrame, method: str = "standard") -> None:
        """
        Fit feature scaler.

        Args:
            X: Feature matrix
            method: Scaling method ('standard', 'robust')
        """
        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        self.scaler.fit(X)
        logger.info(f"Fitted {method} scaler on {X.shape[0]} samples")

    def transform_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Args:
            X: Feature matrix

        Returns:
            Scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler() first.")

        return self.scaler.transform(X)

    def fit_pca(self, X: pd.DataFrame, n_components: int = 10) -> None:
        """
        Fit PCA for dimensionality reduction.

        Args:
            X: Feature matrix
            n_components: Number of components
        """
        # Scale first
        if self.scaler is None:
            self.fit_scaler(X)

        X_scaled = self.transform_features(X)

        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)

        explained_var = np.sum(self.pca.explained_variance_ratio_)
        logger.info(
            f"Fitted PCA with {n_components} components "
            f"(explained variance: {explained_var:.2%})"
        )

    def apply_pca(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply PCA transformation.

        Args:
            X: Feature matrix

        Returns:
            PCA-transformed features
        """
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca() first.")

        X_scaled = self.transform_features(X)
        return self.pca.transform(X_scaled)

    def save(self, filepath: str) -> None:
        """Save feature engineer state"""
        state = {
            "scaler": self.scaler,
            "pca": self.pca,
            "feature_definitions": self.feature_definitions,
        }
        joblib.dump(state, filepath)
        logger.info(f"Saved feature engineer to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "CVDFeatureEngineer":
        """Load feature engineer from file"""
        state = joblib.load(filepath)

        engineer = cls()
        engineer.scaler = state.get("scaler")
        engineer.pca = state.get("pca")
        engineer.feature_definitions = state.get("feature_definitions", engineer.feature_definitions)

        logger.info(f"Loaded feature engineer from {filepath}")
        return engineer


# ============================================================================
# Feature Store
# ============================================================================

class FeatureStore:
    """
    Feature store for managing and versioning VM features.
    """

    def __init__(self, storage_path: str = "./feature_store"):
        """
        Initialize feature store.

        Args:
            storage_path: Path to store features
        """
        self.storage_path = storage_path
        self.feature_cache: Dict[str, pd.DataFrame] = {}

    def store_features(
        self,
        run_id: UUID,
        features: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store features for a run.

        Args:
            run_id: Run ID
            features: Feature DataFrame
            metadata: Optional metadata

        Returns:
            Feature set ID
        """
        feature_id = str(uuid4())

        # Store features
        feature_data = {
            "id": feature_id,
            "run_id": str(run_id),
            "features": features,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat(),
        }

        # Cache in memory
        self.feature_cache[feature_id] = features

        # TODO: Persist to database or file storage

        logger.info(f"Stored features for run {run_id}: {feature_id}")

        return feature_id

    def retrieve_features(
        self,
        run_id: Optional[UUID] = None,
        feature_id: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve features.

        Args:
            run_id: Run ID
            feature_id: Feature set ID

        Returns:
            Features DataFrame or None
        """
        if feature_id and feature_id in self.feature_cache:
            return self.feature_cache[feature_id]

        # TODO: Query from database

        return None

    def list_features(
        self,
        process_mode_id: Optional[UUID] = None,
        recipe_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        List available feature sets.

        Args:
            process_mode_id: Filter by process mode
            recipe_id: Filter by recipe
            start_date: Start date filter
            end_date: End date filter

        Returns:
            List of feature set metadata
        """
        # TODO: Implement database query

        return []
