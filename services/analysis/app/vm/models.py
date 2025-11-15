"""
VM Model Definitions

Defines ML models for virtual metrology prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator


class FilmFamily(str, Enum):
    """Supported film families"""
    SIO2 = "SiO2"
    SI3N4 = "Si3N4"
    W = "W"
    TIN = "TiN"
    GAN = "GaN"
    DLC = "DLC"


class PredictionTarget(str, Enum):
    """Prediction targets"""
    THICKNESS_MEAN = "thickness_mean_nm"
    THICKNESS_UNIFORMITY = "thickness_uniformity_pct"
    STRESS_MEAN = "stress_mpa_mean"
    ADHESION_SCORE = "adhesion_score"
    ADHESION_CLASS = "adhesion_class"


@dataclass
class VMModel:
    """
    Virtual Metrology Model

    Predicts film properties from process parameters and telemetry.
    """
    film_family: FilmFamily
    target: PredictionTarget
    model_type: str = "random_forest"  # "random_forest", "gradient_boosting", "neural_network"

    # ML model components
    scaler: Optional[StandardScaler] = None
    model: Optional[BaseEstimator] = None

    # Feature information
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0

    # Model performance
    train_score: Optional[float] = None
    val_score: Optional[float] = None
    test_score: Optional[float] = None

    # Metadata
    training_date: Optional[str] = None
    version: str = "1.0"
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Fit the model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: Optional list of feature names
        """
        if feature_names is not None:
            self.feature_names = feature_names
            self.n_features = len(feature_names)

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Create and fit model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=self.hyperparameters.get("n_estimators", 100),
                max_depth=self.hyperparameters.get("max_depth", 10),
                min_samples_split=self.hyperparameters.get("min_samples_split", 5),
                min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 2),
                random_state=42,
                n_jobs=-1,
            )

        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=self.hyperparameters.get("n_estimators", 100),
                max_depth=self.hyperparameters.get("max_depth", 5),
                learning_rate=self.hyperparameters.get("learning_rate", 0.1),
                min_samples_split=self.hyperparameters.get("min_samples_split", 5),
                min_samples_leaf=self.hyperparameters.get("min_samples_leaf", 2),
                random_state=42,
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_scaled, y)

        # Compute train score
        self.train_score = self.model.score(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates

        For tree-based ensembles, uncertainty is estimated from
        the standard deviation of individual tree predictions.

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            (predictions, uncertainties)
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained. Call fit() first.")

        X_scaled = self.scaler.transform(X)

        # Get predictions from individual estimators
        if hasattr(self.model, "estimators_"):
            # Tree-based ensemble
            predictions_all = np.array([
                estimator.predict(X_scaled)
                for estimator in self.model.estimators_
            ])

            predictions = np.mean(predictions_all, axis=0)
            uncertainties = np.std(predictions_all, axis=0)

        else:
            # Single model - no uncertainty estimate
            predictions = self.model.predict(X_scaled)
            uncertainties = np.zeros_like(predictions)

        return predictions, uncertainties

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores

        Returns:
            Dictionary of feature_name: importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        if not hasattr(self.model, "feature_importances_"):
            return {}

        importances = self.model.feature_importances_

        if not self.feature_names:
            return {f"feature_{i}": imp for i, imp in enumerate(importances)}

        return {
            name: importance
            for name, importance in zip(self.feature_names, importances)
        }

    def get_top_features(self, n: int = 10) -> List[tuple[str, float]]:
        """
        Get top n most important features

        Args:
            n: Number of features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        importance_dict = self.get_feature_importance()

        sorted_features = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return sorted_features[:n]


def create_vm_model(
    film_family: FilmFamily,
    target: PredictionTarget,
    model_type: str = "random_forest",
    hyperparameters: Optional[Dict[str, Any]] = None,
) -> VMModel:
    """
    Factory function to create VM model

    Args:
        film_family: Film family (SiO2, Si3N4, etc.)
        target: Prediction target
        model_type: Model type ("random_forest", "gradient_boosting")
        hyperparameters: Optional hyperparameter dict

    Returns:
        VMModel instance
    """
    if hyperparameters is None:
        # Default hyperparameters by film family and target
        hyperparameters = get_default_hyperparameters(film_family, target, model_type)

    return VMModel(
        film_family=film_family,
        target=target,
        model_type=model_type,
        hyperparameters=hyperparameters,
    )


def get_default_hyperparameters(
    film_family: FilmFamily,
    target: PredictionTarget,
    model_type: str,
) -> Dict[str, Any]:
    """
    Get default hyperparameters for a given configuration

    These are tuned values based on typical CVD datasets.
    """
    if model_type == "random_forest":
        # Thickness predictions need deeper trees
        if target in [PredictionTarget.THICKNESS_MEAN, PredictionTarget.THICKNESS_UNIFORMITY]:
            return {
                "n_estimators": 150,
                "max_depth": 12,
                "min_samples_split": 4,
                "min_samples_leaf": 2,
            }

        # Stress predictions are more complex
        elif target == PredictionTarget.STRESS_MEAN:
            return {
                "n_estimators": 200,
                "max_depth": 15,
                "min_samples_split": 3,
                "min_samples_leaf": 1,
            }

        # Adhesion is simpler
        else:
            return {
                "n_estimators": 100,
                "max_depth": 8,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            }

    elif model_type == "gradient_boosting":
        return {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
        }

    else:
        return {}
