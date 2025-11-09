"""
Virtual Metrology Models - Session 8

Implements Ridge, Lasso, and XGBoost models for predicting post-process KPIs
from in-situ FDC data and recipe parameters.

Includes:
- Model training with k-fold cross-validation
- Permutation feature importance
- Model cards with metadata
- Artifact persistence

Status: PRODUCTION READY ✅
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelCard:
    """
    Model card with metadata for reproducibility and governance.

    Attributes:
        model_name: Name of the model
        model_type: Type (ridge, lasso, xgboost)
        version: Model version
        target: Target variable name
        features: List of feature names
        n_samples: Number of training samples
        cv_scores: Cross-validation scores
        test_metrics: Test set metrics
        feature_importance: Feature importance scores
        hyperparameters: Model hyperparameters
        training_date: Date/time of training
        trained_by: User who trained the model
        description: Model description
    """
    model_name: str
    model_type: str
    version: str
    target: str
    features: List[str]
    n_samples: int
    cv_scores: Dict[str, float]
    test_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_date: str
    trained_by: str
    description: str


class VirtualMetrologyModel:
    """
    Virtual Metrology model for predicting post-process KPIs.

    Supports Ridge, Lasso, and XGBoost (via GradientBoostingRegressor).
    """

    def __init__(
        self,
        model_type: str = "ridge",
        target: str = "junction_depth",
        **hyperparameters
    ):
        """
        Initialize VM model.

        Args:
            model_type: Type of model ("ridge", "lasso", "xgboost")
            target: Target variable name
            **hyperparameters: Model-specific hyperparameters
        """
        self.model_type = model_type
        self.target = target
        self.hyperparameters = hyperparameters
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names: Optional[List[str]] = None
        self.model_card: Optional[ModelCard] = None

        # Initialize model
        self._init_model()

    def _init_model(self):
        """Initialize the underlying model based on type."""
        if self.model_type == "ridge":
            alpha = self.hyperparameters.get('alpha', 1.0)
            self.model = Ridge(alpha=alpha)

        elif self.model_type == "lasso":
            alpha = self.hyperparameters.get('alpha', 1.0)
            self.model = Lasso(alpha=alpha)

        elif self.model_type == "xgboost":
            # Using GradientBoostingRegressor as XGBoost alternative
            n_estimators = self.hyperparameters.get('n_estimators', 100)
            max_depth = self.hyperparameters.get('max_depth', 3)
            learning_rate = self.hyperparameters.get('learning_rate', 0.1)
            self.model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        test_size: float = 0.2,
        trained_by: str = "system",
        description: str = ""
    ) -> ModelCard:
        """
        Train the model with cross-validation.

        Args:
            X: Feature matrix
            y: Target variable
            cv_folds: Number of CV folds
            test_size: Fraction for test set
            trained_by: User who trained the model
            description: Model description

        Returns:
            ModelCard with training metadata
        """
        self.feature_names = list(X.columns)

        # Split train/test
        n_test = int(len(X) * test_size)
        X_train = X.iloc[:-n_test]
        y_train = y.iloc[:-n_test]
        X_test = X.iloc[-n_test:]
        y_test = y.iloc[-n_test:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Cross-validation
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores_r2 = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=kfold, scoring='r2'
        )
        cv_scores_neg_mse = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=kfold, scoring='neg_mean_squared_error'
        )

        # Train on full training set
        self.model.fit(X_train_scaled, y_train)

        # Test set predictions
        y_pred = self.model.predict(X_test_scaled)

        # Test metrics
        test_metrics = {
            'r2': float(r2_score(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'mae': float(mean_absolute_error(y_test, y_pred))
        }

        # Feature importance
        feature_importance = self._compute_feature_importance(
            X_test_scaled, y_test
        )

        # Create model card
        self.model_card = ModelCard(
            model_name=f"vm_{self.target}_{self.model_type}",
            model_type=self.model_type,
            version="1.0.0",
            target=self.target,
            features=self.feature_names,
            n_samples=len(X_train),
            cv_scores={
                'r2_mean': float(np.mean(cv_scores_r2)),
                'r2_std': float(np.std(cv_scores_r2)),
                'mse_mean': float(-np.mean(cv_scores_neg_mse)),
                'mse_std': float(np.std(cv_scores_neg_mse))
            },
            test_metrics=test_metrics,
            feature_importance=feature_importance,
            hyperparameters=self.hyperparameters,
            training_date=datetime.now().isoformat(),
            trained_by=trained_by,
            description=description or f"VM model for {self.target}"
        )

        return self.model_card

    def _compute_feature_importance(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute permutation feature importance.

        Args:
            X_test: Test features (scaled)
            y_test: Test targets

        Returns:
            Dictionary of feature importances
        """
        result = permutation_importance(
            self.model, X_test, y_test,
            n_repeats=10, random_state=42
        )

        importance_dict = {}
        for i, feature in enumerate(self.feature_names):
            importance_dict[feature] = float(result.importances_mean[i])

        # Sort by importance
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )

        return importance_dict

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Ensure feature order matches training
        X = X[self.feature_names]

        # Scale features
        X_scaled = self.scaler.transform(X)

        return self.model.predict(X_scaled)

    def save(self, artifacts_dir: Path, model_name: Optional[str] = None):
        """
        Save model artifacts.

        Args:
            artifacts_dir: Directory to save artifacts
            model_name: Optional custom model name
        """
        if self.model_card is None:
            raise ValueError("Model not trained. Call train() first.")

        # Create model directory
        name = model_name or self.model_card.model_name
        version = self.model_card.version
        model_dir = artifacts_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        with open(model_dir / "model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

        # Save scaler
        with open(model_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save model card
        with open(model_dir / "model_card.json", 'w') as f:
            json.dump(asdict(self.model_card), f, indent=2)

        # Save feature names
        with open(model_dir / "features.json", 'w') as f:
            json.dump({'features': self.feature_names}, f, indent=2)

    @classmethod
    def load(cls, artifacts_dir: Path, model_name: str, version: str = "1.0.0"):
        """
        Load model from artifacts.

        Args:
            artifacts_dir: Directory containing artifacts
            model_name: Name of the model
            version: Model version

        Returns:
            Loaded VirtualMetrologyModel
        """
        model_dir = artifacts_dir / model_name / version

        # Load model card
        with open(model_dir / "model_card.json", 'r') as f:
            card_dict = json.load(f)

        # Create instance
        instance = cls(
            model_type=card_dict['model_type'],
            target=card_dict['target'],
            **card_dict['hyperparameters']
        )

        # Load model
        with open(model_dir / "model.pkl", 'rb') as f:
            instance.model = pickle.load(f)

        # Load scaler
        with open(model_dir / "scaler.pkl", 'rb') as f:
            instance.scaler = pickle.load(f)

        # Load features
        with open(model_dir / "features.json", 'r') as f:
            instance.feature_names = json.load(f)['features']

        # Reconstruct model card
        instance.model_card = ModelCard(**card_dict)

        return instance


def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    target: str,
    artifacts_dir: Path,
    trained_by: str = "system"
) -> Dict[str, ModelCard]:
    """
    Train ensemble of Ridge, Lasso, and XGBoost models.

    Args:
        X: Feature matrix
        y: Target variable
        target: Target name
        artifacts_dir: Directory to save artifacts
        trained_by: User training the models

    Returns:
        Dictionary of model cards
    """
    model_cards = {}

    # Ridge
    ridge = VirtualMetrologyModel(
        model_type="ridge",
        target=target,
        alpha=1.0
    )
    ridge_card = ridge.train(X, y, trained_by=trained_by)
    ridge.save(artifacts_dir)
    model_cards['ridge'] = ridge_card

    # Lasso
    lasso = VirtualMetrologyModel(
        model_type="lasso",
        target=target,
        alpha=0.1
    )
    lasso_card = lasso.train(X, y, trained_by=trained_by)
    lasso.save(artifacts_dir)
    model_cards['lasso'] = lasso_card

    # XGBoost
    xgb = VirtualMetrologyModel(
        model_type="xgboost",
        target=target,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1
    )
    xgb_card = xgb.train(X, y, trained_by=trained_by)
    xgb.save(artifacts_dir)
    model_cards['xgboost'] = xgb_card

    return model_cards


def get_best_model(
    model_cards: Dict[str, ModelCard],
    metric: str = 'r2'
) -> str:
    """
    Get best model based on test metric.

    Args:
        model_cards: Dictionary of model cards
        metric: Metric to compare ('r2', 'rmse', 'mae')

    Returns:
        Name of best model
    """
    if metric == 'r2':
        # Higher is better
        best_model = max(
            model_cards.items(),
            key=lambda x: x[1].test_metrics['r2']
        )[0]
    else:
        # Lower is better
        best_model = min(
            model_cards.items(),
            key=lambda x: x[1].test_metrics[metric]
        )[0]

    return best_model
