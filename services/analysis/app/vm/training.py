"""
VM Model Training

Training pipeline for Virtual Metrology models.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
import logging

from .models import VMModel, FilmFamily, PredictionTarget, create_vm_model

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for VM model training"""
    # Data split
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42

    # Cross-validation
    use_cv: bool = True
    cv_folds: int = 5

    # Model selection
    model_type: str = "random_forest"
    hyperparameters: Dict = field(default_factory=dict)

    # Training options
    verbose: bool = True


class VMTrainer:
    """
    VM Model Trainer

    Handles data splitting, training, validation, and model selection.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
    ):
        """
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        self.model: Optional[VMModel] = None

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list] = None,
        film_family: FilmFamily = FilmFamily.SI3N4,
        target: PredictionTarget = PredictionTarget.THICKNESS_MEAN,
    ) -> VMModel:
        """
        Train a VM model

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: Optional feature names
            film_family: Film family
            target: Prediction target

        Returns:
            Trained VMModel
        """
        n_samples, n_features = X.shape

        if self.config.verbose:
            logger.info(f"Training VM model for {film_family.value} - {target.value}")
            logger.info(f"Dataset size: {n_samples} samples, {n_features} features")

        # Split data
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        if self.config.val_size > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val,
                test_size=self.config.val_size / (1 - self.config.test_size),
                random_state=self.config.random_state,
            )
        else:
            X_train, y_train = X_train_val, y_train_val
            X_val, y_val = None, None

        if self.config.verbose:
            logger.info(f"Train set: {len(y_train)} samples")
            if X_val is not None:
                logger.info(f"Val set: {len(y_val)} samples")
            logger.info(f"Test set: {len(y_test)} samples")

        # Create model
        self.model = create_vm_model(
            film_family=film_family,
            target=target,
            model_type=self.config.model_type,
            hyperparameters=self.config.hyperparameters,
        )

        # Train model
        self.model.fit(X_train, y_train, feature_names=feature_names)

        # Evaluate on validation set
        if X_val is not None:
            y_val_pred = self.model.predict(X_val)
            val_score = self.model.model.score(self.model.scaler.transform(X_val), y_val)
            self.model.val_score = val_score

            if self.config.verbose:
                logger.info(f"Validation R²: {val_score:.4f}")

        # Cross-validation
        if self.config.use_cv and len(y_train) >= self.config.cv_folds:
            cv_scores = cross_val_score(
                self.model.model,
                self.model.scaler.transform(X_train),
                y_train,
                cv=self.config.cv_folds,
                scoring="r2",
            )

            if self.config.verbose:
                logger.info(f"CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        # Evaluate on test set
        y_test_pred = self.model.predict(X_test)
        test_score = self.model.model.score(self.model.scaler.transform(X_test), y_test)
        self.model.test_score = test_score

        if self.config.verbose:
            logger.info(f"Test R²: {test_score:.4f}")

        # Set training date
        self.model.training_date = datetime.now().isoformat()

        # Show top features
        if self.config.verbose and feature_names:
            logger.info("\nTop 10 important features:")
            for name, importance in self.model.get_top_features(10):
                logger.info(f"  {name}: {importance:.4f}")

        return self.model

    def train_multiple_targets(
        self,
        X: np.ndarray,
        y_dict: Dict[PredictionTarget, np.ndarray],
        feature_names: Optional[list] = None,
        film_family: FilmFamily = FilmFamily.SI3N4,
    ) -> Dict[PredictionTarget, VMModel]:
        """
        Train models for multiple targets using the same features

        Args:
            X: Feature matrix (n_samples, n_features)
            y_dict: Dictionary of target -> values
            feature_names: Optional feature names
            film_family: Film family

        Returns:
            Dictionary of target -> trained model
        """
        models = {}

        for target, y in y_dict.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Training model for target: {target.value}")
            logger.info(f"{'='*70}")

            model = self.train(
                X=X,
                y=y,
                feature_names=feature_names,
                film_family=film_family,
                target=target,
            )

            models[target] = model

        return models


def train_vm_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    film_family: FilmFamily = FilmFamily.SI3N4,
    target: PredictionTarget = PredictionTarget.THICKNESS_MEAN,
    config: Optional[TrainingConfig] = None,
) -> VMModel:
    """
    Convenience function to train a VM model

    Args:
        X: Feature matrix
        y: Target values
        feature_names: Optional feature names
        film_family: Film family
        target: Prediction target
        config: Optional training config

    Returns:
        Trained VMModel
    """
    trainer = VMTrainer(config)
    return trainer.train(X, y, feature_names, film_family, target)
