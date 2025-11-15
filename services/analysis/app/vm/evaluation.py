"""
VM Model Evaluation

Evaluation metrics and diagnostics for VM models.
"""

import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import logging

from .models import VMModel

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for VM model"""
    # Regression metrics
    r2: float
    rmse: float
    mae: float
    mape: float  # Mean Absolute Percentage Error

    # Residual statistics
    mean_residual: float
    std_residual: float

    # Process capability metrics
    # (assumes target has specification limits)
    within_spec_pct: Optional[float] = None
    cpk: Optional[float] = None  # Process capability index


class VMEvaluator:
    """
    VM Model Evaluator

    Computes evaluation metrics and diagnostics.
    """

    def __init__(self, model: VMModel):
        """
        Args:
            model: Trained VM model
        """
        self.model = model

    def evaluate(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        spec_limits: Optional[Dict[str, float]] = None,
    ) -> EvaluationMetrics:
        """
        Evaluate model on test data

        Args:
            X: Feature matrix
            y_true: True target values
            spec_limits: Optional specification limits
                         {"lower": LSL, "upper": USL}

        Returns:
            EvaluationMetrics
        """
        # Make predictions
        y_pred = self.model.predict(X)

        # Calculate metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        # MAPE (handle division by zero)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        # Residuals
        residuals = y_true - y_pred
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)

        # Specification compliance (if provided)
        within_spec_pct = None
        cpk = None

        if spec_limits:
            lsl = spec_limits.get("lower")
            usl = spec_limits.get("upper")

            if lsl is not None and usl is not None:
                within_spec = np.sum((y_pred >= lsl) & (y_pred <= usl))
                within_spec_pct = (within_spec / len(y_pred)) * 100.0

                # Cpk = min((USL - μ) / 3σ, (μ - LSL) / 3σ)
                mu = np.mean(y_pred)
                sigma = np.std(y_pred)

                if sigma > 0:
                    cpu = (usl - mu) / (3 * sigma)
                    cpl = (mu - lsl) / (3 * sigma)
                    cpk = min(cpu, cpl)

        return EvaluationMetrics(
            r2=r2,
            rmse=rmse,
            mae=mae,
            mape=mape,
            mean_residual=mean_residual,
            std_residual=std_residual,
            within_spec_pct=within_spec_pct,
            cpk=cpk,
        )

    def plot_diagnostics(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
    ):
        """
        Generate diagnostic plots (placeholder for actual plotting)

        In production, this would generate:
        - Predicted vs Actual plot
        - Residual plot
        - Residual histogram
        - Q-Q plot

        Args:
            X: Feature matrix
            y_true: True target values
        """
        y_pred = self.model.predict(X)
        residuals = y_true - y_pred

        # Calculate statistics for logging
        logger.info("\nDiagnostic Statistics:")
        logger.info(f"  Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
        logger.info(f"  Actual range: [{y_true.min():.2f}, {y_true.max():.2f}]")
        logger.info(f"  Residual range: [{residuals.min():.2f}, {residuals.max():.2f}]")
        logger.info(f"  Residual mean: {residuals.mean():.4f}")
        logger.info(f"  Residual std: {residuals.std():.4f}")

        # In production, would generate matplotlib plots here
        # For now, just log the statistics

    def check_prediction_uncertainty(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """
        Check prediction uncertainty calibration

        Args:
            X: Feature matrix
            y_true: True values
            confidence_level: Confidence level (default 0.95)

        Returns:
            Dictionary with uncertainty metrics
        """
        y_pred, y_std = self.model.predict_with_uncertainty(X)

        # Calculate prediction intervals
        z_score = 1.96  # For 95% confidence
        if confidence_level == 0.99:
            z_score = 2.576

        lower_bound = y_pred - z_score * y_std
        upper_bound = y_pred + z_score * y_std

        # Check calibration: how many true values fall within intervals
        within_interval = np.sum((y_true >= lower_bound) & (y_true <= upper_bound))
        coverage = within_interval / len(y_true)

        # Average interval width
        avg_interval_width = np.mean(upper_bound - lower_bound)

        return {
            "coverage": coverage,
            "expected_coverage": confidence_level,
            "avg_interval_width": avg_interval_width,
            "avg_uncertainty": np.mean(y_std),
        }


def evaluate_vm_model(
    model: VMModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    spec_limits: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> EvaluationMetrics:
    """
    Convenience function to evaluate a VM model

    Args:
        model: Trained VM model
        X_test: Test features
        y_test: Test target values
        spec_limits: Optional specification limits
        verbose: Print metrics

    Returns:
        EvaluationMetrics
    """
    evaluator = VMEvaluator(model)
    metrics = evaluator.evaluate(X_test, y_test, spec_limits)

    if verbose:
        logger.info(f"\nEvaluation Metrics:")
        logger.info(f"  R²: {metrics.r2:.4f}")
        logger.info(f"  RMSE: {metrics.rmse:.4f}")
        logger.info(f"  MAE: {metrics.mae:.4f}")
        logger.info(f"  MAPE: {metrics.mape:.2f}%")
        logger.info(f"  Mean residual: {metrics.mean_residual:.4f}")
        logger.info(f"  Std residual: {metrics.std_residual:.4f}")

        if metrics.within_spec_pct is not None:
            logger.info(f"  Within spec: {metrics.within_spec_pct:.1f}%")

        if metrics.cpk is not None:
            logger.info(f"  Cpk: {metrics.cpk:.2f}")

    return metrics
