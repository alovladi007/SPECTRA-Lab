"""Model drift monitoring for detecting data and concept drift.

Monitors feature distributions and prediction performance to detect when
models need retraining.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from scipy import stats
import time


# ============================================================================
# Drift Detection Data Structures
# ============================================================================

class DriftType(Enum):
    """Types of drift."""
    FEATURE_DRIFT = "feature_drift"  # Input distribution changed
    PREDICTION_DRIFT = "prediction_drift"  # Output distribution changed
    CONCEPT_DRIFT = "concept_drift"  # Input-output relationship changed
    NO_DRIFT = "no_drift"


class DriftSeverity(Enum):
    """Drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Alert for detected drift."""
    timestamp: float
    drift_type: DriftType
    severity: DriftSeverity
    feature_name: Optional[str] = None
    drift_score: float = 0.0
    p_value: float = 1.0
    message: str = ""
    recommended_action: str = ""


@dataclass
class FeatureStatistics:
    """Statistics for a feature."""
    feature_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float  # 25th percentile
    q75: float  # 75th percentile
    sample_count: int

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "feature_name": self.feature_name,
            "mean": float(self.mean),
            "std": float(self.std),
            "min": float(self.min),
            "max": float(self.max),
            "median": float(self.median),
            "q25": float(self.q25),
            "q75": float(self.q75),
            "sample_count": self.sample_count,
        }


# ============================================================================
# Statistical Drift Tests
# ============================================================================

class DriftDetector:
    """Statistical tests for drift detection."""

    @staticmethod
    def kolmogorov_smirnov_test(
        baseline: np.ndarray,
        current: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float, bool]:
        """Kolmogorov-Smirnov test for distribution drift.

        Args:
            baseline: Baseline sample
            current: Current sample
            alpha: Significance level

        Returns:
            (statistic, p_value, is_drift)
        """
        statistic, p_value = stats.ks_2samp(baseline, current)
        is_drift = p_value < alpha

        return statistic, p_value, is_drift

    @staticmethod
    def population_stability_index(
        baseline: np.ndarray,
        current: np.ndarray,
        num_bins: int = 10
    ) -> float:
        """Population Stability Index (PSI) for drift detection.

        PSI < 0.1: No significant drift
        0.1 <= PSI < 0.25: Moderate drift
        PSI >= 0.25: Significant drift

        Args:
            baseline: Baseline sample
            current: Current sample
            num_bins: Number of bins for histogram

        Returns:
            PSI value
        """
        # Create bins based on baseline quantiles
        bin_edges = np.percentile(baseline, np.linspace(0, 100, num_bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate distributions
        baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
        current_counts, _ = np.histogram(current, bins=bin_edges)

        # Normalize to proportions
        baseline_prop = (baseline_counts + 1e-6) / (np.sum(baseline_counts) + num_bins * 1e-6)
        current_prop = (current_counts + 1e-6) / (np.sum(current_counts) + num_bins * 1e-6)

        # Calculate PSI
        psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))

        return psi

    @staticmethod
    def jensen_shannon_divergence(
        baseline: np.ndarray,
        current: np.ndarray,
        num_bins: int = 50
    ) -> float:
        """Jensen-Shannon divergence for distribution comparison.

        JSD is symmetric version of KL divergence, bounded [0, 1].

        Args:
            baseline: Baseline sample
            current: Current sample
            num_bins: Number of bins

        Returns:
            JSD value (0 = identical, 1 = completely different)
        """
        # Create histogram
        combined = np.concatenate([baseline, current])
        bins = np.histogram_bin_edges(combined, bins=num_bins)

        p, _ = np.histogram(baseline, bins=bins, density=True)
        q, _ = np.histogram(current, bins=bins, density=True)

        # Normalize
        p = (p + 1e-10) / np.sum(p + 1e-10)
        q = (q + 1e-10) / np.sum(q + 1e-10)

        # Calculate JSD
        m = 0.5 * (p + q)
        jsd = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))

        return np.sqrt(jsd)  # Square root for metric property


# ============================================================================
# Feature Drift Monitor
# ============================================================================

class FeatureDriftMonitor:
    """Monitor individual feature distributions for drift."""

    def __init__(
        self,
        feature_name: str,
        window_size: int = 1000,
        alert_threshold_psi: float = 0.25,
        alert_threshold_ks: float = 0.05
    ):
        """Initialize feature drift monitor.

        Args:
            feature_name: Name of feature to monitor
            window_size: Size of rolling window for current data
            alert_threshold_psi: PSI threshold for alerting
            alert_threshold_ks: KS test p-value threshold
        """
        self.feature_name = feature_name
        self.window_size = window_size
        self.alert_threshold_psi = alert_threshold_psi
        self.alert_threshold_ks = alert_threshold_ks

        # Baseline statistics (from training data)
        self.baseline_stats: Optional[FeatureStatistics] = None
        self.baseline_data: Optional[np.ndarray] = None

        # Current data window
        self.current_window: deque = deque(maxlen=window_size)

        # Drift history
        self.drift_scores: List[float] = []
        self.alerts: List[DriftAlert] = []

    def set_baseline(self, data: np.ndarray):
        """Set baseline distribution from training data.

        Args:
            data: Baseline feature values
        """
        self.baseline_data = data
        self.baseline_stats = FeatureStatistics(
            feature_name=self.feature_name,
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            median=float(np.median(data)),
            q25=float(np.percentile(data, 25)),
            q75=float(np.percentile(data, 75)),
            sample_count=len(data)
        )

    def update(self, value: float) -> Optional[DriftAlert]:
        """Update monitor with new value.

        Args:
            value: New feature value

        Returns:
            DriftAlert if drift detected
        """
        self.current_window.append(value)

        # Need baseline and enough current data
        if self.baseline_data is None or len(self.current_window) < 100:
            return None

        # Perform drift tests
        current_data = np.array(self.current_window)

        # PSI test
        psi = DriftDetector.population_stability_index(
            self.baseline_data,
            current_data
        )

        # KS test
        ks_stat, ks_pvalue, ks_drift = DriftDetector.kolmogorov_smirnov_test(
            self.baseline_data,
            current_data,
            alpha=self.alert_threshold_ks
        )

        # JSD
        jsd = DriftDetector.jensen_shannon_divergence(
            self.baseline_data,
            current_data
        )

        # Store drift score
        self.drift_scores.append(psi)

        # Determine severity and alert
        drift_alert = None

        if psi >= self.alert_threshold_psi or ks_drift:
            # Determine severity
            if psi >= 0.5 or ks_pvalue < 0.001:
                severity = DriftSeverity.CRITICAL
                action = "Retrain model immediately"
            elif psi >= 0.35 or ks_pvalue < 0.01:
                severity = DriftSeverity.HIGH
                action = "Schedule model retraining"
            elif psi >= 0.25:
                severity = DriftSeverity.MEDIUM
                action = "Monitor closely, consider retraining"
            else:
                severity = DriftSeverity.LOW
                action = "Continue monitoring"

            drift_alert = DriftAlert(
                timestamp=time.time(),
                drift_type=DriftType.FEATURE_DRIFT,
                severity=severity,
                feature_name=self.feature_name,
                drift_score=psi,
                p_value=ks_pvalue,
                message=f"Feature '{self.feature_name}' drift detected: PSI={psi:.3f}, KS p-value={ks_pvalue:.4f}",
                recommended_action=action
            )

            self.alerts.append(drift_alert)

        return drift_alert

    def get_current_statistics(self) -> Optional[FeatureStatistics]:
        """Get statistics of current window."""
        if len(self.current_window) == 0:
            return None

        data = np.array(self.current_window)
        return FeatureStatistics(
            feature_name=self.feature_name,
            mean=float(np.mean(data)),
            std=float(np.std(data)),
            min=float(np.min(data)),
            max=float(np.max(data)),
            median=float(np.median(data)),
            q25=float(np.percentile(data, 25)),
            q75=float(np.percentile(data, 75)),
            sample_count=len(data)
        )


# ============================================================================
# Model Drift Monitor
# ============================================================================

class ModelDriftMonitor:
    """Monitor model predictions and performance for concept drift."""

    def __init__(
        self,
        model_name: str,
        num_features: int,
        feature_names: Optional[List[str]] = None
    ):
        """Initialize model drift monitor.

        Args:
            model_name: Name of model being monitored
            num_features: Number of input features
            feature_names: Optional feature names
        """
        self.model_name = model_name
        self.num_features = num_features
        self.feature_names = feature_names or [f"feature_{i}" for i in range(num_features)]

        # Feature monitors
        self.feature_monitors: Dict[str, FeatureDriftMonitor] = {}
        for fname in self.feature_names:
            self.feature_monitors[fname] = FeatureDriftMonitor(fname)

        # Prediction monitoring
        self.predictions_baseline: Optional[np.ndarray] = None
        self.predictions_current: deque = deque(maxlen=1000)

        # Performance monitoring (when ground truth available)
        self.errors: deque = deque(maxlen=1000)
        self.baseline_mae: Optional[float] = None

        # Alerts
        self.all_alerts: List[DriftAlert] = []

    def set_baseline(
        self,
        features: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        ground_truth: Optional[np.ndarray] = None
    ):
        """Set baseline from training/validation data.

        Args:
            features: Baseline feature matrix (n_samples, n_features)
            predictions: Baseline predictions
            ground_truth: Ground truth labels for MAE calculation
        """
        # Set feature baselines
        for i, fname in enumerate(self.feature_names):
            self.feature_monitors[fname].set_baseline(features[:, i])

        # Set prediction baseline
        if predictions is not None:
            self.predictions_baseline = predictions

        # Calculate baseline MAE
        if predictions is not None and ground_truth is not None:
            self.baseline_mae = np.mean(np.abs(predictions - ground_truth))

    def update(
        self,
        features: np.ndarray,
        prediction: float,
        ground_truth: Optional[float] = None
    ) -> List[DriftAlert]:
        """Update monitor with new prediction.

        Args:
            features: Feature vector
            prediction: Model prediction
            ground_truth: Optional ground truth for performance monitoring

        Returns:
            List of drift alerts
        """
        alerts = []

        # Update feature monitors
        for i, fname in enumerate(self.feature_names):
            if i < len(features):
                alert = self.feature_monitors[fname].update(features[i])
                if alert:
                    alerts.append(alert)

        # Update prediction distribution
        self.predictions_current.append(prediction)

        # Check prediction drift
        if self.predictions_baseline is not None and len(self.predictions_current) >= 100:
            pred_drift_alert = self._check_prediction_drift()
            if pred_drift_alert:
                alerts.append(pred_drift_alert)

        # Update performance monitoring
        if ground_truth is not None:
            error = abs(prediction - ground_truth)
            self.errors.append(error)

            # Check for concept drift (performance degradation)
            if self.baseline_mae is not None and len(self.errors) >= 100:
                concept_drift_alert = self._check_concept_drift()
                if concept_drift_alert:
                    alerts.append(concept_drift_alert)

        # Store alerts
        self.all_alerts.extend(alerts)

        return alerts

    def _check_prediction_drift(self) -> Optional[DriftAlert]:
        """Check if prediction distribution has drifted."""
        current_preds = np.array(self.predictions_current)

        psi = DriftDetector.population_stability_index(
            self.predictions_baseline,
            current_preds
        )

        if psi >= 0.25:
            severity = DriftSeverity.HIGH if psi >= 0.5 else DriftSeverity.MEDIUM

            return DriftAlert(
                timestamp=time.time(),
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=severity,
                drift_score=psi,
                message=f"Prediction distribution drift: PSI={psi:.3f}",
                recommended_action="Investigate model behavior, consider retraining"
            )

        return None

    def _check_concept_drift(self) -> Optional[DriftAlert]:
        """Check for concept drift (performance degradation)."""
        current_mae = np.mean(list(self.errors))

        # Check if MAE increased significantly
        mae_increase_pct = ((current_mae - self.baseline_mae) / self.baseline_mae) * 100

        if mae_increase_pct > 20:  # 20% increase in error
            if mae_increase_pct > 50:
                severity = DriftSeverity.CRITICAL
            elif mae_increase_pct > 35:
                severity = DriftSeverity.HIGH
            else:
                severity = DriftSeverity.MEDIUM

            return DriftAlert(
                timestamp=time.time(),
                drift_type=DriftType.CONCEPT_DRIFT,
                severity=severity,
                drift_score=mae_increase_pct,
                message=f"Model performance degraded: MAE increased {mae_increase_pct:.1f}%",
                recommended_action="Retrain model with recent data"
            )

        return None

    def get_summary(self) -> Dict:
        """Get drift monitoring summary."""
        summary = {
            "model_name": self.model_name,
            "total_alerts": len(self.all_alerts),
            "feature_drift_count": sum(1 for a in self.all_alerts if a.drift_type == DriftType.FEATURE_DRIFT),
            "prediction_drift_count": sum(1 for a in self.all_alerts if a.drift_type == DriftType.PREDICTION_DRIFT),
            "concept_drift_count": sum(1 for a in self.all_alerts if a.drift_type == DriftType.CONCEPT_DRIFT),
            "recent_alerts": [
                {
                    "timestamp": a.timestamp,
                    "type": a.drift_type.value,
                    "severity": a.severity.value,
                    "message": a.message
                }
                for a in self.all_alerts[-10:]
            ]
        }

        if self.baseline_mae and len(self.errors) > 0:
            current_mae = np.mean(list(self.errors))
            summary["baseline_mae"] = self.baseline_mae
            summary["current_mae"] = current_mae
            summary["mae_change_pct"] = ((current_mae - self.baseline_mae) / self.baseline_mae) * 100

        return summary


# Export
__all__ = [
    "FeatureDriftMonitor",
    "ModelDriftMonitor",
    "DriftDetector",
    "DriftAlert",
    "DriftType",
    "DriftSeverity",
    "FeatureStatistics",
]
