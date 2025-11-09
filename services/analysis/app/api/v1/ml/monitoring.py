"""
ML Monitoring API Router
Provides endpoints for anomaly detection, drift monitoring, and time series forecasting
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import logging
import uuid
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import json

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["ML Monitoring"])

# In-memory storage (use Redis/database in production)
jobs = {}
anomalies_db = []
drift_reports_db = []
forecasts_db = []

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AnomalyDetectionConfig(BaseModel):
    """Configuration for anomaly detection"""
    data_source: str = Field("synthetic", description="Data source: synthetic or uploaded")
    contamination: float = Field(0.1, description="Expected proportion of anomalies (0.0-0.5)")
    n_samples: int = Field(1000, description="Number of samples to analyze")
    feature_names: Optional[List[str]] = Field(None, description="Feature names")

class DriftMonitoringConfig(BaseModel):
    """Configuration for drift monitoring"""
    reference_data_id: Optional[str] = Field(None, description="Reference dataset ID")
    current_data_id: Optional[str] = Field(None, description="Current dataset ID")
    drift_threshold: float = Field(0.05, description="P-value threshold for drift detection")
    methods: List[str] = Field(["psi", "ks"], description="Drift detection methods")

class ForecastConfig(BaseModel):
    """Configuration for time series forecasting"""
    metric_name: str = Field("yield", description="Metric to forecast")
    forecast_periods: int = Field(30, description="Number of periods to forecast")
    historical_periods: int = Field(90, description="Historical periods to use")
    confidence_level: float = Field(0.95, description="Confidence interval level")

class AnomalyResult(BaseModel):
    """Anomaly detection result"""
    id: int
    timestamp: str
    is_anomaly: bool
    anomaly_score: float
    anomaly_type: str
    features: Dict[str, float]
    feature_contributions: Dict[str, float]
    likely_causes: List[str]
    resolved: bool

class DriftResult(BaseModel):
    """Drift monitoring result"""
    id: int
    drift_type: str
    drift_detected: bool
    drift_score: float
    feature_drifts: Dict[str, Any]
    recommended_action: str
    created_at: str

class ForecastPoint(BaseModel):
    """Single forecast point"""
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float
    trend: float

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_synthetic_data(n_samples: int = 1000, n_features: int = 6):
    """Generate synthetic semiconductor manufacturing data"""
    np.random.seed(42)

    # Normal operating conditions
    normal_data = np.random.randn(int(n_samples * 0.9), n_features) * 10 + 100

    # Inject anomalies (10%)
    anomaly_data = np.random.randn(int(n_samples * 0.1), n_features) * 20 + 100
    anomaly_data += np.random.randn(int(n_samples * 0.1), n_features) * 30  # Extra variance

    # Combine
    data = np.vstack([normal_data, anomaly_data])
    np.random.shuffle(data)

    return data

def detect_anomalies(data: np.ndarray, contamination: float = 0.1):
    """Detect anomalies using Isolation Forest"""
    # Train Isolation Forest
    clf = IsolationForest(contamination=contamination, random_state=42)
    predictions = clf.fit_predict(data)
    scores = clf.score_samples(data)

    # Convert to anomaly scores (0 to 1, higher = more anomalous)
    anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())

    return predictions, anomaly_scores

def calculate_feature_contributions(data: np.ndarray, anomaly_indices: np.ndarray):
    """Calculate feature contributions to anomalies"""
    contributions = {}

    for i in range(data.shape[1]):
        feature_values = data[anomaly_indices, i]
        mean_val = np.mean(data[:, i])
        std_val = np.std(data[:, i])

        # Z-score based contribution
        z_scores = np.abs((feature_values - mean_val) / (std_val + 1e-10))
        contributions[f"feature_{i}"] = float(np.mean(z_scores))

    return contributions

def classify_anomaly_type(feature_values: np.ndarray, all_data: np.ndarray):
    """Classify anomaly type: point, contextual, or collective"""
    # Simple heuristic classification
    if np.std(feature_values) > np.std(all_data) * 2:
        return "collective"
    elif np.any(np.abs(feature_values - np.mean(all_data)) > 3 * np.std(all_data)):
        return "point"
    else:
        return "contextual"

def detect_drift_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10):
    """Calculate Population Stability Index (PSI) for drift detection"""
    psi_values = {}

    for i in range(reference.shape[1]):
        ref_feature = reference[:, i]
        cur_feature = current[:, i]

        # Create bins based on reference data
        bin_edges = np.histogram_bin_edges(ref_feature, bins=bins)

        # Calculate distributions
        ref_dist, _ = np.histogram(ref_feature, bins=bin_edges)
        cur_dist, _ = np.histogram(cur_feature, bins=bin_edges)

        # Normalize to probabilities
        ref_dist = (ref_dist + 1) / (np.sum(ref_dist) + bins)  # Laplace smoothing
        cur_dist = (cur_dist + 1) / (np.sum(cur_dist) + bins)

        # Calculate PSI
        psi = np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist))
        psi_values[f"feature_{i}"] = float(psi)

    return psi_values

def detect_drift_ks(reference: np.ndarray, current: np.ndarray):
    """Kolmogorov-Smirnov test for drift detection"""
    ks_results = {}

    for i in range(reference.shape[1]):
        statistic, p_value = stats.ks_2samp(reference[:, i], current[:, i])
        ks_results[f"feature_{i}"] = {
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < 0.05)  # Convert numpy.bool_ to Python bool
        }

    return ks_results

def generate_time_series_forecast(historical_data: np.ndarray, forecast_periods: int = 30):
    """Generate time series forecast using trend + seasonality decomposition"""
    # Simple trend + seasonal model
    n = len(historical_data)

    # Fit linear trend
    x = np.arange(n)
    trend_coeffs = np.polyfit(x, historical_data, 1)
    trend = np.poly1d(trend_coeffs)

    # Detrend
    detrended = historical_data - trend(x)

    # Extract seasonality (assume weekly pattern with 7-day cycle)
    season_length = min(7, n // 2)
    seasonal = np.zeros(season_length)
    for i in range(season_length):
        seasonal[i] = np.mean(detrended[i::season_length])

    # Forecast
    forecast_x = np.arange(n, n + forecast_periods)
    forecast_trend = trend(forecast_x)
    forecast_seasonal = np.tile(seasonal, forecast_periods // season_length + 1)[:forecast_periods]
    forecast_mean = forecast_trend + forecast_seasonal

    # Estimate prediction intervals based on residual variance
    residuals = historical_data - (trend(x) + np.tile(seasonal, n // season_length + 1)[:n])
    residual_std = np.std(residuals)

    # Expanding confidence intervals
    expanding_std = residual_std * (1 + 0.1 * np.arange(forecast_periods))
    forecast_lower = forecast_mean - 1.96 * expanding_std
    forecast_upper = forecast_mean + 1.96 * expanding_std

    return forecast_mean, forecast_lower, forecast_upper, forecast_trend

# ============================================================================
# API ENDPOINTS - ANOMALY DETECTION
# ============================================================================

@router.post("/anomaly-detection/detect")
async def detect_anomalies_endpoint(config: AnomalyDetectionConfig):
    """
    Detect anomalies in semiconductor manufacturing data

    Returns detected anomalies with scores and feature contributions
    """
    try:
        # Generate or load data
        feature_names = config.feature_names or [
            "temperature", "pressure", "flow_rate", "power",
            "gas_concentration", "chamber_pressure"
        ]
        n_features = len(feature_names)

        data = generate_synthetic_data(config.n_samples, n_features)

        # Detect anomalies
        predictions, anomaly_scores = detect_anomalies(data, config.contamination)

        # Build results
        results = []
        anomaly_indices = np.where(predictions == -1)[0]

        for idx in anomaly_indices:
            # Feature values and contributions
            feature_values = {name: float(data[idx, i]) for i, name in enumerate(feature_names)}
            contributions = calculate_feature_contributions(data, np.array([idx]))

            # Anomaly type
            anomaly_type = classify_anomaly_type(data[idx], data)

            # Likely causes based on top contributing features
            top_features = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:2]
            causes = [f"{feat.replace('_', ' ').title()} deviation detected" for feat, _ in top_features]

            anomaly = AnomalyResult(
                id=int(idx),
                timestamp=(datetime.now() - timedelta(hours=len(anomaly_indices) - len(results))).isoformat(),
                is_anomaly=True,
                anomaly_score=float(anomaly_scores[idx]),
                anomaly_type=anomaly_type,
                features=feature_values,
                feature_contributions={k.replace("feature_", feature_names[int(k.split("_")[1])]): v
                                     for k, v in contributions.items()},
                likely_causes=causes,
                resolved=False
            )

            results.append(anomaly.dict())
            anomalies_db.append(anomaly.dict())

        return {
            "total_samples": config.n_samples,
            "anomalies_detected": len(results),
            "anomaly_rate": len(results) / config.n_samples,
            "anomalies": results[:25]  # Return first 25
        }

    except Exception as e:
        logger.error(f"Anomaly detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomaly-detection/list")
async def list_anomalies(limit: int = 50, resolved: Optional[bool] = None):
    """List detected anomalies"""
    filtered = anomalies_db

    if resolved is not None:
        filtered = [a for a in filtered if a["resolved"] == resolved]

    return {
        "total": len(filtered),
        "anomalies": filtered[:limit]
    }

@router.post("/anomaly-detection/{anomaly_id}/resolve")
async def resolve_anomaly(anomaly_id: int):
    """Mark an anomaly as resolved"""
    for anomaly in anomalies_db:
        if anomaly["id"] == anomaly_id:
            anomaly["resolved"] = True
            return {"message": "Anomaly resolved successfully"}

    raise HTTPException(status_code=404, detail="Anomaly not found")

# ============================================================================
# API ENDPOINTS - DRIFT MONITORING
# ============================================================================

@router.post("/drift/detect")
async def detect_drift_endpoint(config: DriftMonitoringConfig):
    """
    Detect model drift between reference and current data

    Returns drift metrics and recommended actions
    """
    try:
        # Generate reference and current datasets
        reference_data = generate_synthetic_data(500, 6)

        # Add drift to current data
        current_data = generate_synthetic_data(500, 6)
        current_data += np.random.randn(500, 6) * 5  # Add drift

        # Detect drift using requested methods
        drift_results = {}
        drift_detected = False

        if "psi" in config.methods:
            psi_values = detect_drift_psi(reference_data, current_data)
            drift_results["psi"] = psi_values
            # PSI > 0.2 indicates significant drift
            if any(v > 0.2 for v in psi_values.values()):
                drift_detected = True

        if "ks" in config.methods:
            ks_results = detect_drift_ks(reference_data, current_data)
            drift_results["ks"] = ks_results
            if any(v["drift_detected"] for v in ks_results.values()):
                drift_detected = True

        # Calculate overall drift score
        if "psi" in drift_results:
            drift_score = np.mean(list(drift_results["psi"].values()))
        elif "ks" in drift_results:
            drift_score = np.mean([v["statistic"] for v in drift_results["ks"].values()])
        else:
            drift_score = 0.0

        # Determine recommended action
        if drift_score > 0.3:
            action = "Retrain model immediately - significant drift detected"
        elif drift_score > 0.2:
            action = "Monitor closely and plan retraining"
        else:
            action = "No action required - drift within acceptable limits"

        # Create drift report
        report = DriftResult(
            id=len(drift_reports_db) + 1,
            drift_type="data_drift",
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            feature_drifts=drift_results,
            recommended_action=action,
            created_at=datetime.now().isoformat()
        )

        drift_reports_db.append(report.dict())

        return report.dict()

    except Exception as e:
        logger.error(f"Drift detection failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/drift/reports")
async def list_drift_reports(limit: int = 50):
    """List drift monitoring reports"""
    return {
        "total": len(drift_reports_db),
        "reports": drift_reports_db[:limit]
    }

# ============================================================================
# API ENDPOINTS - TIME SERIES FORECASTING
# ============================================================================

@router.post("/forecast/generate")
async def generate_forecast_endpoint(config: ForecastConfig):
    """
    Generate time series forecast for semiconductor metrics

    Returns forecast with confidence intervals
    """
    try:
        # Generate historical data
        np.random.seed(42)
        trend = np.linspace(100, 110, config.historical_periods)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(config.historical_periods) / 7)
        noise = np.random.randn(config.historical_periods) * 2
        historical_values = trend + seasonal + noise

        # Generate forecast
        forecast_mean, forecast_lower, forecast_upper, forecast_trend = generate_time_series_forecast(
            historical_values, config.forecast_periods
        )

        # Create historical data points
        historical_data = []
        for i in range(config.historical_periods):
            historical_data.append({
                "timestamp": (datetime.now() - timedelta(days=config.historical_periods - i)).isoformat(),
                "value": float(historical_values[i])
            })

        # Create forecast points
        forecast_points = []
        for i in range(config.forecast_periods):
            point = ForecastPoint(
                ds=(datetime.now() + timedelta(days=i + 1)).isoformat(),
                yhat=float(forecast_mean[i]),
                yhat_lower=float(forecast_lower[i]),
                yhat_upper=float(forecast_upper[i]),
                trend=float(forecast_trend[i])
            )
            forecast_points.append(point.dict())

        result = {
            "metric_name": config.metric_name,
            "forecast_generated_at": datetime.now().isoformat(),
            "historical_data": historical_data,
            "forecast": forecast_points,
            "forecast_summary": {
                "periods": config.forecast_periods,
                "mean_forecast": float(np.mean(forecast_mean)),
                "trend_direction": "increasing" if forecast_trend[-1] > forecast_trend[0] else "decreasing",
                "confidence_level": config.confidence_level
            }
        }

        forecasts_db.append(result)

        return result

    except Exception as e:
        logger.error(f"Forecast generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast/list")
async def list_forecasts(limit: int = 10):
    """List generated forecasts"""
    return {
        "total": len(forecasts_db),
        "forecasts": forecasts_db[:limit]
    }

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check for monitoring service"""
    return {
        "status": "healthy",
        "service": "ml_monitoring",
        "anomalies_tracked": len(anomalies_db),
        "drift_reports": len(drift_reports_db),
        "forecasts": len(forecasts_db)
    }

@router.delete("/clear")
async def clear_all_data():
    """Clear all monitoring data (development only)"""
    global anomalies_db, drift_reports_db, forecasts_db
    anomalies_db = []
    drift_reports_db = []
    forecasts_db = []
    return {"message": "All monitoring data cleared"}
