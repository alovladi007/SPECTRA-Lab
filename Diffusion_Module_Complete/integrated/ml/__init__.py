"""ML module for Virtual Metrology and Forecasting - Session 8."""

from .features import (
    FDCFeatureExtractor,
    ThermalProfileFeatures,
    ProcessStabilityFeatures,
    SpatialFeatures,
    HistoricalFeatures,
    extract_features_from_fdc_data,
)

from .vm import (
    VirtualMetrologyModel,
    ModelCard,
    train_ensemble,
    get_best_model,
)

from .forecast import (
    ARIMAForecaster,
    TreeBasedForecaster,
    NextRunForecaster,
    ForecastResult,
    forecast_with_drift_detection,
)

__all__ = [
    # Features
    "FDCFeatureExtractor",
    "ThermalProfileFeatures",
    "ProcessStabilityFeatures",
    "SpatialFeatures",
    "HistoricalFeatures",
    "extract_features_from_fdc_data",
    # VM Models
    "VirtualMetrologyModel",
    "ModelCard",
    "train_ensemble",
    "get_best_model",
    # Forecasting
    "ARIMAForecaster",
    "TreeBasedForecaster",
    "NextRunForecaster",
    "ForecastResult",
    "forecast_with_drift_detection",
]
