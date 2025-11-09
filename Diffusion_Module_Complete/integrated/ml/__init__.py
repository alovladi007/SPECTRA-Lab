"""ML module for Virtual Metrology, Forecasting, and Calibration - Sessions 8-9."""

# Session 8: Virtual Metrology & Forecasting
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

# Session 9: Calibration & Uncertainty Quantification
from .calibrate import (
    Prior,
    DiffusionPriors,
    OxidationPriors,
    CalibrationResult,
    LeastSquaresCalibrator,
    BayesianCalibrator,
    predict_with_uncertainty,
    calibrate_diffusion_params,
    calibrate_oxidation_params,
)

__all__ = [
    # Session 8: Features
    "FDCFeatureExtractor",
    "ThermalProfileFeatures",
    "ProcessStabilityFeatures",
    "SpatialFeatures",
    "HistoricalFeatures",
    "extract_features_from_fdc_data",
    # Session 8: VM Models
    "VirtualMetrologyModel",
    "ModelCard",
    "train_ensemble",
    "get_best_model",
    # Session 8: Forecasting
    "ARIMAForecaster",
    "TreeBasedForecaster",
    "NextRunForecaster",
    "ForecastResult",
    "forecast_with_drift_detection",
    # Session 9: Calibration
    "Prior",
    "DiffusionPriors",
    "OxidationPriors",
    "CalibrationResult",
    "LeastSquaresCalibrator",
    "BayesianCalibrator",
    "predict_with_uncertainty",
    "calibrate_diffusion_params",
    "calibrate_oxidation_params",
]
