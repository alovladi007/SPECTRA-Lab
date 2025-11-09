"""API endpoints for ML services - Session 8."""

from .ml_endpoints import (
    vm_predict,
    forecast_next,
    VMPredictRequest,
    VMPredictResponse,
    ForecastRequest,
    ForecastResponse,
)

__all__ = [
    "vm_predict",
    "forecast_next",
    "VMPredictRequest",
    "VMPredictResponse",
    "ForecastRequest",
    "ForecastResponse",
]
