"""
ML API Endpoints for Virtual Metrology and Forecasting - Session 8

FastAPI endpoints for:
- /ml/vm/predict: Predict post-process KPIs from FDC data
- /ml/forecast/next: Forecast next-run KPIs and violation probability
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


class FDCDataRequest(BaseModel):
    """FDC time series data for VM prediction."""
    time: List[float] = Field(..., description="Time array (minutes)")
    temperature: List[float] = Field(..., description="Temperature (°C)")
    setpoint: Optional[List[float]] = Field(None, description="Setpoint temp")
    pressure: Optional[List[float]] = Field(None, description="Pressure (Torr)")
    gas_flow_o2: Optional[List[float]] = Field(None, description="O2 flow (sccm)")
    gas_flow_n2: Optional[List[float]] = Field(None, description="N2 flow (sccm)")
    alarms: Optional[List[bool]] = Field(None, description="Alarm flags")


class RecipeParameters(BaseModel):
    """Recipe and process parameters."""
    boat_load_count: int = Field(25, description="Wafers in boat")
    slot_index: int = Field(0, description="Position in boat")


class HistoricalData(BaseModel):
    """Historical wafer/lot data."""
    prior_thermal_budgets: Optional[List[float]] = None
    steps_completed: int = 0
    time_since_last_process: float = 0.0
    lot_age: float = 0.0
    wafer_usage_count: int = 1


class VMPredictRequest(BaseModel):
    """Request for VM prediction."""
    fdc_data: FDCDataRequest
    recipe: Optional[RecipeParameters] = None
    historical: Optional[HistoricalData] = None
    model_type: str = Field("xgboost", description="ridge, lasso, or xgboost")
    target: str = Field("junction_depth", description="Target KPI")


class VMPredictResponse(BaseModel):
    """Response from VM prediction."""
    predicted_value: float
    target: str
    model_type: str
    model_version: str
    features_used: List[str]
    feature_values: Dict[str, float]
    prediction_timestamp: datetime
    confidence_metrics: Dict[str, float]


class ForecastRequest(BaseModel):
    """Request for next-run forecast."""
    historical_kpis: List[float] = Field(..., min_items=10)
    target: str = Field(..., description="KPI name")
    method: str = Field("ensemble", description="arima, tree, or ensemble")
    control_limits: Optional[Dict[str, float]] = Field(
        None, description="{'lcl': x, 'cl': y, 'ucl': z}"
    )
    confidence_level: float = Field(0.95, ge=0.0, le=1.0)


class ForecastResponse(BaseModel):
    """Response from forecast."""
    predicted_value: float
    confidence_interval: Dict[str, float]  # lower, upper
    violation_probability: float
    method: str
    target: str
    forecast_timestamp: datetime
    metadata: Dict[str, Any]


def vm_predict(request: VMPredictRequest) -> VMPredictResponse:
    """
    Predict post-process KPI from FDC data and recipe.

    This would be wrapped in FastAPI:
    @router.post("/ml/vm/predict")
    async def predict_endpoint(request: VMPredictRequest) -> VMPredictResponse:
        return vm_predict(request)
    """
    from session8.ml.features import extract_features_from_fdc_data
    from session8.ml.vm import VirtualMetrologyModel

    # Convert FDC data to dict
    fdc_dict = {
        'time': np.array(request.fdc_data.time),
        'temperature': np.array(request.fdc_data.temperature),
        'setpoint': np.array(request.fdc_data.setpoint) if request.fdc_data.setpoint else None,
        'pressure': np.array(request.fdc_data.pressure) if request.fdc_data.pressure else None,
        'gas_flow_o2': np.array(request.fdc_data.gas_flow_o2) if request.fdc_data.gas_flow_o2 else None,
        'gas_flow_n2': np.array(request.fdc_data.gas_flow_n2) if request.fdc_data.gas_flow_n2 else None,
        'alarms': np.array(request.fdc_data.alarms) if request.fdc_data.alarms else None,
    }

    # Recipe params
    recipe_params = request.recipe.dict() if request.recipe else None

    # Historical data
    hist_data = request.historical.dict() if request.historical else None

    # Extract features
    features = extract_features_from_fdc_data(fdc_dict, recipe_params, hist_data)

    # Load model (in production, would cache models)
    # For now, simulate with mock prediction
    # In real implementation:
    # artifacts_dir = Path("artifacts/vm")
    # model = VirtualMetrologyModel.load(
    #     artifacts_dir,
    #     f"vm_{request.target}_{request.model_type}",
    #     version="1.0.0"
    # )
    # prediction = model.predict(features.to_frame().T)[0]

    # Mock prediction for demonstration
    if request.target == "junction_depth":
        # Use peak temperature and soak integral as simple predictors
        prediction = features['peak_temperature'] * 0.5 + features['soak_integral'] * 0.001
    elif request.target == "oxide_thickness":
        prediction = features['peak_temperature'] * 0.3 + features['time_at_peak'] * 2.0
    else:
        prediction = features['peak_temperature'] * 0.4

    return VMPredictResponse(
        predicted_value=float(prediction),
        target=request.target,
        model_type=request.model_type,
        model_version="1.0.0",
        features_used=list(features.index),
        feature_values=features.to_dict(),
        prediction_timestamp=datetime.now(),
        confidence_metrics={
            'model_r2': 0.92,  # Would come from model card
            'model_rmse': 5.3,
            'feature_importance_top3': {
                features.index[0]: 0.35,
                features.index[1]: 0.22,
                features.index[2]: 0.15
            }
        }
    )


def forecast_next(request: ForecastRequest) -> ForecastResponse:
    """
    Forecast next-run KPI and violation probability.

    This would be wrapped in FastAPI:
    @router.post("/ml/forecast/next")
    async def forecast_endpoint(request: ForecastRequest) -> ForecastResponse:
        return forecast_next(request)
    """
    from session8.ml.forecast import NextRunForecaster

    # Convert control limits
    control_limits = None
    if request.control_limits:
        control_limits = (
            request.control_limits['lcl'],
            request.control_limits['cl'],
            request.control_limits['ucl']
        )

    # Create forecaster
    forecaster = NextRunForecaster(
        method=request.method,
        control_limits=control_limits
    )

    # Fit to historical data
    historical_array = np.array(request.historical_kpis)
    forecaster.fit(historical_array)

    # Forecast next run
    result = forecaster.forecast_next_run(confidence=request.confidence_level)

    return ForecastResponse(
        predicted_value=result.predicted_value,
        confidence_interval={
            'lower': result.confidence_interval[0],
            'upper': result.confidence_interval[1]
        },
        violation_probability=result.violation_probability,
        method=result.method,
        target=request.target,
        forecast_timestamp=datetime.now(),
        metadata=result.metadata
    )


# Example usage for testing
if __name__ == "__main__":
    # Test VM prediction
    fdc_data = FDCDataRequest(
        time=list(range(60)),
        temperature=[900 + i*2 for i in range(60)],
        pressure=[760.0] * 60
    )

    vm_request = VMPredictRequest(
        fdc_data=fdc_data,
        target="junction_depth",
        model_type="xgboost"
    )

    vm_response = vm_predict(vm_request)
    print(f"VM Prediction: {vm_response.predicted_value:.2f} nm")

    # Test forecasting
    forecast_request = ForecastRequest(
        historical_kpis=[100 + np.random.randn()*5 for _ in range(30)],
        target="junction_depth",
        method="ensemble",
        control_limits={'lcl': 90, 'cl': 100, 'ucl': 110}
    )

    forecast_response = forecast_next(forecast_request)
    print(f"Forecast: {forecast_response.predicted_value:.2f}")
    print(f"Violation Prob: {forecast_response.violation_probability:.2%}")
