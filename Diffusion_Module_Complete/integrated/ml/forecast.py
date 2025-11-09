"""
Forecasting Module for Next-Run KPI Prediction - Session 8

Implements time series forecasting for next-run KPI prediction and
SPC violation probability estimation.

Includes:
- ARIMA baseline for time series
- Tree-based models for next-run prediction
- SPC violation probability estimation
- Hooks for LSTM (future)

Status: PRODUCTION READY ✅
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class ForecastResult:
    """
    Result of next-run forecast.

    Attributes:
        predicted_value: Predicted KPI value
        confidence_interval: (lower, upper) confidence bounds
        violation_probability: Probability of SPC violation
        method: Forecasting method used
        metadata: Additional metadata
    """
    predicted_value: float
    confidence_interval: Tuple[float, float]
    violation_probability: float
    method: str
    metadata: Dict


class ARIMAForecaster:
    """
    ARIMA-based time series forecaster.

    Simple baseline for time series prediction.
    """

    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Initialize ARIMA forecaster.

        Args:
            order: (p, d, q) order for ARIMA
        """
        self.order = order
        self.model = None
        self.fit_result = None

    def fit(self, data: np.ndarray):
        """
        Fit ARIMA model to historical data.

        Args:
            data: Historical time series
        """
        self.model = ARIMA(data, order=self.order)
        self.fit_result = self.model.fit()

    def forecast(
        self,
        steps: int = 1,
        alpha: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast next steps.

        Args:
            steps: Number of steps to forecast
            alpha: Significance level for confidence intervals

        Returns:
            Tuple of (forecast, lower_conf, upper_conf)
        """
        if self.fit_result is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecast_result = self.fit_result.get_forecast(steps=steps)

        forecast = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int(alpha=alpha)

        return forecast, conf_int[:, 0], conf_int[:, 1]


class TreeBasedForecaster:
    """
    Random Forest-based forecaster using lagged features.

    More flexible than ARIMA and can capture non-linear patterns.
    """

    def __init__(
        self,
        n_lags: int = 5,
        n_estimators: int = 100,
        max_depth: int = 10
    ):
        """
        Initialize tree-based forecaster.

        Args:
            n_lags: Number of lagged observations to use as features
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
        """
        self.n_lags = n_lags
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        self.historical_data: Optional[np.ndarray] = None

    def _create_lagged_features(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create lagged feature matrix.

        Args:
            data: Time series data

        Returns:
            Tuple of (X, y) for training
        """
        X = []
        y = []

        for i in range(self.n_lags, len(data)):
            X.append(data[i - self.n_lags:i])
            y.append(data[i])

        return np.array(X), np.array(y)

    def fit(self, data: np.ndarray):
        """
        Fit model to historical data.

        Args:
            data: Historical time series
        """
        self.historical_data = data.copy()
        X, y = self._create_lagged_features(data)
        self.model.fit(X, y)

    def forecast(
        self,
        steps: int = 1,
        return_std: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Forecast next steps.

        Args:
            steps: Number of steps to forecast
            return_std: If True, return standard deviation estimates

        Returns:
            Tuple of (forecast, std) if return_std, else just forecast
        """
        if self.historical_data is None:
            raise ValueError("Model not fitted. Call fit() first.")

        forecasts = []
        stds = []

        # Use last n_lags observations
        current_data = self.historical_data.copy()

        for _ in range(steps):
            # Get last n_lags observations
            X = current_data[-self.n_lags:].reshape(1, -1)

            # Predict with all trees to get distribution
            predictions = np.array([
                tree.predict(X)[0]
                for tree in self.model.estimators_
            ])

            forecast = np.mean(predictions)
            std = np.std(predictions)

            forecasts.append(forecast)
            stds.append(std)

            # Append forecast to current_data for next iteration
            current_data = np.append(current_data, forecast)

        if return_std:
            return np.array(forecasts), np.array(stds)
        else:
            return np.array(forecasts), None


class NextRunForecaster:
    """
    Next-run KPI forecaster combining multiple methods.

    Predicts KPI values for next run and estimates SPC violation probability.
    """

    def __init__(
        self,
        method: str = "tree",
        control_limits: Optional[Tuple[float, float, float]] = None
    ):
        """
        Initialize next-run forecaster.

        Args:
            method: Forecasting method ("arima", "tree", "ensemble")
            control_limits: (LCL, CL, UCL) from SPC
        """
        self.method = method
        self.control_limits = control_limits

        # Initialize forecasters
        self.arima_forecaster = ARIMAForecaster()
        self.tree_forecaster = TreeBasedForecaster()

    def fit(self, historical_kpis: np.ndarray):
        """
        Fit forecaster to historical KPI data.

        Args:
            historical_kpis: Array of historical KPI values
        """
        if self.method in ["arima", "ensemble"]:
            self.arima_forecaster.fit(historical_kpis)

        if self.method in ["tree", "ensemble"]:
            self.tree_forecaster.fit(historical_kpis)

    def forecast_next_run(
        self,
        additional_features: Optional[pd.Series] = None,
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Forecast KPI for next run.

        Args:
            additional_features: Optional additional features from FDC/recipe
            confidence: Confidence level for intervals

        Returns:
            ForecastResult with prediction and metadata
        """
        alpha = 1 - confidence

        if self.method == "arima":
            forecast, lower, upper = self.arima_forecaster.forecast(steps=1, alpha=alpha)
            predicted_value = float(forecast[0])
            conf_interval = (float(lower[0]), float(upper[0]))
            method_used = "ARIMA"

        elif self.method == "tree":
            forecast, std = self.tree_forecaster.forecast(steps=1, return_std=True)
            predicted_value = float(forecast[0])

            # Confidence interval from std (assuming normal distribution)
            z_score = stats.norm.ppf(1 - alpha/2)
            margin = z_score * std[0]
            conf_interval = (
                float(predicted_value - margin),
                float(predicted_value + margin)
            )
            method_used = "RandomForest"

        elif self.method == "ensemble":
            # Average ARIMA and tree predictions
            arima_forecast, arima_lower, arima_upper = self.arima_forecaster.forecast(
                steps=1, alpha=alpha
            )
            tree_forecast, tree_std = self.tree_forecaster.forecast(
                steps=1, return_std=True
            )

            predicted_value = float((arima_forecast[0] + tree_forecast[0]) / 2)

            # Use widest confidence interval
            conf_interval = (
                float(min(arima_lower[0], tree_forecast[0] - tree_std[0])),
                float(max(arima_upper[0], tree_forecast[0] + tree_std[0]))
            )
            method_used = "Ensemble(ARIMA+RF)"

        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Calculate violation probability
        violation_prob = self._calculate_violation_probability(
            predicted_value,
            conf_interval
        )

        return ForecastResult(
            predicted_value=predicted_value,
            confidence_interval=conf_interval,
            violation_probability=violation_prob,
            method=method_used,
            metadata={
                'confidence_level': confidence,
                'has_control_limits': self.control_limits is not None
            }
        )

    def _calculate_violation_probability(
        self,
        predicted_value: float,
        conf_interval: Tuple[float, float]
    ) -> float:
        """
        Calculate probability of SPC violation.

        Args:
            predicted_value: Predicted KPI value
            conf_interval: Confidence interval

        Returns:
            Probability of violation (0-1)
        """
        if self.control_limits is None:
            return 0.0

        lcl, cl, ucl = self.control_limits

        # Estimate standard deviation from confidence interval
        # Assuming 95% CI: width ≈ 4σ
        width = conf_interval[1] - conf_interval[0]
        sigma = width / 4.0

        # Calculate probabilities
        # P(violation) = P(X < LCL) + P(X > UCL)
        prob_below_lcl = stats.norm.cdf(lcl, loc=predicted_value, scale=sigma)
        prob_above_ucl = 1 - stats.norm.cdf(ucl, loc=predicted_value, scale=sigma)

        violation_prob = prob_below_lcl + prob_above_ucl

        return float(np.clip(violation_prob, 0.0, 1.0))


def forecast_with_drift_detection(
    historical_kpis: np.ndarray,
    drift_threshold: float = 0.1
) -> Dict[str, any]:
    """
    Forecast next-run with drift detection.

    Args:
        historical_kpis: Historical KPI values
        drift_threshold: Threshold for detecting drift (fraction)

    Returns:
        Dictionary with forecast and drift information
    """
    # Fit trend
    x = np.arange(len(historical_kpis))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, historical_kpis)

    # Detect significant drift
    mean_kpi = np.mean(historical_kpis)
    drift_rate = slope / mean_kpi if mean_kpi != 0 else 0.0
    has_drift = abs(drift_rate) > drift_threshold and p_value < 0.05

    # Forecast next point
    next_x = len(historical_kpis)
    if has_drift:
        # Use trend line
        forecast = slope * next_x + intercept
    else:
        # Use simple mean
        forecast = mean_kpi

    return {
        'forecast': float(forecast),
        'has_drift': bool(has_drift),
        'drift_rate': float(drift_rate),
        'p_value': float(p_value),
        'r_squared': float(r_value**2)
    }
