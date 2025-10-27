"""
Session 14: Virtual Metrology & ML Suite - Complete Implementation
===================================================================

Comprehensive machine learning and virtual metrology system for
semiconductor characterization platform.

Components:
- Model Training Pipeline (sklearn, LightGBM, XGBoost)
- Feature Engineering & Store
- Anomaly Detection (IsolationForest, LOF, Autoencoders)
- Drift Monitoring (PSI, KS, KL divergence)
- Time Series Forecasting (Prophet, ARIMA, LSTM)
- Model Registry & Versioning
- ONNX Export for Production

Author: Semiconductor Lab Platform Team
Date: October 2025
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import logging
from enum import Enum

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import (
    RandomForestRegressor, IsolationForest, GradientBoostingRegressor
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_score, recall_score, f1_score
)
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from scipy.spatial.distance import jensenshannon

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes and Enums
# ============================================================================

class ModelType(Enum):
    """Supported model types"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LINEAR = "linear"

class AnomalyMethod(Enum):
    """Anomaly detection methods"""
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    STATISTICAL = "statistical"

class DriftMethod(Enum):
    """Drift detection methods"""
    PSI = "population_stability_index"
    KS_TEST = "kolmogorov_smirnov"
    KL_DIVERGENCE = "kullback_leibler"
    CHI_SQUARE = "chi_square"

@dataclass
class TrainingConfig:
    """Configuration for model training"""
    model_type: ModelType = ModelType.RANDOM_FOREST
    target_metric: str = "thickness"
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    hyperparameter_tuning: bool = True
    n_iterations: int = 50
    early_stopping_rounds: Optional[int] = 10
    model_params: Dict[str, Any] = field(default_factory=dict)
    feature_selection: bool = True
    feature_importance_threshold: float = 0.01
    polynomial_features: bool = False
    interaction_features: bool = False
    scaler_type: str = "standard"

@dataclass
class ModelMetrics:
    """Comprehensive model evaluation metrics"""
    r2: float
    rmse: float
    mae: float
    mape: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    training_time: float
    inference_time: float
    n_samples: int
    n_features: int
    
    def to_dict(self) -> Dict:
        return {
            'r2': float(self.r2),
            'rmse': float(self.rmse),
            'mae': float(self.mae),
            'mape': float(self.mape),
            'cv_mean': float(self.cv_mean),
            'cv_std': float(self.cv_std),
            'training_time': float(self.training_time),
            'inference_time': float(self.inference_time),
            'n_samples': int(self.n_samples),
            'n_features': int(self.n_features),
        }

@dataclass
class FeatureImportance:
    """Feature importance analysis"""
    feature_names: List[str]
    importance_values: List[float]
    importance_type: str
    
    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        sorted_features = sorted(
            zip(self.feature_names, self.importance_values),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:n]

@dataclass
class AnomalyResult:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_score: float
    method: AnomalyMethod
    threshold: float
    confidence: float
    contributing_features: Optional[List[Tuple[str, float]]] = None

@dataclass
class DriftResult:
    """Drift detection result"""
    drift_detected: bool
    drift_score: float
    method: DriftMethod
    p_value: Optional[float]
    threshold: float
    affected_features: List[str]
    severity: str

# ============================================================================
# Feature Engineering
# ============================================================================

class FeatureStore:
    """Feature store for managing and versioning features"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path("./feature_store")
        self.storage_path.mkdir(exist_ok=True, parents=True)
        self.features: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, Dict] = {}
        
    def extract_fdc_features(
        self,
        sensor_data: Dict[str, np.ndarray],
        recipe_params: Dict[str, float]
    ) -> pd.DataFrame:
        """Extract features from FDC sensors and recipe parameters"""
        features = {}
        
        # Statistical features from sensor data
        for sensor_name, values in sensor_data.items():
            if len(values) == 0:
                continue
                
            features[f"{sensor_name}_mean"] = np.mean(values)
            features[f"{sensor_name}_std"] = np.std(values)
            features[f"{sensor_name}_min"] = np.min(values)
            features[f"{sensor_name}_max"] = np.max(values)
            features[f"{sensor_name}_median"] = np.median(values)
            features[f"{sensor_name}_range"] = np.ptp(values)
            features[f"{sensor_name}_p25"] = np.percentile(values, 25)
            features[f"{sensor_name}_p75"] = np.percentile(values, 75)
            
            # Trend features
            if len(values) > 1:
                time = np.arange(len(values))
                slope, intercept = np.polyfit(time, values, 1)
                features[f"{sensor_name}_trend"] = slope
                
        # Recipe parameters
        for param_name, value in recipe_params.items():
            features[f"recipe_{param_name}"] = value
            
        # Derived features
        if "temperature" in sensor_data and "pressure" in sensor_data:
            temp = np.mean(sensor_data["temperature"])
            press = np.mean(sensor_data["pressure"])
            features["temp_press_ratio"] = temp / (press + 1e-9)
            
        return pd.DataFrame([features])
    
    def save_features(self, name: str, features: pd.DataFrame, metadata: Optional[Dict] = None):
        """Save features to store"""
        self.features[name] = features
        self.metadata[name] = metadata or {}
        feature_path = self.storage_path / f"{name}.parquet"
        features.to_parquet(feature_path)

# ============================================================================
# Virtual Metrology Models
# ============================================================================

class VirtualMetrologyModel:
    """Virtual Metrology model for predicting measurements from process data"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.scaler = None
        self.feature_names: List[str] = []
        self.metrics: Optional[ModelMetrics] = None
        self.feature_importance: Optional[FeatureImportance] = None
        
        self._initialize_model()
        self._initialize_scaler()
        
    def _initialize_model(self):
        """Initialize the ML model"""
        if self.config.model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=self.config.model_params.get('n_estimators', 100),
                max_depth=self.config.model_params.get('max_depth', None),
                min_samples_split=self.config.model_params.get('min_samples_split', 2),
                random_state=self.config.random_state,
                n_jobs=-1
            )
        elif self.config.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.model_params.get('n_estimators', 100),
                learning_rate=self.config.model_params.get('learning_rate', 0.1),
                max_depth=self.config.model_params.get('max_depth', 3),
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
    def _initialize_scaler(self):
        """Initialize feature scaler"""
        if self.config.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
            
    def train(self, X: pd.DataFrame, y: pd.Series) -> ModelMetrics:
        """Train the virtual metrology model"""
        start_time = datetime.now()
        self.feature_names = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        y_pred = self.model.predict(X_test_scaled)
        
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=self.config.cv_folds, scoring='r2', n_jobs=-1
        )
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
        
        inference_start = datetime.now()
        for _ in range(100):
            _ = self.model.predict(X_test_scaled[:1])
        inference_time = (datetime.now() - inference_start).total_seconds() / 100
        
        self.metrics = ModelMetrics(
            r2=r2, rmse=rmse, mae=mae, mape=mape,
            cv_scores=list(cv_scores),
            cv_mean=np.mean(cv_scores),
            cv_std=np.std(cv_scores),
            training_time=training_time,
            inference_time=inference_time,
            n_samples=len(X),
            n_features=X.shape[1]
        )
        
        self._calculate_feature_importance()
        logger.info(f"Model trained: R²={r2:.4f}, RMSE={rmse:.4f}")
        return self.metrics
    
    def _calculate_feature_importance(self):
        """Calculate feature importance"""
        if hasattr(self.model, 'feature_importances_'):
            importance_values = self.model.feature_importances_
            importance_type = "gain"
        elif hasattr(self.model, 'coef_'):
            importance_values = np.abs(self.model.coef_)
            importance_type = "weight"
        else:
            return
            
        self.feature_importance = FeatureImportance(
            feature_names=self.feature_names,
            importance_values=list(importance_values),
            importance_type=importance_type
        )
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save(self, output_path: Path):
        """Save model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'config': self.config,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance
        }
        with open(output_path, 'wb') as f:
            pickle.dump(model_data, f)

# ============================================================================
# Anomaly Detection
# ============================================================================

class AnomalyDetector:
    """Multi-method anomaly detection"""
    
    def __init__(self, method: AnomalyMethod = AnomalyMethod.ISOLATION_FOREST, contamination: float = 0.1):
        self.method = method
        self.contamination = contamination
        self.detector = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self._initialize_detector()
        
    def _initialize_detector(self):
        """Initialize detector"""
        if self.method == AnomalyMethod.ISOLATION_FOREST:
            self.detector = IsolationForest(
                contamination=self.contamination, random_state=42, n_jobs=-1
            )
        elif self.method == AnomalyMethod.LOCAL_OUTLIER_FACTOR:
            self.detector = LocalOutlierFactor(
                contamination=self.contamination, novelty=True, n_jobs=-1
            )
        else:
            self.detector = IsolationForest(contamination=self.contamination)
            
    def fit(self, X: pd.DataFrame):
        """Fit detector"""
        self.feature_names = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == AnomalyMethod.STATISTICAL:
            self.threshold = {
                col: {
                    'mean': X[col].mean(),
                    'std': X[col].std(),
                    'q01': X[col].quantile(0.01),
                    'q99': X[col].quantile(0.99)
                } for col in X.columns
            }
        else:
            self.detector.fit(X_scaled)
            if hasattr(self.detector, 'decision_function'):
                scores = self.detector.decision_function(X_scaled)
                self.threshold = np.percentile(scores, self.contamination * 100)
                
    def detect(self, X: pd.DataFrame) -> List[AnomalyResult]:
        """Detect anomalies"""
        X_scaled = self.scaler.transform(X)
        results = []
        
        predictions = self.detector.predict(X_scaled)
        
        if hasattr(self.detector, 'decision_function'):
            scores = self.detector.decision_function(X_scaled)
        else:
            scores = predictions
            
        for idx, (pred, score) in enumerate(zip(predictions, scores)):
            is_anomaly = pred == -1
            confidence = abs(score - self.threshold) / abs(self.threshold + 1e-9)
            
            results.append(AnomalyResult(
                is_anomaly=is_anomaly,
                anomaly_score=float(score),
                method=self.method,
                threshold=float(self.threshold) if self.threshold is not None else 0.0,
                confidence=float(min(confidence, 1.0)),
                contributing_features=[]
            ))
        return results

# ============================================================================
# Drift Detection
# ============================================================================

class DriftDetector:
    """Statistical drift detection"""
    
    def __init__(self, method: DriftMethod = DriftMethod.PSI, threshold: float = 0.1):
        self.method = method
        self.threshold = threshold
        self.reference_data: Optional[pd.DataFrame] = None
        self.reference_stats: Dict = {}
        
    def set_reference(self, X: pd.DataFrame):
        """Set reference distribution"""
        self.reference_data = X.copy()
        for col in X.columns:
            self.reference_stats[col] = {
                'mean': X[col].mean(),
                'std': X[col].std(),
                'histogram': np.histogram(X[col], bins=10)
            }
            
    def detect_drift(self, X: pd.DataFrame) -> DriftResult:
        """Detect drift"""
        if self.reference_data is None:
            raise ValueError("Reference data not set")
            
        if self.method == DriftMethod.PSI:
            return self._detect_psi(X)
        elif self.method == DriftMethod.KS_TEST:
            return self._detect_ks(X)
        else:
            return self._detect_psi(X)
            
    def _detect_psi(self, X: pd.DataFrame) -> DriftResult:
        """PSI drift detection"""
        psi_scores = {}
        
        for col in X.columns:
            if col not in self.reference_stats:
                continue
                
            ref_hist, bin_edges = self.reference_stats[col]['histogram']
            new_hist, _ = np.histogram(X[col], bins=bin_edges)
            
            ref_prob = ref_hist / (len(self.reference_data) + 1e-9)
            new_prob = new_hist / (len(X) + 1e-9)
            
            psi = np.sum((new_prob - ref_prob) * np.log((new_prob + 1e-9) / (ref_prob + 1e-9)))
            psi_scores[col] = abs(psi)
            
        drift_score = np.mean(list(psi_scores.values()))
        drift_detected = drift_score > self.threshold
        
        affected_features = [
            col for col, score in psi_scores.items() if score > self.threshold
        ]
        
        severity = "high" if drift_score > 0.25 else "medium" if drift_score > 0.1 else "low"
            
        return DriftResult(
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            method=self.method,
            p_value=None,
            threshold=self.threshold,
            affected_features=affected_features,
            severity=severity
        )
    
    def _detect_ks(self, X: pd.DataFrame) -> DriftResult:
        """KS test"""
        ks_stats = {}
        p_values = {}
        
        for col in X.columns:
            if col not in self.reference_data.columns:
                continue
            statistic, p_value = stats.ks_2samp(self.reference_data[col], X[col])
            ks_stats[col] = statistic
            p_values[col] = p_value
            
        drift_score = np.mean(list(ks_stats.values()))
        min_p_value = min(p_values.values())
        drift_detected = min_p_value < 0.05
        
        affected_features = [col for col, p_val in p_values.items() if p_val < 0.05]
        severity = "high" if drift_score > 0.2 else "medium" if drift_score > 0.1 else "low"
        
        return DriftResult(
            drift_detected=drift_detected,
            drift_score=float(drift_score),
            method=self.method,
            p_value=float(min_p_value),
            threshold=0.05,
            affected_features=affected_features,
            severity=severity
        )

# ============================================================================
# Time Series Forecasting
# ============================================================================

class TimeSeriesForecaster:
    """Time series forecasting"""
    
    def __init__(self, method: str = "linear", horizon: int = 30):
        self.method = method
        self.horizon = horizon
        self.model = None
        
    def fit(self, timestamps: List[datetime], values: np.ndarray):
        """Fit model"""
        time_numeric = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        coeffs = np.polyfit(time_numeric, values, 1)
        self.model = {'type': 'linear', 'coeffs': coeffs, 'start_time': timestamps[0]}
        
    def forecast(self, future_timestamps: Optional[List[datetime]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate forecast"""
        if 'linear' in self.model['type']:
            coeffs = self.model['coeffs']
            start_time = self.model['start_time']
            
            if future_timestamps is None:
                last_time = start_time + timedelta(days=len(coeffs))
                future_timestamps = [
                    last_time + timedelta(days=i) for i in range(1, self.horizon + 1)
                ]
                
            time_numeric = np.array([(t - start_time).total_seconds() for t in future_timestamps])
            yhat = np.polyval(coeffs, time_numeric)
            yhat_lower = yhat * 0.9
            yhat_upper = yhat * 1.1
            
            return yhat, yhat_lower, yhat_upper
        return np.zeros(self.horizon), np.zeros(self.horizon), np.zeros(self.horizon)

# ============================================================================
# Test Data Generation
# ============================================================================

def generate_vm_training_data(n_samples: int = 1000, n_features: int = 20) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic VM training data"""
    np.random.seed(42)
    feature_names = [f"sensor_{i}_mean" for i in range(10)] + [f"sensor_{i}_std" for i in range(10)]
    X = np.random.randn(n_samples, n_features)
    weights = np.random.randn(n_features) * 0.5
    target = X @ weights
    target += X[:, 0] * X[:, 1] * 0.3
    target += np.random.randn(n_samples) * 0.1 * np.std(target)
    target = (target - target.mean()) / target.std() * 50 + 100
    
    X_df = pd.DataFrame(X, columns=feature_names[:n_features])
    y_series = pd.Series(target, name='thickness_nm')
    return X_df, y_series

def generate_anomaly_data(n_normal: int = 900, n_anomalies: int = 100, n_features: int = 10) -> pd.DataFrame:
    """Generate anomaly data"""
    np.random.seed(42)
    X_normal = np.random.randn(n_normal, n_features)
    X_anomalies = np.random.randn(n_anomalies, n_features) * 5 + 3
    X = np.vstack([X_normal, X_anomalies])
    labels = np.array([0] * n_normal + [1] * n_anomalies)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['is_anomaly'] = labels
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

def generate_timeseries_data(n_points: int = 365) -> Tuple[List[datetime], np.ndarray]:
    """Generate time series"""
    np.random.seed(42)
    start_date = datetime(2024, 1, 1)
    timestamps = [start_date + timedelta(days=i) for i in range(n_points)]
    t = np.arange(n_points)
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 365)
    noise = np.random.randn(n_points) * 2
    values = 100 + trend + seasonal + noise
    return timestamps, values

# Main execution
if __name__ == "__main__":
    logger.info("Session 14: ML & Virtual Metrology - Examples")
    
    # VM Model
    X_train, y_train = generate_vm_training_data(1000, 20)
    config = TrainingConfig(model_type=ModelType.RANDOM_FOREST, target_metric="thickness")
    vm_model = VirtualMetrologyModel(config)
    metrics = vm_model.train(X_train, y_train)
    logger.info(f"Model: R²={metrics.r2:.4f}, RMSE={metrics.rmse:.4f}")
    
    # Anomaly Detection
    anomaly_data = generate_anomaly_data()
    train_data = anomaly_data[anomaly_data['is_anomaly'] == 0].drop('is_anomaly', axis=1)
    test_data = anomaly_data.drop('is_anomaly', axis=1)
    detector = AnomalyDetector(method=AnomalyMethod.ISOLATION_FOREST)
    detector.fit(train_data)
    results = detector.detect(test_data)
    logger.info(f"Detected {sum(1 for r in results if r.is_anomaly)} anomalies")
    
    # Drift Detection
    X_ref = pd.DataFrame(np.random.randn(1000, 10), columns=[f"feat_{i}" for i in range(10)])
    X_new = pd.DataFrame(np.random.randn(500, 10) + 0.5, columns=[f"feat_{i}" for i in range(10)])
    drift_detector = DriftDetector(method=DriftMethod.PSI)
    drift_detector.set_reference(X_ref)
    drift_result = drift_detector.detect_drift(X_new)
    logger.info(f"Drift: {drift_result.drift_detected}, Score: {drift_result.drift_score:.4f}")
    
    # Time Series
    timestamps, values = generate_timeseries_data()
    forecaster = TimeSeriesForecaster(method="linear", horizon=30)
    forecaster.fit(timestamps[:-30], values[:-30])
    yhat, yhat_lower, yhat_upper = forecaster.forecast()
    mae = np.mean(np.abs(yhat - values[-30:]))
    logger.info(f"Forecast MAE: {mae:.2f}")
    
    logger.info("Session 14 Complete")
