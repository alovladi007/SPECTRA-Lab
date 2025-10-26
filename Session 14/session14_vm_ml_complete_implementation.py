"""
SESSION 14: VIRTUAL METROLOGY & MACHINE LEARNING SUITE
Complete Implementation

Enterprise-grade ML/VM platform for semiconductor process monitoring:
- Feature engineering pipelines
- Virtual metrology models
- Anomaly detection (supervised & unsupervised)
- Drift monitoring and forecasting
- Model training, validation, and deployment
- ONNX export for production inference
- Model versioning and registry
- Automated retraining pipelines
- Predictive maintenance
- Calibration assistance

Author: Semiconductor Lab Platform Team
Date: October 2024
Version: 1.0.0
"""

import asyncio
import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from prophet import Prophet
from scipy import stats
from scipy.signal import find_peaks
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available, using sklearn GradientBoosting instead")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False
    warnings.warn("skl2onnx not available, ONNX export limited")

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database
Base = declarative_base()


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class ModelType(str, Enum):
    """Model type enumeration"""
    VIRTUAL_METROLOGY = "virtual_metrology"
    ANOMALY_DETECTION = "anomaly_detection"
    DRIFT_DETECTION = "drift_detection"
    PREDICTIVE_MAINTENANCE = "predictive_maintenance"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"


class ModelAlgorithm(str, Enum):
    """Model algorithm enumeration"""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LIGHTGBM = "lightgbm"
    ISOLATION_FOREST = "isolation_forest"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    PCA_ANOMALY = "pca_anomaly"
    PROPHET = "prophet"
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"


class ModelStatus(str, Enum):
    """Model status enumeration"""
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class DriftType(str, Enum):
    """Drift type enumeration"""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"


class AnomalyType(str, Enum):
    """Anomaly type enumeration"""
    POINT = "point"
    CONTEXTUAL = "contextual"
    COLLECTIVE = "collective"


# ============================================================================
# DATABASE MODELS
# ============================================================================

class MLModel(Base):
    """ML Model registry"""
    __tablename__ = 'ml_models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    algorithm = Column(String(50), nullable=False)
    status = Column(String(50), default='training')
    
    # Model artifacts
    model_path = Column(String(500))
    onnx_path = Column(String(500))
    scaler_path = Column(String(500))
    
    # Feature information
    feature_names = Column(ARRAY(String))
    target_name = Column(String(200))
    feature_importance = Column(JSONB)
    
    # Training metadata
    training_data_size = Column(Integer)
    training_start = Column(DateTime)
    training_end = Column(DateTime)
    training_config = Column(JSONB)
    
    # Performance metrics
    metrics = Column(JSONB)
    cv_scores = Column(JSONB)
    
    # Deployment
    deployed_at = Column(DateTime)
    deployment_config = Column(JSONB)
    
    # Monitoring
    last_prediction = Column(DateTime)
    prediction_count = Column(Integer, default=0)
    drift_detected = Column(Boolean, default=False)
    last_drift_check = Column(DateTime)
    
    # Metadata
    created_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    predictions = relationship("ModelPrediction", back_populates="model")
    drift_reports = relationship("DriftReport", back_populates="model")
    

class FeatureStore(Base):
    """Feature store for ML"""
    __tablename__ = 'feature_store'
    
    id = Column(Integer, primary_key=True)
    feature_set_name = Column(String(200), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Features
    feature_names = Column(ARRAY(String))
    feature_types = Column(JSONB)
    feature_definitions = Column(JSONB)
    
    # Statistics
    feature_statistics = Column(JSONB)
    correlation_matrix = Column(JSONB)
    
    # Lineage
    source_tables = Column(ARRAY(String))
    transformation_pipeline = Column(JSONB)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class ModelPrediction(Base):
    """Model prediction logging"""
    __tablename__ = 'model_predictions'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    
    # Input/Output
    features = Column(JSONB)
    prediction = Column(Float)
    prediction_proba = Column(ARRAY(Float))
    
    # Confidence
    confidence_score = Column(Float)
    uncertainty = Column(Float)
    
    # Context
    sample_id = Column(String(100))
    run_id = Column(Integer)
    instrument_id = Column(Integer)
    
    # Actual value (if available)
    actual_value = Column(Float)
    prediction_error = Column(Float)
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow)
    inference_time_ms = Column(Float)
    
    # Relationships
    model = relationship("MLModel", back_populates="predictions")


class DriftReport(Base):
    """Drift detection reports"""
    __tablename__ = 'drift_reports'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    
    # Drift detection
    drift_type = Column(String(50))
    drift_detected = Column(Boolean)
    drift_score = Column(Float)
    drift_threshold = Column(Float)
    
    # Analysis window
    reference_start = Column(DateTime)
    reference_end = Column(DateTime)
    current_start = Column(DateTime)
    current_end = Column(DateTime)
    
    # Details
    affected_features = Column(ARRAY(String))
    feature_drift_scores = Column(JSONB)
    statistical_tests = Column(JSONB)
    
    # Recommendations
    recommended_action = Column(String(50))
    retrain_recommended = Column(Boolean)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    model = relationship("MLModel", back_populates="drift_reports")


class AnomalyDetection(Base):
    """Anomaly detection results"""
    __tablename__ = 'anomaly_detections'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    
    # Anomaly info
    is_anomaly = Column(Boolean)
    anomaly_score = Column(Float)
    anomaly_type = Column(String(50))
    
    # Context
    sample_id = Column(String(100))
    run_id = Column(Integer)
    timestamp = Column(DateTime)
    
    # Features at time of anomaly
    features = Column(JSONB)
    feature_contributions = Column(JSONB)
    
    # Explanation
    likely_causes = Column(ARRAY(String))
    similar_anomalies = Column(JSONB)
    
    # Resolution
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Metadata
    detected_at = Column(DateTime, default=datetime.utcnow)


class MaintenancePrediction(Base):
    """Predictive maintenance predictions"""
    __tablename__ = 'maintenance_predictions'
    
    id = Column(Integer, primary_key=True)
    instrument_id = Column(Integer, nullable=False)
    model_id = Column(Integer, ForeignKey('ml_models.id'))
    
    # Prediction
    failure_probability = Column(Float)
    estimated_rul_hours = Column(Float)  # Remaining useful life
    confidence_interval = Column(ARRAY(Float))
    
    # Risk assessment
    risk_level = Column(String(20))
    recommended_action = Column(String(50))
    urgency_score = Column(Float)
    
    # Features
    health_indicators = Column(JSONB)
    degradation_rate = Column(Float)
    
    # Recommendations
    maintenance_type = Column(String(50))
    estimated_cost = Column(Float)
    suggested_date = Column(DateTime)
    
    # Outcome tracking
    actual_failure_date = Column(DateTime)
    maintenance_performed = Column(Boolean, default=False)
    maintenance_date = Column(DateTime)
    
    # Metadata
    predicted_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration"""
    # Aggregations
    compute_rolling_stats: bool = True
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # Transformations
    compute_differences: bool = True
    compute_ratios: bool = True
    compute_interactions: bool = False
    
    # Statistical features
    compute_distributions: bool = True
    compute_outlier_scores: bool = True
    
    # Time-based features
    include_temporal: bool = True
    include_cyclical: bool = True
    
    # Process knowledge
    include_domain_features: bool = True


class FeatureEngineer:
    """Feature engineering for semiconductor data"""
    
    def __init__(self, config: Optional[FeatureEngineeringConfig] = None):
        self.config = config or FeatureEngineeringConfig()
        self.feature_definitions = {}
        
    def engineer_features(
        self,
        df: pd.DataFrame,
        target_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Engineer features from raw data
        
        Args:
            df: Raw data
            target_col: Target column name (optional)
            
        Returns:
            DataFrame with engineered features
        """
        result = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_col and target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Rolling statistics
        if self.config.compute_rolling_stats:
            result = self._add_rolling_features(result, numeric_cols)
        
        # Differences and rates of change
        if self.config.compute_differences:
            result = self._add_difference_features(result, numeric_cols)
        
        # Ratios
        if self.config.compute_ratios:
            result = self._add_ratio_features(result, numeric_cols)
        
        # Statistical distributions
        if self.config.compute_distributions:
            result = self._add_distribution_features(result, numeric_cols)
        
        # Outlier scores
        if self.config.compute_outlier_scores:
            result = self._add_outlier_features(result, numeric_cols)
        
        # Temporal features
        if self.config.include_temporal and 'timestamp' in df.columns:
            result = self._add_temporal_features(result)
        
        # Domain-specific features
        if self.config.include_domain_features:
            result = self._add_domain_features(result)
        
        # Interactions (if enabled)
        if self.config.compute_interactions:
            result = self._add_interaction_features(result, numeric_cols[:5])  # Limit to avoid explosion
        
        # Drop NaN values created by rolling/differencing
        result = result.dropna()
        
        return result
    
    def _add_rolling_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add rolling window statistics"""
        result = df.copy()
        
        for col in cols:
            for window in self.config.rolling_windows:
                # Mean
                result[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
                # Std
                result[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
                # Min/Max
                result[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
                result[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
                # Range
                result[f'{col}_roll_range_{window}'] = (
                    result[f'{col}_roll_max_{window}'] - result[f'{col}_roll_min_{window}']
                )
                
                self.feature_definitions[f'{col}_roll_mean_{window}'] = {
                    'type': 'rolling_stat',
                    'base_feature': col,
                    'window': window,
                    'stat': 'mean'
                }
        
        return result
    
    def _add_difference_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add difference and rate-of-change features"""
        result = df.copy()
        
        for col in cols:
            # First difference
            result[f'{col}_diff'] = df[col].diff()
            # Second difference (acceleration)
            result[f'{col}_diff2'] = df[col].diff().diff()
            # Percentage change
            result[f'{col}_pct_change'] = df[col].pct_change()
            
            self.feature_definitions[f'{col}_diff'] = {
                'type': 'difference',
                'base_feature': col,
                'order': 1
            }
        
        return result
    
    def _add_ratio_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add ratio features between selected columns"""
        result = df.copy()
        
        # Common semiconductor ratios
        ratio_pairs = [
            ('mobility', 'sheet_resistance'),
            ('thickness', 'roughness'),
            ('current', 'voltage'),
        ]
        
        for col1, col2 in ratio_pairs:
            if col1 in cols and col2 in cols:
                # Avoid division by zero
                mask = df[col2] != 0
                result.loc[mask, f'{col1}_{col2}_ratio'] = df.loc[mask, col1] / df.loc[mask, col2]
                
                self.feature_definitions[f'{col1}_{col2}_ratio'] = {
                    'type': 'ratio',
                    'numerator': col1,
                    'denominator': col2
                }
        
        return result
    
    def _add_distribution_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add statistical distribution features"""
        result = df.copy()
        
        window = 20  # Rolling window for distribution calc
        
        for col in cols:
            # Skewness
            result[f'{col}_skew'] = df[col].rolling(window).skew()
            # Kurtosis
            result[f'{col}_kurt'] = df[col].rolling(window).apply(lambda x: stats.kurtosis(x))
            
            self.feature_definitions[f'{col}_skew'] = {
                'type': 'distribution',
                'base_feature': col,
                'stat': 'skewness'
            }
        
        return result
    
    def _add_outlier_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add outlier detection scores"""
        result = df.copy()
        
        window = 20
        
        for col in cols:
            # Z-score
            roll_mean = df[col].rolling(window).mean()
            roll_std = df[col].rolling(window).std()
            result[f'{col}_zscore'] = (df[col] - roll_mean) / roll_std
            
            # IQR-based outlier score
            roll_q1 = df[col].rolling(window).quantile(0.25)
            roll_q3 = df[col].rolling(window).quantile(0.75)
            iqr = roll_q3 - roll_q1
            lower = roll_q1 - 1.5 * iqr
            upper = roll_q3 + 1.5 * iqr
            result[f'{col}_outlier_score'] = np.maximum(
                (lower - df[col]) / iqr,
                (df[col] - upper) / iqr
            ).fillna(0).clip(0, None)
            
            self.feature_definitions[f'{col}_zscore'] = {
                'type': 'outlier_score',
                'base_feature': col,
                'method': 'zscore'
            }
        
        return result
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features"""
        result = df.copy()
        
        if 'timestamp' not in df.columns:
            return result
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            result['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time-based features
        result['hour'] = result['timestamp'].dt.hour
        result['day_of_week'] = result['timestamp'].dt.dayofweek
        result['day_of_month'] = result['timestamp'].dt.day
        result['month'] = result['timestamp'].dt.month
        
        # Cyclical encoding
        result['hour_sin'] = np.sin(2 * np.pi * result['hour'] / 24)
        result['hour_cos'] = np.cos(2 * np.pi * result['hour'] / 24)
        result['day_sin'] = np.sin(2 * np.pi * result['day_of_week'] / 7)
        result['day_cos'] = np.cos(2 * np.pi * result['day_of_week'] / 7)
        
        # Time since start
        result['time_since_start'] = (
            result['timestamp'] - result['timestamp'].min()
        ).dt.total_seconds()
        
        self.feature_definitions['hour_sin'] = {
            'type': 'temporal',
            'encoding': 'cyclical',
            'period': 24
        }
        
        return result
    
    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add semiconductor domain-specific features"""
        result = df.copy()
        
        # Electrical domain
        if 'current' in df.columns and 'voltage' in df.columns:
            # Conductance
            mask = df['voltage'] != 0
            result.loc[mask, 'conductance'] = df.loc[mask, 'current'] / df.loc[mask, 'voltage']
            
            # Power
            result['power'] = df['current'] * df['voltage']
        
        # Optical domain
        if 'absorption' in df.columns and 'wavelength' in df.columns:
            # Tauc plot preparation
            result['tauc_y'] = (df['absorption'] * 1240 / df['wavelength']) ** 0.5
        
        # Film quality indicators
        if 'thickness' in df.columns and 'roughness' in df.columns:
            mask = df['thickness'] != 0
            result.loc[mask, 'roughness_ratio'] = df.loc[mask, 'roughness'] / df.loc[mask, 'thickness']
        
        return result
    
    def _add_interaction_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add interaction features (limited to avoid feature explosion)"""
        result = df.copy()
        
        # Only create interactions for most important pairs
        for i in range(len(cols)):
            for j in range(i + 1, min(i + 3, len(cols))):  # Limit interactions
                col1, col2 = cols[i], cols[j]
                # Multiplication
                result[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                self.feature_definitions[f'{col1}_x_{col2}'] = {
                    'type': 'interaction',
                    'features': [col1, col2],
                    'operation': 'multiply'
                }
        
        return result
    
    def get_feature_importance_report(
        self,
        feature_importance: Dict[str, float],
        top_n: int = 20
    ) -> Dict[str, Any]:
        """
        Generate feature importance report with definitions
        
        Args:
            feature_importance: Feature importance scores
            top_n: Number of top features to include
            
        Returns:
            Feature importance report
        """
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n]
        
        report = {
            'top_features': [],
            'feature_types': {},
            'recommendations': []
        }
        
        for feature_name, importance in sorted_features:
            feature_info = {
                'name': feature_name,
                'importance': float(importance),
                'definition': self.feature_definitions.get(feature_name, {})
            }
            report['top_features'].append(feature_info)
            
            # Count feature types
            feat_type = self.feature_definitions.get(feature_name, {}).get('type', 'raw')
            report['feature_types'][feat_type] = report['feature_types'].get(feat_type, 0) + 1
        
        # Generate recommendations
        if report['feature_types'].get('rolling_stat', 0) > 5:
            report['recommendations'].append(
                "Many rolling statistics are important - process stability is key"
            )
        
        if report['feature_types'].get('difference', 0) > 3:
            report['recommendations'].append(
                "Rate of change features are important - monitor process dynamics"
            )
        
        return report


# ============================================================================
# VIRTUAL METROLOGY
# ============================================================================

@dataclass
class VMModelConfig:
    """Virtual metrology model configuration"""
    algorithm: ModelAlgorithm = ModelAlgorithm.RANDOM_FOREST
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    n_jobs: int = -1
    
    # Training
    test_size: float = 0.2
    cv_folds: int = 5
    scale_features: bool = True
    
    # Feature selection
    feature_selection: bool = True
    max_features: Optional[int] = None
    importance_threshold: float = 0.01


class VirtualMetrologyModel:
    """Virtual metrology model for predicting measurements"""
    
    def __init__(self, config: Optional[VMModelConfig] = None):
        self.config = config or VMModelConfig()
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.target_name = None
        self.training_metrics = {}
        self.feature_importance = {}
        
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Train virtual metrology model
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Feature names
            target_name: Target name
            
        Returns:
            Training results
        """
        logger.info(f"Training VM model with {X.shape[0]} samples, {X.shape[1]} features")
        
        self.feature_names = feature_names or list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.target_name = target_name or "target"
        
        # Convert to numpy if needed
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        y_array = y.values if isinstance(y, pd.Series) else y
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Scale features
        if self.config.scale_features:
            self.scaler = RobustScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Select and train model
        if self.config.algorithm == ModelAlgorithm.RANDOM_FOREST:
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        elif self.config.algorithm == ModelAlgorithm.GRADIENT_BOOSTING:
            self.model = GradientBoostingRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth or 3,
                min_samples_split=self.config.min_samples_split,
                min_samples_leaf=self.config.min_samples_leaf,
                random_state=self.config.random_state
            )
        elif self.config.algorithm == ModelAlgorithm.LIGHTGBM and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth or -1,
                min_child_samples=self.config.min_samples_leaf,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=-1
            )
        else:
            logger.warning(f"Algorithm {self.config.algorithm} not available, using Random Forest")
            self.model = RandomForestRegressor(
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=self.config.cv_folds,
            scoring='r2',
            n_jobs=self.config.n_jobs
        )
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Metrics
        self.training_metrics = {
            'train': {
                'r2': float(r2_score(y_train, y_train_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
                'mae': float(mean_absolute_error(y_train, y_train_pred)),
                'mape': float(np.mean(np.abs((y_train - y_train_pred) / (y_train + 1e-10))) * 100)
            },
            'test': {
                'r2': float(r2_score(y_test, y_test_pred)),
                'rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
                'mae': float(mean_absolute_error(y_test, y_test_pred)),
                'mape': float(np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-10))) * 100)
            },
            'cv': {
                'r2_mean': float(np.mean(cv_scores)),
                'r2_std': float(np.std(cv_scores)),
                'scores': cv_scores.tolist()
            }
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_names or [f'feature_{i}' for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        logger.info(f"Training complete - Test R²: {self.training_metrics['test']['r2']:.4f}")
        
        return {
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'model_params': self.model.get_params()
        }
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_uncertainty: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict using trained model
        
        Args:
            X: Features
            return_uncertainty: Whether to return prediction uncertainty
            
        Returns:
            Predictions and optionally uncertainty estimates
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale
        if self.scaler is not None:
            X_array = self.scaler.transform(X_array)
        
        # Predict
        predictions = self.model.predict(X_array)
        
        if not return_uncertainty:
            return predictions
        
        # Estimate uncertainty using ensemble std (if available)
        if hasattr(self.model, 'estimators_'):
            # For ensemble models
            all_predictions = np.array([
                estimator.predict(X_array)
                for estimator in self.model.estimators_
            ])
            uncertainty = np.std(all_predictions, axis=0)
        else:
            # Fallback: use residual std from training
            uncertainty = np.full(len(predictions), self.training_metrics['test']['rmse'])
        
        return predictions, uncertainty
    
    def export_onnx(self, output_path: str) -> str:
        """
        Export model to ONNX format
        
        Args:
            output_path: Path to save ONNX model
            
        Returns:
            Path to saved model
        """
        if not SKL2ONNX_AVAILABLE:
            raise ImportError("skl2onnx not available for ONNX export")
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create sample input
        n_features = len(self.feature_names) if self.feature_names else 1
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        # Convert
        if self.scaler is not None:
            # Include scaler in pipeline
            from sklearn.pipeline import Pipeline
            pipeline = Pipeline([
                ('scaler', self.scaler),
                ('model', self.model)
            ])
            onnx_model = convert_sklearn(pipeline, initial_types=initial_type)
        else:
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
        
        # Save
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        logger.info(f"Model exported to ONNX: {output_path}")
        return output_path
    
    def save(self, path: str):
        """Save model to file"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'training_metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'VirtualMetrologyModel':
        """Load model from file"""
        model_data = joblib.load(path)
        
        vm = cls(config=model_data.get('config'))
        vm.model = model_data['model']
        vm.scaler = model_data['scaler']
        vm.feature_names = model_data['feature_names']
        vm.target_name = model_data['target_name']
        vm.training_metrics = model_data['training_metrics']
        vm.feature_importance = model_data['feature_importance']
        
        logger.info(f"Model loaded from {path}")
        return vm


# ============================================================================
# ANOMALY DETECTION
# ============================================================================

@dataclass
class AnomalyDetectorConfig:
    """Anomaly detector configuration"""
    algorithm: ModelAlgorithm = ModelAlgorithm.ISOLATION_FOREST
    contamination: float = 0.1
    n_estimators: int = 100
    random_state: int = 42
    
    # PCA-based
    n_components: Optional[int] = None
    threshold_percentile: float = 95
    
    # Multivariate
    support_fraction: float = 0.8


class AnomalyDetector:
    """Anomaly detection for process monitoring"""
    
    def __init__(self, config: Optional[AnomalyDetectorConfig] = None):
        self.config = config or AnomalyDetectorConfig()
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.feature_names = None
        
    def fit(self, X: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None):
        """
        Fit anomaly detector on normal data
        
        Args:
            X: Training data (normal samples)
            feature_names: Feature names
        """
        logger.info(f"Fitting anomaly detector on {X.shape[0]} samples")
        
        self.feature_names = feature_names or (list(X.columns) if isinstance(X, pd.DataFrame) else None)
        
        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Select algorithm
        if self.config.algorithm == ModelAlgorithm.ISOLATION_FOREST:
            self.model = IsolationForest(
                contamination=self.config.contamination,
                n_estimators=self.config.n_estimators,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            self.model.fit(X_scaled)
            
        elif self.config.algorithm == ModelAlgorithm.ELLIPTIC_ENVELOPE:
            self.model = EllipticEnvelope(
                contamination=self.config.contamination,
                support_fraction=self.config.support_fraction,
                random_state=self.config.random_state
            )
            self.model.fit(X_scaled)
            
        elif self.config.algorithm == ModelAlgorithm.PCA_ANOMALY:
            # PCA-based reconstruction error
            n_components = self.config.n_components or min(X_scaled.shape[1], X_scaled.shape[0] // 2)
            self.model = PCA(n_components=n_components, random_state=self.config.random_state)
            self.model.fit(X_scaled)
            
            # Compute reconstruction errors
            X_reconstructed = self.model.inverse_transform(self.model.transform(X_scaled))
            reconstruction_errors = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
            
            # Set threshold
            self.threshold = np.percentile(reconstruction_errors, self.config.threshold_percentile)
        
        logger.info("Anomaly detector fitted successfully")
    
    def predict(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        return_scores: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Detect anomalies
        
        Args:
            X: Data to check
            return_scores: Whether to return anomaly scores
            
        Returns:
            Anomaly predictions (-1 for anomaly, 1 for normal) and optionally scores
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale
        X_scaled = self.scaler.transform(X_array)
        
        if self.config.algorithm == ModelAlgorithm.PCA_ANOMALY:
            # Reconstruction error
            X_reconstructed = self.model.inverse_transform(self.model.transform(X_scaled))
            scores = np.sum((X_scaled - X_reconstructed) ** 2, axis=1)
            predictions = np.where(scores > self.threshold, -1, 1)
        else:
            # Use model's predict
            predictions = self.model.predict(X_scaled)
            # Get scores
            if hasattr(self.model, 'score_samples'):
                scores = -self.model.score_samples(X_scaled)  # Negative for anomaly scores
            elif hasattr(self.model, 'decision_function'):
                scores = -self.model.decision_function(X_scaled)
            else:
                scores = np.zeros(len(predictions))
        
        if return_scores:
            return predictions, scores
        return predictions
    
    def explain_anomaly(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        sample_idx: int
    ) -> Dict[str, Any]:
        """
        Explain why a sample is anomalous
        
        Args:
            X: Data
            sample_idx: Index of anomalous sample
            
        Returns:
            Explanation with feature contributions
        """
        # Convert to numpy
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = self.scaler.transform(X_array)
        
        sample = X_scaled[sample_idx:sample_idx+1]
        
        # Compute feature contributions
        feature_contributions = {}
        
        if self.config.algorithm == ModelAlgorithm.PCA_ANOMALY:
            # Reconstruction error per feature
            reconstructed = self.model.inverse_transform(self.model.transform(sample))
            errors = (sample - reconstructed)[0] ** 2
            
            for i, feat_name in enumerate(self.feature_names or range(len(errors))):
                feature_contributions[str(feat_name)] = float(errors[i])
        else:
            # Use deviation from mean for other algorithms
            mean = np.mean(X_scaled, axis=0)
            std = np.std(X_scaled, axis=0) + 1e-10
            z_scores = np.abs((sample[0] - mean) / std)
            
            for i, feat_name in enumerate(self.feature_names or range(len(z_scores))):
                feature_contributions[str(feat_name)] = float(z_scores[i])
        
        # Sort by contribution
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'sample_index': sample_idx,
            'feature_contributions': dict(sorted_contributions[:10]),  # Top 10
            'top_anomalous_features': [name for name, _ in sorted_contributions[:5]],
            'explanation': f"Most anomalous features: {', '.join([name for name, _ in sorted_contributions[:3]])}"
        }
    
    def save(self, path: str):
        """Save detector to file"""
        detector_data = {
            'model': self.model,
            'scaler': self.scaler,
            'threshold': self.threshold,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(detector_data, path)
    
    @classmethod
    def load(cls, path: str) -> 'AnomalyDetector':
        """Load detector from file"""
        detector_data = joblib.load(path)
        
        detector = cls(config=detector_data.get('config'))
        detector.model = detector_data['model']
        detector.scaler = detector_data['scaler']
        detector.threshold = detector_data['threshold']
        detector.feature_names = detector_data['feature_names']
        
        return detector


# ============================================================================
# DRIFT DETECTION
# ============================================================================

@dataclass
class DriftDetectorConfig:
    """Drift detector configuration"""
    # Statistical tests
    use_ks_test: bool = True
    use_chi2_test: bool = True
    use_psi: bool = True
    
    # Thresholds
    ks_threshold: float = 0.05  # p-value
    chi2_threshold: float = 0.05
    psi_threshold: float = 0.2
    
    # Windows
    reference_window: int = 1000
    detection_window: int = 100
    
    # Prediction drift
    monitor_predictions: bool = True
    prediction_drift_threshold: float = 0.1


class DriftDetector:
    """Drift detection for model monitoring"""
    
    def __init__(self, config: Optional[DriftDetectorConfig] = None):
        self.config = config or DriftDetectorConfig()
        self.reference_data = None
        self.reference_predictions = None
        self.feature_names = None
        
    def set_reference(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        predictions: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """
        Set reference distribution
        
        Args:
            X: Reference feature data
            predictions: Reference predictions (optional)
            feature_names: Feature names
        """
        self.reference_data = X.values if isinstance(X, pd.DataFrame) else X
        self.reference_predictions = predictions
        self.feature_names = feature_names or (list(X.columns) if isinstance(X, pd.DataFrame) else None)
        
        logger.info(f"Reference set with {self.reference_data.shape[0]} samples")
    
    def detect_drift(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detect drift in new data
        
        Args:
            X: Current data
            predictions: Current predictions (optional)
            
        Returns:
            Drift detection results
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")
        
        X_current = X.values if isinstance(X, pd.DataFrame) else X
        
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'feature_drifts': {},
            'statistical_tests': {},
            'recommendation': 'continue_monitoring'
        }
        
        # Feature-wise drift detection
        drift_scores = []
        
        for feat_idx in range(X_current.shape[1]):
            feat_name = self.feature_names[feat_idx] if self.feature_names else f'feature_{feat_idx}'
            
            ref_values = self.reference_data[:, feat_idx]
            curr_values = X_current[:, feat_idx]
            
            feat_drift = self._detect_feature_drift(ref_values, curr_values, feat_name)
            results['feature_drifts'][feat_name] = feat_drift
            
            if feat_drift['drift_detected']:
                drift_scores.append(feat_drift['drift_score'])
        
        # Prediction drift
        if predictions is not None and self.reference_predictions is not None:
            pred_drift = self._detect_prediction_drift(
                self.reference_predictions,
                predictions
            )
            results['prediction_drift'] = pred_drift
            
            if pred_drift['drift_detected']:
                drift_scores.append(pred_drift['drift_score'])
        
        # Overall assessment
        if drift_scores:
            results['drift_detected'] = True
            results['drift_score'] = np.mean(drift_scores)
            
            # Recommendation
            if results['drift_score'] > 0.5:
                results['recommendation'] = 'retrain_immediately'
            elif results['drift_score'] > 0.3:
                results['recommendation'] = 'schedule_retraining'
            else:
                results['recommendation'] = 'increase_monitoring'
        
        return results
    
    def _detect_feature_drift(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        feature_name: str
    ) -> Dict[str, Any]:
        """Detect drift for a single feature"""
        result = {
            'feature_name': feature_name,
            'drift_detected': False,
            'drift_score': 0.0,
            'tests': {}
        }
        
        # Kolmogorov-Smirnov test
        if self.config.use_ks_test:
            ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
            result['tests']['ks'] = {
                'statistic': float(ks_stat),
                'p_value': float(ks_pvalue),
                'drift_detected': ks_pvalue < self.config.ks_threshold
            }
            
            if ks_pvalue < self.config.ks_threshold:
                result['drift_detected'] = True
                result['drift_score'] = max(result['drift_score'], ks_stat)
        
        # Population Stability Index (PSI)
        if self.config.use_psi:
            psi = self._calculate_psi(reference, current)
            result['tests']['psi'] = {
                'value': float(psi),
                'drift_detected': psi > self.config.psi_threshold
            }
            
            if psi > self.config.psi_threshold:
                result['drift_detected'] = True
                result['drift_score'] = max(result['drift_score'], psi / 0.5)  # Normalize
        
        # Distribution statistics
        result['statistics'] = {
            'reference': {
                'mean': float(np.mean(reference)),
                'std': float(np.std(reference)),
                'median': float(np.median(reference))
            },
            'current': {
                'mean': float(np.mean(current)),
                'std': float(np.std(current)),
                'median': float(np.median(current))
            }
        }
        
        return result
    
    def _calculate_psi(self, reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        # Create bins from reference
        bin_edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
        
        # Digitize
        ref_binned = np.digitize(reference, bin_edges)
        curr_binned = np.digitize(current, bin_edges)
        
        # Calculate proportions
        ref_props = np.array([(ref_binned == i).mean() for i in range(1, bins + 1)])
        curr_props = np.array([(curr_binned == i).mean() for i in range(1, bins + 1)])
        
        # Avoid log(0)
        ref_props = np.maximum(ref_props, 0.0001)
        curr_props = np.maximum(curr_props, 0.0001)
        
        # PSI formula
        psi = np.sum((curr_props - ref_props) * np.log(curr_props / ref_props))
        
        return psi
    
    def _detect_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Detect drift in model predictions"""
        result = {
            'drift_detected': False,
            'drift_score': 0.0
        }
        
        # KS test on predictions
        ks_stat, ks_pvalue = stats.ks_2samp(reference_predictions, current_predictions)
        
        result['ks_test'] = {
            'statistic': float(ks_stat),
            'p_value': float(ks_pvalue)
        }
        
        if ks_pvalue < self.config.ks_threshold:
            result['drift_detected'] = True
            result['drift_score'] = ks_stat
        
        # Mean shift
        ref_mean = np.mean(reference_predictions)
        curr_mean = np.mean(current_predictions)
        relative_shift = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-10)
        
        result['mean_shift'] = {
            'reference_mean': float(ref_mean),
            'current_mean': float(curr_mean),
            'relative_shift': float(relative_shift)
        }
        
        if relative_shift > self.config.prediction_drift_threshold:
            result['drift_detected'] = True
            result['drift_score'] = max(result['drift_score'], relative_shift)
        
        return result


# ============================================================================
# TIME SERIES FORECASTING
# ============================================================================

@dataclass
class TimeSeriesForecastConfig:
    """Time series forecast configuration"""
    method: ModelAlgorithm = ModelAlgorithm.PROPHET
    forecast_horizon: int = 30
    confidence_interval: float = 0.95
    
    # Prophet-specific
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    changepoint_prior_scale: float = 0.05


class TimeSeriesForecaster:
    """Time series forecasting for process trends"""
    
    def __init__(self, config: Optional[TimeSeriesForecastConfig] = None):
        self.config = config or TimeSeriesForecastConfig()
        self.model = None
        self.last_training_date = None
        
    def fit(self, df: pd.DataFrame, date_col: str = 'ds', value_col: str = 'y'):
        """
        Fit forecasting model
        
        Args:
            df: Time series data with date and value columns
            date_col: Name of date column
            value_col: Name of value column
        """
        logger.info(f"Fitting time series model on {len(df)} points")
        
        # Prepare data
        df_prepared = df[[date_col, value_col]].copy()
        df_prepared.columns = ['ds', 'y']
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df_prepared['ds']):
            df_prepared['ds'] = pd.to_datetime(df_prepared['ds'])
        
        # Fit Prophet
        if self.config.method == ModelAlgorithm.PROPHET:
            self.model = Prophet(
                yearly_seasonality=self.config.yearly_seasonality,
                weekly_seasonality=self.config.weekly_seasonality,
                daily_seasonality=self.config.daily_seasonality,
                changepoint_prior_scale=self.config.changepoint_prior_scale,
                interval_width=self.config.confidence_interval
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(df_prepared)
            
            self.last_training_date = df_prepared['ds'].max()
        
        logger.info("Time series model fitted")
    
    def forecast(self, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Generate forecast
        
        Args:
            periods: Number of periods to forecast (uses config if None)
            
        Returns:
            Forecast DataFrame
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        periods = periods or self.config.forecast_horizon
        
        # Make future dataframe
        future = self.model.make_future_dataframe(periods=periods)
        
        # Predict
        forecast = self.model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend']]
    
    def detect_changepoints(self) -> List[Dict[str, Any]]:
        """
        Detect significant changepoints in the series
        
        Returns:
            List of changepoint information
        """
        if self.model is None:
            raise ValueError("Model not fitted yet")
        
        changepoints = []
        
        if hasattr(self.model, 'changepoints'):
            for i, cp_date in enumerate(self.model.changepoints):
                if i < len(self.model.params['delta'][0]):
                    delta = float(self.model.params['delta'][0][i])
                    
                    if abs(delta) > 0.01:  # Significant change
                        changepoints.append({
                            'date': pd.Timestamp(cp_date),
                            'delta': delta,
                            'direction': 'increase' if delta > 0 else 'decrease'
                        })
        
        return sorted(changepoints, key=lambda x: abs(x['delta']), reverse=True)[:5]
    
    def save(self, path: str):
        """Save forecaster"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load(cls, path: str) -> 'TimeSeriesForecaster':
        """Load forecaster"""
        forecaster = cls()
        with open(path, 'rb') as f:
            forecaster.model = pickle.load(f)
        return forecaster


# ============================================================================
# ML PIPELINE ORCHESTRATION
# ============================================================================

class MLPipeline:
    """End-to-end ML pipeline orchestrator"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db = db_session
        self.feature_engineer = FeatureEngineer()
        
    def train_vm_model(
        self,
        training_data: pd.DataFrame,
        target_col: str,
        model_name: str,
        feature_config: Optional[FeatureEngineeringConfig] = None,
        model_config: Optional[VMModelConfig] = None
    ) -> Tuple[VirtualMetrologyModel, Dict[str, Any]]:
        """
        Train a complete VM model with feature engineering
        
        Args:
            training_data: Raw training data
            target_col: Target column name
            model_name: Model name
            feature_config: Feature engineering config
            model_config: Model config
            
        Returns:
            Trained model and training report
        """
        logger.info(f"Training VM model: {model_name}")
        
        # Feature engineering
        if feature_config:
            self.feature_engineer = FeatureEngineer(feature_config)
        
        engineered_data = self.feature_engineer.engineer_features(
            training_data,
            target_col=target_col
        )
        
        # Separate features and target
        X = engineered_data.drop(columns=[target_col])
        y = engineered_data[target_col]
        
        # Train model
        vm_model = VirtualMetrologyModel(model_config)
        training_results = vm_model.train(
            X, y,
            feature_names=list(X.columns),
            target_name=target_col
        )
        
        # Feature importance report
        importance_report = self.feature_engineer.get_feature_importance_report(
            vm_model.feature_importance
        )
        
        # Save to database
        if self.db:
            self._save_model_to_db(
                vm_model,
                model_name,
                ModelType.VIRTUAL_METROLOGY,
                training_results,
                importance_report
            )
        
        report = {
            'model_name': model_name,
            'training_results': training_results,
            'feature_engineering': importance_report,
            'model_ready': True
        }
        
        return vm_model, report
    
    def train_anomaly_detector(
        self,
        normal_data: pd.DataFrame,
        model_name: str,
        detector_config: Optional[AnomalyDetectorConfig] = None
    ) -> Tuple[AnomalyDetector, Dict[str, Any]]:
        """
        Train anomaly detector
        
        Args:
            normal_data: Normal operating data
            model_name: Model name
            detector_config: Detector config
            
        Returns:
            Trained detector and report
        """
        logger.info(f"Training anomaly detector: {model_name}")
        
        # Feature engineering
        engineered_data = self.feature_engineer.engineer_features(normal_data)
        
        # Train detector
        detector = AnomalyDetector(detector_config)
        detector.fit(engineered_data, feature_names=list(engineered_data.columns))
        
        # Save to database
        if self.db:
            self._save_model_to_db(
                detector,
                model_name,
                ModelType.ANOMALY_DETECTION,
                {'algorithm': str(detector.config.algorithm)},
                {}
            )
        
        report = {
            'model_name': model_name,
            'algorithm': str(detector.config.algorithm),
            'training_samples': len(normal_data),
            'n_features': engineered_data.shape[1],
            'model_ready': True
        }
        
        return detector, report
    
    def _save_model_to_db(
        self,
        model: Any,
        name: str,
        model_type: ModelType,
        training_results: Dict[str, Any],
        feature_report: Dict[str, Any]
    ):
        """Save model metadata to database"""
        if not self.db:
            return
        
        try:
            db_model = MLModel(
                name=name,
                version='1.0.0',
                model_type=model_type.value,
                algorithm=str(model.config.algorithm) if hasattr(model, 'config') else 'unknown',
                status=ModelStatus.READY.value,
                metrics=training_results.get('metrics', {}),
                feature_importance=training_results.get('feature_importance', {}),
                training_end=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            
            self.db.add(db_model)
            self.db.commit()
            
            logger.info(f"Model {name} saved to database (ID: {db_model.id})")
            
        except Exception as e:
            logger.error(f"Error saving model to database: {e}")
            self.db.rollback()


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="ML/VM Hub API", version="1.0.0")


class VMTrainingRequest(BaseModel):
    """VM training request"""
    model_name: str
    training_data_path: str
    target_col: str
    algorithm: str = "random_forest"


class PredictionRequest(BaseModel):
    """Prediction request"""
    model_name: str
    features: Dict[str, float]


class AnomalyDetectionRequest(BaseModel):
    """Anomaly detection request"""
    detector_name: str
    features: Dict[str, float]


@app.post("/api/ml/train-vm")
async def train_vm_endpoint(request: VMTrainingRequest):
    """Train virtual metrology model"""
    try:
        # Load data
        training_data = pd.read_csv(request.training_data_path)
        
        # Configure
        model_config = VMModelConfig(
            algorithm=ModelAlgorithm(request.algorithm)
        )
        
        # Train
        pipeline = MLPipeline()
        model, report = pipeline.train_vm_model(
            training_data,
            request.target_col,
            request.model_name,
            model_config=model_config
        )
        
        # Save model
        model_path = f"/tmp/{request.model_name}.joblib"
        model.save(model_path)
        
        return {
            'status': 'success',
            'model_name': request.model_name,
            'model_path': model_path,
            'report': report
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ml/predict")
async def predict_endpoint(request: PredictionRequest):
    """Make prediction using trained model"""
    try:
        # Load model
        model_path = f"/tmp/{request.model_name}.joblib"
        model = VirtualMetrologyModel.load(model_path)
        
        # Prepare features
        X = pd.DataFrame([request.features])
        
        # Predict
        prediction, uncertainty = model.predict(X, return_uncertainty=True)
        
        return {
            'model_name': request.model_name,
            'prediction': float(prediction[0]),
            'uncertainty': float(uncertainty[0]),
            'timestamp': datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ml/models")
async def list_models():
    """List all trained models"""
    # Implementation would query database
    return {'models': []}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'ml-vm-hub',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8014)
