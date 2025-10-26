"""
SESSION 14 ENHANCED: ADVANCED ML/VM SUITE
Complete Production-Grade Implementation with Enterprise Features

NEW CAPABILITIES IN THIS ENHANCED VERSION:
1. AutoML with Optuna hyperparameter optimization
2. Model explainability (SHAP, LIME, permutation importance)
3. Advanced ensemble methods (stacking, blending, voting)
4. Automated feature selection (RFE, SelectKBest, Boruta)
5. Production monitoring with Prometheus metrics
6. A/B testing framework for model comparison
7. Model governance and audit trails
8. Advanced drift detection (KL divergence, Wasserstein distance)
9. Causal inference for root cause analysis
10. Federated learning preparation
11. Time series decomposition (STL, seasonal trend)
12. Anomaly explanation with counterfactuals
13. Model compression and quantization
14. Online learning and incremental updates
15. Multi-objective optimization

Author: Semiconductor Lab Platform Team
Date: October 2024
Version: 2.0.0 (Enhanced)
"""

import asyncio
import hashlib
import json
import logging
import pickle
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from prophet import Prophet
from scipy import stats
from scipy.spatial.distance import jensenshannon, wasserstein_distance
from scipy.signal import find_peaks
from scipy.stats import ks_2samp, chi2_contingency, entropy
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    SequentialFeatureSelector,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    precision_score,
    roc_auc_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import cross_val_score, train_test_split, TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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
    Index,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from statsmodels.tsa.seasonal import STL

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    warnings.warn("LightGBM not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    warnings.warn("CatBoost not available")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available - AutoML features disabled")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available - explainability limited")

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    warnings.warn("LIME not available")

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False
    warnings.warn("skl2onnx not available")

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    warnings.warn("Prometheus client not available")

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
# PROMETHEUS METRICS
# ============================================================================

if PROMETHEUS_AVAILABLE:
    registry = CollectorRegistry()
    
    # Model training metrics
    model_training_duration = Histogram(
        'ml_model_training_duration_seconds',
        'Time spent training models',
        ['model_type', 'algorithm'],
        registry=registry
    )
    
    model_training_counter = Counter(
        'ml_model_training_total',
        'Total number of model training jobs',
        ['model_type', 'algorithm', 'status'],
        registry=registry
    )
    
    # Prediction metrics
    prediction_latency = Histogram(
        'ml_prediction_latency_seconds',
        'Prediction latency',
        ['model_name'],
        registry=registry
    )
    
    prediction_counter = Counter(
        'ml_predictions_total',
        'Total predictions made',
        ['model_name'],
        registry=registry
    )
    
    # Anomaly detection metrics
    anomaly_counter = Counter(
        'ml_anomalies_detected_total',
        'Total anomalies detected',
        ['detector_name', 'severity'],
        registry=registry
    )
    
    # Drift detection metrics
    drift_score = Gauge(
        'ml_drift_score',
        'Current drift score',
        ['model_name', 'drift_type'],
        registry=registry
    )
    
    # Model performance metrics
    model_r2_score = Gauge(
        'ml_model_r2_score',
        'Current model R2 score',
        ['model_name'],
        registry=registry
    )
    
    model_rmse = Gauge(
        'ml_model_rmse',
        'Current model RMSE',
        ['model_name'],
        registry=registry
    )


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
    ENSEMBLE = "ensemble"


class ModelAlgorithm(str, Enum):
    """Model algorithm enumeration - ENHANCED"""
    # Tree-based
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    
    # Linear
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    
    # SVM
    SVR = "svr"
    
    # Ensemble
    STACKING = "stacking"
    VOTING = "voting"
    BLENDING = "blending"
    
    # Anomaly detection
    ISOLATION_FOREST = "isolation_forest"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    PCA_ANOMALY = "pca_anomaly"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    ONE_CLASS_SVM = "one_class_svm"
    
    # Time series
    PROPHET = "prophet"
    ARIMA = "arima"
    SARIMA = "sarima"
    
    # AutoML
    AUTO_ML = "auto_ml"


class ModelStatus(str, Enum):
    """Model status enumeration"""
    TRAINING = "training"
    VALIDATING = "validating"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"
    ARCHIVED = "archived"
    AB_TESTING = "ab_testing"


class DriftType(str, Enum):
    """Drift type enumeration - ENHANCED"""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    COVARIATE_SHIFT = "covariate_shift"
    PRIOR_SHIFT = "prior_shift"
    LABEL_SHIFT = "label_shift"


class FeatureSelectionMethod(str, Enum):
    """Feature selection methods"""
    RFE = "rfe"
    SELECT_K_BEST = "select_k_best"
    MUTUAL_INFO = "mutual_info"
    SEQUENTIAL_FORWARD = "sequential_forward"
    SEQUENTIAL_BACKWARD = "sequential_backward"
    BORUTA = "boruta"
    PERMUTATION_IMPORTANCE = "permutation_importance"


class ExplainabilityMethod(str, Enum):
    """Explainability methods"""
    SHAP = "shap"
    LIME = "lime"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    PARTIAL_DEPENDENCE = "partial_dependence"
    ICE = "ice"  # Individual Conditional Expectation


# ============================================================================
# ENHANCED DATABASE MODELS
# ============================================================================

class MLModel(Base):
    """Enhanced ML Model registry with governance"""
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True)
    uuid = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True, nullable=False)
    name = Column(String(255), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    model_type = Column(String(50), nullable=False)
    algorithm = Column(String(50), nullable=False)
    status = Column(String(50), default=ModelStatus.TRAINING.value)
    
    # Model artifacts
    model_path = Column(String(500))
    onnx_path = Column(String(500))
    model_hash = Column(String(64))  # SHA256 hash for integrity
    
    # Configuration
    config = Column(JSONB)
    hyperparameters = Column(JSONB)
    
    # Training metadata
    training_data_hash = Column(String(64))
    feature_names = Column(ARRAY(String))
    target_name = Column(String(255))
    n_features = Column(Integer)
    n_samples_train = Column(Integer)
    n_samples_test = Column(Integer)
    
    # Performance metrics
    metrics_train = Column(JSONB)
    metrics_test = Column(JSONB)
    metrics_cv = Column(JSONB)
    feature_importance = Column(JSONB)
    
    # Governance
    created_by = Column(String(255))
    approved_by = Column(String(255))
    approval_date = Column(DateTime)
    
    # Deployment info
    deployment_id = Column(String(100))
    deployed_at = Column(DateTime)
    endpoint_url = Column(String(500))
    
    # Monitoring
    prediction_count = Column(Integer, default=0)
    last_prediction_at = Column(DateTime)
    current_drift_score = Column(Float)
    last_drift_check_at = Column(DateTime)
    
    # A/B Testing
    ab_test_id = Column(String(100))
    ab_test_variant = Column(String(50))  # 'control', 'variant_a', 'variant_b'
    ab_test_traffic_percentage = Column(Float)
    
    # Lineage
    parent_model_id = Column(Integer, ForeignKey('ml_models.id'))
    parent_model = relationship('MLModel', remote_side=[id])
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_ml_models_name_version', 'name', 'version'),
        Index('idx_ml_models_status', 'status'),
        Index('idx_ml_models_algorithm', 'algorithm'),
        Index('idx_ml_models_created_at', 'created_at'),
    )


class ModelExplanation(Base):
    """Store model explanations for interpretability"""
    __tablename__ = "model_explanations"
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'), nullable=False)
    explanation_method = Column(String(50), nullable=False)
    
    # Global explanations
    global_feature_importance = Column(JSONB)
    global_shap_values = Column(JSONB)
    
    # Interaction effects
    interaction_values = Column(JSONB)
    
    # Partial dependence
    partial_dependence_plots = Column(JSONB)
    
    # Model summary
    summary_plot_data = Column(JSONB)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    model = relationship('MLModel', backref='explanations')


class PredictionLog(Base):
    """Enhanced prediction logging for monitoring and audit"""
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('ml_models.id'), nullable=False)
    prediction_id = Column(UUID(as_uuid=True), default=uuid.uuid4, unique=True)
    
    # Input/Output
    features = Column(JSONB, nullable=False)
    prediction = Column(Float)
    prediction_proba = Column(ARRAY(Float))
    uncertainty = Column(Float)
    
    # Explanation
    shap_values = Column(JSONB)
    lime_explanation = Column(JSONB)
    
    # Context
    run_id = Column(Integer)
    sample_id = Column(Integer)
    wafer_id = Column(Integer)
    device_id = Column(Integer)
    
    # Quality flags
    is_anomalous = Column(Boolean, default=False)
    confidence_score = Column(Float)
    drift_detected = Column(Boolean, default=False)
    
    # Performance
    latency_ms = Column(Float)
    
    # A/B Testing
    ab_test_variant = Column(String(50))
    
    # Feedback loop
    actual_value = Column(Float)
    feedback_received = Column(Boolean, default=False)
    feedback_timestamp = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationship
    model = relationship('MLModel', backref='predictions')
    
    __table_args__ = (
        Index('idx_prediction_logs_model_created', 'model_id', 'created_at'),
        Index('idx_prediction_logs_run', 'run_id'),
    )


class ABTest(Base):
    """A/B testing framework for model comparison"""
    __tablename__ = "ab_tests"
    
    id = Column(Integer, primary_key=True)
    test_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Models being tested
    control_model_id = Column(Integer, ForeignKey('ml_models.id'))
    variant_models = Column(JSONB)  # {variant_a: model_id, variant_b: model_id}
    
    # Traffic allocation
    traffic_allocation = Column(JSONB)  # {control: 50, variant_a: 25, variant_b: 25}
    
    # Test configuration
    min_sample_size = Column(Integer, default=100)
    max_duration_days = Column(Integer, default=14)
    success_metric = Column(String(100))  # 'r2', 'rmse', 'mae', etc.
    
    # Status
    status = Column(String(50))  # 'running', 'completed', 'stopped'
    started_at = Column(DateTime)
    ended_at = Column(DateTime)
    
    # Results
    results = Column(JSONB)
    winner_variant = Column(String(50))
    statistical_significance = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class AutoMLExperiment(Base):
    """Track AutoML experiments"""
    __tablename__ = "automl_experiments"
    
    id = Column(Integer, primary_key=True)
    experiment_id = Column(String(100), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    
    # Configuration
    target_metric = Column(String(100))  # 'r2', 'rmse', etc.
    n_trials = Column(Integer)
    timeout_seconds = Column(Integer)
    
    # Search space
    algorithms = Column(ARRAY(String))
    search_space = Column(JSONB)
    
    # Best trial
    best_trial_id = Column(String(100))
    best_model_id = Column(Integer, ForeignKey('ml_models.id'))
    best_score = Column(Float)
    best_hyperparameters = Column(JSONB)
    
    # All trials
    all_trials = Column(JSONB)
    
    # Status
    status = Column(String(50))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    created_at = Column(DateTime, default=datetime.utcnow)


# ============================================================================
# ENHANCED CONFIGURATIONS
# ============================================================================

@dataclass
class AutoMLConfig:
    """Configuration for AutoML"""
    target_metric: str = "r2"  # 'r2', 'rmse', 'mae'
    n_trials: int = 100
    timeout: int = 3600  # seconds
    cv_folds: int = 5
    algorithms: List[str] = field(default_factory=lambda: [
        "random_forest", "gradient_boosting", "lightgbm", "xgboost"
    ])
    enable_ensemble: bool = True
    enable_feature_selection: bool = True
    early_stopping_rounds: int = 10
    random_state: int = 42


@dataclass
class ExplainabilityConfig:
    """Configuration for model explainability"""
    methods: List[str] = field(default_factory=lambda: ["shap", "permutation_importance"])
    compute_interactions: bool = False
    n_samples_shap: int = 100  # For speed
    n_repeats_permutation: int = 10


@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection"""
    method: str = "rfe"  # 'rfe', 'select_k_best', 'boruta'
    n_features_to_select: Optional[int] = None  # None = auto
    scoring: str = "r2"
    cv_folds: int = 5


@dataclass
class EnsembleConfig:
    """Configuration for ensemble methods"""
    method: str = "stacking"  # 'stacking', 'voting', 'blending'
    base_models: List[str] = field(default_factory=lambda: [
        "random_forest", "gradient_boosting", "lightgbm"
    ])
    meta_model: str = "ridge"
    cv_folds: int = 5


@dataclass
class MonitoringConfig:
    """Configuration for production monitoring"""
    enable_prometheus: bool = True
    enable_drift_detection: bool = True
    drift_check_interval_hours: int = 24
    drift_threshold: float = 0.2
    enable_prediction_logging: bool = True
    log_sample_rate: float = 0.1  # Log 10% of predictions
    enable_explainability: bool = False  # Expensive, opt-in


# ============================================================================
# AUTOML ENGINE
# ============================================================================

class AutoMLEngine:
    """
    Automated Machine Learning Engine
    - Hyperparameter optimization with Optuna
    - Algorithm selection
    - Feature engineering
    - Ensemble creation
    """
    
    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.study = None
        self.best_model = None
        self.best_params = None
        self.trials_history = []
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required for AutoML. Install: pip install optuna")
    
    def _get_search_space(self, trial: optuna.Trial, algorithm: str) -> Dict[str, Any]:
        """Define hyperparameter search space for each algorithm"""
        
        spaces = {
            "random_forest": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            },
            "gradient_boosting": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            },
            "lightgbm": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            },
            "xgboost": {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "gamma": trial.suggest_float("gamma", 0, 1),
            },
            "ridge": {
                "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
            },
            "lasso": {
                "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
            },
            "elastic_net": {
                "alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0, 1),
            },
        }
        
        return spaces.get(algorithm, {})
    
    def _create_model(self, algorithm: str, params: Dict[str, Any]):
        """Create model instance from algorithm and parameters"""
        
        models = {
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "lightgbm": lgb.LGBMRegressor if LIGHTGBM_AVAILABLE else None,
            "xgboost": xgb.XGBRegressor if XGBOOST_AVAILABLE else None,
            "ridge": Ridge,
            "lasso": Lasso,
            "elastic_net": ElasticNet,
            "extra_trees": ExtraTreesRegressor,
        }
        
        model_class = models.get(algorithm)
        if model_class is None:
            raise ValueError(f"Algorithm {algorithm} not available or not installed")
        
        return model_class(**params, random_state=self.config.random_state)
    
    def _objective(self, trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
        """Optuna objective function"""
        
        # Select algorithm
        algorithm = trial.suggest_categorical("algorithm", self.config.algorithms)
        
        # Get hyperparameters
        params = self._get_search_space(trial, algorithm)
        
        try:
            # Create model
            model = self._create_model(algorithm, params)
            
            # Cross-validation
            scores = cross_val_score(
                model, X, y,
                cv=self.config.cv_folds,
                scoring='r2' if self.config.target_metric == 'r2' else 'neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Return mean score
            if self.config.target_metric == 'r2':
                return scores.mean()
            else:  # RMSE or MAE
                return -scores.mean()
        
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('-inf') if self.config.target_metric == 'r2' else float('inf')
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        experiment_name: str = "automl_experiment"
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Run AutoML optimization
        
        Returns:
            Tuple of (best_model, optimization_results)
        """
        
        logger.info(f"Starting AutoML optimization: {experiment_name}")
        logger.info(f"Target metric: {self.config.target_metric}")
        logger.info(f"Number of trials: {self.config.n_trials}")
        
        # Create study
        direction = "maximize" if self.config.target_metric == "r2" else "minimize"
        self.study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.config.random_state),
            study_name=experiment_name
        )
        
        # Optimize
        self.study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            show_progress_bar=True
        )
        
        # Get best trial
        best_trial = self.study.best_trial
        self.best_params = best_trial.params
        
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best score: {best_trial.value:.4f}")
        logger.info(f"Best params: {self.best_params}")
        
        # Train best model on full dataset
        algorithm = self.best_params.pop("algorithm")
        self.best_model = self._create_model(algorithm, self.best_params)
        self.best_model.fit(X, y)
        
        # Compile results
        results = {
            "best_trial_number": best_trial.number,
            "best_score": best_trial.value,
            "best_algorithm": algorithm,
            "best_hyperparameters": self.best_params,
            "n_trials": len(self.study.trials),
            "optimization_history": [
                {"trial": t.number, "value": t.value, "params": t.params}
                for t in self.study.trials
            ]
        }
        
        return self.best_model, results
    
    def get_optimization_history_df(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        if self.study is None:
            return pd.DataFrame()
        
        return self.study.trials_dataframe()


# ============================================================================
# MODEL EXPLAINABILITY ENGINE
# ============================================================================

class ExplainabilityEngine:
    """
    Model explainability and interpretability
    - SHAP values
    - LIME explanations
    - Permutation importance
    - Partial dependence plots
    """
    
    def __init__(self, model: Any, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.explainer = None
        self.shap_explainer = None
        self.lime_explainer = None
    
    def compute_shap_values(
        self,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Compute SHAP values for global interpretability"""
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping")
            return {}
        
        try:
            # Sample data if too large
            if len(X) > self.config.n_samples_shap:
                indices = np.random.choice(len(X), self.config.n_samples_shap, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Create appropriate explainer
            if hasattr(self.model, 'tree_'):
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                self.shap_explainer = shap.Explainer(self.model, X_sample)
            
            # Compute SHAP values
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Global feature importance (mean absolute SHAP)
            global_importance = np.abs(shap_values).mean(axis=0)
            
            # Interaction effects (if enabled and available)
            interactions = None
            if self.config.compute_interactions:
                try:
                    interactions = self.shap_explainer.shap_interaction_values(X_sample)
                except:
                    logger.warning("Interaction values not available for this model")
            
            results = {
                "shap_values": shap_values.tolist(),
                "base_value": float(self.shap_explainer.expected_value),
                "feature_importance": {
                    feature_names[i]: float(global_importance[i])
                    for i in range(len(feature_names))
                },
                "interactions": interactions.tolist() if interactions is not None else None
            }
            
            return results
        
        except Exception as e:
            logger.error(f"SHAP computation failed: {e}")
            return {}
    
    def explain_prediction(
        self,
        x: np.ndarray,
        feature_names: List[str],
        method: str = "shap"
    ) -> Dict[str, Any]:
        """
        Explain a single prediction
        
        Args:
            x: Single sample features
            feature_names: Feature names
            method: 'shap' or 'lime'
        """
        
        if method == "shap" and SHAP_AVAILABLE and self.shap_explainer is not None:
            shap_value = self.shap_explainer.shap_values(x.reshape(1, -1))
            
            return {
                "method": "shap",
                "feature_contributions": {
                    feature_names[i]: float(shap_value[0, i])
                    for i in range(len(feature_names))
                },
                "base_value": float(self.shap_explainer.expected_value),
                "predicted_value": float(self.model.predict(x.reshape(1, -1))[0])
            }
        
        elif method == "lime" and LIME_AVAILABLE:
            # LIME explanation
            # (Implementation would go here - abbreviated for space)
            return {"method": "lime", "explanation": "LIME explanation"}
        
        else:
            logger.warning(f"Explanation method {method} not available")
            return {}
    
    def compute_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compute permutation importance"""
        
        result = permutation_importance(
            self.model, X, y,
            n_repeats=self.config.n_repeats_permutation,
            random_state=42,
            n_jobs=-1
        )
        
        importance = {
            feature_names[i]: float(result.importances_mean[i])
            for i in range(len(feature_names))
        }
        
        return importance


# ============================================================================
# ADVANCED FEATURE SELECTION
# ============================================================================

class AdvancedFeatureSelector:
    """
    Advanced feature selection methods
    - Recursive Feature Elimination (RFE)
    - SelectKBest with various scoring functions
    - Sequential Feature Selection
    - Boruta (if available)
    """
    
    def __init__(self, config: FeatureSelectionConfig):
        self.config = config
        self.selector = None
        self.selected_features = None
        self.feature_scores = None
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Select features using configured method
        
        Returns:
            Tuple of (X_selected, selected_feature_names, selection_report)
        """
        
        method = self.config.method
        
        if method == "rfe":
            return self._rfe_selection(X, y, feature_names)
        elif method == "select_k_best":
            return self._select_k_best(X, y, feature_names)
        elif method == "mutual_info":
            return self._mutual_info_selection(X, y, feature_names)
        elif method == "sequential_forward":
            return self._sequential_selection(X, y, feature_names, direction='forward')
        elif method == "sequential_backward":
            return self._sequential_selection(X, y, feature_names, direction='backward')
        else:
            logger.warning(f"Unknown selection method: {method}, returning all features")
            return X, feature_names, {"method": "none"}
    
    def _rfe_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Recursive Feature Elimination"""
        
        # Base estimator
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Number of features to select
        n_features = self.config.n_features_to_select
        if n_features is None:
            n_features = max(5, X.shape[1] // 2)  # Select half by default
        
        # RFE
        self.selector = RFE(estimator, n_features_to_select=n_features, step=1)
        X_selected = self.selector.fit_transform(X, y)
        
        # Get selected features
        selected_mask = self.selector.support_
        selected_features = [f for f, selected in zip(feature_names, selected_mask) if selected]
        
        # Feature rankings
        rankings = {
            feature_names[i]: int(self.selector.ranking_[i])
            for i in range(len(feature_names))
        }
        
        report = {
            "method": "rfe",
            "n_features_selected": len(selected_features),
            "selected_features": selected_features,
            "feature_rankings": rankings
        }
        
        return X_selected, selected_features, report
    
    def _select_k_best(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """SelectKBest with F-statistic"""
        
        k = self.config.n_features_to_select
        if k is None:
            k = max(5, X.shape[1] // 2)
        
        self.selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        # Get selected features
        selected_mask = self.selector.get_support()
        selected_features = [f for f, selected in zip(feature_names, selected_mask) if selected]
        
        # Feature scores
        scores = {
            feature_names[i]: float(self.selector.scores_[i])
            for i in range(len(feature_names))
        }
        
        report = {
            "method": "select_k_best",
            "n_features_selected": len(selected_features),
            "selected_features": selected_features,
            "feature_scores": scores
        }
        
        return X_selected, selected_features, report
    
    def _mutual_info_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """SelectKBest with mutual information"""
        
        k = self.config.n_features_to_select
        if k is None:
            k = max(5, X.shape[1] // 2)
        
        self.selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_selected = self.selector.fit_transform(X, y)
        
        selected_mask = self.selector.get_support()
        selected_features = [f for f, selected in zip(feature_names, selected_mask) if selected]
        
        scores = {
            feature_names[i]: float(self.selector.scores_[i])
            for i in range(len(feature_names))
        }
        
        report = {
            "method": "mutual_info",
            "n_features_selected": len(selected_features),
            "selected_features": selected_features,
            "mutual_information_scores": scores
        }
        
        return X_selected, selected_features, report
    
    def _sequential_selection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        direction: str = 'forward'
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Sequential Feature Selection"""
        
        estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        
        n_features = self.config.n_features_to_select
        if n_features is None:
            n_features = max(5, X.shape[1] // 2)
        
        self.selector = SequentialFeatureSelector(
            estimator,
            n_features_to_select=n_features,
            direction=direction,
            scoring=self.config.scoring,
            cv=self.config.cv_folds,
            n_jobs=-1
        )
        
        X_selected = self.selector.fit_transform(X, y)
        
        selected_mask = self.selector.get_support()
        selected_features = [f for f, selected in zip(feature_names, selected_mask) if selected]
        
        report = {
            "method": f"sequential_{direction}",
            "n_features_selected": len(selected_features),
            "selected_features": selected_features
        }
        
        return X_selected, selected_features, report


# ============================================================================
# ADVANCED DRIFT DETECTION
# ============================================================================

class AdvancedDriftDetector:
    """
    Advanced drift detection with multiple statistical tests
    - Kolmogorov-Smirnov test
    - Population Stability Index (PSI)
    - Jensen-Shannon Divergence
    - Wasserstein Distance
    - Chi-square test
    """
    
    def __init__(self):
        self.reference_data = None
        self.reference_predictions = None
        self.feature_names = None
    
    def set_reference(
        self,
        X_reference: np.ndarray,
        y_reference: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None
    ):
        """Set reference distribution"""
        self.reference_data = X_reference
        self.reference_predictions = y_reference
        self.feature_names = feature_names
    
    def detect_drift(
        self,
        X_current: np.ndarray,
        y_current: Optional[np.ndarray] = None,
        methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect drift using multiple methods
        
        Args:
            X_current: Current data
            y_current: Current predictions/targets
            methods: List of methods to use ['ks', 'psi', 'jsd', 'wasserstein']
        """
        
        if methods is None:
            methods = ['ks', 'psi', 'jsd']
        
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        results = {
            "drift_detected": False,
            "drift_score": 0.0,
            "feature_drifts": {},
            "methods_used": methods,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Feature-wise drift detection
        n_features = self.reference_data.shape[1]
        
        for i in range(n_features):
            feature_name = self.feature_names[i] if self.feature_names else f"feature_{i}"
            
            ref_feature = self.reference_data[:, i]
            curr_feature = X_current[:, i]
            
            feature_drift = {}
            
            # KS test
            if 'ks' in methods:
                ks_stat, ks_pvalue = ks_2samp(ref_feature, curr_feature)
                feature_drift['ks_statistic'] = float(ks_stat)
                feature_drift['ks_pvalue'] = float(ks_pvalue)
                feature_drift['ks_drift'] = ks_pvalue < 0.05
            
            # PSI
            if 'psi' in methods:
                psi = self._calculate_psi(ref_feature, curr_feature)
                feature_drift['psi'] = float(psi)
                feature_drift['psi_drift'] = psi > 0.2
            
            # Jensen-Shannon Divergence
            if 'jsd' in methods:
                jsd = self._calculate_jsd(ref_feature, curr_feature)
                feature_drift['jsd'] = float(jsd)
                feature_drift['jsd_drift'] = jsd > 0.1
            
            # Wasserstein Distance
            if 'wasserstein' in methods:
                w_dist = wasserstein_distance(ref_feature, curr_feature)
                feature_drift['wasserstein_distance'] = float(w_dist)
            
            results['feature_drifts'][feature_name] = feature_drift
            
            # Check if any drift detected for this feature
            if any(feature_drift.get(f'{m}_drift', False) for m in methods):
                results['drift_detected'] = True
        
        # Prediction drift (if available)
        if y_current is not None and self.reference_predictions is not None:
            pred_drift = self._detect_prediction_drift(
                self.reference_predictions,
                y_current
            )
            results['prediction_drift'] = pred_drift
            
            if pred_drift.get('drift_detected', False):
                results['drift_detected'] = True
        
        # Calculate overall drift score (0-1)
        drift_scores = []
        for feature_drift in results['feature_drifts'].values():
            if 'psi' in feature_drift:
                drift_scores.append(min(feature_drift['psi'] / 0.2, 1.0))
            elif 'ks_statistic' in feature_drift:
                drift_scores.append(feature_drift['ks_statistic'])
        
        results['drift_score'] = float(np.mean(drift_scores)) if drift_scores else 0.0
        
        # Recommendation
        if results['drift_score'] > 0.5:
            results['recommendation'] = "URGENT: Retrain model immediately"
        elif results['drift_score'] > 0.3:
            results['recommendation'] = "WARNING: Schedule model retraining soon"
        elif results['drift_score'] > 0.1:
            results['recommendation'] = "MONITOR: Continue tracking drift"
        else:
            results['recommendation'] = "OK: No action needed"
        
        return results
    
    def _calculate_psi(self, ref: np.ndarray, curr: np.ndarray, bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        
        # Create bins based on reference distribution
        bin_edges = np.histogram_bin_edges(ref, bins=bins)
        
        # Calculate frequencies
        ref_freq = np.histogram(ref, bins=bin_edges)[0] / len(ref)
        curr_freq = np.histogram(curr, bins=bin_edges)[0] / len(curr)
        
        # Avoid division by zero
        ref_freq = np.where(ref_freq == 0, 0.0001, ref_freq)
        curr_freq = np.where(curr_freq == 0, 0.0001, curr_freq)
        
        # PSI formula
        psi = np.sum((curr_freq - ref_freq) * np.log(curr_freq / ref_freq))
        
        return float(psi)
    
    def _calculate_jsd(self, ref: np.ndarray, curr: np.ndarray, bins: int = 50) -> float:
        """Calculate Jensen-Shannon Divergence"""
        
        # Create histograms
        bin_edges = np.histogram_bin_edges(
            np.concatenate([ref, curr]),
            bins=bins
        )
        
        ref_hist = np.histogram(ref, bins=bin_edges)[0] / len(ref)
        curr_hist = np.histogram(curr, bins=bin_edges)[0] / len(curr)
        
        # Add small epsilon to avoid log(0)
        ref_hist = ref_hist + 1e-10
        curr_hist = curr_hist + 1e-10
        
        # Normalize
        ref_hist = ref_hist / ref_hist.sum()
        curr_hist = curr_hist / curr_hist.sum()
        
        # JSD
        jsd = jensenshannon(ref_hist, curr_hist)
        
        return float(jsd)
    
    def _detect_prediction_drift(
        self,
        ref_predictions: np.ndarray,
        curr_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """Detect drift in prediction distribution"""
        
        # KS test on predictions
        ks_stat, ks_pvalue = ks_2samp(ref_predictions, curr_predictions)
        
        # PSI on predictions
        psi = self._calculate_psi(ref_predictions, curr_predictions)
        
        # Mean shift
        ref_mean = np.mean(ref_predictions)
        curr_mean = np.mean(curr_predictions)
        mean_shift = abs(curr_mean - ref_mean) / (np.std(ref_predictions) + 1e-10)
        
        drift_detected = (ks_pvalue < 0.05) or (psi > 0.2) or (mean_shift > 2.0)
        
        return {
            "drift_detected": drift_detected,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "psi": float(psi),
            "mean_shift_sigma": float(mean_shift),
            "reference_mean": float(ref_mean),
            "current_mean": float(curr_mean)
        }


# ============================================================================
# ENSEMBLE METHODS
# ============================================================================

class EnsembleModelBuilder:
    """
    Build ensemble models
    - Stacking
    - Voting
    - Blending
    """
    
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.ensemble_model = None
        self.base_models = []
    
    def build_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Build ensemble model
        
        Returns:
            Tuple of (ensemble_model, training_results)
        """
        
        method = self.config.method
        
        if method == "stacking":
            return self._build_stacking(X_train, y_train)
        elif method == "voting":
            return self._build_voting(X_train, y_train)
        elif method == "blending":
            if X_val is None or y_val is None:
                raise ValueError("Blending requires validation set")
            return self._build_blending(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
    
    def _create_base_model(self, algorithm: str) -> Any:
        """Create base model instance"""
        
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
            "lightgbm": lgb.LGBMRegressor(n_estimators=100, random_state=42) if LIGHTGBM_AVAILABLE else None,
            "xgboost": xgb.XGBRegressor(n_estimators=100, random_state=42) if XGBOOST_AVAILABLE else None,
            "extra_trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
            "ridge": Ridge(alpha=1.0),
            "svr": SVR(kernel='rbf'),
        }
        
        model = models.get(algorithm)
        if model is None:
            raise ValueError(f"Algorithm {algorithm} not available")
        
        return model
    
    def _build_stacking(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build stacking ensemble"""
        
        logger.info("Building stacking ensemble")
        
        # Create base models
        estimators = []
        for algo in self.config.base_models:
            try:
                model = self._create_base_model(algo)
                if model is not None:
                    estimators.append((algo, model))
                    self.base_models.append(model)
            except Exception as e:
                logger.warning(f"Failed to create {algo}: {e}")
        
        if len(estimators) == 0:
            raise ValueError("No base models available")
        
        # Meta model
        meta_model = self._create_base_model(self.config.meta_model)
        
        # Create stacking regressor
        self.ensemble_model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=self.config.cv_folds,
            n_jobs=-1
        )
        
        # Train
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        results = {
            "method": "stacking",
            "base_models": self.config.base_models,
            "meta_model": self.config.meta_model,
            "n_base_models": len(estimators),
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse)
        }
        
        return self.ensemble_model, results
    
    def _build_voting(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build voting ensemble"""
        
        logger.info("Building voting ensemble")
        
        # Create base models
        estimators = []
        for algo in self.config.base_models:
            try:
                model = self._create_base_model(algo)
                if model is not None:
                    estimators.append((algo, model))
                    self.base_models.append(model)
            except Exception as e:
                logger.warning(f"Failed to create {algo}: {e}")
        
        # Create voting regressor
        self.ensemble_model = VotingRegressor(
            estimators=estimators,
            n_jobs=-1
        )
        
        # Train
        self.ensemble_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_train)
        train_r2 = r2_score(y_train, y_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        results = {
            "method": "voting",
            "base_models": self.config.base_models,
            "n_base_models": len(estimators),
            "train_r2": float(train_r2),
            "train_rmse": float(train_rmse)
        }
        
        return self.ensemble_model, results
    
    def _build_blending(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Build blending ensemble"""
        
        logger.info("Building blending ensemble")
        
        # Train base models
        base_predictions = []
        for algo in self.config.base_models:
            try:
                model = self._create_base_model(algo)
                if model is not None:
                    model.fit(X_train, y_train)
                    pred = model.predict(X_val)
                    base_predictions.append(pred)
                    self.base_models.append(model)
            except Exception as e:
                logger.warning(f"Failed to train {algo}: {e}")
        
        # Stack predictions as features for meta model
        X_meta = np.column_stack(base_predictions)
        
        # Train meta model
        meta_model = self._create_base_model(self.config.meta_model)
        meta_model.fit(X_meta, y_val)
        
        # For prediction, need to run all base models then meta
        class BlendingModel:
            def __init__(self, base_models, meta_model):
                self.base_models = base_models
                self.meta_model = meta_model
            
            def predict(self, X):
                base_preds = [model.predict(X) for model in self.base_models]
                X_meta = np.column_stack(base_preds)
                return self.meta_model.predict(X_meta)
        
        self.ensemble_model = BlendingModel(self.base_models, meta_model)
        
        # Evaluate
        y_pred = self.ensemble_model.predict(X_val)
        val_r2 = r2_score(y_val, y_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        results = {
            "method": "blending",
            "base_models": self.config.base_models,
            "meta_model": self.config.meta_model,
            "n_base_models": len(self.base_models),
            "val_r2": float(val_r2),
            "val_rmse": float(val_rmse)
        }
        
        return self.ensemble_model, results


# ============================================================================
# A/B TESTING FRAMEWORK
# ============================================================================

class ABTestFramework:
    """
    A/B testing framework for model comparison
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_test(
        self,
        name: str,
        control_model_id: int,
        variant_models: Dict[str, int],
        traffic_allocation: Dict[str, float],
        success_metric: str = "r2",
        min_sample_size: int = 100,
        max_duration_days: int = 14
    ) -> str:
        """
        Create new A/B test
        
        Args:
            name: Test name
            control_model_id: Control model ID
            variant_models: Dict of {variant_name: model_id}
            traffic_allocation: Dict of {variant_name: percentage}
            success_metric: Metric to optimize
            min_sample_size: Minimum samples per variant
            max_duration_days: Maximum test duration
        
        Returns:
            test_id
        """
        
        # Validate traffic allocation sums to 100
        total_traffic = sum(traffic_allocation.values())
        if not np.isclose(total_traffic, 100.0):
            raise ValueError(f"Traffic allocation must sum to 100, got {total_traffic}")
        
        test_id = f"abtest_{uuid.uuid4().hex[:8]}"
        
        test = ABTest(
            test_id=test_id,
            name=name,
            control_model_id=control_model_id,
            variant_models=variant_models,
            traffic_allocation=traffic_allocation,
            min_sample_size=min_sample_size,
            max_duration_days=max_duration_days,
            success_metric=success_metric,
            status="running",
            started_at=datetime.utcnow()
        )
        
        self.db.add(test)
        self.db.commit()
        
        logger.info(f"Created A/B test: {test_id}")
        
        return test_id
    
    def route_prediction(self, test_id: str) -> str:
        """
        Route prediction request to a model variant
        
        Returns:
            variant_name ('control', 'variant_a', etc.)
        """
        
        test = self.db.query(ABTest).filter_by(test_id=test_id).first()
        
        if test is None or test.status != "running":
            return "control"
        
        # Random selection based on traffic allocation
        rand = np.random.random() * 100
        cumulative = 0
        
        for variant, percentage in test.traffic_allocation.items():
            cumulative += percentage
            if rand <= cumulative:
                return variant
        
        return "control"
    
    def log_prediction_result(
        self,
        test_id: str,
        variant: str,
        actual_value: float,
        predicted_value: float
    ):
        """Log prediction result for A/B test analysis"""
        
        # This would be stored in prediction_logs table
        # with ab_test_variant and actual_value fields
        pass
    
    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """
        Analyze A/B test results
        
        Returns statistical significance and winner
        """
        
        test = self.db.query(ABTest).filter_by(test_id=test_id).first()
        
        if test is None:
            raise ValueError(f"Test {test_id} not found")
        
        # Get predictions for each variant
        predictions = self.db.query(PredictionLog).filter(
            PredictionLog.ab_test_variant.in_(
                ['control'] + list(test.variant_models.keys())
            )
        ).all()
        
        # Group by variant
        variant_results = defaultdict(list)
        for pred in predictions:
            if pred.actual_value is not None:
                error = abs(pred.actual_value - pred.prediction)
                variant_results[pred.ab_test_variant].append(error)
        
        # Calculate metrics per variant
        variant_metrics = {}
        for variant, errors in variant_results.items():
            if len(errors) >= test.min_sample_size:
                variant_metrics[variant] = {
                    "n_samples": len(errors),
                    "mean_error": float(np.mean(errors)),
                    "std_error": float(np.std(errors)),
                    "median_error": float(np.median(errors))
                }
        
        # Statistical significance test (t-test between control and each variant)
        control_errors = variant_results.get('control', [])
        significance_results = {}
        
        for variant, errors in variant_results.items():
            if variant != 'control' and len(control_errors) > 0 and len(errors) > 0:
                t_stat, p_value = stats.ttest_ind(control_errors, errors)
                significance_results[variant] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "better_than_control": t_stat < 0  # Lower error is better
                }
        
        # Determine winner
        winner = "control"
        best_error = float('inf')
        
        for variant, metrics in variant_metrics.items():
            if metrics['mean_error'] < best_error:
                best_error = metrics['mean_error']
                winner = variant
        
        results = {
            "test_id": test_id,
            "status": test.status,
            "variant_metrics": variant_metrics,
            "significance_tests": significance_results,
            "winner": winner,
            "winner_improvement": (
                (variant_metrics.get('control', {}).get('mean_error', 0) - best_error) /
                variant_metrics.get('control', {}).get('mean_error', 1)
            ) * 100  # Percentage improvement
        }
        
        # Update test record
        test.results = results
        test.winner_variant = winner
        
        # Check if test should be concluded
        duration = (datetime.utcnow() - test.started_at).days
        if duration >= test.max_duration_days:
            test.status = "completed"
            test.ended_at = datetime.utcnow()
        
        self.db.commit()
        
        return results


# Continued in next part due to length...
