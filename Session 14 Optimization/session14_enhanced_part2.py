"""
SESSION 14 ENHANCED - PART 2: Additional Advanced Features

This file continues from session14_enhanced_implementation.py with:
- Time series decomposition
- Causal inference
- Model compression
- Online learning
- Multi-objective optimization
- Integration helpers
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# TIME SERIES DECOMPOSITION
# ============================================================================

class TimeSeriesDecomposer:
    """
    Advanced time series decomposition and analysis
    - STL decomposition
    - Trend extraction
    - Seasonality detection
    - Changepoint detection
    """
    
    def __init__(self):
        self.trend = None
        self.seasonal = None
        self.residual = None
        self.changepoints = []
    
    def decompose(
        self,
        data: pd.Series,
        period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Decompose time series using STL
        
        Args:
            data: Time series data
            period: Seasonal period (auto-detect if None)
        
        Returns:
            Decomposition components
        """
        
        if period is None:
            period = self._detect_period(data)
        
        # STL decomposition
        stl = STL(data, period=period, seasonal=13)
        result = stl.fit()
        
        self.trend = result.trend
        self.seasonal = result.seasonal
        self.residual = result.resid
        
        return {
            "trend": self.trend.to_dict(),
            "seasonal": self.seasonal.to_dict(),
            "residual": self.residual.to_dict(),
            "period": period,
            "seasonal_strength": self._calculate_seasonal_strength(),
            "trend_strength": self._calculate_trend_strength()
        }
    
    def _detect_period(self, data: pd.Series) -> int:
        """Auto-detect seasonal period using FFT"""
        
        # Remove trend
        detrended = data - data.rolling(window=7, center=True).mean()
        detrended = detrended.dropna()
        
        # FFT
        fft = np.fft.fft(detrended.values)
        power = np.abs(fft) ** 2
        freqs = np.fft.fftfreq(len(detrended))
        
        # Find dominant frequency (excluding DC component)
        idx = np.argmax(power[1:len(power)//2]) + 1
        dominant_freq = freqs[idx]
        
        if dominant_freq > 0:
            period = int(1 / dominant_freq)
            return max(2, period)
        else:
            return 7  # Default to weekly
    
    def _calculate_seasonal_strength(self) -> float:
        """Calculate strength of seasonality"""
        
        if self.seasonal is None or self.residual is None:
            return 0.0
        
        var_residual = np.var(self.residual)
        var_seasonal_residual = np.var(self.seasonal + self.residual)
        
        if var_seasonal_residual == 0:
            return 0.0
        
        strength = max(0, 1 - var_residual / var_seasonal_residual)
        return float(strength)
    
    def _calculate_trend_strength(self) -> float:
        """Calculate strength of trend"""
        
        if self.trend is None or self.residual is None:
            return 0.0
        
        var_residual = np.var(self.residual)
        var_trend_residual = np.var(self.trend + self.residual)
        
        if var_trend_residual == 0:
            return 0.0
        
        strength = max(0, 1 - var_residual / var_trend_residual)
        return float(strength)
    
    def detect_changepoints(
        self,
        data: pd.Series,
        min_distance: int = 10,
        threshold: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect changepoints in trend
        
        Args:
            data: Time series data
            min_distance: Minimum distance between changepoints
            threshold: Z-score threshold for changepoint detection
        """
        
        if self.trend is None:
            self.decompose(data)
        
        # Calculate first differences of trend
        trend_diff = np.diff(self.trend)
        
        # Z-score of differences
        z_scores = np.abs((trend_diff - np.mean(trend_diff)) / (np.std(trend_diff) + 1e-10))
        
        # Find peaks (potential changepoints)
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(z_scores, height=threshold, distance=min_distance)
        
        changepoints = []
        for i, peak in enumerate(peaks):
            changepoint = {
                "index": int(peak),
                "date": data.index[peak].isoformat() if hasattr(data.index[peak], 'isoformat') else peak,
                "magnitude": float(z_scores[peak]),
                "direction": "increase" if trend_diff[peak] > 0 else "decrease",
                "change_value": float(trend_diff[peak])
            }
            changepoints.append(changepoint)
        
        self.changepoints = changepoints
        return changepoints


# ============================================================================
# MODEL COMPRESSION
# ============================================================================

class ModelCompressor:
    """
    Model compression techniques
    - Pruning
    - Quantization
    - Knowledge distillation
    """
    
    def __init__(self):
        pass
    
    def prune_random_forest(
        self,
        model: Any,
        keep_ratio: float = 0.5
    ) -> Any:
        """
        Prune random forest by keeping only top performing trees
        
        Args:
            model: RandomForestRegressor
            keep_ratio: Ratio of trees to keep (0-1)
        
        Returns:
            Pruned model
        """
        
        from sklearn.ensemble import RandomForestRegressor
        
        if not isinstance(model, RandomForestRegressor):
            raise ValueError("Model must be RandomForestRegressor")
        
        # Get number of trees to keep
        n_trees = len(model.estimators_)
        n_keep = max(1, int(n_trees * keep_ratio))
        
        # Keep top trees (simple version - keep first n_keep)
        # More sophisticated: rank by OOB score or validation performance
        model.estimators_ = model.estimators_[:n_keep]
        model.n_estimators = n_keep
        
        logger.info(f"Pruned model from {n_trees} to {n_keep} trees")
        
        return model
    
    def quantize_model(self, model: Any) -> Any:
        """
        Quantize model weights for smaller size and faster inference
        
        Note: This is a placeholder. Real implementation would use
        ONNX quantization or TensorFlow Lite
        """
        
        logger.warning("Quantization not fully implemented in this version")
        return model


# ============================================================================
# ONLINE LEARNING
# ============================================================================

class OnlineLearner:
    """
    Online/incremental learning for model updates
    """
    
    def __init__(self, model: Any):
        self.model = model
        self.update_count = 0
        self.performance_history = []
    
    def partial_fit(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        learning_rate: float = 0.1
    ) -> Dict[str, Any]:
        """
        Incrementally update model with new data
        
        Args:
            X_new: New features
            y_new: New targets
            learning_rate: Learning rate for update
        
        Returns:
            Update statistics
        """
        
        # Check if model supports partial_fit
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new, y_new)
        else:
            # For models without partial_fit, retrain with combined data
            # This is not true online learning but incremental retraining
            logger.warning("Model doesn't support partial_fit, using full retraining")
            
            # Get previous training data if available
            # (In real implementation, would maintain a sliding window)
            self.model.fit(X_new, y_new)
        
        self.update_count += 1
        
        # Evaluate on new data
        y_pred = self.model.predict(X_new)
        mae = mean_absolute_error(y_new, y_pred)
        rmse = np.sqrt(mean_squared_error(y_new, y_pred))
        
        performance = {
            "update_number": self.update_count,
            "n_samples": len(X_new),
            "mae": float(mae),
            "rmse": float(rmse),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.performance_history.append(performance)
        
        return performance
    
    def should_retrain(self, threshold: float = 0.2) -> bool:
        """
        Determine if full retraining is needed based on performance degradation
        
        Args:
            threshold: Performance degradation threshold
        
        Returns:
            True if retraining recommended
        """
        
        if len(self.performance_history) < 2:
            return False
        
        # Compare recent performance to initial
        initial_mae = self.performance_history[0]['mae']
        recent_mae = np.mean([p['mae'] for p in self.performance_history[-5:]])
        
        degradation = (recent_mae - initial_mae) / (initial_mae + 1e-10)
        
        return degradation > threshold


# ============================================================================
# CAUSAL INFERENCE
# ============================================================================

class CausalAnalyzer:
    """
    Causal inference for root cause analysis
    """
    
    def __init__(self):
        pass
    
    def estimate_treatment_effect(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        confounders: List[str]
    ) -> Dict[str, Any]:
        """
        Estimate average treatment effect using propensity score matching
        
        Args:
            data: DataFrame with all variables
            treatment_col: Treatment variable (binary)
            outcome_col: Outcome variable
            confounders: List of confounder column names
        
        Returns:
            Treatment effect estimate
        """
        
        from sklearn.linear_model import LogisticRegression
        
        # Propensity score model
        X = data[confounders].values
        treatment = data[treatment_col].values
        
        propensity_model = LogisticRegression()
        propensity_model.fit(X, treatment)
        
        # Propensity scores
        propensity_scores = propensity_model.predict_proba(X)[:, 1]
        
        # Inverse probability weighting
        weights = np.where(
            treatment == 1,
            1 / (propensity_scores + 1e-10),
            1 / (1 - propensity_scores + 1e-10)
        )
        
        # Weighted outcome means
        treated_outcome = np.average(
            data[outcome_col].values[treatment == 1],
            weights=weights[treatment == 1]
        )
        
        control_outcome = np.average(
            data[outcome_col].values[treatment == 0],
            weights=weights[treatment == 0]
        )
        
        ate = treated_outcome - control_outcome
        
        return {
            "average_treatment_effect": float(ate),
            "treated_mean": float(treated_outcome),
            "control_mean": float(control_outcome),
            "n_treated": int((treatment == 1).sum()),
            "n_control": int((treatment == 0).sum())
        }
    
    def feature_causal_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: List[str],
        n_interventions: int = 100
    ) -> Dict[str, float]:
        """
        Estimate causal importance of features using interventions
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Feature names
            n_interventions: Number of interventional samples
        
        Returns:
            Dict of feature: causal_importance
        """
        
        baseline_pred = model.predict(X).mean()
        
        causal_importance = {}
        
        for i, feature in enumerate(feature_names):
            # Create interventional samples
            X_intervention = X.copy()
            
            # Set feature to different values
            feature_values = np.linspace(X[:, i].min(), X[:, i].max(), n_interventions)
            
            effects = []
            for value in feature_values:
                X_int = X_intervention.copy()
                X_int[:, i] = value
                pred = model.predict(X_int).mean()
                effect = abs(pred - baseline_pred)
                effects.append(effect)
            
            # Average causal effect
            causal_importance[feature] = float(np.mean(effects))
        
        return causal_importance


# ============================================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ============================================================================

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for model selection
    Balance between accuracy, speed, interpretability, etc.
    """
    
    def __init__(self):
        self.pareto_front = []
    
    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        objectives: List[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on multiple objectives
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            objectives: List of objectives to evaluate
        
        Returns:
            Dict of objective: value
        """
        
        if objectives is None:
            objectives = ['accuracy', 'speed', 'size', 'interpretability']
        
        results = {}
        
        # Accuracy (R2 score)
        if 'accuracy' in objectives:
            y_pred = model.predict(X_test)
            results['accuracy'] = float(r2_score(y_test, y_pred))
        
        # Speed (inference time)
        if 'speed' in objectives:
            import time
            start = time.time()
            _ = model.predict(X_test)
            duration = time.time() - start
            # Normalize: samples per second
            results['speed'] = len(X_test) / (duration + 1e-10)
        
        # Size (number of parameters or model size)
        if 'size' in objectives:
            # Estimate model size
            import sys
            import pickle
            model_bytes = len(pickle.dumps(model))
            results['size'] = model_bytes / 1024  # KB
        
        # Interpretability (heuristic)
        if 'interpretability' in objectives:
            # Simple heuristic based on model type
            if hasattr(model, 'coef_'):  # Linear models
                interpretability = 1.0
            elif hasattr(model, 'feature_importances_'):  # Tree-based
                interpretability = 0.7
            else:
                interpretability = 0.3
            results['interpretability'] = interpretability
        
        return results
    
    def find_pareto_optimal(
        self,
        models: List[Tuple[str, Any, Dict[str, float]]],
        maximize: List[str] = None,
        minimize: List[str] = None
    ) -> List[Tuple[str, Any, Dict[str, float]]]:
        """
        Find Pareto-optimal models
        
        Args:
            models: List of (name, model, objectives_dict)
            maximize: Objectives to maximize
            minimize: Objectives to minimize
        
        Returns:
            List of Pareto-optimal models
        """
        
        if maximize is None:
            maximize = ['accuracy', 'speed', 'interpretability']
        if minimize is None:
            minimize = ['size']
        
        pareto_front = []
        
        for i, (name_i, model_i, obj_i) in enumerate(models):
            is_dominated = False
            
            for j, (name_j, model_j, obj_j) in enumerate(models):
                if i == j:
                    continue
                
                # Check if i is dominated by j
                dominates = True
                strictly_better = False
                
                for obj in maximize:
                    if obj_j.get(obj, 0) < obj_i.get(obj, 0):
                        dominates = False
                        break
                    elif obj_j.get(obj, 0) > obj_i.get(obj, 0):
                        strictly_better = True
                
                for obj in minimize:
                    if obj_j.get(obj, float('inf')) > obj_i.get(obj, float('inf')):
                        dominates = False
                        break
                    elif obj_j.get(obj, float('inf')) < obj_i.get(obj, float('inf')):
                        strictly_better = True
                
                if dominates and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append((name_i, model_i, obj_i))
        
        self.pareto_front = pareto_front
        
        logger.info(f"Found {len(pareto_front)} Pareto-optimal models")
        
        return pareto_front


# ============================================================================
# INTEGRATION UTILITIES
# ============================================================================

class ModelRegistry:
    """
    Centralized model registry for managing all ML models
    """
    
    def __init__(self, db_session):
        self.db = db_session
    
    def register_model(
        self,
        name: str,
        version: str,
        model: Any,
        metadata: Dict[str, Any]
    ) -> int:
        """Register a new model"""
        
        import hashlib
        import pickle
        
        # Serialize model
        model_bytes = pickle.dumps(model)
        model_hash = hashlib.sha256(model_bytes).hexdigest()
        
        # Create record
        from session14_enhanced_implementation import MLModel, ModelStatus
        
        ml_model = MLModel(
            name=name,
            version=version,
            model_type=metadata.get('model_type', 'unknown'),
            algorithm=metadata.get('algorithm', 'unknown'),
            status=ModelStatus.READY.value,
            model_hash=model_hash,
            config=metadata.get('config', {}),
            hyperparameters=metadata.get('hyperparameters', {}),
            metrics_train=metadata.get('metrics_train', {}),
            metrics_test=metadata.get('metrics_test', {}),
            feature_names=metadata.get('feature_names', []),
            n_features=metadata.get('n_features', 0),
            created_by=metadata.get('created_by', 'system')
        )
        
        self.db.add(ml_model)
        self.db.commit()
        
        logger.info(f"Registered model: {name} v{version} (ID: {ml_model.id})")
        
        return ml_model.id
    
    def get_model(self, model_id: int) -> Optional[Any]:
        """Load model from registry"""
        
        from session14_enhanced_implementation import MLModel
        
        ml_model = self.db.query(MLModel).filter_by(id=model_id).first()
        
        if ml_model is None:
            return None
        
        # Load model from path
        if ml_model.model_path:
            import joblib
            model = joblib.load(ml_model.model_path)
            return model
        
        return None
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List all models in registry"""
        
        from session14_enhanced_implementation import MLModel
        
        query = self.db.query(MLModel)
        
        if model_type:
            query = query.filter_by(model_type=model_type)
        if status:
            query = query.filter_by(status=status)
        
        models = query.all()
        
        return [
            {
                "id": m.id,
                "name": m.name,
                "version": m.version,
                "algorithm": m.algorithm,
                "status": m.status,
                "created_at": m.created_at.isoformat() if m.created_at else None,
                "metrics": m.metrics_test
            }
            for m in models
        ]
    
    def promote_model(self, model_id: int, from_status: str, to_status: str):
        """Promote model to new status (e.g., ready -> deployed)"""
        
        from session14_enhanced_implementation import MLModel
        
        ml_model = self.db.query(MLModel).filter_by(id=model_id).first()
        
        if ml_model is None:
            raise ValueError(f"Model {model_id} not found")
        
        if ml_model.status != from_status:
            raise ValueError(f"Model status is {ml_model.status}, expected {from_status}")
        
        ml_model.status = to_status
        
        if to_status == "deployed":
            ml_model.deployed_at = datetime.utcnow()
        
        self.db.commit()
        
        logger.info(f"Promoted model {model_id}: {from_status} -> {to_status}")


class IntegrationHelper:
    """
    Helper functions for integrating ML with other sessions
    """
    
    @staticmethod
    def integrate_with_spc(model: Any, spc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Integrate ML predictions with SPC monitoring
        
        Args:
            model: Trained ML model
            spc_data: SPC data with control limits
        
        Returns:
            Integration results
        """
        
        # Make predictions
        features = spc_data.drop(['timestamp', 'value'], axis=1, errors='ignore')
        predictions = model.predict(features.values)
        
        # Check against control limits
        ucl = spc_data.get('ucl', predictions.mean() + 3 * predictions.std())
        lcl = spc_data.get('lcl', predictions.mean() - 3 * predictions.std())
        
        violations = (predictions > ucl) | (predictions < lcl)
        
        return {
            "predictions": predictions.tolist(),
            "violations": violations.tolist(),
            "n_violations": int(violations.sum()),
            "violation_rate": float(violations.mean())
        }
    
    @staticmethod
    def integrate_with_electrical(
        vm_model: Any,
        electrical_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Integrate VM with electrical measurements
        
        Predict electrical properties from process parameters
        """
        
        # Convert params to feature vector
        feature_names = ['temperature', 'pressure', 'flow_rate', 'power', 'time']
        features = np.array([[electrical_params.get(f, 0) for f in feature_names]])
        
        # Predict
        prediction = vm_model.predict(features)[0]
        
        # Get uncertainty if available
        uncertainty = None
        if hasattr(vm_model, 'predict_std'):
            uncertainty = vm_model.predict_std(features)[0]
        
        return {
            "predicted_thickness": float(prediction),
            "uncertainty": float(uncertainty) if uncertainty else None,
            "input_params": electrical_params
        }


# Export all classes
__all__ = [
    'TimeSeriesDecomposer',
    'ModelCompressor',
    'OnlineLearner',
    'CausalAnalyzer',
    'MultiObjectiveOptimizer',
    'ModelRegistry',
    'IntegrationHelper'
]
