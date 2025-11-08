"""
Auto Model Selection for Semiconductor Manufacturing
Automatically evaluates and selects the best ML algorithm for your data
"""

import numpy as np
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """Represents a candidate model with its performance metrics"""
    name: str
    model: Any
    cv_score: float
    train_score: float
    test_score: float
    inference_time: float
    complexity: int
    params: Dict[str, Any]


class AutoModelSelector:
    """
    Automatically selects the best model for semiconductor manufacturing data
    
    Evaluates multiple algorithms and selects based on:
    - Predictive performance (R², RMSE, MAE)
    - Inference speed
    - Model complexity
    - Cross-validation stability
    """
    
    def __init__(
        self,
        task_type: str = "regression",
        metric: str = "r2",
        cv_folds: int = 5,
        time_budget_seconds: int = 300,
        prioritize_speed: bool = False
    ):
        self.task_type = task_type
        self.metric = metric
        self.cv_folds = cv_folds
        self.time_budget_seconds = time_budget_seconds
        self.prioritize_speed = prioritize_speed
        
        self.candidates = []
        self.best_model = None
        self.best_model_name = None
        
    def _get_candidate_models(self) -> Dict[str, Any]:
        """Returns a dictionary of candidate models for semiconductor processes"""
        if self.task_type == "regression":
            return {
                # Tree-based models (robust to outliers, good for manufacturing data)
                "RandomForest": RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ),
                "DecisionTree": DecisionTreeRegressor(
                    max_depth=10,
                    random_state=42
                ),
                
                # Linear models (interpretable, fast)
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=1.0),
                "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5),
                
                # Non-linear models
                "SVR_RBF": SVR(kernel='rbf', C=1.0, gamma='scale'),
                "MLP_Small": MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                ),
                "MLP_Deep": MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                )
            }
        else:
            raise NotImplementedError(f"Task type {self.task_type} not yet supported")
    
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Fit and evaluate all candidate models
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            
        Returns:
            Dictionary with results and best model info
        """
        logger.info(f"Starting Auto Model Selection with {len(self._get_candidate_models())} candidates")
        logger.info(f"Training data shape: {X_train.shape}, Target shape: {y_train.shape}")
        
        models = self._get_candidate_models()
        self.candidates = []
        
        for model_name, model in models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Cross-validation score
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=self.cv_folds,
                    scoring=self._get_scoring_metric(),
                    n_jobs=-1
                )
                cv_score = np.mean(cv_scores)
                
                # Train the model
                import time
                start_time = time.time()
                model.fit(X_train, y_train)
                inference_time = time.time() - start_time
                
                # Training score
                train_pred = model.predict(X_train)
                train_score = self._calculate_score(y_train, train_pred)
                
                # Test score (if provided)
                test_score = None
                if X_test is not None and y_test is not None:
                    test_pred = model.predict(X_test)
                    test_score = self._calculate_score(y_test, test_pred)
                
                # Model complexity (number of parameters)
                complexity = self._estimate_complexity(model)
                
                candidate = ModelCandidate(
                    name=model_name,
                    model=model,
                    cv_score=cv_score,
                    train_score=train_score,
                    test_score=test_score if test_score else train_score,
                    inference_time=inference_time,
                    complexity=complexity,
                    params=model.get_params()
                )
                
                self.candidates.append(candidate)
                
                logger.info(f"  CV Score: {cv_score:.4f}, Train Score: {train_score:.4f}, " +
                           f"Test Score: {test_score:.4f if test_score else 0:.4f}, " +
                           f"Time: {inference_time:.3f}s")
                
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {str(e)}")
                continue
        
        # Select best model
        self._select_best_model()
        
        return self._get_results_summary()
    
    def _get_scoring_metric(self) -> str:
        """Convert metric name to sklearn scoring string"""
        metric_map = {
            "r2": "r2",
            "rmse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error"
        }
        return metric_map.get(self.metric, "r2")
    
    def _calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate score based on selected metric"""
        if self.metric == "r2":
            return r2_score(y_true, y_pred)
        elif self.metric == "rmse":
            return -np.sqrt(mean_squared_error(y_true, y_pred))
        elif self.metric == "mae":
            return -mean_absolute_error(y_true, y_pred)
        else:
            return r2_score(y_true, y_pred)
    
    def _estimate_complexity(self, model: Any) -> int:
        """Estimate model complexity (number of parameters)"""
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
        else:
            n_features = 0
            
        if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
            # Trees: approximate as n_estimators * max_depth * n_features
            n_estimators = getattr(model, 'n_estimators', 100)
            max_depth = getattr(model, 'max_depth', 10) or 10
            return n_estimators * max_depth * n_features
        elif isinstance(model, MLPRegressor):
            # Neural network: sum of all weights
            if hasattr(model, 'coefs_'):
                return sum(coef.size for coef in model.coefs_)
            return 10000  # Estimate
        else:
            # Linear models: number of features
            return n_features
    
    def _select_best_model(self):
        """Select the best model based on performance and constraints"""
        if not self.candidates:
            raise ValueError("No successful candidates to select from")
        
        # Score each candidate
        scored_candidates = []
        for candidate in self.candidates:
            # Composite score: weighted combination of metrics
            if self.prioritize_speed:
                # Speed is important for real-time semiconductor monitoring
                score = (
                    0.6 * candidate.cv_score +
                    0.2 * (1.0 / (1.0 + candidate.inference_time)) +
                    0.2 * (1.0 / (1.0 + np.log10(candidate.complexity + 1)))
                )
            else:
                # Prioritize accuracy
                score = (
                    0.8 * candidate.cv_score +
                    0.1 * (1.0 / (1.0 + candidate.inference_time)) +
                    0.1 * (1.0 / (1.0 + np.log10(candidate.complexity + 1)))
                )
            
            scored_candidates.append((score, candidate))
        
        # Sort by score (descending)
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        self.best_model = scored_candidates[0][1].model
        self.best_model_name = scored_candidates[0][1].name
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BEST MODEL SELECTED: {self.best_model_name}")
        logger.info(f"  CV Score: {scored_candidates[0][1].cv_score:.4f}")
        logger.info(f"  Test Score: {scored_candidates[0][1].test_score:.4f}")
        logger.info(f"  Inference Time: {scored_candidates[0][1].inference_time:.3f}s")
        logger.info(f"  Complexity: {scored_candidates[0][1].complexity} parameters")
        logger.info(f"{'='*60}\n")
    
    def _get_results_summary(self) -> Dict[str, Any]:
        """Generate a summary of all results"""
        results = {
            "best_model": self.best_model_name,
            "best_score": None,
            "all_candidates": [],
            "recommendation": self._generate_recommendation()
        }
        
        for candidate in sorted(self.candidates, key=lambda x: x.cv_score, reverse=True):
            results["all_candidates"].append({
                "name": candidate.name,
                "cv_score": float(candidate.cv_score),
                "train_score": float(candidate.train_score),
                "test_score": float(candidate.test_score) if candidate.test_score else None,
                "inference_time": float(candidate.inference_time),
                "complexity": int(candidate.complexity)
            })
            
            if candidate.name == self.best_model_name:
                results["best_score"] = float(candidate.cv_score)
        
        return results
    
    def _generate_recommendation(self) -> str:
        """Generate recommendations for the selected model"""
        if not self.best_model_name:
            return "No model selected"
        
        recommendations = {
            "RandomForest": "Excellent for semiconductor manufacturing with robust handling of outliers and non-linear relationships. Good interpretability through feature importance.",
            "GradientBoosting": "High accuracy for complex patterns in process data. May require careful tuning to avoid overfitting.",
            "Ridge": "Fast and interpretable linear model. Best for linear relationships and when inference speed is critical.",
            "MLP_Deep": "Captures complex non-linear patterns. Suitable for large datasets but requires more data and tuning.",
            "SVR_RBF": "Good for non-linear relationships with moderate dataset sizes. Can be slow on large datasets."
        }
        
        return recommendations.get(self.best_model_name, "Model selected based on cross-validation performance")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("No model has been fitted. Call fit() first.")
        return self.best_model.predict(X)
    
    def save(self, filepath: str):
        """Save the best model and selection results"""
        import joblib
        results = self._get_results_summary()
        
        # Save model
        joblib.dump(self.best_model, filepath)
        
        # Save metadata
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Model saved to {filepath}")
        logger.info(f"Metadata saved to {metadata_path}")
