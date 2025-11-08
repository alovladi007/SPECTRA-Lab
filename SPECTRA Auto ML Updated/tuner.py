"""
Hyperparameter Tuning for Semiconductor Manufacturing Models
Automatically optimizes model parameters using Bayesian optimization
"""

import optuna
import numpy as np
from typing import Dict, Any, Optional, Callable, List
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import logging
import json
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search spaces"""
    model_type: str
    search_space: Dict[str, Any]
    fixed_params: Dict[str, Any]


class AutoHyperparameterTuner:
    """
    Automatically tunes hyperparameters for semiconductor manufacturing models
    
    Uses Optuna for Bayesian optimization with:
    - Intelligent search space definition
    - Pruning of unpromising trials
    - Multi-objective optimization support
    - Hardware-aware optimization
    """
    
    def __init__(
        self,
        model_type: str,
        metric: str = "r2",
        n_trials: int = 50,
        cv_folds: int = 5,
        timeout_seconds: Optional[int] = None,
        n_jobs: int = -1,
        pruner: Optional[str] = "median"
    ):
        self.model_type = model_type
        self.metric = metric
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.timeout_seconds = timeout_seconds
        self.n_jobs = n_jobs
        
        # Initialize Optuna study
        pruner_obj = self._get_pruner(pruner)
        direction = "maximize" if metric in ["r2", "accuracy"] else "minimize"
        
        self.study = optuna.create_study(
            direction=direction,
            pruner=pruner_obj,
            study_name=f"{model_type}_optimization"
        )
        
        self.best_params = None
        self.best_score = None
        self.best_model = None
    
    def _get_pruner(self, pruner_name: Optional[str]):
        """Get Optuna pruner"""
        if pruner_name == "median":
            return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        elif pruner_name == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif pruner_name is None:
            return optuna.pruners.NopPruner()
        else:
            return optuna.pruners.MedianPruner()
    
    def _get_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Define search space for different model types
        Optimized for semiconductor manufacturing applications
        """
        
        if self.model_type == "RandomForest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
                "random_state": 42,
                "n_jobs": self.n_jobs
            }
        
        elif self.model_type == "GradientBoosting":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": 42
            }
        
        elif self.model_type == "MLP":
            # Neural network for semiconductor process modeling
            n_layers = trial.suggest_int("n_layers", 1, 4)
            hidden_layers = []
            for i in range(n_layers):
                layer_size = trial.suggest_int(f"layer_{i}_size", 16, 256, step=16)
                hidden_layers.append(layer_size)
            
            return {
                "hidden_layer_sizes": tuple(hidden_layers),
                "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
                "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
                "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "max_iter": 1000,
                "early_stopping": True,
                "validation_fraction": 0.1,
                "random_state": 42
            }
        
        elif self.model_type == "SVR":
            return {
                "kernel": trial.suggest_categorical("kernel", ["rbf", "poly"]),
                "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            }
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _create_model(self, params: Dict[str, Any]):
        """Create model instance with given parameters"""
        if self.model_type == "RandomForest":
            return RandomForestRegressor(**params)
        elif self.model_type == "GradientBoosting":
            return GradientBoostingRegressor(**params)
        elif self.model_type == "MLP":
            return MLPRegressor(**params)
        elif self.model_type == "SVR":
            return SVR(**params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _objective(
        self, 
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> float:
        """Objective function for Optuna optimization"""
        
        # Get hyperparameters for this trial
        params = self._get_search_space(trial)
        
        # Create model
        model = self._create_model(params)
        
        # Evaluate with cross-validation
        try:
            scoring = self._get_scoring_metric()
            scores = cross_val_score(
                model, X_train, y_train,
                cv=self.cv_folds,
                scoring=scoring,
                n_jobs=1  # Optuna handles parallelization
            )
            
            # Return mean score
            score = np.mean(scores)
            
            # Report intermediate value for pruning
            trial.report(score, step=0)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return score
            
        except Exception as e:
            logger.warning(f"Trial failed: {str(e)}")
            return float('-inf') if self.metric in ["r2", "accuracy"] else float('inf')
    
    def _get_scoring_metric(self) -> str:
        """Convert metric name to sklearn scoring string"""
        metric_map = {
            "r2": "r2",
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error"
        }
        return metric_map.get(self.metric, "r2")
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features (optional)
            y_test: Test targets (optional)
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {self.model_type}")
        logger.info(f"Number of trials: {self.n_trials}")
        logger.info(f"CV folds: {self.cv_folds}")
        logger.info(f"Optimization metric: {self.metric}")
        
        # Run optimization
        self.study.optimize(
            lambda trial: self._objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            n_jobs=self.n_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"\n{'='*60}")
        logger.info(f"OPTIMIZATION COMPLETE")
        logger.info(f"Best {self.metric} score: {self.best_score:.4f}")
        logger.info(f"Best parameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"{'='*60}\n")
        
        # Train final model with best parameters
        logger.info("Training final model with optimized parameters...")
        self.best_model = self._create_model(self.best_params)
        self.best_model.fit(X_train, y_train)
        
        # Evaluate on test set if provided
        results = {
            "model_type": self.model_type,
            "best_params": self.best_params,
            "best_cv_score": float(self.best_score),
            "n_trials": len(self.study.trials),
            "optimization_history": self._get_optimization_history()
        }
        
        if X_test is not None and y_test is not None:
            test_pred = self.best_model.predict(X_test)
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            results["test_metrics"] = {
                "r2": float(r2_score(y_test, test_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
                "mae": float(mean_absolute_error(y_test, test_pred))
            }
            
            logger.info("Test set performance:")
            for metric, value in results["test_metrics"].items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials"""
        history = []
        for trial in self.study.trials:
            history.append({
                "trial_number": trial.number,
                "value": float(trial.value) if trial.value is not None else None,
                "params": trial.params,
                "state": trial.state.name
            })
        return history
    
    def get_param_importance(self) -> Dict[str, float]:
        """Calculate parameter importance using fANOVA"""
        try:
            importance = optuna.importance.get_param_importances(self.study)
            logger.info("\nParameter Importance:")
            for param, imp in importance.items():
                logger.info(f"  {param}: {imp:.4f}")
            return importance
        except Exception as e:
            logger.warning(f"Could not calculate parameter importance: {str(e)}")
            return {}
    
    def plot_optimization(self, save_path: Optional[str] = None):
        """Generate optimization visualization plots"""
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            fig1 = plot_optimization_history(self.study)
            fig2 = plot_param_importances(self.study)
            
            if save_path:
                fig1.write_image(f"{save_path}_history.png")
                fig2.write_image(f"{save_path}_importance.png")
                logger.info(f"Plots saved to {save_path}_*.png")
            else:
                fig1.show()
                fig2.show()
                
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualization.")
        except Exception as e:
            logger.warning(f"Could not generate plots: {str(e)}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the optimized model"""
        if self.best_model is None:
            raise ValueError("No model has been optimized. Call optimize() first.")
        return self.best_model.predict(X)
    
    def save(self, filepath: str):
        """Save the optimized model and results"""
        import joblib
        
        # Save model
        joblib.dump(self.best_model, filepath)
        
        # Save optimization results
        results_path = filepath.replace('.pkl', '_optimization_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                "model_type": self.model_type,
                "best_params": self.best_params,
                "best_score": float(self.best_score),
                "n_trials": len(self.study.trials)
            }, f, indent=2)
        
        logger.info(f"Optimized model saved to {filepath}")
        logger.info(f"Results saved to {results_path}")


class MultiObjectiveOptimizer(AutoHyperparameterTuner):
    """
    Multi-objective hyperparameter optimization
    Optimizes for multiple metrics simultaneously (e.g., accuracy + speed)
    """
    
    def __init__(
        self,
        model_type: str,
        metrics: List[str] = ["r2", "inference_time"],
        **kwargs
    ):
        # Don't call parent __init__ directly
        self.model_type = model_type
        self.metrics = metrics
        self.n_trials = kwargs.get("n_trials", 50)
        self.cv_folds = kwargs.get("cv_folds", 5)
        self.timeout_seconds = kwargs.get("timeout_seconds", None)
        self.n_jobs = kwargs.get("n_jobs", -1)
        
        # Create multi-objective study
        self.study = optuna.create_study(
            directions=["maximize"] * len(metrics),  # Can be customized
            study_name=f"{model_type}_multi_objective"
        )
        
        self.best_params = None
        self.best_model = None
    
    def _multi_objective(
        self,
        trial: optuna.Trial,
        X_train: np.ndarray,
        y_train: np.ndarray
    ) -> List[float]:
        """Multi-objective function"""
        import time
        
        params = self._get_search_space(trial)
        model = self._create_model(params)
        
        # Calculate accuracy
        scores = cross_val_score(model, X_train, y_train, cv=self.cv_folds, scoring="r2")
        accuracy = np.mean(scores)
        
        # Calculate inference time
        model.fit(X_train, y_train)
        start = time.time()
        _ = model.predict(X_train[:100])  # Sample for speed test
        inference_time = (time.time() - start) / 100  # Per sample
        
        return [accuracy, -inference_time]  # Negative because we want to minimize time
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Run multi-objective optimization"""
        logger.info(f"Starting multi-objective optimization: {self.metrics}")
        
        self.study.optimize(
            lambda trial: self._multi_objective(trial, X_train, y_train),
            n_trials=self.n_trials,
            timeout=self.timeout_seconds,
            n_jobs=self.n_jobs
        )
        
        # Get Pareto front
        pareto_front = self.study.best_trials
        logger.info(f"Found {len(pareto_front)} Pareto-optimal solutions")
        
        # Select balanced solution (middle of Pareto front)
        if pareto_front:
            middle_idx = len(pareto_front) // 2
            best_trial = pareto_front[middle_idx]
            self.best_params = best_trial.params
            self.best_model = self._create_model(self.best_params)
            self.best_model.fit(X_train, y_train)
        
        return {
            "model_type": self.model_type,
            "best_params": self.best_params,
            "pareto_front_size": len(pareto_front)
        }
