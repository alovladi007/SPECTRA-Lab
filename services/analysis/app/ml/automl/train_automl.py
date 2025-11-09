"""
AutoML Training Pipeline for Semiconductor Manufacturing
Integrates Model Selection, Hyperparameter Tuning, and Neural Architecture Search
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import yaml
import json
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

from app.ml.automl.model_selection.auto_selector import AutoModelSelector
from app.ml.automl.hyperopt.tuner import AutoHyperparameterTuner, MultiObjectiveOptimizer
from app.ml.automl.nas.architecture_search import NeuralArchitectureSearch
from app.ml.data.data_handler import load_semiconductor_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AutoMLPipeline:
    """
    Complete AutoML pipeline for semiconductor manufacturing
    
    Workflow:
    1. Auto Model Selection: Find best algorithm
    2. Hyperparameter Tuning: Optimize selected model
    3. (Optional) Neural Architecture Search: Design custom neural nets
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.results = {}
        
        # Create output directory
        self.output_dir = Path(config.get('output_dir', 'automl_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AutoML Pipeline initialized")
        logger.info(f"Results will be saved to: {self.run_dir}")
    
    def run(self):
        """Execute complete AutoML pipeline"""
        logger.info("="*80)
        logger.info("STARTING AUTOML PIPELINE FOR SEMICONDUCTOR MANUFACTURING")
        logger.info("="*80)
        
        # 1. Load data
        logger.info("\n[STEP 1] Loading semiconductor data...")
        X_train, X_val, X_test, y_train, y_val, y_test = self._load_data()
        
        # 2. Model selection
        if self.config.get('run_model_selection', True):
            logger.info("\n[STEP 2] Auto Model Selection...")
            model_selection_results = self._run_model_selection(
                X_train, y_train, X_test, y_test
            )
            self.results['model_selection'] = model_selection_results
            
            # Get best model type for hyperparameter tuning
            best_model_type = model_selection_results['best_model']
        else:
            best_model_type = self.config.get('model_type', 'RandomForest')
        
        # 3. Hyperparameter tuning
        if self.config.get('run_hyperparameter_tuning', True):
            logger.info(f"\n[STEP 3] Hyperparameter Tuning for {best_model_type}...")
            hyperopt_results = self._run_hyperparameter_tuning(
                best_model_type, X_train, y_train, X_test, y_test
            )
            self.results['hyperparameter_tuning'] = hyperopt_results
        
        # 4. Neural Architecture Search (if enabled)
        if self.config.get('run_nas', False):
            logger.info("\n[STEP 4] Neural Architecture Search...")
            nas_results = self._run_nas(X_train, y_train, X_val, y_val, X_test, y_test)
            self.results['neural_architecture_search'] = nas_results
        
        # 5. Save results
        logger.info("\n[STEP 5] Saving results...")
        self._save_results()
        
        # 6. Generate report
        logger.info("\n[STEP 6] Generating report...")
        self._generate_report()
        
        logger.info("\n" + "="*80)
        logger.info("AUTOML PIPELINE COMPLETE!")
        logger.info(f"Results saved to: {self.run_dir}")
        logger.info("="*80)
        
        return self.results
    
    def _load_data(self):
        """Load semiconductor manufacturing data"""
        data_config = self.config.get('data', {})
        
        data_path = data_config.get('path', None)
        data_type = data_config.get('type', 'synthetic_yield')
        test_size = data_config.get('test_size', 0.2)
        val_size = data_config.get('val_size', 0.1)
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_semiconductor_data(
            data_path=data_path,
            data_type=data_type,
            test_size=test_size,
            val_size=val_size
        )
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Training set: {X_train.shape}")
        logger.info(f"  Validation set: {X_val.shape}")
        logger.info(f"  Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _run_model_selection(self, X_train, y_train, X_test, y_test):
        """Run automatic model selection"""
        model_config = self.config.get('model_selection', {})
        
        selector = AutoModelSelector(
            task_type=model_config.get('task_type', 'regression'),
            metric=model_config.get('metric', 'r2'),
            cv_folds=model_config.get('cv_folds', 5),
            prioritize_speed=model_config.get('prioritize_speed', False)
        )
        
        results = selector.fit(X_train, y_train, X_test, y_test)
        
        # Save best model
        model_path = self.run_dir / 'best_model_selection.pkl'
        selector.save(str(model_path))
        
        return results
    
    def _run_hyperparameter_tuning(self, model_type, X_train, y_train, X_test, y_test):
        """Run hyperparameter optimization"""
        hyperopt_config = self.config.get('hyperparameter_tuning', {})
        
        # Check if multi-objective optimization
        if hyperopt_config.get('multi_objective', False):
            tuner = MultiObjectiveOptimizer(
                model_type=model_type,
                metrics=hyperopt_config.get('metrics', ['r2', 'inference_time']),
                n_trials=hyperopt_config.get('n_trials', 50),
                cv_folds=hyperopt_config.get('cv_folds', 5),
                n_jobs=hyperopt_config.get('n_jobs', -1)
            )
        else:
            tuner = AutoHyperparameterTuner(
                model_type=model_type,
                metric=hyperopt_config.get('metric', 'r2'),
                n_trials=hyperopt_config.get('n_trials', 50),
                cv_folds=hyperopt_config.get('cv_folds', 5),
                timeout_seconds=hyperopt_config.get('timeout_seconds', None),
                n_jobs=hyperopt_config.get('n_jobs', -1)
            )
        
        results = tuner.optimize(X_train, y_train, X_test, y_test)
        
        # Get parameter importance
        try:
            param_importance = tuner.get_param_importance()
            results['param_importance'] = {k: float(v) for k, v in param_importance.items()}
        except:
            pass
        
        # Save optimized model
        model_path = self.run_dir / f'best_model_{model_type}_optimized.pkl'
        tuner.save(str(model_path))
        
        return results
    
    def _run_nas(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Run Neural Architecture Search"""
        nas_config = self.config.get('nas', {})
        
        nas = NeuralArchitectureSearch(
            input_dim=X_train.shape[1],
            output_dim=1,
            task_type=nas_config.get('task_type', 'regression'),
            search_strategy=nas_config.get('search_strategy', 'evolutionary'),
            n_architectures=nas_config.get('n_architectures', 20),
            max_layers=nas_config.get('max_layers', 5),
            max_units_per_layer=nas_config.get('max_units_per_layer', 256)
        )
        
        results = nas.search(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        import torch
        X_test_tensor = torch.FloatTensor(X_test)
        nas.best_model.eval()
        with torch.no_grad():
            y_pred = nas.best_model(X_test_tensor).squeeze().numpy()
        
        from sklearn.metrics import r2_score, mean_squared_error
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results['test_metrics'] = {
            'r2': float(test_r2),
            'rmse': float(test_rmse)
        }
        
        logger.info(f"NAS Test R²: {test_r2:.4f}, RMSE: {test_rmse:.4f}")
        
        # Save NAS model
        model_path = self.run_dir / 'best_nas_model.pth'
        nas.save(str(model_path))
        
        return results
    
    def _save_results(self):
        """Save all results to JSON"""
        results_path = self.run_dir / 'automl_results.json'
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
    
    def _generate_report(self):
        """Generate human-readable report"""
        report_path = self.run_dir / 'automl_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("AUTOML PIPELINE REPORT - SEMICONDUCTOR MANUFACTURING\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model Selection Results
            if 'model_selection' in self.results:
                f.write("-"*80 + "\n")
                f.write("1. AUTO MODEL SELECTION\n")
                f.write("-"*80 + "\n")
                ms = self.results['model_selection']
                f.write(f"Best Model: {ms['best_model']}\n")
                f.write(f"Best Score: {ms['best_score']:.4f}\n")
                f.write(f"Recommendation: {ms['recommendation']}\n\n")
                
                f.write("All Candidates:\n")
                for cand in ms['all_candidates'][:5]:
                    f.write(f"  - {cand['name']}: CV={cand['cv_score']:.4f}, "
                           f"Test={cand['test_score']:.4f}, Time={cand['inference_time']:.3f}s\n")
                f.write("\n")
            
            # Hyperparameter Tuning Results
            if 'hyperparameter_tuning' in self.results:
                f.write("-"*80 + "\n")
                f.write("2. HYPERPARAMETER TUNING\n")
                f.write("-"*80 + "\n")
                ht = self.results['hyperparameter_tuning']
                f.write(f"Model Type: {ht['model_type']}\n")
                f.write(f"Best CV Score: {ht['best_cv_score']:.4f}\n")
                f.write(f"Number of Trials: {ht['n_trials']}\n\n")
                
                f.write("Best Parameters:\n")
                for param, value in ht['best_params'].items():
                    f.write(f"  - {param}: {value}\n")
                
                if 'test_metrics' in ht:
                    f.write("\nTest Set Performance:\n")
                    for metric, value in ht['test_metrics'].items():
                        f.write(f"  - {metric}: {value:.4f}\n")
                f.write("\n")
            
            # NAS Results
            if 'neural_architecture_search' in self.results:
                f.write("-"*80 + "\n")
                f.write("3. NEURAL ARCHITECTURE SEARCH\n")
                f.write("-"*80 + "\n")
                nas = self.results['neural_architecture_search']
                ba = nas['best_architecture']
                f.write(f"Search Strategy: {nas['search_strategy']}\n")
                f.write(f"Architectures Evaluated: {nas['architectures_evaluated']}\n\n")
                
                f.write("Best Architecture:\n")
                f.write(f"  Structure: {ba['architecture_string']}\n")
                f.write(f"  Parameters: {ba['num_params']:,}\n")
                f.write(f"  Performance: {ba['performance']:.4f}\n")
                f.write(f"  Inference Time: {ba['inference_time_ms']:.2f}ms\n\n")
                
                if 'test_metrics' in nas:
                    f.write("Test Set Performance:\n")
                    for metric, value in nas['test_metrics'].items():
                        f.write(f"  - {metric}: {value:.4f}\n")
                f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        logger.info(f"Report saved to {report_path}")
        
        # Also print summary to console
        logger.info("\n" + "="*80)
        logger.info("SUMMARY")
        logger.info("="*80)
        if 'model_selection' in self.results:
            logger.info(f"Best Model: {self.results['model_selection']['best_model']}")
        if 'hyperparameter_tuning' in self.results:
            ht = self.results['hyperparameter_tuning']
            logger.info(f"Optimized {ht['model_type']} - CV Score: {ht['best_cv_score']:.4f}")
            if 'test_metrics' in ht:
                logger.info(f"Test R²: {ht['test_metrics']['r2']:.4f}")
        if 'neural_architecture_search' in self.results:
            nas = self.results['neural_architecture_search']
            if 'test_metrics' in nas:
                logger.info(f"NAS Test R²: {nas['test_metrics']['r2']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='AutoML Pipeline for Semiconductor Manufacturing'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/automl/automl_full.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Run pipeline
    pipeline = AutoMLPipeline(config)
    results = pipeline.run()
    
    return results


if __name__ == '__main__':
    main()
