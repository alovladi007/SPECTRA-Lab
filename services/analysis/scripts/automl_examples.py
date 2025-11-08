"""
Example: AutoML for Semiconductor Wafer Yield Prediction
Demonstrates the complete AutoML pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from automl.model_selection.auto_selector import AutoModelSelector
from automl.hyperopt.tuner import AutoHyperparameterTuner
from semiconductor.data_handler import load_semiconductor_data
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_model_selection():
    """Example 1: Automatic Model Selection"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Automatic Model Selection")
    print("="*80 + "\n")
    
    # Load data
    logger.info("Loading semiconductor wafer yield data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_semiconductor_data(
        data_type="synthetic_yield",
        test_size=0.2,
        val_size=0.1
    )
    
    # Run auto model selection
    logger.info("Running automatic model selection...")
    selector = AutoModelSelector(
        task_type="regression",
        metric="r2",
        cv_folds=5,
        prioritize_speed=False
    )
    
    results = selector.fit(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n" + "-"*80)
    print("RESULTS")
    print("-"*80)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.4f}")
    print(f"\nRecommendation: {results['recommendation']}")
    
    print("\nTop 3 Models:")
    for i, candidate in enumerate(results['all_candidates'][:3], 1):
        print(f"{i}. {candidate['name']}")
        print(f"   CV Score: {candidate['cv_score']:.4f}")
        print(f"   Test Score: {candidate['test_score']:.4f}")
        print(f"   Inference Time: {candidate['inference_time']:.3f}s")
        print()
    
    # Save the best model
    selector.save("models/best_auto_selected_model.pkl")
    
    return selector


def example_hyperparameter_tuning(model_type="RandomForest"):
    """Example 2: Hyperparameter Optimization"""
    print("\n" + "="*80)
    print(f"EXAMPLE 2: Hyperparameter Tuning for {model_type}")
    print("="*80 + "\n")
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_semiconductor_data(
        data_type="synthetic_yield"
    )
    
    # Run hyperparameter tuning
    logger.info(f"Optimizing {model_type} hyperparameters...")
    tuner = AutoHyperparameterTuner(
        model_type=model_type,
        metric="r2",
        n_trials=30,  # Use 30 trials for demo (use 50-100 for production)
        cv_folds=5,
        n_jobs=-1
    )
    
    results = tuner.optimize(X_train, y_train, X_test, y_test)
    
    # Display results
    print("\n" + "-"*80)
    print("OPTIMIZATION RESULTS")
    print("-"*80)
    print(f"Best CV Score: {results['best_cv_score']:.4f}")
    print(f"Number of Trials: {results['n_trials']}")
    
    print("\nBest Parameters:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    if 'test_metrics' in results:
        print("\nTest Set Performance:")
        for metric, value in results['test_metrics'].items():
            print(f"  {metric}: {value:.4f}")
    
    # Get parameter importance
    print("\nParameter Importance:")
    importance = tuner.get_param_importance()
    for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {imp:.4f}")
    
    # Save the optimized model
    tuner.save("models/optimized_model.pkl")
    
    return tuner


def example_complete_pipeline():
    """Example 3: Complete AutoML Pipeline"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Complete AutoML Pipeline")
    print("="*80 + "\n")
    
    # Load data
    logger.info("Step 1: Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_semiconductor_data(
        data_type="synthetic_yield"
    )
    
    # Step 1: Model Selection
    logger.info("Step 2: Auto Model Selection...")
    selector = AutoModelSelector(
        task_type="regression",
        metric="r2",
        cv_folds=5
    )
    selection_results = selector.fit(X_train, y_train, X_test, y_test)
    best_model_type = selection_results['best_model']
    
    print(f"\n✓ Best model identified: {best_model_type}")
    
    # Step 2: Hyperparameter Tuning
    logger.info(f"Step 3: Hyperparameter Tuning for {best_model_type}...")
    tuner = AutoHyperparameterTuner(
        model_type=best_model_type,
        metric="r2",
        n_trials=30,
        cv_folds=5,
        n_jobs=-1
    )
    tuning_results = tuner.optimize(X_train, y_train, X_test, y_test)
    
    print(f"\n✓ Hyperparameters optimized")
    
    # Final Results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"\nBest Model: {best_model_type}")
    print(f"Optimized CV Score: {tuning_results['best_cv_score']:.4f}")
    
    if 'test_metrics' in tuning_results:
        print(f"Test R²: {tuning_results['test_metrics']['r2']:.4f}")
        print(f"Test RMSE: {tuning_results['test_metrics']['rmse']:.4f}")
    
    print("\nModel saved to: models/optimized_model.pkl")
    print("\n✓ AutoML pipeline complete!")
    
    return selector, tuner


def example_custom_data():
    """Example 4: Using Custom Semiconductor Data"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Data Integration")
    print("="*80 + "\n")
    
    print("To use your own semiconductor manufacturing data:")
    print("\n1. Prepare your CSV file with these columns:")
    print("   - Process parameters (temperature, pressure, flow rates, etc.)")
    print("   - Target variable (yield, defect rate, etc.)")
    print("\n2. Update the configuration:")
    print("   data:")
    print("     path: 'path/to/your/data.csv'")
    print("     type: null  # Not using synthetic data")
    print("\n3. Run the pipeline:")
    print("   python src/automl/train_automl.py --config your_config.yaml")
    print("\nExample CSV structure:")
    print("-" * 60)
    print("temp_chamber,pressure,flow_ar,rf_power,yield_percent")
    print("425.3,5.2,55.1,350.5,87.3")
    print("410.8,4.9,58.3,340.2,89.1")
    print("...")
    print("-" * 60)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("AUTOML EXAMPLES FOR SEMICONDUCTOR MANUFACTURING")
    print("="*80)
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Model Selection
        selector = example_model_selection()
        
        # Example 2: Hyperparameter Tuning
        tuner = example_hyperparameter_tuning(
            model_type=selector.best_model_name
        )
        
        # Example 3: Complete Pipeline
        selector, tuner = example_complete_pipeline()
        
        # Example 4: Custom Data
        example_custom_data()
        
        print("\n" + "="*80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY! 🎉")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Example failed: {str(e)}")
        raise
