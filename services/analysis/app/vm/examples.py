"""
Examples for VM Module

Demonstrates VM model training, evaluation, and ONNX export.
"""

import numpy as np
import logging
from pathlib import Path

from .models import FilmFamily, PredictionTarget, create_vm_model
from .training import VMTrainer, TrainingConfig, train_vm_model
from .evaluation import evaluate_vm_model
from .onnx_export import export_to_onnx, export_model_package, ONNXPredictor

# Import physics models for feature extraction
import sys
sys.path.append(str(Path(__file__).parent.parent))

from drivers.physics_models.thickness import DepositionParameters
from drivers.physics_models.stress import ProcessConditions
from drivers.physics_models.adhesion import AdhesionFactors
from drivers.physics_models.vm_features import VMFeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Helper: Generate Synthetic CVD Dataset
# =============================================================================

def generate_synthetic_cvd_dataset(
    n_samples: int = 200,
    film_family: FilmFamily = FilmFamily.SI3N4,
) -> tuple:
    """
    Generate synthetic CVD dataset for demonstration

    Returns:
        (X, y_thickness, y_stress, y_adhesion, feature_names)
    """
    np.random.seed(42)

    # Generate random process parameters
    temperatures = np.random.uniform(750, 830, n_samples)
    pressures = np.random.uniform(0.2, 0.5, n_samples)
    precursor_flows = np.random.uniform(60, 100, n_samples)
    carrier_flows = np.random.uniform(400, 600, n_samples)
    rf_powers = np.random.uniform(0, 150, n_samples) if film_family != FilmFamily.SI3N4 else np.zeros(n_samples)

    # Extract VM features using physics models
    extractor = VMFeatureExtractor()
    feature_list = []
    thickness_list = []
    stress_list = []
    adhesion_list = []

    for i in range(n_samples):
        # Create deposition parameters
        dep_params = DepositionParameters(
            temperature_c=temperatures[i],
            pressure_torr=pressures[i],
            precursor_flow_sccm=precursor_flows[i],
            carrier_gas_flow_sccm=carrier_flows[i],
            rf_power_w=rf_powers[i],
            film_material=film_family.value,
        )

        proc_cond = ProcessConditions(
            temperature_c=temperatures[i],
            pressure_torr=pressures[i],
            deposition_rate_nm_min=50.0,
            rf_power_w=rf_powers[i],
        )

        adhes_fac = AdhesionFactors(
            film_stress_mpa=-250.0,
            pre_clean_quality=0.95,
            deposition_temp_c=temperatures[i],
        )

        # Extract features
        features = extractor.extract_all_features(
            deposition_params=dep_params,
            process_conditions=proc_cond,
            adhesion_factors=adhes_fac,
        )

        feature_list.append(features)

        # Simulate measurements (physics-based + noise)
        # Thickness: roughly linear with temperature and precursor flow
        thickness = 90 + 0.3 * temperatures[i] + 0.1 * precursor_flows[i] + np.random.normal(0, 3)
        thickness_list.append(thickness)

        # Stress: more compressive at lower temperatures
        stress = -280 + 0.4 * temperatures[i] - 0.05 * pressures[i] * 100 + np.random.normal(0, 15)
        stress_list.append(stress)

        # Adhesion: good at moderate temperatures
        temp_factor = 85 - abs(temperatures[i] - 790) * 0.1
        adhesion = max(50, min(100, temp_factor + np.random.normal(0, 3)))
        adhesion_list.append(adhesion)

    # Convert to arrays
    feature_names = list(feature_list[0].keys())
    X = np.array([[f[name] for name in feature_names] for f in feature_list])

    y_thickness = np.array(thickness_list)
    y_stress = np.array(stress_list)
    y_adhesion = np.array(adhesion_list)

    return X, y_thickness, y_stress, y_adhesion, feature_names


# =============================================================================
# Example 1: Train VM Model for Thickness
# =============================================================================

def example_train_thickness_model():
    """
    Train VM model to predict film thickness
    """
    logger.info("=" * 70)
    logger.info("Example 1: Train VM Model for Thickness")
    logger.info("=" * 70)

    # Generate dataset
    X, y_thickness, _, _, feature_names = generate_synthetic_cvd_dataset(
        n_samples=200,
        film_family=FilmFamily.SI3N4,
    )

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Train model
    config = TrainingConfig(
        test_size=0.2,
        val_size=0.1,
        use_cv=True,
        cv_folds=5,
        model_type="random_forest",
        verbose=True,
    )

    trainer = VMTrainer(config)

    model = trainer.train(
        X=X,
        y=y_thickness,
        feature_names=feature_names,
        film_family=FilmFamily.SI3N4,
        target=PredictionTarget.THICKNESS_MEAN,
    )

    return model


# =============================================================================
# Example 2: Train Multiple Targets
# =============================================================================

def example_train_multiple_targets():
    """
    Train VM models for all targets (thickness, stress, adhesion)
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Train Models for Multiple Targets")
    logger.info("=" * 70)

    # Generate dataset
    X, y_thickness, y_stress, y_adhesion, feature_names = generate_synthetic_cvd_dataset(
        n_samples=200,
        film_family=FilmFamily.SI3N4,
    )

    # Train models for all targets
    config = TrainingConfig(
        test_size=0.2,
        val_size=0.1,
        use_cv=True,
        cv_folds=5,
        model_type="random_forest",
        verbose=True,
    )

    trainer = VMTrainer(config)

    y_dict = {
        PredictionTarget.THICKNESS_MEAN: y_thickness,
        PredictionTarget.STRESS_MEAN: y_stress,
        PredictionTarget.ADHESION_SCORE: y_adhesion,
    }

    models = trainer.train_multiple_targets(
        X=X,
        y_dict=y_dict,
        feature_names=feature_names,
        film_family=FilmFamily.SI3N4,
    )

    logger.info(f"\n{'='*70}")
    logger.info("Training Summary:")
    logger.info(f"{'='*70}")

    for target, model in models.items():
        logger.info(f"\n{target.value}:")
        logger.info(f"  Train R²: {model.train_score:.4f}")
        logger.info(f"  Val R²: {model.val_score:.4f}")
        logger.info(f"  Test R²: {model.test_score:.4f}")

    return models


# =============================================================================
# Example 3: Model Evaluation with Diagnostics
# =============================================================================

def example_model_evaluation():
    """
    Detailed model evaluation with diagnostics
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Model Evaluation with Diagnostics")
    logger.info("=" * 70)

    # Generate dataset
    X, y_thickness, _, _, feature_names = generate_synthetic_cvd_dataset(
        n_samples=200,
        film_family=FilmFamily.SI3N4,
    )

    # Train model
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_thickness,
        test_size=0.2,
        random_state=42,
    )

    model = create_vm_model(
        film_family=FilmFamily.SI3N4,
        target=PredictionTarget.THICKNESS_MEAN,
        model_type="random_forest",
    )

    model.fit(X_train, y_train, feature_names)

    # Evaluate with spec limits
    spec_limits = {
        "lower": 95.0,  # 95 nm
        "upper": 105.0,  # 105 nm
    }

    metrics = evaluate_vm_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        spec_limits=spec_limits,
        verbose=True,
    )

    # Check prediction uncertainty
    from .evaluation import VMEvaluator

    evaluator = VMEvaluator(model)

    uncertainty_metrics = evaluator.check_prediction_uncertainty(
        X_test, y_test,
        confidence_level=0.95,
    )

    logger.info("\nPrediction Uncertainty:")
    logger.info(f"  Coverage: {uncertainty_metrics['coverage']:.2%} (expected: {uncertainty_metrics['expected_coverage']:.0%})")
    logger.info(f"  Avg interval width: {uncertainty_metrics['avg_interval_width']:.2f} nm")
    logger.info(f"  Avg uncertainty: {uncertainty_metrics['avg_uncertainty']:.2f} nm")

    return model, metrics


# =============================================================================
# Example 4: ONNX Export
# =============================================================================

def example_onnx_export():
    """
    Export trained model to ONNX format
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: ONNX Model Export")
    logger.info("=" * 70)

    # Generate dataset and train model
    X, y_thickness, _, _, feature_names = generate_synthetic_cvd_dataset(
        n_samples=200,
        film_family=FilmFamily.SI3N4,
    )

    model = train_vm_model(
        X=X,
        y=y_thickness,
        feature_names=feature_names,
        film_family=FilmFamily.SI3N4,
        target=PredictionTarget.THICKNESS_MEAN,
        config=TrainingConfig(verbose=False),
    )

    # Export to ONNX
    output_dir = "/tmp/vm_models"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    success = export_model_package(
        model=model,
        output_dir=output_dir,
        model_name="Si3N4_thickness",
    )

    if success:
        logger.info("\nModel package exported successfully!")

        # Test ONNX inference (if ONNX runtime available)
        try:
            onnx_path = Path(output_dir) / "Si3N4_thickness.onnx"

            predictor = ONNXPredictor(str(onnx_path))

            # Compare predictions
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_thickness,
                test_size=0.2,
                random_state=42,
            )

            sklearn_pred = model.predict(X_test[:5])
            onnx_pred = predictor.predict(X_test[:5])

            logger.info("\nPrediction Comparison (first 5 samples):")
            logger.info("  sklearn | ONNX | difference")
            for i in range(5):
                diff = sklearn_pred[i] - onnx_pred[i]
                logger.info(f"  {sklearn_pred[i]:6.2f} | {onnx_pred[i]:6.2f} | {diff:+.4f}")

        except Exception as e:
            logger.warning(f"ONNX inference test skipped: {e}")


# =============================================================================
# Example 5: Different Film Families
# =============================================================================

def example_different_film_families():
    """
    Train models for different film families
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: Models for Different Film Families")
    logger.info("=" * 70)

    film_families = [
        FilmFamily.SI3N4,
        FilmFamily.SIO2,
        FilmFamily.TIN,
    ]

    models = {}

    for film_family in film_families:
        logger.info(f"\n{'-'*70}")
        logger.info(f"Training model for: {film_family.value}")
        logger.info(f"{'-'*70}")

        # Generate dataset for this film family
        X, y_thickness, _, _, feature_names = generate_synthetic_cvd_dataset(
            n_samples=150,
            film_family=film_family,
        )

        # Train
        model = train_vm_model(
            X=X,
            y=y_thickness,
            feature_names=feature_names,
            film_family=film_family,
            target=PredictionTarget.THICKNESS_MEAN,
            config=TrainingConfig(verbose=False),
        )

        models[film_family] = model

        logger.info(f"  Test R²: {model.test_score:.4f}")

    logger.info(f"\n{'='*70}")
    logger.info("Summary: Models Trained for All Film Families")
    logger.info(f"{'='*70}")

    for film_family, model in models.items():
        logger.info(f"  {film_family.value}: R² = {model.test_score:.4f}")

    return models


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run all VM examples"""
    logger.info("\n" + "=" * 70)
    logger.info("VM Module - Examples")
    logger.info("=" * 70)

    example_train_thickness_model()
    example_train_multiple_targets()
    example_model_evaluation()
    example_onnx_export()
    example_different_film_families()

    logger.info("\n" + "=" * 70)
    logger.info("All VM examples completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
