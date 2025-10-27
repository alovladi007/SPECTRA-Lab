"""
Session 14: ML & Virtual Metrology - Integration Tests
======================================================

Comprehensive test suite for all ML components.

Author: Semiconductor Lab Platform Team
Date: October 2025
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Import from main implementation
from session14_ml_complete_implementation import (
    VirtualMetrologyModel,
    FeatureStore,
    AnomalyDetector,
    DriftDetector,
    TimeSeriesForecaster,
    TrainingConfig,
    ModelType,
    AnomalyMethod,
    DriftMethod,
    generate_vm_training_data,
    generate_anomaly_data,
    generate_timeseries_data,
)


class TestVirtualMetrologyModel:
    """Test VM model training and prediction"""
    
    def test_model_training(self):
        """Test basic model training"""
        X, y = generate_vm_training_data(n_samples=500, n_features=10)
        
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            target_metric="thickness",
            test_size=0.2,
            cv_folds=3,
        )
        
        model = VirtualMetrologyModel(config)
        metrics = model.train(X, y)
        
        # Assertions
        assert metrics.r2 > 0.7, "R² should be > 0.7 for synthetic data"
        assert metrics.rmse < 10, "RMSE should be reasonable"
        assert metrics.training_time > 0, "Training time should be positive"
        assert metrics.n_features == 10, "Should use all features"
        
    def test_model_prediction(self):
        """Test model predictions"""
        X, y = generate_vm_training_data(n_samples=500, n_features=10)
        
        config = TrainingConfig(model_type=ModelType.RANDOM_FOREST)
        model = VirtualMetrologyModel(config)
        model.train(X, y)
        
        # Make predictions
        X_test = X.iloc[:10]
        predictions = model.predict(X_test)
        
        assert len(predictions) == 10, "Should predict for all samples"
        assert np.all(predictions > 0), "Predictions should be positive"
        
    def test_feature_importance(self):
        """Test feature importance calculation"""
        X, y = generate_vm_training_data(n_samples=500, n_features=10)
        
        config = TrainingConfig(model_type=ModelType.RANDOM_FOREST)
        model = VirtualMetrologyModel(config)
        model.train(X, y)
        
        assert model.feature_importance is not None, "Feature importance should be calculated"
        assert len(model.feature_importance.feature_names) == 10, "Should have importance for all features"
        
        top_features = model.feature_importance.get_top_features(5)
        assert len(top_features) == 5, "Should return top 5 features"
        
    def test_cross_validation(self):
        """Test cross-validation"""
        X, y = generate_vm_training_data(n_samples=500, n_features=10)
        
        config = TrainingConfig(
            model_type=ModelType.RANDOM_FOREST,
            cv_folds=5,
        )
        
        model = VirtualMetrologyModel(config)
        metrics = model.train(X, y)
        
        assert len(metrics.cv_scores) == 5, "Should have 5 CV scores"
        assert metrics.cv_std < 0.2, "CV std should be reasonable"


class TestFeatureStore:
    """Test feature engineering and storage"""
    
    def test_fdc_feature_extraction(self):
        """Test FDC feature extraction"""
        store = FeatureStore()
        
        sensor_data = {
            'temperature': np.array([350.0, 351.0, 349.5, 350.5]),
            'pressure': np.array([5.0, 5.1, 4.9, 5.0]),
            'power': np.array([500, 505, 495, 500]),
        }
        
        recipe_params = {
            'time': 300,
            'gas_flow': 100,
        }
        
        features = store.extract_fdc_features(sensor_data, recipe_params)
        
        assert 'temperature_mean' in features.columns
        assert 'pressure_std' in features.columns
        assert 'power_max' in features.columns
        assert 'recipe_time' in features.columns
        assert features.shape[0] == 1, "Should return one row"
        
    def test_feature_storage(self):
        """Test feature saving and loading"""
        store = FeatureStore(storage_path=Path("./test_features"))
        
        features = pd.DataFrame({
            'feat1': [1, 2, 3],
            'feat2': [4, 5, 6],
        })
        
        store.save_features("test_features", features)
        loaded = store.load_features("test_features")
        
        assert loaded.equals(features), "Loaded features should match saved"
        
        # Cleanup
        Path("./test_features").rmdir()


class TestAnomalyDetector:
    """Test anomaly detection"""
    
    def test_isolation_forest_detection(self):
        """Test Isolation Forest anomaly detection"""
        data = generate_anomaly_data(n_normal=900, n_anomalies=100, n_features=10)
        
        train_data = data[data['is_anomaly'] == 0].drop('is_anomaly', axis=1)
        test_data = data.drop('is_anomaly', axis=1)
        true_labels = data['is_anomaly'].values
        
        detector = AnomalyDetector(
            method=AnomalyMethod.ISOLATION_FOREST,
            contamination=0.1
        )
        detector.fit(train_data)
        
        results = detector.detect(test_data)
        predictions = [1 if r.is_anomaly else 0 for r in results]
        
        accuracy = np.mean(np.array(predictions) == true_labels)
        assert accuracy > 0.7, f"Anomaly detection accuracy should be > 70%, got {accuracy:.2%}"
        
    def test_lof_detection(self):
        """Test Local Outlier Factor detection"""
        data = generate_anomaly_data(n_normal=900, n_anomalies=100, n_features=10)
        
        train_data = data[data['is_anomaly'] == 0].drop('is_anomaly', axis=1)
        test_data = data.drop('is_anomaly', axis=1)
        
        detector = AnomalyDetector(
            method=AnomalyMethod.LOCAL_OUTLIER_FACTOR,
            contamination=0.1
        )
        detector.fit(train_data)
        
        results = detector.detect(test_data)
        assert len(results) == len(test_data), "Should detect for all samples"
        
    def test_anomaly_scores(self):
        """Test anomaly score calculation"""
        data = generate_anomaly_data(n_normal=900, n_anomalies=100, n_features=10)
        
        train_data = data[data['is_anomaly'] == 0].drop('is_anomaly', axis=1)
        test_data = data.drop('is_anomaly', axis=1)
        
        detector = AnomalyDetector(method=AnomalyMethod.ISOLATION_FOREST)
        detector.fit(train_data)
        
        results = detector.detect(test_data)
        
        # Check score properties
        for result in results:
            assert result.anomaly_score is not None
            assert 0 <= result.confidence <= 1, "Confidence should be between 0 and 1"


class TestDriftDetector:
    """Test drift detection"""
    
    def test_psi_drift_detection(self):
        """Test PSI drift detection"""
        # Reference data (no drift)
        X_ref = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f"feat_{i}" for i in range(10)]
        )
        
        # New data with drift
        X_drift = pd.DataFrame(
            np.random.randn(500, 10) + 0.5,  # Add drift
            columns=[f"feat_{i}" for i in range(10)]
        )
        
        detector = DriftDetector(method=DriftMethod.PSI, threshold=0.1)
        detector.set_reference(X_ref)
        
        result = detector.detect_drift(X_drift)
        
        assert result.drift_detected, "Should detect drift"
        assert result.drift_score > 0.1, "Drift score should exceed threshold"
        assert len(result.affected_features) > 0, "Should identify affected features"
        
    def test_ks_drift_detection(self):
        """Test Kolmogorov-Smirnov drift detection"""
        X_ref = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f"feat_{i}" for i in range(10)]
        )
        
        X_drift = pd.DataFrame(
            np.random.randn(500, 10) + 0.8,
            columns=[f"feat_{i}" for i in range(10)]
        )
        
        detector = DriftDetector(method=DriftMethod.KS_TEST)
        detector.set_reference(X_ref)
        
        result = detector.detect_drift(X_drift)
        
        assert result.p_value is not None, "Should calculate p-value"
        assert result.drift_detected, "Should detect significant drift"
        
    def test_no_drift_detection(self):
        """Test when there is no drift"""
        X_ref = pd.DataFrame(
            np.random.randn(1000, 10),
            columns=[f"feat_{i}" for i in range(10)]
        )
        
        # New data from same distribution
        X_new = pd.DataFrame(
            np.random.randn(500, 10),
            columns=[f"feat_{i}" for i in range(10)]
        )
        
        detector = DriftDetector(method=DriftMethod.PSI, threshold=0.1)
        detector.set_reference(X_ref)
        
        result = detector.detect_drift(X_new)
        
        # Should not detect drift (or very low drift)
        assert result.drift_score < 0.2, "Drift score should be low for same distribution"


class TestTimeSeriesForecaster:
    """Test time series forecasting"""
    
    def test_linear_forecast(self):
        """Test linear forecasting"""
        timestamps, values = generate_timeseries_data(n_points=365)
        
        forecaster = TimeSeriesForecaster(method="linear", horizon=30)
        forecaster.fit(timestamps[:-30], values[:-30])
        
        yhat, yhat_lower, yhat_upper = forecaster.forecast()
        
        assert len(yhat) == 30, "Should forecast 30 points"
        assert len(yhat_lower) == 30, "Should have lower bounds"
        assert len(yhat_upper) == 30, "Should have upper bounds"
        assert np.all(yhat_lower <= yhat), "Lower bounds should be below forecast"
        assert np.all(yhat <= yhat_upper), "Upper bounds should be above forecast"
        
    def test_forecast_accuracy(self):
        """Test forecast accuracy on known data"""
        timestamps, values = generate_timeseries_data(n_points=365)
        
        forecaster = TimeSeriesForecaster(method="linear", horizon=30)
        forecaster.fit(timestamps[:-30], values[:-30])
        
        yhat, _, _ = forecaster.forecast()
        
        # Calculate MAE on known future values
        actual = values[-30:]
        mae = np.mean(np.abs(yhat - actual))
        
        assert mae < 20, f"Forecast MAE should be < 20, got {mae:.2f}"


class TestIntegrationWorkflow:
    """Test complete ML workflow"""
    
    def test_full_vm_workflow(self):
        """Test complete VM workflow: train -> predict -> monitor"""
        # 1. Train model
        X_train, y_train = generate_vm_training_data(n_samples=1000, n_features=20)
        
        config = TrainingConfig(model_type=ModelType.RANDOM_FOREST)
        model = VirtualMetrologyModel(config)
        metrics = model.train(X_train, y_train)
        
        assert metrics.r2 > 0.7
        
        # 2. Make predictions
        X_test = X_train.iloc[:10]
        predictions = model.predict(X_test)
        
        assert len(predictions) == 10
        
        # 3. Setup anomaly detection
        detector = AnomalyDetector(method=AnomalyMethod.ISOLATION_FOREST)
        detector.fit(X_train)
        
        anomaly_results = detector.detect(X_test)
        assert len(anomaly_results) == 10
        
        # 4. Setup drift detection
        drift_detector = DriftDetector(method=DriftMethod.PSI)
        drift_detector.set_reference(X_train)
        
        # Simulate new data with slight drift
        X_new = X_train.iloc[-100:] * 1.1  # 10% increase
        drift_result = drift_detector.detect_drift(X_new)
        
        assert drift_result.drift_score >= 0
        
    def test_production_pipeline(self):
        """Test production pipeline with all components"""
        # Feature extraction
        store = FeatureStore()
        
        sensor_data = {
            'temperature': np.random.randn(100) * 10 + 350,
            'pressure': np.random.randn(100) * 0.5 + 5.0,
            'power': np.random.randn(100) * 50 + 500,
        }
        
        recipe_params = {'time': 300, 'gas_flow': 100}
        
        features = store.extract_fdc_features(sensor_data, recipe_params)
        
        # Generate more samples for training
        X_train, y_train = generate_vm_training_data(n_samples=1000, n_features=len(features.columns))
        
        # Train model
        config = TrainingConfig(model_type=ModelType.RANDOM_FOREST)
        model = VirtualMetrologyModel(config)
        model.train(X_train, y_train)
        
        # Ensure features match
        features_aligned = features[model.feature_names]
        
        # Make prediction
        prediction = model.predict(features_aligned)
        
        assert len(prediction) == 1
        assert prediction[0] > 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance requirements"""
    
    def test_training_time(self):
        """Test that training completes in reasonable time"""
        X, y = generate_vm_training_data(n_samples=1000, n_features=20)
        
        config = TrainingConfig(model_type=ModelType.RANDOM_FOREST)
        model = VirtualMetrologyModel(config)
        
        metrics = model.train(X, y)
        
        assert metrics.training_time < 60, "Training should complete in < 60 seconds"
        
    def test_inference_time(self):
        """Test that inference is fast enough"""
        X, y = generate_vm_training_data(n_samples=1000, n_features=20)
        
        config = TrainingConfig(model_type=ModelType.RANDOM_FOREST)
        model = VirtualMetrologyModel(config)
        model.train(X, y)
        
        metrics = model.metrics
        assert metrics.inference_time < 0.05, "Inference should be < 50ms"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
