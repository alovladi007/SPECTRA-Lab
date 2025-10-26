"""
SESSION 14: ML/VM HUB - COMPREHENSIVE TEST SUITE

Complete integration and unit tests for ML/VM platform.

Test Coverage:
- Feature engineering
- Virtual metrology models
- Anomaly detection
- Drift detection
- Time series forecasting
- ML pipeline orchestration
- API endpoints
- Performance benchmarks

Author: Semiconductor Lab Platform Team
Date: October 2024
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from session14_vm_ml_complete_implementation import (
    FeatureEngineer, FeatureEngineeringConfig,
    VirtualMetrologyModel, VMModelConfig,
    AnomalyDetector, AnomalyDetectorConfig,
    DriftDetector, DriftDetectorConfig,
    TimeSeriesForecaster, TimeSeriesForecastConfig,
    MLPipeline,
    ModelAlgorithm, ModelType
)

# ==========================================================================
# FIXTURES
# ==========================================================================

@pytest.fixture
def sample_timeseries_data():
    """Generate sample time series data"""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    values = 100 + 10 * np.sin(np.arange(100) * 0.1) + np.random.normal(0, 2, 100)
    return pd.DataFrame({
        'timestamp': dates,
        'value': values,
        'temperature': 25 + np.random.normal(0, 1, 100),
        'pressure': 1000 + np.random.normal(0, 10, 100)
    })

@pytest.fixture
def sample_features():
    """Generate sample feature data"""
    np.random.seed(42)
    return pd.DataFrame({
        'thickness': np.random.normal(100, 5, 200),
        'temperature': np.random.normal(300, 10, 200),
        'pressure': np.random.normal(100, 2, 200),
        'flow_rate': np.random.normal(50, 3, 200),
        'power': np.random.normal(1000, 50, 200)
    })

@pytest.fixture
def sample_vm_data():
    """Generate sample VM training data"""
    np.random.seed(42)
    n = 500
    X = pd.DataFrame({
        'temperature': np.random.normal(300, 20, n),
        'pressure': np.random.normal(100, 10, n),
        'flow_rate': np.random.normal(50, 5, n),
        'power': np.random.normal(1000, 100, n),
        'time': np.random.normal(60, 10, n)
    })
    # Target: thickness is a function of process parameters + noise
    y = pd.Series(
        50 + 0.1 * X['temperature'] + 0.05 * X['pressure'] + 
        0.02 * X['power'] + np.random.normal(0, 2, n),
        name='thickness'
    )
    return X, y

@pytest.fixture
def sample_anomaly_data():
    """Generate sample data with anomalies"""
    np.random.seed(42)
    # Normal data
    normal = np.random.multivariate_normal(
        mean=[100, 50, 25],
        cov=[[25, 0, 0], [0, 9, 0], [0, 0, 4]],
        size=900
    )
    # Anomalies
    anomalies = np.random.multivariate_normal(
        mean=[150, 70, 35],
        cov=[[100, 0, 0], [0, 25, 0], [0, 0, 16]],
        size=100
    )
    
    data = np.vstack([normal, anomalies])
    np.random.shuffle(data)
    
    return pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3'])

# ==========================================================================
# FEATURE ENGINEERING TESTS
# ==========================================================================

class TestFeatureEngineering:
    """Test feature engineering capabilities"""
    
    def test_basic_feature_engineering(self, sample_timeseries_data):
        """Test basic feature engineering"""
        engineer = FeatureEngineer()
        result = engineer.engineer_features(sample_timeseries_data, target_col='value')
        
        # Check that new features were created
        assert result.shape[1] > sample_timeseries_data.shape[1]
        
        # Check that no NaN in final result
        assert not result.isnull().any().any()
    
    def test_rolling_features(self, sample_timeseries_data):
        """Test rolling window statistics"""
        config = FeatureEngineeringConfig(
            compute_rolling_stats=True,
            rolling_windows=[5, 10],
            compute_differences=False,
            compute_ratios=False
        )
        engineer = FeatureEngineer(config)
        result = engineer.engineer_features(sample_timeseries_data)
        
        # Check for rolling features
        rolling_cols = [col for col in result.columns if 'roll' in col]
        assert len(rolling_cols) > 0
    
    def test_difference_features(self, sample_timeseries_data):
        """Test difference and rate-of-change features"""
        config = FeatureEngineeringConfig(
            compute_rolling_stats=False,
            compute_differences=True
        )
        engineer = FeatureEngineer(config)
        result = engineer.engineer_features(sample_timeseries_data)
        
        # Check for difference features
        diff_cols = [col for col in result.columns if 'diff' in col or 'pct_change' in col]
        assert len(diff_cols) > 0
    
    def test_temporal_features(self, sample_timeseries_data):
        """Test temporal feature extraction"""
        config = FeatureEngineeringConfig(include_temporal=True)
        engineer = FeatureEngineer(config)
        result = engineer.engineer_features(sample_timeseries_data)
        
        # Check for temporal features
        assert 'hour' in result.columns
        assert 'hour_sin' in result.columns
        assert 'hour_cos' in result.columns
    
    def test_feature_importance_report(self):
        """Test feature importance reporting"""
        engineer = FeatureEngineer()
        
        feature_importance = {
            'temperature_roll_mean_5': 0.25,
            'pressure_diff': 0.20,
            'flow_rate': 0.15,
            'power_roll_std_10': 0.12,
            'temperature': 0.10
        }
        
        report = engineer.get_feature_importance_report(feature_importance, top_n=5)
        
        assert 'top_features' in report
        assert len(report['top_features']) == 5
        assert 'feature_types' in report

# ==========================================================================
# VIRTUAL METROLOGY TESTS
# ==========================================================================

class TestVirtualMetrology:
    """Test virtual metrology model"""
    
    def test_model_training(self, sample_vm_data):
        """Test VM model training"""
        X, y = sample_vm_data
        
        config = VMModelConfig(
            algorithm=ModelAlgorithm.RANDOM_FOREST,
            n_estimators=50,
            random_state=42
        )
        model = VirtualMetrologyModel(config)
        
        results = model.train(X, y)
        
        # Check results structure
        assert 'metrics' in results
        assert 'train' in results['metrics']
        assert 'test' in results['metrics']
        assert 'cv' in results['metrics']
        
        # Check R² is reasonable
        assert results['metrics']['test']['r2'] > 0.8
    
    def test_prediction(self, sample_vm_data):
        """Test prediction with trained model"""
        X, y = sample_vm_data
        
        model = VirtualMetrologyModel(VMModelConfig(random_state=42))
        model.train(X, y)
        
        # Make predictions
        predictions = model.predict(X.iloc[:10])
        
        assert len(predictions) == 10
        assert not np.isnan(predictions).any()
    
    def test_prediction_uncertainty(self, sample_vm_data):
        """Test prediction with uncertainty estimates"""
        X, y = sample_vm_data
        
        model = VirtualMetrologyModel(VMModelConfig(random_state=42))
        model.train(X, y)
        
        predictions, uncertainties = model.predict(X.iloc[:10], return_uncertainty=True)
        
        assert len(predictions) == len(uncertainties)
        assert all(uncertainties > 0)
    
    def test_feature_importance(self, sample_vm_data):
        """Test feature importance extraction"""
        X, y = sample_vm_data
        
        model = VirtualMetrologyModel(VMModelConfig(random_state=42))
        model.train(X, y)
        
        assert len(model.feature_importance) == X.shape[1]
        assert sum(model.feature_importance.values()) > 0
    
    def test_model_save_load(self, sample_vm_data, tmp_path):
        """Test model serialization"""
        X, y = sample_vm_data
        
        model = VirtualMetrologyModel(VMModelConfig(random_state=42))
        model.train(X, y)
        
        # Save
        model_path = tmp_path / "test_model.joblib"
        model.save(str(model_path))
        
        # Load
        loaded_model = VirtualMetrologyModel.load(str(model_path))
        
        # Compare predictions
        orig_pred = model.predict(X.iloc[:10])
        loaded_pred = loaded_model.predict(X.iloc[:10])
        
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred, decimal=6)
    
    def test_different_algorithms(self, sample_vm_data):
        """Test different ML algorithms"""
        X, y = sample_vm_data
        
        algorithms = [
            ModelAlgorithm.RANDOM_FOREST,
            ModelAlgorithm.GRADIENT_BOOSTING
        ]
        
        for algo in algorithms:
            config = VMModelConfig(algorithm=algo, n_estimators=20, random_state=42)
            model = VirtualMetrologyModel(config)
            results = model.train(X, y)
            
            # All should achieve reasonable performance
            assert results['metrics']['test']['r2'] > 0.7

# ==========================================================================
# ANOMALY DETECTION TESTS
# ==========================================================================

class TestAnomalyDetection:
    """Test anomaly detection"""
    
    def test_isolation_forest(self, sample_anomaly_data):
        """Test Isolation Forest anomaly detection"""
        # Train on normal data (first 900 samples)
        normal_data = sample_anomaly_data.iloc[:900]
        
        config = AnomalyDetectorConfig(
            algorithm=ModelAlgorithm.ISOLATION_FOREST,
            contamination=0.1,
            random_state=42
        )
        detector = AnomalyDetector(config)
        detector.fit(normal_data)
        
        # Detect on all data
        predictions, scores = detector.predict(sample_anomaly_data)
        
        # Should detect most anomalies in last 100 samples
        anomaly_indices = np.where(predictions == -1)[0]
        assert len(anomaly_indices) > 0
    
    def test_pca_anomaly(self, sample_anomaly_data):
        """Test PCA-based anomaly detection"""
        normal_data = sample_anomaly_data.iloc[:900]
        
        config = AnomalyDetectorConfig(
            algorithm=ModelAlgorithm.PCA_ANOMALY,
            n_components=2,
            threshold_percentile=95,
            random_state=42
        )
        detector = AnomalyDetector(config)
        detector.fit(normal_data)
        
        predictions, scores = detector.predict(sample_anomaly_data)
        
        assert len(predictions) == len(sample_anomaly_data)
        assert len(scores) == len(sample_anomaly_data)
    
    def test_anomaly_explanation(self, sample_anomaly_data):
        """Test anomaly explanation"""
        normal_data = sample_anomaly_data.iloc[:900]
        
        detector = AnomalyDetector(AnomalyDetectorConfig(random_state=42))
        detector.fit(normal_data, feature_names=list(sample_anomaly_data.columns))
        
        # Get prediction
        predictions, _ = detector.predict(sample_anomaly_data)
        
        # Find first anomaly
        anomaly_idx = np.where(predictions == -1)[0][0]
        
        # Explain it
        explanation = detector.explain_anomaly(sample_anomaly_data, anomaly_idx)
        
        assert 'feature_contributions' in explanation
        assert 'top_anomalous_features' in explanation
        assert len(explanation['top_anomalous_features']) > 0
    
    def test_detector_save_load(self, sample_anomaly_data, tmp_path):
        """Test detector serialization"""
        normal_data = sample_anomaly_data.iloc[:900]
        
        detector = AnomalyDetector(AnomalyDetectorConfig(random_state=42))
        detector.fit(normal_data)
        
        # Save
        detector_path = tmp_path / "test_detector.joblib"
        detector.save(str(detector_path))
        
        # Load
        loaded_detector = AnomalyDetector.load(str(detector_path))
        
        # Compare predictions
        orig_pred, _ = detector.predict(sample_anomaly_data)
        loaded_pred, _ = loaded_detector.predict(sample_anomaly_data)
        
        np.testing.assert_array_equal(orig_pred, loaded_pred)

# ==========================================================================
# DRIFT DETECTION TESTS
# ==========================================================================

class TestDriftDetection:
    """Test drift detection"""
    
    def test_feature_drift_detection(self):
        """Test feature drift detection"""
        # Reference data
        np.random.seed(42)
        reference = np.random.normal(100, 10, (1000, 5))
        
        # Current data with drift
        current = np.random.normal(110, 10, (100, 5))  # Mean shift
        
        detector = DriftDetector(DriftDetectorConfig())
        detector.set_reference(reference)
        
        result = detector.detect_drift(current)
        
        assert 'drift_detected' in result
        assert 'drift_score' in result
        assert 'feature_drifts' in result
    
    def test_prediction_drift(self):
        """Test prediction drift detection"""
        # Reference predictions
        ref_predictions = np.random.normal(100, 5, 1000)
        
        # Current predictions with drift
        curr_predictions = np.random.normal(105, 5, 100)
        
        detector = DriftDetector(DriftDetectorConfig())
        detector.set_reference(np.zeros((1000, 1)), predictions=ref_predictions)
        
        result = detector.detect_drift(np.zeros((100, 1)), predictions=curr_predictions)
        
        assert 'prediction_drift' in result
    
    def test_psi_calculation(self):
        """Test PSI calculation"""
        reference = np.random.normal(100, 10, 1000)
        
        # No drift
        current_no_drift = np.random.normal(100, 10, 100)
        
        # With drift
        current_with_drift = np.random.normal(120, 10, 100)
        
        detector = DriftDetector(DriftDetectorConfig())
        
        psi_no_drift = detector._calculate_psi(reference, current_no_drift)
        psi_with_drift = detector._calculate_psi(reference, current_with_drift)
        
        # PSI should be higher with drift
        assert psi_with_drift > psi_no_drift

# ==========================================================================
# TIME SERIES FORECASTING TESTS
# ==========================================================================

class TestTimeSeriesForecasting:
    """Test time series forecasting"""
    
    def test_prophet_training(self, sample_timeseries_data):
        """Test Prophet model training"""
        df = sample_timeseries_data[['timestamp', 'value']].copy()
        df.columns = ['ds', 'y']
        
        config = TimeSeriesForecastConfig(
            method=ModelAlgorithm.PROPHET,
            forecast_horizon=10
        )
        forecaster = TimeSeriesForecaster(config)
        forecaster.fit(df)
        
        assert forecaster.model is not None
    
    def test_forecasting(self, sample_timeseries_data):
        """Test forecast generation"""
        df = sample_timeseries_data[['timestamp', 'value']].copy()
        df.columns = ['ds', 'y']
        
        forecaster = TimeSeriesForecaster(TimeSeriesForecastConfig(forecast_horizon=10))
        forecaster.fit(df)
        
        forecast = forecaster.forecast()
        
        assert len(forecast) > len(df)  # Should include historical + future
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns
    
    def test_changepoint_detection(self, sample_timeseries_data):
        """Test changepoint detection"""
        df = sample_timeseries_data[['timestamp', 'value']].copy()
        df.columns = ['ds', 'y']
        
        forecaster = TimeSeriesForecaster()
        forecaster.fit(df)
        
        changepoints = forecaster.detect_changepoints()
        
        assert isinstance(changepoints, list)

# ==========================================================================
# ML PIPELINE TESTS
# ==========================================================================

class TestMLPipeline:
    """Test ML pipeline orchestration"""
    
    def test_vm_pipeline(self, sample_vm_data):
        """Test complete VM training pipeline"""
        X, y = sample_vm_data
        
        # Create DataFrame with target
        df = X.copy()
        df['target'] = y
        
        pipeline = MLPipeline()
        model, report = pipeline.train_vm_model(
            df,
            target_col='target',
            model_name='test_vm_model'
        )
        
        assert model is not None
        assert 'training_results' in report
        assert 'feature_engineering' in report
    
    def test_anomaly_pipeline(self, sample_anomaly_data):
        """Test anomaly detector training pipeline"""
        pipeline = MLPipeline()
        
        detector, report = pipeline.train_anomaly_detector(
            sample_anomaly_data,
            model_name='test_anomaly_detector'
        )
        
        assert detector is not None
        assert 'model_name' in report
        assert 'algorithm' in report

# ==========================================================================
# PERFORMANCE BENCHMARKS
# ==========================================================================

class TestPerformance:
    """Performance and scalability tests"""
    
    def test_feature_engineering_performance(self):
        """Test feature engineering performance on large dataset"""
        import time
        
        # Large dataset
        np.random.seed(42)
        n = 10000
        df = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n),
            'feature2': np.random.normal(0, 1, n),
            'feature3': np.random.normal(0, 1, n),
            'timestamp': pd.date_range('2024-01-01', periods=n, freq='min')
        })
        
        engineer = FeatureEngineer()
        
        start = time.time()
        result = engineer.engineer_features(df)
        elapsed = time.time() - start
        
        print(f"Feature engineering on {n} samples: {elapsed:.3f}s")
        assert elapsed < 10  # Should complete in reasonable time
    
    def test_prediction_performance(self, sample_vm_data):
        """Test prediction throughput"""
        import time
        
        X, y = sample_vm_data
        
        model = VirtualMetrologyModel(VMModelConfig(random_state=42))
        model.train(X, y)
        
        # Test batch prediction
        test_size = 1000
        test_data = X.iloc[:test_size]
        
        start = time.time()
        predictions = model.predict(test_data)
        elapsed = time.time() - start
        
        throughput = test_size / elapsed
        print(f"Prediction throughput: {throughput:.0f} samples/sec")
        assert throughput > 100  # Should be reasonably fast

# ==========================================================================
# INTEGRATION TESTS
# ==========================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_complete_vm_workflow(self, sample_vm_data):
        """Test complete VM workflow"""
        X, y = sample_vm_data
        
        # 1. Feature engineering
        df = X.copy()
        df['target'] = y
        engineer = FeatureEngineer()
        engineered = engineer.engineer_features(df, target_col='target')
        
        # 2. Train model
        X_eng = engineered.drop(columns=['target'])
        y_eng = engineered['target']
        
        model = VirtualMetrologyModel(VMModelConfig(random_state=42))
        results = model.train(X_eng, y_eng)
        
        # 3. Make predictions
        predictions = model.predict(X_eng.iloc[:10])
        
        # 4. Verify results
        assert results['metrics']['test']['r2'] > 0.7
        assert len(predictions) == 10
    
    def test_complete_anomaly_workflow(self, sample_anomaly_data):
        """Test complete anomaly detection workflow"""
        # 1. Feature engineering
        engineer = FeatureEngineer()
        engineered = engineer.engineer_features(sample_anomaly_data)
        
        # 2. Train detector
        normal_data = engineered.iloc[:900]
        detector = AnomalyDetector(AnomalyDetectorConfig(random_state=42))
        detector.fit(normal_data)
        
        # 3. Detect anomalies
        predictions, scores = detector.predict(engineered)
        
        # 4. Explain anomalies
        anomaly_idx = np.where(predictions == -1)[0][0]
        explanation = detector.explain_anomaly(engineered, anomaly_idx)
        
        # 5. Verify
        assert len(predictions) == len(engineered)
        assert 'top_anomalous_features' in explanation

# ==========================================================================
# RUN TESTS
# ==========================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
