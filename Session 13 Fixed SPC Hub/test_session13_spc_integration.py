"""
Session 13: SPC Hub - Integration Tests
Semiconductor Lab Platform

Comprehensive test suite for SPC analysis:
- Control chart calculations
- Rule violation detection
- Process capability analysis
- Alert generation and triage
- Performance benchmarks

Author: Semiconductor Lab Platform Team
Version: 1.0.0
Date: October 2025
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List

# Import the SPC module (adjust path as needed)
import sys
sys.path.insert(0, '../services/analysis/app/methods/spc/')

from session13_spc_complete_implementation import (
    DataPoint, XbarRChart, EWMAChart, CUSUMChart,
    CapabilityAnalysis, SPCManager, ChartType,
    generate_in_control_data, generate_shift_data, generate_trend_data
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def in_control_data():
    """Generate in-control process data"""
    return generate_in_control_data(n_points=50, mean=100.0, sigma=2.0)


@pytest.fixture
def shift_data():
    """Generate data with mean shift"""
    return generate_shift_data(
        n_points=50,
        mean=100.0,
        sigma=2.0,
        shift_point=25,
        shift_amount=4.0
    )


@pytest.fixture
def trend_data():
    """Generate data with linear trend"""
    return generate_trend_data(
        n_points=50,
        mean=100.0,
        sigma=2.0,
        drift_rate=0.1
    )


@pytest.fixture
def spc_manager():
    """Create SPC manager instance"""
    return SPCManager()


# ============================================================================
# Test X-bar/R Chart
# ============================================================================

class TestXbarRChart:
    """Test suite for X-bar and R control charts"""
    
    def test_control_limit_calculation(self, in_control_data):
        """Test that control limits are calculated correctly"""
        chart = XbarRChart(subgroup_size=5)
        xbar_limits, r_limits = chart.compute_control_limits(in_control_data)
        
        # X-bar limits should be symmetric around mean
        values = [p.value for p in in_control_data]
        expected_mean = np.mean(values)
        
        assert abs(xbar_limits.cl - expected_mean) < 0.5, "Centerline should be close to mean"
        assert xbar_limits.ucl > xbar_limits.cl > xbar_limits.lcl, "Control limits should be ordered"
        assert xbar_limits.sigma > 0, "Sigma should be positive"
        
        # R limits
        assert r_limits.ucl > r_limits.lcl >= 0, "R limits should be non-negative"
        assert r_limits.cl > 0, "Average range should be positive"
    
    def test_in_control_no_violations(self, in_control_data):
        """Test that in-control data produces minimal violations"""
        chart = XbarRChart(subgroup_size=5)
        xbar_limits, _ = chart.compute_control_limits(in_control_data)
        alerts = chart.check_rules(in_control_data, xbar_limits)
        
        # In-control data should have few or no violations (random chance < 5%)
        assert len(alerts) <= 2, f"Expected ≤2 alerts, got {len(alerts)}"
    
    def test_shift_detection(self, shift_data):
        """Test that process shift is detected"""
        chart = XbarRChart(subgroup_size=5)
        # Use first 25 points to establish control limits
        baseline = shift_data[:25]
        xbar_limits, _ = chart.compute_control_limits(baseline)
        
        # Check all data for violations
        alerts = chart.check_rules(shift_data, xbar_limits)
        
        # Should detect violations after shift point
        assert len(alerts) > 0, "Shift should be detected"
        
        # At least one alert should be after shift point
        shift_alerts = [a for a in alerts if any(
            p.run_id in [f"run_{i}" for i in range(25, 50)]
            for p in a.data_points
        )]
        assert len(shift_alerts) > 0, "Alert should occur after shift"
    
    def test_trend_detection(self, trend_data):
        """Test that trending pattern is detected"""
        chart = XbarRChart(subgroup_size=5)
        # Use early data for limits
        baseline = trend_data[:20]
        xbar_limits, _ = chart.compute_control_limits(baseline)
        
        # Check for violations
        alerts = chart.check_rules(trend_data, xbar_limits)
        
        # Should detect trend (Rule 5: 6 points trending)
        trend_alerts = [a for a in alerts if 'trend' in a.message.lower()]
        assert len(trend_alerts) > 0, "Trend should be detected"
    
    def test_rule_1_beyond_3sigma(self):
        """Test Rule 1: Point beyond 3σ"""
        chart = XbarRChart(subgroup_size=5)
        
        # Create data with outlier
        base_data = generate_in_control_data(n_points=30, mean=100.0, sigma=2.0)
        
        # Add an outlier
        outlier = DataPoint(
            timestamp=datetime.now(),
            value=115.0,  # Way beyond 3σ
            subgroup="outlier",
            run_id="run_outlier"
        )
        test_data = base_data + [outlier]
        
        xbar_limits, _ = chart.compute_control_limits(base_data)
        alerts = chart.check_rules(test_data, xbar_limits)
        
        # Should detect the outlier
        rule_1_alerts = [a for a in alerts if 'beyond 3' in a.message.lower()]
        assert len(rule_1_alerts) > 0, "Outlier should trigger Rule 1"
    
    def test_rule_4_consecutive_same_side(self):
        """Test Rule 4: 8 consecutive points on same side"""
        chart = XbarRChart(subgroup_size=5)
        
        # Create data with 8 consecutive points above centerline
        base_data = generate_in_control_data(n_points=20, mean=100.0, sigma=2.0)
        xbar_limits, _ = chart.compute_control_limits(base_data)
        
        # Add 8 points above centerline
        consecutive_data = []
        for i in range(8):
            point = DataPoint(
                timestamp=datetime.now() + timedelta(hours=i),
                value=xbar_limits.cl + 0.5,  # Slightly above centerline
                subgroup=f"consecutive_{i}",
                run_id=f"run_consecutive_{i}"
            )
            consecutive_data.append(point)
        
        test_data = base_data + consecutive_data
        alerts = chart.check_rules(test_data, xbar_limits)
        
        # Should detect consecutive points
        rule_4_alerts = [a for a in alerts if '8 consecutive' in a.message.lower()]
        assert len(rule_4_alerts) > 0, "8 consecutive points should trigger Rule 4"


# ============================================================================
# Test EWMA Chart
# ============================================================================

class TestEWMAChart:
    """Test suite for EWMA control charts"""
    
    def test_ewma_calculation(self):
        """Test EWMA value calculation"""
        chart = EWMAChart(lambda_=0.2)
        
        # Simple test data
        data = [
            DataPoint(datetime.now() + timedelta(hours=i), 100 + i, f"sg{i}", f"run{i}")
            for i in range(10)
        ]
        
        ewma_values = chart.calculate_ewma(data, initial_ewma=100.0)
        
        assert len(ewma_values) == len(data), "EWMA values should match data length"
        
        # EWMA should be smooth (less variable than raw data)
        data_std = np.std([p.value for p in data])
        ewma_std = np.std(ewma_values)
        assert ewma_std < data_std, "EWMA should be smoother than raw data"
    
    def test_ewma_sensitivity_to_small_shifts(self):
        """Test that EWMA detects small shifts better than X-bar"""
        chart = EWMAChart(lambda_=0.2)
        
        # Create data with small shift (1σ)
        data = []
        base_time = datetime.now()
        for i in range(50):
            mean = 100.0 if i < 25 else 102.0  # Small shift
            value = mean + np.random.normal(0, 2.0)
            data.append(DataPoint(
                base_time + timedelta(hours=i),
                value,
                f"sg{i}",
                f"run{i}"
            ))
        
        # Use first half for control limits
        limits = chart.compute_control_limits(data[:25])
        
        # Check for violations
        alerts = chart.check_violations(data, limits)
        
        # EWMA should detect the small shift
        assert len(alerts) > 0, "EWMA should detect small shift"


# ============================================================================
# Test CUSUM Chart
# ============================================================================

class TestCUSUMChart:
    """Test suite for CUSUM control charts"""
    
    def test_cusum_calculation(self):
        """Test CUSUM value calculation"""
        chart = CUSUMChart(k=0.5, h=5.0)
        
        data = generate_in_control_data(n_points=30, mean=100.0, sigma=2.0)
        target, sigma, h_limit = chart.compute_control_limits(data)
        
        cusum_high, cusum_low = chart.calculate_cusum(data, target, sigma)
        
        assert len(cusum_high) == len(data), "CUSUM high should match data length"
        assert len(cusum_low) == len(data), "CUSUM low should match data length"
        assert all(ch >= 0 for ch in cusum_high), "CUSUM high should be non-negative"
        assert all(cl >= 0 for cl in cusum_low), "CUSUM low should be non-negative"
    
    def test_cusum_sustained_shift_detection(self):
        """Test that CUSUM detects sustained shifts"""
        chart = CUSUMChart(k=0.5, h=5.0)
        
        # Create data with sustained shift
        data = generate_shift_data(
            n_points=50,
            mean=100.0,
            sigma=2.0,
            shift_point=25,
            shift_amount=3.0
        )
        
        target, sigma, h_limit = chart.compute_control_limits(data[:25])
        alerts = chart.check_violations(data, target, sigma, h_limit)
        
        # CUSUM should detect sustained shift
        assert len(alerts) > 0, "CUSUM should detect sustained shift"


# ============================================================================
# Test Process Capability
# ============================================================================

class TestProcessCapability:
    """Test suite for process capability analysis"""
    
    def test_capability_calculation(self):
        """Test Cp/Cpk calculation"""
        # Generate capable process (6σ)
        data = np.random.normal(100.0, 1.0, 100).tolist()
        capability = CapabilityAnalysis.calculate_capability(
            data,
            lsl=94.0,
            usl=106.0
        )
        
        assert capability.cp > 0, "Cp should be positive"
        assert capability.cpk > 0, "Cpk should be positive"
        assert capability.cpk <= capability.cp, "Cpk ≤ Cp (equality if centered)"
        assert 1.8 < capability.cp < 2.2, "Cp should be around 2.0 for 6σ process"
    
    def test_capability_interpretation(self):
        """Test capability interpretation"""
        # Excellent process
        excellent = CapabilityAnalysis.calculate_capability(
            np.random.normal(100.0, 0.5, 100).tolist(),
            lsl=94.0,
            usl=106.0
        )
        assert "Excellent" in CapabilityAnalysis.interpret_capability(excellent)
        
        # Poor process
        poor = CapabilityAnalysis.calculate_capability(
            np.random.normal(100.0, 3.0, 100).tolist(),
            lsl=94.0,
            usl=106.0
        )
        assert "Poor" in CapabilityAnalysis.interpret_capability(poor)
    
    def test_capability_with_off_center_process(self):
        """Test Cpk for off-center process"""
        # Process mean shifted to upper spec limit
        data = np.random.normal(104.0, 1.0, 100).tolist()
        capability = CapabilityAnalysis.calculate_capability(
            data,
            lsl=94.0,
            usl=106.0
        )
        
        # Cpk should be significantly less than Cp
        assert capability.cpk < capability.cp * 0.8, "Cpk should reflect off-center process"
        assert capability.cpu < capability.cpl, "CPU should be limiting factor"


# ============================================================================
# Test SPC Manager (Integration)
# ============================================================================

class TestSPCManager:
    """Integration tests for SPC Manager"""
    
    def test_analyze_in_control_process(self, spc_manager, in_control_data):
        """Test complete analysis of in-control process"""
        results = spc_manager.analyze_metric(
            metric_name="test_metric",
            data=in_control_data,
            chart_type=ChartType.XBAR_R,
            lsl=94.0,
            usl=106.0
        )
        
        assert "error" not in results, "Should not error on valid data"
        assert results["metric"] == "test_metric"
        assert results["data_count"] == len(in_control_data)
        assert "xbar_limits" in results
        assert "capability" in results
        assert "alerts" in results
        assert "statistics" in results
        
        # In-control process should have good capability
        assert results["capability"]["cpk"] > 1.0, "In-control process should be capable"
    
    def test_analyze_shifted_process(self, spc_manager, shift_data):
        """Test analysis of shifted process"""
        results = spc_manager.analyze_metric(
            metric_name="shifted_metric",
            data=shift_data,
            chart_type=ChartType.XBAR_R,
            lsl=94.0,
            usl=106.0
        )
        
        # Should detect alerts
        assert len(results["alerts"]) > 0, "Shifted process should trigger alerts"
        
        # Alerts should have proper structure
        for alert in results["alerts"]:
            assert "severity" in alert
            assert "message" in alert
            assert "suggested_actions" in alert
            assert len(alert["suggested_actions"]) > 0
    
    def test_analyze_with_ewma(self, spc_manager, trend_data):
        """Test EWMA analysis"""
        results = spc_manager.analyze_metric(
            metric_name="trending_metric",
            data=trend_data,
            chart_type=ChartType.EWMA,
            lsl=94.0,
            usl=106.0
        )
        
        assert "ewma_values" in results, "EWMA analysis should include EWMA values"
        assert len(results["ewma_values"]) == len(trend_data)
    
    def test_analyze_with_cusum(self, spc_manager, shift_data):
        """Test CUSUM analysis"""
        results = spc_manager.analyze_metric(
            metric_name="cusum_metric",
            data=shift_data,
            chart_type=ChartType.CUSUM,
            lsl=94.0,
            usl=106.0
        )
        
        assert "cusum_high" in results, "CUSUM analysis should include CUSUM+ values"
        assert "cusum_low" in results, "CUSUM analysis should include CUSUM- values"
    
    def test_insufficient_data_handling(self, spc_manager):
        """Test handling of insufficient data"""
        sparse_data = generate_in_control_data(n_points=10)
        
        results = spc_manager.analyze_metric(
            metric_name="sparse_metric",
            data=sparse_data,
            chart_type=ChartType.XBAR_R
        )
        
        assert "error" in results, "Should return error for insufficient data"
        assert "Insufficient" in results["message"]


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks for SPC analysis"""
    
    def test_xbar_performance(self, spc_manager, benchmark):
        """Benchmark X-bar/R chart analysis"""
        data = generate_in_control_data(n_points=100)
        
        def run_analysis():
            return spc_manager.analyze_metric(
                metric_name="perf_test",
                data=data,
                chart_type=ChartType.XBAR_R
            )
        
        result = benchmark(run_analysis)
        
        # Should complete in < 1 second
        assert benchmark.stats['mean'] < 1.0, "Analysis should complete in < 1s"
    
    def test_large_dataset_performance(self, spc_manager):
        """Test performance with large dataset"""
        import time
        
        large_data = generate_in_control_data(n_points=1000)
        
        start_time = time.time()
        results = spc_manager.analyze_metric(
            metric_name="large_test",
            data=large_data,
            chart_type=ChartType.XBAR_R
        )
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 5.0, f"Large dataset analysis took {elapsed_time:.2f}s (should be < 5s)"
        assert "error" not in results


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_all_identical_values(self, spc_manager):
        """Test handling of constant data"""
        constant_data = [
            DataPoint(
                datetime.now() + timedelta(hours=i),
                100.0,
                f"sg{i}",
                f"run{i}"
            )
            for i in range(30)
        ]
        
        results = spc_manager.analyze_metric(
            metric_name="constant",
            data=constant_data,
            chart_type=ChartType.XBAR_R
        )
        
        # Should handle gracefully (sigma = 0)
        assert results["capability"]["sigma"] < 0.001, "Sigma should be near zero"
    
    def test_extreme_outliers(self, spc_manager):
        """Test handling of extreme outliers"""
        data = generate_in_control_data(n_points=50, mean=100.0, sigma=2.0)
        
        # Add extreme outliers
        data.append(DataPoint(datetime.now(), 10000.0, "outlier1", "run_out1"))
        data.append(DataPoint(datetime.now(), -10000.0, "outlier2", "run_out2"))
        
        results = spc_manager.analyze_metric(
            metric_name="outlier_test",
            data=data,
            chart_type=ChartType.XBAR_R
        )
        
        # Should detect outliers
        assert len(results["alerts"]) > 0, "Extreme outliers should trigger alerts"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=session13_spc_complete_implementation"])
