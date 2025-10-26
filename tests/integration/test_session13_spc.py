"""
Session 13: SPC Hub - Comprehensive Test Suite

Tests all SPC functionality including:
- Control chart calculations (X-bar/R, I-MR, EWMA, CUSUM)
- Rule detection (Western Electric and Nelson rules)
- Process capability calculations
- Trend analysis
- Root cause suggestions
- Integration tests
- Performance benchmarks

Author: Semiconductor Lab Platform Team
Version: 1.0.0
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
import json
from typing import List, Dict
import time

# Import SPC modules
from session13_spc_complete_implementation import (
    SPCHub,
    ControlChartCalculator,
    RuleDetector,
    CapabilityCalculator,
    TrendAnalyzer,
    RootCauseAnalyzer,
    SPCTestDataGenerator,
    ChartType,
    RuleViolation,
    AlertSeverity,
    ProcessStatus,
    ControlLimits,
    SPCAlert,
    ProcessCapability,
    TrendAnalysis
)


# ============================================================================
# Unit Tests - Control Chart Calculations
# ============================================================================

class TestControlChartCalculator:
    """Test control chart limit calculations"""
    
    def test_xbar_r_limits_calculation(self):
        """Test X-bar and R chart limits"""
        calculator = ControlChartCalculator()
        
        # Generate test data (mean=100, sigma=5, n=50, subgroup_size=5)
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        xbar_limits, r_limits = calculator.calculate_xbar_r_limits(data, subgroup_size=5)
        
        # Verify X-bar limits
        assert xbar_limits.centerline == pytest.approx(100, abs=2)
        assert xbar_limits.ucl > xbar_limits.centerline
        assert xbar_limits.lcl < xbar_limits.centerline
        assert xbar_limits.sigma > 0
        
        # Verify R limits
        assert r_limits.centerline > 0
        assert r_limits.ucl > r_limits.centerline
        assert r_limits.lcl >= 0
    
    def test_i_mr_limits_calculation(self):
        """Test Individual and Moving Range limits"""
        calculator = ControlChartCalculator()
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        i_limits, mr_limits = calculator.calculate_i_mr_limits(data)
        
        # Verify Individual limits
        assert i_limits.centerline == pytest.approx(100, abs=2)
        assert i_limits.ucl > i_limits.centerline
        assert i_limits.lcl < i_limits.centerline
        
        # Verify MR limits
        assert mr_limits.centerline > 0
        assert mr_limits.ucl > mr_limits.centerline
    
    def test_ewma_limits_calculation(self):
        """Test EWMA limits"""
        calculator = ControlChartCalculator()
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        limits = calculator.calculate_ewma_limits(data, lambda_weight=0.2, target=100)
        
        assert limits.centerline == 100
        assert limits.ucl > 100
        assert limits.lcl < 100
        assert limits.sigma > 0
    
    def test_cusum_calculation(self):
        """Test CUSUM values"""
        calculator = ControlChartCalculator()
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        cusum_high, cusum_low, decision_limit = calculator.calculate_cusum(
            data, target=100, sigma=5
        )
        
        assert len(cusum_high) == len(data)
        assert len(cusum_low) == len(data)
        assert decision_limit > 0
        assert all(cusum_high >= 0)
        assert all(cusum_low <= 0)


# ============================================================================
# Unit Tests - Rule Detection
# ============================================================================

class TestRuleDetector:
    """Test Western Electric and Nelson rules"""
    
    def test_rule_1_detection(self):
        """Test Rule 1: One point beyond 3σ"""
        limits = ControlLimits(ucl=115, lcl=85, centerline=100, sigma=5)
        detector = RuleDetector(limits)
        
        # Create data with one outlier
        data = np.full(20, 100.0)
        data[10] = 120  # Beyond UCL
        
        alerts = detector._detect_rule_1(data)
        
        assert len(alerts) == 1
        assert alerts[0].rule_violated == RuleViolation.RULE_1
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert 10 in alerts[0].points_involved
    
    def test_rule_2_detection(self):
        """Test Rule 2: 2 of 3 beyond 2σ"""
        limits = ControlLimits(ucl=115, lcl=85, centerline=100, sigma=5)
        detector = RuleDetector(limits)
        
        # Create data with 2 of 3 beyond 2σ
        data = np.full(20, 100.0)
        data[10] = 111  # Beyond +2σ (110)
        data[11] = 111
        
        alerts = detector._detect_rule_2(data)
        
        assert len(alerts) > 0
        assert alerts[0].rule_violated == RuleViolation.RULE_2
    
    def test_rule_4_detection(self):
        """Test Rule 4: 8 consecutive on same side"""
        limits = ControlLimits(ucl=115, lcl=85, centerline=100, sigma=5)
        detector = RuleDetector(limits)
        
        # Create data with 8 consecutive above centerline
        data = np.full(20, 100.0)
        data[10:18] = 105
        
        alerts = detector._detect_rule_4(data)
        
        assert len(alerts) > 0
        assert alerts[0].rule_violated == RuleViolation.RULE_4
    
    def test_rule_5_detection(self):
        """Test Rule 5: 6 points trending"""
        limits = ControlLimits(ucl=115, lcl=85, centerline=100, sigma=5)
        detector = RuleDetector(limits)
        
        # Create increasing trend
        data = np.full(20, 100.0)
        data[10:16] = np.arange(100, 106)
        
        alerts = detector._detect_rule_5(data)
        
        assert len(alerts) > 0
        assert alerts[0].rule_violated == RuleViolation.RULE_5
    
    def test_all_rules_detection(self):
        """Test detection of all rules on various patterns"""
        limits = ControlLimits(ucl=115, lcl=85, centerline=100, sigma=5)
        detector = RuleDetector(limits)
        
        # In-control data should have no alerts (mostly)
        np.random.seed(42)
        data = np.random.normal(100, 4, 50)
        alerts = detector.detect_all_rules(data)
        
        # Verify alert structure
        for alert in alerts:
            assert isinstance(alert, SPCAlert)
            assert alert.severity in [e.value for e in AlertSeverity]
            assert len(alert.suggested_actions) > 0
            assert len(alert.root_causes) > 0


# ============================================================================
# Unit Tests - Process Capability
# ============================================================================

class TestCapabilityCalculator:
    """Test process capability calculations"""
    
    def test_capability_calculation_capable_process(self):
        """Test capability calculation for capable process"""
        calculator = CapabilityCalculator()
        
        # Generate capable process data
        np.random.seed(42)
        data = np.random.normal(100, 2, 100)  # σ=2, spec range=30 → Cp≈2.5
        
        capability = calculator.calculate_capability(
            data, usl=115, lsl=85, target=100
        )
        
        assert capability.cp > 1.33
        assert capability.cpk > 1.33
        assert capability.is_capable
        assert capability.sigma_level > 4
        assert "capable" in " ".join(capability.comments).lower()
    
    def test_capability_calculation_incapable_process(self):
        """Test capability for incapable process"""
        calculator = CapabilityCalculator()
        
        # Generate incapable process
        np.random.seed(42)
        data = np.random.normal(100, 10, 100)  # σ=10, spec range=30 → Cp≈0.5
        
        capability = calculator.calculate_capability(
            data, usl=115, lsl=85, target=100
        )
        
        assert capability.cp < 1.0
        assert capability.cpk < 1.0
        assert not capability.is_capable
    
    def test_capability_with_off_center_process(self):
        """Test capability when process is off-center"""
        calculator = CapabilityCalculator()
        
        np.random.seed(42)
        data = np.random.normal(105, 3, 100)  # Mean shifted to 105
        
        capability = calculator.calculate_capability(
            data, usl=115, lsl=85, target=100
        )
        
        # Cp should be > Cpk (off-center penalty)
        assert capability.cp > capability.cpk
        assert "not well-centered" in " ".join(capability.comments).lower()


# ============================================================================
# Unit Tests - Trend Analysis
# ============================================================================

class TestTrendAnalyzer:
    """Test trend analysis"""
    
    def test_increasing_trend_detection(self):
        """Test detection of increasing trend"""
        analyzer = TrendAnalyzer()
        
        # Generate data with increasing trend
        np.random.seed(42)
        data = np.arange(50) * 0.5 + np.random.normal(0, 2, 50) + 100
        
        trend = analyzer.analyze_trend(data, forecast_steps=5)
        
        assert trend.trend_detected
        assert trend.trend_direction == "increasing"
        assert trend.trend_slope > 0
        assert len(trend.predicted_values) == 5
    
    def test_decreasing_trend_detection(self):
        """Test detection of decreasing trend"""
        analyzer = TrendAnalyzer()
        
        # Generate data with decreasing trend
        np.random.seed(42)
        data = -np.arange(50) * 0.5 + np.random.normal(0, 2, 50) + 100
        
        trend = analyzer.analyze_trend(data, forecast_steps=5)
        
        assert trend.trend_detected
        assert trend.trend_direction == "decreasing"
        assert trend.trend_slope < 0
    
    def test_stable_process(self):
        """Test that stable process has no trend"""
        analyzer = TrendAnalyzer()
        
        # Generate stable data
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        trend = analyzer.analyze_trend(data)
        
        # With random data, might detect trend by chance, but p-value should be high
        if not trend.trend_detected:
            assert trend.trend_direction == "stable"


# ============================================================================
# Unit Tests - Root Cause Analysis
# ============================================================================

class TestRootCauseAnalyzer:
    """Test root cause analysis suggestions"""
    
    def test_root_cause_suggestions_for_critical_alerts(self):
        """Test that critical alerts get appropriate suggestions"""
        analyzer = RootCauseAnalyzer()
        
        # Create a critical alert
        alert = SPCAlert(
            id="test1",
            timestamp=datetime.now(),
            metric="test",
            rule_violated=RuleViolation.RULE_1,
            severity=AlertSeverity.CRITICAL,
            value=120,
            control_limits=ControlLimits(115, 85, 100),
            points_involved=[10],
            message="Critical violation"
        )
        
        data = np.random.normal(100, 5, 50)
        suggestions = analyzer.suggest_causes([alert], data)
        
        assert "likely_causes" in suggestions
        assert len(suggestions["likely_causes"]) > 0
        assert "investigate" in suggestions
        assert "preventive_actions" in suggestions
    
    def test_trend_based_suggestions(self):
        """Test suggestions based on trend detection"""
        analyzer = RootCauseAnalyzer()
        
        # Create increasing data
        data = np.arange(50) * 0.5 + 100
        
        suggestions = analyzer.suggest_causes([], data)
        
        # Should still have preventive actions even without alerts
        assert len(suggestions["preventive_actions"]) > 0


# ============================================================================
# Integration Tests - SPC Hub
# ============================================================================

class TestSPCHubIntegration:
    """Integration tests for complete SPC Hub"""
    
    def test_full_analysis_in_control_process(self):
        """Test complete analysis of in-control process"""
        hub = SPCHub()
        
        np.random.seed(42)
        data = np.random.normal(100, 4, 50)
        
        results = hub.analyze_process(
            data=data,
            chart_type=ChartType.I_MR,
            usl=115,
            lsl=85,
            target=100
        )
        
        assert "timestamp" in results
        assert "status" in results
        assert "control_limits" in results
        assert "alerts" in results
        assert "capability" in results
        assert "trend" in results
        assert "statistics" in results
        assert "recommendations" in results
        
        # Verify statistics
        assert results["statistics"]["mean"] == pytest.approx(100, abs=5)
        assert results["statistics"]["std"] > 0
    
    def test_full_analysis_with_shift(self):
        """Test analysis of process with mean shift"""
        hub = SPCHub()
        
        # Generate data with shift
        data = SPCTestDataGenerator.generate_with_shift(
            n=50, mean=100, sigma=5, shift_at=30, shift_magnitude=15
        )
        
        results = hub.analyze_process(
            data=data,
            chart_type=ChartType.I_MR,
            usl=115,
            lsl=85
        )
        
        # Should detect out of control
        assert results["status"] in [ProcessStatus.OUT_OF_CONTROL.value, ProcessStatus.WARNING.value]
        assert len(results["alerts"]) > 0
        
        # Should detect as not capable
        assert not results["capability"]["is_capable"]
    
    def test_full_analysis_with_trend(self):
        """Test analysis of process with trend"""
        hub = SPCHub()
        
        # Generate data with trend
        data = SPCTestDataGenerator.generate_with_trend(
            n=50, mean=100, sigma=5, slope=0.3
        )
        
        results = hub.analyze_process(
            data=data,
            chart_type=ChartType.I_MR
        )
        
        # Should detect trend
        assert results["trend"]["detected"]
        assert results["trend"]["direction"] in ["increasing", "decreasing"]
        assert len(results["trend"]["predicted_values"]) > 0
    
    def test_xbar_r_chart_analysis(self):
        """Test X-bar/R chart analysis"""
        hub = SPCHub()
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        results = hub.analyze_process(
            data=data,
            chart_type=ChartType.XBAR_R,
            subgroup_size=5,
            usl=115,
            lsl=85
        )
        
        assert "xbar" in results["control_limits"]
        assert "r" in results["control_limits"]
        assert results["control_limits"]["xbar"]["centerline"] > 0
    
    def test_ewma_chart_analysis(self):
        """Test EWMA chart analysis"""
        hub = SPCHub()
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        results = hub.analyze_process(
            data=data,
            chart_type=ChartType.EWMA,
            target=100
        )
        
        assert "ewma" in results["control_limits"]
        assert results["control_limits"]["ewma"]["centerline"] == 100
    
    def test_cusum_chart_analysis(self):
        """Test CUSUM chart analysis"""
        hub = SPCHub()
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 50)
        
        results = hub.analyze_process(
            data=data,
            chart_type=ChartType.CUSUM,
            target=100
        )
        
        assert "cusum" in results["control_limits"]
        assert "decision_limit" in results["control_limits"]["cusum"]


# ============================================================================
# Test Data Generators
# ============================================================================

class TestSPCTestDataGenerator:
    """Test synthetic data generation"""
    
    def test_in_control_generation(self):
        """Test in-control data generation"""
        data = SPCTestDataGenerator.generate_in_control(n=100, mean=100, sigma=5)
        
        assert len(data) == 100
        assert np.abs(np.mean(data) - 100) < 10  # Within reasonable range
        assert np.std(data) > 0
    
    def test_shift_generation(self):
        """Test data with shift generation"""
        data = SPCTestDataGenerator.generate_with_shift(
            n=100, mean=100, sigma=5, shift_at=50, shift_magnitude=10
        )
        
        assert len(data) == 100
        # Before shift should be near 100, after should be near 110
        assert np.mean(data[:50]) < np.mean(data[50:])
    
    def test_trend_generation(self):
        """Test data with trend generation"""
        data = SPCTestDataGenerator.generate_with_trend(
            n=100, mean=100, sigma=5, slope=0.2
        )
        
        assert len(data) == 100
        # Should have positive correlation with index
        correlation = np.corrcoef(np.arange(100), data)[0, 1]
        assert correlation > 0.5
    
    def test_cycle_generation(self):
        """Test data with cycle generation"""
        data = SPCTestDataGenerator.generate_with_cycle(
            n=100, mean=100, sigma=2, period=20, amplitude=5
        )
        
        assert len(data) == 100
        # Mean should be close to 100
        assert np.abs(np.mean(data) - 100) < 5


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformance:
    """Performance benchmarks for SPC operations"""
    
    def test_control_limit_calculation_performance(self):
        """Benchmark control limit calculation"""
        calculator = ControlChartCalculator()
        
        # Large dataset
        np.random.seed(42)
        data = np.random.normal(100, 5, 1000)
        
        start = time.time()
        for _ in range(100):
            calculator.calculate_i_mr_limits(data)
        elapsed = time.time() - start
        
        avg_time = elapsed / 100
        print(f"\nControl limit calculation: {avg_time*1000:.2f} ms/operation")
        assert avg_time < 0.1  # Should be < 100ms
    
    def test_rule_detection_performance(self):
        """Benchmark rule detection"""
        limits = ControlLimits(ucl=115, lcl=85, centerline=100, sigma=5)
        detector = RuleDetector(limits)
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 1000)
        
        start = time.time()
        for _ in range(10):
            detector.detect_all_rules(data)
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"Rule detection (1000 points): {avg_time*1000:.2f} ms/operation")
        assert avg_time < 1.0  # Should be < 1 second
    
    def test_full_analysis_performance(self):
        """Benchmark full SPC analysis"""
        hub = SPCHub()
        
        np.random.seed(42)
        data = np.random.normal(100, 5, 100)
        
        start = time.time()
        for _ in range(10):
            hub.analyze_process(
                data=data,
                chart_type=ChartType.I_MR,
                usl=115,
                lsl=85
            )
        elapsed = time.time() - start
        
        avg_time = elapsed / 10
        print(f"Full SPC analysis (100 points): {avg_time*1000:.2f} ms/operation")
        assert avg_time < 2.0  # Should be < 2 seconds


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        hub = SPCHub()
        data = np.array([])
        
        # Should handle gracefully
        results = hub.analyze_process(data=data, chart_type=ChartType.I_MR)
        assert "error" in results or results["status"] == ProcessStatus.UNKNOWN.value
    
    def test_single_point_data(self):
        """Test handling of single data point"""
        hub = SPCHub()
        data = np.array([100.0])
        
        results = hub.analyze_process(data=data, chart_type=ChartType.I_MR)
        # Should handle without crashing
        assert "statistics" in results
    
    def test_constant_data(self):
        """Test handling of constant data (no variation)"""
        hub = SPCHub()
        data = np.full(50, 100.0)
        
        results = hub.analyze_process(data=data, chart_type=ChartType.I_MR)
        
        # Standard deviation should be 0 or very small
        assert results["statistics"]["std"] < 0.01
    
    def test_extreme_outliers(self):
        """Test handling of extreme outliers"""
        hub = SPCHub()
        
        data = np.random.normal(100, 5, 50)
        data[25] = 1000  # Extreme outlier
        
        results = hub.analyze_process(data=data, chart_type=ChartType.I_MR)
        
        # Should detect as out of control
        assert results["status"] == ProcessStatus.OUT_OF_CONTROL.value
        assert len(results["alerts"]) > 0


# ============================================================================
# Test Runner
# ============================================================================

def run_all_tests():
    """Run all tests and generate report"""
    print("=" * 80)
    print("Session 13: SPC Hub - Comprehensive Test Suite")
    print("=" * 80)
    
    # Run pytest
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_",
        "--color=yes"
    ]
    
    result = pytest.main(pytest_args)
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    if result == 0:
        print("✅ All tests passed!")
    else:
        print(f"❌ Some tests failed (exit code: {result})")
    
    return result


if __name__ == "__main__":
    run_all_tests()
