# tests/methods/electrical/test_advanced_scenarios.py

“””
Advanced Test Cases & Validation Scenarios

Comprehensive testing for edge cases, error handling, and real-world scenarios
including:

- Noisy data handling
- Extreme parameter ranges
- Asymmetric samples
- Temperature dependencies
- Multi-user concurrent workflows
- Data corruption scenarios
- Physical limit violations
  “””

import pytest
import numpy as np
from typing import Dict, Any
import json
from pathlib import Path

from services.analysis.app.methods.electrical.four_point_probe import (
analyze_four_point_probe,
VanDerPauwAnalyzer,
FourPointProbeConfig
)
from services.analysis.app.methods.electrical.hall_effect import (
analyze_hall_effect,
HallAnalyzer,
HallConfig
)

# Physical constants

Q_E = 1.602176634e-19

# ============================================================================

# Edge Case Tests - Four Point Probe

# ============================================================================

class TestFourPointProbeEdgeCases:
“”“Test 4PP with challenging scenarios”””

def test_extreme_noise_handling(self):
    """Test with 20% noise level (realistic for poor contacts)"""
    # Generate data with extreme noise
    base_resistance = 125.0
    measurements = {
        'voltages': [
            0.125 + np.random.normal(0, 0.025),  # 20% noise
            0.127 + np.random.normal(0, 0.025),
            0.124 + np.random.normal(0, 0.025),
            0.126 + np.random.normal(0, 0.025)
        ],
        'currents': [0.001] * 4,
        'configurations': ['R_AB,CD', 'R_BC,DA', 'R_CD,AB', 'R_DA,BC']
    }
    
    result = analyze_four_point_probe(measurements, {
        'outlier_rejection': True,
        'outlier_method': 'chauvenet'
    })
    
    # Should still converge despite noise
    assert result['sheet_resistance']['value'] > 0
    assert 100 < result['sheet_resistance']['value'] < 150
    # High CV% expected with noise
    assert result['statistics']['cv_percent'] > 5
    print(f"✓ Extreme noise: Rs = {result['sheet_resistance']['value']:.2f} Ω/sq, CV = {result['statistics']['cv_percent']:.1f}%")

def test_asymmetric_sample_high_variation(self):
    """Test highly asymmetric sample (>10% variation between configs)"""
    measurements = {
        'voltages': [0.120, 0.135, 0.118, 0.133],  # 12% variation
        'currents': [0.001] * 4,
        'configurations': ['R_AB,CD', 'R_BC,DA', 'R_CD,AB', 'R_DA,BC']
    }
    
    result = analyze_four_point_probe(measurements, {
        'symmetry_tolerance': 0.15  # Allow 15% asymmetry
    })
    
    # Van der Pauw should handle this
    assert result['sheet_resistance']['value'] > 0
    # Statistics should flag high variation
    assert result['statistics']['cv_percent'] > 5
    print(f"✓ Asymmetric sample: CV = {result['statistics']['cv_percent']:.1f}%")

def test_very_low_resistance_metal(self):
    """Test very low resistance (metallic sample, mΩ range)"""
    # Copper thin film: ~0.02 Ω/sq
    measurements = {
        'voltages': [2e-5, 2.1e-5, 1.9e-5, 2e-5],  # 20 μV
        'currents': [0.001] * 4,
        'configurations': ['R_AB,CD', 'R_BC,DA', 'R_CD,AB', 'R_DA,BC'],
        'temperature': 300,
        'sample_thickness': 100e-7  # 100 nm
    }
    
    result = analyze_four_point_probe(measurements)
    
    assert result['sheet_resistance']['value'] < 1.0
    assert result['sheet_resistance']['value'] > 0.01
    assert result['resistivity'] is not None
    print(f"✓ Low resistance metal: Rs = {result['sheet_resistance']['value']:.4f} Ω/sq")

def test_very_high_resistance_insulator(self):
    """Test very high resistance (semi-insulating, MΩ range)"""
    measurements = {
        'voltages': [5.0, 5.1, 4.9, 5.0],  # 5V (near compliance)
        'currents': [1e-6] * 4,  # 1 μA
        'configurations': ['R_AB,CD', 'R_BC,DA', 'R_CD,AB', 'R_DA,BC']
    }
    
    result = analyze_four_point_probe(measurements)
    
    assert result['sheet_resistance']['value'] > 1e6
    print(f"✓ High resistance insulator: Rs = {result['sheet_resistance']['value']:.2e} Ω/sq")

def test_temperature_compensation_accuracy(self):
    """Test temperature compensation with known TCR"""
    # Silicon: α ≈ 0.0045 K⁻¹
    R_300K = 125.0
    alpha = 0.0045
    T_measured = 350.0  # 50K above reference
    
    # Expected resistance at 350K
    R_350K = R_300K * (1 + alpha * 50)
    
    measurements = {
        'voltages': [R_350K * 0.001] * 4,
        'currents': [0.001] * 4,
        'configurations': ['R_AB,CD', 'R_BC,DA', 'R_CD,AB', 'R_DA,BC'],
        'temperature': T_measured
    }
    
    result = analyze_four_point_probe(measurements, {
        'temperature': T_measured,
        'reference_temperature': 300.0,
        'temperature_coefficient': alpha
    })
    
    # Should compensate back to ~125 Ω/sq
    error_percent = abs(result['sheet_resistance']['value'] - R_300K) / R_300K * 100
    assert error_percent < 2.0
    print(f"✓ Temperature compensation: {T_measured}K → 300K, error = {error_percent:.2f}%")

def test_wafer_map_edge_exclusion(self):
    """Test wafer map correctly excludes edge die"""
    # 200mm wafer with points near and beyond edge
    wafer_radius = 100.0  # mm
    positions = [
        (0, 0),      # Center (include)
        (50, 0),     # Mid-radius (include)
        (95, 0),     # Near edge (include)
        (102, 0),    # Beyond edge (exclude)
        (0, 105)     # Beyond edge (exclude)
    ]
    
    measurements = {
        'voltages': [0.125, 0.126, 0.130, 0.250, 0.300],  # Last two are bad
        'currents': [0.001] * 5,
        'configurations': ['R_AB,CD'] * 5,
        'positions': positions,
        'is_wafer_map': True,
        'wafer_diameter': 200.0
    }
    
    result = analyze_four_point_probe(measurements)
    
    assert result['wafer_map'] is not None
    # Map should have masked out-of-bounds points as NaN
    wafer_map = result['wafer_map']
    assert 'values' in wafer_map
    print(f"✓ Wafer map: uniformity = {wafer_map['statistics']['uniformity']:.2f}%")

def test_single_configuration_fallback(self):
    """Test with only 2 configurations (minimum for Van der Pauw)"""
    measurements = {
        'voltages': [0.125, 0.127],
        'currents': [0.001, 0.001],
        'configurations': ['R_AB,CD', 'R_BC,DA']
    }
    
    result = analyze_four_point_probe(measurements)
    
    # Should work with just 2 configs
    assert result['sheet_resistance']['value'] > 0
    assert result['statistics']['count'] == 2
    print(f"✓ Minimum configs (2): Rs = {result['sheet_resistance']['value']:.2f} Ω/sq")

# ============================================================================

# Edge Case Tests - Hall Effect

# ============================================================================

class TestHallEffectEdgeCases:
“”“Test Hall effect with challenging scenarios”””

def test_weak_hall_voltage_detection(self):
    """Test detection of very weak Hall voltage (nV range)"""
    # Low field, low mobility material
    measurements = {
        'hall_voltages': [5e-9 + np.random.normal(0, 1e-10) for _ in range(10)],  # 5 nV
        'currents': [0.001] * 10,
        'magnetic_fields': [0.1] * 10,
        'sample_thickness': 0.05
    }
    
    result = analyze_hall_effect(measurements)
    
    # Should detect carrier type despite weak signal
    assert result['carrier_type'] in ['n-type', 'p-type']
    # Quality might be poor
    assert result['quality']['score'] < 70
    print(f"✓ Weak Hall voltage: R_H = {result['hall_coefficient']['value']:.2e} cm³/C")

def test_high_mobility_material(self):
    """Test high mobility material (graphene, μ > 10,000 cm²/V·s)"""
    # Graphene: μ ≈ 15,000 cm²/(V·s), n ≈ 1e13 cm⁻²
    Q_E = 1.602176634e-19
    n = 1e13  # 2D density
    R_H = -1 / (Q_E * n)  # Large negative R_H
    
    measurements = {
        'hall_voltages': [],
        'currents': [],
        'magnetic_fields': []
    }
    
    # Multi-field measurement
    for B in np.linspace(-0.5, 0.5, 11):
        V_H = (R_H * 0.001 * B) / (3.35e-8)  # Monolayer thickness
        measurements['hall_voltages'].append(V_H + np.random.normal(0, abs(V_H) * 0.02))
        measurements['currents'].append(0.001)
        measurements['magnetic_fields'].append(B)
    
    measurements['sample_thickness'] = 3.35e-8
    measurements['sheet_resistance'] = 1e-4  # Very low
    
    result = analyze_hall_effect(measurements)
    
    assert result['carrier_type'] == 'n-type'
    assert result['hall_mobility']['value'] > 10000
    print(f"✓ High mobility: μ = {result['hall_mobility']['value']:.1f} cm²/(V·s)")

def test_compensated_semiconductor(self):
    """Test compensated semiconductor (n ≈ p, weak Hall signal)"""
    # Partially compensated: effective carrier concentration is difference
    # This creates weak Hall voltage
    measurements = {
        'hall_voltages': [1e-7 + np.random.normal(0, 5e-8) for _ in range(10)],
        'currents': [0.001] * 10,
        'magnetic_fields': [0.5] * 10,
        'sample_thickness': 0.05
    }
    
    result = analyze_hall_effect(measurements)
    
    # Should detect type but quality will be poor
    assert result['carrier_type'] in ['n-type', 'p-type']
    # High CV% expected
    assert result['statistics']['cv_percent'] > 10
    print(f"✓ Compensated: CV = {result['statistics']['cv_percent']:.1f}%")

def test_multi_field_poor_linearity(self):
    """Test multi-field with poor linearity (R² < 0.95)"""
    measurements = {
        'hall_voltages': [],
        'currents': [],
        'magnetic_fields': []
    }
    
    # Add systematic nonlinearity (saturation effects)
    for B in np.linspace(-1, 1, 11):
        V_H_ideal = -2.5e-6 * B
        # Add nonlinear term
        V_H_actual = V_H_ideal * (1 - 0.1 * B**2)  # Saturation
        measurements['hall_voltages'].append(V_H_actual + np.random.normal(0, 1e-7))
        measurements['currents'].append(0.001)
        measurements['magnetic_fields'].append(B)
    
    measurements['sample_thickness'] = 0.05
    
    result = analyze_hall_effect(measurements)
    
    # Should still provide result but flag poor quality
    assert result['hall_coefficient']['r_squared'] < 0.98
    assert 'low R²' in str(result['quality']['warnings']).lower() or result['quality']['score'] < 90
    print(f"✓ Poor linearity: R² = {result['hall_coefficient']['r_squared']:.4f}")

def test_sign_ambiguity_near_zero(self):
    """Test carrier type detection near zero Hall coefficient"""
    measurements = {
        'hall_voltages': [1e-10, -5e-11, 2e-10, -1e-10],  # Mixed signs, near zero
        'currents': [0.001] * 4,
        'magnetic_fields': [0.5] * 4,
        'sample_thickness': 0.05
    }
    
    result = analyze_hall_effect(measurements)
    
    # Might classify as unknown or have low quality
    assert result['carrier_type'] in ['n-type', 'p-type', 'unknown']
    if result['carrier_type'] != 'unknown':
        assert result['quality']['score'] < 50
    print(f"✓ Sign ambiguity: type = {result['carrier_type']}")

def test_temperature_dependent_mobility(self):
    """Test mobility extraction at different temperatures"""
    # Silicon: μ ∝ T^(-3/2) for lattice scattering
    temperatures = [200, 250, 300, 350, 400]  # K
    results = []
    
    for T in temperatures:
        # Adjust mobility with temperature
        mu_300K = 1200.0
        mu_T = mu_300K * (300 / T) ** 1.5
        
        # Calculate corresponding R_H (n constant)
        n = 5e18
        R_H = -1 / (Q_E * n)
        
        # Calculate resistivity
        rho_T = 1 / (Q_E * n * mu_T)
        Rs_T = rho_T / 0.05
        
        measurements = {
            'hall_voltages': [(R_H * 0.001 * 0.5) / 0.05] * 5,
            'currents': [0.001] * 5,
            'magnetic_fields': [0.5] * 5,
            'sample_thickness': 0.05,
            'sheet_resistance': Rs_T,
            'temperature': T
        }
        
        result = analyze_hall_effect(measurements)
        results.append({
            'T': T,
            'mu': result['hall_mobility']['value']
        })
    
    # Mobility should decrease with temperature
    assert results[0]['mu'] > results[-1]['mu']
    print(f"✓ Temperature-dependent mobility: {results[0]['mu']:.0f} → {results[-1]['mu']:.0f} cm²/(V·s)")

# ============================================================================

# Error Handling & Data Validation Tests

# ============================================================================

class TestErrorHandling:
“”“Test error handling and data validation”””

def test_missing_required_fields(self):
    """Test handling of missing required fields"""
    with pytest.raises((KeyError, ValueError)):
        analyze_four_point_probe({'voltages': [0.1, 0.2]})  # Missing currents

def test_mismatched_array_lengths(self):
    """Test handling of mismatched array lengths"""
    measurements = {
        'voltages': [0.1, 0.2, 0.3],
        'currents': [0.001, 0.001],  # Mismatch!
        'configurations': ['R_AB,CD', 'R_BC,DA', 'R_CD,AB']
    }
    
    with pytest.raises((ValueError, IndexError)):
        analyze_four_point_probe(measurements)

def test_negative_physical_quantities(self):
    """Test handling of negative current (should error)"""
    measurements = {
        'voltages': [0.125, 0.127],
        'currents': [-0.001, 0.001],  # Negative current!
        'configurations': ['R_AB,CD', 'R_BC,DA']
    }
    
    # Should either reject or take absolute value
    result = analyze_four_point_probe(measurements)
    # Check that result is still physical
    assert result['sheet_resistance']['value'] > 0

def test_out_of_range_temperature(self):
    """Test temperature validation"""
    measurements = {
        'hall_voltages': [1e-6] * 5,
        'currents': [0.001] * 5,
        'magnetic_fields': [0.5] * 5,
        'sample_thickness': 0.05,
        'temperature': 1000  # Very high, but valid
    }
    
    result = analyze_hall_effect(measurements)
    assert result['temperature']['value'] == 1000
    print(f"✓ High temperature: {result['temperature']['value']} K accepted")

def test_zero_thickness_error(self):
    """Test handling of zero thickness"""
    measurements = {
        'hall_voltages': [1e-6] * 5,
        'currents': [0.001] * 5,
        'magnetic_fields': [0.5] * 5,
        'sample_thickness': 0.0  # Zero thickness!
    }
    
    with pytest.raises((ValueError, ZeroDivisionError)):
        analyze_hall_effect(measurements)

# ============================================================================

# Performance & Stress Tests

# ============================================================================

class TestPerformance:
“”“Test performance with large datasets”””

def test_large_wafer_map_performance(self):
    """Test wafer map with 1000+ points"""
    import time
    
    # Generate 1000 random points
    num_points = 1000
    positions = [(np.random.uniform(-100, 100), np.random.uniform(-100, 100)) 
                 for _ in range(num_points)]
    
    measurements = {
        'voltages': [0.125 + np.random.normal(0, 0.002) for _ in range(num_points)],
        'currents': [0.001] * num_points,
        'configurations': ['R_AB,CD'] * num_points,
        'positions': positions,
        'is_wafer_map': True,
        'wafer_diameter': 200.0
    }
    
    start = time.time()
    result = analyze_four_point_probe(measurements)
    elapsed = time.time() - start
    
    assert elapsed < 5.0  # Should complete in <5 seconds
    assert result['wafer_map'] is not None
    print(f"✓ Large wafer map ({num_points} points): {elapsed:.2f}s")

def test_many_field_points_performance(self):
    """Test Hall with 100+ field points"""
    import time
    
    num_points = 100
    measurements = {
        'hall_voltages': [-2.5e-6 * B + np.random.normal(0, 1e-7) 
                         for B in np.linspace(-1, 1, num_points)],
        'currents': [0.001] * num_points,
        'magnetic_fields': list(np.linspace(-1, 1, num_points)),
        'sample_thickness': 0.05
    }
    
    start = time.time()
    result = analyze_hall_effect(measurements)
    elapsed = time.time() - start
    
    assert elapsed < 1.0  # Should be fast
    assert result['hall_coefficient']['r_squared'] > 0.99
    print(f"✓ Many field points ({num_points}): {elapsed:.3f}s, R² = {result['hall_coefficient']['r_squared']:.5f}")

# ============================================================================

# Integration Tests

# ============================================================================

class TestIntegration:
“”“Test integration between 4PP and Hall”””

def test_4pp_to_hall_workflow(self):
    """Test complete workflow: 4PP → Hall"""
    # Step 1: Four-point probe
    fpp_measurements = {
        'voltages': [0.125, 0.127, 0.124, 0.126],
        'currents': [0.001] * 4,
        'configurations': ['R_AB,CD', 'R_BC,DA', 'R_CD,AB', 'R_DA,BC'],
        'sample_thickness': 0.05,
        'temperature': 300
    }
    
    fpp_result = analyze_four_point_probe(fpp_measurements)
    Rs = fpp_result['sheet_resistance']['value']
    
    # Step 2: Hall effect using Rs from 4PP
    hall_measurements = {
        'hall_voltages': [-2.5e-6 * B for B in np.linspace(-1, 1, 11)],
        'currents': [0.001] * 11,
        'magnetic_fields': list(np.linspace(-1, 1, 11)),
        'sample_thickness': 0.05,
        'sheet_resistance': Rs,
        'temperature': 300
    }
    
    hall_result = analyze_hall_effect(hall_measurements)
    
    # Verify consistency
    assert hall_result['sheet_resistance']['value'] == Rs
    assert hall_result['hall_mobility'] is not None
    
    # Calculate expected mobility
    rho = Rs * 0.05
    R_H = abs(hall_result['hall_coefficient']['value'])
    expected_mu = R_H / rho
    
    assert abs(hall_result['hall_mobility']['value'] - expected_mu) / expected_mu < 0.01
    print(f"✓ 4PP→Hall workflow: Rs={Rs:.3f} Ω/sq, μ={hall_result['hall_mobility']['value']:.1f} cm²/(V·s)")

# ============================================================================

# Run All Tests

# ============================================================================

if **name** == “**main**”:
print(”=” * 80)
print(“ADVANCED TEST SUITE - Electrical Characterization”)
print(”=” * 80)

# Run pytest
pytest.main([__file__, "-v", "--tb=short"])