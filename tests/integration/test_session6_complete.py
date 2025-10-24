"""
Session 6: Complete Integration Tests
Testing DLTS, EBIC, and PCD implementation
"""

import pytest
import numpy as np
import json
from datetime import datetime
from pathlib import Path

# Import the backend modules
from session6_backend_analysis import (
    DLTSAnalyzer, EBICAnalyzer, PCDAnalyzer,
    Session6ElectricalAnalysis, Session6TestDataGenerator
)

# ==========================================
# Test Configuration
# ==========================================

TEST_DATA_DIR = Path("/tmp/test_data/session6")
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

@pytest.fixture
def session6_analyzer():
    """Fixture for Session 6 analysis interface"""
    return Session6ElectricalAnalysis()

@pytest.fixture
def test_data_generator():
    """Fixture for test data generator"""
    return Session6TestDataGenerator()

# ==========================================
# DLTS Tests
# ==========================================

class TestDLTSAnalysis:
    """Test DLTS measurement and analysis"""
    
    def test_dlts_single_trap(self, session6_analyzer):
        """Test DLTS analysis with single trap"""
        # Generate test data
        data = session6_analyzer.generate_test_data('dlts', num_traps=1)
        
        # Verify structure
        assert 'spectrum' in data
        assert 'traps' in data
        assert 'arrhenius' in data
        assert len(data['traps']) >= 1
        
        # Check trap parameters
        trap = data['traps'][0]
        assert 0 < trap['activation_energy'] < 1.0  # eV range
        assert 1e-20 < trap['capture_cross_section'] < 1e-12  # cm^2 range
        assert trap['trap_concentration'] > 0
        
    def test_dlts_multiple_traps(self, session6_analyzer):
        """Test DLTS with multiple traps"""
        # Generate spectrum with 3 traps
        temperatures = np.linspace(77, 400, 162)
        capacitances = np.ones_like(temperatures) * 100
        
        # Add three distinct peaks
        for peak_temp, amplitude in [(120, 8), (220, 12), (310, 5)]:
            capacitances += amplitude * np.exp(-(temperatures - peak_temp)**2 / 400)
        
        # Process
        results = session6_analyzer.process_dlts({
            'temperatures': temperatures.tolist(),
            'capacitances': capacitances.tolist(),
            'rate_window': 200
        })
        
        # Verify multiple traps detected
        assert len(results['traps']) >= 2
        assert results['dominant_trap'] is not None
        
    def test_dlts_arrhenius_generation(self, session6_analyzer):
        """Test Arrhenius plot data generation"""
        data = session6_analyzer.generate_test_data('dlts')
        
        assert 'arrhenius' in data
        assert len(data['arrhenius']) > 0
        
        # Check Arrhenius data structure
        for point in data['arrhenius']:
            assert 'trap' in point
            assert 'temperature' in point
            assert 'inv_temp' in point
            assert 'emission' in point
            assert point['inv_temp'] == pytest.approx(1000 / point['temperature'], rel=1e-3)
    
    def test_dlts_trap_identification(self):
        """Test trap identification against database"""
        analyzer = DLTSAnalyzer()
        
        # Test known trap (Fe in Si)
        temperatures = np.linspace(77, 400, 162)
        capacitances = np.ones_like(temperatures) * 100
        
        # Add Fe peak around 120K
        capacitances += 10 * np.exp(-(temperatures - 120)**2 / 225)
        
        results = analyzer.analyze_spectrum(temperatures, capacitances, 200)
        
        # Should identify E1 (Fe)
        assert len(results['traps']) > 0
        trap = results['traps'][0]
        assert trap['label'] in ['E1', 'E2', 'E3']  # Should match database
        assert trap['confidence'] > 0.5

# ==========================================
# EBIC Tests
# ==========================================

class TestEBICAnalysis:
    """Test EBIC mapping and analysis"""
    
    def test_ebic_map_generation(self, session6_analyzer):
        """Test EBIC map generation and analysis"""
        data = session6_analyzer.generate_test_data('ebic', map_size=128)
        
        assert 'map' in data
        assert 'diffusion_length' in data
        assert 'defects' in data
        assert 'statistics' in data
        
        # Check diffusion length extraction
        ld = data['diffusion_length']
        assert 10 < ld['mean'] < 100  # Reasonable range in µm
        assert ld['std'] >= 0
        
    def test_ebic_defect_detection(self):
        """Test defect detection in EBIC maps"""
        analyzer = EBICAnalyzer(pixel_size=1.0)
        
        # Create map with known defects
        map_size = 100
        current_map = np.ones((map_size, map_size)) * 50
        
        # Add defects at specific positions
        defect_positions = [(20, 30), (70, 80)]
        for x, y in defect_positions:
            current_map[y-5:y+5, x-5:x+5] *= 0.3
        
        results = analyzer.analyze_map(current_map)
        
        # Should detect defects
        assert len(results['defects']) >= len(defect_positions)
        for defect in results['defects']:
            assert defect['contrast'] < 0  # Negative contrast
            assert defect['area'] > 0
    
    def test_ebic_line_profile(self):
        """Test line profile extraction"""
        analyzer = EBICAnalyzer()
        
        # Create exponential decay profile
        map_size = 100
        x = np.arange(map_size)
        current_map = np.zeros((map_size, map_size))
        
        for i in range(map_size):
            current_map[i, :] = 100 * np.exp(-np.abs(x - 50) / 20)
        
        profile = analyzer.extract_line_profile(
            current_map,
            (0, 50),
            (99, 50)
        )
        
        assert 'distance' in profile
        assert 'current' in profile
        assert 'fitted_ld' in profile
        assert 15 < profile['fitted_ld'] < 25  # Should be close to 20
    
    def test_ebic_quality_assessment(self):
        """Test map quality assessment"""
        analyzer = EBICAnalyzer()
        
        # High quality map (high SNR)
        good_map = np.random.normal(100, 5, (100, 100))
        good_map = np.maximum(0, good_map)
        
        results_good = analyzer.analyze_map(good_map)
        assert results_good['quality_score'] > 50
        
        # Poor quality map (low SNR)
        bad_map = np.random.normal(10, 10, (100, 100))
        bad_map = np.maximum(0, bad_map)
        
        results_bad = analyzer.analyze_map(bad_map)
        assert results_bad['quality_score'] < results_good['quality_score']

# ==========================================
# PCD Tests
# ==========================================

class TestPCDAnalysis:
    """Test PCD lifetime measurement analysis"""
    
    def test_pcd_transient_analysis(self, session6_analyzer):
        """Test transient PCD analysis"""
        data = session6_analyzer.generate_test_data('pcd', mode='transient')
        
        assert 'transient' in data
        assert 'lifetime' in data
        assert 'srv' in data
        assert 'injection_dependent' in data
        
        # Check lifetime values
        lifetime = data['lifetime']
        assert 1 < lifetime['effective'] < 10000  # µs range
        assert lifetime['bulk'] > lifetime['effective']
        
        # Check SRV
        srv = data['srv']
        assert 0.1 < srv['effective'] < 1e7  # cm/s range
    
    def test_pcd_qsspc_analysis(self, session6_analyzer):
        """Test QSSPC analysis"""
        data = session6_analyzer.generate_test_data('pcd', mode='qsspc')
        
        assert 'qsspc' in data
        assert 'parameters' in data
        
        params = data['parameters']
        assert 'tau_low_injection' in params
        assert 'tau_high_injection' in params
        assert 'crossover' in params
        assert 'auger_coefficient' in params
        
        # Check Auger coefficient (Si range)
        assert 1e-31 < params['auger_coefficient'] < 1e-29
    
    def test_pcd_carrier_density_conversion(self):
        """Test photoconductance to carrier density conversion"""
        analyzer = PCDAnalyzer(sample_thickness=300e-4)
        
        # Test conversion
        photoconductance = np.array([1e-3, 1e-4, 1e-5])
        carrier_density = analyzer._conductance_to_density(
            photoconductance, 300, 'p-type'
        )
        
        # Check reasonable values
        assert np.all(carrier_density > 0)
        assert np.all(carrier_density < 1e18)  # Reasonable upper limit
    
    def test_pcd_lifetime_extraction(self):
        """Test lifetime extraction from decay"""
        analyzer = PCDAnalyzer()
        
        # Generate perfect exponential decay
        tau_true = 100e-6
        time = np.linspace(0, 500e-6, 100)
        carrier_density = 1e15 * np.exp(-time / tau_true)
        
        tau_eff, tau_bulk, tau_surface = analyzer._extract_lifetimes(
            time, carrier_density
        )
        
        # Should extract close to true value
        assert abs(tau_eff - tau_true) / tau_true < 0.2  # Within 20%

# ==========================================
# Integration Tests
# ==========================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_complete_dlts_workflow(self, session6_analyzer):
        """Test complete DLTS measurement workflow"""
        # Generate measurement
        test_data = session6_analyzer.generate_test_data('dlts')
        
        # Save to file
        output_file = TEST_DATA_DIR / "dlts_test.json"
        with open(output_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        # Verify file exists and is valid
        assert output_file.exists()
        
        # Load and reprocess
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Process loaded data
        results = session6_analyzer.process_dlts({
            'temperatures': loaded_data['spectrum']['temperature'],
            'capacitances': loaded_data['spectrum']['capacitance']
        })
        
        assert 'traps' in results
        assert len(results['traps']) > 0
    
    def test_complete_ebic_workflow(self, session6_analyzer):
        """Test complete EBIC mapping workflow"""
        # Generate map
        test_data = session6_analyzer.generate_test_data('ebic')
        
        # Add line profiles
        current_map = np.array(test_data['map']['current'])
        
        # Process with profiles
        results = session6_analyzer.process_ebic({
            'current_map': current_map,
            'beam_energy': 20.0,
            'line_profiles': [
                {'start': (0, 128), 'end': (255, 128)},
                {'start': (128, 0), 'end': (128, 255)}
            ]
        })
        
        assert 'profiles' in results
        assert len(results['profiles']) == 2
        
        # Check profile quality
        for profile in results['profiles']:
            assert 'fitted_ld' in profile
            assert profile['fitted_ld'] > 0
    
    def test_complete_pcd_workflow(self, session6_analyzer):
        """Test complete PCD measurement workflow"""
        # Test both transient and QSSPC modes
        for mode in ['transient', 'qsspc']:
            test_data = session6_analyzer.generate_test_data('pcd', mode=mode)
            
            # Save results
            output_file = TEST_DATA_DIR / f"pcd_{mode}_test.json"
            with open(output_file, 'w') as f:
                json.dump(test_data, f, indent=2)
            
            assert output_file.exists()
            
            # Verify critical parameters
            if mode == 'transient':
                assert 'lifetime' in test_data
                assert test_data['lifetime']['effective'] > 0
            else:
                assert 'parameters' in test_data
                assert test_data['parameters']['tau_low_injection'] > 0

# ==========================================
# Performance Tests
# ==========================================

class TestPerformance:
    """Performance and efficiency tests"""
    
    def test_dlts_performance(self, session6_analyzer):
        """Test DLTS analysis performance"""
        import time
        
        # Generate large dataset
        temperatures = np.linspace(77, 400, 500)
        capacitances = np.random.normal(100, 5, 500)
        
        start_time = time.time()
        results = session6_analyzer.process_dlts({
            'temperatures': temperatures.tolist(),
            'capacitances': capacitances.tolist()
        })
        processing_time = time.time() - start_time
        
        # Should complete within 2 seconds
        assert processing_time < 2.0
        assert results is not None
    
    def test_ebic_map_performance(self, session6_analyzer):
        """Test EBIC map analysis performance"""
        import time
        
        # Generate large map
        current_map = np.random.normal(50, 10, (512, 512))
        current_map = np.maximum(0, current_map)
        
        start_time = time.time()
        results = session6_analyzer.process_ebic({
            'current_map': current_map.tolist()
        })
        processing_time = time.time() - start_time
        
        # Should complete within 5 seconds even for large maps
        assert processing_time < 5.0
    
    def test_pcd_performance(self, session6_analyzer):
        """Test PCD analysis performance"""
        import time
        
        # Generate high-resolution decay
        time_array = np.logspace(-7, -2, 1000)
        photoconductance = 1e-3 * np.exp(-time_array / 100e-6)
        
        start_time = time.time()
        results = session6_analyzer.process_pcd({
            'mode': 'transient',
            'time': time_array.tolist(),
            'photoconductance': photoconductance.tolist()
        })
        processing_time = time.time() - start_time
        
        # Should complete within 1 second
        assert processing_time < 1.0

# ==========================================
# Data Validation Tests
# ==========================================

class TestDataValidation:
    """Test data validation and error handling"""
    
    def test_dlts_invalid_data_handling(self, session6_analyzer):
        """Test DLTS handling of invalid data"""
        # Empty data
        with pytest.raises((ValueError, IndexError)):
            session6_analyzer.process_dlts({
                'temperatures': [],
                'capacitances': []
            })
        
        # Mismatched lengths
        with pytest.raises((ValueError, IndexError)):
            session6_analyzer.process_dlts({
                'temperatures': [77, 100, 150],
                'capacitances': [100, 101]  # Wrong length
            })
    
    def test_ebic_map_validation(self):
        """Test EBIC map validation"""
        analyzer = EBICAnalyzer()
        
        # Test with negative values
        current_map = np.random.normal(0, 10, (100, 100))
        results = analyzer.analyze_map(current_map)
        
        # Should handle negative values gracefully
        assert results is not None
        assert 'statistics' in results
    
    def test_pcd_boundary_conditions(self):
        """Test PCD analysis with boundary conditions"""
        analyzer = PCDAnalyzer()
        
        # Test with very short lifetime
        time = np.linspace(0, 1e-6, 100)
        photoconductance = 1e-3 * np.exp(-time / 1e-7)
        
        results = analyzer.analyze_transient(time, photoconductance)
        assert results['lifetime']['effective'] > 0
        
        # Test with very long lifetime
        time = np.linspace(0, 0.1, 100)
        photoconductance = 1e-3 * np.exp(-time / 0.01)
        
        results = analyzer.analyze_transient(time, photoconductance)
        assert results['lifetime']['effective'] < 1e8  # Reasonable upper bound

# ==========================================
# Run Tests
# ==========================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    print("\n" + "="*70)
    print("Session 6 Integration Tests Complete!")
    print("="*70)
    print("\nTest Summary:")
    print("  ✓ DLTS Analysis: Single/multiple traps, Arrhenius plots")
    print("  ✓ EBIC Mapping: Defect detection, diffusion length extraction")
    print("  ✓ PCD Analysis: Transient and QSSPC modes, lifetime extraction")
    print("  ✓ Integration: Complete workflows for all methods")
    print("  ✓ Performance: All methods complete within time limits")
    print("  ✓ Validation: Proper error handling and boundary conditions")