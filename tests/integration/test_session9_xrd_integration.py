"""
Session 9: Integration Tests for XRD Analysis
==============================================
Comprehensive test suite for X-ray diffraction analysis system
"""

import pytest
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, Any, List, Tuple
from datetime import datetime
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Import the main modules
import sys
sys.path.append('/home/claude')
from session9_xrd_complete_implementation import (
    XRDAnalyzer, XRDSimulator, XRDPattern, CrystalStructure,
    Peak, PhaseIdentification, CrystalSystem, PeakProfile
)

# Test fixtures
@pytest.fixture
def xrd_analyzer():
    """Create XRD analyzer instance"""
    return XRDAnalyzer()

@pytest.fixture
def xrd_simulator():
    """Create XRD simulator instance"""
    return XRDSimulator(seed=42)

@pytest.fixture
def sample_si_pattern(xrd_simulator):
    """Generate sample Si XRD pattern"""
    return xrd_simulator.generate_pattern(
        phase='Si',
        crystallite_size=50,
        microstrain=0.001,
        texture_coefficient=1.0
    )

@pytest.fixture
def sample_mixture_pattern(xrd_simulator):
    """Generate mixture XRD pattern"""
    return xrd_simulator.generate_mixture([
        ('Si', 0.6),
        ('SiO2_quartz', 0.4)
    ])

@pytest.fixture
def sample_peaks():
    """Create sample peak list"""
    return [
        Peak(position=28.4, d_spacing=3.136, intensity=1000, fwhm=0.15, area=150),
        Peak(position=47.3, d_spacing=1.920, intensity=600, fwhm=0.18, area=108),
        Peak(position=56.1, d_spacing=1.638, intensity=350, fwhm=0.20, area=70),
        Peak(position=69.1, d_spacing=1.358, intensity=100, fwhm=0.22, area=22),
        Peak(position=76.4, d_spacing=1.246, intensity=150, fwhm=0.24, area=36)
    ]


class TestXRDAnalyzer:
    """Test suite for XRD analyzer core functionality"""
    
    def test_pattern_processing(self, xrd_analyzer, sample_si_pattern):
        """Test XRD pattern processing"""
        # Process pattern
        processed = xrd_analyzer.process_pattern(
            sample_si_pattern,
            smooth=True,
            background_correct=True
        )
        
        # Check processed pattern
        assert isinstance(processed, XRDPattern)
        assert len(processed.two_theta) == len(sample_si_pattern.two_theta)
        assert len(processed.intensity) == len(sample_si_pattern.intensity)
        
        # Background should be reduced
        assert np.min(processed.intensity) < np.min(sample_si_pattern.intensity)
        
        # Check smoothing effect
        original_noise = np.std(np.diff(sample_si_pattern.intensity))
        processed_noise = np.std(np.diff(processed.intensity))
        assert processed_noise < original_noise
    
    def test_peak_finding(self, xrd_analyzer, sample_si_pattern):
        """Test peak finding algorithm"""
        # Find peaks
        peaks = xrd_analyzer.find_peaks(sample_si_pattern)
        
        # Should find major Si peaks
        assert len(peaks) >= 3
        
        # Check peak properties
        for peak in peaks:
            assert isinstance(peak, Peak)
            assert peak.position > 0
            assert peak.d_spacing > 0
            assert peak.intensity > 0
            assert peak.fwhm > 0
            assert peak.area > 0
        
        # First peak should be around 28.4° (Si 111)
        first_peak = min(peaks, key=lambda p: p.position)
        assert 27 < first_peak.position < 30
    
    def test_peak_fitting(self, xrd_analyzer, sample_si_pattern, sample_peaks):
        """Test peak profile fitting"""
        # Test different profile functions
        profiles = [
            PeakProfile.GAUSSIAN,
            PeakProfile.LORENTZIAN,
            PeakProfile.PSEUDO_VOIGT
        ]
        
        for profile in profiles:
            result = xrd_analyzer.fit_peaks(
                sample_si_pattern,
                sample_peaks[:3],
                profile=profile
            )
            
            # Check result structure
            assert 'peaks' in result
            assert 'fitted_pattern' in result
            assert 'r_wp' in result
            assert 'chi_squared' in result
            
            # Check fitted peaks
            assert len(result['peaks']) == 3
            
            for peak in result['peaks']:
                assert 'position' in peak
                assert 'intensity' in peak
                assert 'fwhm' in peak
                assert 'area' in peak
            
            # Check fit quality
            assert result['r_wp'] < 50  # Reasonable R_wp
            assert len(result['fitted_pattern']) == len(sample_si_pattern.intensity)
    
    def test_phase_identification(self, xrd_analyzer, sample_si_pattern):
        """Test phase identification"""
        # Process and find peaks
        processed = xrd_analyzer.process_pattern(sample_si_pattern)
        peaks = xrd_analyzer.find_peaks(processed)
        
        # Identify phases
        phases = xrd_analyzer.identify_phases(processed, peaks)
        
        # Should identify Si
        assert len(phases) > 0
        
        # First match should be Si
        best_match = phases[0]
        assert isinstance(best_match, PhaseIdentification)
        assert 'Silicon' in best_match.phase_name or 'Si' in best_match.formula
        assert best_match.crystal_system == 'cubic'
        assert best_match.score > 50  # Good match score
        assert len(best_match.matched_peaks) >= 3
    
    def test_crystallite_size_scherrer(self, xrd_analyzer, sample_peaks):
        """Test crystallite size calculation using Scherrer equation"""
        wavelength = 1.5418  # Cu Kα
        
        result = xrd_analyzer.calculate_crystallite_size(
            sample_peaks,
            wavelength,
            shape_factor=0.9
        )
        
        # Check result structure
        assert 'mean_size' in result
        assert 'std_size' in result
        assert 'min_size' in result
        assert 'max_size' in result
        
        # Check reasonable values
        assert 1 < result['mean_size'] < 1000  # nm
        assert result['std_size'] >= 0
        assert result['min_size'] <= result['mean_size'] <= result['max_size']
    
    def test_williamson_hall_analysis(self, xrd_analyzer, sample_peaks):
        """Test Williamson-Hall analysis for size and strain"""
        wavelength = 1.5418
        
        result = xrd_analyzer.williamson_hall_analysis(sample_peaks, wavelength)
        
        # Check result structure
        assert 'crystallite_size' in result
        assert 'microstrain' in result
        assert 'strain_percent' in result
        assert 'r_squared' in result
        
        # Check reasonable values
        if result['crystallite_size'] != np.inf:
            assert 1 < result['crystallite_size'] < 1000  # nm
        assert 0 <= abs(result['microstrain']) < 0.1  # Reasonable strain
        assert 0 <= result['r_squared'] <= 1
    
    def test_strain_calculation(self, xrd_analyzer, sample_peaks):
        """Test microstrain calculation"""
        # Create reference peaks (unstrained)
        ref_peaks = [
            {'position': 28.443, 'intensity': 1.0},
            {'position': 47.302, 'intensity': 0.6},
            {'position': 56.122, 'intensity': 0.35}
        ]
        
        result = xrd_analyzer.calculate_strain(
            sample_peaks[:3],
            ref_peaks,
            young_modulus=169  # GPa for Si
        )
        
        # Check result structure
        assert 'mean_strain' in result
        assert 'std_strain' in result
        assert 'mean_stress' in result
        assert 'type' in result
        
        # Check strain type
        assert result['type'] in ['tensile', 'compressive']
    
    def test_texture_analysis(self, xrd_analyzer, sample_si_pattern, sample_peaks):
        """Test texture coefficient calculation"""
        # Get reference intensities
        ref_intensities = [1000, 600, 350, 100, 150]
        
        result = xrd_analyzer.calculate_texture(
            sample_si_pattern,
            sample_peaks,
            ref_intensities
        )
        
        # Check result structure
        assert 'texture_coefficients' in result
        assert 'texture_index' in result
        assert 'is_textured' in result
        assert 'preferred_orientation' in result
        
        # Check values
        assert len(result['texture_coefficients']) == len(sample_peaks)
        assert result['texture_index'] >= 0
        assert isinstance(result['is_textured'], bool)
    
    def test_residual_stress_sin2psi(self, xrd_analyzer):
        """Test residual stress calculation using sin²ψ method"""
        # Simulate measurements at different tilt angles
        d0 = 3.1355  # Unstressed d-spacing
        measurements = [
            (0, d0 + 0.0001),
            (15, d0 + 0.0002),
            (30, d0 + 0.0004),
            (45, d0 + 0.0007),
            (60, d0 + 0.0010)
        ]
        
        result = xrd_analyzer.residual_stress_sin2psi(
            measurements,
            d0,
            young_modulus=169,
            poisson_ratio=0.22
        )
        
        # Check result structure
        assert 'stress' in result
        assert 'stress_mpa' in result
        assert 'type' in result
        assert 'error' in result
        assert 'r_squared' in result
        
        # Check stress calculation
        assert result['type'] in ['tensile', 'compressive']
        assert 0 <= result['r_squared'] <= 1
        
        # With increasing d-spacing, should be tensile
        assert result['stress'] > 0
        assert result['type'] == 'tensile'
    
    def test_mixture_analysis(self, xrd_analyzer, sample_mixture_pattern):
        """Test analysis of multi-phase mixture"""
        # Process pattern
        processed = xrd_analyzer.process_pattern(sample_mixture_pattern)
        
        # Find peaks
        peaks = xrd_analyzer.find_peaks(processed)
        
        # Should find peaks from both phases
        assert len(peaks) > 5
        
        # Identify phases
        phases = xrd_analyzer.identify_phases(processed, peaks)
        
        # Should identify both Si and SiO2
        phase_names = [p.formula for p in phases]
        assert any('Si' in name for name in phase_names)
        # May or may not identify SiO2 depending on peak overlap
    
    def test_rietveld_refinement_simplified(self, xrd_analyzer, sample_si_pattern):
        """Test simplified Rietveld refinement"""
        structure = xrd_analyzer.phase_database['Si']
        
        result = xrd_analyzer.rietveld_refinement_simplified(
            sample_si_pattern,
            structure
        )
        
        # Check result structure
        assert 'calculated_pattern' in result
        assert 'scale_factor' in result
        assert 'r_wp' in result
        assert 'chi_squared' in result
        assert 'converged' in result
        
        # Check calculated pattern
        assert len(result['calculated_pattern']) == len(sample_si_pattern.intensity)
        
        # Scale factor should be positive
        assert result['scale_factor'] > 0


class TestXRDSimulator:
    """Test suite for XRD pattern simulator"""
    
    def test_pattern_generation(self, xrd_simulator):
        """Test synthetic pattern generation"""
        pattern = xrd_simulator.generate_pattern(
            phase='Si',
            crystallite_size=100,
            microstrain=0.001,
            noise_level=0.05
        )
        
        # Check pattern structure
        assert isinstance(pattern, XRDPattern)
        assert len(pattern.two_theta) > 0
        assert len(pattern.intensity) == len(pattern.two_theta)
        
        # Check 2θ range
        assert pattern.two_theta[0] == 20
        assert pattern.two_theta[-1] < 80
        
        # Check intensity is positive
        assert np.all(pattern.intensity >= 0)
        
        # Check metadata
        assert pattern.metadata['phase'] == 'Si'
        assert pattern.metadata['crystallite_size'] == 100
    
    def test_different_phases(self, xrd_simulator):
        """Test pattern generation for different phases"""
        phases = ['Si', 'GaAs', 'GaN_hex', 'SiO2_quartz', 'Al2O3']
        
        for phase in phases:
            pattern = xrd_simulator.generate_pattern(phase=phase)
            
            # Should generate valid pattern
            assert isinstance(pattern, XRDPattern)
            assert len(pattern.intensity) > 0
            assert pattern.metadata['phase'] == phase
            
            # Should have peaks
            assert np.max(pattern.intensity) > np.mean(pattern.intensity) * 2
    
    def test_crystallite_size_effect(self, xrd_simulator):
        """Test effect of crystallite size on peak width"""
        # Generate patterns with different crystallite sizes
        small = xrd_simulator.generate_pattern(phase='Si', crystallite_size=10)
        large = xrd_simulator.generate_pattern(phase='Si', crystallite_size=200)
        
        # Find peaks
        analyzer = XRDAnalyzer()
        peaks_small = analyzer.find_peaks(small)
        peaks_large = analyzer.find_peaks(large)
        
        # Smaller crystallites should give broader peaks
        if len(peaks_small) > 0 and len(peaks_large) > 0:
            avg_fwhm_small = np.mean([p.fwhm for p in peaks_small])
            avg_fwhm_large = np.mean([p.fwhm for p in peaks_large])
            assert avg_fwhm_small > avg_fwhm_large
    
    def test_microstrain_effect(self, xrd_simulator):
        """Test effect of microstrain on pattern"""
        # Generate patterns with different microstrain
        no_strain = xrd_simulator.generate_pattern(phase='Si', microstrain=0)
        with_strain = xrd_simulator.generate_pattern(phase='Si', microstrain=0.005)
        
        # Find peaks
        analyzer = XRDAnalyzer()
        peaks_no_strain = analyzer.find_peaks(no_strain)
        peaks_strain = analyzer.find_peaks(with_strain)
        
        # Strain should broaden peaks and shift positions
        if len(peaks_no_strain) > 0 and len(peaks_strain) > 0:
            # Check broadening
            avg_fwhm_no_strain = np.mean([p.fwhm for p in peaks_no_strain])
            avg_fwhm_strain = np.mean([p.fwhm for p in peaks_strain])
            assert avg_fwhm_strain >= avg_fwhm_no_strain
    
    def test_texture_effect(self, xrd_simulator):
        """Test effect of texture on relative intensities"""
        # Generate patterns with different texture
        random = xrd_simulator.generate_pattern(phase='Si', texture_coefficient=1.0)
        textured = xrd_simulator.generate_pattern(phase='Si', texture_coefficient=3.0)
        
        # Find peaks
        analyzer = XRDAnalyzer()
        peaks_random = analyzer.find_peaks(random)
        peaks_textured = analyzer.find_peaks(textured)
        
        # Texture should change relative intensities
        if len(peaks_random) > 1 and len(peaks_textured) > 1:
            # First peak should be enhanced in textured sample
            ratio_random = peaks_random[0].intensity / peaks_random[1].intensity
            ratio_textured = peaks_textured[0].intensity / peaks_textured[1].intensity
            # Ratios should be different
            assert abs(ratio_textured - ratio_random) > 0.1
    
    def test_mixture_generation(self, xrd_simulator):
        """Test mixture pattern generation"""
        mixture = xrd_simulator.generate_mixture([
            ('Si', 0.7),
            ('GaAs', 0.3)
        ])
        
        # Check pattern
        assert isinstance(mixture, XRDPattern)
        assert len(mixture.intensity) > 0
        
        # Check metadata
        assert 'phases' in mixture.metadata
        assert len(mixture.metadata['phases']) == 2
        
        # Should have more peaks than single phase
        analyzer = XRDAnalyzer()
        peaks_mixture = analyzer.find_peaks(mixture)
        peaks_si = analyzer.find_peaks(
            xrd_simulator.generate_pattern(phase='Si')
        )
        
        # Mixture should have at least as many peaks
        assert len(peaks_mixture) >= len(peaks_si)
    
    def test_reproducibility(self):
        """Test simulator reproducibility with same seed"""
        sim1 = XRDSimulator(seed=123)
        sim2 = XRDSimulator(seed=123)
        
        pattern1 = sim1.generate_pattern('Si')
        pattern2 = sim2.generate_pattern('Si')
        
        # Should be identical
        np.testing.assert_array_equal(pattern1.two_theta, pattern2.two_theta)
        np.testing.assert_array_equal(pattern1.intensity, pattern2.intensity)


class TestCrystalStructure:
    """Test suite for crystal structure handling"""
    
    def test_unit_cell_volume_cubic(self, xrd_analyzer):
        """Test unit cell volume calculation for cubic system"""
        si_structure = xrd_analyzer.phase_database['Si']
        
        volume = si_structure.get_unit_cell_volume()
        
        # Si cubic: V = a³
        expected = 5.43095 ** 3
        assert np.isclose(volume, expected, rtol=0.01)
    
    def test_unit_cell_volume_hexagonal(self, xrd_analyzer):
        """Test unit cell volume calculation for hexagonal system"""
        gan_structure = xrd_analyzer.phase_database['GaN_hex']
        
        volume = gan_structure.get_unit_cell_volume()
        
        # Hexagonal: V = a²c·sin(120°)
        a = 3.189
        c = 5.185
        expected = a * a * c * np.sin(np.radians(120))
        assert np.isclose(volume, expected, rtol=0.01)
    
    def test_crystal_system_enum(self):
        """Test crystal system enumeration"""
        systems = [s.value for s in CrystalSystem]
        
        expected = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal',
                   'trigonal', 'monoclinic', 'triclinic']
        
        assert set(systems) == set(expected)
    
    def test_reference_peaks_calculation(self, xrd_analyzer):
        """Test reference peak calculation for different structures"""
        for phase_name, structure in xrd_analyzer.phase_database.items():
            peaks = xrd_analyzer._calculate_reference_peaks(
                structure,
                wavelength=1.5418
            )
            
            # Should generate peaks
            assert len(peaks) > 0
            
            # Check peak properties
            for peak in peaks:
                assert 'position' in peak
                assert 'intensity' in peak
                assert 'd_spacing' in peak
                assert 'hkl' in peak
                
                # Physical checks
                assert 0 < peak['position'] < 180
                assert peak['intensity'] > 0
                assert peak['d_spacing'] > 0


class TestIntegration:
    """Integration tests for complete XRD workflows"""
    
    @pytest.mark.integration
    def test_complete_si_analysis(self, xrd_analyzer, xrd_simulator):
        """Test complete analysis workflow for Si"""
        # Generate pattern
        pattern = xrd_simulator.generate_pattern(
            phase='Si',
            crystallite_size=45,
            microstrain=0.0015,
            texture_coefficient=1.2
        )
        
        # Process pattern
        processed = xrd_analyzer.process_pattern(pattern)
        
        # Find peaks
        peaks = xrd_analyzer.find_peaks(processed)
        assert len(peaks) >= 3
        
        # Identify phase
        phases = xrd_analyzer.identify_phases(processed, peaks)
        assert len(phases) > 0
        assert 'Si' in phases[0].formula
        
        # Calculate crystallite size
        size_result = xrd_analyzer.calculate_crystallite_size(
            peaks, pattern.wavelength
        )
        assert 20 < size_result['mean_size'] < 100  # Should be close to 45 nm
        
        # Williamson-Hall analysis
        wh_result = xrd_analyzer.williamson_hall_analysis(peaks, pattern.wavelength)
        assert 20 < wh_result['crystallite_size'] < 100
        assert 0.0005 < abs(wh_result['microstrain']) < 0.005
    
    @pytest.mark.integration
    def test_mixture_phase_identification(self, xrd_analyzer, xrd_simulator):
        """Test phase identification for mixture"""
        # Generate two-phase mixture
        mixture = xrd_simulator.generate_mixture([
            ('Si', 0.6),
            ('GaAs', 0.4)
        ])
        
        # Full analysis
        processed = xrd_analyzer.process_pattern(mixture)
        peaks = xrd_analyzer.find_peaks(processed)
        phases = xrd_analyzer.identify_phases(processed, peaks)
        
        # Should identify at least one phase
        assert len(phases) > 0
        
        # Check phase names
        phase_formulas = [p.formula for p in phases[:3]]
        # Si should be identified (dominant phase)
        assert any('Si' in f for f in phase_formulas)
    
    @pytest.mark.integration
    def test_stress_analysis_workflow(self, xrd_analyzer):
        """Test complete stress analysis workflow"""
        # Simulate stressed sample measurements
        d0 = 3.1355  # Unstressed Si (111)
        stress_mpa = 200  # Applied stress
        
        # Generate sin²ψ measurements
        measurements = []
        for psi in [0, 15, 30, 45, 60]:
            # Simulate d-spacing change with tilt
            strain = stress_mpa / 169000 * (1 + 0.22) * np.sin(np.radians(psi))**2
            d = d0 * (1 + strain)
            measurements.append((psi, d))
        
        # Analyze stress
        result = xrd_analyzer.residual_stress_sin2psi(
            measurements, d0,
            young_modulus=169,
            poisson_ratio=0.22
        )
        
        # Should recover stress within error
        assert 150 < result['stress_mpa'] < 250
        assert result['type'] == 'tensile'
        assert result['r_squared'] > 0.9
    
    @pytest.mark.integration
    def test_peak_fitting_workflow(self, xrd_analyzer, xrd_simulator):
        """Test complete peak fitting workflow"""
        # Generate pattern
        pattern = xrd_simulator.generate_pattern('Si')
        
        # Process and find peaks
        processed = xrd_analyzer.process_pattern(pattern)
        peaks = xrd_analyzer.find_peaks(processed)
        
        # Fit with different profiles
        profiles = [PeakProfile.GAUSSIAN, PeakProfile.PSEUDO_VOIGT]
        
        for profile in profiles:
            result = xrd_analyzer.fit_peaks(processed, peaks[:3], profile)
            
            # Check fit quality
            assert result['r_wp'] < 50
            assert result['chi_squared'] < 10
            
            # Check fitted parameters are reasonable
            for i, fitted_peak in enumerate(result['peaks']):
                # Position should be close to original
                assert abs(fitted_peak['position'] - peaks[i].position) < 0.5
                # FWHM should be positive and reasonable
                assert 0.05 < fitted_peak['fwhm'] < 1.0


class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.performance
    def test_large_pattern_processing(self, xrd_analyzer):
        """Test processing of high-resolution pattern"""
        # Create large pattern (0.001° step)
        two_theta = np.arange(10, 120, 0.001)
        intensity = 100 + 50 * np.random.randn(len(two_theta))
        
        # Add some peaks
        for pos in [28.4, 47.3, 56.1]:
            idx = np.argmin(np.abs(two_theta - pos))
            intensity[idx-50:idx+50] += 1000 * np.exp(
                -0.5 * ((two_theta[idx-50:idx+50] - pos) / 0.1)**2
            )
        
        pattern = XRDPattern(
            two_theta=two_theta,
            intensity=intensity,
            wavelength=1.5418
        )
        
        # Process pattern
        start_time = time.time()
        processed = xrd_analyzer.process_pattern(pattern)
        process_time = time.time() - start_time
        
        # Should complete quickly even for large pattern
        assert process_time < 5.0  # seconds
        
        # Find peaks
        start_time = time.time()
        peaks = xrd_analyzer.find_peaks(processed)
        peak_time = time.time() - start_time
        
        assert peak_time < 2.0  # seconds
        assert len(peaks) > 0
    
    @pytest.mark.performance
    def test_many_peaks_fitting(self, xrd_analyzer):
        """Test fitting many peaks simultaneously"""
        # Create pattern with many peaks
        two_theta = np.arange(20, 80, 0.02)
        intensity = np.ones_like(two_theta) * 50
        
        # Add 20 peaks
        peaks = []
        for i in range(20):
            pos = 25 + i * 2.5
            amplitude = 500 + np.random.rand() * 500
            sigma = 0.1 + np.random.rand() * 0.1
            
            intensity += amplitude * np.exp(-0.5 * ((two_theta - pos) / sigma)**2)
            peaks.append(Peak(
                position=pos,
                d_spacing=1.5418 / (2 * np.sin(np.radians(pos/2))),
                intensity=amplitude,
                fwhm=sigma * 2.355,
                area=amplitude * sigma * np.sqrt(2 * np.pi)
            ))
        
        pattern = XRDPattern(
            two_theta=two_theta,
            intensity=intensity,
            wavelength=1.5418
        )
        
        # Fit all peaks
        start_time = time.time()
        result = xrd_analyzer.fit_peaks(pattern, peaks[:10], PeakProfile.GAUSSIAN)
        fit_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert fit_time < 10.0  # seconds
        assert len(result['peaks']) == 10
    
    @pytest.mark.performance
    def test_phase_database_search(self, xrd_analyzer):
        """Test phase identification performance"""
        # Generate complex pattern
        simulator = XRDSimulator()
        pattern = simulator.generate_mixture([
            ('Si', 0.3),
            ('GaAs', 0.3),
            ('SiO2_quartz', 0.2),
            ('Al2O3', 0.2)
        ])
        
        # Find peaks
        peaks = xrd_analyzer.find_peaks(pattern)
        
        # Identify phases
        start_time = time.time()
        phases = xrd_analyzer.identify_phases(pattern, peaks)
        search_time = time.time() - start_time
        
        # Should complete quickly
        assert search_time < 2.0  # seconds
        assert len(phases) > 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_pattern(self, xrd_analyzer):
        """Test handling of empty pattern"""
        empty_pattern = XRDPattern(
            two_theta=np.array([]),
            intensity=np.array([]),
            wavelength=1.5418
        )
        
        # Should handle gracefully
        peaks = xrd_analyzer.find_peaks(empty_pattern)
        assert len(peaks) == 0
        
        phases = xrd_analyzer.identify_phases(empty_pattern)
        assert len(phases) == 0
    
    def test_no_peaks_pattern(self, xrd_analyzer):
        """Test pattern with no peaks"""
        # Flat pattern
        flat_pattern = XRDPattern(
            two_theta=np.linspace(20, 80, 100),
            intensity=np.ones(100) * 100,
            wavelength=1.5418
        )
        
        peaks = xrd_analyzer.find_peaks(flat_pattern)
        assert len(peaks) == 0
        
        # Should handle empty peak list
        size_result = xrd_analyzer.calculate_crystallite_size([], 1.5418)
        assert size_result['mean_size'] == 0
    
    def test_single_peak_williamson_hall(self, xrd_analyzer):
        """Test Williamson-Hall with insufficient peaks"""
        single_peak = [Peak(
            position=28.4,
            d_spacing=3.136,
            intensity=1000,
            fwhm=0.15,
            area=150
        )]
        
        # Should handle gracefully
        result = xrd_analyzer.williamson_hall_analysis(single_peak, 1.5418)
        assert result['crystallite_size'] == 0
        assert result['microstrain'] == 0
    
    def test_invalid_wavelength(self, xrd_analyzer):
        """Test handling of invalid wavelength"""
        pattern = XRDPattern(
            two_theta=np.linspace(20, 80, 100),
            intensity=np.random.rand(100) * 100,
            wavelength=0  # Invalid
        )
        
        # Should not crash
        peaks = xrd_analyzer.find_peaks(pattern)
        # d-spacing calculation will fail but should handle it
        for peak in peaks:
            if pattern.wavelength == 0:
                assert np.isinf(peak.d_spacing) or np.isnan(peak.d_spacing)
    
    def test_stress_with_insufficient_data(self, xrd_analyzer):
        """Test stress analysis with too few measurements"""
        # Only one measurement
        measurements = [(0, 3.136)]
        
        result = xrd_analyzer.residual_stress_sin2psi(
            measurements, 3.136,
            young_modulus=169,
            poisson_ratio=0.22
        )
        
        # Should return zero stress
        assert result['stress'] == 0
        assert result['error'] == np.inf


class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_pattern_data_validation(self):
        """Test XRDPattern data validation"""
        # Mismatched array lengths should raise error
        with pytest.raises(ValueError):
            XRDPattern(
                two_theta=np.array([1, 2, 3]),
                intensity=np.array([1, 2]),  # Wrong length
                wavelength=1.5418
            )
    
    def test_pattern_sorting(self):
        """Test automatic sorting of pattern data"""
        # Create unsorted pattern
        pattern = XRDPattern(
            two_theta=np.array([30, 20, 40, 10]),
            intensity=np.array([100, 200, 150, 50]),
            wavelength=1.5418
        )
        
        # Should be automatically sorted
        assert np.all(np.diff(pattern.two_theta) > 0)
        assert pattern.two_theta[0] == 10
        assert pattern.intensity[0] == 50
    
    def test_negative_intensity_handling(self):
        """Test handling of negative intensities"""
        pattern = XRDPattern(
            two_theta=np.linspace(20, 80, 100),
            intensity=np.random.randn(100) * 100 - 50,  # Some negative
            wavelength=1.5418
        )
        
        # Should clip negative values
        assert np.all(pattern.intensity >= 0)


# Benchmark function
def run_benchmarks():
    """Run performance benchmarks"""
    import cProfile
    import pstats
    from io import StringIO
    
    analyzer = XRDAnalyzer()
    simulator = XRDSimulator()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Benchmark pattern processing
    for _ in range(10):
        pattern = simulator.generate_pattern('Si')
        processed = analyzer.process_pattern(pattern)
        peaks = analyzer.find_peaks(processed)
    
    # Benchmark phase identification
    for _ in range(5):
        phases = analyzer.identify_phases(processed, peaks)
    
    # Benchmark fitting
    for _ in range(5):
        analyzer.fit_peaks(processed, peaks[:5], PeakProfile.PSEUDO_VOIGT)
    
    # Benchmark crystallite size
    for _ in range(10):
        analyzer.calculate_crystallite_size(peaks, 1.5418)
        analyzer.williamson_hall_analysis(peaks, 1.5418)
    
    profiler.disable()
    
    # Generate report
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(15)
    
    print("\nSession 9 XRD Analysis Performance Benchmark:")
    print(stream.getvalue())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("Running Performance Benchmarks")
    print("=" * 80)
    run_benchmarks()
