"""
Session 8: Integration Tests for Ellipsometry, PL, and Raman
=============================================================
Comprehensive test suite for advanced optical characterization methods
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
from session8_complete_implementation import (
    EllipsometryAnalyzer, PhotoluminescenceAnalyzer, RamanAnalyzer,
    OpticalTestDataGeneratorII, EllipsometryData, LayerStack,
    PLSpectrum, RamanSpectrum, DispersionModel, PLMeasurementType, RamanMode
)

# Test fixtures
@pytest.fixture
def ellipsometry_analyzer():
    """Create ellipsometry analyzer instance"""
    return EllipsometryAnalyzer()

@pytest.fixture
def pl_analyzer():
    """Create PL analyzer instance"""
    return PhotoluminescenceAnalyzer()

@pytest.fixture
def raman_analyzer():
    """Create Raman analyzer instance"""
    return RamanAnalyzer()

@pytest.fixture
def data_generator():
    """Create test data generator"""
    return OpticalTestDataGeneratorII(seed=42)

@pytest.fixture
def sample_layer_stack():
    """Create sample layer stack for ellipsometry"""
    return LayerStack(
        layers=[
            {
                'thickness': 100,
                'model': DispersionModel.CAUCHY,
                'params': {'A': 1.46, 'B': 0.00354, 'C': 0, 'k': 0}
            }
        ],
        substrate={'n': 3.85, 'k': 0.02}
    )

@pytest.fixture
def sample_ellipsometry_data(data_generator, sample_layer_stack):
    """Generate sample ellipsometry data"""
    return data_generator.generate_ellipsometry_data(sample_layer_stack)

@pytest.fixture
def sample_pl_spectrum(data_generator):
    """Generate sample PL spectrum"""
    return data_generator.generate_pl_spectrum('GaAs', temperature=10)

@pytest.fixture
def sample_raman_spectrum(data_generator):
    """Generate sample Raman spectrum"""
    return data_generator.generate_raman_spectrum('Si', stress=0)


class TestEllipsometryAnalyzer:
    """Test suite for ellipsometry analyzer"""
    
    def test_fresnel_coefficients(self, ellipsometry_analyzer):
        """Test Fresnel coefficient calculation"""
        n1 = complex(1.0, 0)  # Air
        n2 = complex(1.5, 0)  # Glass
        theta1 = 45  # degrees
        
        r_p, r_s = ellipsometry_analyzer.fresnel_coefficients(n1, n2, theta1)
        
        # Check that coefficients are complex
        assert isinstance(r_p, complex)
        assert isinstance(r_s, complex)
        
        # Check magnitude is <= 1
        assert abs(r_p) <= 1
        assert abs(r_s) <= 1
    
    def test_calculate_rho(self, ellipsometry_analyzer):
        """Test complex reflectance ratio calculation"""
        psi = np.array([45, 50, 55])  # degrees
        delta = np.array([180, 170, 160])  # degrees
        
        rho = ellipsometry_analyzer.calculate_rho(psi, delta)
        
        # Check output shape
        assert len(rho) == len(psi)
        
        # Check that rho is complex
        assert np.iscomplexobj(rho)
        
        # Verify calculation
        expected = np.tan(np.radians(psi)) * np.exp(1j * np.radians(delta))
        np.testing.assert_allclose(rho, expected)
    
    def test_transfer_matrix_single_layer(self, ellipsometry_analyzer, sample_layer_stack):
        """Test transfer matrix method for single layer"""
        wavelength = 633  # nm
        angle = 70  # degrees
        
        r_p, r_s = ellipsometry_analyzer.transfer_matrix_method(
            wavelength, sample_layer_stack, angle
        )
        
        # Check that results are complex
        assert isinstance(r_p, complex)
        assert isinstance(r_s, complex)
        
        # Check physical bounds
        assert abs(r_p) <= 1
        assert abs(r_s) <= 1
    
    def test_calculate_psi_delta(self, ellipsometry_analyzer, sample_layer_stack):
        """Test Psi and Delta calculation for layer stack"""
        wavelengths = np.linspace(400, 800, 50)
        angle = 70
        
        psi, delta = ellipsometry_analyzer.calculate_psi_delta(
            wavelengths, sample_layer_stack, angle
        )
        
        # Check output shape
        assert len(psi) == len(wavelengths)
        assert len(delta) == len(wavelengths)
        
        # Check physical bounds
        assert np.all(psi >= 0) and np.all(psi <= 90)
        assert np.all(delta >= -180) and np.all(delta <= 180)
    
    def test_model_fitting(self, ellipsometry_analyzer, sample_ellipsometry_data, sample_layer_stack):
        """Test ellipsometry model fitting"""
        # Fit thickness only
        fit_params = ['layer0_thickness']
        
        result = ellipsometry_analyzer.fit_model(
            sample_ellipsometry_data,
            sample_layer_stack,
            fit_params
        )
        
        # Check result structure
        assert 'stack' in result
        assert 'parameters' in result
        assert 'psi_fit' in result
        assert 'delta_fit' in result
        assert 'mse' in result
        assert 'r_squared' in result
        
        # Check fit quality metrics
        assert result['mse'] >= 0
        assert 0 <= result['r_squared'] <= 1
        
        # Check that fitted parameters are reasonable
        fitted_thickness = result['parameters']['layer0_thickness']
        assert 0 < fitted_thickness < 10000  # nm
    
    def test_dispersion_models(self, ellipsometry_analyzer):
        """Test different dispersion models"""
        wavelength = 633  # nm
        
        # Test Cauchy model
        cauchy_params = {'A': 1.45, 'B': 0.01, 'C': 0.001, 'k': 0}
        n_cauchy = ellipsometry_analyzer._cauchy_model(wavelength, cauchy_params)
        assert n_cauchy.real > 1
        assert n_cauchy.imag >= 0
        
        # Test Sellmeier model
        sellmeier_params = {'B1': 1.0, 'C1': 0.01, 'B2': 0.5, 'C2': 0.02, 'k': 0}
        n_sellmeier = ellipsometry_analyzer._sellmeier_model(wavelength, sellmeier_params)
        assert n_sellmeier.real > 1
        
        # Test Tauc-Lorentz model
        tl_params = {'A': 100, 'E0': 3.5, 'C': 1, 'Eg': 1.5, 'eps_inf': 1}
        n_tl = ellipsometry_analyzer._tauc_lorentz_model(wavelength, tl_params)
        assert n_tl.real > 0
    
    def test_multilayer_stack(self, ellipsometry_analyzer):
        """Test analysis of multilayer stack"""
        # Create 3-layer stack
        stack = LayerStack(
            layers=[
                {'thickness': 50, 'model': DispersionModel.CAUCHY, 
                 'params': {'A': 1.46, 'B': 0.003, 'C': 0, 'k': 0}},
                {'thickness': 100, 'model': DispersionModel.CAUCHY,
                 'params': {'A': 2.0, 'B': 0.01, 'C': 0, 'k': 0.1}},
                {'thickness': 75, 'model': DispersionModel.CAUCHY,
                 'params': {'A': 1.7, 'B': 0.005, 'C': 0, 'k': 0.05}}
            ],
            substrate={'n': 3.85, 'k': 0.02}
        )
        
        # Calculate optical response
        wavelengths = np.linspace(400, 800, 100)
        psi, delta = ellipsometry_analyzer.calculate_psi_delta(wavelengths, stack, 70)
        
        # Check for interference oscillations
        psi_gradient = np.gradient(psi)
        sign_changes = np.sum(np.diff(np.sign(psi_gradient)) != 0)
        assert sign_changes > 2  # Should see oscillations from multilayer


class TestPhotoluminescenceAnalyzer:
    """Test suite for photoluminescence analyzer"""
    
    def test_spectrum_processing(self, pl_analyzer, sample_pl_spectrum):
        """Test PL spectrum processing"""
        processed = pl_analyzer.process_spectrum(sample_pl_spectrum, smooth=True)
        
        # Check that spectrum is processed
        assert isinstance(processed, PLSpectrum)
        assert len(processed.wavelength) == len(sample_pl_spectrum.wavelength)
        
        # Check normalization by integration time and power
        expected_factor = 1 / (sample_pl_spectrum.integration_time * sample_pl_spectrum.excitation_power)
        intensity_ratio = processed.intensity[0] / sample_pl_spectrum.intensity[0]
        assert np.isclose(intensity_ratio, expected_factor, rtol=0.01)
    
    def test_peak_finding(self, pl_analyzer, sample_pl_spectrum):
        """Test PL peak finding"""
        peaks = pl_analyzer.find_peaks(sample_pl_spectrum)
        
        # Check result structure
        assert 'wavelengths' in peaks
        assert 'energies' in peaks
        assert 'intensities' in peaks
        assert 'fwhm_nm' in peaks
        assert 'fwhm_meV' in peaks
        assert 'types' in peaks
        assert 'count' in peaks
        
        # Should find at least one peak
        assert peaks['count'] > 0
        
        # Check energy conversion
        for wl, energy in zip(peaks['wavelengths'], peaks['energies']):
            assert np.isclose(energy, 1240 / wl, rtol=0.001)
        
        # Check peak types assignment
        assert len(peaks['types']) == peaks['count']
    
    def test_peak_fitting(self, pl_analyzer, sample_pl_spectrum):
        """Test multi-peak fitting"""
        result = pl_analyzer.fit_peaks(sample_pl_spectrum, n_peaks=2)
        
        # Check result structure
        assert 'peaks' in result
        assert 'fitted_spectrum' in result
        assert 'background' in result
        assert 'r_squared' in result
        
        # Check fitted peaks
        if len(result['peaks']) > 0:
            for peak in result['peaks']:
                assert 'amplitude' in peak
                assert 'energy' in peak
                assert 'wavelength' in peak
                assert 'sigma_eV' in peak
                assert 'fwhm_meV' in peak
                assert 'area' in peak
                
                # Check physical bounds
                assert peak['energy'] > 0
                assert peak['wavelength'] > 0
                assert peak['fwhm_meV'] > 0
        
        # Check fit quality
        assert 0 <= result['r_squared'] <= 1
    
    def test_quantum_yield_calculation(self, pl_analyzer, sample_pl_spectrum):
        """Test quantum yield calculation"""
        absorption = 0.9  # 90% absorption
        reference_qy = 0.95  # Reference QY
        
        qy = pl_analyzer.calculate_quantum_yield(
            sample_pl_spectrum, 
            absorption, 
            reference_qy
        )
        
        # Check bounds
        assert 0 <= qy <= 1
    
    def test_temperature_series_analysis(self, pl_analyzer, data_generator):
        """Test temperature-dependent PL analysis"""
        # Generate temperature series
        temperatures = [10, 50, 100, 150, 200, 250, 293]
        spectra = [data_generator.generate_pl_spectrum('GaAs', T) for T in temperatures]
        
        # Analyze series
        result = pl_analyzer.analyze_temperature_series(spectra)
        
        # Check result structure
        assert 'temperatures' in result
        assert 'peak_positions' in result
        assert 'peak_intensities' in result
        assert 'integrated_intensities' in result
        assert 'varshni_params' in result
        assert 'activation_energy' in result
        
        # Check Varshni parameters
        assert 'Eg0' in result['varshni_params']
        assert 'alpha' in result['varshni_params']
        assert 'beta' in result['varshni_params']
        
        # Physical checks
        assert result['varshni_params']['Eg0'] > 0  # Positive bandgap at 0K
        assert result['activation_energy'] >= 0  # Positive activation energy
        
        # Check temperature quenching (intensity should decrease with T)
        intensities = [i for i in result['peak_intensities'] if not np.isnan(i)]
        if len(intensities) > 1:
            assert intensities[0] > intensities[-1]  # Higher at low T
    
    def test_multi_gaussian_fitting(self, pl_analyzer):
        """Test multi-Gaussian peak fitting function"""
        x = np.linspace(0, 10, 100)
        # 2 peaks + background
        params = np.array([
            100, 3, 0.5,  # Peak 1: amplitude, center, sigma
            80, 7, 0.7,   # Peak 2
            1, 10         # Background: slope, offset
        ])
        
        y = pl_analyzer._multi_gaussian(x, params)
        
        # Check that peaks are at expected positions
        peak1_idx = np.argmin(np.abs(x - 3))
        peak2_idx = np.argmin(np.abs(x - 7))
        
        # Local maxima should be near peak centers
        assert y[peak1_idx] > y[peak1_idx - 5]
        assert y[peak2_idx] > y[peak2_idx - 5]
    
    def test_voigt_profile(self, pl_analyzer):
        """Test Voigt profile implementation"""
        x = np.linspace(0, 10, 100)
        params = np.array([100, 5, 0.5, 0, 0])  # Single Voigt peak
        
        y = pl_analyzer._multi_voigt(x, params)
        
        # Check that result is real
        assert np.all(np.isreal(y))
        
        # Check peak is at center
        peak_idx = np.argmax(y)
        assert np.abs(x[peak_idx] - 5) < 0.2


class TestRamanAnalyzer:
    """Test suite for Raman analyzer"""
    
    def test_spectrum_processing(self, raman_analyzer, sample_raman_spectrum):
        """Test Raman spectrum processing"""
        processed = raman_analyzer.process_spectrum(
            sample_raman_spectrum,
            baseline_correct=True,
            normalize=True
        )
        
        # Check processing
        assert isinstance(processed, RamanSpectrum)
        assert len(processed.raman_shift) == len(sample_raman_spectrum.raman_shift)
        
        # Check normalization (max should be 1)
        if np.max(processed.intensity) > 0:
            assert np.isclose(np.max(processed.intensity), 1.0, rtol=0.01)
    
    def test_peak_finding_and_identification(self, raman_analyzer, sample_raman_spectrum):
        """Test Raman peak finding and identification"""
        peaks = raman_analyzer.find_peaks(sample_raman_spectrum)
        
        # Check result structure
        assert 'positions' in peaks
        assert 'intensities' in peaks
        assert 'fwhm' in peaks
        assert 'identifications' in peaks
        
        # Should find Si peak around 520 cm⁻¹
        si_peaks = [p for p in peaks['positions'] if 510 < p < 530]
        assert len(si_peaks) > 0
        
        # Check identification
        for pos, ident in zip(peaks['positions'], peaks['identifications']):
            if 510 < pos < 530 and ident is not None:
                assert ident['material'] == 'Si'
                assert 'TO/LO' in ident['mode']
    
    def test_stress_calculation(self, raman_analyzer):
        """Test stress/strain calculation from Raman shift"""
        # Test compressive stress (blue shift)
        result_comp = raman_analyzer.calculate_stress(
            measured_position=523,  # Shifted up
            reference_position=520.5,
            material='Si'
        )
        
        assert result_comp['shift'] > 0
        assert result_comp['stress'] < 0  # Compressive
        assert result_comp['type'] == 'compressive'
        
        # Test tensile stress (red shift)
        result_tens = raman_analyzer.calculate_stress(
            measured_position=518,  # Shifted down
            reference_position=520.5,
            material='Si'
        )
        
        assert result_tens['shift'] < 0
        assert result_tens['stress'] > 0  # Tensile
        assert result_tens['type'] == 'tensile'
        
        # Check strain calculation
        assert abs(result_comp['strain']) > 0
        assert abs(result_tens['strain']) > 0
    
    def test_crystallinity_analysis(self, raman_analyzer, sample_raman_spectrum):
        """Test crystallinity analysis"""
        result = raman_analyzer.analyze_crystallinity(
            sample_raman_spectrum,
            material='Si'
        )
        
        # Check result structure
        assert 'crystallinity' in result
        assert 'grain_size_nm' in result
        assert 'quality' in result
        
        # Check bounds
        assert 0 <= result['crystallinity'] <= 1
        assert result['grain_size_nm'] >= 0
        assert result['quality'] in ['high', 'medium', 'low']
    
    def test_graphene_analysis(self, raman_analyzer, data_generator):
        """Test Raman analysis for graphene"""
        graphene_spectrum = data_generator.generate_raman_spectrum('Graphene')
        
        # Find peaks
        peaks = raman_analyzer.find_peaks(graphene_spectrum)
        
        # Should find G peak around 1580 cm⁻¹
        g_peaks = [p for p in peaks['positions'] if 1550 < p < 1610]
        assert len(g_peaks) > 0
        
        # Crystallinity analysis
        result = raman_analyzer.analyze_crystallinity(graphene_spectrum, 'Graphene')
        
        # Check D/G ratio calculation
        assert 0 <= result['crystallinity'] <= 1
    
    def test_map_analysis(self, raman_analyzer):
        """Test Raman mapping analysis"""
        # Create synthetic map data (5x5 spatial points)
        nx, ny = 5, 5
        n_points = 100
        positions = np.linspace(500, 540, n_points)
        
        # Generate map with spatial variation
        spectra_map = np.zeros((nx, ny, n_points))
        for i in range(nx):
            for j in range(ny):
                # Add spatial variation in peak position
                shift = (i - nx/2) * 0.5 + (j - ny/2) * 0.3
                peak_pos = 520 + shift
                spectra_map[i, j, :] = 1000 * np.exp(
                    -0.5 * ((positions - peak_pos) / 3)**2
                )
        
        # Analyze map
        result = raman_analyzer.map_analysis(
            spectra_map,
            positions,
            peak_of_interest=520
        )
        
        # Check result structure
        assert 'peak_position_map' in result
        assert 'peak_intensity_map' in result
        assert 'peak_width_map' in result
        assert 'position_mean' in result
        assert 'position_std' in result
        assert 'uniformity' in result
        
        # Check map dimensions
        assert result['peak_position_map'].shape == (nx, ny)
        assert result['peak_intensity_map'].shape == (nx, ny)
        
        # Check statistics
        assert result['position_std'] > 0  # Should have variation
        assert 0 <= result['uniformity'] <= 1
    
    def test_baseline_calculation(self, raman_analyzer):
        """Test baseline calculation methods"""
        # Create test spectrum with baseline
        x = np.linspace(100, 1000, 200)
        baseline_true = 50 + 0.01 * x
        peak = 1000 * np.exp(-0.5 * ((x - 520) / 10)**2)
        y = baseline_true + peak + np.random.normal(0, 5, len(x))
        
        # Calculate baseline
        baseline_calc = raman_analyzer._calculate_baseline(x, y)
        
        # Check that baseline is reasonable
        assert len(baseline_calc) == len(y)
        
        # Baseline should be lower than peaks
        peak_region = (x > 510) & (x < 530)
        assert np.mean(baseline_calc[peak_region]) < np.mean(y[peak_region])


class TestOpticalDataGeneratorII:
    """Test suite for synthetic data generator"""
    
    def test_ellipsometry_data_generation(self, data_generator):
        """Test ellipsometry data generation"""
        # Generate with default stack
        ell_data = data_generator.generate_ellipsometry_data()
        
        # Check data structure
        assert isinstance(ell_data, EllipsometryData)
        assert len(ell_data.wavelength) > 0
        assert len(ell_data.psi) == len(ell_data.wavelength)
        assert len(ell_data.delta) == len(ell_data.wavelength)
        
        # Check physical bounds
        assert np.all(ell_data.psi >= 0) and np.all(ell_data.psi <= 90)
        
        # Custom stack
        custom_stack = LayerStack(
            layers=[{'thickness': 200, 'model': DispersionModel.CAUCHY,
                    'params': {'A': 2.0, 'B': 0.01, 'C': 0, 'k': 0.1}}],
            substrate={'n': 1.5, 'k': 0}
        )
        
        custom_data = data_generator.generate_ellipsometry_data(custom_stack)
        assert custom_data.metadata['stack'] == custom_stack
    
    def test_pl_spectrum_generation(self, data_generator):
        """Test PL spectrum generation for different materials"""
        materials = ['GaAs', 'GaN', 'InP', 'CdTe']
        
        for material in materials:
            spectrum = data_generator.generate_pl_spectrum(material)
            
            # Check structure
            assert isinstance(spectrum, PLSpectrum)
            assert len(spectrum.wavelength) > 0
            assert len(spectrum.intensity) == len(spectrum.wavelength)
            
            # Check that spectrum has features
            assert np.max(spectrum.intensity) > np.min(spectrum.intensity)
            
            # Check metadata
            assert spectrum.metadata['material'] == material
    
    def test_temperature_effects_pl(self, data_generator):
        """Test temperature effects in PL generation"""
        temps = [10, 100, 293]
        spectra = [data_generator.generate_pl_spectrum('GaAs', T) for T in temps]
        
        # Peak intensity should decrease with temperature
        max_intensities = [np.max(s.intensity) for s in spectra]
        assert max_intensities[0] > max_intensities[-1]
        
        # Peak should red-shift with temperature
        peak_indices = [np.argmax(s.intensity) for s in spectra]
        peak_wavelengths = [s.wavelength[i] for s, i in zip(spectra, peak_indices)]
        assert peak_wavelengths[0] < peak_wavelengths[-1]
    
    def test_raman_spectrum_generation(self, data_generator):
        """Test Raman spectrum generation"""
        materials = ['Si', 'GaAs', 'Graphene']
        
        for material in materials:
            spectrum = data_generator.generate_raman_spectrum(material)
            
            # Check structure
            assert isinstance(spectrum, RamanSpectrum)
            assert len(spectrum.raman_shift) > 0
            assert len(spectrum.intensity) == len(spectrum.raman_shift)
            
            # Check metadata
            assert spectrum.metadata['material'] == material
    
    def test_stress_effects_raman(self, data_generator):
        """Test stress effects in Raman generation"""
        stresses = [-2, 0, 2]  # GPa
        spectra = [data_generator.generate_raman_spectrum('Si', stress=s) 
                   for s in stresses]
        
        # Find main peak positions
        peak_positions = []
        for spectrum in spectra:
            peak_idx = np.argmax(spectrum.intensity)
            peak_positions.append(spectrum.raman_shift[peak_idx])
        
        # Peak should shift with stress
        assert peak_positions[0] < peak_positions[1] < peak_positions[2]
    
    def test_reproducibility(self):
        """Test generator reproducibility with same seed"""
        gen1 = OpticalTestDataGeneratorII(seed=123)
        gen2 = OpticalTestDataGeneratorII(seed=123)
        
        # Generate same data
        ell1 = gen1.generate_ellipsometry_data()
        ell2 = gen2.generate_ellipsometry_data()
        
        # Should be identical
        np.testing.assert_array_equal(ell1.psi, ell2.psi)
        np.testing.assert_array_equal(ell1.delta, ell2.delta)


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.integration
    def test_ellipsometry_workflow(self, ellipsometry_analyzer, data_generator):
        """Test complete ellipsometry workflow"""
        # Create sample structure
        stack = LayerStack(
            layers=[
                {'thickness': 95, 'model': DispersionModel.CAUCHY,
                 'params': {'A': 1.48, 'B': 0.004, 'C': 0, 'k': 0}}
            ],
            substrate={'n': 3.85, 'k': 0.02}
        )
        
        # Generate measurement
        data = data_generator.generate_ellipsometry_data(stack)
        
        # Fit model
        fit_params = ['layer0_thickness', 'layer0_params_A']
        result = ellipsometry_analyzer.fit_model(data, stack, fit_params)
        
        # Check convergence
        assert result['success']
        
        # Check fitted parameters are close to true values
        assert abs(result['parameters']['layer0_thickness'] - 100) < 20
        assert abs(result['parameters']['layer0_params_A'] - 1.46) < 0.1
    
    @pytest.mark.integration
    def test_pl_workflow(self, pl_analyzer, data_generator):
        """Test complete PL analysis workflow"""
        # Generate spectrum
        spectrum = data_generator.generate_pl_spectrum('GaAs', temperature=10)
        
        # Process spectrum
        processed = pl_analyzer.process_spectrum(spectrum)
        
        # Find peaks
        peaks = pl_analyzer.find_peaks(processed)
        assert peaks['count'] > 0
        
        # Fit peaks
        fit_result = pl_analyzer.fit_peaks(processed, n_peaks=2)
        assert fit_result['r_squared'] > 0.8
        
        # Check main peak energy is close to GaAs bandgap
        if len(fit_result['peaks']) > 0:
            main_peak_energy = fit_result['peaks'][0]['energy']
            assert 1.3 < main_peak_energy < 1.5  # GaAs ~1.42 eV
    
    @pytest.mark.integration
    def test_raman_workflow(self, raman_analyzer, data_generator):
        """Test complete Raman analysis workflow"""
        # Generate stressed Si spectrum
        stress_applied = 1.5  # GPa
        spectrum = data_generator.generate_raman_spectrum('Si', stress=stress_applied)
        
        # Process spectrum
        processed = raman_analyzer.process_spectrum(spectrum)
        
        # Find peaks
        peaks = raman_analyzer.find_peaks(processed)
        
        # Identify Si peak
        si_peak_idx = np.argmax(peaks['intensities'])
        si_peak_position = peaks['positions'][si_peak_idx]
        
        # Calculate stress
        stress_result = raman_analyzer.calculate_stress(si_peak_position, 520.5, 'Si')
        
        # Stress should be detected
        assert abs(stress_result['stress']) > 0.5  # Should detect significant stress
        
        # Crystallinity analysis
        crystal_result = raman_analyzer.analyze_crystallinity(processed, 'Si')
        assert crystal_result['crystallinity'] > 0.8  # Should be crystalline
    
    @pytest.mark.integration
    def test_temperature_dependent_pl(self, pl_analyzer, data_generator):
        """Test temperature-dependent PL workflow"""
        # Generate temperature series
        temperatures = np.logspace(np.log10(10), np.log10(300), 10)
        spectra = [data_generator.generate_pl_spectrum('GaAs', T) for T in temperatures]
        
        # Analyze series
        result = pl_analyzer.analyze_temperature_series(spectra)
        
        # Check Varshni parameters for GaAs
        Eg0 = result['varshni_params']['Eg0']
        assert 1.4 < Eg0 < 1.5  # GaAs Eg(0) ~1.42 eV
        
        # Check activation energy is reasonable
        assert 0 < result['activation_energy'] < 0.5  # eV


class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.performance
    def test_ellipsometry_large_dataset(self, ellipsometry_analyzer, sample_layer_stack):
        """Test ellipsometry with large wavelength range"""
        wavelengths = np.linspace(200, 2000, 1000)
        
        start_time = time.time()
        psi, delta = ellipsometry_analyzer.calculate_psi_delta(
            wavelengths, sample_layer_stack, 70
        )
        calc_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert calc_time < 5.0  # seconds
        assert len(psi) == len(wavelengths)
    
    @pytest.mark.performance
    def test_pl_multi_peak_fitting(self, pl_analyzer, data_generator):
        """Test fitting many peaks"""
        # Generate complex spectrum
        spectrum = data_generator.generate_pl_spectrum('GaAs')
        
        start_time = time.time()
        result = pl_analyzer.fit_peaks(spectrum, n_peaks=5)
        fit_time = time.time() - start_time
        
        # Should complete quickly
        assert fit_time < 3.0  # seconds
    
    @pytest.mark.performance
    def test_raman_mapping_performance(self, raman_analyzer):
        """Test Raman mapping performance"""
        # Create large map
        nx, ny = 20, 20
        n_points = 200
        positions = np.linspace(400, 600, n_points)
        spectra_map = np.random.rand(nx, ny, n_points) * 1000
        
        start_time = time.time()
        result = raman_analyzer.map_analysis(spectra_map, positions, 520)
        analysis_time = time.time() - start_time
        
        # Should handle large maps efficiently
        assert analysis_time < 10.0  # seconds
        assert result['peak_position_map'].shape == (nx, ny)


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_spectrum(self, pl_analyzer):
        """Test handling of empty spectrum"""
        empty_spectrum = PLSpectrum(
            wavelength=np.array([]),
            intensity=np.array([]),
            excitation_wavelength=532,
            excitation_power=10,
            temperature=293
        )
        
        peaks = pl_analyzer.find_peaks(empty_spectrum)
        assert peaks['count'] == 0
    
    def test_no_peaks(self, raman_analyzer):
        """Test handling when no peaks are found"""
        # Flat spectrum
        flat_spectrum = RamanSpectrum(
            raman_shift=np.linspace(100, 1000, 100),
            intensity=np.ones(100) * 50,
            laser_wavelength=532,
            laser_power=5
        )
        
        peaks = raman_analyzer.find_peaks(flat_spectrum)
        assert len(peaks['positions']) == 0
    
    def test_invalid_layer_stack(self, ellipsometry_analyzer):
        """Test handling of invalid layer parameters"""
        # Negative thickness
        invalid_stack = LayerStack(
            layers=[{'thickness': -100, 'model': DispersionModel.CAUCHY,
                    'params': {'A': 1.5, 'B': 0, 'C': 0, 'k': 0}}],
            substrate={'n': 1.5, 'k': 0}
        )
        
        # Should handle gracefully
        wavelengths = np.linspace(400, 800, 10)
        psi, delta = ellipsometry_analyzer.calculate_psi_delta(
            wavelengths, invalid_stack, 70
        )
        
        # Should return some result (even if unphysical)
        assert len(psi) == len(wavelengths)
    
    def test_convergence_failure(self, ellipsometry_analyzer, sample_ellipsometry_data):
        """Test handling of fitting convergence failure"""
        # Create impossible fitting scenario
        bad_stack = LayerStack(
            layers=[{'thickness': 1e6, 'model': DispersionModel.CAUCHY,
                    'params': {'A': 10, 'B': 0, 'C': 0, 'k': 5}}],
            substrate={'n': 0.1, 'k': 10}
        )
        
        fit_params = ['layer0_thickness']
        
        # Should not crash
        result = ellipsometry_analyzer.fit_model(
            sample_ellipsometry_data,
            bad_stack,
            fit_params
        )
        
        # Check that result is returned even if fit failed
        assert 'mse' in result
        assert 'success' in result


# Performance benchmark function
def run_benchmarks():
    """Run performance benchmarks"""
    import cProfile
    import pstats
    from io import StringIO
    
    # Initialize components
    ellipsometry = EllipsometryAnalyzer()
    pl_analyzer = PhotoluminescenceAnalyzer()
    raman = RamanAnalyzer()
    generator = OpticalTestDataGeneratorII()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Benchmark ellipsometry
    for _ in range(5):
        stack = LayerStack(
            layers=[{'thickness': 100, 'model': DispersionModel.CAUCHY,
                    'params': {'A': 1.5, 'B': 0.01, 'C': 0, 'k': 0}}],
            substrate={'n': 3.85, 'k': 0.02}
        )
        data = generator.generate_ellipsometry_data(stack)
        ellipsometry.fit_model(data, stack, ['layer0_thickness'])
    
    # Benchmark PL
    for _ in range(10):
        spectrum = generator.generate_pl_spectrum('GaAs')
        pl_analyzer.find_peaks(spectrum)
        pl_analyzer.fit_peaks(spectrum)
    
    # Benchmark Raman
    for _ in range(10):
        spectrum = generator.generate_raman_spectrum('Si', stress=1.0)
        raman.find_peaks(spectrum)
        raman.analyze_crystallinity(spectrum, 'Si')
    
    profiler.disable()
    
    # Generate report
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(15)
    
    print("\nSession 8 Performance Benchmark Results:")
    print(stream.getvalue())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("Running Performance Benchmarks")
    print("=" * 80)
    run_benchmarks()
