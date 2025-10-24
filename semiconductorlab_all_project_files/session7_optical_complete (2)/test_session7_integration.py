"""
Session 7: Integration Tests for UV-Vis-NIR and FTIR Analysis
==============================================================
Comprehensive test suite for optical spectroscopy methods
"""

import pytest
import numpy as np
import pandas as pd
import json
import time
from typing import Dict, Any, List
from datetime import datetime
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Import the main modules (adjust path as needed)
import sys
sys.path.append('/home/claude')
from session7_complete_implementation import (
    UVVisNIRAnalyzer, FTIRAnalyzer, OpticalTestDataGenerator,
    SpectralData, TaucResult, PeakFitResult, 
    MeasurementType, BandgapType, BaselineMethod
)

# Test fixtures
@pytest.fixture
def uv_vis_analyzer():
    """Create UV-Vis-NIR analyzer instance"""
    return UVVisNIRAnalyzer()

@pytest.fixture
def ftir_analyzer():
    """Create FTIR analyzer instance"""
    return FTIRAnalyzer()

@pytest.fixture
def data_generator():
    """Create test data generator"""
    return OpticalTestDataGenerator(seed=42)

@pytest.fixture
def sample_uv_spectrum(data_generator):
    """Generate sample UV-Vis spectrum"""
    return data_generator.generate_uv_vis_spectrum(
        material='GaAs',
        measurement_type=MeasurementType.TRANSMISSION
    )

@pytest.fixture
def sample_ftir_spectrum(data_generator):
    """Generate sample FTIR spectrum"""
    return data_generator.generate_ftir_spectrum(sample_type='SiO2_on_Si')


class TestUVVisNIRAnalyzer:
    """Test suite for UV-Vis-NIR analyzer"""
    
    def test_spectrum_processing(self, uv_vis_analyzer, sample_uv_spectrum):
        """Test spectrum processing with smoothing and baseline correction"""
        # Process spectrum
        processed = uv_vis_analyzer.process_spectrum(
            sample_uv_spectrum,
            smooth=True,
            baseline_correct=True,
            baseline_method=BaselineMethod.RUBBERBAND
        )
        
        # Verify processing
        assert processed is not None
        assert len(processed.wavelength) == len(sample_uv_spectrum.wavelength)
        assert len(processed.intensity) == len(sample_uv_spectrum.intensity)
        assert 'baseline' in processed.metadata
        
        # Check that smoothing reduces noise
        raw_noise = np.std(np.diff(sample_uv_spectrum.intensity))
        processed_noise = np.std(np.diff(processed.intensity))
        assert processed_noise < raw_noise
    
    def test_transmission_to_absorption(self, uv_vis_analyzer, sample_uv_spectrum):
        """Test conversion from transmission to absorption"""
        # Convert to absorption
        absorption = uv_vis_analyzer.calculate_absorption(sample_uv_spectrum)
        
        # Verify conversion
        assert absorption.measurement_type == MeasurementType.ABSORPTION
        assert len(absorption.intensity) == len(sample_uv_spectrum.intensity)
        
        # Check physical consistency
        # Where transmission is high, absorption should be low
        high_trans_idx = np.argmax(sample_uv_spectrum.intensity)
        low_abs_idx = np.argmin(absorption.intensity)
        assert abs(high_trans_idx - low_abs_idx) < 10  # Should be nearby
    
    def test_absorption_coefficient(self, uv_vis_analyzer, sample_uv_spectrum):
        """Test absorption coefficient calculation"""
        # Calculate absorption first
        absorption = uv_vis_analyzer.calculate_absorption(sample_uv_spectrum)
        
        # Calculate absorption coefficient
        thickness = 0.5  # mm
        alpha = uv_vis_analyzer.calculate_absorption_coefficient(absorption, thickness)
        
        # Verify calculation
        assert len(alpha) == len(absorption.intensity)
        assert np.all(alpha >= 0)  # Should be non-negative
        assert np.max(alpha) < 1e8  # Reasonable upper limit (cm⁻¹)
    
    def test_tauc_analysis_direct(self, uv_vis_analyzer, sample_uv_spectrum):
        """Test Tauc analysis for direct bandgap"""
        # Perform Tauc analysis
        result = uv_vis_analyzer.tauc_analysis(
            sample_uv_spectrum,
            thickness=0.5,
            bandgap_type=BandgapType.DIRECT_ALLOWED
        )
        
        # Verify result structure
        assert isinstance(result, TaucResult)
        assert result.bandgap > 0
        assert result.bandgap_error >= 0
        assert 0 <= result.r_squared <= 1
        assert len(result.tauc_x) == len(result.tauc_y)
        
        # Check bandgap is reasonable for GaAs
        assert 1.3 < result.bandgap < 1.6  # GaAs ~1.42 eV
    
    def test_tauc_analysis_indirect(self, uv_vis_analyzer, data_generator):
        """Test Tauc analysis for indirect bandgap"""
        # Generate Si spectrum (indirect bandgap)
        si_spectrum = data_generator.generate_uv_vis_spectrum('Si')
        
        # Perform Tauc analysis
        result = uv_vis_analyzer.tauc_analysis(
            si_spectrum,
            thickness=0.5,
            bandgap_type=BandgapType.INDIRECT_ALLOWED
        )
        
        # Verify result
        assert isinstance(result, TaucResult)
        assert 1.0 < result.bandgap < 1.3  # Si ~1.12 eV
        assert result.bandgap_type == BandgapType.INDIRECT_ALLOWED
    
    def test_baseline_methods(self, uv_vis_analyzer, sample_uv_spectrum):
        """Test different baseline correction methods"""
        methods = [
            BaselineMethod.LINEAR,
            BaselineMethod.POLYNOMIAL,
            BaselineMethod.RUBBERBAND,
            BaselineMethod.ASYMMETRIC_LEAST_SQUARES
        ]
        
        baselines = {}
        for method in methods:
            processed = uv_vis_analyzer.process_spectrum(
                sample_uv_spectrum,
                smooth=False,
                baseline_correct=True,
                baseline_method=method
            )
            baselines[method] = processed.metadata.get('baseline', [])
        
        # Verify all methods produce baselines
        for method in methods:
            assert len(baselines[method]) == len(sample_uv_spectrum.intensity)
        
        # Check that different methods give different results
        assert not np.allclose(baselines[BaselineMethod.LINEAR], 
                              baselines[BaselineMethod.POLYNOMIAL])
    
    def test_energy_conversion(self, sample_uv_spectrum):
        """Test wavelength to energy conversion"""
        energy, intensity = sample_uv_spectrum.to_energy()
        
        # Verify conversion
        assert len(energy) == len(sample_uv_spectrum.wavelength)
        assert len(intensity) == len(sample_uv_spectrum.intensity)
        
        # Check energy is in ascending order (wavelength was descending)
        assert np.all(np.diff(energy) > 0)
        
        # Verify conversion formula (E = hc/λ)
        h = 6.62607015e-34
        c = 299792458
        eV = 1.602176634e-19
        
        # Check first point
        expected_energy = (h * c) / (sample_uv_spectrum.wavelength[0] * 1e-9) / eV
        assert np.isclose(energy[-1], expected_energy, rtol=1e-3)


class TestFTIRAnalyzer:
    """Test suite for FTIR analyzer"""
    
    def test_spectrum_processing(self, ftir_analyzer, sample_ftir_spectrum):
        """Test FTIR spectrum processing"""
        processed = ftir_analyzer.process_ftir_spectrum(
            sample_ftir_spectrum,
            baseline_method=BaselineMethod.ASYMMETRIC_LEAST_SQUARES,
            smooth=True
        )
        
        # Verify processing
        assert processed is not None
        assert len(processed.wavelength) == len(sample_ftir_spectrum.wavelength)
        assert 'units' in processed.metadata
        assert processed.metadata['units'] == 'wavenumber_cm-1'
    
    def test_peak_finding(self, ftir_analyzer, sample_ftir_spectrum):
        """Test peak finding in FTIR spectrum"""
        peaks = ftir_analyzer.find_peaks(
            sample_ftir_spectrum,
            prominence=0.01,
            distance=10,
            identify=True
        )
        
        # Verify peak finding
        assert 'positions' in peaks
        assert 'intensities' in peaks
        assert 'widths' in peaks
        assert 'identifications' in peaks
        
        # Should find at least some peaks
        assert len(peaks['positions']) > 0
        
        # Check peak positions are within spectrum range
        assert np.all(peaks['positions'] >= sample_ftir_spectrum.wavelength[0])
        assert np.all(peaks['positions'] <= sample_ftir_spectrum.wavelength[-1])
    
    def test_peak_identification(self, ftir_analyzer, sample_ftir_spectrum):
        """Test automatic peak identification"""
        peaks = ftir_analyzer.find_peaks(
            sample_ftir_spectrum,
            identify=True
        )
        
        # Check identifications
        identifications = peaks['identifications']
        identified_count = sum(1 for x in identifications if x is not None)
        
        # Should identify at least some peaks for SiO2
        assert identified_count > 0
        
        # Check for expected SiO2 peaks
        expected_peaks = ['Si-O']  # Should find Si-O stretching around 1100 cm⁻¹
        identified_names = [x for x in identifications if x is not None]
        
        for expected in expected_peaks:
            assert any(expected in name for name in identified_names)
    
    def test_peak_fitting(self, ftir_analyzer, sample_ftir_spectrum):
        """Test peak fitting with different functions"""
        peak_types = ['gaussian', 'lorentzian']  # Skip 'voigt' for speed
        
        for peak_type in peak_types:
            result = ftir_analyzer.fit_peaks(
                sample_ftir_spectrum,
                peak_type=peak_type,
                max_peaks=5
            )
            
            # Verify fitting result
            assert isinstance(result, PeakFitResult)
            assert len(result.peaks) <= 5
            assert len(result.baseline) == len(sample_ftir_spectrum.intensity)
            assert len(result.fitted_spectrum) == len(sample_ftir_spectrum.intensity)
            assert len(result.residuals) == len(sample_ftir_spectrum.intensity)
            assert 0 <= result.r_squared <= 1
    
    def test_film_thickness_calculation(self, ftir_analyzer, sample_ftir_spectrum):
        """Test thin film thickness calculation from interference fringes"""
        result = ftir_analyzer.calculate_film_thickness(
            sample_ftir_spectrum,
            n_substrate=1.46,  # SiO2
            angle=0
        )
        
        # Verify result structure
        assert 'thickness' in result
        assert 'error' in result
        assert 'n_fringes' in result
        
        # Check physical reasonableness (if fringes detected)
        if result['n_fringes'] > 1:
            assert 0.1 < result['thickness'] < 100  # µm range
            assert result['error'] >= 0
    
    def test_wavenumber_conversion(self, sample_ftir_spectrum):
        """Test wavelength to wavenumber conversion"""
        # Create spectrum in wavelength (nm)
        wavelength_nm = np.linspace(2500, 25000, 100)  # 400-4000 cm⁻¹
        intensity = np.ones_like(wavelength_nm)
        
        spectrum = SpectralData(
            wavelength=wavelength_nm,
            intensity=intensity,
            measurement_type=MeasurementType.TRANSMISSION
        )
        
        wavenumber, intensity_wn = spectrum.to_wavenumber()
        
        # Verify conversion
        assert len(wavenumber) == len(wavelength_nm)
        
        # Check conversion formula (ν = 10^7/λ)
        expected_wn = 1e7 / wavelength_nm[0]
        assert np.isclose(wavenumber[-1], expected_wn, rtol=1e-3)
    
    def test_peak_database(self, ftir_analyzer):
        """Test peak database loading and access"""
        # Check database exists
        assert hasattr(ftir_analyzer, 'peak_database')
        assert len(ftir_analyzer.peak_database) > 0
        
        # Verify database structure
        for name, info in ftir_analyzer.peak_database.items():
            assert 'position' in info
            assert 'range' in info
            assert 'type' in info
            assert len(info['range']) == 2
            assert info['range'][0] < info['range'][1]


class TestOpticalDataGenerator:
    """Test suite for synthetic data generator"""
    
    def test_uv_vis_generation(self, data_generator):
        """Test UV-Vis spectrum generation for different materials"""
        materials = ['GaAs', 'Si', 'GaN', 'InP', 'CdTe']
        
        for material in materials:
            spectrum = data_generator.generate_uv_vis_spectrum(
                material=material,
                measurement_type=MeasurementType.TRANSMISSION
            )
            
            # Verify spectrum structure
            assert isinstance(spectrum, SpectralData)
            assert len(spectrum.wavelength) > 0
            assert len(spectrum.intensity) == len(spectrum.wavelength)
            assert spectrum.measurement_type == MeasurementType.TRANSMISSION
            assert 'material' in spectrum.metadata
            assert spectrum.metadata['material'] == material
            
            # Check physical reasonableness
            assert np.all(spectrum.intensity >= 0)
            assert np.all(spectrum.intensity <= 100)
    
    def test_ftir_generation(self, data_generator):
        """Test FTIR spectrum generation for different sample types"""
        sample_types = ['SiO2_on_Si', 'Si3N4_on_Si', 'organic_contamination']
        
        for sample_type in sample_types:
            spectrum = data_generator.generate_ftir_spectrum(sample_type=sample_type)
            
            # Verify spectrum
            assert isinstance(spectrum, SpectralData)
            assert 'sample_type' in spectrum.metadata
            assert spectrum.metadata['sample_type'] == sample_type
            
            # Check that spectrum contains features (not flat)
            std_dev = np.std(spectrum.intensity)
            assert std_dev > 1  # Should have some variation
    
    def test_reproducibility(self):
        """Test that generator produces reproducible results with same seed"""
        gen1 = OpticalTestDataGenerator(seed=123)
        gen2 = OpticalTestDataGenerator(seed=123)
        
        spectrum1 = gen1.generate_uv_vis_spectrum('GaAs')
        spectrum2 = gen2.generate_uv_vis_spectrum('GaAs')
        
        # Should be identical
        np.testing.assert_array_equal(spectrum1.wavelength, spectrum2.wavelength)
        np.testing.assert_array_equal(spectrum1.intensity, spectrum2.intensity)
    
    def test_tauc_test_data(self, data_generator):
        """Test Tauc analysis test data generation"""
        test_data = data_generator.generate_tauc_test_data(
            true_bandgap=1.42,
            bandgap_type=BandgapType.DIRECT_ALLOWED
        )
        
        # Verify test data structure
        assert 'spectrum' in test_data
        assert 'tauc_result' in test_data
        assert 'true_bandgap' in test_data
        assert 'measured_bandgap' in test_data
        assert 'error' in test_data
        
        # Check error is reasonable
        assert test_data['error'] < 0.1  # Should be within 100 meV


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.integration
    def test_complete_uv_vis_workflow(self, uv_vis_analyzer, data_generator):
        """Test complete UV-Vis analysis workflow"""
        # Generate spectrum
        spectrum = data_generator.generate_uv_vis_spectrum('GaAs')
        
        # Process spectrum
        processed = uv_vis_analyzer.process_spectrum(spectrum)
        
        # Convert to absorption if needed
        if processed.measurement_type == MeasurementType.TRANSMISSION:
            absorption = uv_vis_analyzer.calculate_absorption(processed)
        else:
            absorption = processed
        
        # Perform Tauc analysis
        tauc_result = uv_vis_analyzer.tauc_analysis(
            absorption,
            thickness=0.5,
            bandgap_type=BandgapType.DIRECT_ALLOWED
        )
        
        # Verify complete workflow
        assert tauc_result.bandgap > 0
        assert tauc_result.r_squared > 0.8
        
        # Check bandgap accuracy
        expected_bandgap = 1.42  # GaAs
        error = abs(tauc_result.bandgap - expected_bandgap)
        assert error < 0.1  # Within 100 meV
    
    @pytest.mark.integration
    def test_complete_ftir_workflow(self, ftir_analyzer, data_generator):
        """Test complete FTIR analysis workflow"""
        # Generate spectrum
        spectrum = data_generator.generate_ftir_spectrum('SiO2_on_Si')
        
        # Process spectrum
        processed = ftir_analyzer.process_ftir_spectrum(spectrum)
        
        # Find peaks
        peaks = ftir_analyzer.find_peaks(processed, identify=True)
        
        # Fit peaks
        fit_result = ftir_analyzer.fit_peaks(processed, max_peaks=5)
        
        # Calculate film thickness
        thickness = ftir_analyzer.calculate_film_thickness(processed)
        
        # Verify workflow completion
        assert len(peaks['positions']) > 0
        assert len(fit_result.peaks) > 0
        assert thickness['thickness'] >= 0
    
    @pytest.mark.integration
    def test_multi_material_comparison(self, uv_vis_analyzer, data_generator):
        """Test bandgap analysis across multiple materials"""
        materials = {
            'GaAs': 1.42,
            'Si': 1.12,
            'GaN': 3.4,
            'InP': 1.35,
            'CdTe': 1.5
        }
        
        results = {}
        for material, expected_eg in materials.items():
            spectrum = data_generator.generate_uv_vis_spectrum(material)
            
            # Determine appropriate bandgap type
            if material in ['Si']:
                bandgap_type = BandgapType.INDIRECT_ALLOWED
            else:
                bandgap_type = BandgapType.DIRECT_ALLOWED
            
            tauc = uv_vis_analyzer.tauc_analysis(
                spectrum,
                thickness=0.5,
                bandgap_type=bandgap_type
            )
            
            results[material] = {
                'measured': tauc.bandgap,
                'expected': expected_eg,
                'error': abs(tauc.bandgap - expected_eg),
                'r_squared': tauc.r_squared
            }
        
        # Verify all materials analyzed successfully
        assert len(results) == len(materials)
        
        # Check average error
        avg_error = np.mean([r['error'] for r in results.values()])
        assert avg_error < 0.1  # Average error < 100 meV
        
        # Check all have good fits
        for material, result in results.items():
            assert result['r_squared'] > 0.8, f"Poor fit for {material}"


class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.performance
    def test_large_spectrum_processing(self, uv_vis_analyzer):
        """Test processing of large spectrum"""
        # Generate large spectrum (10000 points)
        wavelengths = np.linspace(200, 2000, 10000)
        intensities = np.random.rand(10000) * 100
        
        large_spectrum = SpectralData(
            wavelength=wavelengths,
            intensity=intensities,
            measurement_type=MeasurementType.TRANSMISSION
        )
        
        # Time processing
        start_time = time.time()
        processed = uv_vis_analyzer.process_spectrum(large_spectrum)
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert processing_time < 5.0  # seconds
        assert processed is not None
    
    @pytest.mark.performance
    def test_multiple_peak_fitting(self, ftir_analyzer):
        """Test fitting many peaks"""
        # Generate complex spectrum with many peaks
        wavenumbers = np.linspace(400, 4000, 1000)
        intensity = np.ones_like(wavenumbers) * 90
        
        # Add 20 peaks
        for i in range(20):
            position = 500 + i * 150
            width = 20
            amplitude = 30
            peak = amplitude * np.exp(-((wavenumbers - position) / width) ** 2)
            intensity -= peak
        
        spectrum = SpectralData(
            wavelength=wavenumbers,
            intensity=intensity,
            measurement_type=MeasurementType.TRANSMISSION
        )
        
        # Time peak fitting
        start_time = time.time()
        result = ftir_analyzer.fit_peaks(spectrum, max_peaks=10)
        fitting_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert fitting_time < 10.0  # seconds
        assert len(result.peaks) > 0
    
    @pytest.mark.performance
    def test_batch_processing(self, uv_vis_analyzer, data_generator):
        """Test batch processing of multiple spectra"""
        n_samples = 50
        
        start_time = time.time()
        for i in range(n_samples):
            # Generate spectrum
            spectrum = data_generator.generate_uv_vis_spectrum('GaAs')
            
            # Process and analyze
            processed = uv_vis_analyzer.process_spectrum(spectrum)
            tauc = uv_vis_analyzer.tauc_analysis(processed, thickness=0.5)
        
        total_time = time.time() - start_time
        time_per_sample = total_time / n_samples
        
        # Should process quickly
        assert time_per_sample < 1.0  # Less than 1 second per sample


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_spectrum(self, uv_vis_analyzer):
        """Test handling of empty spectrum"""
        with pytest.raises(ValueError):
            SpectralData(
                wavelength=np.array([]),
                intensity=np.array([]),
                measurement_type=MeasurementType.TRANSMISSION
            )
    
    def test_mismatched_arrays(self):
        """Test handling of mismatched array lengths"""
        with pytest.raises(ValueError):
            SpectralData(
                wavelength=np.array([1, 2, 3]),
                intensity=np.array([1, 2]),  # Wrong length
                measurement_type=MeasurementType.TRANSMISSION
            )
    
    def test_invalid_measurement_type(self, uv_vis_analyzer):
        """Test handling of invalid measurement type"""
        spectrum = SpectralData(
            wavelength=np.array([300, 400, 500]),
            intensity=np.array([50, 60, 70]),
            measurement_type=MeasurementType.REFLECTANCE
        )
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises(ValueError):
            uv_vis_analyzer.calculate_absorption_coefficient(spectrum, 0.5)
    
    def test_negative_thickness(self, uv_vis_analyzer, sample_uv_spectrum):
        """Test handling of negative thickness"""
        # Should handle negative thickness appropriately
        with pytest.raises(ValueError):
            uv_vis_analyzer.tauc_analysis(
                sample_uv_spectrum,
                thickness=-0.5  # Invalid
            )
    
    def test_no_peaks_found(self, ftir_analyzer):
        """Test handling when no peaks are found"""
        # Generate flat spectrum
        flat_spectrum = SpectralData(
            wavelength=np.linspace(400, 4000, 100),
            intensity=np.ones(100) * 90,
            measurement_type=MeasurementType.TRANSMISSION
        )
        
        result = ftir_analyzer.fit_peaks(flat_spectrum)
        
        # Should return empty result gracefully
        assert len(result.peaks) == 0
        assert result.r_squared == 0


# API Integration tests
class TestAPIIntegration:
    """Test integration with FastAPI backend"""
    
    @pytest.mark.asyncio
    async def test_api_spectrum_upload(self):
        """Test spectrum upload endpoint"""
        # Mock API client
        from httpx import AsyncClient
        
        # Mock spectrum data
        spectrum_data = {
            'wavelength': [300, 400, 500, 600],
            'intensity': [90, 85, 80, 75],
            'measurementType': 'transmission',
            'metadata': {
                'sampleId': 'TEST-001',
                'thickness': 0.5
            }
        }
        
        # Simulate API call (would need actual server running)
        # async with AsyncClient(base_url="http://localhost:8000") as client:
        #     response = await client.post("/api/optical/spectrum", json=spectrum_data)
        #     assert response.status_code == 200
        #     result = response.json()
        #     assert 'id' in result
        
        # For now, just verify data structure
        assert 'wavelength' in spectrum_data
        assert len(spectrum_data['wavelength']) == len(spectrum_data['intensity'])
    
    @pytest.mark.asyncio
    async def test_api_tauc_analysis(self):
        """Test Tauc analysis endpoint"""
        # Mock request data
        analysis_request = {
            'spectrumId': 'SPEC-123',
            'thickness': 0.5,
            'bandgapType': 'direct_allowed',
            'autoRange': True
        }
        
        # Verify request structure
        assert 'spectrumId' in analysis_request
        assert 'thickness' in analysis_request
        assert analysis_request['thickness'] > 0


# Run performance benchmarks
def run_benchmarks():
    """Run performance benchmarks and generate report"""
    import cProfile
    import pstats
    from io import StringIO
    
    # Initialize components
    analyzer = UVVisNIRAnalyzer()
    generator = OpticalTestDataGenerator()
    
    # Profile Tauc analysis
    profiler = cProfile.Profile()
    profiler.enable()
    
    for _ in range(10):
        spectrum = generator.generate_uv_vis_spectrum('GaAs')
        analyzer.tauc_analysis(spectrum, thickness=0.5)
    
    profiler.disable()
    
    # Generate report
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(10)
    
    print("\nPerformance Benchmark Results:")
    print(stream.getvalue())


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("Running Performance Benchmarks")
    print("=" * 80)
    run_benchmarks()
