"""
Session 10: Integration Tests for Microscopy Analysis
======================================================
Comprehensive test suite for SEM, TEM, and AFM analysis systems
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
import cv2
from skimage import measure, morphology

# Import the main modules
import sys
sys.path.append('/home/claude')
from session10_microscopy_complete_implementation import (
    SEMAnalyzer, TEMAnalyzer, AFMAnalyzer, MicroscopySimulator,
    MicroscopyImage, AFMData, Particle, GrainBoundary,
    MicroscopyType, ImagingMode, DetectorType
)

# Test fixtures
@pytest.fixture
def sem_analyzer():
    """Create SEM analyzer instance"""
    return SEMAnalyzer()

@pytest.fixture
def tem_analyzer():
    """Create TEM analyzer instance"""
    return TEMAnalyzer()

@pytest.fixture
def afm_analyzer():
    """Create AFM analyzer instance"""
    return AFMAnalyzer()

@pytest.fixture
def simulator():
    """Create microscopy simulator instance"""
    return MicroscopySimulator(seed=42)

@pytest.fixture
def sample_sem_image(simulator):
    """Generate sample SEM image"""
    return simulator.generate_sem_image('particles', pixel_size=5.0)

@pytest.fixture
def sample_tem_image(simulator):
    """Generate sample TEM image"""
    return simulator.generate_tem_image('lattice', pixel_size=0.1)

@pytest.fixture
def sample_afm_data(simulator):
    """Generate sample AFM data"""
    return simulator.generate_afm_data('rough', scan_size=(1000, 1000))


class TestSEMAnalyzer:
    """Test suite for SEM analysis functionality"""
    
    def test_image_processing(self, sem_analyzer, sample_sem_image):
        """Test SEM image processing"""
        # Process image
        processed = sem_analyzer.process_image(
            sample_sem_image,
            denoise=True,
            enhance_contrast=True
        )
        
        # Check processed image
        assert isinstance(processed, MicroscopyImage)
        assert processed.shape == sample_sem_image.shape
        assert processed.microscopy_type == MicroscopyType.SEM
        
        # Check that processing was applied
        assert 'processed' in processed.metadata
        
        # Verify intensity range
        assert 0 <= np.min(processed.image_data) <= 1
        assert 0 <= np.max(processed.image_data) <= 1
    
    def test_particle_detection(self, sem_analyzer, sample_sem_image):
        """Test particle detection and characterization"""
        # Process first
        processed = sem_analyzer.process_image(sample_sem_image)
        
        # Detect particles
        particles = sem_analyzer.detect_particles(
            processed,
            min_size=10,
            threshold_method='otsu'
        )
        
        # Check results
        assert isinstance(particles, list)
        assert len(particles) > 0
        
        # Verify particle properties
        for particle in particles:
            assert isinstance(particle, Particle)
            assert particle.area > 0
            assert particle.diameter > 0
            assert 0 <= particle.circularity <= 1
            assert particle.aspect_ratio >= 1
            assert -180 <= particle.orientation <= 180
    
    def test_grain_size_analysis(self, sem_analyzer, simulator):
        """Test grain size measurement"""
        # Generate grain structure
        grain_image = simulator.generate_sem_image('grains', pixel_size=10.0)
        
        # Measure grain size
        result = sem_analyzer.measure_grain_size(grain_image)
        
        # Check results
        assert 'num_grains' in result
        assert 'mean_diameter' in result
        assert 'std_diameter' in result
        assert 'grain_diameters' in result
        
        assert result['num_grains'] > 0
        assert result['mean_diameter'] > 0
        assert len(result['grain_diameters']) == result['num_grains']
    
    def test_porosity_analysis(self, sem_analyzer, simulator):
        """Test porosity analysis"""
        # Generate porous structure
        porous_image = simulator.generate_sem_image('porous', pixel_size=5.0)
        
        # Analyze porosity
        result = sem_analyzer.analyze_porosity(porous_image)
        
        # Check results
        assert 'porosity_fraction' in result
        assert 'num_pores' in result
        assert 'mean_pore_diameter' in result
        assert 'pore_areas' in result
        
        assert 0 <= result['porosity_fraction'] <= 1
        assert result['num_pores'] >= 0
        if result['num_pores'] > 0:
            assert result['mean_pore_diameter'] > 0
    
    def test_critical_dimension_measurement(self, sem_analyzer, sample_sem_image):
        """Test critical dimension (CD) measurement"""
        # Measure CD
        result = sem_analyzer.measure_critical_dimension(
            sample_sem_image,
            direction='horizontal'
        )
        
        # Check results
        assert 'mean_cd' in result
        assert 'std_cd' in result
        assert 'num_features' in result
        assert 'uniformity' in result
        
        if result['num_features'] > 0:
            assert result['mean_cd'] >= 0
            assert 0 <= result['uniformity'] <= 1
    
    def test_eds_quantification(self, sem_analyzer):
        """Test EDS spectrum quantification"""
        # Generate synthetic spectrum
        spectrum = np.random.poisson(100, 2048)
        
        # Add peaks for specific elements
        spectrum[174] += 1000  # Si Ka at 1.74 keV
        spectrum[149] += 500   # Al Ka at 1.49 keV
        
        # Quantify
        composition = sem_analyzer.eds_quantification(
            spectrum,
            elements=['Si', 'Al', 'O'],
            beam_energy=20.0
        )
        
        # Check results
        assert 'Si' in composition
        assert 'Al' in composition
        assert composition['Si'] > composition['Al']  # Si peak was stronger
        
        # Check normalization
        total = sum(composition.values())
        assert abs(total - 100) < 1  # Should sum to ~100%
    
    def test_particle_size_distribution(self, sem_analyzer, sample_sem_image):
        """Test particle size distribution analysis"""
        # Detect particles
        particles = sem_analyzer.detect_particles(sample_sem_image)
        
        if len(particles) > 0:
            # Calculate size distribution
            sizes = [p.diameter for p in particles]
            
            # Statistics
            mean_size = np.mean(sizes)
            std_size = np.std(sizes)
            
            assert mean_size > 0
            assert std_size >= 0
            
            # Check distribution
            hist, bins = np.histogram(sizes, bins=10)
            assert len(hist) == 10
            assert np.sum(hist) == len(particles)


class TestTEMAnalyzer:
    """Test suite for TEM analysis functionality"""
    
    def test_hrtem_processing(self, tem_analyzer, sample_tem_image):
        """Test HRTEM image processing"""
        # Process with different filters
        for filter_type in ['wiener', 'fourier']:
            processed = tem_analyzer.process_hrtem(
                sample_tem_image,
                filter_type=filter_type
            )
            
            assert isinstance(processed, MicroscopyImage)
            assert processed.shape == sample_tem_image.shape
            assert 'filtered' in processed.metadata
            assert processed.metadata['filtered'] == filter_type
    
    def test_lattice_spacing_measurement(self, tem_analyzer, simulator):
        """Test lattice spacing measurement from HRTEM"""
        # Generate lattice image
        lattice_image = simulator.generate_tem_image('lattice', pixel_size=0.1)
        
        # Measure lattice spacing
        result = tem_analyzer.measure_lattice_spacing(lattice_image)
        
        # Check results
        assert 'd_spacings' in result
        assert 'fft_magnitude' in result
        assert 'peaks' in result
        
        if len(result['d_spacings']) > 0:
            # Check reasonable d-spacing values
            for d in result['d_spacings']:
                assert 0.5 < d < 10  # Angstroms
            
            if result['mean_spacing']:
                assert 0.5 < result['mean_spacing'] < 10
    
    def test_diffraction_pattern_analysis(self, tem_analyzer, simulator):
        """Test SAED pattern analysis"""
        # Generate diffraction pattern
        diff_image = simulator.generate_tem_image('diffraction', pixel_size=0.01)
        
        # Analyze pattern
        result = tem_analyzer.analyze_diffraction_pattern(
            diff_image,
            calibration=2.0
        )
        
        # Check results
        assert 'num_spots' in result
        assert 'spots' in result
        assert 'd_spacings' in result
        assert 'proposed_structure' in result
        
        assert result['num_spots'] > 0
        assert len(result['d_spacings']) > 0
        
        # Check structure identification
        assert result['proposed_structure'] in ['FCC', 'BCC', 'HCP', 'Unknown', 'Insufficient data']
    
    def test_defect_detection(self, tem_analyzer, simulator):
        """Test crystallographic defect detection"""
        # Generate image with defects
        defect_image = simulator.generate_tem_image('defects', pixel_size=0.1)
        
        # Test different defect types
        for defect_type in ['dislocation', 'stacking_fault', 'grain_boundary']:
            defects = tem_analyzer.detect_defects(defect_image, defect_type)
            
            assert isinstance(defects, list)
            
            for defect in defects:
                assert 'type' in defect
                assert 'id' in defect
                assert defect['type'] == defect_type
    
    def test_thickness_measurement(self, tem_analyzer, sample_tem_image):
        """Test sample thickness estimation"""
        # Test different methods
        for method in ['eels', 'contrast']:
            thickness = tem_analyzer.measure_thickness(sample_tem_image, method)
            
            assert isinstance(thickness, float)
            assert thickness >= 0
            
            # Check reasonable range (nm)
            assert thickness < 1000  # Should be less than 1 μm
    
    def test_lattice_parameter_calculation(self, tem_analyzer, simulator):
        """Test lattice parameter determination"""
        # Generate perfect lattice
        lattice_image = simulator.generate_tem_image('lattice', pixel_size=0.1)
        
        # Measure spacings
        result = tem_analyzer.measure_lattice_spacing(lattice_image)
        
        if result['lattice_parameter']:
            # Check reasonable value for cubic system
            assert 2 < result['lattice_parameter'] < 10  # Angstroms


class TestAFMAnalyzer:
    """Test suite for AFM analysis functionality"""
    
    def test_height_map_processing(self, afm_analyzer, sample_afm_data):
        """Test AFM height map processing"""
        # Process height map
        processed = afm_analyzer.process_height_map(
            sample_afm_data,
            flatten=True,
            remove_outliers=True
        )
        
        # Check results
        assert isinstance(processed, AFMData)
        assert processed.height.shape == sample_afm_data.height.shape
        assert 'processed' in processed.metadata
        
        # Check that flattening reduced tilt
        original_range = np.max(sample_afm_data.height) - np.min(sample_afm_data.height)
        processed_range = np.max(processed.height) - np.min(processed.height)
        assert processed_range <= original_range
    
    def test_roughness_calculation(self, afm_analyzer, sample_afm_data):
        """Test surface roughness parameter calculation"""
        # Process first
        processed = afm_analyzer.process_height_map(sample_afm_data)
        
        # Calculate roughness
        roughness = afm_analyzer.calculate_roughness(processed)
        
        # Check all parameters present
        expected_params = ['Sa', 'Sq', 'Sp', 'Sv', 'Sz', 'Ssk', 'Sku', 'Sdr']
        for param in expected_params:
            assert param in roughness
        
        # Check reasonable values
        assert roughness['Sa'] > 0  # Average roughness
        assert roughness['Sq'] > roughness['Sa']  # RMS > average
        assert roughness['Sp'] > 0  # Peak height
        assert roughness['Sv'] > 0  # Valley depth
        assert roughness['Sz'] == roughness['Sp'] + roughness['Sv']
        assert -5 < roughness['Ssk'] < 5  # Skewness
        assert roughness['Sku'] > 0  # Kurtosis
        assert roughness['Sdr'] >= 0  # Surface area ratio
    
    def test_line_roughness(self, afm_analyzer, sample_afm_data):
        """Test line-by-line roughness analysis"""
        # Calculate line roughness
        roughness = afm_analyzer.calculate_roughness(
            sample_afm_data,
            line_by_line=True
        )
        
        # Check that mean and std are provided for each parameter
        params = ['Ra', 'Rq', 'Rp', 'Rv', 'Rt', 'Rsk', 'Rku']
        for param in params:
            assert f'{param}_mean' in roughness
            assert f'{param}_std' in roughness
            assert roughness[f'{param}_mean'] >= 0
            assert roughness[f'{param}_std'] >= 0
    
    def test_step_height_measurement(self, afm_analyzer, simulator):
        """Test step height measurement"""
        # Generate surface with steps
        step_data = simulator.generate_afm_data('steps', scan_size=(500, 500))
        
        # Measure step height
        result = afm_analyzer.measure_step_height(step_data)
        
        # Check results
        assert 'step_height' in result
        assert 'step_position' in result
        assert 'step_width' in result
        assert 'profile' in result
        
        assert result['step_height'] > 0
        assert result['step_position'] >= 0
        assert result['step_width'] >= 0
    
    def test_grain_structure_analysis(self, afm_analyzer, simulator):
        """Test grain structure analysis from AFM"""
        # Generate grain surface
        grain_data = simulator.generate_afm_data('grains', scan_size=(2000, 2000))
        
        # Analyze grains
        result = afm_analyzer.analyze_grain_structure(grain_data)
        
        # Check results
        assert 'num_grains' in result
        assert 'mean_grain_area' in result
        assert 'mean_grain_diameter' in result
        assert 'grain_areas' in result
        
        assert result['num_grains'] > 0
        assert result['mean_grain_area'] > 0
        assert result['mean_grain_diameter'] > 0
        assert len(result['grain_areas']) == result['num_grains']
    
    def test_power_spectrum_calculation(self, afm_analyzer, sample_afm_data):
        """Test power spectral density calculation"""
        # Calculate PSD
        result = afm_analyzer.calculate_power_spectrum(sample_afm_data)
        
        # Check results
        assert 'psd_2d' in result
        assert 'radial_psd' in result
        assert 'spatial_frequency' in result
        
        # Check dimensions
        assert result['psd_2d'].shape == sample_afm_data.height.shape
        assert len(result['radial_psd']) > 0
        assert len(result['spatial_frequency']) == len(result['radial_psd'])
        
        # Check that PSD is positive
        assert np.all(result['psd_2d'] >= 0)
        assert np.all(result['radial_psd'] >= 0)
    
    def test_force_curve_analysis(self, afm_analyzer, sample_afm_data):
        """Test force curve extraction and analysis"""
        # Extract force curves
        positions = [(100, 100), (200, 200)]
        force_curves = afm_analyzer.extract_force_curves(
            sample_afm_data,
            positions
        )
        
        # Check results
        assert isinstance(force_curves, list)
        
        for curve_analysis in force_curves:
            assert 'position' in curve_analysis
            assert 'contact_point' in curve_analysis
            assert 'stiffness' in curve_analysis
            assert 'young_modulus' in curve_analysis
            assert 'adhesion_force' in curve_analysis
    
    def test_adhesion_measurement(self, afm_analyzer):
        """Test adhesion force measurement"""
        # Generate synthetic force curve
        force_curve = np.zeros(200)
        force_curve[100:150] = np.linspace(0, 1, 50)  # Approach
        force_curve[150:] = np.linspace(1, -0.5, 50)  # Retract with adhesion
        
        # Measure adhesion
        result = afm_analyzer.measure_adhesion(
            force_curve,
            spring_constant=1.0
        )
        
        # Check results
        assert 'adhesion_force' in result
        assert 'snap_in_force' in result
        assert 'pull_off_index' in result
        
        assert result['adhesion_force'] > 0


class TestMicroscopySimulator:
    """Test suite for microscopy image simulation"""
    
    def test_sem_simulation(self, simulator):
        """Test SEM image generation"""
        image_types = ['particles', 'grains', 'porous', 'fibers']
        
        for img_type in image_types:
            image = simulator.generate_sem_image(
                img_type,
                size=(256, 256),
                pixel_size=5.0
            )
            
            assert isinstance(image, MicroscopyImage)
            assert image.shape == (256, 256)
            assert image.microscopy_type == MicroscopyType.SEM
            assert image.metadata['type'] == img_type
            
            # Check intensity range
            assert 0 <= np.min(image.image_data) <= 1
            assert 0 <= np.max(image.image_data) <= 1
    
    def test_tem_simulation(self, simulator):
        """Test TEM image generation"""
        image_types = ['lattice', 'diffraction', 'defects']
        
        for img_type in image_types:
            image = simulator.generate_tem_image(
                img_type,
                size=(256, 256),
                pixel_size=0.1
            )
            
            assert isinstance(image, MicroscopyImage)
            assert image.shape == (256, 256)
            assert image.microscopy_type == MicroscopyType.TEM
            assert image.metadata['type'] == img_type
    
    def test_afm_simulation(self, simulator):
        """Test AFM data generation"""
        surface_types = ['rough', 'steps', 'grains']
        
        for surf_type in surface_types:
            afm_data = simulator.generate_afm_data(
                surf_type,
                size=(128, 128),
                scan_size=(1000, 1000)
            )
            
            assert isinstance(afm_data, AFMData)
            assert afm_data.height.shape == (128, 128)
            assert afm_data.amplitude is not None
            assert afm_data.phase is not None
            assert len(afm_data.force_curves) > 0
            assert afm_data.metadata['type'] == surf_type
    
    def test_noise_addition(self, simulator):
        """Test realistic noise addition"""
        # Generate clean image
        clean_image = np.ones((100, 100)) * 0.5
        
        # Add SEM noise
        noisy = simulator._add_sem_noise(clean_image)
        
        # Check that noise was added
        assert not np.array_equal(clean_image, noisy)
        assert np.std(noisy) > np.std(clean_image)
        
        # Check range
        assert 0 <= np.min(noisy) <= 1
        assert 0 <= np.max(noisy) <= 1
    
    def test_reproducibility(self):
        """Test simulator reproducibility with same seed"""
        sim1 = MicroscopySimulator(seed=123)
        sim2 = MicroscopySimulator(seed=123)
        
        # Generate same images
        img1 = sim1.generate_sem_image('particles')
        img2 = sim2.generate_sem_image('particles')
        
        # Should be identical
        np.testing.assert_array_equal(img1.image_data, img2.image_data)


class TestIntegration:
    """Integration tests for complete microscopy workflows"""
    
    @pytest.mark.integration
    def test_complete_sem_workflow(self, sem_analyzer, simulator):
        """Test complete SEM analysis workflow"""
        # Generate image
        image = simulator.generate_sem_image('particles', pixel_size=5.0)
        
        # Process image
        processed = sem_analyzer.process_image(image)
        
        # Detect particles
        particles = sem_analyzer.detect_particles(processed, min_size=20)
        assert len(particles) > 0
        
        # Calculate statistics
        sizes = [p.diameter for p in particles]
        circularities = [p.circularity for p in particles]
        
        mean_size = np.mean(sizes)
        mean_circ = np.mean(circularities)
        
        assert 10 < mean_size < 100  # nm
        assert 0.5 < mean_circ < 1.0
        
        # Porosity analysis
        porosity_result = sem_analyzer.analyze_porosity(processed)
        assert 0 <= porosity_result['porosity_fraction'] <= 1
    
    @pytest.mark.integration
    def test_complete_tem_workflow(self, tem_analyzer, simulator):
        """Test complete TEM analysis workflow"""
        # Generate HRTEM image
        hrtem_image = simulator.generate_tem_image('lattice', pixel_size=0.1)
        
        # Process image
        processed = tem_analyzer.process_hrtem(hrtem_image, filter_type='wiener')
        
        # Measure lattice spacing
        lattice_result = tem_analyzer.measure_lattice_spacing(processed)
        
        if lattice_result['d_spacings']:
            # Check d-spacings are reasonable
            for d in lattice_result['d_spacings'][:3]:
                assert 1 < d < 5  # Typical range for semiconductors
        
        # Generate diffraction pattern
        diff_image = simulator.generate_tem_image('diffraction', pixel_size=0.01)
        
        # Analyze diffraction
        diff_result = tem_analyzer.analyze_diffraction_pattern(diff_image)
        assert diff_result['num_spots'] > 0
    
    @pytest.mark.integration
    def test_complete_afm_workflow(self, afm_analyzer, simulator):
        """Test complete AFM analysis workflow"""
        # Generate AFM data
        afm_data = simulator.generate_afm_data('rough', scan_size=(1000, 1000))
        
        # Process height map
        processed = afm_analyzer.process_height_map(afm_data, flatten=True)
        
        # Calculate roughness
        roughness = afm_analyzer.calculate_roughness(processed)
        
        # Check roughness values are reasonable
        assert 0 < roughness['Sa'] < 20  # nm
        assert roughness['Sq'] > roughness['Sa']
        
        # Analyze grain structure
        grain_result = afm_analyzer.analyze_grain_structure(processed)
        assert grain_result['num_grains'] > 0
        
        # Calculate power spectrum
        psd_result = afm_analyzer.calculate_power_spectrum(processed)
        assert len(psd_result['radial_psd']) > 0
    
    @pytest.mark.integration
    def test_multi_technique_correlation(self, sem_analyzer, afm_analyzer, simulator):
        """Test correlation between SEM and AFM measurements"""
        # Generate similar structures for both techniques
        # SEM for lateral dimensions
        sem_image = simulator.generate_sem_image('grains', pixel_size=10.0)
        
        # AFM for height information
        afm_data = simulator.generate_afm_data('grains', scan_size=(2000, 2000))
        
        # Analyze with SEM
        sem_grains = sem_analyzer.measure_grain_size(sem_image)
        
        # Analyze with AFM
        afm_grains = afm_analyzer.analyze_grain_structure(afm_data)
        
        # Both should detect similar number of grains
        assert abs(sem_grains['num_grains'] - afm_grains['num_grains']) < 20
        
        # Grain sizes should be correlated
        sem_diameter = sem_grains['mean_diameter']
        afm_diameter = afm_grains['mean_grain_diameter']
        
        # Allow for some difference due to different techniques
        assert 0.5 < sem_diameter / afm_diameter < 2.0


class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.performance
    def test_large_image_processing(self, sem_analyzer):
        """Test processing of large images"""
        # Create large image
        large_image = MicroscopyImage(
            image_data=np.random.rand(2048, 2048),
            microscopy_type=MicroscopyType.SEM,
            imaging_mode=ImagingMode.SE,
            pixel_size=5.0
        )
        
        # Process image
        start_time = time.time()
        processed = sem_analyzer.process_image(large_image)
        process_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert process_time < 10.0  # seconds
        assert processed.shape == (2048, 2048)
    
    @pytest.mark.performance
    def test_many_particles(self, sem_analyzer):
        """Test detection of many particles"""
        # Create image with many particles
        image_data = np.zeros((1024, 1024))
        
        # Add 100 particles
        for _ in range(100):
            x = np.random.randint(20, 1004)
            y = np.random.randint(20, 1004)
            radius = np.random.randint(5, 15)
            
            yy, xx = np.ogrid[:1024, :1024]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
            image_data[mask] = 1
        
        image = MicroscopyImage(
            image_data=image_data,
            microscopy_type=MicroscopyType.SEM,
            imaging_mode=ImagingMode.SE,
            pixel_size=5.0
        )
        
        # Detect particles
        start_time = time.time()
        particles = sem_analyzer.detect_particles(image)
        detect_time = time.time() - start_time
        
        # Should complete quickly
        assert detect_time < 5.0  # seconds
        assert len(particles) > 50  # Should find most particles
    
    @pytest.mark.performance
    def test_afm_large_scan(self, afm_analyzer):
        """Test AFM analysis of large scan"""
        # Create large AFM scan
        large_height = np.random.randn(1024, 1024) * 5
        
        afm_data = AFMData(
            height=large_height,
            scan_size=(5000, 5000),
            scan_rate=0.5
        )
        
        # Calculate roughness
        start_time = time.time()
        roughness = afm_analyzer.calculate_roughness(afm_data)
        calc_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert calc_time < 5.0  # seconds
        assert roughness['Sa'] > 0


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_image(self, sem_analyzer):
        """Test handling of empty/flat image"""
        # Create flat image
        flat_image = MicroscopyImage(
            image_data=np.ones((100, 100)) * 0.5,
            microscopy_type=MicroscopyType.SEM,
            imaging_mode=ImagingMode.SE,
            pixel_size=5.0
        )
        
        # Should handle gracefully
        particles = sem_analyzer.detect_particles(flat_image)
        assert len(particles) == 0
    
    def test_noisy_image(self, tem_analyzer):
        """Test handling of very noisy image"""
        # Create pure noise
        noise_image = MicroscopyImage(
            image_data=np.random.rand(256, 256),
            microscopy_type=MicroscopyType.TEM,
            imaging_mode=ImagingMode.BF,
            pixel_size=0.1
        )
        
        # Should not crash
        result = tem_analyzer.measure_lattice_spacing(noise_image)
        assert 'd_spacings' in result
    
    def test_invalid_parameters(self, afm_analyzer):
        """Test handling of invalid parameters"""
        # Create AFM data with negative heights
        invalid_height = np.random.randn(100, 100) - 10
        
        afm_data = AFMData(
            height=invalid_height,
            scan_size=(1000, 1000)
        )
        
        # Should handle negative values
        roughness = afm_analyzer.calculate_roughness(afm_data)
        assert roughness['Sa'] > 0  # Absolute values used
    
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatches"""
        # Create mismatched data
        with pytest.raises(ValueError):
            MicroscopyImage(
                image_data=np.array([1, 2, 3]),  # 1D array
                microscopy_type=MicroscopyType.SEM,
                imaging_mode=ImagingMode.SE,
                pixel_size=5.0
            )


class TestDataIntegrity:
    """Test data integrity and validation"""
    
    def test_image_normalization(self, sem_analyzer, sample_sem_image):
        """Test that processing preserves data range"""
        processed = sem_analyzer.process_image(sample_sem_image)
        
        # Check normalized range
        assert 0 <= np.min(processed.image_data) <= 1
        assert 0 <= np.max(processed.image_data) <= 1
        
        # Check no NaN or inf values
        assert not np.any(np.isnan(processed.image_data))
        assert not np.any(np.isinf(processed.image_data))
    
    def test_metadata_preservation(self, tem_analyzer, sample_tem_image):
        """Test that metadata is preserved through processing"""
        original_metadata = sample_tem_image.metadata.copy()
        
        processed = tem_analyzer.process_hrtem(sample_tem_image)
        
        # Original metadata should be preserved
        for key in original_metadata:
            assert key in processed.metadata
    
    def test_units_consistency(self, afm_analyzer, sample_afm_data):
        """Test units consistency in calculations"""
        # Calculate roughness
        roughness = afm_analyzer.calculate_roughness(sample_afm_data)
        
        # All roughness parameters should be in nm
        # and have reasonable magnitudes
        for key in ['Sa', 'Sq', 'Sp', 'Sv', 'Sz']:
            assert 0 < roughness[key] < 1000  # nm range


# Benchmark function
def run_benchmarks():
    """Run performance benchmarks"""
    import cProfile
    import pstats
    from io import StringIO
    
    sem_analyzer = SEMAnalyzer()
    tem_analyzer = TEMAnalyzer()
    afm_analyzer = AFMAnalyzer()
    simulator = MicroscopySimulator()
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Benchmark SEM processing
    for _ in range(10):
        image = simulator.generate_sem_image('particles')
        processed = sem_analyzer.process_image(image)
        particles = sem_analyzer.detect_particles(processed)
    
    # Benchmark TEM analysis
    for _ in range(10):
        image = simulator.generate_tem_image('lattice')
        processed = tem_analyzer.process_hrtem(image)
        tem_analyzer.measure_lattice_spacing(processed)
    
    # Benchmark AFM analysis
    for _ in range(10):
        afm_data = simulator.generate_afm_data('rough')
        processed = afm_analyzer.process_height_map(afm_data)
        afm_analyzer.calculate_roughness(processed)
    
    profiler.disable()
    
    # Generate report
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
    stats.print_stats(15)
    
    print("\nSession 10 Microscopy Analysis Performance Benchmark:")
    print(stream.getvalue())


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
    
    # Run benchmarks
    print("\n" + "=" * 80)
    print("Running Performance Benchmarks")
    print("=" * 80)
    run_benchmarks()
