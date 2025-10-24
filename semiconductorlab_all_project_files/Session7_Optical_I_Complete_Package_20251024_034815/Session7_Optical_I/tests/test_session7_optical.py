"""
Test Suite and Data Generators for Session 7: Optical I
UV-Vis-NIR and FTIR Spectroscopy
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import os

# Import the analyzers
from session7_uvvisnir_analyzer import (
    UVVisNIRAnalyzer, TransitionType, BaselineMethod,
    TaucPlotResult, UrbachTailResult, OpticalConstants
)
from session7_ftir_analyzer import (
    FTIRAnalyzer, Peak, FunctionalGroup, FTIRResult
)


# ============================================================================
# Test Data Generators
# ============================================================================

class OpticalDataGenerator:
    """Generate synthetic optical spectroscopy data for testing"""
    
    @staticmethod
    def generate_uvvisnir_spectrum(
        material: str = 'GaAs',
        wavelength_range: Tuple[float, float] = (300, 1000),
        n_points: int = 500,
        noise_level: float = 0.01,
        include_interference: bool = False,
        film_thickness: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic UV-Vis-NIR spectrum
        
        Args:
            material: Material type ('Si', 'GaAs', 'GaN', 'TiO2', 'ZnO')
            wavelength_range: Wavelength range in nm
            n_points: Number of data points
            noise_level: Noise amplitude (0-1)
            include_interference: Add thin film interference fringes
            film_thickness: Film thickness in nm for interference
            
        Returns:
            Dictionary with wavelength and intensity arrays
        """
        # Material band gaps (eV)
        band_gaps = {
            'Si': 1.12,      # Indirect
            'GaAs': 1.42,    # Direct
            'GaN': 3.4,      # Direct
            'TiO2': 3.2,     # Indirect
            'ZnO': 3.37,     # Direct
            'CdTe': 1.5,     # Direct
            'InP': 1.35,     # Direct
            'a-Si': 1.7      # Amorphous silicon
        }
        
        # Transition types
        transition_types = {
            'Si': 'indirect',
            'GaAs': 'direct',
            'GaN': 'direct',
            'TiO2': 'indirect',
            'ZnO': 'direct',
            'CdTe': 'direct',
            'InP': 'direct',
            'a-Si': 'direct'
        }
        
        # Generate wavelength array
        wavelength = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
        
        # Convert to energy
        h = 6.626e-34  # Planck constant
        c = 2.998e8    # Speed of light
        e = 1.602e-19  # Elementary charge
        energy = (h * c) / (wavelength * 1e-9 * e)  # eV
        
        # Get material parameters
        eg = band_gaps.get(material, 2.0)
        transition = transition_types.get(material, 'direct')
        
        # Generate absorption coefficient
        alpha = np.zeros_like(energy)
        
        for i, E in enumerate(energy):
            if E > eg:
                if transition == 'direct':
                    # Direct transition: α ∝ √(E - Eg)
                    alpha[i] = 1e4 * np.sqrt(E - eg)
                else:
                    # Indirect transition: α ∝ (E - Eg)²
                    alpha[i] = 1e3 * (E - eg)**2
                    
        # Add Urbach tail
        urbach_energy = 0.05  # 50 meV
        for i, E in enumerate(energy):
            if E < eg:
                alpha[i] = 100 * np.exp((E - eg) / urbach_energy)
        
        # Convert to transmission (Beer-Lambert)
        if film_thickness:
            thickness_cm = film_thickness * 1e-7
            transmission = np.exp(-alpha * thickness_cm)
        else:
            # Arbitrary thickness
            transmission = np.exp(-alpha / 1e4)
        
        # Add interference fringes if requested
        if include_interference and film_thickness:
            n_film = 2.0  # Refractive index
            # Optical path difference
            opd = 2 * n_film * film_thickness
            # Interference pattern
            phase = 4 * np.pi * opd / wavelength
            interference = 0.1 * np.sin(phase)
            transmission = transmission * (1 + interference)
        
        # Ensure transmission is between 0 and 1
        transmission = np.clip(transmission, 0, 1)
        
        # Convert to percentage
        transmission = transmission * 100
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level * 100, len(transmission))
            transmission = transmission + noise
            transmission = np.clip(transmission, 0, 100)
        
        return {
            'wavelength': wavelength,
            'transmission': transmission,
            'material': material,
            'band_gap': eg,
            'transition_type': transition
        }
    
    @staticmethod
    def generate_ftir_spectrum(
        sample_type: str = 'SiO2',
        wavenumber_range: Tuple[float, float] = (400, 4000),
        n_points: int = 3600,
        noise_level: float = 0.02,
        baseline_drift: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Generate synthetic FTIR spectrum
        
        Args:
            sample_type: Sample type ('SiO2', 'Si3N4', 'polymer', 'protein')
            wavenumber_range: Wavenumber range in cm⁻¹
            n_points: Number of data points
            noise_level: Noise amplitude
            baseline_drift: Baseline curvature
            
        Returns:
            Dictionary with wavenumber and absorbance arrays
        """
        # Generate wavenumber array
        wavenumber = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)
        
        # Initialize spectrum
        absorbance = np.zeros_like(wavenumber)
        
        # Define peak sets for different materials
        peak_sets = {
            'SiO2': [
                (460, 80, 40),    # Si-O-Si bend
                (800, 60, 30),    # Si-O-Si symmetric stretch
                (1080, 100, 50),  # Si-O-Si asymmetric stretch
                (3400, 40, 150),  # O-H stretch (surface)
            ],
            'Si3N4': [
                (490, 70, 35),    # Si-N bend
                (840, 90, 40),    # Si-N stretch
                (1000, 60, 30),   # Si-N overtone
                (3350, 30, 100),  # N-H stretch
            ],
            'polymer': [
                (1050, 50, 30),   # C-O stretch
                (1450, 40, 25),   # CH2 bend
                (1650, 60, 30),   # C=C stretch
                (1730, 80, 25),   # C=O stretch
                (2850, 70, 30),   # CH2 symmetric stretch
                (2920, 90, 35),   # CH2 asymmetric stretch
                (3300, 60, 120),  # O-H stretch
            ],
            'protein': [
                (1540, 80, 40),   # Amide II
                (1650, 100, 45),  # Amide I
                (2930, 60, 40),   # C-H stretch
                (3300, 70, 150),  # N-H/O-H stretch
            ],
            'organic': [
                (1380, 30, 20),   # CH3 bend
                (1460, 35, 25),   # CH2 bend
                (1600, 40, 30),   # Aromatic C=C
                (1710, 70, 25),   # C=O stretch
                (2860, 50, 30),   # CH3 symmetric
                (2930, 60, 35),   # CH3 asymmetric
                (2960, 45, 25),   # CH3 stretch
                (3050, 30, 40),   # Aromatic C-H
            ]
        }
        
        # Get peaks for sample type
        peaks = peak_sets.get(sample_type, peak_sets['SiO2'])
        
        # Add peaks to spectrum
        for position, intensity, width in peaks:
            # Gaussian peak
            peak = intensity * np.exp(-0.5 * ((wavenumber - position) / width)**2)
            absorbance += peak / 100  # Scale to absorbance units
        
        # Add baseline drift
        if baseline_drift > 0:
            # Polynomial baseline
            baseline = baseline_drift * (
                0.1 + 
                0.05 * (wavenumber - 2000) / 1000 +
                0.02 * ((wavenumber - 2000) / 1000)**2
            )
            absorbance += baseline
        
        # Add noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, len(absorbance))
            absorbance += noise
        
        # Ensure positive values
        absorbance = np.maximum(absorbance, 0)
        
        return {
            'wavenumber': wavenumber,
            'absorbance': absorbance,
            'sample_type': sample_type,
            'peaks': peaks
        }
    
    @staticmethod
    def generate_batch_spectra(
        method: str = 'uvvisnir',
        n_samples: int = 10,
        variations: bool = True
    ) -> List[Dict]:
        """
        Generate batch of spectra with variations
        
        Args:
            method: 'uvvisnir' or 'ftir'
            n_samples: Number of samples
            variations: Add sample-to-sample variations
            
        Returns:
            List of spectrum dictionaries
        """
        spectra = []
        
        if method == 'uvvisnir':
            materials = ['GaAs', 'GaN', 'Si', 'ZnO', 'TiO2']
            
            for i in range(n_samples):
                material = materials[i % len(materials)]
                
                # Add variations
                if variations:
                    noise = 0.01 + np.random.uniform(-0.005, 0.005)
                    thickness = np.random.uniform(100, 1000) if i % 2 == 0 else None
                else:
                    noise = 0.01
                    thickness = None
                
                spectrum = OpticalDataGenerator.generate_uvvisnir_spectrum(
                    material=material,
                    noise_level=noise,
                    film_thickness=thickness,
                    include_interference=(thickness is not None)
                )
                
                spectrum['sample_id'] = f"UV_{i+1:03d}"
                spectrum['timestamp'] = datetime.now().isoformat()
                spectra.append(spectrum)
                
        elif method == 'ftir':
            sample_types = ['SiO2', 'Si3N4', 'polymer', 'protein', 'organic']
            
            for i in range(n_samples):
                sample_type = sample_types[i % len(sample_types)]
                
                # Add variations
                if variations:
                    noise = 0.02 + np.random.uniform(-0.01, 0.01)
                    drift = 0.1 + np.random.uniform(-0.05, 0.05)
                else:
                    noise = 0.02
                    drift = 0.1
                
                spectrum = OpticalDataGenerator.generate_ftir_spectrum(
                    sample_type=sample_type,
                    noise_level=noise,
                    baseline_drift=drift
                )
                
                spectrum['sample_id'] = f"FTIR_{i+1:03d}"
                spectrum['timestamp'] = datetime.now().isoformat()
                spectra.append(spectrum)
        
        return spectra


# ============================================================================
# UV-Vis-NIR Analyzer Tests
# ============================================================================

class TestUVVisNIRAnalyzer:
    """Test suite for UV-Vis-NIR analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return UVVisNIRAnalyzer()
    
    @pytest.fixture
    def test_spectrum(self):
        """Generate test spectrum"""
        return OpticalDataGenerator.generate_uvvisnir_spectrum(
            material='GaAs',
            noise_level=0.01
        )
    
    def test_spectrum_processing(self, analyzer, test_spectrum):
        """Test basic spectrum processing"""
        result = analyzer.process_spectrum(
            test_spectrum['wavelength'],
            test_spectrum['transmission'],
            mode='transmission',
            baseline_method=BaselineMethod.ALS
        )
        
        assert 'wavelength' in result
        assert 'corrected' in result
        assert 'absorbance' in result
        assert len(result['wavelength']) == len(test_spectrum['wavelength'])
    
    def test_tauc_plot_direct(self, analyzer, test_spectrum):
        """Test Tauc plot for direct transition"""
        # Convert to absorbance
        absorbance = -np.log10(test_spectrum['transmission'] / 100)
        
        result = analyzer.calculate_tauc_plot(
            test_spectrum['wavelength'],
            absorbance,
            transition_type=TransitionType.DIRECT
        )
        
        assert isinstance(result, TaucPlotResult)
        assert 1.0 < result.band_gap < 2.0  # GaAs ~1.42 eV
        assert result.r_squared > 0.8
        assert result.transition_type == 'direct'
    
    def test_tauc_plot_indirect(self, analyzer):
        """Test Tauc plot for indirect transition"""
        # Generate Si spectrum (indirect)
        spectrum = OpticalDataGenerator.generate_uvvisnir_spectrum(
            material='Si',
            noise_level=0.005
        )
        
        absorbance = -np.log10(spectrum['transmission'] / 100)
        
        result = analyzer.calculate_tauc_plot(
            spectrum['wavelength'],
            absorbance,
            transition_type=TransitionType.INDIRECT
        )
        
        assert isinstance(result, TaucPlotResult)
        assert 0.8 < result.band_gap < 1.5  # Si ~1.12 eV
        assert result.transition_type == 'indirect'
    
    def test_urbach_tail_analysis(self, analyzer, test_spectrum):
        """Test Urbach tail analysis"""
        absorbance = -np.log10(test_spectrum['transmission'] / 100)
        
        result = analyzer.analyze_urbach_tail(
            test_spectrum['wavelength'],
            absorbance
        )
        
        assert isinstance(result, UrbachTailResult)
        assert result.urbach_energy > 0
        assert 0 <= result.r_squared <= 1
        assert result.fit_quality in ['Excellent', 'Good', 'Fair', 'Poor']
    
    def test_optical_constants(self, analyzer, test_spectrum):
        """Test optical constants calculation"""
        transmission = test_spectrum['transmission'] / 100
        
        result = analyzer.calculate_optical_constants(
            test_spectrum['wavelength'],
            transmission=transmission,
            film_thickness=500  # nm
        )
        
        assert isinstance(result, OpticalConstants)
        assert len(result.n) == len(test_spectrum['wavelength'])
        assert len(result.k) == len(test_spectrum['wavelength'])
        assert np.all(result.n >= 0)
        assert np.all(result.k >= 0)
    
    def test_interference_fringe_removal(self, analyzer):
        """Test interference fringe removal"""
        # Generate spectrum with fringes
        spectrum = OpticalDataGenerator.generate_uvvisnir_spectrum(
            material='GaAs',
            include_interference=True,
            film_thickness=1000
        )
        
        # Remove fringes
        corrected = analyzer.remove_interference_fringes(
            spectrum['wavelength'],
            spectrum['transmission'],
            method='fft'
        )
        
        assert len(corrected) == len(spectrum['transmission'])
        # Check that high-frequency components are reduced
        fft_original = np.fft.fft(spectrum['transmission'])
        fft_corrected = np.fft.fft(corrected)
        assert np.sum(np.abs(fft_corrected[100:])) < np.sum(np.abs(fft_original[100:]))
    
    def test_baseline_methods(self, analyzer, test_spectrum):
        """Test different baseline correction methods"""
        methods = [
            BaselineMethod.POLYNOMIAL,
            BaselineMethod.SPLINE,
            BaselineMethod.RUBBERBAND,
            BaselineMethod.ALS
        ]
        
        for method in methods:
            result = analyzer.process_spectrum(
                test_spectrum['wavelength'],
                test_spectrum['transmission'],
                baseline_method=method
            )
            
            assert 'baseline' in result
            assert 'corrected' in result
            # Baseline should be smoother than original
            baseline_roughness = np.std(np.diff(result['baseline']))
            signal_roughness = np.std(np.diff(result['raw']))
            assert baseline_roughness < signal_roughness
    
    def test_batch_processing(self, analyzer):
        """Test batch processing of multiple spectra"""
        # Generate batch
        spectra = OpticalDataGenerator.generate_batch_spectra(
            method='uvvisnir',
            n_samples=5
        )
        
        # Create file paths (simulated)
        file_paths = [f"spectrum_{i}.csv" for i in range(5)]
        
        # Process batch (with mocked file loading)
        results = []
        for spectrum in spectra:
            processed = analyzer.process_spectrum(
                spectrum['wavelength'],
                spectrum['transmission']
            )
            results.append(processed)
        
        assert len(results) == 5
        for result in results:
            assert 'corrected' in result


# ============================================================================
# FTIR Analyzer Tests
# ============================================================================

class TestFTIRAnalyzer:
    """Test suite for FTIR analyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return FTIRAnalyzer()
    
    @pytest.fixture
    def test_spectrum(self):
        """Generate test FTIR spectrum"""
        return OpticalDataGenerator.generate_ftir_spectrum(
            sample_type='SiO2',
            noise_level=0.02
        )
    
    def test_spectrum_processing(self, analyzer, test_spectrum):
        """Test FTIR spectrum processing"""
        result = analyzer.process_spectrum(
            test_spectrum['wavenumber'],
            test_spectrum['absorbance'],
            mode='absorbance',
            baseline_method='als'
        )
        
        assert isinstance(result, FTIRResult)
        assert len(result.wavenumber) == len(test_spectrum['wavenumber'])
        assert result.peaks is not None
        assert result.functional_groups is not None
    
    def test_peak_detection(self, analyzer, test_spectrum):
        """Test peak detection"""
        result = analyzer.process_spectrum(
            test_spectrum['wavenumber'],
            test_spectrum['absorbance']
        )
        
        # Should find major peaks
        assert len(result.peaks) > 0
        
        # Check peak properties
        for peak in result.peaks:
            assert isinstance(peak, Peak)
            assert 400 <= peak.position <= 4000
            assert peak.intensity > 0
            assert peak.width > 0
            assert peak.area > 0
    
    def test_functional_group_identification(self, analyzer):
        """Test functional group identification"""
        # Generate polymer spectrum with known groups
        spectrum = OpticalDataGenerator.generate_ftir_spectrum(
            sample_type='polymer',
            noise_level=0.01
        )
        
        result = analyzer.process_spectrum(
            spectrum['wavenumber'],
            spectrum['absorbance']
        )
        
        # Should identify functional groups
        assert len(result.functional_groups) > 0
        
        # Check for expected groups
        group_names = [g.name for g in result.functional_groups]
        # Polymer should have C-H, C=O, O-H groups
        assert any('C-H' in name for name in group_names)
        assert any('C=O' in name or 'C-O' in name for name in group_names)
    
    def test_atr_correction(self, analyzer, test_spectrum):
        """Test ATR correction"""
        result_no_atr = analyzer.process_spectrum(
            test_spectrum['wavenumber'],
            test_spectrum['absorbance'],
            atr_correction=False
        )
        
        result_with_atr = analyzer.process_spectrum(
            test_spectrum['wavenumber'],
            test_spectrum['absorbance'],
            atr_correction=True,
            atr_crystal='ZnSe'
        )
        
        # ATR correction should modify intensities
        assert not np.array_equal(
            result_no_atr.corrected,
            result_with_atr.corrected
        )
    
    def test_quantitative_analysis(self, analyzer, test_spectrum):
        """Test quantitative analysis"""
        result = analyzer.process_spectrum(
            test_spectrum['wavenumber'],
            test_spectrum['absorbance']
        )
        
        # Perform quantitative analysis
        quant_results = analyzer.quantitative_analysis(result.peaks)
        
        assert isinstance(quant_results, dict)
        # Should have area measurements
        assert any('area' in key for key in quant_results.keys())
    
    def test_spectrum_comparison(self, analyzer):
        """Test spectrum comparison methods"""
        # Generate multiple spectra
        spectra_data = OpticalDataGenerator.generate_batch_spectra(
            method='ftir',
            n_samples=3
        )
        
        # Extract intensity arrays
        spectra = [s['absorbance'] for s in spectra_data]
        wavenumber = spectra_data[0]['wavenumber']
        
        # Test correlation method
        corr_matrix = analyzer.compare_spectra(
            spectra,
            wavenumber,
            method='correlation'
        )
        
        assert corr_matrix.shape == (3, 3)
        assert np.all(np.diag(corr_matrix) > 0.99)  # Self-correlation ~1
        
        # Test PCA method
        pca_results = analyzer.compare_spectra(
            spectra,
            wavenumber,
            method='pca'
        )
        
        assert 'scores' in pca_results
        assert 'variance_explained' in pca_results
        assert pca_results['scores'].shape[0] == 3
    
    def test_quality_metrics(self, analyzer, test_spectrum):
        """Test spectrum quality metrics"""
        result = analyzer.process_spectrum(
            test_spectrum['wavenumber'],
            test_spectrum['absorbance']
        )
        
        metrics = result.quality_metrics
        
        assert 'snr' in metrics
        assert 'baseline_std' in metrics
        assert 'n_peaks' in metrics
        assert metrics['snr'] > 0
        assert metrics['n_peaks'] > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_uvvisnir_workflow(self):
        """Test complete UV-Vis-NIR workflow"""
        # Generate data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_uvvisnir_spectrum(
            material='GaN',
            include_interference=True,
            film_thickness=500
        )
        
        # Create analyzer
        analyzer = UVVisNIRAnalyzer()
        
        # Process spectrum
        processed = analyzer.process_spectrum(
            spectrum['wavelength'],
            spectrum['transmission'],
            mode='transmission',
            baseline_method=BaselineMethod.ALS
        )
        
        # Calculate band gap
        tauc = analyzer.calculate_tauc_plot(
            processed['wavelength'],
            processed['absorbance'],
            transition_type=TransitionType.DIRECT
        )
        
        # Validate results
        assert 3.0 < tauc.band_gap < 3.8  # GaN ~3.4 eV
        assert tauc.r_squared > 0.85
        
        # Calculate optical constants
        optical = analyzer.calculate_optical_constants(
            processed['wavelength'],
            transmission=spectrum['transmission']/100,
            film_thickness=500
        )
        
        assert np.mean(optical.n) > 1.5  # Typical for semiconductors
    
    def test_ftir_workflow(self):
        """Test complete FTIR workflow"""
        # Generate data
        generator = OpticalDataGenerator()
        spectrum = generator.generate_ftir_spectrum(
            sample_type='polymer',
            baseline_drift=0.15
        )
        
        # Create analyzer
        analyzer = FTIRAnalyzer()
        
        # Process spectrum
        result = analyzer.process_spectrum(
            spectrum['wavenumber'],
            spectrum['absorbance'],
            baseline_method='als',
            atr_correction=True
        )
        
        # Validate results
        assert len(result.peaks) >= 5  # Polymer has multiple peaks
        assert len(result.functional_groups) >= 3
        assert result.quality_metrics['snr'] > 10
        
        # Quantitative analysis
        quant = analyzer.quantitative_analysis(result.peaks)
        assert len(quant) > 0
    
    def test_batch_analysis_workflow(self):
        """Test batch analysis workflow"""
        # Generate batch data
        generator = OpticalDataGenerator()
        
        # UV-Vis-NIR batch
        uv_spectra = generator.generate_batch_spectra(
            method='uvvisnir',
            n_samples=10,
            variations=True
        )
        
        uv_analyzer = UVVisNIRAnalyzer()
        uv_results = []
        
        for spectrum in uv_spectra:
            processed = uv_analyzer.process_spectrum(
                spectrum['wavelength'],
                spectrum['transmission']
            )
            
            # Extract band gap
            tauc = uv_analyzer.calculate_tauc_plot(
                processed['wavelength'],
                processed['absorbance']
            )
            
            uv_results.append({
                'sample_id': spectrum['sample_id'],
                'band_gap': tauc.band_gap,
                'r_squared': tauc.r_squared
            })
        
        # FTIR batch
        ftir_spectra = generator.generate_batch_spectra(
            method='ftir',
            n_samples=10,
            variations=True
        )
        
        ftir_analyzer = FTIRAnalyzer()
        ftir_results = []
        
        for spectrum in ftir_spectra:
            result = ftir_analyzer.process_spectrum(
                spectrum['wavenumber'],
                spectrum['absorbance']
            )
            
            ftir_results.append({
                'sample_id': spectrum['sample_id'],
                'n_peaks': len(result.peaks),
                'n_groups': len(result.functional_groups),
                'snr': result.quality_metrics['snr']
            })
        
        # Validate batch results
        assert len(uv_results) == 10
        assert len(ftir_results) == 10
        
        # Check consistency
        band_gaps = [r['band_gap'] for r in uv_results]
        assert np.std(band_gaps) < 1.0  # Reasonable variation


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks"""
    
    def test_uvvisnir_processing_speed(self):
        """Test UV-Vis-NIR processing speed"""
        import time
        
        generator = OpticalDataGenerator()
        analyzer = UVVisNIRAnalyzer()
        
        # Generate large spectrum
        spectrum = generator.generate_uvvisnir_spectrum(
            n_points=10000
        )
        
        start = time.time()
        
        # Process spectrum
        processed = analyzer.process_spectrum(
            spectrum['wavelength'],
            spectrum['transmission']
        )
        
        # Calculate band gap
        tauc = analyzer.calculate_tauc_plot(
            processed['wavelength'],
            processed['absorbance']
        )
        
        elapsed = time.time() - start
        
        assert elapsed < 2.0  # Should complete in under 2 seconds
        print(f"UV-Vis-NIR processing time: {elapsed:.3f}s")
    
    def test_ftir_processing_speed(self):
        """Test FTIR processing speed"""
        import time
        
        generator = OpticalDataGenerator()
        analyzer = FTIRAnalyzer()
        
        # Generate large spectrum
        spectrum = generator.generate_ftir_spectrum(
            n_points=10000
        )
        
        start = time.time()
        
        # Process spectrum
        result = analyzer.process_spectrum(
            spectrum['wavenumber'],
            spectrum['absorbance']
        )
        
        elapsed = time.time() - start
        
        assert elapsed < 3.0  # Should complete in under 3 seconds
        print(f"FTIR processing time: {elapsed:.3f}s")
    
    def test_memory_usage(self):
        """Test memory usage for large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Initial memory
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate and process large batch
        generator = OpticalDataGenerator()
        spectra = generator.generate_batch_spectra(
            method='uvvisnir',
            n_samples=100
        )
        
        analyzer = UVVisNIRAnalyzer()
        results = []
        
        for spectrum in spectra:
            processed = analyzer.process_spectrum(
                spectrum['wavelength'],
                spectrum['transmission']
            )
            results.append(processed)
        
        # Final memory
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_after - mem_before
        
        assert mem_increase < 500  # Should use less than 500 MB
        print(f"Memory increase: {mem_increase:.1f} MB")


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Generate sample data for manual testing
    print("\nGenerating sample data files...")
    
    generator = OpticalDataGenerator()
    
    # UV-Vis-NIR samples
    for material in ['GaAs', 'GaN', 'Si', 'ZnO']:
        spectrum = generator.generate_uvvisnir_spectrum(material=material)
        
        df = pd.DataFrame({
            'Wavelength_nm': spectrum['wavelength'],
            'Transmission_%': spectrum['transmission']
        })
        
        filename = f"test_data/uvvisnir_{material.lower()}.csv"
        os.makedirs('test_data', exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"Created: {filename}")
    
    # FTIR samples
    for sample in ['SiO2', 'polymer', 'protein']:
        spectrum = generator.generate_ftir_spectrum(sample_type=sample)
        
        df = pd.DataFrame({
            'Wavenumber_cm-1': spectrum['wavenumber'],
            'Absorbance': spectrum['absorbance']
        })
        
        filename = f"test_data/ftir_{sample.lower()}.csv"
        df.to_csv(filename, index=False)
        print(f"Created: {filename}")
    
    print("\nTest data generation complete!")
