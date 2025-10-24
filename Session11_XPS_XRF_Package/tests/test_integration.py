"""
Session 11: XPS/XRF Analysis - Integration Tests
Comprehensive test suite for surface and elemental analysis
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os
from datetime import datetime
import asyncio

# Import the modules to test
from session11_xps_xrf_complete_implementation import (
    XPSAnalyzer, XRFAnalyzer, ChemicalSimulator, ElementDatabase,
    XRaySource, PeakShape, Element, XPSPeak, XRFPeak
)

# Fixtures
@pytest.fixture
def element_db():
    """Provide element database"""
    return ElementDatabase()

@pytest.fixture
def xps_analyzer():
    """Create XPS analyzer instance"""
    return XPSAnalyzer(source=XRaySource.AL_KA)

@pytest.fixture
def xrf_analyzer():
    """Create XRF analyzer instance"""
    return XRFAnalyzer(excitation_energy=50.0)

@pytest.fixture
def simulator():
    """Create chemical simulator"""
    return ChemicalSimulator()

@pytest.fixture
def sample_xps_spectrum():
    """Generate sample XPS spectrum"""
    be = np.linspace(0, 1200, 2400)
    intensity = np.random.normal(1000, 100, len(be))
    
    # Add peaks
    intensity += 5000 * np.exp(-((be - 284.5)**2) / 2)  # C 1s
    intensity += 3000 * np.exp(-((be - 532.5)**2) / 3)  # O 1s
    intensity += 2000 * np.exp(-((be - 399.5)**2) / 2)  # N 1s
    
    return be, np.maximum(intensity, 0)

@pytest.fixture
def sample_xrf_spectrum():
    """Generate sample XRF spectrum"""
    energy = np.linspace(0.1, 20, 2000)
    counts = np.random.poisson(100, len(energy))
    
    # Add element peaks
    counts += 5000 * np.exp(-((energy - 1.74)**2) / 0.001)  # Si Kα
    counts += 3000 * np.exp(-((energy - 6.404)**2) / 0.002)  # Fe Kα
    
    return energy, counts

class TestElementDatabase:
    """Test element database functionality"""
    
    def test_element_retrieval(self, element_db):
        """Test getting element properties"""
        carbon = element_db.get_element('C')
        assert carbon is not None
        assert carbon.symbol == 'C'
        assert carbon.atomic_number == 6
        assert 'C-C' in carbon.xps_peaks
        assert 'Kα' in carbon.xrf_lines
    
    def test_sensitivity_factors(self, element_db):
        """Test sensitivity factor retrieval"""
        rsf = element_db.get_sensitivity_factor('Si', '2p', XRaySource.AL_KA)
        assert rsf > 0
        assert rsf < 10
        
        # Test different source correction
        rsf_mg = element_db.get_sensitivity_factor('Si', '2p', XRaySource.MG_KA)
        assert rsf_mg != rsf
    
    def test_all_elements_present(self, element_db):
        """Test that all key elements are in database"""
        expected_elements = ['C', 'O', 'N', 'Si', 'Ga', 'As', 'Al', 'Au', 'Ti', 'Fe']
        for elem in expected_elements:
            assert element_db.get_element(elem) is not None
    
    def test_element_properties_complete(self, element_db):
        """Test completeness of element properties"""
        si = element_db.get_element('Si')
        assert len(si.xps_peaks) > 0
        assert len(si.xrf_lines) > 0
        assert len(si.sensitivity_factors) > 0
        assert si.fluorescence_yield > 0

class TestXPSAnalyzer:
    """Test XPS analysis functionality"""
    
    def test_spectrum_processing(self, xps_analyzer, sample_xps_spectrum):
        """Test basic spectrum processing"""
        be, intensity = sample_xps_spectrum
        be_proc, int_proc = xps_analyzer.process_spectrum(be, intensity)
        
        assert len(be_proc) == len(be)
        assert len(int_proc) == len(intensity)
        assert np.all(int_proc >= 0)
    
    def test_peak_finding(self, xps_analyzer, sample_xps_spectrum):
        """Test peak detection"""
        be, intensity = sample_xps_spectrum
        peaks = xps_analyzer.find_peaks(be, intensity, prominence=0.2)
        
        assert len(peaks) >= 3
        # Check for expected peaks
        c1s_found = any(abs(p['position'] - 284.5) < 5 for p in peaks)
        o1s_found = any(abs(p['position'] - 532.5) < 5 for p in peaks)
        assert c1s_found
        assert o1s_found
    
    def test_shirley_background(self, xps_analyzer):
        """Test Shirley background calculation"""
        be = np.linspace(280, 290, 100)
        intensity = 1000 + 5000 * np.exp(-((be - 285)**2) / 2)
        
        background = xps_analyzer.shirley_background(be, intensity)
        
        assert len(background) == len(intensity)
        assert np.all(background <= intensity)
        assert np.all(background >= 0)
    
    def test_tougaard_background(self, xps_analyzer):
        """Test Tougaard background calculation"""
        be = np.linspace(280, 290, 100)
        intensity = 1000 + 5000 * np.exp(-((be - 285)**2) / 2)
        
        background = xps_analyzer.tougaard_background(be, intensity)
        
        assert len(background) == len(intensity)
        assert np.all(background >= 0)
    
    def test_peak_fitting_gaussian(self, xps_analyzer):
        """Test Gaussian peak fitting"""
        be = np.linspace(280, 290, 100)
        intensity = 1000 + 5000 * np.exp(-((be - 285)**2) / 2) + np.random.normal(0, 50, 100)
        
        result = xps_analyzer.fit_peak(be, intensity, shape=PeakShape.GAUSSIAN)
        
        assert result['success']
        assert abs(result['position'] - 285) < 0.5
        assert result['r_squared'] > 0.9
        assert result['area'] > 0
    
    def test_peak_fitting_voigt(self, xps_analyzer):
        """Test Voigt peak fitting"""
        be = np.linspace(280, 290, 100)
        # Generate Voigt-like peak
        intensity = 1000 + 5000 * np.exp(-((be - 285)**2) / 2)
        
        result = xps_analyzer.fit_peak(be, intensity, shape=PeakShape.VOIGT)
        
        assert result['success']
        assert abs(result['position'] - 285) < 0.5
        assert result['fwhm'] > 0
    
    def test_multiplet_splitting(self, xps_analyzer):
        """Test multiplet splitting calculation"""
        splitting = xps_analyzer.multiplet_splitting('Ti', '2p')
        
        assert splitting is not None
        assert '2p3/2' in splitting
        assert '2p1/2' in splitting
        assert splitting['ratio'] == 2.0
        assert splitting['separation'] > 0
    
    def test_quantification(self, xps_analyzer):
        """Test atomic composition quantification"""
        peaks = [
            XPSPeak(position=284.5, area=10000, fwhm=1.4, shape=PeakShape.VOIGT,
                   orbital='1s', element='C'),
            XPSPeak(position=532.5, area=8000, fwhm=1.6, shape=PeakShape.VOIGT,
                   orbital='1s', element='O'),
            XPSPeak(position=399.5, area=3000, fwhm=1.5, shape=PeakShape.VOIGT,
                   orbital='1s', element='N')
        ]
        
        composition = xps_analyzer.quantification(peaks)
        
        assert 'C' in composition
        assert 'O' in composition
        assert 'N' in composition
        assert abs(sum(composition.values()) - 100) < 0.1
        assert composition['C'] > composition['N']
    
    def test_depth_profile(self, xps_analyzer):
        """Test depth profile analysis"""
        # Generate series of spectra
        spectra = []
        etch_times = np.array([0, 60, 120, 180, 240])
        
        for t in etch_times:
            be = np.linspace(280, 540, 500)
            # Simulate changing composition with depth
            c_intensity = 5000 * np.exp(-t/100) * np.exp(-((be - 284.5)**2) / 2)
            o_intensity = 3000 * (1 - np.exp(-t/100)) * np.exp(-((be - 532.5)**2) / 2)
            intensity = 1000 + c_intensity + o_intensity
            spectra.append((be, intensity))
        
        profile = xps_analyzer.depth_profile(spectra, etch_times, ['C', 'O'])
        
        assert 'depth' in profile
        assert 'C' in profile
        assert 'O' in profile
        assert len(profile['depth']) == len(etch_times)
        # Check that C decreases and O increases with depth
        assert profile['C'][0] > profile['C'][-1]
        assert profile['O'][0] < profile['O'][-1]
    
    def test_chemical_state_analysis(self, xps_analyzer):
        """Test chemical state identification"""
        be = np.linspace(280, 290, 100)
        # Create multi-component C 1s peak
        intensity = (3000 * np.exp(-((be - 284.5)**2) / 1) +  # C-C
                    2000 * np.exp(-((be - 286.0)**2) / 1) +  # C-O
                    1000 * np.exp(-((be - 288.0)**2) / 1))   # C=O
        
        result = xps_analyzer.chemical_state_analysis(be, intensity, 'C', '1s')
        
        assert 'identified_states' in result
        assert len(result['identified_states']) >= 1
        assert 'reference_states' in result

class TestXRFAnalyzer:
    """Test XRF analysis functionality"""
    
    def test_spectrum_processing(self, xrf_analyzer, sample_xrf_spectrum):
        """Test XRF spectrum processing"""
        energy, counts = sample_xrf_spectrum
        energy_proc, counts_proc = xrf_analyzer.process_spectrum(energy, counts)
        
        assert len(energy_proc) == len(energy)
        assert len(counts_proc) == len(counts)
        assert np.all(counts_proc >= 0)
    
    def test_dead_time_correction(self, xrf_analyzer):
        """Test dead time correction"""
        counts = np.array([1000, 2000, 3000, 4000])
        corrected = xrf_analyzer._dead_time_correction(counts)
        
        assert len(corrected) == len(counts)
        assert np.all(corrected >= counts)  # Correction should increase counts
    
    def test_peak_finding(self, xrf_analyzer, sample_xrf_spectrum):
        """Test XRF peak detection"""
        energy, counts = sample_xrf_spectrum
        peaks = xrf_analyzer.find_peaks(energy, counts, prominence=0.1)
        
        assert len(peaks) >= 2
        # Check for Si and Fe peaks
        si_found = any(abs(p['energy'] - 1.74) < 0.1 for p in peaks)
        fe_found = any(abs(p['energy'] - 6.404) < 0.1 for p in peaks)
        assert si_found
        assert fe_found
    
    def test_element_line_identification(self, xrf_analyzer):
        """Test element and line identification"""
        # Test Si Kα
        element_line = xrf_analyzer._identify_element_line(1.74)
        assert element_line is not None
        assert 'Si' in element_line
        assert 'Kα' in element_line
        
        # Test Fe Kα
        element_line = xrf_analyzer._identify_element_line(6.404)
        assert element_line is not None
        assert 'Fe' in element_line
    
    def test_escape_peak_identification(self, xrf_analyzer):
        """Test escape peak detection"""
        energy = np.linspace(0, 10, 1000)
        counts = np.zeros_like(energy)
        
        # Add primary peak at 5 keV
        primary_idx = np.argmin(np.abs(energy - 5.0))
        counts[primary_idx] = 5000
        
        # Add escape peak at 5 - 1.84 = 3.16 keV
        escape_idx = np.argmin(np.abs(energy - 3.16))
        counts[escape_idx] = 500
        
        primary_peaks = [{'energy': 5.0, 'element_line': 'Test Kα'}]
        artifact_peaks = xrf_analyzer._identify_artifact_peaks(energy, counts, primary_peaks)
        
        # Should identify escape peak
        assert len(artifact_peaks) >= 1
        escape_found = any('escape' in p.get('type', '') for p in artifact_peaks)
        assert escape_found
    
    def test_fundamental_parameters_quantification(self, xrf_analyzer):
        """Test fundamental parameters quantification"""
        peaks = [
            XRFPeak(energy=1.74, intensity=5000, fwhm=0.15, element='Si', line='Kα', 
                   escapePeak=False, sumPeak=False),
            XRFPeak(energy=0.525, intensity=3000, fwhm=0.15, element='O', line='Kα',
                   escapePeak=False, sumPeak=False),
            XRFPeak(energy=6.404, intensity=2000, fwhm=0.15, element='Fe', line='Kα',
                   escapePeak=False, sumPeak=False)
        ]
        
        composition = xrf_analyzer.quantification_fundamental_parameters(peaks, matrix='SiO2')
        
        assert 'Si' in composition
        assert 'O' in composition
        assert 'Fe' in composition
        assert abs(sum(composition.values()) - 100) < 0.1
    
    def test_matrix_corrections(self, xrf_analyzer):
        """Test matrix absorption corrections"""
        peaks = [
            XRFPeak(energy=1.74, intensity=5000, fwhm=0.15, element='Si', line='Kα',
                   escapePeak=False, sumPeak=False),
        ]
        
        corrections = xrf_analyzer._calculate_matrix_corrections(peaks, 'SiO2')
        
        assert 'Si' in corrections
        assert corrections['Si'] > 0
        assert corrections['Si'] <= 1.0
    
    def test_detector_efficiency(self, xrf_analyzer):
        """Test detector efficiency calculation"""
        # Low energy (high efficiency)
        eff_low = xrf_analyzer._detector_efficiency(1.0)
        assert eff_low > 0.9
        
        # Medium energy
        eff_med = xrf_analyzer._detector_efficiency(5.0)
        assert 0.5 < eff_med < 0.95
        
        # High energy (lower efficiency)
        eff_high = xrf_analyzer._detector_efficiency(20.0)
        assert eff_high < eff_med
    
    def test_standardless_quantification(self, xrf_analyzer):
        """Test standardless quantification workflow"""
        energy = np.linspace(0, 10, 1000)
        counts = np.random.poisson(100, len(energy))
        
        # Add peaks
        counts += 5000 * np.exp(-((energy - 1.74)**2) / 0.001)  # Si
        counts += 2000 * np.exp(-((energy - 6.404)**2) / 0.002)  # Fe
        
        composition = xrf_analyzer.standardless_quantification(energy, counts)
        
        assert len(composition) > 0
        assert 'Si' in composition or 'Fe' in composition
    
    def test_detection_limits(self, xrf_analyzer, sample_xrf_spectrum):
        """Test detection limit calculation"""
        energy, counts = sample_xrf_spectrum
        mdl = xrf_analyzer.detection_limits(energy, counts, measurement_time=300)
        
        assert len(mdl) > 0
        for element, limit in mdl.items():
            assert limit > 0
            assert limit < 10000  # Should be reasonable ppm values

class TestChemicalSimulator:
    """Test spectrum simulation functionality"""
    
    def test_xps_spectrum_generation(self, simulator):
        """Test XPS spectrum generation"""
        composition = {'C': 50, 'O': 30, 'N': 20}
        be, intensity = simulator.generate_xps_spectrum(composition)
        
        assert len(be) == len(intensity)
        assert np.all(intensity >= 0)
        assert np.max(intensity) > np.mean(intensity) * 2  # Should have peaks
    
    def test_xps_peak_positions(self, simulator):
        """Test that XPS peaks appear at correct positions"""
        composition = {'C': 100}
        be, intensity = simulator.generate_xps_spectrum(composition, resolution=0.5, noise_level=0.001)
        
        # Find C 1s peak around 284.5 eV
        c1s_region = (be > 280) & (be < 290)
        peak_idx = np.argmax(intensity[c1s_region])
        peak_position = be[c1s_region][peak_idx]
        
        assert abs(peak_position - 284.5) < 2.0
    
    def test_xrf_spectrum_generation(self, simulator):
        """Test XRF spectrum generation"""
        composition = {'Si': 40, 'Fe': 30, 'O': 30}
        energy, counts = simulator.generate_xrf_spectrum(composition)
        
        assert len(energy) == len(counts)
        assert np.all(counts >= 0)
        assert np.max(counts) > np.mean(counts) * 5  # Should have peaks
    
    def test_xrf_peak_positions(self, simulator):
        """Test that XRF peaks appear at correct energies"""
        composition = {'Si': 100}
        energy, counts = simulator.generate_xrf_spectrum(composition, noise_level=0.001)
        
        # Find Si Kα peak around 1.74 keV
        si_region = (energy > 1.7) & (energy < 1.8)
        if np.any(si_region):
            peak_idx = np.argmax(counts[si_region])
            peak_energy = energy[si_region][peak_idx]
            assert abs(peak_energy - 1.74) < 0.1
    
    def test_shirley_background_generation(self, simulator):
        """Test Shirley background generation"""
        be = np.linspace(0, 100, 100)
        intensity = np.ones_like(be) * 1000
        
        background = simulator._generate_shirley_background(be, intensity)
        
        assert len(background) == len(intensity)
        assert np.all(background >= 0)
    
    def test_bremsstrahlung_generation(self, simulator):
        """Test Bremsstrahlung background for XRF"""
        energy = np.linspace(0.1, 50, 500)
        background = simulator._generate_bremsstrahlung(energy, 50.0)
        
        assert len(background) == len(energy)
        assert np.all(background >= 0)
        # Background should decrease with energy
        assert background[0] > background[-1]

class TestIntegrationWorkflows:
    """Test complete analysis workflows"""
    
    @pytest.mark.integration
    def test_complete_xps_workflow(self, xps_analyzer, simulator):
        """Test complete XPS analysis workflow"""
        # Generate spectrum
        true_composition = {'C': 40, 'O': 35, 'N': 15, 'Si': 10}
        be, intensity = simulator.generate_xps_spectrum(true_composition)
        
        # Process spectrum
        be_proc, int_proc = xps_analyzer.process_spectrum(be, intensity)
        
        # Find peaks
        peaks = xps_analyzer.find_peaks(be_proc, int_proc)
        assert len(peaks) >= len(true_composition)
        
        # Fit peaks and quantify
        xps_peaks = []
        for peak in peaks[:4]:  # Take first 4 peaks
            region = (be_proc > peak['position'] - 10) & (be_proc < peak['position'] + 10)
            if np.sum(region) > 10:
                fit_result = xps_analyzer.fit_peak(be_proc[region], int_proc[region])
                if fit_result['success']:
                    element = peak.get('element', 'Unknown')
                    if element and element != 'Unknown':
                        elem_symbol = element.split()[0] if ' ' in element else element
                        xps_peaks.append(XPSPeak(
                            position=fit_result['position'],
                            area=fit_result['area'],
                            fwhm=fit_result['fwhm'],
                            shape=PeakShape.VOIGT,
                            orbital='1s',
                            element=elem_symbol
                        ))
        
        if xps_peaks:
            composition = xps_analyzer.quantification(xps_peaks)
            assert len(composition) > 0
    
    @pytest.mark.integration
    def test_complete_xrf_workflow(self, xrf_analyzer, simulator):
        """Test complete XRF analysis workflow"""
        # Generate spectrum
        true_composition = {'Si': 45, 'Fe': 25, 'O': 30}
        energy, counts = simulator.generate_xrf_spectrum(true_composition)
        
        # Process spectrum
        energy_proc, counts_proc = xrf_analyzer.process_spectrum(energy, counts)
        
        # Find peaks
        peaks = xrf_analyzer.find_peaks(energy_proc, counts_proc)
        assert len(peaks) >= 2
        
        # Quantify
        composition = xrf_analyzer.standardless_quantification(energy_proc, counts_proc)
        assert len(composition) > 0
        
        # Calculate detection limits
        mdl = xrf_analyzer.detection_limits(energy_proc, counts_proc)
        assert len(mdl) > 0
    
    @pytest.mark.integration
    def test_depth_profile_workflow(self, xps_analyzer, simulator):
        """Test depth profiling workflow"""
        # Simulate depth-dependent composition
        depths = [0, 20, 40, 60, 80, 100]
        spectra = []
        etch_times = np.array([0, 60, 120, 180, 240, 300])
        
        for d in depths:
            # Composition changes with depth
            comp = {
                'C': 50 * np.exp(-d/30),
                'O': 40 * (1 - np.exp(-d/50)),
                'Si': 10 + 5 * d/100
            }
            be, intensity = simulator.generate_xps_spectrum(comp)
            spectra.append((be, intensity))
        
        # Analyze depth profile
        profile = xps_analyzer.depth_profile(spectra, etch_times, ['C', 'O', 'Si'])
        
        assert 'depth' in profile
        assert len(profile['depth']) == len(depths)
        # Verify trends
        assert profile['C'][0] > profile['C'][-1]  # C decreases
        assert profile['O'][0] < profile['O'][-1]  # O increases
    
    @pytest.mark.integration
    def test_chemical_state_workflow(self, xps_analyzer):
        """Test chemical state analysis workflow"""
        # Create multi-component carbon peak
        be = np.linspace(282, 292, 200)
        intensity = np.zeros_like(be)
        
        # Add different carbon states
        intensity += 5000 * np.exp(-((be - 284.5)**2) / 0.5)  # C-C
        intensity += 3000 * np.exp(-((be - 286.0)**2) / 0.5)  # C-O
        intensity += 1500 * np.exp(-((be - 288.0)**2) / 0.5)  # C=O
        intensity += 500 * np.exp(-((be - 289.0)**2) / 0.5)   # O-C=O
        intensity += np.random.normal(100, 10, len(be))
        
        # Process and analyze
        be_proc, int_proc = xps_analyzer.process_spectrum(be, intensity)
        
        # Calculate background
        background = xps_analyzer.shirley_background(be_proc, int_proc)
        
        # Analyze chemical states
        result = xps_analyzer.chemical_state_analysis(be_proc, int_proc - background, 'C', '1s')
        
        assert 'identified_states' in result
        assert len(result['identified_states']) >= 2

class TestPerformance:
    """Performance and stress tests"""
    
    @pytest.mark.performance
    def test_large_spectrum_processing(self, xps_analyzer):
        """Test processing of large spectra"""
        # Generate large spectrum
        be = np.linspace(0, 1200, 12000)  # 0.1 eV steps
        intensity = np.random.normal(1000, 100, len(be))
        
        import time
        start = time.time()
        be_proc, int_proc = xps_analyzer.process_spectrum(be, intensity)
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should process in under 1 second
        assert len(be_proc) == len(be)
    
    @pytest.mark.performance
    def test_multiple_peak_fitting(self, xps_analyzer):
        """Test fitting multiple peaks efficiently"""
        import time
        
        fit_times = []
        for i in range(10):
            be = np.linspace(280 + i*10, 290 + i*10, 100)
            intensity = 1000 + 5000 * np.exp(-((be - (285 + i*10))**2) / 2)
            
            start = time.time()
            result = xps_analyzer.fit_peak(be, intensity)
            fit_times.append(time.time() - start)
        
        assert all(t < 0.5 for t in fit_times)  # Each fit under 0.5s
        assert np.mean(fit_times) < 0.2  # Average under 0.2s
    
    @pytest.mark.performance
    def test_batch_quantification(self, xrf_analyzer):
        """Test batch processing of multiple spectra"""
        import time
        
        # Generate batch of spectra
        spectra = []
        for i in range(20):
            energy = np.linspace(0, 20, 2000)
            counts = np.random.poisson(100 + i*10, len(energy))
            spectra.append((energy, counts))
        
        start = time.time()
        results = []
        for energy, counts in spectra:
            composition = xrf_analyzer.standardless_quantification(energy, counts)
            results.append(composition)
        elapsed = time.time() - start
        
        assert elapsed < 10.0  # Process 20 spectra in under 10s
        assert len(results) == 20

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_spectrum(self, xps_analyzer):
        """Test handling of empty spectrum"""
        be = np.array([])
        intensity = np.array([])
        
        be_proc, int_proc = xps_analyzer.process_spectrum(be, intensity)
        assert len(be_proc) == 0
        assert len(int_proc) == 0
    
    def test_noisy_spectrum(self, xps_analyzer):
        """Test handling of very noisy spectrum"""
        be = np.linspace(280, 290, 100)
        intensity = np.random.normal(1000, 500, len(be))  # Very noisy
        
        peaks = xps_analyzer.find_peaks(be, intensity, prominence=0.5)
        # Should handle without crashing, might find few or no peaks
        assert isinstance(peaks, list)
    
    def test_invalid_peak_shape(self, xps_analyzer):
        """Test handling of invalid peak shape"""
        be = np.linspace(280, 290, 100)
        intensity = 1000 + 5000 * np.exp(-((be - 285)**2) / 2)
        
        # This should not crash but return an error
        result = xps_analyzer.fit_peak(be, intensity, shape="InvalidShape")
        # Should handle gracefully
        assert 'success' in result
    
    def test_negative_energies(self, xrf_analyzer):
        """Test handling of negative energies"""
        energy = np.linspace(-10, 10, 200)
        counts = np.random.poisson(100, len(energy))
        
        # Should handle without crashing
        energy_proc, counts_proc = xrf_analyzer.process_spectrum(energy, counts)
        assert len(energy_proc) == len(energy)
    
    def test_missing_element(self, element_db):
        """Test handling of missing element"""
        element = element_db.get_element('Unobtainium')
        assert element is None
        
        rsf = element_db.get_sensitivity_factor('Unobtainium', '1s')
        assert rsf == 1.0  # Should return default

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
