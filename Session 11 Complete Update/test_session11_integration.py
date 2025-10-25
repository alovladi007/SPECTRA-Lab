"""
Session 11: Surface Analysis (XPS/XRF) - Integration Tests
===========================================================

Comprehensive test suite for XPS and XRF analysis.

Author: Semiconductor Lab Platform Team
Version: 1.0.0
Date: October 2024
"""

import pytest
import numpy as np
from session11_surface_analysis_complete_implementation import (
    XPSAnalyzer, XPSSpectrum, XPSPeak, XPSPeakShape,
    XRFAnalyzer, XRFSpectrum, XRFPeak,
    SurfaceAnalysisSimulator
)


class TestXPSAnalyzer:
    """Test suite for XPS analysis"""
    
    @pytest.fixture
    def xps_analyzer(self):
        return XPSAnalyzer()
    
    @pytest.fixture
    def simulator(self):
        return SurfaceAnalysisSimulator()
    
    @pytest.fixture
    def sample_spectrum(self, simulator):
        return simulator.simulate_xps_spectrum(element="Si", peak_position=99.0)
    
    def test_analyzer_initialization(self, xps_analyzer):
        """Test XPS analyzer initializes with sensitivity factors"""
        assert len(xps_analyzer.sensitivity_factors) > 0
        assert "Si_2p" in xps_analyzer.sensitivity_factors
    
    def test_shirley_background(self, xps_analyzer, sample_spectrum):
        """Test Shirley background subtraction"""
        background = xps_analyzer.shirley_background(sample_spectrum)
        
        assert len(background) == len(sample_spectrum.intensity)
        assert np.all(background >= 0)
        assert background.max() < sample_spectrum.intensity.max()
    
    def test_peak_fitting(self, xps_analyzer, sample_spectrum):
        """Test Voigt peak fitting"""
        peaks = xps_analyzer.fit_multiple_peaks(
            sample_spectrum,
            peak_positions=[99.0],
            shape=XPSPeakShape.VOIGT
        )
        
        assert len(peaks) >= 1
        peak = peaks[0]
        assert 95 < peak.position < 103
        assert peak.area > 0
        assert peak.fwhm > 0
    
    def test_quantification(self, xps_analyzer):
        """Test atomic % quantification"""
        # Create mock peaks
        peaks = [
            XPSPeak(position=99.0, area=1000, fwhm=1.5, height=500, element="Si", orbital="2p"),
            XPSPeak(position=532.0, area=2000, fwhm=2.0, height=800, element="O", orbital="1s")
        ]
        
        quant = xps_analyzer.quantify(peaks, use_rsf=True)
        
        assert "Si" in quant.composition
        assert "O" in quant.composition
        assert abs(sum(quant.composition.values()) - 100.0) < 0.1


class TestXRFAnalyzer:
    """Test suite for XRF analysis"""
    
    @pytest.fixture
    def xrf_analyzer(self):
        return XRFAnalyzer()
    
    @pytest.fixture
    def simulator(self):
        return SurfaceAnalysisSimulator()
    
    @pytest.fixture
    def sample_spectrum(self, simulator):
        return simulator.simulate_xrf_spectrum(elements=["Ti", "Cu"], concentrations=[60, 40])
    
    def test_analyzer_initialization(self, xrf_analyzer):
        """Test XRF analyzer initialization"""
        assert len(xrf_analyzer.xray_database) > 0
        assert "Ti" in xrf_analyzer.xray_database
    
    def test_peak_identification(self, xrf_analyzer, sample_spectrum):
        """Test element identification from peaks"""
        peaks = xrf_analyzer.identify_peaks(sample_spectrum, threshold=100.0)
        
        assert len(peaks) >= 2
        elements = {peak.element for peak in peaks}
        assert "Ti" in elements or "Cu" in elements
    
    def test_fundamental_parameters(self, xrf_analyzer, sample_spectrum):
        """Test FP quantification"""
        peaks = xrf_analyzer.identify_peaks(sample_spectrum)
        quant = xrf_analyzer.fundamental_parameters(peaks)
        
        assert len(quant.composition) > 0
        assert abs(sum(quant.composition.values()) - 100.0) < 0.1
        assert quant.method == "fundamental_parameters"


class TestSimulator:
    """Test suite for simulator"""
    
    @pytest.fixture
    def simulator(self):
        return SurfaceAnalysisSimulator()
    
    def test_xps_spectrum_generation(self, simulator):
        """Test XPS spectrum simulation"""
        spectrum = simulator.simulate_xps_spectrum(element="Si", peak_position=99.0)
        
        assert len(spectrum.binding_energy) > 0
        assert len(spectrum.intensity) == len(spectrum.binding_energy)
        assert spectrum.intensity.max() > 0
    
    def test_xrf_spectrum_generation(self, simulator):
        """Test XRF spectrum simulation"""
        spectrum = simulator.simulate_xrf_spectrum(elements=["Ti", "Cu"])
        
        assert len(spectrum.energy) > 0
        assert len(spectrum.intensity) == len(spectrum.energy)
        assert spectrum.intensity.max() > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
