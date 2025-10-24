"""
Session 11: XPS/XRF Analysis - Complete Implementation
X-ray Photoelectron Spectroscopy & X-ray Fluorescence
Production-ready implementation with comprehensive analysis capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from scipy import optimize, signal, interpolate, constants
from scipy.special import voigt_profile
import json
import hashlib
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
h = constants.h  # Planck's constant
c = constants.c  # Speed of light
e = constants.e  # Elementary charge
m_e = constants.m_e  # Electron mass
N_A = constants.Avogadro  # Avogadro's number

class XRaySource(Enum):
    """X-ray source types"""
    AL_KA = "Al Kα"  # 1486.6 eV
    MG_KA = "Mg Kα"  # 1253.6 eV
    AG_LA = "Ag Lα"  # 2984.3 eV
    MONOCHROMATIC_AL = "Monochromatic Al Kα"  # 1486.6 eV, ΔE ~0.3 eV
    SYNCHROTRON = "Synchrotron"  # Variable energy

class PeakShape(Enum):
    """Peak profile shapes"""
    GAUSSIAN = "Gaussian"
    LORENTZIAN = "Lorentzian"
    VOIGT = "Voigt"
    DONIACH_SUNJIC = "Doniach-Sunjic"
    PSEUDO_VOIGT = "Pseudo-Voigt"

@dataclass
class Element:
    """Element properties for XPS/XRF"""
    symbol: str
    name: str
    atomic_number: int
    atomic_mass: float
    xps_peaks: Dict[str, float] = field(default_factory=dict)  # orbital: BE (eV)
    xrf_lines: Dict[str, float] = field(default_factory=dict)  # line: energy (keV)
    sensitivity_factors: Dict[str, float] = field(default_factory=dict)  # orbital: RSF
    photoionization_cross_sections: Dict[str, float] = field(default_factory=dict)
    fluorescence_yield: float = 0.0

@dataclass
class XPSPeak:
    """XPS peak parameters"""
    position: float  # Binding energy (eV)
    area: float
    fwhm: float
    shape: PeakShape
    orbital: str
    element: str
    chemical_state: Optional[str] = None
    asymmetry: float = 0.0  # For Doniach-Sunjic
    gaussian_fraction: float = 0.5  # For Pseudo-Voigt

@dataclass
class XRFPeak:
    """XRF peak parameters"""
    energy: float  # keV
    intensity: float
    fwhm: float
    element: str
    line: str  # Kα, Kβ, Lα, etc.
    escape_peak: bool = False
    sum_peak: bool = False

class ElementDatabase:
    """Database of element properties for XPS/XRF"""
    
    def __init__(self):
        self.elements = self._initialize_database()
    
    def _initialize_database(self) -> Dict[str, Element]:
        """Initialize element database with key elements"""
        elements = {}
        
        # Carbon
        elements['C'] = Element(
            symbol='C', name='Carbon', atomic_number=6, atomic_mass=12.011,
            xps_peaks={'1s': 284.5, 'C-C': 284.5, 'C-O': 286.0, 'C=O': 288.0, 'O-C=O': 289.0},
            xrf_lines={'Kα': 0.277},
            sensitivity_factors={'1s': 0.278},
            photoionization_cross_sections={'1s': 1.0},
            fluorescence_yield=0.0026
        )
        
        # Oxygen
        elements['O'] = Element(
            symbol='O', name='Oxygen', atomic_number=8, atomic_mass=15.999,
            xps_peaks={'1s': 532.5, 'O-C': 532.5, 'O-Si': 533.0, 'O-Metal': 530.0},
            xrf_lines={'Kα': 0.525},
            sensitivity_factors={'1s': 0.780},
            photoionization_cross_sections={'1s': 2.93},
            fluorescence_yield=0.0083
        )
        
        # Silicon
        elements['Si'] = Element(
            symbol='Si', name='Silicon', atomic_number=14, atomic_mass=28.085,
            xps_peaks={'2p3/2': 99.3, '2p1/2': 99.9, 'Si-O': 103.4, '2s': 150.0},
            xrf_lines={'Kα': 1.740, 'Kβ': 1.836},
            sensitivity_factors={'2p': 0.339},
            photoionization_cross_sections={'2p': 0.27},
            fluorescence_yield=0.041
        )
        
        # Gallium
        elements['Ga'] = Element(
            symbol='Ga', name='Gallium', atomic_number=31, atomic_mass=69.723,
            xps_peaks={'3d5/2': 18.6, '3d3/2': 19.0, '3p3/2': 104.0, '3p1/2': 107.0, 
                      '3s': 159.0, '2p3/2': 1116.0, '2p1/2': 1143.0},
            xrf_lines={'Kα1': 9.252, 'Kα2': 9.225, 'Kβ1': 10.264, 'Lα': 1.098},
            sensitivity_factors={'3d': 2.65},
            photoionization_cross_sections={'3d': 6.0},
            fluorescence_yield=0.53
        )
        
        # Arsenic
        elements['As'] = Element(
            symbol='As', name='Arsenic', atomic_number=33, atomic_mass=74.922,
            xps_peaks={'3d5/2': 41.0, '3d3/2': 41.7, '3p3/2': 141.0, '3p1/2': 146.0,
                      '3s': 204.0, '2p3/2': 1323.0, '2p1/2': 1358.0},
            xrf_lines={'Kα1': 10.544, 'Kα2': 10.508, 'Kβ1': 11.726, 'Lα': 1.282},
            sensitivity_factors={'3d': 3.0},
            photoionization_cross_sections={'3d': 6.8},
            fluorescence_yield=0.56
        )
        
        # Nitrogen
        elements['N'] = Element(
            symbol='N', name='Nitrogen', atomic_number=7, atomic_mass=14.007,
            xps_peaks={'1s': 399.5, 'N-C': 399.5, 'N-H': 401.0, 'N-O': 403.0},
            xrf_lines={'Kα': 0.392},
            sensitivity_factors={'1s': 0.477},
            photoionization_cross_sections={'1s': 1.80},
            fluorescence_yield=0.0045
        )
        
        # Aluminum
        elements['Al'] = Element(
            symbol='Al', name='Aluminum', atomic_number=13, atomic_mass=26.982,
            xps_peaks={'2p3/2': 72.9, '2p1/2': 73.4, 'Al2O3': 74.5, '2s': 117.8},
            xrf_lines={'Kα': 1.487, 'Kβ': 1.553},
            sensitivity_factors={'2p': 0.234},
            photoionization_cross_sections={'2p': 0.19},
            fluorescence_yield=0.032
        )
        
        # Gold (for calibration)
        elements['Au'] = Element(
            symbol='Au', name='Gold', atomic_number=79, atomic_mass=196.967,
            xps_peaks={'4f7/2': 84.0, '4f5/2': 87.7, '4d5/2': 335.1, '4d3/2': 353.2},
            xrf_lines={'Lα1': 9.713, 'Lα2': 9.628, 'Lβ1': 11.442, 'Mα': 2.123},
            sensitivity_factors={'4f': 8.5},
            photoionization_cross_sections={'4f': 17.1},
            fluorescence_yield=0.95
        )
        
        # Titanium
        elements['Ti'] = Element(
            symbol='Ti', name='Titanium', atomic_number=22, atomic_mass=47.867,
            xps_peaks={'2p3/2': 454.0, '2p1/2': 459.7, 'TiO2': 458.5, '2s': 564.0},
            xrf_lines={'Kα1': 4.511, 'Kα2': 4.505, 'Kβ1': 4.932, 'Lα': 0.452},
            sensitivity_factors={'2p': 1.8},
            photoionization_cross_sections={'2p': 2.3},
            fluorescence_yield=0.223
        )
        
        # Iron
        elements['Fe'] = Element(
            symbol='Fe', name='Iron', atomic_number=26, atomic_mass=55.845,
            xps_peaks={'2p3/2': 707.0, '2p1/2': 720.0, 'Fe2+': 709.5, 'Fe3+': 711.0},
            xrf_lines={'Kα1': 6.404, 'Kα2': 6.391, 'Kβ1': 7.058, 'Lα': 0.705},
            sensitivity_factors={'2p': 2.7},
            photoionization_cross_sections={'2p': 3.5},
            fluorescence_yield=0.347
        )
        
        return elements
    
    def get_element(self, symbol: str) -> Optional[Element]:
        """Get element by symbol"""
        return self.elements.get(symbol)
    
    def get_sensitivity_factor(self, element: str, orbital: str, 
                              source: XRaySource = XRaySource.AL_KA) -> float:
        """Get relative sensitivity factor for quantification"""
        elem = self.get_element(element)
        if elem and orbital in elem.sensitivity_factors:
            # Apply energy correction for different sources
            if source == XRaySource.MG_KA:
                return elem.sensitivity_factors[orbital] * 0.9
            return elem.sensitivity_factors[orbital]
        return 1.0

class XPSAnalyzer:
    """X-ray Photoelectron Spectroscopy analyzer"""
    
    def __init__(self, source: XRaySource = XRaySource.AL_KA):
        self.source = source
        self.source_energy = self._get_source_energy()
        self.element_db = ElementDatabase()
        self.calibration = {'C1s': 284.5}  # Standard C 1s calibration
        self.results = {}
        
    def _get_source_energy(self) -> float:
        """Get X-ray source energy in eV"""
        energies = {
            XRaySource.AL_KA: 1486.6,
            XRaySource.MG_KA: 1253.6,
            XRaySource.AG_LA: 2984.3,
            XRaySource.MONOCHROMATIC_AL: 1486.6,
            XRaySource.SYNCHROTRON: 1486.6  # Default
        }
        return energies.get(self.source, 1486.6)
    
    def process_spectrum(self, binding_energy: np.ndarray, intensity: np.ndarray,
                        smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw XPS spectrum"""
        # Smooth spectrum
        if smooth_window > 1:
            intensity = signal.savgol_filter(intensity, smooth_window, 2)
        
        # Remove negative values
        intensity = np.maximum(intensity, 0)
        
        # Energy calibration (if needed)
        if self.calibration:
            binding_energy = self._apply_calibration(binding_energy, intensity)
        
        return binding_energy, intensity
    
    def _apply_calibration(self, binding_energy: np.ndarray, 
                          intensity: np.ndarray) -> np.ndarray:
        """Apply energy calibration using reference peak"""
        # Find C 1s peak around 284.5 eV
        mask = (binding_energy > 280) & (binding_energy < 290)
        if np.any(mask):
            c1s_region = intensity[mask]
            c1s_be = binding_energy[mask]
            peak_idx = np.argmax(c1s_region)
            measured_c1s = c1s_be[peak_idx]
            
            # Calculate shift
            shift = self.calibration['C1s'] - measured_c1s
            logger.info(f"Applying calibration shift: {shift:.2f} eV")
            return binding_energy + shift
        
        return binding_energy
    
    def shirley_background(self, binding_energy: np.ndarray, 
                          intensity: np.ndarray,
                          endpoints: Optional[Tuple[float, float]] = None,
                          max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
        """Calculate Shirley background"""
        if endpoints is None:
            endpoints = (binding_energy[0], binding_energy[-1])
        
        # Find indices for endpoints
        idx1 = np.argmin(np.abs(binding_energy - endpoints[0]))
        idx2 = np.argmin(np.abs(binding_energy - endpoints[1]))
        
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        # Extract region
        be_region = binding_energy[idx1:idx2+1]
        int_region = intensity[idx1:idx2+1]
        
        # Initial linear background
        background = np.linspace(int_region[0], int_region[-1], len(int_region))
        
        # Iterative Shirley calculation
        for iteration in range(max_iter):
            old_background = background.copy()
            
            # Calculate cumulative intensity
            cumsum = np.cumsum(int_region - background)
            total = cumsum[-1]
            
            if total != 0:
                k = (int_region[-1] - int_region[0]) / total
                background = int_region[0] + k * cumsum
            
            # Check convergence
            if np.max(np.abs(background - old_background)) < tol:
                break
        
        # Return full background array
        full_background = np.zeros_like(intensity)
        full_background[idx1:idx2+1] = background
        
        # Extend background outside region
        if idx1 > 0:
            full_background[:idx1] = background[0]
        if idx2 < len(intensity) - 1:
            full_background[idx2+1:] = background[-1]
        
        return full_background
    
    def tougaard_background(self, binding_energy: np.ndarray,
                           intensity: np.ndarray,
                           B: float = 2866, C: float = 1643,
                           D: float = 1, endpoints: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """Calculate Tougaard universal background"""
        if endpoints is None:
            endpoints = (binding_energy[0], binding_energy[-1])
        
        # Find indices for endpoints
        idx1 = np.argmin(np.abs(binding_energy - endpoints[0]))
        idx2 = np.argmin(np.abs(binding_energy - endpoints[1]))
        
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        
        background = np.zeros_like(intensity)
        
        # Calculate Tougaard background
        for i in range(idx1, idx2+1):
            integral = 0
            for j in range(i+1, idx2+1):
                E_diff = binding_energy[j] - binding_energy[i]
                if E_diff > 0:
                    loss_function = B * E_diff / ((C + E_diff**2)**2 + D * E_diff**2)
                    integral += intensity[j] * loss_function * np.abs(binding_energy[j] - binding_energy[j-1])
            
            background[i] = integral
        
        return background
    
    def find_peaks(self, binding_energy: np.ndarray, intensity: np.ndarray,
                  prominence: float = 0.1, distance: int = 10,
                  width: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """Find peaks in XPS spectrum"""
        # Normalize intensity for peak finding
        norm_intensity = intensity / np.max(intensity)
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            norm_intensity,
            prominence=prominence,
            distance=distance,
            width=width
        )
        
        # Extract peak information
        peak_list = []
        for idx in peaks:
            peak_info = {
                'position': binding_energy[idx],
                'intensity': intensity[idx],
                'prominence': properties['prominences'][np.where(peaks == idx)[0][0]],
                'width': properties.get('widths', [None])[np.where(peaks == idx)[0][0]],
                'element': self._identify_element(binding_energy[idx])
            }
            peak_list.append(peak_info)
        
        return peak_list
    
    def _identify_element(self, binding_energy: float, tolerance: float = 2.0) -> Optional[str]:
        """Identify element and orbital from binding energy"""
        for symbol, element in self.element_db.elements.items():
            for orbital, be in element.xps_peaks.items():
                if abs(binding_energy - be) < tolerance:
                    return f"{symbol} {orbital}"
        return None
    
    def fit_peak(self, binding_energy: np.ndarray, intensity: np.ndarray,
                 initial_params: Optional[Dict[str, float]] = None,
                 shape: PeakShape = PeakShape.VOIGT,
                 background_type: str = 'shirley') -> Dict[str, Any]:
        """Fit single XPS peak with background"""
        # Calculate background
        if background_type == 'shirley':
            background = self.shirley_background(binding_energy, intensity)
        elif background_type == 'tougaard':
            background = self.tougaard_background(binding_energy, intensity)
        else:
            background = np.zeros_like(intensity)
        
        # Subtract background
        peak_intensity = intensity - background
        
        # Initial parameters
        if initial_params is None:
            peak_idx = np.argmax(peak_intensity)
            initial_params = {
                'position': binding_energy[peak_idx],
                'amplitude': peak_intensity[peak_idx],
                'fwhm': 1.5,
                'asymmetry': 0.0
            }
        
        # Define fitting function
        def fit_function(x, *params):
            if shape == PeakShape.GAUSSIAN:
                return self._gaussian(x, *params[:3])
            elif shape == PeakShape.LORENTZIAN:
                return self._lorentzian(x, *params[:3])
            elif shape == PeakShape.VOIGT:
                return self._voigt(x, *params[:4])
            elif shape == PeakShape.DONIACH_SUNJIC:
                return self._doniach_sunjic(x, *params[:4])
            else:  # PSEUDO_VOIGT
                return self._pseudo_voigt(x, *params[:4])
        
        # Prepare initial parameters for fitting
        p0 = [initial_params['position'], initial_params['amplitude'], initial_params['fwhm']]
        if shape in [PeakShape.VOIGT, PeakShape.DONIACH_SUNJIC, PeakShape.PSEUDO_VOIGT]:
            p0.append(initial_params.get('asymmetry', 0.0))
        
        # Fit peak
        try:
            popt, pcov = optimize.curve_fit(fit_function, binding_energy, peak_intensity, p0=p0)
            
            # Calculate fit quality
            fitted_curve = fit_function(binding_energy, *popt)
            residuals = peak_intensity - fitted_curve
            r_squared = 1 - np.sum(residuals**2) / np.sum((peak_intensity - np.mean(peak_intensity))**2)
            
            # Calculate peak area
            area = np.trapz(fitted_curve, binding_energy)
            
            return {
                'success': True,
                'position': popt[0],
                'amplitude': popt[1],
                'fwhm': abs(popt[2]),
                'asymmetry': popt[3] if len(popt) > 3 else 0.0,
                'area': area,
                'r_squared': r_squared,
                'fitted_curve': fitted_curve,
                'background': background,
                'shape': shape.value
            }
            
        except Exception as e:
            logger.error(f"Peak fitting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _gaussian(self, x: np.ndarray, position: float, amplitude: float, fwhm: float) -> np.ndarray:
        """Gaussian peak shape"""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        return amplitude * np.exp(-(x - position)**2 / (2 * sigma**2))
    
    def _lorentzian(self, x: np.ndarray, position: float, amplitude: float, fwhm: float) -> np.ndarray:
        """Lorentzian peak shape"""
        gamma = fwhm / 2
        return amplitude * gamma**2 / ((x - position)**2 + gamma**2)
    
    def _voigt(self, x: np.ndarray, position: float, amplitude: float, 
               fwhm: float, shape_factor: float = 0.5) -> np.ndarray:
        """Voigt peak shape (convolution of Gaussian and Lorentzian)"""
        sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
        gamma = fwhm / 2
        
        # Use scipy's voigt_profile
        z = (x - position + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(voigt_profile(x - position, sigma, gamma))
    
    def _doniach_sunjic(self, x: np.ndarray, position: float, amplitude: float,
                        fwhm: float, asymmetry: float) -> np.ndarray:
        """Doniach-Sunjic peak shape for metals"""
        gamma = fwhm / 2
        
        # Avoid singularities
        epsilon = 1e-10
        x_shifted = x - position
        
        # Calculate Doniach-Sunjic profile
        if abs(asymmetry) < epsilon:
            # Reduce to Lorentzian for asymmetry = 0
            return self._lorentzian(x, position, amplitude, fwhm)
        
        term1 = np.cos(np.pi * asymmetry / 2 + (1 - asymmetry) * np.arctan(x_shifted / gamma))
        term2 = (gamma**2 + x_shifted**2)**((1 - asymmetry) / 2)
        
        return amplitude * term1 / term2
    
    def _pseudo_voigt(self, x: np.ndarray, position: float, amplitude: float,
                     fwhm: float, eta: float = 0.5) -> np.ndarray:
        """Pseudo-Voigt peak shape (linear combination of Gaussian and Lorentzian)"""
        gaussian = self._gaussian(x, position, 1.0, fwhm)
        lorentzian = self._lorentzian(x, position, 1.0, fwhm)
        
        # Normalize
        gaussian /= np.max(gaussian)
        lorentzian /= np.max(lorentzian)
        
        return amplitude * ((1 - eta) * gaussian + eta * lorentzian)
    
    def multiplet_splitting(self, element: str, orbital: str) -> Optional[Dict[str, float]]:
        """Calculate multiplet splitting for p, d, f orbitals"""
        splitting_data = {
            'Si': {'2p': {'2p3/2': 99.3, '2p1/2': 99.9, 'ratio': 2.0, 'separation': 0.6}},
            'Ga': {'3d': {'3d5/2': 18.6, '3d3/2': 19.0, 'ratio': 1.5, 'separation': 0.4}},
            'As': {'3d': {'3d5/2': 41.0, '3d3/2': 41.7, 'ratio': 1.5, 'separation': 0.7}},
            'Ti': {'2p': {'2p3/2': 454.0, '2p1/2': 459.7, 'ratio': 2.0, 'separation': 5.7}},
            'Fe': {'2p': {'2p3/2': 707.0, '2p1/2': 720.0, 'ratio': 2.0, 'separation': 13.0}},
            'Au': {'4f': {'4f7/2': 84.0, '4f5/2': 87.7, 'ratio': 1.33, 'separation': 3.7}}
        }
        
        if element in splitting_data and orbital in splitting_data[element]:
            return splitting_data[element][orbital]
        return None
    
    def quantification(self, peaks: List[XPSPeak]) -> Dict[str, float]:
        """Quantify atomic composition from XPS peaks"""
        compositions = {}
        total_concentration = 0
        
        for peak in peaks:
            # Get sensitivity factor
            rsf = self.element_db.get_sensitivity_factor(
                peak.element, peak.orbital, self.source
            )
            
            # Calculate concentration (simplified)
            # C_i = (I_i / RSF_i) / Σ(I_j / RSF_j)
            normalized_intensity = peak.area / rsf
            compositions[peak.element] = normalized_intensity
            total_concentration += normalized_intensity
        
        # Normalize to atomic %
        if total_concentration > 0:
            for element in compositions:
                compositions[element] = (compositions[element] / total_concentration) * 100
        
        return compositions
    
    def depth_profile(self, spectra: List[Tuple[np.ndarray, np.ndarray]],
                     etch_times: np.ndarray, elements: List[str]) -> Dict[str, np.ndarray]:
        """Analyze depth profile from sequential XPS spectra"""
        depth_profiles = {elem: [] for elem in elements}
        
        for i, (be, intensity) in enumerate(spectra):
            # Find and fit peaks for each element
            peaks = self.find_peaks(be, intensity)
            
            # Quantify composition at this depth
            xps_peaks = []
            for peak in peaks:
                if peak['element']:
                    elem_symbol = peak['element'].split()[0]
                    if elem_symbol in elements:
                        # Simple peak fitting for quantification
                        peak_region = (be > peak['position'] - 5) & (be < peak['position'] + 5)
                        area = np.trapz(intensity[peak_region], be[peak_region])
                        
                        xps_peaks.append(XPSPeak(
                            position=peak['position'],
                            area=area,
                            fwhm=1.5,
                            shape=PeakShape.VOIGT,
                            orbital='1s',  # Simplified
                            element=elem_symbol
                        ))
            
            # Quantify
            composition = self.quantification(xps_peaks)
            
            # Store depth profile data
            for elem in elements:
                depth_profiles[elem].append(composition.get(elem, 0))
        
        # Convert to numpy arrays
        for elem in elements:
            depth_profiles[elem] = np.array(depth_profiles[elem])
        
        # Calculate approximate depth from etch rate (assume 0.1 nm/s)
        etch_rate = 0.1  # nm/s
        depths = etch_times * etch_rate
        depth_profiles['depth'] = depths
        
        return depth_profiles
    
    def chemical_state_analysis(self, binding_energy: np.ndarray, 
                               intensity: np.ndarray,
                               element: str, orbital: str) -> Dict[str, Any]:
        """Analyze chemical states from peak positions"""
        # Get reference chemical states
        elem = self.element_db.get_element(element)
        if not elem:
            return {'error': 'Element not found'}
        
        chemical_states = {}
        for state_name, be in elem.xps_peaks.items():
            if orbital in state_name or state_name == orbital:
                chemical_states[state_name] = be
        
        # Find peaks in the region
        peaks = self.find_peaks(binding_energy, intensity)
        
        # Match peaks to chemical states
        identified_states = []
        for peak in peaks:
            best_match = None
            min_diff = float('inf')
            
            for state_name, ref_be in chemical_states.items():
                diff = abs(peak['position'] - ref_be)
                if diff < min_diff and diff < 2.0:  # 2 eV tolerance
                    min_diff = diff
                    best_match = state_name
            
            if best_match:
                identified_states.append({
                    'state': best_match,
                    'position': peak['position'],
                    'intensity': peak['intensity'],
                    'reference': chemical_states[best_match]
                })
        
        return {
            'element': element,
            'orbital': orbital,
            'identified_states': identified_states,
            'reference_states': chemical_states
        }

class XRFAnalyzer:
    """X-ray Fluorescence analyzer"""
    
    def __init__(self, excitation_energy: float = 50.0):  # keV
        self.excitation_energy = excitation_energy
        self.element_db = ElementDatabase()
        self.detector_resolution = 150  # eV at 5.9 keV (Si detector)
        self.results = {}
    
    def process_spectrum(self, energy: np.ndarray, counts: np.ndarray,
                        smooth_window: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Process raw XRF spectrum"""
        # Smooth spectrum
        if smooth_window > 1:
            counts = signal.savgol_filter(counts, smooth_window, 2)
        
        # Remove negative values
        counts = np.maximum(counts, 0)
        
        # Dead time correction (simplified)
        counts = self._dead_time_correction(counts)
        
        return energy, counts
    
    def _dead_time_correction(self, counts: np.ndarray, 
                             dead_time: float = 10e-6) -> np.ndarray:
        """Apply dead time correction"""
        # Simplified non-paralyzable detector model
        count_rate = np.sum(counts)
        true_rate = count_rate / (1 - count_rate * dead_time)
        correction_factor = true_rate / count_rate if count_rate > 0 else 1.0
        
        return counts * correction_factor
    
    def find_peaks(self, energy: np.ndarray, counts: np.ndarray,
                  prominence: float = 0.05, distance: int = 10) -> List[Dict[str, Any]]:
        """Find peaks in XRF spectrum"""
        # Normalize for peak finding
        norm_counts = counts / np.max(counts) if np.max(counts) > 0 else counts
        
        # Find peaks
        peaks, properties = signal.find_peaks(
            norm_counts,
            prominence=prominence,
            distance=distance
        )
        
        # Identify elements
        peak_list = []
        for idx in peaks:
            element_line = self._identify_element_line(energy[idx])
            peak_list.append({
                'energy': energy[idx],
                'counts': counts[idx],
                'element_line': element_line,
                'prominence': properties['prominences'][np.where(peaks == idx)[0][0]]
            })
        
        # Check for escape and sum peaks
        peak_list.extend(self._identify_artifact_peaks(energy, counts, peak_list))
        
        return peak_list
    
    def _identify_element_line(self, energy: float, tolerance: float = 0.05) -> Optional[str]:
        """Identify element and line from X-ray energy"""
        for symbol, element in self.element_db.elements.items():
            for line, ref_energy in element.xrf_lines.items():
                if abs(energy - ref_energy) < tolerance:
                    return f"{symbol} {line}"
        return None
    
    def _identify_artifact_peaks(self, energy: np.ndarray, counts: np.ndarray,
                                primary_peaks: List[Dict]) -> List[Dict]:
        """Identify escape peaks and sum peaks"""
        artifact_peaks = []
        si_edge = 1.84  # keV, Si K-edge for escape peaks
        
        for peak in primary_peaks:
            # Check for Si escape peak
            escape_energy = peak['energy'] - si_edge
            if escape_energy > 0:
                idx = np.argmin(np.abs(energy - escape_energy))
                if counts[idx] > np.mean(counts) * 2:
                    artifact_peaks.append({
                        'energy': energy[idx],
                        'counts': counts[idx],
                        'element_line': f"{peak['element_line']} (Si escape)",
                        'type': 'escape'
                    })
            
            # Check for sum peaks (simplified)
            for peak2 in primary_peaks:
                sum_energy = peak['energy'] + peak2['energy']
                if sum_energy < energy[-1]:
                    idx = np.argmin(np.abs(energy - sum_energy))
                    if counts[idx] > np.mean(counts) * 1.5:
                        artifact_peaks.append({
                            'energy': energy[idx],
                            'counts': counts[idx],
                            'element_line': f"Sum peak ({peak['element_line']} + {peak2['element_line']})",
                            'type': 'sum'
                        })
        
        return artifact_peaks
    
    def quantification_fundamental_parameters(self, peaks: List[XRFPeak],
                                            matrix: str = 'SiO2',
                                            thickness: Optional[float] = None) -> Dict[str, float]:
        """Quantify composition using fundamental parameters method"""
        compositions = {}
        
        # Matrix correction factors (simplified Sherman equation)
        matrix_corrections = self._calculate_matrix_corrections(peaks, matrix)
        
        for peak in peaks:
            if peak.escape_peak or peak.sum_peak:
                continue
            
            # Get element properties
            elem = self.element_db.get_element(peak.element)
            if not elem:
                continue
            
            # Calculate concentration
            # C_i = I_i / (σ_i * ω_i * ε_i * A_i)
            # Where σ = cross-section, ω = fluorescence yield, ε = detector efficiency, A = absorption
            
            cross_section = self._photoionization_cross_section(
                elem.atomic_number, self.excitation_energy
            )
            fluorescence_yield = elem.fluorescence_yield
            detector_efficiency = self._detector_efficiency(peak.energy)
            absorption = matrix_corrections.get(peak.element, 1.0)
            
            # Simplified concentration calculation
            sensitivity = cross_section * fluorescence_yield * detector_efficiency * absorption
            
            if sensitivity > 0:
                concentration = peak.intensity / sensitivity
                compositions[peak.element] = concentration
        
        # Normalize to 100%
        total = sum(compositions.values())
        if total > 0:
            for element in compositions:
                compositions[element] = (compositions[element] / total) * 100
        
        return compositions
    
    def _calculate_matrix_corrections(self, peaks: List[XRFPeak], 
                                     matrix: str) -> Dict[str, float]:
        """Calculate matrix absorption corrections"""
        corrections = {}
        
        # Simplified matrix corrections based on common matrices
        matrix_factors = {
            'SiO2': {'Si': 0.9, 'O': 0.85, 'Al': 0.8, 'Fe': 0.75},
            'GaAs': {'Ga': 0.95, 'As': 0.95, 'Si': 0.7, 'O': 0.6},
            'Al2O3': {'Al': 0.9, 'O': 0.85, 'Si': 0.8},
        }
        
        default_correction = 0.8
        
        for peak in peaks:
            if peak.element in matrix_factors.get(matrix, {}):
                corrections[peak.element] = matrix_factors[matrix][peak.element]
            else:
                corrections[peak.element] = default_correction
        
        return corrections
    
    def _photoionization_cross_section(self, z: int, energy: float) -> float:
        """Calculate photoionization cross-section (simplified)"""
        # Simplified model: σ ∝ Z^4 / E^3
        return (z**4) / (energy**3) * 1e-24  # barn
    
    def _detector_efficiency(self, energy: float) -> float:
        """Calculate detector efficiency for given energy"""
        # Simplified Si detector efficiency model
        if energy < 1.84:  # Below Si K-edge
            return 0.95
        elif energy < 10:
            return 0.9 - 0.05 * (energy - 1.84) / 8.16
        else:
            return 0.5 * np.exp(-energy / 20)
    
    def standardless_quantification(self, energy: np.ndarray, 
                                   counts: np.ndarray) -> Dict[str, float]:
        """Perform standardless quantification"""
        # Find peaks
        peaks = self.find_peaks(energy, counts)
        
        # Convert to XRFPeak objects
        xrf_peaks = []
        for peak in peaks:
            if peak['element_line'] and not any(x in peak['element_line'] for x in ['escape', 'Sum']):
                element = peak['element_line'].split()[0]
                line = peak['element_line'].split()[1] if len(peak['element_line'].split()) > 1 else 'Kα'
                
                xrf_peaks.append(XRFPeak(
                    energy=peak['energy'],
                    intensity=peak['counts'],
                    fwhm=0.15,  # Typical for Si detector
                    element=element,
                    line=line
                ))
        
        # Quantify using fundamental parameters
        return self.quantification_fundamental_parameters(xrf_peaks)
    
    def detection_limits(self, energy: np.ndarray, counts: np.ndarray,
                        measurement_time: float = 300) -> Dict[str, float]:
        """Calculate detection limits for elements"""
        # Background estimation
        background = signal.savgol_filter(counts, 51, 2)
        
        detection_limits = {}
        
        for symbol, element in self.element_db.elements.items():
            for line, line_energy in element.xrf_lines.items():
                if 'α' in line:  # Focus on primary lines
                    # Find background at line position
                    idx = np.argmin(np.abs(energy - line_energy))
                    if idx < len(background):
                        bg_counts = background[idx]
                        
                        # Calculate MDL using 3σ criterion
                        # MDL = 3 * sqrt(2 * B * t) / (S * t)
                        # Where B = background count rate, t = time, S = sensitivity
                        
                        if bg_counts > 0 and measurement_time > 0:
                            mdl = 3 * np.sqrt(2 * bg_counts / measurement_time)
                            
                            # Convert to concentration (ppm) - simplified
                            sensitivity = self._detector_efficiency(line_energy) * element.fluorescence_yield
                            if sensitivity > 0:
                                mdl_ppm = (mdl / sensitivity) * 1e6
                                detection_limits[f"{symbol}"] = mdl_ppm
        
        return detection_limits

class ChemicalSimulator:
    """Simulator for XPS and XRF spectra"""
    
    def __init__(self):
        self.element_db = ElementDatabase()
        
    def generate_xps_spectrum(self, composition: Dict[str, float],
                             binding_energy: Optional[np.ndarray] = None,
                             source: XRaySource = XRaySource.AL_KA,
                             resolution: float = 0.5,
                             noise_level: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic XPS spectrum"""
        if binding_energy is None:
            binding_energy = np.linspace(0, 1200, 2400)
        
        intensity = np.zeros_like(binding_energy)
        
        for element, concentration in composition.items():
            elem = self.element_db.get_element(element)
            if elem:
                # Add peaks for this element
                for orbital, be in elem.xps_peaks.items():
                    if ':' not in orbital and '-' not in orbital and '=' not in orbital:
                        # Calculate peak parameters
                        amplitude = concentration * elem.sensitivity_factors.get(orbital, 1.0) * 100
                        
                        # Check for multiplet splitting
                        if any(x in orbital for x in ['p', 'd', 'f']) and len(orbital) > 1:
                            # Add multiplet peaks
                            if '3/2' in orbital:
                                fwhm = 1.2 + np.random.normal(0, 0.1)
                                peak = self._generate_peak(binding_energy, be, amplitude, fwhm, 'voigt')
                                intensity += peak
                            elif '1/2' in orbital or '5/2' in orbital:
                                fwhm = 1.3 + np.random.normal(0, 0.1)
                                peak = self._generate_peak(binding_energy, be, amplitude * 0.5, fwhm, 'voigt')
                                intensity += peak
                        else:
                            # Single peak
                            fwhm = resolution + np.random.normal(0, 0.1)
                            peak = self._generate_peak(binding_energy, be, amplitude, fwhm, 'voigt')
                            intensity += peak
        
        # Add Shirley background
        background = self._generate_shirley_background(binding_energy, intensity)
        intensity += background
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.max(intensity), len(intensity))
        intensity += noise
        
        # Ensure positive values
        intensity = np.maximum(intensity, 0)
        
        return binding_energy, intensity
    
    def generate_xrf_spectrum(self, composition: Dict[str, float],
                            energy: Optional[np.ndarray] = None,
                            excitation_energy: float = 50.0,
                            resolution: float = 0.15,
                            noise_level: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic XRF spectrum"""
        if energy is None:
            energy = np.linspace(0.1, 20, 2000)
        
        counts = np.zeros_like(energy)
        
        for element, concentration in composition.items():
            elem = self.element_db.get_element(element)
            if elem:
                # Add XRF lines for this element
                for line, line_energy in elem.xrf_lines.items():
                    if line_energy < excitation_energy:
                        # Calculate intensity based on fluorescence yield and concentration
                        intensity = concentration * elem.fluorescence_yield * 1000
                        
                        # Adjust intensity for different lines
                        if 'β' in line:
                            intensity *= 0.2
                        elif 'L' in line:
                            intensity *= 0.3
                        
                        # Generate peak
                        fwhm = resolution * np.sqrt(line_energy)  # Energy-dependent resolution
                        peak = self._generate_peak(energy, line_energy, intensity, fwhm, 'gaussian')
                        counts += peak
                        
                        # Add Si escape peak (for Si detector)
                        if line_energy > 1.84:
                            escape_energy = line_energy - 1.84
                            escape_intensity = intensity * 0.05
                            escape_peak = self._generate_peak(energy, escape_energy, escape_intensity, fwhm, 'gaussian')
                            counts += escape_peak
        
        # Add Bremsstrahlung background
        background = self._generate_bremsstrahlung(energy, excitation_energy)
        counts += background * 10
        
        # Add Poisson noise
        counts = np.random.poisson(counts)
        
        # Add detector noise
        noise = np.random.normal(0, noise_level * np.max(counts), len(counts))
        counts += noise
        
        # Ensure positive values
        counts = np.maximum(counts, 0)
        
        return energy, counts
    
    def _generate_peak(self, x: np.ndarray, position: float, amplitude: float,
                      fwhm: float, shape: str = 'gaussian') -> np.ndarray:
        """Generate a single peak"""
        if shape == 'gaussian':
            sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
            return amplitude * np.exp(-(x - position)**2 / (2 * sigma**2))
        elif shape == 'lorentzian':
            gamma = fwhm / 2
            return amplitude * gamma**2 / ((x - position)**2 + gamma**2)
        elif shape == 'voigt':
            # Simplified pseudo-Voigt
            gaussian = self._generate_peak(x, position, amplitude, fwhm, 'gaussian')
            lorentzian = self._generate_peak(x, position, amplitude, fwhm, 'lorentzian')
            eta = 0.5
            return eta * lorentzian + (1 - eta) * gaussian
        else:
            return np.zeros_like(x)
    
    def _generate_shirley_background(self, binding_energy: np.ndarray, 
                                    intensity: np.ndarray) -> np.ndarray:
        """Generate Shirley-type background"""
        background = np.zeros_like(intensity)
        
        # Simplified Shirley background
        cumsum = np.cumsum(intensity[::-1])[::-1]
        if np.max(cumsum) > 0:
            background = cumsum / np.max(cumsum) * np.max(intensity) * 0.3
        
        return background
    
    def _generate_bremsstrahlung(self, energy: np.ndarray, 
                                excitation_energy: float) -> np.ndarray:
        """Generate Bremsstrahlung background for XRF"""
        # Kramers' law: I ∝ Z(E_0 - E)/E
        background = np.zeros_like(energy)
        mask = energy < excitation_energy
        background[mask] = (excitation_energy - energy[mask]) / (energy[mask] + 0.1)
        
        return background / np.max(background) if np.max(background) > 0 else background

# Test the implementation
if __name__ == "__main__":
    print("Session 11: XPS/XRF Analysis System")
    print("=" * 50)
    
    # Test XPS
    print("\n1. Testing XPS Analysis...")
    
    # Create simulator and generate test spectrum
    simulator = ChemicalSimulator()
    composition = {'Si': 30, 'O': 50, 'C': 15, 'N': 5}
    be, intensity = simulator.generate_xps_spectrum(composition)
    
    # Analyze with XPS
    xps = XPSAnalyzer()
    be_processed, int_processed = xps.process_spectrum(be, intensity)
    
    # Find peaks
    peaks = xps.find_peaks(be_processed, int_processed)
    print(f"Found {len(peaks)} XPS peaks")
    
    # Fit a peak
    if peaks:
        # Select region around first major peak
        peak_pos = peaks[0]['position']
        mask = (be_processed > peak_pos - 10) & (be_processed < peak_pos + 10)
        fit_result = xps.fit_peak(be_processed[mask], int_processed[mask])
        
        if fit_result['success']:
            print(f"Peak fit: Position={fit_result['position']:.1f} eV, FWHM={fit_result['fwhm']:.2f} eV")
    
    # Test XRF
    print("\n2. Testing XRF Analysis...")
    
    # Generate test XRF spectrum
    energy, counts = simulator.generate_xrf_spectrum(composition)
    
    # Analyze with XRF
    xrf = XRFAnalyzer()
    energy_processed, counts_processed = xrf.process_spectrum(energy, counts)
    
    # Find peaks
    xrf_peaks = xrf.find_peaks(energy_processed, counts_processed)
    print(f"Found {len(xrf_peaks)} XRF peaks")
    
    # Quantification
    quant = xrf.standardless_quantification(energy_processed, counts_processed)
    print("\nQuantification results:")
    for element, conc in quant.items():
        print(f"  {element}: {conc:.1f}%")
    
    # Detection limits
    mdl = xrf.detection_limits(energy_processed, counts_processed)
    print("\nDetection limits (ppm):")
    for element, limit in list(mdl.items())[:5]:
        print(f"  {element}: {limit:.1f} ppm")
    
    print("\n✅ Session 11 XPS/XRF implementation complete!")
