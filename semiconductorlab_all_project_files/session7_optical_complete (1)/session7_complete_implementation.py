"""
Session 7: Optical I - UV-Vis-NIR and FTIR Complete Implementation
=================================================================
Production-ready optical spectroscopy analysis for semiconductor characterization
Including absorption, transmission, reflectance, Tauc plots, and FTIR analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import scipy.signal as signal
import scipy.optimize as optimize
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.integrate import trapz
import warnings
import json
import hashlib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
h = 6.62607015e-34  # Planck constant (J⋅s)
c = 299792458       # Speed of light (m/s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)
e = 1.602176634e-19 # Elementary charge (C)
eV = e              # Electronvolt (J)

class MeasurementType(Enum):
    """Optical measurement types"""
    TRANSMISSION = "transmission"
    ABSORPTION = "absorption"
    REFLECTANCE = "reflectance"
    ATR = "atr"  # Attenuated Total Reflectance
    DIFFUSE_REFLECTANCE = "diffuse_reflectance"

class BandgapType(Enum):
    """Types of band gaps for Tauc analysis"""
    DIRECT_ALLOWED = "direct_allowed"
    DIRECT_FORBIDDEN = "direct_forbidden"
    INDIRECT_ALLOWED = "indirect_allowed"
    INDIRECT_FORBIDDEN = "indirect_forbidden"

class BaselineMethod(Enum):
    """Baseline correction methods"""
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    RUBBERBAND = "rubberband"
    ASYMMETRIC_LEAST_SQUARES = "als"
    SAVITZKY_GOLAY = "savgol"
    AIRPLS = "airpls"  # Adaptive iteratively reweighted penalized least squares

@dataclass
class SpectralData:
    """Container for spectral data"""
    wavelength: np.ndarray  # nm
    intensity: np.ndarray   # a.u. or %
    measurement_type: MeasurementType
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate data consistency"""
        if len(self.wavelength) != len(self.intensity):
            raise ValueError("Wavelength and intensity arrays must have same length")
        
        # Sort by wavelength
        sort_idx = np.argsort(self.wavelength)
        self.wavelength = self.wavelength[sort_idx]
        self.intensity = self.intensity[sort_idx]
    
    def to_energy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert wavelength (nm) to energy (eV)"""
        energy = (h * c) / (self.wavelength * 1e-9) / eV
        return energy[::-1], self.intensity[::-1]  # Reverse for increasing energy
    
    def to_wavenumber(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert wavelength (nm) to wavenumber (cm⁻¹)"""
        wavenumber = 1e7 / self.wavelength  # nm to cm⁻¹
        return wavenumber[::-1], self.intensity[::-1]

@dataclass
class TaucResult:
    """Result of Tauc analysis"""
    bandgap: float  # eV
    bandgap_error: float  # eV
    r_squared: float
    tauc_x: np.ndarray  # Energy (eV)
    tauc_y: np.ndarray  # (αhν)^n
    fit_x: np.ndarray
    fit_y: np.ndarray
    bandgap_type: BandgapType
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PeakFitResult:
    """Result of peak fitting"""
    peaks: List[Dict[str, float]]  # List of peak parameters
    baseline: np.ndarray
    fitted_spectrum: np.ndarray
    residuals: np.ndarray
    r_squared: float
    chi_squared: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class UVVisNIRAnalyzer:
    """
    UV-Vis-NIR Spectroscopy Analysis
    Handles transmission, absorption, reflectance measurements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def process_spectrum(self, data: SpectralData, 
                        smooth: bool = True,
                        baseline_correct: bool = True,
                        baseline_method: BaselineMethod = BaselineMethod.RUBBERBAND) -> SpectralData:
        """
        Process raw spectrum with smoothing and baseline correction
        """
        processed = SpectralData(
            wavelength=data.wavelength.copy(),
            intensity=data.intensity.copy(),
            measurement_type=data.measurement_type,
            metadata=data.metadata.copy()
        )
        
        if smooth:
            processed.intensity = self._smooth_spectrum(processed.intensity)
            
        if baseline_correct:
            baseline = self._calculate_baseline(
                processed.wavelength, 
                processed.intensity,
                method=baseline_method
            )
            processed.intensity = processed.intensity - baseline
            processed.metadata['baseline'] = baseline.tolist()
        
        return processed
    
    def calculate_absorption(self, transmission: SpectralData,
                            reference: Optional[SpectralData] = None) -> SpectralData:
        """
        Calculate absorption from transmission
        A = -log10(T/T0) where T0 is reference (usually 100% or air)
        """
        if transmission.measurement_type != MeasurementType.TRANSMISSION:
            raise ValueError("Input must be transmission data")
        
        # Interpolate reference if provided
        if reference is not None:
            f = interp1d(reference.wavelength, reference.intensity, 
                        kind='cubic', fill_value='extrapolate')
            ref_intensity = f(transmission.wavelength)
        else:
            ref_intensity = 100.0  # Assume 100% reference
        
        # Calculate absorbance
        T = transmission.intensity / ref_intensity
        T = np.clip(T, 1e-10, 1.0)  # Avoid log(0) and T>1
        
        absorption = SpectralData(
            wavelength=transmission.wavelength,
            intensity=-np.log10(T),
            measurement_type=MeasurementType.ABSORPTION,
            metadata={'source': 'calculated_from_transmission'}
        )
        
        return absorption
    
    def calculate_absorption_coefficient(self, absorption: SpectralData,
                                       thickness: float) -> np.ndarray:
        """
        Calculate absorption coefficient α from absorbance
        α = 2.303 * A / d where d is thickness in cm
        Returns α in cm⁻¹
        """
        if absorption.measurement_type != MeasurementType.ABSORPTION:
            raise ValueError("Input must be absorption data")
        
        alpha = 2.303 * absorption.intensity / (thickness / 10)  # Convert mm to cm
        return alpha
    
    def tauc_analysis(self, data: SpectralData,
                     thickness: float,  # mm
                     bandgap_type: BandgapType = BandgapType.DIRECT_ALLOWED,
                     energy_range: Optional[Tuple[float, float]] = None) -> TaucResult:
        """
        Perform Tauc analysis to determine optical bandgap
        (αhν)^n = A(hν - Eg)
        """
        # Get absorption coefficient
        if data.measurement_type == MeasurementType.TRANSMISSION:
            absorption = self.calculate_absorption(data)
        else:
            absorption = data
            
        alpha = self.calculate_absorption_coefficient(absorption, thickness)
        
        # Convert to energy
        energy, _ = absorption.to_energy()
        
        # Determine Tauc exponent
        n_values = {
            BandgapType.DIRECT_ALLOWED: 2,
            BandgapType.DIRECT_FORBIDDEN: 2/3,
            BandgapType.INDIRECT_ALLOWED: 1/2,
            BandgapType.INDIRECT_FORBIDDEN: 1/3
        }
        n = n_values[bandgap_type]
        
        # Calculate Tauc plot
        tauc_y = (alpha * energy) ** n
        
        # Find linear region for fitting
        if energy_range is None:
            energy_range = self._find_linear_region(energy, tauc_y)
        
        # Fit linear region
        mask = (energy >= energy_range[0]) & (energy <= energy_range[1])
        fit_x = energy[mask]
        fit_y = tauc_y[mask]
        
        # Linear fit
        coeffs = np.polyfit(fit_x, fit_y, 1)
        poly = np.poly1d(coeffs)
        
        # Find bandgap (x-intercept)
        bandgap = -coeffs[1] / coeffs[0] if coeffs[0] != 0 else 0
        
        # Calculate R²
        y_pred = poly(fit_x)
        ss_res = np.sum((fit_y - y_pred) ** 2)
        ss_tot = np.sum((fit_y - np.mean(fit_y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Estimate error
        residuals = fit_y - y_pred
        std_error = np.std(residuals)
        bandgap_error = std_error / np.abs(coeffs[0]) if coeffs[0] != 0 else 0
        
        return TaucResult(
            bandgap=bandgap,
            bandgap_error=bandgap_error,
            r_squared=r_squared,
            tauc_x=energy,
            tauc_y=tauc_y,
            fit_x=np.linspace(min(fit_x), max(fit_x) * 1.1, 100),
            fit_y=poly(np.linspace(min(fit_x), max(fit_x) * 1.1, 100)),
            bandgap_type=bandgap_type,
            metadata={
                'fit_range': energy_range,
                'slope': coeffs[0],
                'intercept': coeffs[1]
            }
        )
    
    def _smooth_spectrum(self, intensity: np.ndarray, 
                        window_length: int = 5,
                        polyorder: int = 2) -> np.ndarray:
        """Apply Savitzky-Golay filter for smoothing"""
        if len(intensity) < window_length:
            return intensity
        return signal.savgol_filter(intensity, window_length, polyorder)
    
    def _calculate_baseline(self, x: np.ndarray, y: np.ndarray,
                          method: BaselineMethod) -> np.ndarray:
        """Calculate baseline using specified method"""
        if method == BaselineMethod.LINEAR:
            return self._linear_baseline(x, y)
        elif method == BaselineMethod.POLYNOMIAL:
            return self._polynomial_baseline(x, y)
        elif method == BaselineMethod.RUBBERBAND:
            return self._rubberband_baseline(x, y)
        elif method == BaselineMethod.ASYMMETRIC_LEAST_SQUARES:
            return self._als_baseline(y)
        else:
            return np.zeros_like(y)
    
    def _linear_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Simple linear baseline between endpoints"""
        coeffs = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
        return np.polyval(coeffs, x)
    
    def _polynomial_baseline(self, x: np.ndarray, y: np.ndarray, 
                           degree: int = 3) -> np.ndarray:
        """Polynomial baseline fitting"""
        # Find minima points
        minima_idx = signal.argrelmin(y, order=len(y)//20)[0]
        if len(minima_idx) < degree + 1:
            return self._linear_baseline(x, y)
        
        # Fit polynomial through minima
        coeffs = np.polyfit(x[minima_idx], y[minima_idx], degree)
        return np.polyval(coeffs, x)
    
    def _rubberband_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Rubberband baseline - convex hull approach"""
        from scipy.spatial import ConvexHull
        
        # Create points for convex hull
        points = np.column_stack((x, y))
        
        # Find lower convex hull
        hull = ConvexHull(points)
        vertices = hull.vertices
        
        # Get lower hull points
        lower_hull_idx = []
        for i in range(len(vertices)):
            v1 = vertices[i]
            v2 = vertices[(i+1) % len(vertices)]
            if points[v1, 0] < points[v2, 0]:  # Going left to right
                lower_hull_idx.extend([v1, v2])
        
        lower_hull_idx = sorted(list(set(lower_hull_idx)))
        
        if len(lower_hull_idx) < 2:
            return self._linear_baseline(x, y)
        
        # Interpolate baseline
        f = interp1d(x[lower_hull_idx], y[lower_hull_idx], 
                    kind='linear', fill_value='extrapolate')
        return f(x)
    
    def _als_baseline(self, y: np.ndarray, lam: float = 1e6, 
                     p: float = 0.01, niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares baseline
        P. Eilers and H. Boelens, 2005
        """
        L = len(y)
        D = np.diff(np.eye(L), 2)  # Second derivative matrix
        w = np.ones(L)
        
        for i in range(niter):
            W = np.diag(w)
            Z = W + lam * D.T @ D
            z = np.linalg.solve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
        
        return z
    
    def _find_linear_region(self, x: np.ndarray, y: np.ndarray,
                          window_size: int = 10) -> Tuple[float, float]:
        """Find the most linear region for Tauc plot fitting"""
        # Calculate local R² values
        r2_values = []
        x_centers = []
        
        for i in range(len(x) - window_size):
            x_window = x[i:i+window_size]
            y_window = y[i:i+window_size]
            
            # Fit linear
            coeffs = np.polyfit(x_window, y_window, 1)
            poly = np.poly1d(coeffs)
            y_pred = poly(x_window)
            
            # Calculate R²
            ss_res = np.sum((y_window - y_pred) ** 2)
            ss_tot = np.sum((y_window - np.mean(y_window)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            r2_values.append(r2)
            x_centers.append(np.mean(x_window))
        
        # Find region with highest R²
        if r2_values:
            best_idx = np.argmax(r2_values)
            center = x_centers[best_idx]
            half_window = (x[window_size] - x[0]) / 2
            return (center - half_window, center + half_window)
        
        # Default to middle third
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min
        return (x_min + x_range/3, x_min + 2*x_range/3)


class FTIRAnalyzer:
    """
    FTIR Spectroscopy Analysis
    Handles vibrational spectroscopy for semiconductors
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_peak_database()
    
    def _load_peak_database(self):
        """Load database of common semiconductor FTIR peaks"""
        self.peak_database = {
            'Si-O': {'position': 1100, 'range': (1050, 1150), 'type': 'stretching'},
            'Si-H': {'position': 2100, 'range': (2000, 2200), 'type': 'stretching'},
            'Si-N': {'position': 840, 'range': (800, 880), 'type': 'stretching'},
            'C-H': {'position': 2900, 'range': (2800, 3000), 'type': 'stretching'},
            'O-H': {'position': 3400, 'range': (3200, 3600), 'type': 'stretching'},
            'C=O': {'position': 1700, 'range': (1650, 1750), 'type': 'stretching'},
            'N-H': {'position': 3300, 'range': (3250, 3350), 'type': 'stretching'},
            'Si-C': {'position': 800, 'range': (750, 850), 'type': 'stretching'},
            'Ga-As': {'position': 270, 'range': (250, 290), 'type': 'phonon'},
            'In-P': {'position': 340, 'range': (320, 360), 'type': 'phonon'},
        }
    
    def process_ftir_spectrum(self, data: SpectralData,
                             baseline_method: BaselineMethod = BaselineMethod.ASYMMETRIC_LEAST_SQUARES,
                             smooth: bool = True) -> SpectralData:
        """Process FTIR spectrum with baseline correction"""
        # Convert to wavenumber if needed
        if np.mean(data.wavelength) > 100:  # Likely in nm
            wavenumber, intensity = data.to_wavenumber()
            processed = SpectralData(
                wavelength=wavenumber,  # Using wavelength field for wavenumber
                intensity=intensity,
                measurement_type=data.measurement_type,
                metadata={'units': 'wavenumber_cm-1'}
            )
        else:
            processed = data
        
        # Apply baseline correction
        if baseline_method:
            baseline = self._calculate_ftir_baseline(
                processed.wavelength,
                processed.intensity,
                method=baseline_method
            )
            processed.intensity = processed.intensity - baseline
        
        # Smooth if requested
        if smooth:
            processed.intensity = signal.savgol_filter(
                processed.intensity, 
                window_length=11, 
                polyorder=3
            )
        
        return processed
    
    def find_peaks(self, data: SpectralData,
                  prominence: float = 0.01,
                  distance: int = 10,
                  identify: bool = True) -> Dict[str, Any]:
        """
        Find and identify peaks in FTIR spectrum
        """
        # Find peaks
        peaks, properties = signal.find_peaks(
            data.intensity,
            prominence=prominence * np.max(data.intensity),
            distance=distance
        )
        
        # Get peak positions and intensities
        peak_positions = data.wavelength[peaks]
        peak_intensities = data.intensity[peaks]
        
        # Calculate peak widths
        widths = signal.peak_widths(data.intensity, peaks, rel_height=0.5)
        peak_widths = widths[0] * np.mean(np.diff(data.wavelength))
        
        # Identify peaks if requested
        identifications = []
        if identify:
            for pos in peak_positions:
                match = self._identify_peak(pos)
                identifications.append(match)
        
        return {
            'positions': peak_positions,
            'intensities': peak_intensities,
            'widths': peak_widths,
            'identifications': identifications,
            'indices': peaks
        }
    
    def fit_peaks(self, data: SpectralData,
                 peak_type: str = 'lorentzian',
                 max_peaks: int = 10) -> PeakFitResult:
        """
        Fit peaks with specified function (Gaussian, Lorentzian, Voigt)
        """
        # Find initial peaks
        peak_info = self.find_peaks(data)
        peaks_idx = peak_info['indices'][:max_peaks]
        
        if len(peaks_idx) == 0:
            return PeakFitResult(
                peaks=[],
                baseline=np.zeros_like(data.intensity),
                fitted_spectrum=data.intensity,
                residuals=np.zeros_like(data.intensity),
                r_squared=0,
                chi_squared=0
            )
        
        # Prepare fitting
        x = data.wavelength
        y = data.intensity
        
        # Initial parameters
        params = []
        for idx in peaks_idx:
            amplitude = y[idx]
            position = x[idx]
            width = 10  # Initial guess
            params.extend([amplitude, position, width])
        
        # Add baseline parameters (linear)
        params.extend([0, np.min(y)])
        
        # Fit function
        if peak_type == 'gaussian':
            fit_func = self._multi_gaussian
        elif peak_type == 'lorentzian':
            fit_func = self._multi_lorentzian
        else:
            fit_func = self._multi_voigt
        
        # Optimize
        try:
            popt, pcov = optimize.curve_fit(
                lambda x, *p: fit_func(x, p),
                x, y, p0=params,
                maxfev=5000
            )
        except:
            # Fallback to simple peak list
            return self._simple_peak_fit(data, peak_info)
        
        # Extract results
        fitted_spectrum = fit_func(x, popt)
        residuals = y - fitted_spectrum
        
        # Calculate metrics
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        chi_squared = np.sum(residuals ** 2 / np.abs(fitted_spectrum + 1e-10))
        
        # Extract individual peaks
        n_peaks = len(peaks_idx)
        peaks = []
        for i in range(n_peaks):
            peak_params = popt[i*3:(i+1)*3]
            peaks.append({
                'amplitude': peak_params[0],
                'position': peak_params[1],
                'width': peak_params[2],
                'area': self._calculate_peak_area(peak_params, peak_type)
            })
        
        # Baseline
        baseline = popt[-2] * x + popt[-1]
        
        return PeakFitResult(
            peaks=peaks,
            baseline=baseline,
            fitted_spectrum=fitted_spectrum,
            residuals=residuals,
            r_squared=r_squared,
            chi_squared=chi_squared,
            metadata={'peak_type': peak_type}
        )
    
    def calculate_film_thickness(self, data: SpectralData,
                                n_substrate: float = 3.42,  # e.g., Si
                                angle: float = 0) -> Dict[str, float]:
        """
        Calculate thin film thickness from interference fringes
        """
        # Find maxima and minima
        maxima_idx = signal.argrelmax(data.intensity)[0]
        minima_idx = signal.argrelmin(data.intensity)[0]
        
        if len(maxima_idx) < 2:
            return {'thickness': 0, 'error': np.inf, 'n_fringes': 0}
        
        # Use maxima for calculation
        wavenumbers = 1e7 / data.wavelength[maxima_idx]  # Convert to cm⁻¹
        
        # Calculate thickness from fringe spacing
        # d = m / (2n * Δν) where m is fringe order
        thicknesses = []
        for i in range(len(wavenumbers) - 1):
            delta_nu = np.abs(wavenumbers[i+1] - wavenumbers[i])
            if delta_nu > 0:
                # Assume adjacent fringes differ by 1 order
                thickness = 1 / (2 * n_substrate * delta_nu) * 1e4  # µm
                thicknesses.append(thickness)
        
        if thicknesses:
            thickness_mean = np.mean(thicknesses)
            thickness_std = np.std(thicknesses)
        else:
            thickness_mean = 0
            thickness_std = np.inf
        
        return {
            'thickness': thickness_mean,  # µm
            'error': thickness_std,
            'n_fringes': len(maxima_idx),
            'refractive_index': n_substrate
        }
    
    def _calculate_ftir_baseline(self, x: np.ndarray, y: np.ndarray,
                                method: BaselineMethod) -> np.ndarray:
        """Calculate baseline for FTIR spectrum"""
        analyzer = UVVisNIRAnalyzer()
        return analyzer._calculate_baseline(x, y, method)
    
    def _identify_peak(self, position: float, tolerance: float = 50) -> Optional[str]:
        """Identify peak based on position"""
        for name, info in self.peak_database.items():
            if info['range'][0] <= position <= info['range'][1]:
                return name
        return None
    
    def _multi_gaussian(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Multiple Gaussian peaks with linear baseline"""
        n_peaks = (len(params) - 2) // 3
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i*3]
            center = params[i*3 + 1]
            width = params[i*3 + 2]
            y += amp * np.exp(-0.5 * ((x - center) / width) ** 2)
        
        # Add baseline
        y += params[-2] * x + params[-1]
        return y
    
    def _multi_lorentzian(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Multiple Lorentzian peaks with linear baseline"""
        n_peaks = (len(params) - 2) // 3
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i*3]
            center = params[i*3 + 1]
            width = params[i*3 + 2]
            y += amp * width**2 / ((x - center)**2 + width**2)
        
        # Add baseline
        y += params[-2] * x + params[-1]
        return y
    
    def _multi_voigt(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Multiple Voigt peaks (Gaussian + Lorentzian) with linear baseline"""
        # Simplified pseudo-Voigt (linear combination)
        eta = 0.5  # Mixing parameter
        gaussian = self._multi_gaussian(x, params)
        lorentzian = self._multi_lorentzian(x, params)
        return eta * lorentzian + (1 - eta) * gaussian
    
    def _calculate_peak_area(self, params: List[float], peak_type: str) -> float:
        """Calculate integrated area under peak"""
        amp, center, width = params
        
        if peak_type == 'gaussian':
            # Area = amplitude * width * sqrt(2π)
            return amp * width * np.sqrt(2 * np.pi)
        elif peak_type == 'lorentzian':
            # Area = amplitude * width * π
            return amp * width * np.pi
        else:  # Voigt
            # Approximate
            return amp * width * np.pi
    
    def _simple_peak_fit(self, data: SpectralData, 
                        peak_info: Dict) -> PeakFitResult:
        """Simple peak fitting fallback"""
        peaks = []
        for i, (pos, intensity, width) in enumerate(zip(
            peak_info['positions'],
            peak_info['intensities'],
            peak_info['widths']
        )):
            peaks.append({
                'amplitude': intensity,
                'position': pos,
                'width': width,
                'area': intensity * width
            })
        
        return PeakFitResult(
            peaks=peaks,
            baseline=np.zeros_like(data.intensity),
            fitted_spectrum=data.intensity,
            residuals=np.zeros_like(data.intensity),
            r_squared=0,
            chi_squared=0
        )


class OpticalTestDataGenerator:
    """
    Generate synthetic test data for optical measurements
    Physics-based models for realistic spectra
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_uv_vis_spectrum(self, 
                                material: str = 'GaAs',
                                measurement_type: MeasurementType = MeasurementType.TRANSMISSION,
                                wavelength_range: Tuple[float, float] = (300, 1100),
                                n_points: int = 500) -> SpectralData:
        """Generate synthetic UV-Vis-NIR spectrum"""
        
        # Material parameters
        materials = {
            'GaAs': {'Eg': 1.42, 'A': 5.0, 'n': 3.5},
            'Si': {'Eg': 1.12, 'A': 3.0, 'n': 3.4},
            'GaN': {'Eg': 3.4, 'A': 6.0, 'n': 2.3},
            'InP': {'Eg': 1.35, 'A': 4.5, 'n': 3.1},
            'CdTe': {'Eg': 1.5, 'A': 4.0, 'n': 2.7}
        }
        
        params = materials.get(material, materials['GaAs'])
        
        # Generate wavelength array
        wavelength = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
        energy = (h * c) / (wavelength * 1e-9) / eV
        
        # Generate absorption coefficient
        alpha = np.zeros_like(energy)
        mask = energy > params['Eg']
        alpha[mask] = params['A'] * 1e5 * np.sqrt(energy[mask] - params['Eg'])
        
        # Add Urbach tail
        mask_tail = energy <= params['Eg']
        E0 = 0.05  # Urbach energy
        alpha[mask_tail] = 1e3 * np.exp((energy[mask_tail] - params['Eg']) / E0)
        
        # Calculate transmission/absorption
        thickness = 0.5  # mm
        if measurement_type == MeasurementType.TRANSMISSION:
            T = np.exp(-alpha * thickness / 10)
            # Add interference fringes
            n = params['n']
            phase = 4 * np.pi * n * thickness * 1000 / wavelength
            fringes = 1 + 0.3 * np.cos(phase)
            T = T * fringes
            T = np.clip(T * 100, 0, 100)  # Convert to %
            intensity = T
        else:  # Absorption
            A = alpha * thickness / 10 / 2.303  # Absorbance
            intensity = A
        
        # Add noise
        noise = np.random.normal(0, 0.001 * np.max(intensity), len(intensity))
        intensity = intensity + noise
        
        return SpectralData(
            wavelength=wavelength,
            intensity=intensity,
            measurement_type=measurement_type,
            metadata={'material': material, 'thickness_mm': thickness}
        )
    
    def generate_ftir_spectrum(self,
                              sample_type: str = 'SiO2_on_Si',
                              wavenumber_range: Tuple[float, float] = (400, 4000),
                              n_points: int = 1000) -> SpectralData:
        """Generate synthetic FTIR spectrum"""
        
        # Generate wavenumber array
        wavenumber = np.linspace(wavenumber_range[0], wavenumber_range[1], n_points)
        
        # Initialize spectrum
        intensity = np.ones_like(wavenumber) * 90  # Base transmittance
        
        # Add peaks based on sample type
        if sample_type == 'SiO2_on_Si':
            # Si-O stretching
            self._add_peak(wavenumber, intensity, 1080, 100, 30, 'absorption')
            # Si-O bending
            self._add_peak(wavenumber, intensity, 460, 50, 20, 'absorption')
            # Si-O rocking
            self._add_peak(wavenumber, intensity, 800, 60, 40, 'absorption')
            
        elif sample_type == 'Si3N4_on_Si':
            # Si-N stretching
            self._add_peak(wavenumber, intensity, 850, 80, 50, 'absorption')
            # Si-H stretching
            self._add_peak(wavenumber, intensity, 2160, 40, 30, 'absorption')
            # N-H stretching
            self._add_peak(wavenumber, intensity, 3340, 30, 80, 'absorption')
            
        elif sample_type == 'organic_contamination':
            # C-H stretching
            self._add_peak(wavenumber, intensity, 2920, 20, 40, 'absorption')
            self._add_peak(wavenumber, intensity, 2850, 15, 35, 'absorption')
            # C=O stretching
            self._add_peak(wavenumber, intensity, 1730, 10, 25, 'absorption')
            # O-H stretching (moisture)
            self._add_peak(wavenumber, intensity, 3400, 25, 200, 'absorption')
        
        # Add interference fringes for thin films
        if 'on_Si' in sample_type:
            thickness = 1.0  # µm
            n = 1.46  # SiO2
            period = 1 / (2 * n * thickness / 10000)  # in cm⁻¹
            fringes = 5 * np.sin(2 * np.pi * wavenumber / period)
            intensity = intensity + fringes
        
        # Add baseline drift
        baseline = 0.01 * (wavenumber - wavenumber[0])
        intensity = intensity + baseline
        
        # Add noise
        noise = np.random.normal(0, 0.5, len(intensity))
        intensity = intensity + noise
        
        # Ensure positive values
        intensity = np.maximum(intensity, 0)
        
        return SpectralData(
            wavelength=wavenumber,  # Using wavelength field for wavenumber
            intensity=intensity,
            measurement_type=MeasurementType.TRANSMISSION,
            metadata={'sample_type': sample_type, 'units': 'wavenumber_cm-1'}
        )
    
    def _add_peak(self, x: np.ndarray, y: np.ndarray,
                 position: float, amplitude: float, 
                 width: float, peak_type: str = 'absorption'):
        """Add a peak to spectrum"""
        # Lorentzian peak
        peak = amplitude * width**2 / ((x - position)**2 + width**2)
        
        if peak_type == 'absorption':
            y -= peak
        else:
            y += peak
            
    def generate_tauc_test_data(self, 
                               true_bandgap: float = 1.42,
                               bandgap_type: BandgapType = BandgapType.DIRECT_ALLOWED) -> Dict:
        """Generate test data for Tauc analysis validation"""
        
        # Generate transmission spectrum
        spectrum = self.generate_uv_vis_spectrum('GaAs')
        
        # Perform Tauc analysis
        analyzer = UVVisNIRAnalyzer()
        result = analyzer.tauc_analysis(
            spectrum,
            thickness=0.5,
            bandgap_type=bandgap_type
        )
        
        return {
            'spectrum': spectrum,
            'tauc_result': result,
            'true_bandgap': true_bandgap,
            'measured_bandgap': result.bandgap,
            'error': np.abs(result.bandgap - true_bandgap)
        }


class Session7IntegrationTest:
    """
    Integration tests for Session 7 optical methods
    """
    
    def __init__(self):
        self.uv_vis_analyzer = UVVisNIRAnalyzer()
        self.ftir_analyzer = FTIRAnalyzer()
        self.generator = OpticalTestDataGenerator()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        results = {
            'uv_vis_tests': self.test_uv_vis_analysis(),
            'ftir_tests': self.test_ftir_analysis(),
            'tauc_tests': self.test_tauc_analysis(),
            'performance_tests': self.test_performance(),
            'accuracy_tests': self.test_accuracy()
        }
        
        # Calculate overall metrics
        total_tests = sum(len(v) for v in results.values())
        passed_tests = sum(
            sum(1 for test in v.values() if test.get('passed', False))
            for v in results.values()
        )
        
        results['summary'] = {
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': total_tests - passed_tests,
            'pass_rate': passed_tests / total_tests * 100
        }
        
        return results
    
    def test_uv_vis_analysis(self) -> Dict[str, Any]:
        """Test UV-Vis-NIR analysis pipeline"""
        tests = {}
        
        # Test transmission to absorption conversion
        trans_data = self.generator.generate_uv_vis_spectrum(
            measurement_type=MeasurementType.TRANSMISSION
        )
        abs_data = self.uv_vis_analyzer.calculate_absorption(trans_data)
        
        tests['absorption_conversion'] = {
            'passed': abs_data.measurement_type == MeasurementType.ABSORPTION,
            'absorption_range': (np.min(abs_data.intensity), np.max(abs_data.intensity))
        }
        
        # Test baseline correction
        processed = self.uv_vis_analyzer.process_spectrum(abs_data)
        tests['baseline_correction'] = {
            'passed': 'baseline' in processed.metadata,
            'baseline_removed': np.mean(processed.intensity[:10]) < np.mean(abs_data.intensity[:10])
        }
        
        return tests
    
    def test_ftir_analysis(self) -> Dict[str, Any]:
        """Test FTIR analysis pipeline"""
        tests = {}
        
        # Generate test spectrum
        ftir_data = self.generator.generate_ftir_spectrum('SiO2_on_Si')
        
        # Test peak finding
        peaks = self.ftir_analyzer.find_peaks(ftir_data)
        tests['peak_finding'] = {
            'passed': len(peaks['positions']) > 0,
            'n_peaks': len(peaks['positions']),
            'identified': sum(1 for x in peaks['identifications'] if x is not None)
        }
        
        # Test peak fitting
        fit_result = self.ftir_analyzer.fit_peaks(ftir_data)
        tests['peak_fitting'] = {
            'passed': fit_result.r_squared > 0.8,
            'r_squared': fit_result.r_squared,
            'n_fitted_peaks': len(fit_result.peaks)
        }
        
        # Test thickness calculation
        thickness = self.ftir_analyzer.calculate_film_thickness(ftir_data)
        tests['thickness_calculation'] = {
            'passed': thickness['thickness'] > 0,
            'thickness_um': thickness['thickness'],
            'error_um': thickness['error']
        }
        
        return tests
    
    def test_tauc_analysis(self) -> Dict[str, Any]:
        """Test Tauc bandgap analysis"""
        tests = {}
        
        materials = ['GaAs', 'Si', 'GaN']
        expected_bandgaps = {'GaAs': 1.42, 'Si': 1.12, 'GaN': 3.4}
        
        for material in materials:
            spectrum = self.generator.generate_uv_vis_spectrum(material)
            result = self.uv_vis_analyzer.tauc_analysis(
                spectrum,
                thickness=0.5,
                bandgap_type=BandgapType.DIRECT_ALLOWED
            )
            
            error = np.abs(result.bandgap - expected_bandgaps[material])
            tests[f'tauc_{material}'] = {
                'passed': error < 0.1,  # Within 0.1 eV
                'expected': expected_bandgaps[material],
                'measured': result.bandgap,
                'error': error,
                'r_squared': result.r_squared
            }
        
        return tests
    
    def test_performance(self) -> Dict[str, Any]:
        """Test performance metrics"""
        import time
        tests = {}
        
        # UV-Vis processing time
        spectrum = self.generator.generate_uv_vis_spectrum()
        start = time.time()
        self.uv_vis_analyzer.process_spectrum(spectrum)
        uv_time = time.time() - start
        
        tests['uv_vis_processing'] = {
            'passed': uv_time < 1.0,  # Should complete in < 1 second
            'time_seconds': uv_time
        }
        
        # FTIR processing time
        ftir = self.generator.generate_ftir_spectrum()
        start = time.time()
        self.ftir_analyzer.process_ftir_spectrum(ftir)
        self.ftir_analyzer.find_peaks(ftir)
        ftir_time = time.time() - start
        
        tests['ftir_processing'] = {
            'passed': ftir_time < 2.0,  # Should complete in < 2 seconds
            'time_seconds': ftir_time
        }
        
        return tests
    
    def test_accuracy(self) -> Dict[str, Any]:
        """Test measurement accuracy"""
        tests = {}
        
        # Test absorption coefficient calculation
        spectrum = self.generator.generate_uv_vis_spectrum()
        abs_data = self.uv_vis_analyzer.calculate_absorption(spectrum)
        alpha = self.uv_vis_analyzer.calculate_absorption_coefficient(abs_data, 0.5)
        
        tests['absorption_coefficient'] = {
            'passed': np.all(alpha >= 0),
            'range_cm-1': (np.min(alpha), np.max(alpha))
        }
        
        # Test bandgap accuracy across multiple materials
        errors = []
        for _ in range(5):
            test_data = self.generator.generate_tauc_test_data()
            errors.append(test_data['error'])
        
        mean_error = np.mean(errors)
        tests['bandgap_accuracy'] = {
            'passed': mean_error < 0.05,  # Average error < 50 meV
            'mean_error_eV': mean_error,
            'std_error_eV': np.std(errors)
        }
        
        return tests


def main():
    """
    Main execution and demonstration
    """
    print("=" * 80)
    print("Session 7: Optical I - UV-Vis-NIR and FTIR Analysis")
    print("=" * 80)
    
    # Initialize components
    generator = OpticalTestDataGenerator()
    uv_vis_analyzer = UVVisNIRAnalyzer()
    ftir_analyzer = FTIRAnalyzer()
    
    # Demo UV-Vis-NIR analysis
    print("\n1. UV-Vis-NIR Analysis Demo")
    print("-" * 40)
    
    # Generate test spectrum
    spectrum = generator.generate_uv_vis_spectrum('GaAs', MeasurementType.TRANSMISSION)
    print(f"Generated transmission spectrum: {len(spectrum.wavelength)} points")
    print(f"Wavelength range: {spectrum.wavelength[0]:.1f} - {spectrum.wavelength[-1]:.1f} nm")
    
    # Process and analyze
    processed = uv_vis_analyzer.process_spectrum(spectrum)
    tauc_result = uv_vis_analyzer.tauc_analysis(processed, thickness=0.5)
    
    print(f"Bandgap analysis:")
    print(f"  - Measured bandgap: {tauc_result.bandgap:.3f} ± {tauc_result.bandgap_error:.3f} eV")
    print(f"  - R² of linear fit: {tauc_result.r_squared:.4f}")
    print(f"  - Expected (GaAs): 1.42 eV")
    
    # Demo FTIR analysis
    print("\n2. FTIR Analysis Demo")
    print("-" * 40)
    
    # Generate test spectrum
    ftir_spectrum = generator.generate_ftir_spectrum('SiO2_on_Si')
    print(f"Generated FTIR spectrum: {len(ftir_spectrum.wavelength)} points")
    print(f"Wavenumber range: {ftir_spectrum.wavelength[0]:.1f} - {ftir_spectrum.wavelength[-1]:.1f} cm⁻¹")
    
    # Process and analyze
    processed_ftir = ftir_analyzer.process_ftir_spectrum(ftir_spectrum)
    peaks = ftir_analyzer.find_peaks(processed_ftir)
    
    print(f"Peak analysis:")
    print(f"  - Peaks found: {len(peaks['positions'])}")
    print(f"  - Identified peaks: {sum(1 for x in peaks['identifications'] if x is not None)}")
    
    if peaks['identifications']:
        print(f"  - Peak assignments:")
        for pos, ident in zip(peaks['positions'][:5], peaks['identifications'][:5]):
            if ident:
                print(f"    • {pos:.1f} cm⁻¹: {ident}")
    
    # Film thickness calculation
    thickness_result = ftir_analyzer.calculate_film_thickness(ftir_spectrum)
    print(f"\nFilm thickness from interference fringes:")
    print(f"  - Thickness: {thickness_result['thickness']:.2f} ± {thickness_result['error']:.2f} µm")
    print(f"  - Number of fringes: {thickness_result['n_fringes']}")
    
    # Run integration tests
    print("\n3. Running Integration Tests")
    print("-" * 40)
    
    tester = Session7IntegrationTest()
    test_results = tester.run_all_tests()
    
    print(f"Test Results:")
    print(f"  - Total tests: {test_results['summary']['total_tests']}")
    print(f"  - Passed: {test_results['summary']['passed']}")
    print(f"  - Failed: {test_results['summary']['failed']}")
    print(f"  - Pass rate: {test_results['summary']['pass_rate']:.1f}%")
    
    # Performance metrics
    print("\n4. Performance Metrics")
    print("-" * 40)
    
    for category, tests in test_results.items():
        if category != 'summary':
            print(f"\n{category.replace('_', ' ').title()}:")
            for test_name, result in tests.items():
                if isinstance(result, dict) and 'passed' in result:
                    status = "✓" if result['passed'] else "✗"
                    print(f"  {status} {test_name}: {result.get('passed', False)}")
    
    print("\n" + "=" * 80)
    print("Session 7 Implementation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
