"""
Session 9: X-ray Diffraction (XRD) Analysis - Complete Implementation
=====================================================================
Production-ready implementation of XRD analysis for semiconductor characterization
Including phase identification, crystallite size, texture, and stress analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import scipy.signal as signal
import scipy.optimize as optimize
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.special import voigt_profile
import warnings
import json
import hashlib
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Physical constants
h = 6.62607015e-34      # Planck constant (J⋅s)
c = 299792458           # Speed of light (m/s)
e = 1.602176634e-19     # Elementary charge (C)
m_e = 9.10938356e-31    # Electron mass (kg)
k_B = 1.380649e-23      # Boltzmann constant (J/K)
r_e = 2.8179403262e-15  # Classical electron radius (m)
N_A = 6.02214076e23     # Avogadro's number

# X-ray wavelengths (Å)
XRAY_WAVELENGTHS = {
    'Cu_Ka1': 1.5405980,
    'Cu_Ka2': 1.5444260,
    'Cu_Ka': 1.5418,      # Weighted average
    'Cu_Kb': 1.3922,
    'Co_Ka': 1.7902,
    'Co_Kb': 1.6208,
    'Mo_Ka': 0.7107,
    'Mo_Kb': 0.6323,
    'Cr_Ka': 2.2909,
    'Fe_Ka': 1.9373,
    'Ag_Ka': 0.5609
}

class CrystalSystem(Enum):
    """Crystal systems"""
    CUBIC = "cubic"
    TETRAGONAL = "tetragonal"
    ORTHORHOMBIC = "orthorhombic"
    HEXAGONAL = "hexagonal"
    TRIGONAL = "trigonal"
    MONOCLINIC = "monoclinic"
    TRICLINIC = "triclinic"

class PeakProfile(Enum):
    """Peak profile functions"""
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    PSEUDO_VOIGT = "pseudo_voigt"
    PEARSON_VII = "pearson_vii"

class TextureModel(Enum):
    """Texture models"""
    MARCH_DOLLASE = "march_dollase"
    SPHERICAL_HARMONICS = "spherical_harmonics"
    POLE_FIGURE = "pole_figure"

@dataclass
class XRDPattern:
    """Container for XRD measurement data"""
    two_theta: np.ndarray       # 2θ angles (degrees)
    intensity: np.ndarray       # Intensity (counts)
    wavelength: float          # X-ray wavelength (Å)
    scan_speed: float = 1.0    # degrees/min
    step_size: float = 0.02    # degrees
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate and sort data"""
        if len(self.two_theta) != len(self.intensity):
            raise ValueError("2θ and intensity arrays must have same length")
        
        # Sort by 2θ
        sort_idx = np.argsort(self.two_theta)
        self.two_theta = self.two_theta[sort_idx]
        self.intensity = self.intensity[sort_idx]
        
        # Remove negative intensities
        self.intensity = np.maximum(self.intensity, 0)

@dataclass
class CrystalStructure:
    """Crystal structure information"""
    name: str
    formula: str
    crystal_system: CrystalSystem
    space_group: str
    lattice_params: Dict[str, float]  # a, b, c, alpha, beta, gamma
    atoms: List[Dict[str, Any]]       # Atomic positions
    
    def get_unit_cell_volume(self) -> float:
        """Calculate unit cell volume"""
        a = self.lattice_params['a']
        b = self.lattice_params.get('b', a)
        c = self.lattice_params.get('c', a)
        alpha = np.radians(self.lattice_params.get('alpha', 90))
        beta = np.radians(self.lattice_params.get('beta', 90))
        gamma = np.radians(self.lattice_params.get('gamma', 90))
        
        # General formula for unit cell volume
        volume = a * b * c * np.sqrt(
            1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 +
            2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)
        )
        return volume

@dataclass
class Peak:
    """XRD peak information"""
    position: float         # 2θ position (degrees)
    d_spacing: float       # d-spacing (Å)
    intensity: float       # Peak intensity
    fwhm: float           # Full width at half maximum (degrees)
    area: float           # Integrated intensity
    hkl: Optional[Tuple[int, int, int]] = None  # Miller indices
    phase: Optional[str] = None

@dataclass
class PhaseIdentification:
    """Phase identification result"""
    phase_name: str
    formula: str
    crystal_system: str
    score: float           # Match score (0-100)
    matched_peaks: List[Peak]
    reference_peaks: List[Dict]
    lattice_params: Dict[str, float]


class XRDAnalyzer:
    """
    Comprehensive XRD Analysis System
    Handles phase identification, peak fitting, crystallite size, texture, and stress
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_phase_database()
        
    def _load_phase_database(self):
        """Load crystallographic database (simplified version)"""
        self.phase_database = {
            'Si': CrystalStructure(
                name='Silicon',
                formula='Si',
                crystal_system=CrystalSystem.CUBIC,
                space_group='Fd-3m',
                lattice_params={'a': 5.43095},
                atoms=[
                    {'element': 'Si', 'position': [0, 0, 0]},
                    {'element': 'Si', 'position': [0.25, 0.25, 0.25]}
                ]
            ),
            'GaAs': CrystalStructure(
                name='Gallium Arsenide',
                formula='GaAs',
                crystal_system=CrystalSystem.CUBIC,
                space_group='F-43m',
                lattice_params={'a': 5.65325},
                atoms=[
                    {'element': 'Ga', 'position': [0, 0, 0]},
                    {'element': 'As', 'position': [0.25, 0.25, 0.25]}
                ]
            ),
            'GaN_hex': CrystalStructure(
                name='Gallium Nitride (Wurtzite)',
                formula='GaN',
                crystal_system=CrystalSystem.HEXAGONAL,
                space_group='P63mc',
                lattice_params={'a': 3.189, 'c': 5.185},
                atoms=[
                    {'element': 'Ga', 'position': [0.333, 0.667, 0]},
                    {'element': 'N', 'position': [0.333, 0.667, 0.375]}
                ]
            ),
            'SiO2_quartz': CrystalStructure(
                name='Quartz',
                formula='SiO2',
                crystal_system=CrystalSystem.HEXAGONAL,
                space_group='P3221',
                lattice_params={'a': 4.9133, 'c': 5.4053},
                atoms=[
                    {'element': 'Si', 'position': [0.4697, 0, 0.333]},
                    {'element': 'O', 'position': [0.4135, 0.2669, 0.2144]}
                ]
            ),
            'Al2O3': CrystalStructure(
                name='Alumina (Corundum)',
                formula='Al2O3',
                crystal_system=CrystalSystem.TRIGONAL,
                space_group='R-3c',
                lattice_params={'a': 4.759, 'c': 12.993},
                atoms=[
                    {'element': 'Al', 'position': [0, 0, 0.35216]},
                    {'element': 'O', 'position': [0.30624, 0, 0.25]}
                ]
            ),
            'TiO2_anatase': CrystalStructure(
                name='Anatase',
                formula='TiO2',
                crystal_system=CrystalSystem.TETRAGONAL,
                space_group='I41/amd',
                lattice_params={'a': 3.785, 'c': 9.514},
                atoms=[
                    {'element': 'Ti', 'position': [0, 0, 0]},
                    {'element': 'O', 'position': [0, 0, 0.208]}
                ]
            )
        }
        
        # Pre-calculate reference patterns
        self.reference_patterns = {}
        for phase_name, structure in self.phase_database.items():
            self.reference_patterns[phase_name] = self._calculate_reference_peaks(
                structure, wavelength=1.5418  # Cu Kα
            )
    
    def process_pattern(self, pattern: XRDPattern,
                       smooth: bool = True,
                       background_correct: bool = True) -> XRDPattern:
        """
        Process XRD pattern with smoothing and background correction
        """
        processed = XRDPattern(
            two_theta=pattern.two_theta.copy(),
            intensity=pattern.intensity.copy(),
            wavelength=pattern.wavelength,
            scan_speed=pattern.scan_speed,
            step_size=pattern.step_size,
            metadata=pattern.metadata.copy()
        )
        
        # Smoothing
        if smooth:
            window_size = max(3, int(0.1 / pattern.step_size))
            if window_size % 2 == 0:
                window_size += 1
            processed.intensity = signal.savgol_filter(
                processed.intensity, window_size, polyorder=2
            )
        
        # Background correction
        if background_correct:
            background = self._calculate_background(
                processed.two_theta, processed.intensity
            )
            processed.intensity = processed.intensity - background
            processed.intensity = np.maximum(processed.intensity, 0)
        
        return processed
    
    def find_peaks(self, pattern: XRDPattern,
                  prominence: float = 0.05,
                  min_height: Optional[float] = None,
                  min_distance: int = 5) -> List[Peak]:
        """
        Find and characterize peaks in XRD pattern
        """
        # Normalize for peak finding
        norm_intensity = pattern.intensity / np.max(pattern.intensity)
        
        if min_height is None:
            min_height = 0.05 * np.max(norm_intensity)
        
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            norm_intensity,
            prominence=prominence,
            height=min_height,
            distance=min_distance
        )
        
        peaks = []
        for idx in peak_indices:
            # Get peak position and intensity
            position = pattern.two_theta[idx]
            intensity = pattern.intensity[idx]
            
            # Calculate d-spacing from Bragg's law
            theta_rad = np.radians(position / 2)
            d_spacing = pattern.wavelength / (2 * np.sin(theta_rad))
            
            # Estimate FWHM
            widths = signal.peak_widths(
                pattern.intensity, [idx], rel_height=0.5
            )
            fwhm = widths[0][0] * pattern.step_size
            
            # Calculate area (simple trapezoidal)
            left_idx = max(0, idx - int(widths[0][0] / 2))
            right_idx = min(len(pattern.intensity), idx + int(widths[0][0] / 2))
            area = np.trapz(
                pattern.intensity[left_idx:right_idx],
                pattern.two_theta[left_idx:right_idx]
            )
            
            peaks.append(Peak(
                position=position,
                d_spacing=d_spacing,
                intensity=intensity,
                fwhm=fwhm,
                area=area
            ))
        
        return peaks
    
    def fit_peaks(self, pattern: XRDPattern,
                 peaks: Optional[List[Peak]] = None,
                 profile: PeakProfile = PeakProfile.PSEUDO_VOIGT) -> Dict[str, Any]:
        """
        Fit peaks with specified profile function
        """
        if peaks is None:
            peaks = self.find_peaks(pattern)
        
        if len(peaks) == 0:
            return {'peaks': [], 'fitted_pattern': pattern.intensity}
        
        # Build initial parameters
        params = []
        for peak in peaks:
            params.extend([
                peak.intensity,  # Amplitude
                peak.position,   # Center
                peak.fwhm / 2    # Width (sigma for Gaussian)
            ])
        
        # Add background parameters
        params.extend([0, np.min(pattern.intensity)])  # Linear background
        
        # Select fitting function
        if profile == PeakProfile.GAUSSIAN:
            fit_func = self._multi_gaussian
        elif profile == PeakProfile.LORENTZIAN:
            fit_func = self._multi_lorentzian
        elif profile == PeakProfile.PSEUDO_VOIGT:
            fit_func = self._multi_pseudo_voigt
            # Add mixing parameter for each peak
            for _ in peaks:
                params.insert(len(params) - 2, 0.5)  # eta = 0.5
        else:
            fit_func = self._multi_voigt
        
        # Optimize
        try:
            if profile == PeakProfile.PSEUDO_VOIGT:
                # Special handling for pseudo-Voigt
                popt, pcov = optimize.curve_fit(
                    lambda x, *p: self._multi_pseudo_voigt(x, p),
                    pattern.two_theta, pattern.intensity,
                    p0=params, maxfev=5000
                )
            else:
                popt, pcov = optimize.curve_fit(
                    lambda x, *p: fit_func(x, p),
                    pattern.two_theta, pattern.intensity,
                    p0=params, maxfev=5000
                )
        except:
            # Return original if fitting fails
            return {'peaks': peaks, 'fitted_pattern': pattern.intensity}
        
        # Extract fitted peaks
        fitted_peaks = []
        n_params_per_peak = 3 if profile != PeakProfile.PSEUDO_VOIGT else 4
        n_peaks = len(peaks)
        
        for i in range(n_peaks):
            base_idx = i * n_params_per_peak
            
            if profile == PeakProfile.PSEUDO_VOIGT:
                amplitude = popt[base_idx]
                center = popt[base_idx + 1]
                width = popt[base_idx + 2]
                eta = popt[base_idx + 3]
            else:
                amplitude = popt[base_idx]
                center = popt[base_idx + 1]
                width = popt[base_idx + 2]
                eta = None
            
            # Calculate d-spacing
            theta_rad = np.radians(center / 2)
            d_spacing = pattern.wavelength / (2 * np.sin(theta_rad))
            
            # Calculate FWHM
            if profile == PeakProfile.GAUSSIAN:
                fwhm = 2 * width * np.sqrt(2 * np.log(2))
            elif profile == PeakProfile.LORENTZIAN:
                fwhm = 2 * width
            else:
                fwhm = 2 * width  # Approximate
            
            # Calculate area
            if profile == PeakProfile.GAUSSIAN:
                area = amplitude * width * np.sqrt(2 * np.pi)
            elif profile == PeakProfile.LORENTZIAN:
                area = amplitude * width * np.pi
            else:
                area = amplitude * width * np.pi  # Approximate
            
            fitted_peaks.append({
                'position': center,
                'd_spacing': d_spacing,
                'intensity': amplitude,
                'fwhm': fwhm,
                'area': area,
                'width': width,
                'eta': eta
            })
        
        # Generate fitted pattern
        if profile == PeakProfile.PSEUDO_VOIGT:
            fitted_pattern = self._multi_pseudo_voigt(pattern.two_theta, popt)
        else:
            fitted_pattern = fit_func(pattern.two_theta, popt)
        
        # Calculate R-factors
        r_wp = self._calculate_rwp(pattern.intensity, fitted_pattern)
        r_exp = self._calculate_rexp(pattern.intensity)
        chi_squared = (r_wp / r_exp) ** 2 if r_exp > 0 else np.inf
        
        return {
            'peaks': fitted_peaks,
            'fitted_pattern': fitted_pattern,
            'background': popt[-2] * pattern.two_theta + popt[-1],
            'r_wp': r_wp,
            'r_exp': r_exp,
            'chi_squared': chi_squared,
            'profile': profile.value
        }
    
    def identify_phases(self, pattern: XRDPattern,
                       peaks: Optional[List[Peak]] = None,
                       tolerance: float = 0.1) -> List[PhaseIdentification]:
        """
        Identify crystalline phases by matching peak positions
        """
        if peaks is None:
            peaks = self.find_peaks(pattern)
        
        if len(peaks) == 0:
            return []
        
        identifications = []
        
        for phase_name, ref_peaks in self.reference_patterns.items():
            # Match peaks
            matched_peaks = []
            matched_ref = []
            score = 0
            
            for peak in peaks:
                for ref_peak in ref_peaks:
                    if abs(peak.position - ref_peak['position']) < tolerance:
                        matched_peaks.append(peak)
                        matched_ref.append(ref_peak)
                        # Score based on position match and intensity
                        position_score = 1 - abs(peak.position - ref_peak['position']) / tolerance
                        intensity_score = min(peak.intensity, ref_peak['intensity']) / \
                                        max(peak.intensity, ref_peak['intensity'])
                        score += position_score * intensity_score
                        break
            
            if len(matched_peaks) > 0:
                # Normalize score
                score = 100 * score / len(ref_peaks)
                
                # Get structure info
                structure = self.phase_database[phase_name]
                
                identifications.append(PhaseIdentification(
                    phase_name=structure.name,
                    formula=structure.formula,
                    crystal_system=structure.crystal_system.value,
                    score=score,
                    matched_peaks=matched_peaks,
                    reference_peaks=matched_ref,
                    lattice_params=structure.lattice_params
                ))
        
        # Sort by score
        identifications.sort(key=lambda x: x.score, reverse=True)
        
        return identifications
    
    def calculate_crystallite_size(self, peaks: List[Peak],
                                  wavelength: float,
                                  shape_factor: float = 0.9) -> Dict[str, float]:
        """
        Calculate crystallite size using Scherrer equation
        D = K·λ / (β·cos(θ))
        """
        if len(peaks) == 0:
            return {'mean_size': 0, 'std_size': 0}
        
        sizes = []
        
        for peak in peaks:
            # Convert FWHM to radians
            beta_rad = np.radians(peak.fwhm)
            
            # Bragg angle
            theta_rad = np.radians(peak.position / 2)
            
            # Scherrer equation
            size = shape_factor * wavelength / (beta_rad * np.cos(theta_rad))
            
            # Convert from Å to nm
            size_nm = size / 10
            
            # Physical limit check
            if 0.5 < size_nm < 1000:  # Reasonable range
                sizes.append(size_nm)
        
        if len(sizes) == 0:
            return {'mean_size': 0, 'std_size': 0, 'sizes': []}
        
        return {
            'mean_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'min_size': np.min(sizes),
            'max_size': np.max(sizes),
            'sizes': sizes
        }
    
    def calculate_strain(self, peaks: List[Peak],
                        reference_peaks: List[Dict],
                        young_modulus: float = 169) -> Dict[str, float]:
        """
        Calculate microstrain from peak shifts
        ε = Δd/d = -Δ(2θ)·cot(θ)
        """
        if len(peaks) == 0 or len(reference_peaks) == 0:
            return {'mean_strain': 0, 'mean_stress': 0}
        
        strains = []
        
        for peak in peaks:
            # Find matching reference peak
            ref_peak = None
            for ref in reference_peaks:
                if abs(peak.position - ref['position']) < 0.5:
                    ref_peak = ref
                    break
            
            if ref_peak:
                # Calculate strain
                delta_2theta = peak.position - ref_peak['position']
                theta_rad = np.radians(peak.position / 2)
                
                if np.tan(theta_rad) != 0:
                    strain = -delta_2theta * np.cos(theta_rad) / np.sin(theta_rad) / 2
                    strains.append(strain)
        
        if len(strains) == 0:
            return {'mean_strain': 0, 'mean_stress': 0}
        
        mean_strain = np.mean(strains)
        mean_stress = mean_strain * young_modulus  # GPa
        
        return {
            'mean_strain': mean_strain,
            'std_strain': np.std(strains),
            'mean_stress': mean_stress,
            'type': 'tensile' if mean_strain > 0 else 'compressive'
        }
    
    def williamson_hall_analysis(self, peaks: List[Peak],
                                wavelength: float) -> Dict[str, float]:
        """
        Williamson-Hall analysis to separate size and strain broadening
        β·cos(θ) = K·λ/D + 4·ε·sin(θ)
        """
        if len(peaks) < 2:
            return {'crystallite_size': 0, 'microstrain': 0}
        
        # Prepare data for linear fit
        sin_theta = []
        beta_cos_theta = []
        
        for peak in peaks:
            theta_rad = np.radians(peak.position / 2)
            beta_rad = np.radians(peak.fwhm)
            
            sin_theta.append(np.sin(theta_rad))
            beta_cos_theta.append(beta_rad * np.cos(theta_rad))
        
        # Linear fit
        sin_theta = np.array(sin_theta)
        beta_cos_theta = np.array(beta_cos_theta)
        
        # y = mx + c, where y = β·cos(θ), x = sin(θ)
        # Slope m = 4ε, Intercept c = Kλ/D
        coeffs = np.polyfit(sin_theta, beta_cos_theta, 1)
        slope = coeffs[0]
        intercept = coeffs[1]
        
        # Extract parameters
        microstrain = slope / 4
        
        if intercept > 0:
            crystallite_size = 0.9 * wavelength / intercept / 10  # nm
        else:
            crystallite_size = np.inf
        
        # Calculate R²
        predicted = np.polyval(coeffs, sin_theta)
        ss_res = np.sum((beta_cos_theta - predicted) ** 2)
        ss_tot = np.sum((beta_cos_theta - np.mean(beta_cos_theta)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'crystallite_size': crystallite_size,
            'microstrain': microstrain,
            'strain_percent': microstrain * 100,
            'r_squared': r_squared,
            'slope': slope,
            'intercept': intercept
        }
    
    def calculate_texture(self, pattern: XRDPattern,
                         peaks: List[Peak],
                         reference_intensities: List[float]) -> Dict[str, Any]:
        """
        Calculate texture coefficients and preferred orientation
        TC(hkl) = [I(hkl)/I₀(hkl)] / [(1/n)·Σ(I(hkl)/I₀(hkl))]
        """
        if len(peaks) != len(reference_intensities):
            return {'texture_coefficients': [], 'texture_index': 0}
        
        # Calculate intensity ratios
        ratios = []
        for peak, ref_int in zip(peaks, reference_intensities):
            if ref_int > 0:
                ratio = peak.intensity / ref_int
                ratios.append(ratio)
            else:
                ratios.append(0)
        
        # Calculate texture coefficients
        mean_ratio = np.mean(ratios) if len(ratios) > 0 else 1
        texture_coeffs = []
        
        for ratio in ratios:
            if mean_ratio > 0:
                tc = ratio / mean_ratio
            else:
                tc = 1
            texture_coeffs.append(tc)
        
        # Calculate texture index (degree of preferred orientation)
        # σ = sqrt(Σ(TC - 1)²/n)
        texture_index = np.sqrt(
            np.mean([(tc - 1) ** 2 for tc in texture_coeffs])
        )
        
        # Identify preferred orientation
        max_tc_idx = np.argmax(texture_coeffs)
        preferred_peak = peaks[max_tc_idx] if len(peaks) > 0 else None
        
        return {
            'texture_coefficients': texture_coeffs,
            'texture_index': texture_index,
            'preferred_orientation': preferred_peak,
            'is_textured': texture_index > 0.1,
            'orientation_factor': max(texture_coeffs) if texture_coeffs else 1
        }
    
    def residual_stress_sin2psi(self, measurements: List[Tuple[float, float]],
                              d0: float, young_modulus: float = 169,
                              poisson_ratio: float = 0.22) -> Dict[str, float]:
        """
        Calculate residual stress using sin²ψ method
        ε_ψ = [(1+ν)/E]·σ·sin²ψ - (ν/E)·σ
        """
        if len(measurements) < 2:
            return {'stress': 0, 'error': np.inf}
        
        # Extract data
        psi_values = []
        d_values = []
        
        for psi, d_spacing in measurements:
            psi_values.append(psi)
            d_values.append(d_spacing)
        
        psi_values = np.array(psi_values)
        d_values = np.array(d_values)
        
        # Calculate strain
        strains = (d_values - d0) / d0
        
        # Calculate sin²ψ
        sin2_psi = np.sin(np.radians(psi_values)) ** 2
        
        # Linear fit: ε = m·sin²ψ + c
        coeffs = np.polyfit(sin2_psi, strains, 1)
        slope = coeffs[0]
        
        # Calculate stress
        # slope = (1+ν)/E · σ
        stress = slope * young_modulus / (1 + poisson_ratio)  # GPa
        
        # Calculate error
        predicted = np.polyval(coeffs, sin2_psi)
        residuals = strains - predicted
        error = np.std(residuals)
        
        # R²
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((strains - np.mean(strains)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'stress': stress,
            'stress_mpa': stress * 1000,  # Convert to MPa
            'type': 'tensile' if stress > 0 else 'compressive',
            'error': error,
            'r_squared': r_squared,
            'slope': slope,
            'measurements': len(measurements)
        }
    
    def rietveld_refinement_simplified(self, pattern: XRDPattern,
                                      structure: CrystalStructure,
                                      initial_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Simplified Rietveld refinement (demonstration version)
        Full implementation would require more sophisticated algorithms
        """
        # Calculate theoretical pattern
        calc_peaks = self._calculate_reference_peaks(structure, pattern.wavelength)
        
        # Build calculated pattern
        calc_pattern = np.zeros_like(pattern.intensity)
        
        for peak in calc_peaks:
            # Add Gaussian peak
            amplitude = peak['intensity'] * np.max(pattern.intensity)
            center = peak['position']
            sigma = 0.1  # Default width
            
            calc_pattern += amplitude * np.exp(
                -0.5 * ((pattern.two_theta - center) / sigma) ** 2
            )
        
        # Scale factor optimization
        scale = np.sum(pattern.intensity * calc_pattern) / np.sum(calc_pattern ** 2)
        calc_pattern *= scale
        
        # Calculate R-factors
        r_wp = self._calculate_rwp(pattern.intensity, calc_pattern)
        r_exp = self._calculate_rexp(pattern.intensity)
        chi_squared = (r_wp / r_exp) ** 2 if r_exp > 0 else np.inf
        
        return {
            'calculated_pattern': calc_pattern,
            'scale_factor': scale,
            'r_wp': r_wp,
            'r_exp': r_exp,
            'chi_squared': chi_squared,
            'converged': chi_squared < 2,
            'refined_params': {
                'lattice_a': structure.lattice_params.get('a', 0),
                'scale': scale
            }
        }
    
    def _calculate_reference_peaks(self, structure: CrystalStructure,
                                  wavelength: float,
                                  max_2theta: float = 90) -> List[Dict]:
        """
        Calculate theoretical peak positions for a crystal structure
        """
        peaks = []
        
        # Get lattice parameters
        a = structure.lattice_params.get('a', 5.0)
        b = structure.lattice_params.get('b', a)
        c = structure.lattice_params.get('c', a)
        
        # Generate hkl indices (simplified - should use systematic absences)
        max_h = int(2 * a / wavelength) + 1
        
        for h in range(-max_h, max_h + 1):
            for k in range(-max_h, max_h + 1):
                for l in range(-max_h, max_h + 1):
                    if h == 0 and k == 0 and l == 0:
                        continue
                    
                    # Calculate d-spacing based on crystal system
                    if structure.crystal_system == CrystalSystem.CUBIC:
                        d = a / np.sqrt(h**2 + k**2 + l**2)
                    elif structure.crystal_system == CrystalSystem.TETRAGONAL:
                        d = 1 / np.sqrt((h**2 + k**2) / a**2 + l**2 / c**2)
                    elif structure.crystal_system == CrystalSystem.HEXAGONAL:
                        d = 1 / np.sqrt(4 * (h**2 + h*k + k**2) / (3 * a**2) + l**2 / c**2)
                    else:
                        # Default to cubic
                        d = a / np.sqrt(h**2 + k**2 + l**2)
                    
                    # Calculate 2θ
                    if d > wavelength / 2:
                        sin_theta = wavelength / (2 * d)
                        if sin_theta <= 1:
                            theta = np.arcsin(sin_theta)
                            two_theta = 2 * np.degrees(theta)
                            
                            if two_theta <= max_2theta:
                                # Calculate structure factor (simplified)
                                intensity = self._calculate_structure_factor(
                                    structure, (h, k, l)
                                )
                                
                                if intensity > 0.01:  # Threshold
                                    peaks.append({
                                        'position': two_theta,
                                        'd_spacing': d,
                                        'intensity': intensity,
                                        'hkl': (h, k, l)
                                    })
        
        # Sort by 2θ and normalize intensities
        peaks.sort(key=lambda x: x['position'])
        
        if peaks:
            max_int = max(p['intensity'] for p in peaks)
            for peak in peaks:
                peak['intensity'] /= max_int
        
        # Keep only significant peaks
        peaks = [p for p in peaks if p['intensity'] > 0.01]
        
        return peaks[:20]  # Limit to first 20 peaks
    
    def _calculate_structure_factor(self, structure: CrystalStructure,
                                   hkl: Tuple[int, int, int]) -> float:
        """
        Calculate structure factor (simplified)
        F_hkl = Σ f_j · exp(2πi(h·x_j + k·y_j + l·z_j))
        """
        h, k, l = hkl
        
        # Simplified atomic scattering factors
        scattering_factors = {
            'Si': 14, 'Ga': 31, 'As': 33, 'N': 7,
            'O': 8, 'Al': 13, 'Ti': 22, 'C': 6
        }
        
        f_real = 0
        f_imag = 0
        
        for atom in structure.atoms:
            element = atom['element']
            pos = atom['position']
            
            # Get scattering factor (simplified - should be Q-dependent)
            f = scattering_factors.get(element, 10)
            
            # Calculate phase
            phase = 2 * np.pi * (h * pos[0] + k * pos[1] + l * pos[2])
            
            f_real += f * np.cos(phase)
            f_imag += f * np.sin(phase)
        
        # Return intensity (|F|²)
        return f_real**2 + f_imag**2
    
    def _calculate_background(self, two_theta: np.ndarray,
                            intensity: np.ndarray,
                            method: str = 'polynomial') -> np.ndarray:
        """
        Calculate background using various methods
        """
        if method == 'polynomial':
            # Fit polynomial to minima
            window = int(len(intensity) / 20)
            minima = []
            minima_pos = []
            
            for i in range(0, len(intensity), window):
                end = min(i + window, len(intensity))
                min_idx = i + np.argmin(intensity[i:end])
                minima.append(intensity[min_idx])
                minima_pos.append(two_theta[min_idx])
            
            # Fit polynomial
            if len(minima) > 3:
                coeffs = np.polyfit(minima_pos, minima, 3)
                background = np.polyval(coeffs, two_theta)
            else:
                background = np.ones_like(intensity) * np.min(intensity)
        
        elif method == 'linear':
            # Simple linear background
            slope = (intensity[-1] - intensity[0]) / (two_theta[-1] - two_theta[0])
            background = intensity[0] + slope * (two_theta - two_theta[0])
        
        else:
            # Constant background
            background = np.ones_like(intensity) * np.min(intensity)
        
        return background
    
    def _multi_gaussian(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Multiple Gaussian peaks with linear background"""
        n_peaks = (len(params) - 2) // 3
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i*3]
            center = params[i*3 + 1]
            sigma = params[i*3 + 2]
            y += amp * np.exp(-0.5 * ((x - center) / sigma)**2)
        
        # Add background
        y += params[-2] * x + params[-1]
        return y
    
    def _multi_lorentzian(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Multiple Lorentzian peaks with linear background"""
        n_peaks = (len(params) - 2) // 3
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i*3]
            center = params[i*3 + 1]
            gamma = params[i*3 + 2]
            y += amp * gamma**2 / ((x - center)**2 + gamma**2)
        
        # Add background
        y += params[-2] * x + params[-1]
        return y
    
    def _multi_voigt(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Multiple Voigt peaks (true Voigt profile)"""
        n_peaks = (len(params) - 2) // 3
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i*3]
            center = params[i*3 + 1]
            sigma = params[i*3 + 2]
            gamma = sigma * 0.5  # Relate to Gaussian width
            
            # Use scipy's Voigt profile
            y += amp * voigt_profile(x - center, sigma, gamma)
        
        # Add background
        y += params[-2] * x + params[-1]
        return y
    
    def _multi_pseudo_voigt(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Multiple pseudo-Voigt peaks
        PV = η·L + (1-η)·G
        """
        n_peaks = (len(params) - 2) // 4  # 4 params per peak
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i*4]
            center = params[i*4 + 1]
            width = params[i*4 + 2]
            eta = params[i*4 + 3]  # Lorentzian fraction
            
            # Gaussian component
            gaussian = amp * np.exp(-0.5 * ((x - center) / width)**2)
            
            # Lorentzian component
            lorentzian = amp * width**2 / ((x - center)**2 + width**2)
            
            # Mix
            y += eta * lorentzian + (1 - eta) * gaussian
        
        # Add background
        y += params[-2] * x + params[-1]
        return y
    
    def _calculate_rwp(self, observed: np.ndarray, calculated: np.ndarray) -> float:
        """
        Calculate weighted R-factor (R_wp)
        """
        weights = 1 / np.maximum(observed, 1)  # Avoid division by zero
        numerator = np.sum(weights * (observed - calculated)**2)
        denominator = np.sum(weights * observed**2)
        
        if denominator > 0:
            return 100 * np.sqrt(numerator / denominator)
        return np.inf
    
    def _calculate_rexp(self, observed: np.ndarray) -> float:
        """
        Calculate expected R-factor (R_exp)
        """
        n = len(observed)
        weights = 1 / np.maximum(observed, 1)
        denominator = np.sum(weights * observed**2)
        
        if denominator > 0:
            return 100 * np.sqrt(n / denominator)
        return np.inf


class XRDSimulator:
    """
    Generate synthetic XRD patterns for testing
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.analyzer = XRDAnalyzer()
    
    def generate_pattern(self, phase: str = 'Si',
                        wavelength: float = 1.5418,
                        crystallite_size: float = 100,  # nm
                        microstrain: float = 0.001,
                        texture_coefficient: float = 1.0,
                        noise_level: float = 0.05,
                        two_theta_range: Tuple[float, float] = (20, 80),
                        step_size: float = 0.02) -> XRDPattern:
        """
        Generate synthetic XRD pattern
        """
        # Get structure
        if phase not in self.analyzer.phase_database:
            phase = 'Si'  # Default
        
        structure = self.analyzer.phase_database[phase]
        
        # Generate 2θ array
        two_theta = np.arange(two_theta_range[0], two_theta_range[1], step_size)
        intensity = np.zeros_like(two_theta)
        
        # Calculate reference peaks
        ref_peaks = self.analyzer._calculate_reference_peaks(structure, wavelength)
        
        # Add peaks
        for peak in ref_peaks:
            if two_theta_range[0] <= peak['position'] <= two_theta_range[1]:
                # Peak position
                center = peak['position']
                
                # Add microstrain shift
                center += microstrain * center * np.random.randn()
                
                # Peak width from crystallite size and strain
                theta_rad = np.radians(center / 2)
                size_broadening = 0.9 * wavelength / (crystallite_size / 10) / np.cos(theta_rad)
                strain_broadening = 4 * microstrain * np.tan(theta_rad)
                beta = np.degrees(size_broadening + strain_broadening)
                
                # Convert to Gaussian width
                sigma = beta / 2.355
                
                # Peak intensity with texture
                amplitude = peak['intensity'] * 1000
                if peak == ref_peaks[0]:  # Modify first peak for texture
                    amplitude *= texture_coefficient
                
                # Add peak
                intensity += amplitude * np.exp(-0.5 * ((two_theta - center) / sigma)**2)
        
        # Add background
        background = 50 + 0.5 * two_theta + 10 * np.random.randn(len(two_theta))
        intensity += background
        
        # Add noise
        if noise_level > 0:
            noise = noise_level * np.max(intensity) * np.random.randn(len(intensity))
            intensity += noise
        
        # Ensure positive
        intensity = np.maximum(intensity, 0)
        
        # Add Poisson noise for realism
        intensity = np.random.poisson(intensity)
        
        return XRDPattern(
            two_theta=two_theta,
            intensity=intensity.astype(float),
            wavelength=wavelength,
            step_size=step_size,
            metadata={
                'phase': phase,
                'crystallite_size': crystallite_size,
                'microstrain': microstrain,
                'texture': texture_coefficient
            }
        )
    
    def generate_mixture(self, phases: List[Tuple[str, float]],
                       wavelength: float = 1.5418,
                       **kwargs) -> XRDPattern:
        """
        Generate pattern for mixture of phases
        phases: List of (phase_name, weight_fraction) tuples
        """
        # Normalize fractions
        total = sum(f for _, f in phases)
        phases = [(p, f/total) for p, f in phases]
        
        # Generate individual patterns
        patterns = []
        for phase_name, fraction in phases:
            pattern = self.generate_pattern(phase_name, wavelength, **kwargs)
            patterns.append((pattern, fraction))
        
        # Combine
        two_theta = patterns[0][0].two_theta
        intensity = np.zeros_like(two_theta)
        
        for pattern, fraction in patterns:
            # Interpolate if needed
            if not np.array_equal(pattern.two_theta, two_theta):
                f = interp1d(pattern.two_theta, pattern.intensity,
                           bounds_error=False, fill_value=0)
                pattern_intensity = f(two_theta)
            else:
                pattern_intensity = pattern.intensity
            
            intensity += fraction * pattern_intensity
        
        return XRDPattern(
            two_theta=two_theta,
            intensity=intensity,
            wavelength=wavelength,
            metadata={'phases': phases}
        )


def main():
    """
    Demonstration of Session 9 XRD analysis capabilities
    """
    print("=" * 80)
    print("Session 9: X-ray Diffraction Analysis")
    print("Structural and Crystallographic Characterization")
    print("=" * 80)
    
    # Initialize components
    analyzer = XRDAnalyzer()
    simulator = XRDSimulator()
    
    # Demo 1: Basic Pattern Analysis
    print("\n1. Basic XRD Pattern Analysis")
    print("-" * 40)
    
    # Generate synthetic Si pattern
    pattern = simulator.generate_pattern(
        phase='Si',
        crystallite_size=50,  # nm
        microstrain=0.002,
        texture_coefficient=1.5
    )
    
    print(f"Generated pattern: {len(pattern.two_theta)} points")
    print(f"2θ range: {pattern.two_theta[0]:.1f}° - {pattern.two_theta[-1]:.1f}°")
    
    # Process pattern
    processed = analyzer.process_pattern(pattern)
    
    # Find peaks
    peaks = analyzer.find_peaks(processed)
    print(f"\nFound {len(peaks)} peaks:")
    for i, peak in enumerate(peaks[:5]):
        print(f"  Peak {i+1}: 2θ = {peak.position:.2f}°, "
              f"d = {peak.d_spacing:.3f} Å, "
              f"I = {peak.intensity:.0f}")
    
    # Demo 2: Phase Identification
    print("\n2. Phase Identification")
    print("-" * 40)
    
    phases = analyzer.identify_phases(processed, peaks)
    
    if phases:
        print(f"Identified phases:")
        for phase in phases[:3]:
            print(f"  {phase.phase_name} ({phase.formula})")
            print(f"    Crystal system: {phase.crystal_system}")
            print(f"    Match score: {phase.score:.1f}%")
            print(f"    Matched peaks: {len(phase.matched_peaks)}")
    
    # Demo 3: Crystallite Size Analysis
    print("\n3. Crystallite Size Analysis")
    print("-" * 40)
    
    # Scherrer analysis
    size_result = analyzer.calculate_crystallite_size(
        peaks, pattern.wavelength
    )
    
    print(f"Scherrer analysis:")
    print(f"  Mean crystallite size: {size_result['mean_size']:.1f} nm")
    print(f"  Std deviation: {size_result['std_size']:.1f} nm")
    
    # Williamson-Hall analysis
    wh_result = analyzer.williamson_hall_analysis(peaks, pattern.wavelength)
    
    print(f"\nWilliamson-Hall analysis:")
    print(f"  Crystallite size: {wh_result['crystallite_size']:.1f} nm")
    print(f"  Microstrain: {wh_result['strain_percent']:.3f}%")
    print(f"  R²: {wh_result['r_squared']:.3f}")
    
    # Demo 4: Peak Fitting
    print("\n4. Peak Profile Fitting")
    print("-" * 40)
    
    fit_result = analyzer.fit_peaks(
        processed, peaks[:3],
        profile=PeakProfile.PSEUDO_VOIGT
    )
    
    print(f"Fitting with {fit_result['profile']} profile:")
    print(f"  R_wp: {fit_result['r_wp']:.2f}%")
    print(f"  χ²: {fit_result['chi_squared']:.2f}")
    
    for i, peak in enumerate(fit_result['peaks']):
        print(f"\n  Peak {i+1}:")
        print(f"    Position: {peak['position']:.3f}°")
        print(f"    FWHM: {peak['fwhm']:.3f}°")
        print(f"    Area: {peak['area']:.0f}")
    
    # Demo 5: Texture Analysis
    print("\n5. Texture Analysis")
    print("-" * 40)
    
    # Get reference intensities
    ref_pattern = analyzer.reference_patterns.get('Si', [])
    ref_intensities = [p['intensity'] * 1000 for p in ref_pattern[:len(peaks)]]
    
    texture_result = analyzer.calculate_texture(
        processed, peaks, ref_intensities
    )
    
    print(f"Texture analysis:")
    print(f"  Texture index: {texture_result['texture_index']:.3f}")
    print(f"  Is textured: {texture_result['is_textured']}")
    if texture_result['preferred_orientation']:
        print(f"  Preferred orientation: "
              f"{texture_result['preferred_orientation'].position:.1f}°")
    
    # Demo 6: Stress Analysis (sin²ψ method)
    print("\n6. Residual Stress Analysis")
    print("-" * 40)
    
    # Simulate measurements at different ψ angles
    d0 = 3.1355  # Unstressed d-spacing for Si (111)
    measurements = [
        (0, d0 + 0.0001),    # ψ = 0°
        (15, d0 + 0.0002),
        (30, d0 + 0.0004),
        (45, d0 + 0.0007),
        (60, d0 + 0.0010)
    ]
    
    stress_result = analyzer.residual_stress_sin2psi(
        measurements, d0,
        young_modulus=169,  # GPa for Si
        poisson_ratio=0.22
    )
    
    print(f"sin²ψ analysis:")
    print(f"  Stress: {stress_result['stress_mpa']:.1f} MPa")
    print(f"  Type: {stress_result['type']}")
    print(f"  R²: {stress_result['r_squared']:.3f}")
    
    # Demo 7: Mixture Analysis
    print("\n7. Multi-phase Mixture Analysis")
    print("-" * 40)
    
    # Generate mixture pattern
    mixture = simulator.generate_mixture([
        ('Si', 0.7),
        ('SiO2_quartz', 0.3)
    ])
    
    # Process and identify
    mixture_processed = analyzer.process_pattern(mixture)
    mixture_peaks = analyzer.find_peaks(mixture_processed)
    mixture_phases = analyzer.identify_phases(mixture_processed, mixture_peaks)
    
    print(f"Mixture analysis:")
    print(f"  Total peaks found: {len(mixture_peaks)}")
    print(f"  Phases identified: {len(mixture_phases)}")
    
    for phase in mixture_phases[:2]:
        print(f"\n  {phase.phase_name}:")
        print(f"    Score: {phase.score:.1f}%")
        print(f"    Peaks matched: {len(phase.matched_peaks)}")
    
    # Demo 8: Simplified Rietveld Refinement
    print("\n8. Rietveld Refinement (Simplified)")
    print("-" * 40)
    
    structure = analyzer.phase_database['Si']
    rietveld_result = analyzer.rietveld_refinement_simplified(
        processed, structure
    )
    
    print(f"Rietveld refinement:")
    print(f"  R_wp: {rietveld_result['r_wp']:.2f}%")
    print(f"  χ²: {rietveld_result['chi_squared']:.2f}")
    print(f"  Scale factor: {rietveld_result['scale_factor']:.3f}")
    print(f"  Converged: {rietveld_result['converged']}")
    
    print("\n" + "=" * 80)
    print("Session 9 XRD Analysis Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
