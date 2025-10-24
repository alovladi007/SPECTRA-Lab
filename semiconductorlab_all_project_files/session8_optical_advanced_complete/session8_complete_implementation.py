"""
Session 8: Optical Methods II - Complete Implementation
========================================================
Production-ready implementation of Ellipsometry, Photoluminescence, and Raman Spectroscopy
For semiconductor thin films and materials characterization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import scipy.signal as signal
import scipy.optimize as optimize
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.special import wofz  # Faddeeva function for Voigt
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
m_e = 9.10938356e-31  # Electron mass (kg)
epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)

class DispersionModel(Enum):
    """Dispersion models for ellipsometry"""
    CAUCHY = "cauchy"
    SELLMEIER = "sellmeier"
    TAUC_LORENTZ = "tauc_lorentz"
    CODY_LORENTZ = "cody_lorentz"
    DRUDE = "drude"
    LORENTZ = "lorentz"
    GAUSSIAN = "gaussian"
    EMA = "effective_medium"  # Bruggeman EMA

class PLMeasurementType(Enum):
    """Types of PL measurements"""
    STEADY_STATE = "steady_state"
    TIME_RESOLVED = "time_resolved"
    TEMPERATURE_DEPENDENT = "temperature_dependent"
    POWER_DEPENDENT = "power_dependent"
    MAPPING = "mapping"

class RamanMode(Enum):
    """Raman measurement modes"""
    STOKES = "stokes"
    ANTI_STOKES = "anti_stokes"
    RESONANCE = "resonance"
    SURFACE_ENHANCED = "sers"
    TIP_ENHANCED = "ters"

@dataclass
class EllipsometryData:
    """Container for ellipsometry measurements"""
    wavelength: np.ndarray  # nm
    psi: np.ndarray        # degrees
    delta: np.ndarray      # degrees
    angle_of_incidence: float  # degrees
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate data consistency"""
        if len(self.wavelength) != len(self.psi) or len(self.wavelength) != len(self.delta):
            raise ValueError("Wavelength, Psi, and Delta arrays must have same length")
        
        # Sort by wavelength
        sort_idx = np.argsort(self.wavelength)
        self.wavelength = self.wavelength[sort_idx]
        self.psi = self.psi[sort_idx]
        self.delta = self.delta[sort_idx]

@dataclass
class LayerStack:
    """Multi-layer structure for ellipsometry modeling"""
    layers: List[Dict[str, Any]]  # List of layer properties
    substrate: Dict[str, Any]     # Substrate properties
    ambient: Dict[str, Any] = field(default_factory=lambda: {"n": 1.0, "k": 0.0})
    
    def get_total_thickness(self) -> float:
        """Calculate total stack thickness"""
        return sum(layer.get('thickness', 0) for layer in self.layers)

@dataclass
class PLSpectrum:
    """Photoluminescence spectrum data"""
    wavelength: np.ndarray  # nm
    intensity: np.ndarray   # counts or normalized
    excitation_wavelength: float  # nm
    excitation_power: float  # mW
    temperature: float = 293  # K
    integration_time: float = 1.0  # seconds
    measurement_type: PLMeasurementType = PLMeasurementType.STEADY_STATE
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RamanSpectrum:
    """Raman spectrum data"""
    raman_shift: np.ndarray  # cm⁻¹
    intensity: np.ndarray    # counts
    laser_wavelength: float  # nm
    laser_power: float      # mW
    acquisition_time: float = 1.0  # seconds
    mode: RamanMode = RamanMode.STOKES
    metadata: Dict[str, Any] = field(default_factory=dict)


class EllipsometryAnalyzer:
    """
    Spectroscopic Ellipsometry Analysis
    Handles multi-layer optical modeling and parameter extraction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def calculate_rho(self, psi: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """
        Calculate complex reflectance ratio ρ = tan(Ψ)·exp(iΔ)
        """
        psi_rad = np.radians(psi)
        delta_rad = np.radians(delta)
        return np.tan(psi_rad) * np.exp(1j * delta_rad)
    
    def fresnel_coefficients(self, n1: complex, n2: complex, 
                           theta1: float) -> Tuple[complex, complex]:
        """
        Calculate Fresnel reflection coefficients
        """
        # Snell's law
        sin_theta1 = np.sin(np.radians(theta1))
        sin_theta2 = n1 * sin_theta1 / n2
        
        # Handle total internal reflection
        if np.abs(sin_theta2) > 1:
            sin_theta2 = np.sign(sin_theta2.real)
        
        cos_theta1 = np.sqrt(1 - sin_theta1**2)
        cos_theta2 = np.sqrt(1 - sin_theta2**2)
        
        # Ensure correct branch of square root
        if cos_theta2.imag < 0:
            cos_theta2 = -cos_theta2
        
        # p-polarized (TM)
        r_p = (n2 * cos_theta1 - n1 * cos_theta2) / (n2 * cos_theta1 + n1 * cos_theta2)
        
        # s-polarized (TE)
        r_s = (n1 * cos_theta1 - n2 * cos_theta2) / (n1 * cos_theta1 + n2 * cos_theta2)
        
        return r_p, r_s
    
    def transfer_matrix_method(self, wavelength: float, stack: LayerStack, 
                              angle: float) -> Tuple[complex, complex]:
        """
        Calculate reflection using Transfer Matrix Method for multilayer stack
        """
        # Convert wavelength to nm
        lambda_nm = wavelength
        
        # Start with substrate
        n_substrate = complex(
            stack.substrate.get('n', 1.5),
            stack.substrate.get('k', 0)
        )
        
        # Initialize with substrate
        n_prev = n_substrate
        r_p_total = 0
        r_s_total = 0
        
        # Process layers from substrate to ambient (reverse order)
        for layer in reversed(stack.layers):
            n = self._get_refractive_index(lambda_nm, layer)
            thickness = layer.get('thickness', 0)  # nm
            
            if thickness > 0:
                # Calculate phase change
                cos_theta = np.sqrt(1 - (stack.ambient['n'] * np.sin(np.radians(angle)) / n)**2)
                beta = 2 * np.pi * n * thickness * cos_theta / lambda_nm
                
                # Fresnel coefficients at interface
                r_p, r_s = self.fresnel_coefficients(n_prev, n, angle)
                
                # Update total reflection (simplified)
                phase = np.exp(2j * beta)
                r_p_total = (r_p + r_p_total * phase) / (1 + r_p * r_p_total * phase)
                r_s_total = (r_s + r_s_total * phase) / (1 + r_s * r_s_total * phase)
            
            n_prev = n
        
        # Final interface: top layer to ambient
        n_ambient = complex(stack.ambient['n'], stack.ambient['k'])
        r_p_final, r_s_final = self.fresnel_coefficients(n_prev, n_ambient, angle)
        
        # Combine with accumulated reflection
        if stack.layers:
            r_p_total = (r_p_final + r_p_total) / (1 + r_p_final * r_p_total)
            r_s_total = (r_s_final + r_s_total) / (1 + r_s_final * r_s_total)
        else:
            r_p_total = r_p_final
            r_s_total = r_s_final
        
        return r_p_total, r_s_total
    
    def calculate_psi_delta(self, wavelengths: np.ndarray, stack: LayerStack,
                           angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Ψ and Δ for a layer stack
        """
        psi = np.zeros_like(wavelengths)
        delta = np.zeros_like(wavelengths)
        
        for i, wl in enumerate(wavelengths):
            r_p, r_s = self.transfer_matrix_method(wl, stack, angle)
            
            # Calculate ρ = r_p / r_s
            rho = r_p / r_s if r_s != 0 else 0
            
            # Extract Ψ and Δ
            psi[i] = np.degrees(np.arctan(np.abs(rho)))
            delta[i] = np.degrees(np.angle(rho))
        
        # Unwrap delta to avoid discontinuities
        delta = np.degrees(np.unwrap(np.radians(delta)))
        
        return psi, delta
    
    def fit_model(self, data: EllipsometryData, initial_stack: LayerStack,
                 fit_parameters: List[str], bounds: Optional[Dict] = None) -> Dict:
        """
        Fit ellipsometry model to data
        """
        # Define objective function
        def objective(params, param_names):
            # Update stack with new parameters
            stack_copy = self._update_stack(initial_stack, params, param_names)
            
            # Calculate model
            psi_model, delta_model = self.calculate_psi_delta(
                data.wavelength, stack_copy, data.angle_of_incidence
            )
            
            # Calculate MSE
            mse_psi = np.mean((data.psi - psi_model)**2)
            mse_delta = np.mean((data.delta - delta_model)**2)
            
            return mse_psi + mse_delta / 100  # Weight delta less
        
        # Initial parameter values
        x0 = self._get_initial_params(initial_stack, fit_parameters)
        
        # Set bounds
        if bounds is None:
            bounds = self._get_default_bounds(fit_parameters)
        
        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            args=(fit_parameters,),
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Get final stack
        final_stack = self._update_stack(initial_stack, result.x, fit_parameters)
        
        # Calculate final model
        psi_fit, delta_fit = self.calculate_psi_delta(
            data.wavelength, final_stack, data.angle_of_incidence
        )
        
        # Calculate goodness of fit
        mse = result.fun
        r_squared = 1 - mse / np.var(np.concatenate([data.psi, data.delta]))
        
        return {
            'stack': final_stack,
            'parameters': dict(zip(fit_parameters, result.x)),
            'psi_fit': psi_fit,
            'delta_fit': delta_fit,
            'mse': mse,
            'r_squared': r_squared,
            'success': result.success
        }
    
    def _get_refractive_index(self, wavelength: float, layer: Dict) -> complex:
        """
        Calculate refractive index based on dispersion model
        """
        model = layer.get('model', DispersionModel.CAUCHY)
        params = layer.get('params', {})
        
        if model == DispersionModel.CAUCHY:
            return self._cauchy_model(wavelength, params)
        elif model == DispersionModel.SELLMEIER:
            return self._sellmeier_model(wavelength, params)
        elif model == DispersionModel.TAUC_LORENTZ:
            return self._tauc_lorentz_model(wavelength, params)
        else:
            # Default constant n, k
            return complex(params.get('n', 1.5), params.get('k', 0))
    
    def _cauchy_model(self, wavelength: float, params: Dict) -> complex:
        """Cauchy dispersion model: n = A + B/λ² + C/λ⁴"""
        A = params.get('A', 1.45)
        B = params.get('B', 0.01)
        C = params.get('C', 0)
        k = params.get('k', 0)
        
        lambda_um = wavelength / 1000  # Convert to μm
        n = A + B / lambda_um**2 + C / lambda_um**4
        
        return complex(n, k)
    
    def _sellmeier_model(self, wavelength: float, params: Dict) -> complex:
        """Sellmeier dispersion model"""
        B1 = params.get('B1', 1.0)
        B2 = params.get('B2', 0.0)
        B3 = params.get('B3', 0.0)
        C1 = params.get('C1', 0.0)
        C2 = params.get('C2', 0.0)
        C3 = params.get('C3', 0.0)
        
        lambda_um = wavelength / 1000
        lambda2 = lambda_um**2
        
        n_squared = 1 + (B1 * lambda2) / (lambda2 - C1)
        if B2 > 0 and C2 > 0:
            n_squared += (B2 * lambda2) / (lambda2 - C2)
        if B3 > 0 and C3 > 0:
            n_squared += (B3 * lambda2) / (lambda2 - C3)
        
        n = np.sqrt(max(n_squared, 1))
        k = params.get('k', 0)
        
        return complex(n, k)
    
    def _tauc_lorentz_model(self, wavelength: float, params: Dict) -> complex:
        """Tauc-Lorentz oscillator model for amorphous materials"""
        # Energy in eV
        energy = 1240 / wavelength
        
        # Parameters
        A = params.get('A', 100)  # Amplitude
        E0 = params.get('E0', 3.5)  # Peak energy
        C = params.get('C', 1)  # Broadening
        Eg = params.get('Eg', 1.5)  # Optical gap
        eps_inf = params.get('eps_inf', 1)  # High-frequency dielectric constant
        
        # Calculate epsilon_2 (imaginary part)
        if energy > Eg:
            eps2 = (A * E0 * C * (energy - Eg)**2) / \
                   ((energy**2 - E0**2)**2 + C**2 * energy**2) / energy
        else:
            eps2 = 0
        
        # Kramers-Kronig to get epsilon_1 (real part)
        # Simplified - should integrate properly
        eps1 = eps_inf + 2 * A * E0 * C * Eg / np.pi / (E0**2 - Eg**2)
        
        # Convert to n, k
        eps_complex = complex(eps1, eps2)
        n_complex = np.sqrt(eps_complex)
        
        return n_complex
    
    def _update_stack(self, stack: LayerStack, params: np.ndarray, 
                     param_names: List[str]) -> LayerStack:
        """Update stack with new parameter values"""
        import copy
        new_stack = copy.deepcopy(stack)
        
        for i, (param_name, value) in enumerate(zip(param_names, params)):
            # Parse parameter name (e.g., "layer0_thickness", "layer1_n")
            parts = param_name.split('_')
            if parts[0].startswith('layer'):
                layer_idx = int(parts[0][5:])
                param = '_'.join(parts[1:])
                
                if layer_idx < len(new_stack.layers):
                    if param == 'thickness':
                        new_stack.layers[layer_idx]['thickness'] = value
                    elif param in new_stack.layers[layer_idx].get('params', {}):
                        new_stack.layers[layer_idx]['params'][param] = value
        
        return new_stack
    
    def _get_initial_params(self, stack: LayerStack, 
                          param_names: List[str]) -> np.ndarray:
        """Extract initial parameter values from stack"""
        values = []
        
        for param_name in param_names:
            parts = param_name.split('_')
            if parts[0].startswith('layer'):
                layer_idx = int(parts[0][5:])
                param = '_'.join(parts[1:])
                
                if layer_idx < len(stack.layers):
                    if param == 'thickness':
                        values.append(stack.layers[layer_idx].get('thickness', 100))
                    elif param in stack.layers[layer_idx].get('params', {}):
                        values.append(stack.layers[layer_idx]['params'][param])
                    else:
                        values.append(1.5)  # Default
            else:
                values.append(1.5)  # Default
        
        return np.array(values)
    
    def _get_default_bounds(self, param_names: List[str]) -> List[Tuple[float, float]]:
        """Get default parameter bounds"""
        bounds = []
        
        for param_name in param_names:
            if 'thickness' in param_name:
                bounds.append((0.1, 10000))  # nm
            elif '_n' in param_name or '_A' in param_name:
                bounds.append((1.0, 5.0))  # refractive index
            elif '_k' in param_name:
                bounds.append((0, 2.0))  # extinction coefficient
            elif '_B' in param_name or '_C' in param_name:
                bounds.append((0, 1.0))  # Cauchy coefficients
            else:
                bounds.append((0, 100))  # Generic
        
        return bounds


class PhotoluminescenceAnalyzer:
    """
    Photoluminescence Spectroscopy Analysis
    Handles steady-state and time-resolved PL measurements
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def process_spectrum(self, spectrum: PLSpectrum,
                        background: Optional[PLSpectrum] = None,
                        smooth: bool = True) -> PLSpectrum:
        """
        Process PL spectrum with background subtraction and smoothing
        """
        processed = PLSpectrum(
            wavelength=spectrum.wavelength.copy(),
            intensity=spectrum.intensity.copy(),
            excitation_wavelength=spectrum.excitation_wavelength,
            excitation_power=spectrum.excitation_power,
            temperature=spectrum.temperature,
            integration_time=spectrum.integration_time
        )
        
        # Background subtraction
        if background is not None:
            # Interpolate background to match spectrum wavelengths
            f = interp1d(background.wavelength, background.intensity, 
                        kind='linear', fill_value=0, bounds_error=False)
            bg_intensity = f(processed.wavelength)
            processed.intensity = processed.intensity - bg_intensity
            processed.intensity = np.maximum(processed.intensity, 0)
        
        # Smoothing
        if smooth:
            processed.intensity = signal.savgol_filter(
                processed.intensity, window_length=5, polyorder=2
            )
        
        # Normalize by integration time and power
        processed.intensity = processed.intensity / processed.integration_time
        if processed.excitation_power > 0:
            processed.intensity = processed.intensity / processed.excitation_power
        
        return processed
    
    def find_peaks(self, spectrum: PLSpectrum,
                  prominence: float = 0.1,
                  width: Optional[int] = None) -> Dict[str, Any]:
        """
        Find emission peaks in PL spectrum
        """
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            spectrum.intensity,
            prominence=prominence * np.max(spectrum.intensity),
            width=width
        )
        
        # Extract peak properties
        peak_wavelengths = spectrum.wavelength[peak_indices]
        peak_intensities = spectrum.intensity[peak_indices]
        
        # Convert to energy
        peak_energies = 1240 / peak_wavelengths  # eV
        
        # Calculate FWHM
        widths = signal.peak_widths(spectrum.intensity, peak_indices, rel_height=0.5)
        fwhm_indices = widths[0]
        fwhm_nm = fwhm_indices * np.mean(np.diff(spectrum.wavelength))
        fwhm_meV = 1000 * 1240 * fwhm_nm / peak_wavelengths**2
        
        # Identify peak type (band-edge, defect, etc.)
        peak_types = []
        for energy in peak_energies:
            if energy > 3.0:
                peak_types.append("UV emission")
            elif energy > 1.5:
                peak_types.append("Band-edge")
            elif energy > 0.8:
                peak_types.append("Deep level")
            else:
                peak_types.append("IR emission")
        
        return {
            'wavelengths': peak_wavelengths,
            'energies': peak_energies,
            'intensities': peak_intensities,
            'fwhm_nm': fwhm_nm,
            'fwhm_meV': fwhm_meV,
            'types': peak_types,
            'count': len(peak_indices)
        }
    
    def fit_peaks(self, spectrum: PLSpectrum,
                 n_peaks: Optional[int] = None,
                 peak_type: str = 'gaussian') -> Dict[str, Any]:
        """
        Fit multiple peaks to PL spectrum
        """
        # Find initial peak positions
        peaks = self.find_peaks(spectrum)
        
        if n_peaks is None:
            n_peaks = min(peaks['count'], 5)
        
        if n_peaks == 0:
            return {'peaks': [], 'fitted_spectrum': spectrum.intensity}
        
        # Convert to energy for fitting
        energy = 1240 / spectrum.wavelength
        
        # Initial parameters
        params = []
        for i in range(min(n_peaks, len(peaks['wavelengths']))):
            amplitude = peaks['intensities'][i]
            center = peaks['energies'][i]
            sigma = peaks['fwhm_meV'][i] / 1000 / 2.355  # FWHM to sigma in eV
            params.extend([amplitude, center, sigma])
        
        # Add background
        params.extend([0, np.min(spectrum.intensity)])
        
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
                energy, spectrum.intensity,
                p0=params, maxfev=5000
            )
        except:
            # Return original if fitting fails
            return {'peaks': [], 'fitted_spectrum': spectrum.intensity}
        
        # Extract fitted peaks
        fitted_peaks = []
        for i in range(n_peaks):
            fitted_peaks.append({
                'amplitude': popt[i*3],
                'energy': popt[i*3 + 1],
                'wavelength': 1240 / popt[i*3 + 1],
                'sigma_eV': popt[i*3 + 2],
                'fwhm_meV': popt[i*3 + 2] * 2.355 * 1000,
                'area': popt[i*3] * popt[i*3 + 2] * np.sqrt(2 * np.pi)
            })
        
        # Generate fitted spectrum
        fitted_spectrum = fit_func(energy, popt)
        
        # Calculate R²
        ss_res = np.sum((spectrum.intensity - fitted_spectrum)**2)
        ss_tot = np.sum((spectrum.intensity - np.mean(spectrum.intensity))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'peaks': fitted_peaks,
            'fitted_spectrum': fitted_spectrum,
            'background': popt[-2] * energy + popt[-1],
            'r_squared': r_squared
        }
    
    def calculate_quantum_yield(self, spectrum: PLSpectrum,
                               absorption: float,
                               reference_qy: Optional[float] = None) -> float:
        """
        Calculate internal quantum yield
        """
        # Integrate PL intensity
        integrated_pl = np.trapezoid(spectrum.intensity, spectrum.wavelength)
        
        # Correct for absorption
        if absorption > 0:
            qy_relative = integrated_pl / absorption
        else:
            qy_relative = 0
        
        # If reference QY provided, calculate absolute QY
        if reference_qy is not None:
            qy_absolute = qy_relative * reference_qy
        else:
            qy_absolute = qy_relative
        
        return min(qy_absolute, 1.0)  # QY cannot exceed 1
    
    def analyze_temperature_series(self, spectra: List[PLSpectrum]) -> Dict:
        """
        Analyze temperature-dependent PL series
        """
        temperatures = [s.temperature for s in spectra]
        
        # Find peak position and intensity vs temperature
        peak_positions = []
        peak_intensities = []
        integrated_intensities = []
        
        for spectrum in spectra:
            peaks = self.find_peaks(spectrum)
            if len(peaks['energies']) > 0:
                # Track main peak
                peak_positions.append(peaks['energies'][0])
                peak_intensities.append(peaks['intensities'][0])
            else:
                peak_positions.append(np.nan)
                peak_intensities.append(np.nan)
            
            # Integrated intensity
            integrated_intensities.append(
                np.trapezoid(spectrum.intensity, spectrum.wavelength)
            )
        
        # Fit Varshni equation for bandgap vs temperature
        # Eg(T) = Eg(0) - αT²/(T+β)
        if not any(np.isnan(peak_positions)):
            def varshni(T, Eg0, alpha, beta):
                return Eg0 - alpha * T**2 / (T + beta)
            
            try:
                popt, _ = optimize.curve_fit(
                    varshni, temperatures, peak_positions,
                    p0=[peak_positions[0], 0.0005, 300]
                )
                Eg0, alpha, beta = popt
            except:
                Eg0, alpha, beta = peak_positions[0], 0, 0
        else:
            Eg0, alpha, beta = 0, 0, 0
        
        # Fit Arrhenius for intensity quenching
        # I(T) = I0 / (1 + A*exp(-Ea/kT))
        if not any(np.isnan(peak_intensities)):
            def arrhenius(T, I0, A, Ea):
                return I0 / (1 + A * np.exp(-Ea / (k_B * T / e)))
            
            try:
                popt, _ = optimize.curve_fit(
                    arrhenius, temperatures, peak_intensities,
                    p0=[peak_intensities[0], 100, 0.1]
                )
                I0, A_factor, activation_energy = popt
            except:
                I0, A_factor, activation_energy = peak_intensities[0], 0, 0
        else:
            I0, A_factor, activation_energy = 0, 0, 0
        
        return {
            'temperatures': temperatures,
            'peak_positions': peak_positions,
            'peak_intensities': peak_intensities,
            'integrated_intensities': integrated_intensities,
            'varshni_params': {
                'Eg0': Eg0,
                'alpha': alpha,
                'beta': beta
            },
            'activation_energy': activation_energy
        }
    
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
            width = params[i*3 + 2]
            y += amp * width**2 / ((x - center)**2 + width**2)
        
        # Add background
        y += params[-2] * x + params[-1]
        return y
    
    def _multi_voigt(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Multiple Voigt peaks (using Faddeeva function)"""
        n_peaks = (len(params) - 2) // 3
        y = np.zeros_like(x)
        
        for i in range(n_peaks):
            amp = params[i*3]
            center = params[i*3 + 1]
            sigma = params[i*3 + 2]
            gamma = sigma * 0.5  # Lorentzian width related to Gaussian
            
            # Voigt profile using Faddeeva function
            z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
            voigt = np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
            y += amp * voigt
        
        # Add background
        y += params[-2] * x + params[-1]
        return y


class RamanAnalyzer:
    """
    Raman Spectroscopy Analysis
    Handles peak identification, stress/strain analysis, and phase identification
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._load_raman_database()
    
    def _load_raman_database(self):
        """Load database of Raman peaks for semiconductors"""
        self.raman_database = {
            # Silicon
            'Si': [
                {'position': 520.5, 'mode': 'TO/LO', 'intensity': 'strong'},
                {'position': 300, 'mode': '2TA', 'intensity': 'weak'}
            ],
            # Germanium
            'Ge': [
                {'position': 300, 'mode': 'TO/LO', 'intensity': 'strong'}
            ],
            # GaAs
            'GaAs': [
                {'position': 268, 'mode': 'TO(Γ)', 'intensity': 'strong'},
                {'position': 292, 'mode': 'LO(Γ)', 'intensity': 'strong'}
            ],
            # GaN
            'GaN': [
                {'position': 532, 'mode': 'E2(high)', 'intensity': 'strong'},
                {'position': 568, 'mode': 'E2(high)', 'intensity': 'strong'},
                {'position': 734, 'mode': 'A1(LO)', 'intensity': 'medium'}
            ],
            # Graphene
            'Graphene': [
                {'position': 1580, 'mode': 'G', 'intensity': 'strong'},
                {'position': 2700, 'mode': '2D', 'intensity': 'strong'},
                {'position': 1350, 'mode': 'D', 'intensity': 'variable'}
            ],
            # SiO2
            'SiO2': [
                {'position': 464, 'mode': 'Si-O-Si', 'intensity': 'strong'},
                {'position': 800, 'mode': 'Si-O', 'intensity': 'medium'}
            ]
        }
    
    def process_spectrum(self, spectrum: RamanSpectrum,
                        baseline_correct: bool = True,
                        normalize: bool = True) -> RamanSpectrum:
        """
        Process Raman spectrum with baseline correction and normalization
        """
        processed = RamanSpectrum(
            raman_shift=spectrum.raman_shift.copy(),
            intensity=spectrum.intensity.copy(),
            laser_wavelength=spectrum.laser_wavelength,
            laser_power=spectrum.laser_power,
            acquisition_time=spectrum.acquisition_time
        )
        
        # Baseline correction
        if baseline_correct:
            baseline = self._calculate_baseline(
                processed.raman_shift,
                processed.intensity
            )
            processed.intensity = processed.intensity - baseline
        
        # Normalize
        if normalize:
            max_intensity = np.max(processed.intensity)
            if max_intensity > 0:
                processed.intensity = processed.intensity / max_intensity
        
        return processed
    
    def find_peaks(self, spectrum: RamanSpectrum,
                  prominence: float = 0.05) -> Dict[str, Any]:
        """
        Find and identify Raman peaks
        """
        # Find peaks
        peak_indices, properties = signal.find_peaks(
            spectrum.intensity,
            prominence=prominence * np.max(spectrum.intensity)
        )
        
        # Extract peak properties
        peak_positions = spectrum.raman_shift[peak_indices]
        peak_intensities = spectrum.intensity[peak_indices]
        
        # Calculate FWHM
        widths = signal.peak_widths(spectrum.intensity, peak_indices, rel_height=0.5)
        fwhm = widths[0] * np.mean(np.diff(spectrum.raman_shift))
        
        # Identify peaks
        identifications = []
        for pos in peak_positions:
            match = self._identify_peak(pos)
            identifications.append(match)
        
        return {
            'positions': peak_positions,
            'intensities': peak_intensities,
            'fwhm': fwhm,
            'identifications': identifications,
            'indices': peak_indices
        }
    
    def calculate_stress(self, measured_position: float,
                        reference_position: float,
                        material: str = 'Si') -> Dict[str, float]:
        """
        Calculate stress/strain from Raman shift
        For Si: Δω = -1.8ω₀ε (biaxial stress)
        """
        # Shift in cm⁻¹
        delta_omega = measured_position - reference_position
        
        # Material-specific parameters
        if material == 'Si':
            # Silicon parameters
            omega_0 = 520.5  # cm⁻¹
            k = -1.8  # Strain coefficient
            E = 169  # Young's modulus (GPa)
            nu = 0.22  # Poisson's ratio
            
            # Calculate strain
            strain = delta_omega / (k * omega_0)
            
            # Calculate stress (biaxial)
            stress = E * strain / (1 - nu)  # GPa
            
        elif material == 'GaAs':
            omega_0 = 268  # TO mode
            k = -1.5
            E = 85.5  # GPa
            nu = 0.31
            
            strain = delta_omega / (k * omega_0)
            stress = E * strain / (1 - nu)
            
        else:
            # Generic calculation
            strain = delta_omega / reference_position
            stress = strain * 100  # Approximate GPa
        
        return {
            'shift': delta_omega,
            'strain': strain,
            'stress': stress,
            'type': 'tensile' if delta_omega < 0 else 'compressive'
        }
    
    def analyze_crystallinity(self, spectrum: RamanSpectrum,
                            material: str = 'Si') -> Dict[str, float]:
        """
        Analyze crystallinity from Raman spectrum
        """
        peaks = self.find_peaks(spectrum)
        
        if material == 'Si':
            # Silicon crystallinity from 520 cm⁻¹ peak
            si_peak_idx = np.argmin(np.abs(peaks['positions'] - 520))
            
            if len(peaks['positions']) > 0:
                # Peak position indicates crystallinity
                peak_pos = peaks['positions'][si_peak_idx]
                peak_fwhm = peaks['fwhm'][si_peak_idx]
                
                # Crystalline Si: sharp peak at 520.5 cm⁻¹
                # Amorphous Si: broad band around 480 cm⁻¹
                
                # Estimate crystalline fraction
                if peak_pos > 515 and peak_fwhm < 10:
                    crystallinity = 0.95  # Highly crystalline
                elif peak_pos > 500:
                    crystallinity = 0.7  # Polycrystalline
                else:
                    crystallinity = 0.2  # Mostly amorphous
                
                grain_size = 50 / peak_fwhm  # Empirical relation (nm)
            else:
                crystallinity = 0
                grain_size = 0
                
        elif material == 'Graphene':
            # Graphene quality from D/G ratio
            d_peak_idx = np.argmin(np.abs(peaks['positions'] - 1350))
            g_peak_idx = np.argmin(np.abs(peaks['positions'] - 1580))
            
            if len(peaks['positions']) > max(d_peak_idx, g_peak_idx):
                d_intensity = peaks['intensities'][d_peak_idx]
                g_intensity = peaks['intensities'][g_peak_idx]
                
                # D/G ratio indicates defect density
                d_g_ratio = d_intensity / g_intensity if g_intensity > 0 else 0
                
                # La (crystallite size) = (2.4×10^-10) * λ^4 * (I_G/I_D)
                lambda_nm = spectrum.laser_wavelength
                if d_intensity > 0:
                    grain_size = 2.4e-10 * lambda_nm**4 * (g_intensity / d_intensity)
                else:
                    grain_size = 100  # Large crystallites
                
                crystallinity = 1 / (1 + d_g_ratio)
            else:
                crystallinity = 0.5
                grain_size = 10
        else:
            crystallinity = 0.5
            grain_size = 10
        
        return {
            'crystallinity': crystallinity,
            'grain_size_nm': grain_size,
            'quality': 'high' if crystallinity > 0.8 else 'medium' if crystallinity > 0.5 else 'low'
        }
    
    def map_analysis(self, spectra_map: np.ndarray,
                    positions: np.ndarray,
                    peak_of_interest: float) -> Dict[str, Any]:
        """
        Analyze Raman mapping data
        spectra_map: 3D array (x, y, raman_shift)
        """
        nx, ny, n_points = spectra_map.shape
        
        # Extract peak properties at each position
        peak_positions = np.zeros((nx, ny))
        peak_intensities = np.zeros((nx, ny))
        peak_widths = np.zeros((nx, ny))
        
        for i in range(nx):
            for j in range(ny):
                spectrum = RamanSpectrum(
                    raman_shift=positions,
                    intensity=spectra_map[i, j, :],
                    laser_wavelength=532,  # Default
                    laser_power=1.0
                )
                
                peaks = self.find_peaks(spectrum)
                
                if len(peaks['positions']) > 0:
                    # Find closest peak to interest
                    idx = np.argmin(np.abs(peaks['positions'] - peak_of_interest))
                    peak_positions[i, j] = peaks['positions'][idx]
                    peak_intensities[i, j] = peaks['intensities'][idx]
                    peak_widths[i, j] = peaks['fwhm'][idx]
        
        # Calculate statistics
        return {
            'peak_position_map': peak_positions,
            'peak_intensity_map': peak_intensities,
            'peak_width_map': peak_widths,
            'position_mean': np.mean(peak_positions),
            'position_std': np.std(peak_positions),
            'uniformity': 1 - np.std(peak_intensities) / np.mean(peak_intensities)
        }
    
    def _calculate_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Calculate baseline using asymmetric least squares"""
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
        
        L = len(y)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = 1e6 * D.dot(D.transpose())
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
        
        for i in range(10):
            W.setdiag(w)
            Z = W + D
            z = spsolve(Z, w * y)
            w = 0.01 * (y > z) + (1 - 0.01) * (y < z)
        
        return z
    
    def _identify_peak(self, position: float, tolerance: float = 10) -> Optional[Dict]:
        """Identify Raman peak based on position"""
        for material, peaks in self.raman_database.items():
            for peak in peaks:
                if abs(position - peak['position']) < tolerance:
                    return {
                        'material': material,
                        'mode': peak['mode'],
                        'expected': peak['position']
                    }
        return None


class OpticalTestDataGeneratorII:
    """
    Generate synthetic test data for Session 8 optical methods
    """
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_ellipsometry_data(self, 
                                  stack: Optional[LayerStack] = None,
                                  wavelength_range: Tuple[float, float] = (300, 800),
                                  angle: float = 70.0,
                                  n_points: int = 200) -> EllipsometryData:
        """Generate synthetic ellipsometry data"""
        
        if stack is None:
            # Default SiO2 on Si
            stack = LayerStack(
                layers=[{
                    'thickness': 100,  # nm
                    'model': DispersionModel.CAUCHY,
                    'params': {'A': 1.46, 'B': 0.00354, 'C': 0, 'k': 0}
                }],
                substrate={'n': 3.85, 'k': 0.02}  # Si
            )
        
        # Generate wavelength array
        wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_points)
        
        # Calculate ideal Psi and Delta
        analyzer = EllipsometryAnalyzer()
        psi, delta = analyzer.calculate_psi_delta(wavelengths, stack, angle)
        
        # Add realistic noise
        psi += np.random.normal(0, 0.1, len(psi))
        delta += np.random.normal(0, 0.5, len(delta))
        
        return EllipsometryData(
            wavelength=wavelengths,
            psi=psi,
            delta=delta,
            angle_of_incidence=angle,
            metadata={'stack': stack}
        )
    
    def generate_pl_spectrum(self,
                            material: str = 'GaAs',
                            temperature: float = 293,
                            excitation_wavelength: float = 532,
                            n_points: int = 500) -> PLSpectrum:
        """Generate synthetic PL spectrum"""
        
        # Material parameters
        materials = {
            'GaAs': {'peak': 870, 'fwhm': 30, 'bandgap': 1.42},
            'GaN': {'peak': 365, 'fwhm': 20, 'bandgap': 3.4},
            'InP': {'peak': 920, 'fwhm': 35, 'bandgap': 1.35},
            'CdTe': {'peak': 830, 'fwhm': 25, 'bandgap': 1.5}
        }
        
        params = materials.get(material, materials['GaAs'])
        
        # Generate wavelength array
        wavelengths = np.linspace(
            params['peak'] - 100,
            params['peak'] + 100,
            n_points
        )
        
        # Generate main emission peak
        intensity = 1000 * np.exp(
            -0.5 * ((wavelengths - params['peak']) / (params['fwhm'] / 2.355))**2
        )
        
        # Add phonon replicas
        if material == 'GaAs':
            # LO phonon energy ~36 meV
            phonon_shift = 36 * params['peak']**2 / 1240 / 1000  # nm
            intensity += 300 * np.exp(
                -0.5 * ((wavelengths - (params['peak'] + phonon_shift)) / (params['fwhm'] / 2.355))**2
            )
        
        # Add defect emission
        defect_peak = params['peak'] + 50
        intensity += 200 * np.exp(
            -0.5 * ((wavelengths - defect_peak) / 40)**2
        )
        
        # Temperature broadening
        temp_factor = np.sqrt(temperature / 293)
        intensity = intensity * np.exp(-(wavelengths - params['peak'])**2 * temp_factor / 10000)
        
        # Add noise
        intensity += np.random.poisson(10, len(intensity))
        intensity = np.maximum(intensity, 0)
        
        return PLSpectrum(
            wavelength=wavelengths,
            intensity=intensity,
            excitation_wavelength=excitation_wavelength,
            excitation_power=10.0,
            temperature=temperature,
            integration_time=1.0,
            metadata={'material': material}
        )
    
    def generate_raman_spectrum(self,
                              material: str = 'Si',
                              laser_wavelength: float = 532,
                              stress: float = 0,  # GPa
                              n_points: int = 600) -> RamanSpectrum:
        """Generate synthetic Raman spectrum"""
        
        # Generate Raman shift array
        raman_shift = np.linspace(100, 700, n_points)
        intensity = np.ones_like(raman_shift) * 10  # Background
        
        if material == 'Si':
            # Silicon peaks
            si_peak = 520.5 + stress * 2.5  # Shift due to stress
            si_width = 5 + abs(stress) * 0.5  # Broadening with stress
            
            intensity += 1000 * np.exp(
                -0.5 * ((raman_shift - si_peak) / si_width)**2
            )
            
            # Second order peak
            intensity += 100 * np.exp(
                -0.5 * ((raman_shift - 300) / 20)**2
            )
            
        elif material == 'GaAs':
            # TO and LO modes
            intensity += 800 * np.exp(
                -0.5 * ((raman_shift - 268) / 4)**2
            )
            intensity += 1000 * np.exp(
                -0.5 * ((raman_shift - 292) / 4)**2
            )
            
        elif material == 'Graphene':
            # G peak
            intensity += 1000 * np.exp(
                -0.5 * ((raman_shift - 1580) / 15)**2
            )
            # D peak (defects)
            intensity += 300 * np.exp(
                -0.5 * ((raman_shift - 1350) / 20)**2
            )
            # Extend range for 2D peak
            if raman_shift[-1] < 2700:
                extended_shift = np.linspace(100, 2800, n_points * 4)
                extended_intensity = np.ones_like(extended_shift) * 10
                
                # Re-add peaks on extended range
                extended_intensity += 1000 * np.exp(
                    -0.5 * ((extended_shift - 1580) / 15)**2
                )
                extended_intensity += 300 * np.exp(
                    -0.5 * ((extended_shift - 1350) / 20)**2
                )
                # 2D peak
                extended_intensity += 800 * np.exp(
                    -0.5 * ((extended_shift - 2700) / 25)**2
                )
                
                raman_shift = extended_shift
                intensity = extended_intensity
        
        # Add noise
        intensity += np.random.normal(0, 5, len(intensity))
        intensity = np.maximum(intensity, 0)
        
        return RamanSpectrum(
            raman_shift=raman_shift,
            intensity=intensity,
            laser_wavelength=laser_wavelength,
            laser_power=5.0,
            acquisition_time=10.0,
            metadata={'material': material, 'stress': stress}
        )


def main():
    """
    Main demonstration of Session 8 optical methods
    """
    print("=" * 80)
    print("Session 8: Optical Methods II")
    print("Ellipsometry, Photoluminescence, and Raman Spectroscopy")
    print("=" * 80)
    
    # Initialize components
    ellipsometry = EllipsometryAnalyzer()
    pl_analyzer = PhotoluminescenceAnalyzer()
    raman_analyzer = RamanAnalyzer()
    generator = OpticalTestDataGeneratorII()
    
    # Demo 1: Ellipsometry
    print("\n1. Ellipsometry Analysis")
    print("-" * 40)
    
    # Create a thin film stack
    stack = LayerStack(
        layers=[
            {
                'thickness': 50,  # nm
                'model': DispersionModel.CAUCHY,
                'params': {'A': 1.46, 'B': 0.00354, 'C': 0, 'k': 0}
            },
            {
                'thickness': 150,  # nm
                'model': DispersionModel.CAUCHY,
                'params': {'A': 2.0, 'B': 0.01, 'C': 0, 'k': 0.1}
            }
        ],
        substrate={'n': 3.85, 'k': 0.02}
    )
    
    # Generate measurement data
    ell_data = generator.generate_ellipsometry_data(stack)
    print(f"Generated ellipsometry data: {len(ell_data.wavelength)} points")
    print(f"Angle of incidence: {ell_data.angle_of_incidence}°")
    
    # Fit model
    fit_params = ['layer0_thickness', 'layer1_thickness', 'layer1_params_A']
    fit_result = ellipsometry.fit_model(ell_data, stack, fit_params)
    
    print(f"\nFitting results:")
    for param, value in fit_result['parameters'].items():
        print(f"  {param}: {value:.2f}")
    print(f"  MSE: {fit_result['mse']:.4f}")
    print(f"  R²: {fit_result['r_squared']:.4f}")
    
    # Demo 2: Photoluminescence
    print("\n2. Photoluminescence Analysis")
    print("-" * 40)
    
    # Generate PL spectrum
    pl_spectrum = generator.generate_pl_spectrum('GaAs', temperature=10)
    print(f"Generated PL spectrum for GaAs at {pl_spectrum.temperature}K")
    
    # Process and find peaks
    processed_pl = pl_analyzer.process_spectrum(pl_spectrum)
    pl_peaks = pl_analyzer.find_peaks(processed_pl)
    
    print(f"Found {pl_peaks['count']} emission peaks:")
    for i in range(min(3, pl_peaks['count'])):
        print(f"  Peak {i+1}: {pl_peaks['wavelengths'][i]:.1f} nm "
              f"({pl_peaks['energies'][i]:.3f} eV), "
              f"FWHM: {pl_peaks['fwhm_meV'][i]:.1f} meV")
    
    # Fit peaks
    fit_result = pl_analyzer.fit_peaks(processed_pl, n_peaks=2)
    print(f"\nPeak fitting R²: {fit_result['r_squared']:.4f}")
    
    # Demo 3: Raman Spectroscopy
    print("\n3. Raman Spectroscopy Analysis")
    print("-" * 40)
    
    # Generate Raman spectrum with stress
    raman_spectrum = generator.generate_raman_spectrum('Si', stress=1.0)  # 1 GPa stress
    print(f"Generated Raman spectrum for Si under {raman_spectrum.metadata['stress']} GPa stress")
    
    # Process and find peaks
    processed_raman = raman_analyzer.process_spectrum(raman_spectrum)
    raman_peaks = raman_analyzer.find_peaks(processed_raman)
    
    print(f"Found peaks at:")
    for i, (pos, ident) in enumerate(zip(raman_peaks['positions'][:3], 
                                         raman_peaks['identifications'][:3])):
        if ident:
            print(f"  {pos:.1f} cm⁻¹: {ident['material']} - {ident['mode']}")
        else:
            print(f"  {pos:.1f} cm⁻¹: Unidentified")
    
    # Calculate stress from main Si peak
    if len(raman_peaks['positions']) > 0:
        main_peak = raman_peaks['positions'][np.argmax(raman_peaks['intensities'])]
        stress_result = raman_analyzer.calculate_stress(main_peak, 520.5, 'Si')
        print(f"\nStress analysis:")
        print(f"  Peak shift: {stress_result['shift']:.2f} cm⁻¹")
        print(f"  Calculated stress: {stress_result['stress']:.2f} GPa ({stress_result['type']})")
    
    # Crystallinity analysis
    crystal_result = raman_analyzer.analyze_crystallinity(processed_raman, 'Si')
    print(f"\nCrystallinity analysis:")
    print(f"  Crystalline fraction: {crystal_result['crystallinity']:.2%}")
    print(f"  Grain size: {crystal_result['grain_size_nm']:.1f} nm")
    print(f"  Quality: {crystal_result['quality']}")
    
    # Demo 4: Temperature-dependent PL
    print("\n4. Temperature-Dependent PL Analysis")
    print("-" * 40)
    
    # Generate temperature series
    temperatures = [10, 50, 100, 150, 200, 250, 293]
    pl_series = [generator.generate_pl_spectrum('GaAs', T) for T in temperatures]
    
    # Analyze series
    temp_analysis = pl_analyzer.analyze_temperature_series(pl_series)
    
    print(f"Temperature series analysis:")
    print(f"  Eg(0K): {temp_analysis['varshni_params']['Eg0']:.3f} eV")
    print(f"  Varshni α: {temp_analysis['varshni_params']['alpha']:.6f} eV/K")
    print(f"  Activation energy: {temp_analysis['activation_energy']:.3f} eV")
    
    print("\n" + "=" * 80)
    print("Session 8 Implementation Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
