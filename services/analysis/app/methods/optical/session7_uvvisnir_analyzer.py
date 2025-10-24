"""
UV-Vis-NIR Spectroscopy Analysis Module
Session 7 - Optical I Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from scipy import signal, optimize, interpolate
from scipy.integrate import trapz
from scipy.stats import linregress
import warnings
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransitionType(Enum):
    """Optical transition types for band gap analysis"""
    DIRECT = "direct"
    INDIRECT = "indirect"
    DIRECT_FORBIDDEN = "direct_forbidden"
    INDIRECT_FORBIDDEN = "indirect_forbidden"


class BaselineMethod(Enum):
    """Baseline correction methods"""
    POLYNOMIAL = "polynomial"
    SPLINE = "spline"
    RUBBERBAND = "rubberband"
    ALS = "asymmetric_least_squares"
    MANUAL = "manual"


@dataclass
class OpticalConstants:
    """Optical constants extracted from spectrum"""
    wavelength: np.ndarray
    n: np.ndarray  # Refractive index
    k: np.ndarray  # Extinction coefficient
    alpha: np.ndarray  # Absorption coefficient
    epsilon_real: np.ndarray  # Dielectric constant (real)
    epsilon_imag: np.ndarray  # Dielectric constant (imaginary)


@dataclass
class TaucPlotResult:
    """Tauc plot analysis results"""
    band_gap: float  # eV
    transition_type: str
    r_squared: float
    intercept: float
    photon_energy: np.ndarray
    tauc_values: np.ndarray
    fit_range: Tuple[float, float]
    uncertainty: float  # eV


@dataclass
class UrbachTailResult:
    """Urbach tail analysis results"""
    urbach_energy: float  # meV
    disorder_parameter: float
    r_squared: float
    energy_range: Tuple[float, float]
    fit_quality: str


class UVVisNIRAnalyzer:
    """
    Comprehensive UV-Vis-NIR spectroscopy analyzer
    
    Features:
    - Multiple measurement modes (T, A, R)
    - Advanced baseline correction
    - Band gap extraction via Tauc plots
    - Urbach tail analysis
    - Optical constants calculation
    - Thin film interference handling
    """
    
    def __init__(self, wavelength_range: Tuple[float, float] = (200, 2500)):
        """
        Initialize UV-Vis-NIR analyzer
        
        Args:
            wavelength_range: Wavelength range in nm
        """
        self.wavelength_range = wavelength_range
        self.h = 6.62607e-34  # Planck constant (J⋅s)
        self.c = 2.998e8  # Speed of light (m/s)
        self.e = 1.602e-19  # Elementary charge (C)
        
    def process_spectrum(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        mode: str = 'transmission',
        reference: Optional[np.ndarray] = None,
        baseline_method: BaselineMethod = BaselineMethod.ALS,
        smooth: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Process raw spectrum with corrections
        
        Args:
            wavelength: Wavelength array (nm)
            intensity: Intensity array
            mode: 'transmission', 'absorbance', or 'reflectance'
            reference: Reference spectrum for correction
            baseline_method: Baseline correction method
            smooth: Apply smoothing
            
        Returns:
            Processed spectrum dictionary
        """
        # Input validation
        if len(wavelength) != len(intensity):
            raise ValueError("Wavelength and intensity arrays must have same length")
        
        # Sort by wavelength
        sort_idx = np.argsort(wavelength)
        wavelength = wavelength[sort_idx]
        intensity = intensity[sort_idx]
        
        # Reference correction
        if reference is not None:
            intensity = intensity / reference
            
        # Smoothing
        if smooth:
            window_length = min(21, len(intensity))
            if window_length % 2 == 0:
                window_length -= 1
            if window_length >= 3:
                intensity = signal.savgol_filter(intensity, window_length, 3)
        
        # Baseline correction
        baseline = self._calculate_baseline(wavelength, intensity, baseline_method)
        corrected = intensity - baseline
        
        # Convert to different representations
        results = {
            'wavelength': wavelength,
            'raw': intensity,
            'baseline': baseline,
            'corrected': corrected
        }
        
        if mode == 'transmission':
            results['transmission'] = corrected
            results['absorbance'] = -np.log10(np.clip(corrected, 1e-6, None))
        elif mode == 'absorbance':
            results['absorbance'] = corrected
            results['transmission'] = 10**(-corrected)
        elif mode == 'reflectance':
            results['reflectance'] = corrected
            results['absorbance'] = -np.log10(np.clip(corrected, 1e-6, None))
        
        return results
    
    def _calculate_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: BaselineMethod,
        **kwargs
    ) -> np.ndarray:
        """Calculate baseline using specified method"""
        
        if method == BaselineMethod.POLYNOMIAL:
            return self._polynomial_baseline(x, y, **kwargs)
        elif method == BaselineMethod.SPLINE:
            return self._spline_baseline(x, y, **kwargs)
        elif method == BaselineMethod.RUBBERBAND:
            return self._rubberband_baseline(x, y, **kwargs)
        elif method == BaselineMethod.ALS:
            return self._als_baseline(y, **kwargs)
        else:
            return np.zeros_like(y)
    
    def _polynomial_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 3
    ) -> np.ndarray:
        """Polynomial baseline fitting"""
        coeffs = np.polyfit(x, y, degree)
        return np.polyval(coeffs, x)
    
    def _spline_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_points: int = 10,
        smoothing: float = 1.0
    ) -> np.ndarray:
        """Spline baseline fitting"""
        # Select anchor points
        indices = np.linspace(0, len(x)-1, n_points, dtype=int)
        x_anchors = x[indices]
        y_anchors = y[indices]
        
        # Fit spline
        spline = interpolate.UnivariateSpline(
            x_anchors, y_anchors, s=smoothing
        )
        return spline(x)
    
    def _rubberband_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Rubberband baseline (convex hull)"""
        from scipy.spatial import ConvexHull
        
        # Create points for convex hull
        points = np.column_stack([x, y])
        
        # Get convex hull
        hull = ConvexHull(points)
        
        # Get lower hull points
        hull_points = points[hull.vertices]
        hull_points = hull_points[hull_points[:, 0].argsort()]
        
        # Interpolate baseline
        baseline = np.interp(x, hull_points[:, 0], hull_points[:, 1])
        
        # Ensure baseline is below spectrum
        baseline = np.minimum(baseline, y)
        
        return baseline
    
    def _als_baseline(
        self,
        y: np.ndarray,
        lam: float = 1e6,
        p: float = 0.01,
        n_iter: int = 10
    ) -> np.ndarray:
        """
        Asymmetric Least Squares baseline
        
        Args:
            y: Spectrum intensity
            lam: Smoothness parameter (larger = smoother)
            p: Asymmetry parameter (smaller = more baseline)
            n_iter: Number of iterations
        """
        L = len(y)
        D = np.diff(np.eye(L), 2)  # Second derivative matrix
        D = lam * D.T @ D
        
        w = np.ones(L)
        z = np.zeros(L)
        
        for _ in range(n_iter):
            W = np.diag(w)
            z = np.linalg.solve(W + D, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
            
        return z
    
    def calculate_tauc_plot(
        self,
        wavelength: np.ndarray,
        absorbance: np.ndarray,
        transition_type: TransitionType = TransitionType.DIRECT,
        energy_range: Optional[Tuple[float, float]] = None,
        film_thickness: Optional[float] = None
    ) -> TaucPlotResult:
        """
        Calculate Tauc plot and extract band gap
        
        Args:
            wavelength: Wavelength in nm
            absorbance: Absorbance spectrum
            transition_type: Type of optical transition
            energy_range: Energy range for linear fit (eV)
            film_thickness: Film thickness in nm (for alpha calculation)
            
        Returns:
            TaucPlotResult with band gap and fit quality
        """
        # Convert wavelength to photon energy
        photon_energy = self._wavelength_to_energy(wavelength)
        
        # Calculate absorption coefficient
        if film_thickness:
            # α = 2.303 * A / d (where A is absorbance, d is thickness in cm)
            alpha = 2.303 * absorbance / (film_thickness * 1e-7)
        else:
            # Use absorbance as proxy
            alpha = absorbance
            
        # Calculate Tauc values based on transition type
        if transition_type == TransitionType.DIRECT:
            n = 2  # (αhν)^2
        elif transition_type == TransitionType.INDIRECT:
            n = 0.5  # (αhν)^0.5
        elif transition_type == TransitionType.DIRECT_FORBIDDEN:
            n = 2/3  # (αhν)^(2/3)
        elif transition_type == TransitionType.INDIRECT_FORBIDDEN:
            n = 1/3  # (αhν)^(1/3)
        else:
            n = 2  # Default to direct
            
        tauc_values = (alpha * photon_energy) ** n
        
        # Auto-determine fitting range if not specified
        if energy_range is None:
            # Find steepest region (maximum derivative)
            d_tauc = np.gradient(tauc_values)
            d_energy = np.gradient(photon_energy)
            derivative = d_tauc / d_energy
            
            # Smooth derivative
            derivative_smooth = signal.savgol_filter(derivative, 11, 3)
            
            # Find peak in derivative
            peak_idx = np.argmax(derivative_smooth)
            
            # Fit around peak
            window = int(len(photon_energy) * 0.1)  # 10% window
            start_idx = max(0, peak_idx - window)
            end_idx = min(len(photon_energy), peak_idx + window)
            
            energy_range = (photon_energy[start_idx], photon_energy[end_idx])
        
        # Select data in fitting range
        mask = (photon_energy >= energy_range[0]) & (photon_energy <= energy_range[1])
        fit_energy = photon_energy[mask]
        fit_tauc = tauc_values[mask]
        
        # Linear regression
        if len(fit_energy) < 2:
            raise ValueError("Insufficient data points for linear fit")
            
        slope, intercept, r_value, p_value, std_err = linregress(fit_energy, fit_tauc)
        
        # Calculate band gap (x-intercept)
        band_gap = -intercept / slope if slope != 0 else 0
        
        # Estimate uncertainty
        uncertainty = std_err / abs(slope) if slope != 0 else 0
        
        return TaucPlotResult(
            band_gap=band_gap,
            transition_type=transition_type.value,
            r_squared=r_value**2,
            intercept=intercept,
            photon_energy=photon_energy,
            tauc_values=tauc_values,
            fit_range=energy_range,
            uncertainty=uncertainty
        )
    
    def analyze_urbach_tail(
        self,
        wavelength: np.ndarray,
        absorbance: np.ndarray,
        energy_range: Optional[Tuple[float, float]] = None
    ) -> UrbachTailResult:
        """
        Analyze Urbach tail to quantify disorder
        
        The Urbach tail follows: α = α₀ * exp((E - E₀) / E_u)
        where E_u is the Urbach energy (disorder parameter)
        
        Args:
            wavelength: Wavelength in nm
            absorbance: Absorbance spectrum
            energy_range: Energy range for Urbach tail fit
            
        Returns:
            UrbachTailResult with Urbach energy
        """
        # Convert to photon energy
        photon_energy = self._wavelength_to_energy(wavelength)
        
        # Use log of absorbance for linear fit
        log_abs = np.log(np.clip(absorbance, 1e-10, None))
        
        # Auto-determine Urbach region if not specified
        if energy_range is None:
            # Typically below band gap
            # Find region with exponential behavior
            d2_log = np.gradient(np.gradient(log_abs))
            
            # Look for constant second derivative region
            # (exponential has constant d²(ln α)/dE²)
            smooth_d2 = signal.savgol_filter(d2_log, 11, 3)
            
            # Find flattest region
            variation = np.abs(np.gradient(smooth_d2))
            min_var_idx = np.argmin(variation[len(variation)//4:3*len(variation)//4])
            min_var_idx += len(variation)//4
            
            window = int(len(photon_energy) * 0.1)
            start_idx = max(0, min_var_idx - window//2)
            end_idx = min(len(photon_energy), min_var_idx + window//2)
            
            energy_range = (photon_energy[start_idx], photon_energy[end_idx])
        
        # Select fitting range
        mask = (photon_energy >= energy_range[0]) & (photon_energy <= energy_range[1])
        fit_energy = photon_energy[mask]
        fit_log_abs = log_abs[mask]
        
        # Linear fit to ln(α) vs E
        if len(fit_energy) < 2:
            raise ValueError("Insufficient data for Urbach fit")
            
        slope, intercept, r_value, _, _ = linregress(fit_energy, fit_log_abs)
        
        # Urbach energy (meV)
        urbach_energy = 1000 / abs(slope) if slope != 0 else np.inf
        
        # Disorder parameter
        disorder_parameter = urbach_energy / 25.7  # At room temperature
        
        # Assess fit quality
        if r_value**2 > 0.99:
            fit_quality = "Excellent"
        elif r_value**2 > 0.95:
            fit_quality = "Good"
        elif r_value**2 > 0.90:
            fit_quality = "Fair"
        else:
            fit_quality = "Poor"
            
        return UrbachTailResult(
            urbach_energy=urbach_energy,
            disorder_parameter=disorder_parameter,
            r_squared=r_value**2,
            energy_range=energy_range,
            fit_quality=fit_quality
        )
    
    def calculate_optical_constants(
        self,
        wavelength: np.ndarray,
        transmission: Optional[np.ndarray] = None,
        reflectance: Optional[np.ndarray] = None,
        film_thickness: Optional[float] = None,
        substrate_n: float = 1.5
    ) -> OpticalConstants:
        """
        Calculate optical constants from transmission/reflectance
        
        Args:
            wavelength: Wavelength in nm
            transmission: Transmission spectrum (0-1)
            reflectance: Reflectance spectrum (0-1)
            film_thickness: Film thickness in nm
            substrate_n: Substrate refractive index
            
        Returns:
            OpticalConstants with n, k, α, ε
        """
        if transmission is None and reflectance is None:
            raise ValueError("Need at least transmission or reflectance data")
            
        # Initialize arrays
        n = np.zeros_like(wavelength)
        k = np.zeros_like(wavelength)
        
        if transmission is not None and reflectance is not None:
            # Use both T and R for better accuracy
            # Swanepoel method for thin films
            
            # Find interference extrema
            extrema = self._find_extrema(transmission)
            
            if len(extrema) > 2 and film_thickness:
                # Calculate n from extrema
                n = self._swanepoel_n(
                    wavelength, transmission, extrema, 
                    film_thickness, substrate_n
                )
                
            # Calculate k from T and R
            for i, wl in enumerate(wavelength):
                T = transmission[i]
                R = reflectance[i] if reflectance is not None else 0
                
                # Absorption coefficient
                if T > 0 and T < 1:
                    A = 1 - T - R
                    if A > 0 and film_thickness:
                        alpha_i = -np.log(T) / (film_thickness * 1e-7)
                        k[i] = alpha_i * wl * 1e-7 / (4 * np.pi)
                        
        elif transmission is not None:
            # Only transmission available
            # Simplified calculation
            
            for i, wl in enumerate(wavelength):
                T = transmission[i]
                
                # Estimate n from Fresnel equations
                if T > 0 and T < 1:
                    # Assuming negligible absorption
                    R_est = ((substrate_n - 1) / (substrate_n + 1))**2
                    n[i] = substrate_n * np.sqrt((1 + np.sqrt(T)) / (1 - np.sqrt(T)))
                    
                    # Estimate k from transmission
                    if film_thickness:
                        alpha_i = -np.log(T) / (film_thickness * 1e-7)
                        k[i] = alpha_i * wl * 1e-7 / (4 * np.pi)
                        
        # Calculate derived quantities
        alpha = 4 * np.pi * k / (wavelength * 1e-7)  # cm⁻¹
        epsilon_real = n**2 - k**2
        epsilon_imag = 2 * n * k
        
        return OpticalConstants(
            wavelength=wavelength,
            n=n,
            k=k,
            alpha=alpha,
            epsilon_real=epsilon_real,
            epsilon_imag=epsilon_imag
        )
    
    def _wavelength_to_energy(self, wavelength: np.ndarray) -> np.ndarray:
        """Convert wavelength (nm) to photon energy (eV)"""
        return (self.h * self.c) / (wavelength * 1e-9 * self.e)
    
    def _find_extrema(self, signal: np.ndarray) -> np.ndarray:
        """Find local maxima and minima in signal"""
        # Find peaks and valleys
        peaks, _ = signal.find_peaks(signal)
        valleys, _ = signal.find_peaks(-signal)
        
        # Combine and sort
        extrema = np.sort(np.concatenate([peaks, valleys]))
        
        return extrema
    
    def _swanepoel_n(
        self,
        wavelength: np.ndarray,
        transmission: np.ndarray,
        extrema: np.ndarray,
        thickness: float,
        substrate_n: float
    ) -> np.ndarray:
        """
        Calculate refractive index using Swanepoel method
        
        Based on interference fringes in transmission spectrum
        """
        n = np.ones_like(wavelength) * 1.5  # Initial guess
        
        # Get transmission at extrema
        T_max = []
        T_min = []
        lambda_max = []
        lambda_min = []
        
        for i, idx in enumerate(extrema):
            if i % 2 == 0:  # Maximum
                T_max.append(transmission[idx])
                lambda_max.append(wavelength[idx])
            else:  # Minimum
                T_min.append(transmission[idx])
                lambda_min.append(wavelength[idx])
                
        if len(T_max) > 1 and len(T_min) > 1:
            # Calculate substrate transmission
            T_s = 2 * substrate_n / (substrate_n**2 + 1)
            
            # Calculate refractive index at extrema
            for i in range(min(len(T_max), len(T_min))):
                T_M = T_max[i]
                T_m = T_min[i]
                
                # Substrate corrected values
                N = 2 * substrate_n * (T_M - T_m) / (T_M * T_m) + (substrate_n**2 + 1) / 2
                
                # Refractive index
                n_val = np.sqrt(N + np.sqrt(N**2 - substrate_n**2))
                
                # Interpolate to full wavelength range
                idx_max = np.argmin(np.abs(wavelength - lambda_max[i]))
                idx_min = np.argmin(np.abs(wavelength - lambda_min[i]))
                
                n[idx_max:idx_min] = n_val
                
        return n
    
    def remove_interference_fringes(
        self,
        wavelength: np.ndarray,
        spectrum: np.ndarray,
        method: str = 'fft'
    ) -> np.ndarray:
        """
        Remove thin film interference fringes
        
        Args:
            wavelength: Wavelength array
            spectrum: Spectrum with fringes
            method: 'fft' or 'envelope'
            
        Returns:
            Fringe-corrected spectrum
        """
        if method == 'fft':
            # FFT-based fringe removal
            # Convert to wavenumber space
            wavenumber = 1e7 / wavelength  # cm⁻¹
            
            # Resample to uniform wavenumber spacing
            wn_uniform = np.linspace(wavenumber.min(), wavenumber.max(), len(wavenumber))
            spectrum_uniform = np.interp(wn_uniform, wavenumber[::-1], spectrum[::-1])
            
            # FFT
            fft = np.fft.fft(spectrum_uniform)
            freqs = np.fft.fftfreq(len(spectrum_uniform))
            
            # Filter high frequencies (fringes)
            cutoff = 0.1  # Adjust based on fringe frequency
            fft[np.abs(freqs) > cutoff] = 0
            
            # Inverse FFT
            filtered = np.real(np.fft.ifft(fft))
            
            # Back to wavelength space
            result = np.interp(wavelength, 1e7/wn_uniform[::-1], filtered[::-1])
            
        else:  # envelope method
            # Find upper and lower envelopes
            peaks, _ = signal.find_peaks(spectrum)
            valleys, _ = signal.find_peaks(-spectrum)
            
            if len(peaks) > 2 and len(valleys) > 2:
                # Fit envelopes
                upper_env = np.interp(wavelength, wavelength[peaks], spectrum[peaks])
                lower_env = np.interp(wavelength, wavelength[valleys], spectrum[valleys])
                
                # Average envelope
                result = (upper_env + lower_env) / 2
            else:
                # Not enough fringes, return original
                result = spectrum
                
        return result
    
    def batch_process(
        self,
        file_paths: List[str],
        common_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple spectra with same parameters
        
        Args:
            file_paths: List of spectrum file paths
            common_params: Processing parameters
            
        Returns:
            List of processed results
        """
        results = []
        
        for file_path in file_paths:
            try:
                # Load spectrum (implement based on file format)
                data = self._load_spectrum_file(file_path)
                
                # Process
                processed = self.process_spectrum(
                    data['wavelength'],
                    data['intensity'],
                    **common_params
                )
                
                # Extract band gap if requested
                if common_params.get('extract_band_gap', False):
                    tauc = self.calculate_tauc_plot(
                        processed['wavelength'],
                        processed['absorbance']
                    )
                    processed['band_gap'] = tauc.band_gap
                    
                results.append({
                    'file': file_path,
                    'status': 'success',
                    'data': processed
                })
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                results.append({
                    'file': file_path,
                    'status': 'error',
                    'error': str(e)
                })
                
        return results
    
    def _load_spectrum_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """Load spectrum from file (CSV, TXT, etc.)"""
        # Simplified loader - extend based on actual formats
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return {
                'wavelength': df.iloc[:, 0].values,
                'intensity': df.iloc[:, 1].values
            }
        else:
            # Add more formats as needed
            raise NotImplementedError(f"Format not supported: {file_path}")


# Example usage and validation
if __name__ == "__main__":
    # Create analyzer
    analyzer = UVVisNIRAnalyzer()
    
    # Generate synthetic data for testing
    wavelength = np.linspace(300, 800, 500)
    
    # Simulate GaAs spectrum (Eg = 1.42 eV)
    eg = 1.42  # eV
    hv = analyzer._wavelength_to_energy(wavelength)
    
    # Direct transition absorption
    alpha = np.zeros_like(hv)
    alpha[hv > eg] = 1e4 * np.sqrt(hv[hv > eg] - eg)
    
    # Add some noise
    alpha += np.random.normal(0, 50, len(alpha))
    
    # Convert to absorbance (arbitrary units)
    absorbance = alpha / 1e4
    
    # Process spectrum
    processed = analyzer.process_spectrum(
        wavelength,
        absorbance,
        mode='absorbance',
        baseline_method=BaselineMethod.ALS
    )
    
    # Calculate band gap
    tauc_result = analyzer.calculate_tauc_plot(
        processed['wavelength'],
        processed['absorbance'],
        transition_type=TransitionType.DIRECT
    )
    
    print(f"Extracted band gap: {tauc_result.band_gap:.3f} eV")
    print(f"Expected: 1.42 eV")
    print(f"R-squared: {tauc_result.r_squared:.4f}")
    print(f"Uncertainty: ±{tauc_result.uncertainty:.3f} eV")
    
    # Analyze Urbach tail
    urbach = analyzer.analyze_urbach_tail(
        processed['wavelength'],
        processed['absorbance']
    )
    
    print(f"\nUrbach energy: {urbach.urbach_energy:.1f} meV")
    print(f"Disorder parameter: {urbach.disorder_parameter:.2f}")
    print(f"Fit quality: {urbach.fit_quality}")
