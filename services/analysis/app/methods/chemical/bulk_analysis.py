"""
Session 12: Chemical II (SIMS/RBS/NAA, Chemical Etch) - Complete Implementation
================================================================================

Comprehensive backend implementation for bulk and chemical analysis methods:
- SIMS: Secondary Ion Mass Spectrometry depth profiling
- RBS: Rutherford Backscattering Spectrometry layer fitting
- NAA: Neutron Activation Analysis quantification
- Chemical Etch: Pattern density mapping

Author: Semiconductor Lab Platform Team
Version: 1.0.0
Date: October 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
import json
from datetime import datetime
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter, find_peaks
from scipy.integrate import quad, trapz
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# SIMS (Secondary Ion Mass Spectrometry) Implementation
# ============================================================================

class SIMSMode(Enum):
    """SIMS analysis modes"""
    DYNAMIC = "dynamic"  # Depth profiling
    STATIC = "static"   # Surface analysis
    IMAGING = "imaging"  # 2D/3D mapping


class MatrixEffect(Enum):
    """Matrix effect correction methods"""
    NONE = "none"
    RSF = "relative_sensitivity_factor"
    IMPLANT_STANDARD = "implant_standard"
    MCS = "multi_component_standard"


@dataclass
class SIMSProfile:
    """SIMS depth profile data"""
    time: np.ndarray  # Sputter time (s)
    depth: np.ndarray  # Depth (nm)
    counts: np.ndarray  # Ion counts (cts/s)
    concentration: Optional[np.ndarray] = None  # Atoms/cm³
    element: str = ""
    isotope: int = 0
    matrix: str = "Si"
    mode: SIMSMode = SIMSMode.DYNAMIC
    primary_ion: str = "O2+"
    primary_energy: float = 1.0  # keV
    current: float = 100.0  # nA
    raster_size: float = 250.0  # μm
    analysis_area: float = 60.0  # μm
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SIMSCalibration:
    """SIMS calibration data"""
    element: str
    matrix: str
    rsf: float  # Relative sensitivity factor
    rsf_uncertainty: float = 0.0
    reference_concentration: float = 1e20  # atoms/cm³
    sputter_rate: float = 1.0  # nm/s
    calibration_date: str = ""
    standard_name: str = ""
    notes: str = ""


class SIMSAnalyzer:
    """
    SIMS depth profiling and quantification analyzer
    
    Capabilities:
    - Depth profile conversion (time → depth)
    - Quantification (counts → concentration)
    - Matrix effect corrections
    - Detection limit estimation
    - Interface detection
    - Mixing region analysis
    """
    
    def __init__(self):
        self.calibrations: Dict[str, SIMSCalibration] = {}
        self._load_default_calibrations()
    
    def _load_default_calibrations(self):
        """Load default RSF values for common dopants in Si"""
        default_rsf = {
            ("B", "Si"): 1.8e21,
            ("P", "Si"): 3.0e21,
            ("As", "Si"): 2.5e21,
            ("Sb", "Si"): 2.0e21,
            ("In", "Si"): 1.5e21,
            ("Ga", "Si"): 1.7e21,
            ("Al", "Si"): 2.2e21,
            ("N", "Si"): 5.0e21,
            ("O", "Si"): 1.0e22,
            ("C", "Si"): 3.5e21,
        }
        
        for (element, matrix), rsf in default_rsf.items():
            key = f"{element}_{matrix}"
            self.calibrations[key] = SIMSCalibration(
                element=element,
                matrix=matrix,
                rsf=rsf,
                rsf_uncertainty=rsf * 0.15,  # ~15% uncertainty
                sputter_rate=1.0,
                standard_name="default"
            )
    
    def add_calibration(self, calibration: SIMSCalibration):
        """Add or update calibration"""
        key = f"{calibration.element}_{calibration.matrix}"
        self.calibrations[key] = calibration
    
    def convert_time_to_depth(
        self,
        profile: SIMSProfile,
        sputter_rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Convert sputter time to depth
        
        Args:
            profile: SIMS profile data
            sputter_rate: Sputter rate in nm/s (if None, use calibration)
        
        Returns:
            Depth array in nm
        """
        if sputter_rate is None:
            cal_key = f"{profile.element}_{profile.matrix}"
            if cal_key in self.calibrations:
                sputter_rate = self.calibrations[cal_key].sputter_rate
            else:
                sputter_rate = 1.0  # Default
        
        depth = profile.time * sputter_rate
        return depth
    
    def quantify_profile(
        self,
        profile: SIMSProfile,
        method: MatrixEffect = MatrixEffect.RSF,
        calibration: Optional[SIMSCalibration] = None
    ) -> np.ndarray:
        """
        Convert ion counts to concentration
        
        Args:
            profile: SIMS profile with counts
            method: Matrix effect correction method
            calibration: Calibration data (if None, use stored)
        
        Returns:
            Concentration array in atoms/cm³
        """
        # Get calibration
        if calibration is None:
            cal_key = f"{profile.element}_{profile.matrix}"
            if cal_key not in self.calibrations:
                raise ValueError(f"No calibration found for {cal_key}")
            calibration = self.calibrations[cal_key]
        
        if method == MatrixEffect.RSF:
            # Simple RSF conversion: C = RSF * (I_analyte / I_matrix)
            # For simplicity, assuming matrix counts are normalized
            concentration = calibration.rsf * profile.counts
        
        elif method == MatrixEffect.IMPLANT_STANDARD:
            # Use implant standard with known dose
            # Integrate counts and match to dose
            total_counts = trapz(profile.counts, profile.time)
            known_dose = calibration.reference_concentration  # atoms/cm²
            conversion_factor = known_dose / total_counts
            concentration = profile.counts * conversion_factor
        
        else:  # NONE
            concentration = profile.counts.copy()
        
        return concentration
    
    def estimate_detection_limit(
        self,
        profile: SIMSProfile,
        background_region: Tuple[int, int] = (0, 10)
    ) -> float:
        """
        Estimate detection limit (3σ of background)
        
        Args:
            profile: SIMS profile
            background_region: Indices for background region
        
        Returns:
            Detection limit in atoms/cm³
        """
        start, end = background_region
        background = profile.counts[start:end]
        
        noise_level = 3 * np.std(background)
        
        # Convert to concentration if quantified
        if profile.concentration is not None:
            cal_key = f"{profile.element}_{profile.matrix}"
            if cal_key in self.calibrations:
                rsf = self.calibrations[cal_key].rsf
                detection_limit = noise_level * rsf
            else:
                detection_limit = noise_level
        else:
            detection_limit = noise_level
        
        return float(detection_limit)
    
    def find_interfaces(
        self,
        profile: SIMSProfile,
        threshold_factor: float = 0.5
    ) -> List[Dict[str, float]]:
        """
        Detect interfaces in depth profile
        
        Args:
            profile: SIMS profile
            threshold_factor: Threshold as fraction of max concentration
        
        Returns:
            List of interface positions and widths
        """
        data = profile.concentration if profile.concentration is not None else profile.counts
        depth = profile.depth
        
        # Smooth data
        smoothed = gaussian_filter1d(data, sigma=2)
        
        # Find gradient
        gradient = np.abs(np.gradient(smoothed, depth))
        
        # Find peaks in gradient (interfaces)
        threshold = threshold_factor * np.max(gradient)
        peaks, properties = find_peaks(gradient, height=threshold, distance=10)
        
        interfaces = []
        for peak in peaks:
            # Estimate interface width (FWHM of gradient peak)
            peak_height = gradient[peak]
            half_max = peak_height / 2
            
            # Find left and right half-max points
            left_idx = peak
            while left_idx > 0 and gradient[left_idx] > half_max:
                left_idx -= 1
            
            right_idx = peak
            while right_idx < len(gradient) - 1 and gradient[right_idx] > half_max:
                right_idx += 1
            
            width = depth[right_idx] - depth[left_idx]
            
            interfaces.append({
                'depth': float(depth[peak]),
                'width': float(width),
                'gradient': float(peak_height),
                'concentration': float(data[peak])
            })
        
        return interfaces
    
    def calculate_dose(
        self,
        profile: SIMSProfile,
        depth_range: Optional[Tuple[float, float]] = None
    ) -> float:
        """
        Calculate integrated dose (atoms/cm²)
        
        Args:
            profile: SIMS profile with concentration
            depth_range: Optional depth range (nm) for integration
        
        Returns:
            Dose in atoms/cm²
        """
        if profile.concentration is None:
            raise ValueError("Profile must be quantified first")
        
        depth = profile.depth
        conc = profile.concentration
        
        if depth_range is not None:
            min_depth, max_depth = depth_range
            mask = (depth >= min_depth) & (depth <= max_depth)
            depth = depth[mask]
            conc = conc[mask]
        
        # Integrate concentration over depth (convert nm to cm)
        dose = trapz(conc, depth * 1e-7)  # nm → cm
        
        return float(dose)


# ============================================================================
# RBS (Rutherford Backscattering Spectrometry) Implementation
# ============================================================================

class RBSGeometry(Enum):
    """RBS scattering geometry"""
    CORNELL = "cornell"  # θ = 170°
    IBM = "ibm"  # θ = 160°
    CUSTOM = "custom"


@dataclass
class RBSSpectrum:
    """RBS spectrum data"""
    energy: np.ndarray  # Detected energy (keV)
    counts: np.ndarray  # Yield (counts)
    channel: Optional[np.ndarray] = None  # ADC channel
    incident_energy: float = 2000.0  # keV (typically He+ 2 MeV)
    scattering_angle: float = 170.0  # degrees
    detector_solid_angle: float = 3.0  # msr
    incident_charge: float = 10.0  # μC
    target_normal_angle: float = 0.0  # degrees
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RBSLayer:
    """RBS layer model"""
    element: str
    atomic_fraction: float  # 0-1
    thickness: float  # 1e15 atoms/cm² (areal density)
    density: float = 5.0e22  # atoms/cm³
    
    def thickness_nm(self) -> float:
        """Convert areal density to thickness in nm"""
        return self.thickness / (self.density * 1e-7)


@dataclass
class RBSFitResult:
    """RBS fitting results"""
    layers: List[RBSLayer]
    simulated_spectrum: np.ndarray
    chi_squared: float
    r_factor: float
    fit_range: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


class RBSAnalyzer:
    """
    RBS spectrum analysis and layer fitting
    
    Capabilities:
    - Kinematic factor calculation
    - Energy loss computation
    - Layer composition fitting
    - Stopping power integration
    - Rutherford and non-Rutherford cross-sections
    """
    
    # Atomic masses (amu)
    ATOMIC_MASSES = {
        'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012,
        'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
        'Si': 28.086, 'P': 30.974, 'S': 32.065, 'Ti': 47.867,
        'Cr': 51.996, 'Fe': 55.845, 'Ni': 58.693, 'Cu': 63.546,
        'Ga': 69.723, 'As': 74.922, 'Ge': 72.64, 'Zr': 91.224,
        'Hf': 178.49, 'Ta': 180.948, 'W': 183.84, 'Au': 196.967,
    }
    
    def __init__(self, projectile: str = "He", projectile_mass: float = 4.003):
        self.projectile = projectile
        self.projectile_mass = projectile_mass
        self.projectile_z = 2  # He
    
    def kinematic_factor(
        self,
        target_mass: float,
        scattering_angle: float
    ) -> float:
        """
        Calculate kinematic factor K
        
        K = [(M1*cos(θ) + sqrt(M2² - M1²sin²(θ))) / (M1 + M2)]²
        
        Args:
            target_mass: Target atom mass (amu)
            scattering_angle: Scattering angle (degrees)
        
        Returns:
            Kinematic factor (dimensionless)
        """
        M1 = self.projectile_mass
        M2 = target_mass
        theta = np.radians(scattering_angle)
        
        if M2 < M1:
            # Light target - forward scattering only
            return np.nan
        
        numerator = M1 * np.cos(theta) + np.sqrt(M2**2 - M1**2 * np.sin(theta)**2)
        denominator = M1 + M2
        K = (numerator / denominator)**2
        
        return K
    
    def stopping_power(
        self,
        energy: float,
        element: str,
        density: float = None
    ) -> float:
        """
        Calculate stopping power using Bethe-Bloch (simplified)
        
        Args:
            energy: Projectile energy (keV)
            element: Target element
            density: Atomic density (atoms/cm³)
        
        Returns:
            Stopping power (keV/(atoms/cm²))
        """
        # Simplified stopping power (fit to tabulated values)
        # S ≈ A * Z₂² * (1/E) for E > 100 keV
        
        Z2 = self._get_atomic_number(element)
        
        # Empirical coefficient (order of magnitude)
        A = 100.0  # keV·(atoms/cm²)/(atomic number)²
        
        S = A * Z2**2 / energy
        
        return S
    
    def _get_atomic_number(self, element: str) -> int:
        """Get atomic number from element symbol"""
        periodic_table = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6,
            'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12,
            'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
            'K': 19, 'Ca': 20, 'Ti': 22, 'Cr': 24, 'Fe': 26, 'Ni': 28,
            'Cu': 29, 'Ga': 31, 'As': 33, 'Ge': 32, 'Zr': 40, 'Hf': 72,
            'Ta': 73, 'W': 74, 'Au': 79,
        }
        return periodic_table.get(element, 14)  # Default to Si
    
    def rutherford_cross_section(
        self,
        energy: float,
        target_z: int,
        scattering_angle: float
    ) -> float:
        """
        Calculate Rutherford scattering cross-section
        
        dσ/dΩ = (Z₁Z₂e²/4E)² * 1/sin⁴(θ/2)
        
        Args:
            energy: Projectile energy (keV)
            target_z: Target atomic number
            scattering_angle: Scattering angle (degrees)
        
        Returns:
            Cross-section (barn/sr)
        """
        theta = np.radians(scattering_angle)
        
        # Constants
        Z1 = self.projectile_z
        Z2 = target_z
        e_squared = 1.44  # MeV·fm (in natural units)
        
        # Convert energy to MeV
        E_MeV = energy / 1000.0
        
        # Rutherford formula
        numerator = (Z1 * Z2 * e_squared / (4 * E_MeV))**2
        denominator = np.sin(theta / 2)**4
        
        dsigma_dOmega = numerator / denominator
        
        # Convert fm²/sr to barn/sr (1 barn = 100 fm²)
        dsigma_dOmega_barn = dsigma_dOmega / 100.0
        
        return dsigma_dOmega_barn
    
    def simulate_spectrum(
        self,
        layers: List[RBSLayer],
        spectrum: RBSSpectrum,
        energy_resolution: float = 15.0  # keV FWHM
    ) -> np.ndarray:
        """
        Simulate RBS spectrum for given layer structure
        
        Args:
            layers: List of layers (from surface to substrate)
            spectrum: Spectrum template (for energy axis)
            energy_resolution: Detector resolution (keV FWHM)
        
        Returns:
            Simulated yield array
        """
        E0 = spectrum.incident_energy
        theta = spectrum.scattering_angle
        
        simulated = np.zeros_like(spectrum.counts, dtype=float)
        
        # Process each layer
        depth = 0.0  # Cumulative depth (1e15 atoms/cm²)
        
        for layer in layers:
            # Get element properties
            element = layer.element
            M2 = self.ATOMIC_MASSES.get(element, 28.0)
            Z2 = self._get_atomic_number(element)
            
            # Calculate kinematic factor
            K = self.kinematic_factor(M2, theta)
            if np.isnan(K):
                continue
            
            # Surface energy for this layer
            E_surface = E0 * K
            
            # Energy at layer depth (account for energy loss)
            # Simplified: constant stopping power
            S_in = self.stopping_power(E0, element)
            S_out = self.stopping_power(E_surface, element)
            
            # Energy loss in layer
            dE = (S_in + S_out) * layer.thickness / 2
            
            # Energy range for this layer
            E_front = E_surface
            E_back = E_surface - dE
            
            # Add contribution to spectrum
            # Simplified: uniform distribution across energy range
            mask = (spectrum.energy >= E_back) & (spectrum.energy <= E_front)
            
            # Calculate yield (simplified)
            # Y ∝ σ * Ω * Q * N * Δx
            cross_section = self.rutherford_cross_section(E0, Z2, theta)
            solid_angle = spectrum.detector_solid_angle  # msr
            charge = spectrum.incident_charge  # μC
            
            # Incident particles
            N_incident = charge * 1e-6 / 1.602e-19  # Convert μC to particles
            
            # Yield per channel
            yield_factor = (cross_section * 1e-27 * solid_angle * 1e-3 * 
                          N_incident * layer.atomic_fraction * layer.thickness * 1e15)
            
            # Distribute over energy channels
            n_channels = np.sum(mask)
            if n_channels > 0:
                simulated[mask] += yield_factor / n_channels
            
            depth += layer.thickness
        
        # Apply energy resolution (Gaussian broadening)
        sigma = energy_resolution / 2.355  # FWHM to σ
        sigma_channels = sigma / np.mean(np.diff(spectrum.energy))
        simulated = gaussian_filter1d(simulated, sigma=sigma_channels)
        
        return simulated
    
    def fit_spectrum(
        self,
        spectrum: RBSSpectrum,
        initial_layers: List[RBSLayer],
        fit_range: Optional[Tuple[float, float]] = None,
        fix_composition: bool = False
    ) -> RBSFitResult:
        """
        Fit layer structure to experimental spectrum
        
        Args:
            spectrum: Experimental RBS spectrum
            initial_layers: Initial layer structure guess
            fit_range: Energy range for fitting (keV)
            fix_composition: If True, only fit thicknesses
        
        Returns:
            Fit results with optimized layers
        """
        # Define fit range
        if fit_range is None:
            fit_range = (spectrum.energy.min(), spectrum.energy.max())
        
        mask = (spectrum.energy >= fit_range[0]) & (spectrum.energy <= fit_range[1])
        energy_fit = spectrum.energy[mask]
        counts_fit = spectrum.counts[mask]
        
        # Pack parameters for optimization
        def pack_params(layers):
            params = []
            for layer in layers:
                params.append(layer.thickness)
                if not fix_composition:
                    params.append(layer.atomic_fraction)
            return np.array(params)
        
        def unpack_params(params, template_layers):
            layers = []
            idx = 0
            for template in template_layers:
                thickness = params[idx]
                idx += 1
                
                if fix_composition:
                    fraction = template.atomic_fraction
                else:
                    fraction = params[idx]
                    idx += 1
                
                layers.append(RBSLayer(
                    element=template.element,
                    atomic_fraction=fraction,
                    thickness=thickness,
                    density=template.density
                ))
            return layers
        
        # Objective function
        def objective(params):
            layers = unpack_params(params, initial_layers)
            
            # Simulate spectrum
            sim_full = self.simulate_spectrum(layers, spectrum)
            sim_fit = sim_full[mask]
            
            # Chi-squared
            # Add small constant to avoid division by zero
            variance = np.maximum(counts_fit, 1.0)
            chi_sq = np.sum((counts_fit - sim_fit)**2 / variance)
            
            # Add constraints (fractions sum to 1, positive values)
            penalty = 0.0
            if not fix_composition:
                for layer in layers:
                    if layer.atomic_fraction < 0 or layer.atomic_fraction > 1:
                        penalty += 1e6
            
            return chi_sq + penalty
        
        # Initial guess
        x0 = pack_params(initial_layers)
        
        # Bounds
        bounds = []
        for _ in initial_layers:
            bounds.append((0.1, 1000.0))  # Thickness: 0.1-1000 (1e15 at/cm²)
            if not fix_composition:
                bounds.append((0.0, 1.0))  # Fraction: 0-1
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        # Extract results
        fitted_layers = unpack_params(result.x, initial_layers)
        simulated = self.simulate_spectrum(fitted_layers, spectrum)
        
        # Calculate goodness of fit
        chi_squared = result.fun
        
        # R-factor
        r_factor = np.sum(np.abs(counts_fit - simulated[mask])) / np.sum(counts_fit)
        
        fit_indices = np.where(mask)[0]
        fit_range_indices = (int(fit_indices[0]), int(fit_indices[-1]))
        
        return RBSFitResult(
            layers=fitted_layers,
            simulated_spectrum=simulated,
            chi_squared=float(chi_squared),
            r_factor=float(r_factor),
            fit_range=fit_range_indices,
            metadata={
                'convergence': bool(result.success),
                'iterations': int(result.nit),
                'message': result.message
            }
        )


# ============================================================================
# NAA (Neutron Activation Analysis) Implementation
# ============================================================================

@dataclass
class NAADecayCurve:
    """NAA decay curve data"""
    time: np.ndarray  # Time after irradiation (s)
    counts: np.ndarray  # Counts per measurement
    live_time: np.ndarray  # Live time per measurement (s)
    energy: float  # Gamma energy (keV)
    element: str = ""
    isotope: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NAANuclearData:
    """Nuclear data for NAA"""
    element: str
    isotope: str  # e.g., "Co-60"
    half_life: float  # seconds
    gamma_energy: float  # keV
    gamma_intensity: float  # 0-1
    thermal_cross_section: float  # barns
    activation_product: str  # e.g., "Co-60"
    notes: str = ""


@dataclass
class NAAQuantResult:
    """NAA quantification result"""
    element: str
    concentration: float  # μg/g or atoms/g
    uncertainty: float
    detection_limit: float
    isotope: str
    decay_constant: float
    activity: float  # Bq
    metadata: Dict[str, Any] = field(default_factory=dict)


class NAAAnalyzer:
    """
    Neutron Activation Analysis quantification
    
    Capabilities:
    - Decay curve fitting
    - Comparator method quantification
    - Interference corrections
    - Detection limit estimation
    - Multi-isotope analysis
    """
    
    def __init__(self):
        self.nuclear_data: Dict[str, NAANuclearData] = {}
        self._load_nuclear_data()
    
    def _load_nuclear_data(self):
        """Load common NAA nuclear data"""
        common_isotopes = [
            NAANuclearData(
                element="Na", isotope="Na-24", half_life=53996.0,
                gamma_energy=1368.6, gamma_intensity=1.0,
                thermal_cross_section=0.53, activation_product="Na-24"
            ),
            NAANuclearData(
                element="Mn", isotope="Mn-56", half_life=9287.0,
                gamma_energy=846.8, gamma_intensity=0.989,
                thermal_cross_section=13.3, activation_product="Mn-56"
            ),
            NAANuclearData(
                element="Cu", isotope="Cu-64", half_life=45720.0,
                gamma_energy=1345.8, gamma_intensity=0.005,
                thermal_cross_section=4.5, activation_product="Cu-64"
            ),
            NAANuclearData(
                element="As", isotope="As-76", half_life=95040.0,
                gamma_energy=559.1, gamma_intensity=0.45,
                thermal_cross_section=4.5, activation_product="As-76"
            ),
            NAANuclearData(
                element="Br", isotope="Br-82", half_life=126230.0,
                gamma_energy=776.5, gamma_intensity=0.835,
                thermal_cross_section=6.8, activation_product="Br-82"
            ),
            NAANuclearData(
                element="Au", isotope="Au-198", half_life=232992.0,
                gamma_energy=411.8, gamma_intensity=0.955,
                thermal_cross_section=98.65, activation_product="Au-198"
            ),
        ]
        
        for data in common_isotopes:
            self.nuclear_data[data.element] = data
    
    def decay_constant(self, half_life: float) -> float:
        """Calculate decay constant from half-life"""
        return np.log(2) / half_life
    
    def fit_decay_curve(
        self,
        curve: NAADecayCurve,
        half_life: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Fit exponential decay to data
        
        N(t) = N₀ * exp(-λt)
        
        Args:
            curve: Decay curve data
            half_life: Known half-life (s), if None use fit
        
        Returns:
            Dict with N0, lambda, and fit quality
        """
        # Correct for live time
        count_rate = curve.counts / curve.live_time
        
        # Remove zeros/negatives
        mask = count_rate > 0
        t = curve.time[mask]
        y = count_rate[mask]
        
        if half_life is not None:
            # Fixed decay constant
            lambda_fixed = self.decay_constant(half_life)
            
            # Linear fit: ln(N) = ln(N0) - λt
            ln_y = np.log(y)
            
            # Weighted least squares (weight by 1/σ²)
            weights = y  # Poisson: σ² = N
            
            N0 = np.exp(np.average(ln_y + lambda_fixed * t, weights=weights))
            
            # Calculate residuals
            y_fit = N0 * np.exp(-lambda_fixed * t)
            residuals = y - y_fit
            chi_squared = np.sum((residuals**2) / np.maximum(y, 1.0))
            
            return {
                'N0': float(N0),
                'lambda': float(lambda_fixed),
                'half_life': float(half_life),
                'chi_squared': float(chi_squared),
                'fixed_lambda': True
            }
        
        else:
            # Fit both N0 and lambda
            def decay_func(t, N0, lam):
                return N0 * np.exp(-lam * t)
            
            # Initial guess
            p0 = [y[0], 1e-5]
            
            try:
                popt, pcov = curve_fit(
                    decay_func, t, y,
                    p0=p0,
                    sigma=np.sqrt(np.maximum(y, 1.0)),
                    absolute_sigma=True,
                    maxfev=10000
                )
                
                N0, lam = popt
                N0_err, lam_err = np.sqrt(np.diag(pcov))
                
                # Calculate chi-squared
                y_fit = decay_func(t, *popt)
                chi_squared = np.sum(((y - y_fit)**2) / np.maximum(y, 1.0))
                
                return {
                    'N0': float(N0),
                    'lambda': float(lam),
                    'half_life': float(np.log(2) / lam),
                    'N0_uncertainty': float(N0_err),
                    'lambda_uncertainty': float(lam_err),
                    'chi_squared': float(chi_squared),
                    'fixed_lambda': False
                }
            
            except Exception as e:
                # Fit failed, return simple estimate
                return {
                    'N0': float(y[0]),
                    'lambda': 1e-5,
                    'half_life': np.inf,
                    'chi_squared': np.inf,
                    'fixed_lambda': False,
                    'error': str(e)
                }
    
    def comparator_method(
        self,
        sample_curve: NAADecayCurve,
        standard_curve: NAADecayCurve,
        standard_mass: float,
        sample_mass: float,
        standard_concentration: float,
        element: str
    ) -> NAAQuantResult:
        """
        Quantify using comparator method
        
        C_sample = C_std * (A_sample / A_std) * (m_std / m_sample)
        
        Args:
            sample_curve: Sample decay curve
            standard_curve: Standard decay curve
            standard_mass: Standard mass (g)
            sample_mass: Sample mass (g)
            standard_concentration: Known concentration in standard (μg/g)
            element: Element symbol
        
        Returns:
            Quantification result
        """
        # Get nuclear data
        if element not in self.nuclear_data:
            raise ValueError(f"No nuclear data for {element}")
        
        nuc_data = self.nuclear_data[element]
        
        # Fit decay curves
        fit_sample = self.fit_decay_curve(sample_curve, nuc_data.half_life)
        fit_standard = self.fit_decay_curve(standard_curve, nuc_data.half_life)
        
        # Extract activities (N0)
        A_sample = fit_sample['N0']
        A_standard = fit_standard['N0']
        
        # Calculate concentration
        concentration = (standard_concentration * 
                        (A_sample / A_standard) * 
                        (standard_mass / sample_mass))
        
        # Estimate uncertainty (simplified - propagate counting statistics)
        # δC/C = sqrt((δA_s/A_s)² + (δA_std/A_std)²)
        if 'N0_uncertainty' in fit_sample:
            rel_unc_sample = fit_sample['N0_uncertainty'] / A_sample
        else:
            rel_unc_sample = 1.0 / np.sqrt(A_sample)  # Poisson
        
        if 'N0_uncertainty' in fit_standard:
            rel_unc_std = fit_standard['N0_uncertainty'] / A_standard
        else:
            rel_unc_std = 1.0 / np.sqrt(A_standard)
        
        rel_uncertainty = np.sqrt(rel_unc_sample**2 + rel_unc_std**2)
        uncertainty = concentration * rel_uncertainty
        
        # Detection limit (3σ of background)
        # Simplified: based on counting statistics
        background_rate = sample_curve.counts[-5:].mean() / sample_curve.live_time[-5:].mean()
        detection_limit_counts = 3 * np.sqrt(background_rate * sample_curve.live_time[0])
        detection_limit = (standard_concentration * 
                          (detection_limit_counts / A_standard) * 
                          (standard_mass / sample_mass))
        
        return NAAQuantResult(
            element=element,
            concentration=float(concentration),
            uncertainty=float(uncertainty),
            detection_limit=float(detection_limit),
            isotope=nuc_data.isotope,
            decay_constant=fit_sample['lambda'],
            activity=float(A_sample),
            metadata={
                'sample_fit': fit_sample,
                'standard_fit': fit_standard,
                'gamma_energy': nuc_data.gamma_energy
            }
        )


# ============================================================================
# Chemical Etch Mapping Implementation
# ============================================================================

@dataclass
class EtchProfile:
    """Chemical etch profile data"""
    pattern_density: np.ndarray  # % (0-100)
    etch_rate: np.ndarray  # nm/min
    position_x: Optional[np.ndarray] = None  # mm (for spatial mapping)
    position_y: Optional[np.ndarray] = None  # mm
    chemistry: str = "KOH"
    temperature: float = 80.0  # °C
    concentration: float = 30.0  # wt%
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadingEffect:
    """Loading effect analysis results"""
    nominal_rate: float  # nm/min (at 0% pattern density)
    max_reduction: float  # % (at 100% pattern density)
    critical_density: float  # % (where rate drops to 50%)
    model_type: str = "linear"
    coefficients: Dict[str, float] = field(default_factory=dict)
    r_squared: float = 0.0


class ChemicalEtchAnalyzer:
    """
    Chemical etch mapping and loading effect analysis
    
    Capabilities:
    - Etch rate vs pattern density modeling
    - Loading effect quantification
    - Spatial uniformity analysis
    - Micro-loading vs macro-loading
    """
    
    def fit_loading_effect(
        self,
        profile: EtchProfile,
        model: str = "linear"
    ) -> LoadingEffect:
        """
        Fit loading effect model
        
        Models:
        - linear: R = R₀(1 - α*D)
        - exponential: R = R₀*exp(-α*D)
        - power: R = R₀*(1 - D)^α
        
        Args:
            profile: Etch profile data
            model: Model type
        
        Returns:
            Loading effect parameters
        """
        D = profile.pattern_density / 100.0  # Convert % to fraction
        R = profile.etch_rate
        
        if model == "linear":
            # R = R0(1 - α*D)
            # Linear regression: R = a + b*D
            coeffs = np.polyfit(D, R, deg=1)
            a, b = coeffs
            
            R0 = a  # Intercept = R at D=0
            alpha = -b / a  # Slope / intercept
            
            R_fit = np.polyval(coeffs, D)
            r_squared = 1 - np.sum((R - R_fit)**2) / np.sum((R - R.mean())**2)
            
            # Find critical density (50% reduction)
            critical_density = 0.5 / alpha if alpha > 0 else np.inf
            
            # Max reduction at D=1
            max_reduction = alpha * 100.0
            
            return LoadingEffect(
                nominal_rate=float(R0),
                max_reduction=float(max_reduction),
                critical_density=float(critical_density * 100),
                model_type="linear",
                coefficients={'R0': float(R0), 'alpha': float(alpha)},
                r_squared=float(r_squared)
            )
        
        elif model == "exponential":
            # R = R0*exp(-α*D)
            # Linearize: ln(R) = ln(R0) - α*D
            ln_R = np.log(np.maximum(R, 1e-6))
            coeffs = np.polyfit(D, ln_R, deg=1)
            ln_R0, minus_alpha = coeffs
            
            R0 = np.exp(ln_R0)
            alpha = -minus_alpha
            
            R_fit = R0 * np.exp(-alpha * D)
            r_squared = 1 - np.sum((R - R_fit)**2) / np.sum((R - R.mean())**2)
            
            # Critical density (50% reduction)
            critical_density = np.log(2) / alpha if alpha > 0 else np.inf
            
            # Max reduction at D=1
            max_reduction = (1 - np.exp(-alpha)) * 100.0
            
            return LoadingEffect(
                nominal_rate=float(R0),
                max_reduction=float(max_reduction),
                critical_density=float(critical_density * 100),
                model_type="exponential",
                coefficients={'R0': float(R0), 'alpha': float(alpha)},
                r_squared=float(r_squared)
            )
        
        elif model == "power":
            # R = R0*(1 - D)^α
            # Linearize: ln(R) = ln(R0) + α*ln(1-D)
            mask = D < 0.99  # Avoid log(0)
            D_masked = D[mask]
            R_masked = R[mask]
            
            ln_R = np.log(np.maximum(R_masked, 1e-6))
            ln_one_minus_D = np.log(1 - D_masked)
            
            coeffs = np.polyfit(ln_one_minus_D, ln_R, deg=1)
            alpha, ln_R0 = coeffs
            
            R0 = np.exp(ln_R0)
            
            R_fit = R0 * (1 - D)**alpha
            r_squared = 1 - np.sum((R - R_fit)**2) / np.sum((R - R.mean())**2)
            
            # Critical density (50% reduction)
            critical_density = 1 - 0.5**(1/alpha) if alpha > 0 else np.inf
            
            # Max reduction at D=1
            max_reduction = 100.0  # Full reduction
            
            return LoadingEffect(
                nominal_rate=float(R0),
                max_reduction=float(max_reduction),
                critical_density=float(critical_density * 100),
                model_type="power",
                coefficients={'R0': float(R0), 'alpha': float(alpha)},
                r_squared=float(r_squared)
            )
        
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def calculate_uniformity(
        self,
        profile: EtchProfile
    ) -> Dict[str, float]:
        """
        Calculate etch uniformity metrics
        
        Args:
            profile: Etch profile with spatial data
        
        Returns:
            Uniformity metrics (%, 1σ, 3σ, range)
        """
        R = profile.etch_rate
        
        mean_rate = np.mean(R)
        std_rate = np.std(R)
        min_rate = np.min(R)
        max_rate = np.max(R)
        
        # Uniformity = (1 - σ/mean) * 100%
        uniformity_1sigma = (1 - std_rate / mean_rate) * 100
        
        # Uniformity 3σ = (1 - 3σ/mean) * 100%
        uniformity_3sigma = (1 - 3*std_rate / mean_rate) * 100
        
        # Range uniformity = (1 - (max-min)/(2*mean)) * 100%
        uniformity_range = (1 - (max_rate - min_rate) / (2 * mean_rate)) * 100
        
        return {
            'mean_rate': float(mean_rate),
            'std_rate': float(std_rate),
            'uniformity_1sigma': float(uniformity_1sigma),
            'uniformity_3sigma': float(uniformity_3sigma),
            'uniformity_range': float(uniformity_range),
            'min_rate': float(min_rate),
            'max_rate': float(max_rate),
            'cv_percent': float(std_rate / mean_rate * 100)
        }


# ============================================================================
# Simulator for Test Data Generation
# ============================================================================

class ChemicalBulkSimulator:
    """Generate realistic test data for SIMS, RBS, NAA, and etch"""
    
    @staticmethod
    def simulate_sims_profile(
        element: str = "B",
        matrix: str = "Si",
        peak_depth: float = 100.0,  # nm
        peak_concentration: float = 1e20,  # atoms/cm³
        dose: float = 1e15,  # atoms/cm²
        straggle: float = 30.0,  # nm
        background: float = 1e16,  # atoms/cm³
        n_points: int = 200,
        noise_level: float = 0.05
    ) -> SIMSProfile:
        """Generate Gaussian implant profile"""
        
        # Depth axis
        depth = np.linspace(0, 500, n_points)
        time = depth / 1.0  # 1 nm/s sputter rate
        
        # Gaussian profile
        sigma = straggle
        profile_gauss = (dose / (sigma * np.sqrt(2 * np.pi)) * 
                        np.exp(-((depth - peak_depth)**2) / (2 * sigma**2)))
        
        # Convert to concentration (atoms/cm³)
        # Note: dose is in atoms/cm², need to multiply by factor
        concentration = profile_gauss * 1e7 + background  # nm to cm conversion
        
        # Add noise
        noise = 1 + noise_level * np.random.randn(n_points)
        concentration = concentration * noise
        concentration = np.maximum(concentration, background)
        
        # Convert to counts (for SIMS, counts ∝ concentration / RSF)
        # Assume RSF ~ 1e21 for simplicity
        counts = concentration / 1e21
        
        return SIMSProfile(
            time=time,
            depth=depth,
            counts=counts,
            concentration=concentration,
            element=element,
            matrix=matrix,
            metadata={
                'simulated': True,
                'peak_depth': peak_depth,
                'peak_concentration': float(peak_concentration),
                'dose': dose,
                'straggle': straggle
            }
        )
    
    @staticmethod
    def simulate_rbs_spectrum(
        layers: List[Tuple[str, float, float]],  # (element, fraction, thickness)
        substrate: str = "Si",
        E0: float = 2000.0,  # keV
        theta: float = 170.0,  # degrees
        n_channels: int = 512,
        noise_level: float = 0.03
    ) -> RBSSpectrum:
        """Generate RBS spectrum for multilayer structure"""
        
        # Energy axis
        energy = np.linspace(500, E0 * 0.95, n_channels)
        channel = np.arange(n_channels)
        
        # Initialize counts
        counts = np.zeros(n_channels)
        
        # Use RBS analyzer for simulation
        analyzer = RBSAnalyzer(projectile="He", projectile_mass=4.003)
        
        # Convert input to RBSLayer objects
        layer_objects = []
        for element, fraction, thickness in layers:
            layer_objects.append(RBSLayer(
                element=element,
                atomic_fraction=fraction,
                thickness=thickness,  # 1e15 atoms/cm²
                density=5.0e22
            ))
        
        # Add substrate
        layer_objects.append(RBSLayer(
            element=substrate,
            atomic_fraction=1.0,
            thickness=1000.0,  # Thick substrate
            density=5.0e22
        ))
        
        # Create spectrum template
        spectrum = RBSSpectrum(
            energy=energy,
            counts=counts,
            channel=channel,
            incident_energy=E0,
            scattering_angle=theta
        )
        
        # Simulate
        counts_sim = analyzer.simulate_spectrum(layer_objects, spectrum)
        
        # Add Poisson noise
        counts_sim = np.random.poisson(np.maximum(counts_sim, 1.0))
        
        # Add background
        background = 10 + 5 * np.exp(-energy / 500)
        counts_final = counts_sim + background
        
        return RBSSpectrum(
            energy=energy,
            counts=counts_final,
            channel=channel,
            incident_energy=E0,
            scattering_angle=theta,
            metadata={
                'simulated': True,
                'layers': layers,
                'substrate': substrate
            }
        )
    
    @staticmethod
    def simulate_naa_decay(
        element: str = "Au",
        initial_activity: float = 10000.0,  # counts/s
        irradiation_time: float = 3600.0,  # s
        cooling_time: float = 600.0,  # s
        measurement_time: float = 3600.0,  # s
        n_measurements: int = 20,
        background: float = 50.0,  # counts/s
        noise_level: float = 0.1
    ) -> NAADecayCurve:
        """Generate NAA decay curve"""
        
        # Get nuclear data
        analyzer = NAAAnalyzer()
        if element not in analyzer.nuclear_data:
            raise ValueError(f"No nuclear data for {element}")
        
        nuc_data = analyzer.nuclear_data[element]
        lambda_decay = analyzer.decay_constant(nuc_data.half_life)
        
        # Time points (after cooling)
        time = np.linspace(cooling_time, cooling_time + measurement_time, n_measurements)
        
        # Live time (assume 90% dead time correction)
        live_time = np.full(n_measurements, measurement_time / n_measurements * 0.9)
        
        # Decay curve: A(t) = A0 * exp(-λt)
        # Account for activation during irradiation
        # A0 = saturation_activity * (1 - exp(-λ*t_irr))
        saturation_factor = 1 - np.exp(-lambda_decay * irradiation_time)
        A0 = initial_activity * saturation_factor
        
        # Activity at measurement times
        activity = A0 * np.exp(-lambda_decay * time)
        
        # Counts = activity * live_time
        counts = activity * live_time + background * live_time
        
        # Add Poisson noise
        counts = np.random.poisson(counts)
        
        return NAADecayCurve(
            time=time,
            counts=counts,
            live_time=live_time,
            energy=nuc_data.gamma_energy,
            element=element,
            isotope=nuc_data.isotope,
            metadata={
                'simulated': True,
                'initial_activity': initial_activity,
                'irradiation_time': irradiation_time,
                'cooling_time': cooling_time,
                'background': background
            }
        )
    
    @staticmethod
    def simulate_etch_profile(
        model: str = "linear",
        nominal_rate: float = 100.0,  # nm/min
        alpha: float = 0.3,  # Loading coefficient
        n_points: int = 50,
        noise_level: float = 0.03
    ) -> EtchProfile:
        """Generate etch rate vs pattern density"""
        
        # Pattern density
        pattern_density = np.linspace(0, 100, n_points)
        D = pattern_density / 100.0
        
        # Calculate etch rate based on model
        if model == "linear":
            etch_rate = nominal_rate * (1 - alpha * D)
        elif model == "exponential":
            etch_rate = nominal_rate * np.exp(-alpha * D)
        elif model == "power":
            etch_rate = nominal_rate * (1 - D)**alpha
        else:
            etch_rate = nominal_rate * np.ones_like(D)
        
        # Add noise
        noise = 1 + noise_level * np.random.randn(n_points)
        etch_rate = etch_rate * noise
        etch_rate = np.maximum(etch_rate, 0.0)
        
        return EtchProfile(
            pattern_density=pattern_density,
            etch_rate=etch_rate,
            chemistry="KOH",
            temperature=80.0,
            concentration=30.0,
            metadata={
                'simulated': True,
                'model': model,
                'nominal_rate': nominal_rate,
                'alpha': alpha
            }
        )


# ============================================================================
# FastAPI Integration & Endpoints
# ============================================================================

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import io

app = FastAPI(title="Chemical Bulk Analysis API", version="1.0.0")

# Initialize analyzers
sims_analyzer = SIMSAnalyzer()
rbs_analyzer = RBSAnalyzer()
naa_analyzer = NAAAnalyzer()
etch_analyzer = ChemicalEtchAnalyzer()
simulator = ChemicalBulkSimulator()


# Request/Response models for API
class SIMSAnalysisRequest(BaseModel):
    profile_id: str
    method: str = "RSF"
    sputter_rate: Optional[float] = None


class RBSAnalysisRequest(BaseModel):
    spectrum_id: str
    layers: List[Dict[str, Any]]
    fit_range: Optional[Tuple[float, float]] = None
    fix_composition: bool = False


class NAAAnalysisRequest(BaseModel):
    sample_id: str
    standard_id: str
    sample_mass: float
    standard_mass: float
    standard_concentration: float
    element: str


@app.post("/api/sims/analyze")
async def analyze_sims(request: SIMSAnalysisRequest):
    """Analyze SIMS depth profile"""
    try:
        # In production, load profile from database
        # For demo, create test data
        profile = simulator.simulate_sims_profile(
            element="B", matrix="Si", peak_depth=100, dose=1e15
        )
        
        # Convert time to depth
        depth = sims_analyzer.convert_time_to_depth(profile)
        profile.depth = depth
        
        # Quantify
        method = MatrixEffect[request.method]
        concentration = sims_analyzer.quantify_profile(profile, method=method)
        profile.concentration = concentration
        
        # Find interfaces
        interfaces = sims_analyzer.find_interfaces(profile)
        
        # Calculate dose
        dose = sims_analyzer.calculate_dose(profile)
        
        # Detection limit
        det_limit = sims_analyzer.estimate_detection_limit(profile)
        
        return {
            "status": "success",
            "depth": depth.tolist(),
            "concentration": concentration.tolist(),
            "interfaces": interfaces,
            "total_dose": dose,
            "detection_limit": det_limit,
            "metadata": profile.metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rbs/analyze")
async def analyze_rbs(request: RBSAnalysisRequest):
    """Analyze RBS spectrum"""
    try:
        # Create test spectrum
        test_layers = [("Hf", 0.5, 20), ("O", 0.5, 20)]
        spectrum = simulator.simulate_rbs_spectrum(test_layers, substrate="Si")
        
        # Parse layer guess from request
        initial_layers = []
        for layer_dict in request.layers:
            initial_layers.append(RBSLayer(
                element=layer_dict['element'],
                atomic_fraction=layer_dict['fraction'],
                thickness=layer_dict['thickness']
            ))
        
        # Fit spectrum
        result = rbs_analyzer.fit_spectrum(
            spectrum,
            initial_layers,
            fit_range=request.fit_range,
            fix_composition=request.fix_composition
        )
        
        # Format results
        fitted_layers = []
        for layer in result.layers:
            fitted_layers.append({
                'element': layer.element,
                'atomic_fraction': layer.atomic_fraction,
                'thickness': layer.thickness,
                'thickness_nm': layer.thickness_nm()
            })
        
        return {
            "status": "success",
            "fitted_layers": fitted_layers,
            "simulated_spectrum": result.simulated_spectrum.tolist(),
            "chi_squared": result.chi_squared,
            "r_factor": result.r_factor,
            "metadata": result.metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/naa/analyze")
async def analyze_naa(request: NAAAnalysisRequest):
    """Analyze NAA data using comparator method"""
    try:
        # Create test decay curves
        sample_curve = simulator.simulate_naa_decay(
            element=request.element, initial_activity=5000
        )
        standard_curve = simulator.simulate_naa_decay(
            element=request.element, initial_activity=10000
        )
        
        # Quantify using comparator method
        result = naa_analyzer.comparator_method(
            sample_curve,
            standard_curve,
            request.standard_mass,
            request.sample_mass,
            request.standard_concentration,
            request.element
        )
        
        return {
            "status": "success",
            "element": result.element,
            "concentration": result.concentration,
            "uncertainty": result.uncertainty,
            "detection_limit": result.detection_limit,
            "isotope": result.isotope,
            "activity": result.activity,
            "metadata": result.metadata
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/etch/analyze")
async def analyze_etch():
    """Analyze chemical etch loading effects"""
    try:
        # Create test etch profile
        profile = simulator.simulate_etch_profile(
            model="linear", nominal_rate=100, alpha=0.3
        )
        
        # Fit loading effect
        loading = etch_analyzer.fit_loading_effect(profile, model="linear")
        
        # Calculate uniformity (for spatial case)
        uniformity = etch_analyzer.calculate_uniformity(profile)
        
        return {
            "status": "success",
            "loading_effect": {
                "nominal_rate": loading.nominal_rate,
                "max_reduction": loading.max_reduction,
                "critical_density": loading.critical_density,
                "model": loading.model_type,
                "r_squared": loading.r_squared,
                "coefficients": loading.coefficients
            },
            "uniformity": uniformity,
            "profile": {
                "pattern_density": profile.pattern_density.tolist(),
                "etch_rate": profile.etch_rate.tolist()
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/simulator/sims")
async def simulate_sims_endpoint(
    element: str = "B",
    peak_depth: float = 100,
    dose: float = 1e15,
    straggle: float = 30
):
    """Generate simulated SIMS profile"""
    profile = simulator.simulate_sims_profile(
        element=element,
        peak_depth=peak_depth,
        dose=dose,
        straggle=straggle
    )
    
    return {
        "depth": profile.depth.tolist(),
        "concentration": profile.concentration.tolist() if profile.concentration is not None else [],
        "counts": profile.counts.tolist(),
        "element": profile.element,
        "metadata": profile.metadata
    }


@app.get("/api/simulator/rbs")
async def simulate_rbs_endpoint(
    layer1_element: str = "Hf",
    layer1_thickness: float = 20,
    layer2_element: str = "O",
    layer2_thickness: float = 20
):
    """Generate simulated RBS spectrum"""
    layers = [
        (layer1_element, 0.5, layer1_thickness),
        (layer2_element, 0.5, layer2_thickness)
    ]
    
    spectrum = simulator.simulate_rbs_spectrum(layers)
    
    return {
        "energy": spectrum.energy.tolist(),
        "counts": spectrum.counts.tolist(),
        "metadata": spectrum.metadata
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "chemical-bulk-analysis"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
