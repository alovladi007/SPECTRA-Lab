"""
FTIR Spectroscopy Analysis Module
Session 7 - Optical I Implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from scipy import signal, optimize, interpolate
from scipy.stats import norm
from sklearn.decomposition import PCA
import warnings
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PeakShape(Enum):
    """Peak shape models for fitting"""
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"
    VOIGT = "voigt"
    PSEUDO_VOIGT = "pseudo_voigt"


@dataclass
class Peak:
    """Individual peak information"""
    position: float  # cm⁻¹
    intensity: float
    width: float  # FWHM
    area: float
    shape: str
    assignment: Optional[str] = None
    confidence: float = 0.0


@dataclass
class FunctionalGroup:
    """Functional group identification"""
    name: str
    peak_range: Tuple[float, float]  # cm⁻¹
    typical_position: float
    intensity: str  # 'strong', 'medium', 'weak'
    vibration_type: str  # 'stretch', 'bend', etc.
    compounds: List[str]


@dataclass
class FTIRResult:
    """Complete FTIR analysis results"""
    wavenumber: np.ndarray
    absorbance: np.ndarray
    baseline: np.ndarray
    corrected: np.ndarray
    peaks: List[Peak]
    functional_groups: List[FunctionalGroup]
    quality_metrics: Dict[str, float]


class FunctionalGroupLibrary:
    """Library of common FTIR functional groups"""
    
    def __init__(self):
        self.groups = [
            # O-H groups
            FunctionalGroup(
                name="O-H stretch (free)",
                peak_range=(3500, 3700),
                typical_position=3600,
                intensity="strong",
                vibration_type="stretch",
                compounds=["alcohols", "phenols"]
            ),
            FunctionalGroup(
                name="O-H stretch (H-bonded)",
                peak_range=(3200, 3500),
                typical_position=3350,
                intensity="strong",
                vibration_type="stretch",
                compounds=["alcohols", "carboxylic acids"]
            ),
            
            # C-H groups
            FunctionalGroup(
                name="C-H stretch (alkane)",
                peak_range=(2850, 3000),
                typical_position=2925,
                intensity="strong",
                vibration_type="stretch",
                compounds=["alkanes", "alkyl groups"]
            ),
            FunctionalGroup(
                name="C-H stretch (alkene)",
                peak_range=(3000, 3100),
                typical_position=3050,
                intensity="medium",
                vibration_type="stretch",
                compounds=["alkenes"]
            ),
            FunctionalGroup(
                name="C-H stretch (aromatic)",
                peak_range=(3000, 3100),
                typical_position=3030,
                intensity="medium",
                vibration_type="stretch",
                compounds=["aromatics", "benzene rings"]
            ),
            
            # C=O groups
            FunctionalGroup(
                name="C=O stretch (ketone)",
                peak_range=(1705, 1725),
                typical_position=1715,
                intensity="strong",
                vibration_type="stretch",
                compounds=["ketones"]
            ),
            FunctionalGroup(
                name="C=O stretch (aldehyde)",
                peak_range=(1720, 1740),
                typical_position=1730,
                intensity="strong",
                vibration_type="stretch",
                compounds=["aldehydes"]
            ),
            FunctionalGroup(
                name="C=O stretch (ester)",
                peak_range=(1735, 1750),
                typical_position=1740,
                intensity="strong",
                vibration_type="stretch",
                compounds=["esters"]
            ),
            FunctionalGroup(
                name="C=O stretch (carboxylic acid)",
                peak_range=(1700, 1725),
                typical_position=1710,
                intensity="strong",
                vibration_type="stretch",
                compounds=["carboxylic acids"]
            ),
            FunctionalGroup(
                name="C=O stretch (amide I)",
                peak_range=(1630, 1690),
                typical_position=1650,
                intensity="strong",
                vibration_type="stretch",
                compounds=["amides", "proteins"]
            ),
            
            # C=C groups
            FunctionalGroup(
                name="C=C stretch (alkene)",
                peak_range=(1620, 1680),
                typical_position=1650,
                intensity="medium",
                vibration_type="stretch",
                compounds=["alkenes"]
            ),
            FunctionalGroup(
                name="C=C stretch (aromatic)",
                peak_range=(1450, 1600),
                typical_position=1500,
                intensity="medium",
                vibration_type="stretch",
                compounds=["aromatics"]
            ),
            
            # N-H groups
            FunctionalGroup(
                name="N-H stretch (primary amine)",
                peak_range=(3300, 3500),
                typical_position=3400,
                intensity="medium",
                vibration_type="stretch",
                compounds=["primary amines"]
            ),
            FunctionalGroup(
                name="N-H bend (amide II)",
                peak_range=(1515, 1570),
                typical_position=1540,
                intensity="strong",
                vibration_type="bend",
                compounds=["amides", "proteins"]
            ),
            
            # C-N groups
            FunctionalGroup(
                name="C-N stretch",
                peak_range=(1000, 1250),
                typical_position=1100,
                intensity="medium",
                vibration_type="stretch",
                compounds=["amines"]
            ),
            
            # C-O groups
            FunctionalGroup(
                name="C-O stretch (alcohol)",
                peak_range=(1000, 1100),
                typical_position=1050,
                intensity="strong",
                vibration_type="stretch",
                compounds=["alcohols"]
            ),
            FunctionalGroup(
                name="C-O stretch (ether)",
                peak_range=(1070, 1150),
                typical_position=1100,
                intensity="strong",
                vibration_type="stretch",
                compounds=["ethers"]
            ),
            
            # Inorganic groups
            FunctionalGroup(
                name="Si-O stretch",
                peak_range=(1000, 1150),
                typical_position=1080,
                intensity="strong",
                vibration_type="stretch",
                compounds=["silicates", "SiO2", "glass"]
            ),
            FunctionalGroup(
                name="Si-H stretch",
                peak_range=(2100, 2250),
                typical_position=2150,
                intensity="medium",
                vibration_type="stretch",
                compounds=["silicon hydrides"]
            ),
            FunctionalGroup(
                name="P-O stretch",
                peak_range=(1000, 1250),
                typical_position=1100,
                intensity="strong",
                vibration_type="stretch",
                compounds=["phosphates"]
            ),
            
            # Triple bonds
            FunctionalGroup(
                name="C≡N stretch (nitrile)",
                peak_range=(2210, 2260),
                typical_position=2250,
                intensity="medium",
                vibration_type="stretch",
                compounds=["nitriles"]
            ),
            FunctionalGroup(
                name="C≡C stretch (alkyne)",
                peak_range=(2100, 2250),
                typical_position=2150,
                intensity="weak",
                vibration_type="stretch",
                compounds=["alkynes"]
            ),
        ]
    
    def identify_groups(self, peaks: List[Peak]) -> List[FunctionalGroup]:
        """Identify functional groups from peaks"""
        identified = []
        
        for peak in peaks:
            for group in self.groups:
                if group.peak_range[0] <= peak.position <= group.peak_range[1]:
                    # Calculate confidence based on position match
                    center = group.typical_position
                    range_width = group.peak_range[1] - group.peak_range[0]
                    distance = abs(peak.position - center)
                    confidence = max(0, 1 - (distance / (range_width / 2)))
                    
                    # Update peak assignment
                    if peak.assignment is None or confidence > peak.confidence:
                        peak.assignment = group.name
                        peak.confidence = confidence
                    
                    # Add to identified groups
                    if group not in identified:
                        identified.append(group)
        
        return identified


class FTIRAnalyzer:
    """
    Comprehensive FTIR spectroscopy analyzer
    
    Features:
    - Advanced baseline correction
    - Peak finding and fitting
    - Functional group identification
    - Quantitative analysis
    - ATR correction
    - Library matching
    """
    
    def __init__(self, wavenumber_range: Tuple[float, float] = (400, 4000)):
        """
        Initialize FTIR analyzer
        
        Args:
            wavenumber_range: Wavenumber range in cm⁻¹
        """
        self.wavenumber_range = wavenumber_range
        self.library = FunctionalGroupLibrary()
        
    def process_spectrum(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        mode: str = 'absorbance',
        baseline_method: str = 'als',
        smooth_window: int = 11,
        atr_correction: bool = False,
        atr_crystal: str = 'ZnSe'
    ) -> FTIRResult:
        """
        Process FTIR spectrum with corrections
        
        Args:
            wavenumber: Wavenumber array (cm⁻¹)
            intensity: Intensity array
            mode: 'absorbance' or 'transmittance'
            baseline_method: 'als', 'polynomial', 'manual'
            smooth_window: Window size for smoothing
            atr_correction: Apply ATR correction
            atr_crystal: ATR crystal material
            
        Returns:
            FTIRResult with processed spectrum and analysis
        """
        # Input validation
        if len(wavenumber) != len(intensity):
            raise ValueError("Wavenumber and intensity must have same length")
        
        # Sort by wavenumber
        sort_idx = np.argsort(wavenumber)
        wavenumber = wavenumber[sort_idx]
        intensity = intensity[sort_idx]
        
        # Convert to absorbance if needed
        if mode == 'transmittance':
            intensity = -np.log10(np.clip(intensity / 100, 1e-6, 1))
        
        # Smooth spectrum
        if smooth_window > 0 and len(intensity) > smooth_window:
            if smooth_window % 2 == 0:
                smooth_window += 1
            intensity = signal.savgol_filter(intensity, smooth_window, 3)
        
        # ATR correction
        if atr_correction:
            intensity = self._atr_correction(wavenumber, intensity, atr_crystal)
        
        # Baseline correction
        baseline = self._calculate_baseline(wavenumber, intensity, baseline_method)
        corrected = intensity - baseline
        
        # Find peaks
        peaks = self._find_peaks(wavenumber, corrected)
        
        # Fit peaks
        fitted_peaks = self._fit_peaks(wavenumber, corrected, peaks)
        
        # Identify functional groups
        functional_groups = self.library.identify_groups(fitted_peaks)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            wavenumber, corrected, fitted_peaks
        )
        
        return FTIRResult(
            wavenumber=wavenumber,
            absorbance=intensity,
            baseline=baseline,
            corrected=corrected,
            peaks=fitted_peaks,
            functional_groups=functional_groups,
            quality_metrics=quality_metrics
        )
    
    def _calculate_baseline(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        method: str
    ) -> np.ndarray:
        """Calculate baseline using specified method"""
        
        if method == 'als':
            return self._als_baseline(intensity)
        elif method == 'polynomial':
            return self._polynomial_baseline(wavenumber, intensity)
        elif method == 'rubberband':
            return self._rubberband_baseline(wavenumber, intensity)
        else:
            return np.zeros_like(intensity)
    
    def _als_baseline(
        self,
        y: np.ndarray,
        lam: float = 1e5,
        p: float = 0.05,
        n_iter: int = 10
    ) -> np.ndarray:
        """Asymmetric Least Squares baseline"""
        L = len(y)
        D = np.diff(np.eye(L), 2)
        D = lam * D.T @ D
        
        w = np.ones(L)
        z = np.zeros(L)
        
        for _ in range(n_iter):
            W = np.diag(w)
            z = np.linalg.solve(W + D, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
            
        return z
    
    def _polynomial_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray,
        degree: int = 4
    ) -> np.ndarray:
        """Polynomial baseline fitting"""
        # Find baseline points (local minima)
        minima, _ = signal.find_peaks(-y, prominence=0.01)
        
        if len(minima) < degree + 1:
            # Not enough points, use whole spectrum
            coeffs = np.polyfit(x, y, degree)
        else:
            # Fit to minima only
            coeffs = np.polyfit(x[minima], y[minima], degree)
        
        return np.polyval(coeffs, x)
    
    def _rubberband_baseline(
        self,
        x: np.ndarray,
        y: np.ndarray
    ) -> np.ndarray:
        """Rubberband (convex hull) baseline"""
        from scipy.spatial import ConvexHull
        
        # Create points
        points = np.column_stack([x, y])
        
        # Get convex hull
        hull = ConvexHull(points)
        
        # Get lower hull
        hull_points = points[hull.vertices]
        hull_points = hull_points[hull_points[:, 0].argsort()]
        
        # Select points below the spectrum
        lower_hull = []
        for i in range(len(hull_points) - 1):
            p1 = hull_points[i]
            p2 = hull_points[i + 1]
            
            # Check if segment is below spectrum
            x_segment = x[(x >= p1[0]) & (x <= p2[0])]
            y_segment = y[(x >= p1[0]) & (x <= p2[0])]
            
            # Linear interpolation between hull points
            y_interp = np.interp(x_segment, [p1[0], p2[0]], [p1[1], p2[1]])
            
            if np.all(y_interp <= y_segment):
                lower_hull.append(p1)
        
        if len(lower_hull) > 1:
            lower_hull.append(hull_points[-1])
            lower_hull = np.array(lower_hull)
            baseline = np.interp(x, lower_hull[:, 0], lower_hull[:, 1])
        else:
            # Fallback to linear baseline
            baseline = np.linspace(y[0], y[-1], len(y))
        
        return baseline
    
    def _atr_correction(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        crystal: str = 'ZnSe'
    ) -> np.ndarray:
        """
        Apply ATR correction for penetration depth variation
        
        Args:
            wavenumber: Wavenumber array (cm⁻¹)
            intensity: Absorbance spectrum
            crystal: ATR crystal material
        """
        # Refractive indices of common ATR crystals
        n_crystal = {
            'ZnSe': 2.4,
            'Ge': 4.0,
            'Si': 3.4,
            'Diamond': 2.4,
            'KRS-5': 2.37
        }.get(crystal, 2.4)
        
        # Sample refractive index (approximate)
        n_sample = 1.5
        
        # Angle of incidence (typical)
        theta = 45 * np.pi / 180
        
        # Penetration depth correction
        # dp = λ / (2π * n_crystal * sqrt(sin²θ - (n_sample/n_crystal)²))
        lambda_cm = 1 / wavenumber
        
        sin_theta_eff = np.sqrt(
            np.sin(theta)**2 - (n_sample / n_crystal)**2
        )
        
        dp = lambda_cm / (2 * np.pi * n_crystal * sin_theta_eff)
        
        # Correction factor (normalized)
        correction = dp / dp.mean()
        
        return intensity * correction
    
    def _find_peaks(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        min_prominence: float = 0.01,
        min_distance: int = 10
    ) -> np.ndarray:
        """
        Find peaks in spectrum
        
        Args:
            wavenumber: Wavenumber array
            intensity: Intensity array
            min_prominence: Minimum peak prominence
            min_distance: Minimum distance between peaks
            
        Returns:
            Peak indices
        """
        # Find peaks
        peaks, properties = signal.find_peaks(
            intensity,
            prominence=min_prominence,
            distance=min_distance,
            width=2
        )
        
        # Sort by prominence
        prominences = properties['prominences']
        sorted_idx = np.argsort(prominences)[::-1]
        
        # Return top peaks
        return peaks[sorted_idx[:50]]  # Limit to 50 peaks
    
    def _fit_peaks(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        peak_indices: np.ndarray,
        fit_window: int = 20
    ) -> List[Peak]:
        """
        Fit peaks with Gaussian/Lorentzian profiles
        
        Args:
            wavenumber: Wavenumber array
            intensity: Intensity array  
            peak_indices: Indices of peaks to fit
            fit_window: Window around peak for fitting
            
        Returns:
            List of fitted Peak objects
        """
        fitted_peaks = []
        
        for idx in peak_indices:
            # Define fitting window
            start = max(0, idx - fit_window)
            end = min(len(wavenumber), idx + fit_window)
            
            x_fit = wavenumber[start:end]
            y_fit = intensity[start:end]
            
            if len(x_fit) < 4:
                continue
            
            try:
                # Initial parameters
                p0 = [
                    wavenumber[idx],  # position
                    intensity[idx],   # amplitude
                    10.0             # width
                ]
                
                # Fit Gaussian
                popt, _ = optimize.curve_fit(
                    self._gaussian,
                    x_fit, y_fit,
                    p0=p0,
                    maxfev=1000
                )
                
                # Extract parameters
                position, amplitude, width = popt
                
                # Calculate area
                area = amplitude * width * np.sqrt(2 * np.pi)
                
                # Create Peak object
                peak = Peak(
                    position=position,
                    intensity=amplitude,
                    width=abs(width),
                    area=area,
                    shape='gaussian'
                )
                
                fitted_peaks.append(peak)
                
            except:
                # Fitting failed, use simple parameters
                peak = Peak(
                    position=wavenumber[idx],
                    intensity=intensity[idx],
                    width=10.0,
                    area=intensity[idx] * 10.0,
                    shape='estimated'
                )
                fitted_peaks.append(peak)
        
        return fitted_peaks
    
    def _gaussian(
        self,
        x: np.ndarray,
        position: float,
        amplitude: float,
        width: float
    ) -> np.ndarray:
        """Gaussian peak function"""
        return amplitude * np.exp(-0.5 * ((x - position) / width)**2)
    
    def _lorentzian(
        self,
        x: np.ndarray,
        position: float,
        amplitude: float,
        width: float
    ) -> np.ndarray:
        """Lorentzian peak function"""
        return amplitude * width**2 / ((x - position)**2 + width**2)
    
    def _voigt(
        self,
        x: np.ndarray,
        position: float,
        amplitude: float,
        width_g: float,
        width_l: float
    ) -> np.ndarray:
        """Voigt peak function (convolution of Gaussian and Lorentzian)"""
        from scipy.special import voigt_profile
        
        sigma = width_g / (2 * np.sqrt(2 * np.log(2)))
        gamma = width_l / 2
        
        return amplitude * voigt_profile(x - position, sigma, gamma)
    
    def _calculate_quality_metrics(
        self,
        wavenumber: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Peak]
    ) -> Dict[str, float]:
        """Calculate spectrum quality metrics"""
        
        metrics = {}
        
        # Signal-to-noise ratio
        if len(intensity) > 100:
            # Use high wavenumber region for noise estimation
            noise_region = intensity[-50:]
            noise_std = np.std(noise_region)
            signal_max = np.max(intensity)
            metrics['snr'] = signal_max / noise_std if noise_std > 0 else np.inf
        else:
            metrics['snr'] = 0
        
        # Baseline flatness
        baseline_region = intensity[:20]
        metrics['baseline_std'] = np.std(baseline_region)
        
        # Number of identified peaks
        metrics['n_peaks'] = len(peaks)
        
        # Peak resolution (average)
        if len(peaks) > 1:
            positions = sorted([p.position for p in peaks])
            separations = np.diff(positions)
            widths = [p.width for p in peaks[:-1]]
            resolutions = separations / np.array(widths)
            metrics['avg_resolution'] = np.mean(resolutions)
        else:
            metrics['avg_resolution'] = 0
        
        # Spectral coverage
        actual_range = wavenumber[-1] - wavenumber[0]
        expected_range = self.wavenumber_range[1] - self.wavenumber_range[0]
        metrics['coverage'] = actual_range / expected_range
        
        return metrics
    
    def quantitative_analysis(
        self,
        peaks: List[Peak],
        calibration: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Perform quantitative analysis based on peak areas
        
        Args:
            peaks: List of fitted peaks
            calibration: Calibration factors for concentration
            
        Returns:
            Concentrations or relative amounts
        """
        results = {}
        
        # Group peaks by assignment
        assignments = {}
        for peak in peaks:
            if peak.assignment:
                if peak.assignment not in assignments:
                    assignments[peak.assignment] = []
                assignments[peak.assignment].append(peak)
        
        # Calculate total area for each group
        for group, group_peaks in assignments.items():
            total_area = sum(p.area for p in group_peaks)
            
            if calibration and group in calibration:
                # Apply calibration factor
                concentration = total_area * calibration[group]
                results[f"{group}_concentration"] = concentration
            else:
                # Relative amount
                results[f"{group}_area"] = total_area
        
        # Normalize to percentages
        if not calibration:
            total = sum(results.values())
            if total > 0:
                for key in results:
                    results[f"{key}_percent"] = 100 * results[key] / total
        
        return results
    
    def compare_spectra(
        self,
        spectra: List[np.ndarray],
        wavenumber: np.ndarray,
        method: str = 'correlation'
    ) -> np.ndarray:
        """
        Compare multiple spectra
        
        Args:
            spectra: List of intensity arrays
            wavenumber: Common wavenumber array
            method: 'correlation', 'euclidean', 'pca'
            
        Returns:
            Similarity matrix or PCA results
        """
        n_spectra = len(spectra)
        
        if method == 'correlation':
            # Correlation matrix
            similarity = np.zeros((n_spectra, n_spectra))
            
            for i in range(n_spectra):
                for j in range(n_spectra):
                    similarity[i, j] = np.corrcoef(
                        spectra[i], spectra[j]
                    )[0, 1]
            
            return similarity
            
        elif method == 'euclidean':
            # Euclidean distance matrix
            distance = np.zeros((n_spectra, n_spectra))
            
            for i in range(n_spectra):
                for j in range(n_spectra):
                    distance[i, j] = np.linalg.norm(
                        spectra[i] - spectra[j]
                    )
            
            return distance
            
        elif method == 'pca':
            # PCA analysis
            X = np.array(spectra)
            
            pca = PCA(n_components=min(3, n_spectra))
            scores = pca.fit_transform(X)
            
            return {
                'scores': scores,
                'loadings': pca.components_,
                'variance_explained': pca.explained_variance_ratio_
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def library_match(
        self,
        spectrum: np.ndarray,
        library: List[Dict[str, Any]],
        threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Match spectrum against library
        
        Args:
            spectrum: Unknown spectrum
            library: List of reference spectra
            threshold: Minimum correlation for match
            
        Returns:
            List of matches with scores
        """
        matches = []
        
        for reference in library:
            # Interpolate to common wavenumber grid
            ref_spectrum = np.interp(
                self.wavenumber_range,
                reference['wavenumber'],
                reference['intensity']
            )
            
            # Calculate correlation
            correlation = np.corrcoef(spectrum, ref_spectrum)[0, 1]
            
            if correlation > threshold:
                matches.append({
                    'name': reference['name'],
                    'score': correlation,
                    'metadata': reference.get('metadata', {})
                })
        
        # Sort by score
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return matches[:10]  # Top 10 matches


# Example usage and validation
if __name__ == "__main__":
    # Create analyzer
    analyzer = FTIRAnalyzer()
    
    # Generate synthetic FTIR spectrum
    wavenumber = np.linspace(400, 4000, 3600)
    
    # Create spectrum with multiple peaks
    spectrum = np.zeros_like(wavenumber)
    
    # Add peaks for different functional groups
    peaks_to_add = [
        (1080, 100, 30),  # Si-O stretch
        (1650, 80, 25),   # C=O stretch
        (2150, 40, 20),   # Si-H stretch
        (2925, 60, 35),   # C-H stretch
        (3350, 70, 100),  # O-H stretch (broad)
    ]
    
    for position, intensity, width in peaks_to_add:
        spectrum += analyzer._gaussian(wavenumber, position, intensity, width)
    
    # Add baseline and noise
    baseline = 0.1 + 0.05 * (wavenumber - 2000) / 2000
    noise = np.random.normal(0, 2, len(wavenumber))
    spectrum = spectrum + baseline + noise
    
    # Process spectrum
    result = analyzer.process_spectrum(
        wavenumber,
        spectrum,
        mode='absorbance',
        baseline_method='als'
    )
    
    print("FTIR Analysis Results")
    print("=" * 50)
    print(f"Number of peaks found: {len(result.peaks)}")
    print(f"SNR: {result.quality_metrics['snr']:.1f}")
    
    print("\nIdentified Functional Groups:")
    for group in result.functional_groups:
        print(f"  - {group.name}")
        print(f"    Range: {group.peak_range[0]}-{group.peak_range[1]} cm⁻¹")
        print(f"    Compounds: {', '.join(group.compounds)}")
    
    print("\nPeak Details:")
    for peak in result.peaks[:5]:  # Show first 5 peaks
        print(f"  Position: {peak.position:.1f} cm⁻¹")
        print(f"  Intensity: {peak.intensity:.2f}")
        print(f"  Width: {peak.width:.1f} cm⁻¹")
        print(f"  Assignment: {peak.assignment}")
        print()
