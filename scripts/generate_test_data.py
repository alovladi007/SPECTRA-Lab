# scripts/dev/generate_test_data.py

“””
Test Data Generators and Factory Patterns

Generates realistic synthetic data for all characterization methods:

- Electrical: I-V, C-V, Hall, 4PP, DLTS
- Optical: UV-Vis-NIR, FTIR, Ellipsometry, PL, Raman
- Structural: XRD, SEM/TEM, AFM
- Chemical: XPS, SIMS, RBS

Each generator produces:

- Raw measurement data
- Derived metrics/results
- Metadata with provenance
- Realistic noise and artifacts
  “””

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import json

# ============================================================================

# Base Generator

# ============================================================================

@dataclass
class GeneratorConfig:
“”“Configuration for test data generation”””
add_noise: bool = True
noise_level: float = 0.01  # 1% relative noise
add_outliers: bool = False
outlier_fraction: float = 0.02
add_drift: bool = False
drift_rate: float = 1e-6
seed: Optional[int] = None

class BaseGenerator:
“”“Base class for all data generators”””

def __init__(self, config: Optional[GeneratorConfig] = None):
    self.config = config or GeneratorConfig()
    if self.config.seed is not None:
        np.random.seed(self.config.seed)

def add_noise(self, data: np.ndarray, relative: bool = True) -> np.ndarray:
    """Add Gaussian noise to data"""
    if not self.config.add_noise:
        return data
    
    if relative:
        noise = np.random.normal(0, self.config.noise_level, data.shape) * np.abs(data)
    else:
        noise = np.random.normal(0, self.config.noise_level, data.shape)
    
    return data + noise

def add_outliers(self, data: np.ndarray) -> np.ndarray:
    """Add random outliers to data"""
    if not self.config.add_outliers:
        return data
    
    n_outliers = int(len(data) * self.config.outlier_fraction)
    outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
    outliers = np.random.uniform(0.5, 2.0, n_outliers) * data[outlier_indices]
    data[outlier_indices] = outliers
    
    return data

def generate_metadata(self, method: str, **kwargs) -> Dict[str, Any]:
    """Generate standard metadata"""
    return {
        "method": method,
        "generated_at": datetime.utcnow().isoformat(),
        "generator_version": "1.0.0",
        "config": {
            "noise_level": self.config.noise_level,
            "has_outliers": self.config.add_outliers,
            "has_drift": self.config.add_drift,
        },
        **kwargs
    }

# ============================================================================

# Electrical Data Generators

# ============================================================================

class IVGenerator(BaseGenerator):
“”“Generate I-V curves for diodes, transistors, solar cells”””

def generate_diode(
    self,
    v_range: Tuple[float, float] = (-1.0, 1.0),
    points: int = 200,
    Is: float = 1e-12,
    n: float = 1.5,
    Rs: float = 10.0,
    Rsh: float = 1e6
) -> Dict[str, Any]:
    """Generate diode I-V curve"""
    
    voltage = np.linspace(v_range[0], v_range[1], points)
    
    # Shockley equation with series/shunt resistance
    Vt = 0.026  # Thermal voltage at 300K
    current = np.zeros_like(voltage)
    
    for i, V in enumerate(voltage):
        # Newton-Raphson to solve implicit equation
        I = Is * (np.exp(V / (n * Vt)) - 1)
        for _ in range(5):
            Vd = V - I * Rs
            Id = Is * (np.exp(Vd / (n * Vt)) - 1)
            Ish = Vd / Rsh
            f = I - Id - Ish
            dId_dVd = (Is / (n * Vt)) * np.exp(Vd / (n * Vt))
            df_dI = 1 + Rs * (dId_dVd + 1 / Rsh)
            I_new = I - f / df_dI
            if abs(I_new - I) < 1e-15:
                break
            I = I_new
        current[i] = I
    
    # Add noise
    current = self.add_noise(current)
    current = self.add_outliers(current)
    
    return {
        "voltage": voltage.tolist(),
        "current": current.tolist(),
        "parameters": {
            "Is": Is,
            "n": n,
            "Rs": Rs,
            "Rsh": Rsh,
            "temperature": 300.0
        },
        "metadata": self.generate_metadata("iv_sweep", device_type="diode")
    }

def generate_solar_cell(
    self,
    v_range: Tuple[float, float] = (-0.1, 0.7),
    points: int = 100,
    Jsc: float = 35.0,  # mA/cm²
    Voc: float = 0.65,  # V
    FF: float = 0.80
) -> Dict[str, Any]:
    """Generate solar cell I-V curve under illumination"""
    
    voltage = np.linspace(v_range[0], v_range[1], points)
    
    # Single-diode model
    n = 1.5
    Vt = 0.026
    Rs = 0.5  # ohm-cm²
    Rsh = 1000  # ohm-cm²
    
    # Calculate Is from Voc
    Is = Jsc / (np.exp(Voc / (n * Vt)) - 1)
    
    current = np.zeros_like(voltage)
    for i, V in enumerate(voltage):
        # Light-generated current minus diode current
        I = Jsc
        for _ in range(5):
            Vd = V + I * Rs
            Id = Is * (np.exp(Vd / (n * Vt)) - 1)
            Ish = Vd / Rsh
            f = I - Jsc + Id + Ish
            dId_dVd = (Is / (n * Vt)) * np.exp(Vd / (n * Vt))
            df_dI = 1 + Rs * (dId_dVd + 1 / Rsh)
            I_new = I - f / df_dI
            if abs(I_new - I) < 1e-6:
                break
            I = I_new
        current[i] = I
    
    # Add noise
    current = self.add_noise(current, relative=False)
    
    # Calculate efficiency
    Pmax = np.max(-voltage * current)
    efficiency = Pmax / 100.0  # Assuming 100 mW/cm² illumination
    
    return {
        "voltage": voltage.tolist(),
        "current": current.tolist(),
        "parameters": {
            "Jsc": Jsc,
            "Voc": Voc,
            "FF": FF,
            "efficiency": efficiency,
            "illumination": 100.0  # mW/cm²
        },
        "metadata": self.generate_metadata("iv_sweep", device_type="solar_cell")
    }

class HallGenerator(BaseGenerator):
“”“Generate Hall effect measurement data”””

def generate(
    self,
    material: str = "Si",
    carrier_type: str = "n",
    carrier_density: float = 1e16,  # cm⁻³
    mobility: float = 1500,  # cm²/V/s
    magnetic_fields: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Generate Hall effect data"""
    
    if magnetic_fields is None:
        magnetic_fields = np.linspace(-0.5, 0.5, 21)  # Tesla
    
    # Hall coefficient (cm³/C)
    q = 1.602e-19  # C
    R_H = 1 / (carrier_density * q) if carrier_type == "n" else -1 / (carrier_density * q)
    
    # Convert to SI (m³/C)
    R_H_SI = R_H * 1e-6
    
    # Hall voltage (assuming current = 1 mA, thickness = 500 μm)
    I = 1e-3  # A
    t = 500e-6  # m
    V_H = R_H_SI * I * magnetic_fields / t * 1e3  # mV
    
    # Add noise
    V_H = self.add_noise(V_H, relative=False)
    
    # Calculate conductivity
    sigma = carrier_density * q * mobility * 1e-4  # S/m
    
    return {
        "magnetic_field": magnetic_fields.tolist(),
        "hall_voltage": V_H.tolist(),
        "parameters": {
            "carrier_type": carrier_type,
            "carrier_density": carrier_density,
            "mobility": mobility,
            "hall_coefficient": R_H,
            "conductivity": sigma
        },
        "metadata": self.generate_metadata("hall_effect", material=material)
    }

class FourPointProbeGenerator(BaseGenerator):
“”“Generate 4-point probe sheet resistance data”””

def generate_wafer_map(
    self,
    diameter: float = 150,  # mm
    base_resistance: float = 100,  # ohm/sq
    uniformity: float = 0.05,  # 5% variation
    grid_size: int = 25
) -> Dict[str, Any]:
    """Generate wafer map of sheet resistance"""
    
    # Create grid
    x = np.linspace(-diameter/2, diameter/2, grid_size)
    y = np.linspace(-diameter/2, diameter/2, grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Radial gradient (edge effect)
    R = np.sqrt(X**2 + Y**2)
    radial_factor = 1 + 0.1 * (R / (diameter/2))**2
    
    # Random local variation
    local_variation = np.random.normal(1, uniformity, X.shape)
    
    # Sheet resistance map
    resistance = base_resistance * radial_factor * local_variation
    
    # Mask outside wafer
    resistance[R > diameter/2] = np.nan
    
    # Add noise
    resistance = self.add_noise(resistance)
    
    # Statistics
    valid_points = resistance[~np.isnan(resistance)]
    mean_r = np.mean(valid_points)
    std_r = np.std(valid_points)
    cv = (std_r / mean_r) * 100  # %
    
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "resistance": resistance.tolist(),
        "parameters": {
            "mean": float(mean_r),
            "std": float(std_r),
            "cv_percent": float(cv),
            "min": float(np.min(valid_points)),
            "max": float(np.max(valid_points))
        },
        "metadata": self.generate_metadata(
            "four_point_probe",
            diameter=diameter,
            grid_size=grid_size
        )
    }

# ============================================================================

# Optical Data Generators

# ============================================================================

class UVVisNIRGenerator(BaseGenerator):
“”“Generate UV-Vis-NIR spectroscopy data”””

def generate_semiconductor(
    self,
    material: str = "GaAs",
    wavelength_range: Tuple[float, float] = (300, 1200),  # nm
    points: int = 500,
    band_gap: float = 1.42,  # eV
    thickness: float = 500  # nm
) -> Dict[str, Any]:
    """Generate absorption spectrum for semiconductor"""
    
    wavelength = np.linspace(wavelength_range[0], wavelength_range[1], points)
    
    # Convert wavelength to energy
    h = 6.626e-34  # J·s
    c = 3e8  # m/s
    energy = (h * c) / (wavelength * 1e-9) / 1.602e-19  # eV
    
    # Absorption coefficient (Tauc model for direct band gap)
    alpha = np.zeros_like(energy)
    above_gap = energy > band_gap
    alpha[above_gap] = 1e4 * np.sqrt(energy[above_gap] - band_gap)  # cm⁻¹
    
    # Transmittance
    transmittance = np.exp(-alpha * thickness * 1e-7)  # thickness in cm
    
    # Add noise
    transmittance = self.add_noise(transmittance, relative=False)
    transmittance = np.clip(transmittance, 0, 1)
    
    # Calculate absorption
    absorbance = -np.log10(transmittance + 1e-10)
    
    return {
        "wavelength": wavelength.tolist(),
        "transmittance": transmittance.tolist(),
        "absorbance": absorbance.tolist(),
        "parameters": {
            "material": material,
            "band_gap": band_gap,
            "thickness": thickness
        },
        "metadata": self.generate_metadata("uv_vis_nir", material=material)
    }

class RamanGenerator(BaseGenerator):
“”“Generate Raman spectroscopy data”””

def generate_silicon(
    self,
    wavenumber_range: Tuple[float, float] = (400, 600),  # cm⁻¹
    points: int = 1000,
    strain: float = 0.0,  # GPa
    crystallinity: float = 1.0  # 0=amorphous, 1=crystalline
) -> Dict[str, Any]:
    """Generate Raman spectrum for silicon"""
    
    wavenumber = np.linspace(wavenumber_range[0], wavenumber_range[1], points)
    
    # Silicon phonon peak at ~520 cm⁻¹
    # Strain shifts peak: ~5 cm⁻¹/GPa
    peak_position = 520 - 5 * strain
    peak_width = 3 + 10 * (1 - crystallinity)  # Broader for amorphous
    
    # Lorentzian peak
    intensity = (peak_width / 2)**2 / ((wavenumber - peak_position)**2 + (peak_width / 2)**2)
    intensity = intensity / np.max(intensity) * 1000  # Normalize
    
    # Baseline
    baseline = 50 + 10 * wavenumber / wavenumber_range[1]
    
    # Total spectrum
    spectrum = intensity + baseline
    
    # Add noise
    spectrum = self.add_noise(spectrum, relative=False)
    
    return {
        "wavenumber": wavenumber.tolist(),
        "intensity": spectrum.tolist(),
        "parameters": {
            "peak_position": peak_position,
            "peak_width": peak_width,
            "strain": strain,
            "crystallinity": crystallinity
        },
        "metadata": self.generate_metadata("raman", material="Si")
    }

# ============================================================================

# Structural Data Generators

# ============================================================================

class XRDGenerator(BaseGenerator):
“”“Generate X-ray diffraction patterns”””

def generate_powder(
    self,
    material: str = "Si",
    two_theta_range: Tuple[float, float] = (20, 80),  # degrees
    points: int = 2000,
    crystallite_size: float = 50  # nm
) -> Dict[str, Any]:
    """Generate XRD pattern for powder sample"""
    
    two_theta = np.linspace(two_theta_range[0], two_theta_range[1], points)
    
    # Si peaks: (111) at 28.4°, (220) at 47.3°, (311) at 56.1°
    peaks = [
        {"position": 28.4, "intensity": 100, "hkl": "(111)"},
        {"position": 47.3, "intensity": 55, "hkl": "(220)"},
        {"position": 56.1, "intensity": 30, "hkl": "(311)"}
    ]
    
    # Peak broadening from Scherrer equation
    # FWHM (rad) = K*lambda / (L*cos(theta))
    K = 0.9  # Shape factor
    wavelength = 1.5406  # Å (Cu K-alpha)
    
    intensity = np.zeros_like(two_theta)
    
    for peak in peaks:
        theta_rad = np.radians(peak["position"] / 2)
        fwhm = np.degrees(K * wavelength * 1e-10 / (crystallite_size * 1e-9 * np.cos(theta_rad)))
        
        # Gaussian peak
        peak_intensity = peak["intensity"] * np.exp(
            -0.5 * ((two_theta - peak["position"]) / fwhm)**2
        )
        intensity += peak_intensity
    
    # Background
    background = 50 + 20 * np.exp(-two_theta / 30)
    intensity += background
    
    # Add noise
    intensity = self.add_noise(intensity, relative=False)
    
    return {
        "two_theta": two_theta.tolist(),
        "intensity": intensity.tolist(),
        "parameters": {
            "material": material,
            "crystallite_size": crystallite_size,
            "peaks": peaks
        },
        "metadata": self.generate_metadata("xrd", material=material)
    }

class AFMGenerator(BaseGenerator):
“”“Generate AFM topography data”””

def generate_surface(
    self,
    size: float = 10,  # μm
    points: int = 512,
    rms_roughness: float = 5  # nm
) -> Dict[str, Any]:
    """Generate AFM height map"""
    
    x = np.linspace(0, size, points)
    y = np.linspace(0, size, points)
    X, Y = np.meshgrid(x, y)
    
    # Generate fractal-like surface using multiple frequency components
    height = np.zeros_like(X)
    for i in range(1, 6):
        freq = i * 2 * np.pi / size
        phase_x = np.random.rand() * 2 * np.pi
        phase_y = np.random.rand() * 2 * np.pi
        amplitude = rms_roughness / i
        height += amplitude * np.sin(freq * X + phase_x) * np.sin(freq * Y + phase_y)
    
    # Normalize to desired RMS roughness
    height = height / np.std(height) * rms_roughness
    
    # Add random high-frequency noise
    height += np.random.normal(0, rms_roughness * 0.1, height.shape)
    
    # Calculate statistics
    Ra = np.mean(np.abs(height - np.mean(height)))
    Rq = np.sqrt(np.mean((height - np.mean(height))**2))
    Rsk = np.mean(((height - np.mean(height)) / Rq)**3)
    Rku = np.mean(((height - np.mean(height)) / Rq)**4)
    
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "height": height.tolist(),
        "parameters": {
            "Ra": float(Ra),  # Average roughness
            "Rq": float(Rq),  # RMS roughness
            "Rsk": float(Rsk),  # Skewness
            "Rku": float(Rku),  # Kurtosis
            "size": size,
            "points": points
        },
        "metadata": self.generate_metadata("afm")
    }

# ============================================================================

# Chemical Data Generators

# ============================================================================

class XPSGenerator(BaseGenerator):
“”“Generate XPS spectra”””

def generate_survey(
    self,
    material: str = "SiO2",
    binding_energy_range: Tuple[float, float] = (0, 1200),  # eV
    points: int = 2000
) -> Dict[str, Any]:
    """Generate XPS survey spectrum"""
    
    binding_energy = np.linspace(binding_energy_range[0], binding_energy_range[1], points)
    
    # SiO₂ peaks: O 1s (~532 eV), Si 2p (~103 eV)
    peaks = [
        {"position": 103, "width": 2, "intensity": 100, "element": "Si 2p"},
        {"position": 532, "width": 2, "intensity": 200, "element": "O 1s"}
    ]
    
    # Shirley background
    background = 1000 - 0.5 * binding_energy
    
    intensity = background.copy()
    
    for peak in peaks:
        # Voigt profile (Gaussian + Lorentzian)
        gaussian = peak["intensity"] * np.exp(
            -0.5 * ((binding_energy - peak["position"]) / peak["width"])**2
        )
        intensity += gaussian
    
    # Add noise
    intensity = self.add_noise(intensity, relative=False)
    intensity = np.maximum(intensity, 0)
    
    return {
        "binding_energy": binding_energy.tolist(),
        "intensity": intensity.tolist(),
        "parameters": {
            "material": material,
            "peaks": peaks
        },
        "metadata": self.generate_metadata("xps", material=material)
    }

# ============================================================================

# Batch Generator

# ============================================================================

class TestDataGenerator:
“”“Orchestrate generation of all test data”””

def __init__(self, output_dir: str = "data/test_data", config: Optional[GeneratorConfig] = None):
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(parents=True, exist_ok=True)
    self.config = config or GeneratorConfig()
    
    # Initialize generators
    self.iv_gen = IVGenerator(self.config)
    self.hall_gen = HallGenerator(self.config)
    self.fpp_gen = FourPointProbeGenerator(self.config)
    self.uvvis_gen = UVVisNIRGenerator(self.config)
    self.raman_gen = RamanGenerator(self.config)
    self.xrd_gen = XRDGenerator(self.config)
    self.afm_gen = AFMGenerator(self.config)
    self.xps_gen = XPSGenerator(self.config)

def generate_all(self) -> Dict[str, str]:
    """Generate complete test dataset"""
    
    files = {}
    
    print("Generating test data...")
    
    # Electrical
    print("  Electrical methods...")
    files["diode_iv"] = self._save("electrical/diode_iv.json", self.iv_gen.generate_diode())
    files["solar_cell_iv"] = self._save("electrical/solar_cell_iv.json", self.iv_gen.generate_solar_cell())
    files["hall_si"] = self._save("electrical/hall_si.json", self.hall_gen.generate(material="Si", carrier_type="n"))
    files["4pp_map"] = self._save("electrical/4pp_wafer_map.json", self.fpp_gen.generate_wafer_map())
    
    # Optical
    print("  Optical methods...")
    files["uvvis_gaas"] = self._save("optical/uvvis_gaas.json", self.uvvis_gen.generate_semiconductor(material="GaAs"))
    files["raman_si"] = self._save("optical/raman_si.json", self.raman_gen.generate_silicon())
    
    # Structural
    print("  Structural methods...")
    files["xrd_si"] = self._save("structural/xrd_si.json", self.xrd_gen.generate_powder(material="Si"))
    files["afm_surface"] = self._save("structural/afm_surface.json", self.afm_gen.generate_surface())
    
    # Chemical
    print("  Chemical methods...")
    files["xps_sio2"] = self._save("chemical/xps_sio2.json", self.xps_gen.generate_survey(material="SiO2"))
    
    print(f"✓ Generated {len(files)} test datasets in {self.output_dir}")
    
    # Create manifest
    manifest = {
        "generated_at": datetime.utcnow().isoformat(),
        "config": {
            "noise_level": self.config.noise_level,
            "has_outliers": self.config.add_outliers,
            "seed": self.config.seed
        },
        "files": files
    }
    
    with open(self.output_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    return files

def _save(self, rel_path: str, data: Dict[str, Any]) -> str:
    """Save data to file and return path"""
    filepath = self.output_dir / rel_path
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return str(filepath)

# ============================================================================

# Main

# ============================================================================

if **name** == “**main**”:
config = GeneratorConfig(
add_noise=True,
noise_level=0.02,
add_outliers=False,
seed=42
)

generator = TestDataGenerator(output_dir="data/test_data", config=config)
files = generator.generate_all()

print("\n" + "="*80)
print("Test Data Generation Complete!")
print("="*80)
print(f"\nGenerated files:")
for name, path in files.items():
    print(f"  {name:20s} -> {path}")
print(f"\nManifest: data/test_data/manifest.json")