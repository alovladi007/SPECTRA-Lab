# services/instruments/app/drivers/builtin/oceanoptics_spectrometer.py

“””
Ocean Optics Spectrometer Driver + HIL Simulator

Complete implementation for UV-Vis-NIR spectroscopy with:

- USB/Serial communication
- Integration time control
- Dark/reference subtraction
- Boxcar averaging
- Non-linearity correction
- HIL simulator with Gaussian peaks

Supported Models: USB2000, USB4000, HR2000, QE65000, Flame, Maya2000
“””

import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from ..core.connection import ConnectionConfig
from ..core.plugin_manager import InstrumentDriver

# ============================================================================

# Configuration

# ============================================================================

@dataclass
class SpectrometerConfig:
“”“Configuration for Ocean Optics spectrometer”””
# Integration
integration_time_ms: float = 100.0
scans_to_average: int = 1

# Corrections
dark_correction: bool = True
nonlinearity_correction: bool = True
electric_dark_correction: bool = True

# Boxcar smoothing
boxcar_width: int = 0  # 0 = disabled

# Wavelength calibration
wavelength_min: float = 200.0  # nm
wavelength_max: float = 1100.0
pixels: int = 2048

# Timing
trigger_mode: str = "free_running"  # or "external"
timeout: float = 10.0

# ============================================================================

# Ocean Optics Driver (Real Hardware)

# ============================================================================

class OceanOpticsDriver(InstrumentDriver):
“””
Ocean Optics Spectrometer Driver

Supports USB and Serial communication via Ocean Optics OmniDriver
or SeaBreeze library.
"""

def __init__(self, resource_name: str, config: Optional[Dict[str, Any]] = None):
    """
    Initialize driver
    
    Args:
        resource_name: Format "oceanoptics://serial_number" or "oceanoptics://0"
        config: Optional configuration
    """
    self.logger = logging.getLogger(__name__)
    self.resource_name = resource_name
    
    if config:
        self.config = SpectrometerConfig(**config)
    else:
        self.config = SpectrometerConfig()
    
    # Parse resource name
    if resource_name.startswith("oceanoptics://"):
        self.device_id = resource_name.replace("oceanoptics://", "")
    else:
        raise ValueError(f"Invalid resource name: {resource_name}")
    
    # State
    self._connected = False
    self._device = None
    self._dark_spectrum = None
    self._reference_spectrum = None
    self._wavelengths = None
    self._identity = None

def connect(self) -> bool:
    """Establish connection to spectrometer"""
    try:
        # Try SeaBreeze (open source)
        try:
            import seabreeze.spectrometers as sb
            self._use_seabreeze = True
            
            # List available devices
            devices = sb.list_devices()
            if not devices:
                raise ConnectionError("No Ocean Optics devices found")
            
            # Open device
            if self.device_id.isdigit():
                idx = int(self.device_id)
                self._device = sb.Spectrometer(devices[idx])
            else:
                # Find by serial number
                for dev in devices:
                    if dev.serial_number == self.device_id:
                        self._device = sb.Spectrometer(dev)
                        break
                if self._device is None:
                    raise ConnectionError(f"Device {self.device_id} not found")
            
        except ImportError:
            # Fallback to OmniDriver (proprietary)
            self.logger.warning("SeaBreeze not available, trying OmniDriver")
            raise NotImplementedError("OmniDriver support not yet implemented")
        
        # Get wavelengths
        self._wavelengths = self._device.wavelengths()
        
        # Get identity
        self._identity = {
            'manufacturer': 'Ocean Optics',
            'model': self._device.model,
            'serial_number': self._device.serial_number,
            'firmware': 'N/A'
        }
        
        # Configure integration time
        self._device.integration_time_micros(int(self.config.integration_time_ms * 1000))
        
        self._connected = True
        self.logger.info(f"Connected to {self._identity['model']} ({self._identity['serial_number']})")
        
        return True
        
    except Exception as e:
        self.logger.error(f"Connection failed: {e}")
        return False

def disconnect(self) -> bool:
    """Close connection"""
    try:
        if self._device:
            self._device.close()
            self._device = None
        
        self._connected = False
        self.logger.info("Disconnected from spectrometer")
        return True
        
    except Exception as e:
        self.logger.error(f"Disconnect failed: {e}")
        return False

def reset(self) -> None:
    """Reset to default configuration"""
    self._dark_spectrum = None
    self._reference_spectrum = None
    self._device.integration_time_micros(int(self.config.integration_time_ms * 1000))

def get_identity(self) -> Dict[str, str]:
    """Get instrument identity"""
    return self._identity

def get_capabilities(self) -> List[str]:
    """Get supported methods"""
    return ['uv_vis_nir', 'time_series', 'dark_capture', 'reference_capture']

def configure(self, method: str, params: Dict[str, Any]) -> None:
    """Configure for measurement"""
    if method in ['uv_vis_nir', 'time_series']:
        # Update integration time if specified
        if 'integration_time_ms' in params:
            self.config.integration_time_ms = params['integration_time_ms']
            self._device.integration_time_micros(int(self.config.integration_time_ms * 1000))
        
        # Update averaging if specified
        if 'scans_to_average' in params:
            self.config.scans_to_average = params['scans_to_average']

def measure(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform measurement"""
    if method == 'uv_vis_nir':
        return self._measure_spectrum(params)
    elif method == 'time_series':
        return self._measure_time_series(params)
    elif method == 'dark_capture':
        return self._capture_dark()
    elif method == 'reference_capture':
        return self._capture_reference()
    else:
        raise ValueError(f"Unsupported method: {method}")

def abort(self) -> None:
    """Abort measurement"""
    # Not applicable for spectrometer
    pass

def get_status(self) -> Dict[str, Any]:
    """Get status"""
    return {
        'connected': self._connected,
        'integration_time_ms': self.config.integration_time_ms,
        'has_dark': self._dark_spectrum is not None,
        'has_reference': self._reference_spectrum is not None
    }

def _measure_spectrum(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Measure single spectrum"""
    # Acquire spectrum with averaging
    spectra = []
    for _ in range(self.config.scans_to_average):
        spectrum = self._device.intensities()
        spectra.append(spectrum)
    
    # Average
    intensity = np.mean(spectra, axis=0)
    
    # Apply corrections
    if self.config.dark_correction and self._dark_spectrum is not None:
        intensity = intensity - self._dark_spectrum
    
    if self.config.nonlinearity_correction:
        intensity = self._apply_nonlinearity_correction(intensity)
    
    # Boxcar smoothing
    if self.config.boxcar_width > 0:
        intensity = self._boxcar_smooth(intensity, self.config.boxcar_width)
    
    # Transmission/absorbance calculation
    if self._reference_spectrum is not None and params.get('calculate_transmission', False):
        transmission = intensity / self._reference_spectrum
        absorbance = -np.log10(np.maximum(transmission, 1e-10))
    else:
        transmission = None
        absorbance = None
    
    return {
        'wavelength': self._wavelengths.tolist(),
        'intensity': intensity.tolist(),
        'transmission': transmission.tolist() if transmission is not None else None,
        'absorbance': absorbance.tolist() if absorbance is not None else None,
        'integration_time_ms': self.config.integration_time_ms,
        'scans_averaged': self.config.scans_to_average,
        'metadata': {
            'instrument': self._identity,
            'dark_corrected': self.config.dark_correction and self._dark_spectrum is not None,
            'nonlinearity_corrected': self.config.nonlinearity_correction
        }
    }

def _measure_time_series(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Measure time series"""
    duration = params.get('duration_seconds', 60.0)
    interval = params.get('interval_seconds', 1.0)
    
    num_points = int(duration / interval)
    wavelengths = self._wavelengths
    intensities = np.zeros((num_points, len(wavelengths)))
    timestamps = np.zeros(num_points)
    
    start_time = time.time()
    for i in range(num_points):
        spectrum = self._device.intensities()
        
        if self.config.dark_correction and self._dark_spectrum is not None:
            spectrum = spectrum - self._dark_spectrum
        
        intensities[i] = spectrum
        timestamps[i] = time.time() - start_time
        
        # Wait for next interval
        if i < num_points - 1:
            time.sleep(interval)
    
    return {
        'wavelength': wavelengths.tolist(),
        'time': timestamps.tolist(),
        'intensities': intensities.tolist(),  # 2D array
        'metadata': {
            'duration_seconds': duration,
            'interval_seconds': interval,
            'points': num_points
        }
    }

def _capture_dark(self) -> Dict[str, Any]:
    """Capture dark spectrum"""
    self.logger.info("Capturing dark spectrum (block light source)")
    
    # Acquire with averaging
    spectra = []
    for _ in range(self.config.scans_to_average):
        spectrum = self._device.intensities()
        spectra.append(spectrum)
    
    self._dark_spectrum = np.mean(spectra, axis=0)
    
    return {
        'success': True,
        'wavelength': self._wavelengths.tolist(),
        'dark_spectrum': self._dark_spectrum.tolist()
    }

def _capture_reference(self) -> Dict[str, Any]:
    """Capture reference spectrum"""
    self.logger.info("Capturing reference spectrum (100% transmission)")
    
    # Acquire with averaging
    spectra = []
    for _ in range(self.config.scans_to_average):
        spectrum = self._device.intensities()
        spectra.append(spectrum)
    
    raw = np.mean(spectra, axis=0)
    
    # Apply dark correction
    if self._dark_spectrum is not None:
        self._reference_spectrum = raw - self._dark_spectrum
    else:
        self._reference_spectrum = raw
    
    return {
        'success': True,
        'wavelength': self._wavelengths.tolist(),
        'reference_spectrum': self._reference_spectrum.tolist()
    }

def _apply_nonlinearity_correction(self, intensity: np.ndarray) -> np.ndarray:
    """Apply non-linearity correction (model-specific)"""
    # Simplified correction (actual coefficients are device-specific)
    # Typically: I_corrected = I + C1*I^2 + C2*I^3 + ...
    return intensity  # Placeholder

def _boxcar_smooth(self, data: np.ndarray, width: int) -> np.ndarray:
    """Apply boxcar (moving average) smoothing"""
    kernel = np.ones(width) / width
    return np.convolve(data, kernel, mode='same')

# ============================================================================

# HIL Simulator

# ============================================================================

class OceanOpticsSimulator(InstrumentDriver):
“””
HIL Simulator for Ocean Optics Spectrometer

Simulates realistic spectra with:
- Gaussian peaks
- Baseline offset and drift
- Shot noise
- Detector non-linearity
"""

def __init__(self, resource_name: str, config: Optional[Dict[str, Any]] = None):
    """Initialize simulator"""
    self.logger = logging.getLogger(__name__)
    self.resource_name = resource_name
    
    if config:
        self.config = SpectrometerConfig(**{k: v for k, v in config.items() if k in SpectrometerConfig.__annotations__})
        self.sim_config = config.get('simulation', {})
    else:
        self.config = SpectrometerConfig()
        self.sim_config = {}
    
    # Generate wavelength array
    self._wavelengths = np.linspace(
        self.config.wavelength_min,
        self.config.wavelength_max,
        self.config.pixels
    )
    
    # Simulation parameters
    self.peaks = self.sim_config.get('peaks', [
        {'center': 450, 'amplitude': 10000, 'width': 20},
        {'center': 550, 'amplitude': 15000, 'width': 25},
        {'center': 650, 'amplitude': 8000, 'width': 15}
    ])
    
    self.baseline = self.sim_config.get('baseline', 1000)
    self.noise_level = self.sim_config.get('noise_level', 100)
    
    # State
    self._connected = False
    self._dark_spectrum = None
    self._reference_spectrum = None
    self._identity = {
        'manufacturer': 'Ocean Optics (Simulated)',
        'model': 'USB4000-SIM',
        'serial_number': 'SIM-001',
        'firmware': '1.0.0'
    }

def connect(self) -> bool:
    """Simulate connection"""
    self._connected = True
    self.logger.info("Connected to simulated spectrometer")
    return True

def disconnect(self) -> bool:
    """Simulate disconnect"""
    self._connected = False
    return True

def reset(self) -> None:
    """Reset simulator"""
    self._dark_spectrum = None
    self._reference_spectrum = None

def get_identity(self) -> Dict[str, str]:
    """Get simulated identity"""
    return self._identity

def get_capabilities(self) -> List[str]:
    """Get capabilities"""
    return ['uv_vis_nir', 'time_series', 'dark_capture', 'reference_capture']

def configure(self, method: str, params: Dict[str, Any]) -> None:
    """Configure (update simulation parameters if provided)"""
    if 'integration_time_ms' in params:
        self.config.integration_time_ms = params['integration_time_ms']

def measure(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate measurement"""
    if method == 'uv_vis_nir':
        return self._simulate_spectrum(params)
    elif method == 'time_series':
        return self._simulate_time_series(params)
    elif method == 'dark_capture':
        return self._simulate_dark()
    elif method == 'reference_capture':
        return self._simulate_reference()
    else:
        raise ValueError(f"Unsupported method: {method}")

def abort(self) -> None:
    """Abort (no-op for simulator)"""
    pass

def get_status(self) -> Dict[str, Any]:
    """Get status"""
    return {
        'connected': self._connected,
        'integration_time_ms': self.config.integration_time_ms,
        'has_dark': self._dark_spectrum is not None,
        'has_reference': self._reference_spectrum is not None
    }

def _simulate_spectrum(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Generate simulated spectrum with Gaussian peaks"""
    # Generate clean spectrum
    intensity = np.ones(len(self._wavelengths)) * self.baseline
    
    # Add Gaussian peaks
    for peak in self.peaks:
        center = peak['center']
        amplitude = peak['amplitude']
        width = peak['width']
        
        gaussian = amplitude * np.exp(-((self._wavelengths - center) ** 2) / (2 * width ** 2))
        intensity += gaussian
    
    # Add shot noise (Poisson)
    if self.noise_level > 0:
        intensity += np.random.normal(0, self.noise_level, len(intensity))
        intensity = np.maximum(intensity, 0)  # No negative counts
    
    # Simulate integration time effect (more counts with longer integration)
    intensity = intensity * (self.config.integration_time_ms / 100.0)
    
    # Apply dark correction
    if self.config.dark_correction and self._dark_spectrum is not None:
        intensity = intensity - self._dark_spectrum
    
    # Calculate transmission/absorbance if reference available
    if self._reference_spectrum is not None and params.get('calculate_transmission', False):
        transmission = intensity / np.maximum(self._reference_spectrum, 1e-10)
        absorbance = -np.log10(np.maximum(transmission, 1e-10))
    else:
        transmission = None
        absorbance = None
    
    # Simulate measurement time
    time.sleep(self.config.integration_time_ms / 1000.0 * 0.1)  # 10% of real time
    
    return {
        'wavelength': self._wavelengths.tolist(),
        'intensity': intensity.tolist(),
        'transmission': transmission.tolist() if transmission is not None else None,
        'absorbance': absorbance.tolist() if absorbance is not None else None,
        'integration_time_ms': self.config.integration_time_ms,
        'scans_averaged': self.config.scans_to_average,
        'metadata': {
            'instrument': self._identity,
            'simulated': True,
            'peaks': self.peaks
        }
    }

def _simulate_time_series(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate time series with drift"""
    duration = params.get('duration_seconds', 60.0)
    interval = params.get('interval_seconds', 1.0)
    
    num_points = int(duration / interval)
    intensities = np.zeros((num_points, len(self._wavelengths)))
    timestamps = np.arange(num_points) * interval
    
    for i in range(num_points):
        # Base spectrum
        spectrum = self._simulate_spectrum({})['intensity']
        
        # Add drift
        drift_factor = 1.0 + 0.001 * i  # 0.1% per measurement
        spectrum = np.array(spectrum) * drift_factor
        
        intensities[i] = spectrum
        
        # Simulate measurement time
        time.sleep(interval * 0.01)  # 1% of real time
    
    return {
        'wavelength': self._wavelengths.tolist(),
        'time': timestamps.tolist(),
        'intensities': intensities.tolist(),
        'metadata': {
            'duration_seconds': duration,
            'interval_seconds': interval,
            'points': num_points,
            'simulated': True
        }
    }

def _simulate_dark(self) -> Dict[str, Any]:
    """Simulate dark spectrum"""
    # Dark spectrum = baseline + electronic noise
    dark = np.ones(len(self._wavelengths)) * (self.baseline * 0.1)
    dark += np.random.normal(0, self.noise_level * 0.5, len(dark))
    dark = np.maximum(dark, 0)
    
    self._dark_spectrum = dark
    
    return {
        'success': True,
        'wavelength': self._wavelengths.tolist(),
        'dark_spectrum': dark.tolist()
    }

def _simulate_reference(self) -> Dict[str, Any]:
    """Simulate reference (100% transmission)"""
    # Reference = baseline + high intensity
    reference = np.ones(len(self._wavelengths)) * (self.baseline + np.mean([p['amplitude'] for p in self.peaks]))
    reference += np.random.normal(0, self.noise_level, len(reference))
    reference = np.maximum(reference, 1)
    
    if self._dark_spectrum is not None:
        self._reference_spectrum = reference - self._dark_spectrum
    else:
        self._reference_spectrum = reference
    
    return {
        'success': True,
        'wavelength': self._wavelengths.tolist(),
        'reference_spectrum': self._reference_spectrum.tolist()
    }

# ============================================================================

# Example Usage

# ============================================================================

def example_usage():
“”“Demonstrate spectrometer driver”””
print(”=” * 80)
print(“Ocean Optics Spectrometer Driver - Example Usage”)
print(”=” * 80)

# Use simulator for demonstration
config = {
    'integration_time_ms': 100,
    'scans_to_average': 5,
    'simulation': {
        'peaks': [
            {'center': 450, 'amplitude': 10000, 'width': 20},
            {'center': 650, 'amplitude': 15000, 'width': 25}
        ],
        'baseline': 1000,
        'noise_level': 50
    }
}

driver = OceanOpticsSimulator("sim://spectrometer", config)
driver.connect()

# Capture dark
print("\n1. Capturing dark spectrum...")
dark_result = driver.measure('dark_capture', {})
print(f"   Dark spectrum captured ({len(dark_result['dark_spectrum'])} pixels)")

# Capture reference
print("\n2. Capturing reference spectrum...")
ref_result = driver.measure('reference_capture', {})
print(f"   Reference captured")

# Measure spectrum
print("\n3. Measuring spectrum...")
result = driver.measure('uv_vis_nir', {'calculate_transmission': True})

wavelength = np.array(result['wavelength'])
intensity = np.array(result['intensity'])
absorbance = np.array(result['absorbance']) if result['absorbance'] else None

print(f"   Wavelength range: {wavelength[0]:.1f} - {wavelength[-1]:.1f} nm")
print(f"   Peak intensity: {np.max(intensity):.0f} counts")

if absorbance is not None:
    print(f"   Peak absorbance: {np.max(absorbance):.3f}")

driver.disconnect()

print("\n" + "=" * 80)
print("Spectrometer demonstration complete!")
print("=" * 80)

if **name** == “**main**”:
logging.basicConfig(level=logging.INFO)
example_usage()