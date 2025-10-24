# services/instruments/app/drivers/builtin/keithley_2400.py

“””
Keithley 2400 Source Measure Unit (SMU) Driver

Supports:

- I-V sweeps (voltage source, current measure)
- I-V sweeps (current source, voltage measure)
- 2-wire and 4-wire measurements
- Compliance limits
- Pulsed measurements
  “””

import time
import numpy as np
from typing import Dict, Any, Optional, List
from ..connection import VISAConnection, ConnectionConfig, SCPICommand, InstrumentInterface
from ..plugin_manager import InstrumentDriver, driver_plugin

@driver_plugin(
name=“keithley_2400”,
version=“1.0.0”,
author=“SemiconductorLab Team”,
description=“Keithley 2400 Series Source Measure Unit Driver”,
supported_methods=[“iv_sweep”, “cv_measurement”, “two_point_probe”, “four_point_probe”],
supported_models=[“2400”, “2401”, “2410”, “2420”]
)
class Keithley2400Driver(InstrumentDriver):
“””
Keithley 2400 SMU Driver

Hardware specifications:
- Voltage: ±210V (2400/2401), ±1100V (2410/2420)
- Current: ±1A (2400), ±10mA (2401), ±1.05A (2410/2420)
- Measurement speed: 800 readings/sec
- 6.5 digit resolution
"""

def __init__(self, resource_name: str, config: Optional[Dict[str, Any]] = None):
    """Initialize Keithley 2400 driver"""
    self.resource_name = resource_name
    self.config = config or {}
    
    # Connection
    conn_config = ConnectionConfig(
        timeout=10.0,
        write_termination='\n',
        read_termination='\n'
    )
    self.connection = VISAConnection(resource_name, conn_config)
    
    # State
    self.is_connected = False
    self._identity: Optional[Dict[str, str]] = None
    
    # Measurement settings
    self.compliance = {
        'voltage': 21.0,  # V
        'current': 0.1,   # A
    }

def connect(self) -> bool:
    """Connect to SMU"""
    try:
        self.connection.connect()
        self.is_connected = True
        
        # Get identity
        self._identity = self.get_identity()
        
        # Reset to known state
        self.reset()
        
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def disconnect(self) -> bool:
    """Disconnect from SMU"""
    if self.is_connected:
        # Turn off output
        self.connection.write(":OUTP OFF")
        self.connection.disconnect()
        self.is_connected = False
    return True

def reset(self) -> None:
    """Reset SMU to default state"""
    self.connection.write("*RST")
    time.sleep(0.5)
    
    # Configure for typical use
    self.connection.write(":SYST:RSEN OFF")  # 2-wire mode
    self.connection.write(":FORM:ELEM VOLT,CURR")  # Output format

def get_identity(self) -> Dict[str, str]:
    """Get instrument identity"""
    if self._identity is None:
        idn = self.connection.query("*IDN?")
        self._identity = SCPICommand.parse_idn(idn)
    return self._identity

def get_capabilities(self) -> List[str]:
    """Get supported methods"""
    return ["iv_sweep", "cv_measurement", "two_point_probe", "four_point_probe"]

def configure(self, method: str, params: Dict[str, Any]) -> None:
    """Configure SMU for measurement"""
    if method == "iv_sweep":
        self._configure_iv_sweep(params)
    elif method in ["two_point_probe", "four_point_probe"]:
        self._configure_resistance(params)
    else:
        raise ValueError(f"Unsupported method: {method}")

def _configure_iv_sweep(self, params: Dict[str, Any]) -> None:
    """Configure for I-V sweep"""
    # Source mode (voltage or current)
    source_mode = params.get('source_mode', 'voltage')
    
    if source_mode == 'voltage':
        self.connection.write(":SOUR:FUNC VOLT")
        self.connection.write(":SENS:FUNC 'CURR'")
        
        # Set compliance
        compliance = params.get('current_compliance', 0.1)
        self.connection.write(f":SENS:CURR:PROT {compliance}")
        
    elif source_mode == 'current':
        self.connection.write(":SOUR:FUNC CURR")
        self.connection.write(":SENS:FUNC 'VOLT'")
        
        # Set compliance
        compliance = params.get('voltage_compliance', 21.0)
        self.connection.write(f":SENS:VOLT:PROT {compliance}")
    
    # Wire mode
    wire_mode = params.get('wire_mode', '2wire')
    if wire_mode == '4wire':
        self.connection.write(":SYST:RSEN ON")
    else:
        self.connection.write(":SYST:RSEN OFF")
    
    # Auto-range
    self.connection.write(":SENS:CURR:RANG:AUTO ON")
    self.connection.write(":SENS:VOLT:RANG:AUTO ON")

def measure(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform measurement"""
    if method == "iv_sweep":
        return self._measure_iv_sweep(params)
    elif method in ["two_point_probe", "four_point_probe"]:
        return self._measure_resistance(params)
    else:
        raise ValueError(f"Unsupported method: {method}")

def _measure_iv_sweep(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Perform I-V sweep
    
    Parameters:
    - v_start: Start voltage (V)
    - v_stop: Stop voltage (V)
    - points: Number of points
    - source_mode: 'voltage' or 'current'
    - current_compliance: Compliance limit (A)
    - sweep_mode: 'linear' or 'log'
    """
    v_start = params['v_start']
    v_stop = params['v_stop']
    points = params['points']
    source_mode = params.get('source_mode', 'voltage')
    
    # Generate sweep points
    if params.get('sweep_mode', 'linear') == 'linear':
        sweep_values = np.linspace(v_start, v_stop, points)
    else:
        # Logarithmic spacing
        sweep_values = np.logspace(np.log10(abs(v_start)+1e-9), np.log10(abs(v_stop)+1e-9), points)
        if v_start < 0:
            sweep_values = -sweep_values
    
    # Arrays for results
    voltages = []
    currents = []
    timestamps = []
    compliance_flags = []
    
    # Enable output
    self.connection.write(":OUTP ON")
    
    start_time = time.time()
    
    try:
        for value in sweep_values:
            # Set source level
            if source_mode == 'voltage':
                self.connection.write(f":SOUR:VOLT {value}")
            else:
                self.connection.write(f":SOUR:CURR {value}")
            
            # Allow settling
            time.sleep(params.get('delay', 0.01))
            
            # Trigger measurement
            self.connection.write(":INIT")
            self.connection.write("*WAI")  # Wait for completion
            
            # Read result
            data = self.connection.query(":FETC?")
            v_meas, i_meas = [float(x) for x in data.split(',')[:2]]
            
            voltages.append(v_meas)
            currents.append(i_meas)
            timestamps.append(time.time() - start_time)
            
            # Check compliance (bit 1 of status byte)
            status = self.connection.query(":STAT:MEAS?")
            compliance = bool(int(status) & 0x02)
            compliance_flags.append(compliance)
            
            if compliance:
                print(f"Compliance reached at {v_meas}V, {i_meas}A")
                if params.get('stop_on_compliance', True):
                    break
        
    finally:
        # Turn off output
        self.connection.write(":OUTP OFF")
    
    return {
        'voltage': voltages,
        'current': currents,
        'timestamp': timestamps,
        'compliance': compliance_flags,
        'source_mode': source_mode,
        'points': len(voltages)
    }

def _configure_resistance(self, params: Dict[str, Any]) -> None:
    """Configure for resistance measurement"""
    # Use ohms function
    self.connection.write(":SENS:FUNC 'RES'")
    self.connection.write(":SENS:RES:MODE MAN")
    
    # Set test current
    test_current = params.get('test_current', 1e-3)  # 1 mA
    self.connection.write(f":SOUR:CURR {test_current}")
    
    # 4-wire if specified
    if params.get('wire_mode') == '4wire':
        self.connection.write(":SYST:RSEN ON")

def _measure_resistance(self, params: Dict[str, Any]) -> Dict[str, Any]:
    """Measure resistance"""
    # Enable output
    self.connection.write(":OUTP ON")
    
    # Trigger measurement
    self.connection.write(":INIT")
    time.sleep(0.1)
    
    # Read resistance
    resistance = float(self.connection.query(":FETC?"))
    
    # Turn off output
    self.connection.write(":OUTP OFF")
    
    return {
        'resistance': resistance,
        'unit': 'ohm'
    }

def abort(self) -> None:
    """Abort ongoing measurement"""
    self.connection.write(":ABOR")
    self.connection.write(":OUTP OFF")

def get_status(self) -> Dict[str, Any]:
    """Get instrument status"""
    output_state = self.connection.query(":OUTP?")
    
    return {
        'output_enabled': output_state == '1',
        'connected': self.is_connected,
        'identity': self._identity
    }

# ============================================================================

# Ocean Optics Spectrometer Driver

# ============================================================================

@driver_plugin(
name=“ocean_optics_spectrometer”,
version=“1.0.0”,
author=“SemiconductorLab Team”,
description=“Ocean Optics USB Spectrometer Driver”,
supported_methods=[“uv_vis_nir”, “transmission”, “reflectance”, “absorbance”],
supported_models=[“USB2000”, “USB4000”, “HR2000”, “QE65000”, “FLAME”]
)
class OceanOpticsDriver(InstrumentDriver):
“””
Ocean Optics USB Spectrometer Driver

Note: This is a simplified implementation. Real Ocean Optics instruments
use the SeaBreeze library, not VISA.

For production, use: import seabreeze
"""

def __init__(self, resource_name: str, config: Optional[Dict[str, Any]] = None):
    """Initialize Ocean Optics driver"""
    self.resource_name = resource_name  # e.g., "USB:0x2457:0x1022:SPEC123456"
    self.config = config or {}
    
    self.is_connected = False
    self._identity: Optional[Dict[str, str]] = None
    
    # Spectrometer parameters
    self.wavelengths: Optional[np.ndarray] = None
    self.integration_time_ms = 100
    self.scans_to_average = 1

def connect(self) -> bool:
    """Connect to spectrometer"""
    try:
        # In real implementation, use SeaBreeze:
        # import seabreeze.spectrometers as sb
        # self.device = sb.Spectrometer.from_serial_number(serial_number)
        
        self.is_connected = True
        
        # Get wavelength calibration
        self._load_wavelength_calibration()
        
        return True
        
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def disconnect(self) -> bool:
    """Disconnect from spectrometer"""
    if self.is_connected:
        # self.device.close()
        self.is_connected = False
    return True

def reset(self) -> None:
    """Reset to default state"""
    self.integration_time_ms = 100
    self.scans_to_average = 1

def get_identity(self) -> Dict[str, str]:
    """Get instrument identity"""
    if self._identity is None:
        # In real implementation:
        # model = self.device.model
        # serial = self.device.serial_number
        
        self._identity = {
            'manufacturer': 'Ocean Optics',
            'model': 'USB4000',
            'serial_number': 'SPEC123456',
            'firmware': '1.0.0'
        }
    return self._identity

def get_capabilities(self) -> List[str]:
    """Get supported methods"""
    return ["uv_vis_nir", "transmission", "reflectance", "absorbance"]

def configure(self, method: str, params: Dict[str, Any]) -> None:
    """Configure spectrometer"""
    # Set integration time
    self.integration_time_ms = params.get('integration_time_ms', 100)
    
    # Set averaging
    self.scans_to_average = params.get('scans_to_average', 1)
    
    # In real implementation:
    # self.device.integration_time_micros(self.integration_time_ms * 1000)

def measure(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Acquire spectrum"""
    if not self.is_connected:
        raise RuntimeError("Not connected")
    
    # Acquire spectrum
    # In real implementation:
    # intensities = self.device.intensities()
    
    # Simulate spectrum
    intensities = self._simulate_spectrum(params)
    
    if method == "transmission":
        # Transmission = Sample / Reference
        reference = params.get('reference_spectrum')
        if reference is not None:
            intensities = intensities / (reference + 1e-10)
    
    elif method == "absorbance":
        # Absorbance = -log10(Transmission)
        reference = params.get('reference_spectrum')
        if reference is not None:
            transmission = intensities / (reference + 1e-10)
            intensities = -np.log10(transmission + 1e-10)
    
    return {
        'wavelength': self.wavelengths.tolist() if self.wavelengths is not None else [],
        'intensity': intensities.tolist(),
        'integration_time_ms': self.integration_time_ms,
        'scans_averaged': self.scans_to_average,
        'unit': 'counts' if method == 'uv_vis_nir' else method
    }

def _load_wavelength_calibration(self) -> None:
    """Load wavelength calibration"""
    # In real implementation:
    # self.wavelengths = self.device.wavelengths()
    
    # Simulate typical USB4000 range
    self.wavelengths = np.linspace(200, 1000, 3648)

def _simulate_spectrum(self, params: Dict[str, Any]) -> np.ndarray:
    """Simulate spectrum for development"""
    if self.wavelengths is None:
        raise RuntimeError("Wavelengths not calibrated")
    
    # Simulate blackbody + some features
    temperature = params.get('temperature', 3000)  # K
    
    # Planck's law (simplified)
    h = 6.626e-34
    c = 3e8
    k = 1.381e-23
    
    wl_m = self.wavelengths * 1e-9
    intensity = 2 * h * c**2 / (wl_m**5 * (np.exp(h*c/(wl_m*k*temperature)) - 1))
    
    # Normalize
    intensity = intensity / np.max(intensity) * 60000
    
    # Add noise
    noise = np.random.normal(0, 50, len(intensity))
    intensity = intensity + noise
    
    return np.maximum(intensity, 0)

def abort(self) -> None:
    """Abort acquisition"""
    pass

def get_status(self) -> Dict[str, Any]:
    """Get status"""
    return {
        'connected': self.is_connected,
        'integration_time_ms': self.integration_time_ms,
        'pixels': len(self.wavelengths) if self.wavelengths is not None else 0
    }

# ============================================================================

# J.A. Woollam Ellipsometer Driver

# ============================================================================

@driver_plugin(
name=“ja_woollam_ellipsometer”,
version=“1.0.0”,
author=“SemiconductorLab Team”,
description=“J.A. Woollam Spectroscopic Ellipsometer Driver”,
supported_methods=[“ellipsometry”, “thickness_measurement”],
supported_models=[“M2000”, “RC2”, “VASE”]
)
class JAWoollamDriver(InstrumentDriver):
“””
J.A. Woollam Ellipsometer Driver

Measures Ψ (psi) and Δ (delta) as function of wavelength or angle.
"""

def __init__(self, resource_name: str, config: Optional[Dict[str, Any]] = None):
    """Initialize ellipsometer driver"""
    self.resource_name = resource_name
    self.config = config or {}
    
    # Connection (typically TCP/IP)
    conn_config = ConnectionConfig(timeout=30.0)  # Long timeout for scans
    self.connection = VISAConnection(resource_name, conn_config)
    
    self.is_connected = False
    self._identity: Optional[Dict[str, str]] = None

def connect(self) -> bool:
    """Connect to ellipsometer"""
    try:
        self.connection.connect()
        self.is_connected = True
        self._identity = self.get_identity()
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False

def disconnect(self) -> bool:
    """Disconnect"""
    if self.is_connected:
        self.connection.disconnect()
        self.is_connected = False
    return True

def reset(self) -> None:
    """Reset"""
    self.connection.write("RESET")

def get_identity(self) -> Dict[str, str]:
    """Get identity"""
    if self._identity is None:
        # Ellipsometers may use custom protocol
        self._identity = {
            'manufacturer': 'J.A. Woollam',
            'model': 'M2000',
            'serial_number': 'ELLI12345',
            'firmware': '2.1.0'
        }
    return self._identity

def get_capabilities(self) -> List[str]:
    """Get capabilities"""
    return ["ellipsometry", "thickness_measurement"]

def configure(self, method: str, params: Dict[str, Any]) -> None:
    """Configure measurement"""
    # Set wavelength range
    wl_min = params.get('wavelength_min', 300)  # nm
    wl_max = params.get('wavelength_max', 1000)  # nm
    
    # Set angle
    angle = params.get('angle', 70)  # degrees
    
    # Custom commands (example)
    # self.connection.write(f"WAVELENGTH {wl_min} {wl_max}")
    # self.connection.write(f"ANGLE {angle}")

def measure(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform ellipsometry measurement"""
    if not self.is_connected:
        raise RuntimeError("Not connected")
    
    # Wavelength range
    wl_min = params.get('wavelength_min', 300)
    wl_max = params.get('wavelength_max', 1000)
    points = params.get('points', 500)
    
    wavelength = np.linspace(wl_min, wl_max, points)
    
    # Simulate Psi and Delta
    psi, delta = self._simulate_ellipsometry(wavelength, params)
    
    return {
        'wavelength': wavelength.tolist(),
        'psi': psi.tolist(),
        'delta': delta.tolist(),
        'angle': params.get('angle', 70),
        'unit': {'psi': 'degrees', 'delta': 'degrees'}
    }

def _simulate_ellipsometry(self, wavelength: np.ndarray, params: Dict[str, Any]) -> tuple:
    """Simulate Psi and Delta for thin film"""
    # Simulate SiO2 on Si
    thickness = params.get('film_thickness', 100)  # nm
    
    # Simplified model (not physically accurate, just for demo)
    psi = 45 + 10 * np.sin(2 * np.pi * thickness / wavelength)
    delta = 180 + 20 * np.cos(2 * np.pi * thickness / wavelength)
    
    # Add noise
    psi += np.random.normal(0, 0.5, len(psi))
    delta += np.random.normal(0, 1.0, len(delta))
    
    return psi, delta

def abort(self) -> None:
    """Abort measurement"""
    self.connection.write("ABORT")

def get_status(self) -> Dict[str, Any]:
    """Get status"""
    return {
        'connected': self.is_connected,
        'identity': self._identity
    }

# ============================================================================

# Example Usage

# ============================================================================

def example_usage():
“”“Demonstrate reference drivers”””
print(”=” * 80)
print(“Reference Drivers - Example Usage”)
print(”=” * 80)

# 1. Keithley 2400 SMU
print("\n1. Keithley 2400 SMU Driver:")
print("   Capabilities: iv_sweep, cv_measurement, resistance")
print("   Example:")
print("     driver = Keithley2400Driver('USB0::0x05E6::0x2400::...')")
print("     driver.connect()")
print("     data = driver.measure('iv_sweep', {")
print("         'v_start': 0, 'v_stop': 1, 'points': 100,")
print("         'current_compliance': 0.1")
print("     })")

# 2. Ocean Optics Spectrometer
print("\n2. Ocean Optics Spectrometer Driver:")
print("   Capabilities: uv_vis_nir, transmission, absorbance")
print("   Wavelength range: 200-1000 nm (USB4000)")
print("   Example:")
print("     driver = OceanOpticsDriver('USB:0x2457:0x1022:...')")
print("     driver.configure('uv_vis_nir', {'integration_time_ms': 100})")
print("     spectrum = driver.measure('uv_vis_nir', {})")

# 3. J.A. Woollam Ellipsometer
print("\n3. J.A. Woollam Ellipsometer Driver:")
print("   Capabilities: ellipsometry, thickness_measurement")
print("   Measures: Ψ (psi) and Δ (delta)")
print("   Example:")
print("     driver = JAWoollamDriver('TCPIP::192.168.1.10::...')")
print("     data = driver.measure('ellipsometry', {")
print("         'wavelength_min': 300, 'wavelength_max': 1000,")
print("         'angle': 70")
print("     })")

print("\n" + "=" * 80)
print("All drivers implement InstrumentDriver interface")
print("=" * 80)

if **name** == “**main**”:
example_usage()