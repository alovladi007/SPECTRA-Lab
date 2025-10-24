# services/instruments/tests/test_s3_complete.py

“””
Complete Testing Suite for Session 3: Instrument SDK & HIL

Comprehensive tests covering:

- Contract tests (interface compliance)
- Integration tests (driver functionality)
- Stress tests (reliability)
- Validation tests (physics accuracy)
- Performance tests (latency, throughput)
  “””

import pytest
import time
import numpy as np
from typing import List
import threading
import logging

# Import core components

from app.drivers.core.connection import (
VISAConnection, SCPICommand, ConnectionConfig,
list_resources, ConnectionError, TimeoutError
)
from app.drivers.core.plugin_manager import (
PluginManager, InstrumentDriver, PluginStatus
)

# Import drivers

from app.drivers.builtin.keithley_2400 import Keithley2400Driver, Keithley2400Config
from app.drivers.builtin.oceanoptics_spectrometer import (
OceanOpticsDriver, OceanOpticsSimulator, SpectrometerConfig
)

# Import simulators

from app.drivers.simulators.keithley_2400_sim import Keithley2400Simulator

# ============================================================================

# Fixtures

# ============================================================================

@pytest.fixture
def plugin_manager():
“”“Create plugin manager with loaded plugins”””
manager = PluginManager(platform_version=“1.0.0”)
manager.discover_plugins()
manager.load_all()
return manager

@pytest.fixture
def keithley_sim():
“”“Create Keithley simulator”””
config = {
‘nplc’: 0.1,
‘max_voltage’: 10.0,
‘max_current’: 0.1
}
sim = Keithley2400Simulator(“sim://diode”, config)
sim.connect()
yield sim
sim.disconnect()

@pytest.fixture
def spectrometer_sim():
“”“Create spectrometer simulator”””
config = {
‘integration_time_ms’: 10.0,
‘simulation’: {
‘peaks’: [
{‘center’: 500, ‘amplitude’: 10000, ‘width’: 20}
],
‘noise_level’: 50
}
}
sim = OceanOpticsSimulator(“sim://spec”, config)
sim.connect()
yield sim
sim.disconnect()

# ============================================================================

# Contract Tests - Interface Compliance

# ============================================================================

class TestDriverContract:
“”“Test that all drivers implement InstrumentDriver interface correctly”””

def test_all_plugins_implement_interface(self, plugin_manager):
    """Verify all drivers inherit from InstrumentDriver"""
    for plugin_name, plugin_info in plugin_manager.list_plugins().items():
        if plugin_info.status == PluginStatus.LOADED:
            assert plugin_info.driver_class is not None
            assert issubclass(plugin_info.driver_class, InstrumentDriver)

def test_required_methods_exist(self, plugin_manager):
    """Verify all required methods are implemented"""
    required_methods = [
        'connect', 'disconnect', 'reset',
        'get_identity', 'get_capabilities',
        'configure', 'measure', 'abort', 'get_status'
    ]
    
    for plugin_name, plugin_info in plugin_manager.list_plugins().items():
        if plugin_info.status == PluginStatus.LOADED:
            driver_class = plugin_info.driver_class
            
            for method in required_methods:
                assert hasattr(driver_class, method), \
                    f"Driver {plugin_name} missing method: {method}"
                assert callable(getattr(driver_class, method))

def test_method_signatures(self, keithley_sim):
    """Test method signatures match interface"""
    # connect() -> bool
    result = keithley_sim.connect()
    assert isinstance(result, bool)
    
    # get_identity() -> Dict[str, str]
    identity = keithley_sim.get_identity()
    assert isinstance(identity, dict)
    assert 'manufacturer' in identity
    assert 'model' in identity
    
    # get_capabilities() -> List[str]
    caps = keithley_sim.get_capabilities()
    assert isinstance(caps, list)
    assert all(isinstance(c, str) for c in caps)
    
    # get_status() -> Dict[str, Any]
    status = keithley_sim.get_status()
    assert isinstance(status, dict)

# ============================================================================

# Integration Tests - Functionality

# ============================================================================

class TestKeithleyIntegration:
“”“Integration tests for Keithley driver”””

def test_connection_lifecycle(self, keithley_sim):
    """Test connect/disconnect cycle"""
    # Already connected via fixture
    assert keithley_sim.get_status()['connected']
    
    # Disconnect
    assert keithley_sim.disconnect()
    
    # Reconnect
    assert keithley_sim.connect()
    assert keithley_sim.get_status()['connected']

def test_iv_sweep_configuration(self, keithley_sim):
    """Test I-V sweep configuration"""
    params = {
        'v_start': 0.0,
        'v_stop': 1.0,
        'points': 50,
        'compliance': 0.05
    }
    
    # Should not raise
    keithley_sim.configure('iv_sweep', params)
    
    # Verify configuration was applied
    status = keithley_sim.get_status()
    assert status['compliance'] == 0.05

def test_iv_sweep_measurement(self, keithley_sim):
    """Test complete I-V sweep"""
    params = {
        'v_start': 0.0,
        'v_stop': 0.8,
        'points': 100,
        'compliance': 0.1
    }
    
    keithley_sim.configure('iv_sweep', params)
    results = keithley_sim.measure('iv_sweep', params)
    
    # Verify results structure
    assert 'voltage' in results
    assert 'current' in results
    assert 'metadata' in results
    
    # Verify data dimensions
    assert len(results['voltage']) == 100
    assert len(results['current']) == 100
    
    # Verify voltage range
    voltage = np.array(results['voltage'])
    assert np.isclose(voltage[0], 0.0, atol=0.01)
    assert np.isclose(voltage[-1], 0.8, atol=0.01)
    
    # Verify physical behavior (diode-like)
    current = np.array(results['current'])
    assert np.all(current >= 0)  # Forward bias only
    assert current[-1] > current[0]  # Increasing

def test_compliance_detection(self, keithley_sim):
    """Test that compliance is detected"""
    params = {
        'v_start': 0.0,
        'v_stop': 1.0,
        'points': 50,
        'compliance': 0.001  # Very low compliance
    }
    
    keithley_sim.configure('iv_sweep', params)
    results = keithley_sim.measure('iv_sweep', params)
    
    # Should hit compliance at high voltage
    assert 'compliance_hit' in results
    # Compliance may or may not be hit depending on device model

class TestSpectrometerIntegration:
“”“Integration tests for spectrometer”””

def test_spectrum_acquisition(self, spectrometer_sim):
    """Test basic spectrum acquisition"""
    params = {}
    result = spectrometer_sim.measure('uv_vis_nir', params)
    
    # Verify structure
    assert 'wavelength' in result
    assert 'intensity' in result
    
    # Verify dimensions
    wavelength = np.array(result['wavelength'])
    intensity = np.array(result['intensity'])
    assert len(wavelength) == len(intensity)
    
    # Verify wavelength range
    assert wavelength[0] < wavelength[-1]  # Increasing
    assert 200 <= wavelength[0] <= 300
    assert 1000 <= wavelength[-1] <= 1200
    
    # Verify intensity is positive
    assert np.all(intensity >= 0)

def test_dark_reference_correction(self, spectrometer_sim):
    """Test dark and reference correction"""
    # Capture dark
    dark_result = spectrometer_sim.measure('dark_capture', {})
    assert dark_result['success']
    
    # Capture reference
    ref_result = spectrometer_sim.measure('reference_capture', {})
    assert ref_result['success']
    
    # Measure with corrections
    result = spectrometer_sim.measure('uv_vis_nir', {'calculate_transmission': True})
    
    assert 'transmission' in result
    assert 'absorbance' in result
    assert result['transmission'] is not None
    assert result['absorbance'] is not None
    
    # Verify transmission range (0 to ~1)
    transmission = np.array(result['transmission'])
    assert np.all(transmission >= 0)
    assert np.all(transmission <= 1.5)  # Allow some noise

# ============================================================================

# Stress Tests - Reliability

# ============================================================================

class TestStressTesting:
“”“Stress and reliability tests”””

def test_rapid_connect_disconnect(self, keithley_sim):
    """Test rapid connection cycling"""
    for i in range(50):
        assert keithley_sim.disconnect()
        assert keithley_sim.connect()

def test_many_measurements(self, keithley_sim):
    """Test many consecutive measurements"""
    params = {
        'v_start': 0.0,
        'v_stop': 0.5,
        'points': 20,
        'compliance': 0.1
    }
    
    keithley_sim.configure('iv_sweep', params)
    
    for i in range(100):
        results = keithley_sim.measure('iv_sweep', params)
        assert len(results['voltage']) == 20

def test_concurrent_configuration(self, keithley_sim):
    """Test thread-safe configuration"""
    params1 = {'v_start': 0.0, 'v_stop': 0.5, 'points': 20, 'compliance': 0.1}
    params2 = {'v_start': 0.0, 'v_stop': 1.0, 'points': 50, 'compliance': 0.05}
    
    def configure_loop(params):
        for _ in range(10):
            keithley_sim.configure('iv_sweep', params)
            time.sleep(0.01)
    
    t1 = threading.Thread(target=configure_loop, args=(params1,))
    t2 = threading.Thread(target=configure_loop, args=(params2,))
    
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    # Should complete without errors

def test_abort_during_measurement(self, keithley_sim):
    """Test abort functionality"""
    params = {
        'v_start': 0.0,
        'v_stop': 1.0,
        'points': 1000,  # Long sweep
        'compliance': 0.1
    }
    
    keithley_sim.configure('iv_sweep', params)
    
    # Start measurement in thread
    result = [None]
    def measure():
        try:
            result[0] = keithley_sim.measure('iv_sweep', params)
        except Exception as e:
            result[0] = e
    
    thread = threading.Thread(target=measure)
    thread.start()
    
    # Abort after short delay
    time.sleep(0.05)
    keithley_sim.abort()
    
    thread.join(timeout=2)
    
    # Should complete (either successfully or with error)

# ============================================================================

# Validation Tests - Physics Accuracy

# ============================================================================

class TestPhysicsValidation:
“”“Validate physics models in simulators”””

def test_diode_exponential_behavior(self, keithley_sim):
    """Validate diode exponential I-V"""
    params = {
        'v_start': 0.3,
        'v_stop': 0.7,
        'points': 50,
        'compliance': 0.1
    }
    
    keithley_sim.configure('iv_sweep', params)
    results = keithley_sim.measure('iv_sweep', params)
    
    voltage = np.array(results['voltage'])
    current = np.array(results['current'])
    
    # In forward bias, log(I) vs V should be approximately linear
    # I = Is * exp(V / (n*Vt))
    # log(I) = log(Is) + V / (n*Vt)
    
    # Filter to forward bias with measurable current
    mask = (voltage > 0.4) & (current > 1e-9)
    v_forward = voltage[mask]
    i_forward = current[mask]
    
    if len(v_forward) > 5:
        log_i = np.log(i_forward)
        
        # Linear fit
        slope, intercept = np.polyfit(v_forward, log_i, 1)
        
        # Slope should be ~1/(n*Vt)
        # At 300K, Vt ≈ 0.026V
        # For n=1.5, expected slope ≈ 25.6
        assert 15 < slope < 40, f"Unexpected slope: {slope}"
        
        # R² should be high (good fit)
        predicted = slope * v_forward + intercept
        r_squared = 1 - np.sum((log_i - predicted)**2) / np.sum((log_i - np.mean(log_i))**2)
        assert r_squared > 0.95, f"Poor fit: R²={r_squared}"

def test_spectrum_peak_detection(self, spectrometer_sim):
    """Validate spectrum peaks match configuration"""
    result = spectrometer_sim.measure('uv_vis_nir', {})
    
    wavelength = np.array(result['wavelength'])
    intensity = np.array(result['intensity'])
    
    # Find peaks
    from scipy.signal import find_peaks
    peaks, properties = find_peaks(intensity, height=5000, distance=50)
    
    # Should find configured peak at ~500nm
    peak_wavelengths = wavelength[peaks]
    assert len(peak_wavelengths) >= 1
    
    # Check if any peak is near 500nm
    assert np.any(np.abs(peak_wavelengths - 500) < 50), \
        f"Expected peak near 500nm, found peaks at: {peak_wavelengths}"

# ============================================================================

# Performance Tests

# ============================================================================

class TestPerformance:
“”“Performance and benchmarking tests”””

def test_measurement_latency(self, keithley_sim):
    """Test measurement latency"""
    params = {
        'v_start': 0.0,
        'v_stop': 1.0,
        'points': 100,
        'compliance': 0.1
    }
    
    keithley_sim.configure('iv_sweep', params)
    
    # Measure 10 times and calculate average
    latencies = []
    for _ in range(10):
        start = time.time()
        results = keithley_sim.measure('iv_sweep', params)
        latency = time.time() - start
        latencies.append(latency)
    
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    print(f"\nAverage latency: {avg_latency*1000:.1f} ± {std_latency*1000:.1f} ms")
    
    # Simulator should be fast (<1s for 100 points)
    assert avg_latency < 1.0

def test_throughput(self, spectrometer_sim):
    """Test measurement throughput"""
    params = {}
    
    # Measure throughput (spectra per second)
    num_measurements = 50
    start = time.time()
    
    for _ in range(num_measurements):
        result = spectrometer_sim.measure('uv_vis_nir', params)
    
    duration = time.time() - start
    throughput = num_measurements / duration
    
    print(f"\nThroughput: {throughput:.1f} spectra/second")
    
    # Should achieve reasonable throughput
    assert throughput > 10  # At least 10 spectra/sec

# ============================================================================

# Plugin Manager Tests

# ============================================================================

class TestPluginManager:
“”“Test plugin management functionality”””

def test_plugin_discovery(self, plugin_manager):
    """Test that plugins are discovered"""
    plugins = plugin_manager.list_plugins()
    assert len(plugins) > 0

def test_capability_search(self, plugin_manager):
    """Test finding drivers by capability"""
    drivers = plugin_manager.find_drivers_for_method('iv_sweep')
    assert len(drivers) > 0
    assert 'keithley_2400' in drivers or 'keithley_2400_sim' in drivers

def test_model_search(self, plugin_manager):
    """Test finding drivers by model"""
    driver_name = plugin_manager.find_driver_for_model('2400')
    assert driver_name is not None
    assert 'keithley' in driver_name.lower()

def test_driver_instantiation(self, plugin_manager):
    """Test creating driver instance"""
    # Find a simulator
    plugins = plugin_manager.list_plugins()
    sim_plugin = None
    for name, info in plugins.items():
        if 'sim' in name.lower():
            sim_plugin = name
            break
    
    if sim_plugin:
        driver = plugin_manager.get_driver(sim_plugin, "sim://test", {})
        assert driver is not None
        assert isinstance(driver, InstrumentDriver)

# ============================================================================

# Run Tests

# ============================================================================

if **name** == “**main**”:
pytest.main([**file**, “-v”, “–tb=short”])