# services/analysis/app/methods/electrical/dlts_analysis.py

“””
Deep Level Transient Spectroscopy (DLTS) Analysis

DLTS is used to detect and characterize deep-level defects (traps) in semiconductors.
These defects can significantly affect device performance and reliability.

This module provides:

- DLTS signal processing and baseline correction
- Trap signature identification (peaks in DLTS spectrum)
- Arrhenius plot analysis for activation energy extraction
- Capture cross-section calculation
- Defect concentration determination
- Multi-trap deconvolution

## Physical Principles:

When a reverse-biased junction is pulsed to forward bias, traps capture carriers.
As the junction returns to reverse bias, traps emit carriers with a time constant:
τ = (σ_n * v_th * N_c)^-1 * exp(E_t / kT)

Where:

- σ_n: capture cross-section (cm²)
- v_th: thermal velocity (~10^7 cm/s)
- N_c: effective density of states
- E_t: trap activation energy (eV)

References:

- Lang, J. Appl. Phys. 45, 3023 (1974) - Original DLTS paper
- Schroder, “Semiconductor Material and Device Characterization” Ch. 8
- Blood & Orton, “The Electrical Characterization of Semiconductors” (1992)
  “””

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
from typing import Dict, List, Tuple, Optional, Any
import warnings

# Physical constants

Q_E = 1.602176634e-19  # Elementary charge (C)
K_B = 1.380649e-23      # Boltzmann constant (J/K)
H_PLANCK = 6.62607015e-34  # Planck constant (J·s)

class DLTSAnalysisError(Exception):
“”“Custom exception for DLTS analysis errors”””
pass

def analyze_dlts_spectrum(
temperature: np.ndarray,
dlts_signal: np.ndarray,
config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
“””
Analyze DLTS spectrum to identify trap signatures

Parameters:
-----------
temperature : array_like
    Temperature values (K) during measurement
dlts_signal : array_like
    DLTS signal (capacitance transient difference) in F or normalized units
config : dict, optional
    Analysis configuration:
    - rate_window: Rate window (Hz) used in DLTS measurement (default: 200)
    - baseline_correction: Apply baseline correction (default: True)
    - peak_threshold: Minimum peak height for trap detection (default: 0.05)
    - smooth_data: Apply smoothing filter (default: True)
    - deconvolve_overlapping: Attempt to deconvolve overlapping peaks (default: False)
    
Returns:
--------
dict : Analysis results containing:
    - trap_signatures: List of identified trap peaks with positions and amplitudes
    - baseline: Estimated baseline signal
    - corrected_signal: Baseline-corrected DLTS signal
    - peak_temperatures: Temperatures at peak maxima
    - peak_amplitudes: DLTS signal amplitudes at peaks
    - quality_metrics: Signal-to-noise ratio and other quality indicators
"""

# Default configuration
default_config = {
    'rate_window': 200.0,  # Hz
    'baseline_correction': True,
    'peak_threshold': 0.05,  # Relative to max signal
    'smooth_data': True,
    'smooth_window': 7,
    'smooth_polyorder': 3,
    'deconvolve_overlapping': False
}

if config is None:
    config = {}
cfg = {**default_config, **config}

# Input validation
temperature = np.asarray(temperature, dtype=float)
dlts_signal = np.asarray(dlts_signal, dtype=float)

if len(temperature) != len(dlts_signal):
    raise DLTSAnalysisError("Length mismatch between temperature and signal arrays")

if len(temperature) < 20:
    raise DLTSAnalysisError("Need at least 20 data points for meaningful analysis")

if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(dlts_signal)):
    raise DLTSAnalysisError("Non-finite values detected in input data")

# Ensure temperature is sorted
sort_idx = np.argsort(temperature)
temperature = temperature[sort_idx]
dlts_signal = dlts_signal[sort_idx]

results = {
    'rate_window': cfg['rate_window'],
    'warnings': []
}

# Smooth data if requested
if cfg['smooth_data'] and len(dlts_signal) >= cfg['smooth_window']:
    signal_smooth = savgol_filter(dlts_signal, cfg['smooth_window'], cfg['smooth_polyorder'])
else:
    signal_smooth = dlts_signal.copy()

# Baseline correction
if cfg['baseline_correction']:
    baseline = estimate_baseline(temperature, signal_smooth)
    signal_corrected = signal_smooth - baseline
    results['baseline'] = baseline.tolist()
else:
    signal_corrected = signal_smooth
    results['baseline'] = None

results['corrected_signal'] = signal_corrected.tolist()
results['temperature'] = temperature.tolist()

# Find peaks (trap signatures)
# Normalize signal for peak detection
signal_norm = signal_corrected / np.max(np.abs(signal_corrected)) if np.max(np.abs(signal_corrected)) > 0 else signal_corrected

peak_indices, peak_properties = find_peaks(
    signal_norm,
    height=cfg['peak_threshold'],
    prominence=cfg['peak_threshold'] * 0.5,
    width=3  # Minimum width in data points
)

if len(peak_indices) == 0:
    results['warnings'].append("No trap signatures detected above threshold")
    results['trap_signatures'] = []
else:
    trap_signatures = []
    
    for i, idx in enumerate(peak_indices):
        trap = {
            'trap_id': f"T{i+1}",
            'peak_temperature': float(temperature[idx]),
            'peak_temperature_unit': 'K',
            'dlts_amplitude': float(signal_corrected[idx]),
            'dlts_amplitude_unit': 'a.u.',
            'peak_width': float(peak_properties['widths'][i]) if 'widths' in peak_properties else None,
            'prominence': float(peak_properties['prominences'][i]) if 'prominences' in peak_properties else None
        }
        trap_signatures.append(trap)
    
    results['trap_signatures'] = trap_signatures

# Calculate