# scripts/dev/generate_session6_test_data.py

"""
Test Data Generators for Session 6: Electrical III (DLTS, EBIC, PCD)

Generates realistic synthetic datasets for:
- DLTS (Deep Level Transient Spectroscopy) - trap signatures
- EBIC (Electron Beam Induced Current) - defect mapping
- PCD (Photoconductance Decay) - lifetime measurements

All datasets include physics-based models with realistic noise.
"""

import numpy as np
from pathlib import Path
import json
from datetime import datetime
import uuid
from typing import Dict, List, Any, Tuple
from scipy import signal, ndimage

class DLTSDataGenerator:
    """Generate DLTS test data with realistic trap signatures"""
    
    def __init__(self, material: str = 'Si', seed: int = 42):
        np.random.seed(seed)
        self.material = material
        self.k_B = 8.617333e-5  # eV/K
        
        # Define common trap signatures for Si
        self.trap_database = {
            'E1': {'energy': 0.17, 'sigma': 2e-14, 'type': 'electron', 'defect': 'V-O'},
            'E2': {'energy': 0.23, 'sigma': 1e-15, 'type': 'electron', 'defect': 'V2'},
            'E3': {'energy': 0.38, 'sigma': 1.3e-14, 'type': 'electron', 'defect': 'Fe_i'},
            'E4': {'energy': 0.43, 'sigma': 5e-15, 'type': 'electron', 'defect': 'V2(=)'},
            'H1': {'energy': 0.71, 'sigma': 7e-17, 'type': 'hole', 'defect': 'Fe_i'},
            'H2': {'energy': 0.55, 'sigma': 7e-17, 'type': 'hole', 'defect': 'Au_s'},
        }
    
    def generate_dlts_spectrum(
        self,
        trap_config: List[str] = ['E1', 'E3'],
        temperature_range: Tuple[float, float] = (77, 350),
        num_temps: int = 150,
        rate_windows: List[float] = [20, 50, 100, 200, 500],
        noise_level: float = 0.02
    ) -> Dict[str, Any]:
        """Generate complete DLTS spectrum with multiple traps"""
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_temps)
        
        # Generate capacitance transients for each temperature
        transients = []
        time_points = np.logspace(-6, -1, 200)  # 1µs to 100ms
        
        for temp in temperatures:
            transient = self._generate_transient(temp, time_points, trap_config)
            transients.append(transient.tolist())
        
        # Calculate DLTS signal for each rate window
        dlts_signals = []
        for rw in rate_windows:
            spectrum = self._calculate_dlts_signal(
                temperatures, transients, time_points, rw, trap_config
            )
            dlts_signals.append(spectrum.tolist())
        
        # Add noise
        dlts_signals = np.array(dlts_signals)
        noise = np.random.normal(0, noise_level * np.max(np.abs(dlts_signals)), dlts_signals.shape)
        dlts_signals += noise
        
        # Extract trap signatures (for validation)
        trap_signatures = []
        for trap_id in trap_config:
            trap = self.trap_database[trap_id]
            trap_signatures.append({
                'trap_id': trap_id,
                'activation_energy': trap['energy'],
                'capture_cross_section': trap['sigma'],
                'trap_type': trap['type'],
                'identified_defect': trap['defect']
            })
        
        return {
            'measurement_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'material': self.material,
            'temperatures': temperatures.tolist(),
            'time_points': time_points.tolist(),
            'transients': transients,
            'rate_windows': rate_windows,
            'dlts_signals': dlts_signals.tolist(),
            'trap_signatures': trap_signatures,
            'measurement_parameters': {
                'reverse_voltage': -5.0,
                'pulse_voltage': 0.0,
                'pulse_width': 1e-3,
                'doping_concentration': 1e15
            },
            'quality_metrics': {
                'snr': float(np.mean(dlts_signals) / np.std(noise)),
                'baseline_stability': 0.95,
                'quality_score': 92
            }
        }
    
    def _generate_transient(
        self,
        temperature: float,
        time_points: np.ndarray,
        trap_config: List[str]
    ) -> np.ndarray:
        """Generate capacitance transient at given temperature"""
        
        # Base capacitance
        C0 = 100.0  # pF
        transient = np.ones_like(time_points) * C0
        
        # Add contribution from each trap
        for trap_id in trap_config:
            trap = self.trap_database[trap_id]
            
            # Emission rate at this temperature
            en = trap['sigma'] * 3.25e15 * temperature**2 * np.exp(-trap['energy'] / (self.k_B * temperature))
            
            # Amplitude depends on trap concentration
            amplitude = 0.5 * np.exp(-((temperature - 200) / 100)**2)  # Peak around 200K
            
            # Add exponential transient
            if trap['type'] == 'electron':
                transient -= amplitude * np.exp(-en * time_points)
            else:
                transient += amplitude * np.exp(-en * time_points)
        
        return transient
    
    def _calculate_dlts_signal(
        self,
        temperatures: np.ndarray,
        transients: List[List[float]],
        time_points: np.ndarray,
        rate_window: float,
        trap_config: List[str]
    ) -> np.ndarray:
        """Calculate DLTS signal using rate window"""
        
        spectrum = np.zeros(len(temperatures))
        
        # Boxcar method: S(T) = C(t1) - C(t2)
        t1 = 1.0 / rate_window
        t2 = 2.0 / rate_window
        
        idx1 = np.argmin(np.abs(time_points - t1))
        idx2 = np.argmin(np.abs(time_points - t2))
        
        for i, transient in enumerate(transients):
            spectrum[i] = transient[idx1] - transient[idx2]
        
        # Add characteristic peaks for each trap
        for trap_id in trap_config:
            trap = self.trap_database[trap_id]
            
            # Calculate peak temperature for this rate window
            # From en = σ * v_th * Nc * exp(-Ea/kT)
            # Peak occurs when en = rate_window
            
            # Simplified: add Gaussian peak
            T_peak = trap['energy'] / (self.k_B * np.log(1e15 * trap['sigma'] * rate_window))
            
            if temperature_range[0] < T_peak < temperature_range[1]:
                peak_amplitude = 2.0 if trap['type'] == 'electron' else -2.0
                peak_width = 20.0  # K
                
                spectrum += peak_amplitude * np.exp(-((temperatures - T_peak) / peak_width)**2)
        
        return spectrum
    
    def generate_arrhenius_data(
        self,
        trap_id: str = 'E3',
        rate_windows: List[float] = [20, 50, 100, 200, 500, 1000],
        noise_level: float = 0.05
    ) -> Dict[str, Any]:
        """Generate Arrhenius plot data for trap characterization"""
        
        trap = self.trap_database[trap_id]
        
        # Calculate peak temperatures for each rate window
        peak_temps = []
        emission_rates = []
        
        for rw in rate_windows:
            # Find temperature where en = rw
            # en = σ * v_th * Nc * exp(-Ea/kT)
            # Solve for T
            
            # Simplified calculation
            T_peak = trap['energy'] / (self.k_B * np.log(1e15 * trap['sigma'] * rw))
            
            # Add some noise
            T_peak += np.random.normal(0, noise_level * T_peak)
            
            peak_temps.append(T_peak)
            emission_rates.append(rw)
        
        # Prepare Arrhenius plot data
        x_data = [1000.0 / T for T in peak_temps]  # 1000/T
        y_data = [np.log(en / T**2) for en, T in zip(emission_rates, peak_temps)]
        
        # Linear fit for extraction
        slope, intercept = np.polyfit(x_data, y_data, 1)
        
        activation_energy = -slope * self.k_B / 1000  # eV
        capture_cross_section = np.exp(intercept) / (3.25e15 * np.sqrt(300))
        
        return {
            'trap_id': trap_id,
            'rate_windows': rate_windows,
            'peak_temperatures': peak_temps,
            'arrhenius_x': x_data,  # 1000/T
            'arrhenius_y': y_data,  # ln(en/T²)
            'fitted_energy': activation_energy,
            'fitted_sigma': capture_cross_section,
            'actual_energy': trap['energy'],
            'actual_sigma': trap['sigma'],
            'r_squared': 0.998,
            'identified_defect': trap['defect']
        }


class EBICDataGenerator:
    """Generate EBIC test data with defect maps"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
    
    def generate_ebic_map(
        self,
        map_size: Tuple[int, int] = (256, 256),
        pixel_size: float = 0.5,  # micrometers
        junction_type: str = 'pn',
        num_defects: int = 5,
        diffusion_length: float = 50.0,  # micrometers
        noise_level: float = 0.05
    ) -> Dict[str, Any]:
        """Generate EBIC current map with junction and defects"""
        
        height, width = map_size
        
        # Create base EBIC signal (junction)
        if junction_type == 'pn':
            # Straight junction in middle
            junction_position = width // 2
            x_grid = np.arange(width)
            
            # EBIC signal decays exponentially from junction
            ebic_map = np.zeros((height, width))
            for i in range(width):
                distance = abs(i - junction_position) * pixel_size
                ebic_map[:, i] = np.exp(-distance / diffusion_length)
        
        elif junction_type == 'schottky':
            # Circular junction
            center = (height // 2, width // 2)
            y_grid, x_grid = np.ogrid[:height, :width]
            distance = np.sqrt((x_grid - center[1])**2 + (y_grid - center[0])**2) * pixel_size
            ebic_map = np.exp(-distance / diffusion_length)
        
        else:  # mesa
            # Rectangular junction
            ebic_map = np.zeros((height, width))
            h_start, h_end = height // 4, 3 * height // 4
            w_start, w_end = width // 4, 3 * width // 4
            ebic_map[h_start:h_end, w_start:w_end] = 1.0
            
            # Smooth edges
            ebic_map = ndimage.gaussian_filter(ebic_map, sigma=5)
        
        # Normalize
        ebic_map = ebic_map / np.max(ebic_map)
        
        # Add defects (dark spots)
        defects = []
        for _ in range(num_defects):
            # Random position
            y = np.random.randint(10, height - 10)
            x = np.random.randint(10, width - 10)
            
            # Random size and contrast
            size = np.random.uniform(2, 10)  # pixels
            contrast = np.random.uniform(-0.8, -0.3)  # reduction
            
            # Create defect
            y_grid, x_grid = np.ogrid[:height, :width]
            defect_mask = ((x_grid - x)**2 + (y_grid - y)**2) < size**2
            
            ebic_map[defect_mask] *= (1 + contrast)
            
            defects.append({
                'id': len(defects) + 1,
                'center': [float(x), float(y)],
                'area': float(np.sum(defect_mask)),
                'contrast': float(contrast),
                'type': 'recombination_center'
            })
        
        # Add noise
        noise = np.random.normal(0, noise_level, map_size)
        ebic_map += noise
        
        # Ensure positive values
        ebic_map = np.maximum(ebic_map, 0)
        
        # Scale to realistic current values (nA)
        ebic_map *= 100  # Peak current ~100 nA
        
        # Generate SEM image (simplified)
        sem_image = np.random.uniform(100, 200, map_size).astype(np.uint8)
        
        # Add some structure to SEM
        sem_image = ndimage.gaussian_filter(sem_image, sigma=2)
        
        # Calculate statistics
        statistics = {
            'mean': float(np.mean(ebic_map)),
            'std': float(np.std(ebic_map)),
            'min': float(np.min(ebic_map)),
            'max': float(np.max(ebic_map)),
            'uniformity': float(np.std(ebic_map) / np.mean(ebic_map))
        }
        
        return {
            'measurement_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'current_map': ebic_map.tolist(),
            'sem_image': sem_image.tolist(),
            'map_size': list(map_size),
            'pixel_size': pixel_size,
            'defects': defects,
            'diffusion_length': {
                'mean': diffusion_length,
                'std': diffusion_length * 0.1,
                'unit': 'micrometers'
            },
            'measurement_parameters': {
                'beam_energy': 20.0,  # keV
                'beam_current': 100.0,  # pA
                'temperature': 300.0,  # K
                'bias_voltage': 0.0  # V
            },
            'statistics': statistics,
            'quality_score': 88
        }
    
    def generate_line_profile(
        self,
        profile_length: int = 200,
        pixel_size: float = 0.5,
        diffusion_length: float = 30.0,
        profile_type: str = 'exponential',
        noise_level: float = 0.05
    ) -> Dict[str, Any]:
        """Generate EBIC line profile for diffusion length extraction"""
        
        # Distance array
        distances = np.arange(profile_length) * pixel_size
        
        if profile_type == 'exponential':
            # Simple exponential decay
            profile = 100 * np.exp(-distances / diffusion_length)
        
        elif profile_type == 'erf':
            # Error function profile (for point source)
            profile = 100 * (1 - signal.erf(distances / (2 * diffusion_length)))
        
        else:  # gaussian
            # Gaussian profile
            profile = 100 * np.exp(-(distances / diffusion_length)**2)
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.max(profile), len(profile))
        profile += noise
        
        # Ensure positive
        profile = np.maximum(profile, 0)
        
        return {
            'distances': distances.tolist(),
            'current': profile.tolist(),
            'fitted_diffusion_length': diffusion_length,
            'profile_type': profile_type,
            'fit_quality': 0.95,
            'unit': 'micrometers'
        }


class PCDDataGenerator:
    """Generate PCD test data for lifetime measurements"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.q = 1.602176634e-19
    
    def generate_pcd_transient(
        self,
        tau_bulk: float = 100e-6,  # seconds
        tau_surface: float = 50e-6,  # seconds
        injection_level: str = 'low',
        num_points: int = 500,
        noise_level: float = 0.02
    ) -> Dict[str, Any]:
        """Generate photoconductance decay transient"""
        
        # Time array
        time_max = 5 * max(tau_bulk, tau_surface)
        time = np.linspace(0, time_max, num_points)
        
        # Effective lifetime
        tau_eff = 1 / (1/tau_bulk + 1/tau_surface)
        
        # Initial carrier density based on injection level
        if injection_level == 'low':
            n0 = 1e14  # cm^-3
        elif injection_level == 'mid':
            n0 = 1e15
        else:  # high
            n0 = 1e16
        
        # Simple exponential decay
        carrier_density = n0 * np.exp(-time / tau_eff)
        
        # Add Auger recombination at high injection
        if injection_level == 'high':
            # tau_auger = 1 / (C_aug * n²)
            C_aug = 1e-30  # cm^6/s
            
            # Solve dn/dt = -n/tau_eff - C_aug*n³
            # Simplified: modify effective lifetime
            tau_eff_mod = tau_eff / (1 + C_aug * carrier_density**2 * tau_eff)
            carrier_density = n0 * np.exp(-time / tau_eff_mod)
        
        # Convert to photoconductance
        # ΔG = q * Δn * (μn + μp) * A * W
        mobility_sum = 1500  # cm²/V·s (simplified)
        area = 1.0  # cm²
        thickness = 0.03  # cm
        
        photoconductance = self.q * carrier_density * mobility_sum * area * thickness
        
        # Add noise
        noise = np.random.normal(0, noise_level * np.max(photoconductance), len(photoconductance))
        photoconductance += noise
        
        # Calculate effective lifetime at each point
        dn_dt = np.gradient(carrier_density, time)
        tau_measured = -carrier_density[1:-1] / dn_dt[1:-1]
        
        # Clean up lifetime
        tau_measured = np.clip(tau_measured, 1e-9, 1e-3)
        
        return {
            'measurement_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'time': time.tolist(),
            'photoconductance': photoconductance.tolist(),
            'carrier_density': carrier_density.tolist(),
            'effective_lifetime': [tau_eff] * len(time),
            'measured_lifetime': np.pad(tau_measured, (1, 1), mode='edge').tolist(),
            'injection_level': injection_level,
            'parameters': {
                'tau_bulk': tau_bulk,
                'tau_surface': tau_surface,
                'tau_effective': tau_eff,
                'initial_density': n0,
                'temperature': 300.0,
                'sample_thickness': thickness * 1e4,  # micrometers
                'sample_area': area
            },
            'quality_metrics': {
                'snr': 50.0,
                'dynamic_range': 3.5,
                'quality_score': 90
            }
        }
    
    def generate_injection_dependent_lifetime(
        self,
        material: str = 'Si',
        surface_quality: str = 'good',
        num_points: int = 50,
        noise_level: float = 0.03
    ) -> Dict[str, Any]:
        """Generate injection-dependent lifetime data (QSSPC-like)"""
        
        # Injection level array
        injection = np.logspace(12, 17, num_points)  # cm^-3
        
        # Base bulk lifetime (SRH)
        if material == 'Si':
            tau_srh_low = 500e-6  # Low injection lifetime
            n_crossover = 1e15  # Crossover point
        else:  # GaAs
            tau_srh_low = 10e-6
            n_crossover = 1e16
        
        # SRH lifetime model
        tau_srh = tau_srh_low / (1 + injection / n_crossover)
        
        # Auger recombination at high injection
        if material == 'Si':
            C_aug = 1.66e-30  # cm^6/s for Si
        else:
            C_aug = 7e-30  # Higher for GaAs
        
        tau_auger = 1 / (C_aug * injection**2)
        
        # Surface recombination
        if surface_quality == 'excellent':
            S_eff = 1  # cm/s
        elif surface_quality == 'good':
            S_eff = 10
        elif surface_quality == 'poor':
            S_eff = 100
        else:  # very_poor
            S_eff = 1000
        
        thickness = 0.03  # cm
        tau_surface = thickness / (2 * S_eff)
        
        # Combine all recombination mechanisms
        tau_effective = 1 / (1/tau_srh + 1/tau_auger + 1/tau_surface)
        
        # Add some injection-dependent surface recombination
        # (surface states can be injection-dependent)
        surface_factor = 1 + 0.1 * np.log10(injection / 1e14)
        tau_effective = tau_effective / surface_factor
        
        # Add noise
        noise = np.random.lognormal(0, noise_level, len(tau_effective))
        tau_effective *= noise
        
        # Separate bulk and surface (simplified)
        tau_bulk = 1 / (1/tau_srh + 1/tau_auger)
        tau_surface_array = np.full_like(tau_bulk, tau_surface)
        
        return {
            'measurement_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'injection_level': injection.tolist(),
            'effective_lifetime': tau_effective.tolist(),
            'bulk_lifetime': tau_bulk.tolist(),
            'surface_lifetime': tau_surface_array.tolist(),
            'material': material,
            'surface_quality': surface_quality,
            'srv': {
                'effective': S_eff,
                'front': S_eff,
                'back': S_eff,
                'unit': 'cm/s'
            },
            'mechanisms': {
                'srh': {
                    'tau_low': tau_srh_low,
                    'crossover': n_crossover
                },
                'auger': {
                    'coefficient': C_aug,
                    'dominant_above': 1e17
                },
                'surface': {
                    'srv': S_eff,
                    'thickness': thickness * 1e4  # micrometers
                }
            },
            'quality_score': 92
        }
    
    def generate_qsspc_data(
        self,
        generation_profile: str = 'exponential',
        num_points: int = 200
    ) -> Dict[str, Any]:
        """Generate quasi-steady-state photoconductance data"""
        
        # Time array for slow sweep
        time = np.linspace(0, 10e-3, num_points)  # 10 ms sweep
        
        # Generation rate profile
        if generation_profile == 'exponential':
            # Exponentially decaying flash
            generation = 1e21 * np.exp(-time / 2e-3)  # photons/cm³/s
        elif generation_profile == 'linear':
            # Linear ramp down
            generation = 1e21 * (1 - time / 10e-3)
        else:  # sinusoidal
            # Sinusoidal variation
            generation = 1e21 * (1 + 0.5 * np.sin(2 * np.pi * 100 * time))
        
        # Quasi-steady state: Δn ≈ G * τ
        tau = 100e-6  # 100 µs lifetime
        carrier_density = generation * tau
        
        # Photoconductance
        mobility_sum = 1500  # cm²/V·s
        area = 1.0  # cm²
        thickness = 0.03  # cm
        
        photoconductance = self.q * carrier_density * mobility_sum * area * thickness
        
        # Calculate apparent lifetime
        tau_apparent = carrier_density / generation
        
        return {
            'time': time.tolist(),
            'generation_rate': generation.tolist(),
            'photoconductance': photoconductance.tolist(),
            'carrier_density': carrier_density.tolist(),
            'lifetime': tau_apparent.tolist(),
            'mode': 'quasi-steady-state',
            'generation_profile': generation_profile
        }


def generate_all_session6_data():
    """Generate complete test dataset for Session 6"""
    
    print("\n" + "="*70)
    print("GENERATING SESSION 6 TEST DATA: DLTS, EBIC, PCD")
    print("="*70 + "\n")
    
    # Create output directory
    base_path = Path("data/test_data/electrical_advanced")
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Subdirectories
    (base_path / "dlts").mkdir(exist_ok=True)
    (base_path / "ebic").mkdir(exist_ok=True)
    (base_path / "pcd").mkdir(exist_ok=True)
    
    def save_dataset(data: Dict, filepath: Path):
        """Save dataset as JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  ✓ Generated: {filepath.name}")
    
    # DLTS datasets
    print("1. DLTS Test Data")
    print("-" * 70)
    
    dlts_gen = DLTSDataGenerator()
    
    # Single trap
    data = dlts_gen.generate_dlts_spectrum(trap_config=['E3'])
    save_dataset(data, base_path / "dlts" / "single_trap_Fe.json")
    
    # Multiple traps
    data = dlts_gen.generate_dlts_spectrum(trap_config=['E1', 'E3', 'H1'])
    save_dataset(data, base_path / "dlts" / "multi_trap.json")
    
    # Arrhenius data
    data = dlts_gen.generate_arrhenius_data(trap_id='E3')
    save_dataset(data, base_path / "dlts" / "arrhenius_Fe.json")
    
    # EBIC datasets
    print("\n2. EBIC Test Data")
    print("-" * 70)
    
    ebic_gen = EBICDataGenerator()
    
    # PN junction
    data = ebic_gen.generate_ebic_map(junction_type='pn', num_defects=5)
    save_dataset(data, base_path / "ebic" / "pn_junction_map.json")
    
    # Schottky junction
    data = ebic_gen.generate_ebic_map(junction_type='schottky', num_defects=3)
    save_dataset(data, base_path / "ebic" / "schottky_map.json")
    
    # Line profile
    data = ebic_gen.generate_line_profile(diffusion_length=45.0)
    save_dataset(data, base_path / "ebic" / "line_profile.json")
    
    # PCD datasets
    print("\n3. PCD Test Data")
    print("-" * 70)
    
    pcd_gen = PCDDataGenerator()
    
    # Low injection transient
    data = pcd_gen.generate_pcd_transient(injection_level='low')
    save_dataset(data, base_path / "pcd" / "transient_low_injection.json")
    
    # High injection transient
    data = pcd_gen.generate_pcd_transient(injection_level='high', tau_bulk=200e-6)
    save_dataset(data, base_path / "pcd" / "transient_high_injection.json")
    
    # Injection-dependent lifetime
    data = pcd_gen.generate_injection_dependent_lifetime(surface_quality='good')
    save_dataset(data, base_path / "pcd" / "injection_dependent_Si.json")
    
    # QSSPC data
    data = pcd_gen.generate_qsspc_data()
    save_dataset(data, base_path / "pcd" / "qsspc_measurement.json")
    
    print("\n" + "="*70)
    print("✓ Session 6 test data generation complete!")
    print("="*70)
    print("\nSummary:")
    print("  - DLTS: 3 datasets (spectra, Arrhenius)")
    print("  - EBIC: 3 datasets (maps, profiles)")
    print("  - PCD: 4 datasets (transient, injection-dependent, QSSPC)")
    print(f"\nTotal: 10 datasets")
    print(f"Location: {base_path}")
    print()

if __name__ == "__main__":
    generate_all_session6_data()