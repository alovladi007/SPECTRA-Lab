"""Ion Implantation Hardware-in-Loop (HIL) Simulator with SRIM-like physics."""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging
import math

from app.drivers.ion_implant_driver import (
    IonImplantDriver, IonSpecies, ImplantStatus,
    SourceParameters, BeamParameters, ScanParameters, WaferParameters, DoseParameters
)

logger = logging.getLogger(__name__)


# ============================================================================
# Physical Constants and Models
# ============================================================================

# Ion masses (amu)
ION_MASSES = {
    IonSpecies.BORON: 11.0,
    IonSpecies.PHOSPHORUS: 31.0,
    IonSpecies.ARSENIC: 75.0,
    IonSpecies.ANTIMONY: 122.0,
    IonSpecies.NITROGEN: 14.0,
    IonSpecies.OXYGEN: 16.0,
    IonSpecies.ARGON: 40.0,
    IonSpecies.SILICON: 28.0,
}

# Silicon target properties
SI_DENSITY_G_CM3 = 2.33
SI_ATOMIC_MASS = 28.0855
SI_ATOMIC_NUMBER = 14


@dataclass
class IonProfile:
    """Ion implantation depth profile."""
    depth_nm: np.ndarray  # Depth points (nm)
    concentration_cm3: np.ndarray  # Dopant concentration (cm⁻³)
    projected_range_nm: float  # Mean projected range (Rp)
    range_straggle_nm: float  # Standard deviation (ΔRp)
    lateral_straggle_nm: float  # Lateral spread
    ion_species: IonSpecies
    energy_keV: float
    dose_cm2: float
    tilt_angle_deg: float


class SRIMPhysicsModel:
    """SRIM-like physics model for ion implantation."""

    def __init__(self, random_seed: Optional[int] = None):
        """Initialize physics model with deterministic seed."""
        self.rng = np.random.default_rng(seed=random_seed)

    def calculate_projected_range(
        self,
        ion_species: IonSpecies,
        energy_keV: float,
        tilt_angle_deg: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate projected range (Rp) and range straggle (ΔRp) using LSS theory.

        Based on Lindhard-Scharff-Schiott (LSS) theory for ion stopping in amorphous Si.
        """
        ion_mass = ION_MASSES.get(ion_species, 28.0)

        # Reduced mass
        M1 = ion_mass
        M2 = SI_ATOMIC_MASS
        M_reduced = (M1 * M2) / (M1 + M2)

        # Reduced energy (Lindhard units)
        a_TF = 0.8853 * 0.529 / (SI_ATOMIC_NUMBER**0.23 + SI_ATOMIC_NUMBER**0.23)  # Thomas-Fermi screening length (Å)
        epsilon = 32.53 * M2 * energy_keV / (SI_ATOMIC_NUMBER * SI_ATOMIC_NUMBER * (M1 + M2))  # Reduced energy

        # Electronic and nuclear stopping
        # Simplified model - in real SRIM this is much more complex
        k_e = 0.2  # Electronic stopping coefficient
        k_n = 0.5  # Nuclear stopping coefficient

        if epsilon < 0.1:
            s_n = k_n * 3.441 * np.sqrt(epsilon) * np.log(epsilon + 2.718)
        else:
            s_n = k_n * np.log(1 + 1.1383 * epsilon) / (2 * (epsilon + 0.01321 * epsilon**0.21226 + 0.19593 * epsilon**0.5))

        s_e = k_e * epsilon**0.45
        s_total = s_n + s_e

        # Projected range (simplified Gibbons formula)
        # Rp (nm) ≈ A × E^n where A and n depend on ion-target combination
        if ion_mass < SI_ATOMIC_MASS:  # Light ions (B, N, O)
            A = 1.2 * ion_mass / SI_ATOMIC_MASS
            n = 1.7
        elif ion_mass > SI_ATOMIC_MASS:  # Heavy ions (P, As, Sb)
            A = 0.8 * SI_ATOMIC_MASS / ion_mass
            n = 1.3
        else:  # Similar mass
            A = 1.0
            n = 1.5

        Rp_nm = A * (energy_keV ** n)

        # Range straggle (typically 20-40% of Rp)
        straggle_ratio = 0.3 if ion_mass < SI_ATOMIC_MASS else 0.25
        delta_Rp_nm = straggle_ratio * Rp_nm

        # Account for tilt angle (channeling reduction)
        tilt_rad = np.deg2rad(tilt_angle_deg)
        if tilt_angle_deg > 0:
            # Tilting reduces channeling, increases Rp slightly
            Rp_nm *= (1.0 + 0.1 * np.sin(tilt_rad))
            delta_Rp_nm *= (1.0 + 0.05 * np.sin(tilt_rad))

        return Rp_nm, delta_Rp_nm

    def calculate_lateral_straggle(
        self,
        projected_range_nm: float,
        range_straggle_nm: float
    ) -> float:
        """Calculate lateral straggle (perpendicular to beam direction)."""
        # Lateral straggle is typically 50-70% of longitudinal straggle
        return 0.6 * range_straggle_nm

    def generate_depth_profile(
        self,
        ion_species: IonSpecies,
        energy_keV: float,
        dose_cm2: float,
        tilt_angle_deg: float = 0.0,
        depth_resolution_nm: float = 1.0,
        max_depth_nm: Optional[float] = None
    ) -> IonProfile:
        """
        Generate complete ion implantation depth profile.

        Uses Pearson-IV distribution for accurate profile shape.
        """
        # Calculate range parameters
        Rp, delta_Rp = self.calculate_projected_range(ion_species, energy_keV, tilt_angle_deg)
        lateral_straggle = self.calculate_lateral_straggle(Rp, delta_Rp)

        # Set depth range
        if max_depth_nm is None:
            max_depth_nm = Rp + 5 * delta_Rp

        depth_nm = np.arange(0, max_depth_nm, depth_resolution_nm)

        # Pearson-IV profile (asymmetric Gaussian)
        # Skewness accounts for channeling tails
        skewness = -0.3 if tilt_angle_deg < 1.0 else -0.1  # Channeling creates tail
        kurtosis = 3.0  # Excess kurtosis

        # For simplicity, use modified Gaussian with exponential tail
        gaussian_part = np.exp(-0.5 * ((depth_nm - Rp) / delta_Rp) ** 2)

        # Add channeling tail for low tilt angles
        if tilt_angle_deg < 2.0:
            tail_depth = depth_nm[depth_nm > Rp]
            tail_decay = np.exp(-(tail_depth - Rp) / (3 * delta_Rp))
            channeling_tail = np.zeros_like(depth_nm)
            channeling_tail[depth_nm > Rp] = 0.1 * tail_decay  # 10% channeling fraction
            gaussian_part += channeling_tail

        # Normalize to dose
        # Convert dose (ions/cm²) to peak concentration (ions/cm³)
        peak_concentration = dose_cm2 / (delta_Rp * 1e-7 * np.sqrt(2 * np.pi))  # Convert nm to cm

        concentration_cm3 = peak_concentration * gaussian_part

        return IonProfile(
            depth_nm=depth_nm,
            concentration_cm3=concentration_cm3,
            projected_range_nm=Rp,
            range_straggle_nm=delta_Rp,
            lateral_straggle_nm=lateral_straggle,
            ion_species=ion_species,
            energy_keV=energy_keV,
            dose_cm2=dose_cm2,
            tilt_angle_deg=tilt_angle_deg
        )

    def simulate_dose_integration_noise(
        self,
        true_dose_cm2: float,
        beam_current_mA: float,
        integration_time_s: float,
        wafer_area_cm2: float
    ) -> float:
        """
        Simulate realistic dose integration noise.

        Includes:
        - Shot noise (Poisson statistics)
        - Integrator drift
        - Charge collection efficiency variations
        """
        # Calculate total charge
        charge_coulombs = beam_current_mA * 1e-3 * integration_time_s

        # Number of ions
        n_ions = charge_coulombs / 1.6e-19

        # Shot noise (Poisson): σ = sqrt(N)
        shot_noise_frac = 1.0 / np.sqrt(n_ions) if n_ions > 0 else 0.01
        shot_noise = self.rng.normal(0, shot_noise_frac * true_dose_cm2)

        # Integrator drift (typically 0.1-0.5% over minutes)
        drift_rate_pct_per_min = 0.2
        drift = (drift_rate_pct_per_min / 100) * (integration_time_s / 60) * true_dose_cm2 * self.rng.normal(0, 0.5)

        # Charge collection efficiency variation (±0.5%)
        cce_variation = self.rng.normal(0, 0.005) * true_dose_cm2

        # Total measured dose
        measured_dose = true_dose_cm2 + shot_noise + drift + cce_variation

        return max(0, measured_dose)  # Dose cannot be negative

    def simulate_scan_uniformity_map(
        self,
        wafer_diameter_mm: float,
        scan_params: ScanParameters,
        grid_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Simulate 2D dose uniformity map across wafer.

        Includes:
        - Radial non-uniformity (edge rolloff)
        - Corner bias
        - Scan pattern artifacts
        - Beam divergence effects
        """
        # Create 2D grid
        r = wafer_diameter_mm / 2
        x = np.linspace(-r, r, grid_points)
        y = np.linspace(-r, r, grid_points)
        X, Y = np.meshgrid(x, y)

        # Radial distance from center
        R = np.sqrt(X**2 + Y**2)

        # Base uniformity (start at 1.0 = nominal dose)
        uniformity = np.ones_like(R)

        # Apply wafer mask (circular)
        uniformity[R > r] = 0

        # Edge rolloff (typical 2-5% at edge)
        edge_rolloff_frac = 0.03
        rolloff_width = 0.1 * r  # 10% of radius
        edge_factor = 1.0 - edge_rolloff_frac * np.exp(-(r - R)**2 / (2 * rolloff_width**2))
        uniformity *= edge_factor

        # Scan pattern artifacts (depends on pattern type)
        if scan_params.pattern.value == "raster":
            # Raster creates slight horizontal banding
            band_amplitude = 0.01  # 1% variation
            band_frequency = 2 * np.pi * scan_params.y_frequency_Hz / scan_params.scan_speed_mm_s
            scan_artifact = 1.0 + band_amplitude * np.sin(band_frequency * Y)
            uniformity *= scan_artifact

        elif scan_params.pattern.value == "spiral":
            # Spiral creates radial variations
            spiral_amplitude = 0.008
            spiral_artifact = 1.0 + spiral_amplitude * np.sin(4 * np.arctan2(Y, X))
            uniformity *= spiral_artifact

        # Corner bias (ion optics aberrations)
        corner_bias_strength = 0.02
        angle = np.arctan2(Y, X)
        corner_factor = 1.0 - corner_bias_strength * np.cos(4 * angle) * (R / r)**2
        uniformity *= corner_factor

        # Add random noise (beam jitter, 0.5% RMS)
        noise = self.rng.normal(0, 0.005, uniformity.shape)
        uniformity *= (1.0 + noise)

        # Ensure zero outside wafer
        uniformity[R > r] = 0

        return X, Y, uniformity

    def simulate_beam_jitter(
        self,
        nominal_position: Tuple[float, float],
        dt: float = 0.01
    ) -> Tuple[float, float]:
        """
        Simulate beam position jitter and drift.

        Includes:
        - High-frequency jitter (power supply ripple)
        - Low-frequency drift (thermal effects)
        """
        x_nom, y_nom = nominal_position

        # High-frequency jitter (60 Hz ripple, ~50 μm amplitude)
        jitter_amplitude_mm = 0.05  # 50 μm
        jitter_x = self.rng.normal(0, jitter_amplitude_mm)
        jitter_y = self.rng.normal(0, jitter_amplitude_mm)

        # Low-frequency drift (thermal, ~0.1 mm/hour)
        drift_rate_mm_per_s = 0.1 / 3600  # mm/s
        drift_x = self.rng.normal(0, drift_rate_mm_per_s * dt)
        drift_y = self.rng.normal(0, drift_rate_mm_per_s * dt)

        x_actual = x_nom + jitter_x + drift_x
        y_actual = y_nom + jitter_y + drift_y

        return (x_actual, y_actual)


# ============================================================================
# HIL Driver Implementation
# ============================================================================

class IonImplantHILDriver(IonImplantDriver):
    """Hardware-in-Loop driver with physics simulation."""

    def __init__(
        self,
        equipment_id: str,
        random_seed: Optional[int] = None,
        wafer_diameter_mm: float = 300.0
    ):
        super().__init__(equipment_id)
        self.physics = SRIMPhysicsModel(random_seed=random_seed)
        self.wafer_diameter_mm = wafer_diameter_mm

        # State variables
        self._source_params: Optional[SourceParameters] = None
        self._beam_params: Optional[BeamParameters] = None
        self._scan_params: Optional[ScanParameters] = None
        self._wafer_params: Optional[WaferParameters] = None
        self._dose_params: Optional[DoseParameters] = None
        self._beam_steering = (0.0, 0.0)
        self._current_dose = 0.0
        self._implant_start_time: Optional[datetime] = None
        self._is_scanning = False
        self._run_id: Optional[str] = None

        # Simulated measurements
        self._last_update_time: Optional[datetime] = None
        self._actual_beam_current_mA = 0.0
        self._beam_profile: Optional[IonProfile] = None
        self._uniformity_map: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    async def connect(self) -> bool:
        logger.info(f"Connecting to HIL implanter {self.equipment_id}")
        self._is_connected = True
        self._last_update_time = datetime.now()
        return True

    async def disconnect(self) -> bool:
        logger.info(f"Disconnecting from HIL implanter {self.equipment_id}")
        self._is_connected = False
        self.status = ImplantStatus.IDLE
        return True

    async def initialize(self) -> bool:
        if not self._is_connected:
            raise RuntimeError("Not connected to implanter")

        logger.info("Initializing HIL implanter")
        self.status = ImplantStatus.READY
        return True

    async def shutdown(self) -> bool:
        logger.info("Shutting down HIL implanter")
        await self.source_off()
        await self.stop_scan()
        self.status = ImplantStatus.SHUTDOWN
        return True

    # Source control
    async def source_on(self, params: SourceParameters) -> bool:
        if not self._is_connected:
            raise RuntimeError("Not connected to implanter")

        logger.info(f"HIL: Turning on ion source: {params.ion_species} @ {params.extraction_voltage_kV} kV")
        self._source_params = params

        # Simulate beam current based on source parameters
        # Typical: 1-10 mA depending on species and settings
        base_current = 5.0  # mA
        species_factor = ION_MASSES[params.ion_species] / 30.0  # Heavier ions = lower current
        voltage_factor = (params.extraction_voltage_kV / 30.0) ** 0.5

        self._actual_beam_current_mA = base_current / species_factor * voltage_factor
        self._actual_beam_current_mA += self.physics.rng.normal(0, 0.1 * self._actual_beam_current_mA)  # ±10% variation

        return True

    async def source_off(self) -> bool:
        logger.info("HIL: Turning off ion source")
        self._source_params = None
        self._actual_beam_current_mA = 0.0
        return True

    async def get_source_status(self) -> Dict:
        if self._source_params is None:
            return {
                "is_on": False,
                "ion_species": None,
                "extraction_voltage_kV": 0.0,
                "arc_voltage_V": 0.0,
                "arc_current_A": 0.0,
                "beam_current_mA": 0.0
            }

        # Add some noise to readings
        arc_v_noise = self.physics.rng.normal(0, 2.0)  # ±2V
        arc_i_noise = self.physics.rng.normal(0, 0.5)  # ±0.5A

        return {
            "is_on": True,
            "ion_species": self._source_params.ion_species,
            "extraction_voltage_kV": self._source_params.extraction_voltage_kV,
            "arc_voltage_V": self._source_params.arc_voltage_V + arc_v_noise,
            "arc_current_A": self._source_params.arc_current_A + arc_i_noise,
            "gas_flow_sccm": self._source_params.gas_flow_sccm,
            "beam_current_mA": self._actual_beam_current_mA
        }

    # Beam line control
    async def set_beam_parameters(self, params: BeamParameters) -> bool:
        logger.info(f"HIL: Setting beam parameters: {params.acceleration_voltage_kV} kV")
        self._beam_params = params
        return True

    async def set_analyzer_magnet(self, field_tesla: float) -> bool:
        logger.info(f"HIL: Setting analyzer magnet: {field_tesla} T")
        if self._beam_params:
            self._beam_params.analyzer_magnet_field_T = field_tesla
        return True

    async def get_beam_status(self) -> Dict:
        if self._beam_params is None:
            return {
                "beam_on": False,
                "acceleration_voltage_kV": 0.0,
                "analyzer_field_T": 0.0,
                "beam_current_mA": 0.0
            }

        return {
            "beam_on": self._source_params is not None,
            "acceleration_voltage_kV": self._beam_params.acceleration_voltage_kV,
            "analyzer_field_T": self._beam_params.analyzer_magnet_field_T,
            "beam_current_mA": self._actual_beam_current_mA,
            "focus_voltage_kV": self._beam_params.focus_voltage_kV
        }

    # Beam steering
    async def set_beam_steering(self, x_offset_mm: float, y_offset_mm: float) -> bool:
        logger.info(f"HIL: Setting beam steering: ({x_offset_mm}, {y_offset_mm}) mm")
        self._beam_steering = (x_offset_mm, y_offset_mm)
        return True

    async def get_beam_position(self) -> Tuple[float, float]:
        # Add jitter to position
        if self._last_update_time:
            dt = (datetime.now() - self._last_update_time).total_seconds()
            jittered_pos = self.physics.simulate_beam_jitter(self._beam_steering, dt)
            self._last_update_time = datetime.now()
            return jittered_pos
        return self._beam_steering

    # Scanning
    async def set_scan_pattern(self, params: ScanParameters) -> bool:
        logger.info(f"HIL: Setting scan pattern: {params.pattern}")
        self._scan_params = params

        # Pre-calculate uniformity map
        if self._scan_params:
            self._uniformity_map = self.physics.simulate_scan_uniformity_map(
                self.wafer_diameter_mm,
                self._scan_params
            )

        return True

    async def start_scan(self) -> bool:
        if self._scan_params is None:
            raise RuntimeError("Scan parameters not set")

        logger.info("HIL: Starting beam scan")
        self._is_scanning = True
        return True

    async def stop_scan(self) -> bool:
        logger.info("HIL: Stopping beam scan")
        self._is_scanning = False
        return True

    # Wafer handling
    async def set_wafer_position(self, params: WaferParameters) -> bool:
        logger.info(f"HIL: Setting wafer position: tilt={params.tilt_angle_deg}°")
        self._wafer_params = params
        return True

    async def get_wafer_position(self) -> WaferParameters:
        if self._wafer_params is None:
            return WaferParameters(tilt_angle_deg=0.0, rotation_angle_deg=0.0, rotation_speed_rpm=0.0)
        return self._wafer_params

    # Dose control
    async def start_implant(self, params: DoseParameters) -> str:
        if not self._source_params or not self._beam_params:
            raise RuntimeError("Source and beam parameters must be set before implanting")

        logger.info(f"HIL: Starting implant: {params.target_dose_cm2:.2e} ions/cm²")

        # Calculate depth profile using physics model
        tilt_angle = self._wafer_params.tilt_angle_deg if self._wafer_params else 0.0
        self._beam_profile = self.physics.generate_depth_profile(
            ion_species=self._source_params.ion_species,
            energy_keV=self._beam_params.acceleration_voltage_kV,
            dose_cm2=params.target_dose_cm2,
            tilt_angle_deg=tilt_angle
        )

        logger.info(f"  Projected range: {self._beam_profile.projected_range_nm:.1f} nm")
        logger.info(f"  Range straggle: {self._beam_profile.range_straggle_nm:.1f} nm")

        self._dose_params = params
        self._current_dose = 0.0
        self._implant_start_time = datetime.now()
        self._run_id = f"RUN-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.status = ImplantStatus.RUNNING

        if not self._is_scanning and self._scan_params:
            await self.start_scan()

        return self._run_id

    async def pause_implant(self) -> bool:
        logger.info("HIL: Pausing implant")
        self.status = ImplantStatus.PAUSED
        return True

    async def resume_implant(self) -> bool:
        logger.info("HIL: Resuming implant")
        self.status = ImplantStatus.RUNNING
        return True

    async def stop_implant(self) -> bool:
        logger.info("HIL: Stopping implant")
        await self.stop_scan()
        self.status = ImplantStatus.READY
        self._implant_start_time = None
        self._run_id = None
        return True

    async def get_dose_integrator_reading(self) -> Dict:
        if self._dose_params is None:
            return {
                "current_dose_cm2": 0.0,
                "target_dose_cm2": 0.0,
                "percent_complete": 0.0,
                "integrated_charge_C": 0.0,
                "elapsed_time_s": 0.0,
                "run_id": None,
                "projected_range_nm": None,
                "range_straggle_nm": None
            }

        if self.status == ImplantStatus.RUNNING and self._implant_start_time:
            elapsed = (datetime.now() - self._implant_start_time).total_seconds()

            # Simulate dose accumulation with noise
            ideal_dose_rate = self._actual_beam_current_mA * 1e-3 / (1.6e-19 * self._dose_params.wafer_area_cm2)
            ideal_dose = ideal_dose_rate * elapsed

            # Add integration noise
            self._current_dose = self.physics.simulate_dose_integration_noise(
                true_dose_cm2=min(ideal_dose, self._dose_params.target_dose_cm2),
                beam_current_mA=self._actual_beam_current_mA,
                integration_time_s=elapsed,
                wafer_area_cm2=self._dose_params.wafer_area_cm2
            )

            # Check if done
            if self._current_dose >= self._dose_params.target_dose_cm2:
                await self.stop_implant()

        percent_complete = (self._current_dose / self._dose_params.target_dose_cm2) * 100 if self._dose_params.target_dose_cm2 > 0 else 0.0
        integrated_charge = self._current_dose * self._dose_params.wafer_area_cm2 * 1.6e-19
        elapsed = (datetime.now() - self._implant_start_time).total_seconds() if self._implant_start_time else 0.0

        result = {
            "current_dose_cm2": self._current_dose,
            "target_dose_cm2": self._dose_params.target_dose_cm2,
            "percent_complete": percent_complete,
            "integrated_charge_C": integrated_charge,
            "elapsed_time_s": elapsed,
            "run_id": self._run_id
        }

        # Add profile info if available
        if self._beam_profile:
            result["projected_range_nm"] = self._beam_profile.projected_range_nm
            result["range_straggle_nm"] = self._beam_profile.range_straggle_nm
            result["lateral_straggle_nm"] = self._beam_profile.lateral_straggle_nm

        return result

    # Status and diagnostics
    async def get_status(self) -> ImplantStatus:
        return self.status

    async def get_vacuum_pressure(self) -> Dict[str, float]:
        # Simulate realistic vacuum with noise
        base_pressures = {
            "source_chamber_mTorr": 1e-4,
            "analyzer_chamber_mTorr": 5e-6,
            "process_chamber_mTorr": 1e-5,
            "beamline_mTorr": 2e-6
        }

        # Add 10% noise
        return {
            key: val * (1.0 + self.physics.rng.normal(0, 0.1))
            for key, val in base_pressures.items()
        }

    async def check_interlocks(self) -> Dict[str, bool]:
        """All interlocks OK in simulation."""
        return {
            "chamber_door": True,
            "beam_shutter": True,
            "vacuum_ok": True,
            "cooling_water": True,
            "e_stop": True,
            "x_ray_level": True,
            "ground_fault": True
        }

    # Additional HIL-specific methods
    def get_depth_profile(self) -> Optional[IonProfile]:
        """Get the calculated ion depth profile."""
        return self._beam_profile

    def get_uniformity_map(self) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Get the 2D dose uniformity map."""
        return self._uniformity_map


# Export
__all__ = [
    "IonImplantHILDriver",
    "SRIMPhysicsModel",
    "IonProfile"
]
