"""
Hardware-in-Loop (HIL) CVD Simulator

Physics-based CVD process simulation for testing without real hardware.
Provides realistic deposition rate, thickness, stress, and adhesion predictions.

Features:
- Arrhenius-based deposition kinetics
- Radial thickness distribution modeling
- Film stress calculation (intrinsic + thermal mismatch)
- Adhesion prediction based on process conditions
- Synthetic telemetry with realistic noise and drift
- Fault injection for FDC/SPC testing
"""

import asyncio
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional, Dict, Any, List
from uuid import UUID
import logging

from .cvd_tool import (
    CVDTool,
    ToolState,
    ToolStatus,
    ToolCapabilities,
    CVDTelemetry,
    TelemetryType,
    ToolError,
)

logger = logging.getLogger(__name__)


@dataclass
class PhysicsConfig:
    """Physics model configuration parameters"""

    # Deposition kinetics (Arrhenius: k = A * exp(-Ea/RT))
    pre_exponential_factor: float = 1e12  # A (nm/min at reference conditions)
    activation_energy_kj_mol: float = 120.0  # Ea (kJ/mol)

    # Pressure dependence (rate ∝ P^n)
    pressure_exponent: float = 0.5

    # Flow rate dependence
    flow_rate_coefficient: float = 0.001  # nm/min per sccm

    # Thickness distribution (radial non-uniformity)
    center_edge_ratio: float = 1.1  # Center deposits faster
    radial_profile_exponent: float = 2.0  # Gaussian-like

    # Step coverage / conformality
    # conformality = bottom_thickness / top_thickness
    base_conformality: float = 0.85
    pressure_conformality_factor: float = 0.0001  # Lower pressure → better conformality

    # Film stress model
    # Intrinsic stress (material-dependent)
    intrinsic_stress_mpa: float = -250.0  # Negative = compressive
    intrinsic_stress_std_mpa: float = 50.0

    # Thermal stress: σ = E/(1-ν) * (α_film - α_substrate) * ΔT
    # For Si₃N₄ on Si wafer
    young_modulus_gpa: float = 250.0  # GPa
    poisson_ratio: float = 0.27
    film_cte_ppm_k: float = 2.8  # Coefficient of thermal expansion (ppm/K)
    substrate_cte_ppm_k: float = 2.6  # Silicon CTE
    deposition_temp_c: float = 800.0
    room_temp_c: float = 25.0

    # Stress gradient (through-thickness variation)
    stress_gradient_factor: float = 0.1  # MPa/nm

    # Adhesion model
    # adhesion_score = base_adhesion * surface_quality * stress_penalty
    base_adhesion_score: float = 85.0  # 0-100 scale
    surface_clean_quality: float = 1.0  # 0-1 (1 = perfect clean)
    stress_adhesion_penalty: float = 0.001  # Per MPa of stress magnitude

    # Optical properties (for in-situ monitoring)
    refractive_index_real: float = 2.0  # n
    refractive_index_imag: float = 0.001  # k (absorption)
    optical_bandgap_ev: float = 5.0

    # Material properties
    film_density_g_cm3: float = 3.1  # Si₃N₄
    film_hardness_gpa: float = 20.0
    film_resistivity_ohm_cm: float = 1e14

    # Measurement noise and drift
    temp_noise_std_c: float = 0.5
    temp_drift_rate_c_min: float = 0.1
    pressure_noise_std_torr: float = 0.001
    pressure_drift_rate_torr_min: float = 0.0001
    thickness_noise_std_nm: float = 0.5


@dataclass
class FaultInjectionConfig:
    """Configuration for injecting faults (for FDC testing)"""

    enabled: bool = False

    # Temperature faults
    temp_spike_probability: float = 0.0  # Per second
    temp_spike_magnitude_c: float = 50.0

    # Pressure faults
    pressure_leak_probability: float = 0.0
    pressure_leak_rate_torr_min: float = 0.01

    # Gas flow faults
    flow_blockage_probability: float = 0.0
    flow_reduction_factor: float = 0.5  # 50% reduction

    # Power faults (for PECVD)
    rf_trip_probability: float = 0.0

    # Random process upsets
    random_upset_probability: float = 0.0


class HILCVDSimulator:
    """
    Hardware-in-Loop CVD Simulator

    Implements CVDTool protocol with physics-based process simulation.
    """

    def __init__(
        self,
        tool_id: str,
        vendor: str = "HIL Simulator",
        model: str = "Virtual CVD v1.0",
        mode: str = "LPCVD",
        physics_config: Optional[PhysicsConfig] = None,
        fault_config: Optional[FaultInjectionConfig] = None,
    ):
        self.tool_id = tool_id
        self.vendor = vendor
        self.model = model
        self.mode = mode

        self.physics = physics_config or PhysicsConfig()
        self.faults = fault_config or FaultInjectionConfig()

        # Tool state
        self.state = ToolState.OFFLINE
        self.current_recipe: Optional[Any] = None
        self.current_run_id: Optional[UUID] = None
        self.run_start_time: Optional[datetime] = None
        self.elapsed_time_sec: float = 0.0

        # Process state variables
        self.chamber_temp_c: float = 25.0
        self.chamber_pressure_torr: float = 0.001  # Vacuum
        self.gas_flows_sccm: Dict[str, float] = {}
        self.rf_power_w: float = 0.0

        # Simulated film properties (accumulated during run)
        self.deposited_thickness_nm: float = 0.0
        self.film_stress_mpa: float = 0.0
        self.adhesion_score: float = 0.0

        # Noise and drift accumulators
        self.temp_drift_offset_c: float = 0.0
        self.pressure_drift_offset_torr: float = 0.0

        # Alarms
        self.active_alarms: List[str] = []
        self.active_warnings: List[str] = []

        logger.info(f"HIL CVD Simulator initialized: {tool_id} ({mode})")

    # =========================================================================
    # CVDTool Protocol Implementation
    # =========================================================================

    async def get_capabilities(self) -> ToolCapabilities:
        """Get simulator capabilities"""
        return ToolCapabilities(
            tool_id=self.tool_id,
            vendor=self.vendor,
            model=self.model,
            supported_modes=[self.mode],
            min_temp_c=400.0,
            max_temp_c=1200.0,
            min_pressure_torr=0.001,
            max_pressure_torr=1000.0,
            max_wafer_diameter_mm=200,
            max_batch_size=25,
            available_gas_lines=["SiH4", "NH3", "N2", "H2", "O2", "Ar"],
            max_flow_rate_sccm={
                "SiH4": 500.0,
                "NH3": 2000.0,
                "N2": 5000.0,
                "H2": 3000.0,
                "O2": 1000.0,
                "Ar": 5000.0,
            },
            has_rf_plasma=(self.mode in ["PECVD", "MPCVD", "RPCVD"]),
            rf_frequency_mhz=13.56 if "PECVD" in self.mode else None,
            max_rf_power_w=2000.0 if "PECVD" in self.mode else None,
            has_thickness_monitor=True,
            has_optical_monitor=True,
            has_stress_monitor=True,
            has_oes_monitor=True,
            comm_protocol="HIL_Simulator",
            serial_number=f"SIM-{self.tool_id}",
            firmware_version="1.0.0-sim",
        )

    async def connect(self) -> None:
        """Connect to simulator (always succeeds)"""
        logger.info(f"Connecting to simulator {self.tool_id}")
        await asyncio.sleep(0.1)  # Simulate connection delay
        self.state = ToolState.IDLE
        logger.info(f"Simulator {self.tool_id} connected")

    async def disconnect(self) -> None:
        """Disconnect from simulator"""
        logger.info(f"Disconnecting from simulator {self.tool_id}")
        self.state = ToolState.OFFLINE

    async def configure(self, recipe: Any) -> None:
        """Load recipe into simulator"""
        if self.state not in [ToolState.IDLE, ToolState.ERROR]:
            raise ToolError(f"Cannot configure in state {self.state}")

        logger.info(f"Configuring recipe: {recipe.recipe_name if hasattr(recipe, 'recipe_name') else 'Unknown'}")
        self.state = ToolState.CONFIGURING

        # Validate recipe compatibility
        # (In real implementation, check against capabilities)
        await asyncio.sleep(0.5)  # Simulate configuration time

        self.current_recipe = recipe
        self.state = ToolState.IDLE
        logger.info("Recipe configured successfully")

    async def start_run(self, cvd_run_id: UUID) -> None:
        """Start process execution"""
        if self.state != ToolState.IDLE:
            raise ToolError(f"Cannot start run in state {self.state}")

        if not self.current_recipe:
            raise ToolError("No recipe configured")

        logger.info(f"Starting run {cvd_run_id}")
        self.current_run_id = cvd_run_id
        self.run_start_time = datetime.utcnow()
        self.elapsed_time_sec = 0.0
        self.deposited_thickness_nm = 0.0
        self.state = ToolState.RUNNING

        # Initialize process conditions from recipe
        if hasattr(self.current_recipe, 'target_temp_c'):
            self.chamber_temp_c = self.current_recipe.target_temp_c
        if hasattr(self.current_recipe, 'target_pressure_torr'):
            self.chamber_pressure_torr = self.current_recipe.target_pressure_torr

        logger.info(f"Run {cvd_run_id} started")

    async def stop_run(self, cvd_run_id: UUID) -> None:
        """Stop process (controlled shutdown)"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active")

        logger.info(f"Stopping run {cvd_run_id}")
        self.state = ToolState.STOPPING

        # Simulate ramp-down
        await asyncio.sleep(1.0)

        self.state = ToolState.IDLE
        self.current_run_id = None
        logger.info(f"Run {cvd_run_id} stopped")

    async def pause_run(self, cvd_run_id: UUID) -> None:
        """Pause process"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active")

        logger.info(f"Pausing run {cvd_run_id}")
        self.state = ToolState.PAUSED

    async def resume_run(self, cvd_run_id: UUID) -> None:
        """Resume paused process"""
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active")

        logger.info(f"Resuming run {cvd_run_id}")
        self.state = ToolState.RUNNING

    async def abort_run(self, cvd_run_id: UUID) -> None:
        """Emergency stop"""
        logger.warning(f"ABORTING run {cvd_run_id}")
        self.state = ToolState.ERROR
        self.active_alarms.append("EMERGENCY_STOP")
        self.current_run_id = None

    async def get_status(self, cvd_run_id: Optional[UUID] = None) -> ToolStatus:
        """Get current tool status"""
        return ToolStatus(
            state=self.state,
            cvd_run_id=self.current_run_id,
            current_step=1,
            total_steps=1,
            elapsed_time_sec=self.elapsed_time_sec,
            estimated_remaining_sec=self._estimate_remaining_time(),
            chamber_temp_c=self.chamber_temp_c,
            chamber_pressure_torr=self.chamber_pressure_torr,
            active_alarms=self.active_alarms.copy(),
            active_warnings=self.active_warnings.copy(),
        )

    async def stream_telemetry(
        self,
        cvd_run_id: UUID,
        interval_sec: float = 1.0
    ) -> AsyncIterator[CVDTelemetry]:
        """
        Stream simulated telemetry with physics-based deposition model
        """
        if self.current_run_id != cvd_run_id:
            raise ToolError(f"Run {cvd_run_id} not active")

        logger.info(f"Starting telemetry stream for run {cvd_run_id} (interval={interval_sec}s)")

        while self.state == ToolState.RUNNING:
            # Update elapsed time
            self.elapsed_time_sec += interval_sec

            # === Physics Simulation ===

            # 1. Calculate instantaneous deposition rate
            deposition_rate_nm_min = self._calculate_deposition_rate(
                temp_c=self.chamber_temp_c,
                pressure_torr=self.chamber_pressure_torr,
                flow_rate_sccm=sum(self.gas_flows_sccm.values()),
                rf_power_w=self.rf_power_w,
            )

            # 2. Accumulate thickness
            thickness_increment = deposition_rate_nm_min * (interval_sec / 60.0)
            self.deposited_thickness_nm += thickness_increment

            # 3. Calculate film stress (evolves with thickness)
            self.film_stress_mpa = self._calculate_film_stress(
                thickness_nm=self.deposited_thickness_nm,
                deposition_temp_c=self.chamber_temp_c,
            )

            # 4. Calculate adhesion score
            self.adhesion_score = self._calculate_adhesion_score(
                stress_mpa=self.film_stress_mpa,
                surface_quality=self.physics.surface_clean_quality,
            )

            # === Add noise and drift ===
            temp_noisy = self._add_noise_drift(
                value=self.chamber_temp_c,
                noise_std=self.physics.temp_noise_std_c,
                drift_offset=self.temp_drift_offset_c,
            )
            self.temp_drift_offset_c += self.physics.temp_drift_rate_c_min * (interval_sec / 60.0)

            pressure_noisy = self._add_noise_drift(
                value=self.chamber_pressure_torr,
                noise_std=self.physics.pressure_noise_std_torr,
                drift_offset=self.pressure_drift_offset_torr,
            )
            self.pressure_drift_offset_torr += self.physics.pressure_drift_rate_torr_min * (interval_sec / 60.0)

            thickness_noisy = self._add_noise_drift(
                value=self.deposited_thickness_nm,
                noise_std=self.physics.thickness_noise_std_nm,
                drift_offset=0.0,
            )

            # === Fault injection ===
            if self.faults.enabled:
                temp_noisy, pressure_noisy = self._inject_faults(
                    temp_noisy, pressure_noisy, interval_sec
                )

            # === Generate telemetry ===
            telemetry = CVDTelemetry(
                cvd_run_id=cvd_run_id,
                timestamp=datetime.utcnow(),
                elapsed_time_sec=self.elapsed_time_sec,
                step_number=1,
                step_name="Deposition",
                measurements={
                    TelemetryType.TEMPERATURE: temp_noisy,
                    TelemetryType.PRESSURE: pressure_noisy,
                    TelemetryType.THICKNESS: thickness_noisy,
                    TelemetryType.DEPOSITION_RATE: deposition_rate_nm_min,
                    TelemetryType.STRESS: self.film_stress_mpa,
                },
                gas_flows_sccm=self.gas_flows_sccm.copy(),
                thickness_nm=thickness_noisy,
                deposition_rate_nm_min=deposition_rate_nm_min,
                stress_mpa=self.film_stress_mpa,
                rf_forward_power_w=self.rf_power_w if self.mode == "PECVD" else None,
            )

            yield telemetry

            # Wait for next sample
            await asyncio.sleep(interval_sec)

        logger.info(f"Telemetry stream ended for run {cvd_run_id}")

    async def get_alarms(self) -> List[Dict[str, Any]]:
        """Get active alarms"""
        return [
            {
                "code": alarm,
                "severity": "CRITICAL",
                "message": alarm,
                "timestamp": datetime.utcnow().isoformat(),
            }
            for alarm in self.active_alarms
        ]

    async def clear_alarms(self) -> None:
        """Clear alarms"""
        self.active_alarms.clear()
        self.active_warnings.clear()

    async def run_diagnostics(self) -> Dict[str, Any]:
        """Run self-diagnostics"""
        return {
            "status": "HEALTHY",
            "checks": {
                "chamber_integrity": "PASS",
                "temperature_control": "PASS",
                "pressure_control": "PASS",
                "gas_delivery": "PASS",
                "vacuum_system": "PASS",
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    # =========================================================================
    # Physics Models
    # =========================================================================

    def _calculate_deposition_rate(
        self,
        temp_c: float,
        pressure_torr: float,
        flow_rate_sccm: float,
        rf_power_w: float = 0.0,
    ) -> float:
        """
        Calculate deposition rate using Arrhenius kinetics

        rate = A * exp(-Ea/RT) * P^n * (1 + k_flow * flow) * (1 + k_power * power)

        Args:
            temp_c: Chamber temperature (°C)
            pressure_torr: Chamber pressure (Torr)
            flow_rate_sccm: Total gas flow rate (sccm)
            rf_power_w: RF power (W) for plasma-enhanced CVD

        Returns:
            Deposition rate in nm/min
        """
        # Convert temperature to Kelvin
        temp_k = temp_c + 273.15

        # Arrhenius term: k = A * exp(-Ea/RT)
        R = 8.314  # J/(mol·K)
        Ea = self.physics.activation_energy_kj_mol * 1000  # Convert to J/mol

        arrhenius_factor = self.physics.pre_exponential_factor * math.exp(-Ea / (R * temp_k))

        # Pressure dependence: rate ∝ P^n
        pressure_factor = math.pow(pressure_torr, self.physics.pressure_exponent)

        # Flow rate contribution
        flow_factor = 1.0 + self.physics.flow_rate_coefficient * flow_rate_sccm

        # RF power contribution (for PECVD)
        power_factor = 1.0
        if rf_power_w > 0:
            power_factor = 1.0 + 0.0001 * rf_power_w  # Simple linear model

        # Combined deposition rate
        rate = arrhenius_factor * pressure_factor * flow_factor * power_factor

        # Normalize to realistic values (10-500 nm/min range)
        rate = max(10.0, min(500.0, rate * 1e-9))  # Clamp to realistic range

        return rate

    def _calculate_film_stress(
        self,
        thickness_nm: float,
        deposition_temp_c: float,
    ) -> float:
        """
        Calculate film stress (MPa)

        Total stress = Intrinsic stress + Thermal stress + Gradient stress

        Args:
            thickness_nm: Film thickness (nm)
            deposition_temp_c: Deposition temperature (°C)

        Returns:
            Film stress in MPa (positive=tensile, negative=compressive)
        """
        # 1. Intrinsic stress (material-dependent, process-dependent)
        intrinsic_stress = random.gauss(
            self.physics.intrinsic_stress_mpa,
            self.physics.intrinsic_stress_std_mpa,
        )

        # 2. Thermal stress: σ_thermal = [E/(1-ν)] * Δα * ΔT
        delta_T = deposition_temp_c - self.physics.room_temp_c
        delta_alpha = (
            self.physics.film_cte_ppm_k - self.physics.substrate_cte_ppm_k
        ) * 1e-6  # Convert ppm to fractional

        E_over_1_minus_nu = (
            self.physics.young_modulus_gpa * 1e3 / (1 - self.physics.poisson_ratio)
        )  # Convert GPa to MPa

        thermal_stress = E_over_1_minus_nu * delta_alpha * delta_T

        # 3. Stress gradient (through-thickness variation)
        gradient_stress = self.physics.stress_gradient_factor * thickness_nm

        # Total stress
        total_stress = intrinsic_stress + thermal_stress + gradient_stress

        return total_stress

    def _calculate_adhesion_score(
        self,
        stress_mpa: float,
        surface_quality: float,
    ) -> float:
        """
        Calculate adhesion score (0-100)

        adhesion = base * surface_quality * stress_penalty

        High stress reduces adhesion.

        Args:
            stress_mpa: Film stress magnitude
            surface_quality: Surface cleanliness (0-1)

        Returns:
            Adhesion score (0-100)
        """
        stress_magnitude = abs(stress_mpa)
        stress_penalty = 1.0 / (
            1.0 + self.physics.stress_adhesion_penalty * stress_magnitude
        )

        adhesion = (
            self.physics.base_adhesion_score * surface_quality * stress_penalty
        )

        return max(0.0, min(100.0, adhesion))

    def _calculate_conformality(self, pressure_torr: float) -> float:
        """
        Calculate step coverage conformality

        conformality = bottom_thickness / top_thickness

        Lower pressure → better conformality (more isotropic)
        """
        conformality = self.physics.base_conformality + (
            self.physics.pressure_conformality_factor / (pressure_torr + 0.001)
        )
        return min(1.0, conformality)

    def _calculate_thickness_uniformity(self, radial_position: float) -> float:
        """
        Calculate radial thickness non-uniformity

        Args:
            radial_position: Normalized radius (0=center, 1=edge)

        Returns:
            Thickness multiplier relative to center
        """
        # Gaussian-like radial profile
        # thickness(r) = thickness_center * (center_edge_ratio - (ratio-1)*r^n)
        factor = self.physics.center_edge_ratio - (
            self.physics.center_edge_ratio - 1.0
        ) * math.pow(radial_position, self.physics.radial_profile_exponent)
        return factor

    # =========================================================================
    # Noise and Fault Injection
    # =========================================================================

    def _add_noise_drift(
        self,
        value: float,
        noise_std: float,
        drift_offset: float,
    ) -> float:
        """Add Gaussian noise and systematic drift"""
        noise = random.gauss(0.0, noise_std)
        return value + noise + drift_offset

    def _inject_faults(
        self,
        temp: float,
        pressure: float,
        interval_sec: float,
    ) -> tuple[float, float]:
        """Inject random faults for FDC testing"""

        # Temperature spike
        if random.random() < self.faults.temp_spike_probability * interval_sec:
            logger.warning("FAULT INJECTED: Temperature spike")
            temp += self.faults.temp_spike_magnitude_c
            self.active_alarms.append("TEMP_SPIKE")

        # Pressure leak
        if random.random() < self.faults.pressure_leak_probability * interval_sec:
            logger.warning("FAULT INJECTED: Pressure leak")
            pressure += self.faults.pressure_leak_rate_torr_min * (interval_sec / 60.0)
            self.active_alarms.append("PRESSURE_LEAK")

        # Random process upset
        if random.random() < self.faults.random_upset_probability * interval_sec:
            logger.warning("FAULT INJECTED: Random process upset")
            temp += random.gauss(0, 10.0)
            pressure *= random.uniform(0.9, 1.1)
            self.active_alarms.append("PROCESS_UPSET")

        return temp, pressure

    def _estimate_remaining_time(self) -> Optional[float]:
        """Estimate remaining process time"""
        if not self.current_recipe or not hasattr(self.current_recipe, 'target_thickness_nm'):
            return None

        target_thickness = self.current_recipe.target_thickness_nm
        if self.deposited_thickness_nm >= target_thickness:
            return 0.0

        # Estimate based on current deposition rate
        remaining_thickness = target_thickness - self.deposited_thickness_nm
        current_rate = self._calculate_deposition_rate(
            self.chamber_temp_c,
            self.chamber_pressure_torr,
            sum(self.gas_flows_sccm.values()),
            self.rf_power_w,
        )

        if current_rate > 0:
            remaining_min = remaining_thickness / current_rate
            return remaining_min * 60.0  # Convert to seconds

        return None
