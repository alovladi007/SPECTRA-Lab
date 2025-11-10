"""Hardware-in-the-Loop (HIL) simulators for RTP and Ion Implantation."""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import random
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Simulation fidelity modes."""
    IDEAL = "ideal"  # No noise, perfect response
    REALISTIC = "realistic"  # With noise and delays
    FAULT = "fault"  # Inject faults for testing


@dataclass
class SimulatorConfig:
    """HIL simulator configuration."""
    mode: SimulationMode = SimulationMode.REALISTIC
    update_rate_hz: float = 100.0
    noise_level: float = 0.01  # 1% noise
    response_delay_ms: float = 10.0
    enable_faults: bool = False
    random_seed: Optional[int] = None


class PhysicsModel:
    """Base class for physics simulations."""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.time_step = 1.0 / config.update_rate_hz
        self.t = 0.0
        
        if config.random_seed:
            np.random.seed(config.random_seed)
            random.seed(config.random_seed)
            
    def add_noise(self, value: float, relative: bool = True) -> float:
        """Add measurement noise to value."""
        if self.config.mode == SimulationMode.IDEAL:
            return value
            
        noise = np.random.normal(0, self.config.noise_level)
        if relative:
            return value * (1 + noise)
        else:
            return value + noise
            
    def advance_time(self):
        """Advance simulation time."""
        self.t += self.time_step


class IonBeamPhysics(PhysicsModel):
    """Physics model for ion beam implantation."""
    
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        
        # Beam parameters
        self.energy_keV = 100.0
        self.current_mA = 1.0
        self.species = "P"
        self.mass_amu = 31  # Phosphorus
        
        # Beam optics
        self.tilt_deg = 0.0
        self.twist_deg = 0.0
        self.analyzer_field_T = 0.5
        
        # Vacuum
        self.base_pressure_mTorr = 1e-6
        self.pressure_mTorr = self.base_pressure_mTorr
        
        # Dose integration
        self.total_dose_cm2 = 0.0
        self.dose_rate = 0.0
        self.implant_active = False
        
        # Beam profile (Gaussian)
        self.beam_sigma_cm = 5.0
        self.wafer_radius_cm = 15.0
        
    def calculate_dose_rate(self) -> float:
        """Calculate instantaneous dose rate."""
        if not self.implant_active:
            return 0.0
            
        # ions/s = (mA * 1e-3 A/mA) / (1.602e-19 C/ion)
        ion_flux = self.current_mA * 1e-3 / 1.602e-19
        
        # Account for beam area (cm²)
        beam_area = np.pi * (2 * self.beam_sigma_cm) ** 2
        
        # Dose rate in ions/cm²/s
        dose_rate = ion_flux / beam_area
        
        return dose_rate
        
    def calculate_projected_range(self) -> Tuple[float, float]:
        """Calculate SRIM-like projected range and straggle."""
        # Simplified LSS theory calculation
        # Rp ≈ k * E^n where k and n depend on ion/target
        
        range_coefficients = {
            'B': (0.5, 0.7),
            'P': (0.3, 0.8),
            'As': (0.2, 0.85),
            'Sb': (0.15, 0.9),
        }
        
        k, n = range_coefficients.get(self.species, (0.3, 0.8))
        
        # Projected range in nm
        Rp = k * (self.energy_keV ** n)
        
        # Straggle ≈ 0.3 * Rp (simplified)
        delta_Rp = 0.3 * Rp
        
        # Account for channeling at low tilt angles
        if abs(self.tilt_deg) < 3:
            channeling_factor = 1.5  # Deeper penetration
            Rp *= channeling_factor
            delta_Rp *= channeling_factor * 1.2
            
        return Rp, delta_Rp
        
    def calculate_beam_uniformity(self) -> np.ndarray:
        """Calculate beam uniformity across wafer."""
        # Create wafer grid
        x = np.linspace(-self.wafer_radius_cm, self.wafer_radius_cm, 21)
        y = np.linspace(-self.wafer_radius_cm, self.wafer_radius_cm, 21)
        X, Y = np.meshgrid(x, y)
        
        # Gaussian beam profile
        intensity = np.exp(-(X**2 + Y**2) / (2 * self.beam_sigma_cm**2))
        
        # Mask to wafer area
        mask = (X**2 + Y**2) <= self.wafer_radius_cm**2
        intensity *= mask
        
        # Add scan non-uniformity
        if self.config.mode == SimulationMode.REALISTIC:
            # Simulate mechanical scan imperfections
            scan_variation = 1 + 0.02 * np.sin(X/5) * np.cos(Y/5)
            intensity *= scan_variation
            
        return intensity
        
    def calculate_faraday_currents(self, num_cups: int = 9) -> List[float]:
        """Simulate Faraday cup array measurements."""
        # Cups arranged in 3x3 grid
        positions = np.linspace(-self.wafer_radius_cm * 0.8, 
                               self.wafer_radius_cm * 0.8, 
                               int(np.sqrt(num_cups)))
        
        currents = []
        for x in positions:
            for y in positions:
                # Calculate beam intensity at cup position
                r2 = x**2 + y**2
                intensity = np.exp(-r2 / (2 * self.beam_sigma_cm**2))
                
                # Convert to current
                cup_current = self.current_mA * intensity * 0.1  # 10% intercepted
                cup_current = self.add_noise(cup_current)
                currents.append(max(0, cup_current))
                
        return currents[:num_cups]
        
    def update(self):
        """Update beam physics simulation."""
        self.advance_time()
        
        if self.implant_active:
            # Integrate dose
            self.dose_rate = self.calculate_dose_rate()
            self.total_dose_cm2 += self.dose_rate * self.time_step
            
            # Update pressure (outgassing during implant)
            self.pressure_mTorr = self.base_pressure_mTorr * (1 + 0.1 * self.current_mA)
            self.pressure_mTorr = self.add_noise(self.pressure_mTorr)
            
            # Beam current fluctuations
            if self.config.mode == SimulationMode.REALISTIC:
                self.current_mA *= (1 + np.random.normal(0, 0.001))
                self.current_mA = max(0.1, min(50, self.current_mA))
                
        else:
            self.dose_rate = 0.0
            # Pressure recovery
            self.pressure_mTorr *= 0.99
            self.pressure_mTorr = max(self.base_pressure_mTorr, self.pressure_mTorr)


class ThermalPhysics(PhysicsModel):
    """Physics model for rapid thermal processing."""
    
    def __init__(self, config: SimulatorConfig):
        super().__init__(config)
        
        # Temperature state
        self.temperature_C = 25.0  # Current wafer temperature
        self.setpoint_C = 25.0
        self.ambient_C = 25.0
        
        # Lamp array (12 zones)
        self.num_zones = 12
        self.lamp_powers_pct = [0.0] * self.num_zones
        self.zone_temperatures = [25.0] * self.num_zones
        
        # Thermal properties
        self.wafer_mass_kg = 0.05  # 300mm Si wafer
        self.wafer_cp_J_kg_K = 700  # Specific heat of Si
        self.emissivity = 0.7
        self.stefan_boltzmann = 5.67e-8
        
        # Chamber conditions
        self.pressure_Torr = 760  # Atmospheric
        self.gas_flows = {'N2': 0.0, 'O2': 0.0}
        
        # PID controller state
        self.pid_error = 0.0
        self.pid_integral = 0.0
        self.pid_derivative = 0.0
        self.last_error = 0.0
        
        # Controller gains
        self.Kp = 2.0
        self.Ki = 0.5
        self.Kd = 0.1
        
    def calculate_heating_power(self) -> float:
        """Calculate total heating power from lamps."""
        # Each zone contributes to total power
        # Max power per zone: 5 kW
        max_zone_power_W = 5000
        
        total_power = 0.0
        for i, power_pct in enumerate(self.lamp_powers_pct):
            zone_power = max_zone_power_W * (power_pct / 100.0)
            
            # Account for zone-to-wafer coupling efficiency
            coupling = self._calculate_zone_coupling(i)
            total_power += zone_power * coupling
            
        # Account for lamp efficiency
        lamp_efficiency = 0.6  # 60% of electrical power becomes radiation
        return total_power * lamp_efficiency
        
    def _calculate_zone_coupling(self, zone_index: int) -> float:
        """Calculate coupling efficiency from zone to wafer center."""
        # Zones arranged in rings: center has better coupling
        if zone_index < 4:
            return 0.9  # Inner zones
        elif zone_index < 8:
            return 0.7  # Middle zones
        else:
            return 0.5  # Outer zones
            
    def calculate_heat_loss(self) -> float:
        """Calculate heat loss from wafer."""
        # Radiation loss (dominant at high temp)
        T_K = self.temperature_C + 273.15
        T_amb_K = self.ambient_C + 273.15
        
        # Stefan-Boltzmann law
        wafer_area = np.pi * (0.15 ** 2)  # 300mm diameter
        radiation_loss = (self.emissivity * self.stefan_boltzmann * 
                         wafer_area * (T_K**4 - T_amb_K**4))
        
        # Convection loss (depends on gas and pressure)
        # Simplified model
        h_conv = 10.0  # W/m²K at atmospheric pressure
        
        # Adjust for pressure
        pressure_ratio = self.pressure_Torr / 760.0
        h_conv *= pressure_ratio ** 0.5
        
        # Adjust for gas type
        gas_factors = {'N2': 1.0, 'O2': 1.1, 'He': 3.0, 'H2': 2.5}
        
        total_flow = sum(self.gas_flows.values())
        if total_flow > 0:
            weighted_factor = sum(
                flow * gas_factors.get(gas, 1.0) 
                for gas, flow in self.gas_flows.items()
            ) / total_flow
            h_conv *= weighted_factor
            
        convection_loss = h_conv * wafer_area * (self.temperature_C - self.ambient_C)
        
        return radiation_loss + convection_loss
        
    def calculate_temperature_uniformity(self) -> np.ndarray:
        """Calculate temperature distribution across wafer."""
        # Create wafer grid
        r = np.linspace(0, 0.15, 10)  # Radial positions
        theta = np.linspace(0, 2*np.pi, 36)  # Angular positions
        R, Theta = np.meshgrid(r, theta)
        
        # Base temperature
        T_field = np.ones_like(R) * self.temperature_C
        
        # Add non-uniformity from zones
        for i in range(self.num_zones):
            # Zone angular position
            zone_theta = 2 * np.pi * i / self.num_zones
            
            # Zone contribution
            zone_contribution = self.lamp_powers_pct[i] / 100.0
            
            # Angular proximity
            angular_dist = np.abs(Theta - zone_theta)
            angular_dist = np.minimum(angular_dist, 2*np.pi - angular_dist)
            
            # Radial and angular influence
            if i < 4:  # Inner zones
                radial_influence = np.exp(-(R / 0.05)**2)
            elif i < 8:  # Middle zones
                radial_influence = np.exp(-((R - 0.075) / 0.05)**2)
            else:  # Outer zones
                radial_influence = np.exp(-((R - 0.125) / 0.05)**2)
                
            angular_influence = np.exp(-(angular_dist / (np.pi/6))**2)
            
            # Add zone's temperature contribution
            T_field += 5 * zone_contribution * radial_influence * angular_influence
            
        return T_field
        
    def pid_control(self):
        """PID controller for temperature."""
        # Calculate error
        self.pid_error = self.setpoint_C - self.temperature_C
        
        # Proportional term
        P = self.Kp * self.pid_error
        
        # Integral term with anti-windup
        self.pid_integral += self.pid_error * self.time_step
        self.pid_integral = np.clip(self.pid_integral, -100, 100)
        I = self.Ki * self.pid_integral
        
        # Derivative term
        if self.last_error is not None:
            self.pid_derivative = (self.pid_error - self.last_error) / self.time_step
        else:
            self.pid_derivative = 0
        D = self.Kd * self.pid_derivative
        
        self.last_error = self.pid_error
        
        # Total control output
        output = P + I + D
        
        # Convert to lamp power percentage
        base_power = np.clip(output, 0, 100)
        
        # Distribute power among zones for uniformity
        self.lamp_powers_pct = self._distribute_power(base_power)
        
    def _distribute_power(self, base_power: float) -> List[float]:
        """Distribute power among zones for uniformity."""
        powers = [base_power] * self.num_zones
        
        if self.config.mode == SimulationMode.REALISTIC:
            # Add zone variations for realistic non-uniformity
            variations = [
                1.0, 1.02, 0.98, 1.01,  # Inner zones
                0.99, 1.03, 0.97, 1.0,   # Middle zones
                1.02, 0.98, 1.01, 0.99   # Outer zones
            ]
            powers = [p * v for p, v in zip(powers, variations)]
            
        # Clip to valid range
        powers = [np.clip(p, 0, 100) for p in powers]
        
        return powers
        
    def update(self):
        """Update thermal physics simulation."""
        self.advance_time()
        
        # PID control
        self.pid_control()
        
        # Calculate heating and cooling
        heating_power = self.calculate_heating_power()
        cooling_power = self.calculate_heat_loss()
        
        # Net power
        net_power = heating_power - cooling_power
        
        # Temperature change (dT/dt = P / (m * cp))
        thermal_mass = self.wafer_mass_kg * self.wafer_cp_J_kg_K
        dT_dt = net_power / thermal_mass
        
        # Update temperature
        self.temperature_C += dT_dt * self.time_step
        
        # Add measurement noise
        if self.config.mode == SimulationMode.REALISTIC:
            self.temperature_C = self.add_noise(self.temperature_C, relative=False)
            
        # Clamp to physical limits
        self.temperature_C = np.clip(self.temperature_C, -273, 1400)
        
        # Update zone temperatures (simplified)
        for i in range(self.num_zones):
            zone_target = self.ambient_C + self.lamp_powers_pct[i] * 10
            self.zone_temperatures[i] += (zone_target - self.zone_temperatures[i]) * 0.1


class IonImplantSimulator:
    """HIL simulator for ion implantation system."""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.physics = IonBeamPhysics(config)
        self.running = False
        self._task = None
        
        # State variables
        self.status = {
            'connected': True,
            'beam_on': False,
            'interlocks_ok': True,
            'faults': []
        }
        
        # Telemetry buffer
        self.telemetry_buffer = []
        self.max_buffer_size = 10000
        
    async def start(self):
        """Start simulator."""
        self.running = True
        self._task = asyncio.create_task(self._simulation_loop())
        logger.info("Ion implant simulator started")
        
    async def stop(self):
        """Stop simulator."""
        self.running = False
        if self._task:
            await self._task
        logger.info("Ion implant simulator stopped")
        
    async def _simulation_loop(self):
        """Main simulation loop."""
        while self.running:
            try:
                # Update physics
                self.physics.update()
                
                # Generate telemetry
                telemetry = self._generate_telemetry()
                
                # Store in buffer
                self.telemetry_buffer.append(telemetry)
                if len(self.telemetry_buffer) > self.max_buffer_size:
                    self.telemetry_buffer.pop(0)
                    
                # Check for faults
                if self.config.enable_faults:
                    self._check_faults()
                    
                # Sleep until next update
                await asyncio.sleep(self.physics.time_step)
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                
    def _generate_telemetry(self) -> Dict[str, Any]:
        """Generate telemetry data point."""
        return {
            'timestamp': datetime.now().isoformat(),
            'beam_current_mA': self.physics.add_noise(self.physics.current_mA),
            'pressure_mTorr': self.physics.pressure_mTorr,
            'accel_voltage_kV': self.physics.add_noise(self.physics.energy_keV),
            'analyzer_magnet_T': self.physics.add_noise(self.physics.analyzer_field_T),
            'steering_X': self.physics.add_noise(self.physics.tilt_deg),
            'steering_Y': self.physics.add_noise(self.physics.twist_deg),
            'dose_count_C_cm2': self.physics.total_dose_cm2,
            'dose_rate': self.physics.dose_rate,
            'faraday_currents': self.physics.calculate_faraday_currents(),
        }
        
    def _check_faults(self):
        """Check for and inject faults."""
        # Random fault injection
        if random.random() < 0.001:  # 0.1% chance per update
            fault_types = [
                'BEAM_INSTABILITY',
                'VACUUM_LEAK',
                'FARADAY_CUP_FAILURE',
                'ANALYZER_DRIFT'
            ]
            fault = random.choice(fault_types)
            
            if fault not in self.status['faults']:
                self.status['faults'].append(fault)
                logger.warning(f"Fault injected: {fault}")
                
                # Modify physics based on fault
                if fault == 'BEAM_INSTABILITY':
                    self.physics.current_mA *= 0.5
                elif fault == 'VACUUM_LEAK':
                    self.physics.base_pressure_mTorr *= 10
                    
    def set_beam_parameters(self, energy_keV: float, current_mA: float, species: str):
        """Set beam parameters."""
        self.physics.energy_keV = energy_keV
        self.physics.current_mA = current_mA
        self.physics.species = species
        
        # Update mass for species
        masses = {'B': 11, 'P': 31, 'As': 75, 'Sb': 121}
        self.physics.mass_amu = masses.get(species, 31)
        
    def set_scan_parameters(self, tilt_deg: float, twist_deg: float):
        """Set scan parameters."""
        self.physics.tilt_deg = tilt_deg
        self.physics.twist_deg = twist_deg
        
    def start_implant(self, target_dose_cm2: float):
        """Start implantation."""
        self.physics.implant_active = True
        self.physics.total_dose_cm2 = 0.0
        self.status['beam_on'] = True
        
    def stop_implant(self):
        """Stop implantation."""
        self.physics.implant_active = False
        self.status['beam_on'] = False
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        Rp, dRp = self.physics.calculate_projected_range()
        
        return {
            'status': self.status,
            'beam': {
                'energy_keV': self.physics.energy_keV,
                'current_mA': self.physics.current_mA,
                'species': self.physics.species,
                'dose_cm2': self.physics.total_dose_cm2,
                'dose_rate': self.physics.dose_rate,
            },
            'vacuum': {
                'pressure_mTorr': self.physics.pressure_mTorr,
            },
            'profile': {
                'projected_range_nm': Rp,
                'straggle_nm': dRp,
            },
            'uniformity': np.mean(self.physics.calculate_beam_uniformity()),
        }


class RTPSimulator:
    """HIL simulator for RTP system."""
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.physics = ThermalPhysics(config)
        self.running = False
        self._task = None
        
        # State variables
        self.status = {
            'connected': True,
            'recipe_active': False,
            'interlocks_ok': True,
            'faults': []
        }
        
        # Recipe execution
        self.current_recipe = None
        self.recipe_step = 0
        self.step_start_time = None
        
        # Telemetry buffer
        self.telemetry_buffer = []
        self.max_buffer_size = 10000
        
    async def start(self):
        """Start simulator."""
        self.running = True
        self._task = asyncio.create_task(self._simulation_loop())
        logger.info("RTP simulator started")
        
    async def stop(self):
        """Stop simulator."""
        self.running = False
        if self._task:
            await self._task
        logger.info("RTP simulator stopped")
        
    async def _simulation_loop(self):
        """Main simulation loop."""
        while self.running:
            try:
                # Update physics
                self.physics.update()
                
                # Execute recipe if active
                if self.status['recipe_active'] and self.current_recipe:
                    await self._execute_recipe_step()
                    
                # Generate telemetry
                telemetry = self._generate_telemetry()
                
                # Store in buffer
                self.telemetry_buffer.append(telemetry)
                if len(self.telemetry_buffer) > self.max_buffer_size:
                    self.telemetry_buffer.pop(0)
                    
                # Check for faults
                if self.config.enable_faults:
                    self._check_faults()
                    
                # Sleep until next update
                await asyncio.sleep(self.physics.time_step)
                
            except Exception as e:
                logger.error(f"Simulation error: {e}")
                
    def _generate_telemetry(self) -> Dict[str, Any]:
        """Generate telemetry data point."""
        # Simulate pyrometer reading (with emissivity correction)
        pyrometer_T = self.physics.temperature_C / self.physics.emissivity
        pyrometer_T = self.physics.add_noise(pyrometer_T)
        
        # Simulate TC array
        tc_readings = [
            self.physics.add_noise(self.physics.temperature_C + random.uniform(-2, 2))
            for _ in range(3)
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'setpoint_T_C': self.physics.setpoint_C,
            'pyrometer_T_C': pyrometer_T,
            'tc_T_C': tc_readings,
            'lamp_power_pct': self.physics.lamp_powers_pct,
            'emissivity_used': self.physics.emissivity,
            'chamber_pressure_Torr': self.physics.pressure_Torr,
            'flow_sccm': self.physics.gas_flows,
            'pid_state': {
                'P': self.physics.Kp * self.physics.pid_error,
                'I': self.physics.Ki * self.physics.pid_integral,
                'D': self.physics.Kd * self.physics.pid_derivative,
                'error': self.physics.pid_error,
                'output': np.mean(self.physics.lamp_powers_pct),
            }
        }
        
    async def _execute_recipe_step(self):
        """Execute current recipe step."""
        if not self.current_recipe or 'segments' not in self.current_recipe:
            return
            
        segments = self.current_recipe['segments']
        if self.recipe_step >= len(segments):
            # Recipe complete
            self.status['recipe_active'] = False
            self.recipe_step = 0
            return
            
        segment = segments[self.recipe_step]
        
        # Initialize step
        if self.step_start_time is None:
            self.step_start_time = datetime.now()
            
        # Set temperature setpoint
        self.physics.setpoint_C = segment.get('T_C', 25)
        
        # Check if step complete
        step_duration = timedelta(seconds=segment.get('dwell_s', 0))
        if datetime.now() - self.step_start_time >= step_duration:
            # Temperature within tolerance?
            if abs(self.physics.temperature_C - self.physics.setpoint_C) < 2:
                # Move to next step
                self.recipe_step += 1
                self.step_start_time = None
                
    def _check_faults(self):
        """Check for and inject faults."""
        # Random fault injection
        if random.random() < 0.001:  # 0.1% chance per update
            fault_types = [
                'LAMP_FAILURE',
                'PYROMETER_DRIFT',
                'COOLING_FAILURE',
                'GAS_FLOW_ERROR'
            ]
            fault = random.choice(fault_types)
            
            if fault not in self.status['faults']:
                self.status['faults'].append(fault)
                logger.warning(f"Fault injected: {fault}")
                
                # Modify physics based on fault
                if fault == 'LAMP_FAILURE':
                    # Fail one zone
                    failed_zone = random.randint(0, self.physics.num_zones - 1)
                    self.physics.lamp_powers_pct[failed_zone] = 0
                elif fault == 'PYROMETER_DRIFT':
                    # Add systematic error to pyrometer
                    self.physics.emissivity *= 1.1
                    
    def set_temperature(self, temperature_C: float):
        """Set temperature setpoint."""
        self.physics.setpoint_C = temperature_C
        
    def set_recipe(self, recipe: Dict[str, Any]):
        """Load recipe."""
        self.current_recipe = recipe
        self.recipe_step = 0
        self.step_start_time = None
        
    def start_recipe(self):
        """Start recipe execution."""
        if self.current_recipe:
            self.status['recipe_active'] = True
            
    def stop_recipe(self):
        """Stop recipe execution."""
        self.status['recipe_active'] = False
        self.physics.setpoint_C = 25  # Return to idle
        
    def set_gas_flow(self, gas: str, flow_sccm: float):
        """Set gas flow rate."""
        self.physics.gas_flows[gas] = flow_sccm
        
    def set_pressure(self, pressure_Torr: float):
        """Set chamber pressure."""
        self.physics.pressure_Torr = pressure_Torr
        
    def set_emissivity(self, emissivity: float):
        """Set pyrometer emissivity."""
        self.physics.emissivity = emissivity
        
    def get_status(self) -> Dict[str, Any]:
        """Get current status."""
        uniformity_field = self.physics.calculate_temperature_uniformity()
        uniformity_pct = 100 * (1 - np.std(uniformity_field) / np.mean(uniformity_field))
        
        return {
            'status': self.status,
            'temperature': {
                'setpoint_C': self.physics.setpoint_C,
                'actual_C': self.physics.temperature_C,
                'uniformity_pct': uniformity_pct,
            },
            'lamps': {
                'zone_powers_pct': self.physics.lamp_powers_pct,
                'average_power_pct': np.mean(self.physics.lamp_powers_pct),
            },
            'chamber': {
                'pressure_Torr': self.physics.pressure_Torr,
                'gas_flows_sccm': self.physics.gas_flows,
            },
            'recipe': {
                'active': self.status['recipe_active'],
                'step': self.recipe_step,
            }
        }
