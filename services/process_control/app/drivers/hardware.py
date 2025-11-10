"""Hardware drivers for RTP and Ion Implantation systems."""

import asyncio
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import struct
import logging
from enum import Enum

# Communication protocols
import pyvisa
import serial
import socket
from opcua import Client, ua
import pymodbus.client as ModbusClient

logger = logging.getLogger(__name__)


class DriverStatus(Enum):
    """Driver connection status."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    BUSY = "busy"


@dataclass
class DriverConfig:
    """Base configuration for hardware drivers."""
    name: str
    connection_string: str
    timeout_ms: int = 5000
    retry_count: int = 3
    poll_interval_ms: int = 100


class BaseDriver(ABC):
    """Abstract base class for hardware drivers."""
    
    def __init__(self, config: DriverConfig):
        self.config = config
        self.status = DriverStatus.DISCONNECTED
        self._connection = None
        self._lock = asyncio.Lock()
        self._callbacks = []
        self.last_error: Optional[str] = None
        
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection to hardware."""
        pass
        
    @abstractmethod
    async def disconnect(self) -> bool:
        """Close connection to hardware."""
        pass
        
    @abstractmethod
    async def read_status(self) -> Dict[str, Any]:
        """Read current status from hardware."""
        pass
        
    @abstractmethod
    async def write_command(self, command: str, params: Dict[str, Any]) -> bool:
        """Send command to hardware."""
        pass
        
    async def health_check(self) -> bool:
        """Check if hardware is responsive."""
        try:
            status = await self.read_status()
            return status is not None
        except Exception as e:
            logger.error(f"Health check failed for {self.config.name}: {e}")
            return False
            
    def add_callback(self, callback):
        """Add status change callback."""
        self._callbacks.append(callback)
        
    async def _notify_callbacks(self, event: str, data: Any):
        """Notify all registered callbacks."""
        for callback in self._callbacks:
            try:
                await callback(event, data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class IonImplantDriver(BaseDriver):
    """Driver for ion implantation system."""
    
    # Command definitions
    COMMANDS = {
        'SET_ENERGY': 'BEAM:ENERGY',
        'SET_CURRENT': 'BEAM:CURRENT',
        'SET_DOSE': 'BEAM:DOSE',
        'SET_SPECIES': 'SOURCE:SPECIES',
        'SET_TILT': 'SCAN:TILT',
        'SET_TWIST': 'SCAN:TWIST',
        'START_IMPLANT': 'PROCESS:START',
        'STOP_IMPLANT': 'PROCESS:STOP',
        'READ_PRESSURE': 'VACUUM:PRESSURE?',
        'READ_CURRENT': 'BEAM:CURRENT?',
        'READ_DOSE': 'BEAM:DOSE?',
        'READ_FARADAY': 'DIAG:FARADAY?',
        'BEAM_BLANK': 'BEAM:BLANK',
        'BEAM_UNBLANK': 'BEAM:UNBLANK',
    }
    
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self.rm = pyvisa.ResourceManager()
        self._current_species = None
        self._beam_params = {}
        
    async def connect(self) -> bool:
        """Connect to ion implanter via VISA."""
        async with self._lock:
            try:
                self.status = DriverStatus.CONNECTING
                self._connection = self.rm.open_resource(
                    self.config.connection_string,
                    timeout=self.config.timeout_ms
                )
                
                # Configure connection
                if hasattr(self._connection, 'baud_rate'):
                    self._connection.baud_rate = 115200
                    self._connection.data_bits = 8
                    self._connection.stop_bits = 1
                    self._connection.parity = pyvisa.constants.Parity.none
                
                # Test connection
                self._connection.write('*IDN?')
                idn = self._connection.read()
                logger.info(f"Connected to implanter: {idn}")
                
                # Initialize system
                await self._initialize_system()
                
                self.status = DriverStatus.CONNECTED
                await self._notify_callbacks('connected', {'device': idn})
                return True
                
            except Exception as e:
                self.status = DriverStatus.ERROR
                self.last_error = str(e)
                logger.error(f"Failed to connect to implanter: {e}")
                return False
                
    async def disconnect(self) -> bool:
        """Disconnect from ion implanter."""
        async with self._lock:
            try:
                if self._connection:
                    # Safe shutdown sequence
                    await self.beam_blank()
                    await asyncio.sleep(0.5)
                    self._connection.close()
                    self._connection = None
                    
                self.status = DriverStatus.DISCONNECTED
                await self._notify_callbacks('disconnected', {})
                return True
                
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                return False
                
    async def _initialize_system(self):
        """Initialize implanter subsystems."""
        # Set default parameters
        self._connection.write('SYSTEM:PRESET')
        await asyncio.sleep(1)
        
        # Configure safety interlocks
        self._connection.write('SAFETY:ENABLE ON')
        self._connection.write('INTERLOCK:PRESSURE 1E-5')  # mTorr
        self._connection.write('INTERLOCK:CURRENT 50')  # mA max
        
        # Setup measurement systems
        self._connection.write('DIAG:FARADAY:MODE MULTI')
        self._connection.write('DIAG:PROFILE:ENABLE ON')
        
    async def read_status(self) -> Dict[str, Any]:
        """Read comprehensive status from implanter."""
        async with self._lock:
            if not self._connection:
                return {}
                
            try:
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'beam': await self._read_beam_status(),
                    'vacuum': await self._read_vacuum_status(),
                    'source': await self._read_source_status(),
                    'diagnostics': await self._read_diagnostics(),
                    'interlocks': await self._read_interlocks(),
                }
                return status
                
            except Exception as e:
                logger.error(f"Error reading status: {e}")
                return {}
                
    async def _read_beam_status(self) -> Dict[str, Any]:
        """Read beam parameters."""
        self._connection.write('BEAM:CURRENT?')
        current = float(self._connection.read())
        
        self._connection.write('BEAM:ENERGY?')
        energy = float(self._connection.read())
        
        self._connection.write('BEAM:DOSE?')
        dose = float(self._connection.read())
        
        self._connection.write('ANALYZER:FIELD?')
        analyzer_field = float(self._connection.read())
        
        return {
            'current_mA': current,
            'energy_keV': energy,
            'dose_cm2': dose,
            'analyzer_T': analyzer_field,
            'blanked': await self._is_beam_blanked(),
        }
        
    async def _read_vacuum_status(self) -> Dict[str, Any]:
        """Read vacuum system status."""
        self._connection.write('VACUUM:PRESSURE?')
        pressure = float(self._connection.read())
        
        self._connection.write('VACUUM:GAUGES?')
        gauges_raw = self._connection.read()
        gauges = [float(x) for x in gauges_raw.split(',')]
        
        return {
            'chamber_pressure_mTorr': pressure,
            'gauge_readings': gauges,
            'pumps_on': await self._are_pumps_on(),
        }
        
    async def _read_source_status(self) -> Dict[str, Any]:
        """Read ion source status."""
        self._connection.write('SOURCE:SPECIES?')
        species = self._connection.read().strip()
        
        self._connection.write('SOURCE:TEMP?')
        temp = float(self._connection.read())
        
        self._connection.write('SOURCE:GAS:FLOWS?')
        flows_raw = self._connection.read()
        flows = {}
        for flow in flows_raw.split(';'):
            if ':' in flow:
                gas, rate = flow.split(':')
                flows[gas] = float(rate)
        
        return {
            'species': species,
            'temperature_C': temp,
            'gas_flows_sccm': flows,
        }
        
    async def _read_diagnostics(self) -> Dict[str, Any]:
        """Read diagnostic measurements."""
        self._connection.write('DIAG:FARADAY?')
        faraday_raw = self._connection.read()
        faraday = [float(x) for x in faraday_raw.split(',')]
        
        self._connection.write('DIAG:PROFILE:UNIFORMITY?')
        uniformity = float(self._connection.read())
        
        return {
            'faraday_currents_mA': faraday,
            'beam_uniformity_pct': uniformity,
        }
        
    async def _read_interlocks(self) -> Dict[str, bool]:
        """Read safety interlock status."""
        self._connection.write('INTERLOCK:STATUS?')
        status_raw = self._connection.read()
        
        interlocks = {}
        for item in status_raw.split(';'):
            if ':' in item:
                name, state = item.split(':')
                interlocks[name] = state == 'OK'
                
        return interlocks
        
    async def _is_beam_blanked(self) -> bool:
        """Check if beam is blanked."""
        self._connection.write('BEAM:BLANK?')
        return self._connection.read().strip() == 'ON'
        
    async def _are_pumps_on(self) -> bool:
        """Check if vacuum pumps are running."""
        self._connection.write('VACUUM:PUMPS?')
        return self._connection.read().strip() == 'ON'
        
    async def write_command(self, command: str, params: Dict[str, Any]) -> bool:
        """Execute implanter command."""
        async with self._lock:
            if not self._connection:
                return False
                
            try:
                if command == 'SET_BEAM_PARAMS':
                    return await self.set_beam_parameters(
                        energy_keV=params.get('energy_keV'),
                        current_mA=params.get('current_mA'),
                        species=params.get('species')
                    )
                elif command == 'SET_SCAN_PARAMS':
                    return await self.set_scan_parameters(
                        tilt_deg=params.get('tilt_deg'),
                        twist_deg=params.get('twist_deg')
                    )
                elif command == 'START_IMPLANT':
                    return await self.start_implant(
                        dose_cm2=params.get('dose_cm2')
                    )
                elif command == 'STOP_IMPLANT':
                    return await self.stop_implant()
                elif command == 'BEAM_BLANK':
                    return await self.beam_blank()
                elif command == 'BEAM_UNBLANK':
                    return await self.beam_unblank()
                else:
                    logger.warning(f"Unknown command: {command}")
                    return False
                    
            except Exception as e:
                logger.error(f"Command execution failed: {e}")
                return False
                
    async def set_beam_parameters(
        self,
        energy_keV: Optional[float] = None,
        current_mA: Optional[float] = None,
        species: Optional[str] = None
    ) -> bool:
        """Configure beam parameters."""
        try:
            if energy_keV is not None:
                self._connection.write(f'{self.COMMANDS["SET_ENERGY"]} {energy_keV}')
                self._beam_params['energy'] = energy_keV
                
            if current_mA is not None:
                self._connection.write(f'{self.COMMANDS["SET_CURRENT"]} {current_mA}')
                self._beam_params['current'] = current_mA
                
            if species is not None:
                self._connection.write(f'{self.COMMANDS["SET_SPECIES"]} {species}')
                self._current_species = species
                # Recalibrate mass analyzer for new species
                await self._calibrate_mass_analyzer(species)
                
            await self._notify_callbacks('beam_params_changed', self._beam_params)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set beam parameters: {e}")
            return False
            
    async def set_scan_parameters(
        self,
        tilt_deg: Optional[float] = None,
        twist_deg: Optional[float] = None
    ) -> bool:
        """Configure beam scanning parameters."""
        try:
            if tilt_deg is not None:
                self._connection.write(f'{self.COMMANDS["SET_TILT"]} {tilt_deg}')
                
            if twist_deg is not None:
                self._connection.write(f'{self.COMMANDS["SET_TWIST"]} {twist_deg}')
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to set scan parameters: {e}")
            return False
            
    async def start_implant(self, dose_cm2: float) -> bool:
        """Start implantation process."""
        try:
            # Set target dose
            self._connection.write(f'{self.COMMANDS["SET_DOSE"]} {dose_cm2}')
            
            # Check interlocks
            interlocks = await self._read_interlocks()
            if not all(interlocks.values()):
                logger.error(f"Interlocks not satisfied: {interlocks}")
                return False
                
            # Start implant
            self._connection.write(self.COMMANDS['START_IMPLANT'])
            self.status = DriverStatus.BUSY
            
            await self._notify_callbacks('implant_started', {'dose_target': dose_cm2})
            return True
            
        except Exception as e:
            logger.error(f"Failed to start implant: {e}")
            return False
            
    async def stop_implant(self) -> bool:
        """Stop implantation process."""
        try:
            self._connection.write(self.COMMANDS['STOP_IMPLANT'])
            await self.beam_blank()
            self.status = DriverStatus.CONNECTED
            
            await self._notify_callbacks('implant_stopped', {})
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop implant: {e}")
            return False
            
    async def beam_blank(self) -> bool:
        """Blank the beam (safety)."""
        try:
            self._connection.write(self.COMMANDS['BEAM_BLANK'])
            return True
        except Exception as e:
            logger.error(f"Failed to blank beam: {e}")
            return False
            
    async def beam_unblank(self) -> bool:
        """Unblank the beam."""
        try:
            self._connection.write(self.COMMANDS['BEAM_UNBLANK'])
            return True
        except Exception as e:
            logger.error(f"Failed to unblank beam: {e}")
            return False
            
    async def _calibrate_mass_analyzer(self, species: str) -> bool:
        """Calibrate mass analyzer for ion species."""
        # Mass-to-charge ratios for common ions
        mass_charge = {
            'B': 11, 'P': 31, 'As': 75, 'Sb': 121,
            'Ge': 74, 'Si': 28, 'N': 14, 'C': 12,
            'F': 19, 'H': 1, 'He': 4, 'Ar': 40
        }
        
        if species not in mass_charge:
            logger.warning(f"Unknown species for calibration: {species}")
            return False
            
        try:
            m_q = mass_charge[species]
            # Calculate required magnetic field
            field = 0.144 * np.sqrt(m_q * self._beam_params.get('energy', 100))
            self._connection.write(f'ANALYZER:FIELD {field}')
            await asyncio.sleep(0.5)  # Allow field to stabilize
            return True
            
        except Exception as e:
            logger.error(f"Mass analyzer calibration failed: {e}")
            return False


class RTPDriver(BaseDriver):
    """Driver for Rapid Thermal Processing system."""
    
    # OPC UA node IDs
    NODES = {
        'temperature_setpoint': 'ns=2;s=RTP.Temperature.Setpoint',
        'temperature_actual': 'ns=2;s=RTP.Temperature.Actual',
        'pyrometer': 'ns=2;s=RTP.Pyrometer.Reading',
        'lamp_power': 'ns=2;s=RTP.Lamps.Power',
        'pressure': 'ns=2;s=RTP.Chamber.Pressure',
        'gas_flows': 'ns=2;s=RTP.MFC',
        'emissivity': 'ns=2;s=RTP.Emissivity',
        'recipe_active': 'ns=2;s=RTP.Recipe.Active',
        'recipe_step': 'ns=2;s=RTP.Recipe.Step',
    }
    
    def __init__(self, config: DriverConfig):
        super().__init__(config)
        self._client = None
        self._recipe = None
        self._zone_count = 12  # Number of heating zones
        
    async def connect(self) -> bool:
        """Connect to RTP system via OPC UA."""
        async with self._lock:
            try:
                self.status = DriverStatus.CONNECTING
                
                # Create OPC UA client
                self._client = Client(self.config.connection_string)
                self._client.set_security_string("Basic256Sha256,SignAndEncrypt,certificate.pem,key.pem")
                
                # Connect
                await asyncio.get_event_loop().run_in_executor(
                    None, self._client.connect
                )
                
                # Get root node
                root = self._client.get_root_node()
                logger.info(f"Connected to RTP: {root}")
                
                # Initialize system
                await self._initialize_rtp()
                
                self.status = DriverStatus.CONNECTED
                await self._notify_callbacks('connected', {'system': 'RTP'})
                return True
                
            except Exception as e:
                self.status = DriverStatus.ERROR
                self.last_error = str(e)
                logger.error(f"Failed to connect to RTP: {e}")
                return False
                
    async def disconnect(self) -> bool:
        """Disconnect from RTP system."""
        async with self._lock:
            try:
                if self._client:
                    # Safe shutdown
                    await self.set_temperature(25)  # Room temp
                    await self.stop_recipe()
                    
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._client.disconnect
                    )
                    self._client = None
                    
                self.status = DriverStatus.DISCONNECTED
                await self._notify_callbacks('disconnected', {})
                return True
                
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
                return False
                
    async def _initialize_rtp(self):
        """Initialize RTP subsystems."""
        try:
            # Set safe defaults
            setpoint_node = self._client.get_node(self.NODES['temperature_setpoint'])
            setpoint_node.set_value(25.0)
            
            # Configure pyrometer
            emissivity_node = self._client.get_node(self.NODES['emissivity'])
            emissivity_node.set_value(0.7)  # Default Si emissivity
            
            # Initialize all lamp zones to 0%
            lamp_node = self._client.get_node(self.NODES['lamp_power'])
            lamp_node.set_value([0.0] * self._zone_count)
            
        except Exception as e:
            logger.error(f"RTP initialization failed: {e}")
            
    async def read_status(self) -> Dict[str, Any]:
        """Read comprehensive RTP status."""
        async with self._lock:
            if not self._client:
                return {}
                
            try:
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'temperature': await self._read_temperature(),
                    'lamps': await self._read_lamp_status(),
                    'chamber': await self._read_chamber_status(),
                    'recipe': await self._read_recipe_status(),
                    'safety': await self._read_safety_status(),
                }
                return status
                
            except Exception as e:
                logger.error(f"Error reading RTP status: {e}")
                return {}
                
    async def _read_temperature(self) -> Dict[str, Any]:
        """Read temperature measurements."""
        setpoint_node = self._client.get_node(self.NODES['temperature_setpoint'])
        actual_node = self._client.get_node(self.NODES['temperature_actual'])
        pyrometer_node = self._client.get_node(self.NODES['pyrometer'])
        
        return {
            'setpoint_C': setpoint_node.get_value(),
            'actual_C': actual_node.get_value(),
            'pyrometer_C': pyrometer_node.get_value(),
            'uniformity': await self._calculate_uniformity(),
        }
        
    async def _read_lamp_status(self) -> Dict[str, Any]:
        """Read lamp array status."""
        lamp_node = self._client.get_node(self.NODES['lamp_power'])
        powers = lamp_node.get_value()
        
        return {
            'zone_powers_pct': powers,
            'average_power_pct': np.mean(powers),
            'max_power_pct': np.max(powers),
            'zones_active': sum(1 for p in powers if p > 0),
        }
        
    async def _read_chamber_status(self) -> Dict[str, Any]:
        """Read chamber conditions."""
        pressure_node = self._client.get_node(self.NODES['pressure'])
        flows_node = self._client.get_node(self.NODES['gas_flows'])
        
        flows_dict = {}
        flows_raw = flows_node.get_value()
        if isinstance(flows_raw, dict):
            flows_dict = flows_raw
            
        return {
            'pressure_Torr': pressure_node.get_value(),
            'gas_flows_sccm': flows_dict,
        }
        
    async def _read_recipe_status(self) -> Dict[str, Any]:
        """Read recipe execution status."""
        active_node = self._client.get_node(self.NODES['recipe_active'])
        step_node = self._client.get_node(self.NODES['recipe_step'])
        
        return {
            'active': active_node.get_value(),
            'current_step': step_node.get_value(),
            'recipe_name': self._recipe.get('name') if self._recipe else None,
        }
        
    async def _read_safety_status(self) -> Dict[str, bool]:
        """Read safety interlock status."""
        # Read various safety interlocks
        safety = {
            'over_temp': False,  # Would read actual interlock
            'chamber_door': True,
            'cooling_water': True,
            'exhaust_flow': True,
        }
        return safety
        
    async def _calculate_uniformity(self) -> float:
        """Calculate temperature uniformity across wafer."""
        # In real system, would read multiple TC or pyrometer points
        # For now, return simulated uniformity
        return 98.5  # %
        
    async def write_command(self, command: str, params: Dict[str, Any]) -> bool:
        """Execute RTP command."""
        async with self._lock:
            if not self._client:
                return False
                
            try:
                if command == 'SET_TEMPERATURE':
                    return await self.set_temperature(params.get('temperature_C'))
                elif command == 'SET_RECIPE':
                    return await self.set_recipe(params.get('recipe'))
                elif command == 'START_RECIPE':
                    return await self.start_recipe()
                elif command == 'STOP_RECIPE':
                    return await self.stop_recipe()
                elif command == 'SET_GAS_FLOW':
                    return await self.set_gas_flow(
                        params.get('gas'),
                        params.get('flow_sccm')
                    )
                elif command == 'SET_PRESSURE':
                    return await self.set_pressure(params.get('pressure_Torr'))
                elif command == 'SET_EMISSIVITY':
                    return await self.set_emissivity(params.get('emissivity'))
                else:
                    logger.warning(f"Unknown RTP command: {command}")
                    return False
                    
            except Exception as e:
                logger.error(f"RTP command execution failed: {e}")
                return False
                
    async def set_temperature(self, temperature_C: float) -> bool:
        """Set target temperature."""
        try:
            setpoint_node = self._client.get_node(self.NODES['temperature_setpoint'])
            setpoint_node.set_value(float(temperature_C))
            
            # Calculate lamp powers for uniform heating
            powers = await self._calculate_lamp_powers(temperature_C)
            lamp_node = self._client.get_node(self.NODES['lamp_power'])
            lamp_node.set_value(powers)
            
            await self._notify_callbacks('temperature_set', {'setpoint': temperature_C})
            return True
            
        except Exception as e:
            logger.error(f"Failed to set temperature: {e}")
            return False
            
    async def _calculate_lamp_powers(self, target_C: float) -> List[float]:
        """Calculate individual lamp powers for target temperature."""
        # Simplified model - in reality would use complex thermal model
        base_power = min(100, max(0, (target_C - 25) / 10))
        
        # Add zone corrections for uniformity
        zone_corrections = [
            1.0, 1.02, 0.98, 1.01, 0.99, 1.03,
            0.97, 1.0, 1.02, 0.98, 1.01, 0.99
        ]
        
        powers = [base_power * corr for corr in zone_corrections]
        return powers
        
    async def set_recipe(self, recipe: Dict[str, Any]) -> bool:
        """Load temperature recipe."""
        try:
            self._recipe = recipe
            await self._notify_callbacks('recipe_loaded', {'recipe': recipe['name']})
            return True
            
        except Exception as e:
            logger.error(f"Failed to set recipe: {e}")
            return False
            
    async def start_recipe(self) -> bool:
        """Start recipe execution."""
        try:
            if not self._recipe:
                logger.error("No recipe loaded")
                return False
                
            active_node = self._client.get_node(self.NODES['recipe_active'])
            active_node.set_value(True)
            
            self.status = DriverStatus.BUSY
            
            # Start recipe executor task
            asyncio.create_task(self._execute_recipe())
            
            await self._notify_callbacks('recipe_started', {'recipe': self._recipe['name']})
            return True
            
        except Exception as e:
            logger.error(f"Failed to start recipe: {e}")
            return False
            
    async def stop_recipe(self) -> bool:
        """Stop recipe execution."""
        try:
            active_node = self._client.get_node(self.NODES['recipe_active'])
            active_node.set_value(False)
            
            # Ramp down to idle temperature
            await self.set_temperature(25)
            
            self.status = DriverStatus.CONNECTED
            await self._notify_callbacks('recipe_stopped', {})
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop recipe: {e}")
            return False
            
    async def _execute_recipe(self):
        """Execute temperature recipe segments."""
        if not self._recipe:
            return
            
        try:
            step_node = self._client.get_node(self.NODES['recipe_step'])
            
            for i, segment in enumerate(self._recipe.get('segments', [])):
                # Check if recipe still active
                active_node = self._client.get_node(self.NODES['recipe_active'])
                if not active_node.get_value():
                    break
                    
                step_node.set_value(i)
                
                # Execute segment
                target_temp = segment.get('temperature_C')
                ramp_rate = segment.get('ramp_Cps', 50)
                dwell_time = segment.get('dwell_s', 0)
                
                # Ramp to temperature
                await self._ramp_temperature(target_temp, ramp_rate)
                
                # Dwell at temperature
                if dwell_time > 0:
                    await asyncio.sleep(dwell_time)
                    
            # Recipe complete
            active_node.set_value(False)
            self.status = DriverStatus.CONNECTED
            await self._notify_callbacks('recipe_complete', {})
            
        except Exception as e:
            logger.error(f"Recipe execution error: {e}")
            await self.stop_recipe()
            
    async def _ramp_temperature(self, target_C: float, rate_Cps: float):
        """Ramp temperature at specified rate."""
        current_node = self._client.get_node(self.NODES['temperature_actual'])
        current_temp = current_node.get_value()
        
        temp_diff = abs(target_C - current_temp)
        ramp_time = temp_diff / rate_Cps
        steps = int(ramp_time * 10)  # 10 Hz update rate
        
        for i in range(steps):
            if not self._client:
                break
                
            intermediate_temp = current_temp + (target_C - current_temp) * (i + 1) / steps
            await self.set_temperature(intermediate_temp)
            await asyncio.sleep(0.1)
            
    async def set_gas_flow(self, gas: str, flow_sccm: float) -> bool:
        """Set gas flow rate."""
        try:
            flows_node = self._client.get_node(self.NODES['gas_flows'])
            flows = flows_node.get_value() or {}
            flows[gas] = flow_sccm
            flows_node.set_value(flows)
            return True
            
        except Exception as e:
            logger.error(f"Failed to set gas flow: {e}")
            return False
            
    async def set_pressure(self, pressure_Torr: float) -> bool:
        """Set chamber pressure."""
        try:
            # In real system, would control throttle valve
            logger.info(f"Setting pressure to {pressure_Torr} Torr")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set pressure: {e}")
            return False
            
    async def set_emissivity(self, emissivity: float) -> bool:
        """Set pyrometer emissivity correction."""
        try:
            emissivity_node = self._client.get_node(self.NODES['emissivity'])
            emissivity_node.set_value(float(emissivity))
            return True
            
        except Exception as e:
            logger.error(f"Failed to set emissivity: {e}")
            return False


# Driver factory
def create_driver(driver_type: str, config: DriverConfig) -> BaseDriver:
    """Create hardware driver instance."""
    drivers = {
        'ion_implant': IonImplantDriver,
        'rtp': RTPDriver,
    }
    
    driver_class = drivers.get(driver_type)
    if not driver_class:
        raise ValueError(f"Unknown driver type: {driver_type}")
        
    return driver_class(config)
