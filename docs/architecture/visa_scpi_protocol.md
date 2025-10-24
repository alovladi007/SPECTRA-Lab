# services/instruments/app/drivers/connection.py

“””
VISA/SCPI Core Library for Instrument Communication

Provides:

- VISA resource management (USB, GPIB, TCP/IP)
- SCPI command abstraction
- Response parsing and validation
- Error handling and retry logic
- Timeout management
- Connection pooling
  “””

import pyvisa
import time
import logging
from typing import Optional, Any, Dict, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import re
import numpy as np
from abc import ABC, abstractmethod

# ============================================================================

# Configuration

# ============================================================================

@dataclass
class ConnectionConfig:
“”“Configuration for instrument connection”””
timeout: float = 5.0  # seconds
write_termination: str = ‘\n’
read_termination: str = ‘\n’
encoding: str = ‘ascii’
chunk_size: int = 20480  # bytes
query_delay: float = 0.0  # seconds between write and read

# Retry configuration
max_retries: int = 3
retry_delay: float = 1.0  # seconds

# Advanced
baud_rate: Optional[int] = None  # For serial connections
data_bits: int = 8
parity: str = 'none'
stop_bits: int = 1

class ConnectionType(str, Enum):
“”“Types of instrument connections”””
VISA_USB = “visa_usb”
VISA_GPIB = “visa_gpib”
VISA_TCPIP = “visa_tcpip”
SERIAL = “serial”
USB_RAW = “usb_raw”
SIMULATOR = “simulator”

# ============================================================================

# Exceptions

# ============================================================================

class InstrumentError(Exception):
“”“Base exception for instrument errors”””
pass

class ConnectionError(InstrumentError):
“”“Connection-related errors”””
pass

class CommandError(InstrumentError):
“”“Command execution errors”””
pass

class TimeoutError(InstrumentError):
“”“Timeout errors”””
pass

class ParseError(InstrumentError):
“”“Response parsing errors”””
pass

# ============================================================================

# VISA Connection Manager

# ============================================================================

class VISAConnection:
“””
VISA connection wrapper with automatic retry and error handling

Supports:
- USB (VISA-USB)
- GPIB (VISA-GPIB)
- TCP/IP (VISA-TCPIP)
- Serial (VISA-ASRL)
"""

def __init__(
    self,
    resource_name: str,
    config: Optional[ConnectionConfig] = None,
    logger: Optional[logging.Logger] = None
):
    """
    Initialize VISA connection
    
    Args:
        resource_name: VISA resource string (e.g., 'USB0::0x1234::0x5678::SN12345::INSTR')
        config: Connection configuration
        logger: Logger instance
    """
    self.resource_name = resource_name
    self.config = config or ConnectionConfig()
    self.logger = logger or logging.getLogger(__name__)
    
    self.rm: Optional[pyvisa.ResourceManager] = None
    self.instrument: Optional[pyvisa.Resource] = None
    self.is_connected = False
    
def connect(self) -> None:
    """Establish connection to instrument"""
    try:
        # Create resource manager
        self.rm = pyvisa.ResourceManager()
        
        # Open instrument
        self.instrument = self.rm.open_resource(
            self.resource_name,
            timeout=int(self.config.timeout * 1000)  # Convert to milliseconds
        )
        
        # Configure connection
        self.instrument.write_termination = self.config.write_termination
        self.instrument.read_termination = self.config.read_termination
        self.instrument.encoding = self.config.encoding
        
        # Set baud rate for serial connections
        if self.config.baud_rate and hasattr(self.instrument, 'baud_rate'):
            self.instrument.baud_rate = self.config.baud_rate
        
        self.is_connected = True
        self.logger.info(f"Connected to {self.resource_name}")
        
    except pyvisa.VisaIOError as e:
        raise ConnectionError(f"Failed to connect to {self.resource_name}: {e}")

def disconnect(self) -> None:
    """Close connection to instrument"""
    if self.instrument:
        try:
            self.instrument.close()
            self.logger.info(f"Disconnected from {self.resource_name}")
        except Exception as e:
            self.logger.warning(f"Error during disconnect: {e}")
        finally:
            self.instrument = None
            self.is_connected = False
    
    if self.rm:
        try:
            self.rm.close()
        except Exception:
            pass
        finally:
            self.rm = None

def write(self, command: str) -> None:
    """
    Write command to instrument
    
    Args:
        command: SCPI command string
        
    Raises:
        ConnectionError: If not connected
        CommandError: If write fails
    """
    if not self.is_connected or not self.instrument:
        raise ConnectionError("Not connected to instrument")
    
    try:
        self.instrument.write(command)
        self.logger.debug(f"Write: {command}")
    except pyvisa.VisaIOError as e:
        raise CommandError(f"Write failed: {e}")

def read(self) -> str:
    """
    Read response from instrument
    
    Returns:
        Response string
        
    Raises:
        ConnectionError: If not connected
        TimeoutError: If read times out
        CommandError: If read fails
    """
    if not self.is_connected or not self.instrument:
        raise ConnectionError("Not connected to instrument")
    
    try:
        response = self.instrument.read()
        self.logger.debug(f"Read: {response}")
        return response
    except pyvisa.errors.VisaIOError as e:
        if 'VI_ERROR_TMO' in str(e):
            raise TimeoutError("Read operation timed out")
        else:
            raise CommandError(f"Read failed: {e}")

def query(self, command: str, delay: Optional[float] = None) -> str:
    """
    Write command and read response
    
    Args:
        command: SCPI command string
        delay: Optional delay between write and read (overrides config)
        
    Returns:
        Response string
    """
    self.write(command)
    time.sleep(delay if delay is not None else self.config.query_delay)
    return self.read()

def query_with_retry(self, command: str) -> str:
    """Query with automatic retry on failure"""
    for attempt in range(self.config.max_retries):
        try:
            return self.query(command)
        except (TimeoutError, CommandError) as e:
            if attempt == self.config.max_retries - 1:
                raise
            self.logger.warning(f"Query failed (attempt {attempt + 1}/{self.config.max_retries}): {e}")
            time.sleep(self.config.retry_delay)
    
    raise CommandError("Max retries exceeded")

def read_bytes(self, count: int) -> bytes:
    """Read binary data"""
    if not self.is_connected or not self.instrument:
        raise ConnectionError("Not connected to instrument")
    
    try:
        return self.instrument.read_bytes(count)
    except pyvisa.VisaIOError as e:
        raise CommandError(f"Binary read failed: {e}")

def __enter__(self):
    """Context manager entry"""
    self.connect()
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit"""
    self.disconnect()

# ============================================================================

# SCPI Command Abstraction

# ============================================================================

class SCPICommand:
“””
SCPI command builder and parser

Provides type-safe command construction and response parsing.
"""

# Common SCPI commands (IEEE 488.2)
IDN = "*IDN?"  # Identification query
RST = "*RST"  # Reset
CLS = "*CLS"  # Clear status
ESR = "*ESR?"  # Event status register query
OPC = "*OPC"  # Operation complete
WAI = "*WAI"  # Wait for operations to complete

@staticmethod
def build(subsystem: str, command: str, params: Optional[List[Any]] = None) -> str:
    """
    Build SCPI command
    
    Args:
        subsystem: SCPI subsystem (e.g., 'SOUR', 'MEAS')
        command: Command name (e.g., 'VOLT', 'CURR')
        params: Optional parameters
        
    Returns:
        SCPI command string
        
    Example:
        >>> SCPICommand.build('SOUR', 'VOLT', [1.5])
        'SOUR:VOLT 1.5'
    """
    parts = [subsystem, command]
    cmd = ':'.join(parts)
    
    if params:
        param_str = ','.join(str(p) for p in params)
        cmd = f"{cmd} {param_str}"
    
    return cmd

@staticmethod
def parse_numeric(response: str) -> float:
    """Parse numeric response"""
    try:
        # Remove units and extra whitespace
        cleaned = re.sub(r'[^\d.eE+-]', '', response)
        return float(cleaned)
    except ValueError:
        raise ParseError(f"Cannot parse numeric value from: {response}")

@staticmethod
def parse_numeric_list(response: str, separator: str = ',') -> List[float]:
    """Parse comma-separated numeric list"""
    try:
        parts = response.split(separator)
        return [SCPICommand.parse_numeric(part) for part in parts]
    except (ValueError, ParseError):
        raise ParseError(f"Cannot parse numeric list from: {response}")

@staticmethod
def parse_boolean(response: str) -> bool:
    """Parse boolean response (0/1, ON/OFF, etc.)"""
    response = response.strip().upper()
    if response in ('1', 'ON', 'TRUE'):
        return True
    elif response in ('0', 'OFF', 'FALSE'):
        return False
    else:
        raise ParseError(f"Cannot parse boolean from: {response}")

@staticmethod
def parse_idn(response: str) -> Dict[str, str]:
    """
    Parse *IDN? response
    
    Format: Manufacturer,Model,SerialNumber,FirmwareVersion
    
    Returns:
        Dictionary with manufacturer, model, serial, firmware
    """
    parts = response.split(',')
    if len(parts) < 4:
        raise ParseError(f"Invalid *IDN? response: {response}")
    
    return {
        'manufacturer': parts[0].strip(),
        'model': parts[1].strip(),
        'serial_number': parts[2].strip(),
        'firmware': parts[3].strip()
    }

# ============================================================================

# High-Level Instrument Interface

# ============================================================================

class InstrumentInterface(ABC):
“””
Abstract base class for instrument drivers

All instrument drivers should inherit from this class and implement
the required methods.
"""

def __init__(
    self,
    resource_name: str,
    config: Optional[ConnectionConfig] = None,
    logger: Optional[logging.Logger] = None
):
    """Initialize instrument interface"""
    self.resource_name = resource_name
    self.config = config or ConnectionConfig()
    self.logger = logger or logging.getLogger(__name__)
    
    self.connection: Optional[VISAConnection] = None
    self._identity: Optional[Dict[str, str]] = None

def connect(self) -> None:
    """Establish connection to instrument"""
    self.connection = VISAConnection(
        self.resource_name,
        config=self.config,
        logger=self.logger
    )
    self.connection.connect()
    
    # Query identity
    try:
        idn_response = self.connection.query(SCPICommand.IDN)
        self._identity = SCPICommand.parse_idn(idn_response)
        self.logger.info(f"Connected to {self._identity['manufacturer']} {self._identity['model']}")
    except Exception as e:
        self.logger.warning(f"Could not query identity: {e}")

def disconnect(self) -> None:
    """Close connection to instrument"""
    if self.connection:
        self.connection.disconnect()
        self.connection = None

def reset(self) -> None:
    """Reset instrument to default state"""
    if not self.connection:
        raise ConnectionError("Not connected")
    
    self.connection.write(SCPICommand.RST)
    time.sleep(1)  # Wait for reset to complete

def clear(self) -> None:
    """Clear instrument status registers"""
    if not self.connection:
        raise ConnectionError("Not connected")
    
    self.connection.write(SCPICommand.CLS)

def get_identity(self) -> Dict[str, str]:
    """Get instrument identity"""
    if self._identity is None:
        if not self.connection:
            raise ConnectionError("Not connected")
        idn_response = self.connection.query(SCPICommand.IDN)
        self._identity = SCPICommand.parse_idn(idn_response)
    
    return self._identity

def check_errors(self) -> Optional[str]:
    """Check for instrument errors (must be implemented by subclass)"""
    return None

@abstractmethod
def configure(self, **params) -> None:
    """Configure instrument parameters"""
    pass

@abstractmethod
def measure(self) -> Any:
    """Perform measurement"""
    pass

def __enter__(self):
    """Context manager entry"""
    self.connect()
    return self

def __exit__(self, exc_type, exc_val, exc_tb):
    """Context manager exit"""
    self.disconnect()

# ============================================================================

# Utility Functions

# ============================================================================

def list_resources() -> List[str]:
“””
List all available VISA resources

Returns:
    List of resource strings
"""
try:
    rm = pyvisa.ResourceManager()
    resources = rm.list_resources()
    rm.close()
    return list(resources)
except Exception as e:
    logging.error(f"Failed to list resources: {e}")
    return []

def test_connection(resource_name: str) -> bool:
“””
Test if instrument is reachable

Args:
    resource_name: VISA resource string
    
Returns:
    True if connection successful, False otherwise
"""
try:
    with VISAConnection(resource_name) as conn:
        idn = conn.query(SCPICommand.IDN)
        return len(idn) > 0
except Exception:
    return False

# ============================================================================

# Example Usage

# ============================================================================

def example_usage():
“”“Demonstrate VISA/SCPI core library”””
print(”=” * 80)
print(“VISA/SCPI Core Library - Example Usage”)
print(”=” * 80)

# 1. List available resources
print("\n1. Listing VISA Resources:")
resources = list_resources()
if resources:
    for resource in resources:
        print(f"   Found: {resource}")
else:
    print("   No VISA resources found (this is expected without hardware)")

# 2. Simulate connection (without real hardware)
print("\n2. Connection Management:")
print("   Example resource: 'USB0::0x05E6::0x2400::1234567::INSTR'")
print("   Connection config:")
config = ConnectionConfig(timeout=5.0, max_retries=3)
print(f"     - Timeout: {config.timeout}s")
print(f"     - Max retries: {config.max_retries}")

# 3. SCPI Command Building
print("\n3. SCPI Command Construction:")

cmd1 = SCPICommand.build('SOUR', 'VOLT', [1.5])
print(f"   Set voltage: {cmd1}")

cmd2 = SCPICommand.build('MEAS', 'CURR', ['DC'])
print(f"   Measure current: {cmd2}")

cmd3 = SCPICommand.build('SYST', 'BEEP')
print(f"   System beep: {cmd3}")

# 4. Response Parsing
print("\n4. Response Parsing:")

# Numeric
numeric_response = "+1.23456E-03"
parsed_numeric = SCPICommand.parse_numeric(numeric_response)
print(f"   Numeric: '{numeric_response}' → {parsed_numeric}")

# List
list_response = "1.5,2.3,3.1,4.8"
parsed_list = SCPICommand.parse_numeric_list(list_response)
print(f"   List: '{list_response}' → {parsed_list}")

# Boolean
bool_response = "1"
parsed_bool = SCPICommand.parse_boolean(bool_response)
print(f"   Boolean: '{bool_response}' → {parsed_bool}")

# Identity
idn_response = "Keithley Instruments,Model 2400,1234567,v1.2.3"
parsed_idn = SCPICommand.parse_idn(idn_response)
print(f"   Identity: {parsed_idn['manufacturer']} {parsed_idn['model']}")

# 5. Error Handling
print("\n5. Error Handling:")
print("   ✓ ConnectionError - Connection failures")
print("   ✓ CommandError - Command execution failures")
print("   ✓ TimeoutError - Read/write timeouts")
print("   ✓ ParseError - Response parsing failures")

print("\n" + "=" * 80)
print("VISA/SCPI Core Library demonstration complete!")
print("=" * 80)

if **name** == “**main**”:
# Configure logging
logging.basicConfig(level=logging.INFO)

# Run example
example_usage()