# services/instruments/app/drivers/builtin/keithley_2400.py

"""
Keithley 2400 Source Measure Unit (SMU) Driver

Supports:
- I-V sweeps (voltage source, current measure)
- I-V sweeps (current source, voltage measure)
- 2-wire and 4-wire measurements
- Compliance limits
- Pulsed measurements
"""

import time
import numpy as np
from typing import Dict, Any, Optional, List
from ..connection import VISAConnection, ConnectionConfig, SCPICommand, InstrumentInterface
from ..plugin_manager import InstrumentDriver, driver_plugin

@driver_plugin(
    name="keithley_2400",
    version="1.0.0",
    author="SemiconductorLab Team",
    description="Keithley 2400 Series Source Measure Unit Driver",
    supported_methods=["iv_sweep", "cv_measurement", "two_point_probe", "four_point_probe"],
    supported_models=["2400", "2401", "2410", "2420"]
)
class Keithley2400Driver(InstrumentDriver):
    """
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
            write_termination='\\n',
            read_termination='\\n'
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

    # [Rest of implementation continues...]

if __name__ == "__main__":
    example_usage()
