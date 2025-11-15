"""
CVD Tool Drivers & Hardware-in-Loop Simulators

This package provides:
- CVDTool Protocol interface for real drivers and HIL simulators
- Base driver implementations for various CVD technologies
- Physics-based HIL simulators for testing without hardware
- Communication adapters (SCPI/VISA, OPC-UA, SECS-II/GEM)
"""

# Core protocol and types
from .cvd_tool import (
    CVDTool,
    ToolStatus,
    ToolState,
    ToolCapabilities,
    CVDTelemetry,
    TelemetryType,
    ToolError,
)

# HIL Simulator
from .hil_simulator import (
    HILCVDSimulator,
    PhysicsConfig,
    FaultInjectionConfig,
)

# Thermal CVD Drivers
from .thermal_cvd import (
    ThermalCVDDriverBase,
    APCVDDriver,
    LPCVDDriver,
    UHVCVDDriver,
)

# Plasma CVD Drivers
from .plasma_cvd import (
    PlasmaCVDDriverBase,
    PECVDDriver,
    HDPCVDDriver,
    MPCVDDriver,
    RPCVDDriver,
)

# Specialty CVD Drivers
from .specialty_cvd import (
    MOCVDDriver,
    AACVDDriver,
)

# Communication Adapters
from .comm_adapters import (
    SCPIAdapter,
    OPCUAAdapter,
    SECS2Adapter,
    CommAdapterFactory,
)

__all__ = [
    # Core protocol
    "CVDTool",
    "ToolStatus",
    "ToolState",
    "ToolCapabilities",
    "CVDTelemetry",
    "TelemetryType",
    "ToolError",
    # HIL Simulator
    "HILCVDSimulator",
    "PhysicsConfig",
    "FaultInjectionConfig",
    # Thermal CVD
    "ThermalCVDDriverBase",
    "APCVDDriver",
    "LPCVDDriver",
    "UHVCVDDriver",
    # Plasma CVD
    "PlasmaCVDDriverBase",
    "PECVDDriver",
    "HDPCVDDriver",
    "MPCVDDriver",
    "RPCVDDriver",
    # Specialty CVD
    "MOCVDDriver",
    "AACVDDriver",
    # Communication
    "SCPIAdapter",
    "OPCUAAdapter",
    "SECS2Adapter",
    "CommAdapterFactory",
]
