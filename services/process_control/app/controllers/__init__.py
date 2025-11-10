"""Process control controllers package."""

from .ion import (
    DoseIntegrator,
    ScanUniformityController,
    R2RController,
    BeamDriftDetector,
    DoseIntegrationResult,
    UniformityMap,
    ScanCorrectionParameters,
    R2RState,
    BeamDriftDetection,
    FDCAlert,
)

from .rtp import (
    PIDController,
    MPCController,
    R2RController as RTPR2RController,
    ThermalBudgetCalculator,
    PerformanceAnalyzer,
    PIDGains,
    MPCParameters,
    ThermalBudget,
    RampFidelity,
    ControllerPerformance,
)

__all__ = [
    # Ion controllers
    "DoseIntegrator",
    "ScanUniformityController",
    "R2RController",
    "BeamDriftDetector",
    "DoseIntegrationResult",
    "UniformityMap",
    "ScanCorrectionParameters",
    "R2RState",
    "BeamDriftDetection",
    "FDCAlert",
    # RTP controllers
    "PIDController",
    "MPCController",
    "RTPR2RController",
    "ThermalBudgetCalculator",
    "PerformanceAnalyzer",
    "PIDGains",
    "MPCParameters",
    "ThermalBudget",
    "RampFidelity",
    "ControllerPerformance",
]
