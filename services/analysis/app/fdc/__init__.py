"""
Fault Detection and Classification (FDC) Module

Detects and classifies faults in CVD processes based on patterns in:
- Film thickness (mean, uniformity)
- Film stress (mean, gradient)
- Film adhesion (score, class)

Fault types:
- Gradual drift (thermal, composition, calibration)
- Sudden shifts (contamination, equipment failure)
- Increased variation (instability, hardware degradation)
- Pattern anomalies (recipe errors, software bugs)
"""

from .detector import (
    FDCDetector,
    FaultType,
    FaultSeverity,
    Fault,
    detect_faults,
)

from .classifiers import (
    ThicknessFaultClassifier,
    StressFaultClassifier,
    AdhesionFaultClassifier,
    classify_fault_root_cause,
)

from .patterns import (
    PatternDetector,
    TrendPattern,
    ShiftPattern,
    CyclicPattern,
    detect_all_patterns,
)

__all__ = [
    # Detector
    "FDCDetector",
    "FaultType",
    "FaultSeverity",
    "Fault",
    "detect_faults",

    # Classifiers
    "ThicknessFaultClassifier",
    "StressFaultClassifier",
    "AdhesionFaultClassifier",
    "classify_fault_root_cause",

    # Patterns
    "PatternDetector",
    "TrendPattern",
    "ShiftPattern",
    "CyclicPattern",
    "detect_all_patterns",
]
