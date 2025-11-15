"""
Advanced Physics Models for CVD Process Simulation

This package provides sophisticated physics-based models for:
- Film thickness prediction and uniformity
- Film stress calculation (intrinsic + thermal)
- Adhesion scoring and test simulation

These models are used for:
1. HIL simulator physics
2. VM (Virtual Metrology) feature engineering
3. Process optimization and prediction
"""

from .thickness import (
    ThicknessModel,
    ReactorGeometry,
    DepositionRateCalculator,
    UniformityCalculator,
)

from .stress import (
    StressModel,
    IntrinsicStressCalculator,
    ThermalStressCalculator,
    StressMeasurementMethod,
    wafer_curvature_to_stress,
    xrd_to_stress,
)

from .adhesion import (
    AdhesionModel,
    AdhesionTest,
    AdhesionTestResult,
    simulate_tape_test,
    simulate_scratch_test,
    simulate_nanoindentation,
    simulate_stud_pull,
)

from .reactor_geometry import (
    ReactorType,
    ShowerheadReactor,
    HorizontalFlowReactor,
    BatchFurnaceReactor,
)

__all__ = [
    # Thickness models
    "ThicknessModel",
    "ReactorGeometry",
    "DepositionRateCalculator",
    "UniformityCalculator",
    # Stress models
    "StressModel",
    "IntrinsicStressCalculator",
    "ThermalStressCalculator",
    "StressMeasurementMethod",
    "wafer_curvature_to_stress",
    "xrd_to_stress",
    # Adhesion models
    "AdhesionModel",
    "AdhesionTest",
    "AdhesionTestResult",
    "simulate_tape_test",
    "simulate_scratch_test",
    "simulate_nanoindentation",
    "simulate_stud_pull",
    # Reactor geometry
    "ReactorType",
    "ShowerheadReactor",
    "HorizontalFlowReactor",
    "BatchFurnaceReactor",
]
