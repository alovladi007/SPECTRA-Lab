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
    DepositionRateCalculator,
    UniformityCalculator,
    DepositionParameters,
    CVDMode,
    ArrheniusParameters,
)

from .stress import (
    StressModel,
    IntrinsicStressCalculator,
    ThermalStressCalculator,
    StressMeasurementMethod,
    ProcessConditions,
    MaterialProperties,
    wafer_curvature_to_stress,
    xrd_to_stress,
    get_material_properties,
)

from .adhesion import (
    AdhesionModel,
    AdhesionTest,
    AdhesionTestResult,
    AdhesionFactors,
    AdhesionClass,
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

from .vm_features import (
    VMFeatureExtractor,
    TelemetryData,
    create_training_dataset,
    feature_importance_analysis,
)

__all__ = [
    # Thickness models
    "ThicknessModel",
    "DepositionRateCalculator",
    "UniformityCalculator",
    "DepositionParameters",
    "CVDMode",
    "ArrheniusParameters",
    # Stress models
    "StressModel",
    "IntrinsicStressCalculator",
    "ThermalStressCalculator",
    "StressMeasurementMethod",
    "ProcessConditions",
    "MaterialProperties",
    "wafer_curvature_to_stress",
    "xrd_to_stress",
    "get_material_properties",
    # Adhesion models
    "AdhesionModel",
    "AdhesionTest",
    "AdhesionTestResult",
    "AdhesionFactors",
    "AdhesionClass",
    "simulate_tape_test",
    "simulate_scratch_test",
    "simulate_nanoindentation",
    "simulate_stud_pull",
    # Reactor geometry
    "ReactorType",
    "ShowerheadReactor",
    "HorizontalFlowReactor",
    "BatchFurnaceReactor",
    # VM Features
    "VMFeatureExtractor",
    "TelemetryData",
    "create_training_dataset",
    "feature_importance_analysis",
]
