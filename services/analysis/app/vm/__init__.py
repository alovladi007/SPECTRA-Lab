"""
Virtual Metrology (VM) Module

ML-based prediction of film properties from process telemetry.

Supports film families:
- SiO₂ (Silicon Dioxide)
- Si₃N₄ (Silicon Nitride)
- W (Tungsten)
- TiN (Titanium Nitride)
- GaN (Gallium Nitride)
- DLC (Diamond-Like Carbon)

Prediction targets:
- Thickness (mean_nm, uniformity_pct)
- Stress (mean_MPa)
- Adhesion (score, class)
"""

from .models import (
    VMModel,
    FilmFamily,
    PredictionTarget,
    create_vm_model,
)

from .training import (
    VMTrainer,
    TrainingConfig,
    train_vm_model,
)

from .evaluation import (
    VMEvaluator,
    EvaluationMetrics,
    evaluate_vm_model,
)

from .onnx_export import (
    export_to_onnx,
    load_onnx_model,
    ONNXPredictor,
)

__all__ = [
    # Models
    "VMModel",
    "FilmFamily",
    "PredictionTarget",
    "create_vm_model",

    # Training
    "VMTrainer",
    "TrainingConfig",
    "train_vm_model",

    # Evaluation
    "VMEvaluator",
    "EvaluationMetrics",
    "evaluate_vm_model",

    # ONNX Export
    "export_to_onnx",
    "load_onnx_model",
    "ONNXPredictor",
]
