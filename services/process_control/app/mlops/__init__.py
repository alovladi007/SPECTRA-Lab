"""MLOps module for ML model lifecycle management.

Provides feature store, model registry, drift monitoring, and deployment
infrastructure for Virtual Metrology models.
"""

from .feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureGroup,
    FeatureVector,
    create_ion_implant_feature_group,
    create_rtp_feature_group,
)

from .model_registry import (
    ModelRegistry,
    ModelMetadata,
    ModelCard,
    ModelStage,
    DeploymentStrategy,
)

from .drift_monitor import (
    DriftDetector,
    DriftAlert,
    DriftSeverity,
    FeatureDriftMonitor,
    ModelDriftMonitor,
)

from .onnx_export import (
    ONNXExporter,
    ONNXInferenceRuntime,
    ONNXMetadata,
    ONNXExportConfig,
)

__all__ = [
    # Feature Store
    "FeatureStore",
    "FeatureDefinition",
    "FeatureGroup",
    "FeatureVector",
    "create_ion_implant_feature_group",
    "create_rtp_feature_group",

    # Model Registry
    "ModelRegistry",
    "ModelMetadata",
    "ModelCard",
    "ModelStage",
    "DeploymentStrategy",

    # Drift Monitoring
    "DriftDetector",
    "DriftAlert",
    "DriftSeverity",
    "FeatureDriftMonitor",
    "ModelDriftMonitor",

    # ONNX Export
    "ONNXExporter",
    "ONNXInferenceRuntime",
    "ONNXMetadata",
    "ONNXExportConfig",
]
