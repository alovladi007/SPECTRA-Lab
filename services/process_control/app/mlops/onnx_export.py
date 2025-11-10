"""ONNX export utilities for VM model deployment.

Provides conversion of trained scikit-learn, XGBoost, and PyTorch models
to ONNX format for cross-platform inference.
"""

import os
import json
import pickle
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import numpy as np


# ============================================================================
# ONNX Export Data Structures
# ============================================================================

@dataclass
class ONNXMetadata:
    """Metadata for exported ONNX model."""
    model_name: str
    model_version: str
    model_type: str  # "ion_vm", "rtp_vm"
    framework: str  # "sklearn", "xgboost", "pytorch"

    input_names: List[str]
    input_shapes: List[List[int]]
    input_types: List[str]

    output_names: List[str]
    output_shapes: List[List[int]]
    output_types: List[str]

    opset_version: int = 14
    onnx_file_path: str = ""
    export_date: str = ""

    preprocessing_info: Dict[str, Any] = None  # Feature scaling, encoding info
    postprocessing_info: Dict[str, Any] = None  # Output transformations


@dataclass
class ONNXExportConfig:
    """Configuration for ONNX export."""
    target_opset: int = 14  # ONNX opset version
    optimize: bool = True  # Apply ONNX optimizations
    include_preprocessing: bool = False  # Export preprocessing in ONNX graph
    quantize: bool = False  # Quantize to INT8 for faster inference
    validate: bool = True  # Validate exported model
    test_inputs: Optional[np.ndarray] = None  # For validation


# ============================================================================
# ONNX Exporter
# ============================================================================

class ONNXExporter:
    """Export ML models to ONNX format for production deployment."""

    def __init__(self, export_dir: str = "./onnx_models"):
        """Initialize ONNX exporter.

        Args:
            export_dir: Directory to save ONNX models
        """
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)

    def export_sklearn_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        model_type: str,
        feature_names: List[str],
        config: Optional[ONNXExportConfig] = None
    ) -> ONNXMetadata:
        """Export scikit-learn model to ONNX.

        Args:
            model: Trained scikit-learn model (e.g., RandomForestRegressor)
            model_name: Model name
            model_version: Model version
            model_type: Type (e.g., "ion_vm")
            feature_names: List of feature names
            config: Export configuration

        Returns:
            ONNXMetadata with export details
        """
        config = config or ONNXExportConfig()

        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError(
                "skl2onnx not installed. Install with: pip install skl2onnx"
            )

        # Define input type
        n_features = len(feature_names)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=initial_type,
            target_opset=config.target_opset
        )

        # Save ONNX model
        onnx_filename = f"{model_name}_{model_version}.onnx"
        onnx_path = os.path.join(self.export_dir, onnx_filename)

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Create metadata
        metadata = ONNXMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            framework="sklearn",
            input_names=["float_input"],
            input_shapes=[[None, n_features]],
            input_types=["float32"],
            output_names=["output"],
            output_shapes=[[None, 1]],
            output_types=["float32"],
            opset_version=config.target_opset,
            onnx_file_path=onnx_path,
        )

        # Add preprocessing info if model has scaler
        if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
            scaler = model.named_steps['scaler']
            metadata.preprocessing_info = {
                "scaler_type": type(scaler).__name__,
                "mean": scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
            }

        # Validate if requested
        if config.validate and config.test_inputs is not None:
            self._validate_onnx_model(onnx_path, model, config.test_inputs)

        # Save metadata
        self._save_metadata(metadata)

        return metadata

    def export_xgboost_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        model_type: str,
        feature_names: List[str],
        config: Optional[ONNXExportConfig] = None
    ) -> ONNXMetadata:
        """Export XGBoost model to ONNX.

        Args:
            model: Trained XGBoost model
            model_name: Model name
            model_version: Model version
            model_type: Type (e.g., "ion_vm")
            feature_names: List of feature names
            config: Export configuration

        Returns:
            ONNXMetadata with export details
        """
        config = config or ONNXExportConfig()

        try:
            from onnxmltools.convert import convert_xgboost
            from onnxmltools.convert.common.data_types import FloatTensorType
        except ImportError:
            raise ImportError(
                "onnxmltools not installed. Install with: pip install onnxmltools"
            )

        # Define input type
        n_features = len(feature_names)
        initial_type = [('float_input', FloatTensorType([None, n_features]))]

        # Convert to ONNX
        onnx_model = convert_xgboost(
            model,
            initial_types=initial_type,
            target_opset=config.target_opset
        )

        # Save ONNX model
        onnx_filename = f"{model_name}_{model_version}.onnx"
        onnx_path = os.path.join(self.export_dir, onnx_filename)

        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        # Create metadata
        metadata = ONNXMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            framework="xgboost",
            input_names=["float_input"],
            input_shapes=[[None, n_features]],
            input_types=["float32"],
            output_names=["output"],
            output_shapes=[[None, 1]],
            output_types=["float32"],
            opset_version=config.target_opset,
            onnx_file_path=onnx_path,
        )

        # Validate if requested
        if config.validate and config.test_inputs is not None:
            self._validate_onnx_model(onnx_path, model, config.test_inputs)

        # Save metadata
        self._save_metadata(metadata)

        return metadata

    def export_pytorch_model(
        self,
        model: Any,
        model_name: str,
        model_version: str,
        model_type: str,
        input_shape: List[int],
        config: Optional[ONNXExportConfig] = None
    ) -> ONNXMetadata:
        """Export PyTorch model to ONNX.

        Args:
            model: Trained PyTorch model
            model_name: Model name
            model_version: Model version
            model_type: Type (e.g., "ion_vm")
            input_shape: Input shape (e.g., [1, 10] for batch_size=1, features=10)
            config: Export configuration

        Returns:
            ONNXMetadata with export details
        """
        config = config or ONNXExportConfig()

        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch not installed. Install with: pip install torch"
            )

        # Create dummy input for tracing
        dummy_input = torch.randn(*input_shape)

        # Export to ONNX
        onnx_filename = f"{model_name}_{model_version}.onnx"
        onnx_path = os.path.join(self.export_dir, onnx_filename)

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=config.target_opset,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        # Create metadata
        metadata = ONNXMetadata(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            framework="pytorch",
            input_names=["input"],
            input_shapes=[input_shape],
            input_types=["float32"],
            output_names=["output"],
            output_shapes=[list(input_shape[:1]) + [1]],  # [batch_size, 1]
            output_types=["float32"],
            opset_version=config.target_opset,
            onnx_file_path=onnx_path,
        )

        # Validate if requested
        if config.validate and config.test_inputs is not None:
            self._validate_pytorch_onnx(onnx_path, model, config.test_inputs)

        # Save metadata
        self._save_metadata(metadata)

        return metadata

    def _validate_onnx_model(
        self,
        onnx_path: str,
        original_model: Any,
        test_inputs: np.ndarray
    ):
        """Validate ONNX model against original model.

        Args:
            onnx_path: Path to ONNX model
            original_model: Original scikit-learn/XGBoost model
            test_inputs: Test input data
        """
        try:
            import onnxruntime as ort
        except ImportError:
            print("Warning: onnxruntime not installed. Skipping validation.")
            return

        # Get original predictions
        original_predictions = original_model.predict(test_inputs)

        # Get ONNX predictions
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        onnx_predictions = session.run(None, {input_name: test_inputs.astype(np.float32)})[0]

        # Compare
        max_diff = np.max(np.abs(original_predictions.flatten() - onnx_predictions.flatten()))

        if max_diff > 1e-4:
            print(f"Warning: ONNX validation failed. Max difference: {max_diff}")
        else:
            print(f"✓ ONNX validation passed. Max difference: {max_diff:.2e}")

    def _validate_pytorch_onnx(
        self,
        onnx_path: str,
        original_model: Any,
        test_inputs: np.ndarray
    ):
        """Validate PyTorch ONNX model.

        Args:
            onnx_path: Path to ONNX model
            original_model: Original PyTorch model
            test_inputs: Test input data
        """
        try:
            import torch
            import onnxruntime as ort
        except ImportError:
            print("Warning: torch or onnxruntime not installed. Skipping validation.")
            return

        # Get original predictions
        original_model.eval()
        with torch.no_grad():
            torch_input = torch.from_numpy(test_inputs).float()
            original_predictions = original_model(torch_input).numpy()

        # Get ONNX predictions
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        onnx_predictions = session.run(None, {input_name: test_inputs.astype(np.float32)})[0]

        # Compare
        max_diff = np.max(np.abs(original_predictions.flatten() - onnx_predictions.flatten()))

        if max_diff > 1e-4:
            print(f"Warning: ONNX validation failed. Max difference: {max_diff}")
        else:
            print(f"✓ ONNX validation passed. Max difference: {max_diff:.2e}")

    def _save_metadata(self, metadata: ONNXMetadata):
        """Save ONNX metadata to JSON.

        Args:
            metadata: ONNXMetadata to save
        """
        metadata_filename = f"{metadata.model_name}_{metadata.model_version}_onnx_metadata.json"
        metadata_path = os.path.join(self.export_dir, metadata_filename)

        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2)

    def load_onnx_metadata(self, model_name: str, model_version: str) -> Optional[ONNXMetadata]:
        """Load ONNX metadata from disk.

        Args:
            model_name: Model name
            model_version: Model version

        Returns:
            ONNXMetadata if found
        """
        metadata_filename = f"{model_name}_{model_version}_onnx_metadata.json"
        metadata_path = os.path.join(self.export_dir, metadata_filename)

        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            data = json.load(f)
            return ONNXMetadata(**data)

    def optimize_onnx_model(self, onnx_path: str) -> str:
        """Optimize ONNX model for faster inference.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Path to optimized model
        """
        try:
            from onnxruntime.transformers import optimizer
        except ImportError:
            print("Warning: onnxruntime transformers not available. Skipping optimization.")
            return onnx_path

        optimized_path = onnx_path.replace(".onnx", "_optimized.onnx")

        # Apply optimizations
        optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # Generic optimizations
            num_heads=0,
            hidden_size=0,
            optimization_options=None
        ).save_model_to_file(optimized_path)

        print(f"✓ Optimized model saved to: {optimized_path}")
        return optimized_path

    def quantize_onnx_model(self, onnx_path: str) -> str:
        """Quantize ONNX model to INT8 for faster inference.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Path to quantized model
        """
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
        except ImportError:
            print("Warning: onnxruntime quantization not available.")
            return onnx_path

        quantized_path = onnx_path.replace(".onnx", "_quantized.onnx")

        # Apply dynamic quantization
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8
        )

        print(f"✓ Quantized model saved to: {quantized_path}")
        return quantized_path


# ============================================================================
# ONNX Inference Runtime
# ============================================================================

class ONNXInferenceRuntime:
    """Runtime for ONNX model inference."""

    def __init__(self, onnx_path: str):
        """Initialize inference runtime.

        Args:
            onnx_path: Path to ONNX model
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "onnxruntime not installed. Install with: pip install onnxruntime"
            )

        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            features: Input features (shape: [batch_size, n_features])

        Returns:
            Predictions (shape: [batch_size, 1])
        """
        # Ensure correct dtype
        features = features.astype(np.float32)

        # Run inference
        result = self.session.run(
            [self.output_name],
            {self.input_name: features}
        )[0]

        return result

    def predict_single(self, features: np.ndarray) -> float:
        """Run inference on single sample.

        Args:
            features: Single feature vector (1D array)

        Returns:
            Single prediction value
        """
        # Reshape to batch
        features_batch = features.reshape(1, -1)
        result = self.predict(features_batch)
        return float(result[0, 0])


# Export
__all__ = [
    "ONNXExporter",
    "ONNXInferenceRuntime",
    "ONNXMetadata",
    "ONNXExportConfig",
]
