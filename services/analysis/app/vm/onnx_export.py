"""
ONNX Export for VM Models

Export trained sklearn models to ONNX format for deployment.
"""

import numpy as np
from typing import Optional
import logging
from pathlib import Path

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from .models import VMModel

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: VMModel,
    output_path: str,
    opset_version: int = 12,
) -> bool:
    """
    Export VM model to ONNX format

    Args:
        model: Trained VM model
        output_path: Path to save ONNX file
        opset_version: ONNX opset version

    Returns:
        True if successful, False otherwise
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX export requires: pip install skl2onnx onnx onnxruntime")
        return False

    if model.model is None or model.scaler is None:
        logger.error("Model not trained. Cannot export.")
        return False

    try:
        # Create pipeline with scaler + model
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("scaler", model.scaler),
            ("model", model.model),
        ])

        # Define input type
        n_features = model.n_features if model.n_features > 0 else len(model.feature_names)
        initial_type = [("float_input", FloatTensorType([None, n_features]))]

        # Convert to ONNX
        onnx_model = convert_sklearn(
            pipeline,
            initial_types=initial_type,
            target_opset=opset_version,
        )

        # Save to file
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        logger.info(f"Model exported to ONNX: {output_path}")

        # Verify ONNX model
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification passed")

        return True

    except Exception as e:
        logger.error(f"Error exporting to ONNX: {e}")
        return False


def load_onnx_model(onnx_path: str) -> Optional["ort.InferenceSession"]:
    """
    Load ONNX model for inference

    Args:
        onnx_path: Path to ONNX file

    Returns:
        ONNX Runtime inference session
    """
    if not ONNX_AVAILABLE:
        logger.error("ONNX runtime not available: pip install onnxruntime")
        return None

    try:
        session = ort.InferenceSession(onnx_path)
        logger.info(f"ONNX model loaded from: {onnx_path}")
        return session

    except Exception as e:
        logger.error(f"Error loading ONNX model: {e}")
        return None


class ONNXPredictor:
    """
    Predictor using ONNX runtime

    Faster inference than sklearn for deployment.
    """

    def __init__(self, onnx_path: str):
        """
        Args:
            onnx_path: Path to ONNX model file
        """
        self.session = load_onnx_model(onnx_path)

        if self.session is not None:
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using ONNX model

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        if self.session is None:
            raise ValueError("ONNX session not initialized")

        # Ensure float32 dtype
        X = X.astype(np.float32)

        # Run inference
        result = self.session.run(
            [self.output_name],
            {self.input_name: X},
        )

        return result[0].flatten()

    def predict_single(self, features: np.ndarray) -> float:
        """
        Predict for a single sample

        Args:
            features: Feature vector (n_features,)

        Returns:
            Single prediction value
        """
        # Reshape to (1, n_features)
        X = features.reshape(1, -1)
        predictions = self.predict(X)
        return float(predictions[0])


def save_model_metadata(
    model: VMModel,
    output_path: str,
):
    """
    Save model metadata to JSON file

    Args:
        model: VM model
        output_path: Path to save JSON
    """
    import json

    metadata = {
        "film_family": model.film_family.value,
        "target": model.target.value,
        "model_type": model.model_type,
        "version": model.version,
        "training_date": model.training_date,
        "n_features": model.n_features,
        "feature_names": model.feature_names,
        "hyperparameters": model.hyperparameters,
        "performance": {
            "train_r2": model.train_score,
            "val_r2": model.val_score,
            "test_r2": model.test_score,
        },
    }

    # Get feature importance if available
    try:
        metadata["feature_importance"] = model.get_feature_importance()
    except:
        pass

    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Model metadata saved to: {output_path}")


def export_model_package(
    model: VMModel,
    output_dir: str,
    model_name: Optional[str] = None,
) -> bool:
    """
    Export complete model package (ONNX + metadata)

    Args:
        model: Trained VM model
        output_dir: Directory to save files
        model_name: Optional model name (default: film_target)

    Returns:
        True if successful
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = f"{model.film_family.value}_{model.target.value}"

    # Export ONNX
    onnx_path = output_path / f"{model_name}.onnx"
    success = export_to_onnx(model, str(onnx_path))

    if not success:
        return False

    # Save metadata
    metadata_path = output_path / f"{model_name}_metadata.json"
    save_model_metadata(model, str(metadata_path))

    logger.info(f"Model package exported to: {output_dir}")
    logger.info(f"  ONNX: {onnx_path.name}")
    logger.info(f"  Metadata: {metadata_path.name}")

    return True
