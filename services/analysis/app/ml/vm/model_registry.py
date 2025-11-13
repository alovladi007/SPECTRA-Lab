"""
CVD Virtual Metrology - Model Registry
Model versioning, storage, and lifecycle management
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4
from pathlib import Path
import logging
import json

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


logger = logging.getLogger(__name__)


# ============================================================================
# Model Metadata
# ============================================================================

class ModelStatus(str, Enum):
    """Model lifecycle status"""
    TRAINING = "training"
    VALIDATING = "validating"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"


class ModelType(str, Enum):
    """Type of ML model"""
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    SVR = "svr"
    ENSEMBLE = "ensemble"


@dataclass
class ModelMetadata:
    """Metadata for a VM model"""
    model_id: str
    name: str
    version: str
    model_type: ModelType
    status: ModelStatus

    # Training info
    process_mode_id: Optional[UUID] = None
    recipe_id: Optional[UUID] = None
    target_variable: str = "thickness_nm"

    # Performance metrics
    train_metrics: Dict[str, float] = field(default_factory=dict)
    val_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)

    # Feature info
    feature_names: List[str] = field(default_factory=list)
    n_features: int = 0

    # Training data
    n_train_samples: int = 0
    n_val_samples: int = 0
    n_test_samples: int = 0

    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None

    # Creator/owner
    created_by: Optional[UUID] = None

    # Additional metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""

    # Parent model (for ensemble or transfer learning)
    parent_model_id: Optional[str] = None


@dataclass
class PredictionMetadata:
    """Metadata for a prediction"""
    prediction_id: str
    model_id: str
    run_id: UUID
    predicted_value: float
    confidence: float
    features_used: Dict[str, float]
    prediction_time: datetime = field(default_factory=datetime.utcnow)
    actual_value: Optional[float] = None  # Filled in after measurement


# ============================================================================
# Model Registry
# ============================================================================

class VMModelRegistry:
    """
    Model registry for virtual metrology models.
    Manages model versions, metadata, and lifecycle.
    """

    def __init__(self, storage_path: str = "./model_registry"):
        """
        Initialize model registry.

        Args:
            storage_path: Path to store models
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # In-memory cache
        self.model_cache: Dict[str, Any] = {}
        self.metadata_cache: Dict[str, ModelMetadata] = {}

        logger.info(f"Initialized model registry at {storage_path}")

    # ========================================================================
    # Model Registration
    # ========================================================================

    def register_model(
        self,
        name: str,
        version: str,
        model: Any,
        model_type: ModelType,
        metadata: Optional[ModelMetadata] = None,
        **kwargs,
    ) -> str:
        """
        Register a new model.

        Args:
            name: Model name
            version: Model version
            model: Trained model object
            model_type: Type of model
            metadata: Optional metadata object
            **kwargs: Additional metadata fields

        Returns:
            Model ID
        """
        model_id = str(uuid4())

        # Create metadata if not provided
        if metadata is None:
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                model_type=model_type,
                status=ModelStatus.TRAINING,
            )

            # Update from kwargs
            for key, value in kwargs.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

        # Save model
        model_path = self._get_model_path(model_id)
        joblib.dump(model, model_path)

        # Save metadata
        metadata_path = self._get_metadata_path(model_id)
        self._save_metadata(metadata, metadata_path)

        # Cache
        self.model_cache[model_id] = model
        self.metadata_cache[model_id] = metadata

        logger.info(f"Registered model: {name} v{version} (ID: {model_id})")

        return model_id

    def get_model(self, model_id: str, use_cache: bool = True) -> Optional[Any]:
        """
        Retrieve a model by ID.

        Args:
            model_id: Model ID
            use_cache: Use cached model if available

        Returns:
            Model object or None
        """
        # Check cache
        if use_cache and model_id in self.model_cache:
            return self.model_cache[model_id]

        # Load from disk
        model_path = self._get_model_path(model_id)
        if model_path.exists():
            model = joblib.load(model_path)
            self.model_cache[model_id] = model
            return model

        logger.warning(f"Model not found: {model_id}")
        return None

    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """
        Get model metadata.

        Args:
            model_id: Model ID

        Returns:
            Model metadata or None
        """
        # Check cache
        if model_id in self.metadata_cache:
            return self.metadata_cache[model_id]

        # Load from disk
        metadata_path = self._get_metadata_path(model_id)
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)
                metadata = self._deserialize_metadata(data)
                self.metadata_cache[model_id] = metadata
                return metadata

        logger.warning(f"Metadata not found: {model_id}")
        return None

    # ========================================================================
    # Model Lifecycle
    # ========================================================================

    def update_status(self, model_id: str, new_status: ModelStatus) -> bool:
        """
        Update model status.

        Args:
            model_id: Model ID
            new_status: New status

        Returns:
            Success status
        """
        metadata = self.get_metadata(model_id)
        if metadata is None:
            return False

        old_status = metadata.status
        metadata.status = new_status

        # Update timestamps
        if new_status == ModelStatus.PRODUCTION:
            metadata.deployed_at = datetime.utcnow()
        elif new_status == ModelStatus.ARCHIVED:
            metadata.archived_at = datetime.utcnow()

        # Save updated metadata
        metadata_path = self._get_metadata_path(model_id)
        self._save_metadata(metadata, metadata_path)

        logger.info(f"Updated model {model_id} status: {old_status} -> {new_status}")

        return True

    def promote_to_production(
        self,
        model_id: str,
        process_mode_id: Optional[UUID] = None,
        recipe_id: Optional[UUID] = None,
    ) -> bool:
        """
        Promote model to production.

        Args:
            model_id: Model ID
            process_mode_id: Process mode to deploy for
            recipe_id: Recipe to deploy for

        Returns:
            Success status
        """
        # Validate model
        metadata = self.get_metadata(model_id)
        if metadata is None:
            logger.error(f"Cannot promote: model {model_id} not found")
            return False

        if metadata.status == ModelStatus.PRODUCTION:
            logger.warning(f"Model {model_id} already in production")
            return True

        # Demote existing production models
        current_production = self.list_production_models(
            process_mode_id=process_mode_id,
            recipe_id=recipe_id,
        )

        for prod_model_id in current_production:
            if prod_model_id != model_id:
                self.update_status(prod_model_id, ModelStatus.ARCHIVED)
                logger.info(f"Archived previous production model: {prod_model_id}")

        # Promote new model
        success = self.update_status(model_id, ModelStatus.PRODUCTION)

        if success:
            logger.info(f"Promoted model {model_id} to production")

        return success

    def archive_model(self, model_id: str) -> bool:
        """Archive a model"""
        return self.update_status(model_id, ModelStatus.ARCHIVED)

    # ========================================================================
    # Model Evaluation
    # ========================================================================

    def evaluate_model(
        self,
        model_id: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset_type: str = "test",
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Args:
            model_id: Model ID
            X_test: Test features
            y_test: Test targets
            dataset_type: Dataset type ('train', 'val', 'test')

        Returns:
            Metrics dictionary
        """
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100,
            "n_samples": len(y_test),
        }

        # Update metadata
        metadata = self.get_metadata(model_id)
        if metadata:
            if dataset_type == "train":
                metadata.train_metrics = metrics
            elif dataset_type == "val":
                metadata.val_metrics = metrics
            elif dataset_type == "test":
                metadata.test_metrics = metrics

            metadata_path = self._get_metadata_path(model_id)
            self._save_metadata(metadata, metadata_path)

        logger.info(
            f"Evaluated model {model_id} on {dataset_type}: "
            f"MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}"
        )

        return metrics

    def compare_models(
        self,
        model_ids: List[str],
        metric: str = "mae",
        dataset: str = "test",
    ) -> pd.DataFrame:
        """
        Compare multiple models.

        Args:
            model_ids: List of model IDs
            metric: Metric to compare
            dataset: Dataset to compare on

        Returns:
            Comparison DataFrame
        """
        import pandas as pd

        comparisons = []

        for model_id in model_ids:
            metadata = self.get_metadata(model_id)
            if metadata is None:
                continue

            metrics_dict = {
                "train": metadata.train_metrics,
                "val": metadata.val_metrics,
                "test": metadata.test_metrics,
            }

            metrics = metrics_dict.get(dataset, {})

            comparisons.append({
                "model_id": model_id,
                "name": metadata.name,
                "version": metadata.version,
                "type": metadata.model_type.value,
                "status": metadata.status.value,
                metric: metrics.get(metric, np.nan),
                "r2": metrics.get("r2", np.nan),
                "n_samples": metrics.get("n_samples", 0),
            })

        df = pd.DataFrame(comparisons)

        # Sort by metric (ascending for error metrics, descending for r2)
        ascending = metric != "r2"
        df = df.sort_values(metric, ascending=ascending)

        return df

    # ========================================================================
    # Model Querying
    # ========================================================================

    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        model_type: Optional[ModelType] = None,
        process_mode_id: Optional[UUID] = None,
        recipe_id: Optional[UUID] = None,
    ) -> List[str]:
        """
        List models with filters.

        Args:
            status: Filter by status
            model_type: Filter by type
            process_mode_id: Filter by process mode
            recipe_id: Filter by recipe

        Returns:
            List of model IDs
        """
        model_ids = []

        # Scan storage directory
        for metadata_file in self.storage_path.glob("*.metadata.json"):
            with open(metadata_file, "r") as f:
                data = json.load(f)
                metadata = self._deserialize_metadata(data)

                # Apply filters
                if status and metadata.status != status:
                    continue
                if model_type and metadata.model_type != model_type:
                    continue
                if process_mode_id and metadata.process_mode_id != process_mode_id:
                    continue
                if recipe_id and metadata.recipe_id != recipe_id:
                    continue

                model_ids.append(metadata.model_id)

        return model_ids

    def list_production_models(
        self,
        process_mode_id: Optional[UUID] = None,
        recipe_id: Optional[UUID] = None,
    ) -> List[str]:
        """List production models"""
        return self.list_models(
            status=ModelStatus.PRODUCTION,
            process_mode_id=process_mode_id,
            recipe_id=recipe_id,
        )

    def get_latest_model(
        self,
        name: str,
        status: Optional[ModelStatus] = None,
    ) -> Optional[str]:
        """
        Get latest model by name.

        Args:
            name: Model name
            status: Optional status filter

        Returns:
            Model ID or None
        """
        matching_models = []

        for metadata_file in self.storage_path.glob("*.metadata.json"):
            with open(metadata_file, "r") as f:
                data = json.load(f)
                metadata = self._deserialize_metadata(data)

                if metadata.name == name:
                    if status is None or metadata.status == status:
                        matching_models.append(metadata)

        if not matching_models:
            return None

        # Sort by created_at descending
        matching_models.sort(key=lambda m: m.created_at, reverse=True)

        return matching_models[0].model_id

    # ========================================================================
    # Predictions
    # ========================================================================

    def predict(
        self,
        model_id: str,
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with a model.

        Args:
            model_id: Model ID
            features: Feature array
            feature_names: Feature names

        Returns:
            (predictions, confidence_scores)
        """
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model {model_id} not found")

        predictions = model.predict(features)

        # Calculate confidence (simplified)
        # For ensemble models, use std of predictions
        # For others, use constant or predict_proba if available
        if hasattr(model, "predict_proba"):
            # Classification model
            probas = model.predict_proba(features)
            confidence = np.max(probas, axis=1)
        else:
            # Regression model - use constant high confidence
            # TODO: Implement proper uncertainty estimation
            confidence = np.ones(len(predictions)) * 0.85

        return predictions, confidence

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _get_model_path(self, model_id: str) -> Path:
        """Get model file path"""
        return self.storage_path / f"{model_id}.model.pkl"

    def _get_metadata_path(self, model_id: str) -> Path:
        """Get metadata file path"""
        return self.storage_path / f"{model_id}.metadata.json"

    def _save_metadata(self, metadata: ModelMetadata, path: Path) -> None:
        """Save metadata to JSON"""
        data = self._serialize_metadata(metadata)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _serialize_metadata(self, metadata: ModelMetadata) -> Dict[str, Any]:
        """Convert metadata to JSON-serializable dict"""
        return {
            "model_id": metadata.model_id,
            "name": metadata.name,
            "version": metadata.version,
            "model_type": metadata.model_type.value,
            "status": metadata.status.value,
            "process_mode_id": str(metadata.process_mode_id) if metadata.process_mode_id else None,
            "recipe_id": str(metadata.recipe_id) if metadata.recipe_id else None,
            "target_variable": metadata.target_variable,
            "train_metrics": metadata.train_metrics,
            "val_metrics": metadata.val_metrics,
            "test_metrics": metadata.test_metrics,
            "feature_names": metadata.feature_names,
            "n_features": metadata.n_features,
            "n_train_samples": metadata.n_train_samples,
            "n_val_samples": metadata.n_val_samples,
            "n_test_samples": metadata.n_test_samples,
            "hyperparameters": metadata.hyperparameters,
            "created_at": metadata.created_at.isoformat(),
            "trained_at": metadata.trained_at.isoformat() if metadata.trained_at else None,
            "deployed_at": metadata.deployed_at.isoformat() if metadata.deployed_at else None,
            "archived_at": metadata.archived_at.isoformat() if metadata.archived_at else None,
            "created_by": str(metadata.created_by) if metadata.created_by else None,
            "description": metadata.description,
            "tags": metadata.tags,
            "notes": metadata.notes,
            "parent_model_id": metadata.parent_model_id,
        }

    def _deserialize_metadata(self, data: Dict[str, Any]) -> ModelMetadata:
        """Convert JSON dict to metadata object"""
        return ModelMetadata(
            model_id=data["model_id"],
            name=data["name"],
            version=data["version"],
            model_type=ModelType(data["model_type"]),
            status=ModelStatus(data["status"]),
            process_mode_id=UUID(data["process_mode_id"]) if data.get("process_mode_id") else None,
            recipe_id=UUID(data["recipe_id"]) if data.get("recipe_id") else None,
            target_variable=data.get("target_variable", "thickness_nm"),
            train_metrics=data.get("train_metrics", {}),
            val_metrics=data.get("val_metrics", {}),
            test_metrics=data.get("test_metrics", {}),
            feature_names=data.get("feature_names", []),
            n_features=data.get("n_features", 0),
            n_train_samples=data.get("n_train_samples", 0),
            n_val_samples=data.get("n_val_samples", 0),
            n_test_samples=data.get("n_test_samples", 0),
            hyperparameters=data.get("hyperparameters", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            trained_at=datetime.fromisoformat(data["trained_at"]) if data.get("trained_at") else None,
            deployed_at=datetime.fromisoformat(data["deployed_at"]) if data.get("deployed_at") else None,
            archived_at=datetime.fromisoformat(data["archived_at"]) if data.get("archived_at") else None,
            created_by=UUID(data["created_by"]) if data.get("created_by") else None,
            description=data.get("description", ""),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
            parent_model_id=data.get("parent_model_id"),
        )
