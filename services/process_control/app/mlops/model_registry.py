"""ML Model Registry for managing VM models lifecycle.

Provides versioning, staging, deployment management, and model cards
for Virtual Metrology models.
"""

import json
import os
import shutil
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime
import hashlib


# ============================================================================
# Model Registry Data Structures
# ============================================================================

class ModelStage(Enum):
    """Model lifecycle stages."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DeploymentStrategy(Enum):
    """Model deployment strategies."""
    DIRECT = "direct"  # Replace immediately
    SHADOW = "shadow"  # Run in parallel, log predictions
    CANARY = "canary"  # Route small % of traffic
    BLUE_GREEN = "blue_green"  # Switch between two versions


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    model_name: str
    model_version: str
    model_type: str  # "ion_vm", "rtp_vm"
    stage: ModelStage

    # Training info
    training_date: str
    training_samples: int
    training_r2_score: float
    validation_r2_score: float

    # Model files
    model_file_path: str
    model_file_hash: str  # SHA256

    # Performance metrics
    mae: float = 0.0  # Mean absolute error
    rmse: float = 0.0  # Root mean squared error

    # Deployment info
    deployment_strategy: DeploymentStrategy = DeploymentStrategy.DIRECT
    deployed_date: Optional[str] = None
    deployed_by: Optional[str] = None

    # Metadata
    description: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ModelCard:
    """Model card for documentation and governance.

    Based on Google's Model Card framework for responsible AI.
    """
    model_name: str
    model_version: str

    # Model details
    model_type: str
    intended_use: str
    training_algorithm: str  # e.g., "random_forest", "xgboost", "neural_network"

    # Training data
    training_data_description: str
    training_data_size: int
    training_data_date_range: str
    feature_names: List[str]

    # Performance
    performance_metrics: Dict[str, float]
    performance_by_segment: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Limitations
    known_limitations: List[str] = field(default_factory=list)
    out_of_scope_use_cases: List[str] = field(default_factory=list)

    # Ethical considerations
    fairness_assessment: str = ""
    bias_mitigation: str = ""

    # Maintenance
    update_frequency: str = ""
    responsible_party: str = ""
    contact: str = ""

    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Model registry for managing ML model versions and deployment."""

    def __init__(self, registry_dir: str = "./model_registry"):
        """Initialize model registry.

        Args:
            registry_dir: Directory to store registry data and models
        """
        self.registry_dir = registry_dir
        self.models_dir = os.path.join(registry_dir, "models")
        self.metadata_dir = os.path.join(registry_dir, "metadata")
        self.cards_dir = os.path.join(registry_dir, "model_cards")

        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.cards_dir, exist_ok=True)

        # In-memory cache
        self.metadata_cache: Dict[str, ModelMetadata] = {}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        for filename in os.listdir(self.metadata_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.metadata_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    metadata = ModelMetadata(
                        **data,
                        stage=ModelStage(data["stage"]),
                        deployment_strategy=DeploymentStrategy(data.get("deployment_strategy", "direct"))
                    )
                    key = f"{metadata.model_name}:{metadata.model_version}"
                    self.metadata_cache[key] = metadata

    def register_model(
        self,
        model_file_path: str,
        metadata: ModelMetadata,
        model_card: Optional[ModelCard] = None
    ) -> str:
        """Register a new model version.

        Args:
            model_file_path: Path to model file (pickle, ONNX, etc.)
            metadata: Model metadata
            model_card: Optional model card for documentation

        Returns:
            Model key (name:version)
        """
        # Calculate file hash
        with open(model_file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()

        metadata.model_file_hash = file_hash

        # Copy model file to registry
        dest_filename = f"{metadata.model_name}_{metadata.model_version}.pkl"
        dest_path = os.path.join(self.models_dir, dest_filename)
        shutil.copy(model_file_path, dest_path)
        metadata.model_file_path = dest_path

        # Save metadata
        key = f"{metadata.model_name}:{metadata.model_version}"
        self.metadata_cache[key] = metadata
        self._save_metadata(metadata)

        # Save model card if provided
        if model_card:
            self._save_model_card(model_card)

        return key

    def _save_metadata(self, metadata: ModelMetadata):
        """Save metadata to disk."""
        filename = f"{metadata.model_name}_{metadata.model_version}.json"
        filepath = os.path.join(self.metadata_dir, filename)

        # Convert to dict for JSON serialization
        data = asdict(metadata)
        data["stage"] = metadata.stage.value
        data["deployment_strategy"] = metadata.deployment_strategy.value

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _save_model_card(self, card: ModelCard):
        """Save model card to disk."""
        filename = f"{card.model_name}_{card.model_version}_card.json"
        filepath = os.path.join(self.cards_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(asdict(card), f, indent=2)

    def get_model(self, model_name: str, version: Optional[str] = None, stage: Optional[ModelStage] = None) -> Optional[ModelMetadata]:
        """Get model metadata.

        Args:
            model_name: Model name
            version: Specific version (if None, gets latest in stage)
            stage: Model stage filter

        Returns:
            ModelMetadata if found
        """
        if version:
            key = f"{model_name}:{version}"
            return self.metadata_cache.get(key)

        # Get latest version in stage
        candidates = [
            m for m in self.metadata_cache.values()
            if m.model_name == model_name and (stage is None or m.stage == stage)
        ]

        if not candidates:
            return None

        # Sort by version (assuming semantic versioning)
        candidates.sort(key=lambda m: m.model_version, reverse=True)
        return candidates[0]

    def promote_model(self, model_name: str, version: str, target_stage: ModelStage):
        """Promote model to a new stage.

        Args:
            model_name: Model name
            version: Model version
            target_stage: Target stage
        """
        key = f"{model_name}:{version}"
        metadata = self.metadata_cache.get(key)

        if not metadata:
            raise ValueError(f"Model {key} not found")

        metadata.stage = target_stage
        self._save_metadata(metadata)

    def deploy_model(
        self,
        model_name: str,
        version: str,
        strategy: DeploymentStrategy = DeploymentStrategy.DIRECT,
        deployed_by: str = "system"
    ):
        """Deploy a model to production.

        Args:
            model_name: Model name
            version: Model version
            strategy: Deployment strategy
            deployed_by: User/system deploying the model
        """
        key = f"{model_name}:{version}"
        metadata = self.metadata_cache.get(key)

        if not metadata:
            raise ValueError(f"Model {key} not found")

        metadata.deployment_strategy = strategy
        metadata.deployed_date = datetime.now().isoformat()
        metadata.deployed_by = deployed_by
        metadata.stage = ModelStage.PRODUCTION

        self._save_metadata(metadata)

    def list_models(
        self,
        model_type: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> List[ModelMetadata]:
        """List models in registry.

        Args:
            model_type: Filter by model type
            stage: Filter by stage

        Returns:
            List of ModelMetadata
        """
        models = list(self.metadata_cache.values())

        if model_type:
            models = [m for m in models if m.model_type == model_type]

        if stage:
            models = [m for m in models if m.stage == stage]

        return models

    def get_model_card(self, model_name: str, version: str) -> Optional[ModelCard]:
        """Get model card for documentation.

        Args:
            model_name: Model name
            version: Model version

        Returns:
            ModelCard if exists
        """
        filename = f"{model_name}_{version}_card.json"
        filepath = os.path.join(self.cards_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            data = json.load(f)
            return ModelCard(**data)

    def archive_model(self, model_name: str, version: str):
        """Archive an old model version.

        Args:
            model_name: Model name
            version: Model version
        """
        self.promote_model(model_name, version, ModelStage.ARCHIVED)

    def compare_models(
        self,
        model1_name: str,
        model1_version: str,
        model2_name: str,
        model2_version: str
    ) -> Dict:
        """Compare two models.

        Args:
            model1_name: First model name
            model1_version: First model version
            model2_name: Second model name
            model2_version: Second model version

        Returns:
            Comparison dictionary
        """
        m1 = self.get_model(model1_name, model1_version)
        m2 = self.get_model(model2_name, model2_version)

        if not m1 or not m2:
            raise ValueError("One or both models not found")

        comparison = {
            "model1": {
                "name": m1.model_name,
                "version": m1.model_version,
                "r2_score": m1.validation_r2_score,
                "rmse": m1.rmse,
                "stage": m1.stage.value,
            },
            "model2": {
                "name": m2.model_name,
                "version": m2.model_version,
                "r2_score": m2.validation_r2_score,
                "rmse": m2.rmse,
                "stage": m2.stage.value,
            },
            "winner_by_r2": m1.model_name if m1.validation_r2_score > m2.validation_r2_score else m2.model_name,
            "winner_by_rmse": m1.model_name if m1.rmse < m2.rmse else m2.model_name,
        }

        return comparison


# Export
__all__ = [
    "ModelRegistry",
    "ModelMetadata",
    "ModelCard",
    "ModelStage",
    "DeploymentStrategy",
]
