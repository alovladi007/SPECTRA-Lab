"""Feature Store for managing and serving ML features.

Provides feature engineering, storage, versioning, and serving for
Virtual Metrology models with online and offline access patterns.
"""

import json
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
import pickle


# ============================================================================
# Feature Store Data Structures
# ============================================================================

@dataclass
class FeatureDefinition:
    """Definition of a feature."""
    feature_name: str
    feature_type: str  # "int", "float", "string", "categorical"
    description: str
    source: str  # Where feature originates (e.g., "ion_telemetry", "rtp_recipe")

    # Value constraints
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None

    # Metadata
    version: str = "v1"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    owner: str = ""


@dataclass
class FeatureGroup:
    """Group of related features."""
    group_name: str
    features: List[FeatureDefinition]
    description: str
    primary_keys: List[str]  # Keys to join on (e.g., ["run_id", "wafer_id"])

    version: str = "v1"
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class FeatureVector:
    """Feature vector for a specific entity (run, wafer, etc.)."""
    entity_id: str  # e.g., run_id or wafer_id
    timestamp: float
    features: Dict[str, Any]
    feature_group: str


# ============================================================================
# Feature Store
# ============================================================================

class FeatureStore:
    """Feature store for ML feature management.

    Provides:
    - Feature registration and versioning
    - Online serving (real-time inference)
    - Offline serving (batch training)
    - Feature engineering pipelines
    - Feature validation
    """

    def __init__(self, store_dir: str = "./feature_store"):
        """Initialize feature store.

        Args:
            store_dir: Directory for feature store data
        """
        self.store_dir = store_dir
        self.definitions_dir = os.path.join(store_dir, "definitions")
        self.online_dir = os.path.join(store_dir, "online")  # Latest features for serving
        self.offline_dir = os.path.join(store_dir, "offline")  # Historical features for training

        # Create directories
        os.makedirs(self.definitions_dir, exist_ok=True)
        os.makedirs(self.online_dir, exist_ok=True)
        os.makedirs(self.offline_dir, exist_ok=True)

        # In-memory cache
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.online_cache: Dict[str, Dict] = {}  # entity_id -> features

        self._load_definitions()

    def _load_definitions(self):
        """Load feature definitions from disk."""
        for filename in os.listdir(self.definitions_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.definitions_dir, filename)
                with open(filepath, 'r') as f:
                    data = json.load(f)

                    # Reconstruct FeatureDefinition objects
                    features = [
                        FeatureDefinition(**fd) for fd in data["features"]
                    ]

                    group = FeatureGroup(
                        group_name=data["group_name"],
                        features=features,
                        description=data["description"],
                        primary_keys=data["primary_keys"],
                        version=data.get("version", "v1"),
                        created_date=data.get("created_date", "")
                    )

                    self.feature_groups[group.group_name] = group

    def register_feature_group(self, feature_group: FeatureGroup):
        """Register a new feature group.

        Args:
            feature_group: FeatureGroup to register
        """
        self.feature_groups[feature_group.group_name] = feature_group

        # Save to disk
        filepath = os.path.join(self.definitions_dir, f"{feature_group.group_name}.json")

        data = {
            "group_name": feature_group.group_name,
            "features": [asdict(f) for f in feature_group.features],
            "description": feature_group.description,
            "primary_keys": feature_group.primary_keys,
            "version": feature_group.version,
            "created_date": feature_group.created_date,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def write_online(self, entity_id: str, features: Dict[str, Any], feature_group: str):
        """Write features to online store (for real-time serving).

        Args:
            entity_id: Entity identifier (e.g., run_id)
            features: Feature dictionary
            feature_group: Feature group name
        """
        # Validate features
        if feature_group not in self.feature_groups:
            raise ValueError(f"Feature group '{feature_group}' not registered")

        self._validate_features(features, feature_group)

        # Update online cache
        cache_key = f"{feature_group}:{entity_id}"
        self.online_cache[cache_key] = {
            "entity_id": entity_id,
            "timestamp": datetime.now().isoformat(),
            "features": features,
            "feature_group": feature_group
        }

        # Persist to disk (simple JSON for now; could use Redis/DB in production)
        online_file = os.path.join(self.online_dir, f"{cache_key}.json")
        with open(online_file, 'w') as f:
            json.dump(self.online_cache[cache_key], f)

    def read_online(self, entity_id: str, feature_group: str) -> Optional[Dict[str, Any]]:
        """Read features from online store.

        Args:
            entity_id: Entity identifier
            feature_group: Feature group name

        Returns:
            Feature dictionary if found
        """
        cache_key = f"{feature_group}:{entity_id}"

        # Check cache first
        if cache_key in self.online_cache:
            return self.online_cache[cache_key]["features"]

        # Try loading from disk
        online_file = os.path.join(self.online_dir, f"{cache_key}.json")
        if os.path.exists(online_file):
            with open(online_file, 'r') as f:
                data = json.load(f)
                self.online_cache[cache_key] = data
                return data["features"]

        return None

    def write_offline(self, feature_vectors: List[FeatureVector], feature_group: str):
        """Write features to offline store (for training).

        Args:
            feature_vectors: List of feature vectors
            feature_group: Feature group name
        """
        # Validate
        if feature_group not in self.feature_groups:
            raise ValueError(f"Feature group '{feature_group}' not registered")

        # Append to offline store (could use Parquet/Arrow in production)
        offline_file = os.path.join(self.offline_dir, f"{feature_group}_offline.pkl")

        # Load existing data if any
        if os.path.exists(offline_file):
            with open(offline_file, 'rb') as f:
                existing_data = pickle.load(f)
        else:
            existing_data = []

        # Append new vectors
        existing_data.extend(feature_vectors)

        # Save
        with open(offline_file, 'wb') as f:
            pickle.dump(existing_data, f)

    def read_offline(
        self,
        feature_group: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[FeatureVector]:
        """Read features from offline store for training.

        Args:
            feature_group: Feature group name
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            List of FeatureVector objects
        """
        offline_file = os.path.join(self.offline_dir, f"{feature_group}_offline.pkl")

        if not os.path.exists(offline_file):
            return []

        with open(offline_file, 'rb') as f:
            all_vectors = pickle.load(f)

        # Filter by date if specified
        if start_date or end_date:
            filtered = []
            for vec in all_vectors:
                vec_date = datetime.fromtimestamp(vec.timestamp).isoformat()

                if start_date and vec_date < start_date:
                    continue
                if end_date and vec_date > end_date:
                    continue

                filtered.append(vec)

            return filtered

        return all_vectors

    def _validate_features(self, features: Dict[str, Any], feature_group: str):
        """Validate feature values against definitions.

        Args:
            features: Feature dictionary
            feature_group: Feature group name
        """
        group = self.feature_groups[feature_group]

        for feature_def in group.features:
            fname = feature_def.feature_name

            if fname not in features:
                # Optional feature
                continue

            value = features[fname]

            # Type check
            if feature_def.feature_type == "float":
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Feature '{fname}' must be float, got {type(value)}")

            # Range check
            if feature_def.min_value is not None and value < feature_def.min_value:
                raise ValueError(f"Feature '{fname}' below min: {value} < {feature_def.min_value}")

            if feature_def.max_value is not None and value > feature_def.max_value:
                raise ValueError(f"Feature '{fname}' above max: {value} > {feature_def.max_value}")

            # Allowed values check
            if feature_def.allowed_values and value not in feature_def.allowed_values:
                raise ValueError(f"Feature '{fname}' not in allowed values: {value}")

    def get_feature_group(self, group_name: str) -> Optional[FeatureGroup]:
        """Get feature group definition.

        Args:
            group_name: Feature group name

        Returns:
            FeatureGroup if found
        """
        return self.feature_groups.get(group_name)

    def list_feature_groups(self) -> List[str]:
        """List all registered feature groups.

        Returns:
            List of feature group names
        """
        return list(self.feature_groups.keys())

    def get_training_dataset(
        self,
        feature_groups: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, List[FeatureVector]]:
        """Get training dataset from offline store.

        Args:
            feature_groups: List of feature group names
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dictionary mapping feature groups to feature vectors
        """
        dataset = {}

        for group_name in feature_groups:
            vectors = self.read_offline(group_name, start_date, end_date)
            dataset[group_name] = vectors

        return dataset


# ============================================================================
# Predefined Feature Groups for Ion and RTP
# ============================================================================

def create_ion_implant_feature_group() -> FeatureGroup:
    """Create feature group for Ion Implant VM."""
    features = [
        FeatureDefinition(
            feature_name="ion_species_code",
            feature_type="int",
            description="Encoded ion species",
            source="ion_recipe",
            min_value=0,
            max_value=10
        ),
        FeatureDefinition(
            feature_name="energy_keV",
            feature_type="float",
            description="Ion energy in keV",
            source="ion_recipe",
            min_value=1.0,
            max_value=200.0
        ),
        FeatureDefinition(
            feature_name="dose_cm2",
            feature_type="float",
            description="Implant dose in ions/cm²",
            source="ion_recipe",
            min_value=1e12,
            max_value=1e17
        ),
        FeatureDefinition(
            feature_name="tilt_angle_deg",
            feature_type="float",
            description="Wafer tilt angle in degrees",
            source="ion_recipe",
            min_value=0.0,
            max_value=15.0
        ),
        FeatureDefinition(
            feature_name="beam_current_mA",
            feature_type="float",
            description="Average beam current in mA",
            source="ion_telemetry",
            min_value=0.1,
            max_value=50.0
        ),
        FeatureDefinition(
            feature_name="dose_uniformity_pct",
            feature_type="float",
            description="Dose uniformity percentage",
            source="ion_telemetry",
            min_value=80.0,
            max_value=100.0
        ),
        FeatureDefinition(
            feature_name="rtp_thermal_budget",
            feature_type="float",
            description="RTP thermal budget integral",
            source="rtp_telemetry",
            min_value=0.0
        ),
    ]

    return FeatureGroup(
        group_name="ion_implant_features",
        features=features,
        description="Features for Ion Implant Virtual Metrology",
        primary_keys=["run_id", "wafer_id"]
    )


def create_rtp_feature_group() -> FeatureGroup:
    """Create feature group for RTP VM."""
    features = [
        FeatureDefinition(
            feature_name="peak_temp_C",
            feature_type="float",
            description="Peak temperature in Celsius",
            source="rtp_recipe",
            min_value=400.0,
            max_value=1200.0
        ),
        FeatureDefinition(
            feature_name="dwell_time_s",
            feature_type="float",
            description="Dwell time in seconds",
            source="rtp_recipe",
            min_value=0.0,
            max_value=600.0
        ),
        FeatureDefinition(
            feature_name="ramp_rate_C_per_s",
            feature_type="float",
            description="Ramp rate in °C/s",
            source="rtp_recipe",
            min_value=1.0,
            max_value=200.0
        ),
        FeatureDefinition(
            feature_name="thermal_budget",
            feature_type="float",
            description="Integrated thermal budget",
            source="rtp_telemetry",
            min_value=0.0
        ),
        FeatureDefinition(
            feature_name="avg_tracking_error_C",
            feature_type="float",
            description="Average tracking error in °C",
            source="rtp_telemetry",
            min_value=0.0,
            max_value=50.0
        ),
        FeatureDefinition(
            feature_name="max_overshoot_C",
            feature_type="float",
            description="Maximum overshoot in °C",
            source="rtp_telemetry",
            min_value=0.0,
            max_value=100.0
        ),
    ]

    return FeatureGroup(
        group_name="rtp_features",
        features=features,
        description="Features for RTP Virtual Metrology",
        primary_keys=["run_id", "wafer_id"]
    )


# Export
__all__ = [
    "FeatureStore",
    "FeatureDefinition",
    "FeatureGroup",
    "FeatureVector",
    "create_ion_implant_feature_group",
    "create_rtp_feature_group",
]
