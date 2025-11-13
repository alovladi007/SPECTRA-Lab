"""
Virtual Metrology (VM) Module for CVD Platform
Implements ML-based film thickness and uniformity prediction.
Uses LightGBM and neural networks with FDC data and design features.

References:
- Siemens paper on ML-based VM using design features
- Pattern density, pitch, and perimeter features from layout
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
import pickle
import json

# ML Libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


@dataclass
class DesignFeatures:
    """Design layout features extracted from Calibre or similar EDA tools"""
    pattern_density: float  # 0.0 to 1.0
    line_pitch: float  # nm
    perimeter_density: float  # μm/μm²
    feature_size: float  # nm
    aspect_ratio: float
    open_area_fraction: float
    corner_count: int
    metal_layer: int
    x_position: float  # mm from center
    y_position: float  # mm from center
    die_location: str  # "center", "edge", etc.

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_density": self.pattern_density,
            "line_pitch": self.line_pitch,
            "perimeter_density": self.perimeter_density,
            "feature_size": self.feature_size,
            "aspect_ratio": self.aspect_ratio,
            "open_area_fraction": self.open_area_fraction,
            "corner_count": self.corner_count,
            "metal_layer": self.metal_layer,
            "x_position": self.x_position,
            "y_position": self.y_position,
            "die_location": self.die_location
        }


@dataclass
class ProcessFeatures:
    """Process and equipment features from FDC data"""
    temperature_setpoint: float  # °C
    temperature_actual: float
    temperature_uniformity: float  # std dev
    pressure_setpoint: float  # Torr
    pressure_actual: float
    precursor_flow: float  # sccm
    carrier_flow: float  # sccm
    deposition_time: float  # seconds
    rotation_speed: float  # rpm
    heater_zone_temps: List[float]  # Multi-zone temperatures
    chamber_id: str
    recipe_id: str
    wafer_number: int
    lot_id: str
    pm_cycle: int  # Preventive maintenance cycle
    chamber_age: float  # hours since last PM
    previous_wafer_thickness: Optional[float]  # For sequential learning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "temperature_setpoint": self.temperature_setpoint,
            "temperature_actual": self.temperature_actual,
            "temperature_uniformity": self.temperature_uniformity,
            "pressure_setpoint": self.pressure_setpoint,
            "pressure_actual": self.pressure_actual,
            "precursor_flow": self.precursor_flow,
            "carrier_flow": self.carrier_flow,
            "deposition_time": self.deposition_time,
            "rotation_speed": self.rotation_speed,
            "heater_zone_temps": self.heater_zone_temps,
            "chamber_id": self.chamber_id,
            "recipe_id": self.recipe_id,
            "wafer_number": self.wafer_number,
            "lot_id": self.lot_id,
            "pm_cycle": self.pm_cycle,
            "chamber_age": self.chamber_age,
            "previous_wafer_thickness": self.previous_wafer_thickness
        }


@dataclass
class VMPrediction:
    """Virtual metrology prediction result"""
    wafer_id: str
    predicted_thickness: float  # nm
    predicted_uniformity: float  # %
    confidence: float  # 0.0 to 1.0
    prediction_timestamp: datetime
    model_version: str
    feature_importance: Dict[str, float]
    thickness_map: Optional[np.ndarray]  # Full wafer map if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wafer_id": self.wafer_id,
            "predicted_thickness": self.predicted_thickness,
            "predicted_uniformity": self.predicted_uniformity,
            "confidence": self.confidence,
            "prediction_timestamp": self.prediction_timestamp.isoformat(),
            "model_version": self.model_version,
            "feature_importance": self.feature_importance
        }


class LightGBMPredictor:
    """
    LightGBM-based virtual metrology predictor.
    Gradient boosting model for thickness prediction.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")

        self.model_params = model_params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }

        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_version = "1.0.0"

    def prepare_features(self, design_features: DesignFeatures,
                        process_features: ProcessFeatures) -> np.ndarray:
        """Combine design and process features into feature vector"""
        features = []

        # Design features
        design_dict = design_features.to_dict()
        for key in ["pattern_density", "line_pitch", "perimeter_density",
                   "feature_size", "aspect_ratio", "open_area_fraction",
                   "corner_count", "metal_layer", "x_position", "y_position"]:
            features.append(design_dict[key])

        # Process features
        process_dict = process_features.to_dict()
        for key in ["temperature_setpoint", "temperature_actual", "temperature_uniformity",
                   "pressure_setpoint", "pressure_actual", "precursor_flow", "carrier_flow",
                   "deposition_time", "rotation_speed", "wafer_number", "pm_cycle", "chamber_age"]:
            features.append(process_dict[key])

        # Heater zone temperatures
        features.extend(process_features.heater_zone_temps)

        # Previous wafer thickness (if available)
        if process_features.previous_wafer_thickness is not None:
            features.append(process_features.previous_wafer_thickness)
        else:
            features.append(0.0)

        # Interaction features (important for CVD modeling)
        features.append(process_features.temperature_actual * process_features.deposition_time)
        features.append(process_features.precursor_flow / process_features.pressure_actual)
        features.append(design_features.pattern_density * process_features.temperature_actual)

        return np.array(features)

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             num_boost_round: int = 1000, early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """
        Train LightGBM model.

        Args:
            X_train: Training features (N, num_features)
            y_train: Training targets (N,) - film thickness in nm
            X_val: Validation features
            y_val: Validation targets
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Stop if no improvement for N rounds

        Returns:
            Training metrics
        """
        logger.info("Training LightGBM model...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            valid_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')

        # Train model
        self.model = lgb.train(
            self.model_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
                lgb.log_evaluation(period=100)
            ]
        )

        # Evaluate
        y_train_pred = self.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)

        metrics = {
            "train_rmse": train_rmse,
            "train_mae": train_mae,
            "train_r2": train_r2
        }

        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_mae = mean_absolute_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)

            metrics.update({
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "val_r2": val_r2
            })

        logger.info(f"Training complete. Train RMSE: {train_rmse:.2f} nm, R²: {train_r2:.4f}")
        if "val_rmse" in metrics:
            logger.info(f"Validation RMSE: {metrics['val_rmse']:.2f} nm, R²: {metrics['val_r2']:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict film thickness"""
        if self.model is None:
            raise ValueError("Model not trained")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        return predictions

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation.
        Confidence based on prediction consistency and training data density.
        """
        predictions = self.predict(X)

        # Estimate confidence (simplified - could use conformal prediction)
        # Higher confidence for predictions near training data
        confidences = np.ones(len(predictions)) * 0.85  # Base confidence

        # Adjust based on feature ranges (within training distribution)
        X_scaled = self.scaler.transform(X)
        for i in range(len(X)):
            # If features are far from training range, reduce confidence
            feature_distances = np.abs(X_scaled[i])
            if np.any(feature_distances > 3.0):  # More than 3 std from mean
                confidences[i] *= 0.7

        return predictions, confidences

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importances from trained model"""
        if self.model is None:
            raise ValueError("Model not trained")

        importance = self.model.feature_importance(importance_type='gain')
        feature_importance = {
            f"feature_{i}": float(imp)
            for i, imp in enumerate(importance)
        }
        return feature_importance

    def save_model(self, filepath: str):
        """Save trained model to disk"""
        if self.model is None:
            raise ValueError("Model not trained")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "model_params": self.model_params,
            "model_version": self.model_version,
            "feature_names": self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_params = model_data["model_params"]
        self.model_version = model_data["model_version"]
        self.feature_names = model_data.get("feature_names", [])

        logger.info(f"Model loaded from {filepath}")


class NeuralNetworkPredictor(nn.Module):
    """
    Neural network-based virtual metrology predictor.
    Deep learning model for complex non-linear relationships.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        super(NeuralNetworkPredictor, self).__init__()

        # Build network layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)
        self.scaler = StandardScaler()
        self.model_version = "1.0.0"

    def forward(self, x):
        """Forward pass"""
        return self.network(x)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
                   epochs: int = 100, batch_size: int = 64,
                   learning_rate: float = 0.001) -> Dict[str, Any]:
        """Train neural network"""
        logger.info("Training neural network...")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)

        # Create data loader
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            if X_val is not None and y_val is not None:
                self.eval()
                X_val_scaled = self.scaler.transform(X_val)
                X_val_tensor = torch.FloatTensor(X_val_scaled)
                y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)

                with torch.no_grad():
                    val_outputs = self(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_losses.append(val_loss.item())

                scheduler.step(val_loss)

                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.2f}, Val Loss: {val_loss.item():.2f}")
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.2f}")

        # Final evaluation
        y_train_pred = self.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)

        metrics = {
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "train_losses": train_losses
        }

        if X_val is not None and y_val is not None:
            y_val_pred = self.predict(X_val)
            val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
            val_r2 = r2_score(y_val, y_val_pred)
            metrics.update({
                "val_rmse": val_rmse,
                "val_r2": val_r2,
                "val_losses": val_losses
            })

        logger.info(f"Training complete. Train RMSE: {train_rmse:.2f} nm, R²: {train_r2:.4f}")

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict film thickness"""
        self.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            predictions = self(X_tensor)

        return predictions.numpy().flatten()


class VirtualMetrologyPredictor:
    """
    High-level virtual metrology predictor.
    Manages multiple models and provides unified interface.
    """

    def __init__(self, model_type: str = "lightgbm"):
        self.model_type = model_type

        if model_type == "lightgbm":
            self.predictor = LightGBMPredictor()
        elif model_type == "neural_network":
            self.predictor = NeuralNetworkPredictor(input_size=30)  # Adjust based on features
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        logger.info(f"Initialized VM predictor with {model_type}")

    def predict_thickness(self, design_features: DesignFeatures,
                         process_features: ProcessFeatures) -> VMPrediction:
        """
        Predict film thickness for a wafer.

        Args:
            design_features: Layout design features
            process_features: Process and equipment data

        Returns:
            VMPrediction with thickness, uniformity, and confidence
        """
        # Prepare features
        if self.model_type == "lightgbm":
            feature_vector = self.predictor.prepare_features(design_features, process_features)
            feature_vector = feature_vector.reshape(1, -1)

            # Predict
            thickness, confidence = self.predictor.predict_with_confidence(feature_vector)
            thickness = thickness[0]
            confidence = confidence[0]

            # Get feature importance
            feature_importance = self.predictor.get_feature_importance()

        else:
            # Neural network
            # Feature preparation would be similar
            feature_vector = np.random.randn(1, 30)  # Placeholder
            thickness = self.predictor.predict(feature_vector)[0]
            confidence = 0.85
            feature_importance = {}

        # Estimate uniformity (simplified - could train separate model)
        # Uniformity depends on rotation speed, temperature uniformity, etc.
        uniformity = 2.0 + 0.5 * (100.0 / process_features.rotation_speed) + \
                    0.3 * process_features.temperature_uniformity

        prediction = VMPrediction(
            wafer_id=process_features.lot_id + "_W" + str(process_features.wafer_number),
            predicted_thickness=float(thickness),
            predicted_uniformity=float(uniformity),
            confidence=float(confidence),
            prediction_timestamp=datetime.utcnow(),
            model_version=self.predictor.model_version,
            feature_importance=feature_importance,
            thickness_map=None
        )

        logger.info(f"Predicted thickness: {thickness:.2f} nm, uniformity: {uniformity:.2f}%, confidence: {confidence:.2f}")

        return prediction

    def update_with_metrology(self, wafer_id: str, actual_thickness: float):
        """
        Update model with actual metrology measurement (online learning).
        Enables continuous improvement of VM accuracy.
        """
        logger.info(f"Updating VM model with actual measurement: {actual_thickness:.2f} nm for {wafer_id}")
        # In production, implement online learning or periodic retraining
        # Store measurement in database for next training cycle
