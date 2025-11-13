"""
AI/ML Analytics Engine for Anomaly Detection
Implements sophisticated anomaly detection algorithms for CVD processes:
- Isolation Forest for outlier detection
- Autoencoder for multivariate anomaly detection
- LSTM for time-series anomalies
- Predictive maintenance algorithms
- Root cause analysis
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from enum import Enum
import logging

# ML Libraries
try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""
    POINT_ANOMALY = "point_anomaly"  # Single outlier point
    CONTEXTUAL_ANOMALY = "contextual_anomaly"  # Anomalous in specific context
    COLLECTIVE_ANOMALY = "collective_anomaly"  # Sequence of anomalous points
    DRIFT = "drift"  # Gradual process drift
    SHIFT = "shift"  # Sudden process shift


class AnomalySeverity(Enum):
    """Anomaly severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: datetime
    affected_parameters: List[str]
    anomaly_score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    root_cause: Optional[str]
    recommended_action: str
    predicted_impact: Dict[str, float]


@dataclass
class MaintenancePrediction:
    """Predictive maintenance result"""
    equipment_id: str
    component: str
    predicted_failure_time: datetime
    confidence: float
    current_health_score: float  # 0.0 (failed) to 1.0 (healthy)
    remaining_useful_life: float  # hours
    recommended_maintenance: str
    risk_level: AnomalySeverity


class IsolationForestDetector:
    """
    Isolation Forest for outlier detection.
    Efficient for high-dimensional data.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 100):
        """
        Args:
            contamination: Expected proportion of outliers (default 5%)
            n_estimators: Number of trees in forest
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not installed")

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False

        logger.info(f"Initialized Isolation Forest detector with contamination={contamination}")

    def fit(self, X: np.ndarray):
        """
        Train Isolation Forest on normal data.

        Args:
            X: Training data (N, num_features)
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

        logger.info(f"Isolation Forest trained on {len(X)} samples")

    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in data.

        Args:
            X: Data to check (N, num_features)

        Returns:
            (predictions, anomaly_scores)
            predictions: -1 for anomaly, 1 for normal
            anomaly_scores: Lower scores indicate anomalies
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)

        # Convert to 0-1 range (higher = more anomalous)
        anomaly_scores_normalized = 1 / (1 + np.exp(anomaly_scores))

        logger.info(f"Detected {np.sum(predictions == -1)} anomalies in {len(X)} samples")

        return predictions, anomaly_scores_normalized


class AutoencoderDetector(nn.Module):
    """
    Autoencoder-based anomaly detector.
    Learns to reconstruct normal patterns; high reconstruction error indicates anomaly.
    """

    def __init__(self, input_dim: int, encoding_dim: int = 32):
        """
        Args:
            input_dim: Input feature dimension
            encoding_dim: Latent space dimension
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        super(AutoencoderDetector, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim)
        )

        self.scaler = StandardScaler()
        self.threshold = 0.0
        self.is_fitted = False

        logger.info(f"Initialized Autoencoder detector: {input_dim} -> {encoding_dim} -> {input_dim}")

    def forward(self, x):
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 64,
           learning_rate: float = 0.001):
        """
        Train autoencoder on normal data.

        Args:
            X: Training data (N, num_features) - should be normal samples only
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        logger.info("Training autoencoder...")

        # Scale data
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        # Data loader
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch in dataloader:
                batch_X = batch[0]

                optimizer.zero_grad()
                reconstructed = self(batch_X)
                loss = criterion(reconstructed, batch_X)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Calculate threshold from training data
        self.eval()
        with torch.no_grad():
            reconstructed = self(X_tensor)
            reconstruction_errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()
            # Set threshold at 95th percentile
            self.threshold = np.percentile(reconstruction_errors, 95)

        self.is_fitted = True
        logger.info(f"Training complete. Anomaly threshold: {self.threshold:.6f}")

    def detect(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on reconstruction error.

        Args:
            X: Data to check (N, num_features)

        Returns:
            (predictions, anomaly_scores)
            predictions: 1 for anomaly, 0 for normal
            anomaly_scores: Reconstruction errors
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        self.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled)

        with torch.no_grad():
            reconstructed = self(X_tensor)
            reconstruction_errors = torch.mean((reconstructed - X_tensor) ** 2, dim=1).numpy()

        # Classify as anomaly if error > threshold
        predictions = (reconstruction_errors > self.threshold).astype(int)

        # Normalize scores to 0-1 range
        anomaly_scores = np.clip(reconstruction_errors / (self.threshold * 2), 0, 1)

        logger.info(f"Detected {np.sum(predictions)} anomalies in {len(X)} samples")

        return predictions, anomaly_scores


class LSTMAnomalyDetector(nn.Module):
    """
    LSTM-based time-series anomaly detector.
    Predicts next time step; large prediction error indicates anomaly.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        """
        Args:
            input_dim: Number of features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
        """
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not installed")

        super(LSTMAnomalyDetector, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, input_dim)

        self.scaler = StandardScaler()
        self.threshold = 0.0
        self.is_fitted = False

        logger.info(f"Initialized LSTM detector: input={input_dim}, hidden={hidden_dim}, layers={num_layers}")

    def forward(self, x):
        """Forward pass"""
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        predictions = self.fc(lstm_out[:, -1, :])
        return predictions

    def fit(self, X: np.ndarray, sequence_length: int = 10,
           epochs: int = 50, batch_size: int = 32):
        """
        Train LSTM on time-series data.

        Args:
            X: Time-series data (N, num_features)
            sequence_length: Length of input sequences
            epochs: Number of training epochs
            batch_size: Batch size
        """
        logger.info("Training LSTM...")

        # Scale data
        X_scaled = self.scaler.fit_transform(X)

        # Create sequences
        sequences = []
        targets = []
        for i in range(len(X_scaled) - sequence_length):
            sequences.append(X_scaled[i:i+sequence_length])
            targets.append(X_scaled[i+sequence_length])

        sequences = np.array(sequences)
        targets = np.array(targets)

        # Convert to tensors
        X_tensor = torch.FloatTensor(sequences)
        y_tensor = torch.FloatTensor(targets)

        # Data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        # Training loop
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                predictions = self(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / len(dataloader)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        # Calculate threshold
        self.eval()
        with torch.no_grad():
            predictions = self(X_tensor)
            errors = torch.mean((predictions - y_tensor) ** 2, dim=1).numpy()
            self.threshold = np.percentile(errors, 95)

        self.is_fitted = True
        logger.info(f"Training complete. Anomaly threshold: {self.threshold:.6f}")


class PredictiveMaintenanceEngine:
    """
    Predictive maintenance engine for CVD equipment.
    Predicts component failures based on degradation patterns.
    """

    def __init__(self):
        self.component_health: Dict[str, float] = {}
        self.degradation_rates: Dict[str, float] = {}
        self.maintenance_history: List[Dict[str, Any]] = []

        logger.info("Initialized predictive maintenance engine")

    def calculate_component_health(self,
                                   equipment_id: str,
                                   component: str,
                                   sensor_data: Dict[str, float],
                                   operating_hours: float) -> float:
        """
        Calculate component health score (0-1).

        Args:
            equipment_id: Equipment identifier
            component: Component name (e.g., "heater", "pump", "valve")
            sensor_data: Current sensor readings
            operating_hours: Hours since last maintenance

        Returns:
            Health score (1.0 = perfect, 0.0 = failed)
        """
        # Simplified health model - in production, use ML model trained on failure data

        base_health = 1.0

        # Degradation based on operating hours
        # Typical CVD component lifetimes:
        # - Heaters: 2000-5000 hours
        # - Pumps: 10000-20000 hours
        # - Valves: 5000-10000 hours
        # - Chamber parts: 1000-3000 hours (depends on process)

        lifetime_hours = {
            "heater": 3000,
            "pump": 15000,
            "valve": 7500,
            "chamber": 2000,
            "showerhead": 2500
        }

        expected_lifetime = lifetime_hours.get(component, 5000)

        # Linear degradation model (simplified)
        age_factor = 1.0 - (operating_hours / expected_lifetime)
        age_factor = max(0.0, min(1.0, age_factor))

        # Sensor-based health indicators
        health_factors = [age_factor]

        # Temperature-related degradation
        if "temperature" in sensor_data:
            temp = sensor_data["temperature"]
            if component == "heater":
                # Heaters degrade faster at high temperature
                if temp > 900:  # High temperature operation
                    temp_factor = 0.95
                else:
                    temp_factor = 1.0
                health_factors.append(temp_factor)

        # Pressure-related indicators
        if "pressure" in sensor_data:
            pressure = sensor_data["pressure"]
            if component == "pump":
                # Pump health degrades if pressure unstable
                pressure_variance = sensor_data.get("pressure_variance", 0)
                if pressure_variance > 1.0:  # High variance
                    pressure_factor = 0.90
                else:
                    pressure_factor = 1.0
                health_factors.append(pressure_factor)

        # Combine factors
        health_score = np.prod(health_factors)

        # Store health score
        key = f"{equipment_id}_{component}"
        self.component_health[key] = health_score

        logger.info(f"{key} health score: {health_score:.2f}")

        return health_score

    def predict_failure(self,
                       equipment_id: str,
                       component: str,
                       current_health: float,
                       degradation_rate: float) -> MaintenancePrediction:
        """
        Predict when component will fail.

        Args:
            equipment_id: Equipment identifier
            component: Component name
            current_health: Current health score (0-1)
            degradation_rate: Health decrease per hour

        Returns:
            MaintenancePrediction
        """
        # Calculate remaining useful life
        failure_threshold = 0.2  # Fail below 20% health

        if degradation_rate > 0:
            rul_hours = (current_health - failure_threshold) / degradation_rate
            rul_hours = max(0, rul_hours)
        else:
            rul_hours = float('inf')

        # Predicted failure time
        predicted_failure = datetime.utcnow() + timedelta(hours=rul_hours)

        # Determine risk level
        if rul_hours < 100:
            risk = AnomalySeverity.CRITICAL
            maintenance_action = "Schedule immediate maintenance"
        elif rul_hours < 500:
            risk = AnomalySeverity.HIGH
            maintenance_action = "Schedule maintenance within 1 week"
        elif rul_hours < 1000:
            risk = AnomalySeverity.MEDIUM
            maintenance_action = "Schedule maintenance within 1 month"
        else:
            risk = AnomalySeverity.LOW
            maintenance_action = "Routine maintenance schedule OK"

        # Confidence based on data quality
        confidence = 0.8 if rul_hours < 2000 else 0.6

        prediction = MaintenancePrediction(
            equipment_id=equipment_id,
            component=component,
            predicted_failure_time=predicted_failure,
            confidence=confidence,
            current_health_score=current_health,
            remaining_useful_life=rul_hours,
            recommended_maintenance=maintenance_action,
            risk_level=risk
        )

        logger.info(f"{component} RUL: {rul_hours:.0f} hours, Risk: {risk.value}")

        return prediction


class RootCauseAnalyzer:
    """
    Root cause analysis engine.
    Uses correlation analysis and ML to identify root causes of anomalies.
    """

    def __init__(self):
        self.correlation_matrix: Optional[np.ndarray] = None
        self.feature_names: List[str] = []

        logger.info("Initialized root cause analyzer")

    def fit(self, X: np.ndarray, feature_names: List[str]):
        """
        Build correlation model from historical data.

        Args:
            X: Historical data (N, num_features)
            feature_names: Names of features
        """
        self.feature_names = feature_names
        self.correlation_matrix = np.corrcoef(X, rowvar=False)

        logger.info(f"Built correlation matrix: {X.shape[1]} features")

    def analyze(self,
               anomalous_feature: str,
               current_values: Dict[str, float]) -> Tuple[str, float]:
        """
        Identify most likely root cause of anomaly.

        Args:
            anomalous_feature: Feature showing anomaly
            current_values: Current values of all features

        Returns:
            (root_cause_feature, confidence)
        """
        if self.correlation_matrix is None:
            logger.warning("Correlation matrix not built")
            return "Unknown", 0.0

        # Find index of anomalous feature
        try:
            anomaly_idx = self.feature_names.index(anomalous_feature)
        except ValueError:
            logger.error(f"Feature {anomalous_feature} not found")
            return "Unknown", 0.0

        # Get correlations with anomalous feature
        correlations = np.abs(self.correlation_matrix[anomaly_idx, :])

        # Exclude self-correlation
        correlations[anomaly_idx] = 0

        # Find most correlated feature
        root_cause_idx = np.argmax(correlations)
        root_cause_feature = self.feature_names[root_cause_idx]
        confidence = float(correlations[root_cause_idx])

        logger.info(f"Root cause of {anomalous_feature} anomaly: {root_cause_feature} (confidence: {confidence:.2f})")

        return root_cause_feature, confidence


class AnomalyDetectionEngine:
    """
    High-level anomaly detection engine.
    Combines multiple detection methods and provides unified interface.
    """

    def __init__(self):
        self.isolation_forest = None
        self.autoencoder = None
        self.lstm_detector = None
        self.predictive_maintenance = PredictiveMaintenanceEngine()
        self.root_cause_analyzer = RootCauseAnalyzer()

        self.anomaly_history: List[AnomalyDetection] = []

        logger.info("Initialized anomaly detection engine")

    def initialize_detectors(self, input_dim: int):
        """Initialize all detectors"""
        if SKLEARN_AVAILABLE:
            self.isolation_forest = IsolationForestDetector()

        if PYTORCH_AVAILABLE:
            self.autoencoder = AutoencoderDetector(input_dim=input_dim)
            self.lstm_detector = LSTMAnomalyDetector(input_dim=input_dim)

        logger.info("All detectors initialized")

    def detect_anomalies(self,
                        data: np.ndarray,
                        feature_names: List[str],
                        timestamp: datetime) -> List[AnomalyDetection]:
        """
        Detect anomalies using ensemble of methods.

        Args:
            data: Current data point (num_features,)
            feature_names: Names of features
            timestamp: Timestamp of measurement

        Returns:
            List of detected anomalies
        """
        anomalies = []

        # Reshape for single sample
        data_reshaped = data.reshape(1, -1)

        # Isolation Forest detection
        if self.isolation_forest and self.isolation_forest.is_fitted:
            predictions, scores = self.isolation_forest.detect(data_reshaped)
            if predictions[0] == -1:  # Anomaly detected
                anomaly = AnomalyDetection(
                    anomaly_id=f"IF_{timestamp.strftime('%Y%m%d%H%M%S')}",
                    anomaly_type=AnomalyType.POINT_ANOMALY,
                    severity=self._score_to_severity(scores[0]),
                    timestamp=timestamp,
                    affected_parameters=feature_names,
                    anomaly_score=float(scores[0]),
                    confidence=0.85,
                    description="Outlier detected by Isolation Forest",
                    root_cause=None,
                    recommended_action="Investigate process parameters",
                    predicted_impact={}
                )
                anomalies.append(anomaly)

        # Store anomalies
        self.anomaly_history.extend(anomalies)

        return anomalies

    def _score_to_severity(self, score: float) -> AnomalySeverity:
        """Convert anomaly score to severity level"""
        if score > 0.9:
            return AnomalySeverity.CRITICAL
        elif score > 0.7:
            return AnomalySeverity.HIGH
        elif score > 0.5:
            return AnomalySeverity.MEDIUM
        else:
            return AnomalySeverity.LOW
