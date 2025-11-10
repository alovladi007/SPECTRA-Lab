"""SPC and Virtual Metrology services for process monitoring and prediction."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import asyncio

logger = logging.getLogger(__name__)


class SPCRuleType(Enum):
    """Western Electric and Nelson rules for SPC."""
    RULE_1 = "1_point_beyond_3_sigma"
    RULE_2 = "9_points_same_side"
    RULE_3 = "6_points_trending"
    RULE_4 = "14_points_alternating"
    RULE_5 = "2_of_3_beyond_2_sigma"
    RULE_6 = "4_of_5_beyond_1_sigma"
    RULE_7 = "15_points_within_1_sigma"
    RULE_8 = "8_points_beyond_1_sigma_both_sides"


@dataclass
class ControlLimits:
    """Control limits for SPC charts."""
    UCL: float  # Upper Control Limit
    CL: float   # Center Line
    LCL: float  # Lower Control Limit
    USL: Optional[float] = None  # Upper Spec Limit
    LSL: Optional[float] = None  # Lower Spec Limit
    
    def is_in_control(self, value: float) -> bool:
        """Check if value is within control limits."""
        return self.LCL <= value <= self.UCL
        
    def is_in_spec(self, value: float) -> bool:
        """Check if value is within specification limits."""
        if self.USL is not None and value > self.USL:
            return False
        if self.LSL is not None and value < self.LSL:
            return False
        return True


class SPCCalculator:
    """Calculate SPC statistics and control limits."""
    
    @staticmethod
    def calculate_control_limits(
        data: np.ndarray,
        chart_type: str = "I-MR",
        sigma_level: float = 3.0
    ) -> ControlLimits:
        """Calculate control limits based on chart type."""
        
        if chart_type == "I-MR":
            # Individual and Moving Range chart
            mean = np.mean(data)
            mr = np.abs(np.diff(data))
            mr_bar = np.mean(mr)
            
            # Estimate sigma using moving range
            d2 = 1.128  # for n=2
            sigma_est = mr_bar / d2
            
            return ControlLimits(
                UCL=mean + sigma_level * sigma_est,
                CL=mean,
                LCL=mean - sigma_level * sigma_est
            )
            
        elif chart_type == "Xbar-R":
            # X-bar and Range chart (assumes subgroups)
            # data should be 2D array (samples x subgroup_size)
            if len(data.shape) == 1:
                data = data.reshape(-1, 5)  # Default subgroup size of 5
                
            xbar = np.mean(data, axis=1)
            R = np.ptp(data, axis=1)  # Range of each subgroup
            
            xbar_bar = np.mean(xbar)
            R_bar = np.mean(R)
            
            # Control chart constants
            n = data.shape[1]
            A2 = {2: 1.88, 3: 1.023, 4: 0.729, 5: 0.577}.get(n, 0.577)
            
            return ControlLimits(
                UCL=xbar_bar + A2 * R_bar,
                CL=xbar_bar,
                LCL=xbar_bar - A2 * R_bar
            )
            
        elif chart_type == "EWMA":
            # Exponentially Weighted Moving Average
            lambda_val = 0.2  # EWMA parameter
            mean = np.mean(data)
            std = np.std(data)
            
            # EWMA control limits (simplified)
            L = 3  # Control limit width
            sigma_ewma = std * np.sqrt(lambda_val / (2 - lambda_val))
            
            return ControlLimits(
                UCL=mean + L * sigma_ewma,
                CL=mean,
                LCL=mean - L * sigma_ewma
            )
            
        elif chart_type == "CUSUM":
            # Cumulative Sum chart
            mean = np.mean(data)
            std = np.std(data)
            
            # CUSUM parameters
            k = 0.5 * std  # Reference value
            h = 5 * std    # Decision interval
            
            return ControlLimits(
                UCL=h,
                CL=0,
                LCL=-h
            )
            
        else:
            # Default to simple mean +/- 3 sigma
            mean = np.mean(data)
            std = np.std(data)
            
            return ControlLimits(
                UCL=mean + sigma_level * std,
                CL=mean,
                LCL=mean - sigma_level * std
            )
            
    @staticmethod
    def calculate_ewma(data: np.ndarray, lambda_val: float = 0.2) -> np.ndarray:
        """Calculate EWMA values."""
        ewma = np.zeros_like(data)
        ewma[0] = data[0]
        
        for i in range(1, len(data)):
            ewma[i] = lambda_val * data[i] + (1 - lambda_val) * ewma[i-1]
            
        return ewma
        
    @staticmethod
    def calculate_cusum(data: np.ndarray, target: float, k: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate CUSUM values (positive and negative)."""
        cusum_pos = np.zeros_like(data)
        cusum_neg = np.zeros_like(data)
        
        for i in range(1, len(data)):
            cusum_pos[i] = max(0, data[i] - target - k + cusum_pos[i-1])
            cusum_neg[i] = max(0, target - k - data[i] + cusum_neg[i-1])
            
        return cusum_pos, cusum_neg
        
    @staticmethod
    def calculate_capability_indices(
        data: np.ndarray,
        USL: float,
        LSL: float
    ) -> Dict[str, float]:
        """Calculate process capability indices."""
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        
        # Cp - Process Capability
        Cp = (USL - LSL) / (6 * std)
        
        # Cpk - Process Capability Index
        Cpu = (USL - mean) / (3 * std)
        Cpl = (mean - LSL) / (3 * std)
        Cpk = min(Cpu, Cpl)
        
        # Pp and Ppk (using overall variation)
        Pp = (USL - LSL) / (6 * std)
        Ppu = (USL - mean) / (3 * std)
        Ppl = (mean - LSL) / (3 * std)
        Ppk = min(Ppu, Ppl)
        
        return {
            'Cp': Cp,
            'Cpk': Cpk,
            'Pp': Pp,
            'Ppk': Ppk,
            'mean': mean,
            'std': std
        }


class SPCRuleChecker:
    """Check for SPC rule violations."""
    
    def __init__(self, control_limits: ControlLimits):
        self.limits = control_limits
        self.sigma = (self.limits.UCL - self.limits.CL) / 3
        
    def check_rules(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Check all Western Electric rules."""
        violations = []
        
        # Rule 1: 1 point beyond 3 sigma
        rule1_violations = self._check_rule_1(data)
        if rule1_violations:
            violations.extend(rule1_violations)
            
        # Rule 2: 9 points on same side of center
        rule2_violations = self._check_rule_2(data)
        if rule2_violations:
            violations.extend(rule2_violations)
            
        # Rule 3: 6 points trending
        rule3_violations = self._check_rule_3(data)
        if rule3_violations:
            violations.extend(rule3_violations)
            
        # Rule 5: 2 of 3 points beyond 2 sigma
        rule5_violations = self._check_rule_5(data)
        if rule5_violations:
            violations.extend(rule5_violations)
            
        # Rule 6: 4 of 5 points beyond 1 sigma
        rule6_violations = self._check_rule_6(data)
        if rule6_violations:
            violations.extend(rule6_violations)
            
        return violations
        
    def _check_rule_1(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Check for points beyond 3 sigma."""
        violations = []
        
        for i, value in enumerate(data):
            if value > self.limits.UCL or value < self.limits.LCL:
                violations.append({
                    'rule': SPCRuleType.RULE_1.value,
                    'index': i,
                    'value': value,
                    'description': f'Point {i} beyond 3σ limits'
                })
                
        return violations
        
    def _check_rule_2(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Check for 9 consecutive points on same side."""
        violations = []
        
        if len(data) < 9:
            return violations
            
        for i in range(len(data) - 8):
            subset = data[i:i+9]
            if np.all(subset > self.limits.CL) or np.all(subset < self.limits.CL):
                violations.append({
                    'rule': SPCRuleType.RULE_2.value,
                    'index': i,
                    'value': subset[-1],
                    'description': f'9 consecutive points on same side starting at {i}'
                })
                
        return violations
        
    def _check_rule_3(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Check for 6 points trending."""
        violations = []
        
        if len(data) < 6:
            return violations
            
        for i in range(len(data) - 5):
            subset = data[i:i+6]
            diffs = np.diff(subset)
            
            if np.all(diffs > 0) or np.all(diffs < 0):
                violations.append({
                    'rule': SPCRuleType.RULE_3.value,
                    'index': i,
                    'value': subset[-1],
                    'description': f'6 points trending starting at {i}'
                })
                
        return violations
        
    def _check_rule_5(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Check for 2 of 3 points beyond 2 sigma."""
        violations = []
        
        if len(data) < 3:
            return violations
            
        two_sigma_upper = self.limits.CL + 2 * self.sigma
        two_sigma_lower = self.limits.CL - 2 * self.sigma
        
        for i in range(len(data) - 2):
            subset = data[i:i+3]
            beyond_upper = np.sum(subset > two_sigma_upper)
            beyond_lower = np.sum(subset < two_sigma_lower)
            
            if beyond_upper >= 2 or beyond_lower >= 2:
                violations.append({
                    'rule': SPCRuleType.RULE_5.value,
                    'index': i,
                    'value': subset[-1],
                    'description': f'2 of 3 points beyond 2σ starting at {i}'
                })
                
        return violations
        
    def _check_rule_6(self, data: np.ndarray) -> List[Dict[str, Any]]:
        """Check for 4 of 5 points beyond 1 sigma."""
        violations = []
        
        if len(data) < 5:
            return violations
            
        one_sigma_upper = self.limits.CL + self.sigma
        one_sigma_lower = self.limits.CL - self.sigma
        
        for i in range(len(data) - 4):
            subset = data[i:i+5]
            beyond_upper = np.sum(subset > one_sigma_upper)
            beyond_lower = np.sum(subset < one_sigma_lower)
            
            if beyond_upper >= 4 or beyond_lower >= 4:
                violations.append({
                    'rule': SPCRuleType.RULE_6.value,
                    'index': i,
                    'value': subset[-1],
                    'description': f'4 of 5 points beyond 1σ starting at {i}'
                })
                
        return violations


class MultivariatesSPC:
    """Multivariate Statistical Process Control using T² and PCA."""
    
    def __init__(self, reference_data: np.ndarray):
        """
        Initialize with reference (in-control) data.
        
        Args:
            reference_data: n_samples x n_features array
        """
        self.reference_data = reference_data
        self.mean = np.mean(reference_data, axis=0)
        self.cov = np.cov(reference_data.T)
        self.inv_cov = np.linalg.pinv(self.cov)
        
        # PCA for dimension reduction
        self.pca = PCA()
        self.pca.fit(reference_data)
        
        # Calculate control limits
        n, p = reference_data.shape
        self.calculate_control_limits(n, p)
        
    def calculate_control_limits(self, n: int, p: int, alpha: float = 0.01):
        """Calculate Hotelling T² control limits."""
        # Upper control limit for T²
        F_critical = stats.f.ppf(1 - alpha, p, n - p)
        self.T2_UCL = p * (n - 1) * (n + 1) * F_critical / (n * (n - p))
        
        # Q statistic (SPE) limit for PCA
        # Simplified - normally would use Jackson-Mudholkar approximation
        self.Q_UCL = np.percentile(self._calculate_Q(self.reference_data), 99)
        
    def calculate_T2(self, x: np.ndarray) -> float:
        """Calculate Hotelling T² statistic."""
        diff = x - self.mean
        T2 = diff @ self.inv_cov @ diff.T
        return T2
        
    def _calculate_Q(self, X: np.ndarray) -> np.ndarray:
        """Calculate Q statistic (Squared Prediction Error)."""
        X_reconstructed = self.pca.inverse_transform(self.pca.transform(X))
        residuals = X - X_reconstructed
        Q = np.sum(residuals ** 2, axis=1)
        return Q
        
    def monitor(self, new_data: np.ndarray) -> Dict[str, Any]:
        """
        Monitor new data point(s).
        
        Returns:
            Dictionary with T², Q statistics and violation flags
        """
        if len(new_data.shape) == 1:
            new_data = new_data.reshape(1, -1)
            
        results = []
        for x in new_data:
            T2 = self.calculate_T2(x)
            Q = self._calculate_Q(x.reshape(1, -1))[0]
            
            results.append({
                'T2': T2,
                'Q': Q,
                'T2_violation': T2 > self.T2_UCL,
                'Q_violation': Q > self.Q_UCL,
                'in_control': T2 <= self.T2_UCL and Q <= self.Q_UCL
            })
            
        return results


class VirtualMetrologyEngine:
    """Virtual Metrology prediction engine."""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        
    def register_model(
        self,
        model_id: str,
        model_path: str,
        scaler_path: Optional[str] = None,
        feature_config: Optional[Dict] = None
    ):
        """Register a trained model."""
        try:
            # Load model
            self.models[model_id] = joblib.load(model_path)
            
            # Load scaler if provided
            if scaler_path:
                self.scalers[model_id] = joblib.load(scaler_path)
                
            # Store feature configuration
            if feature_config:
                self.feature_extractors[model_id] = FeatureExtractor(feature_config)
                
            logger.info(f"Model {model_id} registered successfully")
            
        except Exception as e:
            logger.error(f"Failed to register model {model_id}: {e}")
            raise
            
    def extract_features(
        self,
        model_id: str,
        telemetry_data: pd.DataFrame
    ) -> np.ndarray:
        """Extract features from telemetry data."""
        
        if model_id not in self.feature_extractors:
            # Default feature extraction - use all numeric columns
            return telemetry_data.select_dtypes(include=[np.number]).values
            
        extractor = self.feature_extractors[model_id]
        return extractor.extract(telemetry_data)
        
    def predict(
        self,
        model_id: str,
        telemetry_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Make prediction using virtual metrology model."""
        
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not registered")
            
        try:
            # Extract features
            features = self.extract_features(model_id, telemetry_data)
            
            # Scale features if scaler available
            if model_id in self.scalers:
                features = self.scalers[model_id].transform(features)
                
            # Make prediction
            model = self.models[model_id]
            prediction = model.predict(features)
            
            # Calculate prediction intervals if model supports it
            prediction_interval = None
            if hasattr(model, 'predict_proba'):
                # For probabilistic models
                proba = model.predict_proba(features)
                prediction_interval = np.percentile(proba, [5, 95], axis=0)
            elif hasattr(model, 'predict_std'):
                # For models with uncertainty estimation
                std = model.predict_std(features)
                prediction_interval = [
                    prediction - 1.96 * std,
                    prediction + 1.96 * std
                ]
                
            # Calculate feature importance if available
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                
            return {
                'prediction': prediction[0] if len(prediction) == 1 else prediction.tolist(),
                'confidence_interval': prediction_interval,
                'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
                'model_id': model_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed for model {model_id}: {e}")
            raise


class FeatureExtractor:
    """Extract features from raw telemetry data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def extract(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features based on configuration."""
        features = []
        
        for feature_def in self.config.get('features', []):
            feature_name = feature_def['name']
            source_column = feature_def['source']
            aggregation = feature_def.get('aggregation', 'mean')
            window = feature_def.get('window_size', len(data))
            
            if source_column not in data.columns:
                logger.warning(f"Column {source_column} not found")
                continue
                
            # Apply aggregation
            if aggregation == 'mean':
                value = data[source_column].rolling(window).mean().iloc[-1]
            elif aggregation == 'std':
                value = data[source_column].rolling(window).std().iloc[-1]
            elif aggregation == 'min':
                value = data[source_column].rolling(window).min().iloc[-1]
            elif aggregation == 'max':
                value = data[source_column].rolling(window).max().iloc[-1]
            elif aggregation == 'median':
                value = data[source_column].rolling(window).median().iloc[-1]
            elif aggregation == 'range':
                value = (data[source_column].rolling(window).max().iloc[-1] -
                        data[source_column].rolling(window).min().iloc[-1])
            elif aggregation == 'slope':
                # Linear regression slope
                y = data[source_column].iloc[-window:].values
                x = np.arange(len(y))
                if len(y) > 1:
                    slope = np.polyfit(x, y, 1)[0]
                    value = slope
                else:
                    value = 0
            else:
                value = data[source_column].iloc[-1]
                
            features.append(value)
            
        return np.array(features).reshape(1, -1)


class SPCService:
    """High-level SPC service for process monitoring."""
    
    def __init__(self):
        self.calculator = SPCCalculator()
        self.series_data = {}
        self.control_limits = {}
        self.rule_checkers = {}
        
    async def add_point(
        self,
        series_id: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Add new data point to SPC series."""
        
        if timestamp is None:
            timestamp = datetime.now()
            
        # Initialize series if needed
        if series_id not in self.series_data:
            self.series_data[series_id] = []
            
        # Add point
        point = {'value': value, 'timestamp': timestamp}
        self.series_data[series_id].append(point)
        
        # Keep only recent data (e.g., last 1000 points)
        if len(self.series_data[series_id]) > 1000:
            self.series_data[series_id].pop(0)
            
        # Get values for analysis
        values = np.array([p['value'] for p in self.series_data[series_id]])
        
        # Recalculate control limits if needed
        if series_id not in self.control_limits or len(values) % 20 == 0:
            self.control_limits[series_id] = self.calculator.calculate_control_limits(values)
            self.rule_checkers[series_id] = SPCRuleChecker(self.control_limits[series_id])
            
        # Check for violations
        violations = []
        if len(values) > 1:
            violations = self.rule_checkers[series_id].check_rules(values[-20:])
            
        # Calculate statistics
        limits = self.control_limits[series_id]
        
        result = {
            'series_id': series_id,
            'value': value,
            'timestamp': timestamp.isoformat(),
            'in_control': limits.is_in_control(value),
            'violations': violations,
            'statistics': {
                'mean': np.mean(values),
                'std': np.std(values),
                'UCL': limits.UCL,
                'CL': limits.CL,
                'LCL': limits.LCL
            }
        }
        
        # Generate alert if violations detected
        if violations:
            await self._generate_alert(series_id, result)
            
        return result
        
    async def _generate_alert(self, series_id: str, result: Dict[str, Any]):
        """Generate alert for SPC violations."""
        # In production, would send to notification service
        logger.warning(f"SPC Alert for series {series_id}: {result['violations']}")
        
    def get_series_statistics(self, series_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a series."""
        
        if series_id not in self.series_data:
            return {}
            
        values = np.array([p['value'] for p in self.series_data[series_id]])
        limits = self.control_limits.get(series_id)
        
        stats = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'range': np.ptp(values),
        }
        
        if limits and limits.USL and limits.LSL:
            capability = self.calculator.calculate_capability_indices(
                values, limits.USL, limits.LSL
            )
            stats.update(capability)
            
        return stats
