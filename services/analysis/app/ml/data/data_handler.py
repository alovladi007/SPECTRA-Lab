"""
Semiconductor Manufacturing Data Handler
Specialized utilities for processing semiconductor process data
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemiconductorDataProcessor:
    """
    Processes semiconductor manufacturing data with domain-specific handling
    
    Features:
    - Automatic outlier detection (critical for wafer data)
    - Physics-aware feature engineering
    - Missing data imputation
    - Process parameter normalization
    """
    
    def __init__(
        self,
        outlier_method: str = "iqr",
        scaling_method: str = "robust",
        handle_missing: str = "interpolate"
    ):
        self.outlier_method = outlier_method
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        
        self.scaler = None
        self.feature_names = None
        self.is_fitted = False
    
    def fit_transform(
        self,
        data: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit and transform semiconductor data
        
        Args:
            data: Input DataFrame with process parameters
            target_column: Name of target column (if regression/classification)
            
        Returns:
            (X_processed, y) tuple
        """
        logger.info(f"Processing semiconductor data: {data.shape}")

        df = data.copy()

        # Keep target column in dataframe during outlier removal to maintain index alignment
        has_target = target_column and target_column in df.columns

        # Store feature names (excluding target)
        if has_target:
            self.feature_names = [col for col in df.columns if col != target_column]
        else:
            self.feature_names = df.columns.tolist()

        # 1. Handle missing values (on entire dataframe including target)
        df = self._handle_missing_values(df)

        # 2. Remove outliers (on entire dataframe so indices stay aligned)
        df = self._remove_outliers(df)

        # NOW separate features and target after outlier removal
        if has_target:
            y = df[target_column].values
            df = df.drop(columns=[target_column])
        else:
            y = None

        # 3. Feature engineering for semiconductor processes
        df = self._engineer_features(df)

        # 4. Scale features
        X = self._scale_features(df)

        self.is_fitted = True
        logger.info(f"Processing complete: {X.shape}")

        return X, y
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted parameters"""
        if not self.is_fitted:
            raise ValueError("Processor not fitted. Call fit_transform first.")
        
        df = data.copy()
        
        # Ensure same columns
        df = df[self.feature_names]
        
        # Apply same preprocessing
        df = self._handle_missing_values(df)
        df = self._engineer_features(df)
        X = self.scaler.transform(df)
        
        return X
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in semiconductor data"""
        if self.handle_missing == "interpolate":
            # Time-series interpolation (good for sensor data)
            df = df.interpolate(method='linear', limit_direction='both')
        elif self.handle_missing == "median":
            # Fill with median (robust to outliers)
            df = df.fillna(df.median())
        elif self.handle_missing == "drop":
            df = df.dropna()
        
        # Fill any remaining NaNs with 0
        df = df.fillna(0)
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from semiconductor data
        Critical for wafer manufacturing where sensor errors are common
        """
        if self.outlier_method == "iqr":
            # Interquartile range method
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove rows with outliers
            mask = ~((df < lower_bound) | (df > upper_bound)).any(axis=1)
            df = df[mask]
            
            logger.info(f"Removed {(~mask).sum()} outlier samples")
        
        elif self.outlier_method == "zscore":
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(df))
            mask = (z_scores < 3).all(axis=1)
            df = df[mask]
            
            logger.info(f"Removed {(~mask).sum()} outlier samples")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer semiconductor-specific features
        
        Common derived features:
        - Temperature ratios
        - Pressure differentials
        - Flow rate products
        - Process time interactions
        """
        engineered = df.copy()
        
        # Example: If temperature columns exist, create ratios
        temp_cols = [col for col in df.columns if 'temp' in col.lower()]
        if len(temp_cols) >= 2:
            engineered[f'{temp_cols[0]}_to_{temp_cols[1]}_ratio'] = (
                df[temp_cols[0]] / (df[temp_cols[1]] + 1e-10)
            )
        
        # Example: Pressure differentials
        pressure_cols = [col for col in df.columns if 'pressure' in col.lower()]
        if len(pressure_cols) >= 2:
            engineered['pressure_differential'] = (
                df[pressure_cols[0]] - df[pressure_cols[1]]
            )
        
        # Example: Flow rate products (common in CVD processes)
        flow_cols = [col for col in df.columns if 'flow' in col.lower()]
        if len(flow_cols) >= 2:
            engineered['total_flow'] = sum(df[col] for col in flow_cols)
        
        return engineered
    
    def _scale_features(self, df: pd.DataFrame) -> np.ndarray:
        """Scale features using selected method"""
        if self.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.scaling_method == "robust":
            # Robust to outliers (recommended for semiconductor data)
            self.scaler = RobustScaler()
        elif self.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        X = self.scaler.fit_transform(df)
        return X


class SemiconductorDataGenerator:
    """Generate synthetic semiconductor manufacturing data for testing"""
    
    @staticmethod
    def generate_wafer_yield_data(
        n_samples: int = 1000,
        n_features: int = 10,
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate synthetic wafer yield prediction data
        
        Simulates common semiconductor process parameters:
        - Temperature (°C)
        - Pressure (Torr)
        - Gas flow rates (sccm)
        - RF Power (W)
        - Process time (s)
        """
        np.random.seed(42)
        
        data = {}
        
        # Temperature parameters
        data['temperature_chamber'] = np.random.normal(400, 50, n_samples)
        data['temperature_wafer'] = data['temperature_chamber'] + np.random.normal(0, 10, n_samples)
        
        # Pressure parameters
        data['pressure_chamber'] = np.random.uniform(1, 10, n_samples)  # Torr
        
        # Gas flow rates
        data['flow_ar'] = np.random.uniform(10, 100, n_samples)  # sccm
        data['flow_n2'] = np.random.uniform(5, 50, n_samples)
        data['flow_sih4'] = np.random.uniform(1, 20, n_samples)
        
        # RF Power
        data['rf_power'] = np.random.uniform(100, 500, n_samples)  # W
        
        # Process time
        data['process_time'] = np.random.uniform(30, 300, n_samples)  # seconds
        
        # Additional random features
        for i in range(n_features - 8):
            data[f'param_{i}'] = np.random.normal(0, 1, n_samples)
        
        # Generate target: wafer yield (0-100%)
        # Simplified physics-based model
        yield_base = (
            0.3 * (data['temperature_chamber'] - 350) / 100 +
            0.2 * (data['rf_power'] - 300) / 200 +
            0.2 * (data['flow_ar'] - 50) / 50 +
            0.1 * (data['pressure_chamber'] - 5) / 5 +
            0.2 * (data['process_time'] - 150) / 150
        )
        
        # Normalize to 0-100 range and add noise
        data['yield_percent'] = np.clip(
            50 + 30 * yield_base + noise_level * np.random.normal(0, 10, n_samples),
            0, 100
        )
        
        df = pd.DataFrame(data)
        logger.info(f"Generated synthetic wafer data: {df.shape}")
        
        return df
    
    @staticmethod
    def generate_defect_detection_data(
        n_samples: int = 1000,
        defect_rate: float = 0.1
    ) -> pd.DataFrame:
        """Generate synthetic defect detection data (classification)"""
        np.random.seed(42)
        
        data = {}
        
        # Imaging features (simulated)
        data['intensity_mean'] = np.random.normal(128, 30, n_samples)
        data['intensity_std'] = np.random.normal(20, 5, n_samples)
        data['edge_density'] = np.random.uniform(0, 1, n_samples)
        data['texture_contrast'] = np.random.uniform(0, 100, n_samples)
        
        # Process parameters
        data['exposure_time'] = np.random.uniform(1, 10, n_samples)
        data['focus_position'] = np.random.normal(0, 0.5, n_samples)
        data['dose_energy'] = np.random.uniform(5, 50, n_samples)
        
        # Generate defects (binary classification)
        # Defects correlated with extreme parameter values
        defect_prob = (
            0.3 * (np.abs(data['focus_position']) > 0.3) +
            0.3 * (data['exposure_time'] < 2) +
            0.4 * (data['intensity_std'] > 25)
        )
        
        data['is_defect'] = (defect_prob + np.random.random(n_samples) * 0.3 > 0.8).astype(int)
        
        # Ensure minimum defect rate
        n_defects = int(n_samples * defect_rate)
        if data['is_defect'].sum() < n_defects:
            defect_indices = np.random.choice(
                np.where(data['is_defect'] == 0)[0],
                n_defects - data['is_defect'].sum(),
                replace=False
            )
            data['is_defect'][defect_indices] = 1
        
        df = pd.DataFrame(data)
        logger.info(f"Generated defect detection data: {df.shape}, Defect rate: {data['is_defect'].mean():.2%}")
        
        return df


def load_semiconductor_data(
    data_path: Optional[str] = None,
    data_type: str = "synthetic_yield",
    test_size: float = 0.2,
    val_size: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split semiconductor data
    
    Args:
        data_path: Path to data file (CSV) or None for synthetic data
        data_type: Type of synthetic data ('synthetic_yield' or 'synthetic_defect')
        test_size: Fraction for test set
        val_size: Fraction for validation set
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    
    if data_path:
        # Load real data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
    else:
        # Generate synthetic data
        logger.info(f"Generating {data_type} data")
        if data_type == "synthetic_yield":
            df = SemiconductorDataGenerator.generate_wafer_yield_data()
            target_col = "yield_percent"
        elif data_type == "synthetic_defect":
            df = SemiconductorDataGenerator.generate_defect_detection_data()
            target_col = "is_defect"
        else:
            raise ValueError(f"Unknown synthetic data type: {data_type}")
    
    # Process data
    processor = SemiconductorDataProcessor(
        outlier_method="iqr",
        scaling_method="robust",
        handle_missing="interpolate"
    )
    
    # Detect target column
    if data_path:
        target_col = df.columns[-1]  # Assume last column is target
    
    X, y = processor.fit_transform(df, target_column=target_col)
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42
    )
    
    logger.info(f"Data split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test
