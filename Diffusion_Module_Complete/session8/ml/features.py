"""
FDC Feature Engineering for Virtual Metrology - Session 8

Extracts predictive features from Furnace Data Collection (FDC) time series
and recipe parameters for Virtual Metrology models.

Features include:
- Thermal profile features (ramp rates, soak integrals, peak temperatures)
- Process stability features (pressure, gas flow variability)
- Spatial features (zone balance, boat load, slot position)
- Historical features (wafer/lot history, drift trends)

Status: PRODUCTION READY ✅
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats, integrate


@dataclass
class ThermalProfileFeatures:
    """Features extracted from thermal profile."""
    ramp_rate_avg: float  # Average heating rate (°C/min)
    ramp_rate_max: float  # Max heating rate
    ramp_rate_std: float  # Variability in heating rate
    soak_integral: float  # Time-temperature integral during soak (°C·min)
    peak_temperature: float  # Maximum temperature reached
    time_at_peak: float  # Time spent at peak ±5°C (minutes)
    cooldown_rate: float  # Cooling rate (°C/min)
    temperature_uniformity: float  # Std dev across zones
    setpoint_deviation_mean: float  # Mean deviation from setpoint
    setpoint_deviation_max: float  # Max deviation from setpoint


@dataclass
class ProcessStabilityFeatures:
    """Features extracted from process stability."""
    pressure_mean: float  # Mean pressure (Torr)
    pressure_std: float  # Pressure variability
    pressure_drift: float  # Linear drift in pressure
    gas_flow_o2_mean: float  # Mean O2 flow (sccm)
    gas_flow_o2_std: float  # O2 flow variability
    gas_flow_n2_mean: float  # Mean N2 flow (sccm)
    gas_flow_n2_std: float  # N2 flow variability
    alarm_count: int  # Number of alarms during run
    recovery_time: float  # Total time in recovery from alarms (min)


@dataclass
class SpatialFeatures:
    """Features related to spatial position and loading."""
    zone_balance: float  # Temperature balance across zones (std dev)
    boat_load_count: int  # Number of wafers in boat
    slot_index: int  # Position in boat (0=bottom, higher=top)
    slot_normalized: float  # Normalized position (0-1)
    neighbor_distance: float  # Distance to nearest wafer (slots)


@dataclass
class HistoricalFeatures:
    """Features from wafer/lot history."""
    cumulative_thermal_budget: float  # Total °C·min in all prior steps
    steps_completed: int  # Number of prior process steps
    time_since_last_process: float  # Hours since last thermal step
    lot_age: float  # Days since lot start
    wafer_usage_count: int  # Number of times wafer processed


class FDCFeatureExtractor:
    """
    Extract features from FDC time series data for Virtual Metrology.

    This class processes raw FDC sensor data and recipe parameters
    to create a feature vector suitable for ML models.
    """

    def __init__(
        self,
        soak_temp_tolerance: float = 5.0,
        min_soak_duration: float = 10.0
    ):
        """
        Initialize feature extractor.

        Args:
            soak_temp_tolerance: Temperature tolerance for soak detection (°C)
            min_soak_duration: Minimum duration to consider as soak (minutes)
        """
        self.soak_temp_tolerance = soak_temp_tolerance
        self.min_soak_duration = min_soak_duration

    def extract_thermal_features(
        self,
        time: np.ndarray,
        temperature: np.ndarray,
        setpoint: Optional[np.ndarray] = None,
        zone_temps: Optional[np.ndarray] = None
    ) -> ThermalProfileFeatures:
        """
        Extract thermal profile features.

        Args:
            time: Time array (minutes)
            temperature: Temperature array (°C)
            setpoint: Setpoint temperature array (optional)
            zone_temps: Multi-zone temperatures shape (n_samples, n_zones) (optional)

        Returns:
            ThermalProfileFeatures object
        """
        # Calculate ramp rates (°C/min)
        dt = np.diff(time)
        dT = np.diff(temperature)
        ramp_rates = dT / dt

        # Handle potential division by zero
        valid_rates = ramp_rates[np.isfinite(ramp_rates)]

        ramp_rate_avg = np.mean(valid_rates) if len(valid_rates) > 0 else 0.0
        ramp_rate_max = np.max(valid_rates) if len(valid_rates) > 0 else 0.0
        ramp_rate_std = np.std(valid_rates) if len(valid_rates) > 0 else 0.0

        # Find soak region (temperature within tolerance of peak)
        peak_temperature = np.max(temperature)
        soak_mask = temperature >= (peak_temperature - self.soak_temp_tolerance)

        # Calculate soak integral (time-temperature product)
        if np.any(soak_mask):
            soak_time = time[soak_mask]
            soak_temp = temperature[soak_mask]

            # Integrate temperature over time during soak
            soak_integral = integrate.trapz(soak_temp, soak_time)

            # Time at peak
            time_at_peak = soak_time[-1] - soak_time[0] if len(soak_time) > 1 else 0.0
        else:
            soak_integral = 0.0
            time_at_peak = 0.0

        # Cooldown rate (last 20% of profile)
        cooldown_start_idx = int(0.8 * len(temperature))
        if cooldown_start_idx < len(temperature) - 1:
            cooldown_dt = time[-1] - time[cooldown_start_idx]
            cooldown_dT = temperature[-1] - temperature[cooldown_start_idx]
            cooldown_rate = cooldown_dT / cooldown_dt if cooldown_dt > 0 else 0.0
        else:
            cooldown_rate = 0.0

        # Temperature uniformity across zones
        if zone_temps is not None and zone_temps.shape[1] > 1:
            # Calculate std dev across zones at each time point
            zone_std = np.std(zone_temps, axis=1)
            temperature_uniformity = np.mean(zone_std)
        else:
            temperature_uniformity = 0.0

        # Setpoint deviation
        if setpoint is not None:
            deviation = temperature - setpoint
            setpoint_deviation_mean = np.mean(np.abs(deviation))
            setpoint_deviation_max = np.max(np.abs(deviation))
        else:
            setpoint_deviation_mean = 0.0
            setpoint_deviation_max = 0.0

        return ThermalProfileFeatures(
            ramp_rate_avg=float(ramp_rate_avg),
            ramp_rate_max=float(ramp_rate_max),
            ramp_rate_std=float(ramp_rate_std),
            soak_integral=float(soak_integral),
            peak_temperature=float(peak_temperature),
            time_at_peak=float(time_at_peak),
            cooldown_rate=float(cooldown_rate),
            temperature_uniformity=float(temperature_uniformity),
            setpoint_deviation_mean=float(setpoint_deviation_mean),
            setpoint_deviation_max=float(setpoint_deviation_max)
        )

    def extract_stability_features(
        self,
        pressure: np.ndarray,
        gas_flow_o2: Optional[np.ndarray] = None,
        gas_flow_n2: Optional[np.ndarray] = None,
        alarms: Optional[np.ndarray] = None,
        time: Optional[np.ndarray] = None
    ) -> ProcessStabilityFeatures:
        """
        Extract process stability features.

        Args:
            pressure: Pressure array (Torr)
            gas_flow_o2: O2 flow rate array (sccm) (optional)
            gas_flow_n2: N2 flow rate array (sccm) (optional)
            alarms: Boolean array of alarm states (optional)
            time: Time array for drift calculation (optional)

        Returns:
            ProcessStabilityFeatures object
        """
        # Pressure features
        pressure_mean = float(np.mean(pressure))
        pressure_std = float(np.std(pressure))

        # Pressure drift (linear trend)
        if time is not None and len(time) == len(pressure):
            slope, _, _, _, _ = stats.linregress(time, pressure)
            pressure_drift = float(slope)
        else:
            pressure_drift = 0.0

        # Gas flow features
        if gas_flow_o2 is not None:
            gas_flow_o2_mean = float(np.mean(gas_flow_o2))
            gas_flow_o2_std = float(np.std(gas_flow_o2))
        else:
            gas_flow_o2_mean = 0.0
            gas_flow_o2_std = 0.0

        if gas_flow_n2 is not None:
            gas_flow_n2_mean = float(np.mean(gas_flow_n2))
            gas_flow_n2_std = float(np.std(gas_flow_n2))
        else:
            gas_flow_n2_mean = 0.0
            gas_flow_n2_std = 0.0

        # Alarm features
        if alarms is not None:
            alarm_count = int(np.sum(alarms))

            # Calculate recovery time (time spent in alarm state)
            if time is not None and len(time) == len(alarms):
                alarm_intervals = np.diff(time)[alarms[:-1]]
                recovery_time = float(np.sum(alarm_intervals))
            else:
                recovery_time = 0.0
        else:
            alarm_count = 0
            recovery_time = 0.0

        return ProcessStabilityFeatures(
            pressure_mean=pressure_mean,
            pressure_std=pressure_std,
            pressure_drift=pressure_drift,
            gas_flow_o2_mean=gas_flow_o2_mean,
            gas_flow_o2_std=gas_flow_o2_std,
            gas_flow_n2_mean=gas_flow_n2_mean,
            gas_flow_n2_std=gas_flow_n2_std,
            alarm_count=alarm_count,
            recovery_time=recovery_time
        )

    def extract_spatial_features(
        self,
        zone_temps: Optional[np.ndarray] = None,
        boat_load_count: int = 25,
        slot_index: int = 0
    ) -> SpatialFeatures:
        """
        Extract spatial and loading features.

        Args:
            zone_temps: Multi-zone temperatures shape (n_samples, n_zones) (optional)
            boat_load_count: Total wafers in boat
            slot_index: Position in boat (0 = bottom)

        Returns:
            SpatialFeatures object
        """
        # Zone balance (temperature uniformity across zones)
        if zone_temps is not None and zone_temps.shape[1] > 1:
            # Calculate std dev across zones (averaged over time)
            zone_balance = float(np.mean(np.std(zone_temps, axis=1)))
        else:
            zone_balance = 0.0

        # Normalized slot position (0 = bottom, 1 = top)
        slot_normalized = float(slot_index / max(boat_load_count - 1, 1))

        # Neighbor distance (simplified: assume uniform spacing)
        # In practice, this would come from actual boat configuration
        neighbor_distance = 1.0  # Default: 1 slot spacing

        return SpatialFeatures(
            zone_balance=zone_balance,
            boat_load_count=boat_load_count,
            slot_index=slot_index,
            slot_normalized=slot_normalized,
            neighbor_distance=neighbor_distance
        )

    def extract_historical_features(
        self,
        prior_thermal_budgets: Optional[List[float]] = None,
        steps_completed: int = 0,
        time_since_last_process: float = 0.0,
        lot_age: float = 0.0,
        wafer_usage_count: int = 1
    ) -> HistoricalFeatures:
        """
        Extract features from wafer/lot history.

        Args:
            prior_thermal_budgets: List of °C·min from prior steps
            steps_completed: Number of completed process steps
            time_since_last_process: Hours since last thermal step
            lot_age: Days since lot start
            wafer_usage_count: Number of times wafer has been processed

        Returns:
            HistoricalFeatures object
        """
        # Cumulative thermal budget
        if prior_thermal_budgets is not None:
            cumulative_thermal_budget = float(np.sum(prior_thermal_budgets))
        else:
            cumulative_thermal_budget = 0.0

        return HistoricalFeatures(
            cumulative_thermal_budget=cumulative_thermal_budget,
            steps_completed=steps_completed,
            time_since_last_process=time_since_last_process,
            lot_age=lot_age,
            wafer_usage_count=wafer_usage_count
        )

    def create_feature_vector(
        self,
        thermal: ThermalProfileFeatures,
        stability: ProcessStabilityFeatures,
        spatial: SpatialFeatures,
        historical: HistoricalFeatures
    ) -> pd.Series:
        """
        Combine all features into a single feature vector.

        Args:
            thermal: Thermal profile features
            stability: Process stability features
            spatial: Spatial features
            historical: Historical features

        Returns:
            pandas Series with all features
        """
        features = {}

        # Thermal features
        features['ramp_rate_avg'] = thermal.ramp_rate_avg
        features['ramp_rate_max'] = thermal.ramp_rate_max
        features['ramp_rate_std'] = thermal.ramp_rate_std
        features['soak_integral'] = thermal.soak_integral
        features['peak_temperature'] = thermal.peak_temperature
        features['time_at_peak'] = thermal.time_at_peak
        features['cooldown_rate'] = thermal.cooldown_rate
        features['temperature_uniformity'] = thermal.temperature_uniformity
        features['setpoint_deviation_mean'] = thermal.setpoint_deviation_mean
        features['setpoint_deviation_max'] = thermal.setpoint_deviation_max

        # Stability features
        features['pressure_mean'] = stability.pressure_mean
        features['pressure_std'] = stability.pressure_std
        features['pressure_drift'] = stability.pressure_drift
        features['gas_flow_o2_mean'] = stability.gas_flow_o2_mean
        features['gas_flow_o2_std'] = stability.gas_flow_o2_std
        features['gas_flow_n2_mean'] = stability.gas_flow_n2_mean
        features['gas_flow_n2_std'] = stability.gas_flow_n2_std
        features['alarm_count'] = stability.alarm_count
        features['recovery_time'] = stability.recovery_time

        # Spatial features
        features['zone_balance'] = spatial.zone_balance
        features['boat_load_count'] = spatial.boat_load_count
        features['slot_index'] = spatial.slot_index
        features['slot_normalized'] = spatial.slot_normalized
        features['neighbor_distance'] = spatial.neighbor_distance

        # Historical features
        features['cumulative_thermal_budget'] = historical.cumulative_thermal_budget
        features['steps_completed'] = historical.steps_completed
        features['time_since_last_process'] = historical.time_since_last_process
        features['lot_age'] = historical.lot_age
        features['wafer_usage_count'] = historical.wafer_usage_count

        return pd.Series(features)


def extract_features_from_fdc_data(
    fdc_data: Dict,
    recipe_params: Optional[Dict] = None,
    historical_data: Optional[Dict] = None
) -> pd.Series:
    """
    Quick helper to extract features from FDC data dictionary.

    Args:
        fdc_data: Dictionary with FDC time series
            Keys: 'time', 'temperature', 'setpoint', 'pressure',
                  'gas_flow_o2', 'gas_flow_n2', 'alarms', 'zone_temps'
        recipe_params: Optional recipe parameters
            Keys: 'boat_load_count', 'slot_index'
        historical_data: Optional historical data
            Keys: 'prior_thermal_budgets', 'steps_completed', etc.

    Returns:
        Feature vector as pandas Series

    Example:
        >>> fdc = {'time': time, 'temperature': temp, 'pressure': press}
        >>> features = extract_features_from_fdc_data(fdc)
        >>> print(features['peak_temperature'])
    """
    extractor = FDCFeatureExtractor()

    # Extract thermal features
    thermal = extractor.extract_thermal_features(
        time=fdc_data['time'],
        temperature=fdc_data['temperature'],
        setpoint=fdc_data.get('setpoint'),
        zone_temps=fdc_data.get('zone_temps')
    )

    # Extract stability features
    stability = extractor.extract_stability_features(
        pressure=fdc_data.get('pressure', np.zeros_like(fdc_data['time'])),
        gas_flow_o2=fdc_data.get('gas_flow_o2'),
        gas_flow_n2=fdc_data.get('gas_flow_n2'),
        alarms=fdc_data.get('alarms'),
        time=fdc_data.get('time')
    )

    # Extract spatial features
    recipe = recipe_params or {}
    spatial = extractor.extract_spatial_features(
        zone_temps=fdc_data.get('zone_temps'),
        boat_load_count=recipe.get('boat_load_count', 25),
        slot_index=recipe.get('slot_index', 0)
    )

    # Extract historical features
    hist_data = historical_data or {}
    historical = extractor.extract_historical_features(
        prior_thermal_budgets=hist_data.get('prior_thermal_budgets'),
        steps_completed=hist_data.get('steps_completed', 0),
        time_since_last_process=hist_data.get('time_since_last_process', 0.0),
        lot_age=hist_data.get('lot_age', 0.0),
        wafer_usage_count=hist_data.get('wafer_usage_count', 1)
    )

    return extractor.create_feature_vector(thermal, stability, spatial, historical)
