"""
Advanced Film Thickness Modeling

Comprehensive thickness prediction including:
- Arrhenius-based deposition kinetics
- Reactor-specific geometry effects
- Gas depletion and flow patterns
- WIW/WTW uniformity calculation
- VM feature engineering
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from enum import Enum

from .reactor_geometry import (
    ReactorType,
    ShowerheadReactor,
    HorizontalFlowReactor,
    BatchFurnaceReactor,
)


class CVDMode(str, Enum):
    """CVD process modes affecting deposition kinetics"""
    THERMAL = "THERMAL"  # Thermally activated
    PLASMA = "PLASMA"  # Plasma-enhanced
    PHOTO = "PHOTO"  # Photo-assisted
    LASER = "LASER"  # Laser-assisted


@dataclass
class DepositionParameters:
    """Process parameters affecting deposition rate"""
    # Temperature
    temperature_c: float

    # Pressure
    pressure_torr: float

    # Gas flows (sccm)
    precursor_flow_sccm: float
    carrier_gas_flow_sccm: float = 0.0
    dilution_gas_flow_sccm: float = 0.0

    # Plasma parameters (for PECVD)
    rf_power_w: float = 0.0
    rf_frequency_mhz: float = 13.56
    bias_voltage_v: float = 0.0

    # Reactor state
    wafer_diameter_mm: float = 200.0
    rotation_speed_rpm: float = 0.0

    # Film properties
    film_material: str = "SiO2"
    target_thickness_nm: float = 100.0


@dataclass
class ArrheniusParameters:
    """
    Arrhenius kinetic parameters for deposition

    rate = A * exp(-Ea / kT) * f(P, flows, power)
    """
    # Pre-exponential factor (nm/min at reference conditions)
    pre_exponential_A: float = 1e12

    # Activation energy (kJ/mol)
    activation_energy_kj_mol: float = 120.0

    # Pressure dependence (rate ~ P^n)
    pressure_exponent: float = 0.5

    # Flow dependence (rate ~ sqrt(flow))
    flow_exponent: float = 0.5

    # Plasma power dependence (for PECVD)
    power_exponent: float = 0.3

    # Reference conditions for normalization
    ref_temp_c: float = 800.0
    ref_pressure_torr: float = 0.3
    ref_flow_sccm: float = 100.0


class DepositionRateCalculator:
    """
    Calculate deposition rate from process parameters

    Implements Arrhenius kinetics with reactor-specific corrections
    """

    def __init__(
        self,
        mode: CVDMode = CVDMode.THERMAL,
        arrhenius_params: Optional[ArrheniusParameters] = None,
    ):
        self.mode = mode
        self.params = arrhenius_params or ArrheniusParameters()

    def calculate_rate(
        self,
        temperature_c: float,
        pressure_torr: float,
        precursor_flow_sccm: float,
        rf_power_w: float = 0.0,
    ) -> float:
        """
        Calculate deposition rate using Arrhenius model

        rate = A * exp(-Ea/RT) * (P/P0)^n * (F/F0)^m * (Power/Power0)^p

        Args:
            temperature_c: Process temperature (°C)
            pressure_torr: Process pressure (Torr)
            precursor_flow_sccm: Precursor flow rate (sccm)
            rf_power_w: RF power (W) for plasma-enhanced

        Returns:
            Deposition rate (nm/min)
        """
        # Arrhenius temperature term: exp(-Ea / kT)
        temp_k = temperature_c + 273.15
        R = 8.314  # J/(mol·K)
        Ea = self.params.activation_energy_kj_mol * 1000  # Convert to J/mol

        arrhenius_term = math.exp(-Ea / (R * temp_k))

        # Pressure term: (P/P0)^n
        pressure_term = (
            (pressure_torr / self.params.ref_pressure_torr) **
            self.params.pressure_exponent
        )

        # Flow term: (F/F0)^m
        flow_term = (
            (precursor_flow_sccm / self.params.ref_flow_sccm) **
            self.params.flow_exponent
        )

        # Plasma power term (for PECVD)
        if self.mode == CVDMode.PLASMA and rf_power_w > 0:
            ref_power_w = 500.0  # Reference power
            power_term = (rf_power_w / ref_power_w) ** self.params.power_exponent
        else:
            power_term = 1.0

        # Combined rate
        rate = (
            self.params.pre_exponential_A *
            arrhenius_term *
            pressure_term *
            flow_term *
            power_term
        )

        # Normalize to realistic range (10-500 nm/min)
        rate_normalized = rate * 1e-9  # Scale down
        rate_clamped = max(10.0, min(500.0, rate_normalized))

        return rate_clamped

    def calculate_thickness_vs_time(
        self,
        params: DepositionParameters,
        time_points_sec: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate film thickness vs time

        Args:
            params: Deposition parameters
            time_points_sec: Time points (seconds)

        Returns:
            Thickness array (nm) at each time point
        """
        # Calculate constant deposition rate
        rate_nm_min = self.calculate_rate(
            temperature_c=params.temperature_c,
            pressure_torr=params.pressure_torr,
            precursor_flow_sccm=params.precursor_flow_sccm,
            rf_power_w=params.rf_power_w,
        )

        # Convert time to minutes
        time_min = time_points_sec / 60.0

        # Linear thickness growth
        thickness_nm = rate_nm_min * time_min

        return thickness_nm


class UniformityCalculator:
    """
    Calculate film thickness uniformity (WIW and WTW)

    Accounts for:
    - Reactor geometry (showerhead, horizontal flow, batch)
    - Gas depletion effects
    - Wafer rotation
    - Position in batch (for furnaces)
    """

    def __init__(self, reactor: Optional[object] = None):
        """
        Initialize uniformity calculator

        Args:
            reactor: Reactor geometry object (ShowerheadReactor, etc.)
        """
        self.reactor = reactor

    def calculate_wiw_map(
        self,
        base_thickness_nm: float,
        wafer_diameter_mm: float,
        num_points: int = 49,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate WIW (within-wafer) thickness map

        Args:
            base_thickness_nm: Nominal center thickness
            wafer_diameter_mm: Wafer diameter
            num_points: Number of measurement points (49, 121, etc.)

        Returns:
            Tuple of (x_coords_mm, y_coords_mm, thickness_map_nm)
        """
        if self.reactor is None:
            # Default simple model (no reactor specified)
            return self._default_wiw_map(base_thickness_nm, wafer_diameter_mm, num_points)

        # Use reactor-specific model
        if isinstance(self.reactor, ShowerheadReactor):
            return self._showerhead_wiw_map(base_thickness_nm, wafer_diameter_mm, num_points)
        elif isinstance(self.reactor, HorizontalFlowReactor):
            return self._horizontal_flow_wiw_map(base_thickness_nm, wafer_diameter_mm, num_points)
        elif isinstance(self.reactor, BatchFurnaceReactor):
            return self._batch_furnace_wiw_map(base_thickness_nm, wafer_diameter_mm, num_points)
        else:
            return self._default_wiw_map(base_thickness_nm, wafer_diameter_mm, num_points)

    def _default_wiw_map(
        self,
        base_thickness_nm: float,
        wafer_diameter_mm: float,
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Default radial thickness profile"""
        # Create measurement grid
        radius_mm = wafer_diameter_mm / 2.0
        grid_size = int(math.sqrt(num_points))

        x = np.linspace(-radius_mm, radius_mm, grid_size)
        y = np.linspace(-radius_mm, radius_mm, grid_size)
        X, Y = np.meshgrid(x, y)

        # Radial distance from center
        R = np.sqrt(X**2 + Y**2)
        R_norm = R / radius_mm

        # Simple radial profile (center slightly thicker)
        # thickness(r) = center * (1 - 0.1 * r^2)
        thickness_map = base_thickness_nm * (1.0 - 0.1 * R_norm**2)

        # Mask out points outside wafer
        thickness_map[R > radius_mm] = 0.0

        return X.flatten(), Y.flatten(), thickness_map.flatten()

    def _showerhead_wiw_map(
        self,
        base_thickness_nm: float,
        wafer_diameter_mm: float,
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Showerhead reactor thickness map"""
        radius_mm = wafer_diameter_mm / 2.0
        grid_size = int(math.sqrt(num_points))

        x = np.linspace(-radius_mm, radius_mm, grid_size)
        y = np.linspace(-radius_mm, radius_mm, grid_size)
        X, Y = np.meshgrid(x, y)

        R = np.sqrt(X**2 + Y**2)
        R_norm = R / radius_mm

        # Use reactor model for uniformity
        uniformity_factors = np.zeros_like(R)
        for i in range(grid_size):
            for j in range(grid_size):
                r_mm = R[i, j]
                if r_mm <= radius_mm:
                    uniformity_factors[i, j] = self.reactor.calculate_uniformity_factor(
                        radial_position_mm=r_mm,
                        pressure_torr=1.0,  # Nominal
                        temperature_c=300.0,
                    )

        thickness_map = base_thickness_nm * uniformity_factors
        thickness_map[R > radius_mm] = 0.0

        return X.flatten(), Y.flatten(), thickness_map.flatten()

    def _horizontal_flow_wiw_map(
        self,
        base_thickness_nm: float,
        wafer_diameter_mm: float,
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Horizontal flow reactor thickness map (directional gradient)"""
        radius_mm = wafer_diameter_mm / 2.0
        grid_size = int(math.sqrt(num_points))

        x = np.linspace(-radius_mm, radius_mm, grid_size)
        y = np.linspace(-radius_mm, radius_mm, grid_size)
        X, Y = np.meshgrid(x, y)

        R = np.sqrt(X**2 + Y**2)

        # Use reactor model
        uniformity_factors = np.zeros_like(R)
        for i in range(grid_size):
            for j in range(grid_size):
                r_mm = R[i, j]
                if r_mm <= radius_mm:
                    uniformity_factors[i, j] = self.reactor.calculate_uniformity_factor(
                        x_mm=X[i, j],
                        y_mm=Y[i, j],
                        pressure_torr=0.3,
                        flow_velocity_cm_s=10.0,
                    )

        thickness_map = base_thickness_nm * uniformity_factors
        thickness_map[R > radius_mm] = 0.0

        return X.flatten(), Y.flatten(), thickness_map.flatten()

    def _batch_furnace_wiw_map(
        self,
        base_thickness_nm: float,
        wafer_diameter_mm: float,
        num_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batch furnace WIW map (excellent uniformity)"""
        # LPCVD furnaces have excellent WIW uniformity
        radius_mm = wafer_diameter_mm / 2.0
        grid_size = int(math.sqrt(num_points))

        x = np.linspace(-radius_mm, radius_mm, grid_size)
        y = np.linspace(-radius_mm, radius_mm, grid_size)
        X, Y = np.meshgrid(x, y)

        R = np.sqrt(X**2 + Y**2)
        R_norm = R / radius_mm

        # Excellent uniformity: < 2% variation
        thickness_map = base_thickness_nm * (1.0 - 0.01 * R_norm**2)
        thickness_map[R > radius_mm] = 0.0

        return X.flatten(), Y.flatten(), thickness_map.flatten()

    def calculate_wiw_uniformity(
        self,
        thickness_map: np.ndarray,
    ) -> float:
        """
        Calculate WIW uniformity percentage

        uniformity = (max - min) / (2 * mean) * 100%

        Args:
            thickness_map: Thickness values across wafer (excluding zeros)

        Returns:
            Uniformity percentage
        """
        # Remove zeros (outside wafer)
        valid_thickness = thickness_map[thickness_map > 0]

        if len(valid_thickness) == 0:
            return 0.0

        t_mean = np.mean(valid_thickness)
        t_min = np.min(valid_thickness)
        t_max = np.max(valid_thickness)

        uniformity_pct = ((t_max - t_min) / (2 * t_mean)) * 100.0

        return uniformity_pct

    def calculate_wtw_uniformity(
        self,
        wafer_thicknesses: List[float],
    ) -> float:
        """
        Calculate WTW (wafer-to-wafer) uniformity

        uniformity = (max - min) / (2 * mean) * 100%

        Args:
            wafer_thicknesses: Mean thickness for each wafer in batch

        Returns:
            WTW uniformity percentage
        """
        if len(wafer_thicknesses) == 0:
            return 0.0

        thicknesses_arr = np.array(wafer_thicknesses)

        t_mean = np.mean(thicknesses_arr)
        t_min = np.min(thicknesses_arr)
        t_max = np.max(thicknesses_arr)

        uniformity_pct = ((t_max - t_min) / (2 * t_mean)) * 100.0

        return uniformity_pct


class ThicknessModel:
    """
    Comprehensive thickness model combining all effects

    Provides:
    - Deposition rate calculation
    - Thickness vs time
    - WIW/WTW uniformity
    - Reactor-specific corrections
    - VM feature engineering
    """

    def __init__(
        self,
        mode: CVDMode = CVDMode.THERMAL,
        reactor: Optional[object] = None,
        arrhenius_params: Optional[ArrheniusParameters] = None,
    ):
        self.mode = mode
        self.reactor = reactor

        self.rate_calculator = DepositionRateCalculator(mode, arrhenius_params)
        self.uniformity_calculator = UniformityCalculator(reactor)

    def predict_thickness(
        self,
        params: DepositionParameters,
        time_sec: float,
    ) -> Dict[str, any]:
        """
        Predict film thickness and uniformity

        Args:
            params: Deposition parameters
            time_sec: Deposition time (seconds)

        Returns:
            Dictionary with thickness, rate, uniformity metrics
        """
        # Calculate deposition rate
        rate_nm_min = self.rate_calculator.calculate_rate(
            temperature_c=params.temperature_c,
            pressure_torr=params.pressure_torr,
            precursor_flow_sccm=params.precursor_flow_sccm,
            rf_power_w=params.rf_power_w,
        )

        # Calculate mean thickness
        mean_thickness_nm = rate_nm_min * (time_sec / 60.0)

        # Calculate WIW map
        x_coords, y_coords, thickness_map = self.uniformity_calculator.calculate_wiw_map(
            base_thickness_nm=mean_thickness_nm,
            wafer_diameter_mm=params.wafer_diameter_mm,
            num_points=49,
        )

        # Calculate WIW uniformity
        wiw_uniformity_pct = self.uniformity_calculator.calculate_wiw_uniformity(
            thickness_map=thickness_map
        )

        return {
            "mean_thickness_nm": mean_thickness_nm,
            "deposition_rate_nm_min": rate_nm_min,
            "thickness_map": thickness_map,
            "x_coords_mm": x_coords,
            "y_coords_mm": y_coords,
            "wiw_uniformity_pct": wiw_uniformity_pct,
        }

    def extract_vm_features(
        self,
        params: DepositionParameters,
        time_sec: float,
    ) -> Dict[str, float]:
        """
        Extract features for Virtual Metrology models

        Args:
            params: Deposition parameters
            time_sec: Deposition time

        Returns:
            Feature dictionary for ML models
        """
        prediction = self.predict_thickness(params, time_sec)

        features = {
            # Process parameters
            "temperature_c": params.temperature_c,
            "pressure_torr": params.pressure_torr,
            "precursor_flow_sccm": params.precursor_flow_sccm,
            "total_flow_sccm": (
                params.precursor_flow_sccm +
                params.carrier_gas_flow_sccm +
                params.dilution_gas_flow_sccm
            ),
            "rf_power_w": params.rf_power_w,
            "time_sec": time_sec,

            # Derived features
            "deposition_rate_nm_min": prediction["deposition_rate_nm_min"],
            "predicted_thickness_nm": prediction["mean_thickness_nm"],

            # Uniformity features
            "wiw_uniformity_pct": prediction["wiw_uniformity_pct"],

            # Reactor-specific features
            "rotation_speed_rpm": params.rotation_speed_rpm,
            "wafer_diameter_mm": params.wafer_diameter_mm,
        }

        return features
