"""
Reactor Geometry Models

Models different CVD reactor configurations and their impact on
film thickness uniformity and deposition characteristics.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, Callable
import numpy as np


class ReactorType(str, Enum):
    """CVD reactor configurations"""
    SHOWERHEAD = "SHOWERHEAD"  # Top-down injection through perforated plate
    HORIZONTAL_FLOW = "HORIZONTAL_FLOW"  # Side injection, laminar flow
    BATCH_FURNACE = "BATCH_FURNACE"  # Hot-wall tube furnace
    VERTICAL_FLOW = "VERTICAL_FLOW"  # Bottom-to-top flow
    ROTATING_DISK = "ROTATING_DISK"  # High-speed rotation (MOCVD)
    CROSS_FLOW = "CROSS_FLOW"  # Perpendicular flow


@dataclass
class ReactorGeometry:
    """Base reactor geometry parameters"""
    reactor_type: ReactorType
    wafer_diameter_mm: float
    chamber_height_mm: Optional[float] = None
    chamber_diameter_mm: Optional[float] = None

    # Wafer configuration
    is_rotating: bool = False
    rotation_speed_rpm: float = 0.0

    # Flow configuration
    inlet_position: str = "center"  # "center", "edge", "side"
    outlet_position: str = "edge"  # "edge", "center", "side"

    # Heating configuration
    heater_type: str = "susceptor"  # "susceptor", "lamp", "resistive"
    is_hot_wall: bool = False


class ShowerheadReactor:
    """
    Showerhead (top-down injection) reactor model

    Common in PECVD and some LPCVD systems.
    Gas flows through perforated plate above wafer.

    Uniformity factors:
    - Showerhead hole pattern and density
    - Gap between showerhead and wafer
    - Pressure (affects diffusion length)
    """

    def __init__(
        self,
        wafer_diameter_mm: float = 200.0,
        showerhead_diameter_mm: float = 250.0,
        gap_mm: float = 20.0,
        hole_pattern: str = "hexagonal",  # "hexagonal", "concentric", "random"
        hole_density_per_cm2: float = 1.0,
        is_rotating: bool = True,
        rotation_speed_rpm: float = 20.0,
    ):
        self.wafer_diameter_mm = wafer_diameter_mm
        self.showerhead_diameter_mm = showerhead_diameter_mm
        self.gap_mm = gap_mm
        self.hole_pattern = hole_pattern
        self.hole_density = hole_density_per_cm2
        self.is_rotating = is_rotating
        self.rotation_speed_rpm = rotation_speed_rpm

    def calculate_uniformity_factor(
        self,
        radial_position_mm: float,
        pressure_torr: float,
        temperature_c: float,
    ) -> float:
        """
        Calculate thickness uniformity factor at radial position

        Args:
            radial_position_mm: Distance from wafer center (0 to wafer_radius)
            pressure_torr: Process pressure
            temperature_c: Process temperature

        Returns:
            Uniformity factor (1.0 = nominal, <1.0 = thinner, >1.0 = thicker)
        """
        r_norm = radial_position_mm / (self.wafer_diameter_mm / 2.0)

        # Base uniformity from showerhead geometry
        # Center-to-edge ratio depends on gap and pressure
        gap_factor = self.gap_mm / self.wafer_diameter_mm

        # Diffusion length ~ sqrt(D * t_residence) ~ sqrt(D / v)
        # At low pressure, diffusion length increases → better uniformity
        # At high pressure, convection dominates → poorer uniformity
        diffusion_factor = 1.0 / (1.0 + 0.01 * pressure_torr)

        # Showerhead hole pattern effect
        if self.hole_pattern == "hexagonal":
            # Excellent uniformity
            pattern_factor = 1.0 - 0.05 * r_norm**2
        elif self.hole_pattern == "concentric":
            # Good uniformity, slight edge effect
            pattern_factor = 1.0 - 0.1 * r_norm**2
        else:  # "random"
            # Moderate uniformity
            pattern_factor = 1.0 - 0.15 * r_norm**2 + 0.05 * math.sin(r_norm * math.pi)

        # Rotation effect - improves uniformity significantly
        if self.is_rotating:
            rotation_factor = 1.0 - 0.02 * r_norm**2  # Excellent uniformity
        else:
            rotation_factor = pattern_factor

        # Combined uniformity
        uniformity = 1.0 + gap_factor * diffusion_factor * (rotation_factor - 1.0)

        return max(0.5, min(1.5, uniformity))

    def calculate_gas_depletion(
        self,
        radial_position_mm: float,
        deposition_rate_nm_min: float,
        flow_rate_sccm: float,
    ) -> float:
        """
        Calculate gas depletion effect (reduces rate at larger radii)

        Args:
            radial_position_mm: Distance from center
            deposition_rate_nm_min: Base deposition rate
            flow_rate_sccm: Total gas flow rate

        Returns:
            Depletion factor (1.0 = no depletion, <1.0 = depleted)
        """
        # Depletion is minimal in showerhead reactors due to uniform injection
        # But some depletion occurs if flow is insufficient

        area_cm2 = math.pi * (self.wafer_diameter_mm / 10.0)**2
        consumption_rate = deposition_rate_nm_min * area_cm2 / 1000.0  # Arbitrary units

        depletion_ratio = consumption_rate / (flow_rate_sccm + 1.0)

        # Depletion increases slightly with radius
        r_norm = radial_position_mm / (self.wafer_diameter_mm / 2.0)
        depletion = 1.0 - depletion_ratio * 0.1 * r_norm

        return max(0.7, min(1.0, depletion))


class HorizontalFlowReactor:
    """
    Horizontal flow reactor model

    Common in LPCVD and some PECVD.
    Gas flows horizontally across wafer surface.

    Uniformity factors:
    - Flow direction (upstream to downstream depletion)
    - Boundary layer thickness
    - Wafer rotation
    """

    def __init__(
        self,
        wafer_diameter_mm: float = 200.0,
        flow_direction: str = "left_to_right",
        is_rotating: bool = True,
        rotation_speed_rpm: float = 10.0,
        boundary_layer_mm: float = 5.0,
    ):
        self.wafer_diameter_mm = wafer_diameter_mm
        self.flow_direction = flow_direction
        self.is_rotating = is_rotating
        self.rotation_speed_rpm = rotation_speed_rpm
        self.boundary_layer_mm = boundary_layer_mm

    def calculate_uniformity_factor(
        self,
        x_mm: float,
        y_mm: float,
        pressure_torr: float,
        flow_velocity_cm_s: float,
    ) -> float:
        """
        Calculate uniformity at (x,y) position on wafer

        Args:
            x_mm: X position (flow direction)
            y_mm: Y position (perpendicular to flow)
            pressure_torr: Process pressure
            flow_velocity_cm_s: Linear flow velocity

        Returns:
            Uniformity factor
        """
        # Distance along flow direction
        x_norm = x_mm / self.wafer_diameter_mm

        # Gas depletion along flow direction
        # Precursor is consumed as flow travels across wafer
        depletion_factor = math.exp(-0.3 * x_norm)  # Exponential decay

        # Boundary layer effect
        # Thicker boundary layer → slower transport → lower rate
        bl_factor = 1.0 / (1.0 + 0.1 * self.boundary_layer_mm)

        # Rotation smooths out flow direction effects
        if self.is_rotating:
            # Convert to radial coordinate
            r_mm = math.sqrt(x_mm**2 + y_mm**2)
            r_norm = r_mm / (self.wafer_diameter_mm / 2.0)

            # Rotation mixes depletion → radial profile
            rotation_factor = 1.0 - 0.1 * r_norm
            uniformity = 0.7 * rotation_factor + 0.3 * depletion_factor
        else:
            # Without rotation, strong directional gradient
            uniformity = depletion_factor * bl_factor

        return max(0.5, min(1.2, uniformity))

    def calculate_gas_depletion(
        self,
        x_mm: float,
        deposition_rate_nm_min: float,
        flow_rate_sccm: float,
    ) -> float:
        """
        Calculate gas depletion along flow direction

        Significant in horizontal flow - precursor consumed as it flows
        """
        x_norm = x_mm / self.wafer_diameter_mm

        # Stronger depletion at higher deposition rates
        depletion_severity = deposition_rate_nm_min / (flow_rate_sccm + 100.0)

        depletion = math.exp(-depletion_severity * x_norm * 2.0)

        return max(0.6, min(1.0, depletion))


class BatchFurnaceReactor:
    """
    Batch furnace reactor model

    Common in LPCVD (hot-wall tube furnace).
    Multiple wafers in quartz boat, gas flows along tube.

    Uniformity factors:
    - Position along tube (gas depletion)
    - Wafer spacing
    - Temperature gradient along tube
    - Boat-to-boat variation
    """

    def __init__(
        self,
        wafer_diameter_mm: float = 150.0,
        tube_diameter_mm: float = 200.0,
        tube_length_mm: float = 2000.0,
        num_wafers: int = 100,
        wafer_spacing_mm: float = 10.0,
        flow_direction: str = "downstream",  # "downstream", "upstream"
    ):
        self.wafer_diameter_mm = wafer_diameter_mm
        self.tube_diameter_mm = tube_diameter_mm
        self.tube_length_mm = tube_length_mm
        self.num_wafers = num_wafers
        self.wafer_spacing_mm = wafer_spacing_mm
        self.flow_direction = flow_direction

    def calculate_boat_position_factor(
        self,
        wafer_index: int,
        temperature_c: float,
    ) -> float:
        """
        Calculate thickness factor based on wafer position in boat

        Args:
            wafer_index: Wafer position (0 = first, num_wafers-1 = last)
            temperature_c: Nominal temperature

        Returns:
            Position factor (thickness multiplier)
        """
        # Normalize position (0 to 1)
        pos_norm = wafer_index / (self.num_wafers - 1)

        # Temperature gradient along tube
        # Typically center is hottest, ends are cooler
        temp_gradient_c = 5.0  # ±5°C variation
        temp_offset = -temp_gradient_c * (2 * pos_norm - 1)**2
        effective_temp = temperature_c + temp_offset

        # Arrhenius temperature effect on rate
        # rate ~ exp(-Ea/kT)
        Ea_eV = 1.2  # Activation energy
        k_eV_K = 8.617e-5  # Boltzmann constant

        T_nom_K = temperature_c + 273.15
        T_eff_K = effective_temp + 273.15

        temp_factor = math.exp(-Ea_eV / k_eV_K * (1/T_eff_K - 1/T_nom_K))

        # Gas depletion along boat
        # Precursor consumed as it flows past wafers
        if self.flow_direction == "downstream":
            depletion_factor = math.exp(-0.02 * wafer_index)
        else:  # upstream
            depletion_factor = math.exp(-0.02 * (self.num_wafers - wafer_index))

        # Combined factor
        position_factor = temp_factor * depletion_factor

        return position_factor

    def calculate_wiw_uniformity(
        self,
        wafer_index: int,
        radial_position_mm: float,
    ) -> float:
        """
        Calculate WIW (within-wafer) uniformity

        Batch furnaces typically have excellent WIW uniformity
        due to diffusion-controlled regime and wafer stacking symmetry
        """
        r_norm = radial_position_mm / (self.wafer_diameter_mm / 2.0)

        # Excellent WIW uniformity in LPCVD furnaces
        # Slight edge effect due to temperature/flow
        wiw_uniformity = 1.0 - 0.02 * r_norm**2

        # Wafers near tube ends have slightly worse uniformity
        pos_norm = wafer_index / (self.num_wafers - 1)
        end_effect = 0.05 * (2 * pos_norm - 1)**2  # Quadratic, max at ends

        uniformity = wiw_uniformity * (1.0 - end_effect)

        return max(0.95, min(1.05, uniformity))

    def calculate_wtw_uniformity(self) -> float:
        """
        Calculate WTW (wafer-to-wafer) uniformity

        Returns:
            Standard deviation of thickness across boat (as fraction of mean)
        """
        # Simulate thickness for each wafer
        thicknesses = []
        for i in range(self.num_wafers):
            pos_factor = self.calculate_boat_position_factor(i, temperature_c=780.0)
            thicknesses.append(pos_factor)

        thicknesses_arr = np.array(thicknesses)
        mean = np.mean(thicknesses_arr)
        std = np.std(thicknesses_arr)

        wtw_uniformity = (std / mean) * 100.0  # Percent

        return wtw_uniformity


# =============================================================================
# Utility Functions
# =============================================================================

def create_reactor(reactor_type: ReactorType, **kwargs):
    """
    Factory function to create reactor geometry model

    Args:
        reactor_type: Type of reactor
        **kwargs: Reactor-specific parameters

    Returns:
        Reactor geometry instance
    """
    if reactor_type == ReactorType.SHOWERHEAD:
        return ShowerheadReactor(**kwargs)
    elif reactor_type == ReactorType.HORIZONTAL_FLOW:
        return HorizontalFlowReactor(**kwargs)
    elif reactor_type == ReactorType.BATCH_FURNACE:
        return BatchFurnaceReactor(**kwargs)
    else:
        raise ValueError(f"Unsupported reactor type: {reactor_type}")
