"""Process-specific SPC monitors for Ion Implantation and RTP.

Provides default monitoring configurations and real-time SPC tracking
for critical process parameters.
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

from .charts import XbarRChart, EWMAChart, CUSUMChart, SPCAlert, AlertSeverity


# ============================================================================
# Process Parameter Definitions
# ============================================================================

class IonParameter(Enum):
    """Ion implantation parameters for SPC monitoring."""
    BEAM_CURRENT_MA = "beam_current_mA"
    DOSE_UNIFORMITY_PCT = "dose_uniformity_pct"
    SOURCE_PRESSURE_MTORR = "source_pressure_mtorr"
    ANALYZER_PRESSURE_MTORR = "analyzer_pressure_mtorr"
    PROCESS_PRESSURE_MTORR = "process_pressure_mtorr"
    BEAMLINE_PRESSURE_MTORR = "beamline_pressure_mtorr"
    ANALYZER_FIELD_T = "analyzer_field_T"
    EXTRACTION_VOLTAGE_KV = "extraction_voltage_kV"
    ACCELERATION_VOLTAGE_KV = "acceleration_voltage_kV"


class RTPParameter(Enum):
    """RTP parameters for SPC monitoring."""
    RAMP_TRACKING_ERROR_C = "ramp_tracking_error_C"
    OVERSHOOT_PCT = "overshoot_pct"
    LAMP_POWER_PCT = "lamp_power_pct"
    EMISSIVITY_DRIFT = "emissivity_drift"
    GAS_FLOW_DEVIATION_SCCM = "gas_flow_deviation_sccm"
    CHAMBER_PRESSURE_TORR = "chamber_pressure_torr"
    DWELL_STABILITY_C = "dwell_stability_C"
    COOLING_RATE_C_PER_S = "cooling_rate_C_per_s"


@dataclass
class SPCConfiguration:
    """SPC monitoring configuration for a parameter."""
    chart_type: str  # "xbar_r", "ewma", "cusum"
    target: Optional[float] = None
    ucl: Optional[float] = None
    lcl: Optional[float] = None

    # Chart-specific parameters
    subgroup_size: int = 5  # For X-bar/R
    lambda_weight: float = 0.2  # For EWMA
    k_slack: float = 0.5  # For CUSUM
    h_decision: float = 5.0  # For CUSUM

    # Alert configuration
    alert_enabled: bool = True
    critical_threshold: Optional[float] = None


# ============================================================================
# Ion Implantation Monitor
# ============================================================================

class IonImplantMonitor:
    """SPC monitor for Ion Implantation processes.

    Default monitoring parameters:
    - Beam current stability (EWMA)
    - Dose uniformity (X-bar/R)
    - Chamber pressure (CUSUM)
    - Analyzer field (EWMA)
    """

    DEFAULT_CONFIG = {
        IonParameter.BEAM_CURRENT_MA: SPCConfiguration(
            chart_type="ewma",
            lambda_weight=0.2,
            alert_enabled=True,
            critical_threshold=0.15  # ±15% deviation is critical
        ),
        IonParameter.DOSE_UNIFORMITY_PCT: SPCConfiguration(
            chart_type="xbar_r",
            subgroup_size=5,
            target=95.0,  # Target 95% uniformity
            lcl=90.0,     # Critical below 90%
            alert_enabled=True
        ),
        IonParameter.SOURCE_PRESSURE_MTORR: SPCConfiguration(
            chart_type="cusum",
            k_slack=0.5,
            h_decision=5.0,
            alert_enabled=True
        ),
        IonParameter.ANALYZER_PRESSURE_MTORR: SPCConfiguration(
            chart_type="cusum",
            k_slack=0.5,
            h_decision=5.0,
            alert_enabled=True
        ),
        IonParameter.PROCESS_PRESSURE_MTORR: SPCConfiguration(
            chart_type="cusum",
            k_slack=0.5,
            h_decision=5.0,
            alert_enabled=True
        ),
        IonParameter.ANALYZER_FIELD_T: SPCConfiguration(
            chart_type="ewma",
            lambda_weight=0.15,
            alert_enabled=True
        ),
        IonParameter.EXTRACTION_VOLTAGE_KV: SPCConfiguration(
            chart_type="ewma",
            lambda_weight=0.2,
            alert_enabled=True
        ),
    }

    def __init__(self, equipment_id: str, custom_config: Optional[Dict] = None):
        """Initialize Ion Implant SPC monitor.

        Args:
            equipment_id: Equipment identifier
            custom_config: Optional custom SPC configurations
        """
        self.equipment_id = equipment_id
        self.config = self.DEFAULT_CONFIG.copy()

        if custom_config:
            self.config.update(custom_config)

        # Create charts for each parameter
        self.charts: Dict[IonParameter, any] = {}
        self._initialize_charts()

        # Alert history
        self.all_alerts: List[SPCAlert] = []

    def _initialize_charts(self):
        """Initialize SPC charts based on configuration."""
        for param, config in self.config.items():
            param_name = f"{self.equipment_id}_{param.value}"

            if config.chart_type == "ewma":
                self.charts[param] = EWMAChart(
                    parameter_name=param_name,
                    lambda_weight=config.lambda_weight
                )
            elif config.chart_type == "cusum":
                self.charts[param] = CUSUMChart(
                    parameter_name=param_name,
                    target=config.target,
                    k=config.k_slack,
                    h=config.h_decision
                )
            elif config.chart_type == "xbar_r":
                self.charts[param] = XbarRChart(
                    parameter_name=param_name,
                    subgroup_size=config.subgroup_size
                )

    def update(self, measurements: Dict[IonParameter, float], timestamp: Optional[float] = None):
        """Update SPC charts with new measurements.

        Args:
            measurements: Dictionary of parameter measurements
            timestamp: Measurement timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        new_alerts = []

        for param, value in measurements.items():
            if param not in self.charts:
                continue

            chart = self.charts[param]

            # Update chart
            if isinstance(chart, XbarRChart):
                # For X-bar/R, need subgroup - use single point as subgroup of 1
                # In practice, you'd collect multiple samples
                subgroup = np.array([value])
                chart.add_subgroup(subgroup, timestamp)
            else:
                chart.add_point(value, timestamp)

            # Collect alerts
            if chart.state.alerts:
                new_alerts.extend(chart.state.alerts[-10:])  # Get recent alerts

        # Store alerts
        self.all_alerts.extend(new_alerts)

        return new_alerts

    def get_active_alerts(self, lookback_seconds: float = 300) -> List[SPCAlert]:
        """Get active alerts within lookback window.

        Args:
            lookback_seconds: Time window for active alerts

        Returns:
            List of active SPCAlert objects
        """
        current_time = time.time()
        cutoff_time = current_time - lookback_seconds

        active = [
            alert for alert in self.all_alerts
            if alert.timestamp >= cutoff_time and not alert.is_duplicate
        ]

        return active

    def get_control_status(self) -> Dict[IonParameter, bool]:
        """Get in-control status for each parameter.

        Returns:
            Dictionary mapping parameters to in-control status (True = in control)
        """
        status = {}

        for param, chart in self.charts.items():
            # Check if any recent alerts
            recent_alerts = [
                a for a in chart.state.alerts[-5:]
                if not a.is_duplicate
            ]
            status[param] = len(recent_alerts) == 0

        return status

    def get_summary(self) -> Dict:
        """Get monitoring summary statistics."""
        summary = {
            "equipment_id": self.equipment_id,
            "monitored_parameters": len(self.charts),
            "total_alerts": len(self.all_alerts),
            "active_alerts": len(self.get_active_alerts()),
            "parameters_in_control": sum(self.get_control_status().values()),
            "parameters_out_of_control": len(self.charts) - sum(self.get_control_status().values())
        }

        return summary


# ============================================================================
# RTP Monitor
# ============================================================================

class RTPMonitor:
    """SPC monitor for RTP processes.

    Default monitoring parameters:
    - Ramp tracking error ΔT (EWMA)
    - Overshoot % (X-bar/R)
    - Lamp power % (CUSUM)
    - Emissivity drift (CUSUM)
    - Gas flow deviations (EWMA)
    """

    DEFAULT_CONFIG = {
        RTPParameter.RAMP_TRACKING_ERROR_C: SPCConfiguration(
            chart_type="ewma",
            lambda_weight=0.2,
            target=0.0,
            critical_threshold=10.0,  # ±10°C is critical
            alert_enabled=True
        ),
        RTPParameter.OVERSHOOT_PCT: SPCConfiguration(
            chart_type="xbar_r",
            subgroup_size=5,
            target=0.0,
            ucl=5.0,  # >5% overshoot is bad
            alert_enabled=True
        ),
        RTPParameter.LAMP_POWER_PCT: SPCConfiguration(
            chart_type="cusum",
            k_slack=0.5,
            h_decision=5.0,
            target=50.0,  # Nominal lamp power
            alert_enabled=True
        ),
        RTPParameter.EMISSIVITY_DRIFT: SPCConfiguration(
            chart_type="cusum",
            k_slack=0.3,  # More sensitive to drift
            h_decision=4.0,
            target=0.0,
            alert_enabled=True
        ),
        RTPParameter.GAS_FLOW_DEVIATION_SCCM: SPCConfiguration(
            chart_type="ewma",
            lambda_weight=0.2,
            target=0.0,
            critical_threshold=500.0,  # ±500 sccm is critical
            alert_enabled=True
        ),
        RTPParameter.CHAMBER_PRESSURE_TORR: SPCConfiguration(
            chart_type="cusum",
            k_slack=0.5,
            h_decision=5.0,
            alert_enabled=True
        ),
        RTPParameter.DWELL_STABILITY_C: SPCConfiguration(
            chart_type="ewma",
            lambda_weight=0.25,
            target=0.0,  # Want minimal variation
            ucl=5.0,  # >5°C std is bad
            alert_enabled=True
        ),
        RTPParameter.COOLING_RATE_C_PER_S: SPCConfiguration(
            chart_type="ewma",
            lambda_weight=0.2,
            alert_enabled=True
        ),
    }

    def __init__(self, equipment_id: str, custom_config: Optional[Dict] = None):
        """Initialize RTP SPC monitor.

        Args:
            equipment_id: Equipment identifier
            custom_config: Optional custom SPC configurations
        """
        self.equipment_id = equipment_id
        self.config = self.DEFAULT_CONFIG.copy()

        if custom_config:
            self.config.update(custom_config)

        # Create charts
        self.charts: Dict[RTPParameter, any] = {}
        self._initialize_charts()

        # Alert history
        self.all_alerts: List[SPCAlert] = []

        # Recipe tracking
        self.current_recipe_id: Optional[str] = None
        self.recipe_alerts: Dict[str, List[SPCAlert]] = {}

    def _initialize_charts(self):
        """Initialize SPC charts based on configuration."""
        for param, config in self.config.items():
            param_name = f"{self.equipment_id}_{param.value}"

            if config.chart_type == "ewma":
                self.charts[param] = EWMAChart(
                    parameter_name=param_name,
                    lambda_weight=config.lambda_weight
                )
            elif config.chart_type == "cusum":
                self.charts[param] = CUSUMChart(
                    parameter_name=param_name,
                    target=config.target,
                    k=config.k_slack,
                    h=config.h_decision
                )
            elif config.chart_type == "xbar_r":
                self.charts[param] = XbarRChart(
                    parameter_name=param_name,
                    subgroup_size=config.subgroup_size
                )

    def start_recipe(self, recipe_id: str):
        """Start monitoring a new recipe.

        Args:
            recipe_id: Recipe identifier
        """
        self.current_recipe_id = recipe_id
        self.recipe_alerts[recipe_id] = []

    def update(self, measurements: Dict[RTPParameter, float], timestamp: Optional[float] = None):
        """Update SPC charts with new measurements.

        Args:
            measurements: Dictionary of parameter measurements
            timestamp: Measurement timestamp (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()

        new_alerts = []

        for param, value in measurements.items():
            if param not in self.charts:
                continue

            chart = self.charts[param]

            # Update chart
            if isinstance(chart, XbarRChart):
                # For X-bar/R, use single measurement as subgroup
                subgroup = np.array([value])
                chart.add_subgroup(subgroup, timestamp)
            else:
                chart.add_point(value, timestamp)

            # Collect alerts
            if chart.state.alerts:
                new_alerts.extend(chart.state.alerts[-10:])

        # Store alerts
        self.all_alerts.extend(new_alerts)

        # Associate with current recipe
        if self.current_recipe_id:
            self.recipe_alerts[self.current_recipe_id].extend(new_alerts)

        return new_alerts

    def end_recipe(self) -> List[SPCAlert]:
        """End current recipe and return alerts.

        Returns:
            List of alerts from completed recipe
        """
        if not self.current_recipe_id:
            return []

        alerts = self.recipe_alerts.get(self.current_recipe_id, [])
        self.current_recipe_id = None

        return alerts

    def get_recipe_performance(self, recipe_id: str) -> Dict:
        """Get SPC performance metrics for a recipe.

        Args:
            recipe_id: Recipe identifier

        Returns:
            Dictionary with performance metrics
        """
        alerts = self.recipe_alerts.get(recipe_id, [])

        critical_count = sum(1 for a in alerts if a.severity == AlertSeverity.CRITICAL)
        warning_count = sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)

        return {
            "recipe_id": recipe_id,
            "total_alerts": len(alerts),
            "critical_alerts": critical_count,
            "warning_alerts": warning_count,
            "alert_rate": len(alerts) / max(1, len(alerts)),  # Normalize by time
            "passed": critical_count == 0
        }

    def get_active_alerts(self, lookback_seconds: float = 300) -> List[SPCAlert]:
        """Get active alerts within lookback window."""
        current_time = time.time()
        cutoff_time = current_time - lookback_seconds

        active = [
            alert for alert in self.all_alerts
            if alert.timestamp >= cutoff_time and not alert.is_duplicate
        ]

        return active

    def get_control_status(self) -> Dict[RTPParameter, bool]:
        """Get in-control status for each parameter."""
        status = {}

        for param, chart in self.charts.items():
            recent_alerts = [
                a for a in chart.state.alerts[-5:]
                if not a.is_duplicate
            ]
            status[param] = len(recent_alerts) == 0

        return status

    def get_summary(self) -> Dict:
        """Get monitoring summary statistics."""
        summary = {
            "equipment_id": self.equipment_id,
            "monitored_parameters": len(self.charts),
            "total_alerts": len(self.all_alerts),
            "active_alerts": len(self.get_active_alerts()),
            "parameters_in_control": sum(self.get_control_status().values()),
            "parameters_out_of_control": len(self.charts) - sum(self.get_control_status().values()),
            "recipes_monitored": len(self.recipe_alerts)
        }

        return summary


# Export
__all__ = [
    "IonImplantMonitor",
    "RTPMonitor",
    "IonParameter",
    "RTPParameter",
    "SPCConfiguration",
]
