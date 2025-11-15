"""
SPC Series Data Structures

Defines time-series data for SPC monitoring of CVD film properties.
"""

import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class SPCMetric(str, Enum):
    """SPC metric types for CVD film properties"""
    THICKNESS_MEAN = "thickness_mean_nm"
    THICKNESS_UNIFORMITY = "thickness_uniformity_pct"
    STRESS_MEAN = "stress_mpa_mean"
    STRESS_ABS = "stress_mpa_abs"
    ADHESION_SCORE = "adhesion_score"
    ADHESION_CLASS_DIST = "adhesion_class_distribution"


@dataclass
class SPCDataPoint:
    """Single SPC measurement point"""
    timestamp: datetime
    value: float
    run_id: str
    tool_id: Optional[str] = None
    recipe_id: Optional[str] = None
    film_material: Optional[str] = None
    lot_id: Optional[str] = None
    wafer_id: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SPCSeries:
    """
    Time-series data for SPC monitoring

    Tracks a specific metric (e.g., thickness_mean) over time
    for a specific tool/recipe/film combination.
    """
    metric: SPCMetric
    tool_id: Optional[str] = None
    recipe_id: Optional[str] = None
    film_material: Optional[str] = None

    # Data points
    data_points: List[SPCDataPoint] = field(default_factory=list)

    # Control limits (to be calculated by SPC charts)
    center_line: Optional[float] = None
    upper_control_limit: Optional[float] = None
    lower_control_limit: Optional[float] = None
    upper_warning_limit: Optional[float] = None
    lower_warning_limit: Optional[float] = None

    # Statistical parameters
    process_mean: Optional[float] = None
    process_std: Optional[float] = None

    def add_point(self, point: SPCDataPoint):
        """Add a data point to the series"""
        self.data_points.append(point)

    def get_values(self) -> np.ndarray:
        """Get array of values"""
        return np.array([p.value for p in self.data_points])

    def get_timestamps(self) -> List[datetime]:
        """Get list of timestamps"""
        return [p.timestamp for p in self.data_points]

    def get_recent_points(self, n: int) -> List[SPCDataPoint]:
        """Get the n most recent data points"""
        return self.data_points[-n:] if len(self.data_points) >= n else self.data_points

    def get_recent_values(self, n: int) -> np.ndarray:
        """Get array of n most recent values"""
        recent = self.get_recent_points(n)
        return np.array([p.value for p in recent])

    def calculate_statistics(self):
        """Calculate process statistics from data"""
        if len(self.data_points) == 0:
            return

        values = self.get_values()
        self.process_mean = float(np.mean(values))
        self.process_std = float(np.std(values, ddof=1))

    def filter_by_tool(self, tool_id: str) -> 'SPCSeries':
        """Create new series filtered by tool"""
        filtered = SPCSeries(
            metric=self.metric,
            tool_id=tool_id,
            recipe_id=self.recipe_id,
            film_material=self.film_material,
        )
        filtered.data_points = [p for p in self.data_points if p.tool_id == tool_id]
        filtered.calculate_statistics()
        return filtered

    def filter_by_recipe(self, recipe_id: str) -> 'SPCSeries':
        """Create new series filtered by recipe"""
        filtered = SPCSeries(
            metric=self.metric,
            tool_id=self.tool_id,
            recipe_id=recipe_id,
            film_material=self.film_material,
        )
        filtered.data_points = [p for p in self.data_points if p.recipe_id == recipe_id]
        filtered.calculate_statistics()
        return filtered

    def filter_by_material(self, film_material: str) -> 'SPCSeries':
        """Create new series filtered by film material"""
        filtered = SPCSeries(
            metric=self.metric,
            tool_id=self.tool_id,
            recipe_id=self.recipe_id,
            film_material=film_material,
        )
        filtered.data_points = [p for p in self.data_points if p.film_material == film_material]
        filtered.calculate_statistics()
        return filtered


def create_spc_series(
    metric: SPCMetric,
    values: List[float],
    timestamps: List[datetime],
    run_ids: List[str],
    tool_id: Optional[str] = None,
    recipe_id: Optional[str] = None,
    film_material: Optional[str] = None,
    **kwargs,
) -> SPCSeries:
    """
    Factory function to create SPC series from lists

    Args:
        metric: SPC metric type
        values: List of measurement values
        timestamps: List of timestamps
        run_ids: List of run IDs
        tool_id: Optional tool ID
        recipe_id: Optional recipe ID
        film_material: Optional film material
        **kwargs: Additional metadata for all points

    Returns:
        SPCSeries object
    """
    if not (len(values) == len(timestamps) == len(run_ids)):
        raise ValueError("values, timestamps, and run_ids must have same length")

    series = SPCSeries(
        metric=metric,
        tool_id=tool_id,
        recipe_id=recipe_id,
        film_material=film_material,
    )

    for value, timestamp, run_id in zip(values, timestamps, run_ids):
        point = SPCDataPoint(
            timestamp=timestamp,
            value=value,
            run_id=run_id,
            tool_id=tool_id,
            recipe_id=recipe_id,
            film_material=film_material,
            metadata=kwargs,
        )
        series.add_point(point)

    series.calculate_statistics()

    return series
