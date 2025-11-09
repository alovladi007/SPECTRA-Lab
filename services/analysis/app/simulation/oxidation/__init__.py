"""
Oxidation Simulation Module

Provides silicon oxidation simulation capabilities:
- deal_grove: Deal-Grove model for thermal oxidation of silicon (Session 4) ✓
- massoud: Thin-oxide corrections for enhanced accuracy (Session 4) ✓
"""

from .deal_grove import (
    get_rate_constants,
    thickness_at_time,
    time_to_thickness,
    growth_rate,
    linear_regime_thickness,
)

from .massoud import (
    get_correction_params,
    thickness_with_correction,
    correction_magnitude,
    is_correction_significant,
    time_to_thickness_with_correction,
)

__all__ = [
    # Deal-Grove model (Session 4)
    "get_rate_constants",
    "thickness_at_time",
    "time_to_thickness",
    "growth_rate",
    "linear_regime_thickness",
    # Massoud thin-oxide corrections (Session 4)
    "get_correction_params",
    "thickness_with_correction",
    "correction_magnitude",
    "is_correction_significant",
    "time_to_thickness_with_correction",
]
