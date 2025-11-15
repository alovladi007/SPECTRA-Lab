"""
Statistical Process Control (SPC) Module

Provides SPC charts and control rules for monitoring CVD film properties:
- Thickness (mean, uniformity)
- Stress (mean, absolute value)
- Adhesion (score, class distribution)

Includes:
- X-bar/R charts (mean and range)
- EWMA charts (exponentially weighted moving average)
- CUSUM charts (cumulative sum)
- Western Electric rules for anomaly detection
"""

from .charts import (
    XBarRChart,
    EWMAChart,
    CUSUMChart,
    SPCChart,
    SPCChartType,
)

from .rules import (
    WesternElectricRules,
    SPCViolation,
    SPCViolationType,
    check_all_rules,
)

from .series import (
    SPCSeries,
    SPCDataPoint,
    SPCMetric,
    create_spc_series,
)

__all__ = [
    # Charts
    "XBarRChart",
    "EWMAChart",
    "CUSUMChart",
    "SPCChart",
    "SPCChartType",

    # Rules
    "WesternElectricRules",
    "SPCViolation",
    "SPCViolationType",
    "check_all_rules",

    # Series
    "SPCSeries",
    "SPCDataPoint",
    "SPCMetric",
    "create_spc_series",
]
