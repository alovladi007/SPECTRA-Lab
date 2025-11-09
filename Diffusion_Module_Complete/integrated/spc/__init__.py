"""SPC (Statistical Process Control) module - Session 7."""

from .rules import (
    SPCRulesEngine,
    ControlLimits,
    RuleViolation,
    RuleViolationType,
    Severity,
    quick_rule_check,
)

from .ewma import (
    EWMAChart,
    EWMALimits,
    EWMAViolation,
    quick_ewma_check,
    optimal_ewma_params,
)

from .cusum import (
    CUSUMChart,
    CUSUMViolation,
    FastInitialResponse_CUSUM,
    quick_cusum_check,
    optimal_cusum_params,
)

from .changepoint import (
    BOCPD,
    ChangePoint,
    SimplifiedBOCPD,
    quick_bocpd_check,
    constant_hazard,
    discrete_uniform_hazard,
    gaussian_hazard,
)

__all__ = [
    # Rules
    "SPCRulesEngine",
    "ControlLimits",
    "RuleViolation",
    "RuleViolationType",
    "Severity",
    "quick_rule_check",
    # EWMA
    "EWMAChart",
    "EWMALimits",
    "EWMAViolation",
    "quick_ewma_check",
    "optimal_ewma_params",
    # CUSUM
    "CUSUMChart",
    "CUSUMViolation",
    "FastInitialResponse_CUSUM",
    "quick_cusum_check",
    "optimal_cusum_params",
    # Change Points
    "BOCPD",
    "ChangePoint",
    "SimplifiedBOCPD",
    "quick_bocpd_check",
    "constant_hazard",
    "discrete_uniform_hazard",
    "gaussian_hazard",
]
