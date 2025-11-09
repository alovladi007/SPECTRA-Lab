"""
Diffusion Simulation Module

Provides dopant diffusion simulation capabilities:
- fick_fd: Finite difference solver for Fick's second law of diffusion (Session 3) ✓
- massoud: Advanced diffusion model considering clustering effects (Session 1)
- erfc: Complementary error function based analytical solutions (Session 2) ✓
- segregation: Dopant segregation at interfaces with moving boundary tracking (Session 5) ✓
"""

from .erfc import (
    diffusivity,
    constant_source_profile,
    limited_source_profile,
    junction_depth,
    sheet_resistance_estimate,
    two_step_diffusion,
    quick_profile_constant_source,
    quick_profile_limited_source,
)

from .fick_fd import (
    Fick1D,
    quick_solve_constant_D,
)

from .segregation import (
    SegregationModel,
    MovingBoundaryTracker,
    arsenic_pile_up_demo,
    boron_depletion_demo,
    SEGREGATION_COEFFICIENTS,
)

__all__ = [
    # ERFC diffusion functions (Session 2)
    "diffusivity",
    "constant_source_profile",
    "limited_source_profile",
    "junction_depth",
    "sheet_resistance_estimate",
    "two_step_diffusion",
    "quick_profile_constant_source",
    "quick_profile_limited_source",
    # Fick FD solver (Session 3)
    "Fick1D",
    "quick_solve_constant_D",
    # Segregation & moving boundary (Session 5)
    "SegregationModel",
    "MovingBoundaryTracker",
    "arsenic_pile_up_demo",
    "boron_depletion_demo",
    "SEGREGATION_COEFFICIENTS",
]
