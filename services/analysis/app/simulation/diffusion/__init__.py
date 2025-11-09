"""
Diffusion Simulation Module

Provides dopant diffusion simulation capabilities:
- fick_fd: Finite difference solver for Fick's second law of diffusion (Session 3)
- massoud: Advanced diffusion model considering clustering effects (Session 1)
- erfc: Complementary error function based analytical solutions (Session 2) ✓
- segregation: Dopant segregation at interfaces (Session 1)
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
]
