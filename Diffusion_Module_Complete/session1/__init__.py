"""
Diffusion & Thermal Oxidation Module for SemiconductorLab Platform.

This module provides comprehensive dopant diffusion and thermal oxidation modeling
capabilities including:

- Closed-form diffusion solutions (erfc-based)
- Numerical diffusion solvers (Fick's 2nd law)
- Thermal oxidation (Deal-Grove + Massoud)
- Segregation and moving boundary coupling
- Statistical Process Control (SPC) for furnace operations
- Virtual Metrology (VM) for predictive process control
- Parameter calibration with uncertainty quantification

Version: 1.0.0
Author: SemiconductorLab Team
Date: November 2025
"""

__version__ = "1.0.0"
__author__ = "SemiconductorLab Team"

from typing import Optional

# Core physics modules (Session 2-5)
# These will be populated in subsequent sessions
__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Core modules (to be added)
    # "constant_source_profile",  # Session 2
    # "limited_source_profile",   # Session 2
    # "Fick1D",                    # Session 3
    # "DealGrove",                 # Session 4
    # "Massoud",                   # Session 4
    # "SegregationModel",          # Session 5
    
    # SPC modules (Session 7)
    # "WesternElectricRules",
    # "EWMA",
    # "CUSUM",
    # "BOCPD",
    
    # ML modules (Session 8-9)
    # "VirtualMetrology",
    # "ParameterCalibration",
]

def get_version() -> str:
    """Return the current module version."""
    return __version__

def get_info() -> dict:
    """Return module information."""
    return {
        "name": "diffusion_oxidation",
        "version": __version__,
        "author": __author__,
        "description": "Dopant diffusion and thermal oxidation analysis for semiconductor manufacturing",
        "sessions_completed": 1,
        "total_sessions": 12,
    }
