"""
Closed-form diffusion solutions using complementary error function (erfc).

Implements:
- Constant source diffusion (surface concentration held constant)
- Limited source diffusion (Gaussian profile from fixed dose)
- Temperature-dependent diffusivity D(T) = D0 * exp(-Ea / (k*T))
- Concentration-dependent diffusivity (optional)
- Junction depth calculation

Will be implemented in Session 2.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray

# Physical constants
k_B = 8.617333e-5  # Boltzmann constant (eV/K)


def constant_source_profile(
    x: NDArray[np.float64],
    t: float,
    T: float,
    D0: float,
    Ea: float,
    Cs: float,
    NA0: float = 0.0
) -> NDArray[np.float64]:
    """
    Calculate concentration profile for constant-source diffusion.
    
    Uses the complementary error function solution:
    N(x,t) = Cs * erfc(x / (2*sqrt(D*t))) + NA0
    
    Args:
        x: Depth array (nm)
        t: Diffusion time (seconds)
        T: Temperature (Celsius)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        Cs: Surface concentration (atoms/cm³)
        NA0: Background concentration (atoms/cm³)
    
    Returns:
        Concentration profile (atoms/cm³)
    
    Status: STUB - To be implemented in Session 2
    """
    raise NotImplementedError("Session 2: Constant source profile")


def limited_source_profile(
    x: NDArray[np.float64],
    t: float,
    T: float,
    D0: float,
    Ea: float,
    Q: float,
    NA0: float = 0.0,
    limited_model: str = "gaussian"
) -> NDArray[np.float64]:
    """
    Calculate concentration profile for limited-source diffusion.
    
    Uses Gaussian solution:
    N(x,t) = (Q / sqrt(π*D*t)) * exp(-x² / (4*D*t)) + NA0
    
    Args:
        x: Depth array (nm)
        t: Diffusion time (seconds)
        T: Temperature (Celsius)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        Q: Total dose (atoms/cm²)
        NA0: Background concentration (atoms/cm³)
        limited_model: Model type ("gaussian", "delta")
    
    Returns:
        Concentration profile (atoms/cm³)
    
    Status: STUB - To be implemented in Session 2
    """
    raise NotImplementedError("Session 2: Limited source profile")


def diffusivity(
    T: float,
    D0: float,
    Ea: float,
    C: Optional[NDArray[np.float64]] = None,
    alpha: float = 0.0,
    m: float = 1.0
) -> Union[float, NDArray[np.float64]]:
    """
    Calculate diffusivity D(T) or D(T,C).
    
    Temperature-dependent:
    D(T) = D0 * exp(-Ea / (k*T))
    
    Concentration-dependent (optional):
    D(T,C) = D(T) * (1 + alpha * C^m)
    
    Args:
        T: Temperature (Celsius)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        C: Concentration array (atoms/cm³), optional
        alpha: Concentration factor
        m: Concentration exponent
    
    Returns:
        Diffusivity (cm²/s) - scalar or array
    
    Status: STUB - To be implemented in Session 2
    """
    raise NotImplementedError("Session 2: Diffusivity calculation")


def junction_depth(
    C_profile: NDArray[np.float64],
    x: NDArray[np.float64],
    N_background: float
) -> float:
    """
    Calculate junction depth where C(xj) = N_background.
    
    Uses linear interpolation to find the crossing point.
    
    Args:
        C_profile: Concentration profile (atoms/cm³)
        x: Depth array (nm)
        N_background: Background concentration (atoms/cm³)
    
    Returns:
        Junction depth xj (nm)
    
    Status: STUB - To be implemented in Session 2
    """
    raise NotImplementedError("Session 2: Junction depth calculation")


def sheet_resistance_estimate(
    C_profile: NDArray[np.float64],
    x: NDArray[np.float64],
    mobility: float = 1000.0  # cm²/V·s, approximate
) -> float:
    """
    Estimate sheet resistance from dopant profile.
    
    Rs = 1 / (q * integral(μ(x) * N(x) dx))
    
    Args:
        C_profile: Concentration profile (atoms/cm³)
        x: Depth array (nm)
        mobility: Carrier mobility (cm²/V·s)
    
    Returns:
        Sheet resistance (Ω/□)
    
    Status: STUB - To be implemented in Session 2
    """
    raise NotImplementedError("Session 2: Sheet resistance estimation")


__all__ = [
    "constant_source_profile",
    "limited_source_profile",
    "diffusivity",
    "junction_depth",
    "sheet_resistance_estimate",
]
