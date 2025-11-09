"""
Closed-form diffusion solutions using complementary error function (erfc).

Implements:
- Constant source diffusion (surface concentration held constant)
- Limited source diffusion (Gaussian profile from fixed dose)
- Temperature-dependent diffusivity D(T) = D0 * exp(-Ea / (k*T))
- Concentration-dependent diffusivity (optional)
- Junction depth calculation

Session 2 - IMPLEMENTED
"""

from typing import Optional, Tuple, Union, Literal
import numpy as np
from numpy.typing import NDArray
from scipy.special import erfc, erf
from scipy.interpolate import interp1d
import warnings

# Physical constants
k_B = 8.617333e-5  # Boltzmann constant (eV/K)
q = 1.602176634e-19  # Elementary charge (C)


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
        alpha: Concentration factor (dimensionless)
        m: Concentration exponent (dimensionless)
    
    Returns:
        Diffusivity (cm²/s) - scalar or array
    
    Examples:
        >>> # Temperature-dependent only
        >>> D = diffusivity(T=1000, D0=0.76, Ea=3.46)
        >>> print(f"D = {D:.2e} cm²/s")
        
        >>> # Concentration-dependent
        >>> C = np.array([1e19, 1e20])
        >>> D = diffusivity(T=1000, D0=0.76, Ea=3.46, C=C, alpha=1e-20, m=1)
    
    Status: IMPLEMENTED - Session 2
    """
    # Convert Celsius to Kelvin
    T_kelvin = T + 273.15
    
    # Arrhenius temperature dependence
    D_T = D0 * np.exp(-Ea / (k_B * T_kelvin))
    
    # Concentration dependence (if requested)
    if C is not None and alpha != 0.0:
        # Avoid overflow for very high concentrations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            concentration_factor = 1.0 + alpha * np.power(C, m)
            # Clip to reasonable values
            concentration_factor = np.clip(concentration_factor, 0.1, 100.0)
        return D_T * concentration_factor
    else:
        return D_T


def constant_source_profile(
    x: NDArray[np.float64],
    t: float,
    T: float,
    D0: float,
    Ea: float,
    Cs: float,
    NA0: float = 0.0,
    use_concentration_dependent: bool = False,
    alpha: float = 0.0,
    m: float = 1.0
) -> NDArray[np.float64]:
    """
    Calculate concentration profile for constant-source diffusion.
    
    Uses the complementary error function solution:
    N(x,t) = Cs * erfc(x / (2*sqrt(D*t))) + NA0
    
    This solution applies when:
    - Surface concentration is held constant (e.g., dopant gas flow)
    - Substrate is semi-infinite
    - Diffusivity is constant (or weakly concentration-dependent)
    
    Physical interpretation:
    - At surface (x=0): N = Cs (boundary condition)
    - At depth: Concentration decays as erfc
    - Junction depth: where N(xj) = NA0
    
    Args:
        x: Depth array (nm)
        t: Diffusion time (seconds)
        T: Temperature (Celsius)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        Cs: Surface concentration (atoms/cm³)
        NA0: Background concentration (atoms/cm³)
        use_concentration_dependent: Enable D(C) (default: False)
        alpha: Concentration factor for D(C)
        m: Concentration exponent for D(C)
    
    Returns:
        Concentration profile (atoms/cm³)
    
    Examples:
        >>> # Boron diffusion at 1000°C for 30 minutes
        >>> x = np.linspace(0, 1000, 1000)  # nm
        >>> t = 30 * 60  # 30 min -> seconds
        >>> C = constant_source_profile(
        ...     x=x, t=t, T=1000,
        ...     D0=0.76, Ea=3.46,
        ...     Cs=1e20, NA0=1e15
        ... )
        >>> print(f"Surface: {C[0]:.2e} cm⁻³")
        >>> print(f"At 100nm: {C[100]:.2e} cm⁻³")
    
    References:
        - Sze & Lee, "Semiconductor Devices" (2012), Section 1.5
        - Fair & Tsai, J. Electrochem. Soc. 124, 1107 (1977)
    
    Status: IMPLEMENTED - Session 2
    """
    # Input validation
    if t <= 0:
        raise ValueError("Time must be positive")
    if T < 600 or T > 1400:
        warnings.warn(f"Temperature {T}°C outside typical range 600-1400°C")
    if Cs <= NA0:
        raise ValueError("Surface concentration must exceed background")
    
    # Convert x from nm to cm for diffusion calculation
    x_cm = x * 1e-7  # nm -> cm
    
    # Calculate diffusivity (temperature-dependent)
    if use_concentration_dependent:
        # For concentration-dependent D, use average between Cs and NA0
        # This is an approximation for the analytical solution
        C_avg = np.linspace(Cs, NA0, len(x))
        D = diffusivity(T, D0, Ea, C_avg, alpha, m)
        D_eff = np.mean(D)  # Use effective average D
    else:
        D_eff = diffusivity(T, D0, Ea)
    
    # Characteristic diffusion length: sqrt(D*t)
    sqrt_Dt = np.sqrt(D_eff * t)
    
    # Complementary error function solution
    # N(x,t) = Cs * erfc(x / (2*sqrt(D*t))) + NA0
    argument = x_cm / (2 * sqrt_Dt)
    
    # Compute profile
    C = Cs * erfc(argument) + NA0
    
    # Ensure physical constraints
    C = np.clip(C, NA0, Cs)
    
    return C


def limited_source_profile(
    x: NDArray[np.float64],
    t: float,
    T: float,
    D0: float,
    Ea: float,
    Q: float,
    NA0: float = 0.0,
    limited_model: Literal["gaussian", "delta"] = "gaussian"
) -> NDArray[np.float64]:
    """
    Calculate concentration profile for limited-source diffusion.
    
    Uses Gaussian solution:
    N(x,t) = (Q / sqrt(π*D*t)) * exp(-x² / (4*D*t)) + NA0
    
    This solution applies when:
    - Fixed total dose Q is deposited (e.g., ion implantation)
    - No additional dopant is supplied
    - Initial profile is delta function or thin layer
    
    Physical interpretation:
    - Total dose: Q = integral(N(x) dx) (atoms/cm²)
    - Peak at surface (x=0): N_peak = Q / sqrt(π*D*t)
    - Profile spreads and peak decreases with time
    - Maintains constant total dose
    
    Args:
        x: Depth array (nm)
        t: Diffusion time (seconds)
        T: Temperature (Celsius)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        Q: Total dose (atoms/cm²)
        NA0: Background concentration (atoms/cm³)
        limited_model: Model type ("gaussian" or "delta")
    
    Returns:
        Concentration profile (atoms/cm³)
    
    Examples:
        >>> # Phosphorus implant drive-in at 950°C for 20 minutes
        >>> x = np.linspace(0, 500, 500)  # nm
        >>> t = 20 * 60  # seconds
        >>> C = limited_source_profile(
        ...     x=x, t=t, T=950,
        ...     D0=3.85, Ea=3.66,
        ...     Q=1e14, NA0=1e15
        ... )
        >>> print(f"Peak concentration: {C[0]:.2e} cm⁻³")
    
    References:
        - Grove, "Physics and Technology of Semiconductor Devices" (1967)
        - Plummer et al., "Silicon VLSI Technology" (2000), Ch. 7
    
    Status: IMPLEMENTED - Session 2
    """
    # Input validation
    if t <= 0:
        raise ValueError("Time must be positive")
    if Q <= 0:
        raise ValueError("Dose must be positive")
    
    # Convert x from nm to cm
    x_cm = x * 1e-7  # nm -> cm
    
    # Calculate diffusivity
    D = diffusivity(T, D0, Ea)
    
    # Characteristic diffusion length
    sqrt_Dt = np.sqrt(D * t)
    sqrt_pi_Dt = np.sqrt(np.pi * D * t)
    
    # Gaussian solution
    # N(x,t) = (Q / sqrt(π*D*t)) * exp(-x² / (4*D*t))
    exponent = -(x_cm ** 2) / (4 * D * t)
    
    # Prevent overflow/underflow
    exponent = np.clip(exponent, -100, 0)
    
    C = (Q / sqrt_pi_Dt) * np.exp(exponent) + NA0
    
    # Ensure physical constraints
    C = np.maximum(C, NA0)
    
    return C


def junction_depth(
    C_profile: NDArray[np.float64],
    x: NDArray[np.float64],
    N_background: float,
    method: Literal["linear", "log"] = "linear"
) -> float:
    """
    Calculate junction depth where C(xj) = N_background.
    
    Uses interpolation to find the crossing point between the
    concentration profile and background doping level.
    
    Args:
        C_profile: Concentration profile (atoms/cm³)
        x: Depth array (nm)
        N_background: Background concentration (atoms/cm³)
        method: Interpolation method
                - "linear": Linear interpolation
                - "log": Log-scale interpolation (better for steep gradients)
    
    Returns:
        Junction depth xj (nm)
    
    Raises:
        ValueError: If junction depth cannot be found
    
    Examples:
        >>> x = np.linspace(0, 1000, 1000)
        >>> C = constant_source_profile(x, 1800, 1000, 0.76, 3.46, 1e20, 1e15)
        >>> xj = junction_depth(C, x, 1e15)
        >>> print(f"Junction depth: {xj:.1f} nm")
    
    Status: IMPLEMENTED - Session 2
    """
    # Find where C crosses N_background
    # Look for sign change in (C - N_background)
    diff = C_profile - N_background
    
    # Check if junction exists
    if np.all(diff > 0):
        raise ValueError("No junction found - all concentrations above background")
    if np.all(diff < 0):
        raise ValueError("No junction found - all concentrations below background")
    
    # Find crossing points (sign changes)
    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    
    if len(sign_changes) == 0:
        raise ValueError("No junction depth found")
    
    # Take first crossing (shallowest)
    idx = sign_changes[0]
    
    # Interpolate to find exact crossing point
    if method == "log":
        # Log interpolation (better for steep profiles)
        # Avoid log(0) by using max(C, 1e10)
        C_safe = np.maximum(C_profile, 1e10)
        log_C = np.log10(C_safe)
        log_Nb = np.log10(max(N_background, 1e10))
        
        # Linear interpolation in log space
        log_C1, log_C2 = log_C[idx], log_C[idx + 1]
        x1, x2 = x[idx], x[idx + 1]
        
        # Find x where log_C(x) = log_Nb
        if log_C1 != log_C2:
            xj = x1 + (log_Nb - log_C1) / (log_C2 - log_C1) * (x2 - x1)
        else:
            xj = (x1 + x2) / 2
    else:
        # Linear interpolation
        C1, C2 = C_profile[idx], C_profile[idx + 1]
        x1, x2 = x[idx], x[idx + 1]
        
        # Find x where C(x) = N_background
        if C1 != C2:
            xj = x1 + (N_background - C1) / (C2 - C1) * (x2 - x1)
        else:
            xj = (x1 + x2) / 2
    
    return float(xj)


def sheet_resistance_estimate(
    C_profile: NDArray[np.float64],
    x: NDArray[np.float64],
    dopant_type: Literal["n", "p"] = "n",
    mobility_model: Literal["constant", "caughey_thomas"] = "constant",
    mu_constant: float = 1000.0
) -> float:
    """
    Estimate sheet resistance from dopant profile.
    
    Sheet resistance:
    Rs = 1 / (q * integral(μ(x) * N(x) dx))
    
    Where:
    - q is elementary charge
    - μ(x) is carrier mobility (function of doping)
    - N(x) is active doping concentration
    
    Assumptions:
    - 100% dopant activation
    - Simplified mobility model
    - 1D geometry
    
    Args:
        C_profile: Concentration profile (atoms/cm³)
        x: Depth array (nm)
        dopant_type: "n" or "p" type doping
        mobility_model: Mobility model
            - "constant": Fixed mobility
            - "caughey_thomas": Concentration-dependent
        mu_constant: Constant mobility value (cm²/V·s)
    
    Returns:
        Sheet resistance (Ω/□)
    
    Examples:
        >>> x = np.linspace(0, 500, 500)
        >>> C = limited_source_profile(x, 1200, 950, 3.85, 3.66, 1e14, 1e15)
        >>> Rs = sheet_resistance_estimate(C, x, "n")
        >>> print(f"Sheet resistance: {Rs:.1f} Ω/□")
    
    Notes:
        This is a simplified estimate. For accurate Rs:
        - Use detailed mobility models
        - Account for incomplete activation
        - Consider 2D/3D effects
        - Measure experimentally (4-point probe)
    
    References:
        - Caughey & Thomas, Proc. IEEE 55, 2192 (1967)
        - Sze & Ng, "Physics of Semiconductor Devices" (2006)
    
    Status: IMPLEMENTED - Session 2
    """
    # Convert x from nm to cm
    x_cm = x * 1e-7
    
    # Calculate mobility
    if mobility_model == "constant":
        mu = np.full_like(C_profile, mu_constant)
    elif mobility_model == "caughey_thomas":
        # Caughey-Thomas mobility model
        # μ(N) = μ_min + (μ_max - μ_min) / (1 + (N/N_ref)^α)
        if dopant_type == "n":
            mu_min = 92.0  # cm²/V·s
            mu_max = 1414.0
            N_ref = 1.3e17  # cm⁻³
            alpha = 0.91
        else:  # p-type
            mu_min = 47.7
            mu_max = 470.5
            N_ref = 6.3e16
            alpha = 0.76
        
        mu = mu_min + (mu_max - mu_min) / (1 + np.power(C_profile / N_ref, alpha))
    else:
        raise ValueError(f"Unknown mobility model: {mobility_model}")
    
    # Calculate conductivity: σ = q * μ * N
    sigma = q * mu * C_profile
    
    # Integrate conductivity over depth
    # Use trapezoidal rule
    sheet_conductance = np.trapz(sigma, x_cm)  # 1/Ω
    
    # Sheet resistance is inverse of sheet conductance
    if sheet_conductance > 0:
        Rs = 1.0 / sheet_conductance
    else:
        Rs = np.inf
    
    return Rs


def two_step_diffusion(
    x: NDArray[np.float64],
    t1: float,
    T1: float,
    t2: float,
    T2: float,
    D0: float,
    Ea: float,
    Cs: float,
    NA0: float = 0.0
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Two-step diffusion process: pre-deposition + drive-in.
    
    Step 1 (Pre-deposition):
        - Constant source at high temperature
        - Short time to establish surface layer
        - Creates high surface concentration
    
    Step 2 (Drive-in):
        - Limited source (no additional dopant)
        - Longer time at lower/higher temperature
        - Redistributes dopant deeper into substrate
    
    Args:
        x: Depth array (nm)
        t1: Pre-dep time (seconds)
        T1: Pre-dep temperature (°C)
        t2: Drive-in time (seconds)
        T2: Drive-in temperature (°C)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        Cs: Surface concentration during pre-dep (atoms/cm³)
        NA0: Background concentration (atoms/cm³)
    
    Returns:
        (C_after_predep, C_after_drivein) - profiles after each step
    
    Examples:
        >>> # Boron: 900°C/15min predep + 1100°C/30min drive-in
        >>> x = np.linspace(0, 1000, 1000)
        >>> C_predep, C_final = two_step_diffusion(
        ...     x, t1=15*60, T1=900, t2=30*60, T2=1100,
        ...     D0=0.76, Ea=3.46, Cs=1e20, NA0=1e15
        ... )
    
    Status: IMPLEMENTED - Session 2
    """
    # Step 1: Pre-deposition (constant source)
    C_predep = constant_source_profile(x, t1, T1, D0, Ea, Cs, NA0)
    
    # Calculate total dose after pre-dep
    x_cm = x * 1e-7
    Q = np.trapz(C_predep - NA0, x_cm)  # atoms/cm²
    
    # Step 2: Drive-in (limited source)
    # Use dose from pre-dep as source for drive-in
    C_drivein = limited_source_profile(x, t2, T2, D0, Ea, Q, NA0)
    
    return C_predep, C_drivein


def effective_diffusion_time(
    T_profile: NDArray[np.float64],
    time_points: NDArray[np.float64],
    D0: float,
    Ea: float,
    T_ref: float = 1000.0
) -> float:
    """
    Calculate effective diffusion time for time-varying temperature.
    
    For non-isothermal diffusion, the effective time at reference
    temperature T_ref is:
    
    t_eff = integral(D(T(t))/D(T_ref) dt)
    
    This allows treating variable-temperature process as equivalent
    isothermal process.
    
    Args:
        T_profile: Temperature vs time (°C)
        time_points: Time points (seconds)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        T_ref: Reference temperature (°C)
    
    Returns:
        Effective time at T_ref (seconds)
    
    Examples:
        >>> # Ramp from 800°C to 1000°C over 10 min
        >>> time = np.linspace(0, 600, 100)
        >>> T = np.linspace(800, 1000, 100)
        >>> t_eff = effective_diffusion_time(T, time, 0.76, 3.46, 1000)
        >>> print(f"Effective time at 1000°C: {t_eff:.1f} s")
    
    Status: IMPLEMENTED - Session 2
    """
    # Calculate D at reference temperature
    D_ref = diffusivity(T_ref, D0, Ea)
    
    # Calculate D at each temperature point
    D_profile = np.array([diffusivity(T, D0, Ea) for T in T_profile])
    
    # Calculate ratio D(T)/D_ref
    D_ratio = D_profile / D_ref
    
    # Integrate over time
    t_eff = np.trapz(D_ratio, time_points)
    
    return t_eff


# Utility functions for common scenarios

def quick_profile_constant_source(
    depth_max: float = 1000.0,
    n_points: int = 1000,
    time_minutes: float = 30.0,
    temp_celsius: float = 1000.0,
    dopant: Literal["boron", "phosphorus", "arsenic", "antimony"] = "boron",
    Cs: float = 1e20,
    NA0: float = 1e15
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Quick helper for constant-source profile with common dopants.
    
    Args:
        depth_max: Maximum depth (nm)
        n_points: Number of points
        time_minutes: Time (minutes)
        temp_celsius: Temperature (°C)
        dopant: Dopant type
        Cs: Surface concentration (atoms/cm³)
        NA0: Background (atoms/cm³)
    
    Returns:
        (x, C) - depth and concentration arrays
    
    Status: IMPLEMENTED - Session 2
    """
    # Dopant parameters (from config)
    dopant_params = {
        "boron": (0.76, 3.46),
        "phosphorus": (3.85, 3.66),
        "arsenic": (0.066, 3.44),
        "antimony": (0.214, 3.65)
    }
    
    D0, Ea = dopant_params[dopant]
    
    x = np.linspace(0, depth_max, n_points)
    t = time_minutes * 60  # Convert to seconds
    
    C = constant_source_profile(x, t, temp_celsius, D0, Ea, Cs, NA0)
    
    return x, C


def quick_profile_limited_source(
    depth_max: float = 500.0,
    n_points: int = 500,
    time_minutes: float = 20.0,
    temp_celsius: float = 950.0,
    dopant: Literal["boron", "phosphorus", "arsenic", "antimony"] = "boron",
    dose: float = 1e14,
    NA0: float = 1e15
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Quick helper for limited-source profile with common dopants.
    
    Args:
        depth_max: Maximum depth (nm)
        n_points: Number of points
        time_minutes: Time (minutes)
        temp_celsius: Temperature (°C)
        dopant: Dopant type
        dose: Total dose (atoms/cm²)
        NA0: Background (atoms/cm³)
    
    Returns:
        (x, C) - depth and concentration arrays
    
    Status: IMPLEMENTED - Session 2
    """
    dopant_params = {
        "boron": (0.76, 3.46),
        "phosphorus": (3.85, 3.66),
        "arsenic": (0.066, 3.44),
        "antimony": (0.214, 3.65)
    }
    
    D0, Ea = dopant_params[dopant]
    
    x = np.linspace(0, depth_max, n_points)
    t = time_minutes * 60
    
    C = limited_source_profile(x, t, temp_celsius, D0, Ea, dose, NA0)
    
    return x, C


__all__ = [
    "diffusivity",
    "constant_source_profile",
    "limited_source_profile",
    "junction_depth",
    "sheet_resistance_estimate",
    "two_step_diffusion",
    "effective_diffusion_time",
    "quick_profile_constant_source",
    "quick_profile_limited_source",
]
