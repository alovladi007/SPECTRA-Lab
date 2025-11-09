"""
Massoud model for thin-oxide correction to Deal-Grove.

The Massoud model adds an exponential correction term for thin oxides (<~70 nm)
where the Deal-Grove linear-parabolic model underestimates growth rates.

Modified thickness equation:
    x_ox = x_DG + C * exp(-x_DG / L)

Where:
    x_DG: Deal-Grove predicted thickness
    C: Amplitude of correction (nm)
    L: Characteristic length scale (nm)

References:
    Massoud et al., J. Electrochem. Soc. 132, 2685 (1985)
"""

import numpy as np
from typing import Union
from . import deal_grove


# Default Massoud parameters (from literature)
# These are temperature-dependent but we use representative values
C_DEFAULT = 20.0   # nm (amplitude)
L_DEFAULT = 7.0    # nm (characteristic length)


def get_correction_params(T: float, ambient: str = 'dry') -> tuple[float, float]:
    """
    Get Massoud correction parameters C and L for given conditions.
    
    These parameters are somewhat empirical and vary in literature.
    We use temperature-dependent approximations.
    
    Args:
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        
    Returns:
        (C, L): Correction amplitude and length scale (both in nm)
    """
    # Simple temperature scaling (empirical)
    # Higher temperatures -> smaller correction needed
    T_ref = 900.0  # Reference temperature
    
    if ambient.lower() == 'dry':
        # Dry oxidation has more pronounced thin-oxide effects
        C = C_DEFAULT * (1.0 + (T_ref - T) / 500.0)
        L = L_DEFAULT
    elif ambient.lower() == 'wet':
        # Wet oxidation has weaker thin-oxide effects
        C = C_DEFAULT * 0.7 * (1.0 + (T_ref - T) / 500.0)
        L = L_DEFAULT * 0.8
    else:
        raise ValueError(f"Invalid ambient: {ambient}")
    
    # Ensure positive parameters
    C = max(C, 0.0)
    L = max(L, 1.0)
    
    return C, L


def thickness_with_correction(t: Union[float, np.ndarray],
                               T: float,
                               ambient: str = 'dry',
                               pressure: float = 1.0,
                               x_i: float = 0.0,
                               apply_correction: bool = True) -> Union[float, np.ndarray]:
    """
    Calculate oxide thickness with Massoud thin-oxide correction.
    
    Args:
        t: Time or array of times (hours)
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        pressure: Partial pressure of oxidant (atm)
        x_i: Initial oxide thickness (μm)
        apply_correction: If True, apply Massoud correction
        
    Returns:
        Oxide thickness (μm) at time(s) t
    """
    # Get Deal-Grove thickness
    x_dg = deal_grove.thickness_at_time(t, T, ambient, pressure, x_i)
    
    if not apply_correction:
        return x_dg
    
    # Get correction parameters
    C, L = get_correction_params(T, ambient)
    
    # Convert to nm for correction calculation
    x_dg_nm = x_dg * 1000.0  # μm to nm
    
    # Apply Massoud correction
    correction = C * np.exp(-x_dg_nm / L)
    x_massoud_nm = x_dg_nm + correction
    
    # Convert back to μm
    x_massoud = x_massoud_nm / 1000.0
    
    return x_massoud


def correction_magnitude(x_dg: float,
                         T: float,
                         ambient: str = 'dry') -> float:
    """
    Calculate the magnitude of Massoud correction at given thickness.
    
    Args:
        x_dg: Deal-Grove thickness (μm)
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        
    Returns:
        Correction magnitude (nm)
    """
    C, L = get_correction_params(T, ambient)
    x_dg_nm = x_dg * 1000.0  # Convert to nm
    
    return C * np.exp(-x_dg_nm / L)


def is_correction_significant(x_dg: float,
                               T: float,
                               ambient: str = 'dry',
                               threshold: float = 0.05) -> bool:
    """
    Check if Massoud correction is significant (> threshold).
    
    Args:
        x_dg: Deal-Grove thickness (μm)
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        threshold: Relative threshold (e.g., 0.05 = 5%)
        
    Returns:
        True if correction is > threshold fraction of thickness
    """
    if x_dg == 0:
        return True
    
    correction = correction_magnitude(x_dg, T, ambient)
    x_dg_nm = x_dg * 1000.0
    
    return (correction / x_dg_nm) > threshold


def time_to_thickness_with_correction(x_target: float,
                                       T: float,
                                       ambient: str = 'dry',
                                       pressure: float = 1.0,
                                       x_i: float = 0.0,
                                       apply_correction: bool = True,
                                       tol: float = 1e-4) -> float:
    """
    Calculate time to reach target thickness with Massoud correction (inverse problem).
    
    Uses iterative Newton-Raphson method since analytical inversion is not possible.
    
    Args:
        x_target: Target oxide thickness (μm)
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        pressure: Partial pressure of oxidant (atm)
        x_i: Initial oxide thickness (μm)
        apply_correction: If True, apply Massoud correction
        tol: Convergence tolerance (μm)
        
    Returns:
        Time required (hours)
    """
    if not apply_correction:
        # No correction, use analytical Deal-Grove inverse
        return deal_grove.time_to_thickness(x_target, T, ambient, pressure, x_i)
    
    # Initial guess using Deal-Grove
    t_guess = deal_grove.time_to_thickness(x_target, T, ambient, pressure, x_i)
    
    # Newton-Raphson iteration
    max_iter = 50
    for i in range(max_iter):
        # Calculate thickness at current guess
        x_current = thickness_with_correction(t_guess, T, ambient, pressure, x_i, True)
        
        # Check convergence
        error = x_current - x_target
        if abs(error) < tol:
            return t_guess
        
        # Calculate numerical derivative (growth rate)
        dt = 0.001  # Small time step (hours)
        x_plus = thickness_with_correction(t_guess + dt, T, ambient, pressure, x_i, True)
        dxdt = (x_plus - x_current) / dt
        
        # Newton-Raphson update
        if abs(dxdt) > 1e-10:
            t_guess -= error / dxdt
            t_guess = max(0, t_guess)  # Ensure positive time
        else:
            break
    
    # Return best estimate even if not fully converged
    return max(0, t_guess)


if __name__ == "__main__":
    # Example usage and validation
    print("Massoud Thin-Oxide Correction - Examples\n")
    print("=" * 70)
    
    # Test case 1: Correction magnitude vs thickness
    print("\n1. Correction Magnitude vs Thickness (Dry, 1000°C)")
    print("-" * 70)
    T = 1000
    ambient = 'dry'
    C, L = get_correction_params(T, ambient)
    print(f"Massoud parameters: C = {C:.2f} nm, L = {L:.2f} nm\n")
    
    print(f"{'Thickness (nm)':>15}  {'Correction (nm)':>16}  {'Relative (%)':>14}")
    for x_nm in [1, 5, 10, 20, 50, 100, 200]:
        x_um = x_nm / 1000.0
        corr = correction_magnitude(x_um, T, ambient)
        rel_pct = 100 * corr / x_nm if x_nm > 0 else 0
        print(f"{x_nm:>15d}  {corr:>16.3f}  {rel_pct:>14.2f}")
    
    # Test case 2: Deal-Grove vs Massoud comparison
    print(f"\n2. Deal-Grove vs Massoud Comparison (Dry, 900°C)")
    print("-" * 70)
    T = 900
    times = np.array([0.1, 0.5, 1.0, 2.0, 5.0])  # hours
    
    print(f"{'Time (hr)':>10}  {'Deal-Grove (nm)':>16}  {'Massoud (nm)':>14}  {'Diff (nm)':>11}")
    for t in times:
        x_dg = deal_grove.thickness_at_time(t, T, ambient) * 1000  # to nm
        x_mass = thickness_with_correction(t, T, ambient, apply_correction=True) * 1000
        diff = x_mass - x_dg
        print(f"{t:>10.2f}  {x_dg:>16.2f}  {x_mass:>14.2f}  {diff:>11.2f}")
    
    # Test case 3: Inverse problem with correction
    print(f"\n3. Inverse Problem with Massoud Correction")
    print("-" * 70)
    print(f"Target: 50 nm oxide\n")
    print(f"{'Temp (°C)':>10}  {'Ambient':>8}  {'Time DG (hr)':>14}  {'Time Mass (hr)':>16}  {'Diff (hr)':>11}")
    
    target_nm = 50
    target_um = target_nm / 1000.0
    
    for T in [900, 1000, 1100]:
        for amb in ['dry', 'wet']:
            t_dg = deal_grove.time_to_thickness(target_um, T, amb)
            t_mass = time_to_thickness_with_correction(target_um, T, amb, apply_correction=True)
            diff = t_mass - t_dg
            print(f"{T:>10d}  {amb:>8s}  {t_dg:>14.4f}  {t_mass:>16.4f}  {diff:>11.4f}")
    
    # Test case 4: When is correction significant?
    print(f"\n4. Significance of Thin-Oxide Correction (5% threshold)")
    print("-" * 70)
    print(f"Dry oxidation at 1000°C:")
    print(f"{'Thickness (nm)':>15}  {'Significant?':>12}")
    for x_nm in [5, 10, 20, 50, 100, 200]:
        x_um = x_nm / 1000.0
        significant = is_correction_significant(x_um, 1000, 'dry', threshold=0.05)
        print(f"{x_nm:>15d}  {str(significant):>12s}")
