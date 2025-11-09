"""
Deal-Grove model for thermal oxidation of silicon.

The Deal-Grove model describes oxide growth with the equation:
    x_ox^2 + A*x_ox = B*(t + τ)

Where:
    x_ox: oxide thickness (μm)
    t: oxidation time (hours)
    A: linear rate constant (μm)
    B: parabolic rate constant (μm²/hr)
    τ: time shift to account for initial oxide (hours)

References:
    Deal & Grove, J. Appl. Phys. 36, 3770 (1965)
"""

import numpy as np
from typing import Tuple, Union


# Activation energies (eV)
EA_B_DRY = 2.0      # Dry oxidation parabolic rate
EA_B_WET = 0.78     # Wet oxidation parabolic rate
EA_BA_DRY = 1.96    # Dry oxidation linear term
EA_BA_WET = 2.05    # Wet oxidation linear term

# Pre-exponential factors at 1 atm
# B values in μm²/hr
B0_DRY = 7.72e5     # Dry O2
B0_WET = 3.86e8     # H2O (wet)

# B/A values in μm/hr
BA0_DRY = 3.71e6    # Dry O2
BA0_WET = 6.23e8    # H2O (wet)

# Physical constants
K_B = 8.617333e-5   # Boltzmann constant (eV/K)


def arrhenius(T: float, E_a: float, A_0: float) -> float:
    """
    Calculate Arrhenius rate: A_0 * exp(-E_a / (k_B * T))
    
    Args:
        T: Temperature (K)
        E_a: Activation energy (eV)
        A_0: Pre-exponential factor
        
    Returns:
        Rate constant
    """
    return A_0 * np.exp(-E_a / (K_B * T))


def get_rate_constants(T: float, ambient: str = 'dry', pressure: float = 1.0) -> Tuple[float, float]:
    """
    Calculate Deal-Grove rate constants B and B/A for given conditions.
    
    Args:
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        pressure: Partial pressure of oxidant (atm), default 1.0
        
    Returns:
        (B, B_over_A): Parabolic and linear rate constants
            B in μm²/hr
            B/A in μm/hr
    """
    T_K = T + 273.15  # Convert to Kelvin
    
    if ambient.lower() == 'dry':
        B = arrhenius(T_K, EA_B_DRY, B0_DRY)
        B_over_A = arrhenius(T_K, EA_BA_DRY, BA0_DRY)
    elif ambient.lower() == 'wet':
        B = arrhenius(T_K, EA_B_WET, B0_WET)
        B_over_A = arrhenius(T_K, EA_BA_WET, BA0_WET)
    else:
        raise ValueError(f"Invalid ambient: {ambient}. Must be 'dry' or 'wet'")
    
    # Scale by pressure (linear dependence)
    B *= pressure
    B_over_A *= pressure
    
    return B, B_over_A


def thickness_at_time(t: Union[float, np.ndarray], 
                      T: float, 
                      ambient: str = 'dry',
                      pressure: float = 1.0,
                      x_i: float = 0.0) -> Union[float, np.ndarray]:
    """
    Calculate oxide thickness at given time(s) using Deal-Grove model.
    
    Solves: x_ox^2 + A*x_ox = B*(t + τ)
    
    Args:
        t: Time or array of times (hours)
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        pressure: Partial pressure of oxidant (atm)
        x_i: Initial oxide thickness (μm)
        
    Returns:
        Oxide thickness (μm) at time(s) t
    """
    B, B_over_A = get_rate_constants(T, ambient, pressure)
    A = B / B_over_A
    
    # Calculate time shift τ for initial oxide
    if x_i > 0:
        tau = (x_i**2 + A * x_i) / B
    else:
        tau = 0.0
    
    # Solve quadratic: x^2 + A*x - B*(t + τ) = 0
    # Solution: x = (-A + sqrt(A^2 + 4*B*(t + τ))) / 2
    discriminant = A**2 + 4 * B * (t + tau)
    x_ox = (-A + np.sqrt(discriminant)) / 2
    
    return x_ox


def time_to_thickness(x_target: float,
                      T: float,
                      ambient: str = 'dry',
                      pressure: float = 1.0,
                      x_i: float = 0.0) -> float:
    """
    Calculate time required to grow oxide to target thickness (inverse problem).
    
    Args:
        x_target: Target oxide thickness (μm)
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        pressure: Partial pressure of oxidant (atm)
        x_i: Initial oxide thickness (μm)
        
    Returns:
        Time required (hours)
    """
    if x_target < x_i:
        raise ValueError(f"Target thickness {x_target} μm < initial thickness {x_i} μm")
    
    B, B_over_A = get_rate_constants(T, ambient, pressure)
    A = B / B_over_A
    
    # Calculate time shift for initial oxide
    if x_i > 0:
        tau = (x_i**2 + A * x_i) / B
    else:
        tau = 0.0
    
    # From x^2 + A*x = B*(t + τ), solve for t:
    # t = (x^2 + A*x)/B - τ
    t = (x_target**2 + A * x_target) / B - tau
    
    return max(0.0, t)  # Ensure non-negative time


def growth_rate(x_ox: float,
                T: float,
                ambient: str = 'dry',
                pressure: float = 1.0) -> float:
    """
    Calculate instantaneous growth rate dx/dt at given thickness.
    
    From Deal-Grove: dx/dt = B / (2*x + A)
    
    Args:
        x_ox: Current oxide thickness (μm)
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        pressure: Partial pressure of oxidant (atm)
        
    Returns:
        Growth rate (μm/hr)
    """
    B, B_over_A = get_rate_constants(T, ambient, pressure)
    A = B / B_over_A
    
    return B / (2 * x_ox + A)


def linear_regime_thickness(T: float, ambient: str = 'dry', pressure: float = 1.0) -> float:
    """
    Calculate characteristic thickness where growth transitions from linear to parabolic.
    
    This occurs approximately at x ≈ A.
    
    Args:
        T: Temperature (Celsius)
        ambient: 'dry' or 'wet' oxidation
        pressure: Partial pressure of oxidant (atm)
        
    Returns:
        Transition thickness (μm)
    """
    B, B_over_A = get_rate_constants(T, ambient, pressure)
    A = B / B_over_A
    return A


if __name__ == "__main__":
    # Example usage and validation
    print("Deal-Grove Model - Example Calculations\n")
    print("=" * 60)
    
    # Test case 1: Dry oxidation at 1000°C
    T = 1000
    ambient = 'dry'
    times = np.array([0.5, 1.0, 2.0, 4.0, 8.0])  # hours
    
    print(f"\n1. Dry Oxidation at {T}°C")
    print("-" * 60)
    B, B_over_A = get_rate_constants(T, ambient)
    A = B / B_over_A
    print(f"B = {B:.2e} μm²/hr")
    print(f"B/A = {B_over_A:.2e} μm/hr")
    print(f"A = {A:.4f} μm")
    print(f"\nTime (hr)  Thickness (μm)")
    for t in times:
        x = thickness_at_time(t, T, ambient)
        print(f"{t:8.2f}  {x:14.4f}")
    
    # Test case 2: Wet oxidation at 1000°C
    ambient = 'wet'
    print(f"\n2. Wet Oxidation at {T}°C")
    print("-" * 60)
    B, B_over_A = get_rate_constants(T, ambient)
    A = B / B_over_A
    print(f"B = {B:.2e} μm²/hr")
    print(f"B/A = {B_over_A:.2e} μm/hr")
    print(f"A = {A:.4f} μm")
    print(f"\nTime (hr)  Thickness (μm)")
    for t in times:
        x = thickness_at_time(t, T, ambient)
        print(f"{t:8.2f}  {x:14.4f}")
    
    # Test case 3: Inverse problem
    print(f"\n3. Inverse Problem: Time to reach target thickness")
    print("-" * 60)
    target_thickness = 0.5  # μm
    for T in [900, 1000, 1100]:
        for amb in ['dry', 'wet']:
            t_required = time_to_thickness(target_thickness, T, amb)
            print(f"{amb.capitalize():4s} @ {T}°C: {t_required:.3f} hours to reach {target_thickness} μm")
    
    # Test case 4: Growth rate evolution
    print(f"\n4. Growth Rate vs Thickness (Dry, 1000°C)")
    print("-" * 60)
    print(f"Thickness (μm)  Growth Rate (μm/hr)")
    for x in [0.001, 0.01, 0.1, 0.5, 1.0]:
        rate = growth_rate(x, 1000, 'dry')
        print(f"{x:14.3f}  {rate:19.4f}")
