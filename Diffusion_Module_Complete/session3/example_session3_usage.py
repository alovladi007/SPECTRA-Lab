#!/usr/bin/env python3
"""
Quick example script for Session 3 numerical solver.

Demonstrates basic usage of the Crank-Nicolson solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add to path if running standalone
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fick_fd import Fick1D, quick_solve_constant_D
from core.erfc import diffusivity, constant_source_profile, junction_depth


def example_1_basic_comparison():
    """Example 1: Compare numerical vs analytical solution."""
    print("=" * 70)
    print("Example 1: Numerical vs Analytical Comparison")
    print("=" * 70)
    
    # Parameters
    x_max = 1000  # nm
    dx = 2.0
    t_final = 30 * 60  # 30 minutes
    T = 1000  # °C
    Cs = 1e20
    NA0 = 1e15
    
    # Numerical solution
    print("\nSolving numerically...")
    solver = Fick1D(x_max=x_max, dx=dx, refine_surface=False)
    C0 = np.full(solver.n_points, NA0)
    
    D0, Ea = 0.76, 3.46  # Boron
    def D_model(T_val, C):
        return diffusivity(T_val, D0, Ea)
    
    x_num, C_num = solver.solve(
        C0, dt=1.0, steps=int(t_final), T=T, D_model=D_model,
        bc=('dirichlet', 'neumann'), surface_value=Cs
    )
    
    # Analytical solution
    print("Computing analytical solution...")
    C_anal = constant_source_profile(x_num, t_final, T, D0, Ea, Cs, NA0)
    
    # Calculate junction depths
    xj_num = junction_depth(C_num, x_num, NA0)
    xj_anal = junction_depth(C_anal, x_num, NA0)
    
    # Calculate error
    _, _, rel_error = solver.validate_convergence(C_anal, C_num)
    
    print(f"\n{'Results:':<30}")
    print(f"  {'Numerical xⱼ:':<28} {xj_num:.1f} nm")
    print(f"  {'Analytical xⱼ:':<28} {xj_anal:.1f} nm")
    print(f"  {'Difference:':<28} {abs(xj_num - xj_anal):.1f} nm")
    print(f"  {'Relative L2 Error:':<28} {rel_error*100:.2f}%")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_num, C_anal, 'b-', linewidth=2, label='Analytical', alpha=0.7)
    plt.plot(x_num, C_num, 'r--', linewidth=2, label='Numerical')
    plt.axvline(xj_anal, color='b', linestyle=':', alpha=0.5)
    plt.axvline(xj_num, color='r', linestyle=':', alpha=0.5)
    plt.xlabel('Depth (nm)')
    plt.ylabel('Concentration (cm⁻³)')
    plt.title(f'Boron Diffusion: {t_final/60:.0f} min @ {T}°C')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(x_num, C_anal, 'b-', linewidth=2, label='Analytical')
    plt.semilogy(x_num, C_num, 'r--', linewidth=2, label='Numerical')
    plt.xlabel('Depth (nm)')
    plt.ylabel('Concentration (cm⁻³)')
    plt.title('Log Scale')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(1e14, 1e21)
    
    plt.tight_layout()
    plt.savefig('example1_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'example1_comparison.png'")
    plt.show()


def example_2_concentration_dependent():
    """Example 2: Concentration-dependent diffusivity."""
    print("\n" + "=" * 70)
    print("Example 2: Concentration-Dependent Diffusivity")
    print("=" * 70)
    
    # Parameters
    solver = Fick1D(x_max=1000, dx=2.0, refine_surface=True)
    C0 = np.full(solver.n_points, 1e15)
    t_final = 30 * 60
    T = 1000
    Cs = 1e20
    NA0 = 1e15
    
    # Two models
    D0_base = 1e-13
    
    def D_constant(T_val, C):
        return D0_base
    
    def D_enhanced(T_val, C):
        if C is None:
            return D0_base
        # Enhanced at high concentration
        return D0_base * (1 + 5e-20 * C)
    
    print("\nSolving with constant D...")
    x1, C1 = solver.solve(
        C0.copy(), dt=1.0, steps=int(t_final), T=T,
        D_model=D_constant,
        bc=('dirichlet', 'neumann'),
        surface_value=Cs
    )
    
    print("Solving with enhanced D(C)...")
    solver2 = Fick1D(x_max=1000, dx=2.0, refine_surface=True)
    x2, C2 = solver2.solve(
        C0.copy(), dt=1.0, steps=int(t_final), T=T,
        D_model=D_enhanced,
        bc=('dirichlet', 'neumann'),
        surface_value=Cs
    )
    
    # Junction depths
    xj1 = junction_depth(C1, x1, NA0)
    xj2 = junction_depth(C2, x2, NA0)
    
    print(f"\n{'Results:':<30}")
    print(f"  {'Constant D xⱼ:':<28} {xj1:.1f} nm")
    print(f"  {'Enhanced D(C) xⱼ:':<28} {xj2:.1f} nm")
    print(f"  {'Increase:':<28} {xj2 - xj1:.1f} nm ({(xj2/xj1-1)*100:.1f}%)")
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x1, C1, 'b-', linewidth=2, label=f'Constant D (xⱼ={xj1:.0f}nm)')
    plt.plot(x2, C2, 'r--', linewidth=2, label=f'Enhanced D(C) (xⱼ={xj2:.0f}nm)')
    plt.axvline(xj1, color='b', linestyle=':', alpha=0.5)
    plt.axvline(xj2, color='r', linestyle=':', alpha=0.5)
    plt.xlabel('Depth (nm)')
    plt.ylabel('Concentration (cm⁻³)')
    plt.title('Constant vs Enhanced Diffusivity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    D_eff1 = np.full_like(x1, D0_base)
    D_eff2 = D_enhanced(T, C2)
    plt.plot(x1, D_eff1, 'b-', linewidth=2, label='Constant D')
    plt.plot(x2, D_eff2, 'r--', linewidth=2, label='Enhanced D(C)')
    plt.xlabel('Depth (nm)')
    plt.ylabel('Diffusivity (cm²/s)')
    plt.title('Effective Diffusivity Profile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)
    
    plt.tight_layout()
    plt.savefig('example2_concentration_dependent.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'example2_concentration_dependent.png'")
    plt.show()


def example_3_quick_helper():
    """Example 3: Using the quick helper function."""
    print("\n" + "=" * 70)
    print("Example 3: Quick Helper Function")
    print("=" * 70)
    
    print("\nUsing quick_solve_constant_D() for rapid simulation...")
    
    x, C = quick_solve_constant_D(
        x_max=1000,
        dx=2.0,
        t_final=1800,  # 30 minutes
        dt=1.0,
        T=1000,
        D0=0.76,  # Boron
        Ea=3.46,
        Cs=1e20,
        NA0=1e15,
        refine_surface=True
    )
    
    xj = junction_depth(C, x, 1e15)
    
    print(f"\n{'Results:':<30}")
    print(f"  {'Grid points:':<28} {len(x)}")
    print(f"  {'Surface concentration:':<28} {C[0]:.2e} cm⁻³")
    print(f"  {'Junction depth:':<28} {xj:.1f} nm")
    print(f"  {'Deep concentration:':<28} {C[-1]:.2e} cm⁻³")
    
    # Simple plot
    plt.figure(figsize=(10, 5))
    plt.semilogy(x, C, 'b-', linewidth=2)
    plt.axhline(1e15, color='r', linestyle='--', alpha=0.5, label='Background')
    plt.axvline(xj, color='g', linestyle='--', alpha=0.5, label=f'xⱼ = {xj:.0f} nm')
    plt.xlabel('Depth (nm)')
    plt.ylabel('Concentration (cm⁻³)')
    plt.title('Quick Solve: Boron @ 1000°C, 30 min')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(1e14, 1e21)
    
    plt.tight_layout()
    plt.savefig('example3_quick_solve.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as 'example3_quick_solve.png'")
    plt.show()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Session 3: Numerical Solver Examples")
    print("Crank-Nicolson Finite Difference Method for Fick's 2nd Law")
    print("=" * 70)
    
    try:
        example_1_basic_comparison()
        example_2_concentration_dependent()
        example_3_quick_helper()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70)
        print("\nGenerated files:")
        print("  - example1_comparison.png")
        print("  - example2_concentration_dependent.png")
        print("  - example3_quick_solve.png")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
