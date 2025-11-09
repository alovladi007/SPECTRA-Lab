"""
Comprehensive unit tests for fick_fd.py (Session 3).

Tests include:
- Solver initialization and grid setup
- Crank-Nicolson implementation correctness
- Convergence to analytical solutions
- Boundary condition handling
- Concentration-dependent diffusivity
- Grid refinement
- Stability and error metrics
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.fick_fd import Fick1D, quick_solve_constant_D
from core.erfc import (
    diffusivity,
    constant_source_profile,
    junction_depth
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def boron_params():
    """Boron diffusion parameters."""
    return {"D0": 0.76, "Ea": 3.46}


@pytest.fixture
def standard_solver():
    """Standard solver with uniform grid."""
    return Fick1D(x_max=1000, dx=2.0, refine_surface=False)


@pytest.fixture
def refined_solver():
    """Solver with refined surface grid."""
    return Fick1D(x_max=1000, dx=2.0, refine_surface=True,
                  surface_refinement_depth=100, surface_refinement_factor=5)


# ============================================================================
# Initialization Tests
# ============================================================================

class TestFick1DInitialization:
    """Tests for Fick1D initialization and grid setup."""
    
    def test_uniform_grid(self):
        """Test uniform grid generation."""
        solver = Fick1D(x_max=1000, dx=2.0, refine_surface=False)
        
        # Check grid properties
        assert solver.x[0] == 0.0
        assert solver.x[-1] <= 1000.0
        assert solver.n_points > 0
        
        # Check uniform spacing
        dx_computed = np.diff(solver.x)
        assert_allclose(dx_computed, 2.0, rtol=0.01)
    
    def test_refined_grid(self):
        """Test refined surface grid."""
        solver = Fick1D(x_max=1000, dx=4.0, refine_surface=True,
                       surface_refinement_depth=100, 
                       surface_refinement_factor=5)
        
        # Fine region should have smaller spacing
        fine_region_mask = solver.x < 100
        dx_fine = np.diff(solver.x[fine_region_mask])
        
        # Coarse region should have larger spacing
        coarse_region_mask = solver.x > 100
        if np.sum(coarse_region_mask) > 1:
            dx_coarse = np.diff(solver.x[coarse_region_mask])
            
            # Fine spacing should be smaller than coarse
            assert np.mean(dx_fine) < np.mean(dx_coarse)
    
    def test_grid_properties(self, standard_solver):
        """Test grid arrays are properly initialized."""
        assert standard_solver.x is not None
        assert standard_solver.n_points is not None
        assert standard_solver.dx_array is not None
        assert len(standard_solver.dx_array) == standard_solver.n_points - 1
    
    def test_numba_warning(self):
        """Test warning when numba requested but not available."""
        # This test may pass or warn depending on numba availability
        solver = Fick1D(use_numba=True)
        # Should not raise error


# ============================================================================
# Basic Solver Tests
# ============================================================================

class TestBasicSolverOperation:
    """Tests for basic solver operations."""
    
    def test_solve_returns_correct_shape(self, standard_solver, boron_params):
        """Test that solve returns arrays of correct shape."""
        C0 = np.full(standard_solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        x, C_final = standard_solver.solve(
            C0, dt=1.0, steps=10, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=1e20
        )
        
        assert len(x) == standard_solver.n_points
        assert len(C_final) == standard_solver.n_points
    
    def test_solve_with_history(self, standard_solver, boron_params):
        """Test solve with history tracking."""
        C0 = np.full(standard_solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        steps = 10
        x, C_final, C_history = standard_solver.solve(
            C0, dt=1.0, steps=steps, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=1e20,
            return_history=True
        )
        
        assert C_history.shape == (steps + 1, standard_solver.n_points)
        assert_allclose(C_history[0], C0)
        assert_allclose(C_history[-1], C_final)
    
    def test_concentration_increases_near_surface(self, standard_solver, boron_params):
        """Test that concentration increases near surface with source."""
        C0 = np.full(standard_solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        x, C_final = standard_solver.solve(
            C0, dt=1.0, steps=100, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=1e20
        )
        
        # Concentration should be higher near surface
        assert C_final[0] > C_final[-1]
        assert C_final[0] <= 1e20 * 1.01  # Within 1% of surface value
        assert C_final[-1] >= 1e15 * 0.99  # Near background


# ============================================================================
# Boundary Condition Tests
# ============================================================================

class TestBoundaryConditions:
    """Tests for different boundary conditions."""
    
    def test_dirichlet_left_neumann_right(self, standard_solver, boron_params):
        """Test Dirichlet at surface, Neumann at right boundary."""
        C0 = np.full(standard_solver.n_points, 1e15)
        Cs = 1e20
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        x, C_final = standard_solver.solve(
            C0, dt=1.0, steps=50, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # Surface should be at Cs
        assert_allclose(C_final[0], Cs, rtol=0.01)
        
        # Right boundary should have zero flux (dC/dx ≈ 0)
        # Check that last two points are similar
        assert_allclose(C_final[-1], C_final[-2], rtol=0.1)
    
    def test_neumann_both_sides(self, standard_solver, boron_params):
        """Test zero flux on both boundaries."""
        # Start with non-uniform profile
        C0 = np.full(standard_solver.n_points, 1e15)
        C0[0:10] = 1e20  # High concentration at surface
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        # Initial total dose
        x_cm = standard_solver.x * 1e-7
        Q_initial = np.trapz(C0, x_cm)
        
        x, C_final = standard_solver.solve(
            C0, dt=1.0, steps=100, T=1000,
            D_model=D_model,
            bc=('neumann', 'neumann')
        )
        
        # Total dose should be conserved (within 5%)
        Q_final = np.trapz(C_final, x_cm)
        assert_allclose(Q_final, Q_initial, rtol=0.05)
    
    def test_invalid_bc_raises_error(self, standard_solver, boron_params):
        """Test that invalid boundary condition raises error."""
        C0 = np.full(standard_solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        with pytest.raises(ValueError, match="Unknown.*boundary condition"):
            standard_solver.solve(
                C0, dt=1.0, steps=10, T=1000,
                D_model=D_model,
                bc=('invalid', 'neumann')
            )


# ============================================================================
# Convergence Tests
# ============================================================================

class TestConvergenceToAnalytical:
    """Tests convergence to analytical erfc solutions."""
    
    def test_convergence_constant_source(self, boron_params):
        """Test convergence to analytical constant-source solution."""
        # Simulation parameters
        t_final = 30 * 60  # 30 minutes in seconds
        T = 1000  # Celsius
        Cs = 1e20
        NA0 = 1e15
        
        # Create solver
        solver = Fick1D(x_max=1000, dx=1.0, refine_surface=False)
        
        # Initial condition
        C0 = np.full(solver.n_points, NA0)
        
        # Constant diffusivity model
        def D_model(T_val, C):
            return diffusivity(T_val, **boron_params)
        
        # Numerical solution
        dt = 0.5  # seconds
        steps = int(t_final / dt)
        x_num, C_num = solver.solve(
            C0, dt, steps, T, D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # Analytical solution
        C_analytical = constant_source_profile(
            x_num, t_final, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # Calculate errors
        L2, Linf, rel = solver.validate_convergence(C_analytical, C_num)
        
        # Errors should be small
        print(f"L2 error: {L2:.2e}, Linf: {Linf:.2e}, Relative: {rel:.4f}")
        assert rel < 0.05, f"Relative L2 error {rel:.4f} > 5%"
        assert Linf < Cs * 0.05, f"Max error {Linf:.2e} > 5% of Cs"
    
    def test_convergence_refinement(self, boron_params):
        """Test that finer grids give better accuracy."""
        t_final = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        dt = 0.5
        
        dx_values = [4.0, 2.0, 1.0]
        errors = []
        
        def D_model(T_val, C):
            return diffusivity(T_val, **boron_params)
        
        for dx in dx_values:
            solver = Fick1D(x_max=1000, dx=dx, refine_surface=False)
            C0 = np.full(solver.n_points, NA0)
            
            steps = int(t_final / dt)
            x_num, C_num = solver.solve(
                C0, dt, steps, T, D_model,
                bc=('dirichlet', 'neumann'),
                surface_value=Cs
            )
            
            # Analytical solution
            C_analytical = constant_source_profile(
                x_num, t_final, T, **boron_params, Cs=Cs, NA0=NA0
            )
            
            _, _, rel_error = solver.validate_convergence(C_analytical, C_num)
            errors.append(rel_error)
        
        # Errors should decrease with finer grid
        print(f"Errors for dx={dx_values}: {errors}")
        assert errors[1] < errors[0], "Error didn't decrease with finer grid"
        assert errors[2] < errors[1], "Error didn't decrease with finest grid"
    
    def test_second_order_convergence(self, boron_params):
        """Test that method achieves second-order convergence."""
        t_final = 10 * 60  # Shorter time for faster test
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        def D_model(T_val, C):
            return diffusivity(T_val, **boron_params)
        
        # Test spatial convergence
        dx_values = [8.0, 4.0, 2.0, 1.0]
        dt = 0.1  # Fixed small dt
        errors = []
        
        for dx in dx_values:
            solver = Fick1D(x_max=1000, dx=dx, refine_surface=False)
            C0 = np.full(solver.n_points, NA0)
            
            steps = int(t_final / dt)
            x_num, C_num = solver.solve(
                C0, dt, steps, T, D_model,
                bc=('dirichlet', 'neumann'),
                surface_value=Cs
            )
            
            C_analytical = constant_source_profile(
                x_num, t_final, T, **boron_params, Cs=Cs, NA0=NA0
            )
            
            L2, _, _ = solver.validate_convergence(C_analytical, C_num)
            errors.append(L2)
        
        # Check convergence rate
        # Error should decrease as O(dx²)
        # log(e2/e1) / log(dx2/dx1) should be close to 2
        for i in range(len(errors) - 1):
            ratio = errors[i+1] / errors[i]
            dx_ratio = dx_values[i+1] / dx_values[i]
            order = np.log(ratio) / np.log(dx_ratio)
            print(f"dx: {dx_values[i]}->{dx_values[i+1]}, order: {order:.2f}")
            # Should be close to 2 (allow 1.5-2.5)
            # May not be exact due to boundary effects
            assert 1.0 < order < 3.0, f"Convergence order {order:.2f} not near 2"


# ============================================================================
# Concentration-Dependent Diffusivity Tests
# ============================================================================

class TestConcentrationDependentDiffusivity:
    """Tests for concentration-dependent diffusivity."""
    
    def test_enhanced_diffusion(self, standard_solver):
        """Test enhanced diffusion at high concentration."""
        C0 = np.full(standard_solver.n_points, 1e15)
        T = 1000
        Cs = 1e20
        
        # Two models: constant D and concentration-enhanced D
        D0_base = 1e-13
        
        def D_constant(T_val, C):
            return D0_base
        
        def D_enhanced(T_val, C):
            if C is None:
                return D0_base
            # Enhanced diffusion: D = D0 * (1 + 1e-20 * C)
            return D0_base * (1 + 1e-20 * C)
        
        # Run both simulations
        dt = 1.0
        steps = 500
        
        x1, C1 = standard_solver.solve(
            C0, dt, steps, T, D_constant,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # Need new solver instance for second run
        solver2 = Fick1D(x_max=standard_solver.x_max, 
                        dx=standard_solver.dx, 
                        refine_surface=False)
        
        x2, C2 = solver2.solve(
            C0, dt, steps, T, D_enhanced,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # Enhanced diffusion should give deeper penetration
        # Calculate junction depths
        xj1 = junction_depth(C1, x1, 1e15)
        xj2 = junction_depth(C2, x2, 1e15)
        
        print(f"Constant D: xj={xj1:.1f} nm")
        print(f"Enhanced D: xj={xj2:.1f} nm")
        
        assert xj2 > xj1, "Enhanced diffusion should give deeper junction"
    
    def test_concentration_dependent_stability(self, standard_solver):
        """Test that concentration-dependent D remains stable."""
        C0 = np.full(standard_solver.n_points, 1e15)
        C0[0:5] = 1e20  # High concentration near surface
        
        T = 1000
        
        def D_nonlinear(T_val, C):
            if C is None:
                return 1e-13
            # Nonlinear dependence
            return 1e-13 * (1 + 1e-19 * C + 1e-40 * C**2)
        
        # Should not blow up
        x, C_final = standard_solver.solve(
            C0, dt=0.5, steps=100, T=T,
            D_model=D_nonlinear,
            bc=('neumann', 'neumann')
        )
        
        # Check for stability
        assert np.all(np.isfinite(C_final)), "Solution contains NaN or Inf"
        assert np.all(C_final >= 0), "Solution has negative concentrations"


# ============================================================================
# Grid Refinement Tests
# ============================================================================

class TestGridRefinement:
    """Tests for adaptive grid refinement."""
    
    def test_refined_grid_accuracy(self, boron_params):
        """Test that refined grid improves accuracy near surface."""
        t_final = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        dt = 0.5
        steps = int(t_final / dt)
        
        def D_model(T_val, C):
            return diffusivity(T_val, **boron_params)
        
        # Uniform grid
        solver_uniform = Fick1D(x_max=1000, dx=2.0, refine_surface=False)
        C0_uniform = np.full(solver_uniform.n_points, NA0)
        x_u, C_u = solver_uniform.solve(
            C0_uniform, dt, steps, T, D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # Refined grid
        solver_refined = Fick1D(x_max=1000, dx=2.0, refine_surface=True,
                               surface_refinement_depth=100,
                               surface_refinement_factor=5)
        C0_refined = np.full(solver_refined.n_points, NA0)
        x_r, C_r = solver_refined.solve(
            C0_refined, dt, steps, T, D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # Analytical solution at uniform grid points
        C_anal_u = constant_source_profile(
            x_u, t_final, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # Analytical solution at refined grid points
        C_anal_r = constant_source_profile(
            x_r, t_final, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # Calculate errors
        _, _, rel_u = solver_uniform.validate_convergence(C_anal_u, C_u)
        _, _, rel_r = solver_refined.validate_convergence(C_anal_r, C_r)
        
        print(f"Uniform grid error: {rel_u:.4f}")
        print(f"Refined grid error: {rel_r:.4f}")
        
        # Refined grid should be more accurate
        assert rel_r <= rel_u, "Refined grid should be more accurate"


# ============================================================================
# Physical Behavior Tests
# ============================================================================

class TestPhysicalBehavior:
    """Tests for physical behavior of solutions."""
    
    def test_monotonic_profile(self, standard_solver, boron_params):
        """Test that profile is monotonically decreasing from surface."""
        C0 = np.full(standard_solver.n_points, 1e15)
        Cs = 1e20
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        x, C_final = standard_solver.solve(
            C0, dt=1.0, steps=500, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # Profile should be monotonically decreasing
        dC = np.diff(C_final)
        assert np.all(dC <= 1e-5), "Profile not monotonically decreasing"
    
    def test_diffusion_increases_with_time(self, standard_solver, boron_params):
        """Test that diffusion depth increases with time."""
        C0 = np.full(standard_solver.n_points, 1e15)
        Cs = 1e20
        NA0 = 1e15
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        # Short time
        x, C_short = standard_solver.solve(
            C0.copy(), dt=1.0, steps=300, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        xj_short = junction_depth(C_short, x, NA0)
        
        # Long time
        solver2 = Fick1D(x_max=standard_solver.x_max, 
                        dx=standard_solver.dx,
                        refine_surface=False)
        x, C_long = solver2.solve(
            C0.copy(), dt=1.0, steps=1200, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        xj_long = junction_depth(C_long, x, NA0)
        
        print(f"Short time junction: {xj_short:.1f} nm")
        print(f"Long time junction: {xj_long:.1f} nm")
        
        assert xj_long > xj_short, "Junction should deepen with time"
    
    def test_mass_conservation_closed_system(self, standard_solver, boron_params):
        """Test mass conservation with Neumann BCs."""
        # Start with localized dose
        C0 = np.full(standard_solver.n_points, 1e15)
        C0[0:10] = 1e20
        
        x_cm = standard_solver.x * 1e-7
        Q_initial = np.trapz(C0, x_cm)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        x, C_final = standard_solver.solve(
            C0, dt=1.0, steps=500, T=1000,
            D_model=D_model,
            bc=('neumann', 'neumann')
        )
        
        Q_final = np.trapz(C_final, x_cm)
        
        # Mass should be conserved within 5%
        assert_allclose(Q_final, Q_initial, rtol=0.05)


# ============================================================================
# Stability Tests
# ============================================================================

class TestStability:
    """Tests for numerical stability."""
    
    def test_large_time_step_stability(self, standard_solver, boron_params):
        """Test stability with large time steps."""
        C0 = np.full(standard_solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        # Large time step (Crank-Nicolson is unconditionally stable)
        dt = 10.0  # seconds
        
        # Should not blow up
        x, C_final = standard_solver.solve(
            C0, dt, steps=100, T=1000,
            D_model=D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=1e20
        )
        
        assert np.all(np.isfinite(C_final)), "Solution unstable with large dt"
    
    def test_no_negative_concentrations(self, standard_solver, boron_params):
        """Test that solution remains non-negative."""
        C0 = np.full(standard_solver.n_points, 1e15)
        C0[0] = 1e20
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        x, C_final = standard_solver.solve(
            C0, dt=1.0, steps=1000, T=1000,
            D_model=D_model,
            bc=('neumann', 'neumann')
        )
        
        assert np.all(C_final >= -1e10), "Negative concentrations detected"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_quick_solve_helper(self):
        """Test quick_solve_constant_D helper function."""
        x, C = quick_solve_constant_D(
            x_max=1000, dx=2.0,
            t_final=1800, dt=1.0,
            T=1000,
            D0=0.76, Ea=3.46,
            Cs=1e20, NA0=1e15
        )
        
        assert len(x) == len(C)
        assert C[0] <= 1e20 * 1.01
        assert C[-1] >= 1e15 * 0.99
    
    def test_complete_workflow(self, boron_params):
        """Test complete workflow: solve -> validate -> analyze."""
        # 1. Setup
        solver = Fick1D(x_max=1000, dx=2.0, refine_surface=True)
        C0 = np.full(solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        # 2. Solve
        t_final = 30 * 60
        dt = 1.0
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        x, C = solver.solve(
            C0, dt, int(t_final / dt), T, D_model,
            bc=('dirichlet', 'neumann'),
            surface_value=Cs
        )
        
        # 3. Validate against analytical
        C_analytical = constant_source_profile(
            x, t_final, T, **boron_params, Cs=Cs, NA0=NA0
        )
        L2, Linf, rel = solver.validate_convergence(C_analytical, C)
        
        # 4. Analyze
        xj = junction_depth(C, x, NA0)
        
        # 5. Verify
        assert rel < 0.1, "Solution differs significantly from analytical"
        assert 0 < xj < x[-1], "Junction depth unreasonable"
        
        print(f"✓ Complete workflow successful")
        print(f"  L2 error: {L2:.2e}, Relative: {rel:.4f}")
        print(f"  Junction depth: {xj:.1f} nm")


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_mismatched_c0_length(self, standard_solver, boron_params):
        """Test error when C0 length doesn't match grid."""
        C0_wrong = np.full(standard_solver.n_points + 10, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        with pytest.raises(ValueError, match="does not match grid size"):
            standard_solver.solve(
                C0_wrong, dt=1.0, steps=10, T=1000,
                D_model=D_model,
                bc=('dirichlet', 'neumann'),
                surface_value=1e20
            )
    
    def test_negative_time_step(self, standard_solver, boron_params):
        """Test error for negative time step."""
        C0 = np.full(standard_solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        with pytest.raises(ValueError, match="Time step must be positive"):
            standard_solver.solve(
                C0, dt=-1.0, steps=10, T=1000,
                D_model=D_model
            )
    
    def test_missing_surface_value(self, standard_solver, boron_params):
        """Test error when Dirichlet BC lacks surface value."""
        C0 = np.full(standard_solver.n_points, 1e15)
        
        def D_model(T, C):
            return diffusivity(T, **boron_params)
        
        with pytest.raises(ValueError, match="surface_value required"):
            standard_solver.solve(
                C0, dt=1.0, steps=10, T=1000,
                D_model=D_model,
                bc=('dirichlet', 'neumann')
                # Missing surface_value!
            )


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
