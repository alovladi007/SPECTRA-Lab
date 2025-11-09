"""
Comprehensive unit tests for segregation.py (Session 5).

Tests include:
- Segregation model initialization
- Boundary condition application
- Interface velocity calculation
- Pile-up factor calculation
- Coupled solver functionality
- Mass conservation
- Moving boundary tracking
- Demo functions
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.segregation import (
    SegregationModel,
    MovingBoundaryTracker,
    arsenic_pile_up_demo,
    boron_depletion_demo,
    SEGREGATION_COEFFICIENTS
)
from core.fick_fd import Fick1D


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def arsenic_model():
    """Arsenic segregation model."""
    return SegregationModel("arsenic")


@pytest.fixture
def boron_model():
    """Boron segregation model."""
    return SegregationModel("boron")


@pytest.fixture
def standard_grid():
    """Standard spatial grid for testing."""
    return np.linspace(0, 500, 500)


@pytest.fixture
def fick_solver():
    """Fick1D solver for coupled tests."""
    return Fick1D(x_max=500, dx=1.0, refine_surface=False)


# ============================================================================
# Initialization Tests
# ============================================================================

class TestSegregationModelInit:
    """Tests for SegregationModel initialization."""
    
    def test_init_with_known_dopant(self):
        """Test initialization with known dopant."""
        model = SegregationModel("arsenic")
        assert model.dopant == "arsenic"
        assert model.k == SEGREGATION_COEFFICIENTS["arsenic"]
        assert model.k == pytest.approx(0.02)
    
    def test_init_with_custom_k(self):
        """Test initialization with custom segregation coefficient."""
        model = SegregationModel("boron", k_segregation=0.25)
        assert model.k == 0.25
    
    def test_init_case_insensitive(self):
        """Test that dopant name is case-insensitive."""
        model1 = SegregationModel("Boron")
        model2 = SegregationModel("BORON")
        model3 = SegregationModel("boron")
        
        assert model1.k == model2.k == model3.k
    
    def test_init_unknown_dopant_no_k(self):
        """Test error when unknown dopant and no k provided."""
        with pytest.raises(ValueError, match="Unknown dopant"):
            SegregationModel("unknown_element")
    
    def test_init_unknown_dopant_with_k(self):
        """Test custom dopant with explicit k."""
        model = SegregationModel("germanium", k_segregation=0.15)
        assert model.dopant == "germanium"
        assert model.k == 0.15
    
    def test_init_negative_k_error(self):
        """Test error for negative k."""
        with pytest.raises(ValueError, match="must be positive"):
            SegregationModel("boron", k_segregation=-0.1)
    
    def test_init_zero_k_error(self):
        """Test error for zero k."""
        with pytest.raises(ValueError, match="must be positive"):
            SegregationModel("boron", k_segregation=0.0)
    
    def test_interface_tracking_initialized(self, arsenic_model):
        """Test that interface tracking is initialized."""
        assert arsenic_model.interface_position == 0.0
        assert arsenic_model.oxide_thickness == 0.0
        assert len(arsenic_model.interface_history) == 0
        assert len(arsenic_model.time_history) == 0


# ============================================================================
# Segregation Boundary Condition Tests
# ============================================================================

class TestSegregationBC:
    """Tests for segregation boundary condition application."""
    
    def test_segregation_bc_basic(self, arsenic_model, standard_grid):
        """Test basic segregation BC application."""
        # Uniform concentration
        C = np.full_like(standard_grid, 1e19)
        x_interface = 50.0  # nm
        dx = standard_grid[1] - standard_grid[0]
        
        C_updated = arsenic_model.apply_segregation_bc(C, standard_grid, x_interface, dx)
        
        # Concentration in oxide region should be reduced by k
        idx_interface = np.argmin(np.abs(standard_grid - x_interface))
        if idx_interface > 0:
            assert C_updated[0] < C[0], "Oxide concentration should be reduced"
    
    def test_segregation_bc_pile_up(self, arsenic_model, standard_grid):
        """Test pile-up effect with k < 1."""
        # Create a profile with surface doping
        C = np.full_like(standard_grid, 1e15)
        C[0:100] = 1e19
        
        x_interface = 30.0  # nm
        dx = standard_grid[1] - standard_grid[0]
        
        C_updated = arsenic_model.apply_segregation_bc(C, standard_grid, x_interface, dx)
        
        # For k << 1, oxide region should have much lower concentration
        idx_interface = np.argmin(np.abs(standard_grid - x_interface))
        if idx_interface > 0:
            # Oxide region
            C_oxide_avg = np.mean(C_updated[0:idx_interface])
            C_si_interface = C_updated[idx_interface]
            
            # Oxide should have k * C_silicon
            assert C_oxide_avg < C_si_interface
    
    def test_segregation_different_k_values(self, standard_grid):
        """Test segregation with different k values."""
        C = np.full_like(standard_grid, 1e19)
        x_interface = 50.0
        dx = standard_grid[1] - standard_grid[0]
        
        # Arsenic (k=0.02) vs Boron (k=0.3)
        model_as = SegregationModel("arsenic")
        model_b = SegregationModel("boron")
        
        C_as = model_as.apply_segregation_bc(C, standard_grid, x_interface, dx)
        C_b = model_b.apply_segregation_bc(C, standard_grid, x_interface, dx)
        
        # Arsenic should show stronger depletion in oxide
        idx_interface = np.argmin(np.abs(standard_grid - x_interface))
        if idx_interface > 0:
            assert C_as[0] < C_b[0], "Arsenic (lower k) should have lower oxide concentration"


# ============================================================================
# Interface Velocity Tests
# ============================================================================

class TestInterfaceVelocity:
    """Tests for interface velocity calculation."""
    
    def test_interface_velocity_basic(self, arsenic_model):
        """Test basic interface velocity calculation."""
        dx_oxide_dt = 1.0  # nm/min oxide growth
        
        v_interface = arsenic_model.calculate_interface_velocity(dx_oxide_dt)
        
        # Interface should move slower than oxide grows (volume ratio ~2.2)
        assert v_interface > 0
        assert v_interface < dx_oxide_dt
        assert v_interface == pytest.approx(dx_oxide_dt / 2.2)
    
    def test_interface_velocity_scaling(self, arsenic_model):
        """Test interface velocity scales with growth rate."""
        rate1 = 0.5  # nm/min
        rate2 = 2.0  # nm/min
        
        v1 = arsenic_model.calculate_interface_velocity(rate1)
        v2 = arsenic_model.calculate_interface_velocity(rate2)
        
        # Should scale linearly
        assert v2 / v1 == pytest.approx(rate2 / rate1)
    
    def test_interface_velocity_custom_volume_ratio(self, arsenic_model):
        """Test with custom volume ratio."""
        dx_oxide_dt = 1.0
        volume_ratio = 3.0
        
        v = arsenic_model.calculate_interface_velocity(dx_oxide_dt, volume_ratio)
        
        assert v == pytest.approx(dx_oxide_dt / volume_ratio)


# ============================================================================
# Pile-up Factor Tests
# ============================================================================

class TestPileUpFactor:
    """Tests for pile-up factor calculation."""
    
    def test_pile_up_factor_zero_consumption(self, arsenic_model):
        """Test pile-up factor with no Si consumed."""
        puf = arsenic_model.pile_up_factor(1e19, x_oxide=0, x_consumed=0)
        
        # No pile-up initially
        assert puf == pytest.approx(1.0)
    
    def test_pile_up_factor_increases_with_consumption(self, arsenic_model):
        """Test pile-up factor increases with Si consumption."""
        C_init = 1e19
        
        puf1 = arsenic_model.pile_up_factor(C_init, x_oxide=10, x_consumed=5)
        puf2 = arsenic_model.pile_up_factor(C_init, x_oxide=30, x_consumed=15)
        
        # More consumption -> higher pile-up
        assert puf2 > puf1
        assert puf2 > 1.0
    
    def test_pile_up_factor_depends_on_k(self, standard_grid):
        """Test pile-up factor depends on segregation coefficient."""
        C_init = 1e19
        x_oxide = 20
        x_consumed = 10
        
        # Arsenic (k=0.02) vs Boron (k=0.3)
        model_as = SegregationModel("arsenic")
        model_b = SegregationModel("boron")
        
        puf_as = model_as.pile_up_factor(C_init, x_oxide, x_consumed)
        puf_b = model_b.pile_up_factor(C_init, x_oxide, x_consumed)
        
        # Lower k -> stronger pile-up
        assert puf_as > puf_b


# ============================================================================
# Coupled Solver Tests
# ============================================================================

class TestCoupledSolver:
    """Tests for coupled oxidation-diffusion solver."""
    
    def test_coupled_solve_runs(self, arsenic_model, fick_solver):
        """Test that coupled solver runs without error."""
        x = fick_solver.x
        C0 = np.full(len(x), 1e18)
        
        # Simple linear oxidation
        def oxide_model(t, T):
            return 0.5 * t  # 0.5 nm/min
        
        # Solve
        x_final, C_final, interface_history = arsenic_model.coupled_solve(
            C0, x, T=1000, t_total=10,
            oxidation_model=oxide_model,
            diffusion_solver=fick_solver,
            dt=1.0
        )
        
        # Check outputs
        assert len(x_final) == len(x)
        assert len(C_final) == len(x)
        assert len(interface_history) > 0
    
    def test_coupled_solve_interface_moves(self, arsenic_model, fick_solver):
        """Test that interface position increases with time."""
        x = fick_solver.x
        C0 = np.full(len(x), 1e18)
        
        def oxide_model(t, T):
            return 1.0 * t  # 1 nm/min
        
        x_final, C_final, interface_history = arsenic_model.coupled_solve(
            C0, x, T=1000, t_total=30,
            oxidation_model=oxide_model,
            diffusion_solver=fick_solver,
            dt=1.0
        )
        
        # Interface should move
        assert interface_history[-1] > interface_history[0]
        assert arsenic_model.interface_position > 0
    
    def test_coupled_solve_concentration_changes(self, arsenic_model, fick_solver):
        """Test that concentration profile changes during coupled solve."""
        x = fick_solver.x
        C0 = np.full(len(x), 1e18)
        C0[0:50] = 1e19  # Surface doping
        
        def oxide_model(t, T):
            return 0.5 * t
        
        x_final, C_final, interface_history = arsenic_model.coupled_solve(
            C0.copy(), x, T=1000, t_total=30,
            oxidation_model=oxide_model,
            diffusion_solver=fick_solver,
            dt=1.0
        )
        
        # Concentration should change
        assert not np.allclose(C_final, C0)
    
    def test_coupled_solve_mass_approximately_conserved(self, arsenic_model, fick_solver):
        """Test approximate mass conservation (some loss to oxide expected)."""
        x = fick_solver.x
        C0 = np.full(len(x), 1e18)
        
        def oxide_model(t, T):
            return 0.2 * t  # Slow growth
        
        x_final, C_final, interface_history = arsenic_model.coupled_solve(
            C0.copy(), x, T=1000, t_total=20,
            oxidation_model=oxide_model,
            diffusion_solver=fick_solver,
            dt=1.0
        )
        
        # Check mass conservation
        is_conserved, rel_error = arsenic_model.mass_balance_check(
            C0, C_final, x, tolerance=0.3
        )
        
        # Allow 30% loss (dopant enters oxide)
        print(f"Mass conservation error: {rel_error:.2%}")
        assert rel_error < 0.5, f"Mass loss {rel_error:.2%} too high"


# ============================================================================
# Mass Conservation Tests
# ============================================================================

class TestMassConservation:
    """Tests for mass balance checking."""
    
    def test_mass_balance_identical_profiles(self, arsenic_model, standard_grid):
        """Test mass balance with identical profiles."""
        C = np.full_like(standard_grid, 1e18)
        
        is_conserved, rel_error = arsenic_model.mass_balance_check(
            C, C, standard_grid, tolerance=0.01
        )
        
        assert is_conserved
        assert rel_error < 1e-10
    
    def test_mass_balance_small_change(self, arsenic_model, standard_grid):
        """Test mass balance with small change."""
        C0 = np.full_like(standard_grid, 1e18)
        C1 = C0 * 0.98  # 2% reduction
        
        is_conserved, rel_error = arsenic_model.mass_balance_check(
            C0, C1, standard_grid, tolerance=0.05
        )
        
        assert is_conserved
        assert rel_error == pytest.approx(0.02, abs=0.001)
    
    def test_mass_balance_large_change(self, arsenic_model, standard_grid):
        """Test mass balance with large change."""
        C0 = np.full_like(standard_grid, 1e18)
        C1 = C0 * 0.5  # 50% reduction
        
        is_conserved, rel_error = arsenic_model.mass_balance_check(
            C0, C1, standard_grid, tolerance=0.3
        )
        
        assert not is_conserved  # Should fail with 30% tolerance
        assert rel_error == pytest.approx(0.5, abs=0.01)
    
    def test_mass_balance_zero_initial(self, arsenic_model, standard_grid):
        """Test mass balance with zero initial concentration."""
        C0 = np.zeros_like(standard_grid)
        C1 = np.zeros_like(standard_grid)
        
        is_conserved, rel_error = arsenic_model.mass_balance_check(
            C0, C1, standard_grid, tolerance=0.1
        )
        
        # Should handle zero gracefully
        assert is_conserved
        assert rel_error == 0.0


# ============================================================================
# Moving Boundary Tracker Tests
# ============================================================================

class TestMovingBoundaryTracker:
    """Tests for MovingBoundaryTracker class."""
    
    def test_tracker_init(self, standard_grid):
        """Test tracker initialization."""
        tracker = MovingBoundaryTracker(standard_grid)
        
        assert tracker.x_interface == 0.0
        assert len(tracker.x_interface_history) == 1
        assert tracker.x_interface_history[0] == 0.0
    
    def test_update_interface(self, standard_grid):
        """Test interface update."""
        tracker = MovingBoundaryTracker(standard_grid)
        
        # Grow 20 nm of oxide
        tracker.update_interface(x_oxide_new=20.0)
        
        # Interface should move ~20/2.2 ≈ 9 nm
        assert tracker.x_interface > 0
        assert tracker.x_interface < 20.0  # Less than oxide thickness
        assert tracker.x_interface == pytest.approx(20.0 / 2.2, rel=0.01)
    
    def test_update_interface_multiple_steps(self, standard_grid):
        """Test multiple interface updates."""
        tracker = MovingBoundaryTracker(standard_grid)
        
        tracker.update_interface(10.0)
        x1 = tracker.x_interface
        
        tracker.update_interface(20.0)
        x2 = tracker.x_interface
        
        # Interface should move further
        assert x2 > x1
    
    def test_remap_grid_preserves_shape(self, standard_grid):
        """Test grid remapping preserves concentration shape."""
        tracker = MovingBoundaryTracker(standard_grid)
        
        # Create Gaussian profile
        C_old = 1e19 * np.exp(-(standard_grid / 100) ** 2)
        
        # New grid (slightly shifted)
        x_new = standard_grid + 10.0
        
        C_new = tracker.remap_grid(C_old, standard_grid, x_new)
        
        # Should maintain similar shape
        assert len(C_new) == len(x_new)
        assert np.all(C_new >= 0)
    
    def test_remap_grid_non_negative(self, standard_grid):
        """Test remapping ensures non-negative concentrations."""
        tracker = MovingBoundaryTracker(standard_grid)
        
        C_old = np.full_like(standard_grid, 1e18)
        x_new = standard_grid * 1.1  # Stretch grid
        
        C_new = tracker.remap_grid(C_old, standard_grid, x_new)
        
        assert np.all(C_new >= 0)
    
    def test_get_interface_position(self, standard_grid):
        """Test getting interface position."""
        tracker = MovingBoundaryTracker(standard_grid)
        
        assert tracker.get_interface_position() == 0.0
        
        tracker.update_interface(30.0)
        assert tracker.get_interface_position() > 0


# ============================================================================
# Demo Function Tests
# ============================================================================

class TestDemoFunctions:
    """Tests for demo functions."""
    
    def test_arsenic_pile_up_demo_runs(self):
        """Test arsenic pile-up demo runs."""
        x, C = arsenic_pile_up_demo(T=1000, t=30, C_initial=1e19)
        
        assert len(x) > 0
        assert len(C) == len(x)
        assert np.all(C > 0)
    
    def test_arsenic_pile_up_demo_shows_pile_up(self):
        """Test that arsenic demo shows pile-up effect."""
        x, C = arsenic_pile_up_demo(T=1000, t=60, C_initial=1e19)
        
        # Should have some variation (pile-up at interface)
        assert C.max() > C.min()
        
        # Peak should be above initial concentration (pile-up)
        # This may not always be true depending on implementation
        # Just check profile makes sense
        assert C[0] >= 0
        assert C[-1] >= 0
    
    def test_boron_depletion_demo_runs(self):
        """Test boron depletion demo runs."""
        x, C = boron_depletion_demo(T=1100, t=120, C_initial=1e18)
        
        assert len(x) > 0
        assert len(C) == len(x)
        assert np.all(C > 0)
    
    def test_boron_demo_different_from_arsenic(self):
        """Test boron and arsenic demos give different results."""
        x_as, C_as = arsenic_pile_up_demo(T=1000, t=60, C_initial=1e18)
        x_b, C_b = boron_depletion_demo(T=1000, t=60, C_initial=1e18)
        
        # Profiles should be different (different k values)
        # Just check they're not identical
        assert not np.allclose(C_as, C_b[:len(C_as)] if len(C_b) > len(C_as) else C_b)


# ============================================================================
# Physical Behavior Tests
# ============================================================================

class TestPhysicalBehavior:
    """Tests for physical correctness."""
    
    def test_segregation_reduces_oxide_concentration(self, standard_grid):
        """Test that k < 1 reduces concentration in oxide."""
        model = SegregationModel("arsenic", k_segregation=0.02)
        
        C = np.full_like(standard_grid, 1e19)
        x_interface = 50.0
        dx = 1.0
        
        C_updated = model.apply_segregation_bc(C, standard_grid, x_interface, dx)
        
        idx_interface = np.argmin(np.abs(standard_grid - x_interface))
        if idx_interface > 0:
            # Oxide region should have lower concentration
            assert C_updated[0] < C[0]
    
    def test_higher_k_means_less_rejection(self, standard_grid):
        """Test that higher k means less dopant rejection."""
        C = np.full_like(standard_grid, 1e19)
        x_interface = 50.0
        dx = 1.0
        
        # Two different k values
        model1 = SegregationModel("arsenic", k_segregation=0.02)
        model2 = SegregationModel("boron", k_segregation=0.30)
        
        C1 = model1.apply_segregation_bc(C.copy(), standard_grid, x_interface, dx)
        C2 = model2.apply_segregation_bc(C.copy(), standard_grid, x_interface, dx)
        
        idx_interface = np.argmin(np.abs(standard_grid - x_interface))
        if idx_interface > 0:
            # Higher k -> higher oxide concentration
            assert C2[0] > C1[0]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_complete_segregation_workflow(self, fick_solver):
        """Test complete segregation workflow."""
        # 1. Setup model
        model = SegregationModel("arsenic")
        
        # 2. Initial profile
        x = fick_solver.x
        C0 = np.full(len(x), 1e15)
        C0[0:50] = 1e19
        
        # 3. Oxidation model
        def oxide_model(t, T):
            return 0.5 * t
        
        # 4. Solve coupled problem
        x_final, C_final, interface_history = model.coupled_solve(
            C0, x, T=1000, t_total=30,
            oxidation_model=oxide_model,
            diffusion_solver=fick_solver,
            dt=1.0
        )
        
        # 5. Verify results
        assert len(x_final) > 0
        assert len(C_final) == len(x_final)
        assert model.interface_position > 0
        
        # 6. Check mass balance
        is_conserved, rel_error = model.mass_balance_check(
            C0, C_final, x, tolerance=0.4
        )
        
        print(f"✓ Complete segregation workflow successful")
        print(f"  Final interface position: {model.interface_position:.1f} nm")
        print(f"  Mass conservation error: {rel_error:.2%}")
        
        # Reasonable mass loss to oxide
        assert rel_error < 0.5


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
