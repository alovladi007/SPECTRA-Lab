"""
Comprehensive unit tests for erfc.py (Session 2).

Tests include:
- Function correctness
- Edge cases
- Physical constraints
- Monotonicity
- Convergence properties
- Integration with expected behavior
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.erfc import (
    diffusivity,
    constant_source_profile,
    limited_source_profile,
    junction_depth,
    sheet_resistance_estimate,
    two_step_diffusion,
    effective_diffusion_time,
    quick_profile_constant_source,
    quick_profile_limited_source,
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def boron_params():
    """Boron diffusion parameters."""
    return {"D0": 0.76, "Ea": 3.46}


@pytest.fixture
def phosphorus_params():
    """Phosphorus diffusion parameters."""
    return {"D0": 3.85, "Ea": 3.66}


@pytest.fixture
def standard_grid():
    """Standard spatial grid."""
    return np.linspace(0, 1000, 1000)


# ============================================================================
# Diffusivity Tests
# ============================================================================

class TestDiffusivity:
    """Tests for diffusivity function."""
    
    def test_temperature_dependence(self, boron_params):
        """Test that D increases with temperature."""
        T_low = 900
        T_high = 1100
        
        D_low = diffusivity(T_low, **boron_params)
        D_high = diffusivity(T_high, **boron_params)
        
        assert D_high > D_low, "Diffusivity should increase with temperature"
        assert D_low > 0, "Diffusivity must be positive"
        assert D_high > 0, "Diffusivity must be positive"
    
    def test_arrhenius_behavior(self, boron_params):
        """Test Arrhenius temperature dependence."""
        temperatures = np.array([900, 1000, 1100])
        D_values = np.array([diffusivity(T, **boron_params) for T in temperatures])
        
        # ln(D) vs 1/T should be linear
        T_kelvin = temperatures + 273.15
        inv_T = 1.0 / T_kelvin
        log_D = np.log(D_values)
        
        # Check linearity (R² > 0.99)
        coeffs = np.polyfit(inv_T, log_D, 1)
        fit = np.polyval(coeffs, inv_T)
        ss_res = np.sum((log_D - fit) ** 2)
        ss_tot = np.sum((log_D - np.mean(log_D)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        assert r_squared > 0.99, f"Arrhenius fit R² = {r_squared:.4f} < 0.99"
    
    def test_concentration_dependence(self, boron_params):
        """Test concentration-dependent diffusivity."""
        T = 1000
        C_low = np.array([1e18])
        C_high = np.array([1e20])
        
        D_low = diffusivity(T, **boron_params, C=C_low, alpha=1e-20, m=1)
        D_high = diffusivity(T, **boron_params, C=C_high, alpha=1e-20, m=1)
        
        assert D_high > D_low, "D should increase with concentration"
    
    def test_typical_values(self, boron_params):
        """Test that D is in expected range."""
        T = 1000
        D = diffusivity(T, **boron_params)
        
        # At 1000°C, boron D ~ 1e-13 cm²/s
        assert 1e-14 < D < 1e-12, f"D = {D:.2e} outside expected range"
    
    def test_different_dopants(self, boron_params, phosphorus_params):
        """Test different dopants have different D."""
        T = 1000
        
        D_B = diffusivity(T, **boron_params)
        D_P = diffusivity(T, **phosphorus_params)
        
        assert D_B != D_P, "Different dopants should have different D"
        # Phosphorus diffuses faster than boron
        assert D_P > D_B, "Phosphorus should diffuse faster than boron"


# ============================================================================
# Constant Source Tests
# ============================================================================

class TestConstantSourceProfile:
    """Tests for constant-source diffusion."""
    
    def test_surface_concentration(self, standard_grid, boron_params):
        """Test surface concentration equals Cs."""
        t = 30 * 60  # 30 min
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        C = constant_source_profile(
            standard_grid, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # Surface concentration should be Cs
        assert_allclose(C[0], Cs, rtol=1e-6)
    
    def test_monotonic_decay(self, standard_grid, boron_params):
        """Test profile is monotonically decreasing."""
        t = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        C = constant_source_profile(
            standard_grid, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # Check monotonicity
        dC = np.diff(C)
        assert np.all(dC <= 0), "Profile should be monotonically decreasing"
    
    def test_background_approached(self, standard_grid, boron_params):
        """Test concentration approaches background at depth."""
        t = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        C = constant_source_profile(
            standard_grid, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # At large depth, should approach NA0
        assert C[-1] < Cs * 0.01, "Deep concentration should be much less than Cs"
        assert C[-1] >= NA0 * 0.99, "Deep concentration should approach NA0"
    
    def test_time_dependence(self, standard_grid, boron_params):
        """Test longer time gives deeper profiles."""
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        t_short = 10 * 60
        t_long = 60 * 60
        
        C_short = constant_source_profile(
            standard_grid, t_short, T, **boron_params, Cs=Cs, NA0=NA0
        )
        C_long = constant_source_profile(
            standard_grid, t_long, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # At mid-depth, longer time should give higher concentration
        mid_idx = len(standard_grid) // 2
        assert C_long[mid_idx] > C_short[mid_idx], \
            "Longer diffusion should give deeper penetration"
    
    def test_temperature_dependence(self, standard_grid, boron_params):
        """Test higher temperature gives deeper profiles."""
        t = 30 * 60
        Cs = 1e20
        NA0 = 1e15
        
        T_low = 900
        T_high = 1100
        
        C_low = constant_source_profile(
            standard_grid, t, T_low, **boron_params, Cs=Cs, NA0=NA0
        )
        C_high = constant_source_profile(
            standard_grid, t, T_high, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # At mid-depth, higher T should give higher concentration
        mid_idx = len(standard_grid) // 2
        assert C_high[mid_idx] > C_low[mid_idx], \
            "Higher temperature should give deeper diffusion"
    
    def test_scaling_with_sqrt_dt(self, boron_params):
        """Test that profile scales with sqrt(D*t)."""
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        # Two different conditions with same D*t
        t1 = 15 * 60
        t2 = 60 * 60
        
        # At same D*t product, profiles should be similar
        # For this test, we use different temperatures to keep D*t constant
        # D1*t1 = D2*t2 => T1 and T2 chosen accordingly
        # This is complex, so we just check sqrt(D*t) scaling
        
        D = diffusivity(T, **boron_params)
        x1 = np.linspace(0, 500, 500)
        x2 = x1 * np.sqrt(t2 / t1)  # Scale depth by sqrt(t2/t1)
        
        C1 = constant_source_profile(x1, t1, T, **boron_params, Cs=Cs, NA0=NA0)
        C2 = constant_source_profile(x2, t2, T, **boron_params, Cs=Cs, NA0=NA0)
        
        # Profiles should be similar when depth is scaled
        assert_allclose(C1, C2, rtol=0.1)
    
    def test_physical_bounds(self, standard_grid, boron_params):
        """Test concentration stays within physical bounds."""
        t = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        C = constant_source_profile(
            standard_grid, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # All concentrations should be between NA0 and Cs
        assert np.all(C >= NA0), "Concentrations below background"
        assert np.all(C <= Cs), "Concentrations above surface value"
    
    def test_invalid_inputs(self, standard_grid, boron_params):
        """Test error handling for invalid inputs."""
        Cs = 1e20
        NA0 = 1e15
        
        # Negative time
        with pytest.raises(ValueError):
            constant_source_profile(
                standard_grid, -10, 1000, **boron_params, Cs=Cs, NA0=NA0
            )
        
        # Zero time
        with pytest.raises(ValueError):
            constant_source_profile(
                standard_grid, 0, 1000, **boron_params, Cs=Cs, NA0=NA0
            )
        
        # Cs < NA0
        with pytest.raises(ValueError):
            constant_source_profile(
                standard_grid, 1800, 1000, **boron_params, Cs=1e14, NA0=1e15
            )


# ============================================================================
# Limited Source Tests
# ============================================================================

class TestLimitedSourceProfile:
    """Tests for limited-source (Gaussian) diffusion."""
    
    def test_peak_at_surface(self, standard_grid, boron_params):
        """Test peak concentration is at surface."""
        t = 30 * 60
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        C = limited_source_profile(
            standard_grid, t, T, **boron_params, Q=Q, NA0=NA0
        )
        
        # Peak should be at x=0
        assert C[0] == np.max(C), "Peak should be at surface"
    
    def test_gaussian_shape(self, standard_grid, boron_params):
        """Test profile has Gaussian shape."""
        t = 30 * 60
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        C = limited_source_profile(
            standard_grid, t, T, **boron_params, Q=Q, NA0=NA0
        )
        
        # Subtract background
        C_excess = C - NA0
        
        # Take log of excess
        # For Gaussian: ln(C) = ln(C_peak) - x²/(4Dt)
        # So ln(C) should be parabolic in x²
        x_cm = standard_grid * 1e-7
        
        # Find where C_excess is significant
        idx = C_excess > NA0 * 0.01
        if np.sum(idx) > 10:
            log_C = np.log(C_excess[idx])
            x_squared = x_cm[idx] ** 2
            
            # Fit parabola
            coeffs = np.polyfit(x_squared, log_C, 2)
            
            # Linear term should be small (parabola in x², not x)
            # Quadratic term should be negative
            assert coeffs[0] < 0, "Gaussian should be concave in x²"
    
    def test_dose_conservation(self, boron_params):
        """Test total dose is conserved."""
        x = np.linspace(0, 1000, 10000)  # Fine grid
        t = 30 * 60
        T = 950
        Q_input = 1e14
        NA0 = 1e15
        
        C = limited_source_profile(
            x, t, T, **boron_params, Q=Q_input, NA0=NA0
        )
        
        # Integrate to get dose
        x_cm = x * 1e-7
        C_excess = C - NA0
        Q_calculated = np.trapz(C_excess, x_cm)
        
        # Should match input dose within 5%
        assert_allclose(Q_calculated, Q_input, rtol=0.05)
    
    def test_peak_decreases_with_time(self, standard_grid, boron_params):
        """Test peak concentration decreases with longer time."""
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        t_short = 10 * 60
        t_long = 60 * 60
        
        C_short = limited_source_profile(
            standard_grid, t_short, T, **boron_params, Q=Q, NA0=NA0
        )
        C_long = limited_source_profile(
            standard_grid, t_long, T, **boron_params, Q=Q, NA0=NA0
        )
        
        # Peak should decrease (dose spreads)
        assert C_short[0] > C_long[0], \
            "Peak concentration should decrease with time"
    
    def test_spreading_with_time(self, standard_grid, boron_params):
        """Test profile spreads wider with time."""
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        t_short = 10 * 60
        t_long = 60 * 60
        
        C_short = limited_source_profile(
            standard_grid, t_short, T, **boron_params, Q=Q, NA0=NA0
        )
        C_long = limited_source_profile(
            standard_grid, t_long, T, **boron_params, Q=Q, NA0=NA0
        )
        
        # At large depth, longer time should give higher concentration
        deep_idx = len(standard_grid) * 3 // 4
        assert C_long[deep_idx] > C_short[deep_idx], \
            "Profile should spread deeper with time"


# ============================================================================
# Junction Depth Tests
# ============================================================================

class TestJunctionDepth:
    """Tests for junction depth calculation."""
    
    def test_constant_source_junction(self, standard_grid, boron_params):
        """Test junction depth for constant source."""
        t = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        C = constant_source_profile(
            standard_grid, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        xj = junction_depth(C, standard_grid, NA0)
        
        # Junction should be within grid
        assert 0 < xj < standard_grid[-1], "Junction outside grid"
        
        # Concentration at xj should be close to NA0
        idx_xj = np.argmin(np.abs(standard_grid - xj))
        assert_allclose(C[idx_xj], NA0, rtol=0.1)
    
    def test_deeper_junction_with_time(self, standard_grid, boron_params):
        """Test longer time gives deeper junction."""
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        t_short = 10 * 60
        t_long = 60 * 60
        
        C_short = constant_source_profile(
            standard_grid, t_short, T, **boron_params, Cs=Cs, NA0=NA0
        )
        C_long = constant_source_profile(
            standard_grid, t_long, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        xj_short = junction_depth(C_short, standard_grid, NA0)
        xj_long = junction_depth(C_long, standard_grid, NA0)
        
        assert xj_long > xj_short, "Longer time should give deeper junction"
    
    def test_no_junction_error(self, standard_grid):
        """Test error when no junction exists."""
        # Profile entirely above background
        C_high = np.full_like(standard_grid, 1e20, dtype=float)
        
        with pytest.raises(ValueError, match="No junction found"):
            junction_depth(C_high, standard_grid, 1e15)
        
        # Profile entirely below background
        C_low = np.full_like(standard_grid, 1e14, dtype=float)
        
        with pytest.raises(ValueError, match="No junction found"):
            junction_depth(C_low, standard_grid, 1e15)
    
    def test_interpolation_methods(self, standard_grid, boron_params):
        """Test different interpolation methods give similar results."""
        t = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        C = constant_source_profile(
            standard_grid, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        xj_linear = junction_depth(C, standard_grid, NA0, method="linear")
        xj_log = junction_depth(C, standard_grid, NA0, method="log")
        
        # Should be within 10% of each other
        assert_allclose(xj_linear, xj_log, rtol=0.1)


# ============================================================================
# Sheet Resistance Tests
# ============================================================================

class TestSheetResistance:
    """Tests for sheet resistance calculation."""
    
    def test_positive_resistance(self, standard_grid, boron_params):
        """Test sheet resistance is positive."""
        t = 30 * 60
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        C = limited_source_profile(
            standard_grid, t, T, **boron_params, Q=Q, NA0=NA0
        )
        
        Rs = sheet_resistance_estimate(C, standard_grid)
        
        assert Rs > 0, "Sheet resistance must be positive"
        assert np.isfinite(Rs), "Sheet resistance must be finite"
    
    def test_lower_rs_with_higher_dose(self, standard_grid, boron_params):
        """Test higher dose gives lower sheet resistance."""
        t = 30 * 60
        T = 950
        NA0 = 1e15
        
        Q_low = 1e13
        Q_high = 1e14
        
        C_low = limited_source_profile(
            standard_grid, t, T, **boron_params, Q=Q_low, NA0=NA0
        )
        C_high = limited_source_profile(
            standard_grid, t, T, **boron_params, Q=Q_high, NA0=NA0
        )
        
        Rs_low = sheet_resistance_estimate(C_low, standard_grid)
        Rs_high = sheet_resistance_estimate(C_high, standard_grid)
        
        assert Rs_high < Rs_low, \
            "Higher dose should give lower sheet resistance"
    
    def test_typical_range(self, standard_grid, boron_params):
        """Test Rs is in typical range."""
        t = 30 * 60
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        C = limited_source_profile(
            standard_grid, t, T, **boron_params, Q=Q, NA0=NA0
        )
        
        Rs = sheet_resistance_estimate(C, standard_grid)
        
        # Typical diffused layer: 10-1000 Ω/□
        assert 1 < Rs < 10000, f"Rs = {Rs:.1f} Ω/□ outside typical range"
    
    def test_mobility_models(self, standard_grid, boron_params):
        """Test different mobility models."""
        t = 30 * 60
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        C = limited_source_profile(
            standard_grid, t, T, **boron_params, Q=Q, NA0=NA0
        )
        
        Rs_const = sheet_resistance_estimate(
            C, standard_grid, mobility_model="constant"
        )
        Rs_ct = sheet_resistance_estimate(
            C, standard_grid, mobility_model="caughey_thomas"
        )
        
        # Both should be positive and different
        assert Rs_const > 0
        assert Rs_ct > 0
        assert Rs_const != Rs_ct


# ============================================================================
# Two-Step Diffusion Tests
# ============================================================================

class TestTwoStepDiffusion:
    """Tests for two-step diffusion process."""
    
    def test_two_step_profiles(self, standard_grid, boron_params):
        """Test two-step diffusion returns two profiles."""
        C_predep, C_drivein = two_step_diffusion(
            standard_grid,
            t1=15*60, T1=900,
            t2=30*60, T2=1100,
            **boron_params,
            Cs=1e20, NA0=1e15
        )
        
        assert len(C_predep) == len(standard_grid)
        assert len(C_drivein) == len(standard_grid)
    
    def test_drivein_deeper_than_predep(self, standard_grid, boron_params):
        """Test drive-in gives deeper profile than pre-dep."""
        C_predep, C_drivein = two_step_diffusion(
            standard_grid,
            t1=15*60, T1=900,
            t2=30*60, T2=1100,
            **boron_params,
            Cs=1e20, NA0=1e15
        )
        
        # At deep positions, drive-in should have higher concentration
        deep_idx = len(standard_grid) * 3 // 4
        assert C_drivein[deep_idx] > C_predep[deep_idx], \
            "Drive-in should penetrate deeper"
    
    def test_drivein_lower_peak(self, standard_grid, boron_params):
        """Test drive-in has lower peak than pre-dep."""
        C_predep, C_drivein = two_step_diffusion(
            standard_grid,
            t1=15*60, T1=900,
            t2=30*60, T2=1100,
            **boron_params,
            Cs=1e20, NA0=1e15
        )
        
        # Drive-in peak should be lower (dose spreads)
        assert C_drivein[0] < C_predep[0], \
            "Drive-in peak should be lower than pre-dep"


# ============================================================================
# Effective Time Tests
# ============================================================================

class TestEffectiveDiffusionTime:
    """Tests for effective diffusion time calculation."""
    
    def test_isothermal_case(self, boron_params):
        """Test effective time equals actual time for isothermal."""
        T_ref = 1000
        t_total = 30 * 60
        
        # Isothermal at T_ref
        T_profile = np.full(100, T_ref)
        time_points = np.linspace(0, t_total, 100)
        
        t_eff = effective_diffusion_time(
            T_profile, time_points, **boron_params, T_ref=T_ref
        )
        
        # Should equal actual time
        assert_allclose(t_eff, t_total, rtol=0.01)
    
    def test_higher_temp_gives_longer_eff_time(self, boron_params):
        """Test higher temperature gives longer effective time."""
        T_ref = 1000
        t_total = 30 * 60
        
        # Higher temperature
        T_high = np.full(100, 1100)
        T_ref_const = np.full(100, T_ref)
        time_points = np.linspace(0, t_total, 100)
        
        t_eff_high = effective_diffusion_time(
            T_high, time_points, **boron_params, T_ref=T_ref
        )
        t_eff_ref = effective_diffusion_time(
            T_ref_const, time_points, **boron_params, T_ref=T_ref
        )
        
        # Higher T should give longer effective time
        assert t_eff_high > t_eff_ref


# ============================================================================
# Quick Helper Tests
# ============================================================================

class TestQuickHelpers:
    """Tests for quick helper functions."""
    
    def test_quick_constant_source(self):
        """Test quick constant source helper."""
        x, C = quick_profile_constant_source(
            dopant="boron",
            time_minutes=30,
            temp_celsius=1000
        )
        
        assert len(x) == len(C)
        assert np.all(C >= 0)
        assert C[0] > C[-1]  # Decreasing profile
    
    def test_quick_limited_source(self):
        """Test quick limited source helper."""
        x, C = quick_profile_limited_source(
            dopant="phosphorus",
            time_minutes=20,
            temp_celsius=950
        )
        
        assert len(x) == len(C)
        assert np.all(C >= 0)
        assert C[0] == np.max(C)  # Peak at surface


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_complete_workflow_constant_source(self, standard_grid, boron_params):
        """Test complete workflow for constant source."""
        # 1. Generate profile
        t = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        C = constant_source_profile(
            standard_grid, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        
        # 2. Calculate junction depth
        xj = junction_depth(C, standard_grid, NA0)
        assert 0 < xj < standard_grid[-1]
        
        # 3. Calculate sheet resistance
        Rs = sheet_resistance_estimate(C, standard_grid)
        assert Rs > 0
        
        # 4. Verify consistency
        # Concentration at xj should be NA0
        idx_xj = np.argmin(np.abs(standard_grid - xj))
        assert_allclose(C[idx_xj], NA0, rtol=0.2)
    
    def test_complete_workflow_limited_source(self, standard_grid, phosphorus_params):
        """Test complete workflow for limited source."""
        # 1. Generate profile
        t = 20 * 60
        T = 950
        Q = 1e14
        NA0 = 1e15
        
        C = limited_source_profile(
            standard_grid, t, T, **phosphorus_params, Q=Q, NA0=NA0
        )
        
        # 2. Verify dose conservation
        x_cm = standard_grid * 1e-7
        Q_calc = np.trapz(C - NA0, x_cm)
        assert_allclose(Q_calc, Q, rtol=0.05)
        
        # 3. Calculate junction depth
        xj = junction_depth(C, standard_grid, NA0)
        assert 0 < xj < standard_grid[-1]
        
        # 4. Calculate sheet resistance
        Rs = sheet_resistance_estimate(C, standard_grid, "n")
        assert 1 < Rs < 10000


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance and scaling tests."""
    
    def test_large_grid_performance(self, boron_params):
        """Test performance with large grid."""
        import time
        
        # Large grid
        x = np.linspace(0, 5000, 50000)
        t = 30 * 60
        T = 1000
        Cs = 1e20
        NA0 = 1e15
        
        start = time.time()
        C = constant_source_profile(
            x, t, T, **boron_params, Cs=Cs, NA0=NA0
        )
        elapsed = time.time() - start
        
        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Large grid took {elapsed:.2f}s > 1s"
        assert len(C) == len(x)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
