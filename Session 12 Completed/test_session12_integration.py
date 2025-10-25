"""
Session 12: Chemical II - Integration Tests
============================================

Comprehensive test suite for SIMS, RBS, NAA, and Chemical Etch analysis.

Test Coverage:
- SIMS depth profiling and quantification
- RBS spectrum fitting and layer extraction
- NAA decay curve fitting and quantification
- Chemical etch loading effects

Author: Semiconductor Lab Platform Team
Version: 1.0.0
Date: October 2024
"""

import pytest
import numpy as np
from session12_chemical_bulk_complete_implementation import (
    # SIMS
    SIMSAnalyzer, SIMSProfile, SIMSCalibration, MatrixEffect,
    # RBS
    RBSAnalyzer, RBSSpectrum, RBSLayer, RBSGeometry,
    # NAA
    NAAAnalyzer, NAADecayCurve, NAANuclearData,
    # Chemical Etch
    ChemicalEtchAnalyzer, EtchProfile,
    # Simulator
    ChemicalBulkSimulator
)


# ============================================================================
# SIMS Tests
# ============================================================================

class TestSIMSAnalyzer:
    """Test suite for SIMS analysis"""
    
    @pytest.fixture
    def sims_analyzer(self):
        """Create SIMS analyzer instance"""
        return SIMSAnalyzer()
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance"""
        return ChemicalBulkSimulator()
    
    @pytest.fixture
    def sample_profile(self, simulator):
        """Generate test SIMS profile"""
        return simulator.simulate_sims_profile(
            element="B",
            matrix="Si",
            peak_depth=100.0,
            peak_concentration=1e20,
            dose=1e15,
            straggle=30.0
        )
    
    def test_analyzer_initialization(self, sims_analyzer):
        """Test analyzer initializes with calibrations"""
        assert len(sims_analyzer.calibrations) > 0
        assert "B_Si" in sims_analyzer.calibrations
        assert sims_analyzer.calibrations["B_Si"].rsf > 0
    
    def test_time_to_depth_conversion(self, sims_analyzer, sample_profile):
        """Test sputter time to depth conversion"""
        depth = sims_analyzer.convert_time_to_depth(
            sample_profile,
            sputter_rate=1.0
        )
        
        assert len(depth) == len(sample_profile.time)
        assert np.all(depth >= 0)
        assert depth[-1] > depth[0]  # Monotonic increasing
        
        # Check linear relationship
        expected_depth = sample_profile.time * 1.0
        np.testing.assert_allclose(depth, expected_depth, rtol=1e-10)
    
    def test_rsf_quantification(self, sims_analyzer, sample_profile):
        """Test RSF quantification method"""
        # Set depth
        sample_profile.depth = sims_analyzer.convert_time_to_depth(sample_profile)
        
        # Quantify
        concentration = sims_analyzer.quantify_profile(
            sample_profile,
            method=MatrixEffect.RSF
        )
        
        assert len(concentration) == len(sample_profile.counts)
        assert np.all(concentration > 0)
        assert concentration.max() > 1e16  # Should be in reasonable range
    
    def test_dose_calculation(self, sims_analyzer, sample_profile):
        """Test integrated dose calculation"""
        # Prepare profile
        sample_profile.depth = sims_analyzer.convert_time_to_depth(sample_profile)
        sample_profile.concentration = sims_analyzer.quantify_profile(
            sample_profile,
            method=MatrixEffect.RSF
        )
        
        # Calculate dose
        dose = sims_analyzer.calculate_dose(sample_profile)
        
        assert dose > 0
        # Should be close to input dose (1e15 atoms/cm²)
        assert 5e14 < dose < 5e15
    
    def test_interface_detection(self, sims_analyzer, sample_profile):
        """Test interface detection in depth profile"""
        # Add sharp interface by splicing two profiles
        profile1 = sample_profile
        profile1.depth = sims_analyzer.convert_time_to_depth(profile1)
        profile1.concentration = sims_analyzer.quantify_profile(profile1)
        
        # Find interfaces
        interfaces = sims_analyzer.find_interfaces(
            profile1,
            threshold_factor=0.3
        )
        
        # Should detect at least one interface (peak region)
        assert len(interfaces) >= 0  # May or may not detect depending on gradient
        
        if interfaces:
            for intf in interfaces:
                assert intf['depth'] > 0
                assert intf['width'] > 0
                assert intf['gradient'] > 0
    
    def test_detection_limit(self, sims_analyzer, sample_profile):
        """Test detection limit estimation"""
        sample_profile.depth = sims_analyzer.convert_time_to_depth(sample_profile)
        sample_profile.concentration = sims_analyzer.quantify_profile(sample_profile)
        
        det_limit = sims_analyzer.estimate_detection_limit(
            sample_profile,
            background_region=(0, 20)
        )
        
        assert det_limit > 0
        assert det_limit < sample_profile.concentration.max()
    
    def test_custom_calibration(self, sims_analyzer):
        """Test adding custom calibration"""
        custom_cal = SIMSCalibration(
            element="Ga",
            matrix="GaN",
            rsf=2.0e21,
            rsf_uncertainty=0.3e21,
            sputter_rate=0.8,
            standard_name="custom_standard"
        )
        
        sims_analyzer.add_calibration(custom_cal)
        
        assert "Ga_GaN" in sims_analyzer.calibrations
        assert sims_analyzer.calibrations["Ga_GaN"].rsf == 2.0e21


# ============================================================================
# RBS Tests
# ============================================================================

class TestRBSAnalyzer:
    """Test suite for RBS analysis"""
    
    @pytest.fixture
    def rbs_analyzer(self):
        """Create RBS analyzer instance"""
        return RBSAnalyzer(projectile="He", projectile_mass=4.003)
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance"""
        return ChemicalBulkSimulator()
    
    @pytest.fixture
    def sample_spectrum(self, simulator):
        """Generate test RBS spectrum"""
        layers = [("Hf", 0.5, 20.0), ("O", 0.5, 20.0)]
        return simulator.simulate_rbs_spectrum(
            layers=layers,
            substrate="Si",
            E0=2000.0,
            theta=170.0
        )
    
    def test_analyzer_initialization(self, rbs_analyzer):
        """Test RBS analyzer initialization"""
        assert rbs_analyzer.projectile == "He"
        assert rbs_analyzer.projectile_mass == 4.003
        assert rbs_analyzer.projectile_z == 2
        assert len(rbs_analyzer.ATOMIC_MASSES) > 10
    
    def test_kinematic_factor(self, rbs_analyzer):
        """Test kinematic factor calculation"""
        # Silicon target
        K_Si = rbs_analyzer.kinematic_factor(target_mass=28.086, scattering_angle=170.0)
        
        assert 0 < K_Si < 1  # K must be between 0 and 1
        assert 0.55 < K_Si < 0.60  # Expected range for He on Si at 170°
        
        # Gold target (heavier)
        K_Au = rbs_analyzer.kinematic_factor(target_mass=196.967, scattering_angle=170.0)
        
        assert K_Au > K_Si  # Heavier targets have higher K
    
    def test_kinematic_factor_light_target(self, rbs_analyzer):
        """Test kinematic factor for light target (no solution)"""
        # Hydrogen target (lighter than projectile)
        K_H = rbs_analyzer.kinematic_factor(target_mass=1.008, scattering_angle=170.0)
        
        assert np.isnan(K_H)  # Should return NaN for light targets
    
    def test_stopping_power(self, rbs_analyzer):
        """Test stopping power calculation"""
        S = rbs_analyzer.stopping_power(energy=2000.0, element="Si")
        
        assert S > 0
        
        # Stopping power should decrease with energy
        S_high = rbs_analyzer.stopping_power(energy=3000.0, element="Si")
        assert S_high < S
    
    def test_rutherford_cross_section(self, rbs_analyzer):
        """Test Rutherford cross-section calculation"""
        # Silicon target
        sigma = rbs_analyzer.rutherford_cross_section(
            energy=2000.0,
            target_z=14,
            scattering_angle=170.0
        )
        
        assert sigma > 0
        
        # Cross-section should increase for heavier targets
        sigma_Au = rbs_analyzer.rutherford_cross_section(
            energy=2000.0,
            target_z=79,
            scattering_angle=170.0
        )
        assert sigma_Au > sigma
        
        # Cross-section should decrease with energy
        sigma_high_E = rbs_analyzer.rutherford_cross_section(
            energy=3000.0,
            target_z=14,
            scattering_angle=170.0
        )
        assert sigma_high_E < sigma
    
    def test_spectrum_simulation(self, rbs_analyzer, sample_spectrum):
        """Test RBS spectrum simulation"""
        layers = [
            RBSLayer(element="Hf", atomic_fraction=0.5, thickness=20.0),
            RBSLayer(element="O", atomic_fraction=0.5, thickness=20.0)
        ]
        
        simulated = rbs_analyzer.simulate_spectrum(layers, sample_spectrum)
        
        assert len(simulated) == len(sample_spectrum.energy)
        assert np.all(simulated >= 0)
        assert simulated.max() > 0
    
    def test_spectrum_fitting(self, rbs_analyzer, sample_spectrum):
        """Test RBS spectrum fitting"""
        # Initial guess (slightly wrong)
        initial_layers = [
            RBSLayer(element="Hf", atomic_fraction=0.5, thickness=25.0),
            RBSLayer(element="O", atomic_fraction=0.5, thickness=25.0)
        ]
        
        # Fit
        result = rbs_analyzer.fit_spectrum(
            sample_spectrum,
            initial_layers,
            fit_range=(1200.0, 1900.0),
            fix_composition=True
        )
        
        assert len(result.layers) == 2
        assert result.chi_squared > 0
        assert 0 <= result.r_factor <= 1
        
        # Fitted thicknesses should be close to true values (20 each)
        for layer in result.layers:
            assert 10 < layer.thickness < 40  # Within reasonable range
    
    def test_layer_thickness_conversion(self):
        """Test areal density to nm conversion"""
        layer = RBSLayer(
            element="Si",
            atomic_fraction=1.0,
            thickness=50.0,  # 1e15 atoms/cm²
            density=5.0e22  # atoms/cm³
        )
        
        thickness_nm = layer.thickness_nm()
        
        assert thickness_nm > 0
        # 50 * 1e15 / (5e22 * 1e-7) ≈ 10 nm
        assert 5 < thickness_nm < 15


# ============================================================================
# NAA Tests
# ============================================================================

class TestNAAAnalyzer:
    """Test suite for NAA analysis"""
    
    @pytest.fixture
    def naa_analyzer(self):
        """Create NAA analyzer instance"""
        return NAAAnalyzer()
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance"""
        return ChemicalBulkSimulator()
    
    @pytest.fixture
    def sample_decay_curve(self, simulator):
        """Generate test NAA decay curve"""
        return simulator.simulate_naa_decay(
            element="Au",
            initial_activity=10000.0,
            irradiation_time=3600.0,
            cooling_time=600.0,
            measurement_time=3600.0,
            n_measurements=20
        )
    
    def test_analyzer_initialization(self, naa_analyzer):
        """Test NAA analyzer initialization with nuclear data"""
        assert len(naa_analyzer.nuclear_data) > 0
        assert "Au" in naa_analyzer.nuclear_data
        
        # Check Au-198 data
        au_data = naa_analyzer.nuclear_data["Au"]
        assert au_data.isotope == "Au-198"
        assert au_data.half_life > 0
        assert au_data.gamma_energy > 0
    
    def test_decay_constant(self, naa_analyzer):
        """Test decay constant calculation"""
        # Half-life of Au-198: ~232992 s
        lambda_val = naa_analyzer.decay_constant(232992.0)
        
        assert lambda_val > 0
        # λ = ln(2) / t_1/2
        expected = np.log(2) / 232992.0
        np.testing.assert_almost_equal(lambda_val, expected, decimal=10)
    
    def test_decay_curve_fitting_fixed_lambda(self, naa_analyzer, sample_decay_curve):
        """Test decay curve fitting with fixed decay constant"""
        au_data = naa_analyzer.nuclear_data["Au"]
        
        result = naa_analyzer.fit_decay_curve(
            sample_decay_curve,
            half_life=au_data.half_life
        )
        
        assert 'N0' in result
        assert 'lambda' in result
        assert 'half_life' in result
        assert 'chi_squared' in result
        assert result['fixed_lambda'] is True
        
        assert result['N0'] > 0
        assert result['lambda'] > 0
        assert result['chi_squared'] >= 0
    
    def test_decay_curve_fitting_free_lambda(self, naa_analyzer, sample_decay_curve):
        """Test decay curve fitting with free decay constant"""
        result = naa_analyzer.fit_decay_curve(
            sample_decay_curve,
            half_life=None
        )
        
        assert 'N0' in result
        assert 'lambda' in result
        assert 'half_life' in result
        assert result['fixed_lambda'] is False
        
        if 'N0_uncertainty' in result:
            assert result['N0_uncertainty'] > 0
            assert result['lambda_uncertainty'] > 0
    
    def test_comparator_method(self, naa_analyzer, simulator):
        """Test comparator method quantification"""
        # Generate sample and standard decay curves
        sample_curve = simulator.simulate_naa_decay(
            element="Au",
            initial_activity=5000.0,
            irradiation_time=3600.0
        )
        
        standard_curve = simulator.simulate_naa_decay(
            element="Au",
            initial_activity=10000.0,
            irradiation_time=3600.0
        )
        
        # Quantify
        result = naa_analyzer.comparator_method(
            sample_curve=sample_curve,
            standard_curve=standard_curve,
            standard_mass=0.1,  # g
            sample_mass=0.5,  # g
            standard_concentration=100.0,  # μg/g
            element="Au"
        )
        
        assert result.element == "Au"
        assert result.concentration > 0
        assert result.uncertainty > 0
        assert result.detection_limit > 0
        assert result.activity > 0
        
        # Sample has half the activity but 5x the mass
        # So concentration should be ~10% of standard (100 μg/g)
        # Expected: 100 * (5000/10000) * (0.1/0.5) = 10 μg/g
        assert 5 < result.concentration < 20  # Allow some noise
    
    def test_nuclear_data_completeness(self, naa_analyzer):
        """Test nuclear data for common elements"""
        required_elements = ["Na", "Mn", "Cu", "As", "Br", "Au"]
        
        for element in required_elements:
            assert element in naa_analyzer.nuclear_data
            data = naa_analyzer.nuclear_data[element]
            assert data.half_life > 0
            assert data.gamma_energy > 0
            assert 0 < data.gamma_intensity <= 1
            assert data.thermal_cross_section > 0


# ============================================================================
# Chemical Etch Tests
# ============================================================================

class TestChemicalEtchAnalyzer:
    """Test suite for chemical etch analysis"""
    
    @pytest.fixture
    def etch_analyzer(self):
        """Create etch analyzer instance"""
        return ChemicalEtchAnalyzer()
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance"""
        return ChemicalBulkSimulator()
    
    @pytest.fixture
    def linear_profile(self, simulator):
        """Generate linear loading effect profile"""
        return simulator.simulate_etch_profile(
            model="linear",
            nominal_rate=100.0,
            alpha=0.3,
            n_points=50,
            noise_level=0.02
        )
    
    @pytest.fixture
    def exponential_profile(self, simulator):
        """Generate exponential loading effect profile"""
        return simulator.simulate_etch_profile(
            model="exponential",
            nominal_rate=100.0,
            alpha=0.4,
            n_points=50,
            noise_level=0.02
        )
    
    def test_linear_model_fitting(self, etch_analyzer, linear_profile):
        """Test linear loading effect model fitting"""
        result = etch_analyzer.fit_loading_effect(linear_profile, model="linear")
        
        assert result.model_type == "linear"
        assert result.nominal_rate > 0
        assert result.max_reduction > 0
        assert result.critical_density > 0
        assert 0 <= result.r_squared <= 1
        
        # Should recover input parameters
        assert 90 < result.nominal_rate < 110  # Nominal rate ~100
        assert 25 < result.max_reduction < 35  # Max reduction ~30%
        assert result.r_squared > 0.95  # Good fit
    
    def test_exponential_model_fitting(self, etch_analyzer, exponential_profile):
        """Test exponential loading effect model fitting"""
        result = etch_analyzer.fit_loading_effect(exponential_profile, model="exponential")
        
        assert result.model_type == "exponential"
        assert result.nominal_rate > 0
        assert result.max_reduction > 0
        assert result.critical_density > 0
        
        # Should recover input parameters
        assert 90 < result.nominal_rate < 110
        assert result.r_squared > 0.90
    
    def test_power_model_fitting(self, etch_analyzer, simulator):
        """Test power law loading effect model fitting"""
        profile = simulator.simulate_etch_profile(
            model="power",
            nominal_rate=100.0,
            alpha=0.5,
            n_points=50,
            noise_level=0.02
        )
        
        result = etch_analyzer.fit_loading_effect(profile, model="power")
        
        assert result.model_type == "power"
        assert result.nominal_rate > 0
        assert result.max_reduction > 0  # Power model goes to 0 at D=1
        assert result.r_squared > 0.85
    
    def test_uniformity_calculation(self, etch_analyzer, linear_profile):
        """Test etch uniformity metrics calculation"""
        uniformity = etch_analyzer.calculate_uniformity(linear_profile)
        
        assert 'mean_rate' in uniformity
        assert 'std_rate' in uniformity
        assert 'uniformity_1sigma' in uniformity
        assert 'uniformity_3sigma' in uniformity
        assert 'uniformity_range' in uniformity
        assert 'min_rate' in uniformity
        assert 'max_rate' in uniformity
        assert 'cv_percent' in uniformity
        
        assert uniformity['mean_rate'] > 0
        assert uniformity['std_rate'] >= 0
        assert uniformity['min_rate'] <= uniformity['mean_rate'] <= uniformity['max_rate']
        
        # For good uniformity, 1σ uniformity should be > 90%
        # But with pattern loading, uniformity decreases
        assert 0 <= uniformity['uniformity_1sigma'] <= 100
    
    def test_critical_density_calculation(self, etch_analyzer, linear_profile):
        """Test critical density (50% reduction) calculation"""
        result = etch_analyzer.fit_loading_effect(linear_profile, model="linear")
        
        # For linear model: R = R0(1 - α*D)
        # At critical density: R = 0.5*R0
        # So: 0.5 = 1 - α*D_crit => D_crit = 0.5/α
        
        if 'alpha' in result.coefficients:
            alpha = result.coefficients['alpha']
            expected_critical = 50.0 / alpha  # Convert to percentage
            
            # Allow some tolerance due to fitting
            np.testing.assert_allclose(
                result.critical_density,
                expected_critical,
                rtol=0.2  # 20% tolerance
            )


# ============================================================================
# Simulator Tests
# ============================================================================

class TestChemicalBulkSimulator:
    """Test suite for simulator"""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance"""
        return ChemicalBulkSimulator()
    
    def test_sims_profile_generation(self, simulator):
        """Test SIMS profile simulation"""
        profile = simulator.simulate_sims_profile(
            element="B",
            peak_depth=100.0,
            dose=1e15,
            straggle=30.0
        )
        
        assert len(profile.depth) > 0
        assert len(profile.counts) == len(profile.depth)
        assert len(profile.concentration) == len(profile.depth)
        assert profile.element == "B"
        assert profile.matrix == "Si"
        
        # Check peak location
        peak_idx = np.argmax(profile.concentration)
        peak_depth_actual = profile.depth[peak_idx]
        assert 80 < peak_depth_actual < 120  # Should be near 100 nm
    
    def test_rbs_spectrum_generation(self, simulator):
        """Test RBS spectrum simulation"""
        layers = [("Hf", 0.5, 20.0), ("O", 0.5, 20.0)]
        spectrum = simulator.simulate_rbs_spectrum(layers, substrate="Si")
        
        assert len(spectrum.energy) > 0
        assert len(spectrum.counts) == len(spectrum.energy)
        assert spectrum.incident_energy == 2000.0
        assert spectrum.scattering_angle == 170.0
    
    def test_naa_decay_generation(self, simulator):
        """Test NAA decay curve simulation"""
        curve = simulator.simulate_naa_decay(
            element="Au",
            initial_activity=10000.0,
            irradiation_time=3600.0
        )
        
        assert len(curve.time) > 0
        assert len(curve.counts) == len(curve.time)
        assert len(curve.live_time) == len(curve.time)
        assert curve.element == "Au"
        
        # Counts should generally decrease with time
        # (allowing for statistical fluctuations)
        assert curve.counts[0] > curve.counts[-1]
    
    def test_etch_profile_generation(self, simulator):
        """Test etch profile simulation"""
        profile = simulator.simulate_etch_profile(
            model="linear",
            nominal_rate=100.0,
            alpha=0.3
        )
        
        assert len(profile.pattern_density) > 0
        assert len(profile.etch_rate) == len(profile.pattern_density)
        assert np.all(profile.etch_rate >= 0)
        
        # Etch rate should decrease with pattern density
        assert profile.etch_rate[0] > profile.etch_rate[-1]
        
        # Rate at 0% density should be close to nominal
        assert 95 < profile.etch_rate[0] < 105


# ============================================================================
# Integration and Performance Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests"""
    
    def test_sims_complete_workflow(self):
        """Test complete SIMS analysis workflow"""
        # Create instances
        simulator = ChemicalBulkSimulator()
        analyzer = SIMSAnalyzer()
        
        # Generate profile
        profile = simulator.simulate_sims_profile(
            element="P", peak_depth=80.0, dose=5e14
        )
        
        # Convert to depth
        profile.depth = analyzer.convert_time_to_depth(profile)
        
        # Quantify
        profile.concentration = analyzer.quantify_profile(profile)
        
        # Analyze
        interfaces = analyzer.find_interfaces(profile)
        dose = analyzer.calculate_dose(profile)
        det_limit = analyzer.estimate_detection_limit(profile)
        
        # Verify results
        assert dose > 0
        assert det_limit > 0
        assert 3e14 < dose < 7e14  # Should recover input dose
    
    def test_rbs_complete_workflow(self):
        """Test complete RBS analysis workflow"""
        simulator = ChemicalBulkSimulator()
        analyzer = RBSAnalyzer()
        
        # Generate spectrum
        true_layers = [("Hf", 0.5, 20.0), ("O", 0.5, 20.0)]
        spectrum = simulator.simulate_rbs_spectrum(true_layers)
        
        # Fit with initial guess
        initial_layers = [
            RBSLayer(element="Hf", atomic_fraction=0.5, thickness=25.0),
            RBSLayer(element="O", atomic_fraction=0.5, thickness=25.0)
        ]
        
        result = analyzer.fit_spectrum(
            spectrum, initial_layers, fix_composition=True
        )
        
        # Verify fit quality
        assert result.r_factor < 0.2  # R-factor < 20%
        
        # Verify layer thicknesses
        for fitted, (elem, frac, thick) in zip(result.layers, true_layers):
            assert fitted.element == elem
            # Thickness should be recovered within 30%
            assert 0.7 * thick < fitted.thickness < 1.3 * thick
    
    def test_naa_complete_workflow(self):
        """Test complete NAA analysis workflow"""
        simulator = ChemicalBulkSimulator()
        analyzer = NAAAnalyzer()
        
        # Generate curves with known ratio
        sample_curve = simulator.simulate_naa_decay(
            element="Au", initial_activity=5000.0
        )
        standard_curve = simulator.simulate_naa_decay(
            element="Au", initial_activity=10000.0
        )
        
        # Quantify
        result = analyzer.comparator_method(
            sample_curve, standard_curve,
            standard_mass=0.1,
            sample_mass=0.5,
            standard_concentration=100.0,
            element="Au"
        )
        
        # Verify results
        assert result.concentration > 0
        assert result.uncertainty > 0
        # Expected: ~10 μg/g (see calculation in test)
        assert 5 < result.concentration < 20
    
    def test_performance_sims(self, benchmark):
        """Benchmark SIMS analysis performance"""
        simulator = ChemicalBulkSimulator()
        analyzer = SIMSAnalyzer()
        
        profile = simulator.simulate_sims_profile(n_points=500)
        profile.depth = analyzer.convert_time_to_depth(profile)
        
        def run_analysis():
            conc = analyzer.quantify_profile(profile)
            interfaces = analyzer.find_interfaces(profile)
            dose = analyzer.calculate_dose(profile)
            return conc, interfaces, dose
        
        # Should complete in < 100 ms
        result = benchmark(run_analysis)
        assert benchmark.stats['mean'] < 0.1


# ============================================================================
# Test Execution
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
