"""
Examples for Advanced Physics Models

Demonstrates usage of thickness, stress, and adhesion models
for both HIL simulation and VM feature engineering.
"""

import numpy as np
import logging

from .thickness import (
    ThicknessModel,
    DepositionParameters,
    DepositionRateCalculator,
    UniformityCalculator,
    CVDMode,
    ArrheniusParameters,
)

from .stress import (
    StressModel,
    ProcessConditions,
    MaterialProperties,
    IntrinsicStressCalculator,
    ThermalStressCalculator,
    wafer_curvature_to_stress,
    xrd_to_stress,
    get_material_properties,
)

from .adhesion import (
    AdhesionModel,
    AdhesionFactors,
    simulate_tape_test,
    simulate_scratch_test,
    simulate_nanoindentation,
    simulate_stud_pull,
)

from .reactor_geometry import (
    ReactorType,
    ShowerheadReactor,
    HorizontalFlowReactor,
    BatchFurnaceReactor,
)

from .vm_features import (
    VMFeatureExtractor,
    TelemetryData,
    create_training_dataset,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Example 1: Film Thickness Modeling with Reactor Geometry
# =============================================================================

def example_thickness_modeling():
    """
    Demonstrate thickness prediction with reactor-specific uniformity
    """
    logger.info("=" * 70)
    logger.info("Example 1: Film Thickness Modeling")
    logger.info("=" * 70)

    # Create showerhead reactor
    reactor = ShowerheadReactor(
        wafer_diameter_mm=200.0,
        gap_mm=20.0,
        is_rotating=True,
        rotation_speed_rpm=20.0,
    )

    # Create thickness model with reactor
    thickness_model = ThicknessModel(
        mode=CVDMode.THERMAL,
        reactor=reactor,
    )

    # Define deposition parameters
    params = DepositionParameters(
        temperature_c=780.0,
        pressure_torr=0.3,
        precursor_flow_sccm=80.0,
        carrier_gas_flow_sccm=500.0,
        film_material="Si3N4",
        wafer_diameter_mm=200.0,
        rotation_speed_rpm=20.0,
        target_thickness_nm=100.0,
    )

    # Predict thickness after 1 hour
    result = thickness_model.predict_thickness(
        params=params,
        time_sec=3600.0,
    )

    logger.info(f"Mean thickness: {result['mean_thickness_nm']:.1f} nm")
    logger.info(f"Deposition rate: {result['deposition_rate_nm_min']:.2f} nm/min")
    logger.info(f"WIW uniformity: {result['wiw_uniformity_pct']:.2f}%")

    # Extract VM features
    vm_features = thickness_model.extract_vm_features(params, time_sec=3600.0)
    logger.info(f"VM features extracted: {len(vm_features)} features")


# =============================================================================
# Example 2: Film Stress Modeling with Multiple Methods
# =============================================================================

def example_stress_modeling():
    """
    Demonstrate stress calculation and measurement method conversions
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 2: Film Stress Modeling")
    logger.info("=" * 70)

    # Get material properties for Si₃N₄
    material = get_material_properties("Si3N4")

    # Create stress model
    stress_model = StressModel(material=material)

    # Define process conditions
    process = ProcessConditions(
        temperature_c=780.0,
        pressure_torr=0.3,
        deposition_rate_nm_min=50.0,
        rf_power_w=0.0,  # Thermal CVD
        film_thickness_nm=100.0,
    )

    # Calculate total stress
    stress_result = stress_model.calculate_total_stress(
        process=process,
        measurement_temp_c=25.0,
    )

    logger.info(f"Total stress (mean): {stress_result['stress_mean_mpa']:.1f} MPa")
    logger.info(f"Stress type: {stress_result['stress_type'].value}")
    logger.info(f"Intrinsic stress: {stress_result['intrinsic_stress_mpa']:.1f} MPa")
    logger.info(f"Thermal stress: {stress_result['thermal_stress_mpa']:.1f} MPa")
    logger.info(f"Stress gradient: {stress_result['gradient_mpa_per_nm']:.3f} MPa/nm")

    # Simulate wafer curvature measurement
    logger.info("\n--- Wafer Curvature Measurement ---")
    curvature_1_m = 2.5  # Example curvature
    stress_from_curvature = wafer_curvature_to_stress(
        curvature_1_m=curvature_1_m,
        film_thickness_nm=100.0,
    )
    logger.info(f"Stress from curvature ({curvature_1_m} m⁻¹): {stress_from_curvature:.1f} MPa")

    # Extract VM features
    vm_features = stress_model.extract_vm_features(process)
    logger.info(f"\nVM features extracted: {len(vm_features)} features")


# =============================================================================
# Example 3: Film Adhesion Modeling and Test Simulation
# =============================================================================

def example_adhesion_modeling():
    """
    Demonstrate adhesion scoring and test simulations
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 3: Film Adhesion Modeling")
    logger.info("=" * 70)

    # Create adhesion model
    adhesion_model = AdhesionModel()

    # Define adhesion factors
    factors = AdhesionFactors(
        film_stress_mpa=-250.0,  # Compressive
        stress_gradient_mpa_per_nm=0.1,
        pre_clean_quality=0.95,  # Good clean
        surface_roughness_ra_nm=0.8,
        interlayer_type="oxide_on_silicon",
        interlayer_quality=1.0,
        particle_count_per_cm2=5.0,  # Low contamination
        moisture_content_ppm=10.0,
        film_thickness_nm=100.0,
        film_density_g_cm3=3.1,  # Si₃N₄
        deposition_temp_c=780.0,
    )

    # Calculate adhesion score
    adhesion_score, adhesion_class = adhesion_model.calculate_adhesion_score(factors)

    logger.info(f"Adhesion score: {adhesion_score:.1f}/100")
    logger.info(f"Adhesion class: {adhesion_class.value}")

    # Simulate adhesion tests
    logger.info("\n--- Adhesion Test Simulations ---")

    # Tape test
    tape_result = simulate_tape_test(adhesion_score)
    logger.info(f"Tape test: {tape_result.adhesion_class.value}")
    logger.info(f"  {tape_result.notes}")

    # Scratch test
    scratch_result = simulate_scratch_test(
        adhesion_score=adhesion_score,
        film_hardness_gpa=20.0,
        film_thickness_nm=100.0,
    )
    logger.info(f"Scratch test: Critical load = {scratch_result.critical_load_n:.2f} N")
    logger.info(f"  Failure mode: {scratch_result.failure_mode.value}")

    # Nanoindentation
    nano_result = simulate_nanoindentation(
        adhesion_score=adhesion_score,
        film_youngs_modulus_gpa=250.0,
        film_thickness_nm=100.0,
    )
    logger.info(f"Nanoindentation: Gc = {nano_result.interfacial_energy_j_m2:.2f} J/m²")

    # Stud pull
    stud_result = simulate_stud_pull(adhesion_score=adhesion_score)
    logger.info(f"Stud pull: {stud_result.critical_load_n:.1f} N")

    # Extract VM features
    vm_features = adhesion_model.extract_vm_features(factors)
    logger.info(f"\nVM features extracted: {len(vm_features)} features")


# =============================================================================
# Example 4: Reactor Geometry Effects on Uniformity
# =============================================================================

def example_reactor_geometries():
    """
    Compare thickness uniformity across different reactor types
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 4: Reactor Geometry Comparison")
    logger.info("=" * 70)

    # Create different reactors
    showerhead = ShowerheadReactor(wafer_diameter_mm=200.0, is_rotating=True)
    horizontal = HorizontalFlowReactor(wafer_diameter_mm=200.0, is_rotating=True)
    batch = BatchFurnaceReactor(wafer_diameter_mm=150.0, num_wafers=100)

    # Calculate uniformity at edge (95mm from center)
    radial_pos_mm = 95.0

    # Showerhead
    sh_uniformity = showerhead.calculate_uniformity_factor(
        radial_position_mm=radial_pos_mm,
        pressure_torr=1.0,
        temperature_c=300.0,
    )
    logger.info(f"Showerhead uniformity at r={radial_pos_mm}mm: {sh_uniformity:.3f}")

    # Horizontal flow
    hf_uniformity = horizontal.calculate_uniformity_factor(
        x_mm=radial_pos_mm,
        y_mm=0.0,
        pressure_torr=0.3,
        flow_velocity_cm_s=10.0,
    )
    logger.info(f"Horizontal flow uniformity at x={radial_pos_mm}mm: {hf_uniformity:.3f}")

    # Batch furnace WIW
    bf_wiw = batch.calculate_wiw_uniformity(
        wafer_index=50,  # Middle of boat
        radial_position_mm=70.0,
    )
    logger.info(f"Batch furnace WIW uniformity: {bf_wiw:.3f}")

    # Batch furnace WTW
    bf_wtw = batch.calculate_wtw_uniformity()
    logger.info(f"Batch furnace WTW uniformity: {bf_wtw:.2f}%")


# =============================================================================
# Example 5: VM Feature Engineering for ML
# =============================================================================

def example_vm_feature_engineering():
    """
    Demonstrate VM feature extraction for ML model training
    """
    logger.info("\n" + "=" * 70)
    logger.info("Example 5: VM Feature Engineering")
    logger.info("=" * 70)

    # Create feature extractor
    extractor = VMFeatureExtractor()

    # Define process parameters
    deposition_params = DepositionParameters(
        temperature_c=780.0,
        pressure_torr=0.3,
        precursor_flow_sccm=80.0,
        carrier_gas_flow_sccm=500.0,
        film_material="Si3N4",
    )

    process_conditions = ProcessConditions(
        temperature_c=780.0,
        pressure_torr=0.3,
        deposition_rate_nm_min=50.0,
        film_thickness_nm=100.0,
    )

    adhesion_factors = AdhesionFactors(
        film_stress_mpa=-250.0,
        pre_clean_quality=0.95,
        deposition_temp_c=780.0,
    )

    # Create synthetic telemetry
    time_sec = np.linspace(0, 3600, 100)
    telemetry = TelemetryData(
        time_sec=time_sec,
        temperature_c=780.0 + np.random.randn(100) * 0.5,
        pressure_torr=0.3 + np.random.randn(100) * 0.001,
        precursor_flow_sccm=80.0 + np.random.randn(100) * 0.5,
    )

    # Extract all features
    features = extractor.extract_all_features(
        deposition_params=deposition_params,
        process_conditions=process_conditions,
        adhesion_factors=adhesion_factors,
        telemetry=telemetry,
    )

    logger.info(f"Total features extracted: {len(features)}")
    logger.info("\nKey features:")
    for key in list(features.keys())[:20]:  # Show first 20
        logger.info(f"  {key}: {features[key]:.3f}")

    # Create training dataset example
    logger.info("\n--- Training Dataset Creation ---")

    # Simulate multiple runs
    params_list = [
        DepositionParameters(temperature_c=temp, pressure_torr=0.3, precursor_flow_sccm=80.0)
        for temp in [750, 770, 790, 810, 830]
    ]

    # Simulated measurements
    measured_thickness = [95.0, 98.5, 102.0, 105.5, 109.0]

    dataset = create_training_dataset(
        deposition_params_list=params_list,
        measured_thickness_nm_list=measured_thickness,
    )

    logger.info(f"Dataset shape: X={dataset['X'].shape}, y={len(dataset['y_thickness_nm'])}")
    logger.info(f"Feature names: {dataset['feature_names'][:10]}...")


# =============================================================================
# Main: Run All Examples
# =============================================================================

def main():
    """Run all physics model examples"""
    logger.info("\n" + "=" * 70)
    logger.info("CVD Physics Models - Examples")
    logger.info("=" * 70)

    example_thickness_modeling()
    example_stress_modeling()
    example_adhesion_modeling()
    example_reactor_geometries()
    example_vm_feature_engineering()

    logger.info("\n" + "=" * 70)
    logger.info("All examples completed!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
