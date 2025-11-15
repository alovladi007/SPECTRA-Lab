"""
Advanced Film Adhesion Modeling

Comprehensive adhesion prediction and test simulation including:
- Adhesion score calculation (0-100)
- Multiple test methods (tape, scratch, nanoindentation, stud pull)
- Factor analysis (stress, contamination, roughness, interlayer compatibility)
- VM feature engineering
"""

import math
import random
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum


class AdhesionClass(str, Enum):
    """Adhesion quality classification"""
    POOR = "POOR"  # Score 0-40: Delamination likely
    MARGINAL = "MARGINAL"  # Score 40-70: May fail under stress
    GOOD = "GOOD"  # Score 70-85: Acceptable for most applications
    EXCELLENT = "EXCELLENT"  # Score 85-100: Outstanding adhesion


class AdhesionTest(str, Enum):
    """Adhesion test methods"""
    TAPE_TEST = "TAPE_TEST"  # ASTM D3359 (qualitative/semi-quantitative)
    SCRATCH_TEST = "SCRATCH_TEST"  # Progressive load scratch
    NANOINDENTATION = "NANOINDENTATION"  # Interfacial delamination energy
    STUD_PULL = "STUD_PULL"  # Pull-off strength test
    FOUR_POINT_BEND = "FOUR_POINT_BEND"  # Interface toughness
    BLISTER_TEST = "BLISTER_TEST"  # Pressure-induced delamination


class FailureMode(str, Enum):
    """Adhesion failure modes"""
    COHESIVE = "COHESIVE"  # Failure within film
    ADHESIVE = "ADHESIVE"  # Failure at film-substrate interface
    INTERFACIAL = "INTERFACIAL"  # Failure at interlayer
    MIXED = "MIXED"  # Combination of modes


@dataclass
class AdhesionFactors:
    """
    Factors affecting film adhesion

    Adhesion is influenced by:
    - Mechanical: stress, roughness
    - Chemical: surface preparation, interlayer compatibility
    - Structural: grain boundaries, defects
    - Contamination: particles, moisture, organics
    """

    # Stress-related factors
    film_stress_mpa: float = 0.0  # Film stress magnitude
    stress_gradient_mpa_per_nm: float = 0.0  # Through-thickness gradient

    # Surface preparation
    pre_clean_quality: float = 1.0  # 0-1 scale (1 = perfect clean)
    surface_roughness_ra_nm: float = 0.5  # Average roughness
    surface_roughness_rq_nm: float = 0.7  # RMS roughness

    # Interlayer compatibility
    interlayer_type: str = "oxide_on_silicon"  # e.g., "metal_on_oxide", "oxide_on_metal"
    interlayer_quality: float = 1.0  # 0-1 scale

    # Contamination levels
    particle_count_per_cm2: float = 0.0  # Particles >0.1 μm
    moisture_content_ppm: float = 10.0  # Moisture in ppm
    organic_contamination_level: float = 0.0  # 0-1 scale

    # Film microstructure
    film_thickness_nm: float = 100.0
    film_density_g_cm3: float = 2.2
    grain_size_nm: float = 50.0  # Average grain size

    # Deposition conditions
    deposition_temp_c: float = 400.0
    ion_bombardment_energy_ev: float = 0.0  # For PECVD


@dataclass
class AdhesionTestResult:
    """Results from adhesion test"""
    test_method: AdhesionTest
    adhesion_score: float  # 0-100 scale
    adhesion_class: AdhesionClass
    critical_load_n: Optional[float] = None  # For scratch/stud pull
    failure_mode: Optional[FailureMode] = None
    interfacial_energy_j_m2: Optional[float] = None  # For nanoindentation
    notes: str = ""


class AdhesionModel:
    """
    Comprehensive adhesion prediction model

    Combines multiple factors to predict adhesion score and classify quality
    """

    def __init__(self):
        pass

    def calculate_adhesion_score(
        self,
        factors: AdhesionFactors,
    ) -> Tuple[float, AdhesionClass]:
        """
        Calculate overall adhesion score from multiple factors

        Score = base * stress_penalty * surface_factor * interlayer_factor * contamination_factor

        Args:
            factors: Adhesion influencing factors

        Returns:
            Tuple of (adhesion_score, adhesion_class)
        """
        # ====================================================================
        # Base adhesion (material-dependent)
        # ====================================================================
        base_adhesion = 85.0  # Baseline for clean, low-stress film

        # ====================================================================
        # Factor 1: Stress penalty
        # ====================================================================
        # High stress reduces adhesion (promotes delamination)
        stress_magnitude = abs(factors.film_stress_mpa)

        # Critical stress for delamination ~ 200-500 MPa
        stress_penalty = 1.0 / (1.0 + 0.002 * stress_magnitude)

        # Gradient stress effect (high gradients promote edge delamination)
        gradient_penalty = 1.0 / (1.0 + 5.0 * abs(factors.stress_gradient_mpa_per_nm))

        stress_factor = stress_penalty * gradient_penalty

        # ====================================================================
        # Factor 2: Surface preparation
        # ====================================================================
        # Clean surface → better adhesion
        clean_factor = factors.pre_clean_quality

        # Roughness effect (moderate roughness helps mechanical interlocking)
        # Optimal roughness: 0.5-2 nm Ra
        if 0.5 <= factors.surface_roughness_ra_nm <= 2.0:
            roughness_factor = 1.05  # Slight improvement
        elif factors.surface_roughness_ra_nm > 5.0:
            roughness_factor = 0.9  # Too rough (poor conformal coverage)
        else:
            roughness_factor = 1.0

        surface_factor = clean_factor * roughness_factor

        # ====================================================================
        # Factor 3: Interlayer compatibility
        # ====================================================================
        # Chemical compatibility between layers
        interlayer_factor = factors.interlayer_quality

        # Specific interlayer combinations
        if "oxide_on_metal" in factors.interlayer_type:
            interlayer_bonus = 1.0  # Generally good
        elif "metal_on_oxide" in factors.interlayer_type:
            interlayer_bonus = 0.95  # Slightly worse
        elif "metal_on_metal" in factors.interlayer_type:
            interlayer_bonus = 0.9  # Diffusion concerns
        else:
            interlayer_bonus = 1.0

        interlayer_factor *= interlayer_bonus

        # ====================================================================
        # Factor 4: Contamination penalty
        # ====================================================================
        # Particles reduce adhesion
        particle_penalty = 1.0 / (1.0 + 0.01 * factors.particle_count_per_cm2)

        # Moisture penalty
        moisture_penalty = 1.0 / (1.0 + 0.001 * factors.moisture_content_ppm)

        # Organic contamination
        organic_penalty = 1.0 - 0.5 * factors.organic_contamination_level

        contamination_factor = particle_penalty * moisture_penalty * organic_penalty

        # ====================================================================
        # Factor 5: Film microstructure
        # ====================================================================
        # Dense films adhere better
        if factors.film_density_g_cm3 > 2.0:
            density_factor = 1.05
        else:
            density_factor = 0.95

        # Thinner films generally adhere better (less stored energy)
        if factors.film_thickness_nm < 100.0:
            thickness_factor = 1.0
        elif factors.film_thickness_nm > 1000.0:
            thickness_factor = 0.9  # Thick films more prone to delamination
        else:
            thickness_factor = 0.95

        microstructure_factor = density_factor * thickness_factor

        # ====================================================================
        # Factor 6: Deposition process
        # ====================================================================
        # Higher deposition temperature → better interfacial bonding
        if factors.deposition_temp_c > 300.0:
            temp_factor = 1.05
        elif factors.deposition_temp_c < 100.0:
            temp_factor = 0.9
        else:
            temp_factor = 1.0

        # Ion bombardment can improve adhesion (interface mixing)
        # But too much damages substrate
        if 0 < factors.ion_bombardment_energy_ev < 100.0:
            ion_factor = 1.1  # Beneficial
        elif factors.ion_bombardment_energy_ev > 300.0:
            ion_factor = 0.95  # Excessive (substrate damage)
        else:
            ion_factor = 1.0

        process_factor = temp_factor * ion_factor

        # ====================================================================
        # Combined adhesion score
        # ====================================================================
        adhesion_score = (
            base_adhesion *
            stress_factor *
            surface_factor *
            interlayer_factor *
            contamination_factor *
            microstructure_factor *
            process_factor
        )

        # Clamp to [0, 100]
        adhesion_score = max(0.0, min(100.0, adhesion_score))

        # Classify adhesion
        if adhesion_score >= 85.0:
            adhesion_class = AdhesionClass.EXCELLENT
        elif adhesion_score >= 70.0:
            adhesion_class = AdhesionClass.GOOD
        elif adhesion_score >= 40.0:
            adhesion_class = AdhesionClass.MARGINAL
        else:
            adhesion_class = AdhesionClass.POOR

        return adhesion_score, adhesion_class

    def predict_failure_mode(
        self,
        factors: AdhesionFactors,
        adhesion_score: float,
    ) -> FailureMode:
        """
        Predict adhesion failure mode

        Args:
            factors: Adhesion factors
            adhesion_score: Calculated adhesion score

        Returns:
            Predicted failure mode
        """
        # High adhesion → cohesive failure (film fails before interface)
        if adhesion_score > 85.0:
            return FailureMode.COHESIVE

        # Poor adhesion → adhesive failure (interface fails)
        if adhesion_score < 40.0:
            return FailureMode.ADHESIVE

        # High stress → interfacial failure
        if abs(factors.film_stress_mpa) > 300.0:
            return FailureMode.INTERFACIAL

        # Otherwise, mixed mode
        return FailureMode.MIXED

    def extract_vm_features(
        self,
        factors: AdhesionFactors,
    ) -> Dict[str, float]:
        """
        Extract features for Virtual Metrology adhesion prediction

        Args:
            factors: Adhesion factors

        Returns:
            Feature dictionary for ML models
        """
        adhesion_score, adhesion_class = self.calculate_adhesion_score(factors)

        features = {
            # Predicted adhesion
            "predicted_adhesion_score": adhesion_score,

            # Stress factors
            "film_stress_mpa": factors.film_stress_mpa,
            "stress_magnitude_mpa": abs(factors.film_stress_mpa),
            "stress_gradient_mpa_per_nm": factors.stress_gradient_mpa_per_nm,

            # Surface factors
            "pre_clean_quality": factors.pre_clean_quality,
            "surface_roughness_ra_nm": factors.surface_roughness_ra_nm,
            "surface_roughness_rq_nm": factors.surface_roughness_rq_nm,

            # Interlayer
            "interlayer_quality": factors.interlayer_quality,

            # Contamination
            "particle_count_per_cm2": factors.particle_count_per_cm2,
            "moisture_content_ppm": factors.moisture_content_ppm,
            "organic_contamination_level": factors.organic_contamination_level,

            # Microstructure
            "film_thickness_nm": factors.film_thickness_nm,
            "film_density_g_cm3": factors.film_density_g_cm3,
            "grain_size_nm": factors.grain_size_nm,

            # Process
            "deposition_temp_c": factors.deposition_temp_c,
            "ion_bombardment_energy_ev": factors.ion_bombardment_energy_ev,
        }

        return features


# =============================================================================
# Adhesion Test Simulations
# =============================================================================

def simulate_tape_test(
    adhesion_score: float,
    test_type: str = "cross_cut",  # "cross_cut" or "straight_cut"
) -> AdhesionTestResult:
    """
    Simulate tape test (ASTM D3359)

    Cross-cut method:
    - 5B: No peeling or removal
    - 4B: < 5% area removed
    - 3B: 5-15% area removed
    - 2B: 15-35% area removed
    - 1B: 35-65% area removed
    - 0B: > 65% area removed

    Args:
        adhesion_score: Adhesion score (0-100)
        test_type: "cross_cut" or "straight_cut"

    Returns:
        AdhesionTestResult
    """
    # Convert score to classification
    if adhesion_score >= 90:
        classification = "5B"
        notes = "No peeling or removal. Excellent adhesion."
    elif adhesion_score >= 75:
        classification = "4B"
        notes = "< 5% area removed. Good adhesion."
    elif adhesion_score >= 60:
        classification = "3B"
        notes = "5-15% area removed. Fair adhesion."
    elif adhesion_score >= 45:
        classification = "2B"
        notes = "15-35% area removed. Poor adhesion."
    elif adhesion_score >= 30:
        classification = "1B"
        notes = "35-65% area removed. Very poor adhesion."
    else:
        classification = "0B"
        notes = "> 65% area removed. Film delaminated."

    # Determine adhesion class
    if adhesion_score >= 85:
        adhesion_class = AdhesionClass.EXCELLENT
    elif adhesion_score >= 70:
        adhesion_class = AdhesionClass.GOOD
    elif adhesion_score >= 40:
        adhesion_class = AdhesionClass.MARGINAL
    else:
        adhesion_class = AdhesionClass.POOR

    result = AdhesionTestResult(
        test_method=AdhesionTest.TAPE_TEST,
        adhesion_score=adhesion_score,
        adhesion_class=adhesion_class,
        notes=f"Tape test classification: {classification}. {notes}",
    )

    return result


def simulate_scratch_test(
    adhesion_score: float,
    film_hardness_gpa: float = 5.0,
    film_thickness_nm: float = 100.0,
) -> AdhesionTestResult:
    """
    Simulate progressive load scratch test

    Measures critical load at which film delaminates

    Args:
        adhesion_score: Adhesion score (0-100)
        film_hardness_gpa: Film hardness (GPa)
        film_thickness_nm: Film thickness (nm)

    Returns:
        AdhesionTestResult with critical load
    """
    # Critical load correlates with adhesion strength
    # Lc = k * adhesion_score * sqrt(hardness) * thickness
    k = 0.01  # Empirical constant

    base_critical_load = k * adhesion_score * math.sqrt(film_hardness_gpa) * (film_thickness_nm / 100.0)

    # Add realistic noise (±10%)
    noise_factor = random.uniform(0.9, 1.1)
    critical_load_n = base_critical_load * noise_factor

    # Determine failure mode
    if adhesion_score > 85:
        failure_mode = FailureMode.COHESIVE
    elif adhesion_score < 40:
        failure_mode = FailureMode.ADHESIVE
    else:
        failure_mode = FailureMode.MIXED

    # Classify
    if critical_load_n > 5.0:
        adhesion_class = AdhesionClass.EXCELLENT
    elif critical_load_n > 3.0:
        adhesion_class = AdhesionClass.GOOD
    elif critical_load_n > 1.5:
        adhesion_class = AdhesionClass.MARGINAL
    else:
        adhesion_class = AdhesionClass.POOR

    result = AdhesionTestResult(
        test_method=AdhesionTest.SCRATCH_TEST,
        adhesion_score=adhesion_score,
        adhesion_class=adhesion_class,
        critical_load_n=critical_load_n,
        failure_mode=failure_mode,
        notes=f"Critical load: {critical_load_n:.2f} N. Failure mode: {failure_mode.value}.",
    )

    return result


def simulate_nanoindentation(
    adhesion_score: float,
    film_youngs_modulus_gpa: float = 70.0,
    film_thickness_nm: float = 100.0,
) -> AdhesionTestResult:
    """
    Simulate nanoindentation-based adhesion measurement

    Estimates interfacial fracture energy from pop-in events or delamination

    Args:
        adhesion_score: Adhesion score (0-100)
        film_youngs_modulus_gpa: Film modulus (GPa)
        film_thickness_nm: Film thickness (nm)

    Returns:
        AdhesionTestResult with interfacial energy
    """
    # Interfacial fracture energy (J/m²)
    # Typical range: 0.1-10 J/m² for thin films
    # Gc = k * adhesion_score * E * h
    k = 0.0001

    interfacial_energy = k * adhesion_score * film_youngs_modulus_gpa * (film_thickness_nm / 100.0)

    # Add noise
    noise_factor = random.uniform(0.9, 1.1)
    interfacial_energy_j_m2 = interfacial_energy * noise_factor

    # Classify
    if interfacial_energy_j_m2 > 5.0:
        adhesion_class = AdhesionClass.EXCELLENT
    elif interfacial_energy_j_m2 > 2.0:
        adhesion_class = AdhesionClass.GOOD
    elif interfacial_energy_j_m2 > 0.5:
        adhesion_class = AdhesionClass.MARGINAL
    else:
        adhesion_class = AdhesionClass.POOR

    result = AdhesionTestResult(
        test_method=AdhesionTest.NANOINDENTATION,
        adhesion_score=adhesion_score,
        adhesion_class=adhesion_class,
        interfacial_energy_j_m2=interfacial_energy_j_m2,
        notes=f"Interfacial fracture energy: {interfacial_energy_j_m2:.2f} J/m².",
    )

    return result


def simulate_stud_pull(
    adhesion_score: float,
    stud_diameter_mm: float = 2.0,
) -> AdhesionTestResult:
    """
    Simulate stud pull-off strength test

    Measures tensile adhesion strength by pulling bonded stud

    Args:
        adhesion_score: Adhesion score (0-100)
        stud_diameter_mm: Stud diameter (mm)

    Returns:
        AdhesionTestResult with pull-off strength
    """
    # Pull-off strength (MPa) correlates with adhesion
    # Typical range: 1-50 MPa for thin films
    base_strength_mpa = adhesion_score * 0.5

    # Add noise
    noise_factor = random.uniform(0.9, 1.1)
    pull_off_strength_mpa = base_strength_mpa * noise_factor

    # Convert to force (N)
    stud_area_mm2 = math.pi * (stud_diameter_mm / 2.0)**2
    pull_off_force_n = pull_off_strength_mpa * stud_area_mm2

    # Classify
    if pull_off_strength_mpa > 30.0:
        adhesion_class = AdhesionClass.EXCELLENT
    elif pull_off_strength_mpa > 15.0:
        adhesion_class = AdhesionClass.GOOD
    elif pull_off_strength_mpa > 5.0:
        adhesion_class = AdhesionClass.MARGINAL
    else:
        adhesion_class = AdhesionClass.POOR

    result = AdhesionTestResult(
        test_method=AdhesionTest.STUD_PULL,
        adhesion_score=adhesion_score,
        adhesion_class=adhesion_class,
        critical_load_n=pull_off_force_n,
        notes=f"Pull-off strength: {pull_off_strength_mpa:.1f} MPa ({pull_off_force_n:.1f} N).",
    )

    return result
