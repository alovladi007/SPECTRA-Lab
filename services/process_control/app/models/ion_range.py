"""Ion implantation range and profile models.

Provides SRIM-like estimation, channeling risk prediction, and dose-to-sheet-resistance
conversion for ion implantation processes.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

from app.drivers.ion_implant_driver import IonSpecies


# ============================================================================
# Constants and Material Properties
# ============================================================================

# Ion masses (amu)
ION_MASSES = {
    IonSpecies.BORON: 11.0,
    IonSpecies.PHOSPHORUS: 31.0,
    IonSpecies.ARSENIC: 75.0,
    IonSpecies.ANTIMONY: 122.0,
    IonSpecies.NITROGEN: 14.0,
    IonSpecies.OXYGEN: 16.0,
    IonSpecies.ARGON: 40.0,
    IonSpecies.SILICON: 28.0,
}

# Dopant types
class DopantType(str, Enum):
    """Dopant type classification."""
    N_TYPE = "n_type"  # Donors (P, As, Sb)
    P_TYPE = "p_type"  # Acceptors (B)
    NEUTRAL = "neutral"  # Non-dopants (N, O, Ar, Si)


DOPANT_TYPES = {
    IonSpecies.BORON: DopantType.P_TYPE,
    IonSpecies.PHOSPHORUS: DopantType.N_TYPE,
    IonSpecies.ARSENIC: DopantType.N_TYPE,
    IonSpecies.ANTIMONY: DopantType.N_TYPE,
    IonSpecies.NITROGEN: DopantType.NEUTRAL,
    IonSpecies.OXYGEN: DopantType.NEUTRAL,
    IonSpecies.ARGON: DopantType.NEUTRAL,
    IonSpecies.SILICON: DopantType.NEUTRAL,
}

# Silicon properties
SI_DENSITY = 2.33  # g/cm³
SI_ATOMIC_MASS = 28.0855  # amu
SI_ATOMIC_NUMBER = 14
SI_ATOMS_PER_CM3 = 5.0e22  # Silicon atom density

# Activation energies (eV) - dopant diffusion after RTP
ACTIVATION_ENERGIES = {
    IonSpecies.BORON: 3.46,
    IonSpecies.PHOSPHORUS: 3.66,
    IonSpecies.ARSENIC: 4.05,
    IonSpecies.ANTIMONY: 3.98,
}

# Solid solubility limits (cm⁻³) at 1000°C
SOLUBILITY_LIMITS = {
    IonSpecies.BORON: 2.0e20,
    IonSpecies.PHOSPHORUS: 1.3e21,
    IonSpecies.ARSENIC: 2.0e21,
    IonSpecies.ANTIMONY: 7.0e19,
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RangeParameters:
    """Ion range parameters from SRIM calculation."""
    projected_range_nm: float  # Mean projected range (Rp)
    range_straggle_nm: float  # Standard deviation (ΔRp)
    lateral_straggle_nm: float  # Lateral spread
    skewness: float  # Profile skewness (channeling tail)
    kurtosis: float  # Profile kurtosis (excess)


@dataclass
class ChannelingRisk:
    """Channeling risk assessment."""
    risk_level: str  # "low", "medium", "high"
    probability: float  # 0.0 to 1.0
    critical_angle_deg: float  # Critical channeling angle
    recommended_tilt_deg: float  # Recommended tilt to avoid channeling
    warnings: List[str]


@dataclass
class SheetResistanceEstimate:
    """Post-anneal sheet resistance estimate."""
    sheet_resistance_ohm_per_sq: float
    junction_depth_nm: float
    peak_concentration_cm3: float
    activation_fraction: float  # Fraction of dopants electrically active
    carrier_mobility_cm2_per_Vs: float
    confidence_level: str  # "high", "medium", "low"
    assumptions: List[str]


@dataclass
class DepthProfile:
    """Concentration vs depth profile."""
    depth_nm: np.ndarray
    concentration_cm3: np.ndarray
    dose_cm2: float
    ion_species: IonSpecies
    energy_keV: float


# ============================================================================
# SRIM-like Range Estimator
# ============================================================================

class SRIMEstimator:
    """SRIM-like ion range estimator using LSS theory."""

    def __init__(self):
        """Initialize SRIM estimator."""
        pass

    def estimate_range(
        self,
        ion_species: IonSpecies,
        energy_keV: float,
        tilt_angle_deg: float = 0.0,
        twist_angle_deg: float = 0.0,
        target_material: str = "Si"
    ) -> RangeParameters:
        """
        Estimate ion range parameters using LSS theory.

        Args:
            ion_species: Ion species
            energy_keV: Ion energy in keV
            tilt_angle_deg: Wafer tilt angle
            twist_angle_deg: Wafer twist angle
            target_material: Target material (currently only Si supported)

        Returns:
            RangeParameters with Rp, ΔRp, lateral straggle, etc.
        """
        if target_material != "Si":
            raise ValueError("Only Silicon target currently supported")

        ion_mass = ION_MASSES[ion_species]

        # Reduced mass
        M1 = ion_mass
        M2 = SI_ATOMIC_MASS
        M_reduced = (M1 * M2) / (M1 + M2)

        # Thomas-Fermi screening length (Angstroms)
        a_TF = 0.8853 * 0.529 / (SI_ATOMIC_NUMBER**0.23 + SI_ATOMIC_NUMBER**0.23)

        # Reduced energy (Lindhard units)
        epsilon = 32.53 * M2 * energy_keV / (
            SI_ATOMIC_NUMBER * SI_ATOMIC_NUMBER * (M1 + M2)
        )

        # Electronic and nuclear stopping (simplified)
        k_e = 0.15  # Electronic stopping coefficient
        k_n = 0.5   # Nuclear stopping coefficient

        if epsilon < 0.1:
            s_n = k_n * 3.441 * np.sqrt(epsilon) * np.log(epsilon + 2.718)
        else:
            s_n = k_n * np.log(1 + 1.1383 * epsilon) / (
                2 * (epsilon + 0.01321 * epsilon**0.21226 + 0.19593 * epsilon**0.5)
            )

        s_e = k_e * epsilon**0.45
        s_total = s_n + s_e

        # Projected range using empirical formula
        # Rp (nm) ≈ A × E^n
        if ion_mass < SI_ATOMIC_MASS:  # Light ions
            A = 1.2 * ion_mass / SI_ATOMIC_MASS
            n = 1.7
        elif ion_mass > SI_ATOMIC_MASS:  # Heavy ions
            A = 0.8 * SI_ATOMIC_MASS / ion_mass
            n = 1.3
        else:  # Similar mass
            A = 1.0
            n = 1.5

        Rp_nm = A * (energy_keV ** n)

        # Range straggle (typically 20-40% of Rp)
        straggle_ratio = 0.3 if ion_mass < SI_ATOMIC_MASS else 0.25
        delta_Rp_nm = straggle_ratio * Rp_nm

        # Lateral straggle (typically 60% of longitudinal)
        lateral_straggle_nm = 0.6 * delta_Rp_nm

        # Account for tilt angle
        tilt_rad = np.deg2rad(tilt_angle_deg)
        twist_rad = np.deg2rad(twist_angle_deg)

        # Effective tilt (combined tilt and twist)
        eff_tilt = np.sqrt(tilt_angle_deg**2 + twist_angle_deg**2)
        eff_tilt_rad = np.deg2rad(eff_tilt)

        if eff_tilt > 0:
            # Tilting reduces channeling, increases range slightly
            Rp_nm *= (1.0 + 0.1 * np.sin(eff_tilt_rad))
            delta_Rp_nm *= (1.0 + 0.05 * np.sin(eff_tilt_rad))
            lateral_straggle_nm *= (1.0 + 0.03 * np.sin(eff_tilt_rad))

        # Profile shape parameters
        # Low tilt = more channeling = negative skewness (tail to deeper depths)
        skewness = -0.5 if eff_tilt < 2.0 else -0.1

        # Kurtosis (excess) - higher for channeling
        kurtosis = 1.0 if eff_tilt < 2.0 else 0.2

        return RangeParameters(
            projected_range_nm=Rp_nm,
            range_straggle_nm=delta_Rp_nm,
            lateral_straggle_nm=lateral_straggle_nm,
            skewness=skewness,
            kurtosis=kurtosis
        )

    def predict_depth_profile(
        self,
        ion_species: IonSpecies,
        energy_keV: float,
        dose_cm2: float,
        tilt_angle_deg: float = 0.0,
        resolution_nm: float = 1.0
    ) -> DepthProfile:
        """
        Predict full concentration vs depth profile.

        Args:
            ion_species: Ion species
            energy_keV: Ion energy
            dose_cm2: Implanted dose
            tilt_angle_deg: Tilt angle
            resolution_nm: Depth resolution

        Returns:
            DepthProfile with depth and concentration arrays
        """
        # Get range parameters
        range_params = self.estimate_range(ion_species, energy_keV, tilt_angle_deg)

        Rp = range_params.projected_range_nm
        delta_Rp = range_params.range_straggle_nm
        skew = range_params.skewness

        # Create depth array
        max_depth = Rp + 5 * delta_Rp
        depth_nm = np.arange(0, max_depth, resolution_nm)

        # Gaussian profile with channeling tail
        gaussian = np.exp(-0.5 * ((depth_nm - Rp) / delta_Rp) ** 2)

        # Add channeling tail for low tilt
        if tilt_angle_deg < 2.0:
            tail_mask = depth_nm > Rp
            tail_depth = depth_nm[tail_mask]
            channeling_fraction = 0.1  # 10% channeling
            tail_decay = np.exp(-(tail_depth - Rp) / (3 * delta_Rp))
            channeling_tail = np.zeros_like(depth_nm)
            channeling_tail[tail_mask] = channeling_fraction * tail_decay
            profile = gaussian + channeling_tail
        else:
            profile = gaussian

        # Normalize to dose
        peak_concentration = dose_cm2 / (delta_Rp * 1e-7 * np.sqrt(2 * np.pi))
        concentration_cm3 = peak_concentration * profile

        return DepthProfile(
            depth_nm=depth_nm,
            concentration_cm3=concentration_cm3,
            dose_cm2=dose_cm2,
            ion_species=ion_species,
            energy_keV=energy_keV
        )


# ============================================================================
# Channeling Risk Predictor
# ============================================================================

class ChannelingRiskPredictor:
    """Predict channeling risk based on crystal orientation and beam alignment."""

    # Critical angles for major Si crystal planes (degrees)
    CRITICAL_ANGLES = {
        "<100>": 2.0,  # [100] channeling direction
        "<110>": 1.5,  # [110] channeling direction
        "<111>": 1.2,  # [111] channeling direction (strongest)
    }

    def __init__(self):
        """Initialize channeling risk predictor."""
        pass

    def assess_channeling_risk(
        self,
        ion_species: IonSpecies,
        energy_keV: float,
        tilt_angle_deg: float,
        twist_angle_deg: float,
        crystal_orientation: str = "<100>"
    ) -> ChannelingRisk:
        """
        Assess channeling risk for given implant conditions.

        Args:
            ion_species: Ion species
            energy_keV: Ion energy
            tilt_angle_deg: Wafer tilt from normal
            twist_angle_deg: Wafer twist/rotation
            crystal_orientation: Crystal orientation

        Returns:
            ChannelingRisk assessment
        """
        # Get critical angle for this orientation
        if crystal_orientation not in self.CRITICAL_ANGLES:
            crystal_orientation = "<100>"  # Default

        psi_c = self.CRITICAL_ANGLES[crystal_orientation]

        # Effective tilt (combination of tilt and twist)
        eff_tilt = np.sqrt(tilt_angle_deg**2 + twist_angle_deg**2)

        # Calculate channeling probability
        # Lindhard model: P_channeling ≈ (ψ_c / ψ)^2 for ψ < ψ_c
        if eff_tilt < psi_c:
            channeling_prob = (psi_c / max(eff_tilt, 0.1))**2
            channeling_prob = min(channeling_prob, 1.0)
            risk_level = "high" if channeling_prob > 0.5 else "medium"
        else:
            channeling_prob = (psi_c / eff_tilt)**2
            risk_level = "low"

        # Recommended tilt to avoid channeling (typically 7° is standard)
        recommended_tilt = max(7.0, 2 * psi_c)

        # Generate warnings
        warnings = []
        if eff_tilt < 1.0:
            warnings.append("Very low tilt angle - high channeling risk")
        if eff_tilt < psi_c:
            warnings.append(f"Tilt below critical angle ({psi_c:.1f}°) for {crystal_orientation}")
        if tilt_angle_deg < 5.0:
            warnings.append("Tilt < 5° not recommended for production implants")

        # Energy dependence - higher energy = deeper channeling
        if energy_keV > 100 and eff_tilt < 5.0:
            warnings.append("High energy with low tilt - channeling tails may be significant")

        return ChannelingRisk(
            risk_level=risk_level,
            probability=channeling_prob,
            critical_angle_deg=psi_c,
            recommended_tilt_deg=recommended_tilt,
            warnings=warnings
        )


# ============================================================================
# Sheet Resistance Estimator
# ============================================================================

class SheetResistanceEstimator:
    """Estimate post-anneal sheet resistance from implant conditions."""

    def __init__(self):
        """Initialize sheet resistance estimator."""
        self.srim_estimator = SRIMEstimator()

    def estimate_sheet_resistance(
        self,
        ion_species: IonSpecies,
        energy_keV: float,
        dose_cm2: float,
        anneal_temp_C: float,
        anneal_time_s: float,
        tilt_angle_deg: float = 7.0
    ) -> SheetResistanceEstimate:
        """
        Estimate post-anneal sheet resistance.

        Args:
            ion_species: Dopant species
            energy_keV: Implant energy
            dose_cm2: Implanted dose
            anneal_temp_C: Annealing temperature
            anneal_time_s: Annealing time
            tilt_angle_deg: Tilt angle

        Returns:
            SheetResistanceEstimate with Rs, junction depth, etc.
        """
        # Check if this is a dopant
        dopant_type = DOPANT_TYPES[ion_species]
        if dopant_type == DopantType.NEUTRAL:
            raise ValueError(f"{ion_species} is not a dopant species")

        # Get depth profile
        profile = self.srim_estimator.predict_depth_profile(
            ion_species, energy_keV, dose_cm2, tilt_angle_deg
        )

        # Calculate activation fraction
        activation_fraction = self._calculate_activation(
            ion_species, anneal_temp_C, anneal_time_s, profile
        )

        # Active dopant concentration
        active_conc = profile.concentration_cm3 * activation_fraction

        # Calculate junction depth (where concentration drops below 1e15 cm⁻³)
        junction_threshold = 1e15  # cm⁻³
        junction_idx = np.where(active_conc < junction_threshold)[0]
        if len(junction_idx) > 0:
            junction_depth_nm = profile.depth_nm[junction_idx[0]]
        else:
            junction_depth_nm = profile.depth_nm[-1]

        # Peak concentration
        peak_concentration = np.max(active_conc)

        # Calculate mobility (concentration-dependent)
        mobility = self._calculate_mobility(ion_species, peak_concentration, anneal_temp_C)

        # Calculate sheet resistance using integration
        # Rs = 1 / (q * ∫ n(x) * μ(x) dx)
        q = 1.602e-19  # Elementary charge (C)

        # Numerical integration
        dx_cm = (profile.depth_nm[1] - profile.depth_nm[0]) * 1e-7  # Convert nm to cm
        conductivity_integral = np.sum(active_conc * mobility) * dx_cm

        if conductivity_integral > 0:
            sheet_resistance = 1.0 / (q * conductivity_integral)
        else:
            sheet_resistance = 1e6  # Very high resistance (essentially insulating)

        # Confidence assessment
        confidence = self._assess_confidence(
            ion_species, dose_cm2, anneal_temp_C, anneal_time_s
        )

        # Document assumptions
        assumptions = [
            "Uniform implant across wafer",
            "Complete damage annealing",
            "No dopant clustering or precipitation",
            f"Activation fraction: {activation_fraction:.1%}",
            f"Junction depth: {junction_depth_nm:.1f} nm",
            f"Mobility model: concentration-dependent",
        ]

        return SheetResistanceEstimate(
            sheet_resistance_ohm_per_sq=sheet_resistance,
            junction_depth_nm=junction_depth_nm,
            peak_concentration_cm3=peak_concentration,
            activation_fraction=activation_fraction,
            carrier_mobility_cm2_per_Vs=mobility,
            confidence_level=confidence,
            assumptions=assumptions
        )

    def _calculate_activation(
        self,
        ion_species: IonSpecies,
        anneal_temp_C: float,
        anneal_time_s: float,
        profile: DepthProfile
    ) -> float:
        """Calculate dopant activation fraction after annealing."""
        # Activation depends on:
        # 1. Solid solubility limit
        # 2. Anneal temperature and time
        # 3. Peak concentration

        solubility_limit = SOLUBILITY_LIMITS[ion_species]
        peak_conc = np.max(profile.concentration_cm3)

        # If below solubility, activation is primarily limited by anneal conditions
        if peak_conc < solubility_limit:
            # Activation increases with temperature and time
            # Simplified Arrhenius model
            if anneal_temp_C >= 900:
                base_activation = 0.90  # 90% activation for high-temp RTP
            elif anneal_temp_C >= 700:
                base_activation = 0.70  # 70% for medium-temp
            else:
                base_activation = 0.40  # 40% for low-temp

            # Time dependence (logarithmic)
            time_factor = np.log10(anneal_time_s + 1) / 3.0  # Normalize to ~100s
            time_factor = min(time_factor, 1.0)

            activation_fraction = base_activation * (0.7 + 0.3 * time_factor)
        else:
            # Above solubility: activation limited by solid solubility
            activation_fraction = 0.9 * (solubility_limit / peak_conc)
            activation_fraction = min(activation_fraction, 0.95)

        return activation_fraction

    def _calculate_mobility(
        self,
        ion_species: IonSpecies,
        concentration_cm3: float,
        temp_C: float = 25.0
    ) -> float:
        """
        Calculate carrier mobility (concentration-dependent).

        Uses empirical models (Caughey-Thomas or Masetti).
        """
        dopant_type = DOPANT_TYPES[ion_species]

        # Low-field mobilities at 300K (cm²/V·s)
        if dopant_type == DopantType.P_TYPE:  # Holes in n-Si
            mu_min = 44.9
            mu_max = 470.5
            N_ref = 2.23e17
            alpha = 0.719
        else:  # Electrons in p-Si (N_TYPE dopants)
            mu_min = 52.2
            mu_max = 1417.0
            N_ref = 9.68e16
            alpha = 0.680

        # Caughey-Thomas model
        N = concentration_cm3
        mobility = mu_min + (mu_max - mu_min) / (1 + (N / N_ref)**alpha)

        # Temperature dependence (simplified)
        T_ratio = (temp_C + 273.15) / 300.0
        mobility *= T_ratio**(-2.5)

        return mobility

    def _assess_confidence(
        self,
        ion_species: IonSpecies,
        dose_cm2: float,
        anneal_temp_C: float,
        anneal_time_s: float
    ) -> str:
        """Assess confidence level in the estimate."""
        # High confidence conditions:
        # - Standard dopants (B, P, As)
        # - Dose in typical range (1e13 - 1e16 cm⁻²)
        # - Standard RTP conditions (900-1100°C, 10-60s)

        confidence_score = 1.0

        # Dopant check
        if ion_species not in [IonSpecies.BORON, IonSpecies.PHOSPHORUS, IonSpecies.ARSENIC]:
            confidence_score *= 0.7  # Less data for Sb

        # Dose range check
        if dose_cm2 < 1e13 or dose_cm2 > 1e16:
            confidence_score *= 0.8  # Outside typical range

        # Anneal conditions
        if anneal_temp_C < 800 or anneal_temp_C > 1200:
            confidence_score *= 0.7  # Non-standard temperature

        if anneal_time_s < 5 or anneal_time_s > 300:
            confidence_score *= 0.8  # Non-standard time

        if confidence_score > 0.8:
            return "high"
        elif confidence_score > 0.6:
            return "medium"
        else:
            return "low"


# Export
__all__ = [
    "SRIMEstimator",
    "ChannelingRiskPredictor",
    "SheetResistanceEstimator",
    "RangeParameters",
    "ChannelingRisk",
    "SheetResistanceEstimate",
    "DepthProfile",
    "DopantType",
    "ION_MASSES",
    "DOPANT_TYPES",
]
