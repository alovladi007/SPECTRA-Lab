"""Unit tests for ion range models."""

import pytest
import numpy as np
from app.models.ion_range import (
    SRIMEstimator,
    ChannelingRiskPredictor,
    SheetResistanceEstimator,
    IonSpecies,
    DopantType,
)


class TestSRIMEstimator:
    """Test SRIM-like range estimator."""

    def test_estimate_range_boron(self):
        """Test range estimation for boron."""
        estimator = SRIMEstimator()
        result = estimator.estimate_range(
            ion_species=IonSpecies.BORON,
            energy_keV=10.0,
            tilt_angle_deg=7.0
        )

        # Basic sanity checks
        assert result.projected_range_nm > 0
        assert result.range_straggle_nm > 0
        assert result.lateral_straggle_nm > 0
        assert result.range_straggle_nm < result.projected_range_nm
        assert result.tilt_angle_deg == 7.0

    def test_estimate_range_phosphorus(self):
        """Test range estimation for phosphorus."""
        estimator = SRIMEstimator()
        result = estimator.estimate_range(
            ion_species=IonSpecies.PHOSPHORUS,
            energy_keV=30.0,
            tilt_angle_deg=7.0
        )

        assert result.projected_range_nm > 0
        assert result.ion_species == IonSpecies.PHOSPHORUS

    def test_estimate_range_arsenic(self):
        """Test range estimation for arsenic (heavy ion)."""
        estimator = SRIMEstimator()
        result = estimator.estimate_range(
            ion_species=IonSpecies.ARSENIC,
            energy_keV=50.0,
            tilt_angle_deg=7.0
        )

        # Arsenic is heavier, should have shorter range than boron at same energy
        boron_result = estimator.estimate_range(
            ion_species=IonSpecies.BORON,
            energy_keV=50.0,
            tilt_angle_deg=7.0
        )

        assert result.projected_range_nm < boron_result.projected_range_nm

    def test_range_increases_with_energy(self):
        """Test that range increases with energy."""
        estimator = SRIMEstimator()

        range_10keV = estimator.estimate_range(
            IonSpecies.BORON, 10.0, 7.0
        ).projected_range_nm

        range_50keV = estimator.estimate_range(
            IonSpecies.BORON, 50.0, 7.0
        ).projected_range_nm

        assert range_50keV > range_10keV

    def test_predict_depth_profile(self):
        """Test depth profile generation."""
        estimator = SRIMEstimator()
        profile = estimator.predict_depth_profile(
            ion_species=IonSpecies.BORON,
            energy_keV=10.0,
            dose_cm2=1e15,
            tilt_angle_deg=7.0
        )

        assert len(profile.depth_nm) > 0
        assert len(profile.concentration_cm3) == len(profile.depth_nm)
        assert profile.peak_concentration_cm3 > 0
        assert profile.peak_depth_nm > 0

        # Peak should be near projected range
        peak_idx = np.argmax(profile.concentration_cm3)
        assert abs(profile.depth_nm[peak_idx] - profile.peak_depth_nm) < 10.0

        # Integral should approximately equal dose
        integrated_dose = np.trapz(profile.concentration_cm3, profile.depth_nm * 1e-7)  # cm
        assert abs(integrated_dose - 1e15) / 1e15 < 0.5  # Within 50%

    def test_channeling_effect_on_profile(self):
        """Test that low tilt angle increases channeling tail."""
        estimator = SRIMEstimator()

        # High tilt (no channeling)
        profile_high_tilt = estimator.predict_depth_profile(
            IonSpecies.BORON, 10.0, 1e15, tilt_angle_deg=7.0
        )

        # Low tilt (channeling)
        profile_low_tilt = estimator.predict_depth_profile(
            IonSpecies.BORON, 10.0, 1e15, tilt_angle_deg=0.5
        )

        # Low tilt should have longer tail
        assert profile_low_tilt.depth_nm[-1] > profile_high_tilt.depth_nm[-1]


class TestChannelingRiskPredictor:
    """Test channeling risk assessment."""

    def test_assess_channeling_risk_low_tilt(self):
        """Test high channeling risk at low tilt."""
        predictor = ChannelingRiskPredictor()
        risk = predictor.assess_channeling_risk(
            ion_species=IonSpecies.BORON,
            energy_keV=10.0,
            tilt_angle_deg=0.5,  # Low tilt
            twist_angle_deg=0.0
        )

        assert risk.risk_level == "HIGH"
        assert risk.channeling_probability > 0.5

    def test_assess_channeling_risk_high_tilt(self):
        """Test low channeling risk at high tilt."""
        predictor = ChannelingRiskPredictor()
        risk = predictor.assess_channeling_risk(
            ion_species=IonSpecies.BORON,
            energy_keV=10.0,
            tilt_angle_deg=7.0,  # Standard tilt
            twist_angle_deg=0.0
        )

        assert risk.risk_level == "LOW"
        assert risk.channeling_probability < 0.1

    def test_critical_angle_calculation(self):
        """Test critical angle increases with energy."""
        predictor = ChannelingRiskPredictor()

        risk_10keV = predictor.assess_channeling_risk(
            IonSpecies.BORON, 10.0, 1.0, 0.0
        )

        risk_50keV = predictor.assess_channeling_risk(
            IonSpecies.BORON, 50.0, 1.0, 0.0
        )

        # Higher energy -> larger critical angle -> lower risk at same tilt
        assert risk_50keV.channeling_probability < risk_10keV.channeling_probability

    def test_twist_angle_reduces_risk(self):
        """Test that twist angle reduces channeling."""
        predictor = ChannelingRiskPredictor()

        risk_no_twist = predictor.assess_channeling_risk(
            IonSpecies.BORON, 10.0, 1.0, 0.0
        )

        risk_with_twist = predictor.assess_channeling_risk(
            IonSpecies.BORON, 10.0, 1.0, 22.0  # Twist off axis
        )

        assert risk_with_twist.channeling_probability < risk_no_twist.channeling_probability


class TestSheetResistanceEstimator:
    """Test sheet resistance calculation."""

    def test_estimate_sheet_resistance_boron(self):
        """Test Rs calculation for boron."""
        estimator = SheetResistanceEstimator()
        result = estimator.estimate_sheet_resistance(
            ion_species=IonSpecies.BORON,
            energy_keV=10.0,
            dose_cm2=1e15,
            anneal_temp_C=1000.0,
            anneal_time_s=30.0
        )

        assert result.sheet_resistance_ohm_per_sq > 0
        assert result.activation_fraction > 0
        assert result.activation_fraction <= 1.0
        assert result.mobility_cm2_per_Vs > 0

    def test_higher_dose_lower_Rs(self):
        """Test that higher dose gives lower Rs."""
        estimator = SheetResistanceEstimator()

        rs_low_dose = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e14, 1000.0, 30.0
        ).sheet_resistance_ohm_per_sq

        rs_high_dose = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e16, 1000.0, 30.0
        ).sheet_resistance_ohm_per_sq

        assert rs_high_dose < rs_low_dose

    def test_higher_anneal_temp_higher_activation(self):
        """Test that higher anneal temperature increases activation."""
        estimator = SheetResistanceEstimator()

        result_low_temp = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e15, 800.0, 30.0
        )

        result_high_temp = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e15, 1050.0, 30.0
        )

        assert result_high_temp.activation_fraction > result_low_temp.activation_fraction

    def test_longer_anneal_higher_activation(self):
        """Test that longer anneal time increases activation."""
        estimator = SheetResistanceEstimator()

        result_short = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e15, 1000.0, 10.0
        )

        result_long = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e15, 1000.0, 60.0
        )

        assert result_long.activation_fraction >= result_short.activation_fraction

    def test_phosphorus_vs_boron_mobility(self):
        """Test that n-type (P) has higher mobility than p-type (B)."""
        estimator = SheetResistanceEstimator()

        boron = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e15, 1000.0, 30.0
        )

        phosphorus = estimator.estimate_sheet_resistance(
            IonSpecies.PHOSPHORUS, 10.0, 1e15, 1000.0, 30.0
        )

        # Electron mobility > hole mobility
        assert phosphorus.mobility_cm2_per_Vs > boron.mobility_cm2_per_Vs

    def test_high_dose_reduces_mobility(self):
        """Test that high doping reduces mobility (Caughey-Thomas)."""
        estimator = SheetResistanceEstimator()

        result_low = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 1e14, 1000.0, 30.0
        )

        result_high = estimator.estimate_sheet_resistance(
            IonSpecies.BORON, 10.0, 5e15, 1000.0, 30.0
        )

        # Higher concentration -> lower mobility due to scattering
        assert result_high.mobility_cm2_per_Vs < result_low.mobility_cm2_per_Vs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
