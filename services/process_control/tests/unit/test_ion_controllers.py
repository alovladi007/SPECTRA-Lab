"""Unit tests for ion implantation controllers."""

import pytest
import numpy as np
from app.controllers.ion import (
    DoseIntegrator,
    ScanUniformityController,
    R2RController,
    BeamDriftDetector,
)


class TestDoseIntegrator:
    """Test dose integrator (Q = ∫I(t)dt / A)."""

    def test_initialization(self):
        """Test integrator initializes correctly."""
        integrator = DoseIntegrator(
            wafer_area_cm2=707.0,
            charge_state=1
        )

        assert integrator.wafer_area_cm2 == 707.0
        assert integrator.charge_state == 1
        assert integrator.integrated_charge_C == 0.0

    def test_integration_increases_dose(self):
        """Test that integration increases dose."""
        integrator = DoseIntegrator(wafer_area_cm2=707.0)

        dose1 = integrator.get_dose()

        # Integrate for 1 second at 5 mA
        integrator.integrate(beam_current_mA=5.0, timestamp=1.0, dt=1.0)

        dose2 = integrator.get_dose()

        assert dose2 > dose1

    def test_dose_proportional_to_current(self):
        """Test dose is proportional to beam current."""
        int1 = DoseIntegrator(wafer_area_cm2=707.0)
        int2 = DoseIntegrator(wafer_area_cm2=707.0)

        # Integrate at different currents for same time
        for _ in range(10):
            int1.integrate(beam_current_mA=5.0, timestamp=1.0, dt=1.0)
            int2.integrate(beam_current_mA=10.0, timestamp=1.0, dt=1.0)

        dose1 = int1.get_dose()
        dose2 = int2.get_dose()

        # dose2 should be ~2x dose1
        assert abs(dose2 / dose1 - 2.0) < 0.01

    def test_dose_proportional_to_time(self):
        """Test dose is proportional to time."""
        int1 = DoseIntegrator(wafer_area_cm2=707.0)
        int2 = DoseIntegrator(wafer_area_cm2=707.0)

        # Integrate for different times
        int1.integrate(beam_current_mA=5.0, timestamp=1.0, dt=1.0)
        int2.integrate(beam_current_mA=5.0, timestamp=2.0, dt=2.0)

        dose1 = int1.get_dose()
        dose2 = int2.get_dose()

        # dose2 should be ~2x dose1
        assert abs(dose2 / dose1 - 2.0) < 0.01

    def test_dose_inversely_proportional_to_area(self):
        """Test dose is inversely proportional to area."""
        int1 = DoseIntegrator(wafer_area_cm2=707.0)  # 300mm wafer
        int2 = DoseIntegrator(wafer_area_cm2=314.0)  # 200mm wafer

        # Same implant
        for _ in range(10):
            int1.integrate(beam_current_mA=5.0, timestamp=1.0, dt=1.0)
            int2.integrate(beam_current_mA=5.0, timestamp=1.0, dt=1.0)

        dose1 = int1.get_dose()
        dose2 = int2.get_dose()

        # Smaller area -> higher dose
        assert dose2 > dose1

    def test_charge_state_correction(self):
        """Test charge state correction."""
        int1 = DoseIntegrator(wafer_area_cm2=707.0, charge_state=1)
        int2 = DoseIntegrator(wafer_area_cm2=707.0, charge_state=2)

        # Same beam current
        int1.integrate(beam_current_mA=5.0, timestamp=1.0, dt=1.0)
        int2.integrate(beam_current_mA=5.0, timestamp=1.0, dt=1.0)

        dose1 = int1.get_dose()
        dose2 = int2.get_dose()

        # Higher charge state -> fewer ions for same current
        assert dose2 < dose1

    def test_area_correction_factor(self):
        """Test area correction factor application."""
        integrator = DoseIntegrator(wafer_area_cm2=707.0)

        integrator.integrate(beam_current_mA=5.0, timestamp=1.0, dt=10.0)

        dose_uncorrected = integrator.get_dose(apply_corrections=False)

        integrator.set_area_correction(0.95)  # 5% effective area loss
        dose_corrected = integrator.get_dose(apply_corrections=True)

        # Corrected dose should be higher
        assert dose_corrected > dose_uncorrected

    def test_reset(self):
        """Test reset clears integration."""
        integrator = DoseIntegrator(wafer_area_cm2=707.0)

        integrator.integrate(beam_current_mA=5.0, timestamp=1.0, dt=10.0)
        assert integrator.get_dose() > 0

        integrator.reset()
        assert integrator.get_dose() == 0


class TestScanUniformityController:
    """Test scan uniformity controller."""

    def test_analyze_perfect_uniformity(self):
        """Test uniformity analysis for perfect uniform dose."""
        controller = ScanUniformityController()

        # Create perfect uniform dose map
        dose_map = np.ones((50, 50)) * 1e15
        x_pos = np.linspace(-150, 150, 50)
        y_pos = np.linspace(-150, 150, 50)

        result = controller.analyze_uniformity(dose_map, x_pos, y_pos)

        assert result.uniformity_pct > 99.0
        assert abs(result.centroid_x_mm) < 1.0
        assert abs(result.centroid_y_mm) < 1.0

    def test_analyze_nonuniform_dose(self):
        """Test uniformity analysis for non-uniform dose."""
        controller = ScanUniformityController()

        # Create non-uniform dose map (higher on left)
        x = np.linspace(-150, 150, 50)
        y = np.linspace(-150, 150, 50)
        X, Y = np.meshgrid(x, y)

        # Gradient from left to right
        dose_map = 1e15 * (1.0 + 0.2 * X / 150.0)

        result = controller.analyze_uniformity(dose_map, x, y)

        assert result.uniformity_pct < 95.0
        assert result.centroid_x_mm > 0  # Shifted to right

    def test_edge_rolloff_detection(self):
        """Test edge rolloff detection."""
        controller = ScanUniformityController()

        # Create dose map with edge rolloff
        x = np.linspace(-150, 150, 50)
        y = np.linspace(-150, 150, 50)
        X, Y = np.meshgrid(x, y)

        # Radial rolloff
        R = np.sqrt(X**2 + Y**2)
        dose_map = 1e15 * np.exp(-((R - 120) / 30)**2)  # Drops at edges

        result = controller.analyze_uniformity(dose_map, x, y)

        assert result.edge_rolloff_pct > 5.0

    def test_calculate_corrections(self):
        """Test correction calculation."""
        controller = ScanUniformityController()

        # Create tilted dose map
        x = np.linspace(-150, 150, 50)
        y = np.linspace(-150, 150, 50)
        X, Y = np.meshgrid(x, y)

        dose_map = 1e15 * (1.0 + 0.1 * X / 150.0)

        uniformity = controller.analyze_uniformity(dose_map, x, y)
        corrections = controller.calculate_corrections(uniformity)

        # Should recommend steering correction
        assert abs(corrections.steering_x_mm) > 0
        assert corrections.amplitude_correction != 1.0

    def test_apply_corrections_improves_uniformity(self):
        """Test that applying corrections improves uniformity."""
        controller = ScanUniformityController()

        # Initial non-uniform map
        x = np.linspace(-150, 150, 50)
        y = np.linspace(-150, 150, 50)
        X, Y = np.meshgrid(x, y)
        dose_map = 1e15 * (1.0 + 0.2 * X / 150.0)

        uniformity_before = controller.analyze_uniformity(dose_map, x, y)
        corrections = controller.calculate_corrections(uniformity_before)

        # Apply corrections (shift centroid)
        X_corrected = X - corrections.steering_x_mm
        dose_map_corrected = 1e15 * (1.0 + 0.2 * X_corrected / 150.0)

        uniformity_after = controller.analyze_uniformity(dose_map_corrected, x, y)

        # Centroid should be more centered
        assert abs(uniformity_after.centroid_x_mm) < abs(uniformity_before.centroid_x_mm)


class TestR2RController:
    """Test Run-to-Run controller."""

    def test_initialization(self):
        """Test R2R controller initializes."""
        controller = R2RController(alpha=0.3)

        assert controller.state.run_count == 0
        assert len(controller.state.dose_history) == 0

    def test_update_increases_run_count(self):
        """Test update increases run count."""
        controller = R2RController()

        controller.update(
            measured_dose_cm2=1e15,
            target_dose_cm2=1e15,
            measured_uniformity_pct=95.0
        )

        assert controller.state.run_count == 1

    def test_dose_correction_on_error(self):
        """Test dose correction when measured != target."""
        controller = R2RController()

        # Measured dose is 10% low
        controller.update(
            measured_dose_cm2=0.9e15,
            target_dose_cm2=1.0e15,
            measured_uniformity_pct=95.0
        )

        recommendation = controller.get_recommendation()

        # Should recommend increasing dose
        assert recommendation.recommended_dose_adjustment > 1.0

    def test_ewma_smoothing(self):
        """Test EWMA smoothing over multiple runs."""
        controller = R2RController(alpha=0.3)

        # Simulate multiple runs with consistent error
        for _ in range(10):
            controller.update(
                measured_dose_cm2=0.95e15,  # Consistently 5% low
                target_dose_cm2=1.0e15,
                measured_uniformity_pct=95.0
            )

        recommendation = controller.get_recommendation()

        # Should converge to correction
        assert recommendation.recommended_dose_adjustment > 1.0

    def test_uniformity_triggers_tilt_adjustment(self):
        """Test that poor uniformity triggers tilt adjustment."""
        controller = R2RController()

        # Good uniformity
        controller.update(
            measured_dose_cm2=1e15,
            target_dose_cm2=1e15,
            measured_uniformity_pct=98.0
        )

        rec1 = controller.get_recommendation()
        tilt1 = rec1.recommended_tilt_deg

        # Poor uniformity
        for _ in range(5):
            controller.update(
                measured_dose_cm2=1e15,
                target_dose_cm2=1e15,
                measured_uniformity_pct=85.0
            )

        rec2 = controller.get_recommendation()
        tilt2 = rec2.recommended_tilt_deg

        # Should recommend increasing tilt
        assert tilt2 > tilt1


class TestBeamDriftDetector:
    """Test beam drift FDC detector."""

    def test_initialization(self):
        """Test detector initializes."""
        detector = BeamDriftDetector()

        assert len(detector.state.x_history) == 0

    def test_no_drift_on_stable_beam(self):
        """Test no drift detected on stable beam."""
        detector = BeamDriftDetector()

        # Simulate stable beam
        for i in range(100):
            result = detector.update(
                x_position_mm=0.0,
                y_position_mm=0.0,
                timestamp=i * 0.1
            )

        assert result.drift_detected == False
        assert result.recommended_action == "none"

    def test_drift_detection_on_slow_drift(self):
        """Test drift detection on slow drift."""
        detector = BeamDriftDetector(drift_threshold_mm=1.0)

        # Simulate slow drift
        for i in range(200):
            x_pos = 0.01 * i  # 0.01 mm/step drift
            result = detector.update(
                x_position_mm=x_pos,
                y_position_mm=0.0,
                timestamp=i * 0.1
            )

        # Should eventually detect drift
        assert result.drift_detected == True
        assert result.recommended_action in ["compensate", "pause"]

    def test_spike_detection(self):
        """Test spike detection."""
        detector = BeamDriftDetector(spike_threshold_mm=2.0)

        # Simulate stable beam then spike
        for i in range(50):
            detector.update(x_position_mm=0.0, y_position_mm=0.0, timestamp=i * 0.1)

        # Sudden spike
        result = detector.update(x_position_mm=5.0, y_position_mm=0.0, timestamp=5.0)

        assert result.spike_detected == True
        assert result.recommended_action == "pause"

    def test_drift_rate_calculation(self):
        """Test drift rate calculation."""
        detector = BeamDriftDetector()

        # Linear drift
        drift_rate_expected = 0.1  # mm/s
        for i in range(100):
            t = i * 0.1
            x_pos = drift_rate_expected * t
            detector.update(x_position_mm=x_pos, y_position_mm=0.0, timestamp=t)

        result = detector.get_current_state()

        # Should calculate drift rate
        assert abs(result.x_drift_rate_mm_per_s - drift_rate_expected) < 0.05

    def test_compensation_recommendation(self):
        """Test compensation values are calculated."""
        detector = BeamDriftDetector()

        # Simulate drift
        for i in range(100):
            x_pos = 0.02 * i
            detector.update(x_position_mm=x_pos, y_position_mm=0.0, timestamp=i * 0.1)

        result = detector.get_current_state()

        if result.drift_detected:
            # Should recommend steering compensation
            assert result.recommended_steering_x_mm != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
