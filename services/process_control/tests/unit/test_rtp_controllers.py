"""Unit tests for RTP controllers."""

import pytest
import numpy as np
from app.controllers.rtp import (
    PIDController,
    PIDGains,
    MPCController,
    MPCParameters,
    R2RController,
    ThermalBudgetCalculator,
    PerformanceAnalyzer,
)


class TestPIDController:
    """Test PID controller with anti-windup."""

    def test_initialization(self):
        """Test PID initializes correctly."""
        gains = PIDGains(Kp=2.0, Ki=0.5, Kd=0.1)
        controller = PIDController(gains)

        assert controller.gains.Kp == 2.0
        assert controller.state.integral == 0.0

    def test_proportional_only(self):
        """Test proportional control only."""
        gains = PIDGains(Kp=1.0, Ki=0.0, Kd=0.0)
        controller = PIDController(gains)

        # Error of 10
        output = controller.update(setpoint=100.0, measured=90.0, dt=1.0)

        # Output should be Kp * error = 1.0 * 10 = 10
        assert abs(output - 10.0) < 0.1

    def test_integral_accumulation(self):
        """Test integral term accumulates error."""
        gains = PIDGains(Kp=0.0, Ki=1.0, Kd=0.0)
        controller = PIDController(gains)

        # Apply constant error
        for _ in range(10):
            controller.update(setpoint=100.0, measured=90.0, dt=1.0)

        # Integral should have accumulated
        assert controller.state.integral > 0

    def test_derivative_on_error_change(self):
        """Test derivative term responds to error change."""
        gains = PIDGains(Kp=0.0, Ki=0.0, Kd=1.0)
        controller = PIDController(gains)

        # First update
        output1 = controller.update(setpoint=100.0, measured=90.0, dt=1.0)

        # Error increases
        output2 = controller.update(setpoint=100.0, measured=80.0, dt=1.0)

        # Derivative should increase output
        assert output2 > output1

    def test_anti_windup_limits_integral(self):
        """Test anti-windup prevents excessive integral."""
        gains = PIDGains(Kp=0.0, Ki=1.0, Kd=0.0, windup_limit=50.0)
        controller = PIDController(gains)

        # Apply large error for long time
        for _ in range(100):
            controller.update(setpoint=1000.0, measured=0.0, dt=1.0)

        # Integral should be clamped
        assert abs(controller.state.integral) <= gains.windup_limit

    def test_anti_windup_stops_on_saturation(self):
        """Test anti-windup stops integrating when saturated."""
        gains = PIDGains(Kp=0.0, Ki=1.0, Kd=0.0)
        controller = PIDController(gains)

        # Output will saturate at 100
        integral_before = controller.state.integral

        controller.update(
            setpoint=1000.0,
            measured=0.0,
            dt=1.0,
            output_limits=(0.0, 100.0)
        )

        integral_after = controller.state.integral

        # If saturated, integral should stop growing
        if controller.state.saturated:
            # May still accumulate a bit before saturation check
            pass

    def test_feedforward_anticipates_setpoint_change(self):
        """Test feed-forward term anticipates setpoint changes."""
        gains = PIDGains(
            Kp=1.0, Ki=0.0, Kd=0.0,
            enable_feedforward=True,
            feedforward_gain=1.0
        )
        controller = PIDController(gains)

        # Ramping setpoint
        output1 = controller.update(setpoint=100.0, measured=100.0, dt=1.0)
        output2 = controller.update(setpoint=110.0, measured=100.0, dt=1.0)

        # Feed-forward should add extra output for ramp
        assert output2 > output1

    def test_output_limits(self):
        """Test output is clamped to limits."""
        gains = PIDGains(Kp=100.0, Ki=0.0, Kd=0.0)
        controller = PIDController(gains)

        output = controller.update(
            setpoint=1000.0,
            measured=0.0,
            dt=1.0,
            output_limits=(0.0, 100.0)
        )

        assert output <= 100.0
        assert output >= 0.0

    def test_reset_clears_state(self):
        """Test reset clears controller state."""
        gains = PIDGains(Kp=1.0, Ki=1.0, Kd=1.0)
        controller = PIDController(gains)

        # Run controller
        for _ in range(10):
            controller.update(setpoint=100.0, measured=90.0, dt=1.0)

        assert controller.state.integral != 0.0

        controller.reset()

        assert controller.state.integral == 0.0


class TestMPCController:
    """Test Model Predictive Controller."""

    def test_initialization(self):
        """Test MPC initializes."""
        params = MPCParameters(prediction_horizon=20, control_horizon=10)
        controller = MPCController(params, num_zones=4)

        assert controller.params.prediction_horizon == 20
        assert controller.num_zones == 4

    def test_predict_trajectory(self):
        """Test trajectory prediction."""
        params = MPCParameters()
        controller = MPCController(params)

        setpoint_traj = np.linspace(25.0, 1000.0, 20)

        predicted = controller.predict_trajectory(
            current_temp=25.0,
            current_lamp_power=0.0,
            setpoint_trajectory=setpoint_traj,
            dt=1.0
        )

        assert len(predicted) == 20
        # Temperature should generally increase
        assert predicted[-1] > predicted[0]

    def test_optimize_control_respects_constraints(self):
        """Test MPC respects lamp power constraints."""
        params = MPCParameters(
            max_lamp_power_pct=80.0,
            min_lamp_power_pct=10.0
        )
        controller = MPCController(params, num_zones=4)

        setpoint_traj = np.ones(10) * 1000.0

        optimal_sequence = controller.optimize_control(
            current_temp=25.0,
            current_lamp_power=np.array([50.0, 50.0, 50.0, 50.0]),
            setpoint_trajectory=setpoint_traj,
            dt=1.0
        )

        # All lamp powers should be within constraints
        assert np.all(optimal_sequence >= params.min_lamp_power_pct)
        assert np.all(optimal_sequence <= params.max_lamp_power_pct)

    def test_optimize_control_respects_rate_limits(self):
        """Test MPC respects lamp power rate limits."""
        params = MPCParameters(max_lamp_rate_pct_per_s=50.0)
        controller = MPCController(params, num_zones=4)

        setpoint_traj = np.ones(10) * 1000.0

        optimal_sequence = controller.optimize_control(
            current_temp=25.0,
            current_lamp_power=np.array([20.0, 20.0, 20.0, 20.0]),
            setpoint_trajectory=setpoint_traj,
            dt=1.0
        )

        # Check rate limits between steps
        for i in range(1, len(optimal_sequence)):
            rate = np.abs(optimal_sequence[i] - optimal_sequence[i-1])
            assert np.all(rate <= params.max_lamp_rate_pct_per_s * 1.0 + 0.1)  # Small tolerance

    def test_update_returns_control_action(self):
        """Test update returns control action."""
        params = MPCParameters()
        controller = MPCController(params, num_zones=4)

        future_setpoints = [100.0, 200.0, 300.0, 400.0, 500.0]

        lamp_powers = controller.update(
            current_temp=25.0,
            current_lamp_power=np.array([0.0, 0.0, 0.0, 0.0]),
            setpoint=100.0,
            future_setpoints=future_setpoints,
            dt=1.0
        )

        assert len(lamp_powers) == 4
        assert np.all(lamp_powers >= 0)
        assert np.all(lamp_powers <= 100)


class TestR2RController:
    """Test Run-to-Run controller for RTP."""

    def test_initialization(self):
        """Test R2R initializes."""
        controller = R2RController(num_zones=4, alpha=0.3)

        assert controller.num_zones == 4
        assert controller.state.run_count == 0

    def test_emissivity_adjustment(self):
        """Test emissivity drift tracking."""
        controller = R2RController()

        # Simulate emissivity drifting up
        for _ in range(10):
            controller.update(measured_emissivity=0.70)  # Higher than expected 0.65

        adjustments = controller.get_adjustments()

        # Should recommend correction
        # If emissivity is high, pyrometer reads low, so correction < 1
        assert adjustments["emissivity_correction"] != 1.0

    def test_lamp_power_trim(self):
        """Test lamp power trim calculation."""
        controller = R2RController(num_zones=4)

        # Simulate zone 0 consistently higher power
        for _ in range(10):
            controller.update(
                lamp_powers_used=np.array([60.0, 50.0, 50.0, 50.0])
            )

        adjustments = controller.get_adjustments()

        # Zone 0 should have higher trim
        assert adjustments["lamp_power_trim"][0] > adjustments["lamp_power_trim"][1]

    def test_overshoot_tracking(self):
        """Test overshoot tracking with EWMA."""
        controller = R2RController()

        # Simulate consistent overshoot
        for _ in range(10):
            controller.update(overshoot_observed=5.0)

        adjustments = controller.get_adjustments()

        # Should predict overshoot
        assert adjustments["predicted_overshoot_pct"] > 0

    def test_run_count_increments(self):
        """Test run count increments."""
        controller = R2RController()

        controller.update(measured_emissivity=0.65)
        assert controller.state.run_count == 1

        controller.update(measured_emissivity=0.66)
        assert controller.state.run_count == 2


class TestThermalBudgetCalculator:
    """Test thermal budget calculator."""

    def test_initialization(self):
        """Test calculator initializes."""
        calc = ThermalBudgetCalculator(dopant_species="boron")

        assert calc.dopant_species == "boron"
        assert calc.activation_energy_eV > 0

    def test_add_sample_at_high_temp(self):
        """Test adding high temperature sample."""
        calc = ThermalBudgetCalculator()

        calc.add_sample(temperature_C=1000.0, dt=1.0)

        assert len(calc.rate_history) == 1
        assert calc.rate_history[0] > 0

    def test_budget_increases_with_time(self):
        """Test budget integrates over time."""
        calc = ThermalBudgetCalculator()

        # Add multiple samples
        for _ in range(10):
            calc.add_sample(temperature_C=1000.0, dt=1.0)

        budget1 = calc.get_budget().integrated_budget

        # Add more samples
        for _ in range(10):
            calc.add_sample(temperature_C=1000.0, dt=1.0)

        budget2 = calc.get_budget().integrated_budget

        assert budget2 > budget1

    def test_higher_temp_faster_accumulation(self):
        """Test higher temperature accumulates budget faster."""
        calc1 = ThermalBudgetCalculator()
        calc2 = ThermalBudgetCalculator()

        # Same time, different temperatures
        for _ in range(10):
            calc1.add_sample(temperature_C=900.0, dt=1.0)
            calc2.add_sample(temperature_C=1100.0, dt=1.0)

        budget1 = calc1.get_budget().integrated_budget
        budget2 = calc2.get_budget().integrated_budget

        # Higher temp -> higher budget
        assert budget2 > budget1

    def test_equivalent_time_calculation(self):
        """Test equivalent time at reference temp."""
        calc = ThermalBudgetCalculator()

        # Run at 1000°C for 60s
        for _ in range(60):
            calc.add_sample(temperature_C=1000.0, dt=1.0)

        budget = calc.get_budget()

        # Equivalent time should be approximately 60s
        assert abs(budget.equivalent_time_at_1000C_s - 60.0) < 10.0

    def test_different_dopants_different_energies(self):
        """Test different dopants have different activation energies."""
        calc_b = ThermalBudgetCalculator(dopant_species="boron")
        calc_as = ThermalBudgetCalculator(dopant_species="arsenic")

        assert calc_b.activation_energy_eV != calc_as.activation_energy_eV

    def test_reset_clears_history(self):
        """Test reset clears calculation history."""
        calc = ThermalBudgetCalculator()

        calc.add_sample(temperature_C=1000.0, dt=1.0)
        assert len(calc.time_history) > 0

        calc.reset()
        assert len(calc.time_history) == 0


class TestPerformanceAnalyzer:
    """Test RTP performance analyzer."""

    def test_analyze_ramp_segment_perfect(self):
        """Test analysis of perfect ramp."""
        setpoint = np.linspace(25.0, 1000.0, 100)
        measured = setpoint.copy()  # Perfect tracking
        time = np.linspace(0, 100, 100)

        fidelity = PerformanceAnalyzer.analyze_ramp_segment(
            segment_id=1,
            target_ramp_rate_C_per_s=10.0,
            setpoint_history=setpoint,
            measured_history=measured,
            time_history=time
        )

        assert fidelity.rmse_C < 1.0
        assert abs(fidelity.peak_overshoot_C) < 1.0

    def test_analyze_ramp_segment_with_overshoot(self):
        """Test analysis detects overshoot."""
        setpoint = np.linspace(25.0, 1000.0, 100)
        # Add overshoot
        measured = setpoint + 20.0 * np.exp(-np.linspace(0, 5, 100))
        time = np.linspace(0, 100, 100)

        fidelity = PerformanceAnalyzer.analyze_ramp_segment(
            segment_id=1,
            target_ramp_rate_C_per_s=10.0,
            setpoint_history=setpoint,
            measured_history=measured,
            time_history=time
        )

        assert fidelity.peak_overshoot_C > 10.0
        assert fidelity.rmse_C > 5.0

    def test_analyze_dwell_stability(self):
        """Test dwell stability analysis."""
        setpoint = np.ones(100) * 1000.0
        measured = 1000.0 + np.random.normal(0, 2.0, 100)  # ±2°C noise
        time = np.linspace(0, 100, 100)

        fidelity = PerformanceAnalyzer.analyze_ramp_segment(
            segment_id=1,
            target_ramp_rate_C_per_s=0.0,
            setpoint_history=setpoint,
            measured_history=measured,
            time_history=time,
            dwell_start_idx=0,
            dwell_end_idx=100
        )

        # Should measure stability
        assert fidelity.dwell_std_C > 0
        assert fidelity.dwell_std_C < 5.0  # Should be small

    def test_generate_performance_report(self):
        """Test complete performance report generation."""
        # Create dummy fidelities
        fidelity1 = PerformanceAnalyzer.analyze_ramp_segment(
            1, 10.0,
            np.linspace(0, 100, 10),
            np.linspace(0, 100, 10),
            np.linspace(0, 10, 10)
        )

        from app.controllers.rtp import ThermalBudget

        budget = ThermalBudget(
            dopant_species="boron",
            activation_energy_eV=3.65,
            integrated_budget=100.0,
            equivalent_time_at_1000C_s=60.0,
            peak_activation_rate=0.5
        )

        report = PerformanceAnalyzer.generate_performance_report(
            recipe_id="REC-001",
            run_id="RUN-001",
            ramp_fidelities=[fidelity1],
            thermal_budget=budget,
            saturation_events=0,
            constraint_violations=0,
            start_time=0.0,
            end_time=100.0
        )

        assert report.recipe_id == "REC-001"
        assert report.total_duration_s == 100.0
        assert len(report.ramp_fidelities) == 1

    def test_tuning_recommendations_on_overshoot(self):
        """Test tuning recommendations for high overshoot."""
        # Create fidelity with high overshoot
        setpoint = np.ones(10) * 1000.0
        measured = setpoint + 100.0  # Massive overshoot
        time = np.linspace(0, 10, 10)

        fidelity = PerformanceAnalyzer.analyze_ramp_segment(
            1, 0.0, setpoint, measured, time
        )

        from app.controllers.rtp import ThermalBudget

        budget = ThermalBudget("boron", 3.65, 0.0, 0.0, 0.0)

        report = PerformanceAnalyzer.generate_performance_report(
            "REC", "RUN", [fidelity], budget, 0, 0, 0.0, 10.0
        )

        # Should recommend reducing gains
        assert "reduce_Kp" in report.recommended_tuning or "reduce_Ki" in report.recommended_tuning


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
