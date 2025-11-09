"""
Tests for Session 10 API schemas.

Validates Pydantic models with comprehensive validation tests.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from session10.api.schemas import (
    # Diffusion
    DiffusionRequest,
    DiffusionResponse,
    DopantType,
    DiffusionMethod,
    SolverType,
    # Oxidation
    OxidationRequest,
    OxidationResponse,
    AmbientType,
    # SPC
    SPCRequest,
    SPCResponse,
    TimeSeriesPoint,
    RuleViolationDetail,
    SPCRuleType,
    SPCSeverity,
    # VM
    VMRequest,
    VMResponse,
    VMModelType,
    # Calibration
    CalibrationRequest,
    CalibrationResponse,
    CalibrationMethod,
    # Common
    ErrorResponse,
    StatusResponse,
)


class TestDiffusionSchemas:
    """Test diffusion request/response models."""

    def test_valid_constant_source_request(self):
        """Test valid constant source diffusion request."""
        req = DiffusionRequest(
            dopant=DopantType.BORON,
            temp_celsius=1000,
            time_minutes=30,
            method=DiffusionMethod.CONSTANT_SOURCE,
            surface_conc=1e19,
            background=1e15
        )

        assert req.dopant == DopantType.BORON
        assert req.temp_celsius == 1000
        assert req.surface_conc == 1e19

    def test_valid_limited_source_request(self):
        """Test valid limited source diffusion request."""
        req = DiffusionRequest(
            dopant=DopantType.PHOSPHORUS,
            temp_celsius=950,
            time_minutes=60,
            method=DiffusionMethod.LIMITED_SOURCE,
            dose=1e14,
            background=1e15
        )

        assert req.method == DiffusionMethod.LIMITED_SOURCE
        assert req.dose == 1e14

    def test_missing_surface_conc_fails(self):
        """Test that constant_source without surface_conc fails."""
        with pytest.raises(ValidationError):
            DiffusionRequest(
                dopant=DopantType.BORON,
                temp_celsius=1000,
                time_minutes=30,
                method=DiffusionMethod.CONSTANT_SOURCE,
                # Missing surface_conc
                background=1e15
            )

    def test_missing_dose_fails(self):
        """Test that limited_source without dose fails."""
        with pytest.raises(ValidationError):
            DiffusionRequest(
                dopant=DopantType.PHOSPHORUS,
                temp_celsius=950,
                time_minutes=60,
                method=DiffusionMethod.LIMITED_SOURCE,
                # Missing dose
                background=1e15
            )

    def test_temperature_bounds(self):
        """Test temperature validation bounds."""
        # Too low
        with pytest.raises(ValidationError):
            DiffusionRequest(
                dopant=DopantType.BORON,
                temp_celsius=600,  # Below 700
                time_minutes=30,
                method=DiffusionMethod.CONSTANT_SOURCE,
                surface_conc=1e19
            )

        # Too high
        with pytest.raises(ValidationError):
            DiffusionRequest(
                dopant=DopantType.BORON,
                temp_celsius=1400,  # Above 1300
                time_minutes=30,
                method=DiffusionMethod.CONSTANT_SOURCE,
                surface_conc=1e19
            )

    def test_negative_time_fails(self):
        """Test that negative time fails validation."""
        with pytest.raises(ValidationError):
            DiffusionRequest(
                dopant=DopantType.BORON,
                temp_celsius=1000,
                time_minutes=-10,  # Negative
                method=DiffusionMethod.CONSTANT_SOURCE,
                surface_conc=1e19
            )

    def test_diffusion_response(self):
        """Test diffusion response model."""
        resp = DiffusionResponse(
            depth_nm=[0, 50, 100, 150, 200],
            concentration=[1e19, 8e18, 5e18, 2e18, 1e18],
            junction_depth_nm=250.5,
            sheet_resistance_ohm_sq=45.2,
            solver="erfc",
            computation_time_ms=2.3
        )

        assert len(resp.depth_nm) == 5
        assert resp.junction_depth_nm == 250.5


class TestOxidationSchemas:
    """Test oxidation request/response models."""

    def test_valid_dry_oxidation(self):
        """Test valid dry oxidation request."""
        req = OxidationRequest(
            temp_celsius=1000,
            time_hours=2.0,
            ambient=AmbientType.DRY,
            pressure=1.0
        )

        assert req.ambient == AmbientType.DRY
        assert req.time_hours == 2.0

    def test_valid_wet_oxidation(self):
        """Test valid wet oxidation request."""
        req = OxidationRequest(
            temp_celsius=1100,
            time_hours=1.0,
            ambient=AmbientType.WET,
            pressure=1.0,
            initial_thickness_nm=5.0
        )

        assert req.ambient == AmbientType.WET
        assert req.initial_thickness_nm == 5.0

    def test_oxidation_response(self):
        """Test oxidation response model."""
        resp = OxidationResponse(
            final_thickness_nm=125.3,
            growth_thickness_nm=120.3,
            growth_rate_nm_hr=15.2,
            B_parabolic_nm2_hr=2.5e5,
            A_linear_nm=25.0
        )

        assert resp.final_thickness_nm == 125.3
        assert resp.growth_thickness_nm == 120.3


class TestSPCSchemas:
    """Test SPC request/response models."""

    def test_valid_spc_request(self):
        """Test valid SPC request."""
        points = [
            TimeSeriesPoint(timestamp=datetime(2025, 1, 1, i), value=100.0 + i)
            for i in range(10)
        ]

        req = SPCRequest(
            data=points,
            methods=['rules', 'ewma']
        )

        assert len(req.data) == 10
        assert len(req.methods) == 2

    def test_min_data_points(self):
        """Test that minimum 2 data points required."""
        with pytest.raises(ValidationError):
            SPCRequest(
                data=[TimeSeriesPoint(timestamp=datetime.now(), value=100.0)],
                methods=['rules']
            )

    def test_rule_violation_detail(self):
        """Test rule violation detail model."""
        violation = RuleViolationDetail(
            rule=SPCRuleType.RULE_1,
            index=15,
            timestamp=datetime(2025, 1, 1),
            severity=SPCSeverity.CRITICAL,
            description="Point beyond 3σ limit",
            affected_indices=[15],
            metric_value=125.0
        )

        assert violation.rule == SPCRuleType.RULE_1
        assert violation.severity == SPCSeverity.CRITICAL

    def test_spc_response(self):
        """Test SPC response model."""
        resp = SPCResponse(
            summary={
                'n_violations': 2,
                'mean': 100.5,
                'std': 5.2
            },
            violations=[
                RuleViolationDetail(
                    rule=SPCRuleType.RULE_1,
                    index=15,
                    severity=SPCSeverity.CRITICAL,
                    description="Test violation",
                    affected_indices=[15],
                    metric_value=125.0
                )
            ]
        )

        assert resp.summary['n_violations'] == 2
        assert len(resp.violations) == 1


class TestVMSchemas:
    """Test virtual metrology schemas."""

    def test_valid_vm_request(self):
        """Test valid VM request."""
        req = VMRequest(
            features={
                'temp_mean': 1000.0,
                'temp_std': 5.2,
                'time_minutes': 30.0
            },
            model_type=VMModelType.RANDOM_FOREST,
            return_uncertainty=True
        )

        assert len(req.features) == 3
        assert req.return_uncertainty is True

    def test_vm_response(self):
        """Test VM response."""
        resp = VMResponse(
            prediction=125.3,
            uncertainty=2.5,
            model_used="random_forest",
            confidence_interval=[120.3, 130.3]
        )

        assert resp.prediction == 125.3
        assert resp.uncertainty == 2.5


class TestCalibrationSchemas:
    """Test calibration schemas."""

    def test_valid_diffusion_calibration(self):
        """Test valid diffusion calibration request."""
        req = CalibrationRequest(
            x_data=[0, 50, 100, 150, 200],
            y_data=[1e19, 8e18, 5e18, 2e18, 1e18],
            model_type="diffusion",
            temp_celsius=1000,
            time_minutes=30,
            dopant=DopantType.BORON,
            method=CalibrationMethod.LEAST_SQUARES
        )

        assert len(req.x_data) == 5
        assert len(req.y_data) == 5
        assert req.model_type == "diffusion"

    def test_valid_oxidation_calibration(self):
        """Test valid oxidation calibration request."""
        req = CalibrationRequest(
            x_data=[1.0, 2.0, 3.0],
            y_data=[50.0, 80.0, 100.0],
            model_type="oxidation",
            temp_celsius=1000,
            time_hours=2.0,
            ambient=AmbientType.DRY,
            method=CalibrationMethod.MCMC,
            n_samples=500
        )

        assert req.model_type == "oxidation"
        assert req.n_samples == 500

    def test_unequal_length_fails(self):
        """Test that x and y data must have equal length."""
        with pytest.raises(ValidationError):
            CalibrationRequest(
                x_data=[0, 50, 100],
                y_data=[1e19, 8e18],  # Different length
                model_type="diffusion",
                temp_celsius=1000,
                time_minutes=30,
                dopant=DopantType.BORON
            )

    def test_min_data_points(self):
        """Test minimum 3 data points required."""
        with pytest.raises(ValidationError):
            CalibrationRequest(
                x_data=[0, 50],  # Only 2 points
                y_data=[1e19, 8e18],
                model_type="diffusion",
                temp_celsius=1000,
                time_minutes=30,
                dopant=DopantType.BORON
            )

    def test_calibration_response(self):
        """Test calibration response."""
        resp = CalibrationResponse(
            parameters={'D0': 0.75, 'Ea': 3.68},
            uncertainties={'D0': [0.65, 0.85], 'Ea': [3.60, 3.76]},
            method="least_squares",
            r_squared=0.98,
            rmse=0.05
        )

        assert resp.parameters['D0'] == 0.75
        assert resp.r_squared == 0.98


class TestCommonSchemas:
    """Test common schemas."""

    def test_error_response(self):
        """Test error response model."""
        from session10.api.schemas import ErrorDetail

        resp = ErrorResponse(
            error="ValidationError",
            details=[
                ErrorDetail(
                    message="Invalid temperature",
                    field="temp_celsius",
                    code="VALUE_ERROR"
                )
            ]
        )

        assert resp.error == "ValidationError"
        assert len(resp.details) == 1
        assert resp.details[0].field == "temp_celsius"

    def test_status_response(self):
        """Test status response model."""
        resp = StatusResponse(
            status="ok",
            message="System operational"
        )

        assert resp.status == "ok"
        assert resp.message == "System operational"


class TestEnumValidation:
    """Test enum validation."""

    def test_dopant_enums(self):
        """Test dopant type enums."""
        assert DopantType.BORON.value == "boron"
        assert DopantType.B.value == "B"

        # Test enum in request
        req = DiffusionRequest(
            dopant="boron",  # String automatically converted
            temp_celsius=1000,
            time_minutes=30,
            method="constant_source",
            surface_conc=1e19
        )
        assert isinstance(req.dopant, DopantType)

    def test_ambient_enums(self):
        """Test ambient type enums."""
        assert AmbientType.DRY.value == "dry"
        assert AmbientType.WET.value == "wet"

    def test_invalid_enum_fails(self):
        """Test that invalid enum value fails."""
        with pytest.raises(ValidationError):
            DiffusionRequest(
                dopant="invalid_dopant",  # Invalid
                temp_celsius=1000,
                time_minutes=30,
                method="constant_source",
                surface_conc=1e19
            )


class TestJSONSchemaExamples:
    """Test that JSON schema examples are valid."""

    def test_diffusion_example(self):
        """Test diffusion example from schema."""
        example = DiffusionRequest.model_config['json_schema_extra']['example']

        # Should be valid
        req = DiffusionRequest(**example)
        assert req.dopant == "boron"

    def test_oxidation_example(self):
        """Test oxidation example from schema."""
        example = OxidationRequest.model_config['json_schema_extra']['example']

        req = OxidationRequest(**example)
        assert req.ambient == "dry"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
