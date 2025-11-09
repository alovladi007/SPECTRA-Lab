"""
Production-Grade API Schemas - Session 10

Strong Pydantic models with comprehensive validation for all API endpoints.
Includes request/response models for:
- Diffusion simulation
- Oxidation simulation
- SPC monitoring
- Virtual metrology
- Calibration

Status: PRODUCTION READY ✅
"""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime
from enum import Enum


# ============================================================================
# Common Models
# ============================================================================

class ErrorDetail(BaseModel):
    """Error detail for API responses."""
    message: str = Field(..., description="Error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    code: Optional[str] = Field(None, description="Error code")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    details: List[ErrorDetail] = Field(default_factory=list, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class StatusResponse(BaseModel):
    """API status response."""
    status: Literal["ok", "error"] = Field(..., description="Status")
    message: str = Field(..., description="Status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# Diffusion Models
# ============================================================================

class DopantType(str, Enum):
    """Supported dopant types."""
    BORON = "boron"
    B = "B"
    PHOSPHORUS = "phosphorus"
    P = "P"
    ARSENIC = "arsenic"
    AS = "As"
    ANTIMONY = "antimony"
    SB = "Sb"


class DiffusionMethod(str, Enum):
    """Diffusion profile calculation method."""
    CONSTANT_SOURCE = "constant_source"
    LIMITED_SOURCE = "limited_source"


class SolverType(str, Enum):
    """Solver type for diffusion simulation."""
    ERFC = "erfc"
    NUMERICAL = "numerical"


class DiffusionRequest(BaseModel):
    """Request for diffusion profile simulation."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "dopant": "boron",
            "temp_celsius": 1000,
            "time_minutes": 30,
            "method": "constant_source",
            "surface_conc": 1e19,
            "background": 1e15,
            "depth_nm": [0, 50, 100, 150, 200, 250, 300]
        }
    })

    dopant: DopantType = Field(..., description="Dopant species")
    temp_celsius: float = Field(..., ge=700, le=1300, description="Temperature (°C)")
    time_minutes: float = Field(..., gt=0, le=1000, description="Diffusion time (minutes)")
    method: DiffusionMethod = Field(..., description="Diffusion method")

    # Method-specific parameters
    surface_conc: Optional[float] = Field(None, gt=0, description="Surface concentration (cm^-3) for constant_source")
    dose: Optional[float] = Field(None, gt=0, description="Dose (cm^-2) for limited_source")

    background: float = Field(1e15, gt=0, description="Background doping (cm^-3)")
    depth_nm: List[float] = Field(default_factory=lambda: list(range(0, 501, 10)), description="Depth points (nm)")
    solver: SolverType = Field(SolverType.ERFC, description="Solver type")

    @field_validator('surface_conc', 'dose')
    @classmethod
    def validate_method_params(cls, v, info):
        """Validate method-specific parameters."""
        if info.data.get('method') == DiffusionMethod.CONSTANT_SOURCE:
            if info.field_name == 'surface_conc' and v is None:
                raise ValueError("surface_conc required for constant_source method")
        elif info.data.get('method') == DiffusionMethod.LIMITED_SOURCE:
            if info.field_name == 'dose' and v is None:
                raise ValueError("dose required for limited_source method")
        return v


class DiffusionResponse(BaseModel):
    """Response from diffusion profile simulation."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "depth_nm": [0, 50, 100, 150, 200],
            "concentration": [1e19, 8e18, 5e18, 2e18, 1e18],
            "junction_depth_nm": 250.5,
            "sheet_resistance_ohm_sq": 45.2,
            "solver": "erfc",
            "computation_time_ms": 2.3
        }
    })

    depth_nm: List[float] = Field(..., description="Depth values (nm)")
    concentration: List[float] = Field(..., description="Concentration profile (cm^-3)")
    junction_depth_nm: float = Field(..., description="Junction depth (nm)")
    sheet_resistance_ohm_sq: Optional[float] = Field(None, description="Sheet resistance (Ω/sq)")
    solver: str = Field(..., description="Solver used")
    computation_time_ms: Optional[float] = Field(None, description="Computation time (ms)")


# ============================================================================
# Oxidation Models
# ============================================================================

class AmbientType(str, Enum):
    """Oxidation ambient."""
    DRY = "dry"
    WET = "wet"


class OxidationRequest(BaseModel):
    """Request for oxidation simulation."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "temp_celsius": 1000,
            "time_hours": 2.0,
            "ambient": "dry",
            "pressure": 1.0,
            "initial_thickness_nm": 5.0
        }
    })

    temp_celsius: float = Field(..., ge=700, le=1300, description="Temperature (°C)")
    time_hours: float = Field(..., gt=0, le=100, description="Oxidation time (hours)")
    ambient: AmbientType = Field(..., description="Oxidation ambient")
    pressure: float = Field(1.0, gt=0, le=10, description="Partial pressure (atm)")
    initial_thickness_nm: float = Field(0.0, ge=0, description="Initial oxide thickness (nm)")


class OxidationResponse(BaseModel):
    """Response from oxidation simulation."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "final_thickness_nm": 125.3,
            "growth_thickness_nm": 120.3,
            "growth_rate_nm_hr": 15.2,
            "B_parabolic_nm2_hr": 2.5e5,
            "A_linear_nm": 25.0
        }
    })

    final_thickness_nm: float = Field(..., description="Final oxide thickness (nm)")
    growth_thickness_nm: float = Field(..., description="Growth thickness (nm)")
    growth_rate_nm_hr: float = Field(..., description="Growth rate (nm/hr)")
    B_parabolic_nm2_hr: float = Field(..., description="Parabolic rate constant (nm²/hr)")
    A_linear_nm: float = Field(..., description="Linear rate constant (nm)")


# ============================================================================
# SPC Models
# ============================================================================

class SPCMethod(str, Enum):
    """SPC monitoring method."""
    RULES = "rules"
    EWMA = "ewma"
    CUSUM = "cusum"
    BOCPD = "bocpd"


class SPCRuleType(str, Enum):
    """Western Electric & Nelson rule types."""
    RULE_1 = "RULE_1"
    RULE_2 = "RULE_2"
    RULE_3 = "RULE_3"
    RULE_4 = "RULE_4"
    RULE_5 = "RULE_5"
    RULE_6 = "RULE_6"
    RULE_7 = "RULE_7"
    RULE_8 = "RULE_8"


class SPCSeverity(str, Enum):
    """Violation severity."""
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


class TimeSeriesPoint(BaseModel):
    """Single time series data point."""
    timestamp: datetime = Field(..., description="Timestamp")
    value: float = Field(..., description="Metric value")


class SPCRequest(BaseModel):
    """Request for SPC monitoring."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "data": [
                {"timestamp": "2025-01-01T00:00:00", "value": 100.0},
                {"timestamp": "2025-01-01T01:00:00", "value": 102.0},
                {"timestamp": "2025-01-01T02:00:00", "value": 98.5}
            ],
            "methods": ["rules", "ewma"],
            "enabled_rules": ["RULE_1", "RULE_2"]
        }
    })

    data: List[TimeSeriesPoint] = Field(..., min_length=2, description="Time series data")
    methods: List[SPCMethod] = Field(default_factory=lambda: [SPCMethod.RULES], description="SPC methods to apply")
    enabled_rules: Optional[List[SPCRuleType]] = Field(None, description="Specific rules to check (None = all)")
    ewma_lambda: Optional[float] = Field(None, ge=0, le=1, description="EWMA smoothing parameter")
    cusum_threshold: Optional[float] = Field(None, gt=0, description="CUSUM threshold (h)")


class RuleViolationDetail(BaseModel):
    """Details of a rule violation."""
    rule: SPCRuleType = Field(..., description="Rule violated")
    index: int = Field(..., description="Index of violation")
    timestamp: Optional[datetime] = Field(None, description="Timestamp of violation")
    severity: SPCSeverity = Field(..., description="Severity level")
    description: str = Field(..., description="Human-readable description")
    affected_indices: List[int] = Field(..., description="Indices involved")
    metric_value: float = Field(..., description="Metric value at violation")


class ChangePointDetail(BaseModel):
    """Details of a detected change point."""
    index: int = Field(..., description="Index of change point")
    timestamp: Optional[datetime] = Field(None, description="Timestamp")
    probability: float = Field(..., ge=0, le=1, description="Change point probability")
    run_length: Optional[int] = Field(None, description="Run length")
    description: str = Field(..., description="Description")


class SPCResponse(BaseModel):
    """Response from SPC monitoring."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "summary": {
                "n_violations": 2,
                "n_changepoints": 1,
                "mean": 100.5,
                "std": 5.2,
                "process_in_control": False
            },
            "violations": [
                {
                    "rule": "RULE_1",
                    "index": 15,
                    "severity": "CRITICAL",
                    "description": "Point beyond 3σ limit",
                    "affected_indices": [15],
                    "metric_value": 125.0
                }
            ],
            "changepoints": []
        }
    })

    summary: Dict[str, Any] = Field(..., description="Summary statistics")
    violations: List[RuleViolationDetail] = Field(default_factory=list, description="Detected violations")
    changepoints: List[ChangePointDetail] = Field(default_factory=list, description="Detected change points")


# ============================================================================
# Virtual Metrology Models
# ============================================================================

class VMModelType(str, Enum):
    """Virtual metrology model type."""
    LINEAR = "linear"
    RIDGE = "ridge"
    LASSO = "lasso"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"


class VMRequest(BaseModel):
    """Request for virtual metrology prediction."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "features": {
                "temp_mean": 1000.0,
                "temp_std": 5.2,
                "time_minutes": 30.0,
                "pressure_mbar": 100.0
            },
            "model_type": "random_forest",
            "return_uncertainty": True
        }
    })

    features: Dict[str, float] = Field(..., description="FDC features")
    model_type: Optional[VMModelType] = Field(None, description="Model type (None = best available)")
    return_uncertainty: bool = Field(False, description="Return prediction uncertainty")


class VMResponse(BaseModel):
    """Response from virtual metrology prediction."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "prediction": 125.3,
            "uncertainty": 2.5,
            "model_used": "random_forest",
            "confidence_interval": [120.3, 130.3],
            "feature_importance": {
                "temp_mean": 0.45,
                "time_minutes": 0.30,
                "pressure_mbar": 0.15
            }
        }
    })

    prediction: float = Field(..., description="Predicted value")
    uncertainty: Optional[float] = Field(None, description="Prediction uncertainty (std)")
    model_used: str = Field(..., description="Model used for prediction")
    confidence_interval: Optional[List[float]] = Field(None, description="95% confidence interval")
    feature_importance: Optional[Dict[str, float]] = Field(None, description="Feature importance scores")


# ============================================================================
# Calibration Models
# ============================================================================

class CalibrationMethod(str, Enum):
    """Calibration method."""
    LEAST_SQUARES = "least_squares"
    MCMC = "mcmc"


class CalibrationRequest(BaseModel):
    """Request for parameter calibration."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "x_data": [0, 50, 100, 150, 200],
            "y_data": [1e19, 8e18, 5e18, 2e18, 1e18],
            "model_type": "diffusion",
            "temp_celsius": 1000,
            "time_minutes": 30,
            "dopant": "boron",
            "method": "least_squares"
        }
    })

    x_data: List[float] = Field(..., min_length=3, description="Independent variable (e.g., depth)")
    y_data: List[float] = Field(..., min_length=3, description="Dependent variable (e.g., concentration)")
    model_type: Literal["diffusion", "oxidation"] = Field(..., description="Physical model type")

    # Context parameters
    temp_celsius: float = Field(..., ge=700, le=1300, description="Temperature (°C)")
    time_minutes: Optional[float] = Field(None, gt=0, description="Time (minutes) for diffusion")
    time_hours: Optional[float] = Field(None, gt=0, description="Time (hours) for oxidation")
    dopant: Optional[DopantType] = Field(None, description="Dopant (for diffusion)")
    ambient: Optional[AmbientType] = Field(None, description="Ambient (for oxidation)")

    # Method options
    method: CalibrationMethod = Field(CalibrationMethod.LEAST_SQUARES, description="Calibration method")
    n_samples: int = Field(1000, gt=100, le=10000, description="MCMC samples")

    @field_validator('y_data')
    @classmethod
    def validate_equal_length(cls, v, info):
        """Validate x and y have equal length."""
        x_data = info.data.get('x_data')
        if x_data and len(v) != len(x_data):
            raise ValueError("x_data and y_data must have equal length")
        return v


class CalibrationResponse(BaseModel):
    """Response from calibration."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "parameters": {
                "D0": 0.75,
                "Ea": 3.68
            },
            "uncertainties": {
                "D0": [0.65, 0.85],
                "Ea": [3.60, 3.76]
            },
            "method": "least_squares",
            "r_squared": 0.98,
            "rmse": 0.05
        }
    })

    parameters: Dict[str, float] = Field(..., description="Calibrated parameters")
    uncertainties: Dict[str, List[float]] = Field(..., description="Parameter uncertainties (95% CI)")
    method: str = Field(..., description="Calibration method used")
    r_squared: Optional[float] = Field(None, ge=0, le=1, description="R² goodness of fit")
    rmse: Optional[float] = Field(None, ge=0, description="Root mean square error")
    convergence: Optional[bool] = Field(None, description="Convergence status")


# ============================================================================
# Batch Operation Models
# ============================================================================

class BatchDiffusionRequest(BaseModel):
    """Request for batch diffusion simulations."""
    runs: List[DiffusionRequest] = Field(..., min_length=1, max_length=1000, description="List of diffusion runs")


class BatchDiffusionResponse(BaseModel):
    """Response from batch diffusion simulations."""
    results: List[DiffusionResponse] = Field(..., description="Simulation results")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")


class BatchOxidationRequest(BaseModel):
    """Request for batch oxidation simulations."""
    runs: List[OxidationRequest] = Field(..., min_length=1, max_length=1000, description="List of oxidation runs")


class BatchOxidationResponse(BaseModel):
    """Response from batch oxidation simulations."""
    results: List[OxidationResponse] = Field(..., description="Simulation results")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")


# ============================================================================
# Export all models
# ============================================================================

__all__ = [
    # Common
    "ErrorDetail",
    "ErrorResponse",
    "StatusResponse",
    # Diffusion
    "DopantType",
    "DiffusionMethod",
    "SolverType",
    "DiffusionRequest",
    "DiffusionResponse",
    # Oxidation
    "AmbientType",
    "OxidationRequest",
    "OxidationResponse",
    # SPC
    "SPCMethod",
    "SPCRuleType",
    "SPCSeverity",
    "TimeSeriesPoint",
    "SPCRequest",
    "RuleViolationDetail",
    "ChangePointDetail",
    "SPCResponse",
    # VM
    "VMModelType",
    "VMRequest",
    "VMResponse",
    # Calibration
    "CalibrationMethod",
    "CalibrationRequest",
    "CalibrationResponse",
    # Batch
    "BatchDiffusionRequest",
    "BatchDiffusionResponse",
    "BatchOxidationRequest",
    "BatchOxidationResponse",
]
