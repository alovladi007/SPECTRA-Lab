"""Pydantic schemas for RTP and Ion Implantation modules."""

from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, validator
from enum import Enum


# Enums
class IonSpecies(str, Enum):
    """Common ion species for implantation."""
    BORON = "B"
    PHOSPHORUS = "P"
    ARSENIC = "As"
    ANTIMONY = "Sb"
    GERMANIUM = "Ge"
    SILICON = "Si"
    NITROGEN = "N"
    CARBON = "C"
    FLUORINE = "F"
    HYDROGEN = "H"
    HELIUM = "He"
    ARGON = "Ar"


class ChartType(str, Enum):
    """SPC chart types."""
    XBAR_R = "Xbar-R"
    XBAR_S = "Xbar-S"
    I_MR = "I-MR"
    EWMA = "EWMA"
    CUSUM = "CUSUM"
    P_CHART = "P"
    NP_CHART = "NP"
    C_CHART = "C"
    U_CHART = "U"
    MULTIVARIATE = "T2"


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class ModelType(str, Enum):
    """Virtual Metrology model types."""
    NEURAL = "neural"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "rf"
    LINEAR = "linear"
    PHYSICS = "physics"
    HYBRID = "hybrid"


# Ion Implantation Schemas
class ImplantDoseProfileBase(BaseModel):
    """Base schema for implant dose profiles."""
    ion_species: IonSpecies
    isotope: Optional[int] = None
    energy_keV: float = Field(..., gt=0, le=5000)
    tilt_deg: float = Field(..., ge=-10, le=60)
    twist_deg: float = Field(..., ge=-180, le=180)
    dose_cm2: float = Field(..., gt=0, le=1e18)
    
    @validator('dose_cm2')
    def validate_dose(cls, v):
        if v < 1e10 or v > 1e18:
            raise ValueError('Dose must be between 1e10 and 1e18 ions/cm²')
        return v


class ImplantDoseProfileCreate(ImplantDoseProfileBase):
    """Schema for creating implant dose profile."""
    run_id: int
    projected_range_nm: Optional[float] = None
    straggle_nm: Optional[float] = None
    channeling_metric: Optional[float] = None
    damage_metrics: Optional[Dict[str, Any]] = None
    beam_uniformity: Optional[Dict[str, Any]] = None


class ImplantDoseProfileUpdate(BaseModel):
    """Schema for updating implant dose profile."""
    projected_range_nm: Optional[float] = None
    straggle_nm: Optional[float] = None
    channeling_metric: Optional[float] = None
    damage_metrics: Optional[Dict[str, Any]] = None
    beam_uniformity: Optional[Dict[str, Any]] = None
    wafer_map_uri: Optional[str] = None
    sims_profile_uri: Optional[str] = None


class ImplantDoseProfileResponse(ImplantDoseProfileBase):
    """Response schema for implant dose profile."""
    id: int
    org_id: int
    run_id: int
    projected_range_nm: Optional[float]
    straggle_nm: Optional[float]
    channeling_metric: Optional[float]
    damage_metrics: Optional[Dict[str, Any]]
    beam_uniformity: Optional[Dict[str, Any]]
    wafer_map_uri: Optional[str]
    sims_profile_uri: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


class ImplantTelemetryCreate(BaseModel):
    """Schema for creating implant telemetry records."""
    run_id: int
    beam_current_mA: float = Field(..., ge=0, le=50)
    pressure_mTorr: float = Field(..., ge=0, le=1000)
    accel_voltage_kV: float = Field(..., ge=0, le=500)
    analyzer_magnet_T: Optional[float] = Field(None, ge=0, le=2)
    steering_X: Optional[float] = Field(None, ge=-100, le=100)
    steering_Y: Optional[float] = Field(None, ge=-100, le=100)
    dose_count_C_cm2: float = Field(..., ge=0)
    faraday_currents: Optional[List[float]] = None
    gas_flows: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class ImplantTelemetryResponse(BaseModel):
    """Response schema for implant telemetry."""
    id: int
    run_id: int
    ts: datetime
    beam_current_mA: float
    pressure_mTorr: float
    accel_voltage_kV: float
    analyzer_magnet_T: Optional[float]
    steering_X: Optional[float]
    steering_Y: Optional[float]
    dose_count_C_cm2: float
    beam_profile_uri: Optional[str]
    faraday_currents: Optional[List[float]]
    gas_flows: Optional[Dict[str, float]]
    metadata: Optional[Dict[str, Any]]
    
    model_config = ConfigDict(from_attributes=True)


# RTP Schemas
class RTPSegment(BaseModel):
    """Temperature profile segment."""
    time_s: float = Field(..., ge=0)
    T_C: float = Field(..., ge=20, le=1400)
    ramp_Cps: Optional[float] = Field(None, ge=0, le=500)
    dwell_s: Optional[float] = Field(None, ge=0)


class RTPProfileBase(BaseModel):
    """Base schema for RTP profiles."""
    ambient_gas: str = Field(..., max_length=50)
    pressure_Torr: float = Field(..., ge=0, le=760)
    emissivity: float = Field(..., ge=0.1, le=1.0)
    
    @validator('ambient_gas')
    def validate_gas(cls, v):
        allowed = ['N2', 'O2', 'Ar', 'He', 'H2', 'NH3', 'N2O', 'forming_gas']
        if v not in allowed:
            raise ValueError(f'Gas must be one of {allowed}')
        return v


class RTPProfileCreate(RTPProfileBase):
    """Schema for creating RTP profile."""
    run_id: int
    recipe_curve: List[RTPSegment]
    pyrometer_cal_id: Optional[int] = None
    zone_setpoints: Optional[Dict[str, float]] = None
    wafer_rotation_rpm: Optional[float] = Field(None, ge=0, le=100)


class RTPProfileUpdate(BaseModel):
    """Schema for updating RTP profile."""
    recipe_curve: Optional[List[RTPSegment]] = None
    zone_setpoints: Optional[Dict[str, float]] = None
    uniformity_metrics: Optional[Dict[str, Any]] = None


class RTPProfileResponse(RTPProfileBase):
    """Response schema for RTP profile."""
    id: int
    org_id: int
    run_id: int
    recipe_curve: List[RTPSegment]
    peak_T_C: float
    pyrometer_cal_id: Optional[int]
    zone_setpoints: Optional[Dict[str, float]]
    uniformity_metrics: Optional[Dict[str, Any]]
    wafer_rotation_rpm: Optional[float]
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


class RTPTelemetryCreate(BaseModel):
    """Schema for creating RTP telemetry."""
    run_id: int
    setpoint_T_C: float = Field(..., ge=20, le=1400)
    pyrometer_T_C: float = Field(..., ge=20, le=1400)
    tc_T_C: Optional[List[float]] = None
    lamp_power_pct: List[float] = Field(..., min_items=1)
    emissivity_used: float = Field(..., ge=0.1, le=1.0)
    chamber_pressure_Torr: float = Field(..., ge=0)
    flow_sccm: Dict[str, float]
    pid_state: Optional[Dict[str, float]] = None
    mpc_state: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('lamp_power_pct')
    def validate_lamp_power(cls, v):
        if any(p < 0 or p > 100 for p in v):
            raise ValueError('Lamp power must be between 0 and 100%')
        return v


class RTPTelemetryResponse(BaseModel):
    """Response schema for RTP telemetry."""
    id: int
    run_id: int
    ts: datetime
    setpoint_T_C: float
    pyrometer_T_C: float
    tc_T_C: Optional[List[float]]
    lamp_power_pct: List[float]
    emissivity_used: float
    chamber_pressure_Torr: float
    flow_sccm: Dict[str, float]
    pid_state: Optional[Dict[str, float]]
    mpc_state: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    
    model_config = ConfigDict(from_attributes=True)


# SPC Schemas
class ControlLimits(BaseModel):
    """Control limits for SPC charts."""
    UCL: float
    CL: float
    LCL: float
    
    @validator('UCL')
    def validate_limits(cls, v, values):
        if 'LCL' in values and v <= values['LCL']:
            raise ValueError('UCL must be greater than LCL')
        return v


class SPCRule(BaseModel):
    """SPC rule configuration."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = {}


class SPCSeriesCreate(BaseModel):
    """Schema for creating SPC series."""
    name: str = Field(..., min_length=1, max_length=200)
    instrument_id: Optional[int] = None
    parameter: str = Field(..., min_length=1, max_length=100)
    chart_type: ChartType
    control_limits: ControlLimits
    spec_limits: Optional[Dict[str, float]] = None
    rules: List[SPCRule]
    window_size: Optional[int] = Field(None, ge=2, le=100)
    ewma_lambda: Optional[float] = Field(None, gt=0, le=1)


class SPCSeriesUpdate(BaseModel):
    """Schema for updating SPC series."""
    name: Optional[str] = None
    control_limits: Optional[ControlLimits] = None
    spec_limits: Optional[Dict[str, float]] = None
    rules: Optional[List[SPCRule]] = None
    active: Optional[bool] = None


class SPCSeriesResponse(BaseModel):
    """Response schema for SPC series."""
    id: int
    org_id: int
    name: str
    instrument_id: Optional[int]
    parameter: str
    chart_type: ChartType
    control_limits: ControlLimits
    spec_limits: Optional[Dict[str, float]]
    rules: List[SPCRule]
    window_size: Optional[int]
    ewma_lambda: Optional[float]
    active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


class SPCPointCreate(BaseModel):
    """Schema for creating SPC point."""
    series_id: int
    run_id: Optional[int] = None
    value: float
    subgroup_values: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None


class SPCPointResponse(BaseModel):
    """Response schema for SPC point."""
    id: int
    series_id: int
    run_id: Optional[int]
    ts: datetime
    value: float
    subgroup_values: Optional[List[float]]
    moving_range: Optional[float]
    ewma_value: Optional[float]
    cusum_pos: Optional[float]
    cusum_neg: Optional[float]
    violations: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    
    model_config = ConfigDict(from_attributes=True)


class SPCAlertCreate(BaseModel):
    """Schema for creating SPC alert."""
    series_id: int
    point_id: int
    alert_type: str = Field(..., max_length=50)
    severity: AlertSeverity
    rule_violated: str = Field(..., max_length=100)
    description: Optional[str] = None


class SPCAlertAcknowledge(BaseModel):
    """Schema for acknowledging SPC alert."""
    resolution_notes: Optional[str] = None


class SPCAlertResponse(BaseModel):
    """Response schema for SPC alert."""
    id: int
    org_id: int
    series_id: int
    point_id: int
    alert_type: str
    severity: AlertSeverity
    rule_violated: str
    description: Optional[str]
    acknowledged: bool
    acknowledged_by: Optional[int]
    acknowledged_at: Optional[datetime]
    resolution_notes: Optional[str]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Virtual Metrology Schemas
class VMFeatureDefinition(BaseModel):
    """Feature definition for VM."""
    name: str
    source: str  # telemetry field
    aggregation: Literal["mean", "std", "min", "max", "median", "range", "slope"]
    window_size: Optional[int] = None
    lag: Optional[int] = None


class VMFeatureSetCreate(BaseModel):
    """Schema for creating VM feature set."""
    name: str = Field(..., min_length=1, max_length=200)
    instrument_id: Optional[int] = None
    features: List[VMFeatureDefinition]
    target_metrics: Dict[str, str]  # metric_name: source
    preprocessing: Optional[Dict[str, Any]] = None


class VMFeatureSetResponse(BaseModel):
    """Response schema for VM feature set."""
    id: int
    org_id: int
    name: str
    instrument_id: Optional[int]
    features: List[VMFeatureDefinition]
    target_metrics: Dict[str, str]
    preprocessing: Optional[Dict[str, Any]]
    active: bool
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


class VMModelCreate(BaseModel):
    """Schema for creating VM model."""
    feature_set_id: int
    name: str = Field(..., min_length=1, max_length=200)
    version: str = Field(..., min_length=1, max_length=50)
    model_type: ModelType
    hyperparameters: Optional[Dict[str, Any]] = None


class VMModelDeploy(BaseModel):
    """Schema for deploying VM model."""
    deployed: bool


class VMModelResponse(BaseModel):
    """Response schema for VM model."""
    id: int
    org_id: int
    feature_set_id: int
    name: str
    version: str
    model_type: ModelType
    model_uri: Optional[str]
    hyperparameters: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, float]]
    training_runs: Optional[List[int]]
    validation_runs: Optional[List[int]]
    deployed: bool
    approved_by: Optional[int]
    approved_at: Optional[datetime]
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


# Batch operation schemas
class TelemetryQuery(BaseModel):
    """Query parameters for telemetry data."""
    run_id: Optional[int] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(1000, ge=1, le=10000)
    offset: int = Field(0, ge=0)


class SPCAnalysis(BaseModel):
    """SPC analysis request."""
    series_id: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    recalculate_limits: bool = False


class VMPrediction(BaseModel):
    """VM prediction request."""
    model_id: int
    run_id: int


class VMPredictionResponse(BaseModel):
    """VM prediction response."""
    run_id: int
    model_id: int
    predictions: Dict[str, float]
    confidence_intervals: Optional[Dict[str, tuple[float, float]]]
    feature_importance: Optional[Dict[str, float]]
    timestamp: datetime
