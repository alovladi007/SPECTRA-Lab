"""SQLAlchemy models for RTP and Ion Implantation modules."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy import (
    Column, BigInteger, Integer, String, Float, Boolean,
    DateTime, Text, ForeignKey, JSON, ARRAY, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship, Mapped, mapped_column
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

from app.core.database import Base


class ImplantDoseProfile(Base):
    """Ion implantation dose profile and characterization data."""
    
    __tablename__ = "implant_dose_profiles"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("runs.id"), nullable=False)
    
    # Ion beam parameters
    ion_species: Mapped[str] = mapped_column(String(10), nullable=False)  # B, P, As, Sb, etc.
    isotope: Mapped[Optional[int]] = mapped_column(Integer)  # e.g., 11 for B-11
    energy_keV: Mapped[float] = mapped_column(Float, nullable=False)
    tilt_deg: Mapped[float] = mapped_column(Float, nullable=False)  # Beam tilt angle
    twist_deg: Mapped[float] = mapped_column(Float, nullable=False)  # Wafer rotation
    dose_cm2: Mapped[float] = mapped_column(Float, nullable=False)  # ions/cm²
    
    # SRIM/TRIM simulation results
    projected_range_nm: Mapped[Optional[float]] = mapped_column(Float)  # Rp
    straggle_nm: Mapped[Optional[float]] = mapped_column(Float)  # ΔRp
    channeling_metric: Mapped[Optional[float]] = mapped_column(Float)
    
    # Extended metrics
    damage_metrics: Mapped[Optional[Dict]] = mapped_column(JSONB)  # DPA, vacancies, etc.
    beam_uniformity: Mapped[Optional[Dict]] = mapped_column(JSONB)  # Spatial uniformity map
    wafer_map_uri: Mapped[Optional[str]] = mapped_column(String(500))
    sims_profile_uri: Mapped[Optional[str]] = mapped_column(String(500))  # SIMS depth profile
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    organization = relationship("Organization", back_populates="implant_profiles")
    run = relationship("Run", back_populates="implant_profile")
    
    __table_args__ = (
        Index('idx_implant_dose_org_run', 'org_id', 'run_id'),
        Index('idx_implant_dose_species', 'ion_species'),
        Index('idx_implant_dose_deleted', 'deleted_at', postgresql_where='deleted_at IS NULL'),
    )


class ImplantTelemetry(Base):
    """Real-time telemetry data from ion implanter."""
    
    __tablename__ = "implant_telemetry"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("runs.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Beam parameters
    beam_current_mA: Mapped[float] = mapped_column(Float, nullable=False)
    pressure_mTorr: Mapped[float] = mapped_column(Float, nullable=False)
    accel_voltage_kV: Mapped[float] = mapped_column(Float, nullable=False)
    analyzer_magnet_T: Mapped[Optional[float]] = mapped_column(Float)  # Mass analyzer
    
    # Beam steering
    steering_X: Mapped[Optional[float]] = mapped_column(Float)
    steering_Y: Mapped[Optional[float]] = mapped_column(Float)
    
    # Dose integration
    dose_count_C_cm2: Mapped[float] = mapped_column(Float, nullable=False)  # Cumulative dose
    
    # Diagnostics
    beam_profile_uri: Mapped[Optional[str]] = mapped_column(String(500))
    faraday_currents: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))  # Multi-cup array
    gas_flows: Mapped[Optional[Dict]] = mapped_column(JSONB)  # Source gas flows
    metadata: Mapped[Optional[Dict]] = mapped_column(JSONB)
    
    # Relationships
    run = relationship("Run", back_populates="implant_telemetry")
    
    __table_args__ = (
        Index('idx_implant_telem_run_ts', 'run_id', 'ts'),
        Index('idx_implant_telem_ts', 'ts'),
    )


class RTPProfile(Base):
    """Rapid Thermal Processing temperature profiles and parameters."""
    
    __tablename__ = "rtp_profiles"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("runs.id"), nullable=False)
    
    # Temperature profile
    recipe_curve: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # Ramp/soak segments
    peak_T_C: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Process parameters
    ambient_gas: Mapped[str] = mapped_column(String(50), nullable=False)  # N2, O2, NH3, etc.
    pressure_Torr: Mapped[float] = mapped_column(Float, nullable=False)
    emissivity: Mapped[float] = mapped_column(Float, nullable=False)  # Wafer emissivity
    pyrometer_cal_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("calibrations.id"))
    
    # Multi-zone control
    zone_setpoints: Mapped[Optional[Dict]] = mapped_column(JSONB)  # Per-zone lamp settings
    uniformity_metrics: Mapped[Optional[Dict]] = mapped_column(JSONB)
    wafer_rotation_rpm: Mapped[Optional[float]] = mapped_column(Float)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Relationships
    organization = relationship("Organization", back_populates="rtp_profiles")
    run = relationship("Run", back_populates="rtp_profile")
    calibration = relationship("Calibration", back_populates="rtp_profiles")
    
    __table_args__ = (
        Index('idx_rtp_profile_org_run', 'org_id', 'run_id'),
        Index('idx_rtp_profile_deleted', 'deleted_at', postgresql_where='deleted_at IS NULL'),
        Index('idx_rtp_recipe_gin', 'recipe_curve', postgresql_using='gin'),
    )


class RTPTelemetry(Base):
    """Real-time telemetry from RTP system."""
    
    __tablename__ = "rtp_telemetry"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("runs.id"), nullable=False)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Temperature measurements
    setpoint_T_C: Mapped[float] = mapped_column(Float, nullable=False)
    pyrometer_T_C: Mapped[float] = mapped_column(Float, nullable=False)  # Primary measurement
    tc_T_C: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))  # Thermocouple array
    
    # Lamp control
    lamp_power_pct: Mapped[List[float]] = mapped_column(ARRAY(Float), nullable=False)  # Per-zone
    
    # Process parameters
    emissivity_used: Mapped[float] = mapped_column(Float, nullable=False)
    chamber_pressure_Torr: Mapped[float] = mapped_column(Float, nullable=False)
    flow_sccm: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # Gas flows
    
    # Controller states
    pid_state: Mapped[Optional[Dict]] = mapped_column(JSONB)  # P, I, D, error, output
    mpc_state: Mapped[Optional[Dict]] = mapped_column(JSONB)  # MPC state vector
    metadata: Mapped[Optional[Dict]] = mapped_column(JSONB)
    
    # Relationships
    run = relationship("Run", back_populates="rtp_telemetry")
    
    __table_args__ = (
        Index('idx_rtp_telem_run_ts', 'run_id', 'ts'),
        Index('idx_rtp_telem_ts', 'ts'),
        Index('idx_rtp_flow_gin', 'flow_sccm', postgresql_using='gin'),
    )


class SPCSeries(Base):
    """Statistical Process Control chart configuration."""
    
    __tablename__ = "spc_series"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    instrument_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("instruments.id"))
    
    # Chart configuration
    parameter: Mapped[str] = mapped_column(String(100), nullable=False)
    chart_type: Mapped[str] = mapped_column(String(20), nullable=False)  # Xbar-R, EWMA, CUSUM
    control_limits: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # UCL, CL, LCL
    spec_limits: Mapped[Optional[Dict]] = mapped_column(JSONB)  # USL, LSL
    rules: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # Western Electric rules
    
    # Algorithm parameters
    window_size: Mapped[Optional[int]] = mapped_column(Integer)  # For moving window
    ewma_lambda: Mapped[Optional[float]] = mapped_column(Float)  # EWMA smoothing
    
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="spc_series")
    instrument = relationship("Instrument", back_populates="spc_series")
    points = relationship("SPCPoint", back_populates="series", cascade="all, delete-orphan")
    alerts = relationship("SPCAlert", back_populates="series")
    
    __table_args__ = (
        Index('idx_spc_series_org', 'org_id'),
        Index('idx_spc_series_inst', 'instrument_id'),
        Index('idx_spc_series_active', 'active'),
    )


class SPCPoint(Base):
    """Individual SPC data points."""
    
    __tablename__ = "spc_points"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    series_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("spc_series.id"), nullable=False)
    run_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("runs.id"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    
    # Measurements
    value: Mapped[float] = mapped_column(Float, nullable=False)
    subgroup_values: Mapped[Optional[List[float]]] = mapped_column(ARRAY(Float))
    
    # Calculated values
    moving_range: Mapped[Optional[float]] = mapped_column(Float)
    ewma_value: Mapped[Optional[float]] = mapped_column(Float)
    cusum_pos: Mapped[Optional[float]] = mapped_column(Float)
    cusum_neg: Mapped[Optional[float]] = mapped_column(Float)
    
    # Rule violations
    violations: Mapped[Optional[Dict]] = mapped_column(JSONB)
    metadata: Mapped[Optional[Dict]] = mapped_column(JSONB)
    
    # Relationships
    series = relationship("SPCSeries", back_populates="points")
    run = relationship("Run", back_populates="spc_points")
    alerts = relationship("SPCAlert", back_populates="point")
    
    __table_args__ = (
        Index('idx_spc_points_series_ts', 'series_id', 'ts'),
        Index('idx_spc_points_run', 'run_id'),
    )


class SPCAlert(Base):
    """SPC rule violations and alerts."""
    
    __tablename__ = "spc_alerts"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    series_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("spc_series.id"), nullable=False)
    point_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("spc_points.id"), nullable=False)
    
    # Alert details
    alert_type: Mapped[str] = mapped_column(String(50), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)  # warning, critical
    rule_violated: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    
    # Acknowledgment
    acknowledged: Mapped[bool] = mapped_column(Boolean, default=False)
    acknowledged_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="spc_alerts")
    series = relationship("SPCSeries", back_populates="alerts")
    point = relationship("SPCPoint", back_populates="alerts")
    acknowledged_user = relationship("User", back_populates="spc_acknowledgments")
    
    __table_args__ = (
        Index('idx_spc_alerts_org', 'org_id'),
        Index('idx_spc_alerts_series', 'series_id'),
        Index('idx_spc_alerts_ack', 'acknowledged'),
    )


class VMFeatureSet(Base):
    """Virtual Metrology feature engineering configuration."""
    
    __tablename__ = "vm_feature_sets"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    instrument_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("instruments.id"))
    
    # Feature engineering
    features: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # Feature definitions
    target_metrics: Mapped[Dict] = mapped_column(JSONB, nullable=False)  # What to predict
    preprocessing: Mapped[Optional[Dict]] = mapped_column(JSONB)  # Scaling, transforms
    
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="vm_feature_sets")
    instrument = relationship("Instrument", back_populates="vm_feature_sets")
    models = relationship("VMModel", back_populates="feature_set")
    
    __table_args__ = (
        Index('idx_vm_features_org', 'org_id'),
    )


class VMModel(Base):
    """Virtual Metrology predictive models."""
    
    __tablename__ = "vm_models"
    
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    org_id: Mapped[int] = mapped_column(Integer, ForeignKey("organizations.id"), nullable=False)
    feature_set_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("vm_feature_sets.id"), nullable=False)
    
    # Model identification
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)  # neural, xgboost, physics
    
    # Model storage
    model_uri: Mapped[Optional[str]] = mapped_column(String(500))  # S3/Minio path
    hyperparameters: Mapped[Optional[Dict]] = mapped_column(JSONB)
    performance_metrics: Mapped[Optional[Dict]] = mapped_column(JSONB)  # R2, RMSE, MAE
    
    # Training data
    training_runs: Mapped[Optional[List[int]]] = mapped_column(ARRAY(BigInteger))
    validation_runs: Mapped[Optional[List[int]]] = mapped_column(ARRAY(BigInteger))
    
    # Deployment status
    deployed: Mapped[bool] = mapped_column(Boolean, default=False)
    approved_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    approved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="vm_models")
    feature_set = relationship("VMFeatureSet", back_populates="models")
    approved_user = relationship("User", back_populates="vm_approvals")
    
    __table_args__ = (
        Index('idx_vm_models_org', 'org_id'),
        Index('idx_vm_models_deployed', 'deployed'),
    )
