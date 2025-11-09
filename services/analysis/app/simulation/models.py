"""
SQLAlchemy Database Models for Diffusion Module
Provides ORM models for simulation audit trails, batch jobs, KPIs, and maintenance
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
import uuid

Base = declarative_base()


class SimulationAudit(Base):
    """Audit trail for all simulation runs"""
    __tablename__ = 'simulation_audit'

    id = Column(Integer, primary_key=True)
    simulation_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    run_id = Column(String(100))
    recipe_id = Column(String(100))
    simulation_type = Column(String(50), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))

    user_id = Column(String(100))
    user_email = Column(String(255))

    parameters = Column(JSONB, nullable=False)
    results = Column(JSONB)

    status = Column(String(20), default='pending')
    execution_time_ms = Column(Integer)
    memory_mb = Column(Float)

    error_message = Column(Text)
    error_traceback = Column(Text)

    metadata = Column(JSONB)
    tags = Column(ARRAY(Text))

    git_commit = Column(String(40))
    module_version = Column(String(20))

    def __repr__(self):
        return f"<SimulationAudit(id={self.simulation_id}, type={self.simulation_type}, status={self.status})>"


class BatchJob(Base):
    """Batch job management for multiple simulations"""
    __tablename__ = 'batch_jobs'

    id = Column(Integer, primary_key=True)
    job_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    job_name = Column(String(255))

    simulation_type = Column(String(50), nullable=False)
    total_simulations = Column(Integer, nullable=False)
    completed_simulations = Column(Integer, default=0)
    failed_simulations = Column(Integer, default=0)

    status = Column(String(20), default='queued')

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))

    user_id = Column(String(100))
    config = Column(JSONB)
    results_summary = Column(JSONB)
    error_message = Column(Text)

    def __repr__(self):
        return f"<BatchJob(id={self.job_id}, status={self.status}, progress={self.completed_simulations}/{self.total_simulations})>"


class KPIMeasurement(Base):
    """KPI measurements from simulations"""
    __tablename__ = 'kpi_measurements'

    id = Column(Integer, primary_key=True)
    measurement_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)
    simulation_id = Column(UUID(as_uuid=True))

    measured_at = Column(DateTime(timezone=True), server_default=func.now())

    kpi_name = Column(String(100), nullable=False)
    kpi_value = Column(Float, nullable=False)
    kpi_unit = Column(String(50))

    target_value = Column(Float)
    ucl = Column(Float)
    lcl = Column(Float)
    usl = Column(Float)
    lsl = Column(Float)

    within_control = Column(Boolean)
    within_spec = Column(Boolean)

    tool_id = Column(String(100))
    recipe_id = Column(String(100))
    wafer_id = Column(String(100))
    lot_id = Column(String(100))

    metadata = Column(JSONB)

    def __repr__(self):
        return f"<KPIMeasurement(kpi={self.kpi_name}, value={self.kpi_value}, within_control={self.within_control})>"


class SPCViolation(Base):
    """SPC rule violations detected"""
    __tablename__ = 'spc_violations'

    id = Column(Integer, primary_key=True)
    violation_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)

    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    rule_id = Column(String(50), nullable=False)
    rule_name = Column(String(255), nullable=False)
    severity = Column(String(20))

    z_score = Column(Float)
    window_indices = Column(ARRAY(Integer))

    kpi_name = Column(String(100))
    tool_id = Column(String(100))
    recipe_id = Column(String(100))

    suggested_causes = Column(ARRAY(Text))
    recommended_actions = Column(ARRAY(Text))

    acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime(timezone=True))

    resolved = Column(Boolean, default=False)
    resolved_by = Column(String(100))
    resolved_at = Column(DateTime(timezone=True))
    resolution_notes = Column(Text)

    def __repr__(self):
        return f"<SPCViolation(rule={self.rule_id}, severity={self.severity}, resolved={self.resolved})>"


class MaintenanceRecommendation(Base):
    """Predictive maintenance recommendations"""
    __tablename__ = 'maintenance_recommendations'

    id = Column(Integer, primary_key=True)
    recommendation_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    tool_id = Column(String(100), nullable=False)
    component = Column(String(255))

    action = Column(String(255), nullable=False)
    urgency = Column(String(20))

    confidence = Column(Float)
    estimated_downtime_hours = Column(Float)

    tool_health_score = Column(Float)
    failure_probability = Column(Float)

    supporting_data = Column(JSONB)

    status = Column(String(20), default='open')
    scheduled_date = Column(DateTime(timezone=True))
    completed_date = Column(DateTime(timezone=True))

    assigned_to = Column(String(100))
    notes = Column(Text)

    def __repr__(self):
        return f"<MaintenanceRecommendation(tool={self.tool_id}, urgency={self.urgency}, status={self.status})>"


class CalibrationResult(Base):
    """Model calibration results"""
    __tablename__ = 'calibration_results'

    id = Column(Integer, primary_key=True)
    calibration_id = Column(UUID(as_uuid=True), unique=True, default=uuid.uuid4)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    method = Column(String(50), nullable=False)
    dopant = Column(String(50), nullable=False)

    optimized_params = Column(JSONB, nullable=False)
    parameter_uncertainties = Column(JSONB)

    initial_error = Column(Float)
    final_error = Column(Float)

    iterations = Column(Integer)
    convergence_status = Column(String(50))

    experimental_data = Column(JSONB)
    fitted_data = Column(JSONB)

    residuals = Column(ARRAY(Float))
    r_squared = Column(Float)

    user_id = Column(String(100))
    notes = Column(Text)

    def __repr__(self):
        return f"<CalibrationResult(method={self.method}, dopant={self.dopant}, error={self.final_error})>"
