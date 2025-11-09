-- ============================================================================
-- Database Schema for SPECTRA Diffusion Module
-- Session 12: Production Release with Audit Trails
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- Simulation Audit Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS simulation_audit (
    id SERIAL PRIMARY KEY,
    simulation_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    run_id VARCHAR(100),
    recipe_id VARCHAR(100),
    simulation_type VARCHAR(50) NOT NULL CHECK (simulation_type IN ('diffusion', 'oxidation', 'calibration')),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,

    -- User tracking
    user_id VARCHAR(100),
    user_email VARCHAR(255),

    -- Input parameters (stored as JSONB for flexibility)
    parameters JSONB NOT NULL,

    -- Results (stored as JSONB)
    results JSONB,

    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed')),

    -- Performance metrics
    execution_time_ms INTEGER,
    memory_mb REAL,

    -- Error handling
    error_message TEXT,
    error_traceback TEXT,

    -- Metadata
    metadata JSONB,
    tags TEXT[],

    -- Provenance
    git_commit VARCHAR(40),
    module_version VARCHAR(20),

    -- Indexes for common queries
    CONSTRAINT positive_execution_time CHECK (execution_time_ms >= 0)
);

-- Create indexes for performance
CREATE INDEX idx_simulation_audit_created_at ON simulation_audit(created_at DESC);
CREATE INDEX idx_simulation_audit_simulation_id ON simulation_audit(simulation_id);
CREATE INDEX idx_simulation_audit_type ON simulation_audit(simulation_type);
CREATE INDEX idx_simulation_audit_status ON simulation_audit(status);
CREATE INDEX idx_simulation_audit_user_id ON simulation_audit(user_id);
CREATE INDEX idx_simulation_audit_recipe_id ON simulation_audit(recipe_id);
CREATE INDEX idx_simulation_audit_parameters ON simulation_audit USING GIN (parameters);
CREATE INDEX idx_simulation_audit_results ON simulation_audit USING GIN (results);

-- ============================================================================
-- Batch Jobs Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS batch_jobs (
    id SERIAL PRIMARY KEY,
    job_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    job_name VARCHAR(255),

    -- Job configuration
    simulation_type VARCHAR(50) NOT NULL,
    total_simulations INTEGER NOT NULL,
    completed_simulations INTEGER DEFAULT 0,
    failed_simulations INTEGER DEFAULT 0,

    -- Status
    status VARCHAR(20) DEFAULT 'queued' CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,

    -- User tracking
    user_id VARCHAR(100),

    -- Configuration
    config JSONB,

    -- Results summary
    results_summary JSONB,

    -- Error tracking
    error_message TEXT,

    CONSTRAINT positive_total CHECK (total_simulations > 0),
    CONSTRAINT non_negative_completed CHECK (completed_simulations >= 0),
    CONSTRAINT non_negative_failed CHECK (failed_simulations >= 0)
);

CREATE INDEX idx_batch_jobs_job_id ON batch_jobs(job_id);
CREATE INDEX idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX idx_batch_jobs_created_at ON batch_jobs(created_at DESC);
CREATE INDEX idx_batch_jobs_user_id ON batch_jobs(user_id);

-- ============================================================================
-- KPI Tracking Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS kpi_measurements (
    id SERIAL PRIMARY KEY,
    measurement_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    simulation_id UUID REFERENCES simulation_audit(simulation_id),

    -- Timestamp
    measured_at TIMESTAMPTZ DEFAULT NOW(),

    -- KPI type
    kpi_name VARCHAR(100) NOT NULL,
    kpi_value REAL NOT NULL,
    kpi_unit VARCHAR(50),

    -- Control limits
    target_value REAL,
    ucl REAL,  -- Upper control limit
    lcl REAL,  -- Lower control limit
    usl REAL,  -- Upper spec limit
    lsl REAL,  -- Lower spec limit

    -- Flags
    within_control BOOLEAN,
    within_spec BOOLEAN,

    -- Tool/recipe context
    tool_id VARCHAR(100),
    recipe_id VARCHAR(100),
    wafer_id VARCHAR(100),
    lot_id VARCHAR(100),

    -- Additional metadata
    metadata JSONB
);

CREATE INDEX idx_kpi_measurements_measured_at ON kpi_measurements(measured_at DESC);
CREATE INDEX idx_kpi_measurements_kpi_name ON kpi_measurements(kpi_name);
CREATE INDEX idx_kpi_measurements_simulation_id ON kpi_measurements(simulation_id);
CREATE INDEX idx_kpi_measurements_recipe_id ON kpi_measurements(recipe_id);
CREATE INDEX idx_kpi_measurements_tool_id ON kpi_measurements(tool_id);

-- ============================================================================
-- SPC Violations Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS spc_violations (
    id SERIAL PRIMARY KEY,
    violation_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),

    -- Timestamp
    detected_at TIMESTAMPTZ DEFAULT NOW(),

    -- Rule information
    rule_id VARCHAR(50) NOT NULL,
    rule_name VARCHAR(255) NOT NULL,

    -- Violation details
    severity VARCHAR(20) CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    window_start INTEGER,
    window_end INTEGER,
    z_score REAL,

    -- Context
    kpi_name VARCHAR(100),
    recipe_id VARCHAR(100),
    tool_id VARCHAR(100),

    -- Root cause analysis
    suggested_causes TEXT[],
    recommended_actions TEXT[],

    -- Resolution
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,

    -- Metadata
    metadata JSONB
);

CREATE INDEX idx_spc_violations_detected_at ON spc_violations(detected_at DESC);
CREATE INDEX idx_spc_violations_rule_id ON spc_violations(rule_id);
CREATE INDEX idx_spc_violations_severity ON spc_violations(severity);
CREATE INDEX idx_spc_violations_kpi_name ON spc_violations(kpi_name);
CREATE INDEX idx_spc_violations_acknowledged ON spc_violations(acknowledged);
CREATE INDEX idx_spc_violations_resolved ON spc_violations(resolved);

-- ============================================================================
-- Maintenance Recommendations Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS maintenance_recommendations (
    id SERIAL PRIMARY KEY,
    recommendation_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),

    -- Timestamp
    generated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Recommendation details
    action VARCHAR(255) NOT NULL,
    component VARCHAR(255),
    urgency VARCHAR(20) CHECK (urgency IN ('low', 'medium', 'high', 'critical')),
    confidence REAL CHECK (confidence >= 0 AND confidence <= 1),

    -- Impact estimates
    estimated_downtime_hours REAL,
    cost_impact VARCHAR(20),
    expected_improvement TEXT,

    -- Tool context
    tool_id VARCHAR(100),
    tool_health_score REAL,
    failure_probability REAL,

    -- Action tracking
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'scheduled', 'in_progress', 'completed', 'cancelled')),
    scheduled_for TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    completed_by VARCHAR(100),

    -- Results
    actual_downtime_hours REAL,
    actual_improvement TEXT,

    -- Metadata
    fdc_data JSONB,
    metadata JSONB
);

CREATE INDEX idx_maintenance_recs_generated_at ON maintenance_recommendations(generated_at DESC);
CREATE INDEX idx_maintenance_recs_urgency ON maintenance_recommendations(urgency);
CREATE INDEX idx_maintenance_recs_status ON maintenance_recommendations(status);
CREATE INDEX idx_maintenance_recs_tool_id ON maintenance_recommendations(tool_id);

-- ============================================================================
-- Session/Authentication Table (Optional - for multi-user deployments)
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_sessions (
    id SERIAL PRIMARY KEY,
    session_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL,
    user_email VARCHAR(255),

    -- Session tracking
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,

    -- Session data
    ip_address INET,
    user_agent TEXT,
    metadata JSONB
);

CREATE INDEX idx_user_sessions_session_id ON user_sessions(session_id);
CREATE INDEX idx_user_sessions_user_id ON user_sessions(user_id);
CREATE INDEX idx_user_sessions_expires_at ON user_sessions(expires_at);

-- ============================================================================
-- Calibration Results Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS calibration_results (
    id SERIAL PRIMARY KEY,
    calibration_id UUID UNIQUE NOT NULL DEFAULT uuid_generate_v4(),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Method
    method VARCHAR(50) NOT NULL CHECK (method IN ('least_squares', 'bayesian_mcmc')),

    -- Input
    dopant VARCHAR(50) NOT NULL,
    target_metric VARCHAR(50) NOT NULL,
    experimental_data JSONB NOT NULL,

    -- Results
    optimized_params JSONB NOT NULL,
    initial_error REAL,
    final_error REAL,
    improvement_pct REAL,

    -- Convergence
    iterations INTEGER,
    convergence_status VARCHAR(50),

    -- Bayesian-specific (if applicable)
    posterior_samples JSONB,
    credible_intervals JSONB,

    -- Metadata
    user_id VARCHAR(100),
    metadata JSONB
);

CREATE INDEX idx_calibration_results_created_at ON calibration_results(created_at DESC);
CREATE INDEX idx_calibration_results_method ON calibration_results(method);
CREATE INDEX idx_calibration_results_dopant ON calibration_results(dopant);

-- ============================================================================
-- Views for Common Queries
-- ============================================================================

-- Recent simulations view
CREATE OR REPLACE VIEW recent_simulations AS
SELECT
    simulation_id,
    simulation_type,
    status,
    created_at,
    execution_time_ms,
    parameters->>'temperature' as temperature,
    parameters->>'time' as time,
    parameters->>'dopant' as dopant,
    results->>'junction_depth' as junction_depth,
    results->>'sheet_resistance' as sheet_resistance,
    user_id
FROM simulation_audit
ORDER BY created_at DESC
LIMIT 100;

-- SPC summary view
CREATE OR REPLACE VIEW spc_summary AS
SELECT
    kpi_name,
    COUNT(*) as total_measurements,
    AVG(kpi_value) as avg_value,
    STDDEV(kpi_value) as std_dev,
    MIN(kpi_value) as min_value,
    MAX(kpi_value) as max_value,
    COUNT(*) FILTER (WHERE NOT within_control) as out_of_control_count,
    COUNT(*) FILTER (WHERE NOT within_spec) as out_of_spec_count
FROM kpi_measurements
GROUP BY kpi_name;

-- Tool health view
CREATE OR REPLACE VIEW tool_health AS
SELECT
    tool_id,
    COUNT(*) as total_recommendations,
    COUNT(*) FILTER (WHERE urgency = 'critical') as critical_count,
    COUNT(*) FILTER (WHERE urgency = 'high') as high_count,
    COUNT(*) FILTER (WHERE status = 'open') as open_count,
    AVG(tool_health_score) as avg_health_score,
    MAX(generated_at) as last_check
FROM maintenance_recommendations
GROUP BY tool_id;

-- ============================================================================
-- Functions for Data Retention
-- ============================================================================

-- Function to archive old simulations
CREATE OR REPLACE FUNCTION archive_old_simulations(days_to_keep INTEGER DEFAULT 365)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- This would move old records to an archive table in production
    -- For now, just return count of old records
    SELECT COUNT(*) INTO archived_count
    FROM simulation_audit
    WHERE created_at < NOW() - INTERVAL '1 day' * days_to_keep;

    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Grant Permissions (adjust as needed for your deployment)
-- ============================================================================

-- Grant access to diffusion user (if created)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO diffusion_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO diffusion_user;

-- ============================================================================
-- Sample Data (Optional - for testing)
-- ============================================================================

-- Insert sample simulation
INSERT INTO simulation_audit (
    simulation_type,
    parameters,
    results,
    status,
    execution_time_ms,
    user_id,
    module_version
) VALUES (
    'diffusion',
    '{"temperature": 1000, "time": 30, "dopant": "boron", "initial_concentration": 1e20}'::jsonb,
    '{"junction_depth": 717.2, "sheet_resistance": 125.3}'::jsonb,
    'completed',
    85,
    'system',
    '1.12.0'
);

COMMENT ON TABLE simulation_audit IS 'Audit trail for all simulations with full provenance';
COMMENT ON TABLE batch_jobs IS 'Batch job tracking for parameter sweeps';
COMMENT ON TABLE kpi_measurements IS 'Key performance indicator measurements';
COMMENT ON TABLE spc_violations IS 'Statistical process control violations';
COMMENT ON TABLE maintenance_recommendations IS 'Predictive maintenance recommendations';
COMMENT ON TABLE calibration_results IS 'Model calibration results and optimized parameters';
