– Migration: 001_initial_schema.sql
– Description: Core entities for SemiconductorLab platform
– Author: Platform Team
– Date: 2025-10-21

– ============================================================================
– Enable extensions
– ============================================================================
CREATE EXTENSION IF NOT EXISTS “uuid-ossp”;
CREATE EXTENSION IF NOT EXISTS “pgcrypto”;
CREATE EXTENSION IF NOT EXISTS “timescaledb”;

– ============================================================================
– Organizations
– ============================================================================
CREATE TABLE organizations (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
name VARCHAR(255) NOT NULL,
slug VARCHAR(100) NOT NULL UNIQUE,
settings JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_organizations_slug ON organizations(slug);

– ============================================================================
– Users
– ============================================================================
CREATE TYPE user_role AS ENUM (‘admin’, ‘pi’, ‘engineer’, ‘technician’, ‘viewer’);

CREATE TABLE users (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
email VARCHAR(255) NOT NULL UNIQUE,
password_hash VARCHAR(255) NOT NULL,
first_name VARCHAR(100) NOT NULL,
last_name VARCHAR(100) NOT NULL,
role user_role NOT NULL DEFAULT ‘viewer’,
is_active BOOLEAN DEFAULT TRUE,
last_login TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_users_role ON users(role);

– ============================================================================
– Projects
– ============================================================================
CREATE TYPE project_status AS ENUM (‘active’, ‘on_hold’, ‘completed’, ‘archived’);

CREATE TABLE projects (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
name VARCHAR(255) NOT NULL,
description TEXT,
owner_id UUID NOT NULL REFERENCES users(id),
status project_status DEFAULT ‘active’,
started_at TIMESTAMP WITH TIME ZONE,
completed_at TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_projects_organization ON projects(organization_id);
CREATE INDEX idx_projects_owner ON projects(owner_id);
CREATE INDEX idx_projects_status ON projects(status);

– ============================================================================
– Instruments
– ============================================================================
CREATE TYPE instrument_status AS ENUM (‘online’, ‘offline’, ‘maintenance’, ‘error’);
CREATE TYPE connection_type AS ENUM (‘visa_usb’, ‘visa_gpib’, ‘visa_tcpip’, ‘serial’, ‘usb_raw’);

CREATE TABLE instruments (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
name VARCHAR(255) NOT NULL,
model VARCHAR(255) NOT NULL,
vendor VARCHAR(255) NOT NULL,
serial_number VARCHAR(255),
connection_type connection_type NOT NULL,
connection_string VARCHAR(500) NOT NULL,
driver VARCHAR(255) NOT NULL,
capabilities TEXT[], – Array of method names this instrument supports
status instrument_status DEFAULT ‘offline’,
firmware_version VARCHAR(100),
last_seen TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
UNIQUE(organization_id, name)
);

CREATE INDEX idx_instruments_organization ON instruments(organization_id);
CREATE INDEX idx_instruments_status ON instruments(status);
CREATE INDEX idx_instruments_capabilities ON instruments USING GIN(capabilities);

– ============================================================================
– Calibrations
– ============================================================================
CREATE TYPE calibration_status AS ENUM (‘valid’, ‘due’, ‘overdue’, ‘invalid’);

CREATE TABLE calibrations (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
instrument_id UUID NOT NULL REFERENCES instruments(id) ON DELETE CASCADE,
performed_by UUID NOT NULL REFERENCES users(id),
calibration_date TIMESTAMP WITH TIME ZONE NOT NULL,
next_calibration_date TIMESTAMP WITH TIME ZONE NOT NULL,
status calibration_status DEFAULT ‘valid’,
certificate_number VARCHAR(255),
certificate_url TEXT,
standards_used TEXT[],
uncertainty_budget JSONB,
notes TEXT,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_calibrations_instrument ON calibrations(instrument_id);
CREATE INDEX idx_calibrations_status ON calibrations(status);
CREATE INDEX idx_calibrations_next_date ON calibrations(next_calibration_date);

– ============================================================================
– Materials Library
– ============================================================================
CREATE TABLE materials (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
name VARCHAR(255) NOT NULL UNIQUE,
chemical_formula VARCHAR(255),
crystal_structure VARCHAR(100),
lattice_constants JSONB, – {a, b, c, alpha, beta, gamma}
band_gap JSONB, – {value, unit, temperature, type}
refractive_index JSONB, – {wavelength: n+ik}
properties JSONB DEFAULT ‘{}’,
references TEXT[],
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_materials_name ON materials(name);

– ============================================================================
– Samples
– ============================================================================
CREATE TYPE sample_type AS ENUM (‘wafer’, ‘die’, ‘device’, ‘coupon’, ‘test_structure’);

CREATE TABLE samples (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
parent_id UUID REFERENCES samples(id) ON DELETE SET NULL,
name VARCHAR(255) NOT NULL,
type sample_type NOT NULL,
material_id UUID REFERENCES materials(id),
barcode VARCHAR(255),
qr_code VARCHAR(500),
location VARCHAR(255),
custodian_id UUID REFERENCES users(id),
received_date TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
UNIQUE(organization_id, barcode)
);

CREATE INDEX idx_samples_organization ON samples(organization_id);
CREATE INDEX idx_samples_project ON samples(project_id);
CREATE INDEX idx_samples_parent ON samples(parent_id);
CREATE INDEX idx_samples_barcode ON samples(barcode);
CREATE INDEX idx_samples_type ON samples(type);

– ============================================================================
– Methods (Characterization Method Templates)
– ============================================================================
CREATE TYPE method_category AS ENUM (‘electrical’, ‘optical’, ‘structural’, ‘chemical’);

CREATE TABLE methods (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
name VARCHAR(255) NOT NULL UNIQUE,
display_name VARCHAR(255) NOT NULL,
category method_category NOT NULL,
description TEXT,
parameter_schema JSONB NOT NULL, – JSON Schema for method parameters
default_parameters JSONB DEFAULT ‘{}’,
sop_document TEXT, – URL or path to SOP
required_capabilities TEXT[], – Required instrument capabilities
safety_warnings TEXT[],
estimated_duration_minutes INTEGER,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_methods_category ON methods(category);
CREATE INDEX idx_methods_name ON methods(name);

– ============================================================================
– Recipes (User-saved method configurations)
– ============================================================================
CREATE TABLE recipes (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
method_id UUID NOT NULL REFERENCES methods(id) ON DELETE CASCADE,
name VARCHAR(255) NOT NULL,
description TEXT,
parameters JSONB NOT NULL,
owner_id UUID NOT NULL REFERENCES users(id),
is_public BOOLEAN DEFAULT FALSE,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_recipes_organization ON recipes(organization_id);
CREATE INDEX idx_recipes_method ON recipes(method_id);
CREATE INDEX idx_recipes_owner ON recipes(owner_id);

– ============================================================================
– Runs (Experiment Executions)
– ============================================================================
CREATE TYPE run_status AS ENUM (‘pending’, ‘running’, ‘completed’, ‘failed’, ‘aborted’);

CREATE TABLE runs (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
method_id UUID NOT NULL REFERENCES methods(id),
sample_id UUID NOT NULL REFERENCES samples(id) ON DELETE CASCADE,
instrument_id UUID NOT NULL REFERENCES instruments(id),
operator_id UUID NOT NULL REFERENCES users(id),
recipe_id UUID REFERENCES recipes(id),
status run_status DEFAULT ‘pending’,
parameters JSONB NOT NULL,
progress NUMERIC(5,2) DEFAULT 0.0, – 0.00 to 100.00
started_at TIMESTAMP WITH TIME ZONE,
completed_at TIMESTAMP WITH TIME ZONE,
duration_seconds INTEGER,
error_message TEXT,
raw_data_uri TEXT, – S3/MinIO path to raw data file
raw_data_hash VARCHAR(64), – SHA256 hash for integrity
environmental_conditions JSONB, – {temperature, humidity, pressure}
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_runs_organization ON runs(organization_id);
CREATE INDEX idx_runs_project ON runs(project_id);
CREATE INDEX idx_runs_method ON runs(method_id);
CREATE INDEX idx_runs_sample ON runs(sample_id);
CREATE INDEX idx_runs_instrument ON runs(instrument_id);
CREATE INDEX idx_runs_operator ON runs(operator_id);
CREATE INDEX idx_runs_status ON runs(status);
CREATE INDEX idx_runs_started_at ON runs(started_at);

– Convert to hypertable for time-series optimization
SELECT create_hypertable(‘runs’, ‘created_at’, if_not_exists => TRUE);

– ============================================================================
– Measurements (Time-series data points during a run)
– ============================================================================
CREATE TABLE measurements (
time TIMESTAMP WITH TIME ZONE NOT NULL,
run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
sequence_number INTEGER NOT NULL,
values JSONB NOT NULL, – {voltage: 1.5, current: 0.001, …}
metadata JSONB DEFAULT ‘{}’
);

– Create hypertable
SELECT create_hypertable(‘measurements’, ‘time’, if_not_exists => TRUE);

CREATE INDEX idx_measurements_run ON measurements(run_id, time DESC);

– ============================================================================
– Results (Analyzed/derived metrics from runs)
– ============================================================================
CREATE TABLE results (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
metric VARCHAR(255) NOT NULL, – e.g., ‘vth’, ‘ideality_factor’, ‘band_gap’
value NUMERIC NOT NULL,
unit VARCHAR(50) NOT NULL,
uncertainty NUMERIC,
uncertainty_type VARCHAR(50), – ‘absolute’, ‘relative’, ‘k_coverage’
fit_quality JSONB, – {r_squared, rmse, chi_squared}
analysis_method VARCHAR(255), – Algorithm/function used
analysis_version VARCHAR(50),
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_results_run ON results(run_id);
CREATE INDEX idx_results_metric ON results(metric);
CREATE INDEX idx_results_created ON results(created_at);

– Convert to hypertable
SELECT create_hypertable(‘results’, ‘created_at’, if_not_exists => TRUE);

– ============================================================================
– Attachments (Files associated with runs)
– ============================================================================
CREATE TYPE attachment_type AS ENUM (‘raw_data’, ‘image’, ‘report’, ‘notebook’, ‘other’);

CREATE TABLE attachments (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
filename VARCHAR(500) NOT NULL,
mime_type VARCHAR(100) NOT NULL,
file_size BIGINT NOT NULL, – bytes
storage_uri TEXT NOT NULL, – S3/MinIO path
file_hash VARCHAR(64) NOT NULL, – SHA256
attachment_type attachment_type NOT NULL,
description TEXT,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_attachments_run ON attachments(run_id);
CREATE INDEX idx_attachments_type ON attachments(attachment_type);

– ============================================================================
– Notebook Entries (ELN)
– ============================================================================
CREATE TABLE notebook_entries (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
author_id UUID NOT NULL REFERENCES users(id),
title VARCHAR(500) NOT NULL,
content TEXT NOT NULL, – Rich text (HTML or Markdown)
content_format VARCHAR(20) DEFAULT ‘markdown’,
run_ids UUID[], – Associated runs
tags TEXT[],
version INTEGER DEFAULT 1,
is_signed BOOLEAN DEFAULT FALSE,
signed_at TIMESTAMP WITH TIME ZONE,
signed_by UUID REFERENCES users(id),
signature_reason TEXT,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_notebook_organization ON notebook_entries(organization_id);
CREATE INDEX idx_notebook_project ON notebook_entries(project_id);
CREATE INDEX idx_notebook_author ON notebook_entries(author_id);
CREATE INDEX idx_notebook_tags ON notebook_entries USING GIN(tags);

– ============================================================================
– Approvals (Sign-off workflow)
– ============================================================================
CREATE TYPE approval_status AS ENUM (‘pending’, ‘approved’, ‘rejected’);

CREATE TABLE approvals (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
notebook_entry_id UUID REFERENCES notebook_entries(id) ON DELETE CASCADE,
approver_id UUID NOT NULL REFERENCES users(id),
status approval_status DEFAULT ‘pending’,
comments TEXT,
approved_at TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
CONSTRAINT approval_target_check CHECK (
(run_id IS NOT NULL AND notebook_entry_id IS NULL) OR
(run_id IS NULL AND notebook_entry_id IS NOT NULL)
)
);

CREATE INDEX idx_approvals_run ON approvals(run_id);
CREATE INDEX idx_approvals_notebook ON approvals(notebook_entry_id);
CREATE INDEX idx_approvals_approver ON approvals(approver_id);
CREATE INDEX idx_approvals_status ON approvals(status);

– ============================================================================
– Audit Log
– ============================================================================
CREATE TABLE audit_log (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
user_id UUID REFERENCES users(id) ON DELETE SET NULL,
action VARCHAR(100) NOT NULL, – ‘create’, ‘read’, ‘update’, ‘delete’
resource_type VARCHAR(100) NOT NULL, – ‘run’, ‘sample’, ‘instrument’, etc.
resource_id UUID,
changes JSONB, – Before/after values for updates
ip_address INET,
user_agent TEXT,
metadata JSONB DEFAULT ‘{}’,
timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_organization ON audit_log(organization_id);
CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_resource ON audit_log(resource_type, resource_id);
CREATE INDEX idx_audit_timestamp ON audit_log(timestamp DESC);

– Convert to hypertable
SELECT create_hypertable(‘audit_log’, ‘timestamp’, if_not_exists => TRUE);

– ============================================================================
– SPC Control Limits (Pre-computed for performance)
– ============================================================================
CREATE TABLE spc_control_limits (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
metric VARCHAR(255) NOT NULL,
subgroup_column VARCHAR(100), – e.g., ‘wafer_id’, ‘lot_id’
chart_type VARCHAR(50) NOT NULL, – ‘xbar_r’, ‘ewma’, ‘cusum’
ucl NUMERIC NOT NULL, – Upper Control Limit
lcl NUMERIC NOT NULL, – Lower Control Limit
centerline NUMERIC NOT NULL,
sigma NUMERIC,
sample_size INTEGER,
computed_from_runs UUID[], – Run IDs used to compute limits
valid_from TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
valid_until TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_spc_metric ON spc_control_limits(metric);
CREATE INDEX idx_spc_valid ON spc_control_limits(valid_from, valid_until);

– ============================================================================
– ML Models Registry
– ============================================================================
CREATE TYPE model_status AS ENUM (‘training’, ‘deployed’, ‘archived’);

CREATE TABLE ml_models (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
name VARCHAR(255) NOT NULL,
description TEXT,
model_type VARCHAR(100) NOT NULL, – ‘virtual_metrology’, ‘anomaly_detection’, etc.
algorithm VARCHAR(100), – ‘lightgbm’, ‘random_forest’, ‘autoencoder’
target_metric VARCHAR(255),
features TEXT[],
performance_metrics JSONB, – {r_squared, rmse, mae, f1_score, …}
training_runs UUID[], – Runs used for training
model_artifact_uri TEXT, – S3/MinIO path to ONNX file
version VARCHAR(50),
status model_status DEFAULT ‘training’,
deployed_at TIMESTAMP WITH TIME ZONE,
trained_by UUID REFERENCES users(id),
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ml_models_organization ON ml_models(organization_id);
CREATE INDEX idx_ml_models_type ON ml_models(model_type);
CREATE INDEX idx_ml_models_status ON ml_models(status);

– ============================================================================
– Functions & Triggers
– ============================================================================

– Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
NEW.updated_at = NOW();
RETURN NEW;
END;
$$ LANGUAGE plpgsql;

– Apply trigger to all tables with updated_at
CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_instruments_updated_at BEFORE UPDATE ON instruments
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_samples_updated_at BEFORE UPDATE ON samples
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_methods_updated_at BEFORE UPDATE ON methods
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recipes_updated_at BEFORE UPDATE ON recipes
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_runs_updated_at BEFORE UPDATE ON runs
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_notebook_entries_updated_at BEFORE UPDATE ON notebook_entries
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_models_updated_at BEFORE UPDATE ON ml_models
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

– ============================================================================
– Seed Data (Characterization Methods)
– ============================================================================

– Electrical Methods
INSERT INTO methods (name, display_name, category, description, parameter_schema, required_capabilities) VALUES
(‘four_point_probe’, ‘Four-Point Probe’, ‘electrical’, ‘Sheet resistance measurement using Van der Pauw method’,
‘{“type”: “object”, “properties”: {“current”: {“type”: “number”}, “temperature”: {“type”: “number”}}}’,
ARRAY[‘four_point_probe’]),

(‘hall_effect’, ‘Hall Effect’, ‘electrical’, ‘Carrier concentration and mobility measurement’,
‘{“type”: “object”, “properties”: {“current”: {“type”: “number”}, “magnetic_field”: {“type”: “number”}}}’,
ARRAY[‘hall_effect’]),

(‘iv_sweep’, ‘I-V Characterization’, ‘electrical’, ‘Current-voltage characterization of devices’,
‘{“type”: “object”, “properties”: {“v_start”: {“type”: “number”}, “v_stop”: {“type”: “number”}, “v_step”: {“type”: “number”}, “compliance”: {“type”: “number”}}}’,
ARRAY[‘iv_sweep’]),

(‘cv_measurement’, ‘C-V Profiling’, ‘electrical’, ‘Capacitance-voltage measurement for doping profiles’,
‘{“type”: “object”, “properties”: {“v_start”: {“type”: “number”}, “v_stop”: {“type”: “number”}, “frequency”: {“type”: “number”}}}’,
ARRAY[‘cv_measurement’]);

– Optical Methods
INSERT INTO methods (name, display_name, category, description, parameter_schema, required_capabilities) VALUES
(‘uv_vis_nir’, ‘UV-Vis-NIR Spectroscopy’, ‘optical’, ‘Optical absorption and transmission spectroscopy’,
‘{“type”: “object”, “properties”: {“wavelength_start”: {“type”: “number”}, “wavelength_stop”: {“type”: “number”}}}’,
ARRAY[‘spectroscopy’]),

(‘ellipsometry’, ‘Ellipsometry’, ‘optical’, ‘Thin film thickness and optical constants’,
‘{“type”: “object”, “properties”: {“wavelength_range”: {“type”: “array”}, “angle”: {“type”: “number”}}}’,
ARRAY[‘ellipsometry’]);

– Structural Methods
INSERT INTO methods (name, display_name, category, description, parameter_schema, required_capabilities) VALUES
(‘xrd’, ‘X-Ray Diffraction’, ‘structural’, ‘Crystal structure and phase identification’,
‘{“type”: “object”, “properties”: {“theta_start”: {“type”: “number”}, “theta_stop”: {“type”: “number”}}}’,
ARRAY[‘xrd’]);

– ============================================================================
– Views for Common Queries
– ============================================================================

– Active runs with full context
CREATE VIEW active_runs AS
SELECT
r.id,
r.status,
r.progress,
r.started_at,
m.display_name as method_name,
s.name as sample_name,
i.name as instrument_name,
u.first_name || ’ ’ || u.last_name as operator_name
FROM runs r
JOIN methods m ON r.method_id = m.id
JOIN samples s ON r.sample_id = s.id
JOIN instruments i ON r.instrument_id = i.id
JOIN users u ON r.operator_id = u.id
WHERE r.status IN (‘pending’, ‘running’);

– Calibration status summary
CREATE VIEW calibration_status AS
SELECT
i.id as instrument_id,
i.name as instrument_name,
i.model,
c.calibration_date,
c.next_calibration_date,
c.status,
CASE
WHEN c.next_calibration_date < NOW() THEN ‘overdue’
WHEN c.next_calibration_date < NOW() + INTERVAL ‘7 days’ THEN ‘due_soon’
ELSE ‘valid’
END as urgency
FROM instruments i
LEFT JOIN LATERAL (
SELECT * FROM calibrations
WHERE instrument_id = i.id
ORDER BY calibration_date DESC
LIMIT 1
) c ON TRUE;

– ============================================================================
– Comments
– ============================================================================
COMMENT ON TABLE organizations IS ‘Organizations using the platform’;
COMMENT ON TABLE users IS ‘Platform users with role-based access’;
COMMENT ON TABLE instruments IS ‘Registered characterization instruments’;
COMMENT ON TABLE samples IS ‘Samples, wafers, dies, and devices under test’;
COMMENT ON TABLE runs IS ‘Experiment executions with full provenance’;
COMMENT ON TABLE results IS ‘Analyzed metrics extracted from runs’;
COMMENT ON TABLE audit_log IS ‘Immutable audit trail for compliance’;

– ============================================================================
– Grants (adjust based on your access control strategy)
– ============================================================================
– Example: Grant read-only access to viewer role
– GRANT SELECT ON ALL TABLES IN SCHEMA public TO viewer_role;

– ============================================================================
– End of Migration
– ============================================================================