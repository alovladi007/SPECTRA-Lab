– ============================================================================
– Migration: 001_initial_schema.sql
– Description: Complete database schema for SemiconductorLab platform
– Author: Platform Team
– Date: 2025-10-21
– ============================================================================

– Enable extensions
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
capabilities TEXT[],
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
certificate_number VARCHAR(255),
certificate_url TEXT,
standards_used TEXT[],
uncertainty_budget JSONB,
notes TEXT,
status calibration_status DEFAULT ‘valid’,
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
chemical_formula VARCHAR(100),
crystal_structure VARCHAR(100),
lattice_constants JSONB,
band_gap JSONB,
refractive_index JSONB,
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
parent_id UUID REFERENCES samples(id),
name VARCHAR(255) NOT NULL,
type sample_type NOT NULL,
material_id UUID REFERENCES materials(id),
barcode VARCHAR(255),
qr_code VARCHAR(255),
location VARCHAR(255),
custodian_id UUID REFERENCES users(id),
received_date TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_samples_organization ON samples(organization_id);
CREATE INDEX idx_samples_project ON samples(project_id);
CREATE INDEX idx_samples_parent ON samples(parent_id);
CREATE INDEX idx_samples_type ON samples(type);
CREATE INDEX idx_samples_barcode ON samples(barcode);

– ============================================================================
– Methods
– ============================================================================
CREATE TYPE method_category AS ENUM (‘electrical’, ‘optical’, ‘structural’, ‘chemical’);

CREATE TABLE methods (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
name VARCHAR(255) NOT NULL UNIQUE,
display_name VARCHAR(255) NOT NULL,
category method_category NOT NULL,
description TEXT,
parameter_schema JSONB NOT NULL,
default_parameters JSONB DEFAULT ‘{}’,
units JSONB DEFAULT ‘{}’,
safety_warnings TEXT[],
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_methods_category ON methods(category);

– ============================================================================
– Recipes (Method Templates)
– ============================================================================
CREATE TABLE recipes (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
method_id UUID NOT NULL REFERENCES methods(id),
name VARCHAR(255) NOT NULL,
description TEXT,
parameters JSONB NOT NULL,
created_by UUID NOT NULL REFERENCES users(id),
is_template BOOLEAN DEFAULT FALSE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_recipes_organization ON recipes(organization_id);
CREATE INDEX idx_recipes_method ON recipes(method_id);
CREATE INDEX idx_recipes_is_template ON recipes(is_template);

– ============================================================================
– Runs (Experiments)
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
progress NUMERIC(5, 2) DEFAULT 0.0,
started_at TIMESTAMP WITH TIME ZONE,
completed_at TIMESTAMP WITH TIME ZONE,
duration_seconds INTEGER,
error_message TEXT,
raw_data_uri TEXT,
raw_data_hash VARCHAR(64),
environmental_conditions JSONB,
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

– Convert runs to TimescaleDB hypertable
SELECT create_hypertable(‘runs’, ‘created_at’, if_not_exists => TRUE);

– ============================================================================
– Measurements (Time-series data during runs)
– ============================================================================
CREATE TABLE measurements (
time TIMESTAMP WITH TIME ZONE NOT NULL,
run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
sequence_number INTEGER NOT NULL,
values JSONB NOT NULL,
metadata JSONB DEFAULT ‘{}’,
PRIMARY KEY (time, run_id)
);

CREATE INDEX idx_measurements_run ON measurements(run_id);
CREATE INDEX idx_measurements_seq ON measurements(run_id, sequence_number);

– Convert to TimescaleDB hypertable
SELECT create_hypertable(‘measurements’, ‘time’, if_not_exists => TRUE);

– ============================================================================
– Results (Analyzed/derived metrics)
– ============================================================================
CREATE TABLE results (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
metric VARCHAR(255) NOT NULL,
value NUMERIC NOT NULL,
unit VARCHAR(50) NOT NULL,
uncertainty NUMERIC,
uncertainty_type VARCHAR(50),
fit_quality JSONB,
analysis_method VARCHAR(255),
analysis_version VARCHAR(50),
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_results_run ON results(run_id);
CREATE INDEX idx_results_metric ON results(metric);
CREATE INDEX idx_results_created_at ON results(created_at);

– Convert to TimescaleDB hypertable
SELECT create_hypertable(‘results’, ‘created_at’, if_not_exists => TRUE);

– ============================================================================
– Attachments
– ============================================================================
CREATE TYPE attachment_type AS ENUM (‘raw_data’, ‘image’, ‘report’, ‘notebook’, ‘other’);

CREATE TABLE attachments (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
run_id UUID REFERENCES runs(id) ON DELETE CASCADE,
filename VARCHAR(500) NOT NULL,
mime_type VARCHAR(100) NOT NULL,
file_size BIGINT NOT NULL,
storage_uri TEXT NOT NULL,
file_hash VARCHAR(64) NOT NULL,
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
content TEXT NOT NULL,
version INTEGER DEFAULT 1,
is_signed BOOLEAN DEFAULT FALSE,
signed_at TIMESTAMP WITH TIME ZONE,
signature TEXT,
related_runs UUID[],
tags TEXT[],
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_notebook_entries_organization ON notebook_entries(organization_id);
CREATE INDEX idx_notebook_entries_project ON notebook_entries(project_id);
CREATE INDEX idx_notebook_entries_author ON notebook_entries(author_id);
CREATE INDEX idx_notebook_entries_tags ON notebook_entries USING GIN(tags);

– ============================================================================
– Approvals
– ============================================================================
CREATE TYPE approval_status AS ENUM (‘pending’, ‘approved’, ‘rejected’);

CREATE TABLE approvals (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
approver_id UUID NOT NULL REFERENCES users(id),
status approval_status DEFAULT ‘pending’,
comments TEXT,
approved_at TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_approvals_run ON approvals(run_id);
CREATE INDEX idx_approvals_approver ON approvals(approver_id);
CREATE INDEX idx_approvals_status ON approvals(status);

– ============================================================================
– SPC Control Limits
– ============================================================================
CREATE TABLE spc_control_limits (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
method_id UUID NOT NULL REFERENCES methods(id),
instrument_id UUID REFERENCES instruments(id),
metric VARCHAR(255) NOT NULL,
center_line NUMERIC NOT NULL,
ucl NUMERIC NOT NULL,
lcl NUMERIC NOT NULL,
usl NUMERIC,
lsl NUMERIC,
sample_size INTEGER NOT NULL,
confidence_level NUMERIC DEFAULT 0.95,
calculation_method VARCHAR(100),
valid_from TIMESTAMP WITH TIME ZONE NOT NULL,
valid_until TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_spc_limits_organization ON spc_control_limits(organization_id);
CREATE INDEX idx_spc_limits_method ON spc_control_limits(method_id);
CREATE INDEX idx_spc_limits_metric ON spc_control_limits(metric);

– ============================================================================
– ML Models Registry
– ============================================================================
CREATE TYPE model_status AS ENUM (‘training’, ‘deployed’, ‘archived’);

CREATE TABLE ml_models (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
name VARCHAR(255) NOT NULL,
version VARCHAR(50) NOT NULL,
model_type VARCHAR(100) NOT NULL,
target_metric VARCHAR(255) NOT NULL,
features TEXT[] NOT NULL,
hyperparameters JSONB,
training_metrics JSONB,
validation_metrics JSONB,
model_uri TEXT NOT NULL,
status model_status DEFAULT ‘training’,
trained_by UUID NOT NULL REFERENCES users(id),
deployed_at TIMESTAMP WITH TIME ZONE,
metadata JSONB DEFAULT ‘{}’,
created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_ml_models_organization ON ml_models(organization_id);
CREATE INDEX idx_ml_models_status ON ml_models(status);
CREATE INDEX idx_ml_models_name_version ON ml_models(name, version);

– ============================================================================
– Audit Log
– ============================================================================
CREATE TABLE audit_log (
id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
organization_id UUID NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
user_id UUID REFERENCES users(id),
action VARCHAR(100) NOT NULL,
entity_type VARCHAR(100) NOT NULL,
entity_id UUID NOT NULL,
changes JSONB,
ip_address INET,
user_agent TEXT,
timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_log_organization ON audit_log(organization_id);
CREATE INDEX idx_audit_log_user ON audit_log(user_id);
CREATE INDEX idx_audit_log_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp);

– Convert to TimescaleDB hypertable
SELECT create_hypertable(‘audit_log’, ‘timestamp’, if_not_exists => TRUE);

– ============================================================================
– Triggers for updated_at timestamps
– ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
NEW.updated_at = NOW();
RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_projects_updated_at BEFORE UPDATE ON projects
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_instruments_updated_at BEFORE UPDATE ON instruments
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_materials_updated_at BEFORE UPDATE ON materials
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_samples_updated_at BEFORE UPDATE ON samples
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_methods_updated_at BEFORE UPDATE ON methods
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_recipes_updated_at BEFORE UPDATE ON recipes
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_runs_updated_at BEFORE UPDATE ON runs
FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

– ============================================================================
– Seed Data: Methods
– ============================================================================
INSERT INTO methods (name, display_name, category, description, parameter_schema, default_parameters) VALUES
(‘iv_sweep’, ‘I-V Characterization’, ‘electrical’, ‘Current-Voltage sweeps for diodes, transistors, and solar cells’,
‘{“type”: “object”, “properties”: {“v_start”: {“type”: “number”}, “v_stop”: {“type”: “number”}, “points”: {“type”: “integer”}}}’,
‘{“v_start”: 0, “v_stop”: 1, “points”: 100}’),

(‘four_point_probe’, ‘Four-Point Probe’, ‘electrical’, ‘Sheet resistance measurement’,
‘{“type”: “object”, “properties”: {“current”: {“type”: “number”}, “geometry”: {“type”: “string”}}}’,
‘{“current”: 0.001, “geometry”: “linear”}’),

(‘hall_effect’, ‘Hall Effect’, ‘electrical’, ‘Mobility and carrier concentration measurement’,
‘{“type”: “object”, “properties”: {“current”: {“type”: “number”}, “magnetic_field”: {“type”: “number”}}}’,
‘{“current”: 0.001, “magnetic_field”: 0.5}’),

(‘uv_vis_nir’, ‘UV-Vis-NIR Spectroscopy’, ‘optical’, ‘Absorption/transmission/reflectance spectroscopy’,
‘{“type”: “object”, “properties”: {“wavelength_start”: {“type”: “number”}, “wavelength_stop”: {“type”: “number”}}}’,
‘{“wavelength_start”: 300, “wavelength_stop”: 2500}’),

(‘xrd’, ‘X-Ray Diffraction’, ‘structural’, ‘Crystal structure and phase identification’,
‘{“type”: “object”, “properties”: {“2theta_start”: {“type”: “number”}, “2theta_stop”: {“type”: “number”}}}’,
‘{“2theta_start”: 10, “2theta_stop”: 80}’),

(‘afm’, ‘Atomic Force Microscopy’, ‘structural’, ‘Surface topography and roughness’,
‘{“type”: “object”, “properties”: {“scan_size”: {“type”: “number”}, “resolution”: {“type”: “integer”}}}’,
‘{“scan_size”: 10, “resolution”: 512}’),

(‘xps’, ‘X-ray Photoelectron Spectroscopy’, ‘chemical’, ‘Surface chemical composition’,
‘{“type”: “object”, “properties”: {“energy_start”: {“type”: “number”}, “energy_stop”: {“type”: “number”}}}’,
‘{“energy_start”: 0, “energy_stop”: 1200}’);

– ============================================================================
– Views
– ============================================================================

– Active runs summary
CREATE VIEW active_runs AS
SELECT
r.id,
r.status,
r.progress,
m.display_name as method,
s.name as sample,
i.name as instrument,
u.first_name || ’ ’ || u.last_name as operator,
r.started_at
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
– End of Migration
– ============================================================================