"""Initial schema for SPECTRA-Lab Platform Session 17

Revision ID: 0001
Revises: 
Create Date: 2025-10-26 12:00:00.000000

Creates:
- Organizations (multi-tenancy)
- Users with RBAC
- API Keys
- Instruments & Calibrations  
- Materials, Samples, Wafers, Devices
- Recipes & Approvals
- Runs & Results
- Attachments
- ELN Entries & Signatures
- SOPs & Custody Events
- SPC Series, Points & Alerts
- Feature Sets & ML Models
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable UUID extension
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    
    # ========================================================================
    # Organizations
    # ========================================================================
    op.create_table('organizations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('slug', sa.String(100), nullable=False, unique=True),
        sa.Column('settings', postgresql.JSONB, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_organizations_slug', 'organizations', ['slug'], unique=True)
    op.create_index('ix_organizations_is_active', 'organizations', ['is_active'])
    
    # ========================================================================
    # Users
    # ========================================================================
    op.create_table('users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('role', sa.Enum('admin', 'pi', 'engineer', 'technician', 'viewer', name='user_role'), nullable=False),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=sa.text('true')),
        sa.Column('last_login', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_users_email', 'users', ['email'])
    op.create_index('ix_users_org_email', 'users', ['organization_id', 'email'], unique=True)
    op.create_index('ix_users_org_role', 'users', ['organization_id', 'role'])
    
    # ========================================================================
    # API Keys
    # ========================================================================
    op.create_table('api_keys',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('key_hash', sa.String(255), nullable=False, unique=True),
        sa.Column('scopes', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=sa.text('true')),
        sa.Column('expires_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('last_used', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_api_keys_key_hash', 'api_keys', ['key_hash'], unique=True)
    op.create_index('ix_api_keys_user', 'api_keys', ['user_id'])
    
    # ========================================================================
    # Instruments
    # ========================================================================
    op.create_table('instruments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model', sa.String(255), nullable=False),
        sa.Column('serial_number', sa.String(255), nullable=True),
        sa.Column('vendor', sa.String(255), nullable=True),
        sa.Column('category', sa.String(100), nullable=False),
        sa.Column('status', sa.Enum('online', 'offline', 'maintenance', 'error', name='instrument_status'), nullable=False, server_default='offline'),
        sa.Column('location', sa.String(255), nullable=True),
        sa.Column('capabilities', postgresql.JSONB, nullable=True),
        sa.Column('config', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_instruments_org_name', 'instruments', ['organization_id', 'name'])
    op.create_index('ix_instruments_org_category', 'instruments', ['organization_id', 'category'])
    op.create_index('ix_instruments_status', 'instruments', ['status'])
    
    # ========================================================================
    # Calibrations
    # ========================================================================
    op.create_table('calibrations',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id', ondelete='CASCADE'), nullable=False),
        sa.Column('certificate_id', sa.String(255), nullable=False),
        sa.Column('status', sa.Enum('valid', 'expired', name='calibration_status'), nullable=False),
        sa.Column('provider', sa.String(255), nullable=True),
        sa.Column('issued_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('expires_at', sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column('standards_used', postgresql.JSONB, nullable=True),
        sa.Column('uncertainty', postgresql.JSONB, nullable=True),
        sa.Column('attachment_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint('expires_at > issued_at', name='check_cert_dates')
    )
    op.create_index('ix_calibrations_instrument', 'calibrations', ['instrument_id'])
    op.create_index('ix_calibrations_instrument_expires', 'calibrations', ['instrument_id', 'expires_at'])
    op.create_index('ix_calibrations_status', 'calibrations', ['status'])
    
    # ========================================================================
    # Materials
    # ========================================================================
    op.create_table('materials',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('material_type', sa.String(100), nullable=False),
        sa.Column('composition', postgresql.JSONB, nullable=True),
        sa.Column('properties', postgresql.JSONB, nullable=True),
        sa.Column('supplier', sa.String(255), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_materials_org_name', 'materials', ['organization_id', 'name'])
    op.create_index('ix_materials_org_type', 'materials', ['organization_id', 'material_type'])
    
    # ========================================================================
    # Samples
    # ========================================================================
    op.create_table('samples',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('material_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('materials.id'), nullable=True),
        sa.Column('lot_code', sa.String(255), nullable=True),
        sa.Column('barcode', sa.String(255), nullable=True),
        sa.Column('location', sa.String(255), nullable=True),
        sa.Column('dimensions', postgresql.JSONB, nullable=True),
        sa.Column('properties', postgresql.JSONB, nullable=True),
        sa.Column('is_deleted', sa.Boolean, nullable=False, server_default=sa.text('false')),
        sa.Column('deleted_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_samples_org_name', 'samples', ['organization_id', 'name'])
    op.create_index('ix_samples_org_barcode', 'samples', ['organization_id', 'barcode'])
    op.create_index('ix_samples_is_deleted', 'samples', ['is_deleted'])
    
    # ========================================================================
    # Wafers
    # ========================================================================
    op.create_table('wafers',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('sample_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('samples.id', ondelete='CASCADE'), nullable=False),
        sa.Column('wafer_number', sa.Integer, nullable=False),
        sa.Column('orientation', sa.String(50), nullable=True),
        sa.Column('thickness', sa.Float, nullable=True),
        sa.Column('diameter', sa.Float, nullable=True),
        sa.Column('properties', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_wafers_sample', 'wafers', ['sample_id'])
    op.create_index('ix_wafers_sample_number', 'wafers', ['sample_id', 'wafer_number'], unique=True)
    
    # ========================================================================
    # Devices
    # ========================================================================
    op.create_table('devices',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('wafer_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('wafers.id', ondelete='CASCADE'), nullable=False),
        sa.Column('die_x', sa.Integer, nullable=False),
        sa.Column('die_y', sa.Integer, nullable=False),
        sa.Column('device_type', sa.String(100), nullable=False),
        sa.Column('design_id', sa.String(255), nullable=True),
        sa.Column('dimensions', postgresql.JSONB, nullable=True),
        sa.Column('properties', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_devices_wafer', 'devices', ['wafer_id'])
    op.create_index('ix_devices_wafer_position', 'devices', ['wafer_id', 'die_x', 'die_y'], unique=True)
    
    # ========================================================================
    # Recipes
    # ========================================================================
    op.create_table('recipes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('method', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('status', sa.Enum('draft', 'approved', 'retired', name='recipe_status'), nullable=False, server_default='draft'),
        sa.Column('parameters', postgresql.JSONB, nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('created_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_recipes_org_name', 'recipes', ['organization_id', 'name'])
    op.create_index('ix_recipes_org_status', 'recipes', ['organization_id', 'status'])
    op.create_index('ix_recipes_method', 'recipes', ['method'])
    
    # ========================================================================
    # Recipe Approvals
    # ========================================================================
    op.create_table('recipe_approvals',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('recipe_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('recipes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('approver_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('approver_role', sa.Enum('admin', 'pi', 'engineer', 'technician', 'viewer', name='user_role'), nullable=False),
        sa.Column('state', sa.Enum('pending', 'approved', 'rejected', name='approval_state'), nullable=False, server_default='pending'),
        sa.Column('comments', sa.Text, nullable=True),
        sa.Column('approved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("approver_role IN ('admin', 'pi')", name='check_approver_role')
    )
    op.create_index('ix_recipe_approvals_recipe', 'recipe_approvals', ['recipe_id'])
    op.create_index('ix_recipe_approvals_approver', 'recipe_approvals', ['approver_id'])
    
    # ========================================================================
    # Runs
    # ========================================================================
    op.create_table('runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id'), nullable=False),
        sa.Column('sample_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('samples.id'), nullable=True),
        sa.Column('recipe_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('recipes.id'), nullable=True),
        sa.Column('method', sa.String(100), nullable=False),
        sa.Column('status', sa.Enum('queued', 'running', 'succeeded', 'failed', 'blocked', name='run_status'), nullable=False, server_default='queued'),
        sa.Column('operator_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('parameters', postgresql.JSONB, nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('finished_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('blocked_reason', sa.Text, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint('finished_at IS NULL OR finished_at >= started_at', name='check_run_times')
    )
    op.create_index('ix_runs_org_status', 'runs', ['organization_id', 'status'])
    op.create_index('ix_runs_instrument', 'runs', ['instrument_id'])
    op.create_index('ix_runs_sample', 'runs', ['sample_id'])
    op.create_index('ix_runs_created_at', 'runs', ['created_at'])
    
    # ========================================================================
    # Results
    # ========================================================================
    op.create_table('results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('result_type', sa.String(100), nullable=False),
        sa.Column('data', postgresql.JSONB, nullable=False),
        sa.Column('metrics', postgresql.JSONB, nullable=True),
        sa.Column('file_path', sa.String(500), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_results_run', 'results', ['run_id'])
    op.create_index('ix_results_type', 'results', ['result_type'])
    
    # ========================================================================
    # Attachments
    # ========================================================================
    op.create_table('attachments',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('filename', sa.String(500), nullable=False),
        sa.Column('mime_type', sa.String(255), nullable=False),
        sa.Column('size_bytes', sa.BigInteger, nullable=False),
        sa.Column('storage_path', sa.String(1000), nullable=False),
        sa.Column('sha256_hash', sa.String(64), nullable=False),
        sa.Column('uploaded_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_attachments_org', 'attachments', ['organization_id'])
    op.create_index('ix_attachments_hash', 'attachments', ['sha256_hash'])
    
    # ========================================================================
    # ELN Entries
    # ========================================================================
    op.create_table('eln_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('author_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('body_markdown', sa.Text, nullable=False),
        sa.Column('tags', postgresql.ARRAY(sa.String), nullable=True),
        sa.Column('linked_runs', postgresql.ARRAY(postgresql.UUID), nullable=True),
        sa.Column('linked_samples', postgresql.ARRAY(postgresql.UUID), nullable=True),
        sa.Column('is_signed', sa.Boolean, nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_eln_entries_org', 'eln_entries', ['organization_id'])
    op.create_index('ix_eln_entries_author', 'eln_entries', ['author_id'])
    op.create_index('ix_eln_entries_body_fts', 'eln_entries', [sa.text("to_tsvector('english', body_markdown)")], postgresql_using='gin')
    
    # ========================================================================
    # Signatures
    # ========================================================================
    op.create_table('signatures',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('eln_entry_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('eln_entries.id', ondelete='CASCADE'), nullable=False),
        sa.Column('signer_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('signature_hash', sa.String(255), nullable=False),
        sa.Column('content_hash', sa.String(64), nullable=False),
        sa.Column('signed_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_signatures_entry', 'signatures', ['eln_entry_id'])
    op.create_index('ix_signatures_signer', 'signatures', ['signer_id'])
    
    # ========================================================================
    # SOPs
    # ========================================================================
    op.create_table('sops',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('title', sa.String(500), nullable=False),
        sa.Column('sop_id', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('method', sa.String(100), nullable=False),
        sa.Column('content_markdown', sa.Text, nullable=False),
        sa.Column('checklist', postgresql.JSONB, nullable=True),
        sa.Column('approval_status', sa.String(50), nullable=False, server_default='draft'),
        sa.Column('effective_date', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('review_date', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('attachment_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('attachments.id'), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_sops_org', 'sops', ['organization_id'])
    op.create_index('ix_sops_org_sop_id', 'sops', ['organization_id', 'sop_id'])
    op.create_index('ix_sops_method', 'sops', ['method'])
    
    # ========================================================================
    # Custody Events
    # ========================================================================
    op.create_table('custody_events',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('sample_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('samples.id', ondelete='CASCADE'), nullable=False),
        sa.Column('event_type', sa.String(100), nullable=False),
        sa.Column('from_user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('to_user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('from_location', sa.String(255), nullable=True),
        sa.Column('to_location', sa.String(255), nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_custody_events_sample', 'custody_events', ['sample_id'])
    op.create_index('ix_custody_events_created', 'custody_events', ['created_at'])
    
    # ========================================================================
    # SPC Series
    # ========================================================================
    op.create_table('spc_series',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('metric_name', sa.String(255), nullable=False),
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id'), nullable=True),
        sa.Column('method', sa.String(100), nullable=False),
        sa.Column('chart_type', sa.String(50), nullable=False),
        sa.Column('control_limits', postgresql.JSONB, nullable=False),
        sa.Column('rules', postgresql.JSONB, nullable=True),
        sa.Column('is_active', sa.Boolean, nullable=False, server_default=sa.text('true')),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_spc_series_org_name', 'spc_series', ['organization_id', 'name'])
    op.create_index('ix_spc_series_instrument', 'spc_series', ['instrument_id'])
    
    # ========================================================================
    # SPC Points
    # ========================================================================
    op.create_table('spc_points',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('series_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('spc_series.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('runs.id'), nullable=True),
        sa.Column('ts', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('value', sa.Float, nullable=False),
        sa.Column('is_outlier', sa.Boolean, nullable=False, server_default=sa.text('false')),
        sa.Column('triggered_rules', postgresql.ARRAY(sa.String), nullable=True),
    )
    op.create_index('ix_spc_points_series', 'spc_points', ['series_id'])
    op.create_index('ix_spc_points_series_ts', 'spc_points', ['series_id', sa.text('ts DESC')])
    
    # ========================================================================
    # SPC Alerts
    # ========================================================================
    op.create_table('spc_alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('series_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('spc_series.id', ondelete='CASCADE'), nullable=False),
        sa.Column('alert_type', sa.String(100), nullable=False),
        sa.Column('severity', sa.String(50), nullable=False),
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('triggered_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('acknowledged_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('acknowledged_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('resolved_at', sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_spc_alerts_series', 'spc_alerts', ['series_id'])
    op.create_index('ix_spc_alerts_triggered', 'spc_alerts', ['triggered_at'])
    
    # ========================================================================
    # Feature Sets
    # ========================================================================
    op.create_table('feature_sets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('feature_names', postgresql.ARRAY(sa.String), nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_feature_sets_org', 'feature_sets', ['organization_id'])
    
    # ========================================================================
    # ML Models
    # ========================================================================
    op.create_table('ml_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id'), nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('model_type', sa.String(100), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('artifact_path', sa.String(1000), nullable=False),
        sa.Column('metrics', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index('ix_ml_models_org', 'ml_models', ['organization_id'])
    op.create_index('ix_ml_models_type', 'ml_models', ['model_type'])


def downgrade() -> None:
    """Drop all tables in reverse order."""
    op.drop_table('ml_models')
    op.drop_table('feature_sets')
    op.drop_table('spc_alerts')
    op.drop_table('spc_points')
    op.drop_table('spc_series')
    op.drop_table('custody_events')
    op.drop_table('sops')
    op.drop_table('signatures')
    op.drop_table('eln_entries')
    op.drop_table('attachments')
    op.drop_table('results')
    op.drop_table('runs')
    op.drop_table('recipe_approvals')
    op.drop_table('recipes')
    op.drop_table('devices')
    op.drop_table('wafers')
    op.drop_table('samples')
    op.drop_table('materials')
    op.drop_table('calibrations')
    op.drop_table('instruments')
    op.drop_table('api_keys')
    op.drop_table('users')
    op.drop_table('organizations')
    
    # Drop enums
    op.execute('DROP TYPE IF EXISTS user_role')
    op.execute('DROP TYPE IF EXISTS instrument_status')
    op.execute('DROP TYPE IF EXISTS calibration_status')
    op.execute('DROP TYPE IF EXISTS recipe_status')
    op.execute('DROP TYPE IF EXISTS approval_state')
    op.execute('DROP TYPE IF EXISTS run_status')
