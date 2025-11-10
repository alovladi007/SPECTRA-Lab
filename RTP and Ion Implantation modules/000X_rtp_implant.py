"""Add RTP and Ion Implantation modules

Revision ID: 000X_rtp_implant
Revises: previous_migration
Create Date: 2024-11-09 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

# revision identifiers
revision = '000X_rtp_implant'
down_revision = 'previous_migration'
branch_labels = None
depends_on = None


def upgrade():
    # Ion Implantation: Dose Profiles table
    op.create_table(
        'implant_dose_profiles',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.BigInteger(), nullable=False),
        sa.Column('ion_species', sa.String(10), nullable=False),
        sa.Column('isotope', sa.Integer(), nullable=True),
        sa.Column('energy_keV', sa.Float(), nullable=False),
        sa.Column('tilt_deg', sa.Float(), nullable=False),
        sa.Column('twist_deg', sa.Float(), nullable=False),
        sa.Column('dose_cm2', sa.Float(), nullable=False),
        sa.Column('projected_range_nm', sa.Float(), nullable=True),
        sa.Column('straggle_nm', sa.Float(), nullable=True),
        sa.Column('channeling_metric', sa.Float(), nullable=True),
        sa.Column('damage_metrics', postgresql.JSONB(), nullable=True),
        sa.Column('beam_uniformity', postgresql.JSONB(), nullable=True),
        sa.Column('wafer_map_uri', sa.String(500), nullable=True),
        sa.Column('sims_profile_uri', sa.String(500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id']),
    )
    
    # Ion Implantation: Telemetry table
    op.create_table(
        'implant_telemetry',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('run_id', sa.BigInteger(), nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('beam_current_mA', sa.Float(), nullable=False),
        sa.Column('pressure_mTorr', sa.Float(), nullable=False),
        sa.Column('accel_voltage_kV', sa.Float(), nullable=False),
        sa.Column('analyzer_magnet_T', sa.Float(), nullable=True),
        sa.Column('steering_X', sa.Float(), nullable=True),
        sa.Column('steering_Y', sa.Float(), nullable=True),
        sa.Column('dose_count_C_cm2', sa.Float(), nullable=False),
        sa.Column('beam_profile_uri', sa.String(500), nullable=True),
        sa.Column('faraday_currents', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('gas_flows', postgresql.JSONB(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id']),
    )
    
    # RTP: Thermal Profiles table
    op.create_table(
        'rtp_profiles',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.BigInteger(), nullable=False),
        sa.Column('recipe_curve', postgresql.JSONB(), nullable=False),  # [segments: time_s, T_C, ramp_Cps, dwell_s]
        sa.Column('peak_T_C', sa.Float(), nullable=False),
        sa.Column('ambient_gas', sa.String(50), nullable=False),
        sa.Column('pressure_Torr', sa.Float(), nullable=False),
        sa.Column('emissivity', sa.Float(), nullable=False),
        sa.Column('pyrometer_cal_id', sa.BigInteger(), nullable=True),
        sa.Column('zone_setpoints', postgresql.JSONB(), nullable=True),  # Multi-zone lamp settings
        sa.Column('uniformity_metrics', postgresql.JSONB(), nullable=True),
        sa.Column('wafer_rotation_rpm', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.Column('deleted_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id']),
        sa.ForeignKeyConstraint(['pyrometer_cal_id'], ['calibrations.id']),
    )
    
    # RTP: Telemetry table
    op.create_table(
        'rtp_telemetry',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('run_id', sa.BigInteger(), nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('setpoint_T_C', sa.Float(), nullable=False),
        sa.Column('pyrometer_T_C', sa.Float(), nullable=False),
        sa.Column('tc_T_C', postgresql.ARRAY(sa.Float()), nullable=True),  # Multiple TCs
        sa.Column('lamp_power_pct', postgresql.ARRAY(sa.Float()), nullable=False),  # Per-zone power
        sa.Column('emissivity_used', sa.Float(), nullable=False),
        sa.Column('chamber_pressure_Torr', sa.Float(), nullable=False),
        sa.Column('flow_sccm', postgresql.JSONB(), nullable=False),  # {gas_name: flow_rate}
        sa.Column('pid_state', postgresql.JSONB(), nullable=True),  # {P, I, D, error, output}
        sa.Column('mpc_state', postgresql.JSONB(), nullable=True),  # MPC controller state
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id']),
    )
    
    # SPC Series table (if not exists)
    op.create_table(
        'spc_series',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('instrument_id', sa.BigInteger(), nullable=True),
        sa.Column('parameter', sa.String(100), nullable=False),
        sa.Column('chart_type', sa.String(20), nullable=False),  # Xbar-R, EWMA, CUSUM, etc.
        sa.Column('control_limits', postgresql.JSONB(), nullable=False),
        sa.Column('spec_limits', postgresql.JSONB(), nullable=True),
        sa.Column('rules', postgresql.JSONB(), nullable=False),  # Western Electric, custom
        sa.Column('window_size', sa.Integer(), nullable=True),
        sa.Column('ewma_lambda', sa.Float(), nullable=True),
        sa.Column('active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id']),
    )
    
    # SPC Points table
    op.create_table(
        'spc_points',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('series_id', sa.BigInteger(), nullable=False),
        sa.Column('run_id', sa.BigInteger(), nullable=True),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('value', sa.Float(), nullable=False),
        sa.Column('subgroup_values', postgresql.ARRAY(sa.Float()), nullable=True),
        sa.Column('moving_range', sa.Float(), nullable=True),
        sa.Column('ewma_value', sa.Float(), nullable=True),
        sa.Column('cusum_pos', sa.Float(), nullable=True),
        sa.Column('cusum_neg', sa.Float(), nullable=True),
        sa.Column('violations', postgresql.JSONB(), nullable=True),
        sa.Column('metadata', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['series_id'], ['spc_series.id']),
        sa.ForeignKeyConstraint(['run_id'], ['runs.id']),
    )
    
    # SPC Alerts table
    op.create_table(
        'spc_alerts',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('series_id', sa.BigInteger(), nullable=False),
        sa.Column('point_id', sa.BigInteger(), nullable=False),
        sa.Column('alert_type', sa.String(50), nullable=False),
        sa.Column('severity', sa.String(20), nullable=False),  # warning, critical
        sa.Column('rule_violated', sa.String(100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('acknowledged', sa.Boolean(), default=False),
        sa.Column('acknowledged_by', sa.BigInteger(), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['series_id'], ['spc_series.id']),
        sa.ForeignKeyConstraint(['point_id'], ['spc_points.id']),
        sa.ForeignKeyConstraint(['acknowledged_by'], ['users.id']),
    )
    
    # Virtual Metrology: Feature Sets
    op.create_table(
        'vm_feature_sets',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('instrument_id', sa.BigInteger(), nullable=True),
        sa.Column('features', postgresql.JSONB(), nullable=False),  # Feature definitions
        sa.Column('target_metrics', postgresql.JSONB(), nullable=False),
        sa.Column('preprocessing', postgresql.JSONB(), nullable=True),  # Scaling, transforms
        sa.Column('active', sa.Boolean(), default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['instrument_id'], ['instruments.id']),
    )
    
    # Virtual Metrology: Models
    op.create_table(
        'vm_models',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('org_id', sa.Integer(), nullable=False),
        sa.Column('feature_set_id', sa.BigInteger(), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),  # neural, xgboost, physics, hybrid
        sa.Column('model_uri', sa.String(500), nullable=True),  # S3/Minio path
        sa.Column('hyperparameters', postgresql.JSONB(), nullable=True),
        sa.Column('performance_metrics', postgresql.JSONB(), nullable=True),  # R2, RMSE, etc.
        sa.Column('training_runs', postgresql.ARRAY(sa.BigInteger()), nullable=True),
        sa.Column('validation_runs', postgresql.ARRAY(sa.BigInteger()), nullable=True),
        sa.Column('deployed', sa.Boolean(), default=False),
        sa.Column('approved_by', sa.BigInteger(), nullable=True),
        sa.Column('approved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['org_id'], ['organizations.id']),
        sa.ForeignKeyConstraint(['feature_set_id'], ['vm_feature_sets.id']),
        sa.ForeignKeyConstraint(['approved_by'], ['users.id']),
    )
    
    # Create indexes
    op.create_index('idx_implant_dose_org_run', 'implant_dose_profiles', ['org_id', 'run_id'])
    op.create_index('idx_implant_dose_species', 'implant_dose_profiles', ['ion_species'])
    op.create_index('idx_implant_dose_deleted', 'implant_dose_profiles', ['deleted_at'], 
                    postgresql_where=sa.text('deleted_at IS NULL'))
    
    op.create_index('idx_implant_telem_run_ts', 'implant_telemetry', ['run_id', 'ts'])
    op.create_index('idx_implant_telem_ts', 'implant_telemetry', ['ts'])
    
    op.create_index('idx_rtp_profile_org_run', 'rtp_profiles', ['org_id', 'run_id'])
    op.create_index('idx_rtp_profile_deleted', 'rtp_profiles', ['deleted_at'],
                    postgresql_where=sa.text('deleted_at IS NULL'))
    
    op.create_index('idx_rtp_telem_run_ts', 'rtp_telemetry', ['run_id', 'ts'])
    op.create_index('idx_rtp_telem_ts', 'rtp_telemetry', ['ts'])
    
    op.create_index('idx_spc_series_org', 'spc_series', ['org_id'])
    op.create_index('idx_spc_series_inst', 'spc_series', ['instrument_id'])
    op.create_index('idx_spc_series_active', 'spc_series', ['active'])
    
    op.create_index('idx_spc_points_series_ts', 'spc_points', ['series_id', 'ts'])
    op.create_index('idx_spc_points_run', 'spc_points', ['run_id'])
    
    op.create_index('idx_spc_alerts_org', 'spc_alerts', ['org_id'])
    op.create_index('idx_spc_alerts_series', 'spc_alerts', ['series_id'])
    op.create_index('idx_spc_alerts_ack', 'spc_alerts', ['acknowledged'])
    
    op.create_index('idx_vm_features_org', 'vm_feature_sets', ['org_id'])
    op.create_index('idx_vm_models_org', 'vm_models', ['org_id'])
    op.create_index('idx_vm_models_deployed', 'vm_models', ['deployed'])
    
    # GIN indexes for JSONB fields
    op.create_index('idx_implant_damage_gin', 'implant_dose_profiles', ['damage_metrics'],
                    postgresql_using='gin')
    op.create_index('idx_rtp_recipe_gin', 'rtp_profiles', ['recipe_curve'],
                    postgresql_using='gin')
    op.create_index('idx_rtp_flow_gin', 'rtp_telemetry', ['flow_sccm'],
                    postgresql_using='gin')


def downgrade():
    # Drop indexes
    op.drop_index('idx_vm_models_deployed')
    op.drop_index('idx_vm_models_org')
    op.drop_index('idx_vm_features_org')
    op.drop_index('idx_spc_alerts_ack')
    op.drop_index('idx_spc_alerts_series')
    op.drop_index('idx_spc_alerts_org')
    op.drop_index('idx_spc_points_run')
    op.drop_index('idx_spc_points_series_ts')
    op.drop_index('idx_spc_series_active')
    op.drop_index('idx_spc_series_inst')
    op.drop_index('idx_spc_series_org')
    op.drop_index('idx_rtp_flow_gin')
    op.drop_index('idx_rtp_telem_ts')
    op.drop_index('idx_rtp_telem_run_ts')
    op.drop_index('idx_rtp_recipe_gin')
    op.drop_index('idx_rtp_profile_deleted')
    op.drop_index('idx_rtp_profile_org_run')
    op.drop_index('idx_implant_damage_gin')
    op.drop_index('idx_implant_telem_ts')
    op.drop_index('idx_implant_telem_run_ts')
    op.drop_index('idx_implant_dose_deleted')
    op.drop_index('idx_implant_dose_species')
    op.drop_index('idx_implant_dose_org_run')
    
    # Drop tables
    op.drop_table('vm_models')
    op.drop_table('vm_feature_sets')
    op.drop_table('spc_alerts')
    op.drop_table('spc_points')
    op.drop_table('spc_series')
    op.drop_table('rtp_telemetry')
    op.drop_table('rtp_profiles')
    op.drop_table('implant_telemetry')
    op.drop_table('implant_dose_profiles')
