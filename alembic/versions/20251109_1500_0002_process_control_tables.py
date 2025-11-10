"""Add Process Control tables for RTP and Ion Implantation

Revision ID: 20251109_1500_0002
Revises: 20251026_1200_0001
Create Date: 2025-11-09 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20251109_1500_0002'
down_revision = '20251026_1200_0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ========================================================================
    # Update SPC Tables
    # ========================================================================

    # Add new columns to spc_series
    op.add_column('spc_series', sa.Column('chart_type', sa.String(20), server_default='I-MR'))
    op.add_column('spc_series', sa.Column('control_limits', postgresql.JSONB, server_default='{}'))
    op.add_column('spc_series', sa.Column('spec_limits', postgresql.JSONB, server_default='{}'))
    op.add_column('spc_series', sa.Column('rules', postgresql.JSONB, server_default='{}'))
    op.add_column('spc_series', sa.Column('window_size', sa.Integer, nullable=True))
    op.add_column('spc_series', sa.Column('ewma_lambda', sa.Float, nullable=True))

    # Add new columns to spc_points
    op.add_column('spc_points', sa.Column('run_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('spc_points', sa.Column('subgroup_values', postgresql.ARRAY(sa.Float), nullable=True))
    op.add_column('spc_points', sa.Column('moving_range', sa.Float, nullable=True))
    op.add_column('spc_points', sa.Column('ewma_value', sa.Float, nullable=True))
    op.add_column('spc_points', sa.Column('cusum_pos', sa.Float, nullable=True))
    op.add_column('spc_points', sa.Column('cusum_neg', sa.Float, nullable=True))
    op.add_column('spc_points', sa.Column('violations', postgresql.JSONB, server_default='{}'))

    # Add foreign key for run_id in spc_points
    op.create_foreign_key('fk_spc_points_run', 'spc_points', 'runs', ['run_id'], ['id'])
    op.create_index('idx_spc_points_run', 'spc_points', ['run_id'])

    # Add new columns to spc_alerts
    op.add_column('spc_alerts', sa.Column('point_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('spc_alerts', sa.Column('alert_type', sa.String(50), server_default='violation'))
    op.add_column('spc_alerts', sa.Column('rule_violated', sa.String(100), server_default=''))
    op.add_column('spc_alerts', sa.Column('description', sa.Text, nullable=True))
    op.add_column('spc_alerts', sa.Column('acknowledged', sa.Boolean, server_default='false'))
    op.add_column('spc_alerts', sa.Column('acknowledged_by', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('spc_alerts', sa.Column('acknowledged_at', sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column('spc_alerts', sa.Column('resolution_notes', sa.Text, nullable=True))

    # Add foreign keys for spc_alerts
    op.create_foreign_key('fk_spc_alerts_point', 'spc_alerts', 'spc_points', ['point_id'], ['id'], ondelete='CASCADE')
    op.create_foreign_key('fk_spc_alerts_ack_user', 'spc_alerts', 'users', ['acknowledged_by'], ['id'])
    op.create_index('idx_spc_alerts_acknowledged', 'spc_alerts', ['acknowledged'])

    # ========================================================================
    # Ion Implantation Tables
    # ========================================================================

    # implant_dose_profiles table
    op.create_table(
        'implant_dose_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('runs.id'), nullable=False),

        # Ion beam parameters
        sa.Column('ion_species', sa.String(10), nullable=False),
        sa.Column('isotope', sa.Integer, nullable=True),
        sa.Column('energy_keV', sa.Float, nullable=False),
        sa.Column('tilt_deg', sa.Float, nullable=False),
        sa.Column('twist_deg', sa.Float, nullable=False),
        sa.Column('dose_cm2', sa.Float, nullable=False),

        # SRIM results
        sa.Column('projected_range_nm', sa.Float, nullable=True),
        sa.Column('straggle_nm', sa.Float, nullable=True),
        sa.Column('channeling_metric', sa.Float, nullable=True),

        # Extended metrics
        sa.Column('damage_metrics', postgresql.JSONB, server_default='{}'),
        sa.Column('beam_uniformity', postgresql.JSONB, server_default='{}'),
        sa.Column('wafer_map_uri', sa.String(500), nullable=True),
        sa.Column('sims_profile_uri', sa.String(500), nullable=True),

        # Timestamps
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('is_deleted', sa.Boolean, server_default='false'),
        sa.Column('deleted_at', sa.TIMESTAMP(timezone=True), nullable=True),
    )

    op.create_index('idx_implant_dose_org_run', 'implant_dose_profiles', ['organization_id', 'run_id'])
    op.create_index('idx_implant_dose_species', 'implant_dose_profiles', ['ion_species'])
    op.create_index('idx_implant_dose_deleted', 'implant_dose_profiles', ['is_deleted'])

    # implant_telemetry table
    op.create_table(
        'implant_telemetry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('profile_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('implant_dose_profiles.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('runs.id'), nullable=False),
        sa.Column('ts', sa.TIMESTAMP(timezone=True), nullable=False),

        # Beam parameters
        sa.Column('beam_current_mA', sa.Float, nullable=False),
        sa.Column('pressure_mTorr', sa.Float, nullable=False),
        sa.Column('accel_voltage_kV', sa.Float, nullable=False),
        sa.Column('analyzer_magnet_T', sa.Float, nullable=True),

        # Beam steering
        sa.Column('steering_X', sa.Float, nullable=True),
        sa.Column('steering_Y', sa.Float, nullable=True),

        # Dose integration
        sa.Column('dose_count_C_cm2', sa.Float, nullable=False),

        # Diagnostics
        sa.Column('beam_profile_uri', sa.String(500), nullable=True),
        sa.Column('faraday_currents', postgresql.ARRAY(sa.Float), nullable=True),
        sa.Column('gas_flows', postgresql.JSONB, server_default='{}'),
        sa.Column('extra_metadata', postgresql.JSONB, server_default='{}'),
    )

    op.create_index('idx_implant_telem_profile', 'implant_telemetry', ['profile_id'])
    op.create_index('idx_implant_telem_run_ts', 'implant_telemetry', ['run_id', 'ts'])

    # ========================================================================
    # RTP Tables
    # ========================================================================

    # rtp_profiles table
    op.create_table(
        'rtp_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('runs.id'), nullable=False),

        # Temperature profile
        sa.Column('recipe_curve', postgresql.JSONB, nullable=False),
        sa.Column('peak_T_C', sa.Float, nullable=False),

        # Process parameters
        sa.Column('ambient_gas', sa.String(50), nullable=False),
        sa.Column('pressure_Torr', sa.Float, nullable=False),
        sa.Column('emissivity', sa.Float, nullable=False),
        sa.Column('pyrometer_cal_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('calibrations.id'), nullable=True),

        # Multi-zone control
        sa.Column('zone_setpoints', postgresql.JSONB, server_default='{}'),
        sa.Column('uniformity_metrics', postgresql.JSONB, server_default='{}'),
        sa.Column('wafer_rotation_rpm', sa.Float, nullable=True),

        # Timestamps
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.Column('is_deleted', sa.Boolean, server_default='false'),
        sa.Column('deleted_at', sa.TIMESTAMP(timezone=True), nullable=True),
    )

    op.create_index('idx_rtp_profile_org_run', 'rtp_profiles', ['organization_id', 'run_id'])
    op.create_index('idx_rtp_profile_deleted', 'rtp_profiles', ['is_deleted'])
    op.execute("CREATE INDEX idx_rtp_recipe_gin ON rtp_profiles USING gin (recipe_curve)")

    # rtp_telemetry table
    op.create_table(
        'rtp_telemetry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('profile_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('rtp_profiles.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('runs.id'), nullable=False),
        sa.Column('ts', sa.TIMESTAMP(timezone=True), nullable=False),

        # Temperature measurements
        sa.Column('setpoint_T_C', sa.Float, nullable=False),
        sa.Column('pyrometer_T_C', sa.Float, nullable=False),
        sa.Column('tc_T_C', postgresql.ARRAY(sa.Float), nullable=True),

        # Lamp control
        sa.Column('lamp_power_pct', postgresql.ARRAY(sa.Float), nullable=False),

        # Process parameters
        sa.Column('emissivity_used', sa.Float, nullable=False),
        sa.Column('chamber_pressure_Torr', sa.Float, nullable=False),
        sa.Column('flow_sccm', postgresql.JSONB, nullable=False),

        # Controller states
        sa.Column('pid_state', postgresql.JSONB, server_default='{}'),
        sa.Column('mpc_state', postgresql.JSONB, server_default='{}'),
        sa.Column('extra_metadata', postgresql.JSONB, server_default='{}'),
    )

    op.create_index('idx_rtp_telem_profile', 'rtp_telemetry', ['profile_id'])
    op.create_index('idx_rtp_telem_run_ts', 'rtp_telemetry', ['run_id', 'ts'])
    op.execute("CREATE INDEX idx_rtp_flow_gin ON rtp_telemetry USING gin (flow_sccm)")

    # ========================================================================
    # Virtual Metrology Tables
    # ========================================================================

    # vm_feature_sets table
    op.create_table(
        'vm_feature_sets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('instruments.id'), nullable=True),

        # Feature engineering
        sa.Column('features', postgresql.JSONB, nullable=False),
        sa.Column('target_metrics', postgresql.JSONB, nullable=False),
        sa.Column('preprocessing', postgresql.JSONB, server_default='{}'),

        sa.Column('active', sa.Boolean, server_default='true'),
        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_index('idx_vm_features_org', 'vm_feature_sets', ['organization_id'])
    op.create_index('idx_vm_features_active', 'vm_feature_sets', ['active'])

    # vm_models table
    op.create_table(
        'vm_models',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('organization_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('feature_set_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('vm_feature_sets.id', ondelete='CASCADE'), nullable=False),

        # Model identification
        sa.Column('name', sa.String(200), nullable=False),
        sa.Column('version', sa.String(50), nullable=False),
        sa.Column('model_type', sa.String(50), nullable=False),

        # Model storage
        sa.Column('model_uri', sa.String(500), nullable=True),
        sa.Column('hyperparameters', postgresql.JSONB, server_default='{}'),
        sa.Column('performance_metrics', postgresql.JSONB, server_default='{}'),

        # Training data
        sa.Column('training_runs', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('validation_runs', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),

        # Deployment status
        sa.Column('deployed', sa.Boolean, server_default='false'),
        sa.Column('approved_by', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('approved_at', sa.TIMESTAMP(timezone=True), nullable=True),

        sa.Column('created_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.TIMESTAMP(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    op.create_index('idx_vm_models_org', 'vm_models', ['organization_id'])
    op.create_index('idx_vm_models_deployed', 'vm_models', ['deployed'])


def downgrade() -> None:
    # Drop VM tables
    op.drop_table('vm_models')
    op.drop_table('vm_feature_sets')

    # Drop RTP tables
    op.drop_table('rtp_telemetry')
    op.drop_table('rtp_profiles')

    # Drop Ion Implantation tables
    op.drop_table('implant_telemetry')
    op.drop_table('implant_dose_profiles')

    # Remove SPC table additions
    op.drop_column('spc_alerts', 'resolution_notes')
    op.drop_column('spc_alerts', 'acknowledged_at')
    op.drop_column('spc_alerts', 'acknowledged_by')
    op.drop_column('spc_alerts', 'acknowledged')
    op.drop_column('spc_alerts', 'description')
    op.drop_column('spc_alerts', 'rule_violated')
    op.drop_column('spc_alerts', 'alert_type')
    op.drop_column('spc_alerts', 'point_id')

    op.drop_column('spc_points', 'violations')
    op.drop_column('spc_points', 'cusum_neg')
    op.drop_column('spc_points', 'cusum_pos')
    op.drop_column('spc_points', 'ewma_value')
    op.drop_column('spc_points', 'moving_range')
    op.drop_column('spc_points', 'subgroup_values')
    op.drop_column('spc_points', 'run_id')

    op.drop_column('spc_series', 'ewma_lambda')
    op.drop_column('spc_series', 'window_size')
    op.drop_column('spc_series', 'rules')
    op.drop_column('spc_series', 'spec_limits')
    op.drop_column('spc_series', 'control_limits')
    op.drop_column('spc_series', 'chart_type')
