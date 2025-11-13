"""Add CVD module tables

Revision ID: 0001_cvd_module
Revises:
Create Date: 2024-11-13

This migration adds comprehensive CVD support for all variants:
- Process modes (APCVD, LPCVD, UHVCVD, PECVD, HDP-CVD, etc.)
- Recipes with multi-step support
- Runs with telemetry
- Results with SPC/VM integration
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from datetime import datetime

# revision identifiers
revision = '0001_cvd_module'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create CVD module tables"""

    # Enum types for CVD classifications
    pressure_mode_enum = postgresql.ENUM(
        'APCVD', 'LPCVD', 'UHVCVD', 'PECVD', 'HDP_CVD', 'SACVD',
        name='pressure_mode_enum', create_type=True
    )

    energy_mode_enum = postgresql.ENUM(
        'thermal', 'plasma', 'hot_wire', 'laser', 'photo',
        'microwave', 'remote_plasma', 'combustion',
        name='energy_mode_enum', create_type=True
    )

    reactor_type_enum = postgresql.ENUM(
        'cold_wall', 'hot_wall', 'horizontal', 'vertical',
        'pancake', 'showerhead', 'rotating_disk', 'cold_finger',
        name='reactor_type_enum', create_type=True
    )

    chemistry_type_enum = postgresql.ENUM(
        'MOCVD', 'OMCVD', 'HCVD', 'hydride', 'AACVD',
        'standard', 'organometallic', 'halide',
        name='chemistry_type_enum', create_type=True
    )

    run_status_enum = postgresql.ENUM(
        'queued', 'running', 'succeeded', 'failed',
        'aborted', 'blocked', 'paused',
        name='run_status_enum', create_type=True
    )

    # Create cvd_process_modes table
    op.create_table(
        'cvd_process_modes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('pressure_mode', pressure_mode_enum, nullable=False),
        sa.Column('energy_mode', energy_mode_enum, nullable=False),
        sa.Column('reactor_type', reactor_type_enum, nullable=False),
        sa.Column('chemistry_type', chemistry_type_enum, nullable=False),
        sa.Column('variant', sa.String(100), nullable=True),  # ALCVD, pulsed, hybrid, etc.
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('default_recipes', postgresql.JSONB, nullable=True),
        sa.Column('capabilities', postgresql.JSONB, nullable=True),  # Max temp, pressure ranges, etc.
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
        sa.UniqueConstraint('org_id', 'name', name='uq_cvd_process_modes_org_name')
    )

    # Create cvd_recipes table
    op.create_table(
        'cvd_recipes',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('version', sa.Integer, nullable=False, default=1),
        sa.Column('process_mode_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_process_modes.id'), nullable=False),

        # Film targets
        sa.Column('film_target', sa.String(100), nullable=False),  # SiO2, Si3N4, GaN, etc.
        sa.Column('thickness_target_nm', sa.Float, nullable=False),
        sa.Column('uniformity_target_pct', sa.Float, nullable=False),

        # Process parameters (JSONB for flexibility across variants)
        sa.Column('temperature_profile', postgresql.JSONB, nullable=False),
        # e.g., {"zones": [{"zone": 1, "setpoint": 800, "ramp_rate": 10}], "profile": "isothermal"}

        sa.Column('pressure_setpoints', postgresql.JSONB, nullable=False),
        # e.g., {"steps": [{"step": 1, "pressure_torr": 10, "duration_s": 120}]}

        sa.Column('gas_flows', postgresql.JSONB, nullable=False),
        # e.g., {"gases": [{"name": "SiH4", "flow_sccm": 100, "timing": "continuous"}]}

        # Plasma settings (for PECVD, HDP, MPCVD, etc.)
        sa.Column('plasma_settings', postgresql.JSONB, nullable=True),
        # e.g., {"rf_power_w": 300, "frequency_mhz": 13.56, "bias_v": -100, "duty_cycle": 1.0}

        # Hot-wire settings (for HWCVD)
        sa.Column('filament_settings', postgresql.JSONB, nullable=True),
        # e.g., {"filament_temp_c": 1800, "filament_material": "tungsten"}

        # Laser settings (for LCVD, LICVD)
        sa.Column('laser_settings', postgresql.JSONB, nullable=True),
        # e.g., {"wavelength_nm": 308, "power_w": 100, "pulse_rate_hz": 10}

        # Pulsing/Sequential settings (for ALCVD, pulsed CVD, hybrid)
        sa.Column('pulsing_scheme', postgresql.JSONB, nullable=True),
        # e.g., {"cycles": [{"precursor_time": 0.5, "purge_time": 1.0, "repeat": 100}]}

        # Recipe steps (multi-step recipes)
        sa.Column('recipe_steps', postgresql.JSONB, nullable=False),
        # e.g., [{"name": "preheat", "duration_s": 60, ...}, {"name": "deposition", ...}]

        # Safety and compliance
        sa.Column('safety_hazard_level', sa.Integer, nullable=False, default=1),  # 1-5 scale
        sa.Column('required_interlocks', postgresql.JSONB, nullable=True),

        # Metadata
        sa.Column('status', sa.String(50), nullable=False, default='draft'),  # draft, approved, retired
        sa.Column('approval_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('approved_by_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_by_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),

        sa.Index('idx_cvd_recipes_org_name', 'org_id', 'name'),
        sa.Index('idx_cvd_recipes_process_mode', 'process_mode_id'),
        sa.UniqueConstraint('org_id', 'name', 'version', name='uq_cvd_recipes_org_name_version')
    )

    # Create cvd_runs table
    op.create_table(
        'cvd_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('run_number', sa.String(100), nullable=False, unique=True),

        # Relationships
        sa.Column('cvd_recipe_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_recipes.id'), nullable=False),
        sa.Column('process_mode_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_process_modes.id'), nullable=False),
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), nullable=False),  # FK to instruments table
        sa.Column('sample_id', postgresql.UUID(as_uuid=True), nullable=True),  # FK to LIMS samples
        sa.Column('wafer_id', postgresql.UUID(as_uuid=True), nullable=True),

        # Run status
        sa.Column('status', run_status_enum, nullable=False, default='queued'),
        sa.Column('start_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_seconds', sa.Float, nullable=True),

        # Pre/post flight
        sa.Column('preflight_summary', postgresql.JSONB, nullable=True),
        # e.g., {"sop_checks": {"ok": true}, "calibration_valid": true, "chamber_condition": "clean"}

        sa.Column('postflight_summary', postgresql.JSONB, nullable=True),
        # e.g., {"thickness_achieved": 98.5, "uniformity": 2.1, "defects_observed": false}

        # Error tracking
        sa.Column('error_code', sa.String(100), nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('fault_data', postgresql.JSONB, nullable=True),

        # Job queue integration
        sa.Column('celery_task_id', sa.String(255), nullable=True, index=True),
        sa.Column('job_progress', sa.Float, nullable=True, default=0.0),

        # Metadata
        sa.Column('operator_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('metadata', postgresql.JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),

        sa.Index('idx_cvd_runs_status', 'status'),
        sa.Index('idx_cvd_runs_start_time', 'start_time'),
        sa.Index('idx_cvd_runs_recipe', 'cvd_recipe_id'),
    )

    # Create cvd_telemetry table (TimescaleDB hypertable if available)
    op.create_table(
        'cvd_telemetry',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('cvd_run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_runs.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False, index=True),

        # Core measurements
        sa.Column('chamber_pressure_torr', sa.Float, nullable=True),
        sa.Column('temperature_zones_c', postgresql.JSONB, nullable=True),  # {"zone1": 800, "zone2": 805, ...}
        sa.Column('gas_flows_sccm', postgresql.JSONB, nullable=True),  # {"SiH4": 100.2, "N2": 5001.5, ...}

        # Plasma measurements (PECVD, HDP, MPCVD, etc.)
        sa.Column('rf_power_w', sa.Float, nullable=True),
        sa.Column('rf_reflected_w', sa.Float, nullable=True),
        sa.Column('dc_bias_v', sa.Float, nullable=True),
        sa.Column('plasma_impedance_ohm', sa.Float, nullable=True),
        sa.Column('microwave_power_w', sa.Float, nullable=True),

        # Hot-wire measurements (HWCVD)
        sa.Column('filament_temp_c', sa.Float, nullable=True),
        sa.Column('filament_current_a', sa.Float, nullable=True),

        # Laser measurements (LCVD, LICVD)
        sa.Column('laser_power_w', sa.Float, nullable=True),
        sa.Column('laser_energy_mj', sa.Float, nullable=True),

        # In-situ metrology
        sa.Column('qcm_rate_nm_per_s', sa.Float, nullable=True),
        sa.Column('qcm_thickness_nm', sa.Float, nullable=True),
        sa.Column('ellipsometer_thickness_nm', sa.Float, nullable=True),
        sa.Column('reflectometer_thickness_nm', sa.Float, nullable=True),

        # Derived/calculated values
        sa.Column('deposition_rate_nm_per_min', sa.Float, nullable=True),
        sa.Column('plasma_density_proxy', sa.Float, nullable=True),

        # Raw data references
        sa.Column('rga_spectrum_uri', sa.String(500), nullable=True),
        sa.Column('oes_spectrum_uri', sa.String(500), nullable=True),

        # Other signals (flexible)
        sa.Column('other_signals', postgresql.JSONB, nullable=True),

        sa.Index('idx_cvd_telemetry_ts', 'cvd_run_id', 'ts'),
    )

    # Create cvd_results table
    op.create_table(
        'cvd_results',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('cvd_run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_runs.id'), nullable=False, unique=True),

        # Film properties
        sa.Column('film_material', sa.String(100), nullable=False),
        sa.Column('thickness_mean_nm', sa.Float, nullable=True),
        sa.Column('thickness_std_nm', sa.Float, nullable=True),
        sa.Column('thickness_uniformity_pct', sa.Float, nullable=True),  # (std/mean)*100
        sa.Column('thickness_min_nm', sa.Float, nullable=True),
        sa.Column('thickness_max_nm', sa.Float, nullable=True),
        sa.Column('thickness_map', postgresql.JSONB, nullable=True),  # Full wafer map

        # Optical properties
        sa.Column('refractive_index', sa.Float, nullable=True),
        sa.Column('extinction_coefficient', sa.Float, nullable=True),

        # Mechanical properties
        sa.Column('stress_mpa', sa.Float, nullable=True),
        sa.Column('stress_type', sa.String(50), nullable=True),  # tensile, compressive

        # Film quality
        sa.Column('conformality_score', sa.Float, nullable=True),  # 0-1 scale
        sa.Column('selectivity_score', sa.Float, nullable=True),  # For SACVD
        sa.Column('step_coverage_pct', sa.Float, nullable=True),
        sa.Column('defect_density_per_cm2', sa.Float, nullable=True),
        sa.Column('roughness_rms_nm', sa.Float, nullable=True),

        # Composition (for MOCVD/alloys)
        sa.Column('composition', postgresql.JSONB, nullable=True),  # {"In": 0.2, "Ga": 0.8, "N": 1.0}

        # VM predictions
        sa.Column('vm_predictions', postgresql.JSONB, nullable=True),
        # e.g., {"predicted_thickness": 98.2, "confidence": 0.95, "model_version": "v1.2"}

        # SPC snapshot
        sa.Column('spc_snapshot', postgresql.JSONB, nullable=True),
        # e.g., {"cp": 1.45, "cpk": 1.32, "violations": []}

        # References
        sa.Column('report_uri', sa.String(500), nullable=True),
        sa.Column('raw_data_uri', sa.String(500), nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),

        sa.Index('idx_cvd_results_run', 'cvd_run_id'),
        sa.Index('idx_cvd_results_created', 'created_at'),
    )

    # Create cvd_spc_series table (for SPC chart data)
    op.create_table(
        'cvd_spc_series',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('parameter', sa.String(100), nullable=False),  # thickness, uniformity, pressure, etc.
        sa.Column('process_mode_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_process_modes.id'), nullable=True),
        sa.Column('instrument_id', postgresql.UUID(as_uuid=True), nullable=True),

        # Control limits
        sa.Column('target', sa.Float, nullable=True),
        sa.Column('ucl', sa.Float, nullable=True),
        sa.Column('lcl', sa.Float, nullable=True),
        sa.Column('usl', sa.Float, nullable=True),  # Specification limits
        sa.Column('lsl', sa.Float, nullable=True),

        # Statistics
        sa.Column('mean', sa.Float, nullable=True),
        sa.Column('std_dev', sa.Float, nullable=True),
        sa.Column('cp', sa.Float, nullable=True),
        sa.Column('cpk', sa.Float, nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )

    # Create cvd_spc_points table
    op.create_table(
        'cvd_spc_points',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('org_id', postgresql.UUID(as_uuid=True), nullable=False, index=True),
        sa.Column('series_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_spc_series.id'), nullable=False),
        sa.Column('cvd_run_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('cvd_runs.id'), nullable=True),
        sa.Column('value', sa.Float, nullable=False),
        sa.Column('ts', sa.DateTime(timezone=True), nullable=False),
        sa.Column('violation', sa.Boolean, nullable=False, default=False),
        sa.Column('violation_rules', postgresql.JSONB, nullable=True),  # Which rules triggered

        sa.Index('idx_cvd_spc_points_series_ts', 'series_id', 'ts'),
    )


def downgrade() -> None:
    """Drop CVD module tables"""
    op.drop_table('cvd_spc_points')
    op.drop_table('cvd_spc_series')
    op.drop_table('cvd_results')
    op.drop_table('cvd_telemetry')
    op.drop_table('cvd_runs')
    op.drop_table('cvd_recipes')
    op.drop_table('cvd_process_modes')

    # Drop enums
    op.execute('DROP TYPE IF EXISTS run_status_enum')
    op.execute('DROP TYPE IF EXISTS chemistry_type_enum')
    op.execute('DROP TYPE IF EXISTS reactor_type_enum')
    op.execute('DROP TYPE IF EXISTS energy_mode_enum')
    op.execute('DROP TYPE IF EXISTS pressure_mode_enum')
