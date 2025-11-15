"""CVD advanced metrics - thickness, stress, adhesion

Revision ID: 0003_cvd_advanced_metrics
Revises: 0002_process_control_tables
Create Date: 2025-11-14 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20251114_1800_0003'
down_revision = '20251109_1500_0002'
branch_labels = None
depends_on = None


def upgrade():
    """Add advanced CVD metrics for thickness, stress, and adhesion tracking"""

    # Create enum types for adhesion and stress
    adhesion_class_enum = postgresql.ENUM(
        'POOR', 'MARGINAL', 'GOOD', 'EXCELLENT',
        name='adhesion_class_enum',
        create_type=True
    )
    adhesion_class_enum.create(op.get_bind(), checkfirst=True)

    stress_type_enum = postgresql.ENUM(
        'TENSILE', 'COMPRESSIVE', 'MIXED', 'NEUTRAL',
        name='stress_type_enum',
        create_type=True
    )
    stress_type_enum.create(op.get_bind(), checkfirst=True)

    # =========================================================================
    # CVD Process Modes - Add default_targets
    # =========================================================================
    op.add_column('cvd_process_modes',
        sa.Column('default_targets', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Default film types and typical thickness/stress/adhesion ranges')
    )

    # =========================================================================
    # CVD Recipes - Add stress and adhesion targets
    # =========================================================================
    op.add_column('cvd_recipes',
        sa.Column('film_material', sa.String(length=100), nullable=True,
                  comment='Specific film material (e.g., SiO₂, Si₃N₄, TiN, GaN, DLC)')
    )

    op.add_column('cvd_recipes',
        sa.Column('target_stress_mpa', sa.Float(), nullable=True,
                  comment='Target film stress in MPa (tensile>0, compressive<0)')
    )

    op.add_column('cvd_recipes',
        sa.Column('target_stress_type', stress_type_enum, nullable=True,
                  comment='Expected stress type')
    )

    op.add_column('cvd_recipes',
        sa.Column('target_adhesion_class', adhesion_class_enum, nullable=True,
                  comment='Target adhesion class')
    )

    op.add_column('cvd_recipes',
        sa.Column('target_adhesion_score', sa.Float(), nullable=True,
                  comment='Target adhesion score (0-100)')
    )

    op.add_column('cvd_recipes',
        sa.Column('pressure_profile_torr', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Detailed pressure profile vs time')
    )

    # =========================================================================
    # CVD Results - Comprehensive thickness tracking
    # =========================================================================
    op.add_column('cvd_results',
        sa.Column('thickness_map_uri', sa.String(length=500), nullable=True,
                  comment='URI to wafer map file')
    )

    op.add_column('cvd_results',
        sa.Column('conformality_ratio', sa.Float(), nullable=True,
                  comment='Bottom/top step coverage ratio')
    )

    op.add_column('cvd_results',
        sa.Column('thickness_wiw_uniformity_pct', sa.Float(), nullable=True,
                  comment='Within-wafer uniformity percentage')
    )

    op.add_column('cvd_results',
        sa.Column('thickness_wtw_uniformity_pct', sa.Float(), nullable=True,
                  comment='Wafer-to-wafer uniformity percentage')
    )

    # =========================================================================
    # CVD Results - Advanced stress characterization
    # =========================================================================
    op.add_column('cvd_results',
        sa.Column('stress_mpa_mean', sa.Float(), nullable=True,
                  comment='Mean film stress in MPa')
    )

    op.add_column('cvd_results',
        sa.Column('stress_mpa_std', sa.Float(), nullable=True,
                  comment='Standard deviation of stress')
    )

    op.add_column('cvd_results',
        sa.Column('stress_mpa_min', sa.Float(), nullable=True,
                  comment='Minimum stress value')
    )

    op.add_column('cvd_results',
        sa.Column('stress_mpa_max', sa.Float(), nullable=True,
                  comment='Maximum stress value')
    )

    op.add_column('cvd_results',
        sa.Column('stress_distribution_uri', sa.String(length=500), nullable=True,
                  comment='URI to stress distribution map')
    )

    # Rename existing stress_type column if needed, or just ensure it uses enum
    op.alter_column('cvd_results', 'stress_type',
                   existing_type=sa.String(50),
                   type_=stress_type_enum,
                   existing_nullable=True,
                   postgresql_using='stress_type::stress_type_enum'
    )

    op.add_column('cvd_results',
        sa.Column('stress_measurement_method', sa.String(length=100), nullable=True,
                  comment='Method used (e.g., wafer_curvature_Stoney, XRD, nanoindentation)')
    )

    op.add_column('cvd_results',
        sa.Column('stress_gradient_mpa_per_nm', sa.Float(), nullable=True,
                  comment='Stress gradient through film thickness')
    )

    # =========================================================================
    # CVD Results - Adhesion characterization (NEW)
    # =========================================================================
    op.add_column('cvd_results',
        sa.Column('adhesion_score', sa.Float(), nullable=True,
                  comment='Adhesion score (0-100 scale)')
    )

    op.add_column('cvd_results',
        sa.Column('adhesion_class', adhesion_class_enum, nullable=True,
                  comment='Adhesion classification')
    )

    op.add_column('cvd_results',
        sa.Column('adhesion_test_method', sa.String(length=100), nullable=True,
                  comment='Test method (e.g., tape_test, scratch_test, four_point_bend, nanoindentation, stud_pull)')
    )

    op.add_column('cvd_results',
        sa.Column('adhesion_critical_load_n', sa.Float(), nullable=True,
                  comment='Critical load at adhesion failure (Newtons)')
    )

    op.add_column('cvd_results',
        sa.Column('adhesion_failure_mode', sa.String(length=50), nullable=True,
                  comment='Failure mode (cohesive, adhesive, interfacial, mixed)')
    )

    op.add_column('cvd_results',
        sa.Column('adhesion_test_date', sa.DateTime(timezone=True), nullable=True,
                  comment='Date adhesion test was performed')
    )

    op.add_column('cvd_results',
        sa.Column('adhesion_notes', sa.Text(), nullable=True,
                  comment='Additional adhesion test notes')
    )

    # =========================================================================
    # CVD Results - Enhanced optical properties
    # =========================================================================
    op.add_column('cvd_results',
        sa.Column('refractive_index_spectrum', postgresql.JSONB(astext_type=sa.Text()), nullable=True,
                  comment='Refractive index vs wavelength {wavelength_nm: {n: value, k: value}}')
    )

    op.add_column('cvd_results',
        sa.Column('optical_bandgap_ev', sa.Float(), nullable=True,
                  comment='Optical bandgap in eV')
    )

    # =========================================================================
    # CVD Results - Enhanced roughness characterization
    # =========================================================================
    op.add_column('cvd_results',
        sa.Column('roughness_ra_nm', sa.Float(), nullable=True,
                  comment='Average roughness (Ra) in nm')
    )

    op.add_column('cvd_results',
        sa.Column('roughness_rq_nm', sa.Float(), nullable=True,
                  comment='RMS roughness (Rq) in nm')
    )

    op.add_column('cvd_results',
        sa.Column('roughness_rz_nm', sa.Float(), nullable=True,
                  comment='Ten-point height (Rz) in nm')
    )

    op.add_column('cvd_results',
        sa.Column('roughness_measurement_method', sa.String(length=100), nullable=True,
                  comment='Method used (AFM, profilometer, optical, etc.)')
    )

    # =========================================================================
    # CVD Results - Additional quality metrics
    # =========================================================================
    op.add_column('cvd_results',
        sa.Column('density_g_cm3', sa.Float(), nullable=True,
                  comment='Film density in g/cm³')
    )

    op.add_column('cvd_results',
        sa.Column('hardness_gpa', sa.Float(), nullable=True,
                  comment='Film hardness in GPa')
    )

    op.add_column('cvd_results',
        sa.Column('resistivity_ohm_cm', sa.Float(), nullable=True,
                  comment='Film resistivity in Ω·cm')
    )

    op.add_column('cvd_results',
        sa.Column('crystallinity_pct', sa.Float(), nullable=True,
                  comment='Degree of crystallinity (%)')
    )

    op.add_column('cvd_results',
        sa.Column('grain_size_nm', sa.Float(), nullable=True,
                  comment='Average grain size in nm')
    )

    # Create indexes for common queries
    op.create_index('ix_cvd_results_film_material', 'cvd_results', ['film_material'])
    op.create_index('ix_cvd_results_adhesion_class', 'cvd_results', ['adhesion_class'])
    op.create_index('ix_cvd_results_stress_type', 'cvd_results', ['stress_type'])
    op.create_index('ix_cvd_recipes_film_material', 'cvd_recipes', ['film_material'])


def downgrade():
    """Remove advanced CVD metrics"""

    # Drop indexes
    op.drop_index('ix_cvd_recipes_film_material', 'cvd_recipes')
    op.drop_index('ix_cvd_results_stress_type', 'cvd_results')
    op.drop_index('ix_cvd_results_adhesion_class', 'cvd_results')
    op.drop_index('ix_cvd_results_film_material', 'cvd_results')

    # Drop CVD Results columns
    op.drop_column('cvd_results', 'grain_size_nm')
    op.drop_column('cvd_results', 'crystallinity_pct')
    op.drop_column('cvd_results', 'resistivity_ohm_cm')
    op.drop_column('cvd_results', 'hardness_gpa')
    op.drop_column('cvd_results', 'density_g_cm3')
    op.drop_column('cvd_results', 'roughness_measurement_method')
    op.drop_column('cvd_results', 'roughness_rz_nm')
    op.drop_column('cvd_results', 'roughness_rq_nm')
    op.drop_column('cvd_results', 'roughness_ra_nm')
    op.drop_column('cvd_results', 'optical_bandgap_ev')
    op.drop_column('cvd_results', 'refractive_index_spectrum')
    op.drop_column('cvd_results', 'adhesion_notes')
    op.drop_column('cvd_results', 'adhesion_test_date')
    op.drop_column('cvd_results', 'adhesion_failure_mode')
    op.drop_column('cvd_results', 'adhesion_critical_load_n')
    op.drop_column('cvd_results', 'adhesion_test_method')
    op.drop_column('cvd_results', 'adhesion_class')
    op.drop_column('cvd_results', 'adhesion_score')
    op.drop_column('cvd_results', 'stress_gradient_mpa_per_nm')
    op.drop_column('cvd_results', 'stress_measurement_method')

    # Revert stress_type back to String
    op.alter_column('cvd_results', 'stress_type',
                   existing_type=sa.Enum('TENSILE', 'COMPRESSIVE', 'MIXED', 'NEUTRAL', name='stress_type_enum'),
                   type_=sa.String(50),
                   existing_nullable=True
    )

    op.drop_column('cvd_results', 'stress_distribution_uri')
    op.drop_column('cvd_results', 'stress_mpa_max')
    op.drop_column('cvd_results', 'stress_mpa_min')
    op.drop_column('cvd_results', 'stress_mpa_std')
    op.drop_column('cvd_results', 'stress_mpa_mean')
    op.drop_column('cvd_results', 'thickness_wtw_uniformity_pct')
    op.drop_column('cvd_results', 'thickness_wiw_uniformity_pct')
    op.drop_column('cvd_results', 'conformality_ratio')
    op.drop_column('cvd_results', 'thickness_map_uri')

    # Drop CVD Recipes columns
    op.drop_column('cvd_recipes', 'pressure_profile_torr')
    op.drop_column('cvd_recipes', 'target_adhesion_score')
    op.drop_column('cvd_recipes', 'target_adhesion_class')
    op.drop_column('cvd_recipes', 'target_stress_type')
    op.drop_column('cvd_recipes', 'target_stress_mpa')
    op.drop_column('cvd_recipes', 'film_material')

    # Drop CVD Process Modes columns
    op.drop_column('cvd_process_modes', 'default_targets')

    # Drop enum types
    op.execute('DROP TYPE IF EXISTS stress_type_enum CASCADE')
    op.execute('DROP TYPE IF EXISTS adhesion_class_enum CASCADE')
