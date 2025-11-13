"""
CVD Platform - Database Seed Script
Populates database with sample CVD data for testing and demonstration
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from uuid import uuid4
import asyncio

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.base import Base
from app.models.cvd import (
    CVDProcessMode,
    CVDRecipe,
    CVDRun,
    CVDTelemetry,
    CVDResult,
    CVDSPCSeries,
    CVDSPCPoint,
)

# Use environment variable or default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://spectra:spectra@localhost:5433/spectra")


def create_tables(engine):
    """Create all CVD tables"""
    print("Creating CVD tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created")


def seed_process_modes(session: Session, org_id: str) -> dict:
    """Seed CVD process modes"""
    print("\nSeeding Process Modes...")

    process_modes = []

    # 1. LPCVD for silicon nitride
    lpcvd_sin = CVDProcessMode(
        id=uuid4(),
        organization_id=org_id,
        pressure_mode="LPCVD",
        energy_mode="THERMAL",
        reactor_type="HORIZONTAL",
        chemistry_type="INORGANIC",
        variant="LPCVD-Si3N4",
        description="Low pressure CVD for silicon nitride deposition",
        pressure_range_pa={"min": 20, "max": 200},
        temperature_range_c={"min": 650, "max": 850},
        capabilities={
            "multi_zone_heating": True,
            "max_wafer_capacity": 150,
            "uniformity_target_pct": 95,
        },
        materials=["Si3N4", "SiO2", "Poly-Si"],
        is_active=True,
        created_by=org_id,
    )
    process_modes.append(lpcvd_sin)

    # 2. PECVD for SiO2
    pecvd_oxide = CVDProcessMode(
        id=uuid4(),
        organization_id=org_id,
        pressure_mode="PECVD",
        energy_mode="PLASMA",
        reactor_type="SHOWERHEAD",
        chemistry_type="INORGANIC",
        variant="PECVD-SiO2",
        description="Plasma-enhanced CVD for silicon dioxide",
        pressure_range_pa={"min": 100, "max": 1000},
        temperature_range_c={"min": 250, "max": 400},
        capabilities={
            "plasma_power_range_w": {"min": 100, "max": 1000},
            "plasma_frequency_mhz": 13.56,
            "gas_flow_control": "MFC",
        },
        materials=["SiO2", "SiON", "SiN"],
        is_active=True,
        created_by=org_id,
    )
    process_modes.append(pecvd_oxide)

    # 3. MOCVD for GaN
    mocvd_gan = CVDProcessMode(
        id=uuid4(),
        organization_id=org_id,
        pressure_mode="LPCVD",
        energy_mode="THERMAL",
        reactor_type="ROTATING_DISK",
        chemistry_type="METALORGANIC",
        variant="MOCVD-GaN",
        description="Metal-organic CVD for gallium nitride epitaxy",
        pressure_range_pa={"min": 10000, "max": 40000},
        temperature_range_c={"min": 900, "max": 1150},
        capabilities={
            "rotation_rpm": {"min": 100, "max": 1500},
            "precursors": ["TMGa", "NH3", "H2"],
            "v_iii_ratio_range": {"min": 1000, "max": 5000},
        },
        materials=["GaN", "InGaN", "AlGaN"],
        is_active=True,
        created_by=org_id,
    )
    process_modes.append(mocvd_gan)

    # 4. AACVD for ZnO
    aacvd_zno = CVDProcessMode(
        id=uuid4(),
        organization_id=org_id,
        pressure_mode="APCVD",
        energy_mode="THERMAL",
        reactor_type="HORIZONTAL_TUBE",
        chemistry_type="INORGANIC",
        variant="AACVD-ZnO",
        description="Aerosol-assisted CVD for zinc oxide thin films",
        pressure_range_pa={"min": 101325, "max": 101325},
        temperature_range_c={"min": 350, "max": 550},
        capabilities={
            "aerosol_generation": "ultrasonic",
            "atomizer_frequency_khz": 1700,
            "droplet_size_um": 3.0,
        },
        materials=["ZnO", "SnO2", "TiO2"],
        is_active=True,
        created_by=org_id,
    )
    process_modes.append(aacvd_zno)

    for pm in process_modes:
        session.add(pm)

    session.commit()
    print(f"✓ Created {len(process_modes)} process modes")

    return {
        "lpcvd_sin": lpcvd_sin.id,
        "pecvd_oxide": pecvd_oxide.id,
        "mocvd_gan": mocvd_gan.id,
        "aacvd_zno": aacvd_zno.id,
    }


def seed_recipes(session: Session, org_id: str, process_mode_ids: dict) -> dict:
    """Seed CVD recipes"""
    print("\nSeeding Recipes...")

    recipes = []

    # 1. LPCVD Si3N4 Recipe
    lpcvd_recipe = CVDRecipe(
        id=uuid4(),
        name="LPCVD Si3N4 Standard",
        description="Standard silicon nitride deposition for passivation",
        process_mode_id=process_mode_ids["lpcvd_sin"],
        organization_id=org_id,
        temperature_profile={
            "zones": [
                {"zone": 1, "setpoint_c": 780, "ramp_rate_c_per_min": 10},
                {"zone": 2, "setpoint_c": 780, "ramp_rate_c_per_min": 10},
                {"zone": 3, "setpoint_c": 780, "ramp_rate_c_per_min": 10},
            ],
            "soak_time_s": 300,
        },
        gas_flows={
            "gases": [
                {"name": "SiH4", "flow_sccm": 80, "mfc_id": "MFC1"},
                {"name": "NH3", "flow_sccm": 160, "mfc_id": "MFC2"},
            ],
            "carrier_gas": "N2",
            "carrier_flow_sccm": 2000,
        },
        pressure_profile={
            "base_pressure_pa": 133,
            "process_pressure_pa": 133,
            "ramp_rate_pa_per_min": 50,
        },
        recipe_steps=[
            {"step": 1, "name": "Stabilize", "duration_s": 180, "action": "stabilize"},
            {"step": 2, "name": "Deposition", "duration_s": 1800, "action": "deposit"},
            {"step": 3, "name": "Purge", "duration_s": 120, "action": "purge"},
        ],
        process_time_s=2100,
        target_thickness_nm=100.0,
        target_uniformity_pct=95.0,
        tags=["si3n4", "passivation", "lpcvd"],
        version="1.0",
        is_baseline=True,
        is_golden=False,
        is_active=True,
        run_count=0,
        created_by=org_id,
    )
    recipes.append(lpcvd_recipe)

    # 2. PECVD SiO2 Recipe
    pecvd_recipe = CVDRecipe(
        id=uuid4(),
        name="PECVD SiO2 ILD",
        description="Interlayer dielectric oxide deposition",
        process_mode_id=process_mode_ids["pecvd_oxide"],
        organization_id=org_id,
        temperature_profile={
            "zones": [
                {"zone": 1, "setpoint_c": 350, "ramp_rate_c_per_min": 15},
            ],
            "soak_time_s": 120,
        },
        gas_flows={
            "gases": [
                {"name": "SiH4", "flow_sccm": 150, "mfc_id": "MFC1"},
                {"name": "N2O", "flow_sccm": 1000, "mfc_id": "MFC2"},
            ],
            "carrier_gas": "N2",
            "carrier_flow_sccm": 500,
        },
        pressure_profile={
            "base_pressure_pa": 400,
            "process_pressure_pa": 400,
            "ramp_rate_pa_per_min": 100,
        },
        plasma_settings={
            "power_w": 300,
            "frequency_mhz": 13.56,
            "duty_cycle_pct": 100,
        },
        recipe_steps=[
            {"step": 1, "name": "Preheat", "duration_s": 120, "action": "preheat"},
            {"step": 2, "name": "Strike Plasma", "duration_s": 10, "action": "plasma_ignite"},
            {"step": 3, "name": "Deposition", "duration_s": 600, "action": "deposit"},
            {"step": 4, "name": "Cool Down", "duration_s": 180, "action": "cool"},
        ],
        process_time_s=910,
        target_thickness_nm=500.0,
        target_uniformity_pct=92.0,
        tags=["sio2", "ild", "pecvd", "oxide"],
        version="2.1",
        is_baseline=False,
        is_golden=True,
        is_active=True,
        run_count=0,
        created_by=org_id,
    )
    recipes.append(pecvd_recipe)

    # 3. MOCVD GaN Recipe
    mocvd_recipe = CVDRecipe(
        id=uuid4(),
        name="MOCVD GaN Epitaxy",
        description="Gallium nitride epitaxial growth for LEDs",
        process_mode_id=process_mode_ids["mocvd_gan"],
        organization_id=org_id,
        temperature_profile={
            "zones": [
                {"zone": 1, "setpoint_c": 1050, "ramp_rate_c_per_min": 20},
            ],
            "soak_time_s": 600,
        },
        gas_flows={
            "gases": [
                {"name": "TMGa", "flow_sccm": 50, "mfc_id": "MFC1"},
                {"name": "NH3", "flow_sccm": 2000, "mfc_id": "MFC2"},
            ],
            "carrier_gas": "H2",
            "carrier_flow_sccm": 8000,
        },
        pressure_profile={
            "base_pressure_pa": 26664,
            "process_pressure_pa": 26664,
            "ramp_rate_pa_per_min": 1000,
        },
        recipe_steps=[
            {"step": 1, "name": "Heat Up", "duration_s": 900, "action": "heat"},
            {"step": 2, "name": "Nitridation", "duration_s": 300, "action": "nitridation"},
            {"step": 3, "name": "GaN Growth", "duration_s": 3600, "action": "deposit"},
            {"step": 4, "name": "Cool Down", "duration_s": 1200, "action": "cool"},
        ],
        process_time_s=6000,
        target_thickness_nm=2000.0,
        target_uniformity_pct=97.0,
        tags=["gan", "mocvd", "epitaxy", "led"],
        version="1.5",
        is_baseline=True,
        is_golden=True,
        is_active=True,
        run_count=0,
        created_by=org_id,
    )
    recipes.append(mocvd_recipe)

    for recipe in recipes:
        session.add(recipe)

    session.commit()
    print(f"✓ Created {len(recipes)} recipes")

    return {
        "lpcvd_sin": lpcvd_recipe.id,
        "pecvd_oxide": pecvd_recipe.id,
        "mocvd_gan": mocvd_recipe.id,
    }


def seed_runs(session: Session, org_id: str, recipe_ids: dict, process_mode_ids: dict) -> dict:
    """Seed CVD runs"""
    print("\nSeeding Runs...")

    runs = []
    tool_id = str(uuid4())  # Mock tool ID

    # Create some completed runs for the LPCVD recipe
    base_time = datetime.utcnow() - timedelta(days=7)

    for i in range(5):
        run = CVDRun(
            id=uuid4(),
            recipe_id=recipe_ids["lpcvd_sin"],
            process_mode_id=process_mode_ids["lpcvd_sin"],
            tool_id=tool_id,
            organization_id=org_id,
            status="COMPLETED",
            lot_id=f"LOT-2024-00{i+1}",
            wafer_ids=[f"W{j:03d}" for j in range(1, 26)],  # 25 wafers
            operator_id=org_id,
            run_number=f"RUN-{2024000 + i}",
            actual_temperature_c=780.0 + (i * 0.5),
            actual_pressure_pa=133.0 + (i * 2),
            actual_time_s=2100 + (i * 10),
            notes=f"Production run {i+1}",
            start_time=base_time + timedelta(days=i),
            end_time=base_time + timedelta(days=i, minutes=35),
            duration_s=2100,
            created_by=org_id,
        )
        runs.append(run)

    # Create a couple of runs for PECVD
    for i in range(3):
        run = CVDRun(
            id=uuid4(),
            recipe_id=recipe_ids["pecvd_oxide"],
            process_mode_id=process_mode_ids["pecvd_oxide"],
            tool_id=tool_id,
            organization_id=org_id,
            status="COMPLETED",
            lot_id=f"LOT-PECVD-{i+1}",
            wafer_ids=[f"P{j:03d}" for j in range(1, 51)],  # 50 wafers
            operator_id=org_id,
            run_number=f"PECVD-{2024010 + i}",
            actual_temperature_c=350.0,
            actual_pressure_pa=400.0,
            actual_time_s=910,
            start_time=base_time + timedelta(days=i+2, hours=3),
            end_time=base_time + timedelta(days=i+2, hours=3, minutes=15),
            duration_s=910,
            created_by=org_id,
        )
        runs.append(run)

    # Create one in-progress run
    in_progress_run = CVDRun(
        id=uuid4(),
        recipe_id=recipe_ids["pecvd_oxide"],
        process_mode_id=process_mode_ids["pecvd_oxide"],
        tool_id=tool_id,
        organization_id=org_id,
        status="PROCESSING",
        lot_id="LOT-CURRENT",
        wafer_ids=[f"C{j:03d}" for j in range(1, 26)],
        operator_id=org_id,
        run_number="PECVD-2024099",
        start_time=datetime.utcnow() - timedelta(minutes=5),
        created_by=org_id,
    )
    runs.append(in_progress_run)

    for run in runs:
        session.add(run)

    session.commit()
    print(f"✓ Created {len(runs)} runs")

    return {"runs": [r.id for r in runs]}


def seed_spc_series(session: Session, org_id: str, recipe_ids: dict) -> None:
    """Seed SPC control chart series"""
    print("\nSeeding SPC Series...")

    series_list = []

    # Thickness control chart for LPCVD
    thickness_series = CVDSPCSeries(
        id=uuid4(),
        recipe_id=recipe_ids["lpcvd_sin"],
        organization_id=org_id,
        metric_name="thickness_nm",
        chart_type="xbar-r",
        ucl=110.0,
        lcl=90.0,
        center_line=100.0,
        usl=115.0,
        lsl=85.0,
        subgroup_size=5,
        is_active=True,
    )
    series_list.append(thickness_series)

    # Uniformity control chart
    uniformity_series = CVDSPCSeries(
        id=uuid4(),
        recipe_id=recipe_ids["lpcvd_sin"],
        organization_id=org_id,
        metric_name="uniformity_pct",
        chart_type="i-mr",
        ucl=98.0,
        lcl=92.0,
        center_line=95.0,
        is_active=True,
    )
    series_list.append(uniformity_series)

    for series in series_list:
        session.add(series)

    session.commit()
    print(f"✓ Created {len(series_list)} SPC series")


def main():
    """Main seeding function"""
    print("=" * 60)
    print("CVD Platform - Database Seeding")
    print("=" * 60)

    # Create engine
    engine = create_engine(DATABASE_URL)

    # Create tables
    create_tables(engine)

    # Create session
    session = Session(engine)

    try:
        # Use a fixed organization ID for demo
        org_id = str(uuid4())
        print(f"\nUsing Organization ID: {org_id}")

        # Seed data
        process_mode_ids = seed_process_modes(session, org_id)
        recipe_ids = seed_recipes(session, org_id, process_mode_ids)
        run_ids = seed_runs(session, org_id, recipe_ids, process_mode_ids)
        seed_spc_series(session, org_id, recipe_ids)

        print("\n" + "=" * 60)
        print("✓ Database seeding completed successfully!")
        print("=" * 60)
        print("\nSeeded Data Summary:")
        print(f"  - Process Modes: 4")
        print(f"  - Recipes: 3")
        print(f"  - Runs: 9 (8 completed, 1 in progress)")
        print(f"  - SPC Series: 2")
        print("\nYou can now:")
        print("  1. Access the CVD workspace: http://localhost:3012/cvd/workspace")
        print("  2. Browse API docs: http://localhost:8001/docs")
        print("  3. View CVD endpoints: http://localhost:8001/api/v1/cvd")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during seeding: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
