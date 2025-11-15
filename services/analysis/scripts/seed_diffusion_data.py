"""
Diffusion Platform - Database Seed Script
Populates database with sample diffusion data for testing and demonstration
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

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from db.base import Base
from app.models.diffusion import (
    DiffusionFurnace,
    DiffusionRecipe,
    DiffusionRun,
    DiffusionTelemetry,
    DiffusionResult,
    DiffusionSPCSeries,
    DiffusionSPCPoint,
    FurnaceType,
    DopantType,
    DiffusionType,
    DopantSource,
    AmbientGas,
    RunStatus,
)

# Use environment variable or default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://spectra:spectra@localhost:5435/spectra")


def create_tables(engine):
    """Create all diffusion tables"""
    print("Creating Diffusion tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created")


def clean_existing_data(session: Session):
    """Clean existing diffusion data before seeding"""
    print("\nCleaning existing data...")
    try:
        # Delete in order to respect foreign keys
        session.execute(text("DELETE FROM diffusion_spc_points"))
        session.execute(text("DELETE FROM diffusion_spc_series"))
        session.execute(text("DELETE FROM diffusion_results"))
        session.execute(text("DELETE FROM diffusion_telemetry"))
        session.execute(text("DELETE FROM diffusion_runs"))
        session.execute(text("DELETE FROM diffusion_recipes"))
        session.execute(text("DELETE FROM diffusion_furnaces"))
        session.commit()
        print("✓ Existing data cleaned")
    except Exception as e:
        print(f"⚠ Error cleaning data (may not exist yet): {e}")
        session.rollback()


def seed_furnaces(session: Session, org_id: str) -> dict:
    """Seed diffusion furnaces"""
    print("\nSeeding Furnaces...")

    furnaces = []

    # 1. Horizontal Tube Furnace - General Purpose
    furnace_h1 = DiffusionFurnace(
        id=uuid4(),
        org_id=org_id,
        name="Horizontal Tube F1",
        furnace_type=FurnaceType.HORIZONTAL,
        manufacturer="Thermco",
        model="MB-71",
        serial_number="TC-2019-045",
        tube_diameter_mm=200.0,
        tube_length_mm=2000.0,
        num_temperature_zones=3,
        max_wafer_capacity=100,
        max_temperature_c=1200.0,
        supported_dopants=["boron", "phosphorus", "arsenic"],
        supported_sources=["liquid_source", "gas_source", "solid_source"],
        supported_ambients=["N2", "O2", "N2_O2", "Ar"],
        temperature_uniformity_c=2.0,
        capabilities={
            "auto_load": True,
            "wafer_mapping": True,
            "process_monitoring": "real-time",
            "emergency_stop": True,
        },
        is_active=True,
        last_pm_date=datetime.utcnow() - timedelta(days=15),
        next_pm_date=datetime.utcnow() + timedelta(days=75),
        location="Fab 2, Bay 3",
        description="Primary horizontal tube furnace for boron and phosphorus diffusion",
    )
    furnaces.append(furnace_h1)

    # 2. Vertical Furnace - High Volume
    furnace_v1 = DiffusionFurnace(
        id=uuid4(),
        org_id=org_id,
        name="Vertical Furnace V1",
        furnace_type=FurnaceType.VERTICAL,
        manufacturer="ASM International",
        model="A400",
        serial_number="ASM-2020-128",
        tube_diameter_mm=250.0,
        tube_length_mm=1500.0,
        num_temperature_zones=5,
        max_wafer_capacity=150,
        max_temperature_c=1150.0,
        supported_dopants=["phosphorus", "boron"],
        supported_sources=["liquid_source", "spin_on"],
        supported_ambients=["N2", "O2", "N2_O2"],
        temperature_uniformity_c=1.5,
        capabilities={
            "vertical_loading": True,
            "batch_processing": True,
            "zone_control": "independent",
            "recipe_library": 100,
        },
        is_active=True,
        last_pm_date=datetime.utcnow() - timedelta(days=22),
        next_pm_date=datetime.utcnow() + timedelta(days=68),
        location="Fab 2, Bay 5",
        description="High-volume vertical furnace for phosphorus predeposition",
    )
    furnaces.append(furnace_v1)

    # 3. Batch Furnace - Production
    furnace_b1 = DiffusionFurnace(
        id=uuid4(),
        org_id=org_id,
        name="Batch Furnace B1",
        furnace_type=FurnaceType.BATCH,
        manufacturer="Tokyo Electron",
        model="Alpha-8SE",
        serial_number="TEL-2021-089",
        tube_diameter_mm=220.0,
        tube_length_mm=1800.0,
        num_temperature_zones=4,
        max_wafer_capacity=200,
        max_temperature_c=1100.0,
        supported_dopants=["boron", "phosphorus", "arsenic"],
        supported_sources=["solid_source", "gas_source", "ion_implant_anneal"],
        supported_ambients=["N2", "O2", "forming_gas", "Ar"],
        temperature_uniformity_c=1.8,
        capabilities={
            "batch_size": 200,
            "automated_handling": True,
            "inline_metrology": True,
            "spc_integration": True,
        },
        is_active=True,
        last_pm_date=datetime.utcnow() - timedelta(days=8),
        next_pm_date=datetime.utcnow() + timedelta(days=82),
        location="Fab 3, Bay 1",
        description="High-capacity batch furnace for production runs",
    )
    furnaces.append(furnace_b1)

    # 4. Lamp-Heated Furnace - RTP-style
    furnace_l1 = DiffusionFurnace(
        id=uuid4(),
        org_id=org_id,
        name="Lamp Furnace L1",
        furnace_type=FurnaceType.LAMP_HEATED,
        manufacturer="Mattson Technology",
        model="RTP-600",
        serial_number="MAT-2022-014",
        tube_diameter_mm=150.0,
        tube_length_mm=800.0,
        num_temperature_zones=2,
        max_wafer_capacity=25,
        max_temperature_c=1250.0,
        supported_dopants=["boron", "phosphorus", "arsenic", "antimony"],
        supported_sources=["gas_source", "ion_implant_anneal"],
        supported_ambients=["N2", "Ar", "forming_gas"],
        temperature_uniformity_c=3.0,
        capabilities={
            "rapid_heating": True,
            "heating_rate_c_per_s": 50,
            "pyrometry": "multi-wavelength",
            "spike_anneal": True,
        },
        is_active=True,
        last_pm_date=datetime.utcnow() - timedelta(days=5),
        next_pm_date=datetime.utcnow() + timedelta(days=85),
        location="Fab 2, Bay 8",
        description="Lamp-heated furnace for rapid thermal processes and spike anneals",
    )
    furnaces.append(furnace_l1)

    for furnace in furnaces:
        session.add(furnace)

    session.commit()
    print(f"✓ Created {len(furnaces)} furnaces")

    return {
        "horizontal_f1": furnace_h1.id,
        "vertical_v1": furnace_v1.id,
        "batch_b1": furnace_b1.id,
        "lamp_l1": furnace_l1.id,
    }


def seed_recipes(session: Session, org_id: str, furnace_ids: dict) -> dict:
    """Seed diffusion recipes"""
    print("\nSeeding Recipes...")

    recipes = []

    # 1. Boron Predeposition Recipe
    boron_predep = DiffusionRecipe(
        id=uuid4(),
        name="Boron Predeposition - BBr3",
        version=3,
        furnace_id=furnace_ids["horizontal_f1"],
        org_id=org_id,
        diffusion_type=DiffusionType.PREDEPOSITION,
        dopant=DopantType.BORON,
        dopant_source=DopantSource.LIQUID_SOURCE,
        target_conductivity_type="p-type",
        temperature_profile={
            "ramp_rate_c_per_min": 5,
            "hold_temp_c": 950,
            "hold_time_min": 30,
            "cool_rate_c_per_min": 3,
            "zones": {
                "zone1": {"setpoint_c": 950, "tolerance_c": 2},
                "zone2": {"setpoint_c": 950, "tolerance_c": 2},
                "zone3": {"setpoint_c": 950, "tolerance_c": 2},
            },
        },
        ambient_gas=AmbientGas.N2_O2,
        flow_rate_slm=5.0,
        ambient_sequence=[
            {"step": 1, "gas": "N2", "flow_slm": 10.0, "duration_min": 10},
            {"step": 2, "gas": "N2_O2", "flow_slm": 5.0, "duration_min": 30},
            {"step": 3, "gas": "N2", "flow_slm": 10.0, "duration_min": 15},
        ],
        target_junction_depth_um=0.3,
        target_sheet_resistance_ohm_per_sq=50.0,
        target_dose_cm2=1e15,
        recipe_steps=[
            {"step": 1, "name": "Load & Stabilize", "duration_min": 10, "action": "stabilize"},
            {"step": 2, "name": "Ramp to Temp", "duration_min": 190, "action": "ramp"},
            {"step": 3, "name": "BBr3 Predep", "duration_min": 30, "action": "deposit"},
            {"step": 4, "name": "Purge", "duration_min": 10, "action": "purge"},
            {"step": 5, "name": "Cool Down", "duration_min": 300, "action": "cool"},
        ],
        source_temperature_c=25.0,
        carrier_flow_sccm=200.0,
        bubbler_settings={"bubbler_temp_c": 25, "bubbler_pressure_psi": 15},
        safety_hazard_level=3,
        required_interlocks=["toxic_gas_monitor", "scrubber_active", "emergency_n2"],
        max_time_limit_min=600,
        status="approved",
        approval_date=datetime.utcnow() - timedelta(days=30),
        approved_by_id=org_id,
        description="Standard boron predeposition using BBr3 liquid source",
        created_by_id=org_id,
    )
    recipes.append(boron_predep)

    # 2. Phosphorus Predeposition Recipe
    phos_predep = DiffusionRecipe(
        id=uuid4(),
        name="Phosphorus Predeposition - POCl3",
        version=2,
        furnace_id=furnace_ids["vertical_v1"],
        org_id=org_id,
        diffusion_type=DiffusionType.PREDEPOSITION,
        dopant=DopantType.PHOSPHORUS,
        dopant_source=DopantSource.LIQUID_SOURCE,
        target_conductivity_type="n-type",
        temperature_profile={
            "ramp_rate_c_per_min": 7,
            "hold_temp_c": 900,
            "hold_time_min": 45,
            "cool_rate_c_per_min": 4,
            "zones": {
                "zone1": {"setpoint_c": 900, "tolerance_c": 2},
                "zone2": {"setpoint_c": 900, "tolerance_c": 2},
                "zone3": {"setpoint_c": 900, "tolerance_c": 2},
                "zone4": {"setpoint_c": 900, "tolerance_c": 2},
                "zone5": {"setpoint_c": 900, "tolerance_c": 2},
            },
        },
        ambient_gas=AmbientGas.O2,
        flow_rate_slm=3.0,
        ambient_sequence=[
            {"step": 1, "gas": "N2", "flow_slm": 8.0, "duration_min": 15},
            {"step": 2, "gas": "O2", "flow_slm": 3.0, "duration_min": 45},
            {"step": 3, "gas": "N2", "flow_slm": 8.0, "duration_min": 20},
        ],
        target_junction_depth_um=0.4,
        target_sheet_resistance_ohm_per_sq=40.0,
        target_dose_cm2=5e14,
        recipe_steps=[
            {"step": 1, "name": "Load & Purge", "duration_min": 15, "action": "purge"},
            {"step": 2, "name": "Ramp to Temp", "duration_min": 129, "action": "ramp"},
            {"step": 3, "name": "POCl3 Predep", "duration_min": 45, "action": "deposit"},
            {"step": 4, "name": "Oxidation", "duration_min": 10, "action": "oxidize"},
            {"step": 5, "name": "Cool Down", "duration_min": 225, "action": "cool"},
        ],
        source_temperature_c=20.0,
        carrier_flow_sccm=150.0,
        bubbler_settings={"bubbler_temp_c": 20, "bubbler_pressure_psi": 12},
        safety_hazard_level=3,
        required_interlocks=["toxic_gas_monitor", "scrubber_active"],
        max_time_limit_min=480,
        status="approved",
        approval_date=datetime.utcnow() - timedelta(days=45),
        approved_by_id=org_id,
        description="Phosphorus predeposition using POCl3 for n-type emitters",
        created_by_id=org_id,
    )
    recipes.append(phos_predep)

    # 3. Boron Drive-In Recipe
    boron_drive = DiffusionRecipe(
        id=uuid4(),
        name="Boron Drive-In - Inert",
        version=1,
        furnace_id=furnace_ids["batch_b1"],
        org_id=org_id,
        diffusion_type=DiffusionType.DRIVE_IN,
        dopant=DopantType.BORON,
        dopant_source=DopantSource.SOLID_SOURCE,
        target_conductivity_type="p-type",
        temperature_profile={
            "ramp_rate_c_per_min": 4,
            "hold_temp_c": 1100,
            "hold_time_min": 120,
            "cool_rate_c_per_min": 2,
            "zones": {
                "zone1": {"setpoint_c": 1100, "tolerance_c": 3},
                "zone2": {"setpoint_c": 1100, "tolerance_c": 3},
                "zone3": {"setpoint_c": 1100, "tolerance_c": 3},
                "zone4": {"setpoint_c": 1100, "tolerance_c": 3},
            },
        },
        ambient_gas=AmbientGas.N2,
        flow_rate_slm=8.0,
        ambient_sequence=[
            {"step": 1, "gas": "N2", "flow_slm": 8.0, "duration_min": 395},
        ],
        target_junction_depth_um=1.2,
        target_sheet_resistance_ohm_per_sq=200.0,
        recipe_steps=[
            {"step": 1, "name": "Load", "duration_min": 10, "action": "load"},
            {"step": 2, "name": "Ramp", "duration_min": 275, "action": "ramp"},
            {"step": 3, "name": "Drive-In", "duration_min": 120, "action": "drive"},
            {"step": 4, "name": "Cool Down", "duration_min": 550, "action": "cool"},
        ],
        safety_hazard_level=2,
        max_time_limit_min=1000,
        status="approved",
        approval_date=datetime.utcnow() - timedelta(days=20),
        approved_by_id=org_id,
        description="Boron drive-in step for redistribution and deeper junctions",
        created_by_id=org_id,
    )
    recipes.append(boron_drive)

    # 4. Arsenic Ion Implant Anneal
    arsenic_anneal = DiffusionRecipe(
        id=uuid4(),
        name="Arsenic Activation Anneal",
        version=1,
        furnace_id=furnace_ids["lamp_l1"],
        org_id=org_id,
        diffusion_type=DiffusionType.DRIVE_IN,
        dopant=DopantType.ARSENIC,
        dopant_source=DopantSource.ION_IMPLANT_ANNEAL,
        target_conductivity_type="n-type",
        temperature_profile={
            "ramp_rate_c_per_min": 50,
            "hold_temp_c": 1000,
            "hold_time_min": 5,
            "cool_rate_c_per_min": 30,
            "zones": {
                "zone1": {"setpoint_c": 1000, "tolerance_c": 5},
                "zone2": {"setpoint_c": 1000, "tolerance_c": 5},
            },
        },
        ambient_gas=AmbientGas.N2,
        flow_rate_slm=5.0,
        target_junction_depth_um=0.15,
        target_sheet_resistance_ohm_per_sq=80.0,
        recipe_steps=[
            {"step": 1, "name": "Rapid Ramp", "duration_min": 1, "action": "ramp"},
            {"step": 2, "name": "Spike Anneal", "duration_min": 5, "action": "anneal"},
            {"step": 3, "name": "Rapid Cool", "duration_min": 2, "action": "cool"},
        ],
        safety_hazard_level=1,
        max_time_limit_min=30,
        status="approved",
        approval_date=datetime.utcnow() - timedelta(days=10),
        approved_by_id=org_id,
        description="Rapid thermal anneal for arsenic activation after ion implantation",
        created_by_id=org_id,
    )
    recipes.append(arsenic_anneal)

    # 5. Phosphorus Two-Step Process
    phos_twostep = DiffusionRecipe(
        id=uuid4(),
        name="Phosphorus Two-Step Complete",
        version=1,
        furnace_id=furnace_ids["horizontal_f1"],
        org_id=org_id,
        diffusion_type=DiffusionType.TWO_STEP,
        dopant=DopantType.PHOSPHORUS,
        dopant_source=DopantSource.GAS_SOURCE,
        target_conductivity_type="n-type",
        temperature_profile={
            "ramp_rate_c_per_min": 6,
            "hold_temp_c": 850,
            "hold_time_min": 20,
            "zones": {
                "zone1": {"setpoint_c": 850, "tolerance_c": 2},
                "zone2": {"setpoint_c": 850, "tolerance_c": 2},
                "zone3": {"setpoint_c": 850, "tolerance_c": 2},
            },
        },
        ambient_gas=AmbientGas.N2,
        flow_rate_slm=6.0,
        target_junction_depth_um=0.8,
        target_sheet_resistance_ohm_per_sq=60.0,
        recipe_steps=[
            {"step": 1, "name": "Predep Ramp", "duration_min": 142, "action": "ramp"},
            {"step": 2, "name": "PH3 Predep", "duration_min": 20, "action": "deposit"},
            {"step": 3, "name": "Drive Ramp", "duration_min": 42, "action": "ramp"},
            {"step": 4, "name": "Drive-In", "duration_min": 60, "action": "drive"},
            {"step": 5, "name": "Cool", "duration_min": 350, "action": "cool"},
        ],
        safety_hazard_level=3,
        required_interlocks=["toxic_gas_monitor", "scrubber_active", "leak_detector"],
        max_time_limit_min=650,
        status="draft",
        description="Complete two-step phosphorus process with predep and drive-in",
        created_by_id=org_id,
    )
    recipes.append(phos_twostep)

    for recipe in recipes:
        session.add(recipe)

    session.commit()
    print(f"✓ Created {len(recipes)} recipes")

    return {
        "boron_predep": boron_predep.id,
        "phos_predep": phos_predep.id,
        "boron_drive": boron_drive.id,
        "arsenic_anneal": arsenic_anneal.id,
        "phos_twostep": phos_twostep.id,
    }


def seed_runs(session: Session, org_id: str, recipe_ids: dict, furnace_ids: dict) -> dict:
    """Seed diffusion runs"""
    print("\nSeeding Runs...")

    runs = []
    base_time = datetime.utcnow() - timedelta(days=10)

    # Create completed runs for boron predeposition
    for i in range(5):
        run = DiffusionRun(
            id=uuid4(),
            org_id=org_id,
            run_number=f"DIFF-B-{2025000 + i}",
            recipe_id=recipe_ids["boron_predep"],
            furnace_id=furnace_ids["horizontal_f1"],
            lot_id=uuid4(),
            wafer_ids=[str(uuid4()) for _ in range(25)],
            wafer_count=25,
            boat_position_map={"positions": list(range(1, 26))},
            status=RunStatus.SUCCEEDED,
            start_time=base_time + timedelta(days=i),
            end_time=base_time + timedelta(days=i, hours=9),
            duration_seconds=32400.0,
            actual_peak_temp_c=950.0 + (i * 0.3),
            actual_time_at_temp_min=30.0,
            actual_ambient="N2_O2",
            preflight_summary={"checks_passed": 12, "warnings": 0},
            postflight_summary={"wafers_passed": 25, "uniformity_avg": 97.2 + i * 0.2},
            operator_id=org_id,
            notes=f"Standard production run {i+1}",
            custom_metadata={"lot_name": f"LOT-B-{1000+i}", "priority": "normal"},
        )
        runs.append(run)

    # Create completed runs for phosphorus predeposition
    for i in range(4):
        run = DiffusionRun(
            id=uuid4(),
            org_id=org_id,
            run_number=f"DIFF-P-{2025010 + i}",
            recipe_id=recipe_ids["phos_predep"],
            furnace_id=furnace_ids["vertical_v1"],
            lot_id=uuid4(),
            wafer_ids=[str(uuid4()) for _ in range(50)],
            wafer_count=50,
            boat_position_map={"positions": list(range(1, 51))},
            status=RunStatus.SUCCEEDED,
            start_time=base_time + timedelta(days=i+2, hours=4),
            end_time=base_time + timedelta(days=i+2, hours=11),
            duration_seconds=25200.0,
            actual_peak_temp_c=900.0,
            actual_time_at_temp_min=45.0,
            actual_ambient="O2",
            preflight_summary={"checks_passed": 15, "warnings": 0},
            postflight_summary={"wafers_passed": 50, "uniformity_avg": 96.5},
            operator_id=org_id,
            notes=f"Phosphorus predeposition batch {i+1}",
            custom_metadata={"lot_name": f"LOT-P-{2000+i}", "priority": "high"},
        )
        runs.append(run)

    # Create completed runs for boron drive-in
    for i in range(3):
        run = DiffusionRun(
            id=uuid4(),
            org_id=org_id,
            run_number=f"DIFF-BD-{2025020 + i}",
            recipe_id=recipe_ids["boron_drive"],
            furnace_id=furnace_ids["batch_b1"],
            lot_id=uuid4(),
            wafer_ids=[str(uuid4()) for _ in range(100)],
            wafer_count=100,
            status=RunStatus.SUCCEEDED,
            start_time=base_time + timedelta(days=i+5),
            end_time=base_time + timedelta(days=i+5, hours=15, minutes=55),
            duration_seconds=57300.0,
            actual_peak_temp_c=1100.0,
            actual_time_at_temp_min=120.0,
            actual_ambient="N2",
            preflight_summary={"checks_passed": 18, "warnings": 0},
            postflight_summary={"wafers_passed": 100, "uniformity_avg": 95.8},
            operator_id=org_id,
            notes=f"Drive-in batch {i+1}",
            custom_metadata={"lot_name": f"LOT-BD-{3000+i}", "priority": "normal"},
        )
        runs.append(run)

    # Create running run for arsenic anneal
    running_run = DiffusionRun(
        id=uuid4(),
        org_id=org_id,
        run_number="DIFF-AS-2025099",
        recipe_id=recipe_ids["arsenic_anneal"],
        furnace_id=furnace_ids["lamp_l1"],
        lot_id=uuid4(),
        wafer_ids=[str(uuid4()) for _ in range(12)],
        wafer_count=12,
        status=RunStatus.RUNNING,
        start_time=datetime.utcnow() - timedelta(minutes=3),
        actual_peak_temp_c=1000.0,
        actual_ambient="N2",
        job_progress=65.0,
        operator_id=org_id,
        notes="Rapid thermal anneal in progress",
        custom_metadata={"lot_name": "LOT-AS-CURRENT", "priority": "high"},
    )
    runs.append(running_run)

    # Create queued run
    queued_run = DiffusionRun(
        id=uuid4(),
        org_id=org_id,
        run_number="DIFF-P-2025100",
        recipe_id=recipe_ids["phos_predep"],
        furnace_id=furnace_ids["vertical_v1"],
        lot_id=uuid4(),
        wafer_ids=[str(uuid4()) for _ in range(50)],
        wafer_count=50,
        status=RunStatus.QUEUED,
        operator_id=org_id,
        notes="Scheduled for next batch",
        custom_metadata={"lot_name": "LOT-P-NEXT", "priority": "normal"},
    )
    runs.append(queued_run)

    for run in runs:
        session.add(run)

    session.commit()
    print(f"✓ Created {len(runs)} runs")

    # Return IDs of completed runs for adding telemetry/results
    completed_run_ids = [r.id for r in runs if r.status == RunStatus.SUCCEEDED]
    return {"all_runs": [r.id for r in runs], "completed": completed_run_ids}


def seed_telemetry_and_results(session: Session, org_id: str, run_ids: list) -> None:
    """Seed telemetry and results for completed runs"""
    print("\nSeeding Telemetry and Results...")

    telemetry_count = 0
    result_count = 0

    # Add sample telemetry and results to first 3 completed runs
    for run_id in run_ids[:3]:
        # Add telemetry points (sample at 30 second intervals for 1 hour)
        run_obj = session.query(DiffusionRun).filter_by(id=run_id).first()
        if not run_obj or not run_obj.start_time:
            continue

        for i in range(120):  # 120 points = 1 hour of 30-second intervals
            telemetry = DiffusionTelemetry(
                id=uuid4(),
                org_id=org_id,
                run_id=run_id,
                ts=run_obj.start_time + timedelta(seconds=i * 30),
                temperature_zones_c={
                    "zone1": 950.0 + (i * 0.02),
                    "zone2": 950.1 + (i * 0.018),
                    "zone3": 949.9 + (i * 0.022),
                    "wafer_avg": 950.0 + (i * 0.02),
                },
                temperature_setpoint_c=950.0,
                temperature_deviation_c=0.1,
                ambient_gas="N2_O2",
                flow_rate_slm=5.0,
                flow_rate_setpoint_slm=5.0,
                chamber_pressure_torr=760.0,
                source_temperature_c=25.0,
                carrier_flow_sccm=200.0,
                heating_rate_c_per_min=5.0 if i < 20 else 0.0,
                time_at_temperature_min=i * 0.5 if i >= 40 else 0.0,
                pid_output_pct={"zone1": 45.0, "zone2": 46.0, "zone3": 44.5},
            )
            session.add(telemetry)
            telemetry_count += 1

        # Add results for wafers
        for wafer_id in run_obj.wafer_ids[:5]:  # Add results for first 5 wafers
            result = DiffusionResult(
                id=uuid4(),
                org_id=org_id,
                run_id=run_id,
                wafer_id=wafer_id,
                sheet_resistance_ohm_per_sq=50.0 + (hash(wafer_id) % 10) * 0.5,
                sheet_resistance_std_pct=2.5,
                sheet_resistance_uniformity_pct=97.5,
                sheet_resistance_map={"points": [{"x": i, "y": j, "value": 50.0} for i in range(5) for j in range(5)]},
                junction_depth_um=0.3,
                junction_depth_std_um=0.015,
                surface_concentration_cm3=1e20,
                peak_concentration_cm3=1e20,
                dose_cm2=1e15,
                uniformity_score=97.5,
                defect_count=2,
                pass_fail=True,
                measurement_timestamp=run_obj.end_time,
            )
            session.add(result)
            result_count += 1

    session.commit()
    print(f"✓ Created {telemetry_count} telemetry points")
    print(f"✓ Created {result_count} results")


def seed_spc_series(session: Session, org_id: str, recipe_ids: dict, furnace_ids: dict) -> None:
    """Seed SPC control chart series"""
    print("\nSeeding SPC Series...")

    series_list = []

    # Sheet resistance control chart for boron predep
    sr_series = DiffusionSPCSeries(
        id=uuid4(),
        org_id=org_id,
        name="Boron Predep - Sheet Resistance",
        parameter="sheet_resistance",
        recipe_id=recipe_ids["boron_predep"],
        furnace_id=furnace_ids["horizontal_f1"],
        target=50.0,
        ucl=55.0,
        lcl=45.0,
        usl=58.0,
        lsl=42.0,
        mean=50.0,
        std_dev=2.0,
        cp=1.33,
        cpk=1.25,
        is_active=True,
    )
    series_list.append(sr_series)

    # Junction depth control chart
    jd_series = DiffusionSPCSeries(
        id=uuid4(),
        org_id=org_id,
        name="Boron Predep - Junction Depth",
        parameter="junction_depth",
        recipe_id=recipe_ids["boron_predep"],
        furnace_id=furnace_ids["horizontal_f1"],
        target=0.3,
        ucl=0.35,
        lcl=0.25,
        usl=0.38,
        lsl=0.22,
        mean=0.3,
        std_dev=0.015,
        cp=1.78,
        cpk=1.67,
        is_active=True,
    )
    series_list.append(jd_series)

    # Temperature uniformity control chart
    temp_series = DiffusionSPCSeries(
        id=uuid4(),
        org_id=org_id,
        name="Horizontal F1 - Temp Uniformity",
        parameter="temperature_stability",
        furnace_id=furnace_ids["horizontal_f1"],
        target=0.0,
        ucl=2.0,
        lcl=-2.0,
        mean=0.0,
        std_dev=0.5,
        is_active=True,
    )
    series_list.append(temp_series)

    for series in series_list:
        session.add(series)

    session.commit()
    print(f"✓ Created {len(series_list)} SPC series")


def main():
    """Main seeding function"""
    print("=" * 60)
    print("Diffusion Platform - Database Seeding")
    print("=" * 60)

    # Create engine
    engine = create_engine(DATABASE_URL)

    # Create tables
    create_tables(engine)

    # Create session
    session = Session(engine)

    try:
        # Clean existing data
        clean_existing_data(session)

        # Use a fixed organization ID for demo
        org_id = str(uuid4())
        print(f"\nUsing Organization ID: {org_id}")

        # Seed data
        furnace_ids = seed_furnaces(session, org_id)
        recipe_ids = seed_recipes(session, org_id, furnace_ids)
        run_ids = seed_runs(session, org_id, recipe_ids, furnace_ids)
        seed_telemetry_and_results(session, org_id, run_ids["completed"])
        seed_spc_series(session, org_id, recipe_ids, furnace_ids)

        print("\n" + "=" * 60)
        print("✓ Database seeding completed successfully!")
        print("=" * 60)
        print("\nSeeded Data Summary:")
        print(f"  - Furnaces: 4 (Horizontal, Vertical, Batch, Lamp)")
        print(f"  - Recipes: 5 (Boron, Phosphorus, Arsenic processes)")
        print(f"  - Runs: {len(run_ids['all_runs'])} (12 succeeded, 1 running, 1 queued)")
        print(f"  - Telemetry: ~360 points")
        print(f"  - Results: ~15 wafer results")
        print(f"  - SPC Series: 3")
        print("\nYou can now:")
        print("  1. Access the Diffusion page: http://localhost:3012/dashboard/manufacturing/diffusion")
        print("  2. Browse API docs: http://localhost:8001/docs")
        print("  3. View Diffusion endpoints: http://localhost:8001/api/v1/diffusion")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error during seeding: {e}")
        session.rollback()
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
