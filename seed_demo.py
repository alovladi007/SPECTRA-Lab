#!/usr/bin/env python3
"""
scripts/seed_demo.py

Seed demo data for SPECTRA-Lab Platform Session 17.
Creates organizations, users, instruments, samples, recipes, and test runs.
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sqlalchemy.orm import Session
from services.shared.db.base import SessionLocal
from services.shared.db.models import (
    Organization, User, UserRole,
    Instrument, InstrumentStatus, Calibration, CalibrationStatus,
    Material, Sample, Wafer, Device,
    Recipe, RecipeStatus, RecipeApproval, ApprovalState,
    Run, RunStatus, Result,
    SOP, ELNEntry,
    SPCSeries
)
from services.shared.auth.jwt import hash_password

def seed_demo_data(session: Session):
    """Seed complete demo dataset."""
    
    print("🌱 Seeding demo data for Session 17...")
    
    # ========================================================================
    # Organizations
    # ========================================================================
    
    print("\n📁 Creating organizations...")
    
    demo_org = Organization(
        id=uuid4(),
        name="Demo Lab",
        slug="demo-lab",
        settings={
            "timezone": "America/Los_Angeles",
            "measurement_system": "metric",
            "spc_enabled": True
        }
    )
    session.add(demo_org)
    
    test_org = Organization(
        id=uuid4(),
        name="Test Organization",
        slug="test-org",
        settings={}
    )
    session.add(test_org)
    
    session.flush()
    print(f"  ✓ Created organizations: {demo_org.name}, {test_org.name}")
    
    # ========================================================================
    # Users
    # ========================================================================
    
    print("\n👥 Creating users...")
    
    users = []
    user_data = [
        ("admin@demo.lab", "Admin User", UserRole.ADMIN, "admin123"),
        ("pi@demo.lab", "Principal Investigator", UserRole.PI, "pi123"),
        ("engineer@demo.lab", "Process Engineer", UserRole.ENGINEER, "eng123"),
        ("tech@demo.lab", "Lab Technician", UserRole.TECHNICIAN, "tech123"),
        ("viewer@demo.lab", "Data Viewer", UserRole.VIEWER, "view123"),
    ]
    
    for email, name, role, password in user_data:
        user = User(
            id=uuid4(),
            organization_id=demo_org.id,
            email=email,
            name=name,
            role=role,
            password_hash=hash_password(password),
            is_active=True
        )
        users.append(user)
        session.add(user)
    
    session.flush()
    print(f"  ✓ Created {len(users)} users in Demo Lab")
    
    # ========================================================================
    # Instruments
    # ========================================================================
    
    print("\n🔬 Creating instruments...")
    
    instruments = []
    
    # SMU
    smu = Instrument(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Keithley 2400",
        vendor="Keithley",
        model="2400",
        serial="ABC123456",
        interface="visa_gpib",
        location="Room 101",
        status=InstrumentStatus.ONLINE,
        extra_metadata={"capabilities": ["iv_sweep", "cv_measurement"]}
    )
    instruments.append(smu)
    session.add(smu)
    
    # Spectrometer
    spec = Instrument(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Ocean Optics USB4000",
        vendor="Ocean Optics",
        model="USB4000",
        serial="SPEC789",
        interface="visa_usb",
        location="Room 102",
        status=InstrumentStatus.ONLINE,
        extra_metadata={"wavelength_range": [200, 1100]}
    )
    instruments.append(spec)
    session.add(spec)
    
    # Ellipsometer
    ellips = Instrument(
        id=uuid4(),
        organization_id=demo_org.id,
        name="J.A. Woollam M-2000",
        vendor="J.A. Woollam",
        model="M-2000",
        serial="ELL456",
        interface="visa_tcpip",
        location="Room 103",
        status=InstrumentStatus.ONLINE,
        extra_metadata={"angle_range": [45, 90]}
    )
    instruments.append(ellips)
    session.add(ellips)
    
    # XRD
    xrd = Instrument(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Bruker D8",
        vendor="Bruker",
        model="D8 Discover",
        serial="XRD999",
        interface="visa_tcpip",
        location="Room 104",
        status=InstrumentStatus.MAINTENANCE,
        extra_metadata={"theta_range": [10, 80]}
    )
    instruments.append(xrd)
    session.add(xrd)
    
    # SEM
    sem = Instrument(
        id=uuid4(),
        organization_id=demo_org.id,
        name="JEOL JSM-7800F",
        vendor="JEOL",
        model="JSM-7800F",
        serial="SEM777",
        interface="tcp",
        location="Room 105",
        status=InstrumentStatus.ONLINE,
        extra_metadata={"resolution_nm": 1.0}
    )
    instruments.append(sem)
    session.add(sem)
    
    session.flush()
    print(f"  ✓ Created {len(instruments)} instruments")
    
    # ========================================================================
    # Calibrations
    # ========================================================================
    
    print("\n📜 Creating calibration certificates...")
    
    now = datetime.now(timezone.utc)
    
    # Valid calibrations
    for inst in [smu, spec, ellips, sem]:
        cal = Calibration(
            id=uuid4(),
            organization_id=demo_org.id,
            instrument_id=inst.id,
            certificate_id=f"CAL-{inst.serial}-2025",
            issued_at=now - timedelta(days=30),
            expires_at=now + timedelta(days=335),  # Valid for 11 more months
            provider="NIST Traceable Lab",
            status=CalibrationStatus.VALID,
            results={"drift": 0.001, "linearity": 0.9999}
        )
        session.add(cal)
    
    # Expired calibration (XRD)
    expired_cal = Calibration(
        id=uuid4(),
        organization_id=demo_org.id,
        instrument_id=xrd.id,
        certificate_id=f"CAL-{xrd.serial}-2024",
        issued_at=now - timedelta(days=400),
        expires_at=now - timedelta(days=35),  # Expired 35 days ago
        provider="NIST Traceable Lab",
        status=CalibrationStatus.EXPIRED,
        results={}
    )
    session.add(expired_cal)
    
    session.flush()
    print(f"  ✓ Created calibration certificates (1 expired for testing)")
    
    # ========================================================================
    # Materials
    # ========================================================================
    
    print("\n🧪 Creating materials...")
    
    materials = []
    
    silicon = Material(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Silicon",
        type="semiconductor",
        extra_metadata={
            "band_gap_ev": 1.12,
            "lattice_constant_nm": 0.543,
            "refractive_index": 3.42
        }
    )
    materials.append(silicon)
    session.add(silicon)
    
    gaas = Material(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Gallium Arsenide",
        type="semiconductor",
        extra_metadata={
            "band_gap_ev": 1.42,
            "lattice_constant_nm": 0.565
        }
    )
    materials.append(gaas)
    session.add(gaas)
    
    session.flush()
    print(f"  ✓ Created {len(materials)} materials")
    
    # ========================================================================
    # Samples & Wafers
    # ========================================================================
    
    print("\n💎 Creating samples, wafers, and devices...")
    
    samples = []
    
    for i in range(5):
        sample = Sample(
            id=uuid4(),
            organization_id=demo_org.id,
            name=f"Sample-{2025000 + i}",
            material_type="Si",
            material_id=silicon.id,
            lot_code=f"LOT-2025-{i:03d}",
            barcode=f"BARCODE-{i:05d}",
            location="Storage Cabinet A",
            extra_metadata={"thickness_um": 525 + i * 5}
        )
        samples.append(sample)
        session.add(sample)
        
        # Add wafers
        for w in range(2):
            wafer = Wafer(
                id=uuid4(),
                organization_id=demo_org.id,
                sample_id=sample.id,
                wafer_id_code=f"W{i+1}-{w+1}",
                diameter_mm=200.0,
                notch="180",
                map_json={"rows": 10, "cols": 10}
            )
            session.add(wafer)
            
            # Add some devices
            for d in range(5):
                device = Device(
                    id=uuid4(),
                    organization_id=demo_org.id,
                    wafer_id=wafer.id,
                    device_label=f"D{d+1}",
                    coordinates={"row": d, "col": d},
                    extra_metadata={"area_cm2": 0.1}
                )
                session.add(device)
    
    session.flush()
    print(f"  ✓ Created {len(samples)} samples with wafers and devices")
    
    # ========================================================================
    # SOPs
    # ========================================================================
    
    print("\n📋 Creating SOPs...")
    
    sop1 = SOP(
        id=uuid4(),
        organization_id=demo_org.id,
        number="SOP-001",
        title="IV Characterization Standard Procedure",
        version="2.1",
        body_md_uri="s3://spectra-sops/sop-001-v2.1.md",
        hazard_level="low"
    )
    session.add(sop1)
    
    sop2 = SOP(
        id=uuid4(),
        organization_id=demo_org.id,
        number="SOP-002",
        title="XRD Analysis Protocol",
        version="1.5",
        body_md_uri="s3://spectra-sops/sop-002-v1.5.md",
        hazard_level="medium"
    )
    session.add(sop2)
    
    session.flush()
    print(f"  ✓ Created 2 SOPs")
    
    # ========================================================================
    # Recipes
    # ========================================================================
    
    print("\n📝 Creating recipes...")
    
    engineer = users[2]  # Engineer user
    
    # Approved recipe
    recipe1 = Recipe(
        id=uuid4(),
        organization_id=demo_org.id,
        name="MOSFET IV Standard",
        version="1.0",
        status=RecipeStatus.APPROVED,
        owner_id=engineer.id,
        sop_id=sop1.id,
        params={
            "method": "iv_sweep",
            "vgs_start": -2.0,
            "vgs_stop": 2.0,
            "vgs_step": 0.1,
            "vds": 0.1
        }
    )
    session.add(recipe1)
    
    # Add approval
    approval1 = RecipeApproval(
        id=uuid4(),
        organization_id=demo_org.id,
        recipe_id=recipe1.id,
        approver_id=users[1].id,  # PI
        state=ApprovalState.APPROVED,
        comment="Reviewed and approved",
        signed_at=now - timedelta(days=10)
    )
    session.add(approval1)
    
    # Draft recipe
    recipe2 = Recipe(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Solar Cell EQE Scan",
        version="0.5",
        status=RecipeStatus.DRAFT,
        owner_id=engineer.id,
        params={
            "method": "spectroscopy",
            "wavelength_start": 300,
            "wavelength_stop": 1100
        }
    )
    session.add(recipe2)
    
    # Retired recipe
    recipe3 = Recipe(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Old Protocol",
        version="0.9",
        status=RecipeStatus.RETIRED,
        owner_id=engineer.id,
        params={}
    )
    session.add(recipe3)
    
    session.flush()
    print(f"  ✓ Created 3 recipes (1 approved, 1 draft, 1 retired)")
    
    # ========================================================================
    # Runs & Results
    # ========================================================================
    
    print("\n🏃 Creating runs and results...")
    
    # Successful runs
    for i in range(10):
        run = Run(
            id=uuid4(),
            organization_id=demo_org.id,
            recipe_id=recipe1.id if i % 2 == 0 else None,
            instrument_id=smu.id,
            sample_id=samples[i % len(samples)].id,
            method="iv_sweep",
            status=RunStatus.SUCCEEDED,
            started_at=now - timedelta(days=30-i),
            finished_at=now - timedelta(days=30-i, hours=-1),
            created_by=engineer.id,
            log_uri=f"s3://spectra-runs/run-{i:04d}/log.txt"
        )
        session.add(run)
        
        # Add result
        result = Result(
            id=uuid4(),
            organization_id=demo_org.id,
            run_id=run.id,
            result_type="iv_analysis",
            metrics={
                "vth": 0.5 + i * 0.01,
                "gm_max": 0.001 * (1 + i * 0.1),
                "ion_ioff_ratio": 1e6 + i * 1e5
            },
            arrays_uri=f"s3://spectra-runs/run-{i:04d}/data.h5",
            report_uri=f"s3://spectra-runs/run-{i:04d}/report.pdf"
        )
        session.add(result)
    
    # Failed run
    failed_run = Run(
        id=uuid4(),
        organization_id=demo_org.id,
        instrument_id=smu.id,
        sample_id=samples[0].id,
        method="iv_sweep",
        status=RunStatus.FAILED,
        started_at=now - timedelta(days=5),
        finished_at=now - timedelta(days=5, hours=-1),
        created_by=engineer.id,
        log_uri="s3://spectra-runs/run-failed/log.txt"
    )
    session.add(failed_run)
    
    # Blocked run (expired calibration)
    blocked_run = Run(
        id=uuid4(),
        organization_id=demo_org.id,
        instrument_id=xrd.id,
        sample_id=samples[0].id,
        method="xrd_scan",
        status=RunStatus.BLOCKED,
        created_by=engineer.id,
        blocked_reason="Instrument calibration expired"
    )
    session.add(blocked_run)
    
    session.flush()
    print(f"  ✓ Created 12 runs (10 successful, 1 failed, 1 blocked)")
    
    # ========================================================================
    # ELN Entries
    # ========================================================================
    
    print("\n📓 Creating ELN entries...")
    
    eln1 = ELNEntry(
        id=uuid4(),
        organization_id=demo_org.id,
        author_id=engineer.id,
        title="Initial MOSFET Characterization Results",
        body_markdown="# Experiment Summary\n\nTested 5 samples with IV sweeps...",
        linked_entities={"runs": [str(run.id) for run in [run]]},
        signed=True,
        signed_at=now - timedelta(days=20)
    )
    session.add(eln1)
    
    eln2 = ELNEntry(
        id=uuid4(),
        organization_id=demo_org.id,
        author_id=engineer.id,
        title="Solar Cell Performance Analysis",
        body_markdown="# Weekly Report\n\nEQE measurements completed...",
        linked_entities={},
        signed=False
    )
    session.add(eln2)
    
    session.flush()
    print(f"  ✓ Created 2 ELN entries")
    
    # ========================================================================
    # SPC Series
    # ========================================================================
    
    print("\n📊 Creating SPC series...")
    
    spc = SPCSeries(
        id=uuid4(),
        organization_id=demo_org.id,
        name="Threshold Voltage Trend",
        path="mosfet/iv/vth",
        entity_type="method",
        metric="vth",
        subgroup_size=1,
        spec_lcl=0.4,
        spec_ucl=0.6,
        ctrl_lcl=0.45,
        ctrl_ucl=0.55,
        ruleset="western_electric"
    )
    session.add(spc)
    
    session.flush()
    print(f"  ✓ Created 1 SPC series")
    
    # ========================================================================
    # Commit
    # ========================================================================
    
    session.commit()
    print("\n✅ Demo data seeded successfully!")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    print("\n" + "="*60)
    print("📊 DEMO DATA SUMMARY")
    print("="*60)
    print(f"Organizations:     2")
    print(f"Users:             {len(users)}")
    print(f"Instruments:       {len(instruments)}")
    print(f"Calibrations:      {len(instruments) + 1}")
    print(f"Materials:         {len(materials)}")
    print(f"Samples:           {len(samples)}")
    print(f"SOPs:              2")
    print(f"Recipes:           3")
    print(f"Runs:              12")
    print(f"ELN Entries:       2")
    print(f"SPC Series:        1")
    print("="*60)
    print("\n🔑 Demo Credentials:")
    for email, name, role, password in user_data:
        print(f"  • {email:25s} / {password:10s} ({role.value})")
    print("\n")


if __name__ == "__main__":
    try:
        session = SessionLocal()
        seed_demo_data(session)
        session.close()
        print("🎉 Seeding complete!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
