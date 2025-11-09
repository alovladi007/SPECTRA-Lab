"""
Pytest fixtures for Session 17 tests.
"""

import pytest
from datetime import datetime, timedelta, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from fastapi.testclient import TestClient
from typing import Generator
import uuid

from services.shared.db.base import Base
from services.shared.db.models import (
    Organization, User, UserRole, Instrument, InstrumentStatus,
    Calibration, CalibrationStatus, Run, RunStatus
)
from services.shared.auth.jwt import create_token_pair, hash_password


# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def db_engine():
    """Create a fresh database engine for each test."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=False
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Create a fresh database session for each test."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def org1(db_session: Session) -> Organization:
    """Create test organization 1."""
    org = Organization(
        name="Acme Semiconductors",
        slug="acme-semi",
        is_active=True
    )
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


@pytest.fixture
def org2(db_session: Session) -> Organization:
    """Create test organization 2."""
    org = Organization(
        name="Beta Labs",
        slug="beta-labs",
        is_active=True
    )
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


@pytest.fixture
def admin_user(db_session: Session, org1: Organization) -> User:
    """Create admin user for org1."""
    user = User(
        organization_id=org1.id,
        email="admin@acme.com",
        full_name="Admin User",
        role=UserRole.ADMIN,
        password_hash=hash_password("admin123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def pi_user(db_session: Session, org1: Organization) -> User:
    """Create PI user for org1."""
    user = User(
        organization_id=org1.id,
        email="pi@acme.com",
        full_name="PI User",
        role=UserRole.PI,
        password_hash=hash_password("pi123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def engineer_user(db_session: Session, org1: Organization) -> User:
    """Create engineer user for org1."""
    user = User(
        organization_id=org1.id,
        email="engineer@acme.com",
        full_name="Engineer User",
        role=UserRole.ENGINEER,
        password_hash=hash_password("eng123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def technician_user(db_session: Session, org1: Organization) -> User:
    """Create technician user for org1."""
    user = User(
        organization_id=org1.id,
        email="tech@acme.com",
        full_name="Tech User",
        role=UserRole.TECHNICIAN,
        password_hash=hash_password("tech123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def viewer_user(db_session: Session, org1: Organization) -> User:
    """Create viewer user for org1."""
    user = User(
        organization_id=org1.id,
        email="viewer@acme.com",
        full_name="Viewer User",
        role=UserRole.VIEWER,
        password_hash=hash_password("view123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def org2_engineer(db_session: Session, org2: Organization) -> User:
    """Create engineer user for org2 (for testing org isolation)."""
    user = User(
        organization_id=org2.id,
        email="engineer@beta.com",
        full_name="Beta Engineer",
        role=UserRole.ENGINEER,
        password_hash=hash_password("beta123"),
        is_active=True
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def instrument_with_valid_cal(db_session: Session, org1: Organization) -> Instrument:
    """Create instrument with valid calibration."""
    instrument = Instrument(
        organization_id=org1.id,
        name="SIMS-1",
        vendor="Cameca",
        model="IMS-7f",
        serial="SIMS-001",
        interface="visa_gpib",
        location="Clean Room A",
        status=InstrumentStatus.ONLINE
    )
    db_session.add(instrument)
    db_session.commit()
    db_session.refresh(instrument)

    # Add valid calibration
    now = datetime.now(timezone.utc)
    calibration = Calibration(
        instrument_id=instrument.id,
        certificate_id="CAL-2025-001",
        issued_at=now - timedelta(days=30),
        expires_at=now + timedelta(days=335),  # Valid for 11 more months
        status=CalibrationStatus.VALID,
        performed_by="Metrology Lab Inc.",
        extra_metadata={"accuracy": "±0.1%"}
    )
    db_session.add(calibration)
    db_session.commit()

    return instrument


@pytest.fixture
def instrument_with_expired_cal(db_session: Session, org1: Organization) -> Instrument:
    """Create instrument with expired calibration."""
    instrument = Instrument(
        organization_id=org1.id,
        name="XRD-1",
        vendor="Bruker",
        model="D8 Advance",
        serial="XRD-001",
        interface="visa_ethernet",
        location="Analysis Lab",
        status=InstrumentStatus.ONLINE
    )
    db_session.add(instrument)
    db_session.commit()
    db_session.refresh(instrument)

    # Add expired calibration
    now = datetime.now(timezone.utc)
    calibration = Calibration(
        instrument_id=instrument.id,
        certificate_id="CAL-2024-999",
        issued_at=now - timedelta(days=400),
        expires_at=now - timedelta(days=35),  # Expired 35 days ago
        status=CalibrationStatus.EXPIRED,
        performed_by="Metrology Lab Inc."
    )
    db_session.add(calibration)
    db_session.commit()

    return instrument


@pytest.fixture
def instrument_no_cal(db_session: Session, org1: Organization) -> Instrument:
    """Create instrument without any calibration."""
    instrument = Instrument(
        organization_id=org1.id,
        name="SEM-1",
        vendor="Zeiss",
        model="Sigma 500",
        serial="SEM-001",
        interface="visa_usb",
        location="Imaging Lab",
        status=InstrumentStatus.ONLINE
    )
    db_session.add(instrument)
    db_session.commit()
    db_session.refresh(instrument)
    return instrument


@pytest.fixture
def admin_token(admin_user: User) -> dict:
    """Generate JWT token for admin user."""
    return create_token_pair(
        user_id=str(admin_user.id),
        org_id=str(admin_user.organization_id),
        role=admin_user.role.value,
        email=admin_user.email
    )


@pytest.fixture
def pi_token(pi_user: User) -> dict:
    """Generate JWT token for PI user."""
    return create_token_pair(
        user_id=str(pi_user.id),
        org_id=str(pi_user.organization_id),
        role=pi_user.role.value,
        email=pi_user.email
    )


@pytest.fixture
def engineer_token(engineer_user: User) -> dict:
    """Generate JWT token for engineer user."""
    return create_token_pair(
        user_id=str(engineer_user.id),
        org_id=str(engineer_user.organization_id),
        role=engineer_user.role.value,
        email=engineer_user.email
    )


@pytest.fixture
def technician_token(technician_user: User) -> dict:
    """Generate JWT token for technician user."""
    return create_token_pair(
        user_id=str(technician_user.id),
        org_id=str(technician_user.organization_id),
        role=technician_user.role.value,
        email=technician_user.email
    )


@pytest.fixture
def viewer_token(viewer_user: User) -> dict:
    """Generate JWT token for viewer user."""
    return create_token_pair(
        user_id=str(viewer_user.id),
        org_id=str(viewer_user.organization_id),
        role=viewer_user.role.value,
        email=viewer_user.email
    )
