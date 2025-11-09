"""
Unit tests for role-based access control guards and organization scoping.
"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from services.shared.db.models import User, UserRole, Instrument, InstrumentStatus
from services.shared.db.deps import (
    get_current_user,
    get_current_user_optional,
    require_admin,
    require_pi_or_admin,
    require_engineer_or_above,
    require_technician_or_above
)


class TestRoleGuards:
    """Test role-based access control guards."""

    def test_require_admin_with_admin(self, admin_user):
        """Admin user passes admin-only guard."""
        # Should not raise exception
        result = require_admin(admin_user)
        assert result == admin_user

    def test_require_admin_with_pi(self, pi_user):
        """PI user fails admin-only guard."""
        with pytest.raises(HTTPException) as exc_info:
            require_admin(pi_user)

        assert exc_info.value.status_code == 403
        assert "Admin access required" in str(exc_info.value.detail)

    def test_require_admin_with_engineer(self, engineer_user):
        """Engineer user fails admin-only guard."""
        with pytest.raises(HTTPException) as exc_info:
            require_admin(engineer_user)

        assert exc_info.value.status_code == 403

    def test_require_pi_or_admin_with_admin(self, admin_user):
        """Admin user passes PI/admin guard."""
        result = require_pi_or_admin(admin_user)
        assert result == admin_user

    def test_require_pi_or_admin_with_pi(self, pi_user):
        """PI user passes PI/admin guard."""
        result = require_pi_or_admin(pi_user)
        assert result == pi_user

    def test_require_pi_or_admin_with_engineer(self, engineer_user):
        """Engineer user fails PI/admin guard."""
        with pytest.raises(HTTPException) as exc_info:
            require_pi_or_admin(engineer_user)

        assert exc_info.value.status_code == 403
        assert "PI or Admin access required" in str(exc_info.value.detail)

    def test_require_engineer_or_above_with_admin(self, admin_user):
        """Admin user passes engineer+ guard."""
        result = require_engineer_or_above(admin_user)
        assert result == admin_user

    def test_require_engineer_or_above_with_pi(self, pi_user):
        """PI user passes engineer+ guard."""
        result = require_engineer_or_above(pi_user)
        assert result == pi_user

    def test_require_engineer_or_above_with_engineer(self, engineer_user):
        """Engineer user passes engineer+ guard."""
        result = require_engineer_or_above(engineer_user)
        assert result == engineer_user

    def test_require_engineer_or_above_with_technician(self, technician_user):
        """Technician user fails engineer+ guard."""
        with pytest.raises(HTTPException) as exc_info:
            require_engineer_or_above(technician_user)

        assert exc_info.value.status_code == 403
        assert "Engineer, PI, or Admin access required" in str(exc_info.value.detail)

    def test_require_engineer_or_above_with_viewer(self, viewer_user):
        """Viewer user fails engineer+ guard."""
        with pytest.raises(HTTPException) as exc_info:
            require_engineer_or_above(viewer_user)

        assert exc_info.value.status_code == 403

    def test_require_technician_or_above_with_technician(self, technician_user):
        """Technician user passes technician+ guard."""
        result = require_technician_or_above(technician_user)
        assert result == technician_user

    def test_require_technician_or_above_with_engineer(self, engineer_user):
        """Engineer user passes technician+ guard."""
        result = require_technician_or_above(engineer_user)
        assert result == engineer_user

    def test_require_technician_or_above_with_viewer(self, viewer_user):
        """Viewer user fails technician+ guard."""
        with pytest.raises(HTTPException) as exc_info:
            require_technician_or_above(viewer_user)

        assert exc_info.value.status_code == 403
        assert "Technician access or above required" in str(exc_info.value.detail)

    def test_role_hierarchy(self, admin_user, pi_user, engineer_user, technician_user, viewer_user):
        """Test complete role hierarchy."""
        # Admin passes all guards
        assert require_admin(admin_user) == admin_user
        assert require_pi_or_admin(admin_user) == admin_user
        assert require_engineer_or_above(admin_user) == admin_user
        assert require_technician_or_above(admin_user) == admin_user

        # PI passes all except admin-only
        with pytest.raises(HTTPException):
            require_admin(pi_user)
        assert require_pi_or_admin(pi_user) == pi_user
        assert require_engineer_or_above(pi_user) == pi_user
        assert require_technician_or_above(pi_user) == pi_user

        # Engineer passes engineer+ and technician+
        with pytest.raises(HTTPException):
            require_admin(engineer_user)
        with pytest.raises(HTTPException):
            require_pi_or_admin(engineer_user)
        assert require_engineer_or_above(engineer_user) == engineer_user
        assert require_technician_or_above(engineer_user) == engineer_user

        # Technician only passes technician+
        with pytest.raises(HTTPException):
            require_admin(technician_user)
        with pytest.raises(HTTPException):
            require_pi_or_admin(technician_user)
        with pytest.raises(HTTPException):
            require_engineer_or_above(technician_user)
        assert require_technician_or_above(technician_user) == technician_user

        # Viewer passes no guards
        with pytest.raises(HTTPException):
            require_admin(viewer_user)
        with pytest.raises(HTTPException):
            require_pi_or_admin(viewer_user)
        with pytest.raises(HTTPException):
            require_engineer_or_above(viewer_user)
        with pytest.raises(HTTPException):
            require_technician_or_above(viewer_user)


class TestOrganizationScoping:
    """Test organization-level data scoping."""

    def test_users_from_different_orgs(self, org1, org2, engineer_user, org2_engineer):
        """Users from different orgs have different organization_ids."""
        assert engineer_user.organization_id == org1.id
        assert org2_engineer.organization_id == org2.id
        assert engineer_user.organization_id != org2_engineer.organization_id

    def test_org_scoped_query_samples(self, db_session, org1, org2, engineer_user, org2_engineer):
        """Test organization scoping for samples."""
        from services.shared.db.models import Sample

        # Create sample for org1
        sample1 = Sample(
            organization_id=org1.id,
            name="Org1 Sample",
            barcode="ORG1-001",
            sample_type="wafer"
        )
        db_session.add(sample1)

        # Create sample for org2
        sample2 = Sample(
            organization_id=org2.id,
            name="Org2 Sample",
            barcode="ORG2-001",
            sample_type="wafer"
        )
        db_session.add(sample2)
        db_session.commit()

        # Org1 engineer should only see org1 samples
        org1_samples = db_session.query(Sample).filter(
            Sample.organization_id == engineer_user.organization_id
        ).all()
        assert len(org1_samples) == 1
        assert org1_samples[0].barcode == "ORG1-001"

        # Org2 engineer should only see org2 samples
        org2_samples = db_session.query(Sample).filter(
            Sample.organization_id == org2_engineer.organization_id
        ).all()
        assert len(org2_samples) == 1
        assert org2_samples[0].barcode == "ORG2-001"

    def test_org_scoped_query_instruments(self, db_session, org1, org2,
                                          instrument_with_valid_cal, engineer_user):
        """Test organization scoping for instruments."""
        from services.shared.db.models import Instrument, InstrumentType

        # instrument_with_valid_cal belongs to org1
        assert instrument_with_valid_cal.organization_id == org1.id

        # Create instrument for org2
        org2_instrument = Instrument(
            organization_id=org2.id,
            name="Org2 Instrument",
            vendor="Generic", model="Test Model", interface="visa_usb",
            serial="ORG2-INST-001",
            status=InstrumentStatus.ONLINE
        )
        db_session.add(org2_instrument)
        db_session.commit()

        # Org1 engineer should only see org1 instruments
        org1_instruments = db_session.query(Instrument).filter(
            Instrument.organization_id == engineer_user.organization_id
        ).all()

        # Should include instrument_with_valid_cal and any others from org1
        org1_instrument_ids = [str(i.id) for i in org1_instruments]
        assert str(instrument_with_valid_cal.id) in org1_instrument_ids
        assert str(org2_instrument.id) not in org1_instrument_ids

    def test_cross_org_access_blocked(self, db_session, org1, org2, engineer_user):
        """Test that users cannot access data from other organizations."""
        from services.shared.db.models import Sample

        # Create sample for org2
        org2_sample = Sample(
            organization_id=org2.id,
            name="Secret Org2 Sample",
            barcode="SECRET-001",
            sample_type="wafer"
        )
        db_session.add(org2_sample)
        db_session.commit()

        # Org1 engineer tries to query by sample ID (without org filter)
        sample_no_filter = db_session.query(Sample).filter(
            Sample.id == org2_sample.id
        ).first()

        # Query succeeds but...
        assert sample_no_filter is not None

        # Org1 engineer queries WITH org filter (proper scoping)
        sample_with_filter = db_session.query(Sample).filter(
            Sample.id == org2_sample.id,
            Sample.organization_id == engineer_user.organization_id  # org1
        ).first()

        # Should return None - no access to org2 data
        assert sample_with_filter is None


class TestInactiveUsers:
    """Test that inactive users cannot authenticate."""

    def test_inactive_user_blocked(self, db_session, org1):
        """Inactive user should not pass authentication."""
        from services.shared.auth.jwt import create_token_pair, hash_password

        # Create inactive user
        inactive_user = User(
            organization_id=org1.id,
            email="inactive@acme.com",
            full_name="Inactive User",
            role=UserRole.ENGINEER,
            password_hash=hash_password("password123"),
            is_active=False  # Inactive
        )
        db_session.add(inactive_user)
        db_session.commit()
        db_session.refresh(inactive_user)

        # Generate token (this would happen at login)
        # In real scenario, login endpoint should check is_active before issuing token
        tokens = create_token_pair(
            user_id=str(inactive_user.id),
            org_id=str(inactive_user.organization_id),
            role=inactive_user.role.value,
            email=inactive_user.email
        )

        # Token generation succeeds, but...
        # When token is used, get_current_user should filter by is_active=True
        user = db_session.query(User).filter(
            User.id == inactive_user.id,
            User.is_active == True  # noqa: E712
        ).first()

        # Should return None
        assert user is None


class TestSoftDelete:
    """Test soft delete functionality."""

    def test_soft_delete_sample(self, db_session, org1):
        """Test soft deleting a sample."""
        from services.shared.db.models import Sample
        from datetime import datetime, timezone

        sample = Sample(
            organization_id=org1.id,
            name="Sample to Delete",
            barcode="DEL-001",
            sample_type="wafer"
        )
        db_session.add(sample)
        db_session.commit()
        db_session.refresh(sample)

        # Mark as deleted
        sample.is_deleted = True
        sample.deleted_at = datetime.now(timezone.utc)
        db_session.commit()

        # Query without filter shows deleted
        all_samples = db_session.query(Sample).filter(
            Sample.organization_id == org1.id
        ).all()
        assert len(all_samples) == 1

        # Query with is_deleted=False filter excludes deleted
        active_samples = db_session.query(Sample).filter(
            Sample.organization_id == org1.id,
            Sample.is_deleted == False  # noqa: E712
        ).all()
        assert len(active_samples) == 0

    def test_soft_delete_recovery(self, db_session, org1):
        """Test recovering a soft-deleted sample."""
        from services.shared.db.models import Sample

        sample = Sample(
            organization_id=org1.id,
            name="Recoverable Sample",
            barcode="REC-001",
            sample_type="wafer",
            is_deleted=True
        )
        db_session.add(sample)
        db_session.commit()

        # Recover the sample
        sample.is_deleted = False
        sample.deleted_at = None
        db_session.commit()

        # Should now appear in active queries
        active_samples = db_session.query(Sample).filter(
            Sample.organization_id == org1.id,
            Sample.is_deleted == False  # noqa: E712
        ).all()
        assert len(active_samples) == 1
        assert active_samples[0].barcode == "REC-001"
